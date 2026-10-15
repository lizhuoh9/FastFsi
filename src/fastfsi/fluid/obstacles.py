"""The obstacle mask and its topology (mixin)."""

import numpy as np
import taichi as ti


class ObstacleMask:
    """Mixin of IncompressibleFluidSolver; not used on its own.  Contract (solver attributes):
      reads:  _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES, no_slip_component_face_valid_mask,
    no_slip_sampling_obstacle, obstacle, velocity, velocity_dirichlet_boundary_active,
    velocity_dirichlet_boundary_active_component_mask, velocity_dirichlet_boundary_authority,
    velocity_dirichlet_boundary_component_enforcement_weight, velocity_dirichlet_boundary_enforcement_weight,
    velocity_dirichlet_boundary_external_exact_component_mask,
    velocity_dirichlet_boundary_hard_fixed_component_mask, velocity_dirichlet_boundary_owned_component_mask,
    velocity_dirichlet_boundary_owned_row, velocity_dirichlet_boundary_pressure_mobility,
    velocity_dirichlet_boundary_value_mps
      writes: _sst_wall_distance_base_cache_key, _sst_wall_distance_cache_key, _sst_wall_distance_valid"""

    @ti.kernel
    def _build_canonical_no_slip_sampling_obstacle_kernel(self):
        for i, j, k in self.obstacle:
            self.no_slip_sampling_obstacle[i, j, k] = self.obstacle[i, j, k]
            if (
                self.velocity_dirichlet_boundary_active_component_mask[i, j, k]
                & 7
            ) != 0:
                self.no_slip_sampling_obstacle[i, j, k] = 0

    @ti.func
    def _no_slip_component_survives_terminal_cleanup(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        component_axis: ti.i32,
        canonical_authority: ti.template(),
    ):
        """Return whether one stored MAC component survives final clamping."""

        survives = 0
        if component_axis == 0:
            if (
                i > 0
                and self.obstacle[i, j, k] == 0
                and self.obstacle[i - 1, j, k] == 0
            ):
                survives = 1
        elif component_axis == 1:
            if (
                j > 0
                and self.obstacle[i, j, k] == 0
                and self.obstacle[i, j - 1, k] == 0
            ):
                survives = 1
        else:
            if (
                k > 0
                and self.obstacle[i, j, k] == 0
                and self.obstacle[i, j, k - 1] == 0
            ):
                survives = 1
        if self._obstacle_interface_component_is_prescribed(
            i,
            j,
            k,
            component_axis,
            canonical_authority,
        ):
            survives = 1
        return survives

    @ti.kernel
    def _build_no_slip_component_face_valid_mask_kernel(
        self,
        canonical_authority: ti.template(),
    ):
        for i, j, k in self.obstacle:
            valid_mask = 0
            for axis in ti.static(range(3)):
                if self._no_slip_component_survives_terminal_cleanup(
                    i,
                    j,
                    k,
                    axis,
                    canonical_authority,
                ):
                    valid_mask |= 1 << axis
            self.no_slip_component_face_valid_mask[i, j, k] = valid_mask

    def prepare_no_slip_component_face_valid_mask(self):
        """Build and register the canonical derived MAC-face sampling view."""

        if self.velocity_dirichlet_boundary_authority == "canonical":
            self._build_canonical_no_slip_sampling_obstacle_kernel()
        else:
            raise NotImplementedError("only the 'canonical' velocity boundary classification is supported")
        self._build_no_slip_component_face_valid_mask_kernel(
            self._velocity_dirichlet_boundary_authority_code()
        )
        if self.velocity_dirichlet_boundary_authority == "canonical":
            self._register_velocity_dirichlet_component_registry_consumer_generation(
                "no_slip",
                capability=self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES[
                    "no_slip"
                ],
            )
        return self.no_slip_component_face_valid_mask

    def _invalidate_external_obstacle_topology_derived_state(self) -> None:
        """Retire every host contract derived from external obstacle topology."""

        self._invalidate_pressure_reachability()
        self._invalidate_velocity_dirichlet_component_registry()
        self._sst_wall_distance_valid = False
        self._sst_wall_distance_cache_key = None
        self._sst_wall_distance_base_cache_key = None

    @ti.func
    def _obstacle_interface_component_is_prescribed(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        component_axis: ti.i32,
        canonical_authority: ti.template(),
    ):
        """Return whether one backward-MAC obstacle face has an exact owner."""

        is_prescribed = 0
        component_bit = 1 << component_axis
        minus_i = i
        minus_j = j
        minus_k = k
        has_minus_neighbor = 0
        if component_axis == 0:
            has_minus_neighbor = i > 0
            minus_i = ti.max(i - 1, 0)
        elif component_axis == 1:
            has_minus_neighbor = j > 0
            minus_j = ti.max(j - 1, 0)
        else:
            has_minus_neighbor = k > 0
            minus_k = ti.max(k - 1, 0)

        target = self.velocity_dirichlet_boundary_value_mps[i, j, k][
            component_axis
        ]
        storage_is_fluid = self.obstacle[i, j, k] == 0
        minus_is_fluid = self.obstacle[minus_i, minus_j, minus_k] == 0
        physical_interface = (
            (storage_is_fluid and not minus_is_fluid)
            or (not storage_is_fluid and minus_is_fluid)
        )
        contract_tolerance = ti.cast(1.0e-6, ti.f32)
        target_is_finite = not (
            ti.math.isnan(target) or ti.math.isinf(target)
        )
        if ti.static(canonical_authority == 1):
            pressure_mobility = (
                self.velocity_dirichlet_boundary_pressure_mobility[i, j, k][
                    component_axis
                ]
            )
            enforcement_weight = (
                self.velocity_dirichlet_boundary_component_enforcement_weight[
                    i, j, k
                ][component_axis]
            )
            mobility_is_closed = (
                not ti.math.isnan(pressure_mobility)
                and not ti.math.isinf(pressure_mobility)
                and ti.abs(pressure_mobility) <= contract_tolerance
            )
            enforcement_is_exact = (
                not ti.math.isnan(enforcement_weight)
                and not ti.math.isinf(enforcement_weight)
                and ti.abs(enforcement_weight - 1.0) <= contract_tolerance
            )
            if (
                has_minus_neighbor
                and physical_interface
                and (
                    self.velocity_dirichlet_boundary_active_component_mask[
                        i, j, k
                    ]
                    & component_bit
                )
                != 0
                and (
                    self.velocity_dirichlet_boundary_hard_fixed_component_mask[
                        i, j, k
                    ]
                    & component_bit
                )
                != 0
                and (
                    self.velocity_dirichlet_boundary_owned_component_mask[i, j, k]
                    & component_bit
                )
                != 0
                and (
                    self.velocity_dirichlet_boundary_external_exact_component_mask[
                        i, j, k
                    ]
                    & component_bit
                )
                == 0
                and mobility_is_closed
                and enforcement_is_exact
                and target_is_finite
            ):
                is_prescribed = 1
        else:
            enforcement_weight = (
                self.velocity_dirichlet_boundary_enforcement_weight[i, j, k]
            )
            enforcement_is_exact = (
                not ti.math.isnan(enforcement_weight)
                and not ti.math.isinf(enforcement_weight)
                and ti.abs(enforcement_weight - 1.0) <= contract_tolerance
            )
            if (
                has_minus_neighbor
                and physical_interface
                and self.velocity_dirichlet_boundary_active[i, j, k] != 0
                and (
                    self.velocity_dirichlet_boundary_hard_fixed_component_mask[
                        i, j, k
                    ]
                    & component_bit
                )
                != 0
                and self.velocity_dirichlet_boundary_owned_row[i, j, k] != 0
                and enforcement_is_exact
                and target_is_finite
            ):
                # Legacy interface rows still carry direct velocity ownership even
                # though pressure-normal provenance is maintained separately.
                # The generic obstacle cleanup must not erase that exact moving
                # interface after the projection has applied it.
                is_prescribed = 1
        return is_prescribed

    @ti.func
    def _continuity_obstacle_face_velocity(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        axis: ti.template(),
    ):
        """Return the mass-flux velocity of one prescribed fluid-obstacle face."""

        return self.velocity[i, j, k][axis]


@ti.kernel
def _write_obstacle_kernel(obstacle: ti.template(), velocity: ti.template(), velocity_prev: ti.template(),
                           mask: ti.types.ndarray()):
    for i, j, k in obstacle:
        new = mask[i, j, k]
        if new != 0 and obstacle[i, j, k] == 0:
            velocity[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            velocity_prev[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        obstacle[i, j, k] = 1 if new != 0 else 0


def replace_obstacle_cells(fluid, mask, *, pressure_outlet_zmin=True):
    """Set fluid.obstacle to `mask` (i32 0/1, grid shape), zeroing the velocity of newly blocked cells, then retire and
    rebuild every obstacle-derived structure of the solver in the order it requires (nullspace graph and external
    topology invalidated, the velocity boundary registry prepared and sealed, outlet reachability re-marked on the
    single-level path).  Returns the reachability report (unreached cell count)."""
    from fastfsi.fluid.outlet_reachability import single_level_reachability_flood
    mask = np.ascontiguousarray(np.asarray(mask, dtype=np.int32))
    if mask.shape != tuple(fluid.grid.grid_nodes):
        raise ValueError("obstacle mask shape %s != grid %s" % (mask.shape, tuple(fluid.grid.grid_nodes)))
    _write_obstacle_kernel(fluid.obstacle, fluid.velocity, fluid.velocity_prev, mask)
    fluid._invalidate_pressure_nullspace_component_graph()
    fluid._invalidate_external_obstacle_topology_derived_state()
    fluid.prepare_and_seal_velocity_dirichlet_component_registry()
    with single_level_reachability_flood():
        return fluid.mark_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=bool(pressure_outlet_zmin))
