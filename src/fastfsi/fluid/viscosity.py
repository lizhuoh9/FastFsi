"""Momentum diffusion: the laminar viscous terms and the wall-face terms (mixin)."""

import taichi as ti


class MomentumDiffusion:
    """Mixin of IncompressibleFluidSolver; not used on its own.  Contract (solver attributes):
      reads:  canonical_exact_hard_face_component_mask, cell_width_x_m, cell_width_y_m, cell_width_z_m,
    center_distance_x_m, center_distance_y_m, center_distance_z_m,
    external_velocity_boundary_x_face_active_component_mask, external_velocity_boundary_x_face_value_mps,
    external_velocity_boundary_y_face_active_component_mask, external_velocity_boundary_y_face_value_mps,
    external_velocity_boundary_z_face_active_component_mask, external_velocity_boundary_z_face_value_mps,
    laminar_canonical_fluid_patch_exchange, muscl_momentum_dual_volume_m3, nx, ny, nz, obstacle, rho,
    sst_eddy_viscosity_pa_s, sst_obstacle_interface_wall_target_component_mask, velocity,
    velocity_dirichlet_boundary_active, velocity_dirichlet_boundary_active_component_mask,
    velocity_dirichlet_boundary_authority, velocity_dirichlet_boundary_authority_code_device,
    velocity_dirichlet_boundary_component_enforcement_weight, velocity_dirichlet_boundary_enforcement_weight,
    velocity_dirichlet_boundary_external_exact_component_mask,
    velocity_dirichlet_boundary_hard_fixed_component_mask, velocity_dirichlet_boundary_owned_component_mask,
    velocity_dirichlet_boundary_owned_row, velocity_dirichlet_boundary_pressure_mobility,
    velocity_dirichlet_boundary_value_mps, velocity_prev
      writes: -"""

    @ti.func
    def _sst_obstacle_interface_normal_owner_is_exact(
        self,
        i,
        j,
        k,
        normal_axis: ti.template(),
        canonical_authority: ti.template(),
    ):
        """Static-axis form of the exact backward-MAC interface contract."""

        component_bit = 1 << normal_axis
        minus_i = i
        minus_j = j
        minus_k = k
        has_minus_neighbor = False
        if ti.static(normal_axis == 0):
            has_minus_neighbor = i > 0
            minus_i = ti.max(i - 1, 0)
        elif ti.static(normal_axis == 1):
            has_minus_neighbor = j > 0
            minus_j = ti.max(j - 1, 0)
        else:
            has_minus_neighbor = k > 0
            minus_k = ti.max(k - 1, 0)

        target = self.velocity_dirichlet_boundary_value_mps[i, j, k][
            normal_axis
        ]
        storage_is_fluid = self.obstacle[i, j, k] == 0
        minus_is_fluid = self.obstacle[minus_i, minus_j, minus_k] == 0
        physical_interface = (
            (storage_is_fluid and not minus_is_fluid)
            or (not storage_is_fluid and minus_is_fluid)
        )
        contract_tolerance = ti.cast(1.0e-6, ti.f32)
        target_is_finite = target == target and ti.abs(target) < 1.0e30
        is_exact = 0
        if ti.static(canonical_authority == 1):
            pressure_mobility = (
                self.velocity_dirichlet_boundary_pressure_mobility[i, j, k][
                    normal_axis
                ]
            )
            enforcement_weight = (
                self.velocity_dirichlet_boundary_component_enforcement_weight[
                    i, j, k
                ][normal_axis]
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
                and pressure_mobility == pressure_mobility
                and ti.abs(pressure_mobility) <= contract_tolerance
                and enforcement_weight == enforcement_weight
                and ti.abs(enforcement_weight - 1.0) <= contract_tolerance
                and target_is_finite
            ):
                is_exact = 1
        else:
            enforcement_weight = (
                self.velocity_dirichlet_boundary_enforcement_weight[i, j, k]
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
                and enforcement_weight == enforcement_weight
                and ti.abs(enforcement_weight - 1.0) <= contract_tolerance
                and target_is_finite
            ):
                is_exact = 1
        return is_exact

    @ti.kernel
    def _prepare_sst_obstacle_interface_wall_target_masks_kernel(
        self,
        canonical_authority: ti.template(),
    ):
        """Cache exact target-component masks for each physical interface."""

        for i, j, k in self.sst_obstacle_interface_wall_target_component_mask:
            normal_masks = ti.Vector([0, 0, 0])
            for normal_axis in ti.static(range(3)):
                if self._sst_obstacle_interface_normal_owner_is_exact(
                    i, j, k, normal_axis, canonical_authority
                ):
                    component_mask = 0
                    contract_tolerance = ti.cast(1.0e-6, ti.f32)
                    for component in ti.static(range(3)):
                        component_bit = 1 << component
                        target = self.velocity_dirichlet_boundary_value_mps[
                            i, j, k
                        ][component]
                        component_is_exact = 1
                        if ti.static(canonical_authority == 1):
                            pressure_mobility = (
                                self.velocity_dirichlet_boundary_pressure_mobility[
                                    i, j, k
                                ][component]
                            )
                            enforcement_weight = (
                                self.velocity_dirichlet_boundary_component_enforcement_weight[
                                    i, j, k
                                ][component]
                            )
                            if (
                                self.velocity_dirichlet_boundary_active_component_mask[
                                    i, j, k
                                ]
                                & component_bit
                            ) == 0:
                                component_is_exact = 0
                            if (
                                self.velocity_dirichlet_boundary_hard_fixed_component_mask[
                                    i, j, k
                                ]
                                & component_bit
                            ) == 0:
                                component_is_exact = 0
                            if (
                                self.velocity_dirichlet_boundary_owned_component_mask[
                                    i, j, k
                                ]
                                & component_bit
                            ) == 0:
                                component_is_exact = 0
                            if (
                                self.velocity_dirichlet_boundary_external_exact_component_mask[
                                    i, j, k
                                ]
                                & component_bit
                            ) != 0:
                                component_is_exact = 0
                            if (
                                pressure_mobility != pressure_mobility
                                or ti.abs(pressure_mobility) > contract_tolerance
                            ):
                                component_is_exact = 0
                            if (
                                enforcement_weight != enforcement_weight
                                or ti.abs(enforcement_weight - 1.0)
                                > contract_tolerance
                            ):
                                component_is_exact = 0
                        else:
                            if (
                                self.velocity_dirichlet_boundary_hard_fixed_component_mask[
                                    i, j, k
                                ]
                                & component_bit
                            ) == 0:
                                component_is_exact = 0
                            if (
                                self.velocity_dirichlet_boundary_external_exact_component_mask[
                                    i, j, k
                                ]
                                & component_bit
                            ) != 0:
                                component_is_exact = 0
                        if target != target or ti.abs(target) >= 1.0e30:
                            component_is_exact = 0
                        if component_is_exact != 0:
                            component_mask |= component_bit
                    normal_masks[normal_axis] = component_mask
            self.sst_obstacle_interface_wall_target_component_mask[
                i, j, k
            ] = normal_masks

    @ti.func
    def _canonical_exact_component_owner(
        self,
        i,
        j,
        k,
        component: ti.template(),
    ):
        """Return whether a canonical fluid row exactly owns one component."""

        exact_owner = 0
        if (
            self.velocity_dirichlet_boundary_authority_code_device[None] != 0
            and self.obstacle[i, j, k] == 0
        ):
            contract_tolerance = ti.cast(1.0e-6, ti.f32)
            component_bit = 1 << component
            pressure_mobility = (
                self.velocity_dirichlet_boundary_pressure_mobility[i, j, k][
                    component
                ]
            )
            enforcement_weight = (
                self.velocity_dirichlet_boundary_component_enforcement_weight[
                    i, j, k
                ][component]
            )
            target = self.velocity_dirichlet_boundary_value_mps[i, j, k][component]
            if (
                (
                    self.velocity_dirichlet_boundary_active_component_mask[i, j, k]
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
                and pressure_mobility == pressure_mobility
                and ti.abs(pressure_mobility) <= contract_tolerance
                and enforcement_weight == enforcement_weight
                and ti.abs(enforcement_weight - 1.0) <= contract_tolerance
                and target == target
                and ti.abs(target) < 1.0e30
            ):
                exact_owner = 1
        return exact_owner

    @ti.kernel
    def _refresh_canonical_exact_hard_face_component_mask_kernel(self):
        for i, j, k in self.canonical_exact_hard_face_component_mask:
            exact_mask = 0
            if self.velocity_dirichlet_boundary_authority_code_device[None] != 0:
                for component in ti.static(range(3)):
                    if (
                        self._canonical_exact_component_owner(
                            i, j, k, component
                        )
                        != 0
                    ):
                        exact_mask |= 1 << component
            self.canonical_exact_hard_face_component_mask[i, j, k] = exact_mask

    @ti.kernel
    def _clear_canonical_exact_hard_face_component_mask_kernel(self):
        for i, j, k in self.canonical_exact_hard_face_component_mask:
            self.canonical_exact_hard_face_component_mask[i, j, k] = 0

    def _refresh_canonical_exact_hard_face_component_mask(self) -> None:
        authority = self._validated_velocity_dirichlet_boundary_authority(
            self.velocity_dirichlet_boundary_authority
        )
        if authority == "canonical":
            self._refresh_canonical_exact_hard_face_component_mask_kernel()
        else:
            self._clear_canonical_exact_hard_face_component_mask_kernel()

    @ti.func
    def _canonical_hard_face_target_component_mask(
        self,
        i,
        j,
        k,
        normal_axis,
    ):
        """Return exact components owned by one canonical fluid-row face."""

        exact_component_mask = 0
        for component in ti.static(range(3)):
            if self._canonical_exact_component_owner(
                i, j, k, component
            ) != 0:
                exact_component_mask |= 1 << component
        normal_bit = 1 << normal_axis
        if (exact_component_mask & normal_bit) == 0:
            exact_component_mask = 0
        return exact_component_mask

    @ti.func
    def _momentum_canonical_wall_target_component_mask(
        self, i, j, k, normal_axis, molecular_only: ti.template(),
    ):
        # Laminar transport uses the exact face registry independently of SST.
        # Resolved SST retains its existing correlation-dependent wall geometry.
        component_mask = 0
        if ti.static(molecular_only):
            component_mask = self._canonical_hard_face_target_component_mask(
                i, j, k, normal_axis,
            )
        else:
            ti.static_assert(False, "turbulence models are not supported: the fluid solver is laminar")
        return component_mask

    @ti.func
    def _sst_momentum_component_row_contract(
        self,
        i,
        j,
        k,
        component,
        pressure_outlet_zmin,
        velocity_inlet_zmax_mode,
        molecular_only: ti.template() = False,
    ):
        """Return ``(kind, target)`` for one packed backward-MAC row.

        ``kind`` is zero for inactive solid storage, one for a free momentum
        unknown, and two for an exact Dirichlet row.  The contract is
        component-native: an immersed face remains exact whether its canonical
        owner is stored on the fluid (A) or obstacle (B) scalar row.  Minimum
        external normal faces are represented by packed row zero itself and
        are therefore identities rather than half-cell penalty equations.
        """

        plus_fluid = 1 if self.obstacle[i, j, k] == 0 else 0
        has_minus = 0
        minus_fluid = 0
        coordinate = i
        if component == 0:
            if i > 0:
                has_minus = 1
                minus_fluid = 1 if self.obstacle[i - 1, j, k] == 0 else 0
        elif component == 1:
            coordinate = j
            if j > 0:
                has_minus = 1
                minus_fluid = 1 if self.obstacle[i, j - 1, k] == 0 else 0
        else:
            coordinate = k
            if k > 0:
                has_minus = 1
                minus_fluid = 1 if self.obstacle[i, j, k - 1] == 0 else 0

        kind = 0
        target = ti.cast(0.0, ti.f64)
        if has_minus == 0:
            if plus_fluid != 0:
                kind = 1
        elif minus_fluid == plus_fluid:
            if plus_fluid != 0:
                kind = 1
        else:
            # A component-normal immersed face is a physical boundary, not a
            # diffused scalar-cell row.  Missing exact provenance remains an
            # impermeable zero target; a validated owner supplies wall motion.
            kind = 2
            component_bit = 1 << component
            exact_owner = (
                self.sst_obstacle_interface_wall_target_component_mask[
                    i, j, k
                ][component]
                & component_bit
            ) != 0
            if exact_owner:
                target = ti.cast(
                    self.velocity_dirichlet_boundary_value_mps[i, j, k][
                        component
                    ],
                    ti.f64,
                )

        component_bit = 1 << component
        canonical_target_mask = (
            self._momentum_canonical_wall_target_component_mask(
                i,
                j,
                k,
                component,
                molecular_only,
            )
        )
        if (canonical_target_mask & component_bit) != 0:
            # A canonical exact interface component is an identity MAC row even
            # when both adjacent scalar storage cells remain fluid.  Leaving
            # it free would assemble a shared edge through the physical wall.
            kind = 2
            target = ti.cast(
                self.velocity_dirichlet_boundary_value_mps[i, j, k][component],
                ti.f64,
            )

        if coordinate == 0 and plus_fluid != 0:
            external = self._physical_exterior_normal_contract(
                self.velocity, component, 0, i, j, k,
                pressure_outlet_zmin, velocity_inlet_zmax_mode,
            )
            if external[0] != 0:
                kind = 2
                target = ti.cast(external[1], ti.f64)

        return ti.Vector([ti.cast(kind, ti.f64), target])

    @ti.func
    def _momentum_kinematic_viscosity_at(self, i, j, k, molecular_nu_m2_s, molecular_only: ti.template()):
        viscosity = molecular_nu_m2_s
        if ti.static(not molecular_only):
            viscosity += self.sst_eddy_viscosity_pa_s[i, j, k] / self.rho
        return viscosity

    @ti.func
    def _laminar_momentum_diffusion_at(
        self, i, j, k, base_velocity, dt_s, molecular_nu_m2_s,
        wall_flags, pressure_outlet_zmin, velocity_inlet_zmax_mode,
    ):
        """Explicit molecular diffusion on the same MAC volumes as Helmholtz.

        Read only velocity_prev; exact A/B and exterior rows return their target.
        Normal edges use whole scalar widths; transverse faces retain both half
        patches, including their individual wall targets.
        """
        result = base_velocity
        for component in range(3):
            contract = self._sst_momentum_component_row_contract(
                i, j, k, component, pressure_outlet_zmin, velocity_inlet_zmax_mode, True,
            )
            kind = ti.cast(contract[0], ti.i32)
            if kind != 1:
                result[component] = ti.cast(contract[1], ti.f32)
            else:
                center = ti.cast(self.velocity_prev[i, j, k][component], ti.f64)
                increment = ti.cast(0.0, ti.f64)
                for axis_index in range(3):
                    for side_code in range(2):
                        side = 2 * side_code - 1
                        ni, nj, nk = i, j, k
                        if axis_index == 0:
                            ni += side
                        elif axis_index == 1:
                            nj += side
                        else:
                            nk += side
                        in_domain = 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz
                        neighbor_kind = 0
                        neighbor_target = ti.cast(0.0, ti.f64)
                        if in_domain:
                            neighbor = self._sst_momentum_component_row_contract(
                                ni, nj, nk, component, pressure_outlet_zmin, velocity_inlet_zmax_mode, True,
                            )
                            neighbor_kind = ti.cast(neighbor[0], ti.i32)
                            neighbor_target = neighbor[1]
                        boundary = ti.Vector([0.0, 0.0], dt=ti.f64)
                        if axis_index == component:
                            boundary = self._sst_momentum_normal_boundary_face_terms(
                                i, j, k, component, side, dt_s, molecular_nu_m2_s,
                                in_domain, neighbor_kind, neighbor_target,
                                pressure_outlet_zmin, velocity_inlet_zmax_mode, True,
                            )
                        else:
                            boundary = self._sst_momentum_transverse_boundary_face_terms(
                                i, j, k, component, axis_index, side, dt_s, molecular_nu_m2_s,
                                wall_flags[2 * axis_index + side_code], neighbor_kind, neighbor_target, True,
                            )
                        increment += boundary[1] - boundary[0] * center
                        if in_domain and neighbor_kind == 1:
                            coefficient = ti.cast(0.0, ti.f64)
                            if side_code == 0:
                                coefficient = self._sst_momentum_shared_edge_coefficient(
                                    ni, nj, nk, i, j, k, component, axis_index, dt_s, molecular_nu_m2_s, True,
                                )
                            else:
                                coefficient = self._sst_momentum_shared_edge_coefficient(
                                    i, j, k, ni, nj, nk, component, axis_index, dt_s, molecular_nu_m2_s, True,
                                )
                            increment += coefficient * (ti.cast(self.velocity_prev[ni, nj, nk][component], ti.f64) - center)
                volume = ti.cast(self.muscl_momentum_dual_volume_m3[i, j, k, component], ti.f64)
                if volume > 0.0:
                    result[component] += ti.cast(increment / volume, ti.f32)
        return result

    @ti.func
    def _sst_momentum_shared_edge_coefficient(
        self,
        i,
        j,
        k,
        ni,
        nj,
        nk,
        component,
        axis_index,
        dt_s,
        molecular_nu_m2_s,
        molecular_only: ti.template() = False,
    ):
        """Return the physical shared transmissibility of one positive edge.

        For a component-normal edge, the two MAC faces exchange momentum
        through the scalar cell between them.  A transverse dual face is the
        sum of the component-minus and component-plus half-cell patches; each
        patch interpolates its own scalar-cell viscosity.  Using only the
        plus-storage scalar viscosity is non-conservative when ``nu_eff``
        varies across the component direction.
        """

        coefficient = ti.cast(0.0, ti.f64)
        if axis_index == component:
            nu_cell = self._momentum_kinematic_viscosity_at(i, j, k, molecular_nu_m2_s, molecular_only)
            area = self.cell_width_y_m[j] * self.cell_width_z_m[k]
            distance = self.cell_width_x_m[i]
            if component == 1:
                area = self.cell_width_x_m[i] * self.cell_width_z_m[k]
                distance = self.cell_width_y_m[j]
            elif component == 2:
                area = self.cell_width_x_m[i] * self.cell_width_y_m[j]
                distance = self.cell_width_z_m[k]
            coefficient = (
                ti.cast(dt_s, ti.f64)
                * ti.cast(nu_cell, ti.f64)
                * ti.cast(area, ti.f64)
                / ti.cast(distance, ti.f64)
            )
        else:
            center_distance = self.center_distance_x_m[ni]
            other_width = self.cell_width_z_m[k]
            if axis_index == 0:
                if component == 2:
                    other_width = self.cell_width_y_m[j]
            elif axis_index == 1:
                center_distance = self.center_distance_y_m[nj]
                other_width = self.cell_width_z_m[k]
                if component == 2:
                    other_width = self.cell_width_x_m[i]
            elif axis_index == 2:
                center_distance = self.center_distance_z_m[nk]
                other_width = self.cell_width_y_m[j]
                if component == 1:
                    other_width = self.cell_width_x_m[i]

            component_coordinate = i
            plus_component_width = self.cell_width_x_m[i]
            if component == 1:
                component_coordinate = j
                plus_component_width = self.cell_width_y_m[j]
            elif component == 2:
                component_coordinate = k
                plus_component_width = self.cell_width_z_m[k]

            plus_patch_canonical_wall = 0
            if (
                self._momentum_canonical_wall_target_component_mask(
                    ni,
                    nj,
                    nk,
                    axis_index,
                    molecular_only,
                )
                != 0
            ):
                plus_patch_canonical_wall = 1
            if ti.static(molecular_only):
                if self.laminar_canonical_fluid_patch_exchange[None] != 0:
                    plus_patch_canonical_wall = 0
            if (
                self.obstacle[i, j, k] == 0
                and self.obstacle[ni, nj, nk] == 0
                and plus_patch_canonical_wall == 0
            ):
                nu_current = self._momentum_kinematic_viscosity_at(i, j, k, molecular_nu_m2_s, molecular_only)
                nu_neighbor = self._momentum_kinematic_viscosity_at(ni, nj, nk, molecular_nu_m2_s, molecular_only)
                patch_area = 0.5 * plus_component_width * other_width
                coefficient += (
                    ti.cast(dt_s, ti.f64)
                    * ti.cast(0.5 * (nu_current + nu_neighbor), ti.f64)
                    * ti.cast(patch_area, ti.f64)
                    / ti.cast(center_distance, ti.f64)
                )

            if component_coordinate > 0:
                mi = i - 1 if component == 0 else i
                mj = j - 1 if component == 1 else j
                mk = k - 1 if component == 2 else k
                mni = ni - 1 if component == 0 else ni
                mnj = nj - 1 if component == 1 else nj
                mnk = nk - 1 if component == 2 else nk
                minus_component_width = self.cell_width_x_m[mi]
                if component == 1:
                    minus_component_width = self.cell_width_y_m[mj]
                elif component == 2:
                    minus_component_width = self.cell_width_z_m[mk]
                minus_patch_canonical_wall = 0
                if (
                    self._momentum_canonical_wall_target_component_mask(
                        mni,
                        mnj,
                        mnk,
                        axis_index,
                        molecular_only,
                    )
                    != 0
                ):
                    minus_patch_canonical_wall = 1
                if ti.static(molecular_only):
                    if self.laminar_canonical_fluid_patch_exchange[None] != 0:
                        minus_patch_canonical_wall = 0
                if (
                    self.obstacle[mi, mj, mk] == 0
                    and self.obstacle[mni, mnj, mnk] == 0
                    and minus_patch_canonical_wall == 0
                ):
                    nu_current = self._momentum_kinematic_viscosity_at(mi, mj, mk, molecular_nu_m2_s, molecular_only)
                    nu_neighbor = self._momentum_kinematic_viscosity_at(mni, mnj, mnk, molecular_nu_m2_s, molecular_only)
                    patch_area = 0.5 * minus_component_width * other_width
                    coefficient += (
                        ti.cast(dt_s, ti.f64)
                        * ti.cast(0.5 * (nu_current + nu_neighbor), ti.f64)
                        * ti.cast(patch_area, ti.f64)
                        / ti.cast(center_distance, ti.f64)
                    )
        return coefficient

    @ti.func
    def _sst_momentum_normal_boundary_face_terms(
        self,
        i,
        j,
        k,
        component,
        side,
        dt_s,
        molecular_nu_m2_s,
        neighbor_in_domain,
        neighbor_row_kind,
        neighbor_row_target,
        pressure_outlet_zmin,
        velocity_inlet_zmax_mode,
        molecular_only: ti.template() = False,
    ):
        """Return exact-identity normal-MAC boundary coefficient and RHS."""

        integrated_diagonal = ti.cast(0.0, ti.f64)
        integrated_rhs = ti.cast(0.0, ti.f64)
        apply_boundary = 0
        target = ti.cast(0.0, ti.f64)
        si = i
        sj = j
        sk = k

        if neighbor_in_domain:
            if neighbor_row_kind != 1:
                apply_boundary = 1
                target = neighbor_row_target
                if side < 0:
                    if component == 0:
                        si -= 1
                    elif component == 1:
                        sj -= 1
                    else:
                        sk -= 1
        elif side > 0:
            external = self._physical_exterior_normal_contract(
                self.velocity, component, 1, i, j, k,
                pressure_outlet_zmin, velocity_inlet_zmax_mode,
            )
            if external[0] != 0:
                apply_boundary = 1
                target = ti.cast(external[1], ti.f64)

        if apply_boundary != 0:
            nu_cell = self._momentum_kinematic_viscosity_at(si, sj, sk, molecular_nu_m2_s, molecular_only)
            area = self.cell_width_y_m[sj] * self.cell_width_z_m[sk]
            distance = self.cell_width_x_m[si]
            if component == 1:
                area = self.cell_width_x_m[si] * self.cell_width_z_m[sk]
                distance = self.cell_width_y_m[sj]
            elif component == 2:
                area = self.cell_width_x_m[si] * self.cell_width_y_m[sj]
                distance = self.cell_width_z_m[sk]
            integrated_diagonal = (
                ti.cast(dt_s, ti.f64)
                * ti.cast(nu_cell, ti.f64)
                * ti.cast(area, ti.f64)
                / ti.cast(distance, ti.f64)
            )
            integrated_rhs = integrated_diagonal * target
        return ti.Vector([integrated_diagonal, integrated_rhs])

    @ti.func
    def _sst_momentum_transverse_boundary_face_terms(
        self,
        i,
        j,
        k,
        component,
        axis_index,
        side,
        dt_s,
        molecular_nu_m2_s,
        external_no_slip,
        neighbor_row_kind,
        neighbor_row_target,
        molecular_only: ti.template() = False,
    ):
        """Integrate transverse Dirichlet terms over both MAC half patches."""

        integrated_diagonal = ti.cast(0.0, ti.f64)
        integrated_rhs = ti.cast(0.0, ti.f64)
        component_coordinate = i
        if component == 1:
            component_coordinate = j
        elif component == 2:
            component_coordinate = k

        for patch_code in ti.static(range(2)):
            patch_valid = 1
            pi = i
            pj = j
            pk = k
            if ti.static(patch_code == 1):
                if component_coordinate <= 0:
                    patch_valid = 0
                elif component == 0:
                    pi -= 1
                elif component == 1:
                    pj -= 1
                else:
                    pk -= 1

            if patch_valid != 0 and self.obstacle[pi, pj, pk] == 0:
                component_width = self.cell_width_x_m[pi]
                other_width = self.cell_width_z_m[pk]
                if component == 1:
                    component_width = self.cell_width_y_m[pj]
                elif component == 2:
                    component_width = self.cell_width_z_m[pk]
                if axis_index == 0:
                    if component == 2:
                        other_width = self.cell_width_y_m[pj]
                elif axis_index == 1:
                    if component == 2:
                        other_width = self.cell_width_x_m[pi]
                else:
                    other_width = self.cell_width_y_m[pj]
                    if component == 1:
                        other_width = self.cell_width_x_m[pi]
                patch_area = 0.5 * component_width * other_width

                qi = pi
                qj = pj
                qk = pk
                if axis_index == 0:
                    qi += side
                elif axis_index == 1:
                    qj += side
                else:
                    qk += side
                neighbor_scalar_in_domain = (
                    0 <= qi < self.nx
                    and 0 <= qj < self.ny
                    and 0 <= qk < self.nz
                )
                nu_current = self._momentum_kinematic_viscosity_at(pi, pj, pk, molecular_nu_m2_s, molecular_only)

                patch_coefficient = ti.cast(0.0, ti.f64)
                patch_target = ti.cast(0.0, ti.f64)
                patch_is_dirichlet = 0
                canonical_wall_mask = 0
                owner_i, owner_j, owner_k = pi, pj, pk
                if side > 0:
                    owner_i, owner_j, owner_k = qi, qj, qk
                if ti.static(molecular_only):
                    # With the fluid-patch exchange only a face to an obstacle
                    # cell closes as a canonical wall; a fluid-fluid patch keeps
                    # the neighbour row (shared edge or identity) below.
                    if neighbor_scalar_in_domain and (
                        self.laminar_canonical_fluid_patch_exchange[None] == 0
                        or self.obstacle[qi, qj, qk] != 0
                    ):
                        canonical_wall_mask = (
                            self._canonical_hard_face_target_component_mask(
                                owner_i, owner_j, owner_k, axis_index,
                            )
                        )
                if canonical_wall_mask != 0:
                    # A laminar canonical face blocks both half-patch exchange
                    # paths. Only its owned components prescribe wall traction;
                    # an unowned tangent retains the established zero-flux rule.
                    if (canonical_wall_mask & (1 << component)) != 0:
                        wall_distance = 0.5 * self.cell_width_x_m[pi]
                        if axis_index == 1:
                            wall_distance = 0.5 * self.cell_width_y_m[pj]
                        elif axis_index == 2:
                            wall_distance = 0.5 * self.cell_width_z_m[pk]
                        patch_coefficient = (
                            ti.cast(dt_s, ti.f64)
                            * ti.cast(nu_current, ti.f64)
                            * ti.cast(patch_area, ti.f64)
                            / ti.cast(wall_distance, ti.f64)
                        )
                        patch_target = ti.cast(
                            self.velocity_dirichlet_boundary_value_mps[
                                owner_i, owner_j, owner_k
                            ][component],
                            ti.f64,
                        )
                        patch_is_dirichlet = 1
                elif (
                    neighbor_scalar_in_domain
                    and self.obstacle[qi, qj, qk] == 0
                ):
                    if neighbor_row_kind != 1:
                        center_distance = self.center_distance_x_m[
                            pi if side < 0 else qi
                        ]
                        if axis_index == 1:
                            center_distance = self.center_distance_y_m[
                                pj if side < 0 else qj
                            ]
                        elif axis_index == 2:
                            center_distance = self.center_distance_z_m[
                                pk if side < 0 else qk
                            ]
                        nu_neighbor = self._momentum_kinematic_viscosity_at(qi, qj, qk, molecular_nu_m2_s, molecular_only)
                        patch_coefficient = (
                            ti.cast(dt_s, ti.f64)
                            * ti.cast(0.5 * (nu_current + nu_neighbor), ti.f64)
                            * ti.cast(patch_area, ti.f64)
                            / ti.cast(center_distance, ti.f64)
                        )
                        patch_target = neighbor_row_target
                        patch_is_dirichlet = 1
                elif neighbor_scalar_in_domain:
                    wall_distance = 0.5 * self.cell_width_x_m[pi]
                    if axis_index == 1:
                        wall_distance = 0.5 * self.cell_width_y_m[pj]
                    elif axis_index == 2:
                        wall_distance = 0.5 * self.cell_width_z_m[pk]
                    patch_coefficient = (
                        ti.cast(dt_s, ti.f64)
                        * ti.cast(nu_current, ti.f64)
                        * ti.cast(patch_area, ti.f64)
                        / ti.cast(wall_distance, ti.f64)
                    )
                    owner_i = pi
                    owner_j = pj
                    owner_k = pk
                    if side > 0:
                        owner_i = qi
                        owner_j = qj
                        owner_k = qk
                    component_bit = 1 << component
                    exact_wall_component = (
                        self.sst_obstacle_interface_wall_target_component_mask[
                            owner_i, owner_j, owner_k
                        ][axis_index]
                        & component_bit
                    ) != 0
                    if exact_wall_component:
                        patch_target = ti.cast(
                            self.velocity_dirichlet_boundary_value_mps[
                                owner_i, owner_j, owner_k
                            ][component],
                            ti.f64,
                        )
                    patch_is_dirichlet = 1
                else:
                    side_index = 0 if side < 0 else 1
                    exact_mask = 0
                    exact_target = ti.cast(0.0, ti.f64)
                    if axis_index == 0:
                        exact_mask = (
                            self.external_velocity_boundary_x_face_active_component_mask[
                                side_index, pj, pk
                            ]
                            & (1 << component)
                        )
                        exact_target = ti.cast(
                            self.external_velocity_boundary_x_face_value_mps[
                                side_index, pj, pk
                            ][component],
                            ti.f64,
                        )
                    elif axis_index == 1:
                        exact_mask = (
                            self.external_velocity_boundary_y_face_active_component_mask[
                                side_index, pi, pk
                            ]
                            & (1 << component)
                        )
                        exact_target = ti.cast(
                            self.external_velocity_boundary_y_face_value_mps[
                                side_index, pi, pk
                            ][component],
                            ti.f64,
                        )
                    else:
                        exact_mask = (
                            self.external_velocity_boundary_z_face_active_component_mask[
                                side_index, pi, pj
                            ]
                            & (1 << component)
                        )
                        exact_target = ti.cast(
                            self.external_velocity_boundary_z_face_value_mps[
                                side_index, pi, pj
                            ][component],
                            ti.f64,
                        )
                    if exact_mask != 0 or external_no_slip != 0:
                        wall_distance = 0.5 * self.cell_width_x_m[pi]
                        if axis_index == 1:
                            wall_distance = 0.5 * self.cell_width_y_m[pj]
                        elif axis_index == 2:
                            wall_distance = 0.5 * self.cell_width_z_m[pk]
                        patch_coefficient = (
                            ti.cast(dt_s, ti.f64)
                            * ti.cast(nu_current, ti.f64)
                            * ti.cast(patch_area, ti.f64)
                            / ti.cast(wall_distance, ti.f64)
                        )
                        patch_target = (
                            exact_target if exact_mask != 0 else 0.0
                        )
                        patch_is_dirichlet = 1

                if ti.static(not molecular_only):
                    ti.static_assert(False, "turbulence models are not supported: the fluid solver is laminar")

                if patch_is_dirichlet != 0:
                    integrated_diagonal += patch_coefficient
                    integrated_rhs += patch_coefficient * patch_target

        return ti.Vector([integrated_diagonal, integrated_rhs])

    @ti.kernel
    def _laminar_momentum_diffusion_kernel(
        self,
        dt_s: ti.f32,
        molecular_nu_m2_s: ti.f32,
        no_slip_xmin: ti.i32,
        no_slip_xmax: ti.i32,
        no_slip_ymin: ti.i32,
        no_slip_ymax: ti.i32,
        no_slip_zmin: ti.i32,
        no_slip_zmax: ti.i32,
        pressure_outlet_zmin: ti.i32,
        velocity_inlet_zmax_mode: ti.i32,
    ):
        for i, j, k in self.velocity:
            self.velocity[i, j, k] = self._laminar_momentum_diffusion_at(
                i, j, k, self.velocity_prev[i, j, k], dt_s, molecular_nu_m2_s,
                ti.Vector([no_slip_xmin, no_slip_xmax, no_slip_ymin, no_slip_ymax, no_slip_zmin, no_slip_zmax]),
                pressure_outlet_zmin, velocity_inlet_zmax_mode,
            )
