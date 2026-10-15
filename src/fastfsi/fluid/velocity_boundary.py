"""Velocity boundary conditions: the Dirichlet face classification and application, inlet and exterior faces (mixin)."""

import taichi as ti


class VelocityBoundaryConditions:
    """Mixin of IncompressibleFluidSolver; not used on its own.  Contract (solver attributes):
      reads:  _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMERS,
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES,
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_PROJECTION_DEPENDENCIES,
    _velocity_dirichlet_component_registry_consumer_capabilities,
    _velocity_dirichlet_component_registry_consumer_generations,
    external_velocity_boundary_x_face_active_component_mask, external_velocity_boundary_x_face_value_mps,
    external_velocity_boundary_y_face_active_component_mask, external_velocity_boundary_y_face_value_mps,
    external_velocity_boundary_z_face_active_component_mask, external_velocity_boundary_z_face_value_mps, nx, ny,
    nz, obstacle, report_external_exact_zmax_normal_row_count, report_velocity_dirichlet_boundary_cells,
    report_velocity_dirichlet_boundary_delta_max, report_velocity_dirichlet_boundary_delta_sum,
    report_velocity_dirichlet_boundary_momentum_delta_n_s,
    report_velocity_dirichlet_hard_fixed_component_face_count, report_velocity_dirichlet_hard_fixed_row_count, rho,
    velocity, velocity_dirichlet_boundary_active, velocity_dirichlet_boundary_active_component_mask,
    velocity_dirichlet_boundary_authority, velocity_dirichlet_boundary_authority_code_device,
    velocity_dirichlet_boundary_component_enforcement_weight,
    velocity_dirichlet_boundary_external_exact_component_mask,
    velocity_dirichlet_boundary_hard_fixed_component_mask, velocity_dirichlet_boundary_owned_component_mask,
    velocity_dirichlet_boundary_owned_row, velocity_dirichlet_boundary_value_mps,
    velocity_dirichlet_component_registry_generation, velocity_dirichlet_component_registry_sealed,
    velocity_dirichlet_face_symmetric, velocity_dirichlet_pressure_hard_fixed_component_mask
      writes: _velocity_dirichlet_component_registry_consumer_capabilities,
    _velocity_dirichlet_component_registry_consumer_generations, velocity_dirichlet_boundary_authority,
    velocity_dirichlet_component_registry_generation, velocity_dirichlet_component_registry_sealed"""

    @ti.func
    def _velocity_dirichlet_pressure_barrier(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
    ):
        # Scalar protection predicate for cleanup/conversion policies only.
        # Pressure connectivity itself is orientation-specific and must use
        # _pressure_cells_connected(); a row with one hard component can still
        # communicate through either tangential component.
        return (
            self.velocity_dirichlet_pressure_hard_fixed_component_mask[i, j, k]
            & 7
        ) != 0

    @classmethod
    def _validated_velocity_dirichlet_boundary_authority(
        cls,
        authority: object,
    ) -> str:
        if not isinstance(authority, str):
            raise TypeError(
                "velocity Dirichlet boundary authority must be 'legacy' or "
                "'canonical'"
            )
        if authority not in cls._VELOCITY_DIRICHLET_BOUNDARY_AUTHORITIES:
            raise ValueError(
                "velocity Dirichlet boundary authority must be 'legacy' or "
                f"'canonical', got {authority!r}"
            )
        return authority

    def _velocity_dirichlet_boundary_authority_code(self) -> int:
        return (
            1
            if self._validated_velocity_dirichlet_boundary_authority(
                self.velocity_dirichlet_boundary_authority
            )
            == "canonical"
            else 0
        )

    def set_velocity_dirichlet_boundary_authority(self, authority: object) -> None:
        authority_name = self._validated_velocity_dirichlet_boundary_authority(
            authority
        )
        if (
            authority_name == "canonical"
            and int(self.velocity_dirichlet_face_symmetric) != 0
        ):
            raise ValueError(
                "canonical velocity Dirichlet boundary authority forbids the "
                "legacy velocity_dirichlet_face_symmetric compatibility path"
            )
        authority_code = 1 if authority_name == "canonical" else 0
        if authority_name == self.velocity_dirichlet_boundary_authority:
            self.velocity_dirichlet_boundary_authority_code_device[None] = (
                authority_code
            )
            return
        # Authority participates in both pressure connectivity and the exact
        # interface operator graph. Invalidate both before changing either the
        # host identity or its device mirror.
        self._invalidate_pressure_reachability()
        self._invalidate_velocity_dirichlet_component_registry()
        self.velocity_dirichlet_boundary_authority = authority_name
        self.velocity_dirichlet_boundary_authority_code_device[None] = authority_code

    def _invalidate_velocity_dirichlet_component_registry(self) -> int:
        # Retire host authority before the first fallible device operation.
        generation = int(self.velocity_dirichlet_component_registry_generation) + 1
        self.velocity_dirichlet_component_registry_generation = generation
        self.velocity_dirichlet_component_registry_sealed = False
        self._velocity_dirichlet_component_registry_consumer_generations = {}
        self._velocity_dirichlet_component_registry_consumer_capabilities = {}
        self._invalidate_pressure_nullspace_component_graph()
        try:
            object.__getattribute__(
                self,
                "canonical_exact_hard_face_component_mask",
            )
        except AttributeError:
            # Host-only contract probes construct an intentionally incomplete
            # owner; production solvers always own and clear this device mask.
            pass
        else:
            self._clear_canonical_exact_hard_face_component_mask_kernel()
        return generation

    def _register_velocity_dirichlet_component_registry_consumer_generation(
        self,
        consumer: str,
        *,
        capability: object,
    ) -> None:
        if consumer not in self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMERS:
            raise ValueError(
                f"unknown canonical velocity boundary consumer {consumer!r}"
            )
        expected_capability = (
            self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES.get(
                consumer
            )
        )
        if expected_capability is None or capability is not expected_capability:
            raise RuntimeError(
                "canonical velocity boundary consumer is not backed by its "
                f"dedicated migrated prepare capability: {consumer!r}"
            )
        generation_value = int(
            self.velocity_dirichlet_component_registry_generation
        )
        self._velocity_dirichlet_component_registry_consumer_generations = {
            **self._velocity_dirichlet_component_registry_consumer_generations,
            consumer: int(generation_value),
        }
        self._velocity_dirichlet_component_registry_consumer_capabilities = {
            **self._velocity_dirichlet_component_registry_consumer_capabilities,
            consumer: capability,
        }

    def _velocity_dirichlet_component_registry_generation_errors(
        self,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        registered = self._velocity_dirichlet_component_registry_consumer_generations
        registered_capabilities = getattr(
            self,
            "_velocity_dirichlet_component_registry_consumer_capabilities",
            {},
        )
        required = self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMERS
        missing = sorted(required - set(registered))
        unexpected = sorted(set(registered) - required)
        generation = int(self.velocity_dirichlet_component_registry_generation)
        mismatched = sorted(
            consumer
            for consumer in required & set(registered)
            if int(registered[consumer]) != generation
        )
        invalid_capabilities = sorted(
            consumer
            for consumer in required & set(registered)
            if (
                self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES.get(
                    consumer
                )
                is None
                or registered_capabilities.get(consumer)
                is not self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES.get(
                    consumer
                )
            )
        )
        return missing, unexpected, mismatched, invalid_capabilities

    def _require_host_macro_step_restore_complete(self) -> None:
        if getattr(self, "_host_macro_step_restore_pending", False):
            raise RuntimeError("host macro-step restore is incomplete")

    def _require_marker_compatibility_closure_complete(self) -> None:
        VelocityBoundaryConditions._require_host_macro_step_restore_complete(self)
        if bool(getattr(self, "_marker_compatibility_closure_pending", False)):
            raise RuntimeError(
                "interface marker compatibility closure is pending; "
                "intermediate band geometry cannot be sealed or consumed"
            )

    def seal_velocity_dirichlet_component_registry(self) -> None:
        self._require_marker_compatibility_closure_complete()
        authority = self._validated_velocity_dirichlet_boundary_authority(
            self.velocity_dirichlet_boundary_authority
        )
        if authority != "canonical":
            raise RuntimeError(
                "only canonical velocity Dirichlet boundary authority can seal "
                "the component registry"
            )
        if int(self.velocity_dirichlet_face_symmetric) != 0:
            raise RuntimeError(
                "canonical velocity Dirichlet boundary authority forbids the "
                "legacy velocity_dirichlet_face_symmetric compatibility path"
            )
        missing, unexpected, mismatched, invalid_capabilities = (
            self._velocity_dirichlet_component_registry_generation_errors()
        )
        if missing or unexpected or mismatched or invalid_capabilities:
            self.velocity_dirichlet_component_registry_sealed = False
            raise RuntimeError(
                "canonical velocity Dirichlet component registry consumer "
                "generations are incomplete or mismatched: "
                f"missing={missing}, unexpected={unexpected}, "
                f"mismatched={mismatched}, invalid_capabilities="
                f"{invalid_capabilities}, generation="
                f"{int(self.velocity_dirichlet_component_registry_generation)}"
            )
        self._refresh_canonical_exact_hard_face_component_mask()
        self.velocity_dirichlet_component_registry_sealed = True

    def _require_velocity_dirichlet_component_registry_sealed(self) -> None:
        self._require_marker_compatibility_closure_complete()
        authority = self._validated_velocity_dirichlet_boundary_authority(
            self.velocity_dirichlet_boundary_authority
        )
        if authority == "legacy":
            return
        if int(self.velocity_dirichlet_face_symmetric) != 0:
            raise RuntimeError(
                "canonical velocity Dirichlet boundary authority forbids the "
                "legacy velocity_dirichlet_face_symmetric compatibility path"
            )
        missing, unexpected, mismatched, invalid_capabilities = (
            self._velocity_dirichlet_component_registry_generation_errors()
        )
        if (
            not bool(self.velocity_dirichlet_component_registry_sealed)
            or missing
            or unexpected
            or mismatched
            or invalid_capabilities
        ):
            raise RuntimeError(
                "canonical velocity Dirichlet component registry is not sealed "
                "for its current generation"
            )

    def _require_canonical_velocity_dirichlet_boundary_authority(self) -> None:
        authority = self._validated_velocity_dirichlet_boundary_authority(
            self.velocity_dirichlet_boundary_authority
        )
        if authority != "canonical":
            raise RuntimeError(
                "legacy velocity Dirichlet boundary authority forbids this "
                "canonical boundary writer"
            )

    def prepare_velocity_dirichlet_component_registry_apply(self) -> None:
        """Register the migrated componentwise boundary-application consumer."""

        if self.velocity_dirichlet_boundary_authority != "canonical":
            return
        self._register_velocity_dirichlet_component_registry_consumer_generation(
            "apply",
            capability=self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES[
                "apply"
            ],
        )

    def prepare_velocity_dirichlet_component_registry_reference(self) -> None:
        """Register the exact eight-field canonical reference consumer."""

        if self.velocity_dirichlet_boundary_authority != "canonical":
            return
        self._register_velocity_dirichlet_component_registry_consumer_generation(
            "reference",
            capability=self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES[
                "reference"
            ],
        )

    def prepare_velocity_dirichlet_component_registry_snapshot(self) -> None:
        """Register the exact schema-6 canonical snapshot consumer."""

        if self.velocity_dirichlet_boundary_authority != "canonical":
            return
        self._register_velocity_dirichlet_component_registry_consumer_generation(
            "snapshot",
            capability=self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES[
                "snapshot"
            ],
        )

    def prepare_velocity_dirichlet_component_registry_fv_operator(self) -> None:
        """Register the migrated fine-grid FV pressure-operator consumer."""

        if self.velocity_dirichlet_boundary_authority != "canonical":
            return
        self._register_velocity_dirichlet_component_registry_consumer_generation(
            "fv_operator",
            capability=self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES[
                "fv_operator"
            ],
        )

    def prepare_velocity_dirichlet_component_registry_gradient(self) -> None:
        """Register the migrated componentwise pressure-gradient consumer."""

        if self.velocity_dirichlet_boundary_authority != "canonical":
            return
        self._register_velocity_dirichlet_component_registry_consumer_generation(
            "gradient",
            capability=self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES[
                "gradient"
            ],
        )

    def prepare_velocity_dirichlet_component_registry_divergence(self) -> None:
        """Register the migrated axis-local divergence consumer."""

        if self.velocity_dirichlet_boundary_authority != "canonical":
            return
        self._register_velocity_dirichlet_component_registry_consumer_generation(
            "divergence",
            capability=self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES[
                "divergence"
            ],
        )

    def prepare_velocity_dirichlet_component_registry_reachability(self) -> None:
        """Build and register the axis-local pressure-connectivity consumer."""

        if self.velocity_dirichlet_boundary_authority != "canonical":
            return
        # The flood consumes this derived component-face mask.  Rebuild it
        # before certifying the consumer and invalidate every older flood result:
        # neither a checksum nor ``last_*_valid`` certifies this generation.
        self._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask()
        self._invalidate_pressure_reachability()
        self._register_velocity_dirichlet_component_registry_consumer_generation(
            "reachability",
            capability=self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES[
                "reachability"
            ],
        )

    def prepare_velocity_dirichlet_component_registry_projection(self) -> None:
        """Register projection only after every physical child is current."""

        if self.velocity_dirichlet_boundary_authority != "canonical":
            return
        generation = int(self.velocity_dirichlet_component_registry_generation)
        missing: list[str] = []
        mismatched: list[str] = []
        invalid_capabilities: list[str] = []
        capabilities = self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES
        for consumer in sorted(
            self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_PROJECTION_DEPENDENCIES
        ):
            registered_generation = (
                self._velocity_dirichlet_component_registry_consumer_generations.get(
                    consumer
                )
            )
            if registered_generation is None:
                missing.append(consumer)
            elif int(registered_generation) != generation:
                mismatched.append(consumer)
            if (
                self._velocity_dirichlet_component_registry_consumer_capabilities.get(
                    consumer
                )
                is not capabilities[consumer]
            ):
                invalid_capabilities.append(consumer)
        if missing or mismatched or invalid_capabilities:
            raise RuntimeError(
                "canonical projection dependencies are incomplete or stale: "
                f"missing={missing}, mismatched={mismatched}, "
                f"invalid_capabilities={invalid_capabilities}, "
                f"generation={generation}"
            )
        self._register_velocity_dirichlet_component_registry_consumer_generation(
            "projection",
            capability=capabilities["projection"],
        )

    def prepare_and_seal_velocity_dirichlet_component_registry(self) -> None:
        """Prepare every canonical consumer and seal the current generation."""

        self._require_marker_compatibility_closure_complete()
        self._require_canonical_velocity_dirichlet_boundary_authority()
        self._invalidate_pressure_reachability()
        self.prepare_velocity_dirichlet_component_registry_apply()
        self.prepare_velocity_dirichlet_component_registry_divergence()
        self.prepare_velocity_dirichlet_component_registry_reachability()
        self.prepare_velocity_dirichlet_component_registry_fv_operator()
        self.prepare_velocity_dirichlet_component_registry_gradient()
        self.prepare_velocity_dirichlet_component_registry_multigrid()
        self.prepare_velocity_dirichlet_component_registry_projection()
        self.prepare_no_slip_component_face_valid_mask()
        self.prepare_velocity_dirichlet_component_registry_reference()
        self.prepare_velocity_dirichlet_component_registry_snapshot()
        self.seal_velocity_dirichlet_component_registry()
        self._require_velocity_dirichlet_component_registry_sealed()

    @ti.func
    def _velocity_dirichlet_pressure_effective_component_mask_at(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        canonical_authority: ti.template(),
    ) -> ti.i32:
        mask = 0
        if ti.static(canonical_authority == 1):
            if self.obstacle[i, j, k] == 0:
                direct_mask = (
                    self.velocity_dirichlet_boundary_active_component_mask[i, j, k]
                    & self.velocity_dirichlet_boundary_hard_fixed_component_mask[
                        i, j, k
                    ]
                    & 7
                )
                owned_mask = (
                    self.velocity_dirichlet_boundary_owned_component_mask[i, j, k]
                    & 7
                )
                external_mask = (
                    self.velocity_dirichlet_boundary_external_exact_component_mask[
                        i, j, k
                    ]
                    & 7
                )
                # Keep the full componentwise hard clamp for velocity, but let
                # an external-exact row close only its interface-owned components or
                # registered external normals.  Ordinary canonical hard rows
                # without external provenance retain their direct pressure
                # topology; mixed rows preserve ownership component by component.
                mask = direct_mask
                if external_mask != 0:
                    mask = direct_mask & (owned_mask | external_mask)
        else:
            if (
                self.obstacle[i, j, k] == 0
                and self.velocity_dirichlet_boundary_active[i, j, k] != 0
            ):
                direct_mask = (
                    self.velocity_dirichlet_boundary_hard_fixed_component_mask[i, j, k]
                    & 7
                )
                external_mask = (
                    self.velocity_dirichlet_boundary_external_exact_component_mask[
                        i, j, k
                    ]
                    & 7
                )
                # Preserve the legacy scalar-row pressure topology byte for
                # byte. Canonical rows use the component-provenance branch above.
                if (
                    self.velocity_dirichlet_boundary_owned_row[i, j, k] == 0
                    and external_mask != 0
                ):
                    mask = direct_mask & external_mask
                else:
                    mask = direct_mask
        return mask

    @ti.kernel
    def _refresh_velocity_dirichlet_pressure_hard_fixed_component_mask_kernel(
        self,
        face_symmetric: ti.i32,
        canonical_authority: ti.template(),
    ):
        self.report_velocity_dirichlet_hard_fixed_row_count[None] = 0
        self.report_velocity_dirichlet_hard_fixed_component_face_count[None] = 0
        for i, j, k in self.velocity_dirichlet_pressure_hard_fixed_component_mask:
            mask = 0
            own_active = False
            if ti.static(canonical_authority == 1):
                own_active = (
                    self.obstacle[i, j, k] == 0
                    and (
                        self.velocity_dirichlet_boundary_active_component_mask[i, j, k]
                        & 7
                    )
                    != 0
                )
            else:
                own_active = (
                    self.obstacle[i, j, k] == 0
                    and self.velocity_dirichlet_boundary_active[i, j, k] != 0
                )
            if own_active:
                mask = self._velocity_dirichlet_pressure_effective_component_mask_at(
                    i, j, k, canonical_authority
                )
            elif (
                ti.static(canonical_authority == 0)
                and self.obstacle[i, j, k] == 0
                and face_symmetric == 1
            ):
                # Velocity components are stored on the orientation-specific
                # negative face of this cell.  Mirror only the matching hard
                # pressure-effective component from the predecessor row; the
                # destination may remain pressure-correctable in its other two
                # components.  In particular, external exact tangential
                # velocity bits must not become mirrored pressure closures.
                if (
                    i > 0
                    and (
                        self._velocity_dirichlet_pressure_effective_component_mask_at(
                            i - 1, j, k, canonical_authority
                        )
                        & 1
                    )
                    != 0
                ):
                    mask |= 1
                if (
                    j > 0
                    and (
                        self._velocity_dirichlet_pressure_effective_component_mask_at(
                            i, j - 1, k, canonical_authority
                        )
                        & 2
                    )
                    != 0
                ):
                    mask |= 2
                if (
                    k > 0
                    and (
                        self._velocity_dirichlet_pressure_effective_component_mask_at(
                            i, j, k - 1, canonical_authority
                        )
                        & 4
                    )
                    != 0
                ):
                    mask |= 4
            self.velocity_dirichlet_pressure_hard_fixed_component_mask[i, j, k] = mask
            if mask != 0:
                ti.atomic_add(
                    self.report_velocity_dirichlet_hard_fixed_row_count[None],
                    1,
                )
                for axis in ti.static(range(3)):
                    if (mask & (1 << axis)) != 0:
                        ti.atomic_add(
                            self.report_velocity_dirichlet_hard_fixed_component_face_count[
                                None
                            ],
                            1,
                        )

    def _refresh_velocity_dirichlet_pressure_hard_fixed_component_mask(
        self,
    ) -> tuple[int, int]:
        authority = self._validated_velocity_dirichlet_boundary_authority(
            self.velocity_dirichlet_boundary_authority
        )
        self._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask_kernel(
            int(self.velocity_dirichlet_face_symmetric),
            1 if authority == "canonical" else 0,
        )
        return (
            int(self.report_velocity_dirichlet_hard_fixed_row_count[None]),
            int(
                self.report_velocity_dirichlet_hard_fixed_component_face_count[None]
            ),
        )

    def _require_face_symmetric_owned_soft_compatibility(self) -> None:
        if int(self.velocity_dirichlet_face_symmetric) != 1:
            return
        raise NotImplementedError("velocity_dirichlet_face_symmetric must be 0")

    @ti.kernel
    def _apply_canonical_velocity_dirichlet_boundary_rows_kernel(
        self,
        read_report: ti.i32,
        preserve_projected_rows: ti.i32,
    ):
        self.report_velocity_dirichlet_boundary_cells[None] = 0
        self.report_velocity_dirichlet_boundary_delta_sum[None] = 0.0
        self.report_velocity_dirichlet_boundary_delta_max[None] = 0.0
        self.report_velocity_dirichlet_boundary_momentum_delta_n_s[None] = ti.Vector(
            [0.0, 0.0, 0.0]
        )
        for i, j, k in self.velocity:
            active_mask = (
                self.velocity_dirichlet_boundary_active_component_mask[i, j, k]
                & 7
            )
            if active_mask != 0:
                old_velocity = self.velocity[i, j, k]
                new_velocity = old_velocity
                target_velocity = self.velocity_dirichlet_boundary_value_mps[i, j, k]
                component_weight = (
                    self.velocity_dirichlet_boundary_component_enforcement_weight[
                        i, j, k
                    ]
                )
                hard_mask = (
                    self.velocity_dirichlet_boundary_hard_fixed_component_mask[
                        i, j, k
                    ]
                    & active_mask
                )
                for axis in ti.static(range(3)):
                    if (active_mask & (1 << axis)) != 0:
                        weight = ti.min(1.0, ti.max(0.0, component_weight[axis]))
                        if (hard_mask & (1 << axis)) != 0:
                            weight = 1.0
                        elif preserve_projected_rows != 0:
                            weight = 0.0
                        new_velocity[axis] = old_velocity[axis] + weight * (
                            target_velocity[axis] - old_velocity[axis]
                        )
                velocity_delta = new_velocity - old_velocity
                self.velocity[i, j, k] = new_velocity
                if read_report != 0:
                    delta = velocity_delta.norm()
                    momentum_delta = (
                        self.rho * velocity_delta * self._cell_volume_m3(i, j, k)
                    )
                    ti.atomic_add(
                        self.report_velocity_dirichlet_boundary_cells[None],
                        1,
                    )
                    ti.atomic_add(
                        self.report_velocity_dirichlet_boundary_delta_sum[None],
                        delta,
                    )
                    ti.atomic_max(
                        self.report_velocity_dirichlet_boundary_delta_max[None],
                        delta,
                    )
                    self._atomic_add_report_vector(
                        self.report_velocity_dirichlet_boundary_momentum_delta_n_s,
                        momentum_delta,
                    )

    def _apply_velocity_dirichlet_boundary_rows_dispatch(
        self,
        *,
        read_report: bool,
        preserve_projected_rows: bool,
    ) -> None:
        authority = self._validated_velocity_dirichlet_boundary_authority(
            self.velocity_dirichlet_boundary_authority
        )
        if authority == "canonical":
            if int(self.velocity_dirichlet_face_symmetric) != 0:
                raise RuntimeError(
                    "canonical velocity Dirichlet authority forbids the legacy "
                    "face_symmetric application path"
                )
            self._apply_canonical_velocity_dirichlet_boundary_rows_kernel(
                1 if read_report else 0,
                1 if preserve_projected_rows else 0,
            )
            return
        raise NotImplementedError("only the 'canonical' velocity boundary classification is supported")

    @ti.kernel
    def _count_external_exact_zmax_normal_rows_kernel(
        self,
    ):
        self.report_external_exact_zmax_normal_row_count[None] = 0
        for i, j in ti.ndrange(self.nx, self.ny):
            k = self.nz - 1
            if (
                self.obstacle[i, j, k] == 0
                and (
                    self.external_velocity_boundary_z_face_active_component_mask[
                        1, i, j
                    ]
                    & 4
                )
                != 0
            ):
                ti.atomic_add(
                    self.report_external_exact_zmax_normal_row_count[None],
                    1,
                )

    def _resolve_velocity_inlet_zmax_topology_mode(
        self,
        velocity_inlet_zmax: bool | None,
        canonical_authority: int,
    ) -> int:
        # The exact external boundary registry is the source of truth.  None
        # derives the per-face topology from that registry; True is the legacy
        # whole-zmax prescribed-flux mode.  Explicit False is an assertion that
        # no exact z-normal rows exist and therefore fails on a contradiction.
        if velocity_inlet_zmax is None:
            return 2
        if bool(velocity_inlet_zmax):
            return 1
        self._count_external_exact_zmax_normal_rows_kernel()
        exact_row_count = int(
            self.report_external_exact_zmax_normal_row_count[None]
        )
        if exact_row_count != 0:
            raise ValueError(
                "velocity_inlet_zmax=False conflicts with "
                f"{exact_row_count} exact zmax normal velocity boundary rows"
            )
        return 0

    def _resolve_physical_face_topology(
        self,
        *,
        pressure_outlet_zmin: bool,
        velocity_inlet_zmax: bool | None,
        wall_flag_codes: tuple[int, int, int, int, int, int],
    ) -> int:
        """Validate one physical-face write without retaining topology state."""
        if pressure_outlet_zmin and wall_flag_codes[4] != 0:
            raise ValueError(
                "no_slip_zmin=True conflicts with pressure_outlet_zmin=True"
            )
        if velocity_inlet_zmax and wall_flag_codes[5] != 0:
            raise ValueError(
                "no_slip_zmax=True conflicts with velocity_inlet_zmax=True"
            )
        return self._resolve_velocity_inlet_zmax_topology_mode(
            velocity_inlet_zmax,
            canonical_authority=self._velocity_dirichlet_boundary_authority_code(),
        )

    @ti.func
    def _physical_exterior_normal_contract(
        self,
        source: ti.template(),
        axis: ti.i32,
        side: ti.i32,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        pressure_outlet_zmin: ti.i32,
        velocity_inlet_zmax_mode: ti.i32,
    ):
        """Return [prescribed, absolute normal] for this call's stage source.

        Undeclared exterior faces are closed.  Exact normal data overrides
        the no-slip default; the only compact-row fallbacks are the explicit
        zmin pressure outlet and legacy whole-plane zmax inlet.  A zero-valued
        fallback remains extrapolated, whereas an exact zero is prescribed.
        This rule does not describe the relative Q of an interior moving wall.
        """

        prescribed = 1
        value = 0.0
        if self.obstacle[i, j, k] == 0:
            if axis == 0:
                if (
                    self.external_velocity_boundary_x_face_active_component_mask[
                        side, j, k
                    ] & 1
                ) != 0:
                    value = self.external_velocity_boundary_x_face_value_mps[
                        side, j, k
                    ].x
            elif axis == 1:
                if (
                    self.external_velocity_boundary_y_face_active_component_mask[
                        side, i, k
                    ] & 2
                ) != 0:
                    value = self.external_velocity_boundary_y_face_value_mps[
                        side, i, k
                    ].y
            else:
                if side == 0:
                    if (
                        self.external_velocity_boundary_z_face_active_component_mask[
                            side, i, j
                        ] & 4
                    ) != 0:
                        value = self.external_velocity_boundary_z_face_value_mps[
                            side, i, j
                        ].z
                    elif pressure_outlet_zmin != 0:
                        # This minimum compact row is a physical MAC face.
                        prescribed = 0
                        value = source[i, j, k].z
                elif velocity_inlet_zmax_mode != 0:
                    if (
                        self.external_velocity_boundary_z_face_active_component_mask[
                            side, i, j
                        ] & 4
                    ) != 0:
                        value = self.external_velocity_boundary_z_face_value_mps[
                            side, i, j
                        ].z
                    elif velocity_inlet_zmax_mode == 1:
                        # Only explicit True permits this legacy extrapolation.
                        prescribed = 0
                        value = source[i, j, k].z
        return ti.Vector([ti.cast(prescribed, ti.f32), value])

    @ti.func
    def _physical_exterior_normal_velocity(
        self,
        source: ti.template(),
        axis: ti.template(),
        side: ti.template(),
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        pressure_outlet_zmin: ti.i32,
        velocity_inlet_zmax_mode: ti.i32,
    ):
        return self._physical_exterior_normal_contract(
            source, axis, side, i, j, k,
            pressure_outlet_zmin, velocity_inlet_zmax_mode,
        )[1]

    @ti.func
    def _velocity_dirichlet_pressure_face_weight(
        self,
        velocity_dirichlet_boundary_active: ti.template(),
        velocity_dirichlet_boundary_projection_weight: ti.template(),
        velocity_dirichlet_pressure_hard_fixed_component_mask: ti.template(),
        canonical_active_component_mask: ti.template(),
        canonical_pressure_mobility: ti.template(),
        canonical_hard_fixed_component_mask: ti.template(),
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        component_axis: ti.i32,
        canonical_authority: ti.template(),
    ):
        """Return the one authoritative pressure mobility for a MAC face."""

        component_bit = 1 << component_axis
        face_weight = ti.cast(1.0, ti.f64)
        if ti.static(canonical_authority == 1):
            active_mask = canonical_active_component_mask[i, j, k] & 7
            if (active_mask & component_bit) != 0:
                pressure_hard_mask = (
                    velocity_dirichlet_pressure_hard_fixed_component_mask[i, j, k]
                    & 7
                )
                raw_hard_mask = canonical_hard_fixed_component_mask[i, j, k] & 7
                if (
                    pressure_hard_mask & component_bit
                ) != 0:
                    face_weight = ti.cast(0.0, ti.f64)
                elif (raw_hard_mask & component_bit) != 0:
                    # The exact composite projection is D(B(u-dt/rho Gp)).
                    # B clamps every raw-hard component after the pressure
                    # update, so that face has zero pressure mobility even when
                    # external-normal provenance deliberately omits it from the
                    # derived pressure-hard topology mask.
                    face_weight = ti.cast(0.0, ti.f64)
                else:
                    face_weight = ti.cast(
                        ti.min(
                            ti.max(
                                canonical_pressure_mobility[i, j, k][
                                    component_axis
                                ],
                                0.0,
                            ),
                            1.0,
                        ),
                        ti.f64,
                    )
        else:
            if (
                velocity_dirichlet_pressure_hard_fixed_component_mask[i, j, k]
                & component_bit
            ) != 0:
                face_weight = ti.cast(0.0, ti.f64)
            elif velocity_dirichlet_boundary_active[i, j, k] != 0:
                # Preserve the legacy scalar-row behavior exactly.
                face_weight = ti.cast(
                    ti.min(
                        ti.max(
                            velocity_dirichlet_boundary_projection_weight[i, j, k],
                            0.0,
                        ),
                        1.0,
                    ),
                    ti.f64,
                )
        return face_weight
