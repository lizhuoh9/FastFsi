"""CUDA-only finite copies of the original three host controllers.

No iteration cadence, fallback, restart, mathematical operator or CPU method
body is replaced. Scalar arithmetic executes in private f64 device kernels.
"""
import math
from math import sqrt
from .solver import CartesianFluidSolver, UNIFIED_PHYSICS, CG_PRECONDITIONER_CHOICES, _diagnostic_int
from simulation_core.fluids.shared_definitions import scalar_maximum


def solve_with_native_regions(owner, **arguments):
    from simulation_core.native_dispatch.regions import region_dispatch, region_methods
    original, _ = region_methods(_DevicePressureMethods)
    methods = dict(original)
    methods.update(_solve_pressure_poisson_fv_cg=solve_cg,
                   _continue_pressure_poisson_fv_bicgstab=continue_bicgstab)
    with region_dispatch(owner, enabled=True, mode='graph', host_methods=methods):
        return owner._solve_pressure_poisson_fv_cg(**arguments)


def _solve_pressure_poisson_fv_cg(
    self,
    *,
    iterations: int,
    rhs_scale: float,
    pressure_outlet_zmin: bool,
    tolerance: float,
    preconditioner: str = "auto",
    remove_nullspace_mean: bool = True,
    pressure_nullspace_component_count: int = 0,
    pressure_interface_matrix_active: bool = False,
    pressure_components_use_operator_graph: bool | None = None,
    report_requested: bool = True,
    pressure_velocity_nullspace_projector: object | None = None,
    closed_neumann_velocity_inlet_zmax_mode: int | None = None,
) -> None:
    if (UNIFIED_PHYSICS.execution.native_dispatch
            and getattr(self, '_k7_active_session', None) is None):
        from simulation_core.fluids.gpu_fluid_host import solve_with_native_regions
        return solve_with_native_regions(self,
            iterations=iterations,
            rhs_scale=rhs_scale,
            pressure_outlet_zmin=pressure_outlet_zmin,
            tolerance=tolerance,
            preconditioner=preconditioner,
            remove_nullspace_mean=remove_nullspace_mean,
            pressure_nullspace_component_count=pressure_nullspace_component_count,
            pressure_interface_matrix_active=pressure_interface_matrix_active,
            pressure_components_use_operator_graph=pressure_components_use_operator_graph,
            report_requested=report_requested,
            pressure_velocity_nullspace_projector=pressure_velocity_nullspace_projector,
            closed_neumann_velocity_inlet_zmax_mode=closed_neumann_velocity_inlet_zmax_mode)
    max_iters = max(1, int(iterations))
    relative_tolerance = scalar_maximum(0.0, float(tolerance))
    preconditioner_name = str(preconditioner)
    if preconditioner_name not in CG_PRECONDITIONER_CHOICES:
        raise ValueError(f"unsupported cg_preconditioner: {preconditioner!r}")
    outlet = 1 if pressure_outlet_zmin else 0
    self.last_cg_closed_neumann_physical_rhs_measured = False
    self.last_cg_closed_neumann_physical_rhs_mean = 0.0
    self.last_cg_closed_neumann_physical_rhs_integral = 0.0
    self.last_cg_closed_neumann_physical_component_budgets = ()
    self.last_cg_iterations = 0
    self.last_cg_initial_relative_residual = math.inf
    self.last_cg_relative_residual = math.inf
    self.last_cg_converged = False
    self.last_cg_breakdown = ""
    self.last_cg_host_residual_checks = 0
    self.last_cg_mean_projection_count = 0
    self.last_cg_unreached_set_mean_projection_count = 0
    self.last_cg_componentwise_mean_projection_count = 0
    self.last_cg_restart_count = 0
    self.last_cg_restart_count_measured = True
    self.last_cg_restart_policy = "periodic_exact_residual"
    self.last_cg_breakdown_dAd = 0.0
    self.last_cg_preconditioner_requested = preconditioner_name
    self.last_cg_preconditioner_effective = "not_applied"
    self.last_cg_rhs_nonzero = False
    self.last_cg_multigrid_apply_count = 0
    self.last_cg_multigrid_to_jacobi_fallback_count = 0
    self.last_cg_exact_residual_confirmation_count = 0
    self.last_cg_exact_relative_residual = math.inf
    self.last_cg_recursive_relative_residual = math.inf
    self.last_cg_pcg_iterations = 0
    self.last_cg_bicgstab_iterations = 0
    self.last_cg_multigrid_pre_smooth_iterations = 0
    self.last_cg_multigrid_coarse_smooth_iterations = 0
    self.last_cg_multigrid_post_smooth_iterations = 0
    self.last_cg_operator_apply_count = 0
    self.cg_breakdown_code[None] = 0
    self.cg_breakdown_dAd[None] = 0.0
    physical_outlet_component_count = (
        int(self._hibm_pressure_unreached_component_count)
        if bool(pressure_outlet_zmin)
        and int(self._hibm_pressure_unreached_count) > 0
        else 0
    )
    if pressure_interface_matrix_active:
        # Protect direct low-level callers too; interface validity cannot
        # depend on whether an outlet/nullspace component graph exists.
        self._preflight_pressure_interface_operator_storage(
            context="fv-cg",
        )
    operator_graph_requested = pressure_components_use_operator_graph
    if operator_graph_requested is None:
        # Compatibility for direct low-level outlet tests. Production
        # project() always supplies this decision explicitly.
        operator_graph_requested = bool(
            pressure_outlet_zmin
            and pressure_interface_matrix_active
        )
    operator_graph_requested = bool(operator_graph_requested)
    if operator_graph_requested and not pressure_interface_matrix_active:
        raise RuntimeError(
            "pressure operator-graph projection requires active interface "
            "matrix terms"
        )
    use_pressure_outlet_operator_graph = bool(
        operator_graph_requested
        and pressure_outlet_zmin
    )
    if use_pressure_outlet_operator_graph:
        expected_source_component_count = max(
            0,
            int(
                getattr(
                    self,
                    "last_hibm_pressure_unreached_component_raw_count",
                    0,
                )
            ),
            int(getattr(self, "_hibm_pressure_unreached_component_count", 0)),
        )
        graph_matches = bool(
            self._pressure_outlet_nullspace_graph_valid
            and self._pressure_outlet_nullspace_graph_context
            == "outlet-disconnected"
            and int(self._pressure_outlet_nullspace_source_component_count)
            == expected_source_component_count
            and int(self._pressure_outlet_operator_component_count)
            == int(self._pressure_outlet_nullspace_component_count)
        )
        if not graph_matches:
            _, outlet_component_count = (
                self._prepare_pressure_outlet_nullspace_component_graph()
            )
        else:
            outlet_component_count = int(
                self._pressure_outlet_operator_component_count
            )
    else:
        outlet_component_count = physical_outlet_component_count
    closed_component_count = (
        max(0, int(pressure_nullspace_component_count))
        if not bool(pressure_outlet_zmin)
        else 0
    )
    use_closed_operator_graph = bool(
        operator_graph_requested
        and not pressure_outlet_zmin
        and closed_component_count > 0
    )
    if use_closed_operator_graph:
        graph_matches = bool(
            self._pressure_outlet_nullspace_graph_valid
            and self._pressure_outlet_nullspace_graph_context
            == "closed-Neumann"
            and int(self._pressure_outlet_nullspace_component_count)
            == closed_component_count
            and int(self._pressure_outlet_nullspace_source_component_count) > 0
        )
        if not graph_matches:
            raise self._pressure_operator_graph_missing_or_stale_error(
                context="closed-Neumann-fv-cg",
            )
    use_pressure_operator_graph = bool(
        use_pressure_outlet_operator_graph or use_closed_operator_graph
    )
    if not use_pressure_operator_graph:
        self._invalidate_pressure_nullspace_component_graph()
    pressure_component_count = max(
        int(outlet_component_count),
        int(closed_component_count),
    )
    project_pressure_components = pressure_component_count > 0

    def project_component_means(field: object) -> None:
        if use_pressure_operator_graph:
            self._subtract_pressure_outlet_nullspace_component_means_device(
                field,
            )
        else:
            self._subtract_pressure_component_means_device(
                field,
                pressure_component_count,
            )
        if bool(pressure_outlet_zmin):
            self.last_cg_unreached_set_mean_projection_count += 1
        else:
            self.last_cg_componentwise_mean_projection_count += 1
        self.last_cg_mean_projection_count += 1
    self.last_hibm_unreached_cells_with_interface_diagonal = 0
    self.last_hibm_unreached_cells_with_interface_coupling = 0
    self.last_hibm_unreached_components_with_interface_hits = 0
    if physical_outlet_component_count > 0 and report_requested:
        # R2-H1: before the unreached-set mean subtraction anchors the
        # flood-disconnected components, measure how much of that set the
        # interface matrix terms already anchor/connect. Overlap here means
        # the zero-mean projection perturbs a non-singular subsystem. Only
        # the project() report consumes these three counters, so skip the
        # scan on the hot path when reporting isn't requested.
        self._record_unreached_interface_hit_diagnostics()

    # REQ_373: the same no-free-face bookkeeping as the CPU solver (rhs scale for the audit; removed-cell list that
    # the CPU dispatcher pins to p = 0 after this solve).
    self._no_free_face_rhs_scale = float(rhs_scale)
    self._no_free_face_removed_cells = []
    self._prepare_fv_multigrid_rhs(rhs_scale)
    self._fv_diagonal_kernel(
        self.fv_diag,
        outlet,
        self._velocity_dirichlet_boundary_authority_code(),
    )
    if project_pressure_components:
        if use_pressure_operator_graph:
            self._measure_pressure_outlet_nullspace_rhs_incompatibility(
                self._mg_rhs[0],
            )
        else:
            self._measure_unreached_component_rhs_incompatibility(
                self._mg_rhs[0],
                component_count=pressure_component_count,
            )
        if (
            not pressure_outlet_zmin
            and closed_neumann_velocity_inlet_zmax_mode is not None
        ):
            self._measure_closed_component_physical_rhs_incompatibility(
                component_count=(
                    int(self._pressure_outlet_nullspace_source_component_count)
                    if use_closed_operator_graph else pressure_component_count
                ),
                mapped_components=use_closed_operator_graph,
                rhs_scale=rhs_scale,
                velocity_inlet_zmax_mode=closed_neumann_velocity_inlet_zmax_mode,
            )
        project_component_means(self._mg_rhs[0])
    elif remove_nullspace_mean and not pressure_outlet_zmin:
        self._measure_closed_neumann_rhs_incompatibility(
            self._mg_rhs[0], rhs_scale=rhs_scale,
            velocity_inlet_zmax_mode=closed_neumann_velocity_inlet_zmax_mode,
        )
    else:
        self.last_hibm_unreached_incompatible_component_count = 0
        self.last_hibm_unreached_component_rhs_mean_max_abs = 0.0
        self.last_hibm_unreached_component_rhs_integral_max_abs = 0.0
    if remove_nullspace_mean:
        self._weighted_mean_to_cg_field_kernel(self._mg_rhs[0], -1.0)
        self.last_cg_mean_projection_count += 1
        self._cg_build_positive_rhs_from_cg_mean_kernel(
            self._mg_rhs[0],
            self.cg_z,
        )
    else:
        self._cg_build_positive_rhs_kernel(
            self._mg_rhs[0],
            self.cg_z,
            0.0,
        )
    self._copy_scalar_field_kernel(self.cg_rhs, self.cg_z)
    b_norm = self._h2_fluid_scalars.rhs(self)
    self.last_cg_rhs_nonzero = b_norm > 1.0e-30
    if b_norm <= 1.0e-30:
        self._copy_scalar_field_kernel(self.cg_r, self.cg_z)
        self._clear_pressure_kernel()
        self.last_cg_iterations = 0
        self.last_cg_initial_relative_residual = 0.0
        self.last_cg_relative_residual = 0.0
        self.last_cg_converged = True
        return

    # Put the initial guess (accepted pressure or a warm-start slot) in this
    # solve's gauge: an unanchored component's stale constant has zero
    # residual, so an immediate exit would otherwise keep it.
    if project_pressure_components:
        project_component_means(self.pressure)
    if remove_nullspace_mean:
        self._subtract_weighted_mean_device(self.pressure)
        self.last_cg_mean_projection_count += 1

    self._apply_fv_pressure_operator(
        self.pressure,
        self.cg_Ad,
        outlet,
        self._velocity_dirichlet_boundary_authority_code(),
        pressure_velocity_nullspace_projector,
    )
    self._axpby_scalar_field_kernel(self.cg_r, 1.0, self.cg_z, -1.0, self.cg_Ad)
    if project_pressure_components:
        project_component_means(self.cg_r)
    if remove_nullspace_mean:
        self._subtract_weighted_mean_device(self.cg_r)
        self.last_cg_mean_projection_count += 1
    self._weighted_dot_to_field_kernel(self.cg_r, self.cg_r, self.cg_rr)
    self.last_cg_host_residual_checks += 1
    r_norm = self._h2_fluid_scalars.poll(self, relative_tolerance)
    initial_relative_residual = float(self._h2_cg_packet[1])
    self.last_cg_initial_relative_residual = initial_relative_residual
    self.last_cg_relative_residual = initial_relative_residual
    if bool(self._h2_cg_packet[4]):
        self.last_cg_converged = True
        return

    explicit_interface_multigrid = (
        bool(pressure_interface_matrix_active)
        and preconditioner_name in {"fv_multigrid", "fv_multigrid_light"}
    )
    # The current MG hierarchy restricts the interface diagonal but omits
    # row-list off-diagonals. Keep interface use explicit-only: the outer
    # CG fine matvec and fail-closed confirmation still apply the exact row
    # list, while `auto` remains Jacobi until a Galerkin hierarchy is valid.
    use_mg_preconditioner = len(self._mg_shapes) > 1 and (
        explicit_interface_multigrid
        or (
            not bool(pressure_interface_matrix_active)
            and (
                preconditioner_name in {"fv_multigrid", "fv_multigrid_light"}
                or (preconditioner_name == "auto" and not self.grid.is_uniform)
            )
        )
    )
    if use_mg_preconditioner:
        self.last_cg_preconditioner_effective = (
            "fv_multigrid"
            if preconditioner_name == "auto"
            else preconditioner_name
        )
    elif preconditioner_name == "jacobi":
        self.last_cg_preconditioner_effective = "jacobi_explicit"
    elif preconditioner_name == "auto":
        self.last_cg_preconditioner_effective = "jacobi_auto"
    elif len(self._mg_shapes) <= 1:
        self.last_cg_preconditioner_effective = "jacobi_fallback_no_hierarchy"
    else:
        self.last_cg_preconditioner_effective = (
            "jacobi_fallback_interface_matrix_policy"
        )
    mg_pre_smooth_iterations = 1 if preconditioner_name == "fv_multigrid_light" else 2
    mg_coarse_smooth_iterations = 12 if preconditioner_name == "fv_multigrid_light" else 24
    mg_post_smooth_iterations = 1 if preconditioner_name == "fv_multigrid_light" else 2
    if use_mg_preconditioner:
        self.last_cg_multigrid_pre_smooth_iterations = mg_pre_smooth_iterations
        self.last_cg_multigrid_coarse_smooth_iterations = (
            mg_coarse_smooth_iterations
        )
        self.last_cg_multigrid_post_smooth_iterations = mg_post_smooth_iterations
    # The level operators depend only on this solve's fixed masks.
    mg_level_operators_prepared = False

    def apply_cg_preconditioner() -> None:
        nonlocal mg_level_operators_prepared
        if use_mg_preconditioner:
            self.last_cg_multigrid_apply_count += 1
            self._apply_fv_multigrid_preconditioner(
                self.cg_r,
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                pre_smooth_iterations=mg_pre_smooth_iterations,
                coarse_smooth_iterations=mg_coarse_smooth_iterations,
                post_smooth_iterations=mg_post_smooth_iterations,
                prepare_level_operators=not mg_level_operators_prepared,
            )
            mg_level_operators_prepared = True
        else:
            self._apply_jacobi_preconditioner_kernel(self.cg_r, self.cg_z)

    apply_cg_preconditioner()
    if project_pressure_components:
        project_component_means(self.cg_z)
    if remove_nullspace_mean:
        self._subtract_weighted_mean_device(self.cg_z)
        self.last_cg_mean_projection_count += 1
    self._weighted_dot_to_field_kernel(self.cg_r, self.cg_z, self.cg_rz)
    rz = self._h2_fluid_scalars.rz(self)
    if not bool(self._h2_cg_packet[6]) or rz <= 1.0e-300:
        if use_mg_preconditioner:
            self.last_cg_multigrid_to_jacobi_fallback_count += 1
        use_mg_preconditioner = False
        self.last_cg_preconditioner_effective = (
            "jacobi_fallback_nonpositive_rz"
        )
        self._apply_jacobi_preconditioner_kernel(self.cg_r, self.cg_z)
        if project_pressure_components:
            project_component_means(self.cg_z)
        if remove_nullspace_mean:
            self._subtract_weighted_mean_device(self.cg_z)
            self.last_cg_mean_projection_count += 1
        self._weighted_dot_to_field_kernel(self.cg_r, self.cg_z, self.cg_rz)
        rz = self._h2_fluid_scalars.rz(self)
    if not bool(self._h2_cg_packet[6]) or rz <= 0.0:
        self.last_cg_breakdown = "initial residual/preconditioner dot is not positive finite"
        return
    self._copy_scalar_field_kernel(self.cg_d, self.cg_z)

    residual_check_interval = 16
    residual_restart_interval = 256 if bool(pressure_interface_matrix_active) else 512
    from .gpu_cg_window import nonpoll_window
    window_completed_through = 0
    for iteration in range(1, max_iters + 1):
        if iteration <= window_completed_through:
            continue
        window_count = nonpoll_window(self, first_iteration=iteration,
            max_iters=max_iters, outlet=outlet,
            authority=self._velocity_dirichlet_boundary_authority_code(),
            projector=pressure_velocity_nullspace_projector,
            use_mg=use_mg_preconditioner,
            project_components=project_pressure_components,
            remove_mean=remove_nullspace_mean,
            apply_preconditioner=apply_cg_preconditioner,
            project_component_means=project_component_means)
        if window_count:
            window_completed_through = iteration + window_count - 1
            continue
        self.last_cg_pcg_iterations = iteration
        self._apply_fv_pressure_operator(
            self.cg_d,
            self.cg_Ad,
            outlet,
            self._velocity_dirichlet_boundary_authority_code(),
            pressure_velocity_nullspace_projector,
        )
        self._cg_alpha_step_kernel()
        if project_pressure_components and iteration % 16 == 0:
            project_component_means(self.pressure)
            project_component_means(self.cg_r)
        if remove_nullspace_mean and iteration % 16 == 0:
            self._subtract_weighted_mean_device(self.pressure)
            self._subtract_weighted_mean_device(self.cg_r)
            self.last_cg_mean_projection_count += 2

        self.last_cg_iterations = iteration
        should_check_residual = (
            iteration % residual_check_interval == 0 or iteration == max_iters
        )
        if should_check_residual:
            self._weighted_dot_to_field_kernel(self.cg_r, self.cg_r, self.cg_rr)
            self.last_cg_host_residual_checks += 1
            r_norm = self._h2_fluid_scalars.poll(self, relative_tolerance)
            relative_residual = float(self._h2_cg_packet[1])
            self.last_cg_relative_residual = relative_residual
            if bool(self._h2_cg_packet[5]):
                self.last_cg_converged = True
                return
            breakdown_code = int(self._h2_cg_packet[2])
            if breakdown_code != 0:
                self.last_cg_breakdown_dAd = float(self._h2_cg_packet[3])
                if iteration < max_iters:
                    self._apply_fv_pressure_operator(
                        self.pressure,
                        self.cg_Ad,
                        outlet,
                        self._velocity_dirichlet_boundary_authority_code(),
                        pressure_velocity_nullspace_projector,
                    )
                    self._axpby_scalar_field_kernel(
                        self.cg_r,
                        1.0,
                        self.cg_rhs,
                        -1.0,
                        self.cg_Ad,
                    )
                    if project_pressure_components:
                        project_component_means(self.cg_r)
                    if remove_nullspace_mean:
                        self._subtract_weighted_mean_device(self.cg_r)
                        self.last_cg_mean_projection_count += 1
                    self._weighted_dot_to_field_kernel(self.cg_r, self.cg_r, self.cg_rr)
                    self.last_cg_host_residual_checks += 1
                    r_norm = self._h2_fluid_scalars.poll(self, relative_tolerance)
                    relative_residual = float(self._h2_cg_packet[1])
                    self.last_cg_relative_residual = relative_residual
                    if bool(self._h2_cg_packet[5]):
                        self.last_cg_converged = True
                        return
                    apply_cg_preconditioner()
                    if project_pressure_components:
                        project_component_means(self.cg_z)
                    if remove_nullspace_mean:
                        self._subtract_weighted_mean_device(self.cg_z)
                        self.last_cg_mean_projection_count += 1
                    self._weighted_dot_to_field_kernel(self.cg_r, self.cg_z, self.cg_rz)
                    rz = self._h2_fluid_scalars.rz(self)
                    if bool(self._h2_cg_packet[6]) and rz > 0.0:
                        self._copy_scalar_field_kernel(self.cg_d, self.cg_z)
                        self.cg_breakdown_code[None] = 0
                        self.cg_breakdown_dAd[None] = 0.0
                        self.last_cg_restart_count += 1
                        continue
                self.last_cg_breakdown = "device-side CG scalar update failed"
                return
            if (
                iteration < max_iters
                and residual_restart_interval > 0
                and iteration % residual_restart_interval == 0
            ):
                self._apply_fv_pressure_operator(
                    self.pressure,
                    self.cg_Ad,
                    outlet,
                    self._velocity_dirichlet_boundary_authority_code(),
                    pressure_velocity_nullspace_projector,
                )
                self._axpby_scalar_field_kernel(
                    self.cg_r,
                    1.0,
                    self.cg_rhs,
                    -1.0,
                    self.cg_Ad,
                )
                if project_pressure_components:
                    project_component_means(self.cg_r)
                if remove_nullspace_mean:
                    self._subtract_weighted_mean_device(self.cg_r)
                    self.last_cg_mean_projection_count += 1
                self._weighted_dot_to_field_kernel(self.cg_r, self.cg_r, self.cg_rr)
                self.last_cg_host_residual_checks += 1
                r_norm = self._h2_fluid_scalars.poll(self, relative_tolerance)
                relative_residual = float(self._h2_cg_packet[1])
                self.last_cg_relative_residual = relative_residual
                if bool(self._h2_cg_packet[5]):
                    self.last_cg_converged = True
                    return
                apply_cg_preconditioner()
                if project_pressure_components:
                    project_component_means(self.cg_z)
                if remove_nullspace_mean:
                    self._subtract_weighted_mean_device(self.cg_z)
                    self.last_cg_mean_projection_count += 1
                self._weighted_dot_to_field_kernel(self.cg_r, self.cg_z, self.cg_rz)
                rz = self._h2_fluid_scalars.rz(self)
                if not bool(self._h2_cg_packet[6]) or rz <= 0.0:
                    self.last_cg_breakdown = (
                        "restarted residual/preconditioner dot is not positive finite"
                    )
                    return
                self._copy_scalar_field_kernel(self.cg_d, self.cg_z)
                self.last_cg_restart_count += 1
                continue

        apply_cg_preconditioner()
        if project_pressure_components:
            project_component_means(self.cg_z)
        if remove_nullspace_mean:
            self._subtract_weighted_mean_device(self.cg_z)
            self.last_cg_mean_projection_count += 1
        self._cg_beta_step_kernel(1 if use_mg_preconditioner else 0)
    if (
        bool(pressure_interface_matrix_active)
        and int(self.pressure_interface_row_count[None]) > 0
    ):
        self._continue_pressure_poisson_fv_bicgstab(
            start_iteration=max_iters,
            max_iters=max_iters,
            b_norm=b_norm,
            outlet=outlet,
            tolerance=relative_tolerance,
            pressure_component_count=pressure_component_count,
            pressure_components_are_outlet=bool(pressure_outlet_zmin),
            pressure_components_use_operator_graph=bool(
                use_pressure_operator_graph
            ),
            remove_nullspace_mean=remove_nullspace_mean,
            pressure_velocity_nullspace_projector=(
                pressure_velocity_nullspace_projector
            ),
        )


def _continue_pressure_poisson_fv_bicgstab(
    self,
    *,
    start_iteration: int,
    max_iters: int,
    b_norm: float,
    outlet: int,
    tolerance: float,
    pressure_component_count: int,
    pressure_components_are_outlet: bool,
    pressure_components_use_operator_graph: bool,
    remove_nullspace_mean: bool,
    pressure_velocity_nullspace_projector: object | None = None,
) -> None:
    self._h2_fluid_scalars.set_rhs(b_norm)
    fallback_iters = max(1, int(max_iters))
    relative_tolerance = scalar_maximum(0.0, float(tolerance))
    component_count = max(0, int(pressure_component_count))
    self.last_cg_restart_policy = "periodic_exact_residual+bicgstab_fallback"
    self.last_cg_breakdown = ""
    self.cg_breakdown_code[None] = 0
    self.cg_breakdown_dAd[None] = 0.0

    def subtract_component_means(field: object) -> None:
        if bool(pressure_components_use_operator_graph):
            self._subtract_pressure_outlet_nullspace_component_means_device(
                field,
            )
        else:
            self._subtract_pressure_component_means_device(
                field,
                component_count,
            )
        if bool(pressure_components_are_outlet):
            self.last_cg_unreached_set_mean_projection_count += 1
        else:
            self.last_cg_componentwise_mean_projection_count += 1
        self.last_cg_mean_projection_count += 1

    def project_field(field: object) -> None:
        if component_count > 0:
            subtract_component_means(field)
        if remove_nullspace_mean:
            self._subtract_weighted_mean_device(field)
            self.last_cg_mean_projection_count += 1

    def project_pressure_and_residual() -> None:
        if component_count > 0:
            subtract_component_means(self.pressure)
            subtract_component_means(self.cg_r)
        if remove_nullspace_mean:
            self._subtract_weighted_mean_device(self.pressure)
            self._subtract_weighted_mean_device(self.cg_r)
            self.last_cg_mean_projection_count += 2

    def rebuild_exact_residual() -> None:
        self._apply_fv_pressure_operator(
            self.pressure,
            self.cg_Ad,
            outlet,
            self._velocity_dirichlet_boundary_authority_code(),
            pressure_velocity_nullspace_projector,
        )
        self._axpby_scalar_field_kernel(
            self.cg_r,
            1.0,
            self.cg_rhs,
            -1.0,
            self.cg_Ad,
        )
        project_field(self.cg_r)

    rebuild_exact_residual()
    self._weighted_dot_to_field_kernel(self.cg_r, self.cg_r, self.cg_rr)
    self.last_cg_host_residual_checks += 1
    r_norm = self._h2_fluid_scalars.poll(self, relative_tolerance)
    self.last_cg_relative_residual = float(self._h2_cg_packet[1])
    if self.last_cg_relative_residual <= relative_tolerance:
        self.last_cg_converged = True
        return

    self._bicgstab_reset_direction_kernel()
    rho_prev = 1.0
    alpha = 1.0
    omega = 1.0
    scalar_floor = 1.0e-300

    def reset_recurrence() -> None:
        nonlocal rho_prev, alpha, omega
        self._bicgstab_reset_direction_kernel()
        rho_prev = alpha = omega = 1.0
        self.last_cg_restart_count += 1

    for iteration in range(1, fallback_iters + 1):
        self.last_cg_bicgstab_iterations = iteration
        self._weighted_dot_to_field_kernel(
            self.cg_r_old,
            self.cg_r,
            self.cg_rz_new,
        )
        rho = float(self.cg_rz_new[None])
        if not self._h2_fluid_scalars.finite_value(rho) or self._h2_fluid_scalars.abs_value(rho) <= scalar_floor:
            if iteration >= fallback_iters:
                self.last_cg_breakdown = (
                    "BiCGSTAB shadow residual dot is not finite"
                )
                return
            rebuild_exact_residual()
            self._weighted_dot_to_field_kernel(self.cg_r, self.cg_r, self.cg_rr)
            self.last_cg_host_residual_checks += 1
            r_norm = self._h2_fluid_scalars.poll(self, relative_tolerance)
            self.last_cg_iterations = int(start_iteration) + iteration
            self.last_cg_relative_residual = float(self._h2_cg_packet[1])
            if self.last_cg_relative_residual <= relative_tolerance:
                self.last_cg_converged = True
                return
            reset_recurrence()
            continue
        if iteration == 1:
            beta = 0.0
        else:
            if (
                not self._h2_fluid_scalars.finite_value(rho_prev)
                or self._h2_fluid_scalars.abs_value(rho_prev) <= scalar_floor
                or not self._h2_fluid_scalars.finite_value(omega)
                or self._h2_fluid_scalars.abs_value(omega) <= scalar_floor
            ):
                if iteration >= fallback_iters:
                    self.last_cg_breakdown = (
                        "BiCGSTAB recurrence scalar is not finite"
                    )
                    return
                rebuild_exact_residual()
                reset_recurrence()
                continue
            beta = self._h2_fluid_scalars.divide_multiply(rho, rho_prev, alpha, omega)
            if not self._h2_fluid_scalars.finite_value(beta):
                if iteration >= fallback_iters:
                    self.last_cg_breakdown = "BiCGSTAB beta is not finite"
                    return
                rebuild_exact_residual()
                reset_recurrence()
                continue

        self._bicgstab_update_direction_kernel(float(beta), float(omega))
        self._apply_jacobi_preconditioner_kernel(self.cg_d, self.cg_z)
        project_field(self.cg_z)
        self._apply_fv_pressure_operator(
            self.cg_z,
            self.cg_Ad,
            outlet,
            self._velocity_dirichlet_boundary_authority_code(),
            pressure_velocity_nullspace_projector,
        )
        self._weighted_dot_to_field_kernel(
            self.cg_r_old,
            self.cg_Ad,
            self.cg_dAd,
        )
        alpha_denominator = float(self.cg_dAd[None])
        if (
            not self._h2_fluid_scalars.finite_value(alpha_denominator)
            or self._h2_fluid_scalars.abs_value(alpha_denominator) <= scalar_floor
        ):
            if iteration >= fallback_iters:
                self.last_cg_breakdown = (
                    "BiCGSTAB alpha denominator is not finite"
                )
                return
            rebuild_exact_residual()
            reset_recurrence()
            continue
        alpha = self._h2_fluid_scalars.divide(rho, alpha_denominator)
        if not self._h2_fluid_scalars.finite_value(alpha):
            if iteration >= fallback_iters:
                self.last_cg_breakdown = "BiCGSTAB alpha is not finite"
                return
            rebuild_exact_residual()
            reset_recurrence()
            continue

        self._bicgstab_apply_alpha_and_compute_s_kernel(float(alpha))
        project_field(self.bicgstab_s)
        self._weighted_dot_to_field_kernel(
            self.bicgstab_s,
            self.bicgstab_s,
            self.cg_rr,
        )
        self.last_cg_host_residual_checks += 1
        s_norm = self._h2_fluid_scalars.poll(self, relative_tolerance)
        self.last_cg_iterations = int(start_iteration) + iteration
        self.last_cg_relative_residual = float(self._h2_cg_packet[1])
        if self.last_cg_relative_residual <= relative_tolerance:
            self._copy_scalar_field_kernel(self.cg_r, self.bicgstab_s)
            self.last_cg_converged = True
            return

        self._apply_jacobi_preconditioner_kernel(self.bicgstab_s, self.cg_z)
        project_field(self.cg_z)
        self._apply_fv_pressure_operator(
            self.cg_z,
            self.bicgstab_t,
            outlet,
            self._velocity_dirichlet_boundary_authority_code(),
            pressure_velocity_nullspace_projector,
        )
        self._weighted_dot_to_field_kernel(
            self.bicgstab_t,
            self.bicgstab_s,
            self.cg_beta_numerator,
        )
        t_dot_s = float(self.cg_beta_numerator[None])
        self._weighted_dot_to_field_kernel(
            self.bicgstab_t,
            self.bicgstab_t,
            self.cg_dAd,
        )
        t_dot_t = float(self.cg_dAd[None])
        if not self._h2_fluid_scalars.finite_value(t_dot_t) or self._h2_fluid_scalars.abs_value(t_dot_t) <= scalar_floor:
            if iteration >= fallback_iters:
                self.last_cg_breakdown = (
                    "BiCGSTAB omega denominator is not finite"
                )
                return
            self._copy_scalar_field_kernel(self.cg_r, self.bicgstab_s)
            project_field(self.cg_r)
            reset_recurrence()
            continue
        omega = self._h2_fluid_scalars.divide(t_dot_s, t_dot_t)
        if not self._h2_fluid_scalars.finite_value(omega) or self._h2_fluid_scalars.abs_value(omega) <= scalar_floor:
            if iteration >= fallback_iters:
                self.last_cg_breakdown = "BiCGSTAB omega is not finite"
                return
            self._copy_scalar_field_kernel(self.cg_r, self.bicgstab_s)
            project_field(self.cg_r)
            reset_recurrence()
            continue

        self._bicgstab_apply_omega_kernel(float(omega))
        if iteration % 16 == 0:
            project_pressure_and_residual()
        self._weighted_dot_to_field_kernel(self.cg_r, self.cg_r, self.cg_rr)
        self.last_cg_host_residual_checks += 1
        r_norm = self._h2_fluid_scalars.poll(self, relative_tolerance)
        self.last_cg_iterations = int(start_iteration) + iteration
        self.last_cg_relative_residual = float(self._h2_cg_packet[1])
        if self.last_cg_relative_residual <= relative_tolerance:
            self.last_cg_converged = True
            return
        rho_prev = rho

    self.last_cg_breakdown = "BiCGSTAB fallback did not converge"


def confirm_exact(
    self,
    *,
    pressure_outlet_zmin: bool,
    tolerance: float,
    remove_nullspace_mean: bool,
    pressure_nullspace_component_count: int = 0,
    pressure_components_use_operator_graph: bool = False,
    pressure_velocity_nullspace_projector: object | None = None,
) -> None:
    """Replace recurrence-only convergence with an exact fine-grid check."""

    if bool(pressure_components_use_operator_graph):
        graph_valid = bool(self._pressure_outlet_nullspace_graph_valid)
        graph_context = getattr(
            self,
            "_pressure_outlet_nullspace_graph_context",
            "",
        )
        source_component_count = _diagnostic_int(
            getattr(
                self,
                "_pressure_outlet_nullspace_source_component_count",
                -1,
            )
        )
        solve_component_count = _diagnostic_int(
            getattr(
                self,
                "_pressure_outlet_nullspace_component_count",
                -1,
            )
        )
        if bool(pressure_outlet_zmin):
            operator_component_count = _diagnostic_int(
                getattr(
                    self,
                    "_pressure_outlet_operator_component_count",
                    -1,
                )
            )
            component_counts_valid = bool(
                source_component_count >= 0
                and 0
                <= solve_component_count
                <= self.pressure_component_capacity
                and operator_component_count == solve_component_count
            )
            current_source_component_count = max(
                0,
                _diagnostic_int(
                    getattr(
                        self,
                        "last_hibm_pressure_unreached_component_raw_count",
                        0,
                    )
                ),
                _diagnostic_int(
                    getattr(
                        self,
                        "_hibm_pressure_unreached_component_count",
                        0,
                    )
                ),
            )
            graph_semantics_match = bool(
                graph_context == "outlet-disconnected"
                and source_component_count
                == current_source_component_count
            )
        else:
            component_counts_valid = bool(
                source_component_count > 0
                and 0 <= solve_component_count <= source_component_count
            )
            requested_solve_component_count = _diagnostic_int(
                pressure_nullspace_component_count
            )
            graph_semantics_match = bool(
                graph_context == "closed-Neumann"
                and requested_solve_component_count > 0
                and solve_component_count
                == requested_solve_component_count
            )
        if not bool(
            graph_valid
            and component_counts_valid
            and graph_semantics_match
        ):
            raise self._pressure_operator_graph_missing_or_stale_error(
                context="exact-pressure-residual",
            )

    outlet = 1 if pressure_outlet_zmin else 0
    self._apply_fv_pressure_operator(
        self.pressure,
        self.cg_Ad,
        outlet,
        self._velocity_dirichlet_boundary_authority_code(),
        pressure_velocity_nullspace_projector,
    )
    self._axpby_scalar_field_kernel(
        self.cg_r,
        1.0,
        self.cg_rhs,
        -1.0,
        self.cg_Ad,
    )
    outlet_component_count = 0
    if bool(pressure_outlet_zmin):
        if bool(pressure_components_use_operator_graph):
            outlet_component_count = int(
                self._pressure_outlet_nullspace_component_count
            )
        elif int(self._hibm_pressure_unreached_count) > 0:
            outlet_component_count = int(
                self._hibm_pressure_unreached_component_count
            )
    closed_component_count = (
        max(0, int(pressure_nullspace_component_count))
        if not bool(pressure_outlet_zmin)
        else 0
    )
    pressure_component_count = max(
        int(outlet_component_count),
        int(closed_component_count),
    )
    project_pressure_components = pressure_component_count > 0

    def project_component_means(field: object) -> None:
        if bool(pressure_components_use_operator_graph):
            self._subtract_pressure_outlet_nullspace_component_means_device(
                field,
            )
        else:
            self._subtract_pressure_component_means_device(
                field,
                pressure_component_count,
            )
        if bool(pressure_outlet_zmin):
            self.last_cg_unreached_set_mean_projection_count += 1
        else:
            self.last_cg_componentwise_mean_projection_count += 1
        self.last_cg_mean_projection_count += 1
    if project_pressure_components:
        project_component_means(self.cg_r)
    if remove_nullspace_mean:
        self._subtract_weighted_mean_device(self.cg_r)
        self.last_cg_mean_projection_count += 1
    exact_packet = self._h2_fluid_scalars.exact(self, scalar_maximum(0.0, float(tolerance)))
    residual_norm = float(exact_packet[0])
    rhs_norm = float(exact_packet[1])
    exact_relative_residual = float(exact_packet[2])
    self.last_cg_host_residual_checks += 1
    self.last_cg_exact_residual_confirmation_count += 1
    self.last_cg_recursive_relative_residual = self.last_cg_relative_residual
    self.last_cg_exact_relative_residual = exact_relative_residual
    self.last_cg_relative_residual = exact_relative_residual
    recurrence_reported_converged = bool(self.last_cg_converged)
    self.last_cg_converged = (
        bool(exact_packet[7])
    )
    if self.last_cg_converged:
        self.last_cg_breakdown = ""
    elif recurrence_reported_converged:
        self.last_cg_breakdown = (
            "recursive residual convergence failed exact fine-grid confirmation"
        )


solve_cg = _solve_pressure_poisson_fv_cg
continue_bicgstab = _continue_pressure_poisson_fv_bicgstab
# REQ_373 runtime coverage guard marker (= simulation_core.fluids.solver.NO_FREE_FACE_RULE_REVISION).
_solve_pressure_poisson_fv_cg.no_free_face_rule = "REQ369-R1b-v2-all-sites"


class _DevicePressureMethods(CartesianFluidSolver):
    """Source carrier only; never instantiated, preserving existing graph reuse.

    The original region parser proves its own inverse on these finite CUDA
    controllers. Scalar reads and original host error gates remain boundaries.
    """
    _solve_pressure_poisson_fv_cg = solve_cg
    _continue_pressure_poisson_fv_bicgstab = continue_bicgstab
