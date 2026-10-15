"""The CUDA controllers of the conjugate-gradient pressure solve: solve_cg and confirm_exact are
fluid/pressure_conjugate_gradient.py's _solve_pressure_poisson_fv_cg and _confirm_pressure_poisson_fv_cg_exact_residual
with their host scalar arithmetic moved into f64 device kernels (fastfsi.fluid.device_arithmetic): every norm, ratio
and convergence test is evaluated on the device and read as one packet per poll.  No iteration cadence, restart,
fallback or operator changes.  Installed per instance on CUDA by fastfsi.fluid.device_controls; the CPU methods stay.
"""
from math import sqrt
import math
import taichi as ti
from fastfsi.fluid.common import CG_DOT_GROUPS, CG_DOT_LANES, PressureSolveConvergenceError, _diagnostic_int
from fastfsi.fluid.constants import CG_PRECONDITIONER_CHOICES
from fastfsi.fluid.device_reduction import scalar_maximum


def solve_cg(
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
        int(self._pressure_unreached_component_count)
        if bool(pressure_outlet_zmin)
        and int(self._pressure_unreached_count) > 0
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
                    "last_pressure_unreached_component_raw_count",
                    0,
                )
            ),
            int(getattr(self, "_pressure_unreached_component_count", 0)),
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
        raise NotImplementedError("closed domains without an outlet are not supported yet")
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
    self.last_unreached_cells_with_interface_diagonal = 0
    self.last_unreached_cells_with_interface_coupling = 0
    self.last_unreached_components_with_interface_hits = 0
    if physical_outlet_component_count > 0 and report_requested:
        # R2-H1: before the unreached-set mean subtraction anchors the
        # flood-disconnected components, measure how much of that set the
        # interface matrix terms already anchor/connect. Overlap here means
        # the zero-mean projection perturbs a non-singular subsystem. Only
        # the project() report consumes these three counters, so skip the
        # scan on the hot path when reporting isn't requested.
        raise NotImplementedError("per-call solver reports (read_report=True) are not supported; use compute_divergence() and the simulation monitors")

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
            raise NotImplementedError("closed domains without an outlet are not supported yet")
        project_component_means(self._mg_rhs[0])
    elif remove_nullspace_mean and not pressure_outlet_zmin:
        raise NotImplementedError("closed domains without an outlet are not supported yet")
    else:
        self.last_unreached_incompatible_component_count = 0
        self.last_unreached_component_rhs_mean_max_abs = 0.0
        self.last_unreached_component_rhs_integral_max_abs = 0.0
    if remove_nullspace_mean:
        raise NotImplementedError("closed domains without an outlet are not supported yet")
    else:
        self._cg_build_positive_rhs_kernel(
            self._mg_rhs[0],
            self.cg_z,
            0.0,
        )
    self._copy_scalar_field_kernel(self.cg_rhs, self.cg_z)
    b_norm = self._device_arithmetic.rhs(self)
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
        raise NotImplementedError("closed domains without an outlet are not supported yet")

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
        raise NotImplementedError("closed domains without an outlet are not supported yet")
    self._weighted_dot_to_field_kernel(self.cg_r, self.cg_r, self.cg_rr)
    self.last_cg_host_residual_checks += 1
    r_norm = self._device_arithmetic.poll(self, relative_tolerance)
    initial_relative_residual = float(self._device_cg_packet[1])
    self.last_cg_initial_relative_residual = initial_relative_residual
    self.last_cg_relative_residual = initial_relative_residual
    if bool(self._device_cg_packet[4]):
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
        raise NotImplementedError("closed domains without an outlet are not supported yet")
    self._weighted_dot_to_field_kernel(self.cg_r, self.cg_z, self.cg_rz)
    rz = self._device_arithmetic.rz(self)
    if not bool(self._device_cg_packet[6]) or rz <= 1.0e-300:
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
            raise NotImplementedError("closed domains without an outlet are not supported yet")
        self._weighted_dot_to_field_kernel(self.cg_r, self.cg_z, self.cg_rz)
        rz = self._device_arithmetic.rz(self)
    if not bool(self._device_cg_packet[6]) or rz <= 0.0:
        self.last_cg_breakdown = "initial residual/preconditioner dot is not positive finite"
        return
    self._copy_scalar_field_kernel(self.cg_d, self.cg_z)

    residual_check_interval = 16
    residual_restart_interval = 256 if bool(pressure_interface_matrix_active) else 512
    for iteration in range(1, max_iters + 1):
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
            raise NotImplementedError("closed domains without an outlet are not supported yet")

        self.last_cg_iterations = iteration
        should_check_residual = (
            iteration % residual_check_interval == 0 or iteration == max_iters
        )
        if should_check_residual:
            self._weighted_dot_to_field_kernel(self.cg_r, self.cg_r, self.cg_rr)
            self.last_cg_host_residual_checks += 1
            r_norm = self._device_arithmetic.poll(self, relative_tolerance)
            relative_residual = float(self._device_cg_packet[1])
            self.last_cg_relative_residual = relative_residual
            if bool(self._device_cg_packet[5]):
                self.last_cg_converged = True
                return
            breakdown_code = int(self._device_cg_packet[2])
            if breakdown_code != 0:
                self.last_cg_breakdown_dAd = float(self._device_cg_packet[3])
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
                        raise NotImplementedError("closed domains without an outlet are not supported yet")
                    self._weighted_dot_to_field_kernel(self.cg_r, self.cg_r, self.cg_rr)
                    self.last_cg_host_residual_checks += 1
                    r_norm = self._device_arithmetic.poll(self, relative_tolerance)
                    relative_residual = float(self._device_cg_packet[1])
                    self.last_cg_relative_residual = relative_residual
                    if bool(self._device_cg_packet[5]):
                        self.last_cg_converged = True
                        return
                    apply_cg_preconditioner()
                    if project_pressure_components:
                        project_component_means(self.cg_z)
                    if remove_nullspace_mean:
                        raise NotImplementedError("closed domains without an outlet are not supported yet")
                    self._weighted_dot_to_field_kernel(self.cg_r, self.cg_z, self.cg_rz)
                    rz = self._device_arithmetic.rz(self)
                    if bool(self._device_cg_packet[6]) and rz > 0.0:
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
                    raise NotImplementedError("closed domains without an outlet are not supported yet")
                self._weighted_dot_to_field_kernel(self.cg_r, self.cg_r, self.cg_rr)
                self.last_cg_host_residual_checks += 1
                r_norm = self._device_arithmetic.poll(self, relative_tolerance)
                relative_residual = float(self._device_cg_packet[1])
                self.last_cg_relative_residual = relative_residual
                if bool(self._device_cg_packet[5]):
                    self.last_cg_converged = True
                    return
                apply_cg_preconditioner()
                if project_pressure_components:
                    project_component_means(self.cg_z)
                if remove_nullspace_mean:
                    raise NotImplementedError("closed domains without an outlet are not supported yet")
                self._weighted_dot_to_field_kernel(self.cg_r, self.cg_z, self.cg_rz)
                rz = self._device_arithmetic.rz(self)
                if not bool(self._device_cg_packet[6]) or rz <= 0.0:
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
            raise NotImplementedError("closed domains without an outlet are not supported yet")
        self._cg_beta_step_kernel(1 if use_mg_preconditioner else 0)
    if (
        bool(pressure_interface_matrix_active)
        and int(self.pressure_interface_row_count[None]) > 0
    ):
        raise NotImplementedError("pressure-interface coupling rows are not supported (only a pressure-interface diagonal, as set by ColumnPressureChamber)")


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
                        "last_pressure_unreached_component_raw_count",
                        0,
                    )
                ),
                _diagnostic_int(
                    getattr(
                        self,
                        "_pressure_unreached_component_count",
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
        elif int(self._pressure_unreached_count) > 0:
            outlet_component_count = int(
                self._pressure_unreached_component_count
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
        raise NotImplementedError("closed domains without an outlet are not supported yet")
    exact_packet = self._device_arithmetic.exact(self, scalar_maximum(0.0, float(tolerance)))
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
