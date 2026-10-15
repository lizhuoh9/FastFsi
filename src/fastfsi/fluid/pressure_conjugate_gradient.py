"""The finite-volume conjugate-gradient pressure solve with the exact-residual confirmation (mixin)."""

from math import sqrt
import math

import taichi as ti

from fastfsi.fluid.common import CG_DOT_GROUPS, CG_DOT_LANES, PressureSolveConvergenceError, _diagnostic_int
from fastfsi.fluid.constants import CG_PRECONDITIONER_CHOICES


class ConjugateGradientSolver:
    """Mixin of IncompressibleFluidSolver; not used on its own.  Contract (solver attributes):
      reads:  _pressure_unreached_component_count, _pressure_unreached_count,
    _ki3_pressure_reduction_groups, _ki3_pressure_reduction_parts, _mg_rhs, _mg_shapes,
    _pressure_outlet_nullspace_component_count, _pressure_outlet_nullspace_graph_context,
    _pressure_outlet_nullspace_graph_valid, _pressure_outlet_nullspace_source_component_count,
    _pressure_outlet_operator_component_count, cg_Ad, cg_alpha, cg_beta, cg_beta_numerator, cg_breakdown_code,
    cg_breakdown_dAd, cg_d, cg_dAd, cg_r, cg_r_old, cg_rhs, cg_rr, cg_rz, cg_rz_new, cg_z, fv_diag, grid,
    last_cg_converged, last_cg_relative_residual, obstacle, pressure, pressure_interface_row_count
      writes: last_cg_bicgstab_iterations, last_cg_breakdown, last_cg_breakdown_dAd,
    last_cg_closed_neumann_physical_component_budgets, last_cg_closed_neumann_physical_rhs_integral,
    last_cg_closed_neumann_physical_rhs_mean, last_cg_closed_neumann_physical_rhs_measured,
    last_cg_componentwise_mean_projection_count, last_cg_converged, last_cg_exact_relative_residual,
    last_cg_exact_residual_confirmation_count, last_cg_host_residual_checks, last_cg_initial_relative_residual,
    last_cg_iterations, last_cg_mean_projection_count, last_cg_multigrid_apply_count,
    last_cg_multigrid_coarse_smooth_iterations, last_cg_multigrid_post_smooth_iterations,
    last_cg_multigrid_pre_smooth_iterations, last_cg_multigrid_to_jacobi_fallback_count,
    last_cg_operator_apply_count, last_cg_pcg_iterations, last_cg_preconditioner_effective,
    last_cg_preconditioner_requested, last_cg_recursive_relative_residual, last_cg_relative_residual,
    last_cg_restart_count, last_cg_restart_count_measured, last_cg_restart_policy, last_cg_rhs_nonzero,
    last_cg_unreached_set_mean_projection_count, last_unreached_cells_with_interface_coupling,
    last_unreached_cells_with_interface_diagonal, last_unreached_component_rhs_integral_max_abs,
    last_unreached_component_rhs_mean_max_abs, last_unreached_components_with_interface_hits,
    last_unreached_incompatible_component_count"""

    @ti.kernel
    def _cg_build_positive_rhs_kernel(
        self,
        legacy_rhs: ti.template(),
        rhs_out: ti.template(),
        rhs_weighted_mean: ti.f64,
    ):
        for i, j, k in rhs_out:
            if self.obstacle[i, j, k] == 1:
                rhs_out[i, j, k] = 0.0
            else:
                rhs_out[i, j, k] = -legacy_rhs[i, j, k] - rhs_weighted_mean

    @ti.kernel
    def _pcg_prepare_mg_level0_kernel(
        self,
        positive_rhs: ti.template(),
        legacy_rhs: ti.template(),
        pressure: ti.template(),
        tmp: ti.template(),
        residual: ti.template(),
    ):
        for i, j, k in positive_rhs:
            pressure[i, j, k] = 0.0
            tmp[i, j, k] = 0.0
            residual[i, j, k] = 0.0
            if self.obstacle[i, j, k] == 1:
                legacy_rhs[i, j, k] = 0.0
            else:
                legacy_rhs[i, j, k] = -positive_rhs[i, j, k]

    @ti.kernel
    def _apply_jacobi_preconditioner_kernel(
        self,
        residual: ti.template(),
        preconditioned: ti.template(),
    ):
        for i, j, k in preconditioned:
            if self.obstacle[i, j, k] == 1:
                preconditioned[i, j, k] = 0.0
            else:
                denominator = self.fv_diag[i, j, k]
                if denominator > 0.0:
                    preconditioned[i, j, k] = residual[i, j, k] / denominator
                else:
                    preconditioned[i, j, k] = 0.0

    @ti.kernel
    def _axpby_scalar_field_kernel(
        self,
        out: ti.template(),
        a: ti.f64,
        x: ti.template(),
        b: ti.f64,
        y: ti.template(),
    ):
        for i, j, k in out:
            if self.obstacle[i, j, k] == 1:
                out[i, j, k] = 0.0
            else:
                out[i, j, k] = a * x[i, j, k] + b * y[i, j, k]

    @ti.func
    def _ki3_ordered_dot(self, a: ti.template(), b: ti.template()):
        # Identical per-lane operands and arithmetic to the existing native dot.
        for lane in range(CG_DOT_LANES):
            self._ki3_pressure_reduction_parts[0, lane] = self._weighted_dot_lane_partial(a, b, lane)
        for group in range(CG_DOT_GROUPS):
            subtotal = ti.cast(0.0, ti.f64)
            for step in range(CG_DOT_LANES // CG_DOT_GROUPS):
                subtotal += self._ki3_pressure_reduction_parts[0, step * CG_DOT_GROUPS + group]
            self._ki3_pressure_reduction_groups[0, group] = subtotal
        total = ti.cast(0.0, ti.f64)
        ti.loop_config(serialize=True)
        for group in range(CG_DOT_GROUPS):
            total += self._ki3_pressure_reduction_groups[0, group]
        return total

    @ti.kernel
    def _weighted_dot_kernel(self, a: ti.template(), b: ti.template()) -> ti.f64:
        total = ti.cast(0.0, ti.f64)
        if ti.static(ti.lang.impl.current_cfg().arch == ti.cuda):
            total = self._ki3_ordered_dot(a, b)
        else:
            for i, j, k in a:
                if self.obstacle[i, j, k] == 0:
                    volume_m3 = self._cell_volume_f64_m3(i, j, k)
                    total += ti.cast(a[i, j, k], ti.f64) * ti.cast(b[i, j, k], ti.f64) * volume_m3
        return total

    @ti.func
    def _weighted_dot_lane_partial(self, a: ti.template(), b: ti.template(), lane):
        """Sum of one lane's cells: every CG_DOT_LANES-th cell in linear order.

        Neighboring lanes read neighboring cells, and the global f64 sum takes
        one atomic per lane instead of one per cell (f64 atomics on a single
        address serialize: ~10x slower for a 55k-cell field).  Only the
        summation order differs from the per-cell atomic sum, which was
        already order-nondeterministic.
        """

        nx, ny, nz = ti.static(a.shape)
        partial = ti.cast(0.0, ti.f64)
        cell = lane
        while cell < nx * ny * nz:
            i = cell // (ny * nz)
            j = (cell // nz) % ny
            k = cell % nz
            if self.obstacle[i, j, k] == 0:
                volume_m3 = self._cell_volume_f64_m3(i, j, k)
                partial += ti.cast(a[i, j, k], ti.f64) * ti.cast(b[i, j, k], ti.f64) * volume_m3
            cell += CG_DOT_LANES
        return partial

    @ti.kernel
    def _weighted_dot_to_field_kernel(
        self,
        a: ti.template(),
        b: ti.template(),
        out: ti.template(),
    ):
        out[None] = ti.cast(0.0, ti.f64)
        if ti.static(ti.lang.impl.current_cfg().arch == ti.cuda):
            out[None] = self._ki3_ordered_dot(a, b)
        else:
            for lane in range(CG_DOT_LANES):
                ti.atomic_add(out[None], self._weighted_dot_lane_partial(a, b, lane))

    @ti.kernel
    def _cg_alpha_step_kernel(self):
        """One launch for dot(d, Ad), the alpha update and its application.

        The loops and scalar logic are those of ``_weighted_dot_to_field_kernel``
        (d, Ad), ``_cg_compute_alpha_kernel`` and ``_cg_apply_alpha_kernel``; the
        PCG loop is launch-bound, so only the launch count changes.
        """

        self.cg_dAd[None] = ti.cast(0.0, ti.f64)
        if ti.static(ti.lang.impl.current_cfg().arch == ti.cuda):
            self.cg_dAd[None] = self._ki3_ordered_dot(self.cg_d, self.cg_Ad)
        else:
            for lane in range(CG_DOT_LANES):
                ti.atomic_add(
                    self.cg_dAd[None],
                    self._weighted_dot_lane_partial(self.cg_d, self.cg_Ad, lane),
                )
        self.cg_alpha[None] = 0.0
        if self.cg_dAd[None] <= 0.0:
            if self.cg_breakdown_code[None] == 0:
                self.cg_breakdown_dAd[None] = self.cg_dAd[None]
            self.cg_breakdown_code[None] = 1
        else:
            self.cg_alpha[None] = self.cg_rz[None] / self.cg_dAd[None]
        alpha = self.cg_alpha[None]
        for i, j, k in self.pressure:
            if self.obstacle[i, j, k] == 1:
                self.cg_r_old[i, j, k] = 0.0
                self.pressure[i, j, k] = 0.0
                self.cg_r[i, j, k] = 0.0
            else:
                self.cg_r_old[i, j, k] = self.cg_r[i, j, k]
                self.pressure[i, j, k] += alpha * self.cg_d[i, j, k]
                self.cg_r[i, j, k] -= alpha * self.cg_Ad[i, j, k]

    @ti.kernel
    def _cg_beta_step_kernel(self, use_mg_preconditioner: ti.template()):
        """One launch for dot(r, z), the flexible-CG numerator, beta and d.

        Same loops and scalar logic as ``_weighted_dot_to_field_kernel`` (r, z),
        ``_axpby_scalar_field_kernel`` (Ad = r - r_old), the (z, Ad) dot,
        ``_cg_compute_beta_kernel`` and ``_cg_update_direction_and_rz_kernel``.
        The multigrid branch is static so its loops stay top-level (parallel).
        """

        self.cg_rz_new[None] = ti.cast(0.0, ti.f64)
        if ti.static(ti.lang.impl.current_cfg().arch == ti.cuda):
            self.cg_rz_new[None] = self._ki3_ordered_dot(self.cg_r, self.cg_z)
        else:
            for lane in range(CG_DOT_LANES):
                ti.atomic_add(
                    self.cg_rz_new[None],
                    self._weighted_dot_lane_partial(self.cg_r, self.cg_z, lane),
                )
        if ti.static(use_mg_preconditioner != 0):
            for i, j, k in self.cg_Ad:
                if self.obstacle[i, j, k] == 1:
                    self.cg_Ad[i, j, k] = 0.0
                else:
                    self.cg_Ad[i, j, k] = (
                        ti.cast(1.0, ti.f64) * self.cg_r[i, j, k]
                        + ti.cast(-1.0, ti.f64) * self.cg_r_old[i, j, k]
                    )
            self.cg_beta_numerator[None] = ti.cast(0.0, ti.f64)
            if ti.static(ti.lang.impl.current_cfg().arch == ti.cuda):
                self.cg_beta_numerator[None] = self._ki3_ordered_dot(self.cg_z, self.cg_Ad)
            else:
                for lane in range(CG_DOT_LANES):
                    ti.atomic_add(
                        self.cg_beta_numerator[None],
                        self._weighted_dot_lane_partial(self.cg_z, self.cg_Ad, lane),
                    )
        self.cg_beta[None] = 0.0
        if self.cg_rz[None] <= 1.0e-300 or self.cg_rz_new[None] <= 0.0:
            self.cg_breakdown_code[None] = 2
        else:
            if ti.static(use_mg_preconditioner != 0):
                flexible_beta = self.cg_beta_numerator[None] / self.cg_rz[None]
                if flexible_beta < 0.0:
                    flexible_beta = 0.0
                self.cg_beta[None] = flexible_beta
            else:
                self.cg_beta[None] = self.cg_rz_new[None] / self.cg_rz[None]
        beta = self.cg_beta[None]
        for i, j, k in self.cg_d:
            if self.obstacle[i, j, k] == 1:
                self.cg_d[i, j, k] = 0.0
            else:
                self.cg_d[i, j, k] = self.cg_z[i, j, k] + beta * self.cg_d[i, j, k]
        self.cg_rz[None] = self.cg_rz_new[None]

    def _pressure_operator_graph_missing_or_stale_error(
        self,
        *,
        context: str,
    ) -> PressureSolveConvergenceError:
        """Build the one fail-closed contract for stale graph consumers."""

        return PressureSolveConvergenceError(
            f"{context} pressure operator graph is missing or stale",
            diagnostics={
                "stage": "pressure_nullspace_component_graph",
                "context": str(context),
                "reason": "operator_graph_missing_or_stale",
                "count": _diagnostic_int(
                    getattr(
                        self,
                        "_pressure_outlet_nullspace_source_component_count",
                        0,
                    )
                ),
                "capacity": int(self.pressure_component_capacity),
                "labels_converged": bool(
                    getattr(
                        self,
                        "last_pressure_component_labels_converged",
                        False,
                    )
                ),
                "component_overflow": bool(
                    getattr(
                        self,
                        "last_pressure_unreached_component_overflow",
                        False,
                    )
                ),
                "published_graph_context": getattr(
                    self,
                    "_pressure_outlet_nullspace_graph_context",
                    "",
                ),
            },
        )

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
        max_iters = max(1, int(iterations))
        relative_tolerance = max(0.0, float(tolerance))
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
        b_norm = sqrt(max(float(self._weighted_dot_kernel(self.cg_z, self.cg_z)), 0.0))
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
        r_norm = sqrt(max(float(self.cg_rr[None]), 0.0))
        initial_relative_residual = r_norm / b_norm
        self.last_cg_initial_relative_residual = initial_relative_residual
        self.last_cg_relative_residual = initial_relative_residual
        if r_norm <= relative_tolerance * b_norm:
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
        rz = float(self.cg_rz[None])
        if not math.isfinite(rz) or rz <= 1.0e-300:
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
            rz = float(self.cg_rz[None])
        if not math.isfinite(rz) or rz <= 0.0:
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
                r_norm = sqrt(max(float(self.cg_rr[None]), 0.0))
                relative_residual = r_norm / b_norm
                self.last_cg_relative_residual = relative_residual
                if relative_residual <= relative_tolerance:
                    self.last_cg_converged = True
                    return
                breakdown_code = int(self.cg_breakdown_code[None])
                if breakdown_code != 0:
                    self.last_cg_breakdown_dAd = float(self.cg_breakdown_dAd[None])
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
                        r_norm = sqrt(max(float(self.cg_rr[None]), 0.0))
                        relative_residual = r_norm / b_norm
                        self.last_cg_relative_residual = relative_residual
                        if relative_residual <= relative_tolerance:
                            self.last_cg_converged = True
                            return
                        apply_cg_preconditioner()
                        if project_pressure_components:
                            project_component_means(self.cg_z)
                        if remove_nullspace_mean:
                            raise NotImplementedError("closed domains without an outlet are not supported yet")
                        self._weighted_dot_to_field_kernel(self.cg_r, self.cg_z, self.cg_rz)
                        rz = float(self.cg_rz[None])
                        if math.isfinite(rz) and rz > 0.0:
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
                    r_norm = sqrt(max(float(self.cg_rr[None]), 0.0))
                    relative_residual = r_norm / b_norm
                    self.last_cg_relative_residual = relative_residual
                    if relative_residual <= relative_tolerance:
                        self.last_cg_converged = True
                        return
                    apply_cg_preconditioner()
                    if project_pressure_components:
                        project_component_means(self.cg_z)
                    if remove_nullspace_mean:
                        raise NotImplementedError("closed domains without an outlet are not supported yet")
                    self._weighted_dot_to_field_kernel(self.cg_r, self.cg_z, self.cg_rz)
                    rz = float(self.cg_rz[None])
                    if not math.isfinite(rz) or rz <= 0.0:
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

    def _confirm_pressure_poisson_fv_cg_exact_residual(
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
        residual_norm = sqrt(
            max(float(self._weighted_dot_kernel(self.cg_r, self.cg_r)), 0.0)
        )
        rhs_norm = sqrt(
            max(float(self._weighted_dot_kernel(self.cg_rhs, self.cg_rhs)), 0.0)
        )
        if rhs_norm <= 1.0e-30:
            exact_relative_residual = 0.0 if residual_norm <= 1.0e-30 else math.inf
        else:
            exact_relative_residual = residual_norm / rhs_norm
        self.last_cg_host_residual_checks += 1
        self.last_cg_exact_residual_confirmation_count += 1
        self.last_cg_recursive_relative_residual = self.last_cg_relative_residual
        self.last_cg_exact_relative_residual = exact_relative_residual
        self.last_cg_relative_residual = exact_relative_residual
        recurrence_reported_converged = bool(self.last_cg_converged)
        self.last_cg_converged = (
            math.isfinite(exact_relative_residual)
            and exact_relative_residual <= max(0.0, float(tolerance))
        )
        if self.last_cg_converged:
            self.last_cg_breakdown = ""
        elif recurrence_reported_converged:
            self.last_cg_breakdown = (
                "recursive residual convergence failed exact fine-grid confirmation"
            )
