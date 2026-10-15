"""The incompressible fluid solver of fastfsi: IncompressibleFluidSolver composes the topic mixins of
this package (advection, viscosity, velocity boundary conditions, obstacles, pressure projection,
conjugate gradient, multigrid, outlet reachability, pressure nullspace) into one Taichi
data-oriented object.  Split from a single-file solver by a mechanical extraction."""

from collections.abc import Mapping
from types import MappingProxyType
import math

import numpy as np
import taichi as ti

from fastfsi.fluid.advection import MomentumAdvection
from fastfsi.fluid.common import CG_DOT_GROUPS, CG_DOT_LANES, MUSCL_MOMENTUM_INITIAL_CFL_TARGET, MUSCL_MOMENTUM_MAX_CONSECUTIVE_STAGE_RETRIES, MUSCL_MOMENTUM_STAGE_CFL_LIMIT, MUSCL_MOMENTUM_STAGE_RETRY_SAFETY, OBSTACLE_NORMAL_VELOCITY_POLICIES, PREDICTOR_ADVECTION_CFL_FAIL_FAST_LIMIT, PRESSURE_INTERFACE_MATRIX_FV_CG_FORCE_REASON, PRESSURE_INTERFACE_MATRIX_FV_CG_REQUIRED_SOLVERS, PRESSURE_OUTLET_BACKFLOW_POLICIES, PressureSolveConvergenceError, _SSTWallDistanceBaseCacheKey, _SSTWallDistanceCacheKey, _finite_float, _pressure_poisson_coefficients, _run_pre_projection_velocity_transaction, _validated_count, _validated_pressure_solve_context
from fastfsi.fluid.constants import CG_PRECONDITIONER_CHOICES, PRESSURE_COMPONENT_CAPACITY, PRESSURE_OUTLET_AUTO_CLEANUP_ACCEPTANCE_MARGIN, PRESSURE_OUTLET_AUTO_CLEANUP_MAX_PASSES, PRESSURE_OUTLET_AUTO_CLEANUP_MIN_L2, PRESSURE_OUTLET_AUTO_CLEANUP_NO_REGRESSION_MARGIN, PRESSURE_OUTLET_AUTO_CLEANUP_TARGET_REDUCTION
from fastfsi.fluid.interface_constants import PRESSURE_INTERFACE_COUPLING_EXTRA_SLOTS
from fastfsi.fluid.obstacles import ObstacleMask
from fastfsi.fluid.outlet_budget import pressure_outlet_cleanup_iteration_budget
from fastfsi.fluid.outlet_reachability import OutletReachability
from fastfsi.fluid.pressure_conjugate_gradient import ConjugateGradientSolver
from fastfsi.fluid.pressure_multigrid import MultigridPreconditioner
from fastfsi.fluid.pressure_nullspace import PressureNullspace
from fastfsi.fluid.pressure_projection import PressureProjection
from fastfsi.fluid.time_stepping import minimum_admissible_physical_substep_s, physical_time_roundoff_tolerance_s, require_zero_remaining_physical_time_s
from fastfsi.fluid.velocity_boundary import VelocityBoundaryConditions
from fastfsi.fluid.viscosity import MomentumDiffusion
from fastfsi.grid.cartesian_grid import CartesianGrid
from fastfsi.grid.specification import FluidDomainSpecification
from fastfsi.io.checkpoint import Checkpointable
from fastfsi.simulation.taichi_runtime import TaichiRuntimeConfig, init_taichi


@ti.data_oriented
class IncompressibleFluidSolver(Checkpointable, MomentumAdvection, ObstacleMask, OutletReachability, ConjugateGradientSolver, MultigridPreconditioner, PressureNullspace, PressureProjection, VelocityBoundaryConditions, MomentumDiffusion):
    """Cell-centred pressure and scalars, MAC velocity, for coupled FSI."""

    # checkpoint component (fastfsi.io.checkpoint; state census): the velocity and the pressure (the CG
    # initial guess) and the host diagnostics and counters below -- every one created by the constructor, so a
    # freshly built solver loads any checkpoint of the same code.  Warm-start slots and increment accumulation are
    # not checkpointed: a run that uses them is refused at the checkpoint.
    CHECKPOINT_FIELDS = ("velocity", "pressure")
    CHECKPOINT_HOST = ("_last_momentum_advection_accepted_time_s", "_last_momentum_advection_cfl",
        "_last_momentum_advection_max_substep_cfl", "_last_momentum_advection_rejected_trial_count",
        "_last_momentum_advection_remaining_unadvanced_time_s", "_last_momentum_advection_requested_time_s",
        "_last_momentum_advection_scheme", "_last_momentum_advection_substeps",
        "_pressure_marker_nullspace_operator_apply_count",
        "_pressure_marker_nullspace_velocity_correction_apply_count", "_pressure_outlet_nullspace_component_count",
        "_pressure_outlet_nullspace_graph_context", "_pressure_outlet_nullspace_graph_valid",
        "_pressure_outlet_nullspace_source_component_count", "_pressure_outlet_operator_component_count",
        "_pressure_warmstart_valid", "_sst_last_diffusion_cfl", "_sst_last_diffusion_substeps",
        "_sst_last_momentum_diffusion_accepted_time_s", "_sst_last_momentum_diffusion_cfl",
        "_sst_last_momentum_diffusion_integrator", "_sst_last_momentum_diffusion_remaining_unadvanced_time_s",
        "_sst_last_momentum_diffusion_requested_time_s", "_sst_last_momentum_diffusion_substeps",
        "_sst_last_momentum_helmholtz_converged", "_sst_last_momentum_helmholtz_iterations",
        "_sst_last_momentum_helmholtz_iterations_total", "_sst_last_momentum_helmholtz_rejected_trial_count",
        "_sst_last_momentum_helmholtz_relative_residual", "_sst_last_transport_rejected_trial_count",
        "last_air_backed_cell_count", "last_air_backed_cell_volume_m3", "last_air_backed_component_count",
        "last_cg_bicgstab_iterations", "last_cg_breakdown", "last_cg_breakdown_dAd",
        "last_cg_closed_neumann_physical_component_budgets", "last_cg_closed_neumann_physical_rhs_integral",
        "last_cg_closed_neumann_physical_rhs_mean", "last_cg_closed_neumann_physical_rhs_measured",
        "last_cg_componentwise_mean_projection_count", "last_cg_converged", "last_cg_exact_relative_residual",
        "last_cg_exact_residual_confirmation_count", "last_cg_host_residual_checks",
        "last_cg_initial_relative_residual", "last_cg_iterations", "last_cg_mean_projection_count",
        "last_cg_multigrid_apply_count", "last_cg_multigrid_coarse_smooth_iterations",
        "last_cg_multigrid_post_smooth_iterations", "last_cg_multigrid_pre_smooth_iterations",
        "last_cg_multigrid_to_jacobi_fallback_count", "last_cg_operator_apply_count", "last_cg_pcg_iterations",
        "last_cg_preconditioner_effective", "last_cg_preconditioner_requested",
        "last_cg_recursive_relative_residual", "last_cg_relative_residual", "last_cg_restart_count",
        "last_cg_restart_count_measured", "last_cg_restart_policy", "last_cg_rhs_nonzero",
        "last_cg_unreached_set_mean_projection_count", "last_divergence_report_host_reads",
        "last_pressure_component_labels_converged", "last_pressure_filled_cell_count",
        "last_pressure_outlet_report_host_reads", "last_pressure_reachability_converged",
        "last_pressure_reachability_frontier_overflow", "last_pressure_reachability_reused",
        "last_pressure_reachability_sweeps", "last_pressure_unreached_cell_count",
        "last_pressure_unreached_component_count", "last_pressure_unreached_component_largest_cell_count",
        "last_pressure_unreached_component_overflow", "last_pressure_unreached_component_raw_count",
        "last_pressure_unreached_component_singleton_count", "last_pressure_unreached_component_small_cell_count",
        "last_pressure_unreached_component_small_count", "last_pressure_unreached_component_small_threshold_cells",
        "last_pressure_unreached_raw_cell_count", "last_project_cg_breakdown", "last_project_cg_breakdown_code",
        "last_project_cg_breakdown_count", "last_project_cg_breakdown_dAd",
        "last_project_cg_componentwise_mean_projection_count", "last_project_cg_converged_all",
        "last_project_cg_exact_relative_residual_max", "last_project_cg_exact_residual_confirmation_count",
        "last_project_cg_host_residual_checks", "last_project_cg_initial_relative_residual_max",
        "last_project_cg_iterations_max", "last_project_cg_iterations_total",
        "last_project_cg_mean_projection_count", "last_project_cg_multigrid_apply_count",
        "last_project_cg_multigrid_to_jacobi_fallback_count",
        "last_project_cg_nonzero_rhs_multigrid_to_jacobi_fallback_count",
        "last_project_cg_nonzero_rhs_preconditioner_effective",
        "last_project_cg_nonzero_rhs_preconditioner_requested", "last_project_cg_nonzero_rhs_project_calls",
        "last_project_cg_operator_apply_count", "last_project_cg_preconditioner_effective",
        "last_project_cg_preconditioner_requested", "last_project_cg_project_calls",
        "last_project_cg_relative_residual_max", "last_project_cg_restart_count",
        "last_project_cg_restart_count_measured", "last_project_cg_restart_policy",
        "last_project_cg_unreached_set_mean_projection_count",
        "last_project_pressure_nullspace_component_rhs_integral_max_abs",
        "last_project_pressure_nullspace_component_rhs_mean_max_abs",
        "last_project_pressure_nullspace_incompatible_component_count",
        "last_project_pressure_projection_physical_failure",
        "last_project_pressure_projection_physical_failure_action",
        "last_project_pressure_projection_physical_failure_reason", "last_reachability_revision",
        "last_reachability_valid", "last_row_cloud_orphan_cell_count", "last_row_cloud_orphan_component_count",
        "last_solid_band_enclosed_water_cells", "last_solid_band_interior_cells", "last_solid_band_marked_increment",
        "last_solid_band_mask_protected_cells", "last_solid_band_velocity_dirichlet_protected_cells",
        "last_unreached_cells_with_interface_coupling", "last_unreached_cells_with_interface_diagonal",
        "last_unreached_component_rhs_integral_max_abs", "last_unreached_component_rhs_mean_max_abs",
        "last_unreached_components_with_interface_hits", "last_unreached_divergence_raw_stats",
        "last_unreached_divergence_stats", "last_unreached_incompatible_component_count",
        "last_velocity_constraint_impulse_host_reads")

    def _refuse_unsupported_checkpoint(self):
        if any(getattr(self, "_pressure_warmstart_valid", ())):
            raise ValueError("a pressure warm-start slot is in use; warm-start slots are not checkpointed")

    def state_array(self, name):
        self._refuse_unsupported_checkpoint()
        return super().state_array(name)

    def state_host(self):
        self._refuse_unsupported_checkpoint()
        return super().state_host()

    def load_state_host(self, values):
        names, expected = set(values), set(self.CHECKPOINT_HOST)
        if names != expected:
            raise ValueError("checkpoint fluid host attributes differ from the solver's: missing %s, unknown %s"
                             % (sorted(expected - names), sorted(names - expected)))
        super().load_state_host(values)

    EXTERNAL_FACE_XMIN = 0
    EXTERNAL_FACE_XMAX = 1
    EXTERNAL_FACE_YMIN = 2
    EXTERNAL_FACE_YMAX = 3
    EXTERNAL_FACE_ZMIN = 4
    EXTERNAL_FACE_ZMAX = 5
    EXTERNAL_FACE_COUNT = 6

    _VELOCITY_DIRICHLET_BOUNDARY_AUTHORITIES = frozenset(
        {"legacy", "canonical"}
    )
    # A canonical component registry is not production-sealable until every
    # physical consumer has been migrated.  Keeping the full required set here
    # prevents a small public-surface subset from being mistaken for end-to-end
    # authority.  A consumer may be registered only through a dedicated
    # migration capability installed next to that consumer's real prepare path.
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMERS = frozenset(
        {
            "apply",
            "projection",
            "reference",
            "snapshot",
            "divergence",
            "reachability",
            "fv_operator",
            "gradient",
            "multigrid",
            "no_slip",
        }
    )
    # Opaque capabilities are installed only beside a migrated consumer's real
    # prepare path.  They are deliberately not user-constructible generation
    # tokens: the registrar verifies identity against these class-owned objects.
    # All physical consumers prepare the same generation before global sealing;
    # stale or partially prepared registries remain fail-closed.
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_APPLY_CAPABILITY = object()
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_REFERENCE_CAPABILITY = object()
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_NO_SLIP_CAPABILITY = object()
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_FV_OPERATOR_CAPABILITY = object()
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_GRADIENT_CAPABILITY = object()
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_SNAPSHOT_CAPABILITY = object()
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_DIVERGENCE_CAPABILITY = object()
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_REACHABILITY_CAPABILITY = object()
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_MULTIGRID_CAPABILITY = object()
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_PROJECTION_CAPABILITY = object()
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_PROJECTION_DEPENDENCIES = frozenset(
        {
            "apply",
            "divergence",
            "reachability",
            "fv_operator",
            "gradient",
            "multigrid",
        }
    )
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES = MappingProxyType(
        {
            "apply": _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_APPLY_CAPABILITY,
            "reference": _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_REFERENCE_CAPABILITY,
            "no_slip": _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_NO_SLIP_CAPABILITY,
            "fv_operator": _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_FV_OPERATOR_CAPABILITY,
            "gradient": _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_GRADIENT_CAPABILITY,
            "snapshot": _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_SNAPSHOT_CAPABILITY,
            "divergence": _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_DIVERGENCE_CAPABILITY,
            "reachability": _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_REACHABILITY_CAPABILITY,
            "multigrid": _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_MULTIGRID_CAPABILITY,
            "projection": _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_PROJECTION_CAPABILITY,
        }
    )

    DEFAULT_MULTIGRID_CYCLES = 12
    DEFAULT_NONUNIFORM_MULTIGRID_CYCLES = 24
    # project(warm_start_slot=...) stash slot count: enough for a caller with
    # several distinct project() call sites per physical step (e.g. predictor
    # substeps + dirichlet-consistency + post-solid) to each keep an
    # independent CG initial-guess history instead of colliding on one
    # shared buffer. See invalidate_pressure_warmstart() for scoping.
    PRESSURE_WARMSTART_SLOT_COUNT = 5

    no_free_face_isolated_cells_solid = False

    MG_JACOBI_SWEEP_PAIRS_PER_LAUNCH = 4
    # batched multigrid prolongation launches (CUDA results identical; a warm 3-D run 953 s -> 780 s); False keeps
    # the unbatched launches for comparisons
    BATCH_MG_PROLONGATION = True

    @staticmethod
    def _validate_device_cartesian_grid_geometry(grid: CartesianGrid) -> None:
        """Require host geometry to survive the solver's f32 device registry."""

        axes = (
            (
                "x",
                grid.cell_widths_x_m,
                grid.cell_faces_x_m,
                grid.cell_centers_x_m,
                grid.center_distances_x_m,
            ),
            (
                "y",
                grid.cell_widths_y_m,
                grid.cell_faces_y_m,
                grid.cell_centers_y_m,
                grid.center_distances_y_m,
            ),
            (
                "z",
                grid.cell_widths_z_m,
                grid.cell_faces_z_m,
                grid.cell_centers_z_m,
                grid.center_distances_z_m,
            ),
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            for axis, widths, faces, centers, center_distances in axes:
                for quantity_name, values in (
                    ("cell widths", widths),
                    ("center distances", center_distances),
                ):
                    device_values = np.asarray(values, dtype=np.float32)
                    if not bool(
                        np.all(np.isfinite(device_values))
                        and np.all(device_values > 0.0)
                    ):
                        raise ValueError(
                            f"CartesianGrid {axis}-axis {quantity_name} must "
                            "remain finite and strictly positive in f32 device "
                            "geometry"
                        )

                device_faces = np.asarray(faces, dtype=np.float32)
                device_centers = np.asarray(centers, dtype=np.float32)
                if not bool(
                    np.all(np.isfinite(device_faces))
                    and np.all(np.isfinite(device_centers))
                ):
                    raise ValueError(
                        f"CartesianGrid {axis}-axis faces and centers must remain "
                        "finite in f32 device geometry"
                    )
                if bool(np.any(np.diff(device_faces) <= 0.0)) or bool(
                    np.any(np.diff(device_centers) <= 0.0)
                ):
                    raise ValueError(
                        f"CartesianGrid {axis}-axis faces and centers must remain "
                        "strictly ordered in f32 device geometry"
                    )

    @property
    def pressure_component_capacity(self):
        """Immutable legacy storage bound; enabled subclasses may size to grid."""
        return PRESSURE_COMPONENT_CAPACITY

    def __init__(
        self,
        spec: FluidDomainSpecification,
        runtime: TaichiRuntimeConfig | None = None,
        *,
        multigrid_semi_coarsening: bool = False,
    ):
        if not isinstance(multigrid_semi_coarsening, bool):
            raise TypeError("multigrid_semi_coarsening must be a bool")
        # Per-axis multigrid coarsening (halve only axes with >= 4 cells): the legacy
        # rule stops at 2 levels on thin spanwise grids (nx=4), leaving a coarse level
        # with thousands of cells that 24 Jacobi sweeps cannot solve.
        self.multigrid_semi_coarsening = bool(multigrid_semi_coarsening)
        self._validate_device_cartesian_grid_geometry(spec.cartesian_grid)
        init_taichi(runtime)
        self.spec = spec
        self.grid = spec.cartesian_grid
        self.nx, self.ny, self.nz = self.grid.grid_nodes
        if self.grid.is_uniform:
            self.dx, self.dy, self.dz = self.grid.uniform_spacing_m
        else:
            self.dx, self.dy, self.dz = (
                min(self.grid.cell_widths_x_m),
                min(self.grid.cell_widths_y_m),
                min(self.grid.cell_widths_z_m),
            )
        self.bounds_min = tuple(float(v) for v in self.grid.bounds_min_m)
        self.bounds_max = tuple(float(v) for v in self.grid.bounds_max_m)
        self.rho = float(spec.density_kgm3)
        self.mu = float(spec.viscosity_pa_s)
        self.dt = float(spec.dt_s)

        shape = self.grid.grid_nodes
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.velocity_prev = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        # Scratch ownership for the conservative SSP-RK2 momentum transport.
        # This is deliberately separate from velocity_prev: the latter is the
        # current RK stage while this field must retain q^n through both stages.
        self.velocity_transport_base = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        # Reconstruct once per RK stage.  Keeping slope/extremum metadata in
        # fields avoids inlining the five-point smooth-extremum detector into
        # every x/y/z face kernel, which otherwise makes Taichi JIT impractical.
        self.muscl_velocity_slope = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=shape
        )
        self.muscl_velocity_smooth_mask = ti.Matrix.field(
            3, 3, dtype=ti.i32, shape=shape
        )
        self.muscl_velocity_extension_slope = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=shape
        )
        self.muscl_momentum_flux_x = ti.Vector.field(
            3, dtype=ti.f32, shape=(self.nx + 1, self.ny, self.nz)
        )
        self.muscl_momentum_flux_y = ti.Vector.field(
            3, dtype=ti.f32, shape=(self.nx, self.ny + 1, self.nz)
        )
        self.muscl_momentum_flux_z = ti.Vector.field(
            3, dtype=ti.f32, shape=(self.nx, self.ny, self.nz + 1)
        )
        # Native backward-MAC momentum control volumes.  Each packed velocity
        # row owns three different dual cells, so neither its volume nor its
        # transverse face flux can be represented by one scalar-cell registry.
        self.muscl_momentum_dual_volume_m3 = ti.field(
            dtype=ti.f32, shape=(self.nx, self.ny, self.nz, 3)
        )
        self.muscl_momentum_volume_flux_x_half_m3_s = ti.field(
            dtype=ti.f32, shape=(self.nx + 1, self.ny, self.nz, 3, 2)
        )
        self.muscl_momentum_volume_flux_y_half_m3_s = ti.field(
            dtype=ti.f32, shape=(self.nx, self.ny + 1, self.nz, 3, 2)
        )
        self.muscl_momentum_volume_flux_z_half_m3_s = ti.field(
            dtype=ti.f32, shape=(self.nx, self.ny, self.nz + 1, 3, 2)
        )
        self.muscl_normal_velocity_x = ti.field(
            dtype=ti.f32, shape=(self.nx + 1, self.ny, self.nz)
        )
        self.muscl_normal_velocity_y = ti.field(
            dtype=ti.f32, shape=(self.nx, self.ny + 1, self.nz)
        )
        self.muscl_normal_velocity_z = ti.field(
            dtype=ti.f32, shape=(self.nx, self.ny, self.nz + 1)
        )
        self.saved_velocity = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.saved_velocity_prev = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        # Exact external-face data is stored independently from the compact
        # backward-MAC velocity rows.  Each axis has a two-sided plane field;
        # this avoids a six-times-volume allocation and preserves independent
        # values for each incident face at edges and corners.
        self.external_velocity_boundary_x_face_active_component_mask = ti.field(
            dtype=ti.i32,
            shape=(2, self.ny, self.nz),
        )
        self.external_velocity_boundary_x_face_value_mps = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(2, self.ny, self.nz),
        )
        self.external_velocity_boundary_y_face_active_component_mask = ti.field(
            dtype=ti.i32,
            shape=(2, self.nx, self.nz),
        )
        self.external_velocity_boundary_y_face_value_mps = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(2, self.nx, self.nz),
        )
        self.external_velocity_boundary_z_face_active_component_mask = ti.field(
            dtype=ti.i32,
            shape=(2, self.nx, self.ny),
        )
        self.external_velocity_boundary_z_face_value_mps = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(2, self.nx, self.ny),
        )
        # Strong-coupling rollback owns the complete physical boundary map.
        # Keep the directed planes on-device beside the flow snapshot so a
        # rejected trial never restores velocity/pressure against a different
        # external-face profile.
        self.saved_external_velocity_boundary_x_face_active_component_mask = ti.field(
            dtype=ti.i32,
            shape=(2, self.ny, self.nz),
        )
        self.saved_external_velocity_boundary_x_face_value_mps = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(2, self.ny, self.nz),
        )
        self.saved_external_velocity_boundary_y_face_active_component_mask = ti.field(
            dtype=ti.i32,
            shape=(2, self.nx, self.nz),
        )
        self.saved_external_velocity_boundary_y_face_value_mps = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(2, self.nx, self.nz),
        )
        self.saved_external_velocity_boundary_z_face_active_component_mask = ti.field(
            dtype=ti.i32,
            shape=(2, self.nx, self.ny),
        )
        self.saved_external_velocity_boundary_z_face_value_mps = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(2, self.nx, self.ny),
        )
        self.pressure = ti.field(dtype=ti.f64, shape=shape)
        self.saved_pressure = ti.field(dtype=ti.f64, shape=shape)
        # SST-2003 state is allocated beside the primary flow fields so the
        # closure remains Taichi-resident and participates in the same
        # strong-coupling lifecycle.  The default model stays laminar; these
        # fields are inert until configure_sst_2003() is called.
        self.turbulence_model = "laminar"
        # Fluent's near-wall correlation is opt-in.  Keep resolved walls as
        # the default so existing SST cases retain their exact path.
        self._sst_near_wall_treatment = "resolved"
        self._sst_no_slip_domain_walls = (
            False,
            False,
            False,
            False,
            False,
            False,
        )
        self._sst_inlet_turbulent_kinetic_energy_m2_s2 = 0.0
        self._sst_inlet_specific_dissipation_rate_s = 0.0
        self._sst_backflow_turbulent_kinetic_energy_m2_s2 = 0.0
        self._sst_backflow_specific_dissipation_rate_s = 0.0
        self._sst_inlet_face_code = self.EXTERNAL_FACE_ZMAX
        self._sst_outlet_face_code = self.EXTERNAL_FACE_ZMIN
        self._sst_wall_distance_valid = False
        self._sst_saved_wall_distance_valid = False
        self._sst_wall_distance_cache_key: _SSTWallDistanceCacheKey | None = None
        self._sst_wall_distance_base_cache_key: (
            _SSTWallDistanceBaseCacheKey | None
        ) = None
        self._sst_saved_wall_distance_cache_key: _SSTWallDistanceCacheKey | None = None
        self._sst_saved_no_slip_domain_walls = self._sst_no_slip_domain_walls
        self._sst_saved_no_slip_domain_wall_mask = 0
        self._sst_last_diffusion_substeps = 0
        self._sst_last_diffusion_cfl = 0.0
        self._sst_last_transport_rejected_trial_count = 0
        self._sst_last_momentum_diffusion_integrator = "none"
        self._sst_last_momentum_diffusion_substeps = 0
        self._sst_last_momentum_diffusion_cfl = 0.0
        self._sst_last_momentum_diffusion_requested_time_s = 0.0
        self._sst_last_momentum_diffusion_accepted_time_s = 0.0
        self._sst_last_momentum_diffusion_remaining_unadvanced_time_s = 0.0
        self._sst_last_momentum_helmholtz_converged = True
        self._sst_last_momentum_helmholtz_iterations = 0
        self._sst_last_momentum_helmholtz_iterations_total = 0
        self._sst_last_momentum_helmholtz_relative_residual = 0.0
        self._sst_last_momentum_helmholtz_rejected_trial_count = 0
        self._last_momentum_advection_scheme = "none"
        self._last_momentum_advection_substeps = 0
        self._last_momentum_advection_cfl = 0.0
        self._last_momentum_advection_max_substep_cfl = 0.0
        self._last_momentum_advection_rejected_trial_count = 0
        self._last_momentum_advection_requested_time_s = 0.0
        self._last_momentum_advection_accepted_time_s = 0.0
        self._last_momentum_advection_remaining_unadvanced_time_s = 0.0
        self._transport_max_automatic_substeps = 4096
        self._sst_max_automatic_substeps = 4096
        self.sst_turbulent_kinetic_energy = ti.field(dtype=ti.f32, shape=shape)
        self.sst_specific_dissipation_rate = ti.field(dtype=ti.f32, shape=shape)
        self.sst_eddy_viscosity_pa_s = ti.field(dtype=ti.f32, shape=shape)
        self.sst_wall_distance_m = ti.field(dtype=ti.f32, shape=shape)
        self.sst_turbulent_kinetic_energy_prev = ti.field(dtype=ti.f32, shape=shape)
        self.sst_specific_dissipation_rate_prev = ti.field(dtype=ti.f32, shape=shape)
        self.sst_turbulent_kinetic_energy_next = ti.field(dtype=ti.f32, shape=shape)
        self.sst_specific_dissipation_rate_next = ti.field(dtype=ti.f32, shape=shape)
        self.sst_turbulent_kinetic_energy_transport_base = ti.field(
            dtype=ti.f32, shape=shape
        )
        self.sst_specific_dissipation_rate_transport_base = ti.field(
            dtype=ti.f32, shape=shape
        )
        self.muscl_sst_k_slope = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.muscl_sst_k_smooth_mask = ti.Vector.field(
            3, dtype=ti.i32, shape=shape
        )
        self.muscl_sst_k_extension_slope = ti.Vector.field(
            3, dtype=ti.f32, shape=shape
        )
        self.muscl_sst_omega_slope = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.muscl_sst_omega_smooth_mask = ti.Vector.field(
            3, dtype=ti.i32, shape=shape
        )
        self.muscl_sst_omega_extension_slope = ti.Vector.field(
            3, dtype=ti.f32, shape=shape
        )
        self.sst_wall_distance_next_m = ti.field(dtype=ti.f32, shape=shape)
        self.sst_nearest_obstacle_cell = ti.Vector.field(3, dtype=ti.i32, shape=shape)
        self.sst_nearest_obstacle_cell_next = ti.Vector.field(
            3, dtype=ti.i32, shape=shape
        )
        self.sst_no_slip_domain_wall_mask = ti.field(dtype=ti.i32, shape=())
        self.sst_near_wall_correlation_enabled = ti.field(dtype=ti.i32, shape=())
        # Laminar viscous patches between two fluid cells keep their shared
        # exchange across a canonical hard face (0 = legacy: treated as a wall
        # at half a cell with the hard row's target).  A canonical row is an
        # algebraic constraint at its own MAC coordinate, not a wall between the
        # two fluid half-cells; the legacy rule made affine fields diffuse.
        self.laminar_canonical_fluid_patch_exchange = ti.field(dtype=ti.i32, shape=())
        self.sst_blending_f1 = ti.field(dtype=ti.f32, shape=shape)
        self.sst_blending_f2 = ti.field(dtype=ti.f32, shape=shape)
        self.sst_sigma_k = ti.field(dtype=ti.f32, shape=shape)
        self.sst_sigma_omega = ti.field(dtype=ti.f32, shape=shape)
        self.sst_beta = ti.field(dtype=ti.f32, shape=shape)
        self.sst_gamma = ti.field(dtype=ti.f32, shape=shape)
        self.sst_strain_rate_magnitude_s = ti.field(dtype=ti.f32, shape=shape)
        # ``velocity`` is a compact backward-MAC field: its x/y/z entries
        # live on three different negative faces.  Scalar-cell operators reuse
        # one serial scratch reconstruction: current velocity for SST strain,
        # then frozen ``velocity_prev`` for the explicit transpose stress.
        self.sst_cell_center_velocity_mps = ti.Vector.field(
            3, dtype=ti.f32, shape=shape
        )
        self.sst_momentum_transpose_gradient_s = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=shape
        )
        # This serial scratch is rebuilt as transpose-stress divergence before
        # the momentum update.  During coefficient assembly it temporarily
        # stores exact same-cell MAC normal derivatives, avoiding another
        # persistent field in this SNode-budget-constrained solver.
        self.sst_momentum_transpose_divergence_cell_mps2 = ti.Vector.field(
            3, dtype=ti.f32, shape=shape
        )
        # For each backward-MAC storage row and physical interface normal,
        # cache the xyz bits whose wall targets passed the strict canonical or
        # legacy ownership contract.  Keeping only masks (not copied vectors)
        # limits the persistent SST scratch cost while the coefficient kernel
        # can safely read the authoritative target field without re-running a
        # deeply inlined provenance graph.
        self.sst_obstacle_interface_wall_target_component_mask = ti.Vector.field(
            3, dtype=ti.i32, shape=shape
        )
        self.saved_sst_turbulent_kinetic_energy = ti.field(dtype=ti.f32, shape=shape)
        self.saved_sst_specific_dissipation_rate = ti.field(dtype=ti.f32, shape=shape)
        self.saved_sst_eddy_viscosity_pa_s = ti.field(dtype=ti.f32, shape=shape)
        self.saved_sst_wall_distance_m = ti.field(dtype=ti.f32, shape=shape)
        self.saved_fsi_pressure = ti.field(dtype=ti.f64, shape=shape)
        self.sst_reduction_max_diffusivity_m2_s = ti.field(dtype=ti.f64, shape=())
        self.sst_reduction_min_k = ti.field(dtype=ti.f64, shape=())
        self.sst_reduction_max_k = ti.field(dtype=ti.f64, shape=())
        self.sst_reduction_min_omega = ti.field(dtype=ti.f64, shape=())
        self.sst_reduction_max_omega = ti.field(dtype=ti.f64, shape=())
        self.sst_reduction_max_mu_t = ti.field(dtype=ti.f64, shape=())
        self.sst_reduction_nonfinite_or_nonpositive_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.sst_reduction_eddy_viscosity_cap_count = ti.field(dtype=ti.i32, shape=())
        self.sst_reduction_wall_omega_guard_count = ti.field(dtype=ti.i32, shape=())
        self.sst_reduction_wall_omega_target_max_s = ti.field(dtype=ti.f64, shape=())
        self.sst_reduction_explicit_candidate_invalid_count = ti.field(
            dtype=ti.i32, shape=()
        )
        self.sst_reduction_explicit_candidate_min_k = ti.field(
            dtype=ti.f64, shape=()
        )
        self.sst_reduction_explicit_candidate_min_omega = ti.field(
            dtype=ti.f64, shape=()
        )
        # Strong-coupling rollback includes the moving interface topology. The
        # previous composite obstacle controls fresh-fluid reconstruction;
        # dynamic volume/carve fields control the next composite mask.
        self.saved_obstacle = ti.field(dtype=ti.i32, shape=shape)
        self.saved_interface_air_cell = ti.field(dtype=ti.i32, shape=shape)
        self.saved_dynamic_solid_volume_obstacle = ti.field(
            dtype=ti.i32, shape=shape
        )

        self.saved_dynamic_solid_volume_external_carve = ti.field(
            dtype=ti.i32, shape=shape
        )
        self.saved_fresh_fluid_cell = ti.field(dtype=ti.i32, shape=shape)
        # Cleanup snapshot must NOT share buffers with save_state/restore_state
        # (FSI strong-coupling base state) — sharing clobbered the Picard base
        # mid-trial when pressure-outlet auto-cleanup ran inside project().
        self.cleanup_saved_velocity = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.cleanup_saved_pressure = ti.field(dtype=ti.f64, shape=shape)
        self.cleanup_saved_outlet_flux = ti.Vector.field(3, dtype=ti.f64, shape=())
        self.cleanup_saved_pressure_interface_matrix_rhs = ti.field(
            dtype=ti.f64, shape=shape
        )
        self.cleanup_saved_pressure_interface_projection_divergence_s = ti.field(
            dtype=ti.f64, shape=shape
        )
        self.fsi_pressure = ti.field(dtype=ti.f64, shape=shape)
        self.pressure_tmp = ti.field(dtype=ti.f64, shape=shape)
        self.pressure_accum = ti.field(dtype=ti.f64, shape=shape)
        # Step-level pressure carry for increment-mode re-projections
        # (project(accumulate_pressure_into_previous=True)): holds the
        # step's accumulated pressure while a re-projection solves its own
        # increment from a clean zero field.
        self.pressure_step_carry = ti.field(dtype=ti.f64, shape=shape)
        # project(warm_start_slot=...) CG initial-guess stash (opt-in; slot
        # -1, the default everywhere, never touches these fields or adds a
        # kernel launch). Each slot is a full pressure-shaped f64 buffer plus
        # a Python-side validity flag so a caller with several logically
        # distinct project() call sites per step can warm-start each one
        # from its own history without cross-contamination. Warm-starting
        # only changes the CG solve's INITIAL GUESS: with tolerance-based
        # convergence exits the converged pressure field is unchanged: only
        # the iteration count to reach it changes. A budget-CAPPED
        # (non-converging) solve would instead become initial-guess
        # dependent, so pair warm-starting with tolerance-based exits.
        self.pressure_warmstart_slot0 = ti.field(dtype=ti.f64, shape=shape)
        self.pressure_warmstart_slot1 = ti.field(dtype=ti.f64, shape=shape)
        self.pressure_warmstart_slot2 = ti.field(dtype=ti.f64, shape=shape)
        self.pressure_warmstart_slot3 = ti.field(dtype=ti.f64, shape=shape)
        self.pressure_warmstart_slot4 = ti.field(dtype=ti.f64, shape=shape)
        self._pressure_warmstart_slots = (
            self.pressure_warmstart_slot0,
            self.pressure_warmstart_slot1,
            self.pressure_warmstart_slot2,
            self.pressure_warmstart_slot3,
            self.pressure_warmstart_slot4,
        )
        assert len(self._pressure_warmstart_slots) == self.PRESSURE_WARMSTART_SLOT_COUNT
        self._pressure_warmstart_valid = [False] * len(self._pressure_warmstart_slots)
        # Quasi-2D filter (opt-in, -1 = off): after every predictor step and
        # pressure projection the velocity component along this axis is set
        # to zero.  The exact solution of a plane-strain problem between two
        # symmetry planes has no velocity along that axis; keeping it exactly
        # zero removes residual spanwise noise from every interpolated
        # boundary target.  Set by the case, never by the solver itself.
        self.quasi_2d_velocity_axis = -1
        self.divergence = ti.field(dtype=ti.f32, shape=shape)
        self.pressure_interface_projection_divergence_s = ti.field(
            dtype=ti.f64,
            shape=shape,
        )
        self.volume_source_s = ti.field(dtype=ti.f32, shape=shape)
        self.saved_volume_source_s = ti.field(dtype=ti.f32, shape=shape)
        self.pressure_interface_matrix_diagonal = ti.field(dtype=ti.f64, shape=shape)
        self.pressure_interface_matrix_rhs = ti.field(dtype=ti.f64, shape=shape)
        # Full-operator storage validation is a solve precondition, including
        # the all-outlet-reachable path where no nullspace graph is built.
        # Components: diagonal, RHS, row endpoint, row transmissibility,
        # legacy active-count, legacy endpoint, legacy transmissibility,
        # cell-local diagonal/edge provenance, and rank-one row support.
        self.pressure_interface_preflight_invalid_counts = ti.Vector.field(
            9,
            dtype=ti.i32,
            shape=(),
        )
        self.pressure_interface_preflight_incident_diagonal_integral = ti.field(
            dtype=ti.f64,
            shape=shape,
        )
        self.pressure_interface_preflight_incident_term_count = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self.pressure_interface_coupling_active = ti.field(dtype=ti.i32, shape=shape)
        self.pressure_interface_coupling_neighbor = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=shape,
        )
        self.pressure_interface_coupling_coefficient = ti.field(
            dtype=ti.f64,
            shape=shape,
        )
        extra_coupling_shape = shape + (PRESSURE_INTERFACE_COUPLING_EXTRA_SLOTS,)
        self.pressure_interface_coupling_extra_neighbor = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=extra_coupling_shape,
        )
        self.pressure_interface_coupling_extra_coefficient = ti.field(
            dtype=ti.f64,
            shape=extra_coupling_shape,
        )
        self.pressure_interface_row_diagonal_density = ti.field(
            dtype=ti.f64,
            shape=shape,
        )
        self.pressure_interface_row_capacity = int(self.nx * self.ny * self.nz)
        self.pressure_interface_row_count = ti.field(dtype=ti.i32, shape=())
        self.pressure_interface_row_owner = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=self.pressure_interface_row_capacity,
        )
        self.pressure_interface_row_neighbor = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=self.pressure_interface_row_capacity,
        )
        self.pressure_interface_row_transmissibility = ti.field(
            dtype=ti.f64,
            shape=self.pressure_interface_row_capacity,
        )
        # count == 0 retains an explicitly supplied legacy two-point row.
        self.pressure_interface_row_support_count = ti.field(
            dtype=ti.i32, shape=self.pressure_interface_row_capacity,
        )
        self.pressure_interface_row_support_cell = ti.Vector.field(
            3, dtype=ti.i32, shape=(self.pressure_interface_row_capacity, 9),
        )
        self.pressure_interface_row_support_coefficient = ti.field(
            dtype=ti.f64, shape=(self.pressure_interface_row_capacity, 9),
        )
        self.pressure_interface_row_source_node = ti.Vector.field(
            3, dtype=ti.i32, shape=self.pressure_interface_row_capacity,
        )
        self.base_obstacle = ti.field(dtype=ti.i32, shape=shape)
        # Physical solid volume reconstructed from the moving MPM body.  Keep
        # this separate from both the static base geometry and the transient
        # interface node classification so a narrow interface search does not make
        # the interior porous or inflate the projected obstacle envelope.
        self.dynamic_solid_volume_obstacle = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        # Persist the latest external-IB carve across the solid-volume refresh
        # that happens after an MPM step.  The following predictor therefore
        # sees the same immersed-fluid owner topology as the last projection,
        # rather than a temporarily resealed thick obstacle.
        self.dynamic_solid_volume_external_carve = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self.pressure_outlet_reachable = ti.field(dtype=ti.i32, shape=shape)
        self.pressure_outlet_reachable_next = ti.field(dtype=ti.i32, shape=shape)
        # Sparse, level-synchronous pressure-outlet flood frontiers.  Each
        # active cell can enter one frontier at most once, so exact cell count
        # is both sufficient and the fail-closed capacity bound.
        self.pressure_reachability_frontier_capacity = int(
            self.nx * self.ny * self.nz
        )
        self.pressure_reachability_frontier_a = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=self.pressure_reachability_frontier_capacity,
        )
        self.pressure_reachability_frontier_b = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=self.pressure_reachability_frontier_capacity,
        )
        self.pressure_reachability_frontier_a_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.pressure_reachability_frontier_b_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.pressure_reachability_frontier_overflow = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        # Device-side count of non-empty BFS levels expanded by the batched
        # flood; mirrors the host ``frontier_levels`` counter exactly.
        self.pressure_reachability_frontier_levels = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.pressure_unreached_component_label = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        # Exact outlet pressure-operator components are independent from the
        # physical six-neighbour reachability flood above.  In particular,
        # active interface rows may join flood components (or join one to the
        # outlet-anchored region), so operator-nullspace consumers must never
        # reuse the physical flood labels as capacity-indexed solve labels.
        self.pressure_outlet_operator_component_label = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self.pressure_outlet_operator_raw_component_cell_count = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self.pressure_outlet_operator_raw_component_to_compact = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self.pressure_outlet_operator_raw_component_anchored = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self.pressure_outlet_operator_raw_component_diagonal_excess_integral = (
            ti.field(dtype=ti.f64, shape=shape)
        )
        self.pressure_outlet_operator_raw_component_diagonal_scale_integral = (
            ti.field(dtype=ti.f64, shape=shape)
        )
        self.pressure_outlet_operator_raw_component_diagonal_term_count = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self.pressure_outlet_operator_raw_component_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.pressure_outlet_operator_empty_singleton_component_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.pressure_outlet_operator_active_unanchored_component_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.unreached_raw_component_cell_count = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self.unreached_component_compact_raw_label = ti.field(
            dtype=ti.i32,
            shape=self.pressure_component_capacity,
        )
        self.report_unreached_component_raw_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_unreached_component_largest_cell_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_unreached_component_singleton_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_unreached_component_small_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_unreached_component_small_cell_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.pressure_reachability_barrier = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self.reachability_checksum_snapshot = ti.Vector.field(
            4,
            dtype=ti.i64,
            shape=(),
        )
        self.cg_unreached_component_sum = ti.field(
            dtype=ti.f64,
            shape=self.pressure_component_capacity,
        )
        self.cg_unreached_component_volume = ti.field(
            dtype=ti.f64,
            shape=self.pressure_component_capacity,
        )
        # The outlet flood components remain physical/report state.  The exact
        # fine pressure operator may additionally connect or anchor those
        # components through interface rows, so FV-CG keeps a component-level
        # solve graph instead of mutating the physical flood labels.
        self.pressure_solve_component_parent = ti.field(
            dtype=ti.i32,
            shape=self.pressure_component_capacity,
        )
        self.pressure_solve_component_anchored = ti.field(
            dtype=ti.i32,
            shape=self.pressure_component_capacity,
        )
        self.pressure_solve_component_diagonal_excess_integral = ti.field(
            dtype=ti.f64,
            shape=self.pressure_component_capacity,
        )
        self.pressure_solve_component_diagonal_scale_integral = ti.field(
            dtype=ti.f64,
            shape=self.pressure_component_capacity,
        )
        self.pressure_solve_component_diagonal_term_count = ti.field(
            dtype=ti.i32,
            shape=self.pressure_component_capacity,
        )
        self.pressure_solve_component_count = ti.field(dtype=ti.i32, shape=())
        self.pressure_solve_component_invalid = ti.field(dtype=ti.i32, shape=())
        # declared air-backed closure regions: per-cell air tag,
        # fixed-capacity unreached-component selection mask (written by
        # the marker-side far-side seed scan), and conversion reports.
        self.interface_air_cell = ti.field(dtype=ti.i32, shape=shape)
        self.air_component_selected = ti.field(
            dtype=ti.i32,
            shape=self.pressure_component_capacity,
        )
        self.air_component_converted = ti.field(
            dtype=ti.i32,
            shape=self.pressure_component_capacity,
        )
        self.report_air_backed_cells = ti.field(dtype=ti.i32, shape=())
        self.report_air_backed_components = ti.field(dtype=ti.i32, shape=())
        self.report_air_backed_cell_volume_m3 = ti.field(
            dtype=ti.f64,
            shape=(),
        )
        self.row_cloud_component_cell_count = ti.field(
            dtype=ti.i32,
            shape=self.pressure_component_capacity,
        )
        self.row_cloud_component_touches_row = ti.field(
            dtype=ti.i32,
            shape=self.pressure_component_capacity,
        )
        self.row_cloud_component_touches_protected = ti.field(
            dtype=ti.i32,
            shape=self.pressure_component_capacity,
        )
        self.row_cloud_component_selected = ti.field(
            dtype=ti.i32,
            shape=self.pressure_component_capacity,
        )
        self.report_row_cloud_orphan_cells = ti.field(dtype=ti.i32, shape=())
        self.report_row_cloud_orphan_components = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        # R2-H1 observability: overlap between the flood-unreachable anchoring
        # set and rows touched by pressure-interface matrix terms. The cell-hit
        # grid deduplicates coupling-edge hits (a row can be the target of
        # several edges); the fixed-capacity array deduplicates per component.
        self.unreached_interface_cell_hit = ti.field(dtype=ti.i32, shape=shape)
        self.unreached_interface_component_hit = ti.field(
            dtype=ti.i32,
            shape=self.pressure_component_capacity,
        )
        self.report_unreached_interface_diagonal_cells = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_unreached_interface_coupling_cells = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_unreached_interface_component_hits = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.fv_diag = ti.field(dtype=ti.f64, shape=shape)
        self.cg_r = ti.field(dtype=ti.f64, shape=shape)
        self.cg_rhs = ti.field(dtype=ti.f64, shape=shape)
        self.cg_z = ti.field(dtype=ti.f64, shape=shape)
        self.cg_d = ti.field(dtype=ti.f64, shape=shape)
        self.cg_Ad = ti.field(dtype=ti.f64, shape=shape)
        self.cg_r_old = ti.field(dtype=ti.f64, shape=shape)
        self.cg_mg_rhs = ti.field(dtype=ti.f64, shape=shape)
        self.cg_mg_residual = ti.field(dtype=ti.f64, shape=shape)
        self.bicgstab_s = ti.field(dtype=ti.f64, shape=shape)
        self.bicgstab_t = ti.field(dtype=ti.f64, shape=shape)
        # Marker-preserving pressure projection is opt-in.  Keep its three
        # full-grid f64 face vectors and audit scalars lazy so projector=None
        # retains the legacy memory footprint as well as the legacy kernels.
        self.pressure_velocity_actuation_weight = None
        self.pressure_velocity_raw_correction = None
        self.pressure_velocity_projected_correction = None
        self.pressure_velocity_actuation_invalid_count = None
        self.pressure_velocity_correction_invalid_count = None
        self._pressure_velocity_nullspace_resources_allocated = False
        self._pressure_velocity_nullspace_resource_bytes = 0
        self.pressure_velocity_actuation_generation = 0
        self._pressure_velocity_nullspace_transaction_active = False
        self._pressure_marker_nullspace_operator_apply_count = 0
        self._pressure_marker_nullspace_velocity_correction_apply_count = 0
        self.cg_rz = ti.field(dtype=ti.f64, shape=())
        self.cg_rz_new = ti.field(dtype=ti.f64, shape=())
        self.cg_dAd = ti.field(dtype=ti.f64, shape=())
        self.cg_rr = ti.field(dtype=ti.f64, shape=())
        self.cg_beta_numerator = ti.field(dtype=ti.f64, shape=())
        self.cg_alpha = ti.field(dtype=ti.f64, shape=())
        self.cg_beta = ti.field(dtype=ti.f64, shape=())
        self.cg_weighted_mean = ti.field(dtype=ti.f64, shape=())
        # CUDA only: each lane owns one subtotal; final accumulation visits the
        # same lanes in a fixed order. CPU field inventory and arithmetic stay
        # byte-identical. The scratch is transient and is included in captures.
        if ti.lang.impl.current_cfg().arch == ti.cuda:
            self._ki3_pressure_reduction_parts = ti.field(dtype=ti.f64, shape=(2, CG_DOT_LANES))
            self._ki3_pressure_reduction_groups = ti.field(dtype=ti.f64, shape=(2, CG_DOT_GROUPS))

        self.cg_breakdown_code = ti.field(dtype=ti.i32, shape=())
        self.cg_breakdown_dAd = ti.field(dtype=ti.f64, shape=())
        self._pcg_mg_rhs = [self.cg_mg_rhs]
        self._pcg_mg_residual = [self.cg_mg_residual]
        self._pcg_mg_pressure = [self.cg_z]
        self._pcg_mg_tmp = [self.cg_Ad]
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.obstacle = ti.field(dtype=ti.i32, shape=shape)
        self.dynamic_obstacle_candidate = ti.field(dtype=ti.i32, shape=shape)
        self.report_dynamic_obstacle_cell_count = ti.field(dtype=ti.i32, shape=())
        self.report_dynamic_obstacle_added_cell_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_dynamic_obstacle_removed_cell_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.fresh_fluid_cell = ti.field(dtype=ti.i32, shape=shape)
        self.fresh_fluid_reconstruction_pending = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self.fresh_fluid_reconstruction_pending_snapshot = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self.solid_band_protection_cell = ti.field(dtype=ti.i32, shape=shape)
        self.solid_band_candidate_cell = ti.field(dtype=ti.i32, shape=shape)
        self.no_slip_sampling_obstacle = ti.field(dtype=ti.i32, shape=shape)
        # Complete per-component validity for diagnostic MAC sampling.  A bit
        # means the matching velocity storage component is physically readable;
        # it is deliberately not an ownership mask.  Ordinary fluid rows expose
        # all three components, while a canonical boundary claim can expose an
        # individual interface face even when its storage row is an obstacle.
        self.no_slip_component_face_valid_mask = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self.velocity_constraint_sum = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.velocity_constraint_weight = ti.field(dtype=ti.f32, shape=shape)
        self.velocity_constraint_primary_sum = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.velocity_constraint_primary_weight = ti.field(dtype=ti.f32, shape=shape)
        self.velocity_constraint_secondary_sum = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.velocity_constraint_secondary_weight = ti.field(dtype=ti.f32, shape=shape)
        self.marker_feedback_target_sum = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.marker_feedback_target_weight = ti.field(dtype=ti.f32, shape=shape)
        self.marker_feedback_owned = ti.field(dtype=ti.i32, shape=shape)
        self.marker_feedback_region_min = ti.field(dtype=ti.i32, shape=shape)
        self.marker_feedback_region_max = ti.field(dtype=ti.i32, shape=shape)
        self.velocity_dirichlet_boundary_active = ti.field(dtype=ti.i32, shape=shape)
        # Opt-in (default 0): derive the staggered velocity-Dirichlet FACE
        # constraints symmetrically from the cell mask (a backward MAC face is
        # constrained if EITHER adjacent cell is active) so a mirror-symmetric
        # Dirichlet cell set yields a mirror-symmetric face constraint. Off by
        # default to preserve the validated flap and production behaviour;
        # y-symmetric cases (e.g. Turek-Hron) opt in. See
        # _apply_velocity_dirichlet_boundary_rows_kernel.
        # 0 = off (legacy); 1 = all Dirichlet rows (experimental — over-seals
        # thin marker rows, unstable); 2 = static rows only (region_id == -1:
        # channel walls + inlet) — restores the mirror symmetry of the
        # face-constraint set without sealing thin marker geometry.
        self.velocity_dirichlet_face_symmetric = 0
        self.velocity_dirichlet_boundary_value_mps = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=shape,
        )
        self.velocity_dirichlet_boundary_projection_weight = ti.field(
            dtype=ti.f32,
            shape=shape,
        )
        # interface-owned soft rows need two distinct weights.  Projection weight is
        # the fluid-pressure mobility alpha; enforcement weight is the affine
        # wall blend 1-alpha.  Non-owned/direct rows retain their legacy
        # projection-weight enforcement so existing boundary writers remain
        # source compatible.  Hard component masks still override either value.
        self.velocity_dirichlet_boundary_enforcement_weight = ti.field(
            dtype=ti.f32,
            shape=shape,
        )
        self.velocity_dirichlet_boundary_marker_region_id = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        # Bit 0/1/2 mark x/y/z velocity components whose values are prescribed
        # independently of pressure.  This provenance is deliberately separate
        # from reconstruction alpha and marker-region diagnostics: mixed-region
        # rows still remain hard velocity constraints.
        self.velocity_dirichlet_boundary_hard_fixed_component_mask = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        # Bit 0/1/2 identify the domain-face normals that make a non-owned
        # exact row an external boundary condition.  This is deliberately not
        # inferred from coordinates: a reconstructed immersed-boundary row may
        # touch a domain face without opening that face, while edge/corner rows
        # may own any explicitly registered combination of external normals.
        # The direct hard mask above remains the source of truth for velocity
        # reclamping; this provenance only controls pressure topology.
        self.velocity_dirichlet_boundary_external_exact_component_mask = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        # Persistent provenance for rows owned by a moving/reconstructed
        # boundary.  Unlike the boundary builder's transient registry, this field
        # survives a portable preflow snapshot so a fresh process can clear the
        # old geometry before stamping the current one.
        self.velocity_dirichlet_boundary_owned_row = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        # Component registry consumed by canonical boundary assembly, projection,
        # and the other registered physical consumers. Canonical authority
        # requires every consumer to prepare and seal the current generation.
        self.velocity_dirichlet_boundary_active_component_mask = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self.velocity_dirichlet_boundary_pressure_mobility = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=shape,
        )
        self.velocity_dirichlet_boundary_component_enforcement_weight = (
            ti.Vector.field(
                3,
                dtype=ti.f32,
                shape=shape,
            )
        )
        self.velocity_dirichlet_boundary_component_region_id = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=shape,
        )
        self.velocity_dirichlet_boundary_owned_component_mask = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        # Device-resident, generation-scoped hard-face registry consumed by the
        # legacy predictor.  It is rebuilt only after a canonical component
        # registry seals and is cleared on every invalidation, so unprepared or
        # legacy authority can never expose stale immersed walls to transport.
        self.canonical_exact_hard_face_component_mask = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        # Taichi kernels cannot branch on the host authority string. Keep the
        # validated authority mirrored on device: 0=legacy, 1=canonical.
        self.velocity_dirichlet_boundary_authority_code_device = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        # Authority is host state: every public dispatch checks it before a
        # kernel launch.  The canonical registry intentionally starts unsealed;
        # this change does not invent a canonical claim writer or promote the
        # existing shadow fields.
        self.velocity_dirichlet_boundary_authority = "legacy"
        self.velocity_dirichlet_component_registry_generation = 0
        self._marker_compatibility_closure_pending = False
        self.velocity_dirichlet_component_registry_sealed = False
        self._velocity_dirichlet_component_registry_consumer_generations = {}
        self._velocity_dirichlet_component_registry_consumer_capabilities = {}
        # Exact device-resident reference registry used by the velocity-only
        # projection state machine.  A summary count or scalar checksum cannot
        # prove that the same staggered boundary rows, values, weights and
        # provenance were reused; these mirrors let the runner fail closed on
        # any per-row change without copying the production grid to the host.
        self._velocity_dirichlet_registry_reference_active = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self._velocity_dirichlet_registry_reference_value_mps = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=shape,
        )
        self._velocity_dirichlet_registry_reference_projection_weight = ti.field(
            dtype=ti.f32,
            shape=shape,
        )
        self._velocity_dirichlet_registry_reference_enforcement_weight = ti.field(
            dtype=ti.f32,
            shape=shape,
        )
        self._velocity_dirichlet_registry_reference_marker_region_id = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self._velocity_dirichlet_registry_reference_hard_fixed_component_mask = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self._velocity_dirichlet_registry_reference_external_exact_component_mask = (
            ti.field(dtype=ti.i32, shape=shape)
        )
        self._velocity_dirichlet_registry_reference_owned_row = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        # Canonical reference mirrors are intentionally distinct from the
        # legacy scalar-row mirrors above.  Sharing either representation would
        # make an authority switch look byte-identical while silently changing
        # which staggered components are physical boundary faces.
        self._velocity_dirichlet_component_registry_reference_active_component_mask = (
            ti.field(dtype=ti.i32, shape=shape)
        )
        self._velocity_dirichlet_component_registry_reference_value_mps = (
            ti.Vector.field(3, dtype=ti.f32, shape=shape)
        )
        self._velocity_dirichlet_component_registry_reference_pressure_mobility = (
            ti.Vector.field(3, dtype=ti.f32, shape=shape)
        )
        self._velocity_dirichlet_component_registry_reference_component_enforcement_weight = (
            ti.Vector.field(3, dtype=ti.f32, shape=shape)
        )
        self._velocity_dirichlet_component_registry_reference_component_region_id = (
            ti.Vector.field(3, dtype=ti.i32, shape=shape)
        )
        self._velocity_dirichlet_component_registry_reference_hard_fixed_component_mask = (
            ti.field(dtype=ti.i32, shape=shape)
        )
        self._velocity_dirichlet_component_registry_reference_external_exact_component_mask = (
            ti.field(dtype=ti.i32, shape=shape)
        )
        self._velocity_dirichlet_component_registry_reference_owned_component_mask = (
            ti.field(dtype=ti.i32, shape=shape)
        )
        # The projection map also depends on which pressure cells are fluid.
        # Keep that effective topology on-device beside the row registry so a
        # row-identical rebuild cannot be mistaken for the same map after the
        # obstacle field changes.
        self._velocity_dirichlet_registry_reference_obstacle = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        # Exact map references include the independent physical-face planes.
        # These mirrors are intentionally device-resident: reference checks
        # must not introduce a CPU round trip into the coupling loop.
        self._velocity_dirichlet_registry_reference_external_x_face_active_component_mask = ti.field(
            dtype=ti.i32,
            shape=(2, self.ny, self.nz),
        )
        self._velocity_dirichlet_registry_reference_external_x_face_value_mps = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(2, self.ny, self.nz),
        )
        self._velocity_dirichlet_registry_reference_external_y_face_active_component_mask = ti.field(
            dtype=ti.i32,
            shape=(2, self.nx, self.nz),
        )
        self._velocity_dirichlet_registry_reference_external_y_face_value_mps = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(2, self.nx, self.nz),
        )
        self._velocity_dirichlet_registry_reference_external_z_face_active_component_mask = ti.field(
            dtype=ti.i32,
            shape=(2, self.nx, self.ny),
        )
        self._velocity_dirichlet_registry_reference_external_z_face_value_mps = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(2, self.nx, self.ny),
        )
        self.report_velocity_dirichlet_registry_mismatch_rows = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self._velocity_dirichlet_registry_reference_generation = 0
        self._velocity_dirichlet_registry_reference_valid = False
        self._velocity_dirichlet_registry_reference_face_symmetric = 0
        self._velocity_dirichlet_registry_reference_authority = "legacy"
        self._velocity_dirichlet_registry_reference_component_generation = -1
        # Projection-effective mask additionally contains component faces
        # mirrored onto an adjacent storage row by face_symmetric=1.  It is
        # transient and rebuilt from the direct mask before every projection.
        self.velocity_dirichlet_pressure_hard_fixed_component_mask = ti.field(
            dtype=ti.i32,
            shape=shape,
        )
        self.report_velocity_dirichlet_hard_fixed_row_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_velocity_dirichlet_hard_fixed_component_face_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_external_exact_zmax_normal_row_count = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.reduction_sum = ti.field(dtype=ti.f64, shape=())
        self.reduction_max = ti.field(dtype=ti.f64, shape=())
        self.reduction_count = ti.field(dtype=ti.i32, shape=())
        self.cfl_max_abs_velocity_x_mps = ti.field(dtype=ti.f64, shape=())
        self.cfl_max_abs_velocity_y_mps = ti.field(dtype=ti.f64, shape=())
        self.cfl_max_abs_velocity_z_mps = ti.field(dtype=ti.f64, shape=())
        self.cfl_nonfinite_velocity_component_count = ti.field(dtype=ti.i32, shape=())
        self.report_flow_state_obstacle_cells = ti.field(dtype=ti.i32, shape=())
        self.report_flow_state_fluid_cells = ti.field(dtype=ti.i32, shape=())
        self.report_flow_state_velocity_peak_mps = ti.field(dtype=ti.f64, shape=())
        self.report_flow_state_pressure_min_pa = ti.field(dtype=ti.f64, shape=())
        self.report_flow_state_pressure_max_pa = ti.field(dtype=ti.f64, shape=())
        self.divergence_report_snapshot = ti.Vector.field(3, dtype=ti.f64, shape=())
        # 18 slots: 16 partition slots + 2 anchored-unreached slots; must
        # match the static slot range of _divergence_final_report_kernel.
        self.divergence_combined_sum = ti.field(dtype=ti.f64, shape=18)
        self.divergence_combined_max = ti.field(dtype=ti.f64, shape=18)
        self.divergence_combined_count = ti.field(dtype=ti.i32, shape=18)
        self.divergence_final_report_snapshot = ti.Vector.field(
            24,
            dtype=ti.f64,
            shape=3,
        )
        self.divergence_dirichlet_partition_snapshot = ti.Vector.field(
            12,
            dtype=ti.f64,
            shape=(),
        )
        self.cleanup_target_l2_sq = ti.field(dtype=ti.f64, shape=())
        self.cleanup_acceptance_l2_sq = ti.field(dtype=ti.f64, shape=())
        self.cleanup_required = ti.field(dtype=ti.i32, shape=())
        self.cleanup_rejected = ti.field(dtype=ti.i32, shape=())
        self.report_surface_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_grid_force_n = ti.Vector.field(3, dtype=ti.f32, shape=())
        # total fluid force on the obstacle-mask surface (pressure/form part),
        # integrated over every obstacle<->fluid face; see
        # compute_obstacle_surface_pressure_force_n
        self.report_obstacle_surface_pressure_force_n = ti.Vector.field(
            3, dtype=ti.f64, shape=()
        )
        # viscous/friction part of the obstacle-mask surface force; see
        # compute_obstacle_surface_viscous_force_n
        self.report_obstacle_surface_viscous_force_n = ti.Vector.field(
            3, dtype=ti.f64, shape=()
        )
        self.report_force_spread_relative_error = ti.field(dtype=ti.f32, shape=())
        self.report_active_force_cells = ti.field(dtype=ti.i32, shape=())
        self.report_grid_impulse_n_s = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_momentum_delta_n_s = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_impulse_relative_error = ti.field(dtype=ti.f32, shape=())
        self.report_active_velocity_cells = ti.field(dtype=ti.i32, shape=())
        self.report_velocity_constraint_cells = ti.field(dtype=ti.i32, shape=())
        self.report_velocity_constraint_delta_sum = ti.field(dtype=ti.f32, shape=())
        self.report_velocity_constraint_delta_max = ti.field(dtype=ti.f32, shape=())
        self.report_velocity_constraint_momentum_delta_n_s = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_velocity_constraint_primary_momentum_delta_n_s = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_velocity_constraint_secondary_momentum_delta_n_s = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.report_marker_feedback_marker_count = ti.field(dtype=ti.i32, shape=())
        self.report_marker_feedback_active_cell_count = ti.field(dtype=ti.i32, shape=())
        self.report_marker_feedback_cleared_cell_count = ti.field(dtype=ti.i32, shape=())
        self.report_marker_feedback_obstacle_cell_count = ti.field(dtype=ti.i32, shape=())
        self.report_marker_feedback_non_obstacle_cell_count = ti.field(dtype=ti.i32, shape=())
        self.report_marker_feedback_constraint_active_cell_count = ti.field(dtype=ti.i32, shape=())
        self.report_marker_feedback_residual_before_max_mps = ti.field(dtype=ti.f32, shape=())
        self.report_marker_feedback_residual_after_max_mps = ti.field(dtype=ti.f32, shape=())
        self.report_marker_feedback_projected_residual_max_mps = ti.field(dtype=ti.f32, shape=())
        self.velocity_constraint_primary_impulse_n_s = ti.Vector.field(3, dtype=ti.f64, shape=())
        self.velocity_constraint_secondary_impulse_n_s = ti.Vector.field(3, dtype=ti.f64, shape=())
        self.report_velocity_dirichlet_boundary_cells = ti.field(dtype=ti.i32, shape=())
        self.report_velocity_dirichlet_boundary_delta_sum = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_velocity_dirichlet_boundary_delta_max = ti.field(
            dtype=ti.f32,
            shape=(),
        )
        self.report_velocity_dirichlet_boundary_momentum_delta_n_s = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(),
        )
        self.report_source_volume_flux_m3s = ti.field(dtype=ti.f64, shape=())
        self.report_positive_source_volume_flux_m3s = ti.field(dtype=ti.f64, shape=())
        self.report_abs_source_volume_flux_m3s = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_reachable_source_volume_flux_m3s = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_unreached_source_volume_flux_m3s = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_reachable_source_cell_count = ti.field(dtype=ti.i32, shape=())
        self.report_zmin_unreached_source_cell_count = ti.field(dtype=ti.i32, shape=())
        self.report_zmin_unreached_source_abs_flux_m3s = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_unreached_source_weighted_x_m = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_unreached_source_weighted_y_m = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_unreached_source_weighted_z_m = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_unreached_source_min_x_m = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_unreached_source_min_y_m = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_unreached_source_min_z_m = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_unreached_source_max_x_m = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_unreached_source_max_y_m = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_unreached_source_max_z_m = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_pressure_outlet_flux_m3s = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_velocity_outlet_flux_m3s = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_pressure_outlet_flux_ratio = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_velocity_outlet_flux_ratio = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_projection_pre_velocity_outlet_flux_m3s = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_pressure_step_pre_velocity_outlet_flux_m3s = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_projection_post_pressure_velocity_outlet_flux_m3s = ti.field(dtype=ti.f64, shape=())
        self.report_zmin_projection_post_boundary_velocity_outlet_flux_m3s = ti.field(dtype=ti.f64, shape=())
        self.report_zmax_inlet_boundary_active_cells = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_zmax_inlet_boundary_obstacle_cells = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.pressure_outlet_report_snapshot = ti.Vector.field(10, dtype=ti.f64, shape=())
        self.report_pressure_interface_matrix_diagonal_integral = ti.field(dtype=ti.f64, shape=())
        self.report_pressure_interface_matrix_rhs_integral = ti.field(dtype=ti.f64, shape=())
        self.report_pressure_interface_matrix_max_abs_diagonal = ti.field(dtype=ti.f64, shape=())
        self.report_pressure_interface_matrix_active_cells = ti.field(dtype=ti.i32, shape=())
        self.report_pressure_interface_matrix_row_count = ti.field(dtype=ti.i32, shape=())
        self.report_pressure_interface_matrix_active_row_count = ti.field(dtype=ti.i32, shape=())
        self.report_pressure_interface_matrix_invalid_row_count = ti.field(dtype=ti.i32, shape=())
        self.report_pressure_interface_matrix_overflow_row_count = ti.field(dtype=ti.i32, shape=())
        self.report_pressure_interface_matrix_row_diagonal_integral = ti.field(dtype=ti.f64, shape=())
        self.report_pressure_interface_matrix_row_diagonal_abs_mismatch = ti.field(dtype=ti.f64, shape=())
        self.report_pressure_interface_matrix_row_diagonal_max_abs_density_mismatch = ti.field(dtype=ti.f64, shape=())
        self.report_pressure_interface_matrix_row_max_transmissibility = ti.field(dtype=ti.f64, shape=())
        self.report_internal_obstacle_cells = ti.field(dtype=ti.i32, shape=())
        self.report_fresh_fluid_cells = ti.field(dtype=ti.i32, shape=())
        self.report_solid_band_nonprojectable_cells = ti.field(dtype=ti.i32, shape=())
        self.report_solid_band_interior_cells = ti.field(dtype=ti.i32, shape=())
        self.report_solid_band_enclosed_water_cells = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_solid_band_velocity_dirichlet_protected_cells = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_solid_band_mask_protected_cells = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        self.report_pressure_disconnected_nonprojectable_cells = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        # Compatibility mirror of the same fluid, non-barrier, unreached
        # population; both report keys share one device reduction.
        self.report_pressure_disconnected_unreached_raw_cells = ti.field(
            dtype=ti.i32,
            shape=(),
        )
        # post-projection pressure fill into interface-converted cells
        # (obstacle != 0 and base_obstacle == 0): Jacobi double
        # buffers (filled flag + pressure) and the dedicated stress
        # sampling view (base geometry + classified row-cloud envelope
        # dry; converted sealed water samplable).
        self.pressure_filled = ti.field(dtype=ti.i32, shape=shape)
        self.pressure_filled_next = ti.field(dtype=ti.i32, shape=shape)
        self.pressure_fill_next = ti.field(dtype=ti.f64, shape=shape)
        self.sampling_obstacle = ti.field(dtype=ti.i32, shape=shape)
        self.report_pressure_filled_cells = ti.field(dtype=ti.i32, shape=())
        self._base_obstacle_initialized = False
        self.dynamic_solid_volume_enabled = False
        self._saved_dynamic_solid_volume_enabled = False
        self.cell_width_x_m = ti.field(dtype=ti.f32, shape=self.nx)
        self.cell_width_y_m = ti.field(dtype=ti.f32, shape=self.ny)
        self.cell_width_z_m = ti.field(dtype=ti.f32, shape=self.nz)
        self.cell_face_x_m = ti.field(dtype=ti.f32, shape=self.nx + 1)
        self.cell_face_y_m = ti.field(dtype=ti.f32, shape=self.ny + 1)
        self.cell_face_z_m = ti.field(dtype=ti.f32, shape=self.nz + 1)
        self.cell_center_x_m = ti.field(dtype=ti.f32, shape=self.nx)
        self.cell_center_y_m = ti.field(dtype=ti.f32, shape=self.ny)
        self.cell_center_z_m = ti.field(dtype=ti.f32, shape=self.nz)
        self.center_distance_x_m = ti.field(dtype=ti.f32, shape=self.nx)
        self.center_distance_y_m = ti.field(dtype=ti.f32, shape=self.ny)
        self.center_distance_z_m = ti.field(dtype=ti.f32, shape=self.nz)

        # Materialize the already-complete fine-grid solver state before
        # allocating the variable-depth multigrid hierarchy.  Taichi limits
        # SNodes per tree; putting every fine-grid diagnostic plus every
        # possible coarse level in one tree made otherwise valid deeper
        # hierarchies fail during construction.  Kernels may access fields
        # across trees, so this is only an allocation boundary, not a solver or
        # data-flow boundary.  Loading the Cartesian axes is also the normal
        # initialization work for these fields and avoids internal Taichi APIs.
        self._load_cartesian_grid_fields()

        self._mg_shapes = self._build_multigrid_shapes(
            shape,
            semi_coarsening=bool(getattr(self, "multigrid_semi_coarsening", False)),
        )
        self._mg_level_factors = self._build_multigrid_level_factors(self._mg_shapes)
        self._mg_axis_widths_m = self._build_multigrid_axis_widths(self.grid, self._mg_shapes)
        self._mg_cell_width_x_m = [self.cell_width_x_m]
        self._mg_cell_width_y_m = [self.cell_width_y_m]
        self._mg_cell_width_z_m = [self.cell_width_z_m]
        self._mg_center_distance_x_m = [self.center_distance_x_m]
        self._mg_center_distance_y_m = [self.center_distance_y_m]
        self._mg_center_distance_z_m = [self.center_distance_z_m]
        for level_shape in self._mg_shapes[1:]:
            self._mg_cell_width_x_m.append(ti.field(dtype=ti.f32, shape=level_shape[0]))
            self._mg_cell_width_y_m.append(ti.field(dtype=ti.f32, shape=level_shape[1]))
            self._mg_cell_width_z_m.append(ti.field(dtype=ti.f32, shape=level_shape[2]))
            self._mg_center_distance_x_m.append(ti.field(dtype=ti.f32, shape=level_shape[0]))
            self._mg_center_distance_y_m.append(ti.field(dtype=ti.f32, shape=level_shape[1]))
            self._mg_center_distance_z_m.append(ti.field(dtype=ti.f32, shape=level_shape[2]))
        self._mg_pressure = [self.pressure]
        self._mg_tmp = [self.pressure_tmp]
        self._mg_obstacle = [self.obstacle]
        self._mg_velocity_dirichlet_boundary_active = [
            self.velocity_dirichlet_boundary_active
        ]
        self._mg_velocity_dirichlet_boundary_projection_weight = [
            self.velocity_dirichlet_boundary_projection_weight
        ]
        self._mg_velocity_dirichlet_boundary_marker_region_id = [
            self.velocity_dirichlet_boundary_marker_region_id
        ]
        self._mg_velocity_dirichlet_pressure_hard_fixed_component_mask = [
            self.velocity_dirichlet_pressure_hard_fixed_component_mask
        ]
        # Canonical multigrid authority is component-face local.  Level zero
        # aliases the current fine-grid registry exactly; every coarse level owns
        # independent fields built from geometrically coplanar fine subfaces.
        self._mg_velocity_dirichlet_boundary_active_component_mask = [
            self.velocity_dirichlet_boundary_active_component_mask
        ]
        self._mg_velocity_dirichlet_boundary_pressure_mobility = [
            self.velocity_dirichlet_boundary_pressure_mobility
        ]
        self._mg_velocity_dirichlet_boundary_hard_fixed_component_mask = [
            self.velocity_dirichlet_boundary_hard_fixed_component_mask
        ]
        self._mg_velocity_dirichlet_boundary_owned_component_mask = [
            self.velocity_dirichlet_boundary_owned_component_mask
        ]
        self._mg_velocity_dirichlet_boundary_external_exact_component_mask = [
            self.velocity_dirichlet_boundary_external_exact_component_mask
        ]
        self._mg_pressure_interface_matrix_diagonal = [
            self.pressure_interface_matrix_diagonal
        ]
        self._mg_rhs = []
        self._mg_residual = []
        for level_index, level_shape in enumerate(self._mg_shapes):
            level_dtype = ti.f64 if level_index == 0 else ti.f32
            self._mg_rhs.append(ti.field(dtype=level_dtype, shape=level_shape))
            self._mg_residual.append(ti.field(dtype=level_dtype, shape=level_shape))
        for level_index, level_shape in enumerate(self._mg_shapes[1:], start=1):
            self._mg_pressure.append(ti.field(dtype=ti.f32, shape=level_shape))
            self._mg_tmp.append(ti.field(dtype=ti.f32, shape=level_shape))
            self._mg_obstacle.append(ti.field(dtype=ti.i32, shape=level_shape))
            self._mg_velocity_dirichlet_boundary_active.append(
                ti.field(dtype=ti.i32, shape=level_shape)
            )
            self._mg_velocity_dirichlet_boundary_projection_weight.append(
                ti.field(dtype=ti.f32, shape=level_shape)
            )
            self._mg_velocity_dirichlet_boundary_marker_region_id.append(
                ti.field(dtype=ti.i32, shape=level_shape)
            )
            self._mg_velocity_dirichlet_pressure_hard_fixed_component_mask.append(
                ti.field(dtype=ti.i32, shape=level_shape)
            )
            self._mg_velocity_dirichlet_boundary_active_component_mask.append(
                ti.field(dtype=ti.i32, shape=level_shape)
            )
            self._mg_velocity_dirichlet_boundary_pressure_mobility.append(
                ti.Vector.field(3, dtype=ti.f32, shape=level_shape)
            )
            self._mg_velocity_dirichlet_boundary_hard_fixed_component_mask.append(
                ti.field(dtype=ti.i32, shape=level_shape)
            )
            self._mg_velocity_dirichlet_boundary_owned_component_mask.append(
                ti.field(dtype=ti.i32, shape=level_shape)
            )
            self._mg_velocity_dirichlet_boundary_external_exact_component_mask.append(
                ti.field(dtype=ti.i32, shape=level_shape)
            )
            self._mg_pressure_interface_matrix_diagonal.append(
                ti.field(dtype=ti.f32, shape=level_shape)
            )
            # Standalone multigrid and the PCG multigrid preconditioner are
            # mutually exclusive pressure-solve paths.  Their coarse-level
            # RHS/residual/correction/smoothing buffers therefore have
            # identical lifetimes and can share storage.  Level zero remains
            # deliberately independent: PCG must preserve the physical
            # pressure and fine-grid RHS while it builds ``cg_z``.
            #
            # Keeping duplicate coarse fields used four additional SNodes per
            # level.  A perfectly ordinary 8^3 grid (8 -> 4 -> 2), especially
            # when composed with interface marker fields, could consequently exceed
            # Taichi's global SNode limit before the first kernel ran.  Sharing
            # the serial scratch preserves the complete hierarchy instead of
            # silently truncating multigrid depth.
            self._pcg_mg_rhs.append(self._mg_rhs[level_index])
            self._pcg_mg_residual.append(self._mg_residual[level_index])
            self._pcg_mg_pressure.append(self._mg_pressure[level_index])
            self._pcg_mg_tmp.append(self._mg_tmp[level_index])

        self._load_multigrid_axis_fields()
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
        self.last_cg_restart_count_measured = False
        self.last_cg_restart_policy = ""
        self.last_cg_breakdown_dAd = 0.0
        self.last_cg_preconditioner_requested = "auto"
        self.last_cg_preconditioner_effective = "not_run"
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
        self.last_pressure_unreached_cell_count = 0
        self.last_pressure_unreached_raw_cell_count = 0
        self.last_pressure_reachability_converged = True
        self.last_pressure_reachability_sweeps = 0
        self.last_pressure_reachability_frontier_overflow = False
        self.last_pressure_reachability_reused = False
        self._pressure_unreached_count = 0
        self._reachability_checksum = None
        self.reachability_revision = 0
        self.last_reachability_revision = 0
        self.last_reachability_valid = False
        # Monotonic host epoch for obstacle changes that originate outside a
        # sharp-interface classification/cleanup transaction.  Boundary caches use
        # this to reject stale classified topology even when marker geometry is
        # unchanged.  Reachability floods deliberately do not advance it.
        self.external_obstacle_topology_revision = 0
        self.last_pressure_unreached_component_count = 0
        self._pressure_unreached_component_count = 0
        self.last_pressure_unreached_component_overflow = False
        # The exact pressure-operator nullspace graph is derived state. It is
        # published only after every source label, edge and diagonal passes
        # validation, and is invalidated whenever any graph input changes.
        self._pressure_outlet_nullspace_graph_valid = False
        self._pressure_outlet_nullspace_source_component_count = 0
        self._pressure_outlet_nullspace_component_count = 0
        self._pressure_outlet_operator_component_count = 0
        self._pressure_outlet_nullspace_graph_context = ""
        self._reset_pressure_unreached_component_distribution_stats()
        self.last_pressure_component_labels_converged = True
        self.last_unreached_cells_with_interface_diagonal = 0
        self.last_unreached_cells_with_interface_coupling = 0
        self.last_unreached_components_with_interface_hits = 0
        self.last_unreached_incompatible_component_count = 0
        self.last_unreached_component_rhs_mean_max_abs = 0.0
        self.last_unreached_component_rhs_integral_max_abs = 0.0
        self.last_solid_band_marked_increment = 0
        # -1 means "the last band sweep ran without a population split"
        # (legacy unclassified sweep); >= 0 are measured per-sweep
        # populations.
        self.last_solid_band_interior_cells = -1
        self.last_solid_band_enclosed_water_cells = -1
        self.last_solid_band_velocity_dirichlet_protected_cells = -1
        self.last_solid_band_mask_protected_cells = -1
        # -1 means "the post-projection converted-cell pressure fill has
        # not run since construction / restore_state". The fill
        # is never mounted inside project(): the interface assemble calls it
        # explicitly after the projection returns and before stress
        # sampling, so project()'s report key mirrors the PREVIOUS fill.
        self.last_pressure_filled_cell_count = -1
        # -1 means "the air-backed classification has not run since
        # construction / restore_state" (armed per closure region
        # by the interface assemble; default off).
        self.last_air_backed_cell_count = -1
        self.last_air_backed_component_count = -1
        self.last_air_backed_cell_volume_m3 = -1.0
        self.last_row_cloud_orphan_cell_count = 0
        self.last_row_cloud_orphan_component_count = 0
        self.last_unreached_divergence_raw_stats = {
            "max_abs": 0.0,
            "l2": 0.0,
            "count": 0,
        }
        self.last_unreached_divergence_stats = {
            "max_abs": 0.0,
            "l2": 0.0,
            "count": 0,
        }
        self.last_project_cg_project_calls = 0
        self.last_project_cg_iterations_total = 0
        self.last_project_cg_operator_apply_count = 0
        self.last_project_cg_iterations_max = 0
        self.last_project_cg_host_residual_checks = 0
        self.last_project_cg_mean_projection_count = 0
        self.last_project_cg_unreached_set_mean_projection_count = 0
        self.last_project_cg_componentwise_mean_projection_count = 0
        self.last_project_cg_restart_count = 0
        self.last_project_cg_restart_count_measured = False
        self.last_project_cg_restart_policy = ""
        self.last_project_cg_preconditioner_requested = "auto"
        self.last_project_cg_preconditioner_effective = "not_run"
        self.last_project_cg_nonzero_rhs_project_calls = 0
        self.last_project_cg_nonzero_rhs_preconditioner_requested = (
            "not_applicable"
        )
        self.last_project_cg_nonzero_rhs_preconditioner_effective = (
            "not_applicable"
        )
        self.last_project_cg_nonzero_rhs_multigrid_to_jacobi_fallback_count = 0
        self.last_project_cg_multigrid_apply_count = 0
        self.last_project_cg_multigrid_to_jacobi_fallback_count = 0
        self.last_project_cg_exact_residual_confirmation_count = 0
        self.last_project_cg_exact_relative_residual_max = 0.0
        self.last_project_cg_initial_relative_residual_max = 0.0
        self.last_project_cg_relative_residual_max = 0.0
        self.last_project_cg_converged_all = True
        self.last_project_cg_breakdown_count = 0
        self.last_project_cg_breakdown_code = 0
        self.last_project_cg_breakdown_dAd = 0.0
        self.last_project_cg_breakdown = ""
        self.last_project_pressure_projection_physical_failure = False
        self.last_project_pressure_projection_physical_failure_reason = ""
        self.last_project_pressure_projection_physical_failure_action = "none"
        self.last_project_pressure_nullspace_incompatible_component_count = 0
        self.last_project_pressure_nullspace_component_rhs_mean_max_abs = 0.0
        self.last_project_pressure_nullspace_component_rhs_integral_max_abs = 0.0
        self.last_cg_closed_neumann_physical_rhs_measured = False
        self.last_cg_closed_neumann_physical_rhs_mean = 0.0
        self.last_cg_closed_neumann_physical_rhs_integral = 0.0
        self.last_cg_closed_neumann_physical_component_budgets = ()
        self.last_pressure_outlet_report_host_reads = 0
        self.last_divergence_report_host_reads = 0
        self.last_velocity_constraint_impulse_host_reads = 0
        # safeguard of the removed no-free-face rule: the reachability marking counts the
        # outlet-unreached water cells with no free face and no interface term (slot 0) and the first one (slot 1,
        # linear index); any such cell is a hard stop (expected never)
        self.no_free_face_stats = ti.field(dtype=ti.i32, shape=2)
        self.no_free_face_checks = 0
        # the unreached-pocket distribution counters exist from the constructor on; the first
        # reachability marking resets them again before any read
        self._reset_pressure_unreached_component_distribution_stats()
        # only CUDA installs the device scalar / component controllers (fastfsi.fluid.device_controls)
        if ti.lang.impl.current_cfg().arch == ti.cuda:
            from fastfsi.fluid.device_controls import install_fluid_device_control
            install_fluid_device_control(self)
        self.clear()

    @staticmethod
    def _load_axis_field(field, values: tuple[float, ...]) -> None:
        field.from_numpy(np.asarray(values, dtype=np.float32))

    def _load_cartesian_grid_fields(self) -> None:
        self._load_axis_field(self.cell_width_x_m, self.grid.cell_widths_x_m)
        self._load_axis_field(self.cell_width_y_m, self.grid.cell_widths_y_m)
        self._load_axis_field(self.cell_width_z_m, self.grid.cell_widths_z_m)
        self._load_axis_field(self.cell_face_x_m, self.grid.cell_faces_x_m)
        self._load_axis_field(self.cell_face_y_m, self.grid.cell_faces_y_m)
        self._load_axis_field(self.cell_face_z_m, self.grid.cell_faces_z_m)
        self._load_axis_field(self.cell_center_x_m, self.grid.cell_centers_x_m)
        self._load_axis_field(self.cell_center_y_m, self.grid.cell_centers_y_m)
        self._load_axis_field(self.cell_center_z_m, self.grid.cell_centers_z_m)
        self._load_axis_field(self.center_distance_x_m, self.grid.center_distances_x_m)
        self._load_axis_field(self.center_distance_y_m, self.grid.center_distances_y_m)
        self._load_axis_field(self.center_distance_z_m, self.grid.center_distances_z_m)

    @ti.kernel
    def _clear_kernel(self):
        self.pressure_interface_row_count[None] = 0
        for row in self.pressure_interface_row_support_count:
            self.pressure_interface_row_support_count[row] = 0
            self.pressure_interface_row_source_node[row] = ti.Vector([-1, -1, -1])
        self.pressure_outlet_operator_raw_component_count[None] = 0
        for i, j, k in self.velocity:
            self.velocity[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_prev[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_transport_base[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.saved_velocity[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.saved_velocity_prev[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.pressure[i, j, k] = 0.0
            self.saved_pressure[i, j, k] = 0.0
            self.sst_turbulent_kinetic_energy[i, j, k] = 0.0
            self.sst_specific_dissipation_rate[i, j, k] = 1.0
            self.sst_eddy_viscosity_pa_s[i, j, k] = 0.0
            self.sst_wall_distance_m[i, j, k] = 1.0e20
            self.sst_turbulent_kinetic_energy_prev[i, j, k] = 0.0
            self.sst_specific_dissipation_rate_prev[i, j, k] = 1.0
            self.sst_turbulent_kinetic_energy_next[i, j, k] = 0.0
            self.sst_specific_dissipation_rate_next[i, j, k] = 1.0
            self.sst_turbulent_kinetic_energy_transport_base[i, j, k] = 0.0
            self.sst_specific_dissipation_rate_transport_base[i, j, k] = 1.0
            self.sst_wall_distance_next_m[i, j, k] = 1.0e20
            self.sst_blending_f1[i, j, k] = 0.0
            self.sst_blending_f2[i, j, k] = 0.0
            self.sst_sigma_k[i, j, k] = 1.0
            self.sst_sigma_omega[i, j, k] = 0.856
            self.sst_beta[i, j, k] = 0.0828
            self.sst_gamma[i, j, k] = 0.44
            self.sst_strain_rate_magnitude_s[i, j, k] = 0.0
            self.sst_cell_center_velocity_mps[i, j, k] = ti.Vector(
                [0.0, 0.0, 0.0]
            )
            self.sst_momentum_transpose_gradient_s[i, j, k] = ti.Matrix.zero(
                ti.f32, 3, 3
            )
            self.sst_momentum_transpose_divergence_cell_mps2[
                i, j, k
            ] = ti.Vector([0.0, 0.0, 0.0])
            self.sst_obstacle_interface_wall_target_component_mask[
                i, j, k
            ] = ti.Vector([0, 0, 0])
            self.saved_sst_turbulent_kinetic_energy[i, j, k] = 0.0
            self.saved_sst_specific_dissipation_rate[i, j, k] = 1.0
            self.saved_sst_eddy_viscosity_pa_s[i, j, k] = 0.0
            self.saved_sst_wall_distance_m[i, j, k] = 1.0e20
            self.fsi_pressure[i, j, k] = 0.0
            self.saved_fsi_pressure[i, j, k] = 0.0
            self.pressure_tmp[i, j, k] = 0.0
            self.pressure_accum[i, j, k] = 0.0
            self.divergence[i, j, k] = 0.0
            self.pressure_interface_projection_divergence_s[i, j, k] = 0.0
            self.volume_source_s[i, j, k] = 0.0
            self.saved_volume_source_s[i, j, k] = 0.0
            self.pressure_interface_matrix_diagonal[i, j, k] = 0.0
            self.pressure_interface_matrix_rhs[i, j, k] = 0.0
            self.pressure_interface_coupling_active[i, j, k] = 0
            self.pressure_interface_coupling_neighbor[i, j, k] = ti.Vector([0, 0, 0])
            self.pressure_interface_coupling_coefficient[i, j, k] = 0.0
            self.pressure_interface_row_diagonal_density[i, j, k] = 0.0
            for slot in ti.static(range(PRESSURE_INTERFACE_COUPLING_EXTRA_SLOTS)):
                self.pressure_interface_coupling_extra_neighbor[
                    i,
                    j,
                    k,
                    slot,
                ] = ti.Vector([0, 0, 0])
                self.pressure_interface_coupling_extra_coefficient[
                    i,
                    j,
                    k,
                    slot,
                ] = 0.0
            self.fv_diag[i, j, k] = 0.0
            self.cg_r[i, j, k] = 0.0
            self.cg_rhs[i, j, k] = 0.0
            self.cg_z[i, j, k] = 0.0
            self.cg_d[i, j, k] = 0.0
            self.cg_Ad[i, j, k] = 0.0
            self.cg_r_old[i, j, k] = 0.0
            self.cg_mg_rhs[i, j, k] = 0.0
            self.cg_mg_residual[i, j, k] = 0.0
            self.bicgstab_s[i, j, k] = 0.0
            self.bicgstab_t[i, j, k] = 0.0
            self.force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.obstacle[i, j, k] = 0
            self.pressure_outlet_operator_component_label[i, j, k] = 1 << 30
            self.pressure_outlet_operator_raw_component_cell_count[i, j, k] = 0
            self.pressure_outlet_operator_raw_component_to_compact[i, j, k] = -1
            self.pressure_outlet_operator_raw_component_anchored[i, j, k] = 0
            self.pressure_outlet_operator_raw_component_diagonal_excess_integral[
                i, j, k
            ] = 0.0
            self.pressure_outlet_operator_raw_component_diagonal_scale_integral[
                i, j, k
            ] = 0.0
            self.pressure_outlet_operator_raw_component_diagonal_term_count[
                i, j, k
            ] = 0
            self.dynamic_obstacle_candidate[i, j, k] = 0
            self.base_obstacle[i, j, k] = 0
            self.dynamic_solid_volume_obstacle[i, j, k] = 0
            self.dynamic_solid_volume_external_carve[i, j, k] = 0
            self.pressure_outlet_reachable[i, j, k] = 0
            self.pressure_outlet_reachable_next[i, j, k] = 0
            self.pressure_reachability_barrier[i, j, k] = 0
            self.fresh_fluid_cell[i, j, k] = 0
            self.fresh_fluid_reconstruction_pending[i, j, k] = 0
            self.interface_air_cell[i, j, k] = 0
            self.pressure_filled[i, j, k] = 0
            self.pressure_filled_next[i, j, k] = 0
            self.pressure_fill_next[i, j, k] = 0.0
            self.sampling_obstacle[i, j, k] = 0
            self.velocity_constraint_sum[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_constraint_weight[i, j, k] = 0.0
            self.velocity_constraint_primary_sum[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_constraint_primary_weight[i, j, k] = 0.0
            self.velocity_constraint_secondary_sum[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_constraint_secondary_weight[i, j, k] = 0.0
            self.marker_feedback_target_sum[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.marker_feedback_target_weight[i, j, k] = 0.0
            self.marker_feedback_owned[i, j, k] = 0
            self.marker_feedback_region_min[i, j, k] = 2147483647
            self.marker_feedback_region_max[i, j, k] = -2147483648
            self.velocity_dirichlet_boundary_active[i, j, k] = 0
            self.velocity_dirichlet_boundary_value_mps[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_dirichlet_boundary_projection_weight[i, j, k] = 0.0
            self.velocity_dirichlet_boundary_enforcement_weight[i, j, k] = 0.0
            self.velocity_dirichlet_boundary_marker_region_id[i, j, k] = -1
            self.velocity_dirichlet_boundary_hard_fixed_component_mask[i, j, k] = 0
            self.velocity_dirichlet_boundary_external_exact_component_mask[
                i, j, k
            ] = 0
            self.velocity_dirichlet_boundary_owned_row[i, j, k] = 0
            self.velocity_dirichlet_boundary_active_component_mask[i, j, k] = 0
            self.velocity_dirichlet_boundary_pressure_mobility[i, j, k] = ti.Vector(
                [1.0, 1.0, 1.0]
            )
            self.velocity_dirichlet_boundary_component_enforcement_weight[
                i, j, k
            ] = ti.Vector([0.0, 0.0, 0.0])
            self.velocity_dirichlet_boundary_component_region_id[
                i, j, k
            ] = ti.Vector([-1, -1, -1])
            self.velocity_dirichlet_boundary_owned_component_mask[i, j, k] = 0
            self.velocity_dirichlet_pressure_hard_fixed_component_mask[i, j, k] = 0
        self.pressure_reachability_frontier_a_count[None] = 0
        self.pressure_reachability_frontier_b_count[None] = 0
        self.pressure_reachability_frontier_overflow[None] = 0
        self.report_surface_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_grid_force_n[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_force_spread_relative_error[None] = 0.0
        self.report_active_force_cells[None] = 0
        self.report_grid_impulse_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_impulse_relative_error[None] = 0.0
        self.report_active_velocity_cells[None] = 0
        self.report_velocity_constraint_cells[None] = 0
        self.report_velocity_constraint_delta_sum[None] = 0.0
        self.report_velocity_constraint_delta_max[None] = 0.0
        self.report_velocity_constraint_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_velocity_constraint_primary_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_velocity_constraint_secondary_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_marker_feedback_marker_count[None] = 0
        self.report_marker_feedback_active_cell_count[None] = 0
        self.report_marker_feedback_cleared_cell_count[None] = 0
        self.report_marker_feedback_obstacle_cell_count[None] = 0
        self.report_marker_feedback_non_obstacle_cell_count[None] = 0
        self.report_marker_feedback_constraint_active_cell_count[None] = 0
        self.report_marker_feedback_residual_before_max_mps[None] = 0.0
        self.report_marker_feedback_residual_after_max_mps[None] = 0.0
        self.report_marker_feedback_projected_residual_max_mps[None] = 0.0
        self.velocity_constraint_primary_impulse_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.velocity_constraint_secondary_impulse_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_velocity_dirichlet_boundary_cells[None] = 0
        self.report_velocity_dirichlet_boundary_delta_sum[None] = 0.0
        self.report_velocity_dirichlet_boundary_delta_max[None] = 0.0
        self.report_velocity_dirichlet_boundary_momentum_delta_n_s[None] = ti.Vector([0.0, 0.0, 0.0])
        self.report_source_volume_flux_m3s[None] = 0.0
        self.report_positive_source_volume_flux_m3s[None] = 0.0
        self.report_abs_source_volume_flux_m3s[None] = 0.0
        self.report_zmin_reachable_source_volume_flux_m3s[None] = 0.0
        self.report_zmin_unreached_source_volume_flux_m3s[None] = 0.0
        self.report_zmin_reachable_source_cell_count[None] = 0
        self.report_zmin_unreached_source_cell_count[None] = 0
        self.report_zmin_unreached_source_abs_flux_m3s[None] = 0.0
        self.report_zmin_unreached_source_weighted_x_m[None] = 0.0
        self.report_zmin_unreached_source_weighted_y_m[None] = 0.0
        self.report_zmin_unreached_source_weighted_z_m[None] = 0.0
        self.report_zmin_unreached_source_min_x_m[None] = 1.0e30
        self.report_zmin_unreached_source_min_y_m[None] = 1.0e30
        self.report_zmin_unreached_source_min_z_m[None] = 1.0e30
        self.report_zmin_unreached_source_max_x_m[None] = -1.0e30
        self.report_zmin_unreached_source_max_y_m[None] = -1.0e30
        self.report_zmin_unreached_source_max_z_m[None] = -1.0e30
        self.report_zmin_pressure_outlet_flux_m3s[None] = 0.0
        self.report_zmin_velocity_outlet_flux_m3s[None] = 0.0
        self.report_zmin_pressure_outlet_flux_ratio[None] = 0.0
        self.report_zmin_velocity_outlet_flux_ratio[None] = 0.0
        self.report_zmin_projection_pre_velocity_outlet_flux_m3s[None] = 0.0
        self.report_zmin_pressure_step_pre_velocity_outlet_flux_m3s[None] = 0.0
        self.report_zmin_projection_post_pressure_velocity_outlet_flux_m3s[None] = 0.0
        self.report_zmin_projection_post_boundary_velocity_outlet_flux_m3s[None] = 0.0
        self.pressure_outlet_report_snapshot[None] = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.cg_rz[None] = 0.0
        self.cg_rz_new[None] = 0.0
        self.cg_dAd[None] = 0.0
        self.cg_rr[None] = 0.0
        self.cg_beta_numerator[None] = 0.0
        self.cg_alpha[None] = 0.0
        self.cg_beta[None] = 0.0
        self.cg_weighted_mean[None] = 0.0
        self.cg_breakdown_code[None] = 0
        self.cg_breakdown_dAd[None] = 0.0
        self.sst_reduction_max_diffusivity_m2_s[None] = 0.0
        self.sst_reduction_min_k[None] = 0.0
        self.sst_reduction_max_k[None] = 0.0
        self.sst_reduction_min_omega[None] = 0.0
        self.sst_reduction_max_omega[None] = 0.0
        self.sst_reduction_max_mu_t[None] = 0.0
        self.sst_reduction_nonfinite_or_nonpositive_count[None] = 0
        self.report_internal_obstacle_cells[None] = 0
        self.report_fresh_fluid_cells[None] = 0
        self.report_solid_band_nonprojectable_cells[None] = 0
        self.report_solid_band_interior_cells[None] = 0
        self.report_solid_band_enclosed_water_cells[None] = 0
        self.report_solid_band_velocity_dirichlet_protected_cells[None] = 0
        self.report_solid_band_mask_protected_cells[None] = 0
        self.report_pressure_disconnected_nonprojectable_cells[None] = 0
        self.report_pressure_filled_cells[None] = 0
        self.report_air_backed_cells[None] = 0
        self.report_air_backed_components[None] = 0
        self.report_air_backed_cell_volume_m3[None] = 0.0
        self.report_dynamic_obstacle_cell_count[None] = 0
        self.report_dynamic_obstacle_added_cell_count[None] = 0
        self.report_dynamic_obstacle_removed_cell_count[None] = 0
        self.report_unreached_component_raw_count[None] = 0
        self.report_unreached_component_largest_cell_count[None] = 0
        self.report_unreached_component_singleton_count[None] = 0
        self.report_unreached_component_small_count[None] = 0
        self.report_unreached_component_small_cell_count[None] = 0
        for i, j, k in self.unreached_raw_component_cell_count:
            self.unreached_raw_component_cell_count[i, j, k] = 0
        for slot in self.air_component_selected:
            self.air_component_selected[slot] = 0
            self.unreached_component_compact_raw_label[slot] = 1 << 30
            self.row_cloud_component_cell_count[slot] = 0
            self.row_cloud_component_touches_row[slot] = 0
            self.row_cloud_component_touches_protected[slot] = 0
            self.row_cloud_component_selected[slot] = 0
        self.report_row_cloud_orphan_cells[None] = 0
        self.report_row_cloud_orphan_components[None] = 0

    def clear(self) -> None:
        # Retire every obstacle-derived host contract before the first device
        # write.  A failed scalar write or partial clear kernel must never
        # leave old reachability or a sealed canonical registry reusable.
        self.external_obstacle_topology_revision += 1
        self._invalidate_external_obstacle_topology_derived_state()
        self.velocity_dirichlet_boundary_authority_code_device[None] = (
            self._velocity_dirichlet_boundary_authority_code()
        )
        self._clear_kernel()
        self.turbulence_model = "laminar"
        self._sst_no_slip_domain_walls = (False,) * 6
        self._sst_saved_no_slip_domain_walls = (False,) * 6
        self.sst_no_slip_domain_wall_mask[None] = 0
        self._sst_saved_no_slip_domain_wall_mask = 0
        self._sst_wall_distance_valid = False
        self._sst_saved_wall_distance_valid = False
        self._sst_wall_distance_cache_key = None
        self._sst_wall_distance_base_cache_key = None
        self._sst_saved_wall_distance_cache_key = None
        self._sst_last_diffusion_substeps = 0
        self._sst_last_diffusion_cfl = 0.0
        self.external_velocity_boundary_x_face_active_component_mask.fill(0)
        self.external_velocity_boundary_x_face_value_mps.fill(0.0)
        self.external_velocity_boundary_y_face_active_component_mask.fill(0)
        self.external_velocity_boundary_y_face_value_mps.fill(0.0)
        self.external_velocity_boundary_z_face_active_component_mask.fill(0)
        self.external_velocity_boundary_z_face_value_mps.fill(0.0)
        self._base_obstacle_initialized = False
        self.dynamic_solid_volume_enabled = False
        self.reachability_revision = 0
        self.last_reachability_revision = 0
        self.last_reachability_valid = False
        self._reachability_checksum = None
        self.last_pressure_reachability_frontier_overflow = False

    @ti.kernel
    def _clear_volume_source_kernel(self):
        for i, j, k in self.volume_source_s:
            self.volume_source_s[i, j, k] = 0.0

    @ti.kernel
    def _zero_velocity_component_kernel(self, axis: ti.i32):
        for i, j, k in self.velocity:
            self.velocity[i, j, k][axis] = 0.0
            self.velocity_prev[i, j, k][axis] = 0.0

    def enforce_quasi_2d_velocity(self) -> bool:
        """Zero the configured quasi-2D velocity component; returns whether it acted."""
        axis = int(self.quasi_2d_velocity_axis)
        if axis < 0:
            return False
        if axis not in (0, 1, 2):
            raise ValueError("quasi_2d_velocity_axis must be -1, 0, 1 or 2")
        self._zero_velocity_component_kernel(axis)
        return True

    def clear_volume_source(self) -> None:
        self._clear_volume_source_kernel()

    @ti.func
    def _accumulate_cfl_velocity_component(self, value: ti.f32, axis: ti.template()):
        # All velocity/target storage is f32. Inspect its exponent before any
        # arithmetic: fast_math comparisons and max reductions can hide NaNs.
        exponent = ti.u32(0x7F800000)
        finite = (ti.bit_cast(value, ti.u32) & exponent) != exponent
        if not finite:
            ti.atomic_add(self.cfl_nonfinite_velocity_component_count[None], 1)
        else:
            absolute = ti.abs(ti.cast(value, ti.f64))
            if ti.static(axis == 0):
                ti.atomic_max(self.cfl_max_abs_velocity_x_mps[None], absolute)
            elif ti.static(axis == 1):
                ti.atomic_max(self.cfl_max_abs_velocity_y_mps[None], absolute)
            else:
                ti.atomic_max(self.cfl_max_abs_velocity_z_mps[None], absolute)

    @ti.kernel
    def _max_fluid_speed_kernel(self) -> ti.f64:
        self.cfl_max_abs_velocity_x_mps[None] = 0.0
        self.cfl_max_abs_velocity_y_mps[None] = 0.0
        self.cfl_max_abs_velocity_z_mps[None] = 0.0
        self.cfl_nonfinite_velocity_component_count[None] = 0
        for i, j, k in self.velocity:
            boundary_mask = self.canonical_exact_hard_face_component_mask[i, j, k]
            if self.velocity_dirichlet_boundary_authority_code_device[None] != 0:
                boundary_mask |= self.velocity_dirichlet_boundary_active_component_mask[i, j, k]
            elif self.velocity_dirichlet_boundary_active[i, j, k] != 0:
                boundary_mask |= 7
            for axis in ti.static(range(3)):
                active = self.obstacle[i, j, k] == 0
                # A backward-MAC component can belong to fluid on its minus
                # side even when its storage cell lies inside an obstacle.
                if ti.static(axis == 0):
                    if i > 0:
                        active = active or self.obstacle[i - 1, j, k] == 0
                elif ti.static(axis == 1):
                    if j > 0:
                        active = active or self.obstacle[i, j - 1, k] == 0
                else:
                    if k > 0:
                        active = active or self.obstacle[i, j, k - 1] == 0
                if active:
                    self._accumulate_cfl_velocity_component(self.velocity[i, j, k][axis], axis)
                if (boundary_mask & (1 << axis)) != 0:
                    self._accumulate_cfl_velocity_component(self.velocity_dirichlet_boundary_value_mps[i, j, k][axis], axis)
        # Directed external faces have independent active component masks.
        for side, j, k in self.external_velocity_boundary_x_face_active_component_mask:
            mask = self.external_velocity_boundary_x_face_active_component_mask[side, j, k]
            for axis in ti.static(range(3)):
                if (mask & (1 << axis)) != 0:
                    self._accumulate_cfl_velocity_component(self.external_velocity_boundary_x_face_value_mps[side, j, k][axis], axis)
        for side, i, k in self.external_velocity_boundary_y_face_active_component_mask:
            mask = self.external_velocity_boundary_y_face_active_component_mask[side, i, k]
            for axis in ti.static(range(3)):
                if (mask & (1 << axis)) != 0:
                    self._accumulate_cfl_velocity_component(self.external_velocity_boundary_y_face_value_mps[side, i, k][axis], axis)
        for side, i, j in self.external_velocity_boundary_z_face_active_component_mask:
            mask = self.external_velocity_boundary_z_face_active_component_mask[side, i, j]
            for axis in ti.static(range(3)):
                if (mask & (1 << axis)) != 0:
                    self._accumulate_cfl_velocity_component(self.external_velocity_boundary_z_face_value_mps[side, i, j][axis], axis)
        max_x = self.cfl_max_abs_velocity_x_mps[None]
        max_y = self.cfl_max_abs_velocity_y_mps[None]
        max_z = self.cfl_max_abs_velocity_z_mps[None]
        return ti.sqrt(max_x * max_x + max_y * max_y + max_z * max_z)

    def _checked_max_fluid_speed_mps(self) -> float:
        maximum = float(self._max_fluid_speed_kernel())
        invalid_count = int(self.cfl_nonfinite_velocity_component_count[None])
        if invalid_count:
            raise FloatingPointError(
                "fluid velocity contains non-finite active components: "
                f"invalid_component_count={invalid_count}"
            )
        return maximum

    def predict(
        self,
        dt_s: float | None = None,
        *,
        advection_scheme: str = "euler",
        kinematic_viscosity_m2_s: float | None = None,
        no_slip_domain_walls: tuple[bool, bool, bool, bool, bool, bool] | None = None,
        pressure_outlet_zmin: bool = False,
        velocity_inlet_zmax: bool | None = None,
    ) -> None:
        step_dt_s = _finite_float(
            self.dt if dt_s is None else dt_s,
            name="dt_s",
            minimum=0.0,
            inclusive=False,
        )
        density_kgm3 = _finite_float(
            self.rho,
            name="density_kgm3",
            minimum=0.0,
            inclusive=False,
        )
        dynamic_viscosity_pa_s = _finite_float(
            self.mu,
            name="viscosity_pa_s",
            minimum=0.0,
            inclusive=True,
        )
        scheme = str(advection_scheme).lower()
        if scheme not in {"euler", "rk2", "muscl_tvd"}:
            raise ValueError(f"unsupported advection_scheme: {advection_scheme!r}")
        nu_m2_s = (
            dynamic_viscosity_pa_s / density_kgm3
            if kinematic_viscosity_m2_s is None
            else float(kinematic_viscosity_m2_s)
        )
        if not math.isfinite(nu_m2_s) or nu_m2_s < 0.0:
            raise ValueError("kinematic_viscosity_m2_s must be finite and non-negative")
        self._last_momentum_advection_requested_time_s = float(step_dt_s)
        self._last_momentum_advection_accepted_time_s = 0.0
        self._last_momentum_advection_remaining_unadvanced_time_s = float(step_dt_s)
        self._require_velocity_dirichlet_component_registry_sealed()
        max_speed_mps = self._checked_max_fluid_speed_mps()
        if scheme != "muscl_tvd":
            advection_cfl = max_speed_mps * step_dt_s / max(
                min(self.dx, self.dy, self.dz),
                1.0e-30,
            )
            if (
                not math.isfinite(advection_cfl)
                or advection_cfl > PREDICTOR_ADVECTION_CFL_FAIL_FAST_LIMIT
            ):
                raise ValueError(
                    "velocity field is too large for explicit predictor advection "
                    "stability; "
                    f"advection_cfl={advection_cfl:g}, "
                    f"limit={PREDICTOR_ADVECTION_CFL_FAIL_FAST_LIMIT:g}"
                )
        diffusion_cfl = 2.0 * nu_m2_s * step_dt_s * (
            1.0 / max(self.dx * self.dx, 1.0e-30)
            + 1.0 / max(self.dy * self.dy, 1.0e-30)
            + 1.0 / max(self.dz * self.dz, 1.0e-30)
        )
        if (
            scheme != "muscl_tvd"
            and self.turbulence_model == "laminar"
            and diffusion_cfl > 1.0
        ):
            raise ValueError(
                "kinematic_viscosity_m2_s is too large for explicit predictor "
                f"diffusion stability; diffusion_cfl={diffusion_cfl:g}"
            )
        if no_slip_domain_walls is None:
            wall_flags = (
                self._sst_no_slip_domain_walls
                if self.turbulence_model == "sst_2003"
                else (False, False, False, False, False, False)
            )
        else:
            wall_flags = tuple(bool(flag) for flag in no_slip_domain_walls)
            if len(wall_flags) != 6:
                raise ValueError(
                    "no_slip_domain_walls must contain six booleans: "
                    "(xmin, xmax, ymin, ymax, zmin, zmax)"
                )
        wall_flag_codes = tuple(1 if flag else 0 for flag in wall_flags)
        velocity_inlet_zmax_mode = self._resolve_physical_face_topology(
            pressure_outlet_zmin=pressure_outlet_zmin,
            velocity_inlet_zmax=velocity_inlet_zmax,
            wall_flag_codes=wall_flag_codes,
        )
        if self.turbulence_model == "laminar" and nu_m2_s > 0.0:
            self._prepare_sst_obstacle_interface_wall_target_masks_kernel(
                self._velocity_dirichlet_boundary_authority_code()
            )
            if scheme != "muscl_tvd":
                self._compute_muscl_momentum_dual_geometry_kernel()
        if scheme == "muscl_tvd":
            # Conservative finite-volume transport uses the same backward-MAC
            # normal mass flux as the projection operator.  SSP-RK2 supplies
            # second-order time accuracy; automatic substeps make the public
            # operation safe for long physical steps without repeated
            # semi-Lagrangian remapping.
            # The MAC dual volumes depend only on the grid and obstacle mask,
            # neither of which changes inside one predictor transaction.  A
            # production step may evaluate three flux states for each of many
            # automatic substeps, so prepare this full-grid geometry once.
            self._compute_muscl_momentum_dual_geometry_kernel()
            remaining_advection_dt_s = float(step_dt_s)
            minimum_advection_substep_s = minimum_admissible_physical_substep_s(
                requested_time_s=float(step_dt_s)
            )
            advection_substeps = 0
            rejected_stage_trials = 0
            conservative_cfl = 0.0
            maximum_substep_cfl = 0.0
            reusable_base_rate_s: float | None = None
            accepted_advection_trial_dts_s: list[float] = []
            while remaining_advection_dt_s > minimum_advection_substep_s:
                if advection_substeps >= self._transport_max_automatic_substeps:
                    raise ValueError(
                        "MUSCL momentum advection substeps exceed the configured "
                        f"safety limit={self._transport_max_automatic_substeps}, "
                        f"remaining_dt_s={remaining_advection_dt_s:g}"
                    )
                stage_dt_cap_s = remaining_advection_dt_s
                consecutive_stage_retries = 0
                while True:
                    self._copy_velocity_to_transport_base_kernel()
                    if reusable_base_rate_s is None:
                        self._compute_muscl_momentum_fluxes(
                            self.velocity_transport_base,
                            wall_flag_codes,
                            dual_geometry_prepared=True,
                            pressure_outlet_zmin=pressure_outlet_zmin,
                            velocity_inlet_zmax=velocity_inlet_zmax,
                        )
                        advection_rate_s = (
                            self._muscl_momentum_advection_rate_s()
                        )
                    else:
                        advection_rate_s = reusable_base_rate_s
                        reusable_base_rate_s = None
                    if (
                        not math.isfinite(advection_rate_s)
                        or advection_rate_s < 0.0
                    ):
                        raise ValueError(
                            "MUSCL momentum transport produced an invalid rate: "
                            f"rate_s={advection_rate_s:g}; "
                            f"{self._muscl_invalid_rate_diagnostics()}"
                        )
                    if advection_substeps == 0 and consecutive_stage_retries == 0:
                        conservative_cfl = advection_rate_s * step_dt_s
                    advection_dt_s = (
                        min(remaining_advection_dt_s, stage_dt_cap_s)
                        if advection_rate_s <= 0.0
                        else min(
                            remaining_advection_dt_s,
                            stage_dt_cap_s,
                            MUSCL_MOMENTUM_INITIAL_CFL_TARGET / advection_rate_s,
                        )
                    )
                    if advection_dt_s <= minimum_advection_substep_s:
                        raise ValueError(
                            "MUSCL SSP-RK2 reached the minimum representable "
                            "physical substep before trial execution: "
                            f"advection_dt_s={advection_dt_s:g}, "
                            "remaining_advection_dt_s="
                            f"{remaining_advection_dt_s:g}, "
                            f"advection_rate_s={advection_rate_s:g}"
                        )
                    self._muscl_momentum_ssp_stage_kernel(
                        self.velocity_transport_base,
                        float(advection_dt_s),
                        0,
                        wall_flag_codes,
                        pressure_outlet_zmin=pressure_outlet_zmin,
                        velocity_inlet_zmax=velocity_inlet_zmax,
                    )
                    self._apply_velocity_dirichlet_boundary_rows_dispatch(
                        read_report=False,
                        preserve_projected_rows=True,
                    )
                    self._copy_velocity_to_prev_kernel()
                    self._compute_muscl_momentum_fluxes(
                        self.velocity_prev,
                        wall_flag_codes,
                        dual_geometry_prepared=True,
                        pressure_outlet_zmin=pressure_outlet_zmin,
                        velocity_inlet_zmax=velocity_inlet_zmax,
                    )
                    stage_rate_s = self._muscl_momentum_advection_rate_s()
                    stage_rate_is_valid = (
                        math.isfinite(stage_rate_s) and stage_rate_s >= 0.0
                    )
                    stage_cfl = (
                        stage_rate_s * advection_dt_s
                        if stage_rate_is_valid
                        else float("inf")
                    )
                    trial_failure = ""
                    retry_rate_s = stage_rate_s
                    final_rate_s = float("nan")
                    final_cfl = float("inf")
                    if (
                        not stage_rate_is_valid
                        or stage_cfl > MUSCL_MOMENTUM_STAGE_CFL_LIMIT
                    ):
                        trial_failure = (
                            "intermediate_stage_rate_or_cfl"
                        )
                    else:
                        self._muscl_momentum_ssp_stage_kernel(
                            self.velocity_prev,
                            float(advection_dt_s),
                            1,
                            wall_flag_codes,
                            pressure_outlet_zmin=pressure_outlet_zmin,
                            velocity_inlet_zmax=velocity_inlet_zmax,
                        )
                        self._apply_velocity_dirichlet_boundary_rows_dispatch(
                            read_report=False,
                            preserve_projected_rows=True,
                        )
                        # The second RK stage is still a trial until its own
                        # physical-face rate is finite and stable.  Without
                        # this commit gate, a bad final stage contaminates the
                        # next slice's rollback base and cannot be recovered.
                        self._compute_muscl_momentum_fluxes(
                            self.velocity,
                            wall_flag_codes,
                            dual_geometry_prepared=True,
                            pressure_outlet_zmin=pressure_outlet_zmin,
                            velocity_inlet_zmax=velocity_inlet_zmax,
                        )
                        final_rate_s = self._muscl_momentum_advection_rate_s()
                        final_rate_is_valid = (
                            math.isfinite(final_rate_s) and final_rate_s >= 0.0
                        )
                        final_cfl = (
                            final_rate_s * advection_dt_s
                            if final_rate_is_valid
                            else float("inf")
                        )
                        retry_rate_s = final_rate_s
                        if (
                            not final_rate_is_valid
                            or final_cfl > MUSCL_MOMENTUM_STAGE_CFL_LIMIT
                        ):
                            trial_failure = "completed_stage_rate_or_cfl"

                    if not trial_failure:
                        maximum_substep_cfl = max(
                            maximum_substep_cfl,
                            advection_rate_s * advection_dt_s,
                            stage_cfl,
                            final_cfl,
                        )
                        # The accepted final state is the next slice's exact
                        # transport base.  Its flux fields and rate are already
                        # resident, so reuse them once instead of rebuilding
                        # the same state before the next SSP-RK2 stage.
                        reusable_base_rate_s = final_rate_s
                        break

                    self._restore_velocity_from_transport_base_kernel()
                    reusable_base_rate_s = None
                    consecutive_stage_retries += 1
                    rejected_stage_trials += 1
                    if (
                        consecutive_stage_retries
                        > MUSCL_MOMENTUM_MAX_CONSECUTIVE_STAGE_RETRIES
                    ):
                        raise ValueError(
                            "MUSCL SSP-RK2 could not commit a stable substep "
                            "after automatic rollback retries: "
                            f"reason={trial_failure}, "
                            f"retries={consecutive_stage_retries}, "
                            f"input_rate_s={advection_rate_s:g}, "
                            f"stage_rate_s={stage_rate_s:g}, "
                            f"final_rate_s={final_rate_s:g}, "
                            f"last_substep_dt_s={advection_dt_s:g}, "
                            f"stage_cfl={stage_cfl:g}, final_cfl={final_cfl:g}"
                        )
                    stage_dt_cap_s = 0.5 * advection_dt_s
                    if math.isfinite(retry_rate_s) and retry_rate_s > 0.0:
                        stage_dt_cap_s = min(
                            stage_dt_cap_s,
                            MUSCL_MOMENTUM_STAGE_RETRY_SAFETY
                            * MUSCL_MOMENTUM_STAGE_CFL_LIMIT
                            / retry_rate_s,
                        )
                    if stage_dt_cap_s <= minimum_advection_substep_s:
                        raise ValueError(
                            "MUSCL SSP-RK2 automatic CFL retry reached the "
                            "minimum representable physical substep: "
                            f"reason={trial_failure}, "
                            f"stage_dt_cap_s={stage_dt_cap_s:g}, "
                            f"retry_rate_s={retry_rate_s:g}"
                        )

                advection_substeps += 1
                accepted_advection_trial_dts_s.append(float(advection_dt_s))
                remaining_advection_dt_s = max(
                    0.0, remaining_advection_dt_s - advection_dt_s
                )
            require_zero_remaining_physical_time_s(
                remaining_time_s=remaining_advection_dt_s,
                component="MUSCL momentum advection",
            )
            accepted_advection_time_s = math.fsum(accepted_advection_trial_dts_s)
            remaining_unadvanced_advection_time_s = (
                float(step_dt_s) - accepted_advection_time_s
            )
            if (
                not math.isfinite(accepted_advection_time_s)
                or not math.isfinite(remaining_unadvanced_advection_time_s)
            ):
                raise FloatingPointError(
                    "MUSCL momentum advection physical-time accounting is non-finite: "
                    f"requested_advection_time_s={step_dt_s:g}, "
                    f"accepted_advection_time_s={accepted_advection_time_s:g}"
                )
            advection_time_tolerance_s = physical_time_roundoff_tolerance_s(
                requested_time_s=float(step_dt_s),
                accepted_time_s=accepted_advection_time_s,
                accepted_substep_count=len(accepted_advection_trial_dts_s),
            )
            if abs(remaining_unadvanced_advection_time_s) > advection_time_tolerance_s:
                raise FloatingPointError(
                    "MUSCL momentum advection physical-time accounting did not close: "
                    f"requested_advection_time_s={step_dt_s:g}, "
                    f"accepted_advection_time_s={accepted_advection_time_s:g}, "
                    "remaining_unadvanced_advection_time_s="
                    f"{remaining_unadvanced_advection_time_s:g}, "
                    f"tolerance_s={advection_time_tolerance_s:g}"
                )
            reported_remaining_unadvanced_advection_time_s = (
                0.0
                if abs(remaining_unadvanced_advection_time_s)
                <= advection_time_tolerance_s
                else remaining_unadvanced_advection_time_s
            )
            self._last_momentum_advection_scheme = "muscl_tvd"
            self._last_momentum_advection_substeps = int(advection_substeps)
            self._last_momentum_advection_cfl = float(conservative_cfl)
            self._last_momentum_advection_max_substep_cfl = float(
                maximum_substep_cfl
            )
            self._last_momentum_advection_rejected_trial_count = int(
                rejected_stage_trials
            )
            self._last_momentum_advection_accepted_time_s = float(
                accepted_advection_time_s
            )
            self._last_momentum_advection_remaining_unadvanced_time_s = float(
                reported_remaining_unadvanced_advection_time_s
            )
        else:
            raise NotImplementedError("advection_scheme must be 'muscl_tvd'")

        if self.turbulence_model == "sst_2003":
            raise NotImplementedError("turbulence models are not supported: the fluid solver is laminar")
        elif scheme == "muscl_tvd" and nu_m2_s > 0.0:
            diffusion_substeps = max(1, int(math.ceil(diffusion_cfl / 0.45)))
            if diffusion_substeps > self._sst_max_automatic_substeps:
                raise ValueError(
                    "laminar momentum diffusion substeps exceed the configured "
                    f"safety limit: required={diffusion_substeps}, "
                    f"limit={self._sst_max_automatic_substeps}, "
                    f"diffusion_cfl={diffusion_cfl:g}"
                )
            diffusion_dt_s = step_dt_s / float(diffusion_substeps)
            for _ in range(diffusion_substeps):
                self._copy_velocity_to_prev_kernel()
                self._laminar_momentum_diffusion_kernel(
                    float(diffusion_dt_s),
                    float(nu_m2_s),
                    *wall_flag_codes,
                    1 if pressure_outlet_zmin else 0,
                    velocity_inlet_zmax_mode,
                )

    @ti.kernel
    def _apply_body_force_kernel(
        self,
        dt_s: ti.f32,
        density_kgm3: ti.f32,
    ):
        # the velocity update only (speed item 1): the impulse report sums over every cell, with atomics on one
        # address, cost tens of ms per step on large grids and are never read (read_report=True is unsupported)
        for i, j, k in self.velocity:
            if self.obstacle[i, j, k] == 0:
                delta_v = self.force[i, j, k] * (dt_s / density_kgm3)
                self.velocity[i, j, k] += delta_v

    def apply_body_force(
        self,
        dt_s: float | None = None,
        *,
        read_report: bool = True,
    ) -> None:
        step_dt_s = _finite_float(
            self.dt if dt_s is None else dt_s,
            name="dt_s",
            minimum=0.0,
            inclusive=False,
        )
        density_kgm3 = _finite_float(
            self.rho,
            name="density_kgm3",
            minimum=0.0,
            inclusive=False,
        )
        self._apply_body_force_kernel(
            step_dt_s,
            density_kgm3,
        )
        if not read_report:
            return None
        raise NotImplementedError("per-call solver reports (read_report=True) are not supported; use compute_divergence() and the simulation monitors")

    @ti.kernel
    def _compute_divergence_kernel(
        self,
        pressure_outlet_zmin: ti.i32,
        velocity_inlet_zmax_mode: ti.i32,
        canonical_authority: ti.template(),
    ):
        for i, j, k in self.divergence:
            if self.obstacle[i, j, k] == 1:
                self.divergence[i, j, k] = 0.0
            else:
                left_velocity_x = 0.0
                right_velocity_x = 0.0
                if i == 0:
                    left_velocity_x = self._physical_exterior_normal_velocity(
                        self.velocity, 0, 0, i, j, k,
                        pressure_outlet_zmin, velocity_inlet_zmax_mode,
                    )
                elif i > 0:
                    if self.obstacle[i - 1, j, k] == 0:
                        left_velocity_x = self.velocity[i, j, k].x
                    elif self._obstacle_interface_component_is_prescribed(
                        i, j, k, 0, canonical_authority
                    ):
                        # Backward-MAC orientation A: this storage row is
                        # fluid and its minus-side cell is obstacle, so the
                        # prescribed wall face is the cell's left face.
                        left_velocity_x = self._continuity_obstacle_face_velocity(
                            i, j, k, 0
                        )
                if i == self.nx - 1:
                    right_velocity_x = self._physical_exterior_normal_velocity(
                        self.velocity, 0, 1, i, j, k,
                        pressure_outlet_zmin, velocity_inlet_zmax_mode,
                    )
                elif i < self.nx - 1:
                    if self.obstacle[i + 1, j, k] == 0:
                        right_velocity_x = self.velocity[i + 1, j, k].x
                    elif self._obstacle_interface_component_is_prescribed(
                        i + 1, j, k, 0, canonical_authority
                    ):
                        # Backward-MAC orientation B: the obstacle storage
                        # row owns this fluid cell's prescribed right face.
                        right_velocity_x = self._continuity_obstacle_face_velocity(
                            i + 1, j, k, 0
                        )

                back_velocity_y = 0.0
                front_velocity_y = 0.0
                if j == 0:
                    back_velocity_y = self._physical_exterior_normal_velocity(
                        self.velocity, 1, 0, i, j, k,
                        pressure_outlet_zmin, velocity_inlet_zmax_mode,
                    )
                elif j > 0:
                    if self.obstacle[i, j - 1, k] == 0:
                        back_velocity_y = self.velocity[i, j, k].y
                    elif self._obstacle_interface_component_is_prescribed(
                        i, j, k, 1, canonical_authority
                    ):
                        back_velocity_y = self._continuity_obstacle_face_velocity(
                            i, j, k, 1
                        )
                if j == self.ny - 1:
                    front_velocity_y = self._physical_exterior_normal_velocity(
                        self.velocity, 1, 1, i, j, k,
                        pressure_outlet_zmin, velocity_inlet_zmax_mode,
                    )
                elif j < self.ny - 1:
                    if self.obstacle[i, j + 1, k] == 0:
                        front_velocity_y = self.velocity[i, j + 1, k].y
                    elif self._obstacle_interface_component_is_prescribed(
                        i, j + 1, k, 1, canonical_authority
                    ):
                        front_velocity_y = self._continuity_obstacle_face_velocity(
                            i, j + 1, k, 1
                        )

                bottom_velocity_z = 0.0
                top_velocity_z = 0.0
                if k == 0:
                    bottom_velocity_z = self._physical_exterior_normal_velocity(
                        self.velocity, 2, 0, i, j, k,
                        pressure_outlet_zmin, velocity_inlet_zmax_mode,
                    )
                elif k > 0:
                    if self.obstacle[i, j, k - 1] == 0:
                        bottom_velocity_z = self.velocity[i, j, k].z
                    elif self._obstacle_interface_component_is_prescribed(
                        i, j, k, 2, canonical_authority
                    ):
                        bottom_velocity_z = self._continuity_obstacle_face_velocity(
                            i, j, k, 2
                        )
                if k == self.nz - 1:
                    top_velocity_z = self._physical_exterior_normal_velocity(
                        self.velocity, 2, 1, i, j, k,
                        pressure_outlet_zmin, velocity_inlet_zmax_mode,
                    )
                elif k < self.nz - 1:
                    if self.obstacle[i, j, k + 1] == 0:
                        top_velocity_z = self.velocity[i, j, k + 1].z
                    elif self._obstacle_interface_component_is_prescribed(
                        i, j, k + 1, 2, canonical_authority
                    ):
                        top_velocity_z = self._continuity_obstacle_face_velocity(
                            i, j, k + 1, 2
                        )

                self.divergence[i, j, k] = (
                    (right_velocity_x - left_velocity_x) / self.cell_width_x_m[i]
                    + (front_velocity_y - back_velocity_y) / self.cell_width_y_m[j]
                    + (top_velocity_z - bottom_velocity_z) / self.cell_width_z_m[k]
                )

    def compute_divergence(
        self,
        *,
        pressure_outlet_zmin: bool = False,
        velocity_inlet_zmax: bool | None = None,
    ) -> None:
        self._require_velocity_dirichlet_component_registry_sealed()
        canonical_authority = self._velocity_dirichlet_boundary_authority_code()
        velocity_inlet_zmax_mode = self._resolve_velocity_inlet_zmax_topology_mode(
            velocity_inlet_zmax,
            canonical_authority=canonical_authority,
        )
        self._compute_divergence_with_topology_mode(
            pressure_outlet_zmin=pressure_outlet_zmin,
            velocity_inlet_zmax_mode=velocity_inlet_zmax_mode,
            canonical_authority=canonical_authority,
        )

    def _compute_divergence_with_topology_mode(
        self,
        *,
        pressure_outlet_zmin: bool,
        velocity_inlet_zmax_mode: int,
        canonical_authority: int,
    ) -> None:
        self._compute_divergence_kernel(
            1 if pressure_outlet_zmin else 0,
            int(velocity_inlet_zmax_mode),
            int(canonical_authority),
        )

    def project(
        self,
        iterations: int = 40,
        pressure_outlet_zmin: bool = False,
        velocity_inlet_zmax: bool | None = None,
        pressure_outlet_backflow_policy: str = "clamp",
        obstacle_normal_velocity_policy: str = "face_clamp",
        dt_s: float | None = None,
        preserve_velocity_constraints: bool = False,
        velocity_dirichlet_soft_rows_already_applied: bool = False,
        velocity_constraint_blend: float = 1.0,
        velocity_constraint_solid_mobility_ratio: float = 0.0,
        divergence_cleanup_iterations: int = 0,
        divergence_cleanup_relaxation: float = 0.7,
        reset_pressure: bool = False,
        accumulate_pressure_into_previous: bool = False,
        pressure_solver: str = "jacobi",
        multigrid_cycles: int | None = None,
        cg_tolerance: float = 1.0e-6,
        cg_preconditioner: str = "auto",
        pressure_solve_failure_policy: str = "report",
        pressure_solve_context: Mapping[str, object] | None = None,
        tiny_unreached_cleanup_component_cells: int = 0,
        pressure_reachability_prepared: bool = False,
        pre_projection_velocity_projector: object | None = None,
        read_report: bool = True,
        # Opt-in CG initial-guess warm start (perf only; see
        # invalidate_pressure_warmstart() for the stash-lifetime contract).
        # -1 (default) never touches the stash and adds zero kernel
        # launches, so every existing caller is byte-for-byte unchanged.
        # 0..PRESSURE_WARMSTART_SLOT_COUNT-1 copies that slot's stashed
        # pressure into self.pressure before solving (if the slot has ever
        # been written) and copies the solved pressure back into the slot
        # afterward.
        warm_start_slot: int = -1,
        homogenize_pressure_interface_rhs_for_increment: bool = False,
        pressure_velocity_nullspace_projector: object | None = None,
    ) -> dict[str, float]:
        iterations = _validated_count(
            iterations,
            name="iterations",
            allow_zero=False,
        )
        divergence_cleanup_iterations = _validated_count(
            divergence_cleanup_iterations,
            name="divergence_cleanup_iterations",
            allow_zero=True,
        )
        step_dt_s = _finite_float(
            self.dt if dt_s is None else dt_s,
            name="dt_s",
            minimum=0.0,
            inclusive=False,
        )
        density_kgm3 = _finite_float(
            self.rho,
            name="density_kgm3",
            minimum=0.0,
            inclusive=False,
        )
        rhs_scale, inv_dx2, inv_dy2, inv_dz2 = _pressure_poisson_coefficients(
            density_kgm3=density_kgm3,
            dt_s=step_dt_s,
            spacing_m=(self.dx, self.dy, self.dz),
        )
        if pressure_outlet_zmin:
            self._validate_pressure_outlet_zmin_boundary()
        self._require_velocity_dirichlet_component_registry_sealed()
        homogenize_pressure_interface_rhs_for_increment_enabled = bool(
            homogenize_pressure_interface_rhs_for_increment
        )
        if (
            homogenize_pressure_interface_rhs_for_increment_enabled
            and not bool(accumulate_pressure_into_previous)
        ):
            raise ValueError(
                "homogenize_pressure_interface_rhs_for_increment requires "
                "accumulate_pressure_into_previous=True"
            )
        pressure_solve_context_payload = _validated_pressure_solve_context(
            pressure_solve_context
        )
        canonical_authority = self._velocity_dirichlet_boundary_authority_code()
        velocity_inlet_zmax_mode = self._resolve_velocity_inlet_zmax_topology_mode(
            velocity_inlet_zmax,
            canonical_authority=canonical_authority,
        )

        def compute_current_divergence() -> None:
            self._compute_divergence_with_topology_mode(
                pressure_outlet_zmin=pressure_outlet_zmin,
                velocity_inlet_zmax_mode=velocity_inlet_zmax_mode,
                canonical_authority=canonical_authority,
            )
        pressure_outlet_backflow_policy_name = str(pressure_outlet_backflow_policy)
        if pressure_outlet_backflow_policy_name not in PRESSURE_OUTLET_BACKFLOW_POLICIES:
            raise ValueError(
                "pressure_outlet_backflow_policy must be 'clamp' or 'allow'"
            )
        obstacle_normal_velocity_policy_name = str(obstacle_normal_velocity_policy)
        if obstacle_normal_velocity_policy_name not in OBSTACLE_NORMAL_VELOCITY_POLICIES:
            raise ValueError(
                "obstacle_normal_velocity_policy must be 'face_clamp' "
                "or 'cell_zero_only'"
            )
        constraint_blend = float(velocity_constraint_blend)
        if not math.isfinite(constraint_blend) or not 0.0 <= constraint_blend <= 1.0:
            raise ValueError("velocity_constraint_blend must be in [0, 1]")
        constraint_solid_mobility_ratio = float(velocity_constraint_solid_mobility_ratio)
        if (
            not math.isfinite(constraint_solid_mobility_ratio)
            or constraint_solid_mobility_ratio < 0.0
        ):
            raise ValueError(
                "velocity_constraint_solid_mobility_ratio must be a finite non-negative number"
            )
        cleanup_relaxation = float(divergence_cleanup_relaxation)
        if not math.isfinite(cleanup_relaxation) or not 0.0 <= cleanup_relaxation <= 1.0:
            raise ValueError("divergence_cleanup_relaxation must be in [0, 1]")
        cg_relative_tolerance = float(cg_tolerance)
        if not math.isfinite(cg_relative_tolerance) or cg_relative_tolerance < 0.0:
            raise ValueError("cg_tolerance must be a finite non-negative number")
        cg_preconditioner_requested_name = str(cg_preconditioner)
        cg_preconditioner_name = cg_preconditioner_requested_name
        if cg_preconditioner_requested_name not in CG_PRECONDITIONER_CHOICES:
            raise ValueError(f"unsupported cg_preconditioner: {cg_preconditioner!r}")
        pressure_solve_failure_policy_name = str(pressure_solve_failure_policy)
        if pressure_solve_failure_policy_name not in {"report", "raise"}:
            raise ValueError(
                "pressure_solve_failure_policy must be 'report' or 'raise'"
            )
        pressure_solver_requested_name = str(pressure_solver)
        pressure_solver_name = pressure_solver_requested_name
        if pressure_solver_name not in {"jacobi", "compact_jacobi", "fv_jacobi", "fv_multigrid", "fv_cg"}:
            raise ValueError(f"unsupported pressure_solver: {pressure_solver!r}")
        if not self.grid.is_uniform:
            if pressure_solver_name not in {"fv_jacobi", "fv_multigrid", "fv_cg"}:
                raise ValueError(
                    "non-uniform CartesianGrid projection requires an FV pressure solver"
                )
            if divergence_cleanup_iterations > 0:
                raise ValueError(
                    "non-uniform CartesianGrid divergence cleanup requires non-uniform cleanup operators"
                )
        if multigrid_cycles is not None and int(multigrid_cycles) <= 0:
            raise ValueError("multigrid_cycles must be positive")
        warm_start_slot_index = int(warm_start_slot)
        if warm_start_slot_index != -1 and not (
            0 <= warm_start_slot_index < len(self._pressure_warmstart_slots)
        ):
            raise ValueError(
                "warm_start_slot must be -1 (disabled) or in "
                f"[0, {len(self._pressure_warmstart_slots)}), "
                f"got {warm_start_slot_index}"
            )
        report_requested = bool(read_report)
        reachability_prepared = bool(pressure_reachability_prepared)
        pressure_marker_nullspace_report: dict[str, object] = {
            "pressure_marker_nullspace_enabled": bool(
                pressure_velocity_nullspace_projector is not None
            ),
            "pressure_marker_nullspace_prepared": False,
            "pressure_marker_nullspace_active_constraint_count": 0,
            "pressure_marker_nullspace_independent_constraint_count": 0,
            "pressure_marker_nullspace_dependent_constraint_count": 0,
            "pressure_marker_nullspace_unactuated_constraint_count": 0,
            "pressure_marker_nullspace_apply_count": 0,
            "pressure_marker_nullspace_pressure_actuation_generation": 0,
            "pressure_marker_nullspace_min_factor_pivot": 0.0,
            "pressure_marker_nullspace_max_dependent_normalized_pivot": 0.0,
            "pressure_marker_nullspace_max_input_constraint_mps": 0.0,
            "pressure_marker_nullspace_max_unactuated_input_constraint_mps": 0.0,
            "pressure_marker_nullspace_max_constraint_residual_mps": 0.0,
            "pressure_marker_nullspace_solver_scratch_resource_bytes": 0,
            "pressure_marker_nullspace_marker_operator_resource_bytes": 0,
            "pressure_marker_nullspace_resource_bytes": 0,
            "pressure_marker_nullspace_actuation_invalid_count": 0,
            "pressure_marker_nullspace_correction_invalid_count": 0,
            "pressure_marker_nullspace_operator_apply_count": 0,
            "pressure_marker_nullspace_velocity_correction_apply_count": 0,
            "pressure_marker_nullspace_all_velocity_paths_projected": False,
        }
        if reachability_prepared and not bool(pressure_outlet_zmin):
            raise ValueError(
                "pressure_reachability_prepared requires pressure_outlet_zmin=True"
            )
        self._require_face_symmetric_owned_soft_compatibility()
        self.last_divergence_report_host_reads = 0
        empty_stats = {"max_abs": math.nan, "l2": math.nan}
        (
            velocity_dirichlet_hard_fixed_routing_row_count,
            velocity_dirichlet_hard_fixed_routing_component_face_count,
        ) = self._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask()
        # This preflight is deliberately unconditional: NaN-only or malformed
        # all-reachable storage must fail before the policy probe can call it
        # inactive and before any pressure dispatcher/matvec is entered.
        self._preflight_pressure_interface_operator_storage(context="project")
        if homogenize_pressure_interface_rhs_for_increment_enabled:
            # A correction solve reuses the assembled interface operator but
            # solves for delta-p.  Its fixed pressure-Neumann affine term is
            # therefore homogeneous.  Clear only after the unconditional
            # storage preflight so NaN/Inf cannot be hidden by this mode.
            raise NotImplementedError("homogenize_pressure_interface_rhs_for_increment=True is not supported")
        pressure_interface_policy_report = self._pressure_interface_matrix_policy_report()
        pressure_interface_matrix_active = (
            int(pressure_interface_policy_report["active_cells"]) > 0
            or int(pressure_interface_policy_report["row_count"]) > 0
            or int(pressure_interface_policy_report["row_active_count"]) > 0
            or float(pressure_interface_policy_report["max_abs_diagonal"]) > 0.0
        )
        pressure_solver_forced_to_fv_cg = False
        pressure_solver_force_reason = ""
        pressure_solver_force_reasons: list[str] = []
        closed_neumann_requires_fv_cg = (
            not bool(pressure_outlet_zmin)
            and pressure_solver_requested_name != "fv_cg"
        )
        if closed_neumann_requires_fv_cg:
            # A closed pressure graph has a constant nullspace and ordinary
            # obstacles can split it into multiple disconnected components.
            # Only the FV-CG path measures per-component compatibility and
            # applies a separate zero-mean projection to each component.
            pressure_solver_name = "fv_cg"
            pressure_solver_forced_to_fv_cg = True
            pressure_solver_force_reason = (
                "closed_neumann_domain_requires_component_aware_fv_cg"
            )
            pressure_solver_force_reasons.append(pressure_solver_force_reason)
            if pressure_solver_requested_name == "fv_multigrid":
                cg_preconditioner_name = "fv_multigrid"
        hard_fixed_faces_require_fv_cg = (
            velocity_dirichlet_hard_fixed_routing_component_face_count > 0
            and (
                pressure_solver_requested_name == "fv_multigrid"
                or (
                    not bool(pressure_outlet_zmin)
                    and pressure_solver_requested_name != "fv_cg"
                )
            )
        )
        if hard_fixed_faces_require_fv_cg:
            # Exact orientation-specific faces remain on the fine grid.
            # FV-CG retains the exact fine-grid residual and, for a closed
            # Neumann domain, supplies component compatibility and nullspace
            # handling when hard faces split the pressure graph.
            pressure_solver_name = "fv_cg"
            pressure_solver_forced_to_fv_cg = True
            pressure_solver_force_reason = (
                "hard_fixed_velocity_faces_require_fine_grid_exact_fv_cg"
            )
            pressure_solver_force_reasons.append(pressure_solver_force_reason)
            if pressure_solver_requested_name == "fv_multigrid":
                # Multigrid remains safe as a preconditioner because the
                # convergence gate is evaluated with the exact fine operator.
                cg_preconditioner_name = "fv_multigrid"
        if (
            pressure_interface_matrix_active
            and pressure_solver_requested_name
            in PRESSURE_INTERFACE_MATRIX_FV_CG_REQUIRED_SOLVERS
        ):
            pressure_solver_name = "fv_cg"
            pressure_solver_forced_to_fv_cg = True
            pressure_solver_force_reason = PRESSURE_INTERFACE_MATRIX_FV_CG_FORCE_REASON
            if pressure_solver_force_reason not in pressure_solver_force_reasons:
                pressure_solver_force_reasons.append(pressure_solver_force_reason)
        if pressure_velocity_nullspace_projector is not None:
            raise NotImplementedError("a pressure_velocity_nullspace_projector is not supported")
        # A positive interface diagonal is not by itself a pressure-to-ground
        # anchor: conservative row-list edges contribute matching positive
        # diagonals and negative off-diagonals.  Closed-domain anchoring is
        # classified below from diagonal excess on the full operator graph.
        pressure_system_anchored_by_interface_matrix = False
        pressure_nullspace_zero_mean_projection_applied = (
            not bool(pressure_outlet_zmin)
            and pressure_solver_name == "fv_cg"
        )
        if pressure_outlet_zmin:
            pressure_nullspace_policy = "pressure_outlet_dirichlet"
            pressure_nullspace_compatibility_measured = True
        elif pressure_solver_name == "fv_cg":
            pressure_nullspace_policy = "closed_neumann_fv_cg_zero_mean"
            pressure_nullspace_compatibility_measured = True
        else:
            pressure_nullspace_policy = "closed_neumann_non_cg_unmeasured"
            pressure_nullspace_compatibility_measured = False
        pressure_graph_component_count = (
            1
            if not bool(pressure_outlet_zmin)
            and pressure_solver_name == "fv_cg"
            else 0
        )
        pressure_nullspace_component_count = (
            1
            if not bool(pressure_outlet_zmin)
            and pressure_solver_name == "fv_cg"
            else 0
        )
        pressure_nullspace_component_labels_converged = True
        pressure_nullspace_component_overflow = False
        pressure_nullspace_componentwise_projection_applied = False
        outlet_pressure_operator_graph_prepared = False
        pressure_nullspace_incompatible_component_count = 0
        pressure_nullspace_component_rhs_mean_max_abs = 0.0
        pressure_nullspace_component_rhs_integral_max_abs = 0.0
        self.last_project_cg_project_calls = 0
        self.last_project_cg_iterations_total = 0
        self.last_project_cg_operator_apply_count = 0
        self.last_cg_operator_apply_count = 0
        self.last_project_cg_iterations_max = 0
        self.last_project_cg_host_residual_checks = 0
        self.last_project_cg_mean_projection_count = 0
        self.last_project_cg_unreached_set_mean_projection_count = 0
        self.last_project_cg_componentwise_mean_projection_count = 0
        self.last_project_cg_restart_count = 0
        self.last_project_cg_restart_count_measured = pressure_solver_name == "fv_cg"
        self.last_project_cg_restart_policy = (
            "periodic_exact_residual"
            if pressure_solver_name == "fv_cg"
            else "not_applicable_non_cg"
        )
        self.last_project_cg_initial_relative_residual_max = 0.0
        self.last_project_cg_relative_residual_max = 0.0
        self.last_project_cg_converged_all = True
        self.last_project_cg_breakdown_count = 0
        self.last_project_cg_breakdown_code = 0
        self.last_project_cg_breakdown_dAd = 0.0
        self.last_project_cg_breakdown = ""
        self.last_project_cg_preconditioner_requested = (
            cg_preconditioner_requested_name
        )
        self.last_project_cg_preconditioner_effective = "not_run"
        self.last_project_cg_nonzero_rhs_project_calls = 0
        self.last_project_cg_nonzero_rhs_preconditioner_requested = (
            "not_applicable"
        )
        self.last_project_cg_nonzero_rhs_preconditioner_effective = (
            "not_applicable"
        )
        self.last_project_cg_nonzero_rhs_multigrid_to_jacobi_fallback_count = 0
        self.last_project_cg_multigrid_apply_count = 0
        self.last_project_cg_multigrid_to_jacobi_fallback_count = 0
        self.last_project_cg_exact_residual_confirmation_count = 0
        self.last_project_cg_exact_relative_residual_max = 0.0
        self.last_project_pressure_projection_physical_failure = False
        self.last_project_pressure_projection_physical_failure_reason = ""
        self.last_project_pressure_projection_physical_failure_action = "none"
        self.last_project_pressure_nullspace_incompatible_component_count = 0
        self.last_project_pressure_nullspace_component_rhs_mean_max_abs = 0.0
        self.last_project_pressure_nullspace_component_rhs_integral_max_abs = 0.0
        self.last_cg_closed_neumann_physical_rhs_measured = False
        self.last_cg_closed_neumann_physical_rhs_mean = 0.0
        self.last_cg_closed_neumann_physical_rhs_integral = 0.0
        self.last_cg_closed_neumann_physical_component_budgets = ()
        pressure_solve_failed = False
        pressure_solve_failure_action = "none"
        pressure_projection_physical_failure = False
        pressure_projection_physical_failure_reason = ""
        pressure_projection_physical_failure_action = "none"
        pressure_projection_unreached_incompatible_component_count = 0
        pressure_projection_unreached_rhs_mean_max_abs = 0.0
        pressure_projection_unreached_rhs_integral_max_abs = 0.0
        projection_overflow_singleton_cleanup_cell_count = 0
        projection_overflow_singleton_cleanup_component_count = 0
        projection_tiny_unreached_cleanup_cell_count = 0
        projection_tiny_unreached_cleanup_component_count = 0
        projection_topology_mutated = False

        def record_cg_stats() -> None:
            nonlocal pressure_solve_failed
            nonlocal pressure_nullspace_incompatible_component_count
            nonlocal pressure_nullspace_component_rhs_mean_max_abs
            nonlocal pressure_nullspace_component_rhs_integral_max_abs
            if pressure_solver_name != "fv_cg":
                return
            self.last_project_cg_project_calls += 1
            if self.last_project_cg_project_calls == 1:
                self.last_project_cg_preconditioner_effective = str(
                    self.last_cg_preconditioner_effective
                )
            elif self.last_project_cg_preconditioner_effective != str(
                self.last_cg_preconditioner_effective
            ):
                self.last_project_cg_preconditioner_effective = "mixed"
            self.last_project_cg_multigrid_apply_count += int(
                self.last_cg_multigrid_apply_count
            )
            self.last_project_cg_multigrid_to_jacobi_fallback_count += int(
                self.last_cg_multigrid_to_jacobi_fallback_count
            )
            if bool(self.last_cg_rhs_nonzero):
                self.last_project_cg_nonzero_rhs_project_calls += 1
                if self.last_project_cg_nonzero_rhs_project_calls == 1:
                    self.last_project_cg_nonzero_rhs_preconditioner_requested = str(
                        self.last_cg_preconditioner_requested
                    )
                    self.last_project_cg_nonzero_rhs_preconditioner_effective = str(
                        self.last_cg_preconditioner_effective
                    )
                else:
                    if (
                        self.last_project_cg_nonzero_rhs_preconditioner_requested
                        != str(self.last_cg_preconditioner_requested)
                    ):
                        self.last_project_cg_nonzero_rhs_preconditioner_requested = (
                            "mixed"
                        )
                    if (
                        self.last_project_cg_nonzero_rhs_preconditioner_effective
                        != str(self.last_cg_preconditioner_effective)
                    ):
                        self.last_project_cg_nonzero_rhs_preconditioner_effective = (
                            "mixed"
                        )
                (
                    self.last_project_cg_nonzero_rhs_multigrid_to_jacobi_fallback_count
                ) += int(self.last_cg_multigrid_to_jacobi_fallback_count)
            self.last_project_cg_exact_residual_confirmation_count += int(
                self.last_cg_exact_residual_confirmation_count
            )
            if math.isfinite(float(self.last_cg_exact_relative_residual)):
                self.last_project_cg_exact_relative_residual_max = max(
                    float(self.last_project_cg_exact_relative_residual_max),
                    float(self.last_cg_exact_relative_residual),
                )
            else:
                self.last_project_cg_exact_relative_residual_max = math.inf
            self.last_project_cg_iterations_total += int(self.last_cg_iterations)
            self.last_project_cg_operator_apply_count += int(
                self.last_cg_operator_apply_count
            )
            self.last_project_cg_iterations_max = max(
                int(self.last_project_cg_iterations_max),
                int(self.last_cg_iterations),
            )
            self.last_project_cg_host_residual_checks += int(self.last_cg_host_residual_checks)
            self.last_project_cg_mean_projection_count += int(
                self.last_cg_mean_projection_count
            )
            self.last_project_cg_unreached_set_mean_projection_count += int(
                self.last_cg_unreached_set_mean_projection_count
            )
            self.last_project_cg_componentwise_mean_projection_count += int(
                self.last_cg_componentwise_mean_projection_count
            )
            if (
                pressure_nullspace_componentwise_projection_applied
                or pressure_nullspace_zero_mean_projection_applied
            ):
                pressure_nullspace_incompatible_component_count = max(
                    int(pressure_nullspace_incompatible_component_count),
                    int(self.last_unreached_incompatible_component_count),
                )
                pressure_nullspace_component_rhs_mean_max_abs = max(
                    float(pressure_nullspace_component_rhs_mean_max_abs),
                    float(self.last_unreached_component_rhs_mean_max_abs),
                )
                pressure_nullspace_component_rhs_integral_max_abs = max(
                    float(pressure_nullspace_component_rhs_integral_max_abs),
                    float(self.last_unreached_component_rhs_integral_max_abs),
                )
            self.last_project_cg_restart_count += int(self.last_cg_restart_count)
            self.last_project_cg_restart_count_measured = bool(
                self.last_cg_restart_count_measured
            )
            self.last_project_cg_restart_policy = str(self.last_cg_restart_policy)
            if math.isfinite(float(self.last_cg_initial_relative_residual)):
                self.last_project_cg_initial_relative_residual_max = max(
                    float(self.last_project_cg_initial_relative_residual_max),
                    float(self.last_cg_initial_relative_residual),
                )
            if math.isfinite(float(self.last_cg_relative_residual)):
                self.last_project_cg_relative_residual_max = max(
                    float(self.last_project_cg_relative_residual_max),
                    float(self.last_cg_relative_residual),
                )
            if not bool(self.last_cg_converged):
                pressure_solve_failed = True
                self.last_project_cg_converged_all = False
                self.last_project_cg_breakdown_count += 1
                self.last_project_cg_breakdown_code = max(
                    int(self.last_project_cg_breakdown_code),
                    int(self.cg_breakdown_code[None]),
                )
                if math.isfinite(float(self.last_cg_breakdown_dAd)):
                    self.last_project_cg_breakdown_dAd = float(
                        self.last_cg_breakdown_dAd
                    )
                if self.last_cg_breakdown:
                    self.last_project_cg_breakdown = str(self.last_cg_breakdown)

        def record_projection_physical_stats() -> None:
            nonlocal pressure_projection_physical_failure
            nonlocal pressure_projection_physical_failure_reason
            nonlocal pressure_projection_unreached_incompatible_component_count
            nonlocal pressure_projection_unreached_rhs_mean_max_abs
            nonlocal pressure_projection_unreached_rhs_integral_max_abs
            if pressure_solver_name != "fv_cg":
                return
            if (
                pressure_nullspace_componentwise_projection_applied
                or pressure_nullspace_zero_mean_projection_applied
            ):
                if pressure_nullspace_incompatible_component_count > 0:
                    pressure_projection_physical_failure = True
                    pressure_projection_physical_failure_reason = (
                        "unreached_component_rhs_incompatible"
                        if pressure_outlet_zmin
                        else "closed_neumann_component_rhs_incompatible"
                    )
                if pressure_outlet_zmin:
                    pressure_projection_unreached_incompatible_component_count = max(
                        int(
                            pressure_projection_unreached_incompatible_component_count
                        ),
                        int(pressure_nullspace_incompatible_component_count),
                    )
                    pressure_projection_unreached_rhs_mean_max_abs = max(
                        float(pressure_projection_unreached_rhs_mean_max_abs),
                        float(pressure_nullspace_component_rhs_mean_max_abs),
                    )
                    pressure_projection_unreached_rhs_integral_max_abs = max(
                        float(pressure_projection_unreached_rhs_integral_max_abs),
                        float(pressure_nullspace_component_rhs_integral_max_abs),
                    )
                return
            if not pressure_outlet_zmin:
                return
            if outlet_pressure_operator_graph_prepared:
                # Every final operator root is anchored, so the physical flood
                # partition (including a legitimate diagnostic overflow) is
                # not a pressure nullspace and must not fail this gate.
                return
            incompatible_count = int(
                self.last_unreached_incompatible_component_count
            )
            component_overflow = bool(
                self.last_pressure_unreached_component_overflow
            )
            labels_converged = bool(
                self.last_pressure_component_labels_converged
            )
            pressure_projection_unreached_incompatible_component_count = max(
                int(pressure_projection_unreached_incompatible_component_count),
                incompatible_count,
            )
            pressure_projection_unreached_rhs_mean_max_abs = max(
                float(pressure_projection_unreached_rhs_mean_max_abs),
                float(self.last_unreached_component_rhs_mean_max_abs),
            )
            pressure_projection_unreached_rhs_integral_max_abs = max(
                float(pressure_projection_unreached_rhs_integral_max_abs),
                float(self.last_unreached_component_rhs_integral_max_abs),
            )
            if not labels_converged:
                pressure_projection_physical_failure = True
                pressure_projection_physical_failure_reason = (
                    "unreached_component_labels_not_converged"
                )
            elif component_overflow:
                pressure_projection_physical_failure = True
                pressure_projection_physical_failure_reason = (
                    "unreached_component_label_overflow"
                )
            elif incompatible_count > 0:
                pressure_projection_physical_failure = True
                pressure_projection_physical_failure_reason = (
                    "unreached_component_rhs_incompatible"
                )

        def record_projection_structural_preflight() -> None:
            """Fail closed on invalid component labels before the pressure solve."""

            nonlocal pressure_projection_physical_failure
            nonlocal pressure_projection_physical_failure_reason
            if (
                pressure_solver_name != "fv_cg"
                or not pressure_outlet_zmin
                or pressure_nullspace_componentwise_projection_applied
                or outlet_pressure_operator_graph_prepared
            ):
                return
            if not bool(self.last_pressure_component_labels_converged):
                pressure_projection_physical_failure = True
                pressure_projection_physical_failure_reason = (
                    "unreached_component_labels_not_converged"
                )
            elif bool(self.last_pressure_unreached_component_overflow):
                pressure_projection_physical_failure = True
                pressure_projection_physical_failure_reason = (
                    "unreached_component_label_overflow"
                )

        def handle_pressure_solve_failure() -> None:
            nonlocal pressure_solve_failure_action
            if pressure_solver_name != "fv_cg" or bool(self.last_cg_converged):
                return
            if pressure_solve_failure_policy_name != "raise":
                raise NotImplementedError("pressure_solve_failure_policy='report' is not supported: a pressure solve that does not converge raises (use 'raise')")
            diagnostics = self._pressure_solve_failure_diagnostics(
                tolerance=cg_relative_tolerance,
                pressure_interface_matrix_active=pressure_interface_matrix_active,
                context=pressure_solve_context_payload,
            )
            raise PressureSolveConvergenceError(
                "FV-CG pressure solve did not converge before pressure-gradient "
                "velocity correction "
                f"(iterations={int(self.last_cg_iterations)}, "
                f"relative_residual={float(self.last_cg_relative_residual):.6g}, "
                f"breakdown={self.last_cg_breakdown!r}, "
                f"preconditioner_requested={self.last_cg_preconditioner_requested!r}, "
                f"preconditioner_effective={self.last_cg_preconditioner_effective!r}, "
                f"multigrid_apply_count={int(self.last_cg_multigrid_apply_count)}, "
                "multigrid_to_jacobi_fallback_count="
                f"{int(self.last_cg_multigrid_to_jacobi_fallback_count)}, "
                "exact_residual_confirmation_count="
                f"{int(self.last_cg_exact_residual_confirmation_count)})",
                diagnostics=diagnostics,
            )

        def handle_projection_physical_failure() -> None:
            nonlocal pressure_projection_physical_failure_action
            self.last_project_pressure_projection_physical_failure = bool(
                pressure_projection_physical_failure
            )
            self.last_project_pressure_projection_physical_failure_reason = str(
                pressure_projection_physical_failure_reason
            )
            self.last_project_pressure_nullspace_incompatible_component_count = int(
                pressure_nullspace_incompatible_component_count
            )
            self.last_project_pressure_nullspace_component_rhs_mean_max_abs = float(
                pressure_nullspace_component_rhs_mean_max_abs
            )
            self.last_project_pressure_nullspace_component_rhs_integral_max_abs = float(
                pressure_nullspace_component_rhs_integral_max_abs
            )
            if not bool(pressure_projection_physical_failure):
                self.last_project_pressure_projection_physical_failure_action = (
                    "none"
                )
                return
            if pressure_solve_failure_policy_name != "raise":
                raise NotImplementedError("pressure_solve_failure_policy='report' is not supported: a pressure solve that does not converge raises (use 'raise')")
            pressure_projection_physical_failure_action = "raised"
            self.last_project_pressure_projection_physical_failure_action = "raised"
            compatibility_measured = False
            incompatible_component_count = 0
            rhs_mean_max_abs = 0.0
            rhs_integral_max_abs = 0.0
            if (
                pressure_projection_physical_failure_reason
                == "unreached_component_label_overflow"
            ):
                message = (
                    "FV-CG pressure projection physical gate failed: "
                    "unreached component labels exceeded diagnostic capacity; "
                    "sealed-component RHS compatibility is unverified"
                )
            elif (
                pressure_projection_physical_failure_reason
                == "unreached_component_labels_not_converged"
            ):
                message = (
                    "FV-CG pressure projection physical gate failed: "
                    "unreached component labels did not converge; "
                    "sealed-component RHS compatibility is unverified"
                )
            elif (
                pressure_projection_physical_failure_reason
                == "closed_neumann_component_rhs_incompatible"
            ):
                compatibility_measured = bool(
                    pressure_nullspace_compatibility_measured
                )
                incompatible_component_count = int(
                    pressure_nullspace_incompatible_component_count
                )
                rhs_mean_max_abs = float(
                    pressure_nullspace_component_rhs_mean_max_abs
                )
                rhs_integral_max_abs = float(
                    pressure_nullspace_component_rhs_integral_max_abs
                )
                message = (
                    "FV-CG pressure projection physical gate failed: "
                    "closed-Neumann component RHS is incompatible with a "
                    "sealed incompressible component "
                    f"(components={int(pressure_nullspace_incompatible_component_count)}, "
                    "max_abs_rhs_mean="
                    f"{float(pressure_nullspace_component_rhs_mean_max_abs):.6g})"
                )
            else:
                compatibility_measured = True
                incompatible_component_count = int(
                    pressure_projection_unreached_incompatible_component_count
                )
                rhs_mean_max_abs = float(
                    pressure_projection_unreached_rhs_mean_max_abs
                )
                rhs_integral_max_abs = float(
                    pressure_projection_unreached_rhs_integral_max_abs
                )
                message = (
                    "FV-CG pressure projection physical gate failed: "
                    "unreached component RHS is incompatible with a sealed "
                    "incompressible component "
                    f"(components={int(pressure_projection_unreached_incompatible_component_count)}, "
                    f"max_abs_rhs_mean={float(pressure_projection_unreached_rhs_mean_max_abs):.6g})"
                )
            diagnostics = self._pressure_projection_physical_failure_diagnostics(
                tolerance=cg_relative_tolerance,
                pressure_interface_matrix_active=pressure_interface_matrix_active,
                context=pressure_solve_context_payload,
                reason=pressure_projection_physical_failure_reason,
                action=pressure_projection_physical_failure_action,
                compatibility_measured=compatibility_measured,
                incompatible_component_count=incompatible_component_count,
                rhs_mean_max_abs=rhs_mean_max_abs,
                rhs_integral_max_abs=rhs_integral_max_abs,
            )
            diagnostics.update(
                {
                    "pressure_outlet_zmin": bool(pressure_outlet_zmin),
                    **self._closed_neumann_rhs_compatibility_diagnostics(),
                    "pressure_graph_component_count": int(
                        pressure_graph_component_count
                    ),
                    "pressure_nullspace_component_count": int(
                        pressure_nullspace_component_count
                    ),
                    "pressure_nullspace_component_labels_converged": bool(
                        pressure_nullspace_component_labels_converged
                    ),
                    "pressure_nullspace_component_overflow": bool(
                        pressure_nullspace_component_overflow
                    ),
                }
            )
            raise PressureSolveConvergenceError(
                message,
                diagnostics=diagnostics,
            )

        def warm_start_copy_in() -> None:
            # Seeds self.pressure from the slot's stash (if the slot has
            # ever been written) immediately before the CG solve so it
            # starts from a warm initial guess instead of whatever
            # clear/reset_pressure/accumulate-mode left behind. No-op (zero
            # kernel launches) when warm-starting is disabled (slot -1) or
            # the slot has never been populated.
            if warm_start_slot_index < 0:
                return
            if not self._pressure_warmstart_valid[warm_start_slot_index]:
                return
            self._copy_scalar_field_kernel(
                self.pressure,
                self._pressure_warmstart_slots[warm_start_slot_index],
            )

        def warm_start_copy_out() -> None:
            # Stashes self.pressure for next time. Called once, after all
            # solving/cleanup for this project() call has finished mutating
            # self.pressure (see call sites below): in accumulate mode that
            # is the solved INCREMENT, captured before
            # _add_pressure_step_carry_kernel folds the carry back in, so
            # the next warm start seeds another increment-mode solve with
            # an increment, not a stale total. No-op when warm-starting is
            # disabled (slot -1).
            if warm_start_slot_index < 0:
                return
            self._copy_scalar_field_kernel(
                self._pressure_warmstart_slots[warm_start_slot_index],
                self.pressure,
            )
            self._pressure_warmstart_valid[warm_start_slot_index] = True

        velocity_dirichlet_boundary_apply_calls = 0
        velocity_dirichlet_boundary_active_cells_total = 0
        velocity_dirichlet_boundary_active_cells_max = 0
        velocity_dirichlet_boundary_delta_sum_mps = 0.0
        velocity_dirichlet_boundary_delta_max_mps = 0.0
        velocity_dirichlet_boundary_momentum_delta_n_s = (0.0, 0.0, 0.0)

        def apply_velocity_boundary_conditions(
            *,
            preserve_projected_rows: bool = False,
        ) -> None:
            nonlocal velocity_dirichlet_boundary_apply_calls
            nonlocal velocity_dirichlet_boundary_active_cells_total
            nonlocal velocity_dirichlet_boundary_active_cells_max
            nonlocal velocity_dirichlet_boundary_delta_sum_mps
            nonlocal velocity_dirichlet_boundary_delta_max_mps
            nonlocal velocity_dirichlet_boundary_momentum_delta_n_s
            self._apply_velocity_dirichlet_boundary_rows_dispatch(
                read_report=bool(report_requested),
                preserve_projected_rows=bool(preserve_projected_rows),
            )
            if report_requested:
                raise NotImplementedError("per-call solver reports (read_report=True) are not supported; use compute_divergence() and the simulation monitors")
            if (
                pressure_outlet_zmin
                and pressure_outlet_backflow_policy_name == "clamp"
            ):
                self._apply_zmin_no_backflow_kernel()
            if pressure_outlet_zmin:
                self._apply_closed_boundary_no_normal_flow_kernel(1)
            if obstacle_normal_velocity_policy_name == "face_clamp":
                self._apply_obstacle_no_normal_flow_kernel(canonical_authority)
            else:
                raise NotImplementedError("obstacle_normal_velocity_policy must be 'face_clamp'")

        def reject_canonical_projection_topology_mutation() -> None:
            if canonical_authority == 0:
                return
            raise RuntimeError(
                "interface topology cleanup mutated obstacle topology inside project() "
                "under canonical velocity-boundary authority; run cleanup before "
                "row assembly, then rebuild and seal the canonical registry"
            )

        final_closed_boundary_cleanup = not pressure_outlet_zmin

        self._reset_zmin_projection_flux_report_kernel()
        # Applied interface flux belongs to the accumulated pressure state.
        # Homogenization controls only the new increment's affine RHS.  A
        # nonhomogeneous increment runs on rows rebuilt for a moved interface
        # (post-solid); cells those rows no longer touch keep no carried flux.
        if not accumulate_pressure_into_previous:
            self._clear_pressure_interface_projection_divergence_kernel()
        elif not homogenize_pressure_interface_rhs_for_increment_enabled:
            raise NotImplementedError("accumulate_pressure_into_previous=True (pressure-increment projection) is not supported")
        # Blend soft reconstructed rows once before forming the pressure RHS;
        # exact hard components clamp directly to their targets.  A follow-up
        # consistency projection may declare that the same soft map was already
        # applied in this fluid substep.  In that case only hard components are
        # re-clamped; pressure mobility still remains active for every soft row.
        apply_velocity_boundary_conditions(
            preserve_projected_rows=bool(
                velocity_dirichlet_soft_rows_already_applied
            )
        )
        if bool(pressure_outlet_zmin):
            if reachability_prepared:
                self._require_prepared_pressure_reachability_current()
            else:
                self.mark_pressure_outlet_disconnected_nonprojectable_cells(
                    pressure_outlet_zmin=True,
                )
            converted_overflow_singletons = (
                self.convert_row_cloud_orphan_components(
                    max_component_cells=1,
                    overflow_singletons_only=True,
                    protect_velocity_dirichlet_radius_cells=2,
                )
            )
            if int(converted_overflow_singletons) > 0:
                projection_topology_mutated = True
                projection_overflow_singleton_cleanup_cell_count += int(
                    converted_overflow_singletons
                )
                projection_overflow_singleton_cleanup_component_count += int(
                    self.last_row_cloud_orphan_component_count
                )
                reject_canonical_projection_topology_mutation()
                self.mark_pressure_outlet_disconnected_nonprojectable_cells(
                    pressure_outlet_zmin=True,
                )
            tiny_unreached_cleanup_threshold = max(
                0,
                int(tiny_unreached_cleanup_component_cells),
            )
            if tiny_unreached_cleanup_threshold > 0:
                for _tiny_unreached_cleanup_pass in range(8):
                    converted_tiny_unreached = (
                        self.convert_row_cloud_orphan_components(
                            max_component_cells=tiny_unreached_cleanup_threshold,
                            convert_unstamped_small_components=True,
                            protect_velocity_dirichlet_radius_cells=0,
                            protect_solid_band_mask=False,
                        )
                    )
                    if int(converted_tiny_unreached) <= 0:
                        break
                    projection_topology_mutated = True
                    projection_tiny_unreached_cleanup_cell_count += int(
                        converted_tiny_unreached
                    )
                    projection_tiny_unreached_cleanup_component_count += int(
                        self.last_row_cloud_orphan_component_count
                    )
                    reject_canonical_projection_topology_mutation()
                    self.mark_pressure_outlet_disconnected_nonprojectable_cells(
                        pressure_outlet_zmin=True,
                    )
            if (
                projection_topology_mutated
                and pressure_interface_matrix_active
            ):
                raise RuntimeError(
                    "interface topology cleanup mutated obstacle topology inside "
                    "project() after pressure matrix terms were already active; "
                    "move cleanup before interface row/matrix assembly and rebuild "
                    "velocity/pressure rows before projection"
                )
        # Row-cloud cleanup may have converted cells to obstacles after the
        # conservative routing decision.  Rebuild the actual projection mask
        # so the operator and final diagnostics describe the current topology.
        (
            velocity_dirichlet_hard_fixed_row_count,
            velocity_dirichlet_hard_fixed_component_face_count,
        ) = self._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask()
        if (
            bool(pressure_outlet_zmin)
            and pressure_solver_name == "fv_cg"
            and bool(pressure_interface_matrix_active)
        ):
            (
                pressure_graph_component_count,
                pressure_nullspace_component_count,
            ) = self._prepare_pressure_outlet_nullspace_component_graph()
            outlet_pressure_operator_graph_prepared = True
            pressure_nullspace_component_labels_converged = True
            pressure_nullspace_component_overflow = False
            pressure_nullspace_componentwise_projection_applied = bool(
                pressure_nullspace_component_count > 0
            )
            pressure_nullspace_compatibility_measured = True
            pressure_nullspace_policy = (
                "outlet_disconnected_fv_cg_operator_componentwise_zero_mean"
                if pressure_nullspace_componentwise_projection_applied
                else "pressure_outlet_dirichlet_operator_anchored"
            )
        closed_pressure_graph_may_be_partitioned = False
        if not bool(pressure_outlet_zmin) and pressure_solver_name == "fv_cg":
            raise NotImplementedError("closed domains without an outlet are not supported yet")
        if closed_pressure_graph_may_be_partitioned:
            raise NotImplementedError("closed domains without an outlet are not supported yet")
        # Component labels and overflow are already final at this point.  Do
        # not spend a full pressure solve on a topology whose compatibility
        # cannot be verified; the post-solve gate below remains responsible
        # for the RHS compatibility measurement itself.
        record_projection_structural_preflight()
        handle_projection_physical_failure()
        pre_projection_velocity_projector_report = (
            _run_pre_projection_velocity_transaction(
                pre_projection_velocity_projector=(
                    pre_projection_velocity_projector
                ),
                fluid=self,
                pressure_solve_context=pressure_solve_context_payload,
            )
        )
        if pressure_velocity_nullspace_projector is not None:
            raise NotImplementedError("a pressure_velocity_nullspace_projector is not supported")
        if pressure_outlet_zmin:
            self._record_zmin_velocity_flux_kernel(self.report_zmin_projection_pre_velocity_outlet_flux_m3s)
        compute_current_divergence()
        if pressure_outlet_zmin:
            # Cleanup acceptance is physical solver state, not report state.
            # Keep the target on-device so read_report only controls diagnostics.
            self._divergence_residual_stats_kernel(
                0,
                1 if int(self._pressure_unreached_count) > 0 else 0,
            )
            self._store_cleanup_target_l2_sq_from_reduction_kernel(
                PRESSURE_OUTLET_AUTO_CLEANUP_TARGET_REDUCTION
                * PRESSURE_OUTLET_AUTO_CLEANUP_TARGET_REDUCTION,
                PRESSURE_OUTLET_AUTO_CLEANUP_MIN_L2
                * PRESSURE_OUTLET_AUTO_CLEANUP_MIN_L2,
            )
        if report_requested:
            raise NotImplementedError("per-call solver reports (read_report=True) are not supported; use compute_divergence() and the simulation monitors")
        else:
            pre_projection_raw_stats = empty_stats
            pre_projection_stats = empty_stats
        if accumulate_pressure_into_previous:
            # Fractional-step increment mode: this projection acts on an
            # already-projected velocity, so its solution is a pressure
            # INCREMENT relative to the step's accumulated pressure, not the
            # step's full pressure. Save the incoming total and solve the
            # increment from a clean zero field; the total is restored as
            # carry + increment just before returning.
            raise NotImplementedError("accumulate_pressure_into_previous=True (pressure-increment projection) is not supported")
        elif reset_pressure:
            self._clear_pressure_kernel()
        warm_start_copy_in()
        self._solve_pressure_poisson_with_solver(
            closed_neumann_velocity_inlet_zmax_mode=velocity_inlet_zmax_mode,
            iterations=int(iterations),
            rhs_scale=float(rhs_scale),
            pressure_outlet_zmin=bool(pressure_outlet_zmin),
            pressure_solver=pressure_solver_name,
            multigrid_cycles=multigrid_cycles,
            cg_tolerance=cg_relative_tolerance,
            cg_preconditioner=cg_preconditioner_name,
            remove_nullspace_mean=(
                pressure_nullspace_zero_mean_projection_applied
                and not pressure_nullspace_componentwise_projection_applied
            ),
            pressure_nullspace_component_count=(
                pressure_nullspace_component_count
                if pressure_nullspace_componentwise_projection_applied
                else 0
            ),
            pressure_interface_matrix_active=pressure_interface_matrix_active,
            pressure_components_use_operator_graph=bool(
                pressure_interface_matrix_active
                and (
                    (
                        pressure_outlet_zmin
                        and outlet_pressure_operator_graph_prepared
                    )
                    or (
                        not pressure_outlet_zmin
                        and pressure_nullspace_componentwise_projection_applied
                    )
                )
            ),
            report_requested=report_requested,
            pressure_velocity_nullspace_projector=(
                pressure_velocity_nullspace_projector
            ),
        )
        record_cg_stats()
        record_projection_physical_stats()
        handle_pressure_solve_failure()
        handle_projection_physical_failure()
        if pressure_outlet_zmin:
            self._record_zmin_velocity_flux_kernel(self.report_zmin_pressure_step_pre_velocity_outlet_flux_m3s)
        current_pressure_marker_nullspace_report = (
            self._apply_fv_pressure_velocity_correction(
                float(step_dt_s / self.rho),
                1 if pressure_outlet_zmin else 0,
                self._velocity_dirichlet_boundary_authority_code(),
                pressure_velocity_nullspace_projector,
            )
        )
        if current_pressure_marker_nullspace_report:
            pressure_marker_nullspace_report = (
                current_pressure_marker_nullspace_report
            )
        if accumulate_pressure_into_previous:
            raise NotImplementedError("accumulate_pressure_into_previous=True (pressure-increment projection) is not supported")
        else:
            self._update_pressure_interface_projection_divergence_kernel(
                float(step_dt_s / self.rho),
            )
        if pressure_outlet_zmin:
            self._accumulate_zmin_pressure_correction_flux_kernel()
        apply_velocity_boundary_conditions(preserve_projected_rows=True)
        compute_current_divergence()
        if report_requested:
            raise NotImplementedError("per-call solver reports (read_report=True) are not supported; use compute_divergence() and the simulation monitors")
        else:
            projection_raw_stats = empty_stats
            projection_stats = empty_stats
        if pressure_outlet_zmin:
            self._divergence_residual_stats_kernel(
                0,
                1 if int(self._pressure_unreached_count) > 0 else 0,
            )
            self._tighten_cleanup_target_l2_sq_from_reduction_kernel(
                PRESSURE_OUTLET_AUTO_CLEANUP_NO_REGRESSION_MARGIN
                * PRESSURE_OUTLET_AUTO_CLEANUP_NO_REGRESSION_MARGIN,
                PRESSURE_OUTLET_AUTO_CLEANUP_MIN_L2
                * PRESSURE_OUTLET_AUTO_CLEANUP_MIN_L2,
            )
        post_boundary_stats = projection_stats
        post_boundary_raw_stats = projection_raw_stats
        if pressure_outlet_zmin:
            self._copy_pressure_to_accum_kernel()
            apply_velocity_boundary_conditions(preserve_projected_rows=True)
            compute_current_divergence()
            self._divergence_residual_stats_kernel(
                0,
                1 if int(self._pressure_unreached_count) > 0 else 0,
            )
            self._update_cleanup_required_from_reduction_kernel()
            cleanup_status = int(self.cleanup_required[None])
            if cleanup_status == 2:
                raise FloatingPointError("nonfinite pressure-outlet cleanup base metric")
            cleanup_needed = bool(cleanup_status)
            if report_requested:
                raise NotImplementedError("per-call solver reports (read_report=True) are not supported; use compute_divergence() and the simulation monitors")
            else:
                post_boundary_raw_stats = empty_stats
                post_boundary_stats = empty_stats
            cleanup_iterations = pressure_outlet_cleanup_iteration_budget(
                iterations=int(iterations),
                pressure_interface_matrix_active=pressure_interface_matrix_active,
            )
            for cleanup_pass_index in range(PRESSURE_OUTLET_AUTO_CLEANUP_MAX_PASSES):
                if not cleanup_needed:
                    break
                self._divergence_residual_stats_kernel(
                    0,
                    1 if int(self._pressure_unreached_count) > 0 else 0,
                )
                self._store_cleanup_acceptance_l2_sq_from_reduction_kernel(
                    PRESSURE_OUTLET_AUTO_CLEANUP_ACCEPTANCE_MARGIN
                    * PRESSURE_OUTLET_AUTO_CLEANUP_ACCEPTANCE_MARGIN,
                )
                self._snapshot_projection_cleanup_state_kernel()
                cleanup_marker_nullspace_report = pressure_marker_nullspace_report
                try:
                    self._clear_pressure_correction_kernel()
                    self._clear_pressure_interface_matrix_rhs_kernel()
                    self._solve_pressure_poisson_with_solver(
                        iterations=int(cleanup_iterations),
                        rhs_scale=float(rhs_scale),
                        pressure_outlet_zmin=bool(pressure_outlet_zmin),
                        pressure_solver=pressure_solver_name,
                        multigrid_cycles=multigrid_cycles,
                        cg_tolerance=cg_relative_tolerance,
                        cg_preconditioner=cg_preconditioner_name,
                        remove_nullspace_mean=(
                            pressure_nullspace_zero_mean_projection_applied
                            and not pressure_nullspace_componentwise_projection_applied
                        ),
                        pressure_nullspace_component_count=(
                            pressure_nullspace_component_count
                            if pressure_nullspace_componentwise_projection_applied
                            else 0
                        ),
                        pressure_interface_matrix_active=pressure_interface_matrix_active,
                        pressure_components_use_operator_graph=bool(
                            pressure_interface_matrix_active
                            and (
                                (
                                    pressure_outlet_zmin
                                    and outlet_pressure_operator_graph_prepared
                                )
                                or (
                                    not pressure_outlet_zmin
                                    and pressure_nullspace_componentwise_projection_applied
                                )
                            )
                        ),
                        report_requested=report_requested,
                        pressure_velocity_nullspace_projector=(
                            pressure_velocity_nullspace_projector
                        ),
                    )
                    record_cg_stats()
                    record_projection_physical_stats()
                    handle_pressure_solve_failure()
                    handle_projection_physical_failure()
                    if pressure_outlet_zmin:
                        self._record_zmin_velocity_flux_kernel(self.report_zmin_pressure_step_pre_velocity_outlet_flux_m3s)
                    current_pressure_marker_nullspace_report = (
                        self._apply_fv_pressure_velocity_correction(
                            float(step_dt_s / self.rho),
                            1 if pressure_outlet_zmin else 0,
                            self._velocity_dirichlet_boundary_authority_code(),
                            pressure_velocity_nullspace_projector,
                        )
                    )
                    if current_pressure_marker_nullspace_report:
                        pressure_marker_nullspace_report = (
                            current_pressure_marker_nullspace_report
                        )
                    self._accumulate_pressure_interface_projection_divergence_kernel(
                        float(step_dt_s / self.rho),
                    )
                    if pressure_outlet_zmin:
                        self._accumulate_zmin_pressure_correction_flux_kernel()
                    self._accumulate_pressure_correction_kernel()
                    apply_velocity_boundary_conditions(preserve_projected_rows=True)
                    compute_current_divergence()
                    self._divergence_residual_stats_kernel(
                        0,
                        1 if int(self._pressure_unreached_count) > 0 else 0,
                    )
                    self._update_cleanup_rejected_from_reduction_kernel()
                    cleanup_status = int(self.cleanup_rejected[None])
                    if cleanup_status == 2:
                        # This raise stays inside the candidate transaction so
                        # the accepted main projection is restored below.
                        raise FloatingPointError("nonfinite pressure-outlet cleanup candidate metric")
                    cleanup_rejected = bool(cleanup_status)
                except BaseException as failure:
                    cleanup_state_restored = False
                    try:
                        self._restore_projection_cleanup_state_kernel()
                        compute_current_divergence()
                        cleanup_state_restored = True
                    except BaseException as restore_failure:
                        # Keep the original solve/cancellation failure primary.
                        failure.projection_cleanup_restore_error = restore_failure
                    if isinstance(failure, PressureSolveConvergenceError):
                        failure.diagnostics.update({
                            "projection_substage": "outlet_cleanup",
                            "cleanup_pass_index": int(cleanup_pass_index) + 1,
                            "main_projection_correction_applied": True,
                            "correction_applied": False if cleanup_state_restored else None,
                            "correction_applied_scope": "current_cleanup_candidate",
                            "cleanup_candidate_state": "restored" if cleanup_state_restored else "unknown",
                            "cleanup_state_restored": cleanup_state_restored,
                        })
                        if not cleanup_state_restored:
                            restore_failure = failure.projection_cleanup_restore_error
                            failure.diagnostics["cleanup_restore_failure"] = {
                                "type": type(restore_failure).__name__,
                                "message": str(restore_failure),
                            }
                    raise
                if cleanup_rejected:
                    pressure_marker_nullspace_report = cleanup_marker_nullspace_report
                    self._restore_projection_cleanup_state_kernel()
                    compute_current_divergence()
                    if report_requested:
                        raise NotImplementedError("per-call solver reports (read_report=True) are not supported; use compute_divergence() and the simulation monitors")
                    break
                self._update_cleanup_required_from_reduction_kernel()
                cleanup_status = int(self.cleanup_required[None])
                if cleanup_status == 2:
                    raise FloatingPointError("nonfinite pressure-outlet cleanup target metric")
                cleanup_needed = bool(cleanup_status)
                if report_requested:
                    raise NotImplementedError("per-call solver reports (read_report=True) are not supported; use compute_divergence() and the simulation monitors")
                else:
                    post_boundary_raw_stats = empty_stats
                    post_boundary_stats = empty_stats
            if report_requested:
                raise NotImplementedError("per-call solver reports (read_report=True) are not supported; use compute_divergence() and the simulation monitors")
        if preserve_velocity_constraints:
            raise NotImplementedError("preserve_velocity_constraints=True is not supported")
        if final_closed_boundary_cleanup and int(divergence_cleanup_iterations) > 0:
            self._apply_closed_boundary_no_normal_flow_kernel(0)
        for _ in range(max(0, int(divergence_cleanup_iterations))):
            raise NotImplementedError("divergence_cleanup_iterations > 0 is not supported")
        if final_closed_boundary_cleanup:
            self._apply_closed_boundary_no_normal_flow_kernel(0)
        quasi_2d_velocity_filter_applied = self.enforce_quasi_2d_velocity()
        if pressure_outlet_zmin:
            self._record_zmin_velocity_flux_kernel(self.report_zmin_projection_post_boundary_velocity_outlet_flux_m3s)
        if not report_requested:
            warm_start_copy_out()
            if accumulate_pressure_into_previous:
                raise NotImplementedError("accumulate_pressure_into_previous=True (pressure-increment projection) is not supported")
            return {}
        raise NotImplementedError("per-call solver reports (read_report=True) are not supported; use compute_divergence() and the simulation monitors")


def enforce_planar_velocity(fluid):
    """The fluid's opt-in quasi-2-D velocity filter (zero x velocity, x-uniform fields) after a predictor / projection."""
    enforce = getattr(fluid, "enforce_quasi_2d_velocity", None)
    if callable(enforce):
        enforce()
