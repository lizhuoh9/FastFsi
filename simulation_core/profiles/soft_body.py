"""Shared soft-body FSI construction and immutable method identity.

Low-level legacy constructors remain usable for numerical regression fixtures.
Production entry points bind this object before allocating any numerical state.
Geometry-specific operators specialize on primitive topology, never a case name.
W1 selects one native Cauchy trace for both geometries. Experimental load
functionals live outside this production selector and have separate evidence.
"""
from dataclasses import asdict, dataclass, field, fields, replace
import hashlib
import json
from types import MappingProxyType, SimpleNamespace


@dataclass(frozen=True)
class WallPressure:
    method: str = "W1_native_per_face_Cauchy_trace_v1"
    start_cells: float = 4.0
    spacing_cells: float = 1.0
    rung_count: int = 4
    min_fluid_weight: float = 0.95

    def parameters(self):
        return {k: v for k, v in asdict(self).items() if k != "method"}

    def apply(self, markers):
        if self.method != "W1_native_per_face_Cauchy_trace_v1":
            raise ValueError("alternate wall loads require an explicit off-profile experiment")
        # Preserve the original native sampler, including its support gates,
        # molecular viscous derivative and geometry-certified area transpose.
        # A29's parameters/gates remain intact for explicit low-level tests.
        markers.set_wall_pressure_extrapolation(False, **self.parameters())


@dataclass(frozen=True)
class ExecutionPolicy:
    """Delivered performance/accounting choices, identical for both cases.

    Direct low-level fixtures can exercise alternate dispatch implementations;
    production adapters expose none of these as environment or case options.
    """
    multigrid_prolongation_batch: bool = True
    device_visibility: bool = False
    cg_mean_batch: bool = False
    exact_factor_cache: bool = False
    # KI1 wave-1 migrations: one vector for every geometry, all default off.
    device_exact_arithmetic: bool = False
    device_trace_repair: bool = False
    device_transaction_buffers: bool = False
    device_component_graph: bool = False
    device_component_solve: bool = False
    device_execution_batches: bool = False
    native_dispatch: bool = False
    device_material_certificates: bool = False
    device_face_ledger: bool = False
    device_transaction_audits: bool = False
    device_observation_packing: bool = False
    compile_only_budget_accounting: bool = True
    empty_trace_specialization_warmup: bool = True


WAVE1_SWITCHES = (
    "device_exact_arithmetic", "device_trace_repair", "device_transaction_buffers",
    "device_component_graph", "device_component_solve", "device_execution_batches",
    "native_dispatch",
    "device_material_certificates", "device_transaction_audits",
    "device_observation_packing", "device_face_ledger",
)

MATERIAL_AUTHORITY_REVISION = "S36-material-authority-before-hard-proposals-v4"


_CANONICAL = (
    "static_junction_authority", "3d_author_arbitration", "cad_bonded_dry",
    "junction_redistribution", "wet_pressure_extension", "bond_layer_guard",
    "complete_pressure_components", "consistent_volume_refresh",
    "tangential_face_trace", "marker_constraint_projection",
    "pressure_support_reanchoring", "material_face_ownership",
)

# These compatibility fields are no longer case physics inputs. Their old
# defaults are accepted as unspecified at the public API, then resolved here.
# A nondefault conflicting method is rejected, not silently run under U1.
_TUREK_FIXED = MappingProxyType(dict(
    solid_nyquist_damping_ratio=0.0,
    solid_velocity_transfer_flip_blend=0.95,
    ib_wet_surface_tolerance_cells=0.0,
    ib_tip_corner_vertices=True,
    ib_quadratic_normal_reconstruction=False,
    fluid_laminar_canonical_patch_exchange=False,
    ib_incremental_neumann_target=False,
    flow_hibm_marker_mac_constraint_iterations=256,
    flow_hibm_marker_mac_constraint_absolute_tolerance_mps=1.e-6,
    flow_hibm_marker_mac_constraint_relative_audit=0.0,
    flow_hibm_marker_mac_constraint_ulp_audit=0.0,
    flow_hibm_terminal_no_slip_tolerance_mps=1.e-6,
    flow_hibm_pressure_nullspace_relative_audit_cap_mps=0.0,
    flow_hibm_pressure_nullspace_dense_max_constraints=512,
    flow_pressure_solver='fv_cg', flow_cg_tolerance=1.e-6,
    flow_cg_preconditioner='fv_multigrid', flow_multigrid_semi_coarsening=False,
    flow_reprojection_cg_tolerance=1.e-4,
    surface_transfer_method='cartesian_reference_adjoint_v1',
    ledger_relocation_merge_velocity_tolerance_mps=0.0,
    ledger_relocation_merge_velocity_tolerance_rel=0.0,
    flow_pressure_warm_start=False,
    ledger_corner_reconstruction_fallback=False,
    stress_one_sided_extended_probe=True,
    stress_one_sided_missing_side_fallback=True,
    fluid_advection_scheme="muscl_tvd",
    interpolate_velocity_dirichlet_with_interior=True,
    interface_velocity_mode="step_average",
    velocity_damping=1.0,
    fsi_iqn_reuse_previous_step_history=False,
    fsi_coupling_tolerance=1.e-3,
    fsi_coupling_initial_relaxation=0.3,
    fsi_coupling_absolute_tolerance_mps=0.0,
    fsi_coupling_max_marker_tolerance_mps=1.e-6,
    ib_anisotropic_envelope=True,
    classify_far_internal_nodes=True,
    one_sided_pressure_sampling_excludes_ib_envelope=True,
    stress_wall_extrapolation=False,
    marker_reseed_interval_steps=None,
))

_SQUID_FIXED = MappingProxyType(dict(
    canonical_static_clamped_rim=False,
    solid_model="volumetric_mpm", squid_grid="volumetric",
    fluid_advection_scheme="muscl_tvd",
    solid_mpm_flip_blend=0.95, solid_mpm_velocity_damping=1.0,
    mooney_membrane_force_scale=1.0,
    accumulate_reprojection_pressure=True,
    pressure_solve_failure_policy="raise",
    diagnostic_disable_pressure_neumann_matrix_rows=False,
    interface_reaction_relaxation=0.3,
    fsi_marker_coupling_tolerance_mps=1.e-6,
    pressure_solver='fv_cg', cg_tolerance=1.e-6, reprojection_cg_tolerance=1.e-4,
    cg_preconditioner='fv_multigrid',
    one_sided_probe_max_multiplier=12.0, two_sided_probe_max_multiplier=12.0,
    far_pressure_inside_probe_max_multiplier=12.0,
    pressure_outlet_source_ratio_tolerance=0.1, projection_divergence_tolerance=0.01,
    min_outlet_to_main_volume_flux_ratio=0.1,
    hibm_post_dirichlet_consistency_projections=1,
    divergence_cleanup_iterations=0, divergence_cleanup_relaxation=0.7,
    neo_fixed_node_lock_policy=None,
))


@dataclass(frozen=True)
class SoftBodyPhysics:
    revision: str = "M7_soft_body_fsi_M6_S40_S41_v1"
    wall_pressure: WallPressure = WallPressure()
    execution: ExecutionPolicy = ExecutionPolicy()
    pressure_anchor_rule: str = field(default="minimum_owner_key", init=False)

    @staticmethod
    def runtime_identity():
        """Canonical native arithmetic, separate from the physical identity.

        Launchers verify this registration in each fresh workload process.
        Keeping it separate preserves historical physical/configuration hashes;
        KI3 checkpoint sidecars additionally require this exact registration.
        """
        from hibm_native_reference import registration
        return dict(native_arithmetic=registration(),
                    cuda_pressure_reduction_order=(
                        "g1_8192_lanes_then_64_groups_stride64_x128_ascending_then_groups_0_to_63_ascending_v1"))

    def identity(self):
        from simulation_core.coupling.hibm_mpm.closest_surface_tie import identity as closest_surface_tie_identity
        from simulation_core.coupling.hibm_mpm.relocation_target_rule import identity as relocation_target_rule_identity
        data = dict(
            revision=self.revision,
            component_face_target_tie=closest_surface_tie_identity(),
            relocation_target_rule=relocation_target_rule_identity(),
            pressure_nullspace_triangular_solve=dict(
                revision="K12-host-superlu-triangular-v1",
                placement="host_cpu_scipy_superlu_unit_triangles_at_or_above_20000_entries",
                arithmetic="f64_same_factor_entries_column_oriented_non_bitwise",
                factor_check="identity_permutations_unchanged_entries_no_fill_else_raise",
                authority="REQ_370 K11 + Root 2026-10-11 K12"),
            no_free_face_cells=dict(rule="solid_for_projection",
                                    scope="isolated_unanchored_singleton_after_interface_merge",
                                    revision="REQ369-R1b-v2-all-sites",
                                    sites="cpu+cuda outlet-graph/unreached-flood/closed-physical; coverage guard",
                                    authority="REQ_369 user 2026-10-11 无自由面格子计为固体"),
            pressure_anchor=dict(rule=self.pressure_anchor_rule,
                eligible_rows="successfully_admitted_only",
                key="(i*ny+j)*nz+k", key_type="signed_i64",
                publication="after_assembly"),
            enabled_features=list(_CANONICAL),
            wall_pressure=asdict(self.wall_pressure),
            wall_load_definition=dict(
                pressure="native_visible_wet_side_trace",
                viscous="native_molecular_marker_gradient",
                quadrature="native_material_P1_pressure_and_lumped_viscous",
                junction="certified_wet_area_transpose_and_bounded_wet_extension",
                body_transfer="native_material_transpose",
                complete_discrete_reaction=False,
                alternatives="explicit_off_profile_experiments_only"),
            fluid="CompleteConsistentVolumeFluid",
            runtime="HibmMpmFsiRuntime",
            advection="muscl_tvd", velocity_transfer_flip_blend=0.95,
            normal_reconstruction="linear", pressure_target="native",
            # Preserve the delivered M7 identity when every new switch is off.
            # An enabled execution option is always included in the binding.
            execution={k: v for k, v in asdict(self.execution).items()
                       if k not in WAVE1_SWITCHES or v},
            material_face_ownership=dict(
                revision=MATERIAL_AUTHORITY_REVISION,
                admission="closed_material_MAC_face_or_pressure_link_intersection",
                ordering="material_authority_before_hard_proposals",
                relocation="independent_kinematic_routes_preserve_exact_hard_merge_gate",
                primitives=["registered_triangles", "extruded_registered_segments"],
                absent_primitives="native_exact_identity",
                intersection_tolerance_m=0.0,
                direct_segment_bridge="closest_point_source_provenance_v1"),
            material_face_reconstruction={'revision': 'S41_declared_support_volume_visibility_v1', 'owner': 'closest_registered_material_at_MAC_face', 'fluid_side': 'exact_incident_facet_outward_normal_cone', 'crossing': 'reject', 'multiple_surfaces': 'exact_geometry_and_velocity_or_reject', 'solid_side_interface': 'visible_pressure_link_material_boundary_flux', 'disjoint_obstacle_interface': 'visible_material_in_adjacent_obstacle_cell', 'rigid_interface': 'source_surface_and_declared_rigid_kinematics', 'target_merge': 'bounded_winner_target', 'tolerance_added': True, 'target_merge_cap_mps': 1e-05, 'target_merge_authority': 'REQ_355 user 2026-10-11 bounded tolerance; winner target unchanged'},
            marker_projection=self.marker_projection_identity(),
            interface_velocity="step_average",
            coupling=dict(relative_tolerance=1.e-3, absolute_tolerance_mps=0.0,
                          max_marker_tolerance_mps=1.e-6,
                          initial_relaxation=0.3, iqn_reuse_previous_step_history=False),
            relocation_merge_tolerance_mps=0.0, relocation_merge_tolerance_rel=0.0,
            case_physics_overrides=False,
            advance_options=self.advance_options(),
            adapter_contracts=dict(turek=dict(_TUREK_FIXED), squid=dict(_SQUID_FIXED)),
            deferred_methods=["T11_affine_corner", "T13_normal_momentum", "T14_cubic", "T15_LS4",
                              "A29_linear_clean_rungs_v1", "A29_short_affine_extension",
                              "short_quadratic_LS", "T19_pressure_only_SBP"],
        )
        data["sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return data

    @staticmethod
    def marker_projection_identity():
        return dict(revision="U3_shared_material_QP_factor_solve_v1", absolute_tolerance_mps=1.e-6,
                    max_iterations=256, rank_revealing_direct=True,
                    pressure_dense_apply="f64_forward_backward_factor_solve",
                    relative_audit=0.0, ulp_audit=0.0, relative_audit_cap_mps=0.0)

    def fluid_type(self):
        self.prepare_execution()
        from simulation_core.fluids.consistent_volume_refresh import CompleteConsistentVolumeFluid
        return CompleteConsistentVolumeFluid

    def prepare_execution(self):
        if self.execution.native_dispatch:
            from simulation_core.native_dispatch.wave2 import install
            install()

    @staticmethod
    def advance_options():
        return dict(run_fluid_predictor=True, pressure_solver='fv_cg',
            pressure_solve_failure_policy='raise', fluid_advection_scheme='muscl_tvd',
            cg_tolerance=1.e-6, reset_pressure=False, pressure_warm_start=False,
            cg_preconditioner='fv_multigrid', stress_viscosity_pa_s_override=None,
            one_sided_probe_max_multiplier=12.0, two_sided_probe_max_multiplier=12.0,
            far_pressure_inside_probe_max_multiplier=12.0,
            # G1: the converged coupled projection passes without the legacy
            # velocity-only relaxation, which has no pressure/body-impulse commit.
            # Apply one policy on uniform and graded grids.
            divergence_cleanup_iterations=0, divergence_cleanup_relaxation=0.7,
            post_dirichlet_consistency_projection_iterations=1,
            accumulate_reprojection_pressure=True, reprojection_cg_tolerance=1.e-4,
            interpolate_velocity_dirichlet_with_interior=True,
            classify_far_internal_nodes=True, convert_internal_nodes_to_obstacles=True,
            one_sided_pressure_sampling_excludes_ib_envelope=True,
            diagnostic_disable_pressure_neumann_matrix_rows=False,
            interface_velocity_mode='step_average')

    def advance(self, **kwargs):
        from simulation_core.coupling.hibm_mpm import advance_hibm_mpm_sharp_mpm_step
        # The case contributes geometry, BCs, loads, dt and iteration budgets.
        # All equation/staging choices are published by this single entry.
        kwargs.update(self.advance_options())
        return advance_hibm_mpm_sharp_mpm_step(**kwargs)

    def advance_coupling(self, coupling, **kwargs):
        return self.advance(markers=coupling.markers, ib_search=coupling.ib_search,
            ib_boundary=coupling.ib_boundary,
            marker_pressure_neumann_gradient_pa_per_m_field=coupling.marker_pressure_neumann_gradient_pa_per_m,
            **kwargs)

    @staticmethod
    def boundary_policy(*, triangle_count, bonded_contact_count=0, junction_edge_count=0,
                        static_obstacle_present=False):
        """Specialize absent material/primitive operators to the identity.

        Counts are geometry, not on/off switches. Segment material ownership
        is constructed by build_boundary; triangle quadrature and junctions
        have no work for that primitive type. Empty topology returns the native
        class without optional allocations.
        Fluid volume, component completeness, wall trace and Q/P are shared.
        """
        from simulation_core.coupling.hibm_mpm.boundary_policy import BoundaryPolicy
        if min(triangle_count, bonded_contact_count, junction_edge_count) < 0:
            raise ValueError("surface topology counts must be nonnegative")
        if triangle_count == 0:
            if bonded_contact_count or junction_edge_count:
                raise ValueError("triangle contact/junction data requires triangle primitives")
            return BoundaryPolicy()
        bonded = bonded_contact_count > 0
        junction = junction_edge_count > 0
        if junction and not bonded:
            raise ValueError("junction edges require a material bond certificate")
        return BoundaryPolicy(
            static_junction_authority=bool(static_obstacle_present),
            author_3d_arbitration=True, cad_bonded_dry=bonded,
            junction_redistribution=junction, wet_pressure_extension=junction,
            bond_layer_guard=bonded, complete_pressure_components=True,
            consistent_volume_refresh=True, tangential_face_trace=True,
            pressure_support_reanchoring=junction, material_face_ownership=True, material_face_reconstruction=True,
        )

    def configure_coupling(self, *, markers, boundary):
        self.prepare_execution()
        from simulation_core.coupling.hibm_mpm.pressure_anchor_election import configure
        configure(boundary, markers, self.pressure_anchor_rule)
        self.wall_pressure.apply(markers)
        markers.one_sided_missing_side_base_pressure_fallback = True
        markers.no_slip_component_coupling_tolerance_mps = 1.e-6
        markers.no_slip_component_relative_audit = 0.0
        markers.no_slip_component_ulp_audit = 0.0
        boundary.relocation_merge_velocity_tolerance_mps = 0.0
        boundary.relocation_merge_velocity_tolerance_rel = 0.0
        # REQ_355 (user 2026-10-11): bounded relocation-merge audit cap; kernel merge tolerances stay 0.
        boundary.relocation_merge_audit_cap_mps = 1.e-5
        # Shared linear/native choices. No reconstruction math is changed.
        boundary.quadratic_normal_reconstruction_enabled[None] = 0
        markers.quadratic_normal_trace_enabled[None] = 0
        # REQ_366 (user 2026-10-11, 「取最近表面点的速度」): closest-surface-point tie rule for canonical
        # component-face target conflicts, shared by Turek (segments) and the squid (S18 triangles).
        boundary.enable_closest_surface_tie_rule(True)
        # REQ_376 (user 2026-10-11, 「推广最近表面点规则」): an enabled relocation target conflict is decided by the
        # closest registered-surface author (REQ_366 settles an exact distance tie); policy boundaries only.
        if hasattr(boundary, 'enable_relocation_target_rule'):
            boundary.enable_relocation_target_rule(True)

    def build_boundary(self, *, markers, **kwargs):
        self.prepare_execution()
        from simulation_core.coupling.hibm_mpm import HibmMpmIbBoundaryConditions
        from simulation_core.coupling.hibm_mpm.policy_boundary import PolicyHibmMpmIbBoundaryConditions
        policy = self.boundary_policy(triangle_count=markers.projection_triangle_count)
        if not policy.enabled:
            if markers.projection_segment_count > 0:
                from simulation_core.coupling.hibm_mpm.segment_material_ownership import MaterialOwnedSegmentBoundary
                return MaterialOwnedSegmentBoundary(**kwargs)
            return HibmMpmIbBoundaryConditions(**kwargs)
        return PolicyHibmMpmIbBoundaryConditions(policy=policy, **kwargs)

    def marker_projector(self, *, coupling, fluid, primary_region_id, secondary_region_id):
        from simulation_core.coupling.hibm_mpm import (
            HibmMpmMarkerMacConstraintOperator, HibmMpmMarkerMacConstraintProjector,
        )
        markers = coupling.markers
        self.configure_coupling(markers=markers, boundary=coupling.ib_boundary)
        operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=fluid.grid.grid_nodes, marker_capacity=markers.marker_capacity)
        operator.marker_correction_relative_audit = 0.0
        operator.marker_correction_ulp_audit = 0.0
        operator.pressure_nullspace_relative_audit = 0.0
        operator.pressure_nullspace_relative_audit_cap_mps = 0.0
        return HibmMpmMarkerMacConstraintProjector(
            markers=markers, operator=operator, max_iterations=256,
            absolute_tolerance_mps=1.e-6, primary_region_id=primary_region_id,
            secondary_region_id=secondary_region_id, rank_revealing_direct=True)

    @staticmethod
    def coupling_config(max_iterations):
        from simulation_core.drivers.generic_fsi_solver import FsiCouplingConfig
        return FsiCouplingConfig(
            max_iterations=int(max_iterations), relative_tolerance=1.e-3,
            absolute_tolerance_mps=0.0, max_marker_tolerance_mps=1.e-6,
            initial_relaxation=0.3, iqn_reuse_previous_step_history=False)

    def bind_turek(self, config):
        self.require_production_environment()
        defaults = {f.name: f.default for f in fields(config)}
        for name, value in _TUREK_FIXED.items():
            supplied = getattr(config, name)
            if supplied != defaults[name] and supplied != value:
                raise ValueError(f"{name} is solver-owned by {self.revision}; expected {value!r}")
        resolved = dict(_TUREK_FIXED)
        # Plane strain is a case boundary condition. Its redundant span rows
        # are derived from that condition, not separately selectable physics.
        resolved.update(marker_constraint_spanwise_uniform=bool(config.enforce_plane_strain_x),
                        fluid_quasi_2d_x=bool(config.enforce_plane_strain_x))
        result = replace(config, **resolved)
        object.__setattr__(result, "_unified_physics", self.identity())
        return result

    def bind_squid(self, args):
        self.require_production_environment()
        result = SimpleNamespace(**vars(args))
        supplied = set(getattr(args, "_explicit_options", ()))
        values = {"canonical_" + name: True for name in _CANONICAL}
        values.update(_SQUID_FIXED)
        for name, value in values.items():
            option = "--" + name.replace("_", "-")
            if (option in supplied or '--no-' + option[2:] in supplied) and getattr(args, name, value) != value:
                raise ValueError(f"{option} is solver-owned by {self.revision}; expected {value!r}")
            setattr(result, name, value)
        result.physics_profile = self.identity()
        return result

    @staticmethod
    def require_production_environment():
        """Old low-level experimental paths cannot silently select case physics."""
        import os
        for name in ('HIBM_BAND_COUNT_ONLY', 'HIBM_BAND_INTERIOR_ONLY',
                     'HIBM_ENABLE_REACHABILITY_REUSE'):
            if os.environ.get(name) == '1':
                raise ValueError(f'{name} is unavailable in the shared production profile')


UNIFIED_PHYSICS = SoftBodyPhysics()
