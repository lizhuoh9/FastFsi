"""Keep unchanged, wholly excluded numerical pockets across raw recomposition.

An obstacle written by cleanup is not a moving material cell. Reopening an
entire unchanged excluded component creates fluid with no prior state. Retain
that exclusion only with a closed-component certificate. Never initialize a
genuinely uncovered cell here; native reconstruction remains authoritative.
"""
import json
import inspect
from functools import wraps
import numpy as np

import taichi as ti

from .solver import CartesianFluidSolver
from .complete_pressure_components import CompletePressureComponentFluid


WITNESS_LIMIT = 32
OFFSETS = ((0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0),
           (0, 1, 0), (0, 0, -1), (0, 0, 1))
BOUNDARY_REVISION = 'F2-native-exterior-normal-contract-v1'
_PROJECT_SIGNATURE = inspect.signature(CartesianFluidSolver.project)
_PREDICT_SIGNATURE = inspect.signature(CartesianFluidSolver.predict)


class RetiredCleanupSupport:
    """A restore retires selection authority, never grants the native fallback."""
    def select(self, **kwargs):
        raise RuntimeError('numerical cleanup requires rebuilt committed topology support after restore')


def policy_identity():
    return dict(revision="S30-sealed-exclusion-support-after-band-v4",
        scope="whole_previously_excluded_raw_fluid_components",
        certificate="committed_cleanup_owned_classification_support_unchanged_halo_certified_exterior",
        boundary_contract=BOUNDARY_REVISION,
        reconstruction="native_for_all_other_cells", witness_limit=WITNESS_LIMIT)


class ConsistentVolumeRefresh:
    # Instance lookup is required by Taichi's kernel descriptor; super() does
    # not pass through its data_oriented __getattribute__ wrapper.
    _s28_native_store = CartesianFluidSolver._store_hibm_dynamic_solid_volume_candidate_kernel
    _s28_native_classify = CartesianFluidSolver._apply_hibm_internal_obstacles_kernel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_consistent_volume_refresh()

    def _initialize_consistent_volume_refresh(self):
        shape = (self.nx, self.ny, self.nz)
        # Transaction-local scratch; overwritten before each native mutation.
        self.s28_previous = ti.field(ti.i32, shape=shape)
        self.s28_keep = ti.field(ti.i32, shape=shape)
        self.s28_next = ti.field(ti.i32, shape=shape)
        self.s28_count = ti.field(ti.i32, shape=())
        self.s28_volume = ti.field(ti.f64, shape=())
        self.s28_exclusion = ti.field(ti.i32, shape=shape)
        self.s28_exclusion_raw = ti.field(ti.i32, shape=shape)
        self.s28_accepted_kind = ti.field(ti.i32, shape=shape)
        self.s28_accepted_support = ti.field(ti.i32, shape=shape)
        self._s28_exclusion_version = 0
        self._s28_saved_exclusions = None
        self._s28_cleanup_probe = None
        self._s28_boundary_context = None  # Unknown until live solve options arrive.
        self._s28_boundary_version = 0
        self.s28_boundary_changes = ti.field(ti.i32, shape=())
        # Codes: unknown=0, prescribed zero=1, nonzero/invalid=2, outlet=3,
        # explicit inlet=4. These are owned provenance, not velocity donors.
        self.s28_boundary_x = ti.field(ti.i32, shape=(2, self.ny, self.nz))
        self.s28_boundary_y = ti.field(ti.i32, shape=(2, self.nx, self.nz))
        self.s28_boundary_z = ti.field(ti.i32, shape=(2, self.nx, self.ny))
        self.s28_witness_cell = ti.Vector.field(3, ti.i32, shape=WITNESS_LIMIT)
        self.s28_witness_state = ti.Vector.field(10, ti.i32, shape=(WITNESS_LIMIT, 7))
        self.s28_witness_velocity = ti.Vector.field(3, ti.f64, shape=(WITNESS_LIMIT, 7))
        self.s28_witness_target = ti.Vector.field(3, ti.f64, shape=WITNESS_LIMIT)
        self.s28_witness_position = ti.Vector.field(3, ti.f64, shape=WITNESS_LIMIT)
        self._s28_transaction = None
        self.last_consistent_volume_refresh = None

        if ti.lang.impl.current_cfg().arch == ti.cuda:
            from .gpu_volume_host import install_volume_device_control
            install_volume_device_control(self)

    _s28_state_fields = ('s28_exclusion', 's28_exclusion_raw',
                         's28_accepted_kind', 's28_accepted_support',
                         's28_boundary_x', 's28_boundary_y', 's28_boundary_z')

    def configure_exclusion_domain_boundaries(self, *, pressure_outlet_zmin,
                                              velocity_inlet_zmax):
        """Bind the very options passed to the native fluid solve, before refresh.

        No geometry-derived wall claims or user closure flags are accepted.
        Native Cartesian defaults close all undeclared faces; the live face
        planes override their normal components. None is native per-face mode.
        """
        if (type(pressure_outlet_zmin) is not bool
                or (velocity_inlet_zmax is not None and type(velocity_inlet_zmax) is not bool)):
            raise ValueError('invalid numerical-exclusion boundary solve options')
        context = dict(pressure_outlet_zmin=pressure_outlet_zmin,
                       velocity_inlet_zmax=velocity_inlet_zmax)
        if context != self._s28_boundary_context:
            self._s28_boundary_context = context
            self._s28_boundary_version += 1
            self._s28_cleanup_probe = None
        self._s28_refresh_boundary_contracts()

    @wraps(CartesianFluidSolver.project)
    def project(self, *args, **kwargs):
        options = _PROJECT_SIGNATURE.bind(self, *args, **kwargs)
        options.apply_defaults()
        self.configure_exclusion_domain_boundaries(**{
            key: options.arguments[key] for key in ('pressure_outlet_zmin', 'velocity_inlet_zmax')})
        return super().project(*args, **kwargs)

    @wraps(CartesianFluidSolver.predict)
    def predict(self, *args, **kwargs):
        options = _PREDICT_SIGNATURE.bind(self, *args, **kwargs)
        options.apply_defaults()
        self.configure_exclusion_domain_boundaries(**{
            key: options.arguments[key] for key in ('pressure_outlet_zmin', 'velocity_inlet_zmax')})
        return super().predict(*args, **kwargs)

    @wraps(CartesianFluidSolver.refresh_external_velocity_boundary_face_uniform)
    def refresh_external_velocity_boundary_face_uniform(self, *args, **kwargs):
        try:
            return super().refresh_external_velocity_boundary_face_uniform(*args, **kwargs)
        finally:
            # A change followed by a change back must not resurrect a retired
            # exclusion, even if no recomposition happened between the writes.
            self._s28_refresh_boundary_contracts()

    @wraps(CartesianFluidSolver.refresh_zmax_inlet_boundary)
    def refresh_zmax_inlet_boundary(self, *args, **kwargs):
        try:
            return super().refresh_zmax_inlet_boundary(*args, **kwargs)
        finally:
            self._s28_refresh_boundary_contracts()

    @wraps(CartesianFluidSolver.refresh_zmax_inlet_boundary_canonical)
    def refresh_zmax_inlet_boundary_canonical(self, *args, **kwargs):
        try:
            return super().refresh_zmax_inlet_boundary_canonical(*args, **kwargs)
        finally:
            self._s28_refresh_boundary_contracts()

    @ti.kernel
    def _s28_boundary_contracts(self, known: ti.i32, outlet: ti.i32, inlet: ti.i32):
        self.s28_boundary_changes[None] = 0
        for axis in ti.static(range(3)):
            reference = ti.static((self.s28_boundary_x, self.s28_boundary_y,
                                   self.s28_boundary_z)[axis])
            for q in ti.grouped(reference):
                masks = ti.static((self.external_velocity_boundary_x_face_active_component_mask,
                    self.external_velocity_boundary_y_face_active_component_mask,
                    self.external_velocity_boundary_z_face_active_component_mask)[axis])
                values = ti.static((self.external_velocity_boundary_x_face_value_mps,
                    self.external_velocity_boundary_y_face_value_mps,
                    self.external_velocity_boundary_z_face_value_mps)[axis])
                # Same exterior normal rule as _physical_exterior_normal_contract,
                # evaluated even while a cleanup obstacle hides the fluid cell.
                code = 0
                if known != 0:
                    code = 1
                    mask = masks[q]
                    if mask < 0 or mask > 7:
                        code = 0
                    elif ((mask & (1 << axis)) != 0
                            and (ti.bit_cast(values[q][axis], ti.u32) & ti.u32(0x7fffffff)) != 0):
                        # Bitwise zero also fails closed for NaN/Inf under
                        # Taichi/LLVM fast-math floating comparison rules.
                        code = 2
                    if ti.static(axis == 2):
                        # Open/inlet contracts cannot close a pocket, even when
                        # the instantaneous/exact normal velocity happens to be zero.
                        if q[0] == 0 and outlet != 0:
                            code = 3
                        if q[0] == 1 and inlet == 0 and (mask & 4) != 0:
                            code = 0  # Explicit False contradicts exact zmax normal rows.
                        if q[0] == 1 and inlet == 1:
                            code = 4
                if code != reference[q]:
                    c = ti.Vector([q[1], q[2], q[0] * (self.nz - 1)])
                    if ti.static(axis == 0):
                        c = ti.Vector([q[0] * (self.nx - 1), q[1], q[2]])
                    elif ti.static(axis == 1):
                        c = ti.Vector([q[1], q[0] * (self.ny - 1), q[2]])
                    self.s28_exclusion[c] = 0
                    ti.atomic_add(self.s28_boundary_changes[None], 1)
                reference[q] = code

    def _s28_refresh_boundary_contracts(self):
        context = self._s28_boundary_context
        # Other model-specific/open-boundary families have no certificate here.
        known = context is not None and self.turbulence_model in ('laminar', 'none')
        self._s28_boundary_contracts(int(known),
            int(bool(context and context['pressure_outlet_zmin'])),
            -1 if context is None or context['velocity_inlet_zmax'] is None
            else int(context['velocity_inlet_zmax']))
        if int(self.s28_boundary_changes[None]):
            self._s28_boundary_version += 1
            self._s28_cleanup_probe = None

    def _s28_boundary_provenance(self):
        return dict(revision=BOUNDARY_REVISION, version=self._s28_boundary_version,
                    solve_options=None if self._s28_boundary_context is None
                    else dict(self._s28_boundary_context))

    def exclusion_state(self):
        self._s28_refresh_boundary_contracts()
        return dict(version=self._s28_exclusion_version,
                    boundary=self._s28_boundary_provenance(),
                    arrays={name: getattr(self, name).to_numpy().copy()
                            for name in self._s28_state_fields})

    def validate_exclusion_state(self, state):
        if not isinstance(state, dict) or type(state.get('version')) is not int or state['version'] < 0:
            raise ValueError('invalid numerical-exclusion version')
        boundary = state.get('boundary')
        if (not isinstance(boundary, dict) or set(boundary) != {'revision', 'version', 'solve_options'}
                or boundary['revision'] != BOUNDARY_REVISION
                or type(boundary['version']) is not int or boundary['version'] < 0):
            raise ValueError('missing or invalid numerical-exclusion boundary provenance')
        context = boundary['solve_options']
        if context is not None and (not isinstance(context, dict)
                or set(context) != {'pressure_outlet_zmin', 'velocity_inlet_zmax'}
                or type(context['pressure_outlet_zmin']) is not bool
                or (context['velocity_inlet_zmax'] is not None and type(context['velocity_inlet_zmax']) is not bool)
                or boundary['version'] == 0):
            raise ValueError('invalid numerical-exclusion boundary solve options')
        arrays = state.get('arrays', {})
        if set(arrays) != set(self._s28_state_fields):
            raise ValueError('missing numerical-exclusion provenance')
        for name, value in arrays.items():
            if value.shape != getattr(self, name).shape or value.dtype != np.dtype('i4'):
                raise ValueError('invalid numerical-exclusion field '+name)
        for axis in 'xyz':
            codes = arrays['s28_boundary_'+axis]
            if not np.isin(codes, (0,) if context is None else (0, 1, 2, 3, 4)).all():
                raise ValueError('invalid numerical-exclusion boundary certificate')
        if (np.any(arrays['s28_exclusion'] < 0)
                or np.any(arrays['s28_exclusion'] > state['version'])
                or not np.isin(arrays['s28_exclusion_raw'], (0, 2, 4, 6, 8, 10, 12, 14)).all()
                or not np.isin(arrays['s28_accepted_support'], (0, 1)).all()):
            raise ValueError('invalid numerical-exclusion certificate')
        return dict(version=state['version'], arrays={k: v.copy() for k, v in arrays.items()},
                    boundary=dict(boundary, solve_options=None if context is None else dict(context)))

    def restore_exclusion_state(self, state):
        state = self.validate_exclusion_state(state)
        for name, value in state['arrays'].items():
            getattr(self, name).from_numpy(value)
        self._s28_exclusion_version = state['version']
        self._s28_boundary_version = state['boundary']['version']
        self._s28_boundary_context = state['boundary']['solve_options']
        self._s28_transaction = None
        self._s28_cleanup_probe = None
        # Retention reads the owned, restored support certificate, never this
        # discarded selector. Cleanup itself still requires a new selector.
        self._hibm_policy_row_cloud_selector = RetiredCleanupSupport()

    def save_state(self) -> None:
        super().save_state()
        self._s28_saved_exclusions = self.exclusion_state()

    def restore_state(self) -> None:
        if self._s28_saved_exclusions is None:
            raise RuntimeError('missing saved numerical-exclusion provenance')
        self.validate_exclusion_state(self._s28_saved_exclusions)
        super().restore_state()
        self.restore_exclusion_state(self._s28_saved_exclusions)

    @ti.kernel
    def _s28_copy_support(self, support: ti.template()):
        for c in ti.grouped(self.obstacle):
            self.s28_accepted_support[c] = ti.cast(support[c] != 0, ti.i32)
            if support[c] != 0:
                self.s28_exclusion[c] = 0

    def accept_exclusion_support(self, selector):
        b = selector.boundary
        if (selector.epoch != b._topology_support_epoch
                or selector.generation != int(self.velocity_dirichlet_component_ledger_generation)
                or not self.velocity_dirichlet_component_ledger_sealed):
            raise RuntimeError('numerical exclusion requires current committed topology support')
        self._s28_copy_support(b.topology_support_mask())
        self._s28_cleanup_probe = None

    @ti.kernel
    def _s28_commit_cleanup(self, version: ti.i32):
        for c in ti.grouped(self.obstacle):
            if self.s28_exclusion[c] > 0:
                for offset in ti.static(OFFSETS):
                    n = c + ti.Vector(offset)
                    if self._s28_inside(n):
                        if self.s28_exclusion_raw[n] != (self._s28_state(n) & 14):
                            self.s28_exclusion[c] = 0
            # This transition is observed only around the approved native
            # selector's successful commit, never inferred during refresh.
            if (self.s28_previous[c] == 0 and self._s28_state(c) == 1
                    and self.s28_accepted_kind[c] == 0
                    and self.s28_accepted_support[c] == 0):
                self.s28_exclusion[c] = version
            elif self.s28_previous[c] != self._s28_state(c):
                self.s28_exclusion[c] = 0
        for c in ti.grouped(self.obstacle):
            self.s28_exclusion_raw[c] = self._s28_state(c) & 14

    def _run_hibm_row_cloud_orphan_component_selection(self, **kwargs):
        self._s28_refresh_boundary_contracts()
        commit = bool(kwargs['commit_obstacle'])
        if commit:
            probe = self._s28_cleanup_probe
            if probe is None or probe[0] != kwargs:
                raise RuntimeError('numerical exclusion cleanup has no matching successful probe')
            self._s28_cleanup_probe = None
            self._s28_snapshot()
        count = super()._run_hibm_row_cloud_orphan_component_selection(**kwargs)
        if not commit:
            self._s28_cleanup_probe = (dict(kwargs, commit_obstacle=True), count)
        else:
            if probe[1] != count:
                raise RuntimeError('numerical exclusion cleanup has no matching successful probe')
            self._s28_exclusion_version += 1
            self._s28_commit_cleanup(self._s28_exclusion_version)
        return count

    @ti.func
    def _s28_state(self, c):
        return ((self.obstacle[c] != 0) + 2 * (self.hibm_base_obstacle[c] != 0)
            + 4 * (self.hibm_dynamic_solid_volume_obstacle[c] != 0)
            + 8 * (self.hibm_dynamic_solid_volume_external_carve[c] != 0))

    @ti.func
    def _s28_inside(self, c):
        return 0 <= c.x < self.nx and 0 <= c.y < self.ny and 0 <= c.z < self.nz

    @ti.kernel
    def _s28_snapshot(self):
        for c in ti.grouped(self.obstacle):
            self.s28_previous[c] = self._s28_state(c)

    @ti.kernel
    def _s28_candidates(self, support: ti.template(), kinds: ti.template(),
                        has_support: ti.template(), has_kinds: ti.template()):
        self.s28_count[None] = 0
        for c in ti.grouped(self.obstacle):
            eligible = (self.s28_exclusion[c] > 0
                and self.s28_previous[c] == 1 and self._s28_state(c) == 0)
            for side in ti.static(range(2)):
                if c.x == side * (self.nx-1) and self.s28_boundary_x[side, c.y, c.z] != 1:
                    eligible = False
                if c.y == side * (self.ny-1) and self.s28_boundary_y[side, c.x, c.z] != 1:
                    eligible = False
                if c.z == side * (self.nz-1) and self.s28_boundary_z[side, c.x, c.y] != 1:
                    eligible = False
            if ti.static(has_kinds):
                eligible = eligible and kinds[c] == 0
            if ti.static(has_support):
                eligible = eligible and support[c] == 0
            eligible = (eligible and self.hibm_solid_band_protection_cell[c] == 0
                and self.volume_source_s[c] == 0.0 and self.hibm_air_cell[c] == 0)
            for offset in ti.static(OFFSETS):
                n = c + ti.Vector(offset)
                if self._s28_inside(n):
                    if ((self.s28_previous[n] & 14) != (self._s28_state(n) & 14)
                            or self.s28_exclusion_raw[n] != (self._s28_state(n) & 14)
                            or self.velocity_dirichlet_boundary_active[n] != 0
                            or self.velocity_dirichlet_boundary_active_component_mask[n] != 0
                            or self.velocity_dirichlet_pressure_hard_fixed_component_mask[n] != 0):
                        eligible = False
            self.s28_keep[c] = ti.cast(eligible, ti.i32)
            if not eligible:
                self.s28_exclusion[c] = 0
            if eligible:
                ti.atomic_add(self.s28_count[None], 1)

    @ti.kernel
    def _s28_prune_open_components(self):
        self.s28_count[None] = 0
        for c in ti.grouped(self.obstacle):
            keep = self.s28_keep[c]
            if keep != 0:
                for offset in ti.static(OFFSETS[1:]):
                    n = c + ti.Vector(offset)
                    if self._s28_inside(n):
                        if self.obstacle[n] == 0 and self.s28_keep[n] == 0:
                            keep = 0
            self.s28_next[c] = keep
            if keep != self.s28_keep[c]:
                ti.atomic_add(self.s28_count[None], 1)
        for c in ti.grouped(self.obstacle):
            self.s28_keep[c] = self.s28_next[c]

    @ti.kernel
    def _s28_retain_and_count(self):
        self.s28_count[None] = 0
        self.s28_volume[None] = 0.0
        self.reduction_count[None] = 0
        for c in ti.grouped(self.obstacle):
            if self.s28_keep[c] != 0:
                self.obstacle[c] = 1
                # This was already excluded before the transaction. No new
                # fluid state exists and no velocity/pressure is assigned.
                self.hibm_fresh_fluid_cell[c] = 0
                self.fresh_fluid_reconstruction_pending[c] = 0
                ti.atomic_add(self.report_hibm_fresh_fluid_cells[None], -1)
                ti.atomic_add(self.s28_count[None], 1)
                ti.atomic_add(self.s28_volume[None], self._cell_volume_f64_m3(c.x, c.y, c.z))
            else:
                self.s28_exclusion[c] = 0
            if (self.s28_previous[c] & 1) != (self.obstacle[c] != 0):
                ti.atomic_add(self.reduction_count[None], 1)

    def _s28_retain_unchanged_pockets(self):
        self._s28_refresh_boundary_contracts()
        self._s28_candidates(self.s28_accepted_support, self.s28_accepted_kind, True, True)
        candidates = int(self.s28_count[None])
        # Monotone erosion: each productive sweep removes >=1 candidate.
        # At the fixed point no retained component touches any surviving fluid.
        for _ in range(candidates + 1):
            self._s28_prune_open_components()
            if int(self.s28_count[None]) == 0:
                break
        else:
            raise RuntimeError("S28 numerical-pocket certificate did not converge")
        self._s28_retain_and_count()
        count, volume = int(self.s28_count[None]), float(self.s28_volume[None])
        report = dict(revision=policy_identity()["revision"], stage=self._s28_transaction,
            exclusion_version=self._s28_exclusion_version,
            boundary_provenance=self._s28_boundary_provenance(),
            candidate_cells=candidates, retained_cells=count, retained_excluded_volume_m3=volume,
            accounting_scope="retained_numerical_exclusions_only; native physical transitions excluded",
            native_fresh_cells_after_retention=int(self.report_hibm_fresh_fluid_cells[None]),
            created_fluid_mass_kg=0.0, created_fluid_momentum_kg_mps=[0.0, 0.0, 0.0])
        if count:
            report["witness"] = self._s28_witness(retained=True)
            print("S28_VOLUME_REFRESH " + json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
        self.last_consistent_volume_refresh = report

    def _store_hibm_dynamic_solid_volume_candidate_kernel(self):
        self._s28_cleanup_probe = None
        self._s28_transaction = "dynamic_volume"
        self._s28_snapshot()
        self._s28_native_store()
        self._s28_retain_unchanged_pockets()

    def _apply_hibm_internal_obstacles_kernel(self, node_kind_code, internal_node_code,
            external_node_code, carve_external_nodes_from_dynamic_volume):
        self._s28_cleanup_probe = None
        self._s28_transaction = "classification"
        self.s28_accepted_kind.copy_from(node_kind_code)
        self._s28_snapshot()
        self._s28_native_classify(node_kind_code, internal_node_code,
            external_node_code, carve_external_nodes_from_dynamic_volume)
        self._s28_retain_unchanged_pockets()

    @ti.kernel
    def _s28_capture(self, retained: ti.i32, kinds: ti.template(), has_kinds: ti.template()):
        self.s28_count[None] = 0
        for c in ti.grouped(self.obstacle):
            selected = self.fresh_fluid_reconstruction_pending[c] != 0 and self.obstacle[c] == 0
            if retained != 0:
                selected = self.s28_keep[c] != 0
            if selected:
                slot = ti.atomic_add(self.s28_count[None], 1)
                if slot < WITNESS_LIMIT:
                    self.s28_witness_cell[slot] = c
                    self.s28_witness_position[slot] = ti.Vector([
                        self.cell_center_x_m[c.x], self.cell_center_y_m[c.y], self.cell_center_z_m[c.z]])
                    self.s28_witness_target[slot] = ti.cast(self.velocity_dirichlet_boundary_value_mps[c], ti.f64)
                    for q in ti.static(range(7)):
                        n = c + ti.Vector(OFFSETS[q])
                        state = ti.Vector([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
                        velocity = ti.Vector([0., 0., 0.], dt=ti.f64)
                        if self._s28_inside(n):
                            kind = -1
                            if ti.static(has_kinds):
                                kind = kinds[n]
                            state = ti.Vector([self.s28_previous[n], self._s28_state(n),
                                self.fresh_fluid_reconstruction_pending[n],
                                self.fresh_fluid_reconstruction_pending_snapshot[n],
                                self.velocity_dirichlet_boundary_active[n],
                                self.velocity_dirichlet_pressure_hard_fixed_component_mask[n], kind,
                                self.velocity_dirichlet_boundary_active_component_mask[n],
                                self.hibm_solid_band_protection_cell[n], self.hibm_air_cell[n]])
                            velocity = ti.cast(self.velocity[n], ti.f64)
                        self.s28_witness_state[slot, q] = state
                        self.s28_witness_velocity[slot, q] = velocity

    def _s28_witness(self, *, retained=False):
        self._s28_capture(int(retained), self.s28_accepted_kind, True)
        count = int(self.s28_count[None])
        cells = self.s28_witness_cell.to_numpy()[:min(count, WITNESS_LIMIT)]
        states = self.s28_witness_state.to_numpy()[:len(cells)]
        velocities = self.s28_witness_velocity.to_numpy()[:len(cells)]
        positions = self.s28_witness_position.to_numpy()[:len(cells)]
        targets = self.s28_witness_target.to_numpy()[:len(cells)]
        def unpack(value):
            return None if value < 0 else dict(composite=bool(value & 1), base=bool(value & 2),
                dynamic=bool(value & 4), carve=bool(value & 8))
        rows = []
        for c, group, vv, pos, target in zip(cells, states, velocities, positions, targets):
            contacts = []
            for axis, size in enumerate((self.nx, self.ny, self.nz)):
                for side in (0, 1):
                    if c[axis] == side * (size - 1):
                        q = (side, *(int(c[a]) for a in range(3) if a != axis))
                        code = int(getattr(self, 's28_boundary_'+'xyz'[axis])[q])
                        contacts.append(dict(face='xyz'[axis]+('min', 'max')[side],
                                             contract_code=code, certified_zero_normal=code == 1))
            neighbors = []
            for offset, state, velocity in zip(OFFSETS, group, vv):
                old, new, pending, snapshot, active, hard, kind, components, protected, air = map(int, state)
                neighbors.append(dict(cell=(c + offset).tolist(), previous=unpack(old), current=unpack(new),
                    pending=pending, pending_snapshot=snapshot, scalar_dirichlet_active=active,
                    pressure_hard_mask=hard, node_kind=kind, active_component_mask=components,
                    solid_band_protected=protected, air_tag=air,
                    native_neighbor_donor=bool(new >= 0 and not (new & 1) and snapshot == 0),
                    velocity_mps=[float(v) if abs(v) < 1.e30 else None for v in velocity]))
            rows.append(dict(cell=c.tolist(), position_m=pos.tolist(), state=neighbors[0],
                boundary_contacts=contacts,
                neighbors=neighbors[1:], native_dirichlet_donor=bool(group[0, 4] > 0),
                dirichlet_target_mps=[float(v) if abs(v) < 1.e30 else None for v in target]))
        rows.sort(key=lambda row: row["cell"])
        return dict(revision="S28-fresh-fluid-witness-v1", stage=self._s28_transaction,
            total_cells=count, captured_cells=len(rows), truncated=count > len(rows), limit=WITNESS_LIMIT,
            selection="first_device_slots_sorted_by_cell", cells=rows,
            donor_rule="native six-face fluid + previous-sweep resolved, or scalar Dirichlet; native does not filter hard faces",
            retained=retained)

    def _reconstruct_fresh_fluid_cells(self) -> None:
        try:
            super()._reconstruct_fresh_fluid_cells()
        except RuntimeError as error:
            if self._s28_transaction is not None:
                try:
                    witness = self._s28_witness()
                    error.fresh_fluid_witness = witness
                    error.args = (str(error) + " [S28_FRESH_FLUID "
                        + json.dumps(witness, sort_keys=True, allow_nan=False) + "]",)
                except Exception as diagnostic_error:
                    error.fresh_fluid_witness_error = type(diagnostic_error).__name__
            raise
        finally:
            self._s28_transaction = None


@ti.data_oriented
class ConsistentVolumeFluid(ConsistentVolumeRefresh, CartesianFluidSolver):
    pass


@ti.data_oriented
class CompleteConsistentVolumeFluid(ConsistentVolumeRefresh, CompletePressureComponentFluid):
    pressure_component_capacity = CompletePressureComponentFluid.__dict__["pressure_component_capacity"]
    # REQ_369 (user 2026-10-11, 「无自由面的格子计为固体」, Root R1b): the shared profile's fluid.
    no_free_face_isolated_cells_solid = True
