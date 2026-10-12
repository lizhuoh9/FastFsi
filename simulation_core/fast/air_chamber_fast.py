"""REQ_401 FAST MODE air chamber: air cells above the membrane and the "shifted penalty" pressure boundary on the
unmodified production CartesianFluidSolver (Helper 1, 2026-10-12; contract 01 §3, Root acceptance conditions
2026-10-12, REQ_404 device-only per-step path).

REGION.  Same footprint, lid and per-column logic as the sharp path's AirChamberColumns
(cases/squid_soft_robot/air_chamber.py, footprint from volumetric_solid.air_chamber_footprint, chamber geometry from
volumetric_solid.source_config_squid_air_chamber); only the lower bound changes: per footprint column it is the TOP
air-face particle (body.tag bit TAG_AIR_FACE) instead of the mean membrane height.  A cell is air iff its column is in the
footprint, its centre z lies above that column's top air-face particle and below the lid, and it is not an obstacle.
`air_cell` (ti.field i32 on the grid) is recomputed every step on the device from body.x / body.tag.

PRESSURE BOUNDARY WITHOUT EDITING THE SOLVER.  The solver's pressure system (fv_cg, preconditioner fv_multigrid) is
    sum_nb T p_nb - (den + D) p = rhs_scale (div* + s - volume_source) - R,   rhs_scale = rho / dt,
with D = fluid.pressure_interface_matrix_diagonal and R = fluid.pressure_interface_matrix_rhs (f64, per cell); den is
the cell's own FV denominator.  The CG stops on ||r|| / ||b||, so no large value may enter R.  The solver therefore
solves the shifted pressure p~ = p - p_air(t):
  1. air cells: D = c_pen * den (penalty factor c_pen, default DEFAULT_PENALTY_FACTOR), R = 0  ->  p~ ~ 0 there;
     the deviation is ~ (local jump of p~) / (c_pen * den / T) (see the tests for the measured values);
  2. z-min outlet cells (k = 0, fluid, not air): R = T_out * p~_out with p~_out = -p_air(t), where T_out is the solver's own
     outlet ghost coefficient, simulation_core/fluids/solver.py, _fv_pressure_neighbor_sum_denominator (def at line
     21863), lines 22010-22027:
         if pressure_outlet_zmin == 1 and k == 0:
             outlet_face_weight = self._fv_pressure_velocity_face_weight(..., i, j, k, 4, canonical_authority)
             denominator += outlet_face_weight * 2.0 / (cell_width_z_m[k] * cell_width_z_m[k])
     i.e. T_out = w_face * 2 / dz_0^2 (ghost pressure at the face, half a cell from the centre); the solver itself uses
     ghost value 0 there, and moving T_out p~_out to the right side gives R = T_out p~_out;
  3. after fluid.project(...): the solver corrected the outlet face with ghost 0, solver.py,
     _subtract_pressure_gradient_kernel (def at line 27365), lines 27398-27399 and 27420:
         elif pressure_outlet_zmin == 1 and k == 0:
             grad.z = 2.0 * center / self.cell_width_z_m[k]
         ...
         self.velocity[i, j, k] -= ti.cast(correction, ti.f32)    # correction = w_face * dt_over_rho * grad
     so the missing term is  u_z(i, j, 0) += w_face * (dt / rho) * 2 p~_out / dz_0 = (dt / rho) * R * dz_0  (after_project).
     The solver books R as an interface flux: _update_pressure_interface_projection_divergence_kernel (line 27463)
     sets s = (dt / rho) (D p - R), and its divergence residual (solver.py 20689-20730, div + s - volume_source) is zero
     in the k = 0 cells, so the pressure-outlet auto-cleanup passes see nothing to clean.  after_project moves that
     flux onto the face (u_z(i, j, 0) as above) and removes it from s (s += (dt / rho) R), keeping div + s = 0 with the
     face itself divergence-free.  The correction uses this module's own copy of R (a cleanup pass may clear the
     solver's R; its homogeneous increment is correct with ghost 0).
  4. physical pressure: p = p~ + p_air(t) (physical_pressure_into / physical_pressure_field); report() gives physical p.
Preflight (verified, solver.py): _validate_pressure_interface_operator_storage_kernel (line 18723) only rejects a
non-finite or negative diagonal and a non-finite R; _validate_pressure_interface_local_diagonal_provenance_kernel
(line 18996) rejects only a diagonal SMALLER than the incident interface-row transmissibilities (local_excess <
-tolerance), so a positive diagonal without interface rows passes.  With an explicit cg_preconditioner="fv_multigrid"
and an active interface diagonal the MG preconditioner stays on (solver.py 28842-28858, explicit_interface_multigrid),
and every solve re-restricts the fine diagonal to the coarse levels (_apply_fv_multigrid_preconditioner, def at line
28138: _mg_restrict_residual_kernel call at 28230, coarse diagonal = volume average at 26997-27020) and re-prepares the
level operators on its first V-cycle (prepare_level_operator at 28210 / 28261, mg_level_operators_prepared at
28886-28900): nothing else must be refreshed per step (no topology rebuild, no reachability re-mark: air cells stay
fluid cells).  Note: with an active interface diagonal the CG restarts every 256 instead of 512 iterations (line 28933);
it checks the residual every 16 iterations (line 28932), so reported iteration counts are multiples of 16.

DRIVER REQUIREMENTS: project(..., pressure_outlet_zmin=True, pressure_solver="fv_cg", cg_preconditioner="fv_multigrid",
pressure_outlet_backflow_policy="allow", accumulate_pressure_into_previous=False); see PROJECT_REQUIREMENTS.  With the
default backflow policy "clamp" the solver zeroes positive (inflow) outlet face velocities computed with ghost 0, which
is the normal case in the shifted variable (p~_0 < 0 whenever p_air > 0), and the outlet correction would be wrong.
Call order per step: update(t_mid, p_air) -> project -> after_project(dt, p_air) with the same p_air.

PER STEP (REQ_404): update() is ONE kernel launch (column clear, top air-face particle per column by atomic max,
air mask + penalty diagonal + outlet RHS + step counters); after_project() is ONE kernel launch; no host reads or writes.
Hard stops (column without an air-face particle, no air cell at all) are counted on the device and raised by report().

Reuse check (REQ_400): the region rule reuses the sharp path's code (cases/squid_soft_robot/air_chamber.py
AirChamberColumns: footprint / face / centre fields, lid and its _locate column search; volumetric_solid.py
air_chamber_footprint), the operator coefficients are the production solver's own ti.funcs
(_fv_pressure_face_products_and_denominator, _fv_pressure_velocity_face_weight), and the waveform follows
cases/squid_soft_robot/schedules.py pressure_schedule_pa (clamped piecewise-linear; tested equal).  Looked at for the
Dirichlet / penalty treatment (read-only local copies of the upstream sources, reference_libs/): petsc-v3.26.0/src/mat/interface/matrix.c
MatZeroRowsColumns (Dirichlet elimination: known values times the eliminated columns move into b) - the outlet R term
above is exactly this elimination for the solver's own ghost value; ibamr-v0.19.0/include/ibamr/
BrinkmanPenalizationStrategy.h + src/IB/BrinkmanPenalizationStrategy.cpp (penalisation of the MOMENTUM equation) and
ibtk/src/solvers/impls/CCPoisson*.cpp (Dirichlet through SAMRAI Robin boundary coefficients on patch boundaries).
Neither row elimination nor patch-boundary coefficients exist for interior air cells of the fixed production operator,
which only exposes a diagonal and an affine term, so the diagonal penalty on the shifted pressure is new code (algorithm
only; no source copied, no licence obligation).

No `from __future__ import annotations` (Taichi kernel module).
"""
import math

import numpy as np
import taichi as ti

AIR_CHAMBER_FAST_REVISION = "REQ401-fast-air-chamber-shifted-penalty-v1"
HARD_STOP_CODES = {"air_chamber_column_without_membrane", "air_chamber_no_air_cells"}
TAG_AIR_FACE = 1                 # body.tag bit 1: main-membrane air-face (top-layer) particle
DEFAULT_PENALTY_FACTOR = 1.0e7   # |p~_air| <~ jump / (2 c) <= 5e-8 p_air; CG / MG counts unchanged (duct test)
PROJECT_REQUIREMENTS = dict(pressure_outlet_zmin=True, pressure_solver="fv_cg", cg_preconditioner="fv_multigrid",
                            pressure_outlet_backflow_policy="allow", accumulate_pressure_into_previous=False)
# stats slots (one f64 field, read once by report())
S_HOLE_EVENTS, S_HOLE_T, S_HOLE_I, S_HOLE_J, S_NO_AIR_EVENTS, S_NO_AIR_T = 0, 1, 2, 3, 4, 5
S_STEP_AIR_CELLS, S_STEP_HOLES = 6, 7
S_AIR_CELLS, S_AIR_VOLUME, S_AIR_PMAX, S_OUTLET_FLUX, S_WATER_PMIN, S_WATER_PMAX = 8, 9, 10, 11, 12, 13
S_DIV_WATER, S_DIV_OUTLET, S_OUTLET_CELLS, S_STEPS = 14, 15, 16, 17
STATS_SLOTS = 20


class FastModeHardStop(RuntimeError):
    """A fast-mode hard stop with its machine-readable code (one of HARD_STOP_CODES) and witness."""

    def __init__(self, code, witness):
        if code not in HARD_STOP_CODES:
            raise ValueError("unknown fast-mode hard-stop code %r" % (code,))
        super().__init__("%s: %s" % (code, witness))
        self.code = code
        self.witness = witness


class AirPressureWaveform:
    """p_air(t): clamped piecewise-linear schedule (the sharp path's pressure_schedule_pa rule for any node count)."""

    def __init__(self, times_s, values_pa):
        times = np.asarray(times_s, dtype=np.float64).reshape(-1)
        values = np.asarray(values_pa, dtype=np.float64).reshape(-1)
        if times.size < 2 or times.shape != values.shape:
            raise ValueError("the pressure schedule needs >= 2 times and one value per time")
        if not (np.all(np.isfinite(times)) and np.all(np.isfinite(values))):
            raise ValueError("the pressure schedule must be finite")
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("pressure schedule times must be strictly increasing")
        self.times_s = times
        self.values_pa = values

    @classmethod
    def from_config(cls, config, load_index=None):
        """The air-face Pressure load of a source config (contract: loads[0].params 'Pressure Schedule Times s' /
        'Pressure Schedule Values Pa', Direction -z = the air above the membrane pushes down; no direction switching)."""
        loads = list(config.get("loads", []))
        candidates = range(len(loads)) if load_index is None else [int(load_index)]
        for index in candidates:
            load = loads[index]
            params = load.get("params", {})
            if load.get("type") != "Pressure" or "Pressure Schedule Times s" not in params:
                continue
            if str(params.get("Direction", "-z")) != "-z":
                raise ValueError("the air-face load must act along -z (air above the membrane)")
            if float(params.get("Direction Switch Period s", 0.0)) != 0.0:
                raise ValueError("direction switching is not supported by the fast-mode air chamber")
            return cls(params["Pressure Schedule Times s"], params["Pressure Schedule Values Pa"])
        raise ValueError("no scheduled Pressure load in the config")

    def __call__(self, t_s):
        return float(np.interp(float(t_s), self.times_s, self.values_pa))


@ti.data_oriented
class AirChamberFast:
    """Air cells above the membrane's top air-face particles and the shifted-penalty pressure boundary (contract §3)."""

    def __init__(self, fluid, axes, config_chamber, body, *, penalty_factor=DEFAULT_PENALTY_FACTOR,
                 tag_bit=TAG_AIR_FACE):
        from cases.squid_setup.air_chamber import AirChamberColumns
        from cases.squid_setup.volumetric_solid import air_chamber_footprint
        grid = fluid.grid
        shape = tuple(int(v) for v in grid.grid_nodes)
        if axes is not None and tuple(int(v) for v in axes.n) != shape:
            raise ValueError("axes %s do not match the fluid grid %s" % (tuple(axes.n), shape))
        penalty = float(penalty_factor)
        if not math.isfinite(penalty) or penalty <= 0.0:
            raise ValueError("penalty_factor must be finite and positive")
        footprint = air_chamber_footprint(config_chamber, cell_centers_x_m=np.asarray(grid.cell_centers_x_m),
                                          cell_centers_y_m=np.asarray(grid.cell_centers_y_m))
        # the sharp path's holder: validated footprint, face / centre coordinates, lid and the _locate column search
        self.columns = AirChamberColumns(footprint, cell_faces_x_m=np.asarray(grid.cell_faces_x_m),
                                         cell_faces_y_m=np.asarray(grid.cell_faces_y_m),
                                         cell_centers_z_m=np.asarray(grid.cell_centers_z_m),
                                         lid_z_m=float(config_chamber.lid_z_m), membrane_region_id=-1)
        self.fluid = fluid
        self.axes = axes
        self.body = body
        self.shape = shape
        self.penalty_factor = penalty
        self.tag_bit = int(tag_bit)
        self.lid_z_m = float(config_chamber.lid_z_m)
        self.rho = float(fluid.rho)
        self.authority = int(fluid._velocity_dirichlet_boundary_authority_code())
        self.footprint_columns = int(footprint.sum())
        self.air_cell = ti.field(ti.i32, shape=shape)
        self.outlet_rhs = ti.field(ti.f64, shape=shape[:2])      # this module's copy of R at k = 0
        self.stats = ti.field(ti.f64, shape=STATS_SLOTS)
        self.stats.fill(0.0)
        self._physical_pressure = None
        self.last_p_air_pa = None
        self.last_t_s = None
        self.steps = 0

    # ------------------------------------------------------------------------------------------------ kernels
    @ti.kernel
    def _update_kernel(self, t_s: ti.f64, p_air: ti.f64, penalty: ti.f64, lid_z: ti.f64, tag_bit: ti.i32,
                       authority: ti.template()):
        for i, j in self.columns.column_sum:
            self.columns.column_sum[i, j] = -1.0e30
            self.columns.column_count_field[i, j] = 0
        for p in range(self.body.n):
            if (self.body.tag[p] & tag_bit) != 0:
                position = self.body.x[p].cast(ti.f64)
                i = self.columns._locate(self.columns.face_x, self.shape[0], position.x)
                j = self.columns._locate(self.columns.face_y, self.shape[1], position.y)
                if i >= 0 and j >= 0:
                    ti.atomic_max(self.columns.column_sum[i, j], position.z)
                    ti.atomic_add(self.columns.column_count_field[i, j], 1)
        self.stats[S_STEP_AIR_CELLS] = 0.0
        self.stats[S_STEP_HOLES] = 0.0
        self.stats[S_STEPS] += 1.0
        for i, j, k in self.air_cell:
            air = 0
            in_column = self.columns.footprint[i, j] != 0 and self.columns.column_count_field[i, j] > 0
            if in_column and self.fluid.obstacle[i, j, k] == 0:
                z = self.columns.centre_z[k]
                if z > self.columns.column_sum[i, j] and z < lid_z:      # above the top air-face particle
                    air = 1
            self.air_cell[i, j, k] = air
            diagonal = ti.cast(0.0, ti.f64)
            rhs = ti.cast(0.0, ti.f64)
            if air == 1:
                products, denominator = self.fluid._fv_pressure_face_products_and_denominator(
                    self.fluid.obstacle, self.fluid.velocity_dirichlet_boundary_active,
                    self.fluid.velocity_dirichlet_boundary_projection_weight,
                    self.fluid.velocity_dirichlet_pressure_hard_fixed_component_mask,
                    self.fluid.velocity_dirichlet_boundary_active_component_mask,
                    self.fluid.velocity_dirichlet_boundary_pressure_mobility,
                    self.fluid.velocity_dirichlet_boundary_hard_fixed_component_mask,
                    self.fluid.cell_width_x_m, self.fluid.cell_width_y_m, self.fluid.cell_width_z_m,
                    self.fluid.center_distance_x_m, self.fluid.center_distance_y_m, self.fluid.center_distance_z_m,
                    i, j, k, self.shape[0], self.shape[1], self.shape[2], 1, authority)
                diagonal = penalty * denominator
                ti.atomic_add(self.stats[S_STEP_AIR_CELLS], 1.0)
            elif k == 0 and self.fluid.obstacle[i, j, k] == 0:
                weight = self.fluid._fv_pressure_velocity_face_weight(
                    self.fluid.velocity_dirichlet_boundary_active,
                    self.fluid.velocity_dirichlet_boundary_projection_weight,
                    self.fluid.velocity_dirichlet_pressure_hard_fixed_component_mask,
                    self.fluid.velocity_dirichlet_boundary_active_component_mask,
                    self.fluid.velocity_dirichlet_boundary_pressure_mobility,
                    self.fluid.velocity_dirichlet_boundary_hard_fixed_component_mask, i, j, k, 4, authority)
                dz = ti.cast(self.fluid.cell_width_z_m[k], ti.f64)
                rhs = weight * 2.0 / (dz * dz) * (-p_air)          # R = T_out p~_out, p~_out = -p_air
            self.fluid.pressure_interface_matrix_diagonal[i, j, k] = diagonal
            self.fluid.pressure_interface_matrix_rhs[i, j, k] = rhs
            if k == 0:
                self.outlet_rhs[i, j] = rhs
        for i, j in self.columns.footprint:
            if self.columns.footprint[i, j] != 0 and self.columns.column_count_field[i, j] == 0:
                ti.atomic_add(self.stats[S_STEP_HOLES], 1.0)
                first = ti.atomic_add(self.stats[S_HOLE_EVENTS], 1.0)
                if first == 0.0:
                    self.stats[S_HOLE_T] = t_s
                    self.stats[S_HOLE_I] = ti.cast(i, ti.f64)
                    self.stats[S_HOLE_J] = ti.cast(j, ti.f64)
        if self.stats[S_STEP_AIR_CELLS] == 0.0:
            if self.stats[S_NO_AIR_EVENTS] == 0.0:
                self.stats[S_NO_AIR_T] = t_s
            self.stats[S_NO_AIR_EVENTS] += 1.0

    @ti.kernel
    def _after_project_kernel(self, dt_over_rho_f32: ti.f32, dt_over_rho: ti.f64):
        for i, j in self.outlet_rhs:
            rhs = self.outlet_rhs[i, j]
            dz = ti.cast(self.fluid.cell_width_z_m[0], ti.f64)
            velocity = self.fluid.velocity[i, j, 0]
            velocity.z += ti.cast(ti.cast(dt_over_rho_f32, ti.f64) * rhs * dz, ti.f32)
            self.fluid.velocity[i, j, 0] = velocity
            self.fluid.pressure_interface_projection_divergence_s[i, j, 0] += dt_over_rho * rhs

    @ti.kernel
    def _physical_pressure_kernel(self, out: ti.template(), p_air: ti.f64):
        for i, j, k in out:
            value = ti.cast(0.0, ti.f64)
            if self.fluid.obstacle[i, j, k] == 0:
                value = ti.cast(self.fluid.pressure[i, j, k], ti.f64) + p_air
            out[i, j, k] = value

    @ti.kernel
    def _report_kernel(self, with_divergence: ti.i32):
        self.stats[S_AIR_CELLS] = 0.0
        self.stats[S_AIR_VOLUME] = 0.0
        self.stats[S_AIR_PMAX] = 0.0
        self.stats[S_OUTLET_FLUX] = 0.0
        self.stats[S_WATER_PMIN] = 1.0e300
        self.stats[S_WATER_PMAX] = -1.0e300
        self.stats[S_DIV_WATER] = 0.0
        self.stats[S_DIV_OUTLET] = 0.0
        self.stats[S_OUTLET_CELLS] = 0.0
        for i, j, k in self.air_cell:
            if self.fluid.obstacle[i, j, k] == 0:
                volume = (ti.cast(self.fluid.cell_width_x_m[i], ti.f64) * ti.cast(self.fluid.cell_width_y_m[j], ti.f64)
                          * ti.cast(self.fluid.cell_width_z_m[k], ti.f64))
                p_tilde = ti.cast(self.fluid.pressure[i, j, k], ti.f64)
                if self.air_cell[i, j, k] != 0:
                    ti.atomic_add(self.stats[S_AIR_CELLS], 1.0)
                    ti.atomic_add(self.stats[S_AIR_VOLUME], volume)
                    ti.atomic_max(self.stats[S_AIR_PMAX], ti.abs(p_tilde))
                else:
                    ti.atomic_min(self.stats[S_WATER_PMIN], p_tilde)
                    ti.atomic_max(self.stats[S_WATER_PMAX], p_tilde)
                    if with_divergence != 0:
                        residual = ti.abs(ti.cast(self.fluid.divergence[i, j, k], ti.f64)
                                          + self.fluid.pressure_interface_projection_divergence_s[i, j, k]
                                          - ti.cast(self.fluid.volume_source_s[i, j, k], ti.f64))
                        ti.atomic_max(self.stats[S_DIV_WATER], residual)
                        if k == 0:
                            ti.atomic_max(self.stats[S_DIV_OUTLET], residual)
                    if k == 0:
                        area = ti.cast(self.fluid.cell_width_x_m[i], ti.f64) * ti.cast(self.fluid.cell_width_y_m[j], ti.f64)
                        ti.atomic_add(self.stats[S_OUTLET_FLUX], ti.cast(self.fluid.velocity[i, j, 0].z, ti.f64) * area)
                        ti.atomic_add(self.stats[S_OUTLET_CELLS], 1.0)

    # ------------------------------------------------------------------------------------------------ host API
    def update(self, t_s, p_air_pa):
        """Recompute air_cell from body.x / body.tag and write the penalty diagonal and outlet RHS (one kernel; no host
        reads).  p_air_pa is the air pressure for this step's solve (contract: p_air(t_mid))."""
        p_air = float(p_air_pa)
        if not math.isfinite(p_air):
            raise ValueError("p_air must be finite")
        self._update_kernel(float(t_s), p_air, self.penalty_factor, self.lid_z_m, self.tag_bit, self.authority)
        self.last_p_air_pa = p_air
        self.last_t_s = float(t_s)
        self.steps += 1

    def after_project(self, dt_s, p_air_pa):
        """Outlet-face correction for the nonzero ghost value p~_out = -p_air (one kernel; no host reads)."""
        if self.last_p_air_pa is None or float(p_air_pa) != self.last_p_air_pa:
            raise ValueError("after_project must use the p_air of the preceding update() (%r vs %r)"
                             % (p_air_pa, self.last_p_air_pa))
        dt_over_rho = float(dt_s) / self.rho
        if not (math.isfinite(dt_over_rho) and dt_over_rho > 0.0):
            raise ValueError("dt_s must be finite and positive")
        self._after_project_kernel(dt_over_rho, dt_over_rho)

    def physical_pressure_into(self, out_field):
        """out = p~ + p_air (physical pressure, Pa) in fluid cells, 0 in obstacle cells (device only)."""
        if self.last_p_air_pa is None:
            raise RuntimeError("physical pressure requested before the first update()")
        self._physical_pressure_kernel(out_field, self.last_p_air_pa)
        return out_field

    def physical_pressure_field(self):
        """A module-owned f64 field holding the current physical pressure (refreshed on every call)."""
        if self._physical_pressure is None:
            self._physical_pressure = ti.field(ti.f64, shape=self.shape)
        return self.physical_pressure_into(self._physical_pressure)

    def report(self, *, divergence=False):
        """Air cells, air volume, max |p~| in air, outlet flux, physical water pressure range (and, with
        divergence=True, max |div + s| in water and at the outlet after fluid.compute_divergence); one device read.
        Raises the hard stops counted on the device since the start."""
        if divergence:
            self.fluid.compute_divergence(pressure_outlet_zmin=True)
        self._report_kernel(1 if divergence else 0)
        s = self.stats.to_numpy()
        if s[S_HOLE_EVENTS] > 0.0:
            raise FastModeHardStop("air_chamber_column_without_membrane", dict(
                events=int(s[S_HOLE_EVENTS]), first_t_s=float(s[S_HOLE_T]),
                first_column_ij=[int(s[S_HOLE_I]), int(s[S_HOLE_J])], footprint_columns=self.footprint_columns))
        if s[S_NO_AIR_EVENTS] > 0.0:
            raise FastModeHardStop("air_chamber_no_air_cells", dict(
                events=int(s[S_NO_AIR_EVENTS]), first_t_s=float(s[S_NO_AIR_T])))
        p_air = 0.0 if self.last_p_air_pa is None else self.last_p_air_pa
        out = dict(revision=AIR_CHAMBER_FAST_REVISION, t_s=self.last_t_s, p_air_pa=p_air,
                   penalty_factor=self.penalty_factor, steps=int(s[S_STEPS]),
                   air_cell_count=int(s[S_AIR_CELLS]), air_volume_m3=float(s[S_AIR_VOLUME]),
                   p_tilde_air_max_abs_pa=float(s[S_AIR_PMAX]),
                   p_tilde_air_max_abs_over_p_air=(float(s[S_AIR_PMAX]) / abs(p_air)) if p_air else 0.0,
                   outlet_flux_m3s=float(s[S_OUTLET_FLUX]), outlet_cells=int(s[S_OUTLET_CELLS]),
                   water_pressure_min_pa=float(s[S_WATER_PMIN]) + p_air,
                   water_pressure_max_pa=float(s[S_WATER_PMAX]) + p_air)
        if divergence:
            out.update(divergence_residual_water_max_1ps=float(s[S_DIV_WATER]),
                       divergence_residual_outlet_max_1ps=float(s[S_DIV_OUTLET]))
        return out
