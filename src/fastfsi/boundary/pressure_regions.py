"""Per-cell Dirichlet pressures of an occupancy air region (inlet-connected air and sealed air pockets, each with its own
pressure) on the unmodified incompressible projection.  The per-cell generalisation of ColumnPressureChamber's shifted
penalty (boundary/pressure_chamber.py, whose outlet treatment it repeats); design note section 3.1.

REFERENCE FIELD.  The projection solves p~ = p - p_ref with the piecewise-constant reference
    p_ref = p_k        on the cells of air pocket k (its gas pressure, a table set by the caller),
    p_ref = p_in(t)    on every other cell (inlet-connected air, water, material, outlet cells, cells outside the box).
  1. DIRICHLET ROWS (box code 1: inlet-connected air; code 2 + k: pocket k): D = c_pen * den, so p~ ~ 0 and p ~ p_ref.
  2. Every fluid row next to a jump of p_ref: R = sum_f T_f (p_ref,nb - p_ref,c) with T_f the solver's own face products
     (_fv_pressure_face_products_and_denominator).  This is the operator applied to p_ref moved to the right side; it
     is of the size T dp, never of the penalty's, so the relative stopping rule of the solve is unaffected.
  3. z-min OUTLET rows (not Dirichlet): R += T_out p~_out with p~_out = -p_in(t), as in ColumnPressureChamber.
  4. after_project(): the face terms the solver did not apply,
       u_f -= (dt / rho) w_f (p_ref,c - p_ref,c-) / (centre distance)   on every face where p_ref jumps (each face once,
       from the cell on its + side), and the outlet face term of ColumnPressureChamber; then s += (dt / rho) R on every row
       with R != 0, so the solver's divergence bookkeeping div + s closes with the corrected faces (the face terms
       change the row's divergence by exactly -(dt / rho) R).
  5. PHYSICAL PRESSURE p = p~ + p_ref (physical_pressure_into); report() gives physical values.
Per step: set_regions(codes, pocket gauge pressures) when the classification changed (host -> device), update(t,
p_in) -> project -> after_project(dt, p_in) with the same p_in.  update() and after_project() are one kernel each and
read nothing back.  No pocket and every code 0 or 1 reproduces ColumnPressureChamber's rows for the same air cells (the
reference field is p_in everywhere: every R from rule 2 is zero).

DRIVER REQUIREMENTS: project(..., **PROJECT_REQUIREMENTS) of pressure_chamber (z-min pressure outlet, conjugate gradient
with the multigrid preconditioner or the hypre path, outlet backflow allowed, no pressure accumulation).

No `from __future__ import annotations` (Taichi kernel module).
"""
import math

import numpy as np
import taichi as ti

from fastfsi.boundary.pressure_chamber import DEFAULT_PENALTY_FACTOR, PROJECT_REQUIREMENTS  # noqa: F401 (re-export)
from fastfsi.io.checkpoint import Checkpointable

DEFAULT_MAX_POCKETS = 4096
# statistics slots (read by report() only)
S_INLET_CELLS, S_POCKET_CELLS, S_DIRICHLET_PMAX, S_WATER_DIV, S_JUMP_ROWS, S_OUTLET_FLUX = 0, 1, 2, 3, 4, 5
STATS_SLOTS = 6


@ti.data_oriented
class PressureRegions(Checkpointable):
    """Inlet air and air pockets of a box as per-cell Dirichlet pressures (see the module docstring)."""

    CHECKPOINT_FIELDS = ("box_code", "pocket_gauge")
    CHECKPOINT_HOST = ("pocket_count", "last_p_in_pa", "last_t_s", "steps")

    def __init__(self, fluid, box_cells, *, penalty_factor=DEFAULT_PENALTY_FACTOR, max_pockets=DEFAULT_MAX_POCKETS):
        shape = tuple(int(v) for v in fluid.grid.grid_nodes)
        ranges = tuple((int(lo), int(hi)) for lo, hi in box_cells)
        if len(ranges) != 3 or any(not 0 <= lo < hi <= n for (lo, hi), n in zip(ranges, shape)):
            raise ValueError("box_cells must be three half-open cell ranges inside the grid, got %r" % (box_cells,))
        penalty = float(penalty_factor)
        if not math.isfinite(penalty) or penalty <= 0.0:
            raise ValueError("penalty_factor must be finite and positive")
        self.fluid = fluid
        self.shape = shape
        self.box_cells = ranges
        self.box_shape = tuple(hi - lo for lo, hi in ranges)
        self.offset = tuple(lo for lo, _ in ranges)
        self.penalty_factor = penalty
        self.rho = float(fluid.rho)
        self.authority = int(fluid._velocity_dirichlet_boundary_authority_code())
        self.max_pockets = int(max_pockets)
        self.box_code = ti.field(ti.i32, shape=self.box_shape)
        self.pocket_gauge = ti.field(ti.f64, shape=self.max_pockets)
        self.p_ref = ti.field(ti.f64, shape=shape)
        self.r_jump = ti.field(ti.f64, shape=shape)
        self.outlet_rhs = ti.field(ti.f64, shape=shape[:2])
        self.stats = ti.field(ti.f64, shape=STATS_SLOTS)
        self.pocket_count = 0
        self.last_p_in_pa = None
        self.last_t_s = None
        self.steps = 0
        self._physical_pressure = None

    # ------------------------------------------------------------------------------------------------ helpers
    @ti.func
    def _code(self, i, j, k):
        code = 0
        a, b, c = i - self.offset[0], j - self.offset[1], k - self.offset[2]
        if 0 <= a < self.box_shape[0] and 0 <= b < self.box_shape[1] and 0 <= c < self.box_shape[2]:
            code = self.box_code[a, b, c]
        return code

    @ti.func
    def _products(self, i, j, k):
        return self.fluid._fv_pressure_face_products_and_denominator(
            self.fluid.obstacle, self.fluid.velocity_dirichlet_boundary_active,
            self.fluid.velocity_dirichlet_boundary_projection_weight,
            self.fluid.velocity_dirichlet_pressure_hard_fixed_component_mask,
            self.fluid.velocity_dirichlet_boundary_active_component_mask,
            self.fluid.velocity_dirichlet_boundary_pressure_mobility,
            self.fluid.velocity_dirichlet_boundary_hard_fixed_component_mask,
            self.fluid.cell_width_x_m, self.fluid.cell_width_y_m, self.fluid.cell_width_z_m,
            self.fluid.center_distance_x_m, self.fluid.center_distance_y_m, self.fluid.center_distance_z_m,
            i, j, k, self.shape[0], self.shape[1], self.shape[2], 1, self.authority)

    @ti.func
    def _face_weight(self, i, j, k, component: ti.template()):
        return self.fluid._fv_pressure_velocity_face_weight(
            self.fluid.velocity_dirichlet_boundary_active, self.fluid.velocity_dirichlet_boundary_projection_weight,
            self.fluid.velocity_dirichlet_pressure_hard_fixed_component_mask,
            self.fluid.velocity_dirichlet_boundary_active_component_mask,
            self.fluid.velocity_dirichlet_boundary_pressure_mobility,
            self.fluid.velocity_dirichlet_boundary_hard_fixed_component_mask, i, j, k, component, self.authority)

    @ti.func
    def _face_divergence(self, i, j, k):
        """Divergence of the current face velocities of cell (i, j, k): closed walls except the z-min outlet face.
        Computed here because after_project() changes faces after the solver's own divergence pass."""
        hi_x = ti.cast(0.0, ti.f64)
        lo_x = ti.cast(0.0, ti.f64)
        hi_y = ti.cast(0.0, ti.f64)
        lo_y = ti.cast(0.0, ti.f64)
        hi_z = ti.cast(0.0, ti.f64)
        if i < self.shape[0] - 1:
            hi_x = ti.cast(self.fluid.velocity[i + 1, j, k].x, ti.f64)
        if i > 0:
            lo_x = ti.cast(self.fluid.velocity[i, j, k].x, ti.f64)
        if j < self.shape[1] - 1:
            hi_y = ti.cast(self.fluid.velocity[i, j + 1, k].y, ti.f64)
        if j > 0:
            lo_y = ti.cast(self.fluid.velocity[i, j, k].y, ti.f64)
        if k < self.shape[2] - 1:
            hi_z = ti.cast(self.fluid.velocity[i, j, k + 1].z, ti.f64)
        lo_z = ti.cast(self.fluid.velocity[i, j, k].z, ti.f64)
        return ((hi_x - lo_x) / ti.cast(self.fluid.cell_width_x_m[i], ti.f64)
                + (hi_y - lo_y) / ti.cast(self.fluid.cell_width_y_m[j], ti.f64)
                + (hi_z - lo_z) / ti.cast(self.fluid.cell_width_z_m[k], ti.f64))

    # ------------------------------------------------------------------------------------------------ kernels
    @ti.kernel
    def _update_kernel(self, p_in: ti.f64, penalty: ti.f64):
        for i, j, k in self.p_ref:
            code = self._code(i, j, k)
            value = p_in
            if code >= 2:
                value = self.pocket_gauge[code - 2]
            self.p_ref[i, j, k] = value
        for s in ti.static((S_INLET_CELLS, S_POCKET_CELLS, S_JUMP_ROWS)):
            self.stats[s] = 0.0
        for i, j, k in self.p_ref:
            diagonal = ti.cast(0.0, ti.f64)
            jump = ti.cast(0.0, ti.f64)
            rhs = ti.cast(0.0, ti.f64)
            if self.fluid.obstacle[i, j, k] == 0:
                products, denominator = self._products(i, j, k)
                here = self.p_ref[i, j, k]
                if i > 0:
                    jump += products[0] * (self.p_ref[i - 1, j, k] - here)
                if i < self.shape[0] - 1:
                    jump += products[1] * (self.p_ref[i + 1, j, k] - here)
                if j > 0:
                    jump += products[2] * (self.p_ref[i, j - 1, k] - here)
                if j < self.shape[1] - 1:
                    jump += products[3] * (self.p_ref[i, j + 1, k] - here)
                if k > 0:
                    jump += products[4] * (self.p_ref[i, j, k - 1] - here)
                if k < self.shape[2] - 1:
                    jump += products[5] * (self.p_ref[i, j, k + 1] - here)
                code = self._code(i, j, k)
                if code >= 1:
                    diagonal = penalty * denominator
                    if code == 1:
                        ti.atomic_add(self.stats[S_INLET_CELLS], 1.0)
                    else:
                        ti.atomic_add(self.stats[S_POCKET_CELLS], 1.0)
                elif k == 0:
                    weight = self._face_weight(i, j, k, 4)
                    dz = ti.cast(self.fluid.cell_width_z_m[k], ti.f64)
                    rhs = weight * 2.0 / (dz * dz) * (-here)        # R = T_out p~_out, p~_out = 0 - p_ref
                if jump != 0.0:
                    ti.atomic_add(self.stats[S_JUMP_ROWS], 1.0)
            self.r_jump[i, j, k] = jump
            self.fluid.pressure_interface_matrix_diagonal[i, j, k] = diagonal
            self.fluid.pressure_interface_matrix_rhs[i, j, k] = rhs + jump
            if k == 0:
                self.outlet_rhs[i, j] = rhs

    @ti.kernel
    def _after_project_kernel(self, dt_over_rho_f32: ti.f32, dt_over_rho: ti.f64):
        for i, j in self.outlet_rhs:                                          # ColumnPressureChamber's outlet face term
            rhs = self.outlet_rhs[i, j]
            dz = ti.cast(self.fluid.cell_width_z_m[0], ti.f64)
            velocity = self.fluid.velocity[i, j, 0]
            velocity.z += ti.cast(ti.cast(dt_over_rho_f32, ti.f64) * rhs * dz, ti.f32)
            self.fluid.velocity[i, j, 0] = velocity
            self.fluid.pressure_interface_projection_divergence_s[i, j, 0] += dt_over_rho * rhs
        for i, j, k in self.p_ref:
            if self.fluid.obstacle[i, j, k] == 0:
                here = self.p_ref[i, j, k]
                scale = ti.cast(dt_over_rho_f32, ti.f64)
                velocity = self.fluid.velocity[i, j, k]
                if i > 0 and self.fluid.obstacle[i - 1, j, k] == 0 and self.p_ref[i - 1, j, k] != here:
                    velocity.x -= ti.cast(scale * self._face_weight(i, j, k, 1) * (here - self.p_ref[i - 1, j, k])
                                          / ti.cast(self.fluid.center_distance_x_m[i], ti.f64), ti.f32)
                if j > 0 and self.fluid.obstacle[i, j - 1, k] == 0 and self.p_ref[i, j - 1, k] != here:
                    velocity.y -= ti.cast(scale * self._face_weight(i, j, k, 2) * (here - self.p_ref[i, j - 1, k])
                                          / ti.cast(self.fluid.center_distance_y_m[j], ti.f64), ti.f32)
                if k > 0 and self.fluid.obstacle[i, j, k - 1] == 0 and self.p_ref[i, j, k - 1] != here:
                    velocity.z -= ti.cast(scale * self._face_weight(i, j, k, 4) * (here - self.p_ref[i, j, k - 1])
                                          / ti.cast(self.fluid.center_distance_z_m[k], ti.f64), ti.f32)
                self.fluid.velocity[i, j, k] = velocity
                if self.r_jump[i, j, k] != 0.0:
                    self.fluid.pressure_interface_projection_divergence_s[i, j, k] += dt_over_rho * self.r_jump[i, j, k]

    @ti.kernel
    def _physical_pressure_kernel(self, out: ti.template()):
        for i, j, k in out:
            value = ti.cast(0.0, ti.f64)
            if self.fluid.obstacle[i, j, k] == 0:
                value = ti.cast(self.fluid.pressure[i, j, k], ti.f64) + self.p_ref[i, j, k]
            out[i, j, k] = value

    @ti.kernel
    def _report_kernel(self):
        self.stats[S_DIRICHLET_PMAX] = 0.0
        self.stats[S_WATER_DIV] = 0.0
        self.stats[S_OUTLET_FLUX] = 0.0
        for i, j, k in self.p_ref:
            if self.fluid.obstacle[i, j, k] == 0:
                p_tilde = ti.cast(self.fluid.pressure[i, j, k], ti.f64)
                if self._code(i, j, k) >= 1:
                    ti.atomic_max(self.stats[S_DIRICHLET_PMAX], ti.abs(p_tilde))
                else:
                    residual = ti.abs(self._face_divergence(i, j, k)
                                      + self.fluid.pressure_interface_projection_divergence_s[i, j, k]
                                      - ti.cast(self.fluid.volume_source_s[i, j, k], ti.f64))
                    ti.atomic_max(self.stats[S_WATER_DIV], residual)
                if k == 0 and self._code(i, j, k) == 0:
                    area = ti.cast(self.fluid.cell_width_x_m[i], ti.f64) * ti.cast(self.fluid.cell_width_y_m[j], ti.f64)
                    ti.atomic_add(self.stats[S_OUTLET_FLUX], ti.cast(self.fluid.velocity[i, j, 0].z, ti.f64) * area)

    # ------------------------------------------------------------------------------------------------ host API
    def set_regions(self, codes, pocket_gauge_pa):
        """Upload the box's Dirichlet codes (0 none, 1 inlet air, 2 + k pocket k) and the pockets' gauge pressures (Pa)."""
        codes = np.asarray(codes)
        gauge = np.asarray(pocket_gauge_pa, dtype=np.float64).ravel()
        if codes.shape != self.box_shape or not np.issubdtype(codes.dtype, np.integer):
            raise ValueError("codes must be an integer array of the box shape %s" % (self.box_shape,))
        if codes.min(initial=0) < 0 or (codes.size and codes.max() >= 2 + len(gauge)):
            raise ValueError("a code refers to a pocket without a pressure")
        if len(gauge) > self.max_pockets or not np.isfinite(gauge).all():
            raise ValueError("pocket pressures must be finite and at most max_pockets")
        self.box_code.from_numpy(np.ascontiguousarray(codes, dtype=np.int32))
        table = np.zeros(self.max_pockets, dtype=np.float64)
        table[:len(gauge)] = gauge
        self.pocket_gauge.from_numpy(table)
        self.pocket_count = int(len(gauge))

    def set_pocket_pressures(self, pocket_gauge_pa):
        """Upload new pocket gauge pressures for unchanged codes (the identical-topology path)."""
        gauge = np.asarray(pocket_gauge_pa, dtype=np.float64).ravel()
        if len(gauge) != self.pocket_count or not np.isfinite(gauge).all():
            raise ValueError("pocket pressures must be finite, one per pocket of the current codes")
        table = np.zeros(self.max_pockets, dtype=np.float64)
        table[:len(gauge)] = gauge
        self.pocket_gauge.from_numpy(table)

    def update(self, t_s, p_in_pa):
        """Reference field, penalty diagonal and right-hand side for this step's solve (one kernel, no host reads);
        p_in_pa is the inlet gauge pressure at the solve (the step midpoint)."""
        p_in = float(p_in_pa)
        if not math.isfinite(p_in):
            raise ValueError("p_in must be finite")
        self._update_kernel(p_in, self.penalty_factor)
        self.last_p_in_pa = p_in
        self.last_t_s = float(t_s)
        self.steps += 1

    def after_project(self, dt_s, p_in_pa):
        """The face terms of the reference jumps and the outlet ghost (one kernel, no host reads)."""
        if self.last_p_in_pa is None or float(p_in_pa) != self.last_p_in_pa:
            raise ValueError("after_project must use the p_in of the preceding update() (%r vs %r)"
                             % (p_in_pa, self.last_p_in_pa))
        dt_over_rho = float(dt_s) / self.rho
        if not (math.isfinite(dt_over_rho) and dt_over_rho > 0.0):
            raise ValueError("dt_s must be finite and positive")
        self._after_project_kernel(dt_over_rho, dt_over_rho)

    def physical_pressure_into(self, out_field):
        """out = p~ + p_ref (physical gauge pressure, Pa) in fluid cells, 0 in obstacle cells (device only)."""
        if self.last_p_in_pa is None:
            raise RuntimeError("physical pressure requested before the first update()")
        self._physical_pressure_kernel(out_field)
        return out_field

    def physical_pressure_field(self):
        if self._physical_pressure is None:
            self._physical_pressure = ti.field(ti.f64, shape=self.shape)
        return self.physical_pressure_into(self._physical_pressure)

    def report(self):
        """Inlet / pocket cell counts, the largest |p~| in Dirichlet cells, the largest water divergence residual
        (div + s - volume source), the rows with a reference jump and the outlet flux (m3/s)."""
        self._report_kernel()
        s = self.stats.to_numpy()
        return dict(inlet_cells=int(s[S_INLET_CELLS]), pocket_cells=int(s[S_POCKET_CELLS]),
                    pockets=self.pocket_count, dirichlet_p_tilde_max_pa=float(s[S_DIRICHLET_PMAX]),
                    water_divergence_max_1ps=float(s[S_WATER_DIV]), jump_rows=int(s[S_JUMP_ROWS]),
                    outlet_flux_m3s=float(s[S_OUTLET_FLUX]), p_in_pa=self.last_p_in_pa, t_s=self.last_t_s)
