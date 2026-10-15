"""The projection contract and the exactly-once / invariant guard of the shifted z-min pressure outlet, shared by every
chamber that solves a shifted pressure (ColumnPressureChamber, PressureRegions and through it PneumaticChamber).

THE SPLIT OUTLET CORRECTION.  The projection solves p~ = p - p_ref, so the outlet ghost value is p~_out = -p_ref
(physical 0 at the outlet face).  The solver itself corrects the outlet face with ghost 0 and books the interface flux
s = (dt / rho)(D p~ - R) on every row; the chamber's after_project() then adds the missing part, (dt / rho) R dz0 on the
face and (dt / rho) R on s.  Between the two halves an outlet row satisfies div + s - volume_source = 0 (the solver's
half), and the chamber's half keeps it (div and s change by opposite amounts).  Between the halves the outlet face is
often INFLOW (for p_ref > 0 the solver's ghost-0 correction pushes it up by (dt / rho) 2 p_ref / dz0).  Anything that
changes an outlet face between the halves breaks the first equality.  Examples: the solver's default
pressure_outlet_backflow_policy="clamp" (zeroes positive outlet faces), a velocity boundary re-apply, or a divergence
cleanup that uses div without s.  The chamber's half then lands on a changed face, and the outlet cell keeps a
divergence of the size (dt / rho) |R|.  The next step's right-hand side carries it, so the outlet ghost doubles and the
water near the outlet settles near -p_ref instead of 0.

THIS MODULE (one code path for every shifted-outlet chamber):
  - PROJECT_REQUIREMENTS: the projection contract of the shifted outlet (z-min outlet, conjugate gradient with the
    multigrid preconditioner, outlet backflow ALLOWED, no pressure accumulation).  Chambers carry it as the class
    attribute PROJECT_REQUIREMENTS.
  - merge_project_requirements(chamber, kwargs): the caller's project() kwargs with the chamber's contract merged in.
    It raises for a chamber whose class declares no contract and for a caller value that conflicts with the contract
    (the multigrid preconditioner family counts as one value).  Never a silent default.
  - OutletShiftGuard: arm() in the chamber's update(); before() at the start of after_project() consumes the arming
    (exactly one after_project() per update(): a second one raises) and measures the outlet rows before the chamber's
    half; after() measures them after it.  Interval maxima live on the device (no host read per step); report() reads
    them once and resets them.  Outlet rows checked: k = 0, i and j off the domain edges, fluid with fluid neighbours
    (four lateral, one above) and no Dirichlet diagonal (D = 0).  Measured: pre / post = max |div + s - volume_source|
    (1/s, the solver's face divergence of interior faces, the outlet face as stored), shift = max |(dt / rho) R_out|.
    ratio = pre / shift: about the solver tolerance when the halves meet; about 1 when the solver's half was changed.
  - Strict mode (environment FASTFSI_STRICT_INVARIANTS=1, used by the test suites): report() raises SimulationHardStop
    "outlet_shift_invariant" when ratio > INVARIANT_RATIO_MAX.  Otherwise the caller reports it (a run-time warning).

No `from __future__ import annotations` (Taichi kernel module).
"""
import math
import os

import taichi as ti

from fastfsi.errors import SimulationHardStop

PROJECT_REQUIREMENTS = dict(pressure_outlet_zmin=True, pressure_solver="fv_cg", cg_preconditioner="fv_multigrid",
                            pressure_outlet_backflow_policy="allow", accumulate_pressure_into_previous=False)
MULTIGRID_PRECONDITIONERS = frozenset(("fv_multigrid", "fv_multigrid_light"))
STRICT_ENVIRONMENT = "FASTFSI_STRICT_INVARIANTS"
INVARIANT_RATIO_MAX = 1.0e-2           # pre / shift; the halves meet at the solver tolerance (tests: < 1e-4)
HARD_STOP_CODES = {"outlet_shift_invariant"}
G_PRE, G_POST, G_SHIFT, G_CALLS = 0, 1, 2, 3
G_SLOTS = 4


def merge_project_requirements(chamber, project_kwargs):
    """project_kwargs with the chamber class's PROJECT_REQUIREMENTS merged in (a new dict).  Raises ValueError when the
    class declares no contract or when a caller value conflicts with it; the multigrid preconditioner family is one
    value (the contract needs a multigrid preconditioner, either variant)."""
    name = type(chamber).__name__
    requirements = getattr(type(chamber), "PROJECT_REQUIREMENTS", None)
    if not isinstance(requirements, dict) or not requirements:
        raise ValueError("%s declares no PROJECT_REQUIREMENTS class attribute: a shifted-pressure chamber must carry "
                         "its projection contract" % name)
    merged = dict(project_kwargs)
    for key, value in requirements.items():
        if (key == "cg_preconditioner" and value in MULTIGRID_PRECONDITIONERS
                and merged.get(key) in MULTIGRID_PRECONDITIONERS):
            continue
        if key in merged and merged[key] != value:
            raise ValueError("%s requires %s=%r for its shifted outlet; the caller has %r" % (name, key, value,
                                                                                            merged[key]))
        merged[key] = value
    return merged


@ti.data_oriented
class OutletShiftGuard:
    """See the module docstring."""

    def __init__(self, fluid, *, strict=None):
        self.fluid = fluid
        self.shape = tuple(int(v) for v in fluid.grid.grid_nodes)
        if self.shape[2] < 2:
            raise ValueError("the outlet guard needs at least two cell layers in z")
        self.stats = ti.field(ti.f64, shape=G_SLOTS)
        self.stats.fill(0.0)
        self.armed = False
        self.strict = (os.environ.get(STRICT_ENVIRONMENT, "") == "1") if strict is None else bool(strict)

    @ti.func
    def _checkable(self, i, j):
        ok = 1
        if self.fluid.obstacle[i, j, 0] != 0 or self.fluid.obstacle[i, j, 1] != 0:
            ok = 0
        if (self.fluid.obstacle[i - 1, j, 0] != 0 or self.fluid.obstacle[i + 1, j, 0] != 0
                or self.fluid.obstacle[i, j - 1, 0] != 0 or self.fluid.obstacle[i, j + 1, 0] != 0):
            ok = 0
        if self.fluid.pressure_interface_matrix_diagonal[i, j, 0] != 0.0:
            ok = 0
        return ok

    @ti.func
    def _residual(self, i, j):
        here = self.fluid.velocity[i, j, 0]
        div = ((ti.cast(self.fluid.velocity[i + 1, j, 0].x, ti.f64) - ti.cast(here.x, ti.f64))
               / ti.cast(self.fluid.cell_width_x_m[i], ti.f64)
               + (ti.cast(self.fluid.velocity[i, j + 1, 0].y, ti.f64) - ti.cast(here.y, ti.f64))
               / ti.cast(self.fluid.cell_width_y_m[j], ti.f64)
               + (ti.cast(self.fluid.velocity[i, j, 1].z, ti.f64) - ti.cast(here.z, ti.f64))
               / ti.cast(self.fluid.cell_width_z_m[0], ti.f64))
        return ti.abs(div + self.fluid.pressure_interface_projection_divergence_s[i, j, 0]
                      - ti.cast(self.fluid.volume_source_s[i, j, 0], ti.f64))

    @ti.kernel
    def _measure_kernel(self, outlet_rhs: ti.template(), dt_over_rho: ti.f64, slot: ti.i32, with_shift: ti.i32):
        for i, j in ti.ndrange((1, self.shape[0] - 1), (1, self.shape[1] - 1)):
            if self._checkable(i, j) != 0:
                ti.atomic_max(self.stats[slot], self._residual(i, j))
                if with_shift != 0:
                    ti.atomic_max(self.stats[G_SHIFT], ti.abs(dt_over_rho * outlet_rhs[i, j]))
        if with_shift != 0:
            self.stats[G_CALLS] += 1.0

    def arm(self):
        """update(): this step's outlet rows are written; exactly one after_project() may follow."""
        self.armed = True

    def before(self, outlet_rhs, dt_over_rho):
        """Start of after_project(): consume the arming (raises without it) and measure the solver's half."""
        if not self.armed:
            raise RuntimeError("after_project() without a preceding update(): the outlet face term would be applied "
                               "twice to one projection")
        self.armed = False
        self._measure_kernel(outlet_rhs, float(dt_over_rho), G_PRE, 1)

    def after(self, outlet_rhs, dt_over_rho):
        """End of after_project(): measure the outlet rows after the chamber's half."""
        self._measure_kernel(outlet_rhs, float(dt_over_rho), G_POST, 0)

    def report(self):
        """Interval maxima since the last report (one device read; reset), the ratio pre / shift (None while no outlet
        shift was applied) and the strict-mode hard stop."""
        s = self.stats.to_numpy()
        self.stats.fill(0.0)
        pre, post, shift, calls = float(s[G_PRE]), float(s[G_POST]), float(s[G_SHIFT]), int(s[G_CALLS])
        ratio = pre / shift if shift > 0.0 and math.isfinite(pre) else None
        out = dict(outlet_guard_pre_max_1ps=pre, outlet_guard_post_max_1ps=post, outlet_guard_shift_max_1ps=shift,
                   outlet_guard_ratio=ratio, outlet_guard_calls=calls)
        if self.strict and (not math.isfinite(pre) or (ratio is not None and ratio > INVARIANT_RATIO_MAX)):
            raise SimulationHardStop("outlet_shift_invariant", dict(out, ratio_max=INVARIANT_RATIO_MAX))
        return out
