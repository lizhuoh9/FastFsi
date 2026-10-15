"""Excess mass of a solid denser than the fluid, carried by partner points tied to the material points by springs.

The single-velocity immersed-boundary coupling moves every material point with the fluid, and the fluid fills the body,
so each point already has the fluid's inertia rho_f V0.  For rho_s > rho_f the rest, M_ex = (rho_s - rho_f) V0
(reference volume V0; kg per metre of span for planar points), sits on a partner point with displacement Y - X0 and
velocity V.  A spring of fixed stiffness K = M_ex omega0**2 ties the partner to its material point X (penalty method of
Kim & Peskin 2007, as IBAMR's PenaltyIBMethod).  Each step the spring force F = K (Y - X) is computed once, rounded to
f32 (the precision of the fluid force field), spread to the fluid as a point force and applied with the opposite sign
to the partner, so fluid and partner receive exactly opposite impulses.  K is set once; it never changes with dt or
with the volume ratio J.

The partner is integrated with the same rule as the material points:
  explicit Euler (coupler.move_with_fluid): V -= dt F / M_ex, then Y += dt V (symplectic Euler, first order);
  midpoint rule (coupler.midpoint_predict / midpoint_correct, IBAMR MIDPOINT_RULE): Y drifts dt/2 with V, F is taken
  at the half-step positions of Y and X, V gets the full kick, Y drifts the second dt/2 with the new V
  (Stormer-Verlet, second order).

Stability: the spring and the fluid next to a point oscillate with frequency omega0 sqrt(1 + beta), where beta is the
largest ratio of partner mass to fluid mass seen through the kernel overlap: a row sum over all points and fluid faces
with absolute kernel weights (including the one-sided normalisation and graded-grid weights), at the positions where
the force is spread.  A step needs dt omega0 sqrt(1 + beta) <= OVERLAP_STABILITY_LIMIT, for either integrator (both
have the same explicit limit).  This bounds the explicit spring kick only; it does not cover the pressure projection
with inlet, outlet or air constraints.

Not modelled: gravity on the excess mass, i.e. the net weight (rho_s - rho_f) V0 g of a body denser than the fluid
(about 1 Pa on a 1 mm sheet with rho_s - rho_f = 100 kg/m^3).  Approximations: the spring is finite, so a partner lags
its point (the position and velocity slip are reported), and the mass uses V0, not the current volume J V0.

State: Y - X0 and V are checkpointed.  For planar points the material displacement u = x - X0 is kept here in f64 and
checkpointed, because an f32 coordinate near 1 m cannot hold sub-micrometre steps; points.x is its f32 rounding,
rebuilt on load.  Every other field is rewritten before it is read.  Checkpoints are allowed only between steps.  The
midpoint predictor keeps an exact copy of the accepted (Y, V) (planar: also u); a stop raised by the step's own checks
restores the partner from it, and abort_step() rolls the partner back to it.

Cost per step: three or four kernel launches and one device read.  accept() runs one kernel that checks every partner,
spring force and material point (3-D: also the coupler's count of points without a valid fluid face) and returns the
result with the overlap bound; only when something is wrong are the detailed checks run, which raise the usual hard
stop with its witness.  The bound is evaluated once per step, inside a kernel the step launches anyway, and reused while
the positions and the face mask stay unchanged (coupler.overlap_bound_key(): its geometry_epoch and the mask's
version).  After an explicit Euler step accept() evaluates it at the accepted positions, where the next spread happens;
a midpoint step evaluates it with its spring force at the half-step positions and checks it at accept().  Code that
moves points or edits a mask other than through the coupler and mask.refresh() must call
coupler.invalidate_overlap_bound().

No `from __future__ import annotations` (Taichi kernel module).
"""
import hashlib
import math

import numpy as np
import taichi as ti

from fastfsi.coupling.kernels import evaluate_ib3_weight, evaluate_ib4_weight
from fastfsi.errors import SimulationHardStop
from fastfsi.io.checkpoint import Checkpointable
from fastfsi.solid.material_points import point_state_flags

EXCESS_MASS_MODES = ("auto", "off")            # off: neutral inertia, a diagnostic only
# omega0 = OMEGA_DT_FRACTION / elastic dt: one spring period spans 2 pi / 0.1 ~ 63 elastic steps, so the partner
# follows its point on the solid's own time scale.
OMEGA_DT_FRACTION = .1
# dt omega0 sqrt(1 + beta) <= 0.5: a factor 4 inside the explicit limit 2 of the spring-fluid oscillation.
OVERLAP_STABILITY_LIMIT = .5
# A dt chosen from one evaluation of the bound must pass another evaluation of the same bound whose f64 sums were added
# in a different (atomic) order; the two differ by a few ulps, far below this.
DT_LIMIT_RELATIVE_TOLERANCE = 1e-7
# Midpoint drivers choose dt from the last evaluated bound (candidate_dt_limit, taken at the previous step's half-step
# positions); the step's own bound, checked at accept(), may then be up to 10 % tighter without stopping the run.
CANDIDATE_DT_MARGIN = .9

# bits of the step check in accept()
STATUS_PARTNER = 1          # a partner displacement or velocity is not finite
STATUS_SPRING = 2           # a spring force is not finite
STATUS_MATERIAL = 4         # a material point x, u, v, F is not finite or J <= 0; planar: the f64 material
                            # displacement is not finite or points.x is not its f32 rounding
STATUS_EMPTY_SUPPORT = 8    # a point component found no valid fluid face in its kernel support (3-D couplers)


def validate_densities(points, fluid, fluid_density_kgm3=None):
    """(rho_s, rho_f) of the solid points and the fluid; an explicit fluid density must equal the fluid's own."""
    actual = float(fluid.rho)
    rho_f = actual if fluid_density_kgm3 is None else float(fluid_density_kgm3)
    if rho_f != actual:
        raise ValueError("coupling fluid density must exactly match the fluid's momentum density")
    rho_s = float(points.rho_s)
    if not all(math.isfinite(rho) and rho > 0.0 for rho in (rho_s, rho_f)):
        raise ValueError("IB solid and fluid densities must be positive and finite")
    if rho_s < rho_f:
        raise ValueError("rho_s < rho_f is unsupported by positive reference excess mass")
    return rho_s, rho_f


@ti.func
def penalty_node_weight_3d(wa: ti.f32, wb: ti.f32, we: ti.f32, inv_D: ti.f32) -> ti.f32:
    """Node weight wa wb we / D of a 3-D body with excess mass: the product is formed in f64 and rounded to f32 once.

    The spread, the gather and the overlap bound all call this, so they use the identical f32 weight (a plain f32
    product may be contracted differently by the compiler in different kernels).  Equal-density bodies keep the
    original f32 expression.
    """
    product = ti.cast(wa, ti.f64) * ti.cast(wb, ti.f64) * ti.cast(we, ti.f64) * ti.cast(inv_D, ti.f64)
    return ti.cast(product, ti.f32)


@ti.func
def _nonfinite(value):
    return ti.math.isnan(value) or ti.math.isinf(value)


def _json_safe_values(values):
    """Finite numbers, or their repr for inf / nan, including nested lists (strict JSON witnesses)."""
    if isinstance(values, list):
        return [_json_safe_values(value) for value in values]
    value = float(values)
    return value if math.isfinite(value) else repr(value)


@ti.data_oriented
class PenaltyExcessMass(Checkpointable):
    """Partner points of the material points of one coupler: displacement Y - X0 (m), velocity (m/s), reference
    excess mass (kg; kg/m per unit span for planar points)."""
    CHECKPOINT_FIELDS = ("displacement", "velocity")
    CHECKPOINT_HOST = ("last_beta", "last_accepted_dt")

    def __init__(self, coupler, *, dim, rho_f, min_spacing_m, cfl_elastic=.3, omega_rad_s=None):
        self.coupler, self.points, self.dim = coupler, coupler.points, int(dim)
        self.n, self.rho_f = self.points.n, float(rho_f)
        cfl = float(cfl_elastic)
        if not (math.isfinite(cfl) and cfl > 0.):
            raise ValueError("excess mass needs a positive finite elastic CFL")
        self.neutral_elastic_dt = cfl * float(min_spacing_m) / math.sqrt(self.points.mu / self.rho_f)
        self.omega_rad_s = OMEGA_DT_FRACTION / self.neutral_elastic_dt if omega_rad_s is None else float(omega_rad_s)
        if not (math.isfinite(self.omega_rad_s) and self.omega_rad_s > 0.):
            raise ValueError("excess mass omega_rad_s must be positive and finite")
        volumes = (self.points.area if self.dim == 2 else self.points.vol).to_numpy().astype(np.float64)
        if not np.all(np.isfinite(volumes) & (volumes > 0.)):
            raise ValueError("excess mass requires positive finite reference point areas/volumes")
        mass = (self.points.rho_s - self.rho_f) * volumes
        # Both clamps still move their tethered points, so every point carries its excess mass.
        if not np.all(np.isfinite(mass)):
            raise ValueError("reference excess mass is not finite")
        self.reference_mass_sum = float(np.sum(mass, dtype=np.float64))
        if not math.isfinite(self.reference_mass_sum):
            raise ValueError("total reference excess mass is not finite")
        self.mass = ti.field(ti.f64, shape=self.n)                 # set once
        self.displacement = ti.Vector.field(self.dim, ti.f64, shape=self.n)
        self.velocity = ti.Vector.field(self.dim, ti.f64, shape=self.n)
        # the accepted partner state at the start of a midpoint step (written by every midpoint predictor)
        self.displacement_n = ti.Vector.field(self.dim, ti.f64, shape=self.n)
        self.velocity_n = ti.Vector.field(self.dim, ti.f64, shape=self.n)
        if self.dim == 2:
            # planar material displacement u = x - X0 in f64 (points.x is its f32 rounding) and its value at the start
            # of a midpoint step (written by every midpoint predictor)
            self.material_displacement = ti.Vector.field(2, ti.f64, shape=self.n)
            self.material_displacement_n = ti.Vector.field(2, ti.f64, shape=self.n)
            self.CHECKPOINT_FIELDS = ("displacement", "velocity", "material_displacement")
        # written before every read
        self.spring_force = ti.Vector.field(self.dim, ti.f64, shape=self.n)   # this step's force, f32 values
        # partner mass per fluid face; planar fluids are uniform along x, so one x layer holds it
        self.face_mass_density = ti.Vector.field(3, ti.f64, shape=(1 if self.dim == 2 else coupler.nx, coupler.ny,
                                                                    coupler.nz))
        self.bound_beta = ti.field(ti.f64, shape=2)     # overlap bound: [0] at this step's spread, [1] after accept
        self.bound_empty = ti.field(ti.i32, shape=2)    # 1: that evaluation met a point with no valid fluid face
        self.status = ti.field(ti.i32, shape=())        # STATUS_* bits of the last step check
        self.reduce = ti.field(ti.f64, shape=10)        # diagnostics
        self.mass.from_numpy(mass)
        self.last_beta, self.last_accepted_dt = 0., None
        self.last_integrator = None                    # of the last accepted step
        self._pending_dt = None                        # dt of the step in progress; None between steps
        # None, "drifted" (midpoint predictor), "spread", "moved", "stopped" (a check failed; a midpoint step's partner
        # is back at its copy): any stage but None refuses checkpoints and new steps
        self._stage = None
        self._midpoint = False
        self._spread_beta = None                       # bound checked at this step's spread (None: read at accept)
        self._spread_key = None
        self._bound_cache = None                       # (coupler.overlap_bound_key(), beta) of the last evaluation
        self._initialise_kernel()
        self.validate_state()

    # ------------------------------------------------------------------------------------------------ point functions
    @ti.func
    def current_displacement(self, p):
        delta = ti.Vector.zero(ti.f64, self.dim)
        for q in ti.static(range(self.dim)):
            if ti.static(self.dim == 3):
                delta[q] = ti.cast(self.points.u[p][q], ti.f64)
            else:
                delta[q] = self.material_displacement[p][q]
        return delta

    @ti.func
    def _point_bits(self, p, check_force: ti.template()):
        """STATUS_PARTNER, STATUS_SPRING (if check_force) and, for planar points, STATUS_MATERIAL bits of point p."""
        bits = 0
        for q in ti.static(range(self.dim)):
            if _nonfinite(self.displacement[p][q]) or _nonfinite(self.velocity[p][q]):
                bits |= STATUS_PARTNER
            if ti.static(check_force):
                if _nonfinite(self.spring_force[p][q]):
                    bits |= STATUS_SPRING
            if ti.static(self.dim == 2):
                material = self.material_displacement[p][q]
                expected_x = ti.cast(ti.cast(self.points.X0[p][q], ti.f64) + material, ti.f32)
                if (_nonfinite(material) or _nonfinite(expected_x) or _nonfinite(self.points.x[p][q]) or
                        self.points.x[p][q] != expected_x):
                    bits |= STATUS_MATERIAL
        return bits

    @ti.func
    def _planar_weight(self, c: ti.template(), yp, zp, a, b):
        ib = self.coupler
        sy = (yp - ib.origin[1]) / ib.h[1] - (0. if c == 1 else .5)
        sz = (zp - ib.origin[2]) / ib.h[2] - (0. if c == 2 else .5)
        j, k = ti.cast(ti.floor(sy), ti.i32) - 1 + a, ti.cast(ti.floor(sz), ti.i32) - 1 + b
        w = 0.
        if 0 <= j < ib.ny and 0 <= k < ib.nz:
            if ti.static(ib.ib3):
                w = evaluate_ib3_weight(sy - j) * evaluate_ib3_weight(sz - k)
            else:
                w = evaluate_ib4_weight(sy - j) * evaluate_ib4_weight(sz - k)
        return j, k, w

    @ti.func
    def _spread_absolute_mass(self, p, slot: ti.template(), record: ti.template()):
        """Partner mass of point p spread with absolute weights; a component without valid faces is recorded in the
        coupler's empty-support witnesses (record) or only flagged in bound_empty[slot]."""
        if self.mass[p] > 0.:
            xp = self.points.x[p]
            if ti.static(self.dim == 3):
                interior = 0
                if ti.static(self.coupler.interior_shortcut):
                    interior = self.coupler._interior(xp)
                for c in ti.static(range(3)):
                    ix, iy, iz, wx, wy, wz, dx, dy, dz = self.coupler._axis(c, xp)
                    D, gD = 1.0, ti.Vector([0.0, 0.0, 0.0])
                    if interior == 0:
                        D, gD = self.coupler._denominator(c, ix, iy, iz, wx, wy, wz, dx, dy, dz)
                    if D <= 0.:
                        if ti.static(record):
                            self.coupler._record_empty(p, c, xp)
                        else:
                            self.bound_empty[slot] = 1
                    else:
                        inv_D = 1.0 / D
                        for a in ti.static(range(4)):
                            for b, e in ti.ndrange(4, 4):
                                i, j, k = ix + a, iy + b, iz + e
                                if self.coupler._in_grid(i, j, k):
                                    m = 1.0
                                    if interior == 0:
                                        m = ti.cast(self.coupler.mask.face_valid(c, i, j, k), ti.f32)
                                    if m > 0.0:
                                        # Match _spread_node/_gather_node, including f32 multiplication order.
                                        w = penalty_node_weight_3d(wx[a], wy[b], wz[e], inv_D)
                                        density = self.mass[p] * ti.cast(ti.abs(w), ti.f64) / self.coupler.weights.face_volume(c, i, j, k)
                                        ti.atomic_add(self.face_mass_density[i, j, k][c], density)
            else:
                for c in ti.static((1, 2)):
                    for a, b in ti.static(ti.ndrange(4, 4)):
                        j, k, w = self._planar_weight(c, xp[0], xp[1], a, b)
                        if w != 0.:
                            density = self.mass[p] * ti.cast(ti.abs(w), ti.f64) / (self.coupler.h[1] * self.coupler.h[2])
                            ti.atomic_add(self.face_mass_density[0, j, k][c], density)

    @ti.func
    def _overlap_bound(self, p, slot: ti.template()):
        """Overlap row sums of point p's components into bound_beta[slot] (maximum)."""
        if self.mass[p] > 0.:
            xp = self.points.x[p]
            if ti.static(self.dim == 3):
                interior = 0
                if ti.static(self.coupler.interior_shortcut):
                    interior = self.coupler._interior(xp)
                for c in ti.static(range(3)):
                    ix, iy, iz, wx, wy, wz, dx, dy, dz = self.coupler._axis(c, xp)
                    D, gD = 1.0, ti.Vector([0.0, 0.0, 0.0])
                    if interior == 0:
                        D, gD = self.coupler._denominator(c, ix, iy, iz, wx, wy, wz, dx, dy, dz)
                    if D > 0.:
                        inv_D = 1.0 / D
                        beta = ti.cast(0., ti.f64)
                        for a in ti.static(range(4)):
                            for b, e in ti.ndrange(4, 4):
                                i, j, k = ix + a, iy + b, iz + e
                                if self.coupler._in_grid(i, j, k):
                                    m = 1.0
                                    if interior == 0:
                                        m = ti.cast(self.coupler.mask.face_valid(c, i, j, k), ti.f32)
                                    if m > 0.0:
                                        # Match _spread_node/_gather_node, including f32 multiplication order.
                                        w = penalty_node_weight_3d(wx[a], wy[b], wz[e], inv_D)
                                        beta += ti.cast(ti.abs(w), ti.f64) * self.face_mass_density[i, j, k][c] / self.rho_f
                        ti.atomic_max(self.bound_beta[slot], beta)
            else:
                for c in ti.static((1, 2)):
                    beta = ti.cast(0., ti.f64)
                    for a, b in ti.static(ti.ndrange(4, 4)):
                        j, k, w = self._planar_weight(c, xp[0], xp[1], a, b)
                        if w != 0.:
                            beta += ti.cast(ti.abs(w), ti.f64) * self.face_mass_density[0, j, k][c] / self.rho_f
                    ti.atomic_max(self.bound_beta[slot], beta)

    @ti.func
    def _spring(self, p):
        self.spring_force[p] = ti.Vector.zero(ti.f64, self.dim)
        if self.mass[p] > 0.:
            raw = self.mass[p] * self.omega_rad_s ** 2 * (self.displacement[p] - self.current_displacement(p))
            # Keep exactly the f32 value the fluid spread uses, for both fluid and partner.
            self.spring_force[p] = ti.cast(ti.cast(raw, ti.f32), ti.f64)

    # ------------------------------------------------------------------------------------------------ kernels
    @ti.kernel
    def _initialise_kernel(self):
        for p in range(self.n):
            self.displacement[p] = ti.Vector.zero(ti.f64, self.dim)
            self.velocity[p] = ti.Vector.zero(ti.f64, self.dim)
            self.spring_force[p] = ti.Vector.zero(ti.f64, self.dim)
            if ti.static(self.dim == 2):
                for q in ti.static(range(2)):
                    self.material_displacement[p][q] = ti.cast(self.points.x[p][q], ti.f64) - ti.cast(self.points.X0[p][q], ti.f64)
            if self.mass[p] > 0.:
                self.displacement[p] = self.current_displacement(p)
                if ti.static(self.dim == 3):
                    v, G = self.coupler._gather(self.coupler.fluid.velocity, self.coupler.fluid.velocity,
                                                0., p, self.points.x[p])
                    self.velocity[p] = ti.cast(v, ti.f64)
                else:
                    v = self.coupler._gather_face_vector(self.coupler.fluid.velocity,
                                                        self.points.x[p][0], self.points.x[p][1])
                    self.velocity[p] = ti.cast(v, ti.f64)

    @ti.kernel
    def _dt_limit_kernel(self) -> ti.types.vector(2, ti.f64):
        """The overlap bound at the current positions into bound_beta[0]; returns (empty-support count, beta)."""
        for i, j, k in self.face_mass_density:
            self.face_mass_density[i, j, k] = ti.Vector.zero(ti.f64, 3)
        self.bound_beta[0] = 0.
        self.bound_empty[0] = 0
        for p in range(self.n):
            self._spread_absolute_mass(p, 0, True)
        for p in range(self.n):
            self._overlap_bound(p, 0)
        count = 0
        if ti.static(self.coupler.TRACKS_EMPTY_SUPPORT):
            count = self.coupler.count_D0[None]
        return ti.Vector([ti.cast(count, ti.f64), self.bound_beta[0]])

    @ti.kernel
    def _prepare_kernel(self, evaluate_bound: ti.template()):
        """This step's spring force; with evaluate_bound, first the overlap bound at the current positions."""
        if ti.static(evaluate_bound):
            for i, j, k in self.face_mass_density:
                self.face_mass_density[i, j, k] = ti.Vector.zero(ti.f64, 3)
            self.bound_beta[0] = 0.
            self.bound_empty[0] = 0
            for p in range(self.n):
                self._spread_absolute_mass(p, 0, True)
            for p in range(self.n):
                self._overlap_bound(p, 0)
        for p in range(self.n):
            self._spring(p)

    @ti.kernel
    def _accept_kernel(self, prefetch: ti.template()) -> ti.types.vector(4, ti.f64):
        """The step check: (STATUS_* bits, bound at the spread, bound after accept, its empty flag).  With prefetch,
        first the overlap bound at the accepted positions into bound_beta[1]."""
        if ti.static(prefetch):
            for i, j, k in self.face_mass_density:
                self.face_mass_density[i, j, k] = ti.Vector.zero(ti.f64, 3)
            self.bound_beta[1] = 0.
            self.bound_empty[1] = 0
            for p in range(self.n):
                self._spread_absolute_mass(p, 1, False)
            for p in range(self.n):
                self._overlap_bound(p, 1)
        self.status[None] = 0
        for p in range(self.n):
            bits = self._point_bits(p, True)
            flags, J = point_state_flags(self.points, p, ti.static(self.dim), ti.static(self.dim == 3), True)
            if flags != 0:
                bits |= STATUS_MATERIAL
            if bits != 0:
                ti.atomic_or(self.status[None], bits)
        if ti.static(self.coupler.TRACKS_EMPTY_SUPPORT):
            if self.coupler.count_D0[None] > 0:
                self.status[None] |= STATUS_EMPTY_SUPPORT
        return ti.Vector([ti.cast(self.status[None], ti.f64), self.bound_beta[0], self.bound_beta[1],
                          ti.cast(self.bound_empty[1], ti.f64)])

    @ti.kernel
    def _first_invalid_kernel(self, mask: ti.i32, check_force: ti.template()) -> ti.i32:
        first = self.n
        for p in range(self.n):
            if (self._point_bits(p, check_force) & mask) != 0:
                ti.atomic_min(first, p)
        return first

    @ti.kernel
    def _kick_drift_kernel(self, dt: ti.f64):
        for p in range(self.n):
            if self.mass[p] > 0.:
                self.velocity[p] -= dt * self.spring_force[p] / self.mass[p]
                self.displacement[p] += dt * self.velocity[p]

    @ti.kernel
    def _begin_midpoint_kernel(self, half_dt: ti.f64):
        """Copy the accepted state, then Y^{n+1/2} = Y^n + dt/2 V^n."""
        for p in range(self.n):
            self.displacement_n[p] = self.displacement[p]
            self.velocity_n[p] = self.velocity[p]
            if ti.static(self.dim == 2):
                self.material_displacement_n[p] = self.material_displacement[p]
            if self.mass[p] > 0.:
                self.displacement[p] += half_dt * self.velocity[p]

    @ti.kernel
    def _finish_midpoint_kernel(self, dt: ti.f64):
        """V^{n+1} = V^n - dt F / M_ex with the half-step force, then Y^{n+1} = Y^{n+1/2} + dt/2 V^{n+1}."""
        for p in range(self.n):
            if self.mass[p] > 0.:
                self.velocity[p] = self.velocity_n[p] - dt * self.spring_force[p] / self.mass[p]
                self.displacement[p] += (.5 * dt) * self.velocity[p]

    @ti.kernel
    def _restore_step_start_kernel(self, material: ti.template()):
        """(Y, V) back to the copy of the midpoint predictor; with material, also the planar material displacement and
        its f32 position."""
        for p in range(self.n):
            self.displacement[p] = self.displacement_n[p]
            self.velocity[p] = self.velocity_n[p]
            if ti.static(material and self.dim == 2):
                self.material_displacement[p] = self.material_displacement_n[p]
                self.points.x[p] = ti.cast(ti.cast(self.points.X0[p], ti.f64) + self.material_displacement[p], ti.f32)

    @ti.kernel
    def _diagnostics_kernel(self):
        for s in self.reduce:
            self.reduce[s] = 0.
        for p in range(self.n):
            if self.mass[p] > 0.:
                slip = self.displacement[p] - self.current_displacement(p)
                ti.atomic_max(self.reduce[0], slip.norm())
                dv = self.velocity[p] - ti.cast(self.points.v[p], ti.f64)
                ti.atomic_max(self.reduce[1], dv.norm())
                for q in ti.static(range(self.dim)):
                    ti.atomic_add(self.reduce[2 + q], self.mass[p] * self.velocity[p][q])
                ti.atomic_add(self.reduce[5], .5 * self.mass[p] * self.velocity[p].norm_sqr())
                ti.atomic_add(self.reduce[6], .5 * self.mass[p] * self.omega_rad_s ** 2 * slip.norm_sqr())

    @ti.kernel
    def _refresh_positions_kernel(self):
        for p in range(self.n):
            self.points.x[p] = ti.cast(ti.cast(self.points.X0[p], ti.f64) + self.material_displacement[p], ti.f32)

    # ------------------------------------------------------------------------------------------------ checks
    def validate_state(self, *, check_force=False):
        """Raise SimulationHardStop at the first point whose partner displacement or velocity (and spring force, if
        check_force) is not finite or, for planar points, whose material displacement is not finite or whose points.x
        is not its f32 rounding."""
        self._raise_first_invalid(STATUS_PARTNER | STATUS_MATERIAL | (STATUS_SPRING if check_force else 0),
                                  bool(check_force))

    def _raise_first_invalid(self, mask, check_force):
        p = int(self._first_invalid_kernel(mask, check_force))
        if p >= self.n:
            return
        witness = dict(component="excess_mass", point=p,
                       displacement_m=_json_safe_values(list(self.displacement[p])),
                       velocity_mps=_json_safe_values(list(self.velocity[p])))
        if self.dim == 2:
            material = [float(v) for v in self.material_displacement[p]]
            reference = [float(v) for v in self.points.X0[p]]
            with np.errstate(over="ignore", invalid="ignore"):
                expected = (np.asarray(reference) + np.asarray(material)).astype(np.float32)
            witness.update(material_displacement_m=_json_safe_values(material),
                           position_m=_json_safe_values(list(self.points.x[p])),
                           expected_position_m=_json_safe_values(expected.tolist()))
        if check_force:
            witness["spring_force"] = _json_safe_values(list(self.spring_force[p]))
        raise SimulationHardStop("nonfinite_point_state", witness)

    @staticmethod
    def _check_bound(beta):
        if not (math.isfinite(beta) and beta >= 0.):
            raise SimulationHardStop("nonfinite_point_state", dict(component="excess_mass_bound", beta=repr(beta)))

    def _limit(self, beta):
        return OVERLAP_STABILITY_LIMIT / (self.omega_rad_s * math.sqrt(1. + beta))

    def _check_dt(self, dt, beta):
        limit = self._limit(beta)
        if dt > limit * (1. + DT_LIMIT_RELATIVE_TOLERANCE):
            raise SimulationHardStop("excess_mass_dt_limit", dict(dt_s=dt, limit_s=limit, beta=beta,
                                                                  omega_rad_s=self.omega_rad_s))

    def _raise_step_failure(self, status, dt, beta):
        """After a failed step check: the detailed checks, in the order in which the step produced the values; each
        raises its hard stop with the witness."""
        if status & STATUS_EMPTY_SUPPORT:
            self.coupler.one_sided_report(reset=False)
        self._check_bound(beta)
        self._check_dt(dt, beta)
        if status & STATUS_SPRING:
            self._raise_first_invalid(STATUS_SPRING, True)
        if status & STATUS_PARTNER:
            self._raise_first_invalid(STATUS_PARTNER, False)
        if status & STATUS_MATERIAL:
            self.points.validate_state()
            self._raise_first_invalid(STATUS_MATERIAL, False)
        raise RuntimeError("excess-mass step check reported status %d but no detailed check failed" % status)

    def _stop_step(self, raise_witness, *args):
        """Raise the hard stop (raise_witness builds it from the failing values); a midpoint step's partner then returns
        to its step-start copy.  The step stays unaccepted until restore_state(), load_state_arrays() or abort_step()."""
        try:
            raise_witness(*args)
        finally:
            if self._midpoint:
                self._restore_step_start_kernel(False)
            self._stage = "stopped"

    # ------------------------------------------------------------------------------------------------ bound
    def _cached_beta(self, key):
        if key is not None and self._bound_cache is not None and self._bound_cache[0] == key:
            return self._bound_cache[1]
        return None

    def dt_limit(self):
        """Overlap limit of the spring kick at the current positions and face mask, evaluated at most once while they
        stay unchanged; raises the empty-support or non-finite-bound hard stop."""
        key = self.coupler.overlap_bound_key()
        beta = self._cached_beta(key)
        if beta is None:
            result = self._dt_limit_kernel()                    # one launch, one device read
            if int(result[0]):
                self.coupler.one_sided_report(reset=False)      # empty-support witness; never use an invalid bound
            beta = float(result[1])
            self._check_bound(beta)
            self._bound_cache = None if key is None else (key, beta)
        self.last_beta = beta
        return self._limit(beta)

    def candidate_dt_limit(self):
        """The limit of the last evaluated bound (no device work).  Midpoint drivers choose dt from it with
        CANDIDATE_DT_MARGIN; the step's own bound is checked at accept()."""
        self._validate_host(self.state_host())
        return self._limit(self.last_beta)

    # ------------------------------------------------------------------------------------------------ step
    @staticmethod
    def _step_dt(dt_s):
        dt = float(dt_s) if dt_s is not None else float("nan")
        if not (math.isfinite(dt) and dt > 0.):
            raise ValueError("excess mass requires the full positive finite step dt_s")
        return dt

    def begin_midpoint(self, dt_s):
        """Midpoint predictor: copy the accepted (Y, V) (planar: also the material displacement), then Y drifts dt/2
        with V.  Call before the material points move to their half-step positions."""
        if self._pending_dt is not None:
            raise ValueError("an excess-mass step is already in progress; accept, abort or restore it first")
        dt = self._step_dt(dt_s)
        self._begin_midpoint_kernel(.5 * dt)
        self._pending_dt, self._stage, self._midpoint = dt, "drifted", True

    def prepare_force(self, dt_s):
        """This step's spring force at the current positions (the half-step positions of a midpoint step), before
        the spread.  A known bound for these positions is checked now; otherwise the bound is evaluated in the same
        launch and checked at accept()."""
        midpoint = self._stage == "drifted"
        if self._pending_dt is not None and not midpoint:
            raise ValueError("excess-mass spring impulse already pending; advance or restore before another spread")
        dt = self._step_dt(dt_s)
        if midpoint and dt != self._pending_dt:
            raise ValueError("the midpoint spread must use the dt_s of its predictor")
        key = self.coupler.overlap_bound_key()
        beta = self._cached_beta(key)
        if beta is not None:
            self.last_beta = beta
            if midpoint and dt > self._limit(beta) * (1. + DT_LIMIT_RELATIVE_TOLERANCE):
                self._stop_step(self._check_dt, dt, beta)        # the predictor has already moved the partner
            self._check_dt(dt, beta)                             # explicit Euler: nothing has moved yet
        self._prepare_kernel(beta is None)
        self._pending_dt, self._stage, self._midpoint = dt, "spread", midpoint
        self._spread_beta, self._spread_key = beta, key

    def advance(self, dt_s):
        """Explicit Euler step of the partner: V -= dt F / M_ex, then Y += dt V."""
        if self._midpoint:
            raise ValueError("a midpoint step is in progress; finish it with the midpoint correction")
        dt = float(dt_s)
        if self._stage != "spread" or dt != self._pending_dt:
            raise ValueError("excess mass must consume exactly the full dt_s of its saved spring impulse")
        self._kick_drift_kernel(dt)
        self._stage = "moved"

    def finish_midpoint(self, dt_s):
        """Midpoint corrector: V^{n+1} = V^n - dt F / M_ex with the half-step spring force, then Y drifts the second
        dt/2 with V^{n+1}."""
        dt = float(dt_s)
        if self._stage != "spread" or not self._midpoint or dt != self._pending_dt:
            raise ValueError("finish_midpoint needs the spread of a midpoint step with the same dt_s")
        self._finish_midpoint_kernel(dt)
        self._stage = "moved"

    def accept(self, dt_s):
        """End of the step, after the material points moved: one launch checks every partner, spring force and
        material point (3-D: also the coupler's empty-support count) and returns the overlap bounds; this is the only
        device read of a healthy step.  After an explicit Euler step the same launch first evaluates the bound at
        the accepted positions, where the next spread happens."""
        dt = float(dt_s)
        if self._stage != "moved" or dt != self._pending_dt:
            raise ValueError("excess mass acceptance must match its complete pending dt_s")
        key = self.coupler.overlap_bound_key()
        prefetch = key is not None and not self._midpoint
        result = self._accept_kernel(prefetch)
        status = int(result[0])
        beta = float(result[1]) if self._spread_beta is None else self._spread_beta
        if (status or not (math.isfinite(beta) and beta >= 0.) or
                dt > self._limit(beta) * (1. + DT_LIMIT_RELATIVE_TOLERANCE)):
            self._stop_step(self._raise_step_failure, status, dt, beta)
        if self._spread_beta is None and self._spread_key is not None:
            self._bound_cache = (self._spread_key, beta)
        beta_next = float(result[2])
        if prefetch and int(result[3]) == 0 and math.isfinite(beta_next) and beta_next >= 0.:
            self._bound_cache = (key, beta_next)
        self.last_beta, self.last_accepted_dt = beta, dt
        self.last_integrator = "stormer_verlet" if self._midpoint else "symplectic_euler"
        self._end_step()

    def abort_step(self):
        """Roll the partner back to the accepted state of the step in progress: a midpoint step from the predictor's
        exact copy (planar points also restore the material displacement and points.x); an explicit Euler step only
        before advance().  The caller restores the other material point fields and the fluid."""
        if self._pending_dt is None:
            return
        if self._midpoint:
            self._restore_step_start_kernel(True)
        elif self._stage != "spread":
            raise ValueError("an explicit Euler step cannot be rolled back after advance(); restore a saved state")
        self._end_step()

    def _end_step(self):
        self._pending_dt = self._stage = self._spread_beta = self._spread_key = None
        self._midpoint = False

    # ------------------------------------------------------------------------------------------------ output
    def diagnostics(self):
        self._diagnostics_kernel()
        r = self.reduce.to_numpy()
        return dict(target_reference_density_ratio=self.points.rho_s / self.rho_f,
                    reference_excess_mass=self.reference_mass_sum, mass_units="kg/m" if self.dim == 2 else "kg",
                    method="fixed_penalty_spring_partner", integrator=self.last_integrator,
                    omega_rad_s=self.omega_rad_s, max_position_slip_m=float(r[0]), max_velocity_slip_mps=float(r[1]),
                    momentum=[float(r[2 + q]) for q in range(self.dim)], kinetic_energy=float(r[5]),
                    spring_energy=float(r[6]), last_beta=self.last_beta)

    # ------------------------------------------------------------------------------------------------ checkpoint
    def state_array_names(self):
        if self._pending_dt is not None:
            raise ValueError("excess-mass checkpoints require an accepted step boundary")
        self.validate_state()
        return super().state_array_names()

    def state_array(self, name):
        self.state_array_names()
        return super().state_array(name)

    def load_state_arrays(self, arrays):
        # Check shape, dtype and finiteness of every array before writing any of them.
        if set(arrays) != set(self.CHECKPOINT_FIELDS):
            raise ValueError("excess-mass checkpoint fields do not match")
        for name in self.CHECKPOINT_FIELDS:
            value = np.asarray(arrays[name])
            if value.shape != (self.n, self.dim) or value.dtype != np.dtype(np.float64) or not np.all(np.isfinite(value)):
                raise ValueError("invalid excess-mass checkpoint %s" % name)
        if self.dim == 2:
            with np.errstate(over="ignore", invalid="ignore"):
                expected_x = (self.points.X0.to_numpy().astype(np.float64) +
                              arrays["material_displacement"]).astype(np.float32)
            if not np.all(np.isfinite(expected_x)):
                raise ValueError("excess-mass checkpoint material displacement has no finite f32 position")
        for name in self.CHECKPOINT_FIELDS:
            getattr(self, name).from_numpy(arrays[name])
        if self.dim == 2:
            self._refresh_positions_kernel()           # points.x is rebuilt from the loaded material displacement
        self._end_step()
        self._bound_cache = None

    @staticmethod
    def _validate_host(values):
        if set(values) != {"last_beta", "last_accepted_dt"}:
            raise ValueError("excess-mass checkpoint host fields do not match")
        beta, dt = values["last_beta"], values["last_accepted_dt"]
        if type(beta) is not float or not (math.isfinite(beta) and beta >= 0.):
            raise ValueError("excess-mass checkpoint beta is invalid")
        if dt is not None and (type(dt) is not float or not (math.isfinite(dt) and dt > 0.)):
            raise ValueError("excess-mass checkpoint accepted dt is invalid")

    def load_state_host(self, values):
        self._validate_host(values)
        super().load_state_host(values)

    def save_state(self):
        return dict(arrays={name: self.state_array(name).copy() for name in self.state_array_names()},
                    host=self.state_host())

    def restore_state(self, saved):
        self._validate_host(saved["host"])
        self.load_state_arrays(saved["arrays"])
        self.load_state_host(saved["host"])

    def fingerprint(self):
        digest = hashlib.sha256()
        digest.update(self.mass.to_numpy().tobytes())
        digest.update(self.points.X0.to_numpy().tobytes())
        # the label is part of the checkpoint identity; it stays unchanged so that existing checkpoints still match
        digest.update(repr((self.dim, self.rho_f, self.points.rho_s, self.omega_rad_s,
                            self.coupler.clamp_mode, "fixed_penalty_symplectic_euler_reference_v0")).encode())
        return digest.hexdigest()


class ExcessMassCouplerMixin:
    """Density handling and the point time stepping shared by the 3-D and planar couplers.

    The coupler supplies points, fluid, the kernels _interpolate_update_kernel(velocity, dt) and
    _midpoint_correct_kernel(velocity_old, velocity_new, dt), overlap_bound_key(), and TRACKS_EMPTY_SUPPORT (True for
    a coupler that counts points without a valid fluid face in count_D0 and reports them with one_sided_report()).
    Without excess mass every method is the original single-velocity update.
    """
    TRACKS_EMPTY_SUPPORT = False

    def _init_densities(self, points, fluid, fluid_density_kgm3, excess_mass_mode):
        self.rho_s, self.rho_f = validate_densities(points, fluid, fluid_density_kgm3)
        if excess_mass_mode not in EXCESS_MASS_MODES:
            raise ValueError("excess_mass_mode must be auto or diagnostic off")
        self.excess_mass_mode = excess_mass_mode
        self.excess_mass_enabled = self.rho_s > self.rho_f and excess_mass_mode == "auto"
        self.excess_mass = None
        self.geometry_epoch = 0          # changes whenever point positions or kernel supports may have changed

    def _init_excess_mass(self, *, dim, min_spacing_m, cfl_elastic, omega_rad_s):
        if self.excess_mass_enabled:
            self.excess_mass = PenaltyExcessMass(self, dim=dim, rho_f=self.rho_f, min_spacing_m=min_spacing_m,
                                                 cfl_elastic=cfl_elastic, omega_rad_s=omega_rad_s)

    def invalidate_overlap_bound(self):
        """Call after writing point positions or a face mask other than through this coupler and mask.refresh()."""
        self.geometry_epoch += 1

    def move_with_fluid(self, dt_s):
        """Explicit Euler update of the points with the current fluid velocity (partner: symplectic Euler)."""
        if self.excess_mass is not None:
            self.excess_mass.advance(dt_s)
        self._interpolate_update_kernel(self.fluid.velocity, float(dt_s))
        self.geometry_epoch += 1
        if self.excess_mass is not None:
            self.excess_mass.accept(dt_s)

    def midpoint_predict(self, dt_s):
        """MIDPOINT_RULE first half: keep the step-start state and move to X^{n+1/2}, F^{n+1/2} with u^n (partner:
        copy of the accepted state, then a dt/2 drift)."""
        if self.excess_mass is not None:
            self.excess_mass.begin_midpoint(dt_s)
        self.points.save_state()
        self._interpolate_update_kernel(self.fluid.velocity, 0.5 * float(dt_s))
        self.geometry_epoch += 1

    def midpoint_correct(self, velocity_old, dt_s):
        """MIDPOINT_RULE second half, from the step start with U^{n+1/2} = (u^n + u^{n+1}) / 2 at X^{n+1/2} (partner:
        full kick, then the second dt/2 drift; then the step check)."""
        if self.excess_mass is not None:
            self.excess_mass.finish_midpoint(dt_s)
        self._midpoint_correct_kernel(velocity_old, self.fluid.velocity, float(dt_s))
        self.geometry_epoch += 1
        if self.excess_mass is not None:
            self.excess_mass.accept(dt_s)

    def excess_mass_dt_limit(self):
        """Overlap limit of the spring kick at the current positions (infinity without excess mass)."""
        return float("inf") if self.excess_mass is None else self.excess_mass.dt_limit()

    def excess_mass_candidate_dt_limit(self):
        """Limit of the last evaluated bound, without device work (infinity without excess mass)."""
        return float("inf") if self.excess_mass is None else self.excess_mass.candidate_dt_limit()

    def density_report(self):
        """Density set-up of the coupling, for run metadata."""
        enabled = self.excess_mass is not None
        return dict(mode=self.excess_mass_mode, target_reference_density_ratio=self.rho_s / self.rho_f,
                    effective_reference_density_kgm3=self.rho_s if enabled else self.rho_f,
                    effective_reference_density_ratio=self.rho_s / self.rho_f if enabled else 1.,
                    reference_excess_mass=0. if not enabled else self.excess_mass.reference_mass_sum,
                    omega_rad_s=None if not enabled else self.excess_mass.omega_rad_s,
                    method="fixed_penalty_spring_partner" if enabled else "neutral_single_velocity")

    def _excess_mass_diagnostics(self, material_momentum):
        """The partner diagnostics and the reference body momentum rho_f V0 U + M_ex V (U: interpolated material
        velocity, V: partner velocity; per unit span for planar points; clamp forces are external reactions, not
        partner impulses), as new keys for the coupler's diagnostics (none without excess mass)."""
        if self.excess_mass is None:
            return {}
        partner = self.excess_mass.diagnostics()
        scale = self.rho_f / self.rho_s
        return dict(excess_mass=partner,
                    combined_reference_body_momentum=[m * scale + partner["momentum"][q]
                                                      for q, m in enumerate(material_momentum)])
