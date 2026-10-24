"""Single-velocity immersed-boundary coupling of material-point solids with an incompressible fluid on a MAC grid.

Elastic stress, tether pull, external load and extra point force are spread to the fluid MAC grid; the material
points move with interpolated velocity and F follows its gradient, by explicit Euler (move_with_fluid) or by the midpoint
rule (midpoint_predict / midpoint_correct).  Equal-density solids use the original single-velocity path.  For
rho_s > rho_f the excess mass (rho_s - rho_f) V0 sits on partner points tied to the material points by fixed springs
(coupling.excess_mass: equal and opposite impulses; symplectic Euler with the Euler update, Stormer-Verlet with the
midpoint rule).  The explicit limits use the elastic wave speed with rho_f and the spring's mass-overlap bound.
rho_s < rho_f is refused.  Eulerian pressure inside an incompressible body represents its hydrostatic stress.

The point state lives in solid.material_points (MaterialPoints, PlanarMaterialPoints); the rigid clamp by direct
forcing in coupling.direct_forcing.

ImmersedBoundaryCoupler (3-D) accesses the grid through kernel weights (coupling.kernel_weights: axis_weights,
face_volume, n, min_spacing_m, on the grid.axes geometry) and a face-validity mask (coupling.face_mask: face_valid).
ONE-SIDED NORMALISATION: with m the face validity, D = sum w m, w~ = w m / D and grad w~ = (m grad w - w~ grad D) / D,
grad D = sum m grad w.  The same w~ is used to spread and to interpolate, so the pair stays adjoint (power identity) and
sum w~ = 1 keeps both the spread force and the internal-stress force balance (sum grad w~ = 0) exact next to invalid
faces.  D <= 0 raises the hard stop 'one_sided_empty_support' with a witness.

Interior shortcut (interior_shortcut=True): points whose whole kernel support lies on valid fluid faces skip the mask
lookup and normalisation; mathematically identical, bit-identical by test.  refresh_support_flags() marks the cells
within SUPPORT_REACH of an invalid face, after every mask refresh.  Further physics-neutral speed choices: min D is a
separate diagnostics pass; the stress is computed inside the spread kernel; the caller may clear the force field in a
fused pass (spread_force(clear=False)); the inner node loops are dynamic by default (unroll_static=True unrolls all 64).

Clamp: clamp="penalty" pulls tethered points back to their anchors with stiffness tether_stiffness_pa_per_m;
clamp="direct" makes them a rigid clamp by direct forcing (coupling.direct_forcing.DirectForcingClamp), applied at the
end of every spread.

Held deformation (3-D, Root 2026-10-24): the direct clamp holds a tethered point's interpolated VELOCITY near 0, but F
integrates the interpolated gradient, so a held region that does not move still accumulates strain and spreads its
stress.  held_f="frozen" skips the F update of tethered points: their F stays I (set at construction and again after a
checkpoint load, the largest |F - I| removed kept in held_f_reset), so their stress is 0 and the clamp alone supplies
the edge's reaction.  The default held_f="evolve" updates every point, as before.

Point tags (fastfsi.point_tags): the min D of the TAG_CHAMBER_SURFACE points and of the free TAG_CONTACT_RIM points are
reported separately.

PlanarImmersedBoundaryCoupler is the quasi-2-D version (one body in the (y, z) plane, fields uniform along x) on a
uniform grid, with the IB_4 or IB_3 kernel evaluated directly.

References (algorithms; no source copied): IBAMR (BSD-3-Clause) IBExplicitHierarchyIntegrator and LEInteractor
(explicit IB stepping, side-centred interpolate / spread, the 1 / (dx dy dz) spread scaling); taichi_mpm (MIT)
mls-mpm88 (the stress term f_i = -sum_p V0_p tau_p grad w_ip and the F update).

No `from __future__ import annotations` (Taichi kernel module).
"""
import hashlib
import math

import numpy as np
import taichi as ti

from fastfsi.coupling.direct_forcing import DirectForcingClamp, PlanarDirectForcingClamp
from fastfsi.coupling.excess_mass import ExcessMassCouplerMixin, penalty_node_weight_3d
from fastfsi.coupling.kernels import (evaluate_ib3_weight, evaluate_ib3_weight_derivative, evaluate_ib4_weight,
                                      evaluate_ib4_weight_derivative, validate_kernel_name)
from fastfsi.errors import HARD_STOP_CODES as REGISTERED_HARD_STOP_CODES, SimulationHardStop
from fastfsi.io.checkpoint import Checkpointable
from fastfsi.point_tags import TAG_CHAMBER_SURFACE, TAG_CONTACT_RIM
from fastfsi.solid.materials import evaluate_neo_hookean_kirchhoff, evaluate_neo_hookean_kirchhoff_planar

CLAMP_MODES = ("penalty", "direct")
HELD_F_MODES = ("evolve", "frozen")
WITNESS_SLOTS = 16
SUPPORT_REACH = 3         # cells: covers both node families of the 4-node support around the host cell
HARD_STOP_CODES = set(REGISTERED_HARD_STOP_CODES)


def _validated_clamp(clamp):
    if clamp not in CLAMP_MODES:
        raise ValueError("clamp must be one of %s, got %r" % (CLAMP_MODES, clamp))
    return clamp


@ti.data_oriented
class ImmersedBoundaryCoupler(ExcessMassCouplerMixin, Checkpointable):
    """Couples MaterialPoints with a 3-D fluid; grid access through `weights`, validity through `mask`."""

    # checkpoint (state census): the empty-support witness count and rows recorded since the last check; the
    # points are their own component; every other field is rewritten before use
    CHECKPOINT_FIELDS = ("count_D0", "witness")
    TRACKS_EMPTY_SUPPORT = True     # count_D0 / witness / one_sided_report()

    def __init__(self, *, fluid, weights, mask, points, tether_stiffness_pa_per_m, interior_shortcut=True,
                 unroll_static=False, clamp="penalty", clamp_density_kgm3=None, fluid_density_kgm3=None,
                 excess_mass_mode="auto", excess_mass_omega_rad_s=None, cfl_elastic=.3, held_f="evolve"):
        if held_f not in HELD_F_MODES:
            raise ValueError("held_f must be one of %s, got %r" % (HELD_F_MODES, held_f))
        self.held_f = held_f
        self.freeze_held_f = held_f == "frozen"
        self.held_f_reset = None
        self.fluid = fluid
        self.weights = weights
        self.mask = mask
        self.points = points
        self.n = points.n
        self._init_densities(points, fluid, fluid_density_kgm3, excess_mass_mode)
        self.interior_shortcut = bool(interior_shortcut)
        self.unroll_static = bool(unroll_static)
        self.nx, self.ny, self.nz = (int(v) for v in weights.n)
        self.tether_k = float(tether_stiffness_pa_per_m)
        self.min_D = ti.field(ti.f32, shape=())
        self.count_D0 = ti.field(ti.i32, shape=())
        self.min_D_point = ti.field(ti.i32, shape=())
        self.min_D_free = ti.field(ti.f32, shape=())
        self.min_D_contact_rim_free = ti.field(ti.f32, shape=())        # free TAG_CONTACT_RIM points
        self.min_D_chamber_value = ti.field(ti.f32, shape=())             # TAG_CHAMBER_SURFACE points
        self.witness = ti.Vector.field(5, ti.f32, shape=WITNESS_SLOTS)   # point, component, x, y, z
        self.reaction = ti.field(ti.f64, shape=4)                         # tether pull x, y, z; clamp max speed
        grid_shape = (self.nx, self.ny, self.nz)
        self.support_flag = ti.field(ti.u8, shape=grid_shape)  # 1: a cell within SUPPORT_REACH of an invalid face
        self._flag_a = ti.field(ti.u8, shape=grid_shape)
        self._flag_b = ti.field(ti.u8, shape=grid_shape)
        self.clamp_mode = _validated_clamp(clamp)
        self.direct_clamp = clamp == "direct"
        self.clamp = None
        if self.direct_clamp:
            self.clamp = DirectForcingClamp(self, density_kgm3=(self.rho_f if clamp_density_kgm3 is None
                                                                else clamp_density_kgm3))
        self.support_flag.fill(1)            # conservative until the first refresh_support_flags()
        self.reset_one_sided_stats()
        self._init_excess_mass(dim=3, min_spacing_m=weights.min_spacing_m, cfl_elastic=cfl_elastic,
                               omega_rad_s=excess_mass_omega_rad_s)
        if self.excess_mass is not None and int(self.count_D0[None]):
            self.one_sided_report(reset=False)          # the partners' initial gather met an empty support
        if self.freeze_held_f:
            self._held_f_removed = ti.field(ti.f32, shape=2)                # largest |F - I|, held points
            self.reset_held_deformation()

    # ------------------------------------------------------------------------------------------------ helpers
    @ti.func
    def _in_grid(self, i, j, k):
        return 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz

    @ti.func
    def _record_empty(self, p, c, xp):
        slot = ti.atomic_add(self.count_D0[None], 1)
        if slot < WITNESS_SLOTS:
            self.witness[slot] = ti.Vector([ti.cast(p, ti.f32), ti.cast(c, ti.f32), xp[0], xp[1], xp[2]])

    @ti.func
    def _axis(self, c: ti.template(), xp):
        """Per-axis IB_4 weights of MAC component c at xp."""
        ix, wx, dwx = self.weights.axis_weights(0, 1 if c == 0 else 0, xp[0])
        iy, wy, dwy = self.weights.axis_weights(1, 1 if c == 1 else 0, xp[1])
        iz, wz, dwz = self.weights.axis_weights(2, 1 if c == 2 else 0, xp[2])
        return ix, iy, iz, wx, wy, wz, dwx, dwy, dwz

    @ti.func
    def _interior(self, xp):
        """1 if the host cell of xp is not flagged (its whole support is valid)."""
        hx, wcx, dcx = self.weights.axis_weights(0, 0, xp[0])
        hy, wcy, dcy = self.weights.axis_weights(1, 0, xp[1])
        hz, wcz, dcz = self.weights.axis_weights(2, 0, xp[2])
        i, j, k = hx + 1, hy + 1, hz + 1
        result = 0
        if self._in_grid(i, j, k):
            if self.support_flag[i, j, k] == 0:
                result = 1
        return result

    @ti.func
    def _denominator(self, c: ti.template(), ix, iy, iz, wx, wy, wz, dwx, dwy, dwz):
        D = 0.0
        gD = ti.Vector([0.0, 0.0, 0.0])
        for a in ti.static(range(4)):
            for b, e in ti.ndrange(4, 4):
                i, j, k = ix + a, iy + b, iz + e
                if self._in_grid(i, j, k):
                    m = ti.cast(self.mask.face_valid(c, i, j, k), ti.f32)
                    D += m * wx[a] * wy[b] * wz[e]
                    gD += m * ti.Vector([dwx[a] * wy[b] * wz[e], wx[a] * dwy[b] * wz[e], wx[a] * wy[b] * dwz[e]])
        return D, gD

    # ------------------------------------------------------------------------------------------------ kernels
    @ti.kernel
    def _flag_base_kernel(self):
        """Base flag: the two cells adjacent to every invalid MAC face (backward face (c, i) joins cells i - e_c, i),
        and the outermost cell layer of every axis (supports near the domain edges reach unstored nodes: index < 0 or
        the far face n_a)."""
        for i, j, k in self._flag_a:
            edge = i == 0 or j == 0 or k == 0 or i == self.nx - 1 or j == self.ny - 1 or k == self.nz - 1
            self._flag_a[i, j, k] = ti.u8(1) if edge else ti.u8(0)
        for i, j, k in self._flag_a:
            for c in ti.static(range(3)):
                if self.mask.face_valid(c, i, j, k) == 0:
                    self._flag_a[i, j, k] = ti.u8(1)
                    o = ti.Vector([i, j, k])
                    o[c] -= 1
                    if o[c] >= 0:
                        self._flag_a[o[0], o[1], o[2]] = ti.u8(1)

    @ti.kernel
    def _dilate_kernel(self, src: ti.template(), dst: ti.template(), axis: ti.template()):
        for i, j, k in dst:
            hit = ti.u8(0)
            for s in range(-SUPPORT_REACH, SUPPORT_REACH + 1):
                o = ti.Vector([i, j, k])
                o[axis] += s
                if self._in_grid(o[0], o[1], o[2]):
                    if src[o[0], o[1], o[2]] != 0:
                        hit = ti.u8(1)
            dst[i, j, k] = hit

    @ti.kernel
    def _spread_kernel(self, force: ti.template(), tether_k: ti.f32, load_scale: ti.f32, mu: ti.f32, kappa: ti.f32):
        for p in range(self.n):
            xp = self.points.x[p]
            tau = evaluate_neo_hookean_kirchhoff(self.points.F[p], mu, kappa)
            self.points.tau[p] = tau
            v0 = self.points.vol[p]
            pull = ti.Vector([0.0, 0.0, 0.0])
            if ti.static(not self.direct_clamp):               # the direct clamp replaces the tether
                if self.points.tether[p] != 0:
                    pull = -tether_k * v0 * self.points.u[p]
            point_force = pull + load_scale * self.points.f_ext[p] + self.points.f_extra[p]
            if ti.static(self.excess_mass_enabled):
                point_force += ti.cast(self.excess_mass.spring_force[p], ti.f32)
            self.points.f_extra[p] = ti.Vector([0.0, 0.0, 0.0])
            interior = 0
            if ti.static(self.interior_shortcut):
                interior = self._interior(xp)
            for c in ti.static(range(3)):
                ix, iy, iz, wx, wy, wz, dwx, dwy, dwz = self._axis(c, xp)
                D, gD = 1.0, ti.Vector([0.0, 0.0, 0.0])
                if interior == 0:
                    D, gD = self._denominator(c, ix, iy, iz, wx, wy, wz, dwx, dwy, dwz)
                if D <= 0.0:
                    self._record_empty(p, c, xp)
                else:
                    inv_D = 1.0 / D
                    if ti.static(self.unroll_static):
                        for a, b, e in ti.static(ti.ndrange(4, 4, 4)):
                            self._spread_node(force, c, interior, ix + a, iy + b, iz + e, wx[a], wy[b], wz[e],
                                              dwx[a], dwy[b], dwz[e], inv_D, gD, tau, v0, point_force)
                    else:
                        for a in ti.static(range(4)):
                            for b, e in ti.ndrange(4, 4):
                                self._spread_node(force, c, interior, ix + a, iy + b, iz + e, wx[a], wy[b], wz[e],
                                                  dwx[a], dwy[b], dwz[e], inv_D, gD, tau, v0, point_force)

    @ti.func
    def _spread_node(self, force: ti.template(), c: ti.template(), interior, i, j, k, wa, wb, we, da, db, de, inv_D,
                     gD, tau, v0, point_force):
        if self._in_grid(i, j, k):
            m = 1.0
            if interior == 0:
                m = ti.cast(self.mask.face_valid(c, i, j, k), ti.f32)
            if m > 0.0:
                w = wa * wb * we * inv_D
                if ti.static(self.excess_mass_enabled):
                    w = penalty_node_weight_3d(wa, wb, we, inv_D)
                g = (ti.Vector([da * wb * we, wa * db * we, wa * wb * de]) - w * gD) * inv_D
                f = -v0 * (tau[c, 0] * g[0] + tau[c, 1] * g[1] + tau[c, 2] * g[2]) + point_force[c] * w
                ti.atomic_add(force[i, j, k][c], f / self.weights.face_volume(c, i, j, k))

    @ti.func
    def _gather(self, velocity: ti.template(), velocity_b: ti.template(), blend: ti.f32, p, xp):
        """One-sided interpolated velocity and gradient of u = (1 - blend) velocity + blend velocity_b."""
        v = ti.Vector([0.0, 0.0, 0.0])
        G = ti.Matrix.zero(ti.f32, 3, 3)
        interior = 0
        if ti.static(self.interior_shortcut):
            interior = self._interior(xp)
        for c in ti.static(range(3)):
            ix, iy, iz, wx, wy, wz, dwx, dwy, dwz = self._axis(c, xp)
            D, gD = 1.0, ti.Vector([0.0, 0.0, 0.0])
            if interior == 0:
                D, gD = self._denominator(c, ix, iy, iz, wx, wy, wz, dwx, dwy, dwz)
            if D <= 0.0:
                self._record_empty(p, c, xp)
            else:
                inv_D = 1.0 / D
                vc = 0.0
                gc = ti.Vector([0.0, 0.0, 0.0])
                if ti.static(self.unroll_static):
                    for a, b, e in ti.static(ti.ndrange(4, 4, 4)):
                        dv, dg = self._gather_node(velocity, velocity_b, blend, c, interior, ix + a, iy + b, iz + e,
                                                   wx[a], wy[b], wz[e], dwx[a], dwy[b], dwz[e], inv_D, gD)
                        vc += dv
                        gc += dg
                else:
                    for a in ti.static(range(4)):
                        for b, e in ti.ndrange(4, 4):
                            dv, dg = self._gather_node(velocity, velocity_b, blend, c, interior, ix + a, iy + b,
                                                       iz + e, wx[a], wy[b], wz[e], dwx[a], dwy[b], dwz[e], inv_D, gD)
                            vc += dv
                            gc += dg
                v[c] = vc
                G[c, 0] = gc[0]
                G[c, 1] = gc[1]
                G[c, 2] = gc[2]
        return v, G

    @ti.func
    def _gather_node(self, velocity: ti.template(), velocity_b: ti.template(), blend, c: ti.template(), interior, i, j,
                     k, wa, wb, we, da, db, de, inv_D, gD):
        dv = 0.0
        dg = ti.Vector([0.0, 0.0, 0.0])
        if self._in_grid(i, j, k):
            m = 1.0
            if interior == 0:
                m = ti.cast(self.mask.face_valid(c, i, j, k), ti.f32)
            if m > 0.0:
                w = wa * wb * we * inv_D
                if ti.static(self.excess_mass_enabled):
                    w = penalty_node_weight_3d(wa, wb, we, inv_D)
                g = (ti.Vector([da * wb * we, wa * db * we, wa * wb * de]) - w * gD) * inv_D
                uval = (1.0 - blend) * velocity[i, j, k][c] + blend * velocity_b[i, j, k][c]
                dv = uval * w
                dg = uval * g
        return dv, dg

    @ti.kernel
    def _interpolate_update_kernel(self, velocity: ti.template(), dt: ti.f32):
        for p in range(self.n):
            xq = self.points.x[p]
            v, G = self._gather(velocity, velocity, 0.0, p, xq)
            self.points.v[p] = v
            self.points.grad_v[p] = G
            self.points.u[p] += dt * v
            self.points.x[p] = self.points.X0[p] + self.points.u[p]
            if ti.static(self.freeze_held_f):                 # a held point's F stays I (module docstring)
                if self.points.tether[p] == 0:
                    self.points.F[p] = (ti.Matrix.identity(ti.f32, 3) + dt * G) @ self.points.F[p]
            else:
                self.points.F[p] = (ti.Matrix.identity(ti.f32, 3) + dt * G) @ self.points.F[p]

    @ti.kernel
    def _midpoint_correct_kernel(self, velocity_old: ti.template(), velocity_new: ti.template(), dt: ti.f32):
        """IBAMR MIDPOINT_RULE: X^{n+1} = X^n + dt U^{n+1/2}(X^{n+1/2}), U^{n+1/2} = (u^n + u^{n+1}) / 2; the same
        for F with the interpolated gradient.  x currently holds X^{n+1/2}, F the predicted F^{n+1/2}."""
        for p in range(self.n):
            xq = self.points.x[p]
            v, G = self._gather(velocity_old, velocity_new, 0.5, p, xq)
            self.points.v[p] = v
            self.points.grad_v[p] = G
            self.points.u[p] = self.points.u_n[p] + dt * v
            self.points.x[p] = self.points.X0[p] + self.points.u[p]
            if ti.static(self.freeze_held_f):
                if self.points.tether[p] == 0:
                    self.points.F[p] = self.points.F_n[p] + dt * (G @ self.points.F[p])
            else:
                self.points.F[p] = self.points.F_n[p] + dt * (G @ self.points.F[p])

    @ti.kernel
    def _reset_held_F_kernel(self):
        self._held_f_removed[0] = 0.0
        self._held_f_removed[1] = 0.0
        for p in range(self.n):
            if self.points.tether[p] != 0:
                ti.atomic_max(self._held_f_removed[0], (self.points.F[p] - ti.Matrix.identity(ti.f32, 3)).norm())
                ti.atomic_add(self._held_f_removed[1], 1.0)
                self.points.F[p] = ti.Matrix.identity(ti.f32, 3)

    @ti.kernel
    def _min_D_kernel(self):
        """min over points and components of the one-sided denominator D at the current positions (diagnostics pass;
        kept out of the spread / gather hot loops)."""
        self.min_D[None] = 1.0e30
        self.min_D_free[None] = 1.0e30
        self.min_D_contact_rim_free[None] = 1.0e30
        self.min_D_chamber_value[None] = 1.0e30
        for p in range(self.n):
            local = 1.0e30
            xq = self.points.x[p]
            for c in ti.static(range(3)):
                ix, iy, iz, wx, wy, wz, dwx, dwy, dwz = self._axis(c, xq)
                D, gD = self._denominator(c, ix, iy, iz, wx, wy, wz, dwx, dwy, dwz)
                local = ti.min(local, D)
            ti.atomic_min(self.min_D[None], local)
            if (self.points.tag[p] & TAG_CHAMBER_SURFACE) != 0:
                ti.atomic_min(self.min_D_chamber_value[None], local)
            if self.points.tether[p] == 0:
                ti.atomic_min(self.min_D_free[None], local)
                if (self.points.tag[p] & TAG_CONTACT_RIM) != 0:
                    ti.atomic_min(self.min_D_contact_rim_free[None], local)

    @ti.kernel
    def _min_D_locate_kernel(self, free_only: ti.i32):
        """Index of one point whose smallest component denominator equals min D (free points only if free_only;
        diagnostics only)."""
        self.min_D_point[None] = -1
        target = self.min_D[None]
        if free_only != 0:
            target = self.min_D_free[None]
        for p in range(self.n):
            if free_only != 0 and self.points.tether[p] != 0:
                continue
            local = 1.0e30
            xq = self.points.x[p]
            for c in ti.static(range(3)):
                ix, iy, iz, wx, wy, wz, dwx, dwy, dwz = self._axis(c, xq)
                D, gD = self._denominator(c, ix, iy, iz, wx, wy, wz, dwx, dwy, dwz)
                local = ti.min(local, D)
            if local <= target:
                self.min_D_point[None] = p

    @ti.kernel
    def _reaction_kernel(self):
        """Force of the clamp on the tethered points, summed (the penalty pull, or the direct-forcing reaction), and the
        largest interpolated speed at the clamped points (direct clamp)."""
        for s in range(4):
            self.reaction[s] = 0.0
        for p in range(self.n):
            if self.points.tether[p] != 0:
                pull = -self.tether_k * self.points.vol[p] * self.points.u[p]
                if ti.static(self.direct_clamp):               # the direct-forcing reaction
                    pull = self.clamp.point_force[p]
                    ti.atomic_max(self.reaction[3], ti.cast(self.clamp.interp_velocity[p].norm(), ti.f64))
                for q in ti.static(range(3)):
                    ti.atomic_add(self.reaction[q], ti.cast(pull[q], ti.f64))

    @ti.kernel
    def _clear_force_kernel(self, force: ti.template()):
        for i, j, k in force:
            force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

    # ------------------------------------------------------------------------------------------------ host API
    def refresh_support_flags(self):
        """Mark the cells whose 4-node support may touch an invalid face (call after every mask refresh)."""
        self.geometry_epoch += 1
        if not self.interior_shortcut:
            return
        self._flag_base_kernel()
        self._dilate_kernel(self._flag_a, self._flag_b, 0)
        self._dilate_kernel(self._flag_b, self._flag_a, 1)
        self._dilate_kernel(self._flag_a, self.support_flag, 2)

    def reset_one_sided_stats(self):
        self.min_D[None] = 1.0e30
        self.count_D0[None] = 0

    def one_sided_report(self, reset=True):
        """Empty-support witnesses recorded by the hot kernels since the last reset (raises the hard stop on any D <= 0)
        and min D over all points and components at the current positions (separate diagnostics pass)."""
        count = int(self.count_D0[None])
        self._min_D_kernel()
        min_d = float(self.min_D[None])
        if count:
            rows = self.witness.to_numpy()[:min(count, WITNESS_SLOTS)]
            witness = [dict(point=int(r[0]), component=int(r[1]), position_m=[float(r[2]), float(r[3]), float(r[4])])
                       for r in rows]
            raise SimulationHardStop("one_sided_empty_support", dict(count=count, witnesses=witness))
        if reset:
            self.count_D0[None] = 0
        min_free = float(self.min_D_free[None])
        rim = float(self.min_D_contact_rim_free[None])
        return dict(min_D=min_d, min_D_free=min_free, min_D_below_0p25=bool(min_d < 0.25),
                    min_D_free_below_0p25=bool(min_free < 0.25),
                    min_D_contact_rim_free=rim if rim < 1.0e29 else float('nan'))

    def min_D_witness(self, free_only=False):
        """Where min D occurs (over free points only if free_only): point index, position, body, tether, tag
        (diagnostics; two passes)."""
        self._min_D_kernel()
        self._min_D_locate_kernel(1 if free_only else 0)
        p = int(self.min_D_point[None])
        if p < 0:
            return dict(min_D=float(self.min_D[None]), point=None)
        points = self.points
        x = points.X0.to_numpy()[p].astype(np.float64) + points.u.to_numpy()[p].astype(np.float64)
        return dict(min_D=float(self.min_D[None]), min_D_free=float(self.min_D_free[None]), point=p,
                    position_m=x.tolist(), body_id=int(points.body_id.to_numpy()[p]),
                    tether=int(points.tether.to_numpy()[p]), tag=int(points.tag.to_numpy()[p]))

    def min_D_chamber_surface(self):
        """min one-sided D over the TAG_CHAMBER_SURFACE points at the current positions (inf if there are none)."""
        self._min_D_kernel()
        value = float(self.min_D_chamber_value[None])
        return value if value < 1.0e29 else float("inf")

    def spread_force(self, load_scale=1.0, clear=True, dt_s=None):
        """One-sided IB force density of stress + tether + scaled external load + extra point force into fluid.force
        (f_extra is consumed and zeroed).  clear=False when the caller already zeroed fluid.force in a fused pass.
        With clamp="direct" the clamp forces of this step (dt_s required) are computed from the current velocity
        and the force density just spread, then spread too; fluid.force must hold only this step's forces."""
        if self.excess_mass is not None:
            self.excess_mass.prepare_force(dt_s)
        if clear:
            self._clear_force_kernel(self.fluid.force)
        self._spread_kernel(self.fluid.force, self.tether_k, float(load_scale), self.points.mu, self.points.kappa)
        if self.clamp is not None:
            self.clamp.apply(dt_s)

    spread_elastic_force = spread_force

    def overlap_bound_key(self):
        """What the excess-mass overlap bound depends on: point positions and support flags (geometry_epoch) and the
        face mask (its version; None for a mask without one, so the bound is evaluated at every use)."""
        version = getattr(self.mask, "version", None)
        return None if version is None else (self.geometry_epoch, version)

    def _after_load_state(self):
        self.invalidate_overlap_bound()               # a restart may also have loaded new point positions
        if self.freeze_held_f:                        # the points component loads before the coupler (its own F)
            self.reset_held_deformation()

    def reset_held_deformation(self):
        """held_f="frozen" only: every held point's F to I.  Returns, and keeps as held_f_reset, the largest
        |F - I| (Frobenius) removed and the number of held points."""
        if not self.freeze_held_f:
            raise ValueError("reset_held_deformation needs held_f='frozen'")
        self._reset_held_F_kernel()
        removed = self._held_f_removed.to_numpy()
        self.held_f_reset = dict(max_abs_f_minus_i=float(removed[0]), held_points=int(removed[1]))
        return self.held_f_reset

    def clamp_state(self):
        """Interpolated velocities and direct-forcing forces of the clamped points at the last spread (None for the
        penalty clamp)."""
        return None if self.clamp is None else self.clamp.state()

    def diagnostics(self, reference_plane_z_m=0.0):
        """The solid diagnostics of the points (MaterialPoints.diagnostics) with the clamp force on the tethered points
        (tether_pull_n) and, for the direct clamp, the largest interpolated speed at the clamped points."""
        solid = self.points.diagnostics(reference_plane_z_m)
        self._reaction_kernel()
        r = self.reaction.to_numpy()
        out = {}
        for key, value in solid.items():
            out[key] = value
            if key == "tether_max_displacement_m":
                out["tether_pull_n"] = [float(r[0]), float(r[1]), float(r[2])]
        if self.direct_clamp:                                  # how well the clamp holds the fluid
            out['clamp_max_interp_speed_mps'] = float(r[3])
        return {**out, **self._excess_mass_diagnostics(solid["momentum"])}

    def elastic_dt_limit(self, cfl):
        """CFL on the shear wave over the smallest cell edge of the grid (conservative for graded grids)."""
        return float(cfl) * self.weights.min_spacing_m / math.sqrt(self.points.mu / self.rho_f)

    def fingerprint(self):
        """Static setup arrays that a checkpoint must match (reference positions, volumes, tethers, ids, tags, loads)
        and the coupling parameters."""
        points = self.points
        digest = hashlib.sha256()
        for field in (points.X0, points.vol, points.tether, points.body_id, points.tag, points.f_ext,
                      points.near_surface):
            digest.update(np.ascontiguousarray(field.to_numpy()).tobytes())
        digest.update(repr((self.n, tuple(self.weights.n), points.rho_s, points.mu, points.kappa,
                            self.tether_k, self.rho_f, self.excess_mass_mode,
                            None if self.excess_mass is None else self.excess_mass.fingerprint())).encode())
        return digest.hexdigest()


@ti.data_oriented
class PlanarImmersedBoundaryCoupler(ExcessMassCouplerMixin):
    """Couples PlanarMaterialPoints with a quasi-2-D fluid (fields uniform along x) on a uniform grid."""

    def __init__(self, *, fluid, points, tether_stiffness_pa_per_m, grid_origin_m, grid_spacing_m, grid_nodes, span_m,
                 ib_kernel="ib4", clamp="penalty", clamp_density_kgm3=None, fluid_density_kgm3=None,
                 excess_mass_mode="auto", excess_mass_omega_rad_s=None, cfl_elastic=.3):
        self.ib_kernel = validate_kernel_name(ib_kernel)       # compile-time kernel choice
        self.ib3 = self.ib_kernel == "ib3"
        self.fluid = fluid
        self.points = points
        self.n = points.n
        self._init_densities(points, fluid, fluid_density_kgm3, excess_mass_mode)
        self.nx, self.ny, self.nz = (int(v) for v in grid_nodes)
        self.origin = tuple(float(v) for v in grid_origin_m)
        self.h = tuple(float(v) for v in grid_spacing_m)
        self.span_m = float(span_m)
        self.tether_k = float(tether_stiffness_pa_per_m)
        self.reaction = ti.field(ti.f64, shape=3)              # tether pull y, z; clamp max speed
        self.clamp_mode = _validated_clamp(clamp)
        self.direct_clamp = clamp == "direct"
        self.clamp = None
        if self.direct_clamp:
            self.clamp = PlanarDirectForcingClamp(self, density_kgm3=(self.rho_f if clamp_density_kgm3 is None
                                                                      else clamp_density_kgm3))
        self._init_excess_mass(dim=2, min_spacing_m=min(self.h[1], self.h[2]), cfl_elastic=cfl_elastic,
                               omega_rad_s=excess_mass_omega_rad_s)

    # ------------------------------------------------------------------------------------------------ kernels
    @ti.func
    def _face_position(self, c, j, k):
        """(y, z) of the backward MAC face of cell (., j, k) storing component c (1 = y, 2 = z)."""
        y = self.origin[1] + (j + 0.5) * self.h[1]
        z = self.origin[2] + (k + 0.5) * self.h[2]
        if c == 1:
            y = self.origin[1] + j * self.h[1]
        if c == 2:
            z = self.origin[2] + k * self.h[2]
        return y, z

    @ti.func
    def _gather_face_vector(self, field: ti.template(), yp, zp):
        """Interpolated (y, z) components of a face field (velocity or force density) at (yp, zp): the interpolation
        statements of the update kernels without the gradient."""
        hy, hz = self.h[1], self.h[2]
        v = ti.Vector([0.0, 0.0])
        for c in ti.static((1, 2)):
            sy = (yp - self.origin[1]) / hy - (0.0 if c == 1 else 0.5)
            sz = (zp - self.origin[2]) / hz - (0.0 if c == 2 else 0.5)
            bj = ti.cast(ti.floor(sy), ti.i32) - 1
            bk = ti.cast(ti.floor(sz), ti.i32) - 1
            for a, b in ti.static(ti.ndrange(4, 4)):
                j, k = bj + a, bk + b
                if 0 <= j < self.ny and 0 <= k < self.nz:
                    ry = sy - j
                    rz = sz - k
                    w = 0.0
                    if ti.static(self.ib3):
                        w = evaluate_ib3_weight(ry) * evaluate_ib3_weight(rz)
                    else:
                        w = evaluate_ib4_weight(ry) * evaluate_ib4_weight(rz)
                    v[c - 1] += field[0, j, k][c] * w
        return v

    @ti.kernel
    def _stress_kernel(self, mu: ti.f32, kappa: ti.f32):
        for p in range(self.n):
            self.points.tau[p] = evaluate_neo_hookean_kirchhoff_planar(self.points.F[p], mu, kappa)

    @ti.kernel
    def _clear_force_kernel(self, force: ti.template()):
        for i, j, k in force:
            force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _spread_kernel(self, force: ti.template(), tether_k: ti.f32):
        hy, hz = self.h[1], self.h[2]
        face_area = hy * hz
        for p in range(self.n):
            yp, zp = self.points.x[p][0], self.points.x[p][1]
            tau = self.points.tau[p]
            a0 = self.points.area[p]
            pull = ti.Vector([0.0, 0.0])
            if ti.static(not self.direct_clamp):               # the direct clamp replaces the tether
                if self.points.tether[p] != 0:
                    pull = tether_k * a0 * (self.points.X0[p] - self.points.x[p])
            if ti.static(self.excess_mass_enabled):
                pull += ti.cast(self.excess_mass.spring_force[p], ti.f32)
            for c in ti.static((1, 2)):
                # face-grid index space of component c
                sy = (yp - self.origin[1]) / hy - (0.0 if c == 1 else 0.5)
                sz = (zp - self.origin[2]) / hz - (0.0 if c == 2 else 0.5)
                bj = ti.cast(ti.floor(sy), ti.i32) - 1
                bk = ti.cast(ti.floor(sz), ti.i32) - 1
                for a, b in ti.static(ti.ndrange(4, 4)):
                    j, k = bj + a, bk + b
                    if 0 <= j < self.ny and 0 <= k < self.nz:
                        ry = sy - j
                        rz = sz - k
                        if ti.static(self.ib3):                # IB_3 option, same 4-node frame
                            wy, wz = evaluate_ib3_weight(ry), evaluate_ib3_weight(rz)
                            gy = evaluate_ib3_weight_derivative(ry) * wz / hy
                            gz = wy * evaluate_ib3_weight_derivative(rz) / hz
                            row = c - 1
                            f = -a0 * (tau[row, 0] * gy + tau[row, 1] * gz) + pull[row] * wy * wz
                            density = f / face_area
                            for i in range(self.nx):
                                ti.atomic_add(force[i, j, k][c], density)
                        else:                                  # IB_4 (default)
                            wy, wz = evaluate_ib4_weight(ry), evaluate_ib4_weight(rz)
                            # grad of the node weight w_i(x_p) with respect to x_p
                            gy = evaluate_ib4_weight_derivative(ry) * wz / hy
                            gz = wy * evaluate_ib4_weight_derivative(rz) / hz
                            row = c - 1
                            f = -a0 * (tau[row, 0] * gy + tau[row, 1] * gz) + pull[row] * wy * wz
                            density = f / face_area
                            for i in range(self.nx):
                                ti.atomic_add(force[i, j, k][c], density)

    @ti.kernel
    def _interpolate_update_kernel(self, velocity: ti.template(), dt: ti.f32):
        hy, hz = self.h[1], self.h[2]
        for p in range(self.n):
            yp, zp = self.points.x[p][0], self.points.x[p][1]
            v = ti.Vector([0.0, 0.0])
            G = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            for c in ti.static((1, 2)):
                sy = (yp - self.origin[1]) / hy - (0.0 if c == 1 else 0.5)
                sz = (zp - self.origin[2]) / hz - (0.0 if c == 2 else 0.5)
                bj = ti.cast(ti.floor(sy), ti.i32) - 1
                bk = ti.cast(ti.floor(sz), ti.i32) - 1
                for a, b in ti.static(ti.ndrange(4, 4)):
                    j, k = bj + a, bk + b
                    if 0 <= j < self.ny and 0 <= k < self.nz:
                        ry = sy - j
                        rz = sz - k
                        if ti.static(self.ib3):                # IB_3 option
                            wy, wz = evaluate_ib3_weight(ry), evaluate_ib3_weight(rz)
                            u = velocity[0, j, k][c]
                            row = c - 1
                            v[row] += u * wy * wz
                            G[row, 0] += u * evaluate_ib3_weight_derivative(ry) * wz / hy
                            G[row, 1] += u * wy * evaluate_ib3_weight_derivative(rz) / hz
                        else:                                  # IB_4 (default)
                            wy, wz = evaluate_ib4_weight(ry), evaluate_ib4_weight(rz)
                            u = velocity[0, j, k][c]
                            row = c - 1
                            v[row] += u * wy * wz
                            G[row, 0] += u * evaluate_ib4_weight_derivative(ry) * wz / hy
                            G[row, 1] += u * wy * evaluate_ib4_weight_derivative(rz) / hz
            self.points.v[p] = v
            if ti.static(self.excess_mass_enabled):
                self.excess_mass.material_displacement[p] += ti.cast(dt, ti.f64) * ti.cast(v, ti.f64)
                self.points.x[p] = ti.cast(ti.cast(self.points.X0[p], ti.f64) + self.excess_mass.material_displacement[p], ti.f32)
            else:
                self.points.x[p] += dt * v
            self.points.F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * G) @ self.points.F[p]

    @ti.kernel
    def _midpoint_correct_kernel(self, velocity_old: ti.template(), velocity_new: ti.template(), dt: ti.f32):
        """IBAMR MIDPOINT_RULE: X^{n+1} = X^n + dt U^{n+1/2}(X^{n+1/2}), U^{n+1/2} = (u^n + u^{n+1}) / 2; the same
        for F with the interpolated gradient.  x currently holds X^{n+1/2}."""
        hy, hz = self.h[1], self.h[2]
        for p in range(self.n):
            yp, zp = self.points.x[p][0], self.points.x[p][1]
            v = ti.Vector([0.0, 0.0])
            G = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            for c in ti.static((1, 2)):
                sy = (yp - self.origin[1]) / hy - (0.0 if c == 1 else 0.5)
                sz = (zp - self.origin[2]) / hz - (0.0 if c == 2 else 0.5)
                bj = ti.cast(ti.floor(sy), ti.i32) - 1
                bk = ti.cast(ti.floor(sz), ti.i32) - 1
                for a, b in ti.static(ti.ndrange(4, 4)):
                    j, k = bj + a, bk + b
                    if 0 <= j < self.ny and 0 <= k < self.nz:
                        ry = sy - j
                        rz = sz - k
                        if ti.static(self.ib3):                # IB_3 option
                            wy, wz = evaluate_ib3_weight(ry), evaluate_ib3_weight(rz)
                            u = 0.5 * (velocity_old[0, j, k][c] + velocity_new[0, j, k][c])
                            row = c - 1
                            v[row] += u * wy * wz
                            G[row, 0] += u * evaluate_ib3_weight_derivative(ry) * wz / hy
                            G[row, 1] += u * wy * evaluate_ib3_weight_derivative(rz) / hz
                        else:                                  # IB_4 (default)
                            wy, wz = evaluate_ib4_weight(ry), evaluate_ib4_weight(rz)
                            u = 0.5 * (velocity_old[0, j, k][c] + velocity_new[0, j, k][c])
                            row = c - 1
                            v[row] += u * wy * wz
                            G[row, 0] += u * evaluate_ib4_weight_derivative(ry) * wz / hy
                            G[row, 1] += u * wy * evaluate_ib4_weight_derivative(rz) / hz
            self.points.v[p] = v
            # F currently holds the predicted F^{n+1/2}
            if ti.static(self.excess_mass_enabled):    # the f64 material displacement from its step-start value
                material = self.excess_mass.material_displacement_n[p] + ti.cast(dt, ti.f64) * ti.cast(v, ti.f64)
                self.excess_mass.material_displacement[p] = material
                self.points.x[p] = ti.cast(ti.cast(self.points.X0[p], ti.f64) + material, ti.f32)
            else:
                self.points.x[p] = self.points.x_n[p] + dt * v
            self.points.F[p] = self.points.F_n[p] + dt * (G @ self.points.F[p])

    @ti.kernel
    def _reaction_kernel(self):
        """Force of the clamp on the tethered points per unit span, summed (the penalty pull, or the direct-forcing
        reaction), and the largest interpolated speed at the clamped points (direct clamp)."""
        for s in range(3):
            self.reaction[s] = 0.0
        for p in range(self.n):
            if self.points.tether[p] != 0:
                pull = self.tether_k * self.points.area[p] * (self.points.X0[p] - self.points.x[p])
                if ti.static(self.direct_clamp):               # the direct-forcing reaction
                    pull = self.clamp.point_force[p]
                    ti.atomic_max(self.reaction[2], ti.cast(self.clamp.interp_velocity[p].norm(), ti.f64))
                ti.atomic_add(self.reaction[0], ti.cast(pull[0], ti.f64))
                ti.atomic_add(self.reaction[1], ti.cast(pull[1], ti.f64))

    # ------------------------------------------------------------------------------------------------ host API
    def spread_elastic_force(self, dt_s=None):
        """Stress + tether spread into fluid.force; with clamp="direct" then the clamp forces of this step (dt_s
        required) from the current velocity and the force density just spread."""
        if self.excess_mass is not None:
            self.excess_mass.prepare_force(dt_s)
        self._stress_kernel(self.points.mu, self.points.kappa)
        self._clear_force_kernel(self.fluid.force)
        self._spread_kernel(self.fluid.force, self.tether_k)
        if self.clamp is not None:
            self.clamp.apply(dt_s)

    def overlap_bound_key(self):
        """What the excess-mass overlap bound depends on: the point positions (geometry_epoch); no face mask."""
        return (self.geometry_epoch,)

    def fingerprint(self):
        """Static set-up arrays that a checkpoint must match (reference positions, areas, tethers, near-surface marks)
        and the coupling parameters."""
        points = self.points
        digest = hashlib.sha256()
        for field in (points.X0, points.area, points.tether, points.near_surface):
            digest.update(np.ascontiguousarray(field.to_numpy()).tobytes())
        digest.update(repr((self.n, (self.nx, self.ny, self.nz), self.origin, self.h, self.span_m, points.rho_s,
                            points.mu, points.kappa, self.tether_k, self.rho_f, self.excess_mass_mode, self.ib_kernel,
                            self.clamp_mode,
                            None if self.excess_mass is None else self.excess_mass.fingerprint())).encode())
        return digest.hexdigest()

    def diagnostics(self):
        """The solid diagnostics of the points (PlanarMaterialPoints.diagnostics) with the clamp force on the tethered
        points (tether_pull_y / tether_pull_z, per unit span) and, for the direct clamp, the largest interpolated speed
        at the clamped points."""
        solid = self.points.diagnostics()
        self._reaction_kernel()
        r = self.reaction.to_numpy()
        out = {}
        for key, value in solid.items():
            out[key] = value
            if key == "tether_max_displacement_m":
                out["tether_pull_y"] = float(r[0])
                out["tether_pull_z"] = float(r[1])
        if self.direct_clamp:                                  # how well the clamp holds the fluid
            out['clamp_max_interp_speed_mps'] = float(r[2])
        return {**out, **self._excess_mass_diagnostics((solid["momentum_y"], solid["momentum_z"]))}

    def elastic_dt_limit(self, cfl):
        """CFL on the shear wave over the smallest cell where the solid lives (uniform grid: min(hy, hz))."""
        return float(cfl) * min(self.h[1], self.h[2]) / math.sqrt(self.points.mu / self.rho_f)

    def clamp_state(self):
        """Interpolated velocities and direct-forcing forces of the clamped points at the last spread (None for the
        penalty clamp)."""
        return None if self.clamp is None else self.clamp.state()
