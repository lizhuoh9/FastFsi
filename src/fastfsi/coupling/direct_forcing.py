"""Rigid clamp of the tethered material points by direct forcing (multi-direct forcing).

After the other forces of a step are spread, each tethered point p gets the point force

    F_p = rho V_p (-u_p / dt - U~_p) / dt,     U~_p = U_p + dt A_p,

with U_p the interpolated fluid velocity, A_p the interpolated force density divided by rho and u_p the point's
displacement from its anchor (so the clamp also pulls a drifted point back).  F_p is spread with the same one-sided
weights as every immersed-boundary force; ITERATIONS passes re-interpolate and correct the increment, which converges
the clamp velocity.  rho is the fluid density (the force acts on the fluid in the clamp's support).

DirectForcingClamp works on an ImmersedBoundaryCoupler (3-D: one-sided normalised weights on graded axes),
PlanarDirectForcingClamp on a PlanarImmersedBoundaryCoupler (uniform quasi-2-D channel; V_p is the area per unit span).
The coupler creates its clamp (clamp="direct") and calls apply(dt) at the end of its spread; the kernels use the
coupler's own interpolation and spreading functions, so the clamp forces see exactly the coupler's weights.

References (algorithms; no source copied): direct forcing with a regularised delta function, M. Uhlmann,
J. Comput. Phys. 209 (2005) 448-476; its iterated form (multi-direct forcing), K. Luo, Z. Wang, J. Fan, K. Cen,
Phys. Rev. E 76 (2007) 066709.

No `from __future__ import annotations` (Taichi kernel module).
"""
import math

import numpy as np
import taichi as ti

from fastfsi.coupling.kernels import evaluate_ib3_weight, evaluate_ib4_weight

ITERATIONS = 5             # multi-direct forcing passes (each re-interpolates and corrects the increment)


def _validated_density(density_kgm3):
    rho = float(density_kgm3)
    if not (math.isfinite(rho) and rho > 0.0):
        raise ValueError("the direct clamp needs a positive finite density")
    return rho


def _validated_dt(dt_s):
    if dt_s is None or not (math.isfinite(float(dt_s)) and float(dt_s) > 0.0):
        raise ValueError("the direct clamp needs the step dt_s > 0")
    return float(dt_s)


@ti.data_oriented
class DirectForcingClamp:
    """Direct-forcing clamp of the tethered points of an ImmersedBoundaryCoupler (3-D)."""

    def __init__(self, coupler, *, density_kgm3, iterations=ITERATIONS):
        self.coupler = coupler
        self.points = coupler.points
        self.n = self.points.n
        self.rho = _validated_density(density_kgm3)
        self.iterations = int(iterations)
        n = self.n
        self.interp_velocity = ti.Vector.field(3, ti.f32, shape=n)   # interpolated velocity at the clamped points
        self.point_force = ti.Vector.field(3, ti.f32, shape=n)       # their direct-forcing point force (N), last spread
        self.increment = ti.Vector.field(3, ti.f32, shape=n)         # the increment of the current forcing pass
        self.interp_velocity.fill(0.0)
        self.point_force.fill(0.0)

    @ti.kernel
    def _force_kernel(self, velocity: ti.template(), force: ti.template(), inv_dt: ti.f32, rho: ti.f32,
                      first: ti.i32):
        """F_p = rho V_p (-u_p / dt - (U_p + dt A_p)) / dt for every clamped point, from the
        current velocity and the force density already spread this step (gathers only; spread separately)."""
        for p in range(self.n):
            if self.points.tether[p] != 0:
                xq = self.points.x[p]
                v, G = self.coupler._gather(velocity, velocity, 0.0, p, xq)
                a, Ga = self.coupler._gather(force, force, 0.0, p, xq)
                u_tilde = v + a / (rho * inv_dt)
                target = -self.points.u[p] * inv_dt
                df = rho * self.points.vol[p] * (target - u_tilde) * inv_dt   # one pass; A_p includes earlier passes
                self.increment[p] = df
                if first != 0:
                    self.interp_velocity[p] = v
                    self.point_force[p] = df
                else:
                    self.point_force[p] += df

    @ti.kernel
    def _spread_kernel(self, force: ti.template()):
        """spread the clamp forces with the same normalised weights as every IB force."""
        zero = ti.Matrix.zero(ti.f32, 3, 3)
        for p in range(self.n):
            if self.points.tether[p] != 0:
                xp = self.points.x[p]
                fp = self.increment[p]
                interior = 0
                if ti.static(self.coupler.interior_shortcut):
                    interior = self.coupler._interior(xp)
                for c in ti.static(range(3)):
                    ix, iy, iz, wx, wy, wz, dwx, dwy, dwz = self.coupler._axis(c, xp)
                    D, gD = 1.0, ti.Vector([0.0, 0.0, 0.0])
                    if interior == 0:
                        D, gD = self.coupler._denominator(c, ix, iy, iz, wx, wy, wz, dwx, dwy, dwz)
                    if D > 0.0:                                  # D <= 0 is recorded by the main spread
                        inv_D = 1.0 / D
                        for a in ti.static(range(4)):
                            for b, e in ti.ndrange(4, 4):
                                self.coupler._spread_node(force, c, interior, ix + a, iy + b, iz + e, wx[a], wy[b],
                                                          wz[e], dwx[a], dwy[b], dwz[e], inv_D, gD, zero, 0.0, fp)

    def apply(self, dt_s):
        """The clamp passes of this step into the coupler's fluid force (call after the other forces are spread)."""
        inv_dt = 1.0 / _validated_dt(dt_s)
        fluid = self.coupler.fluid
        for k in range(self.iterations):                   # multi-direct forcing: converge the clamp velocity
            self._force_kernel(fluid.velocity, fluid.force, inv_dt, self.rho, 1 if k == 0 else 0)
            self._spread_kernel(fluid.force)

    def state(self):
        """Interpolated velocities and direct-forcing forces of the clamped points at the last spread."""
        mask = self.points.tether.to_numpy() != 0
        return dict(index=np.flatnonzero(mask), u=self.interp_velocity.to_numpy()[mask],
                    f=self.point_force.to_numpy()[mask])


@ti.data_oriented
class PlanarDirectForcingClamp:
    """Direct-forcing clamp of the tethered points of a PlanarImmersedBoundaryCoupler (forces per unit span)."""

    def __init__(self, coupler, *, density_kgm3, iterations=ITERATIONS):
        self.coupler = coupler
        self.points = coupler.points
        self.n = self.points.n
        self.rho = _validated_density(density_kgm3)
        self.iterations = int(iterations)
        n = self.n
        self.interp_velocity = ti.Vector.field(2, ti.f32, shape=n)   # interpolated (y, z) velocity at the clamped points
        self.point_force = ti.Vector.field(2, ti.f32, shape=n)       # their direct-forcing force per unit span (N/m)
        self.increment = ti.Vector.field(2, ti.f32, shape=n)         # the increment of the current forcing pass
        self.interp_velocity.fill(0.0)
        self.point_force.fill(0.0)

    @ti.kernel
    def _force_kernel(self, velocity: ti.template(), force: ti.template(), inv_dt: ti.f32, rho: ti.f32,
                      first: ti.i32):
        """F_p = rho A_p (-(x_p - X0_p) / dt - (U_p + dt A_p)) / dt per unit span from
        the current velocity and the force density already spread this step (gathers only)."""
        for p in range(self.n):
            if self.points.tether[p] != 0:
                yp, zp = self.points.x[p][0], self.points.x[p][1]
                v = self.coupler._gather_face_vector(velocity, yp, zp)
                a = self.coupler._gather_face_vector(force, yp, zp)
                u_tilde = v + a / (rho * inv_dt)
                target = (self.points.X0[p] - self.points.x[p]) * inv_dt
                df = rho * self.points.area[p] * (target - u_tilde) * inv_dt  # one pass; A_p includes earlier passes
                self.increment[p] = df
                if first != 0:
                    self.interp_velocity[p] = v
                    self.point_force[p] = df
                else:
                    self.point_force[p] += df

    @ti.kernel
    def _spread_kernel(self, force: ti.template()):
        """spread the clamp forces (per unit span) with the kernel weights, uniformly along x like every force."""
        hy, hz = self.coupler.h[1], self.coupler.h[2]
        face_area = hy * hz
        for p in range(self.n):
            if self.points.tether[p] != 0:
                yp, zp = self.points.x[p][0], self.points.x[p][1]
                fp = self.increment[p]
                for c in ti.static((1, 2)):
                    sy = (yp - self.coupler.origin[1]) / hy - (0.0 if c == 1 else 0.5)
                    sz = (zp - self.coupler.origin[2]) / hz - (0.0 if c == 2 else 0.5)
                    bj = ti.cast(ti.floor(sy), ti.i32) - 1
                    bk = ti.cast(ti.floor(sz), ti.i32) - 1
                    for a, b in ti.static(ti.ndrange(4, 4)):
                        j, k = bj + a, bk + b
                        if 0 <= j < self.coupler.ny and 0 <= k < self.coupler.nz:
                            ry = sy - j
                            rz = sz - k
                            w = 0.0
                            if ti.static(self.coupler.ib3):
                                w = evaluate_ib3_weight(ry) * evaluate_ib3_weight(rz)
                            else:
                                w = evaluate_ib4_weight(ry) * evaluate_ib4_weight(rz)
                            density = fp[c - 1] * w / face_area
                            for i in range(self.coupler.nx):
                                ti.atomic_add(force[i, j, k][c], density)

    def apply(self, dt_s):
        """The clamp passes of this step into the coupler's fluid force (call after the other forces are spread)."""
        inv_dt = 1.0 / _validated_dt(dt_s)
        fluid = self.coupler.fluid
        for k in range(self.iterations):                   # multi-direct forcing: converge the clamp velocity
            self._force_kernel(fluid.velocity, fluid.force, inv_dt, self.rho, 1 if k == 0 else 0)
            self._spread_kernel(fluid.force)

    def state(self):
        """Interpolated velocities and direct-forcing forces of the clamped points at the last spread."""
        mask = self.points.tether.to_numpy() != 0
        return dict(index=np.flatnonzero(mask), u=self.interp_velocity.to_numpy()[mask],
                    f=self.point_force.to_numpy()[mask])
