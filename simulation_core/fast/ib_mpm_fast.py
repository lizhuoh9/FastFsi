"""REQ_379 FAST MODE prototype (Claude 2026-10-11; design note FAST_MODE_DESIGN_NOTE_01, approved by Root).

Diffuse immersed-boundary coupling of an MPM solid with ONE velocity field (IB / IBFE style): the solid has no momentum
equation of its own; its elastic stress is spread to the fluid grid as a body force, its material points move with the
interpolated divergence-free fluid velocity, and F follows the interpolated velocity gradient.  No partitioned trials,
no traction sampling, hence no added-mass instability; the explicit limit is the elastic shear-wave CFL.

This first version is quasi-2-D (fields uniform along solver x) on a uniform grid, for the Turek-Hron prototype.

Adapted algorithms (no source copied):
- IBAMR (BSD-3-Clause), src/IB/IBExplicitHierarchyIntegrator.cpp: explicit IB time stepping (FORWARD_EULER here):
  compute the Lagrangian force, spread it, solve the incompressible NS step, interpolate the velocity, move X.
- IBAMR ibtk/LEInteractor: the IB_4 regularized delta function, one kernel for spreading and interpolation (adjoints).
- taichi_mpm (MIT), mls-mpm88 / taichi mpm88: the P2G stress term f_i = -sum_p V0_p tau_p grad w_ip and the update
  F <- (I + dt grad v) F.
Nothing in simulation_core/coupling/hibm_mpm is used.
"""
import math

import numpy as np
import taichi as ti

FAST_MODE_REVISION = "REQ379-fast-ib-mpm-quasi2d-prototype-v1"


@ti.func
def ib4_phi(r):
    """Peskin 4-point regularized delta (IBAMR IBTK IB_4), support |r| < 2, unit sum and zero first moment."""
    a = ti.abs(r)
    value = 0.0
    if a < 1.0:
        value = (3.0 - 2.0 * a + ti.sqrt(ti.max(1.0 + 4.0 * a - 4.0 * a * a, 0.0))) / 8.0
    elif a < 2.0:
        value = (5.0 - 2.0 * a - ti.sqrt(ti.max(-7.0 + 12.0 * a - 4.0 * a * a, 0.0))) / 8.0
    return value


@ti.func
def ib4_dphi(r):
    """d phi / d r of IB_4 (continuous, zero at |r| = 2)."""
    a = ti.abs(r)
    slope = 0.0
    if a < 1.0:
        root = ti.sqrt(ti.max(1.0 + 4.0 * a - 4.0 * a * a, 1.0e-30))
        slope = (-2.0 + (2.0 - 4.0 * a) / root) / 8.0
    elif a < 2.0:
        root = ti.sqrt(ti.max(-7.0 + 12.0 * a - 4.0 * a * a, 1.0e-30))
        slope = (-2.0 - (6.0 - 4.0 * a) / root) / 8.0
    sign = 1.0
    if r < 0.0:
        sign = -1.0
    return sign * slope


@ti.func
def neo_hookean_kirchhoff_2d(F2, mu, kappa):
    """Plane-strain Kirchhoff stress (yz block) of the neo-Hookean solid with modified invariants plus a numerical
    volumetric stabiliser: W = mu/2 (I1b - 3) + kappa/2 (J - 1)^2 (same form as solids/neo_hookean_mpm.py
    yeoh_kirchhoff_stress with c1 = mu/2, c2 = c3 = 0).  F = diag(1, F2)."""
    J = F2.determinant()
    b = F2 @ F2.transpose()
    i1 = b.trace() + 1.0
    jm = ti.pow(J, -2.0 / 3.0)
    identity = ti.Matrix.identity(ti.f32, 2)
    return mu * jm * (b - (i1 / 3.0) * identity) + kappa * (J - 1.0) * J * identity


@ti.data_oriented
class FastIbMpmQuasi2D:
    """Material points of one elastic body in the (y, z) plane of a quasi-2-D fluid (fields uniform along x)."""

    def __init__(self, *, fluid, positions_yz_m, areas_m2, tether_mask, density_kgm3, shear_modulus_pa,
                 kappa_pa, tether_stiffness_pa_per_m, grid_origin_m, grid_spacing_m, grid_nodes, span_m,
                 near_surface_mask=None):
        positions = np.asarray(positions_yz_m, dtype=np.float32)
        n = len(positions)
        if n == 0:
            raise ValueError("fast IB-MPM needs material points")
        self.fluid = fluid
        self.n = n
        self.nx, self.ny, self.nz = (int(v) for v in grid_nodes)
        self.origin = tuple(float(v) for v in grid_origin_m)
        self.h = tuple(float(v) for v in grid_spacing_m)
        self.span_m = float(span_m)
        self.rho_s = float(density_kgm3)
        self.mu = float(shear_modulus_pa)
        self.kappa = float(kappa_pa)
        self.tether_k = float(tether_stiffness_pa_per_m)
        self.x = ti.Vector.field(2, ti.f32, shape=n)
        self.x_n = ti.Vector.field(2, ti.f32, shape=n)          # MIDPOINT_RULE: X^n and F^n
        self.F_n = ti.Matrix.field(2, 2, ti.f32, shape=n)
        self.X0 = ti.Vector.field(2, ti.f32, shape=n)
        self.v = ti.Vector.field(2, ti.f32, shape=n)
        self.F = ti.Matrix.field(2, 2, ti.f32, shape=n)
        self.tau = ti.Matrix.field(2, 2, ti.f32, shape=n)
        self.area = ti.field(ti.f32, shape=n)
        self.tether = ti.field(ti.i32, shape=n)
        # Root REQ_382 volume-drift statistics: reference points within 2 cells of the body surface
        self.near_surface = ti.field(ti.i32, shape=n)
        self.reduce = ti.field(ti.f64, shape=16)
        self.x.from_numpy(positions)
        self.X0.from_numpy(positions)
        self.v.fill(0.0)
        self.F.from_numpy(np.tile(np.eye(2, dtype=np.float32), (n, 1, 1)))
        self.area.from_numpy(np.asarray(areas_m2, dtype=np.float32))
        self.tether.from_numpy(np.asarray(tether_mask, dtype=np.int32))
        self.near_surface.from_numpy(np.zeros(n, np.int32) if near_surface_mask is None
                                     else np.asarray(near_surface_mask, dtype=np.int32))

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

    @ti.kernel
    def _stress_kernel(self, mu: ti.f32, kappa: ti.f32):
        for p in range(self.n):
            self.tau[p] = neo_hookean_kirchhoff_2d(self.F[p], mu, kappa)

    @ti.kernel
    def _clear_force_kernel(self, force: ti.template()):
        for i, j, k in force:
            force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _spread_kernel(self, force: ti.template(), tether_k: ti.f32):
        hy, hz = self.h[1], self.h[2]
        face_area = hy * hz
        for p in range(self.n):
            yp, zp = self.x[p][0], self.x[p][1]
            tau = self.tau[p]
            a0 = self.area[p]
            pull = ti.Vector([0.0, 0.0])
            if self.tether[p] != 0:
                pull = tether_k * a0 * (self.X0[p] - self.x[p])
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
                        wy, wz = ib4_phi(ry), ib4_phi(rz)
                        # grad of the node weight w_i(x_p) with respect to x_p
                        gy = ib4_dphi(ry) * wz / hy
                        gz = wy * ib4_dphi(rz) / hz
                        row = c - 1
                        f = -a0 * (tau[row, 0] * gy + tau[row, 1] * gz) + pull[row] * wy * wz
                        density = f / face_area
                        for i in range(self.nx):
                            ti.atomic_add(force[i, j, k][c], density)

    @ti.kernel
    def _interpolate_update_kernel(self, velocity: ti.template(), dt: ti.f32):
        hy, hz = self.h[1], self.h[2]
        for p in range(self.n):
            yp, zp = self.x[p][0], self.x[p][1]
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
                        wy, wz = ib4_phi(ry), ib4_phi(rz)
                        u = velocity[0, j, k][c]
                        row = c - 1
                        v[row] += u * wy * wz
                        G[row, 0] += u * ib4_dphi(ry) * wz / hy
                        G[row, 1] += u * wy * ib4_dphi(rz) / hz
            self.v[p] = v
            self.x[p] += dt * v
            self.F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * G) @ self.F[p]

    @ti.kernel
    def _save_state_kernel(self):
        for p in range(self.n):
            self.x_n[p] = self.x[p]
            self.F_n[p] = self.F[p]

    @ti.kernel
    def _midpoint_correct_kernel(self, velocity_old: ti.template(), velocity_new: ti.template(), dt: ti.f32):
        """IBAMR MIDPOINT_RULE: X^{n+1} = X^n + dt U^{n+1/2}(X^{n+1/2}), U^{n+1/2} = (u^n + u^{n+1}) / 2; the same
        for F with the interpolated gradient.  x currently holds X^{n+1/2}."""
        hy, hz = self.h[1], self.h[2]
        for p in range(self.n):
            yp, zp = self.x[p][0], self.x[p][1]
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
                        wy, wz = ib4_phi(ry), ib4_phi(rz)
                        u = 0.5 * (velocity_old[0, j, k][c] + velocity_new[0, j, k][c])
                        row = c - 1
                        v[row] += u * wy * wz
                        G[row, 0] += u * ib4_dphi(ry) * wz / hy
                        G[row, 1] += u * wy * ib4_dphi(rz) / hz
            self.v[p] = v
            # F currently holds the predicted F^{n+1/2}
            self.x[p] = self.x_n[p] + dt * v
            self.F[p] = self.F_n[p] + dt * (G @ self.F[p])

    @ti.kernel
    def _diagnostics_kernel(self):
        for s in range(16):
            self.reduce[s] = 0.0
        self.reduce[0] = 1.0e30
        self.reduce[1] = -1.0e30
        for p in range(self.n):
            J = ti.cast(self.F[p].determinant(), ti.f64)
            ti.atomic_min(self.reduce[0], J)
            ti.atomic_max(self.reduce[1], J)
            mass = ti.cast(self.rho_s * self.area[p], ti.f64)
            ti.atomic_add(self.reduce[2], mass * self.v[p][0])
            ti.atomic_add(self.reduce[3], mass * self.v[p][1])
            if self.tether[p] != 0:
                d = (self.x[p] - self.X0[p]).norm()
                ti.atomic_max(self.reduce[4], ti.cast(d, ti.f64))
                pull = self.tether_k * self.area[p] * (self.X0[p] - self.x[p])
                ti.atomic_add(self.reduce[5], ti.cast(pull[0], ti.f64))
                ti.atomic_add(self.reduce[6], ti.cast(pull[1], ti.f64))
            bad = 0
            for q in ti.static(range(2)):
                if ti.math.isnan(self.x[p][q]) or ti.math.isinf(self.x[p][q]):
                    bad = 1
            ti.atomic_add(self.reduce[7], ti.cast(bad, ti.f64))
            if self.tether[p] == 0:
                # Root REQ_382: volume drift of the free (non-tethered) body
                a0 = ti.cast(self.area[p], ti.f64)
                ti.atomic_add(self.reduce[8], a0)
                ti.atomic_add(self.reduce[9], a0 * J)
                ti.atomic_add(self.reduce[10], 1.0)
                if ti.abs(J - 1.0) > 5.0e-3:
                    ti.atomic_add(self.reduce[11], 1.0)
                    if self.near_surface[p] != 0:
                        ti.atomic_add(self.reduce[12], 1.0)

    # ------------------------------------------------------------------------------------------------ host API
    def spread_elastic_force(self):
        self._stress_kernel(self.mu, self.kappa)
        self._clear_force_kernel(self.fluid.force)
        self._spread_kernel(self.fluid.force, self.tether_k)

    def move_with_fluid(self, dt_s):
        self._interpolate_update_kernel(self.fluid.velocity, float(dt_s))

    def midpoint_predict(self, dt_s):
        """MIDPOINT_RULE first half: save X^n, F^n and move to X^{n+1/2}, F^{n+1/2} with u^n."""
        self._save_state_kernel()
        self._interpolate_update_kernel(self.fluid.velocity, 0.5 * float(dt_s))

    def midpoint_correct(self, velocity_old, dt_s):
        self._midpoint_correct_kernel(velocity_old, self.fluid.velocity, float(dt_s))

    def diagnostics(self):
        self._diagnostics_kernel()
        r = self.reduce.to_numpy()
        free = max(float(r[10]), 1.0)
        deviated = float(r[11])
        return dict(J_min=float(r[0]), J_max=float(r[1]), momentum_y=float(r[2]), momentum_z=float(r[3]),
                    tether_max_displacement_m=float(r[4]), tether_pull_y=float(r[5]), tether_pull_z=float(r[6]),
                    nonfinite_points=int(r[7]),
                    volume_error=(float(r[9]) / float(r[8]) - 1.0) if r[8] > 0 else 0.0,
                    J_deviated_fraction=deviated / free,
                    J_deviated_near_surface_share=(float(r[12]) / deviated) if deviated > 0 else 0.0)

    def elastic_dt_limit(self, cfl):
        """CFL on the shear wave over the smallest cell where the solid lives (uniform grid: min(hy, hz))."""
        c_s = math.sqrt(self.mu / self.rho_s)
        return float(cfl) * min(self.h[1], self.h[2]) / c_s

    def positions(self):
        return self.x.to_numpy()


@ti.data_oriented
class ControlVolumeForce:
    """Hydrodynamic force on everything inside a fixed (y, z) box (cylinder + beam) by a momentum balance:
    F = surface integral of (sigma n - rho u u.n) - d/dt (momentum inside the box) + d/dt (solid momentum inside)."""

    def __init__(self, fluid, *, j_range, k_range, origin_m, spacing_m, grid_nodes, span_m):
        self.fluid = fluid
        self.j0, self.j1 = (int(v) for v in j_range)
        self.k0, self.k1 = (int(v) for v in k_range)
        self.nx, self.ny, self.nz = (int(v) for v in grid_nodes)
        self.h = tuple(float(v) for v in spacing_m)
        self.span_m = float(span_m)
        self.out = ti.field(ti.f64, shape=4)

    @ti.func
    def _centre_u(self, velocity: ti.template(), j, k):
        """Cell-centre (uy, uz) of cell (0, j, k) from its backward and forward MAC faces."""
        uy = 0.5 * (velocity[0, j, k][1] + velocity[0, ti.min(j + 1, self.ny - 1), k][1])
        uz = 0.5 * (velocity[0, j, k][2] + velocity[0, j, ti.min(k + 1, self.nz - 1)][2])
        return ti.Vector([ti.cast(uy, ti.f64), ti.cast(uz, ti.f64)])

    @ti.kernel
    def _measure_kernel(self, velocity: ti.template(), pressure: ti.template(), obstacle: ti.template(),
                        rho: ti.f64, mu: ti.f64):
        for s in range(4):
            self.out[s] = 0.0
        hy, hz = self.h[1], self.h[2]
        # momentum inside the box (per unit span)
        for j, k in ti.ndrange((self.j0, self.j1), (self.k0, self.k1)):
            if obstacle[0, j, k] == 0:
                u = self._centre_u(velocity, j, k)
                ti.atomic_add(self.out[2], rho * u[0] * hy * hz)
                ti.atomic_add(self.out[3], rho * u[1] * hz * hy)
        # surface integrals: faces z = k0 (normal -z), z = k1 (normal +z), y = j0 (normal -y), y = j1 (normal +y)
        for side, j in ti.ndrange(2, (self.j0, self.j1)):
            k = self.k0 if side == 0 else self.k1 - 1
            n = -1.0 if side == 0 else 1.0
            u = self._centre_u(velocity, j, k)
            kk = ti.max(0, ti.min(k + (1 if side == 1 else -1), self.nz - 1))
            un = self._centre_u(velocity, j, kk)
            dz = n * hz
            duy_dz = (un[0] - u[0]) / dz
            duz_dz = (un[1] - u[1]) / dz
            jp, jm = ti.min(j + 1, self.ny - 1), ti.max(j - 1, 0)
            duz_dy = (self._centre_u(velocity, jp, k)[1] - self._centre_u(velocity, jm, k)[1]) / ((jp - jm) * hy)
            p = ti.cast(pressure[0, j, k], ti.f64)
            # traction sigma.n with n = (0, n): sigma_yz, sigma_zz
            t_y = mu * (duy_dz + duz_dy) * n
            t_z = (-p + 2.0 * mu * duz_dz) * n
            flux = rho * u * (u[1] * n)
            ti.atomic_add(self.out[0], (t_y - flux[0]) * hy)
            ti.atomic_add(self.out[1], (t_z - flux[1]) * hy)
        for side, k in ti.ndrange(2, (self.k0, self.k1)):
            j = self.j0 if side == 0 else self.j1 - 1
            n = -1.0 if side == 0 else 1.0
            u = self._centre_u(velocity, j, k)
            jj = ti.max(0, ti.min(j + (1 if side == 1 else -1), self.ny - 1))
            un = self._centre_u(velocity, jj, k)
            dy = n * hy
            duy_dy = (un[0] - u[0]) / dy
            duz_dy = (un[1] - u[1]) / dy
            kp, km = ti.min(k + 1, self.nz - 1), ti.max(k - 1, 0)
            duy_dz = (self._centre_u(velocity, j, kp)[0] - self._centre_u(velocity, j, km)[0]) / ((kp - km) * hz)
            p = ti.cast(pressure[0, j, k], ti.f64)
            t_y = (-p + 2.0 * mu * duy_dy) * n
            t_z = mu * (duz_dy + duy_dz) * n
            flux = rho * u * (u[0] * n)
            ti.atomic_add(self.out[0], (t_y - flux[0]) * hz)
            ti.atomic_add(self.out[1], (t_z - flux[1]) * hz)

    def measure(self):
        fluid = self.fluid
        self._measure_kernel(fluid.velocity, fluid.pressure, fluid.obstacle, float(fluid.rho), float(fluid.mu))
        r = self.out.to_numpy()
        return dict(surface_y=float(r[0]), surface_z=float(r[1]), momentum_y=float(r[2]), momentum_z=float(r[3]))
