"""REQ_379 FAST MODE, full 3-D body (Claude 2026-10-12; Root REQ_394/399/401/404: gap list 01 items A1-A3, contract 01 §2).

Diffuse immersed-boundary coupling of an MPM solid with ONE velocity field (IB/IBFE style).  The solid has no momentum
equation of its own: its elastic stress, tether pull and any extra point force are spread to the fluid MAC grid as a
body force density with the IB_4 kernel; its material points move with the interpolated velocity; F follows the
interpolated velocity gradient (F <- (I + dt grad v) F, IBAMR MIDPOINT_RULE).  The explicit limit is the elastic
shear-wave CFL.

Grid access goes through an axes object (contract §1.1: axis_weights / face_volume / index map; UniformAxes in
fluid_hooks.py, GradedAxes in graded_kernel.py) and a face-validity mask (contract §1.2: face_valid; AllValidMask /
FaceMask).  ONE-SIDED NORMALISATION (B2): with m the face validity, D = sum w m, w~ = w m / D and
grad w~ = (m grad w - w~ grad D) / D, grad D = sum m grad w.  The SAME w~ is used to spread and to interpolate, so the
pair stays adjoint (power identity) and sum w~ = 1 keeps both the spread force and the internal-stress force balance
(sum grad w~ = 0) exact next to invalid faces.  D <= 0 is a hard stop 'one_sided_empty_support' with a witness.

SPEED (Root REQ_404, physics-neutral, each covered by an equivalence test):
- interior fast path: refresh_support_flags() marks every cell within SUPPORT_REACH cells of an invalid face (base flag
  on the two cells adjacent to each invalid face, then a separable dilation); a point whose host cell is unflagged has
  an all-valid support, where D = sum w = 1 and grad D = sum grad w = 0 hold (partition of unity / zero derivative sum
  of the axis weights), so it takes ONE pass without mask lookups or normalisation;
- min D is a separate diagnostics pass (no contended atomic in the hot loops);
- the stress is computed inside the spread kernel; the force field can be cleared by the caller in the same grid pass
  that saves the old velocity (spread_force(clear=False));
- unroll: by default the outer node index is static and the two inner ones dynamic (16 static bodies per component and
  pass instead of 64), selectable with unroll_static=True for the fully static form (smoke measures both).
POSITIONS are integrated as the displacement u = x - X0 (f32): an absolute f32 coordinate near 1 m has an ulp of
1.2e-7 m, so slow motions (dt v below half an ulp) would be lost in x += dt v; x = X0 + u is kept as the current-position
field for kernels and other modules, and positions() sums X0 + u in float64.

Reuse check (REQ_400): IBAMR v0.19.0 (BSD-3-Clause) ibtk/src/lagrangian/LEInteractor.cpp, side-centred
interpolate()/spread(): each MAC component is treated as cell-centred data on a grid shifted by half a cell along its
own axis (x_lower_axis[axis] -= 0.5 dx), i.e. the node index along the component's axis is the face index and along the
other axes the cell-centre index — the same index map as axis_weights(staggered); the IB_4 weights and the 1/(dx dy dz)
spread scaling follow ibtk/src/lagrangian/fortran/lagrangian_interaction3d.f.m4 / lagrangian_delta.f.m4 (algorithm
reused, rewritten as Taichi kernels; no Fortran copied).  taichi_mpm (MIT) mls-mpm88: the P2G stress term
f_i = -sum_p V0_p tau_p grad w_ip and the F update.  Nothing in simulation_core/coupling/hibm_mpm is used.
IBAMR copyright (c) 2014 - 2026 by the IBAMR developers, BSD-3-Clause; taichi_mpm copyright (c) Yuanming Hu, MIT.
"""
import hashlib
import math

import numpy as np
import taichi as ti

FAST3D_REVISION = "REQ404-fast-ib-mpm-3d-v4-perbody-diag"
STATE_FIELDS = ("u", "u_n", "F", "F_n", "v")
REDUCE_SLOTS = 32
WITNESS_SLOTS = 16
SUPPORT_REACH = 3         # cells: covers both node families of the 4-node support around the host cell
TAG_AIR_FACE = 1          # bit 1: main-membrane air-face (top-layer) particle (air chamber lower bound)
TAG_VALVE_RIM = 2         # bit 2: valve-flap lower-surface rim particle, r >= 28.0 mm (contact diagnostics)
HARD_STOP_CODES = {"one_sided_empty_support", "nonfinite_point_state"}


class FastModeHardStop(RuntimeError):
    """A fast-mode hard stop with its machine-readable code and witness."""

    def __init__(self, code, witness):
        super().__init__("%s: %s" % (code, witness))
        self.code = code
        self.witness = witness


@ti.func
def neo_hookean_kirchhoff_3d(F, mu, kappa):
    """Kirchhoff stress of W = mu/2 (I1b - 3) + kappa/2 (J - 1)^2 (modified invariants; the 3-D form of the energy used
    by ib_mpm_fast.neo_hookean_kirchhoff_2d)."""
    J = F.determinant()
    b = F @ F.transpose()
    jm = ti.pow(J, -2.0 / 3.0)
    identity = ti.Matrix.identity(ti.f32, 3)
    return mu * jm * (b - (b.trace() / 3.0) * identity) + kappa * (J - 1.0) * J * identity


@ti.data_oriented
class FastIbMpm3D:
    """Material points of the deformable bodies in a 3-D fluid; grid access through `axes`, validity through `mask`."""

    def __init__(self, *, fluid, axes, mask, positions_m, volumes_m3, tether_mask, density_kgm3, shear_modulus_pa,
                 kappa_pa, tether_stiffness_pa_per_m, body_id=None, tag=None, external_force_n=None,
                 near_surface_mask=None, probe_indices=(), fast_path=True, unroll_static=False):
        positions = np.asarray(positions_m, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
            raise ValueError("fast 3-D IB-MPM needs an (n, 3) array of material points")
        n = len(positions)
        volumes = np.asarray(volumes_m3, dtype=np.float32)
        tether = np.asarray(tether_mask, dtype=np.int32)
        if volumes.shape != (n,) or tether.shape != (n,):
            raise ValueError("volumes and tether mask must have one entry per point")
        if not np.all(volumes > 0.0):
            raise ValueError("point volumes must be positive")
        self.fluid = fluid
        self.axes = axes
        self.mask = mask
        self.n = n
        self.fast_path = bool(fast_path)
        self.unroll_static = bool(unroll_static)
        self.nx, self.ny, self.nz = (int(v) for v in axes.n)
        self.rho_s = float(density_kgm3)
        self.mu = float(shear_modulus_pa)
        self.kappa = float(kappa_pa)
        self.tether_k = float(tether_stiffness_pa_per_m)
        self.x = ti.Vector.field(3, ti.f32, shape=n)            # current position X0 + u (derived)
        self.X0 = ti.Vector.field(3, ti.f32, shape=n)
        self.u = ti.Vector.field(3, ti.f32, shape=n)            # displacement x - X0 (integrated: f32 precision)
        self.u_n = ti.Vector.field(3, ti.f32, shape=n)          # MIDPOINT_RULE: u^n and F^n
        self.v = ti.Vector.field(3, ti.f32, shape=n)
        self.F = ti.Matrix.field(3, 3, ti.f32, shape=n)
        self.F_n = ti.Matrix.field(3, 3, ti.f32, shape=n)
        self.tau = ti.Matrix.field(3, 3, ti.f32, shape=n)
        self.grad_v = ti.Matrix.field(3, 3, ti.f32, shape=n)    # last interpolated velocity gradient (diagnostics)
        self.vol = ti.field(ti.f32, shape=n)
        self.tether = ti.field(ti.i32, shape=n)
        self.body_id = ti.field(ti.i32, shape=n)
        self.tag = ti.field(ti.i32, shape=n)
        self.near_surface = ti.field(ti.i32, shape=n)
        self.f_ext = ti.Vector.field(3, ti.f32, shape=n)        # external load (N) at load scale 1
        self.f_extra = ti.Vector.field(3, ti.f32, shape=n)      # extra point force (N), consumed and zeroed per spread
        self.reduce = ti.field(ti.f64, shape=REDUCE_SLOTS)
        self.min_D = ti.field(ti.f32, shape=())
        self.count_D0 = ti.field(ti.i32, shape=())
        self.min_D_point = ti.field(ti.i32, shape=())
        self.min_D_free = ti.field(ti.f32, shape=())
        self.min_D_rim_free = ti.field(ti.f32, shape=())        # free TAG_VALVE_RIM points (Root 2026-10-12, item 2)
        self.witness = ti.Vector.field(5, ti.f32, shape=WITNESS_SLOTS)   # point, component, x, y, z
        grid_shape = (self.nx, self.ny, self.nz)
        self.support_flag = ti.field(ti.u8, shape=grid_shape)  # 1: a cell within SUPPORT_REACH of an invalid face
        self._flag_a = ti.field(ti.u8, shape=grid_shape)
        self._flag_b = ti.field(ti.u8, shape=grid_shape)
        probes = np.asarray(probe_indices, dtype=np.int32).reshape(-1)
        if np.any((probes < 0) | (probes >= n)):
            raise ValueError("probe index out of range")
        self.n_probe = max(1, len(probes))
        self.probe_index = ti.field(ti.i32, shape=self.n_probe)
        self.probe_x = ti.Vector.field(3, ti.f32, shape=self.n_probe)
        self.probe_index.from_numpy(probes if len(probes) else np.zeros(1, np.int32))
        self.x.from_numpy(positions)
        self.X0.from_numpy(positions)
        self.u.fill(0.0)
        self.u_n.fill(0.0)
        self.v.fill(0.0)
        self.grad_v.fill(0.0)
        eye = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        self.F.from_numpy(eye)
        self.F_n.from_numpy(eye)
        self.vol.from_numpy(volumes)
        self.tether.from_numpy(tether)
        self.body_id.from_numpy(np.zeros(n, np.int32) if body_id is None else np.asarray(body_id, np.int32))
        self.tag.from_numpy(np.zeros(n, np.int32) if tag is None else np.asarray(tag, np.int32))
        self.near_surface.from_numpy(np.zeros(n, np.int32) if near_surface_mask is None
                                     else np.asarray(near_surface_mask, dtype=np.int32))
        ext = np.zeros((n, 3), np.float32) if external_force_n is None else np.asarray(external_force_n, np.float32)
        if ext.shape != (n, 3):
            raise ValueError("external force must be (n, 3)")
        self.f_ext.from_numpy(ext)
        self.f_extra.fill(0.0)
        self.support_flag.fill(1)            # conservative until the first refresh_support_flags()
        self.reset_one_sided_stats()

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
        ix, wx, dwx = self.axes.axis_weights(0, 1 if c == 0 else 0, xp[0])
        iy, wy, dwy = self.axes.axis_weights(1, 1 if c == 1 else 0, xp[1])
        iz, wz, dwz = self.axes.axis_weights(2, 1 if c == 2 else 0, xp[2])
        return ix, iy, iz, wx, wy, wz, dwx, dwy, dwz

    @ti.func
    def _interior(self, xp):
        """1 if the host cell of xp is not flagged (its whole support is valid)."""
        hx, wcx, dcx = self.axes.axis_weights(0, 0, xp[0])
        hy, wcy, dcy = self.axes.axis_weights(1, 0, xp[1])
        hz, wcz, dcz = self.axes.axis_weights(2, 0, xp[2])
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
        the far face n_a; Helper 1 review 2026-10-12)."""
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
            xp = self.x[p]
            tau = neo_hookean_kirchhoff_3d(self.F[p], mu, kappa)
            self.tau[p] = tau
            v0 = self.vol[p]
            pull = ti.Vector([0.0, 0.0, 0.0])
            if self.tether[p] != 0:
                pull = -tether_k * v0 * self.u[p]
            point_force = pull + load_scale * self.f_ext[p] + self.f_extra[p]
            self.f_extra[p] = ti.Vector([0.0, 0.0, 0.0])
            interior = 0
            if ti.static(self.fast_path):
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
                g = (ti.Vector([da * wb * we, wa * db * we, wa * wb * de]) - w * gD) * inv_D
                f = -v0 * (tau[c, 0] * g[0] + tau[c, 1] * g[1] + tau[c, 2] * g[2]) + point_force[c] * w
                ti.atomic_add(force[i, j, k][c], f / self.axes.face_volume(c, i, j, k))

    @ti.func
    def _gather(self, velocity: ti.template(), velocity_b: ti.template(), blend: ti.f32, p, xp):
        """One-sided interpolated velocity and gradient of u = (1 - blend) velocity + blend velocity_b."""
        v = ti.Vector([0.0, 0.0, 0.0])
        G = ti.Matrix.zero(ti.f32, 3, 3)
        interior = 0
        if ti.static(self.fast_path):
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
                g = (ti.Vector([da * wb * we, wa * db * we, wa * wb * de]) - w * gD) * inv_D
                uval = (1.0 - blend) * velocity[i, j, k][c] + blend * velocity_b[i, j, k][c]
                dv = uval * w
                dg = uval * g
        return dv, dg

    @ti.kernel
    def _interpolate_update_kernel(self, velocity: ti.template(), dt: ti.f32):
        for p in range(self.n):
            v, G = self._gather(velocity, velocity, 0.0, p, self.x[p])
            self.v[p] = v
            self.grad_v[p] = G
            self.u[p] += dt * v
            self.x[p] = self.X0[p] + self.u[p]
            self.F[p] = (ti.Matrix.identity(ti.f32, 3) + dt * G) @ self.F[p]

    @ti.kernel
    def _refresh_x_kernel(self):
        for p in range(self.n):
            self.x[p] = self.X0[p] + self.u[p]

    @ti.kernel
    def _save_state_kernel(self):
        for p in range(self.n):
            self.u_n[p] = self.u[p]
            self.F_n[p] = self.F[p]

    @ti.kernel
    def _midpoint_correct_kernel(self, velocity_old: ti.template(), velocity_new: ti.template(), dt: ti.f32):
        """IBAMR MIDPOINT_RULE: X^{n+1} = X^n + dt U^{n+1/2}(X^{n+1/2}), U^{n+1/2} = (u^n + u^{n+1}) / 2; the same
        for F with the interpolated gradient.  x currently holds X^{n+1/2}, F the predicted F^{n+1/2}."""
        for p in range(self.n):
            v, G = self._gather(velocity_old, velocity_new, 0.5, p, self.x[p])
            self.v[p] = v
            self.grad_v[p] = G
            self.u[p] = self.u_n[p] + dt * v
            self.x[p] = self.X0[p] + self.u[p]
            self.F[p] = self.F_n[p] + dt * (G @ self.F[p])

    @ti.kernel
    def _min_D_kernel(self):
        """min over points and components of the one-sided denominator D at the current positions (diagnostics pass;
        kept out of the spread / gather hot loops, Root review 2026-10-12)."""
        self.min_D[None] = 1.0e30
        self.min_D_free[None] = 1.0e30
        self.min_D_rim_free[None] = 1.0e30
        for p in range(self.n):
            local = 1.0e30
            for c in ti.static(range(3)):
                ix, iy, iz, wx, wy, wz, dwx, dwy, dwz = self._axis(c, self.x[p])
                D, gD = self._denominator(c, ix, iy, iz, wx, wy, wz, dwx, dwy, dwz)
                local = ti.min(local, D)
            ti.atomic_min(self.min_D[None], local)
            if self.tether[p] == 0:
                ti.atomic_min(self.min_D_free[None], local)
                if (self.tag[p] & TAG_VALVE_RIM) != 0:
                    ti.atomic_min(self.min_D_rim_free[None], local)

    @ti.kernel
    def _min_D_locate_kernel(self, free_only: ti.i32):
        """Index of one point whose smallest component denominator equals min D (free points only if free_only;
        diagnostics only)."""
        self.min_D_point[None] = -1
        target = self.min_D[None]
        if free_only != 0:
            target = self.min_D_free[None]
        for p in range(self.n):
            if free_only != 0 and self.tether[p] != 0:
                continue
            local = 1.0e30
            for c in ti.static(range(3)):
                ix, iy, iz, wx, wy, wz, dwx, dwy, dwz = self._axis(c, self.x[p])
                D, gD = self._denominator(c, ix, iy, iz, wx, wy, wz, dwx, dwy, dwz)
                local = ti.min(local, D)
            if local <= target:
                self.min_D_point[None] = p

    @ti.kernel
    def _probe_kernel(self):
        for q in range(self.n_probe):
            self.probe_x[q] = self.x[self.probe_index[q]]

    @ti.kernel
    def _diagnostics_kernel(self, rim_land_z: ti.f32):
        for s in range(REDUCE_SLOTS):
            self.reduce[s] = 0.0
        self.reduce[0] = 1.0e30
        self.reduce[1] = -1.0e30
        self.reduce[18] = 1.0e30
        self.reduce[19] = 1.0e30
        self.reduce[25] = -1.0e30
        for p in range(self.n):
            J = ti.cast(self.F[p].determinant(), ti.f64)
            ti.atomic_min(self.reduce[0], J)
            ti.atomic_max(self.reduce[1], J)
            mass = ti.cast(self.rho_s * self.vol[p], ti.f64)
            for q in ti.static(range(3)):
                ti.atomic_add(self.reduce[2 + q], mass * ti.cast(self.v[p][q], ti.f64))
            bad = 0
            for q in ti.static(range(3)):
                if ti.math.isnan(self.u[p][q]) or ti.math.isinf(self.u[p][q]):
                    bad = 1
            if ti.math.isnan(J) or ti.math.isinf(J):
                bad = 1
            ti.atomic_add(self.reduce[5], ti.cast(bad, ti.f64))
            d = ti.cast(self.u[p].norm(), ti.f64)
            if self.tether[p] != 0:
                ti.atomic_max(self.reduce[6], d)
                if self.body_id[p] == 1:                   # flap bonded band: lift under suction (Root item 4)
                    ti.atomic_max(self.reduce[24], d)
                    ti.atomic_max(self.reduce[25], ti.cast(self.u[p][2], ti.f64))
                elif self.body_id[p] == 2:
                    ti.atomic_max(self.reduce[26], d)
                pull = -self.tether_k * self.vol[p] * self.u[p]
                for q in ti.static(range(3)):
                    ti.atomic_add(self.reduce[7 + q], ti.cast(pull[q], ti.f64))
            else:
                v0 = ti.cast(self.vol[p], ti.f64)
                ti.atomic_add(self.reduce[10], v0)
                ti.atomic_add(self.reduce[11], v0 * J)
                if self.body_id[p] == 1:                   # per-body volume (Root item 4: attribute any stop)
                    ti.atomic_add(self.reduce[20], v0)
                    ti.atomic_add(self.reduce[21], v0 * J)
                elif self.body_id[p] == 2:
                    ti.atomic_add(self.reduce[22], v0)
                    ti.atomic_add(self.reduce[23], v0 * J)
                ti.atomic_add(self.reduce[12], 1.0)
                ti.atomic_max(self.reduce[13], d)
                if ti.abs(J - 1.0) > 5.0e-3:
                    ti.atomic_add(self.reduce[14], 1.0)
                    if self.near_surface[p] != 0:
                        ti.atomic_add(self.reduce[15], 1.0)
                ti.atomic_add(self.reduce[16], 0.5 * mass * ti.cast(self.v[p].norm_sqr(), ti.f64))
                ti.atomic_max(self.reduce[17], ti.cast(self.v[p].norm(), ti.f64))
                if self.body_id[p] == 2:
                    ti.atomic_min(self.reduce[18], ti.cast(self.u[p][2], ti.f64))
            if (self.tag[p] & TAG_VALVE_RIM) != 0:
                ti.atomic_min(self.reduce[19], ti.cast(self.X0[p][2] + self.u[p][2] - rim_land_z, ti.f64))

    # ------------------------------------------------------------------------------------------------ host API
    def refresh_support_flags(self):
        """Mark the cells whose 4-node support may touch an invalid face (call after every mask refresh)."""
        if not self.fast_path:
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
            raise FastModeHardStop("one_sided_empty_support", dict(count=count, witnesses=witness))
        if reset:
            self.count_D0[None] = 0
        min_free = float(self.min_D_free[None])
        rim = float(self.min_D_rim_free[None])
        return dict(min_D=min_d, min_D_free=min_free, min_D_below_0p25=bool(min_d < 0.25),
                    min_D_free_below_0p25=bool(min_free < 0.25),
                    min_D_rim_free=rim if rim < 1.0e29 else float('nan'))

    def min_D_witness(self, free_only=False):
        """Where min D occurs (over free points only if free_only): point index, position, body, tether, tag
        (diagnostics; two passes)."""
        self._min_D_kernel()
        self._min_D_locate_kernel(1 if free_only else 0)
        p = int(self.min_D_point[None])
        if p < 0:
            return dict(min_D=float(self.min_D[None]), point=None)
        x = self.X0.to_numpy()[p].astype(np.float64) + self.u.to_numpy()[p].astype(np.float64)
        return dict(min_D=float(self.min_D[None]), min_D_free=float(self.min_D_free[None]), point=p,
                    position_m=x.tolist(), body_id=int(self.body_id.to_numpy()[p]),
                    tether=int(self.tether.to_numpy()[p]), tag=int(self.tag.to_numpy()[p]))

    @ti.kernel
    def _clear_force_kernel(self, force: ti.template()):
        for i, j, k in force:
            force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

    def spread_force(self, load_scale=1.0, clear=True):
        """One-sided IB force density of stress + tether + scaled external load + extra point force into fluid.force
        (f_extra is consumed and zeroed).  clear=False when the caller already zeroed fluid.force in a fused pass."""
        if clear:
            self._clear_force_kernel(self.fluid.force)
        self._spread_kernel(self.fluid.force, self.tether_k, float(load_scale), self.mu, self.kappa)

    spread_elastic_force = spread_force

    def move_with_fluid(self, dt_s):
        self._interpolate_update_kernel(self.fluid.velocity, float(dt_s))

    def midpoint_predict(self, dt_s):
        """MIDPOINT_RULE first half: save u^n, F^n and move to X^{n+1/2}, F^{n+1/2} with u^n."""
        self._save_state_kernel()
        self._interpolate_update_kernel(self.fluid.velocity, 0.5 * float(dt_s))

    def midpoint_correct(self, velocity_old, dt_s):
        self._midpoint_correct_kernel(velocity_old, self.fluid.velocity, float(dt_s))

    def probes(self):
        self._probe_kernel()
        return self.probe_x.to_numpy()

    def positions(self):
        """Current positions X0 + u, summed in float64 on the host."""
        return self.X0.to_numpy().astype(np.float64) + self.u.to_numpy().astype(np.float64)

    def velocities(self):
        return self.v.to_numpy()

    def diagnostics(self, rim_land_z_m=0.0):
        self._diagnostics_kernel(float(rim_land_z_m))
        r = self.reduce.to_numpy()
        free = max(float(r[12]), 1.0)
        deviated = float(r[14])
        return dict(J_min=float(r[0]), J_max=float(r[1]), momentum=[float(r[2]), float(r[3]), float(r[4])],
                    nonfinite_points=int(r[5]), tether_max_displacement_m=float(r[6]),
                    tether_pull_n=[float(r[7]), float(r[8]), float(r[9])],
                    volume_error=(float(r[11]) / float(r[10]) - 1.0) if r[10] > 0 else 0.0,
                    free_max_displacement_m=float(r[13]), J_deviated_fraction=deviated / free,
                    J_deviated_near_surface_share=(float(r[15]) / deviated) if deviated > 0 else 0.0,
                    kinetic_energy_j=float(r[16]), free_max_speed_mps=float(r[17]),
                    membrane_min_uz_m=float(r[18]) if r[18] < 1.0e29 else float('nan'),
                    rim_min_gap_m=float(r[19]) if r[19] < 1.0e29 else float('nan'),
                    volume_error_flap=(float(r[21]) / float(r[20]) - 1.0) if r[20] > 0 else float('nan'),
                    volume_error_membrane=(float(r[23]) / float(r[22]) - 1.0) if r[22] > 0 else float('nan'),
                    free_volume_flap_m3=float(r[20]), free_volume_membrane_m3=float(r[22]),
                    tether_max_displacement_flap_m=float(r[24]),
                    tether_max_uz_flap_m=float(r[25]) if r[25] > -1.0e29 else float('nan'),
                    tether_max_displacement_membrane_m=float(r[26]))

    def elastic_dt_limit(self, cfl):
        """CFL on the shear wave over the smallest cell edge of the grid (conservative for graded grids)."""
        c_s = math.sqrt(self.mu / self.rho_s)
        return float(cfl) * float(self.axes.min_spacing_m) / c_s

    def state_arrays(self):
        """The evolving point state (checkpoint payload)."""
        return {name: getattr(self, name).to_numpy() for name in STATE_FIELDS}

    def load_state_arrays(self, arrays):
        """Restore the evolving point state after a strict shape / dtype check of every array (all-or-nothing)."""
        staged = []
        for name in STATE_FIELDS:
            if name not in arrays:
                raise ValueError("checkpoint is missing body field %s" % name)
            field = getattr(self, name)
            expected = field.to_numpy()
            value = np.asarray(arrays[name])
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError("checkpoint body field %s is %s %s, expected %s %s"
                                 % (name, value.shape, value.dtype, expected.shape, expected.dtype))
            staged.append((field, value))
        for field, value in staged:
            field.from_numpy(value)
        self._refresh_x_kernel()

    def fingerprint(self):
        """Static setup arrays that a checkpoint must match (reference positions, volumes, tethers, ids, tags, loads)."""
        digest = hashlib.sha256()
        for field in (self.X0, self.vol, self.tether, self.body_id, self.tag, self.f_ext, self.near_surface):
            digest.update(np.ascontiguousarray(field.to_numpy()).tobytes())
        digest.update(repr((self.n, tuple(self.axes.n), self.rho_s, self.mu, self.kappa, self.tether_k)).encode())
        return digest.hexdigest()
