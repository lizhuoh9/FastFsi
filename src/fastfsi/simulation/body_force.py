"""Whole-body force: the force a fixed body and the solids it holds exert on the body's mount (what a load cell under the
body reads), measured read-only two ways (Root 2026-10-24, the USER's whole-robot thrust record).

A. CONTROL VOLUME.  Boxes of whole cells around the body (cell index ranges; a box face may lie on a domain wall).
   Newton for the box's contents, the mount being the only other force on them:
       F = -[ dP/dt + sum_open faces (rho u (u.n) + p n - tau.n) A + sum_wall faces (p n + mu u_t / (w/2)) A ].
   P is rho times the water's cell-centred momentum inside the box plus the excess-mass partners' momentum (every
   partner must lie inside every box).  A box face on the domain's top wall over the body's footprint (top_footprint)
   is the body itself and carries no term; elsewhere a face on a domain wall carries the wall's pressure and its
   no-slip shear on the box's water, and no flux.  Open faces: p and the advected components averaged to the face,
   u.n the MAC face velocity, tau = mu (grad u + grad u^T) from central differences of cell-centred velocities.  A face
   whose two cells are not both water (the box cuts the body or the air) is counted (cut_faces) and skipped: a valid box
   has none.
B. DIRECT SUM.  The pressure (water or air) and the no-slip shear (water) on every face between a non-obstacle cell and
   an obstacle cell, and on the domain's top-wall faces over top_footprint (a lid that is the domain wall), minus the
   forces the held solids receive from the body: the direct clamp's point forces, and the point forces the caller adds
   (add_point_forces: e.g. the contact, a covered inlet's traction; one slot each).
Intervals are impulse-exact: every step accumulate(dt) adds F dt of the surface, traction and clamp terms on the device
(the pressure is the physical gauge pressure); report() takes P at the interval's end, so the interval's dP/dt
telescopes, returns the interval means with one host read, and starts the next interval.  Nothing here writes a field
that the physics reads.

No `from __future__ import annotations` (Taichi kernel module).
"""
import numpy as np
import taichi as ti

A_FLUX, A_PRESSURE, A_VISCOUS, A_WALL_P, A_WALL_SHEAR = range(5)
A_PARTS = ("flux", "pressure", "viscous", "wall_pressure", "wall_shear")
B_OBSTACLE_P, B_OBSTACLE_SHEAR, B_LID_P, B_LID_SHEAR, B_CLAMP = range(5)
B_PARTS = ("obstacle_pressure", "obstacle_shear", "lid_pressure", "lid_shear", "clamp")


@ti.data_oriented
class WholeBodyForce:
    """See the module docstring.  boxes: a list of dicts with integer cell ranges i=(i0, i1), j=(j0, j1), k=(k0, k1)
    (half-open) and a name; top_footprint: an (nx, ny) array, nonzero where the domain's top wall belongs to the body;
    air_cell: the gas mask (None: none); clamp: a DirectForcingClamp (None: no clamp term); excess_mass: the coupler's
    excess-mass partners (None: none); point_slots: the names of the add_point_forces slots."""

    def __init__(self, fluid, axes, boxes, *, top_footprint=None, air_cell=None, clamp=None, excess_mass=None,
                 point_forces=None, point_slots=()):
        self.fluid = fluid
        self.n = tuple(int(v) for v in axes.n)
        nx, ny, nz = self.n
        self.rho, self.mu = float(fluid.rho), float(fluid.mu)
        self.boxes = []
        for b in boxes:
            r = tuple(tuple(int(v) for v in b[a]) for a in ("i", "j", "k"))
            for a, (lo, hi) in enumerate(r):
                if not 0 <= lo < hi <= self.n[a]:
                    raise ValueError("box %s: range %s outside 0..%d along axis %d" % (b.get("name"), (lo, hi),
                                                                                    self.n[a], a))
            self.boxes.append(dict(name=str(b.get("name", "box%d" % len(self.boxes))), range=r))
        nb = len(self.boxes)
        self.box_range = ti.field(ti.i32, shape=(nb, 3, 2))
        self.box_range.from_numpy(np.array([b["range"] for b in self.boxes], np.int32))
        faces = [np.asarray(f, np.float64) for f in axes.faces_m]
        self.centre = [ti.field(ti.f64, shape=self.n[a]) for a in range(3)]
        self.width = [ti.field(ti.f64, shape=self.n[a]) for a in range(3)]
        for a in range(3):
            self.centre[a].from_numpy(0.5 * (faces[a][1:] + faces[a][:-1]))
            self.width[a].from_numpy(np.diff(faces[a]))
        self.footprint = ti.field(ti.i32, shape=(nx, ny))
        if top_footprint is not None:
            self.footprint.from_numpy((np.asarray(top_footprint) != 0).astype(np.int32))
        self.air = air_cell if air_cell is not None else ti.field(ti.i32, shape=self.n)
        self.clamp = clamp
        self.excess = excess_mass
        self.point_forces = point_forces
        self.slots = tuple(point_slots)
        self.acc_a = ti.field(ti.f64, shape=(nb, len(A_PARTS), 3))
        self.acc_b = ti.field(ti.f64, shape=(len(B_PARTS), 3))
        self.acc_points = ti.field(ti.f64, shape=(max(1, len(self.slots)), 3))
        self.acc_time = ti.field(ti.f64, shape=())
        self.cut = ti.field(ti.i32, shape=nb)
        self.momentum = ti.field(ti.f64, shape=(nb + 1, 3))           # the boxes' water, then the partners
        self._own_pressure = None
        self._p_start = None
        self._t_start = None

    # ------------------------------------------------------------------------------------------------ helpers
    @ti.func
    def _inside(self, i, j, k):
        return 0 <= i < self.n[0] and 0 <= j < self.n[1] and 0 <= k < self.n[2]

    @ti.func
    def _water(self, i, j, k):
        ok = False
        if self._inside(i, j, k):
            ok = self.fluid.obstacle[i, j, k] == 0 and self.air[i, j, k] == 0
        return ok

    @ti.func
    def _uc(self, i, j, k):
        """Cell-centred velocity from the backward-MAC faces (a face beyond the last cell reads 0, a wall)."""
        u = ti.Vector([0.0, 0.0, 0.0], ti.f64)
        for a in ti.static(range(3)):
            nxt = ti.Vector([i, j, k])
            nxt[a] += 1
            plus = ti.cast(0.0, ti.f64)
            if nxt[a] < self.n[a]:
                plus = ti.cast(self.fluid.velocity[nxt[0], nxt[1], nxt[2]][a], ti.f64)
            u[a] = 0.5 * (ti.cast(self.fluid.velocity[i, j, k][a], ti.f64) + plus)
        return u

    @ti.func
    def _grad(self, i, j, k):
        """grad u at a cell centre (G[m, a] = d u_m / d x_a) by central differences of the cell-centred velocities
        (one-sided at the domain's edge)."""
        G = ti.Matrix.zero(ti.f64, 3, 3)
        for a in ti.static(range(3)):
            lo = ti.Vector([i, j, k])
            hi = ti.Vector([i, j, k])
            lo[a] = ti.max(lo[a] - 1, 0)
            hi[a] = ti.min(hi[a] + 1, self.n[a] - 1)
            dx = self.centre[a][hi[a]] - self.centre[a][lo[a]]
            if dx > 0.0:
                du = self._uc(hi[0], hi[1], hi[2]) - self._uc(lo[0], lo[1], lo[2])
                for m in ti.static(range(3)):
                    G[m, a] = du[m] / dx
        return G

    @ti.func
    def _area(self, a: ti.template(), i, j, k):
        c = ti.Vector([i, j, k])
        area = ti.cast(1.0, ti.f64)
        for b in ti.static(range(3)):
            if ti.static(b != a):
                area *= self.width[b][c[b]]
        return area

    @ti.func
    def _box_face(self, box, a: ti.template(), side, f, i, j, k, pressure: ti.template(), dt):
        """One face cell of box `box`'s plane normal to axis a at face index f (side -1: the box's lower face, +1: its
        upper face; (i, j, k): the cell inside the box next to the face)."""
        e = ti.Vector([0.0, 0.0, 0.0], ti.f64)
        e[a] = ti.cast(side, ti.f64)
        area = self._area(a, i, j, k)
        if f == 0 or f == self.n[a]:                         # the face lies on a domain wall
            body = False
            if ti.static(a == 2):
                if f == self.n[2]:
                    body = self.footprint[i, j] != 0
            if not body:
                if self._water(i, j, k):
                    p = ti.cast(pressure[i, j, k], ti.f64)
                    u = self._uc(i, j, k)
                    u[a] = 0.0
                    shear = ti.cast(self.mu, ti.f64) * u / (0.5 * self._width(a, i, j, k))
                    for m in ti.static(range(3)):
                        ti.atomic_add(self.acc_a[box, A_WALL_P, m], p * e[m] * area * dt)
                        ti.atomic_add(self.acc_a[box, A_WALL_SHEAR, m], shear[m] * area * dt)
                elif self.air[i, j, k] != 0:
                    ti.atomic_add(self.cut[box], 1)
        else:
            out = ti.Vector([i, j, k])
            out[a] += side
            if self._water(i, j, k) and self._water(out[0], out[1], out[2]):
                hi = ti.Vector([i, j, k])
                if side > 0:
                    hi = out
                un = ti.cast(self.fluid.velocity[hi[0], hi[1], hi[2]][a], ti.f64)    # the MAC face velocity
                u = 0.5 * (self._uc(i, j, k) + self._uc(out[0], out[1], out[2]))
                u[a] = un
                p = 0.5 * (ti.cast(pressure[i, j, k], ti.f64) + ti.cast(pressure[out[0], out[1], out[2]], ti.f64))
                G = 0.5 * (self._grad(i, j, k) + self._grad(out[0], out[1], out[2]))
                tau = ti.cast(self.mu, ti.f64) * (G + G.transpose())
                flux = ti.cast(self.rho, ti.f64) * u * (un * side) * area
                for m in ti.static(range(3)):
                    ti.atomic_add(self.acc_a[box, A_FLUX, m], flux[m] * dt)
                    ti.atomic_add(self.acc_a[box, A_PRESSURE, m], p * e[m] * area * dt)
                    ti.atomic_add(self.acc_a[box, A_VISCOUS, m], tau[m, a] * side * area * dt)
            else:
                ti.atomic_add(self.cut[box], 1)

    @ti.func
    def _width(self, a: ti.template(), i, j, k):
        c = ti.Vector([i, j, k])
        return self.width[a][c[a]]

    # ------------------------------------------------------------------------------------------------ kernels
    @ti.kernel
    def _surface_kernel(self, pressure: ti.template(), dt: ti.f64):
        nb = self.box_range.shape[0]
        for a in ti.static(range(3)):
            b, c = ti.static((a + 1) % 3, (a + 2) % 3)
            for s in ti.static((-1, 1)):
                for box, q, r in ti.ndrange(nb, self.n[b], self.n[c]):       # one parallel loop per box face
                    if (self.box_range[box, b, 0] <= q < self.box_range[box, b, 1]
                            and self.box_range[box, c, 0] <= r < self.box_range[box, c, 1]):
                        f = self.box_range[box, a, ti.static(0 if s < 0 else 1)]
                        idx = ti.Vector([0, 0, 0])
                        idx[a] = f if ti.static(s < 0) else f - 1        # the box's own cell next to the face
                        idx[b] = q
                        idx[c] = r
                        self._box_face(box, a, s, f, idx[0], idx[1], idx[2], pressure, dt)

    @ti.kernel
    def _traction_kernel(self, pressure: ti.template(), dt: ti.f64):
        nx, ny, nz = ti.static(self.n)
        for i, j, k in ti.ndrange(nx, ny, nz):
            if self.fluid.obstacle[i, j, k] == 0:
                gas = self.air[i, j, k] != 0
                p = ti.cast(pressure[i, j, k], ti.f64)
                u = self._uc(i, j, k)
                for a in ti.static(range(3)):
                    for s in ti.static((-1, 1)):
                        nb = ti.Vector([i, j, k])
                        nb[a] += s
                        lid = False
                        if ti.static(a == 2 and s == 1):
                            lid = k == nz - 1 and self.footprint[i, j] != 0
                        hit = lid
                        if self._inside(nb[0], nb[1], nb[2]):
                            hit = self.fluid.obstacle[nb[0], nb[1], nb[2]] != 0
                        if hit:
                            area = self._area(a, i, j, k)
                            half = 0.5 * self._width(a, i, j, k)
                            pslot = B_LID_P if lid else B_OBSTACLE_P
                            sslot = B_LID_SHEAR if lid else B_OBSTACLE_SHEAR
                            ti.atomic_add(self.acc_b[pslot, a], p * s * area * dt)
                            if not gas:                              # no-slip shear of the water on the face
                                for m in ti.static(range(3)):
                                    if ti.static(m != a):
                                        ti.atomic_add(self.acc_b[sslot, m],
                                                      ti.cast(self.mu, ti.f64) * u[m] / half * area * dt)

    @ti.kernel
    def _clamp_kernel(self, force: ti.template(), dt: ti.f64):
        for p in force:
            for m in ti.static(range(3)):
                ti.atomic_add(self.acc_b[B_CLAMP, m], ti.cast(force[p][m], ti.f64) * dt)

    @ti.kernel
    def _points_kernel(self, force: ti.template(), slot: ti.i32, dt: ti.f64):
        for p in force:
            for m in ti.static(range(3)):
                ti.atomic_add(self.acc_points[slot, m], ti.cast(force[p][m], ti.f64) * dt)

    @ti.kernel
    def _time_kernel(self, dt: ti.f64):
        self.acc_time[None] += dt

    @ti.kernel
    def _momentum_kernel(self):
        for b, m in ti.ndrange(self.momentum.shape[0], 3):
            self.momentum[b, m] = 0.0
        nx, ny, nz = ti.static(self.n)
        for i, j, k in ti.ndrange(nx, ny, nz):
            if self._water(i, j, k):
                mom = ti.cast(self.rho, ti.f64) * self._uc(i, j, k) * (self.width[0][i] * self.width[1][j]
                                                                       * self.width[2][k])
                for box in range(self.box_range.shape[0]):
                    if (self.box_range[box, 0, 0] <= i < self.box_range[box, 0, 1]
                            and self.box_range[box, 1, 0] <= j < self.box_range[box, 1, 1]
                            and self.box_range[box, 2, 0] <= k < self.box_range[box, 2, 1]):
                        for m in ti.static(range(3)):
                            ti.atomic_add(self.momentum[box, m], mom[m])

    @ti.kernel
    def _partner_momentum_kernel(self, mass: ti.template(), velocity: ti.template(), slot: ti.i32):
        for p in mass:
            for m in ti.static(range(3)):
                ti.atomic_add(self.momentum[slot, m], mass[p] * velocity[p][m])

    @ti.kernel
    def _reset_kernel(self):
        for idx in ti.grouped(self.acc_a):
            self.acc_a[idx] = 0.0
        for idx in ti.grouped(self.acc_b):
            self.acc_b[idx] = 0.0
        for idx in ti.grouped(self.acc_points):
            self.acc_points[idx] = 0.0
        for b in self.cut:
            self.cut[b] = 0
        self.acc_time[None] = 0.0

    @ti.kernel
    def _shifted_pressure_kernel(self, out: ti.template(), shift: ti.f64):
        for idx in ti.grouped(out):
            out[idx] = ti.cast(self.fluid.pressure[idx], ti.f64) + shift

    # ------------------------------------------------------------------------------------------------ host API
    def _momenta(self):
        self._momentum_kernel()
        if self.excess is not None:
            self._partner_momentum_kernel(self.excess.mass, self.excess.velocity, len(self.boxes))
        return self.momentum.to_numpy()

    def begin(self, t_s):
        """Start the first interval at time t_s (after the initial state or a restart's load)."""
        self._reset_kernel()
        self._p_start = self._momenta()
        self._t_start = float(t_s)

    def pressure_field(self, p_shift_pa):
        """The solver's pressure plus a uniform shift (physical gauge pressure where no pocket differs), into an own
        field."""
        if self._own_pressure is None:
            self._own_pressure = ti.field(ti.f64, shape=self.n)
        self._shifted_pressure_kernel(self._own_pressure, float(p_shift_pa))
        return self._own_pressure

    def add_point_forces(self, slot, dt_s, force=None):
        """Add sum(force) dt to slot `slot` (a name of point_slots); force: a point vector field (default the
        constructor's point_forces, e.g. points.f_extra right after the contact has added to it)."""
        self._points_kernel(self.point_forces if force is None else force, self.slots.index(slot), float(dt_s))

    def accumulate(self, dt_s, pressure):
        """One step: F dt of the surface (A), traction and clamp (B) terms, the pressure being the physical gauge
        pressure field of the step (pressure_field(), or a gas-region chamber's physical_pressure_field())."""
        dt = float(dt_s)
        self._surface_kernel(pressure, dt)
        self._traction_kernel(pressure, dt)
        if self.clamp is not None:
            self._clamp_kernel(self.clamp.point_force, dt)
        self._time_kernel(dt)

    def report(self, t_s):
        """The interval means since begin() or the last report (one host read of the accumulators and the momenta),
        then start the next interval."""
        if self._p_start is None:
            raise RuntimeError("begin() was not called")
        p_end = self._momenta()
        acc_a, acc_b = self.acc_a.to_numpy(), self.acc_b.to_numpy()
        acc_points, cut = self.acc_points.to_numpy(), self.cut.to_numpy()
        span = float(self.acc_time.to_numpy())
        out = dict(t_start_s=self._t_start, t_end_s=float(t_s), interval_s=span, boxes={}, b={}, points={})
        if span <= 0.0:
            self.begin(t_s)
            return out
        partners = (p_end[-1] - self._p_start[-1]) / span
        for n, box in enumerate(self.boxes):
            dpdt = (p_end[n] - self._p_start[n]) / span + partners
            parts = {name: acc_a[n, q] / span for q, name in enumerate(A_PARTS)}
            bracket = dpdt + parts["flux"] + parts["pressure"] - parts["viscous"] + parts["wall_pressure"] \
                + parts["wall_shear"]
            out["boxes"][box["name"]] = dict(force_n=(-bracket).tolist(), momentum_rate_n=dpdt.tolist(),
                                             parts_n={k: v.tolist() for k, v in parts.items()},
                                             cut_faces=int(cut[n]))
        out["b"] = {name: (acc_b[q] / span).tolist() for q, name in enumerate(B_PARTS)}
        out["points"] = {name: (acc_points[s] / span).tolist() for s, name in enumerate(self.slots)}
        self._reset_kernel()
        self._p_start, self._t_start = p_end, float(t_s)
        return out
