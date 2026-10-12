"""REQ_401 FAST MODE: graded-grid IB_4 kernel (G1 + first-order moment correction) and MAC face-validity mask
(Helper 1, 2026-10-12; contract 01 §1.1, §1.2, design note FAST_MODE_DESIGN_NOTE_01).

GradedAxes is the axes object of the production CartesianFluidSolver's graded MAC grid (the same API as
fluid_hooks.UniformAxes: n, min_spacing_m, axis_weights, face_volume).  FaceMask holds the one-sided normalisation
validity of every MAC face (the body applies w~ = w m / D through face_valid).

G1, the index-space graded IB_4 kernel.  Along axis a, for one node family (staggered = 1: the faces faces_a[i], i.e.
the MAC component's own axis; staggered = 0: the cell centres), a physical coordinate is mapped to the node-index
coordinate xi by the monotone piecewise-linear map through the nodes, xi(node_i) = i.  The map is extended by two ghost
nodes per side at the edge cell width (mirrored ghost cells) and linearly beyond.  The four nodes i0 = floor(xi) - 1 ..
i0 + 3 get the index-space weights

    w_q = phi(xi - (i0 + q)),   dw_q = phi'(xi - (i0 + q)) * dxi/dx,   dxi/dx = 1 / (local node spacing)   [1/m]

with phi = IB_4 (ib_mpm_fast.ib4_phi / ib4_dphi, imported, not copied).

First-order moment correction (graded axes, default on; Root / integrator 2026-10-12).  On a stretched axis the
index-space weights do not reproduce linear fields in physical space: the interpolated gradient of u = a x has a
relative error up to ~(r - 1) / 2 for an adjacent-cell stretch ratio r (measured without the correction: 0.025 / 0.050 /
0.100 at r = 1.05 / 1.1 / 1.2).  Per axis and node family the four weights are therefore corrected, reproducing-kernel
style, as w~_q = w_q (alpha + beta d_q) with d_q = (x_q - x) / s (s the local node spacing), where alpha, beta solve the
2 x 2 moment system

    sum_q w~_q = 1,   sum_q w~_q (x_q - x) = 0,

and dw~_q is the EXACT physical derivative d w~_q / dx (alpha, beta and d_q differentiated), which then satisfies
sum_q dw~_q = 0 and sum_q dw~_q (x_q - x) = 1 identically.  So the interpolated value and gradient of a linear field are
exact on any grading (per axis; tensor products are exact in 3-D away from invalid faces), the derivative weights are the
gradient of the corrected weights (variational consistency of the spread stress force and the interpolated velocity
gradient), and force conservation, the power identity and the internal-force balance (sum dw~ = 0) still hold because the
body uses the same w~, dw~ to spread and to gather.  The correction uses all four nodes with their physical (ghost)
positions before out-of-range nodes are zeroed; the domain edge and invalid faces are left to the body's one-sided
normalisation.  On a uniform axis the moments already hold, so no correction is applied and the contract's uniform
reduction is used verbatim: xi = (coord - origin) / h - (0 if staggered else 0.5), dw = phi' / h, the same expressions as
fluid_hooks.UniformAxes (bit-for-bit equal, tested).  GradedAxes(..., moment_correction=False) gives raw G1.

Tested (tests/fast/test_graded_kernel_cpu.py): the moments above; dw = central differences of w (physical derivative);
the linear-field value and gradient table at r = 1.0 / 1.05 / 1.1 / 1.2 (raw within the summation-by-parts bound,
corrected at f32 roundoff); min w~ and sum |w~| (boundedness) at r <= 1.2; T7 items 1-2 (force conservation, power
identity) for point force and stress term on uniform and graded grids and with one-sided normalisation next to an
obstacle, air cells and the domain edge.  stretch_in_box() reports r around the body for the run metadata.

Root review items (2026-10-12), measured:
  1. linear reproduction: value and gradient exact to f32 roundoff on any grading; a rigid rotation sampled on the MAC
     faces of a grid with 1.2 band transitions gives max |G - W| = 6.1e-7, max |sym G| = 5.2e-7 for |omega| = 0.91
     (raw G1: 5.0e-2 / 2.5e-2), i.e. zero strain;
  2. force conservation (sum w~ = 1) and the power identity hold with the same corrected w~, dw~ in spread and gather;
  3. interior fast path of the integrator's body: for every coordinate whose four nodes are stored, sum w~ = 1 and
     sum dw~ = 0 per axis (to roundoff), so D = 1 and grad D = 0 away from invalid faces and the unnormalised fast path
     equals the one-sided path (tests/fast/test_graded_body_fast_path_cpu.py, FastIbMpm3D on these axes).  Nodes beyond
     the FAR domain faces (index n_a, not stored) are out of range, so there D < 1: the body's support flags must
     cover the cells within the support reach of the far faces as well (xfail test in that file);
  4. no grid locking: w~ is Lipschitz-continuous in x (a 10x finer path shrinks the largest per-step change of every
     nodal weight and of an interpolated smooth field ~9.4-10x at r <= 1.2); min w~ >= 0 at every tested position and
     ratio (nonnegative, hence sum |w~| = 1).  Because the index map is piecewise linear, dw~ has a kink where the node
     spacing changes: max |dw(x+) - dw(x-)| / max |dw| = 0.010 / 0.017 / 0.039 at r = 1.05 / 1.1 / 1.2 (raw G1:
     0.048 / 0.091 / 0.166; 0.008 is the probe-offset floor; h1/tools/measure_dw_jump_01.py), i.e. the spread stress
     force kinks by ~3 % at node crossings at r = 1.2 (a C1 index map would remove it).  The IB_4 identity
     sum w^2 = 3/8 deviates by up to 0.8 % at r = 1.2 (0.2 % at 1.05).

f32 precision: node coordinates are stored relative to the f32-rounded grid origin, so the map adds no error beyond the
coordinate's own f32 rounding.  An absolute f32 position near 1 m (squid z ~ 1.01 m) carries an ulp of 1.2e-7 m
(2e-4 of a 0.6 mm cell); the integrator integrates the displacement instead (contract note 2026-10-12).

face_volume(c, i, j, k): centre distance along c (half the edge cell at index 0) times the cell widths along the other
two axes, i.e. the MAC momentum control volume of that face.

FaceMask (contract §1.2): a MAC face of component c at (i, j, k) (between cells (i, j, k) - e_c and (i, j, k)) is valid
iff neither adjacent cell is an obstacle (fluid.obstacle != 0; the solver uses 0/1), the two cells are not both air
cells, and the face is not the domain-edge face (index 0 along c).  The bits are stored as one u8 per cell (bit c).

Reuse check (REQ_400), paths looked at (read-only local copies of the upstream sources, reference_libs/):
- ibamr-v0.19.0/ibtk/src/lagrangian/LEInteractor.cpp (side-centred interpolate()/spread(): each MAC component treated
  as cell-centred data shifted by half a cell along its own axis): REUSED AS ALGORITHM, the staggered / centred node
  index map of axis_weights.  IBAMR patches are uniform (refinement is AMR), so there is no graded-grid kernel to reuse.
- ibamr-v0.19.0/ibtk/src/lagrangian/fortran/lagrangian_delta.f.m4 (lagrangian_ib_4_delta) and
  lagrangian_interaction3d.f.m4 (lagrangian_ib_4_spread3d / interp3d, the closed-form four weights from one sqrt):
  the IB_4 function itself is the existing ib_mpm_fast.ib4_phi / ib4_dphi (contract: import, do not copy); IBAMR's
  1 / (dx dy dz) spread scaling is the 1 / face_volume of the body's spread.
- Moment correction: ALGORITHM from A. Pinelli, I. Z. Naqavi, U. Piomelli, J. Favier, "Immersed-boundary methods for
  general finite-difference and finite-volume Navier-Stokes solvers", J. Comput. Phys. 229 (2010) 9073-9091
  (reproducing-kernel correction of the delta weights on non-uniform grids); no code copied (none in reference_libs).
  The exact-derivative weights are our own construction.
- IBAMR has no one-sided (masked, renormalised) delta next to solid cells; its physical-boundary handling folds ghost
  values back (LEInteractor / IBTK ghost accumulation), so FaceMask is new code.  SAMRAI / libMesh not used (no LGPL).
IBAMR is copyright (c) 2011 - 2026 by the IBAMR developers, BSD-3-Clause; no IBAMR source is copied here.

No `from __future__ import annotations` (Taichi kernel module).
"""
import math

import numpy as np
import taichi as ti

from simulation_core.fast.ib_mpm_fast import ib4_dphi, ib4_phi

GRADED_KERNEL_REVISION = "REQ401-fast-graded-ib4-g1-moment-corrected-facemask-v2"
HARD_STOP_CODES = {"one_sided_empty_support"}
UNIFORM_RTOL = 1.0e-12               # the uniformity test of fluids.grid.CartesianGrid._axis_is_uniform
NONFINITE_INDEX_COORDINATE = -100.0  # a non-finite / absurd coordinate maps here: every node out of range, w = 0
FINITE_INDEX_BOUND = 1.0e7           # |xi| at or beyond this (or NaN) is treated as non-finite
GHOST_NODES = 2                      # mirrored ghost nodes per side in the device node tables (covers the 4-node support)


class FastModeHardStop(RuntimeError):
    """A fast-mode hard stop with its machine-readable code (one of HARD_STOP_CODES) and witness."""

    def __init__(self, code, witness):
        if code not in HARD_STOP_CODES:
            raise ValueError("unknown fast-mode hard-stop code %r" % (code,))
        super().__init__("%s: %s" % (code, witness))
        self.code = code
        self.witness = witness


def _validated_faces(faces, name):
    values = np.asarray(faces, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("faces_%s must be a 1-D array of at least two face coordinates" % name)
    if not np.all(np.isfinite(values)):
        raise ValueError("faces_%s must be finite" % name)
    if not np.all(np.diff(values) > 0.0):
        raise ValueError("faces_%s must be strictly increasing" % name)
    return values


def _axis_is_uniform(widths):
    first = float(widths[0])
    return all(math.isclose(float(w), first, rel_tol=UNIFORM_RTOL, abs_tol=0.0) for w in widths)


@ti.func
def first_order_moment_correction(w, dw, d, spacing):
    """Reproducing-kernel correction of four 1-D weights (Pinelli et al. 2010 algorithm): w~ = w (alpha + beta d) with
    sum w~ = 1 and sum w~ d = 0, d_q = (x_q - x) / spacing; dw~ is the exact x-derivative (d d_q / dx = -1 / spacing),
    so sum dw~ = 0 and sum dw~ (x_q - x) = 1.  w, dw, d: Vector(4); dw in 1/m."""
    inv_s = 1.0 / spacing
    m0 = w.sum()
    m1 = (w * d).sum()
    m2 = (w * d * d).sum()
    dm0 = dw.sum()
    dm1 = (dw * d).sum() - m0 * inv_s
    dm2 = (dw * d * d).sum() - 2.0 * m1 * inv_s
    det = m0 * m2 - m1 * m1
    ddet = dm0 * m2 + m0 * dm2 - 2.0 * m1 * dm1
    alpha = m2 / det
    beta = -m1 / det
    dalpha = (dm2 - alpha * ddet) / det
    dbeta = (-dm1 - beta * ddet) / det
    shape = alpha + beta * d
    return w * shape, dw * shape + w * (dalpha + dbeta * d - beta * inv_s)


@ti.data_oriented
class GradedAxes:
    """IB_4 axes object of a (possibly) graded MAC grid given by its face coordinates (m, float64, n + 1 per axis)."""

    def __init__(self, faces_x, faces_y, faces_z, *, moment_correction=True):
        faces = tuple(_validated_faces(f, name) for f, name in zip((faces_x, faces_y, faces_z), "xyz"))
        widths = tuple(np.diff(f) for f in faces)
        centres = tuple(0.5 * (f[1:] + f[:-1]) for f in faces)
        self.n = tuple(int(len(w)) for w in widths)
        self.min_spacing_m = float(min(float(w.min()) for w in widths))
        self.moment_correction = bool(moment_correction)
        self.faces_m = faces                 # host copies (float64), read by other modules (air chamber, valve)
        self.centres_m = centres
        self.widths_m = widths
        self.origin = tuple(float(f[0]) for f in faces)
        self.uniform = tuple(_axis_is_uniform(w) for w in widths)
        self.h = tuple(float((f[-1] - f[0]) / (len(f) - 1)) for f in faces)    # used on uniform axes only
        self.max_adjacent_stretch = tuple(
            float(np.max(np.maximum(w[1:] / w[:-1], w[:-1] / w[1:]))) if len(w) > 1 else 1.0 for w in widths)
        length = max(self.n) + 2 * GHOST_NODES + 1
        nodes = np.full((3, 2, length), np.float32(3.0e38), dtype=np.float32)
        spacing = np.ones((3, 2, length), dtype=np.float32)
        counts = []
        for a in range(3):
            f, w, c = faces[a], widths[a], centres[a]
            origin32 = float(np.float32(self.origin[a]))      # the device subtracts the f32-rounded origin
            ghosts = np.arange(GHOST_NODES, 0, -1, dtype=np.float64)
            families = (np.concatenate([c[0] - ghosts * w[0], c, c[-1] + ghosts[::-1] * w[-1]]),   # 0: centres
                        np.concatenate([f[0] - ghosts * w[0], f, f[-1] + ghosts[::-1] * w[-1]]))   # 1: faces
            for s, family in enumerate(families):
                nodes[a, s, :len(family)] = (family - origin32).astype(np.float32)
                spacing[a, s, :len(family) - 1] = np.diff(family).astype(np.float32)
            counts.append((len(families[0]), len(families[1])))
        self._count = tuple(counts)
        face_distance = np.ones((3, max(self.n)), dtype=np.float32)
        cell_width = np.ones((3, max(self.n)), dtype=np.float32)
        for a in range(3):
            distance = np.concatenate([[0.5 * widths[a][0]], np.diff(centres[a])])
            face_distance[a, :self.n[a]] = distance.astype(np.float32)
            cell_width[a, :self.n[a]] = widths[a].astype(np.float32)
        self._nodes = ti.field(ti.f32, shape=(3, 2, length))
        self._spacing = ti.field(ti.f32, shape=(3, 2, length))
        self._face_distance = ti.field(ti.f32, shape=(3, max(self.n)))
        self._cell_width = ti.field(ti.f32, shape=(3, max(self.n)))
        self._nodes.from_numpy(nodes)
        self._spacing.from_numpy(spacing)
        self._face_distance.from_numpy(face_distance)
        self._cell_width.from_numpy(cell_width)

    @classmethod
    def from_fluid(cls, fluid, *, moment_correction=True):
        """Axes of fluid.grid (CartesianGrid: bounds_min_m + cell_widths_{x,y,z}_m)."""
        grid = fluid.grid
        return cls(grid.cell_faces_x_m, grid.cell_faces_y_m, grid.cell_faces_z_m,
                   moment_correction=moment_correction)

    def stretch_in_box(self, lo_m, hi_m, margin_cells=2):
        """Largest adjacent-cell width ratio per axis among the cells meeting the box [lo_m, hi_m] widened by
        `margin_cells` (the IB_4 support reaches 2 cells).  With the moment correction the linear-field value and
        gradient are exact for any ratio; raw G1 has a gradient error up to ~(ratio - 1) / 2 there."""
        ratios = []
        for a in range(3):
            faces, widths = self.faces_m[a], self.widths_m[a]
            first = max(int(np.searchsorted(faces, float(lo_m[a]), side="right")) - 1 - int(margin_cells), 0)
            last = min(int(np.searchsorted(faces, float(hi_m[a]), side="left")) + int(margin_cells), self.n[a])
            window = widths[first:last]
            ratios.append(float(np.max(np.maximum(window[1:] / window[:-1], window[:-1] / window[1:])))
                          if len(window) > 1 else 1.0)
        return tuple(ratios)

    def identity(self):
        """JSON-able description (revision, sizes, uniform flags, stretch ratios) for run metadata."""
        return dict(revision=GRADED_KERNEL_REVISION, n=list(self.n), min_spacing_m=self.min_spacing_m,
                    uniform=list(self.uniform), moment_correction=self.moment_correction,
                    max_adjacent_stretch=list(self.max_adjacent_stretch), origin_m=list(self.origin),
                    extent_m=[float(f[-1] - f[0]) for f in self.faces_m])

    # ------------------------------------------------------------------------------------------------ device API
    @ti.func
    def _graded_index_coordinate(self, a: ti.template(), family: ti.template(), r):
        """(xi, local node spacing) of the origin-relative coordinate r on a graded axis (binary search)."""
        count = ti.static(self._count[a][family])
        lo = 0
        hi = count - 1
        if r < self._nodes[a, family, 0]:
            hi = 1                                   # below the first ghost node: extend the first segment
        elif r >= self._nodes[a, family, count - 1]:
            lo = count - 2                           # beyond the last ghost node: extend the last segment
        else:
            while hi - lo > 1:                       # nodes[lo] <= r < nodes[hi]
                middle = (lo + hi) // 2
                if self._nodes[a, family, middle] <= r:
                    lo = middle
                else:
                    hi = middle
        local = self._spacing[a, family, lo]
        xi = ti.cast(lo - GHOST_NODES, ti.f32) + (r - self._nodes[a, family, lo]) / local
        return xi, local

    @ti.func
    def _node_offset(self, a: ti.template(), family: ti.template(), i, r):
        """x_i - x (m) for node index i of a family, x = r + origin (linear extrapolation beyond the ghost nodes)."""
        count = ti.static(self._count[a][family])
        g = i + GHOST_NODES
        clamped = ti.max(0, ti.min(g, count - 1))
        position = self._nodes[a, family, clamped]
        if g < 0:
            position -= ti.cast(-g, ti.f32) * self._spacing[a, family, 0]
        elif g > count - 1:
            position += ti.cast(g - (count - 1), ti.f32) * self._spacing[a, family, count - 2]
        return position - r

    @ti.func
    def index_coordinate(self, a: ti.template(), staggered: ti.template(), coord):
        """(xi, local node spacing in m) of the physical coordinate `coord` along axis a for the node family
        `staggered` (1: faces, 0: centres).  A NaN / infinite / absurd xi is replaced by NONFINITE_INDEX_COORDINATE,
        where every node is out of range (zero weights, hence the body's D = 0 hard stop)."""
        family = ti.static(1 if staggered else 0)
        xi = 0.0
        local = 1.0
        if ti.static(self.uniform[a]):
            xi = (coord - self.origin[a]) / self.h[a] - (0.0 if staggered else 0.5)
            local = self.h[a]
        else:
            xi, local = self._graded_index_coordinate(a, family, coord - self.origin[a])
        if not (ti.abs(xi) < FINITE_INDEX_BOUND):
            xi = NONFINITE_INDEX_COORDINATE
        return xi, local

    @ti.func
    def axis_weights(self, a: ti.template(), staggered: ti.template(), coord):
        """IB_4 weights of the 4 nodes along axis a around physical coordinate `coord` (contract §1.1).
        Returns (i0: i32, w: Vector(4, f32), dw: Vector(4, f32)) for nodes i0 + 0..3, dw = dw/dcoord in 1/m (the
        physical derivative); moment-corrected on graded axes; nodes outside 0 <= i < n_a are returned with w = dw = 0."""
        i0 = 0
        w = ti.Vector([0.0, 0.0, 0.0, 0.0])
        dw = ti.Vector([0.0, 0.0, 0.0, 0.0])
        if ti.static(self.uniform[a]):
            # the contract's uniform reduction, written exactly as fluid_hooks.UniformAxes (compile-time h)
            s = (coord - self.origin[a]) / self.h[a] - (0.0 if staggered else 0.5)
            if not (ti.abs(s) < FINITE_INDEX_BOUND):
                s = NONFINITE_INDEX_COORDINATE
            i0 = ti.cast(ti.floor(s), ti.i32) - 1
            for q in ti.static(range(4)):
                i = i0 + q
                if 0 <= i < self.n[a]:
                    r = s - ti.cast(i, ti.f32)
                    w[q] = ib4_phi(r)
                    dw[q] = ib4_dphi(r) / self.h[a]
        else:
            family = ti.static(1 if staggered else 0)
            relative = coord - self.origin[a]
            xi, local = self._graded_index_coordinate(a, family, relative)
            if not (ti.abs(xi) < FINITE_INDEX_BOUND):
                xi = NONFINITE_INDEX_COORDINATE
            i0 = ti.cast(ti.floor(xi), ti.i32) - 1
            for q in ti.static(range(4)):
                r = xi - ti.cast(i0 + q, ti.f32)
                w[q] = ib4_phi(r)
                dw[q] = ib4_dphi(r) / local
            if ti.static(self.moment_correction):
                d = ti.Vector([0.0, 0.0, 0.0, 0.0])
                for q in ti.static(range(4)):
                    d[q] = self._node_offset(a, family, i0 + q, relative) / local
                w, dw = first_order_moment_correction(w, dw, d, local)
            for q in ti.static(range(4)):
                i = i0 + q
                if i < 0 or i >= self.n[a]:
                    w[q] = 0.0
                    dw[q] = 0.0
        return i0, w, dw

    @ti.func
    def face_volume(self, c: ti.template(), i, j, k):
        """Control volume (m^3) of the MAC face of component c at (i, j, k): centre distance along c (half the edge
        cell at index 0) times the cell widths along the other two axes."""
        index = ti.Vector([i, j, k])
        volume = self._face_distance[c, index[c]]
        for a in ti.static(range(3)):
            if ti.static(a != c):
                volume *= self._cell_width[a, index[a]]
        return volume


@ti.data_oriented
class FaceMask:
    """One-sided normalisation validity of every MAC face (contract §1.2), from fluid.obstacle and the air cells."""

    def __init__(self, fluid, axes):
        shape = tuple(int(v) for v in fluid.obstacle.shape)
        if shape != tuple(int(v) for v in axes.n):
            raise ValueError("fluid grid %s does not match the axes %s" % (shape, tuple(axes.n)))
        self.fluid = fluid
        self.n = shape
        self.bits = ti.field(ti.u8, shape=shape)          # bit c set: the face of component c at (i, j, k) is valid
        self.valid_count = ti.field(ti.i32, shape=3)
        self.with_air = False
        self.refresh()

    @ti.kernel
    def _refresh_kernel(self, obstacle: ti.template(), air: ti.template(), use_air: ti.template()):
        for c in ti.static(range(3)):
            self.valid_count[c] = 0
        for i, j, k in obstacle:
            index = ti.Vector([i, j, k])
            bits = 0
            if obstacle[i, j, k] == 0:
                for c in ti.static(range(3)):
                    if index[c] >= 1:                               # index 0 along c: the domain-edge face
                        neighbour = index - ti.Vector.unit(3, c, ti.i32)
                        if obstacle[neighbour] == 0:
                            valid = 1
                            if ti.static(use_air):
                                if air[i, j, k] != 0 and air[neighbour] != 0:
                                    valid = 0
                            if valid != 0:
                                bits |= 1 << c
                                ti.atomic_add(self.valid_count[c], 1)
            self.bits[i, j, k] = ti.cast(bits, ti.u8)

    def refresh(self, air_cell_field=None):
        """Recompute the validity bits; call after any obstacle or air-cell change."""
        if air_cell_field is None:
            self.with_air = False
            self._refresh_kernel(self.fluid.obstacle, self.fluid.obstacle, False)
            return
        if tuple(int(v) for v in air_cell_field.shape) != self.n:
            raise ValueError("air cell field %s does not match the grid %s" % (tuple(air_cell_field.shape), self.n))
        self.with_air = True
        self._refresh_kernel(self.fluid.obstacle, air_cell_field, True)

    @ti.func
    def face_valid(self, c: ti.template(), i, j, k):
        """1 if the MAC face of component c at (i, j, k) may carry IB weight, else 0 (i32)."""
        return (ti.cast(self.bits[i, j, k], ti.i32) >> c) & 1

    def counts(self):
        """Valid faces per component after the last refresh (diagnostics)."""
        values = self.valid_count.to_numpy()
        return dict(valid_faces=[int(v) for v in values], with_air=bool(self.with_air))
