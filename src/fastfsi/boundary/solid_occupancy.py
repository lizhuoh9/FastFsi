"""Occupancy of a deformable body on a box of the fluid grid: own-F boxes united with the untangled reference-lattice
cells, with coverage fractions and a time hysteresis of the cell topology.

MATERIAL.  The union of
  (1) every body point's OWN-F BOX: x + F (xi o s) with |xi_i| <= 1/2, its reference lattice box (sides s = its
      builder spacing) mapped by its own deformation gradient F;
  (2) every REFERENCE-LATTICE CELL whose 8 corners are body points, built from the corners' CURRENT positions and split
      into the 6 Kuhn tetrahedra 000 -> e_a -> e_a + e_b -> 111 (one per axis permutation).  Every cell uses the same
      split, so neighbouring cells share their face triangles and the union of the cells is watertight while none
      inverts.  A cell is LEFT OUT when its geometric volume ratio J_cell = sum(current) / sum(|reference|) over its
      tetrahedra exceeds stretch_factor x the mean F-based J of its 8 corners: a lattice cell that much larger than its
      points' own material volume has lost its neighbours and is not body volume.  Inverted cells STAY: they are a
      folded sheet, and leaving them out re-opens the fold.

SAMPLING.  Each fluid cell of the box is split into SUB^3 equal sub-cells; a sub-cell is HIT when its centre lies inside
a box or a kept tetrahedron (local coordinates within INSIDE_TOLERANCE).  The hits of a cell are the bits of
ceil(SUB^3 / 32) i32 words (atomic or), so a sub-cell hit by several primitives counts once.  Per cell
  coverage  = hit sub-cells / SUB^3 (the union's volume fraction by centre sampling: unbiased, quantum 1 / SUB^3 of the
              cell; a smooth ramp per primitive is NOT used, because the maximum of per-primitive ramps under-counts the
              sub-cells on the shared internal faces of the tiling primitives);
  material  = a hit sub-cell, OR one of the body points' box samples (SAMPLES^3 per point on the box's cell-centred
              sub-grid, the point itself included) lies in the cell; every point's own cell is therefore material.
Distances and local coordinates are taken relative to each primitive's own origin (differences in f64, then f32).

TOPOLOGY HYSTERESIS (time).  topo = 1 at once when the cell is material; a topo cell turns free only after it has not
been material for `hysteresis_steps` consecutive updates (age counter).  One-cell flicker then never splits, merges or
reconnects air regions; a real gap must stay open for that many accepted steps.  `flipped` counts the cells whose topo
changed in the last update: 0 lets a caller reuse its last labelling.

Per update(): the clear, box, lattice and reduce kernels and ONE host read (the statistics).  The topology, material and
coverage arrays are read only on request.  Checkpoint state: the hysteresis fields (topo, age); everything else is
rewritten by every update().  Setup refuses non-positive spacings and lattice cells whose corners are not lattice
neighbours (consecutive coordinates must be (s_a + s_b) / 2 apart; such cells are counted and left out).

Design: the rule (own-F boxes united with the untangled lattice cells, stretched cells dropped at a factor 3) was
chosen by replaying stored deformed states with the factors 2, 3 and 5.

No `from __future__ import annotations` (Taichi kernel module).
"""
import itertools
import math

import numpy as np
import taichi as ti

from fastfsi.io.checkpoint import Checkpointable

DEFAULT_SUB = 3
DEFAULT_STRETCH_FACTOR = 3.0
DEFAULT_BOX_SAMPLES = 5
DEFAULT_HYSTERESIS_STEPS = 5
INSIDE_TOLERANCE = 1.0e-6          # local-coordinate tolerance (f32): a centre on a shared face belongs to both primitives
DEGENERATE_RELATIVE = 1.0e-12      # |current volume| below this x |reference volume|: the tetrahedron is skipped
KEY_QUANTUM_M = 1.0e-8             # reference lattice coordinates are identified to 10 nm
AABB_MARGIN_M = 1.0e-9             # candidate boxes grow by this much: a centre exactly on a face is never lost to rounding
ADJACENCY_RTOL = 1.0e-6            # consecutive lattice coordinates must be (s_a + s_b) / 2 apart to this tolerance
# statistics slots (one f64 field, read once per update)
S_BOXES_SINGULAR, S_CELLS_DROPPED, S_CELLS_INVERTED, S_TETS_DEGENERATE = 0, 1, 2, 3
S_MATERIAL_CELLS, S_TOPO_CELLS, S_FLIPPED, S_UNION_VOLUME, S_V0J = 4, 5, 6, 7, 8
STATS_SLOTS = 9


def kuhn_tetrahedra():
    """The 6 Kuhn tetrahedra of a cell as corner numbers c = 4 di + 2 dj + dk (paths 000 -> e_a -> e_a + e_b -> 111)."""
    tets = []
    for perm in itertools.permutations(range(3)):
        corner, path = [0, 0, 0], [0]
        for axis in perm:
            corner[axis] = 1
            path.append(4 * corner[0] + 2 * corner[1] + corner[2])
        tets.append(tuple(path))
    return tuple(tets)


TETS = kuhn_tetrahedra()


def reference_lattice_cells(reference_positions_m, spacing_m):
    """Host setup: the reference-lattice cells whose 8 corners are points.  Returns (cells (H, 8) int64 of point numbers,
    corner c = 4 di + 2 dj + dk, and the count of candidate cells refused because two consecutive lattice coordinates
    are not (s_a + s_b) / 2 apart).  Points must sit on a tensor-product lattice (one point per lattice index)."""
    positions = np.asarray(reference_positions_m, dtype=np.float64)
    spacing = np.asarray(spacing_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or spacing.shape != positions.shape:
        raise ValueError("reference positions and spacing must both be (n, 3)")
    if not (np.isfinite(positions).all() and np.isfinite(spacing).all() and (spacing > 0.0).all()):
        raise ValueError("reference positions must be finite and spacings finite and positive")
    keys = np.round(positions / KEY_QUANTUM_M).astype(np.int64)
    unique = [np.unique(keys[:, axis]) for axis in range(3)]
    index = np.stack([np.searchsorted(unique[axis], keys[:, axis]) for axis in range(3)], axis=1)
    table = np.full(tuple(len(u) for u in unique), -1, dtype=np.int64)
    table[index[:, 0], index[:, 1], index[:, 2]] = np.arange(len(positions))
    if int((table >= 0).sum()) != len(positions):
        raise ValueError("two points share a reference lattice index")
    nx, ny, nz = table.shape
    corners = np.stack([table[di:di + nx - 1, dj:dj + ny - 1, dk:dk + nz - 1].ravel()
                        for di, dj, dk in itertools.product((0, 1), repeat=3)], axis=1)
    corners = corners[np.all(corners >= 0, axis=1)]
    adjacent = np.ones(len(corners), dtype=bool)
    for axis, (low, high) in enumerate(((0, 4), (0, 2), (0, 1))):
        gap = positions[corners[:, high], axis] - positions[corners[:, low], axis]
        want = 0.5 * (spacing[corners[:, low], axis] + spacing[corners[:, high], axis])
        adjacent &= np.abs(gap - want) <= ADJACENCY_RTOL * want
    return corners[adjacent], int((~adjacent).sum())


def _subdivide(faces, sub):
    faces = np.asarray(faces, dtype=np.float64)
    fractions = np.arange(sub) / sub
    return np.append((faces[:-1, None] + np.diff(faces)[:, None] * fractions[None, :]).ravel(), faces[-1])


@ti.data_oriented
class SolidOccupancy(Checkpointable):
    """Material, coverage and topology of one deformable body on a box of the fluid grid (see the module docstring)."""

    # the topology hysteresis, and what a refresh that skips the rasterisation (a gas-region raster interval > 1)
    # reuses: the coverage, the column sums and the last statistics, so a restart continues the cadence bit for bit
    CHECKPOINT_FIELDS = ("topo", "age", "coverage_field", "column_mass_field", "column_ke_field")
    CHECKPOINT_HOST = ("updates", "last")

    def __init__(self, axes, box_cells, points, point_indices, reference_positions_m, spacing_m, *, sub=DEFAULT_SUB,
                 stretch_factor=DEFAULT_STRETCH_FACTOR, box_samples=DEFAULT_BOX_SAMPLES,
                 hysteresis_steps=DEFAULT_HYSTERESIS_STEPS, point_mass_kg=None):
        faces = tuple(np.asarray(f, dtype=np.float64) for f in axes.faces_m)
        ranges = tuple((int(lo), int(hi)) for lo, hi in box_cells)
        if len(ranges) != 3 or any(not 0 <= lo < hi <= len(f) - 1 for (lo, hi), f in zip(ranges, faces)):
            raise ValueError("box_cells must be three half-open cell ranges inside the grid, got %r" % (box_cells,))
        sub, samples, steps = int(sub), int(box_samples), int(hysteresis_steps)
        factor = float(stretch_factor)
        if not 1 <= sub <= 6 or samples < 1 or samples % 2 != 1 or steps < 0:
            raise ValueError("1 <= sub <= 6, an odd box_samples >= 1 and hysteresis_steps >= 0 are required")
        if not (math.isfinite(factor) and factor > 1.0):
            raise ValueError("stretch_factor must be finite and > 1")
        indices = np.asarray(point_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0 or indices.min() < 0 or indices.max() >= points.n:
            raise ValueError("point_indices must be a non-empty list of valid point numbers")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("point_indices must be distinct")
        spacing = np.asarray(spacing_m, dtype=np.float64)
        reference = np.asarray(reference_positions_m, dtype=np.float64)
        if reference.shape != (indices.size, 3):
            raise ValueError("reference_positions_m must be (len(point_indices), 3)")
        cells, refused = reference_lattice_cells(reference, spacing)
        self.box_cells = ranges
        self.shape = tuple(hi - lo for lo, hi in ranges)
        self.sub = sub
        self.words = (sub ** 3 + 31) // 32
        self.stretch_factor = factor
        self.box_samples = samples
        self.hysteresis_steps = steps
        self.points = points
        self.n_body = int(indices.size)
        self.n_cells = int(len(cells))
        self.refused_non_adjacent = refused
        self.updates = 0
        cell_faces = [f[lo:hi + 1] for (lo, hi), f in zip(ranges, faces)]
        sub_faces = [_subdivide(f, sub) for f in cell_faces]
        self.cell_faces_m = tuple(cell_faces)
        self.sub_faces_m = tuple(sub_faces)
        self.cell_volume_m3 = (np.diff(cell_faces[0])[:, None, None] * np.diff(cell_faces[1])[None, :, None]
                               * np.diff(cell_faces[2])[None, None, :])
        # device setup data
        self._faces = [ti.field(ti.f64, shape=len(f)) for f in cell_faces]
        self._sub_faces = [ti.field(ti.f64, shape=len(f)) for f in sub_faces]
        for field, values in zip(self._faces + self._sub_faces, cell_faces + sub_faces):
            field.from_numpy(values)
        self.body_index = ti.field(ti.i32, shape=self.n_body)
        self.body_index.from_numpy(indices.astype(np.int32))
        self.spacing = ti.Vector.field(3, ti.f32, shape=self.n_body)
        self.spacing.from_numpy(spacing.astype(np.float32))
        self.v0 = ti.field(ti.f64, shape=self.n_body)
        self.v0.from_numpy(np.prod(spacing, axis=1))
        self.with_columns = point_mass_kg is not None
        self.mass = ti.field(ti.f64, shape=self.n_body)
        if self.with_columns:
            masses = np.asarray(point_mass_kg, dtype=np.float64)
            if masses.shape != (self.n_body,) or not (np.isfinite(masses).all() and (masses > 0.0).all()):
                raise ValueError("point_mass_kg must be finite, positive and one per body point")
            self.mass.from_numpy(masses)
        tets = np.asarray(TETS, dtype=np.int64)
        if self.n_cells:
            vertices = reference[cells[:, tets]]                              # (H, 6, 4, 3)
            e1, e2, e3 = (vertices[:, :, q] - vertices[:, :, 0] for q in (1, 2, 3))
            ref_volumes = np.einsum("hti,hti->ht", e1, np.cross(e2, e3)) / 6.0
            if np.any(ref_volumes == 0.0):
                raise ValueError("a reference lattice tetrahedron has zero volume")
        else:
            ref_volumes = np.zeros((0, 6))
        self.reference_cell_volume_m3 = float(np.abs(ref_volumes).sum())
        count = max(self.n_cells, 1)
        self.cell_corners = ti.field(ti.i32, shape=(count, 8))
        self.cell_reference = ti.field(ti.f64, shape=(count, 6))
        if self.n_cells:
            self.cell_corners.from_numpy(indices[cells].astype(np.int32))      # global point numbers
            self.cell_reference.from_numpy(ref_volumes)
        # per-update fields (scratch) and the hysteresis state
        self.bits = ti.field(ti.i32, shape=self.shape + (self.words,))
        self.sample_hit = ti.field(ti.i32, shape=self.shape)
        self.material = ti.field(ti.i32, shape=self.shape)
        self.coverage_field = ti.field(ti.f32, shape=self.shape)
        self.topo = ti.field(ti.i32, shape=self.shape)
        self.age = ti.field(ti.i32, shape=self.shape)
        self.stats = ti.field(ti.f64, shape=STATS_SLOTS)
        self.column_mass_field = ti.field(ti.f64, shape=self.shape[:2])
        self.column_ke_field = ti.field(ti.f64, shape=self.shape[:2])
        self.last = None

    # ------------------------------------------------------------------------------------------------ device helpers
    @ti.func
    def _locate(self, faces: ti.template(), value: ti.f64) -> ti.i32:
        """Index of the interval of `faces` holding value, clipped to the first / last interval."""
        n = faces.shape[0] - 1
        low, high = 0, n
        if value < faces[0]:
            high = 1
        elif value >= faces[n]:
            low = n - 1
        while high - low > 1:
            middle = (low + high) // 2
            if faces[middle] <= value:
                low = middle
            else:
                high = middle
        return low

    @ti.func
    def _first_centre_at_or_above(self, faces: ti.template(), value: ti.f64) -> ti.i32:
        n = faces.shape[0] - 1
        result = n
        if value <= 0.5 * (faces[0] + faces[1]):
            result = 0
        elif value <= 0.5 * (faces[n - 1] + faces[n]):
            s = self._locate(faces, value)
            if 0.5 * (faces[s] + faces[s + 1]) < value:
                s += 1
            result = s
        return result

    @ti.func
    def _last_centre_at_or_below(self, faces: ti.template(), value: ti.f64) -> ti.i32:
        n = faces.shape[0] - 1
        result = -1
        if value >= 0.5 * (faces[n - 1] + faces[n]):
            result = n - 1
        elif value >= 0.5 * (faces[0] + faces[1]):
            s = self._locate(faces, value)
            if 0.5 * (faces[s] + faces[s + 1]) > value:
                s -= 1
            result = s
        return result

    @ti.func
    def _position(self, p):
        return self.points.X0[p].cast(ti.f64) + self.points.u[p].cast(ti.f64)

    @ti.func
    def _sub_centre(self, i, j, k):
        return ti.Vector([0.5 * (self._sub_faces[0][i] + self._sub_faces[0][i + 1]),
                          0.5 * (self._sub_faces[1][j] + self._sub_faces[1][j + 1]),
                          0.5 * (self._sub_faces[2][k] + self._sub_faces[2][k + 1])])

    @ti.func
    def _hit(self, i, j, k):
        n = ((i % self.sub) * self.sub + j % self.sub) * self.sub + k % self.sub
        ti.atomic_or(self.bits[i // self.sub, j // self.sub, k // self.sub, n // 32], 1 << (n % 32))

    # ------------------------------------------------------------------------------------------------ kernels
    @ti.kernel
    def _clear_kernel(self):
        for I in ti.grouped(self.bits):
            self.bits[I] = 0
        for I in ti.grouped(self.sample_hit):
            self.sample_hit[I] = 0
        for I in ti.grouped(self.column_mass_field):
            self.column_mass_field[I] = 0.0
            self.column_ke_field[I] = 0.0
        for s in range(STATS_SLOTS):
            self.stats[s] = 0.0

    @ti.kernel
    def _box_kernel(self, samples: ti.i32):
        for b in range(self.n_body):
            p = self.body_index[b]
            x = self._position(p)
            F = self.points.F[p]
            s = self.spacing[b]
            a0 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]]) * s[0]               # the box's edge vectors
            a1 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]]) * s[1]
            a2 = ti.Vector([F[0, 2], F[1, 2], F[2, 2]]) * s[2]
            volume = a0.dot(a1.cross(a2))
            ti.atomic_add(self.stats[S_V0J], self.v0[b] * ti.cast(F.determinant(), ti.f64))
            if ti.static(self.with_columns):
                if (x[0] >= self._faces[0][0] and x[0] < self._faces[0][self.shape[0]]
                        and x[1] >= self._faces[1][0] and x[1] < self._faces[1][self.shape[1]]):
                    ci = self._locate(self._faces[0], x[0])
                    cj = self._locate(self._faces[1], x[1])
                    v = self.points.v[p].cast(ti.f64)
                    ti.atomic_add(self.column_mass_field[ci, cj], self.mass[b])
                    ti.atomic_add(self.column_ke_field[ci, cj], 0.5 * self.mass[b] * v.dot(v))
            if ti.abs(volume) > 0.0:                                         # (False for zero and NaN)
                E = ti.Matrix.cols([a0, a1, a2])
                A = E.inverse()                                              # q = A (y - x), |q_i| <= 1/2 inside
                half = 0.5 * (ti.abs(a0) + ti.abs(a1) + ti.abs(a2))
                lo = x - half.cast(ti.f64) - AABB_MARGIN_M
                hi = x + half.cast(ti.f64) + AABB_MARGIN_M
                i0 = self._first_centre_at_or_above(self._sub_faces[0], lo[0])
                i1 = self._last_centre_at_or_below(self._sub_faces[0], hi[0])
                j0 = self._first_centre_at_or_above(self._sub_faces[1], lo[1])
                j1 = self._last_centre_at_or_below(self._sub_faces[1], hi[1])
                k0 = self._first_centre_at_or_above(self._sub_faces[2], lo[2])
                k1 = self._last_centre_at_or_below(self._sub_faces[2], hi[2])
                for i in range(i0, i1 + 1):
                    for j in range(j0, j1 + 1):
                        for k in range(k0, k1 + 1):
                            q = A @ (self._sub_centre(i, j, k) - x).cast(ti.f32)
                            if (ti.abs(q[0]) <= 0.5 + INSIDE_TOLERANCE and ti.abs(q[1]) <= 0.5 + INSIDE_TOLERANCE
                                    and ti.abs(q[2]) <= 0.5 + INSIDE_TOLERANCE):
                                self._hit(i, j, k)
                # the point's own box samples on the cell level (the point itself is the centre sample)
                for n in range(samples * samples * samples):
                    xi = ti.Vector([(n // (samples * samples)) % samples, (n // samples) % samples, n % samples])
                    frac = (xi.cast(ti.f32) + 0.5) / samples - 0.5
                    y = x + (E @ frac).cast(ti.f64)
                    if (y[0] >= self._faces[0][0] and y[0] < self._faces[0][self.shape[0]]
                            and y[1] >= self._faces[1][0] and y[1] < self._faces[1][self.shape[1]]
                            and y[2] >= self._faces[2][0] and y[2] < self._faces[2][self.shape[2]]):
                        self.sample_hit[self._locate(self._faces[0], y[0]), self._locate(self._faces[1], y[1]),
                                        self._locate(self._faces[2], y[2])] = 1
            else:
                ti.atomic_add(self.stats[S_BOXES_SINGULAR], 1.0)

    @ti.kernel
    def _lattice_kernel(self, factor: ti.f64):
        for h in range(self.n_cells):
            j_points = ti.cast(0.0, ti.f64)
            for c in ti.static(range(8)):
                j_points += ti.cast(self.points.F[self.cell_corners[h, c]].determinant(), ti.f64)
            j_points /= 8.0
            current = ti.Vector.zero(ti.f64, 6)
            reference_sum = ti.cast(0.0, ti.f64)
            signed_sum = ti.cast(0.0, ti.f64)
            inverted = 0
            for t in ti.static(range(6)):
                v0 = self._position(self.cell_corners[h, TETS[t][0]])
                v1 = self._position(self.cell_corners[h, TETS[t][1]])
                v2 = self._position(self.cell_corners[h, TETS[t][2]])
                v3 = self._position(self.cell_corners[h, TETS[t][3]])
                current[t] = (v1 - v0).dot((v2 - v0).cross(v3 - v0)) / 6.0
                sign = ti.select(self.cell_reference[h, t] > 0.0, 1.0, -1.0)
                reference_sum += ti.abs(self.cell_reference[h, t])
                signed_sum += current[t] * sign
                if current[t] * sign <= 0.0:
                    inverted = 1
            if inverted != 0:
                ti.atomic_add(self.stats[S_CELLS_INVERTED], 1.0)
            if signed_sum / reference_sum > factor * j_points:
                ti.atomic_add(self.stats[S_CELLS_DROPPED], 1.0)
            else:
                for t in ti.static(range(6)):
                    if ti.abs(current[t]) <= DEGENERATE_RELATIVE * ti.abs(self.cell_reference[h, t]):
                        ti.atomic_add(self.stats[S_TETS_DEGENERATE], 1.0)
                    else:
                        v0 = self._position(self.cell_corners[h, TETS[t][0]])
                        v1 = self._position(self.cell_corners[h, TETS[t][1]])
                        v2 = self._position(self.cell_corners[h, TETS[t][2]])
                        v3 = self._position(self.cell_corners[h, TETS[t][3]])
                        E = ti.Matrix.cols([(v1 - v0).cast(ti.f32), (v2 - v0).cast(ti.f32), (v3 - v0).cast(ti.f32)])
                        A = E.inverse()                                          # barycentric 1..3 = A (y - v0)
                        lo = ti.min(ti.min(v0, v1), ti.min(v2, v3)) - AABB_MARGIN_M
                        hi = ti.max(ti.max(v0, v1), ti.max(v2, v3)) + AABB_MARGIN_M
                        i0 = self._first_centre_at_or_above(self._sub_faces[0], lo[0])
                        i1 = self._last_centre_at_or_below(self._sub_faces[0], hi[0])
                        j0 = self._first_centre_at_or_above(self._sub_faces[1], lo[1])
                        j1 = self._last_centre_at_or_below(self._sub_faces[1], hi[1])
                        k0 = self._first_centre_at_or_above(self._sub_faces[2], lo[2])
                        k1 = self._last_centre_at_or_below(self._sub_faces[2], hi[2])
                        for i in range(i0, i1 + 1):
                            for j in range(j0, j1 + 1):
                                for k in range(k0, k1 + 1):
                                    lam = A @ (self._sub_centre(i, j, k) - v0).cast(ti.f32)
                                    if (lam[0] >= -INSIDE_TOLERANCE and lam[1] >= -INSIDE_TOLERANCE
                                            and lam[2] >= -INSIDE_TOLERANCE
                                            and lam[0] + lam[1] + lam[2] <= 1.0 + INSIDE_TOLERANCE):
                                        self._hit(i, j, k)

    @ti.kernel
    def _reduce_kernel(self, steps: ti.i32):
        for ci, cj, ck in self.material:
            hits = 0
            for word in range(self.words):
                hits += ti.math.popcnt(self.bits[ci, cj, ck, word])
            coverage = ti.cast(hits, ti.f32) / (self.sub * self.sub * self.sub)
            self.coverage_field[ci, cj, ck] = coverage
            material = ti.select((hits != 0) | (self.sample_hit[ci, cj, ck] != 0), 1, 0)
            self.material[ci, cj, ck] = material
            volume = ((self._faces[0][ci + 1] - self._faces[0][ci]) * (self._faces[1][cj + 1] - self._faces[1][cj])
                      * (self._faces[2][ck + 1] - self._faces[2][ck]))
            ti.atomic_add(self.stats[S_UNION_VOLUME], ti.cast(coverage, ti.f64) * volume)
            old = self.topo[ci, cj, ck]
            new = old
            if material != 0:
                new = 1
                self.age[ci, cj, ck] = 0
            elif old != 0:
                age = self.age[ci, cj, ck] + 1
                self.age[ci, cj, ck] = age
                if age >= steps:
                    new = 0
                    self.age[ci, cj, ck] = 0
            self.topo[ci, cj, ck] = new
            if new != old:
                ti.atomic_add(self.stats[S_FLIPPED], 1.0)
            if material != 0:
                ti.atomic_add(self.stats[S_MATERIAL_CELLS], 1.0)
            if new != 0:
                ti.atomic_add(self.stats[S_TOPO_CELLS], 1.0)

    # ------------------------------------------------------------------------------------------------ host API
    def update(self):
        """Rasterise the current body, refresh coverage, material and the topology hysteresis; one host read.  Returns
        the statistics: boxes_singular, lattice_cells_dropped / _inverted, tets_degenerate, material_cells,
        topo_cells, flipped (cells whose topology changed), union_volume_m3, sum_v0_j_m3, overfill."""
        self._clear_kernel()
        self._box_kernel(self.box_samples)
        if self.n_cells:
            self._lattice_kernel(self.stretch_factor)
        self._reduce_kernel(self.hysteresis_steps)
        s = self.stats.to_numpy()
        self.updates += 1
        v0j = float(s[S_V0J])
        self.last = dict(boxes_singular=int(s[S_BOXES_SINGULAR]), lattice_cells=self.n_cells,
                         lattice_cells_dropped=int(s[S_CELLS_DROPPED]), lattice_cells_inverted=int(s[S_CELLS_INVERTED]),
                         tets_degenerate=int(s[S_TETS_DEGENERATE]), material_cells=int(s[S_MATERIAL_CELLS]),
                         topo_cells=int(s[S_TOPO_CELLS]), flipped=int(s[S_FLIPPED]),
                         union_volume_m3=float(s[S_UNION_VOLUME]), sum_v0_j_m3=v0j,
                         overfill=(float(s[S_UNION_VOLUME]) / v0j if v0j > 0.0 else float("nan")))
        return dict(self.last)

    def topology_mask(self):
        """Host copy of the topology (hysteresis) material mask over the box, bool."""
        return self.topo.to_numpy() != 0

    def material_mask(self):
        """Host copy of this update's raw material mask (no hysteresis), bool."""
        return self.material.to_numpy() != 0

    def coverage(self):
        """Host copy of the cell coverage fractions (hit sub-cells / SUB^3, f32)."""
        return self.coverage_field.to_numpy()

    def column_mass(self):
        """Host copy of the body mass per x-y column of the box (kg), at the last update (needs point_mass_kg)."""
        if not self.with_columns:
            raise RuntimeError("column sums need point_mass_kg")
        return self.column_mass_field.to_numpy()

    def column_kinetic_energy(self):
        """Host copy of the body kinetic energy per x-y column of the box (J), at the last update."""
        if not self.with_columns:
            raise RuntimeError("column sums need point_mass_kg")
        return self.column_ke_field.to_numpy()

    def reset_topology(self):
        """Start the hysteresis from this update's raw material (a cold start; a restart loads topo / age instead)."""
        self.topo.from_numpy(self.material.to_numpy().astype(np.int32))
        self.age.from_numpy(np.zeros(self.shape, dtype=np.int32))
