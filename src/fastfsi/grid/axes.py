"""Geometry of the axes of a MAC grid: cell and face coordinates, spacings, the node-index map of each node family, and
the control volume of every MAC face.  GradedAxes handles graded (stretched) grids, UniformAxes uniform ones.

Node families: along axis a, staggered = 1 are the faces of the MAC component along its own axis, staggered = 0 the
cell centres.  On a graded axis a physical coordinate is mapped to the node-index coordinate xi by the monotone
piecewise-linear map through the nodes, xi(node_i) = i, extended by two ghost nodes per side at the edge cell width
(mirrored ghost cells) and linearly beyond (index_coordinate, _graded_index_coordinate, _node_offset).  On a uniform axis
xi = (coord - origin) / h - (0 if staggered else 0.5).

f32 precision: node coordinates are stored relative to the f32-rounded grid origin, so the map adds no error beyond the
coordinate's own f32 rounding (an absolute f32 position near 1 m carries an ulp of 1.2e-7 m; integrate displacements
rather than positions).

face_volume(c, i, j, k): the centre distance along c (half the edge cell at index 0) times the cell widths along the
other two axes, i.e. the MAC momentum control volume of that face.

The kernel weights of the immersed-boundary coupling are evaluated on these axes in coupling.kernel_weights.

No `from __future__ import annotations` (Taichi kernel module).
"""
import math

import numpy as np
import taichi as ti

UNIFORM_RTOL = 1.0e-12               # an axis is uniform when every cell width equals the first to this relative tolerance
NONFINITE_INDEX_COORDINATE = -100.0  # a non-finite / absurd coordinate maps here: every node out of range
FINITE_INDEX_BOUND = 1.0e7           # |xi| at or beyond this (or NaN) is treated as non-finite
GHOST_NODES = 2                      # mirrored ghost nodes per side in the device node tables (covers a 4-node support)


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


@ti.data_oriented
class GradedAxes:
    """Axes of a (possibly) graded MAC grid given by its face coordinates (m, float64, n + 1 per axis)."""

    def __init__(self, faces_x, faces_y, faces_z):
        faces = tuple(_validated_faces(f, name) for f, name in zip((faces_x, faces_y, faces_z), "xyz"))
        widths = tuple(np.diff(f) for f in faces)
        centres = tuple(0.5 * (f[1:] + f[:-1]) for f in faces)
        self.n = tuple(int(len(w)) for w in widths)
        self.min_spacing_m = float(min(float(w.min()) for w in widths))
        self.faces_m = faces                 # host copies (float64), read by other modules (pressure chamber, contact)
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
    def from_fluid(cls, fluid):
        """Axes of fluid.grid (cell_faces_{x,y,z}_m)."""
        grid = fluid.grid
        return cls(grid.cell_faces_x_m, grid.cell_faces_y_m, grid.cell_faces_z_m)

    def stretch_in_box(self, lo_m, hi_m, margin_cells=2):
        """Largest adjacent-cell width ratio per axis among the cells meeting the box [lo_m, hi_m] widened by
        `margin_cells`."""
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
        """JSON-able description (sizes, uniform flags, stretch ratios) for run metadata."""
        return dict(n=list(self.n), min_spacing_m=self.min_spacing_m, uniform=list(self.uniform),
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
        where every node is out of range."""
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
class UniformAxes:
    """Axes of a uniform MAC grid with origin `origin_m`, spacing `spacing_m` and `nodes` cells."""

    def __init__(self, origin_m, spacing_m, nodes):
        self.origin = tuple(float(v) for v in origin_m)
        self.h = tuple(float(v) for v in spacing_m)
        self.n = tuple(int(v) for v in nodes)
        if min(self.h) <= 0.0 or min(self.n) <= 0:
            raise ValueError("uniform axes need positive spacing and node counts")
        self.min_spacing_m = min(self.h)
        self.cell_volume_m3 = self.h[0] * self.h[1] * self.h[2]

    @classmethod
    def from_fluid(cls, fluid):
        grid = fluid.grid
        if not grid.is_uniform:
            raise ValueError("UniformAxes needs a uniform fluid grid; use GradedAxes")
        return cls(grid.bounds_min_m, grid.uniform_spacing_m, grid.grid_nodes)

    @ti.func
    def face_volume(self, c: ti.template(), i, j, k):
        return self.cell_volume_m3
