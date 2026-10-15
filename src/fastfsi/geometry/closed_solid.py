"""Exact cell classification of closed triangulated solids on a Cartesian grid.

A cell is solid when its centre lies inside a closed triangulated solid.  Lines along each grid axis pass through the
cell-centre rows; the crossing parity on a line classifies its centres, and the three axes vote.  A wall thinner than
the local centre spacing contains no centre, so with close_thin_walls=True the cell holding the midpoint of that wall
interval is marked solid as well: no fluid face then joins the two sides of a thin wall.

The line / triangle test evaluates every projected edge in its canonical (lexicographic) orientation and breaks exact
zeros by one global symbolic perturbation of the line position.  A line through a shared edge or vertex therefore
crosses each surface sheet exactly once, and every line through a closed surface has an even number of crossings,
which is checked.  `grid` is any object with cell_faces_{x,y,z}_m and cell_centers_{x,y,z}_m (float64 arrays).
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

_PAIR_CHUNK = 1 << 16


def weld_exact_vertices(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index a triangle soup by bit-identical vertex coordinates."""
    corners = np.ascontiguousarray(
        np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
    )
    points, inverse = np.unique(corners, axis=0, return_inverse=True)
    return points, np.asarray(inverse, dtype=np.int64).reshape(-1, 3)


def open_edge_count(faces: np.ndarray) -> int:
    """Edges not shared by exactly two faces (zero for a closed two-manifold)."""
    edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts != 2))


def _orient2d(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
    """(b - a) x (p - a) with an exact rational fallback when rounding could flip its sign."""
    left = (b[:, 0] - a[:, 0]) * (p[:, 1] - a[:, 1])
    right = (b[:, 1] - a[:, 1]) * (p[:, 0] - a[:, 0])
    det = left - right
    bound = 8.0 * np.finfo(np.float64).eps * (np.abs(left) + np.abs(right))
    for i in np.flatnonzero(np.abs(det) <= bound):
        ax, ay, bx, by, px, py = (
            Fraction(float(value))
            for value in (a[i, 0], a[i, 1], b[i, 0], b[i, 1], p[i, 0], p[i, 1])
        )
        det[i] = float((bx - ax) * (py - ay) - (by - ay) * (px - ax))
    return det


def _oriented_edge(
    start: np.ndarray,
    end: np.ndarray,
    point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Edge function of start->end at point, and its sign with ties broken.

    The value is computed from the canonical endpoint order, so the two faces
    sharing an edge evaluate bit-identical magnitudes of opposite sign. An exact
    zero takes the sign of the derivative along the perturbation (1, eta),
    eta -> 0+, which every edge shares.
    """
    swap = (start[:, 0] > end[:, 0]) | (
        (start[:, 0] == end[:, 0]) & (start[:, 1] > end[:, 1])
    )
    low = np.where(swap[:, None], end, start)
    high = np.where(swap[:, None], start, end)
    dv = high[:, 1] - low[:, 1]
    canonical = _orient2d(low, high, point)
    tie = np.where(dv != 0.0, -dv, 1.0)
    orientation = np.where(swap, -1.0, 1.0)
    signed = orientation * np.where(canonical != 0.0, canonical, tie)
    return orientation * canonical, signed


def _line_crossings(
    tri_uv: np.ndarray,
    tri_w: np.ndarray,
    line_u: np.ndarray,
    line_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Crossings of the lines (line_u[a], line_v[b], w) with the triangles.

    Returns the flat line index ``a * len(line_v) + b`` and the crossing
    coordinate ``w`` of every crossing.
    """
    area2 = _orient2d(tri_uv[:, 0], tri_uv[:, 1], tri_uv[:, 2])
    a_start = np.searchsorted(line_u, tri_uv[:, :, 0].min(axis=1), side="left")
    a_stop = np.searchsorted(line_u, tri_uv[:, :, 0].max(axis=1), side="right")
    b_start = np.searchsorted(line_v, tri_uv[:, :, 1].min(axis=1), side="left")
    b_stop = np.searchsorted(line_v, tri_uv[:, :, 1].max(axis=1), side="right")
    a_count = np.maximum(a_stop - a_start, 0)
    b_count = np.maximum(b_stop - b_start, 0)
    # A face parallel to the lines has no crossing for any perturbed line.
    pair_count = np.where(area2 != 0.0, a_count * b_count, 0).astype(np.int64)
    faces = np.flatnonzero(pair_count)
    line_ids: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    ends = np.cumsum(pair_count[faces])
    starts = ends - pair_count[faces]
    total = int(ends[-1]) if ends.size else 0
    for begin in range(0, total, _PAIR_CHUNK):
        pair = np.arange(begin, min(begin + _PAIR_CHUNK, total), dtype=np.int64)
        position = np.searchsorted(ends, pair, side="right")
        face = faces[position]
        offset = pair - starts[position]
        a = a_start[face] + offset // b_count[face]
        b = b_start[face] + offset % b_count[face]
        point = np.stack((line_u[a], line_v[b]), axis=1)
        e01, s01 = _oriented_edge(tri_uv[face, 0], tri_uv[face, 1], point)
        e12, s12 = _oriented_edge(tri_uv[face, 1], tri_uv[face, 2], point)
        e20, s20 = _oriented_edge(tri_uv[face, 2], tri_uv[face, 0], point)
        sigma = np.sign(area2[face])
        hit = (sigma * s01 > 0.0) & (sigma * s12 > 0.0) & (sigma * s20 > 0.0)
        face = face[hit]
        w = (
            e12[hit] * tri_w[face, 0]
            + e20[hit] * tri_w[face, 1]
            + e01[hit] * tri_w[face, 2]
        ) / area2[face]
        w = np.clip(w, tri_w[face].min(axis=1), tri_w[face].max(axis=1))
        line_ids.append(a[hit] * line_v.size + b[hit])
        coordinates.append(w)
    if not line_ids:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    return np.concatenate(line_ids), np.concatenate(coordinates)


def _axis_classification(
    triangles: np.ndarray,
    centres: tuple[np.ndarray, np.ndarray, np.ndarray],
    axis: int,
) -> tuple[np.ndarray, int, tuple[np.ndarray, ...]]:
    """Centre parity along ``axis`` (in (x, y, z) order), the odd-line count and
    the inside intervals that hold no centre: (line, first centre above, enter, leave)."""
    u_axis, v_axis = (value for value in range(3) if value != axis)
    line_u, line_v, line_w = centres[u_axis], centres[v_axis], centres[axis]
    line, w = _line_crossings(
        triangles[:, :, [u_axis, v_axis]],
        triangles[:, :, axis],
        line_u,
        line_v,
    )
    line_count = line_u.size * line_v.size
    order = np.lexsort((w, line))
    line = line[order]
    w = w[order]
    per_line = np.bincount(line, minlength=line_count)
    odd_lines = int(np.count_nonzero(per_line % 2))
    shape = (line_u.size, line_v.size, line_w.size)
    inside = np.zeros(shape, dtype=bool)
    empty_intervals: tuple[np.ndarray, ...] = (np.zeros(0, dtype=np.int64),) * 2 + (
        np.zeros(0),
    ) * 2
    if odd_lines == 0 and w.size:
        # Consecutive crossings on a line bound the inside intervals.
        enter = w[0::2]
        leave = w[1::2]
        interval_line = line[0::2]
        first = np.searchsorted(line_w, enter, side="right")
        last = np.searchsorted(line_w, leave, side="right")
        coverage = np.zeros((line_count, line_w.size + 1), dtype=np.int32)
        np.add.at(coverage, (interval_line, first), 1)
        np.add.at(coverage, (interval_line, last), -1)
        np.cumsum(coverage[:, :-1], axis=1, dtype=np.int32, out=coverage[:, :-1])
        inside = (coverage[:, :-1] > 0).reshape(shape)
        # Zero-length (tangent) intervals hold no material.
        empty = (first == last) & (leave > enter)
        empty_intervals = (interval_line[empty], first[empty], enter[empty], leave[empty])
    order_to_xyz = tuple(int(value) for value in np.argsort((u_axis, v_axis, axis)))
    return inside.transpose(order_to_xyz), odd_lines, empty_intervals


def _thin_wall_cells(
    by_centre: np.ndarray,
    cell_faces: tuple[np.ndarray, np.ndarray, np.ndarray],
    axis: int,
    empty_intervals: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Cells closing walls that lie between two fluid centres on a line.

    The cell holding the wall interval's midpoint becomes solid unless one of
    the two centres bracketing the interval is already solid (nothing to close).
    """
    u_axis, v_axis = (value for value in range(3) if value != axis)
    shape = by_centre.shape
    count_w = shape[axis]
    line_count = shape[u_axis] * shape[v_axis]
    occupied = by_centre.transpose(u_axis, v_axis, axis).reshape(line_count, count_w)
    line, first, enter, leave = empty_intervals
    interior = (first > 0) & (first < count_w)
    redundant = np.zeros(line.size, dtype=bool)
    redundant[interior] = (
        occupied[line[interior], first[interior] - 1] | occupied[line[interior], first[interior]]
    )
    line, enter, leave = line[~redundant], enter[~redundant], leave[~redundant]
    cell = np.searchsorted(cell_faces[axis], 0.5 * (enter + leave), side="right") - 1
    in_grid = (cell >= 0) & (cell < count_w)
    thin = np.zeros((line_count, count_w), dtype=bool)
    thin[line[in_grid], cell[in_grid]] = True
    order_to_xyz = tuple(int(value) for value in np.argsort((u_axis, v_axis, axis)))
    return thin.reshape(shape[u_axis], shape[v_axis], count_w).transpose(order_to_xyz)


def classify_closed_solid_cells(
    triangles_m: np.ndarray,
    grid,
    *,
    close_thin_walls: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solid cells of one closed triangulated solid (triangles in metres).

    ``close_thin_walls=False`` returns the cell-centre classification alone
    (material points), without the cells that seal walls between centres.
    """
    points, faces = weld_exact_vertices(triangles_m)
    open_edges = open_edge_count(faces)
    if open_edges:
        raise ValueError(
            f"rigid solid surface is not closed: {open_edges} edges are not shared "
            "by exactly two faces"
        )
    triangles = points[faces]
    centres = (
        np.asarray(grid.cell_centers_x_m, dtype=np.float64),
        np.asarray(grid.cell_centers_y_m, dtype=np.float64),
        np.asarray(grid.cell_centers_z_m, dtype=np.float64),
    )
    cell_faces = (
        np.asarray(grid.cell_faces_x_m, dtype=np.float64),
        np.asarray(grid.cell_faces_y_m, dtype=np.float64),
        np.asarray(grid.cell_faces_z_m, dtype=np.float64),
    )
    shape = tuple(int(value.size) for value in centres)
    votes = np.zeros(shape, dtype=np.int8)
    odd_lines: list[int] = []
    empty_intervals: list[tuple[np.ndarray, ...]] = []
    for axis in range(3):
        inside, odd, empty = _axis_classification(triangles, centres, axis)
        votes += inside
        odd_lines.append(odd)
        empty_intervals.append(empty)
    if any(odd_lines):
        raise ValueError(
            "closed-solid line parity failed: lines with an odd crossing count "
            f"per axis {tuple(odd_lines)}"
        )
    by_centre = votes >= 2
    thin = np.zeros(shape, dtype=bool)
    if close_thin_walls:
        for axis in range(3):
            thin |= _thin_wall_cells(by_centre, cell_faces, axis, empty_intervals[axis])
    thin_only = thin & ~by_centre
    solid = by_centre | thin_only
    volume = float(
        np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2]),
        ).sum()
        / 6.0
    )
    widths = np.einsum(
        "i,j,k->ijk",
        np.diff(cell_faces[0]),
        np.diff(cell_faces[1]),
        np.diff(cell_faces[2]),
    )
    return solid, {
        "method": "closed_solid_cell_centre_parity_three_axis_vote",
        "face_count": int(faces.shape[0]),
        "welded_vertex_count": int(points.shape[0]),
        "enclosed_volume_m3": abs(volume),
        "solid_cell_count": int(np.count_nonzero(solid)),
        "solid_cell_volume_m3": float(widths[solid].sum()),
        "centre_inside_cell_count": int(np.count_nonzero(by_centre)),
        "thin_wall_cell_count": int(np.count_nonzero(thin_only)),
        "axis_vote_disagreement_cell_count": int(
            np.count_nonzero((votes != 0) & (votes != 3))
        ),
    }
