"""Signed distance to rigid triangulated solids at the cell centres of a Cartesian grid, its on-disk cache, and the
distance to the obstacle faces of a cell mask.

build_signed_distance: phi (m, float64, grid shape; negative inside a solid) to the union of closed rigid solids.  The
magnitude is exact (point-triangle distance) within band_cells local cells of the surfaces (and, with reach_points_m,
within reach_margin_m of one of those points); elsewhere it is the nearest surface-sample distance, an upper bound with
an error below two sample spacings, meant for places where no contact force can act.  The exact part is complete for
any sample spacing (radius check); the spacing only sets the search cost.  The sign comes from the closed-solid cell
classification (geometry.closed_solid) or, with sign_mask, from a given cell mask (e.g. the run's obstacle mask, so that
the contact geometry and the fluid's rigid geometry agree cell by cell).

load_or_build_signed_distance: the same with an on-disk cache keyed by the SHA-256 of every input (algorithm version,
parameters, triangles, grid faces, reach points, sign mask); a cache file is written create-only.

obstacle_face_distance / wall_distance_on_obstacle_faces: for a cell mask, the distance of every cell centre to the
nearest cell of the other kind (box distance, exact on graded grids), and a signed wall distance whose zero level lies
on the obstacle faces the fluid uses: min(|phi|, face distance) in fluid cells, -(face distance) in obstacle cells.

The exact point-triangle distance is written here (projection and barycentric test, else the nearest edge); scipy's
cKDTree finds the candidate triangles.
"""
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np

OBSTACLE_FACE_WINDOW = 2       # obstacle-face distance searched within 2 cells (the contact stencil's reach)
BAND_CELLS = 2                 # exact SDF within 2 local cells: the trilinear stencil of any point with phi < d_c
REACH_MARGIN_M = 0.005         # exact SDF only within this distance of a body point (reference positions)
SAMPLE_SPACING_CELLS = 2.0     # candidate-search surface samples every 2 smallest cells (search cost only)
CANDIDATE_SAMPLES = 64         # nearest surface samples examined per cell (quadrupled until provably complete)
PAIR_BUDGET = 1_000_000        # point-triangle pairs per exact-distance batch (bounds the transient host memory)
SDF_CACHE_VERSION = "rigid-contact-sdf-v1"   # bump whenever build_signed_distance changes its result


def _axis_faces(origin, widths):
    """Face coordinates (n+1) of one axis, accumulated sequentially in float64 (as the fluid grid does)."""
    return float(origin) + np.concatenate([[0.0], np.cumsum(np.asarray(widths, dtype=np.float64))])


def _cell_index(faces, coord):
    i = int(np.searchsorted(faces, float(coord), side="right")) - 1
    return min(max(i, 0), len(faces) - 2)


def point_triangle_distance(p, a, b, c):
    """Exact Euclidean distance from points p to triangles (a, b, c), all (m, 3) float64: the projection onto the
    plane when it falls inside the triangle, otherwise the nearest of the three edges (degenerate faces: edges)."""
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    area2 = np.linalg.norm(normal, axis=1)
    safe = area2 > 1.0e-300
    unit = np.zeros_like(normal)
    unit[safe] = normal[safe] / area2[safe, None]
    height = np.einsum("ij,ij->i", p - a, unit)
    q = p - height[:, None] * unit
    u = np.einsum("ij,ij->i", np.cross(b - q, c - q), unit)
    v = np.einsum("ij,ij->i", np.cross(c - q, a - q), unit)
    w = area2 - u - v
    inside = safe & (u >= 0.0) & (v >= 0.0) & (w >= 0.0)

    def segment(s0, s1):
        e = s1 - s0
        length2 = np.einsum("ij,ij->i", e, e)
        t = np.where(length2 > 0.0, np.einsum("ij,ij->i", p - s0, e) / np.where(length2 > 0.0, length2, 1.0), 0.0)
        t = np.clip(t, 0.0, 1.0)
        return np.linalg.norm(p - (s0 + t[:, None] * e), axis=1)

    edge = np.minimum(np.minimum(segment(a, b), segment(b, c)), segment(c, a))
    return np.where(inside, np.abs(height), edge)


def surface_samples(triangles, spacing_m):
    """Area-uniform points on every triangle (barycentric lattice with ceil(|AB|/s) x ceil(|AC|/s) steps) and the
    index of the triangle each lies on.  Every point of a triangle lies within 2 s of one of its samples (one lattice
    cell, both cell edges <= s).  Unlike a lattice sized by the longest edge in both directions, sliver triangles are
    not oversampled, so the sample count scales with the area."""
    tri = np.asarray(triangles, dtype=np.float64)
    a, ab, ac = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    nu = np.maximum(1, np.ceil(np.linalg.norm(ab, axis=1) / spacing_m)).astype(np.int64)
    nv = np.maximum(1, np.ceil(np.linalg.norm(ac, axis=1) / spacing_m)).astype(np.int64)
    points, owners = [], []
    keys = nu * (int(nv.max()) + 1) + nv
    for key in np.unique(keys):
        group = np.flatnonzero(keys == key)
        n1, n2 = int(nu[group[0]]), int(nv[group[0]])
        i, j = np.meshgrid(np.arange(n1 + 1), np.arange(n2 + 1), indexing="ij")
        u, v = (i / n1).ravel(), (j / n2).ravel()
        keep = u + v <= 1.0 + 1.0e-12
        u, v = u[keep], v[keep]
        points.append((a[group][:, None] + u[None, :, None] * ab[group][:, None]
                       + v[None, :, None] * ac[group][:, None]).reshape(-1, 3))
        owners.append(np.repeat(group, u.size))
    return np.concatenate(points), np.concatenate(owners)


def _as_region_dict(rigid_triangles_m):
    if isinstance(rigid_triangles_m, dict):
        items = [(str(key), np.asarray(value, dtype=np.float64)) for key, value in rigid_triangles_m.items()]
    else:
        items = [("rigid", np.asarray(rigid_triangles_m, dtype=np.float64))]
    for name, tri in items:
        if tri.ndim != 3 or tri.shape[1:] != (3, 3) or len(tri) == 0 or not np.isfinite(tri).all():
            raise ValueError("rigid region %s: need a non-empty finite (n, 3, 3) triangle array in metres" % name)
    return items


def build_signed_distance(rigid_triangles_m, grid, *, band_cells=BAND_CELLS, sample_spacing_m=None,
                          reach_points_m=None, reach_margin_m=None, sign_mask=None):
    """Signed distance (m, float64, grid shape; negative inside a rigid solid) at the cell centres of `grid`
    (CartesianGrid) to the union of the closed rigid solids, and a report.  Exact within band_cells local cells of
    the surfaces (and, when reach_points_m is given, within reach_margin_m of one of those points), nearest-sample
    distance elsewhere (an upper bound, error below 2 sample spacings, used only where no contact force can act)."""
    from scipy.spatial import cKDTree

    from fastfsi.geometry.closed_solid import classify_closed_solid_cells

    regions = _as_region_dict(rigid_triangles_m)
    faces = [np.asarray(f, dtype=np.float64) for f in (grid.cell_faces_x_m, grid.cell_faces_y_m, grid.cell_faces_z_m)]
    centres = [0.5 * (f[:-1] + f[1:]) for f in faces]
    widths = [np.diff(f) for f in faces]
    shape = tuple(len(c) for c in centres)
    inside = np.zeros(shape, dtype=bool)
    region_reports = {}
    for name, tri in regions:
        solid, report = classify_closed_solid_cells(tri, grid, close_thin_walls=False)
        inside |= solid
        region_reports[name] = dict(faces=int(len(tri)), inside_cells=int(solid.sum()),
                                    enclosed_volume_m3=float(report["enclosed_volume_m3"]))
    # candidate-search samples: completeness of the exact distance is guaranteed by the radius check below for any
    # spacing, so the spacing only trades the search cost against the far-field upper-bound error (< 2 spacings)
    spacing = (SAMPLE_SPACING_CELLS * float(min(w.min() for w in widths))) if sample_spacing_m is None \
        else float(sample_spacing_m)
    band_max = band_cells * float(max(w.max() for w in widths))
    lo = np.array([f[0] for f in faces]) - band_max
    hi = np.array([f[-1] for f in faces]) + band_max
    tri = np.concatenate([t for _, t in regions])
    near_grid = np.all(tri.max(axis=1) >= lo, axis=1) & np.all(tri.min(axis=1) <= hi, axis=1)
    tri = tri[near_grid]
    if len(tri) == 0:
        raise ValueError("no rigid triangle lies within the band of the fluid grid")
    started = time.perf_counter()
    samples, owners = surface_samples(tri, spacing)
    tree = cKDTree(samples)
    cx, cy, cz = np.meshgrid(*centres, indexing="ij")
    points = np.column_stack([cx.ravel(), cy.ravel(), cz.ravel()])
    wx, wy, wz = np.meshgrid(*widths, indexing="ij")
    band = band_cells * np.maximum(np.maximum(wx, wy), wz).ravel()
    d_sample, _ = tree.query(points, k=1)
    distance = d_sample.copy()
    coverage = 2.0 * spacing          # every triangle point lies within one lattice cell (<= 2 s) of a sample
    near = np.flatnonzero(d_sample < band + coverage)
    if reach_points_m is not None:
        reach_tree = cKDTree(np.asarray(reach_points_m, dtype=np.float64).reshape(-1, 3))
        d_reach, _ = reach_tree.query(points[near], k=1, distance_upper_bound=float(reach_margin_m) * (1 + 1e-12))
        near = near[np.isfinite(d_reach)]
    exact_pairs = 0
    chunk = 5000
    for start in range(0, len(near), chunk):
        rows = near[start:start + chunk]
        pending = np.arange(len(rows))
        k = CANDIDATE_SAMPLES
        best = np.full(len(rows), np.inf)
        while len(pending):
            k_eff = min(k, len(samples))
            still = []
            step = max(1, PAIR_BUDGET // k_eff)                  # rows per batch: at most PAIR_BUDGET pairs at once
            for s0 in range(0, len(pending), step):
                sub = pending[s0:s0 + step]
                dist, idx = tree.query(points[rows[sub]], k=k_eff)
                dist = dist.reshape(len(sub), k_eff)
                idx = idx.reshape(len(sub), k_eff)
                cand = owners[idx]                               # (m, k) triangle ids
                p = np.repeat(points[rows[sub]], k_eff, axis=0)
                t = tri[cand.ravel()]
                d = point_triangle_distance(p, t[:, 0], t[:, 1], t[:, 2]).reshape(len(sub), k_eff)
                exact_pairs += d.size
                best[sub] = np.minimum(best[sub], d.min(axis=1))
                # complete when every sample closer than (best exact distance) + coverage was examined: the nearest
                # triangle has a sample within (true distance <= best) + coverage of the cell centre
                complete = (dist[:, -1] >= best[sub] + coverage) | (k_eff == len(samples))
                still.append(sub[~complete])
                del dist, idx, cand, p, t, d
            pending = np.concatenate(still) if still else pending[:0]
            k *= 4
        distance[rows] = best
    cad_inside = inside.ravel()
    if sign_mask is None:
        negative, sign_source = cad_inside, "CAD closed-solid cell-centre parity"
    else:
        negative = np.asarray(sign_mask).astype(bool)
        if negative.shape != shape:
            raise ValueError("sign mask shape %s != grid %s" % (negative.shape, shape))
        negative, sign_source = negative.ravel(), "run static obstacle mask (fluid.obstacle)"
    phi = np.where(negative, -distance, distance).reshape(shape)
    return phi, dict(sign_source=sign_source, cad_inside_fluid_cells=int(np.count_nonzero(cad_inside & ~negative)),
                     cad_outside_obstacle_cells=int(np.count_nonzero(~cad_inside & negative)),
                     method="exact point-triangle distance within %d local cells%s, nearest-sample distance "
                            "elsewhere; magnitude = distance to the CAD surface"
                            % (band_cells, "" if reach_points_m is None else
                               " and within %.4g m of a body point" % float(reach_margin_m)),
                     regions=region_reports, triangles_near_grid=int(len(tri)), surface_samples=int(len(samples)),
                     sample_spacing_m=spacing, exact_cells=int(len(near)), exact_pairs=int(exact_pairs),
                     inside_cells=int(inside.sum()), band_cells=band_cells,
                     reach_margin_m=None if reach_points_m is None else float(reach_margin_m),
                     distance_build_s=time.perf_counter() - started)


def obstacle_face_distance(grid, obstacle, window=OBSTACLE_FACE_WINDOW):
    """For every cell centre of `grid`, the Euclidean distance (m, float64) to the nearest cell of the OTHER kind
    (obstacle vs fluid, `obstacle` bool, grid shape) within `window` cells along each axis, measured to that cell's box
    (per axis max(0, |c - c'| - w'/2)); inf where there is none.  Graded grids are handled exactly (per-cell widths).
    Only the cells within `window` of the other kind are evaluated (binary dilation), the rest stay inf."""
    from scipy.ndimage import binary_dilation

    faces = [np.asarray(f, dtype=np.float64) for f in (grid.cell_faces_x_m, grid.cell_faces_y_m, grid.cell_faces_z_m)]
    centres = [0.5 * (f[:-1] + f[1:]) for f in faces]
    half = [0.5 * np.diff(f) for f in faces]
    solid = np.asarray(obstacle).astype(bool)
    shape = solid.shape
    if shape != tuple(len(c) for c in centres):
        raise ValueError("obstacle mask %s does not match the grid %s" % (shape, tuple(len(c) for c in centres)))
    box = np.ones((2 * window + 1,) * 3, dtype=bool)
    near = (binary_dilation(solid, structure=box) & ~solid) | (binary_dilation(~solid, structure=box) & solid)
    idx = np.argwhere(near)
    best = np.full(len(idx), np.inf)
    n = np.array(shape)
    for di in range(-window, window + 1):
        for dj in range(-window, window + 1):
            for dk in range(-window, window + 1):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                nb = idx + np.array([di, dj, dk])
                ok = np.all((nb >= 0) & (nb < n), axis=1)
                rows = np.flatnonzero(ok)
                a, b = idx[rows], nb[rows]
                other = solid[a[:, 0], a[:, 1], a[:, 2]] != solid[b[:, 0], b[:, 1], b[:, 2]]
                rows, a, b = rows[other], a[other], b[other]
                if not len(rows):
                    continue
                d2 = np.zeros(len(rows))
                for axis, delta in enumerate((di, dj, dk)):
                    if delta != 0:
                        gap = np.abs(centres[axis][a[:, axis]] - centres[axis][b[:, axis]]) - half[axis][b[:, axis]]
                        d2 += np.maximum(gap, 0.0) ** 2
                best[rows] = np.minimum(best[rows], np.sqrt(d2))
    out = np.full(shape, np.inf)
    out[idx[:, 0], idx[:, 1], idx[:, 2]] = best
    return out, dict(window_cells=int(window), evaluated_cells=int(len(idx)),
                     finite_cells=int(np.isfinite(best).sum()))


def wall_distance_on_obstacle_faces(phi, grid, obstacle, window=OBSTACLE_FACE_WINDOW):
    """A signed wall distance whose zero level lies on the obstacle faces the fluid uses: min(|phi|, obstacle-face
    distance) at fluid cell centres,
    -(distance to the nearest fluid cell) at obstacle cell centres within `window` cells, phi elsewhere."""
    solid = np.asarray(obstacle).astype(bool)
    face, report = obstacle_face_distance(grid, solid, window)
    out = np.asarray(phi, dtype=np.float64).copy()
    fluid_near = ~solid & np.isfinite(face)
    out[fluid_near] = np.minimum(np.abs(out[fluid_near]), face[fluid_near])
    solid_near = solid & np.isfinite(face)
    out[solid_near] = -face[solid_near]
    report.update(fluid_cells_changed=int(np.count_nonzero(out[~solid] != np.asarray(phi)[~solid])),
                  obstacle_cells_changed=int(np.count_nonzero(out[solid] != np.asarray(phi)[solid])))
    return out, report


def signed_distance_cache_key(rigid_triangles_m, grid, *, band_cells=BAND_CELLS, sample_spacing_m=None, reach_points_m=None,
                  reach_margin_m=None, sign_mask=None):
    """SHA-256 over every input of build_signed_distance: algorithm version, parameters, region names and triangles,
    the grid's cell faces, the reach points and the sign mask (each length-prefixed)."""
    h = hashlib.sha256()

    def add(tag, data):
        h.update(tag.encode())
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    add("version", SDF_CACHE_VERSION.encode())
    # band_cells as the float the build uses (int() made 0.1 and 0.9 one key)
    add("params", json.dumps([float(band_cells), None if sample_spacing_m is None else float(sample_spacing_m),
                              None if reach_margin_m is None else float(reach_margin_m)]).encode())
    for name, tri in _as_region_dict(rigid_triangles_m):
        add("region", name.encode())
        add("triangles", np.ascontiguousarray(tri, dtype=np.float64).tobytes())
    for faces in (grid.cell_faces_x_m, grid.cell_faces_y_m, grid.cell_faces_z_m):
        add("faces", np.ascontiguousarray(faces, dtype=np.float64).tobytes())
    add("reach", b"none" if reach_points_m is None else
        np.ascontiguousarray(reach_points_m, dtype=np.float64).tobytes())
    # the mask as the build reads it (nonzero = inside): a uint8 cast wraps 256 to 0
    add("sign", b"cad" if sign_mask is None else
        np.ascontiguousarray(np.asarray(sign_mask).astype(bool), dtype=np.uint8).tobytes())
    return h.hexdigest()


def load_or_build_signed_distance(cache_dir, rigid_triangles_m, grid, **kwargs):
    """build_signed_distance with an on-disk cache keyed by signed_distance_cache_key (build once, reuse in later
    runs).  Hit: the stored phi and report (report['cache'] = 'hit').  Miss: build, write a private temporary file in
    the cache directory, then publish it create-only by a hard link (an existing file is never overwritten, a
    concurrent writer's file is left alone, and a write that fails half-way never appears under the cache name; a file
    system without hard links leaves the cache unwritten)."""
    cache_dir = Path(cache_dir)
    key = signed_distance_cache_key(rigid_triangles_m, grid, **kwargs)
    path = cache_dir / ("sdf_%s.npz" % key[:40])
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            if str(data["key"]) != key:
                raise ValueError("SDF cache key mismatch in %s" % path)
            phi = np.array(data["phi"], dtype=np.float64)
            report = json.loads(str(data["report_json"]))
        report.update(cache="hit", cache_path=str(path), cache_key=key)
        return phi, report
    phi, report = build_signed_distance(rigid_triangles_m, grid, **kwargs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=cache_dir)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez(stream, phi=phi, report_json=json.dumps(report), key=key)
        try:
            os.link(temporary, path)
            state = "miss (written)"
        except FileExistsError:
            state = "miss (file appeared meanwhile; kept)"
        except OSError as error:
            state = "miss (not cached: no hard link, %s)" % type(error).__name__
    finally:
        os.unlink(temporary)
    return phi, dict(report, cache=state, cache_path=str(path), cache_key=key)
