"""Physical contact between the deformable IB-MPM points and the rigid regions (Helper 2, 2026-10-12; USER decision
01:35Z "no ideal model, show the real behaviour"; Root REQ_403; per-step path per REQ_404).

MODEL.  No seal assumption, no obstacle change, no switching.  The rigid regions (source config
analysis_settings.solid_obstacle_surface_only_region_ids = [3, 4]: nozzle body and bottom plate) are represented, as
for the static obstacle, by their source-CAD triangles.  A signed distance field phi (m; < 0 inside a rigid solid) is
built once on the host at the fluid cell centres and interpolated trilinearly (with its gradient) at the points.  Every
non-tethered point p (body.tether == 0) whose distance is below its contact distance gets the repulsive penalty force
    F_p = k_c V0_p (d_c,p(n_p) - phi_p) n_p,   n_p = grad phi / |grad phi|,   only while phi_p < d_c,p(n_p),
ADDED to body.f_extra (the body spreads it with the other point forces and zeroes f_extra).  The contact is normal
only, FRICTIONLESS (no tangential force) and NON-ADHESIVE (no force for phi >= d_c, never attractive), with NO damping
(the contact adds no dissipation; dissipated energy = 0).  Whether the valve flap touches the plate, how hard, and how
much water still passes the refill path all follow from the computed flow; nothing is prescribed.

PARAMETERS (reasons; nothing is tuned):
  * k_c (N/m^4: force per unit point volume and unit penetration) = rho_s (0.25/dt)^2, omega dt = 0.25: the convention
    of the existing tether k (the integrator passes the same value as k_c): the contact period 8 pi dt ~ 25 steps is
    resolved by the explicit step (omega dt << 2), and the contact is as stiff as the tethers.
  * d_c,p(n) = a_p . |n|: a point stands for a material box centred on it, of volume V0_p and with the aspect ratio
    of the fluid cell holding its reference position (half extents a_p = 0.5 dx_cell (V0 / V_cell)^(1/3) per axis;
    the squid lattice is the split-2 sub-lattice of the graded grid, so a_p = a quarter of the local cell widths, and
    near the valve rim the boxes are flat, ~0.72 x 0.72 x 0.30 mm).  d_c(n) is the box's support distance along n: the
    repulsion starts exactly when that material box touches the rigid surface (0.15 mm = dz/4 for the rim on the
    land; an isotropic 0.5 V0^(1/3) would start 0.12 mm early there).  Point-commensurate and hence
    grid-commensurate (a quarter cell); never tuned.  A fixed d_c_m may be given instead.
  * the SDF is exact (point-triangle distance) within BAND_CELLS = 2 local cells of the rigid surfaces (the trilinear
    stencil of any point with phi < d_c, d_c <= a quarter cell, lies within 2 cells of the surface) and within
    reach_margin_m of a body point at its reference position (default 5 mm: the flap's whole travel to the land is
    the 2.626 mm gap); elsewhere it is the nearest surface-sample distance (an upper bound, error below two sample
    spacings, used only where no force can act).  The exact part is complete for any sample spacing (radius check);
    the spacing (2 smallest cells) only sets the search cost and the far-field error.
  * SIGN = the run's static obstacle mask (fluid.obstacle at construction: CAD solids + the user's chamber-wall
    groove + thin-wall seals): phi < 0 exactly in obstacle cells, so the contact geometry and the fluid's rigid
    geometry agree cell by cell.  MAGNITUDE = distance to the CAD surface (integrator option (a)).  Where the mask
    differs from the CAD (setup report: cad_inside_fluid_cells, e.g. the 9,820 groove cells r 32-35 mm;
    cad_outside_obstacle_cells, the thin-wall seals) the zero level follows the mask boundary through the sign
    change, and the CAD wall surface left inside the carved groove (r = 32 mm, z 1007.2-1018 mm) still repels points
    within d_c of it (the real chamber wall; no deformable point comes within 2 mm of it).  sign_from_obstacle=False
    takes the sign from the CAD classification instead.

PER-STEP PATH (REQ_404).  apply_point_forces(dt) is ONE kernel launch, dt a kernel argument, no host access and no
Python loop: SDF lookup + penalty force into body.f_extra, and on-device accumulation of the interval statistics
(refill-path volume, closed time, closed-state leakage volumes split by resolution, contact impulse, max penetration,
min rim gap, contact begin/end events with their time in the interval).  It evaluates the state it is given (call it
right before body.spread_force(): points at X^{n+1/2}, fluid velocity u^n), so the interval flows are left Riemann
sums.  report() runs one measure kernel, reads ONE small stats field once, and resets the interval accumulators on
the device.

VALVE DIAGNOSTICS (read-only; they never change the physics), report(t, dt, p_shift_pa):
  * rim (body.tag & TAG_VALVE_RIM: flap lower-surface points r >= 28.0 mm): gap to the land plane
    g = z - a_z - z_land (min / mean / max; a_z the vertical half extent of the material box), rim contact fraction
    (share of rim points with phi < d_c(n));
  * contact: point count, area (sum over contacting points of the material-box area projected along n: the touching
    faces), total normal force sum |F| (and the vector sum), contact energy sum 1/2 k_c V0 (d_c - phi)^2, min phi
    and max penetration max(0, -phi) over all non-tethered points (current state; the penetration also over the
    interval);
  * refill-path volume flow (m^3/s, + = upward toward the flap) through the z faces at refill_plane_z_m (inside the
    plate, z ~ 1003.5 mm) of the 6 ports (columns meeting r 23-28 mm) and of the radial channel (r 9-10 mm):
    current value and interval mean;
  * physical pressure (p = p_field + p_shift_pa) above the land annulus (2.5 cells above the rim's upper surface when
    on the land) and in the ports; dp_rim = p_above - p_ports;
  * closed-state leakage: a step is CLOSED when the rim contact fraction is >= 0.5; the refill-path flow of a closed
    step is leakage past the flap.  It is split by the largest rim gap of that step: UNDER-RESOLVED while
    max rim gap < 2 dz_land (the IB_4 half width in the gap-normal direction: the opening holds less than the kernel
    half support, the flow through it is numerically dominated) and RESOLVED otherwise.  closed_leakage_m3s (= the
    under-resolved + resolved parts) are interval means (closed-step volume / interval time); *_volume_m3 cumulative;
  * rim shut at grid resolution, with or without contact: steps whose largest rim gap is below 2 dz_land.  The
    refill-path flow of those steps (refill_flow_rim_underresolved_m3s, interval mean; volume cumulative) passes an
    opening the grid does not resolve.  In single-velocity diffuse IB the flap can shut the path on a sub-kernel fluid
    film whose fluid lies inside the flap's kernel support and cannot drain, without the contact law engaging (seen
    in the CPU rig); this flow is numerically dominated and is what the grid comparison (T9) must separate;
  * contact begin / end events (closed state changes between steps), appended to the create-only JSONL
    event_log_path (attribute; assigning a path creates the file with mode 'x').

REUSE CHECK (REQ_400).  Looked at, read-only, in reference_libs/ (local copies of the upstream sources):
  * ibamr-v0.19.0 examples (grep valve, leaflet, contact, penalty) and src/include: no contact model and no valve or
    leaflet example in v0.19.0.  src/IB/IBStandardForceGen.cpp (target points, ~lines 1240-1275: F = K (X_t - X) -
    E U) and examples/IBFE/explicit/ex5 (penalty tethers kappa_s (X - x) - eta_s U, KAPPA_S = 7.5 DX/DT^2): REUSED as
    algorithm (rewritten, no code copied): a penalty spring on the IB points whose force is spread with the IB kernel,
    here one-sided (repulsive only) along the SDF normal, without the damper.
  * taichi_mpm-3a680ab0 src/transfer.cpp (~lines 666-676, MLS-CPIC): normal-only penetration penalty against rigid
    level sets (signed distance + boundary normal per particle).  REUSED as algorithm: signed-distance normal and
    normal-only penalty; the CPIC compatibility machinery is not used.
  * Repository code reused: cases/squid_soft_robot/closed_solid_cells.classify_closed_solid_cells (sign) and scipy
    cKDTree (candidate triangles); the triangles come from cases/squid_soft_robot/setup._region_triangles_m (passed in
    by the integrator).  closed_solid_cells.triangle_surface_samples was used first and replaced by the area-uniform
    surface_samples here: its lattice is sized by the longest edge in both directions, so the CAD's sliver
    triangles were oversampled (squid regions 3+4: 3.18 M samples, 448 M exact pairs, 898 s).  trimesh's exact
    proximity needs rtree, which is not installed (no installs): the exact point-triangle distance is written here
    (projection + barycentric test, else nearest edge).
  IBAMR copyright (c) 2002 - 2026 by the IBAMR developers, BSD-3-Clause; taichi_mpm copyright (c) 2018 Taichi MPM
  Authors, MIT (algorithms adapted, attribution retained).
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import taichi as ti

RIGID_CONTACT_REVISION = "REQ404-rigid-contact-v5-sdf-cache"
TAG_VALVE_RIM = 2              # bit 2 of body.tag; same value as simulation_core.fast.ib_mpm_fast3d.TAG_VALVE_RIM
CONTACT_OMEGA_DT = 0.25        # tether convention k = rho_s (0.25 / dt)^2
BAND_CELLS = 2                 # exact SDF within 2 local cells: the trilinear stencil of any point with phi < d_c
REACH_MARGIN_M = 0.005         # exact SDF only within this distance of a body point (reference positions)
SAMPLE_SPACING_CELLS = 2.0     # candidate-search surface samples every 2 smallest cells (search cost only)
KERNEL_HALF_WIDTH_CELLS = 2    # IB_4 half support
CONTACT_EVENT_FRACTION = 0.5   # rim contact fraction at which a step counts as closed (contact begin / end)
CANDIDATE_SAMPLES = 64         # nearest surface samples examined per cell (quadrupled until provably complete)
PAIR_BUDGET = 1_000_000        # point-triangle pairs per exact-distance batch (bounds the transient host memory)
SDF_CACHE_VERSION = "rigid-contact-sdf-v1"   # bump whenever build_signed_distance changes its result
EVENT_SLOTS = 16               # contact events recorded on the device between two reports
EVENT_WIDTH = 6                # kind, time in interval, rim fraction, max rim gap, refill flow, contact force
HARD_STOP_CODES = {
    "contact_setup_invalid",      # SDF / geometry inconsistent (construction)
    "contact_nonfinite_state",    # nonfinite phi at a point or nonfinite refill flow
}

# stats slots (f64): per-step scratch, interval accumulators (reset by report), persistent state, measurement
S_RIM_CONTACT, S_RIM_GAP_MAX, S_Q, S_FORCE, S_POINTS, S_UNDEF, S_NONFINITE = range(7)
A_STEPS, A_TIME, A_REFILL_VOL, A_CLOSED_TIME, A_CLOSED_VOL, A_UNDER_VOL, A_RES_VOL = range(10, 17)
A_IMPULSE, A_MAX_PEN, A_MIN_RIM_GAP, A_NONFINITE, A_EVENTS, A_UNDEF = range(17, 23)
P_PREV_CLOSED = 24
A_GAP_TIME, A_GAP_VOL = 25, 26     # steps with max rim gap < kernel half width (shut at grid resolution), any contact
M_FORCE, M_POINTS, M_AREA, M_ENERGY, M_MIN_PHI, M_MAX_PEN, M_FX, M_FY, M_FZ = range(30, 39)
M_UNDEF, M_NONFINITE, M_RIM_CONTACT, M_RIM_GAP_MIN, M_RIM_GAP_MAX, M_RIM_GAP_SUM = range(39, 45)
M_Q_PORTS, M_Q_CHANNEL, M_P_ABOVE, M_P_PORTS = range(45, 49)
EV_BASE = 64
STATS_SLOTS = EV_BASE + EVENT_SLOTS * EVENT_WIDTH


class FastModeHardStop(RuntimeError):
    """A fast-mode hard stop with its machine-readable code and witness (same signature as ib_mpm_fast3d's)."""

    def __init__(self, code, witness):
        super().__init__("%s: %s" % (code, witness))
        self.code = code
        self.witness = witness


# ---------------------------------------------------------------------------------------------------- host geometry
def _axis_faces(origin, widths):
    """Face coordinates (n+1) of one axis, accumulated sequentially in float64 like CartesianGrid._axis_faces."""
    return float(origin) + np.concatenate([[0.0], np.cumsum(np.asarray(widths, dtype=np.float64))])


def _cell_index(faces, coord):
    i = int(np.searchsorted(faces, float(coord), side="right")) - 1
    return min(max(i, 0), len(faces) - 2)


def _column_radius_bounds(faces_x, faces_y, axis_xy):
    """Minimum and maximum distance from the axis to each closed column footprint, shape (nx, ny)."""
    ax, ay = axis_xy
    x0, x1 = faces_x[:-1], faces_x[1:]
    y0, y1 = faces_y[:-1], faces_y[1:]
    dx_min = np.maximum(np.maximum(x0 - ax, ax - x1), 0.0)
    dy_min = np.maximum(np.maximum(y0 - ay, ay - y1), 0.0)
    dx_max = np.maximum(np.abs(x0 - ax), np.abs(x1 - ax))
    dy_max = np.maximum(np.abs(y0 - ay), np.abs(y1 - ay))
    return (np.sqrt(dx_min[:, None] ** 2 + dy_min[None, :] ** 2),
            np.sqrt(dx_max[:, None] ** 2 + dy_max[None, :] ** 2))


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

    from cases.squid_setup.closed_solid_cells import classify_closed_solid_cells

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


def sdf_cache_key(rigid_triangles_m, grid, *, band_cells=BAND_CELLS, sample_spacing_m=None, reach_points_m=None,
                  reach_margin_m=None, sign_mask=None):
    """SHA-256 over every input of build_signed_distance: algorithm version, parameters, region names and triangles,
    the grid's cell faces, the reach points and the sign mask (each length-prefixed)."""
    h = hashlib.sha256()

    def add(tag, data):
        h.update(tag.encode())
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    add("version", SDF_CACHE_VERSION.encode())
    add("params", json.dumps([int(band_cells), None if sample_spacing_m is None else float(sample_spacing_m),
                              None if reach_margin_m is None else float(reach_margin_m)]).encode())
    for name, tri in _as_region_dict(rigid_triangles_m):
        add("region", name.encode())
        add("triangles", np.ascontiguousarray(tri, dtype=np.float64).tobytes())
    for faces in (grid.cell_faces_x_m, grid.cell_faces_y_m, grid.cell_faces_z_m):
        add("faces", np.ascontiguousarray(faces, dtype=np.float64).tobytes())
    add("reach", b"none" if reach_points_m is None else
        np.ascontiguousarray(reach_points_m, dtype=np.float64).tobytes())
    add("sign", b"cad" if sign_mask is None else np.ascontiguousarray(sign_mask, dtype=np.uint8).tobytes())
    return h.hexdigest()


def cached_signed_distance(cache_dir, rigid_triangles_m, grid, **kwargs):
    """build_signed_distance with an on-disk cache keyed by sdf_cache_key (Root 2026-10-12: build once, reuse in later
    runs).  Hit: the stored phi and report (report['cache'] = 'hit').  Miss: build, then write the file create-only
    (mode 'x'; an existing file is never overwritten, a concurrent writer's file is left alone)."""
    cache_dir = Path(cache_dir)
    key = sdf_cache_key(rigid_triangles_m, grid, **kwargs)
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
    try:
        with path.open("xb") as stream:
            np.savez(stream, phi=phi, report_json=json.dumps(report), key=key)
        state = "miss (written)"
    except FileExistsError:
        state = "miss (file appeared meanwhile; kept)"
    return phi, dict(report, cache=state, cache_path=str(path), cache_key=key)


# ---------------------------------------------------------------------------------------------------- contact
@ti.data_oriented
class RigidContact:
    """Repulsive frictionless penalty contact of the non-tethered IB points against the rigid CAD regions, plus the
    read-only valve diagnostics (module docstring)."""

    def __init__(self, fluid, axes, body, rigid_triangles_m, *, d_c_m=None, k_c=None,
                 axis_xy_m=(-0.0313309, 0.0159178), land_z_m=1.007000, land_r_m=(0.0280, 0.02999),
                 port_r_m=(0.0230, 0.0280), channel_r_m=(0.0090, 0.0100), refill_plane_z_m=1.0035,
                 flap_thickness_m=0.00250, solid_density_kgm3=None, band_cells=BAND_CELLS, sample_spacing_m=None,
                 reach_margin_m=REACH_MARGIN_M, sign_from_obstacle=True, event_log_path=None, sdf_cache_dir=None):
        """fluid: CartesianFluidSolver; axes: contract 1.1 axes (.n checked); body: ti fields x, vol, tether, tag,
        f_extra (FastIbMpm3D); rigid_triangles_m: dict region -> (n, 3, 3) triangles in metres (e.g.
        cases.squid_soft_robot.setup._region_triangles_m(config, (3, 4))) or one array.  k_c None -> rho_s (0.25/dt)^2
        at each apply_point_forces(dt).  d_c_m None -> the support distance of each point's material box along the
        contact normal.  reach_margin_m None -> exact SDF in the whole band.  Defaults of the valve diagnostics are
        the squid valve (VALVE_FACTS_01); channel_r_m=None disables the channel term."""
        self.fluid, self.axes, self.body = fluid, axes, body
        self.axis_xy_m = tuple(float(v) for v in axis_xy_m)
        self.land_z_m = float(land_z_m)
        self.land_r_m = tuple(float(v) for v in land_r_m)
        self.port_r_m = tuple(float(v) for v in port_r_m)
        self.channel_r_m = None if channel_r_m is None else tuple(float(v) for v in channel_r_m)
        self.refill_plane_z_m = float(refill_plane_z_m)
        self.flap_thickness_m = float(flap_thickness_m)
        grid = fluid.grid
        self.faces = [_axis_faces(grid.bounds_min_m[a], w) for a, w in
                      enumerate((grid.cell_widths_x_m, grid.cell_widths_y_m, grid.cell_widths_z_m))]
        self.nodes = tuple(len(f) - 1 for f in self.faces)
        if tuple(int(v) for v in axes.n) != self.nodes:
            raise ValueError("axes.n %s != fluid grid %s" % (tuple(axes.n), self.nodes))
        if k_c is not None and not (math.isfinite(float(k_c)) and float(k_c) > 0.0):
            raise ValueError("k_c must be finite and positive")
        self.k_c = None if k_c is None else float(k_c)
        rho_s = solid_density_kgm3 if solid_density_kgm3 is not None else getattr(body, "rho_s", None)
        if self.k_c is None and (rho_s is None or not float(rho_s) > 0.0):
            raise ValueError("k_c or a solid density (solid_density_kgm3 / body.rho_s) is needed")
        self.rho_s = 0.0 if rho_s is None else float(rho_s)
        self._log, self._event_log_path = None, None
        self.event_log_path = event_log_path
        reach = None
        if reach_margin_m is not None:
            reach = np.asarray((body.X0 if hasattr(body, "X0") else body.x).to_numpy(), dtype=np.float64)
        sign_mask = (np.asarray(fluid.obstacle.to_numpy()) != 0) if sign_from_obstacle else None
        sdf_kwargs = dict(band_cells=band_cells, sample_spacing_m=sample_spacing_m, reach_points_m=reach,
                          reach_margin_m=reach_margin_m, sign_mask=sign_mask)
        try:
            if sdf_cache_dir is None:
                phi, self.sdf_report = build_signed_distance(rigid_triangles_m, grid, **sdf_kwargs)
            else:
                phi, self.sdf_report = cached_signed_distance(sdf_cache_dir, rigid_triangles_m, grid, **sdf_kwargs)
        except ValueError as error:
            raise FastModeHardStop("contact_setup_invalid", dict(reason=str(error))) from None
        self._setup_points(phi, d_c_m)
        self._setup_diagnostics()
        self.stats = ti.field(ti.f64, shape=STATS_SLOTS)
        self._init_stats_kernel()
        self._events = []
        self._last_report_t = None
        self._totals = dict(time=0.0, closed_time=0.0, closed=0.0, underresolved=0.0, resolved=0.0, refill=0.0,
                            gap_time=0.0, gap_volume=0.0)

    # ------------------------------------------------------------------------------------------------ setup
    def _setup_points(self, phi, d_c_m):
        body = self.body
        self.n = int(body.n) if hasattr(body, "n") else int(body.x.shape[0])
        nx, ny, nz = self.nodes
        self.phi = ti.field(ti.f32, shape=self.nodes)
        self.phi.from_numpy(phi.astype(np.float32))
        self.centre_x = ti.field(ti.f32, shape=nx)
        self.centre_y = ti.field(ti.f32, shape=ny)
        self.centre_z = ti.field(ti.f32, shape=nz)
        for field, f in zip((self.centre_x, self.centre_y, self.centre_z), self.faces):
            field.from_numpy((0.5 * (f[:-1] + f[1:])).astype(np.float32))
        vol = np.asarray(body.vol.to_numpy(), dtype=np.float64)
        if vol.shape != (self.n,) or not (vol > 0.0).all():
            raise FastModeHardStop("contact_setup_invalid", dict(reason="point volumes must be positive"))
        # material box of each point: volume V0, aspect ratio of the fluid cell holding its reference position
        x0 = np.asarray((body.X0 if hasattr(body, "X0") else body.x).to_numpy(), dtype=np.float64)
        widths = []
        for a, f in enumerate(self.faces):
            i = np.clip(np.searchsorted(f, x0[:, a], side="right") - 1, 0, len(f) - 2)
            widths.append(np.diff(f)[i])
        widths = np.column_stack(widths)
        half = 0.5 * widths * np.cbrt(vol / np.prod(widths, axis=1))[:, None]
        self.half_extent = ti.Vector.field(3, ti.f32, shape=self.n)
        self.half_extent.from_numpy(half.astype(np.float32))
        if d_c_m is None:
            self.fixed_dc = 0.0
            self.d_c_rule = ("support distance of the point's material box along the normal, d_c = a . |n| (box of "
                             "volume V0 with the aspect ratio of the fluid cell holding X0)")
            self.d_c_range_m = (float(half.min()), float(half.sum(axis=1).max()))
        else:
            if not (math.isfinite(float(d_c_m)) and float(d_c_m) > 0.0):
                raise ValueError("d_c_m must be finite and positive")
            self.fixed_dc = float(d_c_m)
            self.d_c_rule = "fixed %.6g m" % self.fixed_dc
            self.d_c_range_m = (self.fixed_dc, self.fixed_dc)
        self.half_extent_z_range_m = (float(half[:, 2].min()), float(half[:, 2].max()))
        self._half_np = half

    def _setup_diagnostics(self):
        fx, fy, fz = self.faces
        obstacle = np.asarray(self.fluid.obstacle.to_numpy(), dtype=np.int32)
        area = np.diff(fx)[:, None] * np.diff(fy)[None, :]
        r_min, r_max = _column_radius_bounds(fx, fy, self.axis_xy_m)
        cx, cy = 0.5 * (fx[:-1] + fx[1:]), 0.5 * (fy[:-1] + fy[1:])
        r_c = np.hypot(cx[:, None] - self.axis_xy_m[0], cy[None, :] - self.axis_xy_m[1])
        tag = np.asarray(self.body.tag.to_numpy())
        rim = np.flatnonzero((tag & TAG_VALVE_RIM) != 0).astype(np.int32)
        self.rim_count = int(rim.size)
        self.rim_index = ti.field(ti.i32, shape=max(1, rim.size))
        self.rim_half = ti.field(ti.f32, shape=max(1, rim.size))     # vertical half extent of the material box
        if rim.size:
            self.rim_index.from_numpy(rim)
            self.rim_half.from_numpy(self._half_np[rim, 2].astype(np.float32))
        k_land = _cell_index(fz, self.land_z_m + 1.0e-9)
        self.dz_land_m = float(fz[k_land + 1] - fz[k_land])
        self.kernel_half_width_m = KERNEL_HALF_WIDTH_CELLS * self.dz_land_m
        # refill faces: z faces at the refill plane, both adjacent cells fluid (the plate holes)
        k_f = int(np.argmin(np.abs(fz - self.refill_plane_z_m)))
        k_f = min(max(k_f, 1), len(fz) - 2)
        self.k_refill_face = k_f
        open_face = (obstacle[:, :, k_f] == 0) & (obstacle[:, :, k_f - 1] == 0)
        port_cols = (r_min <= self.port_r_m[1]) & (r_max >= self.port_r_m[0]) & open_face
        chan_cols = np.zeros_like(port_cols)
        if self.channel_r_m is not None:
            chan_cols = (r_min <= self.channel_r_m[1]) & (r_max >= self.channel_r_m[0]) & open_face & ~port_cols
        faces = np.concatenate([np.argwhere(port_cols), np.argwhere(chan_cols)]).astype(np.int32)
        self.n_port_faces, self.n_channel_faces = int(port_cols.sum()), int(chan_cols.sum())
        if self.n_port_faces == 0:
            raise FastModeHardStop("contact_setup_invalid", dict(reason="no open port face at the refill plane",
                                                                 k=k_f, port_r_m=self.port_r_m))
        self.n_faces = len(faces)
        # pressure samples: above the land annulus 2.5 cells above the rim's upper surface on the land; in the ports
        k_top = _cell_index(fz, self.land_z_m + self.flap_thickness_m)
        self.k_above = min(k_top + KERNEL_HALF_WIDTH_CELLS + 1, len(fz) - 2)
        above = np.argwhere((r_c >= self.land_r_m[0]) & (r_c <= self.land_r_m[1])
                            & (obstacle[:, :, self.k_above] == 0)).astype(np.int32)
        below = np.argwhere(port_cols).astype(np.int32)
        self.n_above, self.n_below = len(above), len(below)
        dzc = np.diff(fz)
        self.vol_above_m3 = float((area[above[:, 0], above[:, 1]] * dzc[self.k_above]).sum()) if len(above) else 0.0
        self.vol_below_m3 = float((area[below[:, 0], below[:, 1]] * dzc[k_f]).sum()) if len(below) else 0.0

        def cells(cols, k, extra):
            out = np.zeros((max(1, len(cols)), 4), np.float64)
            if len(cols):
                out[:, 0], out[:, 1], out[:, 2] = cols[:, 0], cols[:, 1], k
                out[:, 3] = extra
            return out

        # (i, j, k, weight): face area for the refill faces, cell volume for the pressure samples
        self.face_cell = ti.Vector.field(4, ti.f64, shape=max(1, self.n_faces))
        self.above_cell = ti.Vector.field(4, ti.f64, shape=max(1, self.n_above))
        self.below_cell = ti.Vector.field(4, ti.f64, shape=max(1, self.n_below))
        self.face_cell.from_numpy(cells(faces, k_f, area[faces[:, 0], faces[:, 1]] if len(faces) else 0.0))
        self.above_cell.from_numpy(cells(above, self.k_above,
                                         area[above[:, 0], above[:, 1]] * dzc[self.k_above] if len(above) else 0.0))
        self.below_cell.from_numpy(cells(below, k_f, area[below[:, 0], below[:, 1]] * dzc[k_f] if len(below) else 0.0))

    # ------------------------------------------------------------------------------------------------ kernels
    @ti.func
    def _locate(self, centres: ti.template(), n: ti.template(), coord):
        """Centre interval i (centres[i] <= coord < centres[i+1], clamped) and the fraction t in [0, 1]."""
        i = 0
        t = 0.0
        if coord <= centres[0]:
            i, t = 0, 0.0
        elif coord >= centres[n - 1]:
            i, t = n - 2, 1.0
        else:
            lo, hi = 0, n - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if centres[mid] <= coord:
                    lo = mid
                else:
                    hi = mid
            i = lo
            t = (coord - centres[lo]) / (centres[lo + 1] - centres[lo])
        return i, t

    @ti.func
    def _phi_grad(self, xp):
        """Trilinear phi and its exact gradient at xp."""
        i, tx = self._locate(self.centre_x, self.nodes[0], xp[0])
        j, ty = self._locate(self.centre_y, self.nodes[1], xp[1])
        k, tz = self._locate(self.centre_z, self.nodes[2], xp[2])
        hx = self.centre_x[i + 1] - self.centre_x[i]
        hy = self.centre_y[j + 1] - self.centre_y[j]
        hz = self.centre_z[k + 1] - self.centre_z[k]
        phi = 0.0
        g = ti.Vector([0.0, 0.0, 0.0])
        for a, b, c in ti.static(ti.ndrange(2, 2, 2)):
            wx = tx if a == 1 else 1.0 - tx
            wy = ty if b == 1 else 1.0 - ty
            wz = tz if c == 1 else 1.0 - tz
            sx = 1.0 if a == 1 else -1.0
            sy = 1.0 if b == 1 else -1.0
            sz = 1.0 if c == 1 else -1.0
            v = self.phi[i + a, j + b, k + c]
            phi += wx * wy * wz * v
            g += ti.Vector([sx * wy * wz / hx, wx * sy * wz / hy, wx * wy * sz / hz]) * v
        return phi, g

    @ti.func
    def _contact_distance(self, p, nrm):
        """Contact distance of point p along the unit normal: the support distance a . |n| of its material box, or
        the fixed d_c_m."""
        d = 0.0
        if ti.static(self.fixed_dc > 0.0):
            d = self.fixed_dc
        else:
            a = self.half_extent[p]
            d = a[0] * ti.abs(nrm[0]) + a[1] * ti.abs(nrm[1]) + a[2] * ti.abs(nrm[2])
        return d

    @ti.func
    def _contact_area(self, p, nrm):
        """Area of the material box of point p projected along the unit normal (its touching face for an
        axis-aligned contact)."""
        a = self.half_extent[p]
        return 4.0 * (a[1] * a[2] * ti.abs(nrm[0]) + a[0] * a[2] * ti.abs(nrm[1]) + a[0] * a[1] * ti.abs(nrm[2]))

    @ti.func
    def _face_flow(self, velocity: ti.template(), s):
        c = self.face_cell[s]
        u = velocity[ti.cast(c[0], ti.i32), ti.cast(c[1], ti.i32), ti.cast(c[2], ti.i32)][2]
        return ti.cast(u, ti.f64) * c[3]

    @ti.kernel
    def _init_stats_kernel(self):
        for s in range(STATS_SLOTS):
            self.stats[s] = 0.0
        self.stats[A_MIN_RIM_GAP] = 1.0e30
        self.stats[P_PREV_CLOSED] = -1.0

    @ti.kernel
    def _reset_interval_kernel(self):
        for s in range(A_STEPS, A_UNDEF + 1):
            self.stats[s] = 0.0
        for s in range(EV_BASE, STATS_SLOTS):
            self.stats[s] = 0.0
        self.stats[A_MIN_RIM_GAP] = 1.0e30
        self.stats[A_GAP_TIME] = 0.0
        self.stats[A_GAP_VOL] = 0.0

    @ti.kernel
    def _step_kernel(self, x: ti.template(), vol: ti.template(), tether: ti.template(), f_extra: ti.template(),
                     velocity: ti.template(), dt: ti.f32, k_fixed: ti.f32, rho_s: ti.f32, z_land: ti.f32,
                     half_width: ti.f64):
        k_c = k_fixed
        if k_c <= 0.0:
            omega = CONTACT_OMEGA_DT / dt
            k_c = rho_s * omega * omega
        for s in range(S_NONFINITE + 1):
            self.stats[s] = 0.0
        self.stats[S_RIM_GAP_MAX] = -1.0e30
        for p in range(self.n):
            if tether[p] == 0:
                phi, g = self._phi_grad(x[p])
                gn = g.norm()
                if gn > 1.0e-6:
                    nrm = g / gn
                    dc = self._contact_distance(p, nrm)
                    if phi < dc:
                        mag = k_c * vol[p] * (dc - phi)
                        f = f_extra[p]
                        f += mag * nrm
                        f_extra[p] = f
                        ti.atomic_add(self.stats[S_FORCE], ti.cast(mag, ti.f64))
                        ti.atomic_add(self.stats[S_POINTS], 1.0)
                else:
                    ti.atomic_add(self.stats[S_UNDEF], 1.0)
                ti.atomic_max(self.stats[A_MAX_PEN], ti.cast(ti.max(-phi, 0.0), ti.f64))
                if ti.math.isnan(phi) or ti.math.isinf(phi):
                    ti.atomic_add(self.stats[S_NONFINITE], 1.0)
        for q in range(self.rim_count):
            p = self.rim_index[q]
            phi, g = self._phi_grad(x[p])
            gn = g.norm()
            if gn > 1.0e-6:
                if phi < self._contact_distance(p, g / gn):
                    ti.atomic_add(self.stats[S_RIM_CONTACT], 1.0)
            gap = ti.cast(x[p][2] - self.rim_half[q] - z_land, ti.f64)
            ti.atomic_max(self.stats[S_RIM_GAP_MAX], gap)
            ti.atomic_min(self.stats[A_MIN_RIM_GAP], gap)
        for s in range(self.n_faces):
            ti.atomic_add(self.stats[S_Q], self._face_flow(velocity, s))
        # serial accumulation of the interval statistics
        dt64 = ti.cast(dt, ti.f64)
        q = self.stats[S_Q]
        tau = self.stats[A_TIME] + 0.5 * dt64
        self.stats[A_STEPS] += 1.0
        self.stats[A_TIME] += dt64
        self.stats[A_REFILL_VOL] += q * dt64
        self.stats[A_IMPULSE] += self.stats[S_FORCE] * dt64
        self.stats[A_NONFINITE] += self.stats[S_NONFINITE]
        self.stats[A_UNDEF] += self.stats[S_UNDEF]
        closed = 0.0
        if ti.static(self.rim_count > 0):
            if self.stats[S_RIM_CONTACT] >= CONTACT_EVENT_FRACTION * self.rim_count:
                closed = 1.0
        if closed > 0.5:
            self.stats[A_CLOSED_TIME] += dt64
            self.stats[A_CLOSED_VOL] += q * dt64
            if self.stats[S_RIM_GAP_MAX] >= half_width:
                self.stats[A_RES_VOL] += q * dt64
            else:
                self.stats[A_UNDER_VOL] += q * dt64
        if ti.static(self.rim_count > 0):
            if self.stats[S_RIM_GAP_MAX] < half_width:
                self.stats[A_GAP_TIME] += dt64
                self.stats[A_GAP_VOL] += q * dt64
        prev = self.stats[P_PREV_CLOSED]
        if prev >= 0.0 and closed != prev:
            slot = ti.cast(self.stats[A_EVENTS], ti.i32)
            if slot < EVENT_SLOTS:
                base = EV_BASE + slot * EVENT_WIDTH
                self.stats[base] = ti.select(closed > 0.5, 1.0, -1.0)
                self.stats[base + 1] = tau
                self.stats[base + 2] = self.stats[S_RIM_CONTACT] / ti.max(1.0, ti.cast(self.rim_count, ti.f64))
                self.stats[base + 3] = self.stats[S_RIM_GAP_MAX]
                self.stats[base + 4] = q
                self.stats[base + 5] = self.stats[S_FORCE]
            self.stats[A_EVENTS] += 1.0
        self.stats[P_PREV_CLOSED] = closed

    @ti.kernel
    def _measure_kernel(self, x: ti.template(), vol: ti.template(), tether: ti.template(), k_c: ti.f32,
                        pressure: ti.template(), velocity: ti.template(), z_land: ti.f32):
        for s in range(M_FORCE, M_P_PORTS + 1):
            self.stats[s] = 0.0
        self.stats[M_MIN_PHI] = 1.0e30
        self.stats[M_RIM_GAP_MIN] = 1.0e30
        self.stats[M_RIM_GAP_MAX] = -1.0e30
        for p in range(self.n):
            if tether[p] == 0:
                phi, g = self._phi_grad(x[p])
                phi64 = ti.cast(phi, ti.f64)
                ti.atomic_min(self.stats[M_MIN_PHI], phi64)
                ti.atomic_max(self.stats[M_MAX_PEN], ti.max(-phi64, 0.0))
                gn = g.norm()
                if gn > 1.0e-6:
                    nrm = g / gn
                    dc = self._contact_distance(p, nrm)
                    if phi < dc:
                        depth = ti.cast(dc - phi, ti.f64)
                        mag = ti.cast(k_c * vol[p], ti.f64) * depth
                        ti.atomic_add(self.stats[M_FORCE], mag)
                        ti.atomic_add(self.stats[M_POINTS], 1.0)
                        ti.atomic_add(self.stats[M_AREA], ti.cast(self._contact_area(p, nrm), ti.f64))
                        ti.atomic_add(self.stats[M_ENERGY], 0.5 * mag * depth)
                        ti.atomic_add(self.stats[M_FX], mag * ti.cast(nrm[0], ti.f64))
                        ti.atomic_add(self.stats[M_FY], mag * ti.cast(nrm[1], ti.f64))
                        ti.atomic_add(self.stats[M_FZ], mag * ti.cast(nrm[2], ti.f64))
                else:
                    ti.atomic_add(self.stats[M_UNDEF], 1.0)
                if ti.math.isnan(phi) or ti.math.isinf(phi):
                    ti.atomic_add(self.stats[M_NONFINITE], 1.0)
        for q in range(self.rim_count):
            p = self.rim_index[q]
            phi, g = self._phi_grad(x[p])
            gn = g.norm()
            if gn > 1.0e-6:
                if phi < self._contact_distance(p, g / gn):
                    ti.atomic_add(self.stats[M_RIM_CONTACT], 1.0)
            gap = ti.cast(x[p][2] - self.rim_half[q] - z_land, ti.f64)
            ti.atomic_min(self.stats[M_RIM_GAP_MIN], gap)
            ti.atomic_max(self.stats[M_RIM_GAP_MAX], gap)
            ti.atomic_add(self.stats[M_RIM_GAP_SUM], gap)
        for s in range(self.n_faces):
            if s < self.n_port_faces:
                ti.atomic_add(self.stats[M_Q_PORTS], self._face_flow(velocity, s))
            else:
                ti.atomic_add(self.stats[M_Q_CHANNEL], self._face_flow(velocity, s))
        for s in range(self.n_above):
            c = self.above_cell[s]
            ti.atomic_add(self.stats[M_P_ABOVE], ti.cast(pressure[ti.cast(c[0], ti.i32), ti.cast(c[1], ti.i32),
                                                                   ti.cast(c[2], ti.i32)], ti.f64) * c[3])
        for s in range(self.n_below):
            c = self.below_cell[s]
            ti.atomic_add(self.stats[M_P_PORTS], ti.cast(pressure[ti.cast(c[0], ti.i32), ti.cast(c[1], ti.i32),
                                                                   ti.cast(c[2], ti.i32)], ti.f64) * c[3])

    # ------------------------------------------------------------------------------------------------ host API
    def contact_stiffness(self, dt_s):
        """k_c in N/m^4 (force per unit point volume and unit penetration)."""
        if self.k_c is not None:
            return self.k_c
        omega = CONTACT_OMEGA_DT / float(dt_s)
        return self.rho_s * omega * omega

    def apply_point_forces(self, dt_s):
        """ONE kernel launch: add the contact force of every non-tethered point with phi < d_c to body.f_extra and
        accumulate the interval statistics on the device.  Call right before body.spread_force()."""
        self._step_kernel(self.body.x, self.body.vol, self.body.tether, self.body.f_extra, self.fluid.velocity,
                          dt_s, 0.0 if self.k_c is None else self.k_c, self.rho_s, self.land_z_m,
                          self.kernel_half_width_m)

    def report(self, t_s, dt_s, p_shift_pa=0.0):
        """Diagnostics at the current state plus the interval since the previous report (one device read)."""
        t, dt, shift = float(t_s), float(dt_s), float(p_shift_pa)
        k = self.contact_stiffness(dt)
        self._measure_kernel(self.body.x, self.body.vol, self.body.tether, k, self.fluid.pressure,
                             self.fluid.velocity, self.land_z_m)
        s = self.stats.to_numpy()                                          # the one device read of a report
        self._reset_interval_kernel()
        if s[M_NONFINITE] > 0 or s[A_NONFINITE] > 0 or not np.isfinite(s[:EV_BASE]).all():
            raise FastModeHardStop("contact_nonfinite_state", dict(nonfinite_phi_points=int(s[M_NONFINITE]),
                                                                   interval_nonfinite=int(s[A_NONFINITE]), t_s=t))
        span = float(s[A_TIME])
        n = self.rim_count
        out = dict(t_s=t, k_c=k, interval_steps=int(s[A_STEPS]), interval_time_s=span,
                   contact_force_n=float(s[M_FORCE]), contact_points=int(s[M_POINTS]),
                   contact_area_m2=float(s[M_AREA]), contact_energy_j=float(s[M_ENERGY]),
                   contact_force_vector_n=[float(s[M_FX]), float(s[M_FY]), float(s[M_FZ])],
                   contact_force_interval_mean_n=float(s[A_IMPULSE]) / span if span > 0 else None,
                   normal_undefined_points=int(s[M_UNDEF]),
                   min_phi_m=float(s[M_MIN_PHI]) if s[M_MIN_PHI] < 1.0e29 else None,
                   max_penetration_m=max(float(s[M_MAX_PEN]), float(s[A_MAX_PEN])),
                   max_penetration_now_m=float(s[M_MAX_PEN]), dissipated_energy_j=0.0, d_c_rule=self.d_c_rule)
        fraction = float(s[M_RIM_CONTACT]) / n if n else 0.0
        out.update(rim_contact_fraction=fraction,
                   rim_gap_min_m=float(s[M_RIM_GAP_MIN]) if n else None,
                   rim_gap_max_m=float(s[M_RIM_GAP_MAX]) if n else None,
                   rim_gap_mean_m=float(s[M_RIM_GAP_SUM]) / n if n else None,
                   rim_gap_min_interval_m=float(s[A_MIN_RIM_GAP]) if (n and s[A_MIN_RIM_GAP] < 1.0e29) else None)
        q_ports, q_chan = float(s[M_Q_PORTS]), float(s[M_Q_CHANNEL])
        out.update(refill_flow_m3s=q_ports + q_chan, refill_flow_ports_m3s=q_ports, refill_flow_channel_m3s=q_chan,
                   refill_flow_interval_mean_m3s=float(s[A_REFILL_VOL]) / span if span > 0 else None)
        p_above = float(s[M_P_ABOVE]) / self.vol_above_m3 + shift if self.n_above else None
        p_ports = float(s[M_P_PORTS]) / self.vol_below_m3 + shift if self.n_below else None
        out.update(p_above_land_pa=p_above, p_ports_pa=p_ports, p_shift_pa=shift,
                   dp_rim_pa=(p_above - p_ports) if (p_above is not None and p_ports is not None) else None)
        tot = self._totals
        tot["time"] += span
        tot["closed_time"] += float(s[A_CLOSED_TIME])
        tot["refill"] += float(s[A_REFILL_VOL])
        tot["closed"] += float(s[A_CLOSED_VOL])
        tot["underresolved"] += float(s[A_UNDER_VOL])
        tot["resolved"] += float(s[A_RES_VOL])
        tot["gap_time"] += float(s[A_GAP_TIME])
        tot["gap_volume"] += float(s[A_GAP_VOL])
        out.update(rim_shut_by_grid_time_fraction_interval=float(s[A_GAP_TIME]) / span if span > 0 else 0.0,
                   refill_flow_rim_underresolved_m3s=float(s[A_GAP_VOL]) / span if span > 0 else 0.0,
                   rim_shut_by_grid_time_s=tot["gap_time"], refill_volume_rim_underresolved_m3=tot["gap_volume"])
        out.update(closed=bool(n and fraction >= CONTACT_EVENT_FRACTION), kernel_half_width_m=self.kernel_half_width_m,
                   closed_time_fraction_interval=float(s[A_CLOSED_TIME]) / span if span > 0 else 0.0,
                   closed_leakage_m3s=float(s[A_CLOSED_VOL]) / span if span > 0 else 0.0,
                   closed_leakage_underresolved_m3s=float(s[A_UNDER_VOL]) / span if span > 0 else 0.0,
                   closed_leakage_resolved_m3s=float(s[A_RES_VOL]) / span if span > 0 else 0.0,
                   closed_time_s=tot["closed_time"], refill_volume_m3=tot["refill"],
                   closed_leakage_volume_m3=tot["closed"], closed_leakage_underresolved_volume_m3=tot["underresolved"],
                   closed_leakage_resolved_volume_m3=tot["resolved"])
        count = int(s[A_EVENTS])
        start = t - span
        for slot in range(min(count, EVENT_SLOTS)):
            e = s[EV_BASE + slot * EVENT_WIDTH: EV_BASE + (slot + 1) * EVENT_WIDTH]
            self._event(dict(t_s=start + float(e[1]), kind="contact_begin" if e[0] > 0 else "contact_end",
                             rim_contact_fraction=float(e[2]), rim_gap_max_m=float(e[3]),
                             refill_flow_m3s=float(e[4]), contact_force_n=float(e[5]), reported_at_t_s=t))
        out.update(events=len(self._events), events_lost_interval=max(0, count - EVENT_SLOTS),
                   normal_undefined_steps_interval=int(s[A_UNDEF]))
        self._last_report_t = t
        return out

    @property
    def event_log_path(self):
        return self._event_log_path

    @event_log_path.setter
    def event_log_path(self, path):
        """Assigning a path creates the JSONL file at once with mode 'x' (create-only: an existing file fails here,
        not at the first event); None stops logging."""
        if self._log is not None:
            self._log.close()
            self._log = None
        self._event_log_path = None
        if path is not None:
            self._log = open(str(path), "x", encoding="utf-8")
            self._event_log_path = str(path)

    def _event(self, row):
        self._events.append(row)
        if self._log is not None:
            self._log.write(json.dumps(row, sort_keys=True) + "\n")
            self._log.flush()

    def events(self):
        return [dict(e) for e in self._events]

    def setup_report(self):
        return dict(revision=RIGID_CONTACT_REVISION, sdf=self.sdf_report, d_c_rule=self.d_c_rule,
                    d_c_range_m=list(self.d_c_range_m), half_extent_z_range_m=list(self.half_extent_z_range_m),
                    k_c=self.k_c, rim_points=self.rim_count,
                    dz_land_m=self.dz_land_m, kernel_half_width_m=self.kernel_half_width_m,
                    refill_face_index=self.k_refill_face, port_faces=self.n_port_faces,
                    channel_faces=self.n_channel_faces, pressure_cells=dict(above=self.n_above, ports=self.n_below),
                    friction="none (normal only)", adhesion="none (repulsive only)", damping="none")

    def close(self):
        if self._log is not None:
            self._log.close()
            self._log = None
