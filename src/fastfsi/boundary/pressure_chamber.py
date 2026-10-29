"""A pressure chamber on the fluid grid: a gas region at a prescribed pressure above a deformable surface, imposed on an
unmodified incompressible projection through a shifted penalty.

REGION.  The chamber is a set of grid columns (the footprint, given by the caller) below a lid height.  In each footprint
column the chamber cells are the cells whose centre lies above the lower bound of the column and below the lid, and that
are not obstacles.  The lower bound is the highest point of the chamber surface in the column: the points tagged
TAG_CHAMBER_SURFACE (point_tags) of the deformable body that faces the chamber.  `air_cell` (i32 on the grid) is
recomputed every step on the device from the current point positions and tags.

SURFACE FALLBACK.  A stretched surface can leave footprint columns without a surface point.
  (1) A column WITH surface points uses the rule above.
  (2) A column WITHOUT points takes its lower bound from the surface itself: a Delaunay triangulation of the surface
      points' reference xy (built once at setup, the top point where several share a reference xy), evaluated on the
      CURRENT positions; the triangle whose current xy covers the column centre gives the height by barycentric
      (piecewise-linear) interpolation of the current vertex z.  A triangle is kept only if its circumradius is at most
      half the diagonal of the column holding its centroid (CIRCUMRADIUS_OVER_COLUMN_DIAGONAL; lattice triangles have a
      quarter of the diagonal, hull slivers much more), and counts only while all three vertices are surface points.
  (3) A column centre outside the current surface (no kept triangle covers it) or under an inverted triangle (current
      signed xy area <= 0) raises the hard stop pressure_chamber_column_without_surface; the witness names the reason
      and the distance to the nearest current surface boundary edge.  edge_extension_max_m (default None = off) lets a
      column whose centre lies outside the surface but within that distance of it take the height of the nearest
      boundary edge (linear along the edge).
  Candidate triangles per column come from a table built at setup (triangles whose reference xy box grown by
  candidate_margin_m contains the column centre), re-checked on the current positions; if no candidate covers the
  centre every kept triangle is searched (exact; counted as a table miss), so the margin sets only the speed.
  Each report() writes one JSONL line to fallback_events_path (create-only) when a fallback happened since the last
  report, and returns a cumulative summary.

MATERIAL-TOP BOUND (option lower_bound="material", default "surface").  Where the deformable body lines a vertical wall its
chamber-facing surface turns sideways into the wall column, and the next column inward holds the body's own material
but only low surface points; with the surface rule the cells holding that material become chamber cells and the
one-sided interpolation support of the body empties from the chamber side.  With "material" the lower bound of every
footprint column that HAS a bound (a surface point or a fallback height) becomes max(that bound, the highest point of
body material_body_id, any layer, in the column): the chamber cannot lie inside the material column.  Unchanged: a
column without a bound still raises the hard stop, and gas below an overhang of the surface is not detected (those
cells stay liquid).  Identical to the surface rule wherever the highest material point of each column is a surface
point.  Counters (report()["material_top_bound"]): raised column-steps and the largest raise per report interval and
in total, the raised columns of the last step.

PRESSURE BOUNDARY WITHOUT EDITING THE SOLVER.  The projection solves

    sum_nb T p_nb - (den + D) p = rhs_scale (div* + s - volume_source) - R,   rhs_scale = rho / dt,

with D = fluid.pressure_interface_matrix_diagonal and R = fluid.pressure_interface_matrix_rhs (f64, per cell) and den the
cell's own finite-volume denominator.  The conjugate-gradient stopping rule is relative (||r|| / ||b||), so no large
value may enter R.  The solver therefore solves the shifted pressure p~ = p - p_chamber(t):
  1. chamber cells: D = c_pen * den (penalty factor c_pen, default DEFAULT_PENALTY_FACTOR), R = 0, so p~ ~ 0 there;
     the deviation is ~ (local jump of p~) / (c_pen * den / T);
  2. z-min outlet cells (k = 0, liquid, not chamber): R = T_out * p~_out with p~_out = -p_chamber(t), where
     T_out = w_face * 2 / dz_0^2 is the solver's own outlet ghost coefficient (ghost pressure at the face, half a cell
     from the centre); the solver uses ghost value 0 there, and moving T_out p~_out to the right side gives R; with
     outlet_inflow="total_pressure" an inflow face (stored w > 0) has p~_out = -p_chamber(t) - rho w^2 / 2 (total
     pressure 0; outlet_guard.outlet_ghost_pressure, the module docstring there);
  3. after the projection the solver has corrected the outlet face with ghost 0, so the missing term
     u_z(i, j, 0) += w_face * (dt / rho) * 2 p~_out / dz_0 = (dt / rho) * R * dz_0 is added by after_project(), which
     also moves the interface flux the solver booked as s = (dt / rho) (D p - R) onto the face (s += (dt / rho) R),
     keeping div + s = 0 with the face divergence-free;
  4. physical pressure: p = p~ + p_chamber(t) (physical_pressure_into / physical_pressure_field); report() gives
     physical values.
The positive diagonal without interface rows passes the solver's operator checks; with the multigrid preconditioner
the fine diagonal is restricted to the coarse levels on every solve, so nothing else has to be refreshed per step.

DRIVER REQUIREMENTS: project(..., **PROJECT_REQUIREMENTS) (z-min pressure outlet, conjugate gradient with the multigrid
preconditioner, outlet backflow allowed: in the shifted variable an inflow outlet face is the normal case, and a clamp
of inflow faces would make the outlet correction wrong; no pressure accumulation).  Per step: update(t_mid, p_chamber)
-> project -> after_project(dt, p_chamber) with the same p_chamber.

Per step update() and after_project() are one kernel launch each with no host reads; the hard stops (a column without
surface and without a fallback height, no chamber cell at all) are counted on the device and raised by report().

The Dirichlet treatment follows the idea of row elimination (PETSc MatZeroRowsColumns: known values times the
eliminated columns move into b) for the outlet ghost value; the diagonal penalty on the shifted pressure for interior
chamber cells is this library's construction (IBAMR penalises the momentum equation instead, BrinkmanPenalization).
scipy.spatial.Delaunay (Qhull) builds the setup triangulation.

No `from __future__ import annotations` (Taichi kernel module).
"""
import json
import math

import numpy as np
import taichi as ti

from fastfsi.boundary.outlet_guard import PROJECT_REQUIREMENTS as SHIFTED_OUTLET_REQUIREMENTS
from fastfsi.boundary.outlet_guard import OutletShiftGuard, merge_project_requirements  # noqa: F401 (re-export)
from fastfsi.boundary.outlet_guard import check_outlet_inflow, outlet_ghost_pressure, outlet_half_rho_field
from fastfsi.errors import SimulationHardStop
from fastfsi.io.checkpoint import Checkpointable
from fastfsi.point_tags import TAG_CHAMBER_SURFACE

HARD_STOP_CODES = {"pressure_chamber_column_without_surface", "pressure_chamber_no_chamber_cells"}
DEFAULT_PENALTY_FACTOR = 1.0e7   # |p~_air| <~ jump / (2 c) <= 5e-8 p_air; CG / MG counts unchanged (duct test)
DEFAULT_CANDIDATE_MARGIN_M = 2.0e-3      # candidate-table margin (m): affects the search speed only, never the result
CIRCUMRADIUS_OVER_COLUMN_DIAGONAL = 0.5  # keep triangles with circumradius <= 0.5 x their column's diagonal
                                         # (sub-lattice triangles: 0.25; hull slivers / long hull triangles: much more)
BARYCENTRIC_TOLERANCE = 1.0e-9           # relative: a centre on a shared edge belongs to both triangles
PROJECT_REQUIREMENTS = SHIFTED_OUTLET_REQUIREMENTS   # the shifted outlet's projection contract (outlet_guard.py)
HOLE_REASONS = {1: "no_surface_point_and_outside_the_current_surface", 2: "no_surface_point_and_inverted_covering_triangle"}
# stats slots (one f64 field, read once by report())
S_HOLE_EVENTS, S_HOLE_T, S_HOLE_I, S_HOLE_J, S_NO_AIR_EVENTS, S_NO_AIR_T = 0, 1, 2, 3, 4, 5
S_STEP_AIR_CELLS, S_STEP_HOLES = 6, 7
S_AIR_CELLS, S_AIR_VOLUME, S_AIR_PMAX, S_OUTLET_FLUX, S_WATER_PMIN, S_WATER_PMAX = 8, 9, 10, 11, 12, 13
S_DIV_WATER, S_DIV_OUTLET, S_OUTLET_CELLS, S_STEPS = 14, 15, 16, 17
# fallback counters (interval = since the last report, reset by report(); total = since the start)
S_HOLE_REASON, S_HOLE_DIST = 20, 21
S_FB_STEP_COLS, S_FB_INT_COLSTEPS, S_FB_INT_MAX_COLS, S_FB_INT_MAX_DIST, S_FB_INT_MISSES, S_FB_INT_EDGE = 22, 23, 24, 25, 26, 27
S_FB_TOTAL, S_FB_FIRST_T, S_FB_MAX_DIST, S_FB_MISSES, S_FB_EDGE, S_FB_MAX_COLS = 28, 29, 30, 31, 32, 33
# material-top bound counters (only written when lower_bound == "material")
S_F1_STEP_COLS, S_F1_INT_COLSTEPS, S_F1_TOTAL, S_F1_INT_MAX_RAISE, S_F1_MAX_RAISE, S_F1_FIRST_T = 34, 35, 36, 37, 38, 39
STATS_SLOTS = 40
LOWER_BOUND_RULES = ("surface", "material")   # surface: top surface point; material: max(that, top material point)


class PressureWaveform:
    """p_chamber(t): clamped piecewise-linear schedule through (times_s, values_pa)."""

    def __init__(self, times_s, values_pa):
        times = np.asarray(times_s, dtype=np.float64).reshape(-1)
        values = np.asarray(values_pa, dtype=np.float64).reshape(-1)
        if times.size < 2 or times.shape != values.shape:
            raise ValueError("the pressure schedule needs >= 2 times and one value per time")
        if not (np.all(np.isfinite(times)) and np.all(np.isfinite(values))):
            raise ValueError("the pressure schedule must be finite")
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("pressure schedule times must be strictly increasing")
        self.times_s = times
        self.values_pa = values

    @classmethod
    def from_csv(cls, path):
        """A schedule from a CSV file with the header t_s,p_pa (one row per node, strictly increasing t_s)."""
        import csv
        with open(path, newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or "t_s" not in rows[0] or "p_pa" not in rows[0]:
            raise ValueError("the pressure waveform CSV needs the header t_s,p_pa: %s" % path)
        return cls([float(r["t_s"]) for r in rows], [float(r["p_pa"]) for r in rows])

    def __call__(self, t_s):
        return float(np.interp(float(t_s), self.times_s, self.values_pa))


def circular_footprint(axis_xy_m, radius_m, cell_centers_x_m, cell_centers_y_m):
    """Grid columns whose centre lies strictly inside the circle of radius radius_m around axis_xy_m (bool (nx, ny))."""
    x = np.asarray(cell_centers_x_m)[:, None] - axis_xy_m[0]
    y = np.asarray(cell_centers_y_m)[None, :] - axis_xy_m[1]
    return np.hypot(x, y) < radius_m


@ti.data_oriented
class ChamberColumns:
    """Footprint, column faces, cell-centre heights, lid and the per-column accumulators of a pressure chamber."""

    def __init__(self, footprint, *, cell_faces_x_m, cell_faces_y_m, cell_centers_z_m, lid_z_m):
        footprint = np.asarray(footprint, dtype=bool)
        if footprint.ndim != 2 or not footprint.any():
            raise ValueError("the chamber footprint must be a non-empty (nx, ny) mask")
        nx, ny = footprint.shape
        nz = len(cell_centers_z_m)
        if len(cell_faces_x_m) != nx + 1 or len(cell_faces_y_m) != ny + 1:
            raise ValueError("cell faces do not match the footprint")
        self.shape = (nx, ny, nz)
        self.column_count = int(footprint.sum())
        self.lid_z_m = float(lid_z_m)
        self.footprint = ti.field(ti.i32, shape=(nx, ny))
        self.footprint.from_numpy(footprint.astype(np.int32))
        self.face_x = ti.field(ti.f64, shape=nx + 1)
        self.face_y = ti.field(ti.f64, shape=ny + 1)
        self.centre_z = ti.field(ti.f64, shape=nz)
        self.face_x.from_numpy(np.asarray(cell_faces_x_m, dtype=np.float64))
        self.face_y.from_numpy(np.asarray(cell_faces_y_m, dtype=np.float64))
        self.centre_z.from_numpy(np.asarray(cell_centers_z_m, dtype=np.float64))
        self.column_sum = ti.field(ti.f64, shape=(nx, ny))
        self.column_count_field = ti.field(ti.i32, shape=(nx, ny))

    @ti.func
    def _locate(self, faces: ti.template(), n: ti.i32, value: ti.f64) -> ti.i32:
        index = -1
        if value >= faces[0] and value < faces[n]:
            low, high = 0, n
            while high - low > 1:
                middle = (low + high) // 2
                if faces[middle] <= value:
                    low = middle
                else:
                    high = middle
            index = low
        return index


# ---------------------------------------------------------------------------------------------------- surface setup
def triangulate_chamber_surface(positions_m, tags, faces_x, faces_y, tag_bit=TAG_CHAMBER_SURFACE):
    """Setup triangulation of the chamber surface (SURFACE FALLBACK (2) of the module docstring) on the columns of the grid
    with faces faces_x / faces_y: (vertex point indices (m,), triangles as global point indices (n, 3), boundary edges
    as global point indices (e, 2), report)."""
    from scipy.spatial import Delaunay
    X = np.asarray(positions_m, dtype=np.float64)
    air = np.flatnonzero((np.asarray(tags) & int(tag_bit)) != 0)
    report = dict(surface_points=int(air.size))
    empty = (np.zeros(0, np.int64), np.zeros((0, 3), np.int64), np.zeros((0, 2), np.int64))
    if air.size < 3:
        return empty + (dict(report, triangles=0, reason="fewer than 3 surface points"),)
    order = np.lexsort((-X[air, 2], X[air, 1], X[air, 0]))         # by x, y; the top z first within an xy
    xy = X[air[order], :2]
    first = np.ones(len(order), bool)
    first[1:] = np.any(xy[1:] != xy[:-1], axis=1)
    vertices = air[order[first]]
    report.update(distinct_xy_vertices=int(vertices.size), lower_points_at_shared_xy=int(air.size - vertices.size))
    vxy = X[vertices, :2]
    try:
        tri = Delaunay(vxy)
    except Exception as error:                                       # degenerate (collinear) point sets
        return empty + (dict(report, triangles=0, reason="Delaunay failed: %s" % str(error)[:200]),)
    simplices = np.asarray(tri.simplices, dtype=np.int64)
    p = vxy[simplices]
    a = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    b = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    c = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    circumradius = np.where(area > 0.0, a * b * c / (4.0 * np.maximum(area, 1.0e-300)), np.inf)
    fx, fy = np.asarray(faces_x, dtype=np.float64), np.asarray(faces_y, dtype=np.float64)
    centroid = p.mean(axis=1)
    i = np.clip(np.searchsorted(fx, centroid[:, 0], side="right") - 1, 0, len(fx) - 2)
    j = np.clip(np.searchsorted(fy, centroid[:, 1], side="right") - 1, 0, len(fy) - 2)
    limit = CIRCUMRADIUS_OVER_COLUMN_DIAGONAL * np.hypot(np.diff(fx)[i], np.diff(fy)[j])
    keep = circumradius <= limit
    kept = simplices[keep]
    # orient every kept triangle counter-clockwise in the reference (positive signed area)
    q = vxy[kept]
    signed = (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1]) - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])
    flip = signed < 0.0
    kept[flip] = kept[flip][:, [0, 2, 1]]
    # boundary edges of the kept triangulation (used once)
    pairs = np.concatenate([kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [2, 0]]])
    canon = np.sort(pairs, axis=1)
    unique, counts = np.unique(canon, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    report.update(delaunay_triangles=int(len(simplices)), dropped=int((~keep).sum()), triangles=int(len(kept)),
                  boundary_edges=int(len(boundary)),
                  rule="Delaunay of the reference xy (top point per xy); keep triangles with circumradius <= %g x the "
                       "diagonal of the fluid column holding the centroid" % CIRCUMRADIUS_OVER_COLUMN_DIAGONAL)
    return vertices, vertices[kept], vertices[boundary], report


def build_candidate_table(boxes_lo, boxes_hi, centres_x, centres_y, footprint, margin_m):
    """CSR table column (i * ny + j) -> item ids whose box grown by margin_m contains the column centre (footprint
    columns only); items sorted by id within a column."""
    nx, ny = len(centres_x), len(centres_y)
    lo = np.asarray(boxes_lo, dtype=np.float64) - margin_m
    hi = np.asarray(boxes_hi, dtype=np.float64) + margin_m
    i0 = np.searchsorted(centres_x, lo[:, 0], side="left")
    i1 = np.searchsorted(centres_x, hi[:, 0], side="right")
    j0 = np.searchsorted(centres_y, lo[:, 1], side="left")
    j1 = np.searchsorted(centres_y, hi[:, 1], side="right")
    ni, nj = np.maximum(i1 - i0, 0), np.maximum(j1 - j0, 0)
    count = ni * nj
    total = int(count.sum())
    item = np.repeat(np.arange(len(lo)), count)
    offset = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
    nj_rep = np.repeat(np.maximum(nj, 1), count)
    i = np.repeat(i0, count) + offset // nj_rep
    j = np.repeat(j0, count) + offset % nj_rep
    column = i * ny + j
    inside = np.asarray(footprint, dtype=bool).reshape(-1)[column] if total else np.zeros(0, bool)
    column, item = column[inside], item[inside]
    order = np.lexsort((item, column))
    column, item = column[order], item[order]
    start = np.zeros(nx * ny + 1, np.int64)
    np.add.at(start, column + 1, 1)
    return np.cumsum(start).astype(np.int32), item.astype(np.int32)


@ti.data_oriented
class ColumnPressureChamber(Checkpointable):
    """Chamber cells above the deformable surface and the shifted-penalty pressure boundary (see the module docstring)."""

    # checkpoint (state census): the statistics (per-step, interval and all-time slots), the interval's fallback
    # column flags and the host counters; every other field is setup or rewritten by _update_kernel before use
    CHECKPOINT_FIELDS = ("stats", "fallback_seen")
    CHECKPOINT_HOST = ("steps", "_fb_columns_seen", "_fb_lines", "last_p_air_pa", "last_t_s")
    PROJECT_REQUIREMENTS = PROJECT_REQUIREMENTS   # the contract travels with the class (merged strictly)

    def __init__(self, fluid, axes, footprint, lid_z_m, points, *, penalty_factor=DEFAULT_PENALTY_FACTOR,
                 tag_bit=TAG_CHAMBER_SURFACE, fallback_events_path=None, candidate_margin_m=DEFAULT_CANDIDATE_MARGIN_M,
                 edge_extension_max_m=None, lower_bound="surface", material_body_id=0, outlet_inflow="static"):
        grid = fluid.grid
        shape = tuple(int(v) for v in grid.grid_nodes)
        if axes is not None and tuple(int(v) for v in axes.n) != shape:
            raise ValueError("axes %s do not match the fluid grid %s" % (tuple(axes.n), shape))
        penalty = float(penalty_factor)
        if not math.isfinite(penalty) or penalty <= 0.0:
            raise ValueError("penalty_factor must be finite and positive")
        self.outlet_inflow = check_outlet_inflow(outlet_inflow)
        self.outlet_total_pressure = self.outlet_inflow == "total_pressure"
        margin = float(candidate_margin_m)
        if not (math.isfinite(margin) and margin >= 0.0):
            raise ValueError("candidate_margin_m must be finite and >= 0")
        if edge_extension_max_m is not None and not (math.isfinite(float(edge_extension_max_m))
                                                      and float(edge_extension_max_m) > 0.0):
            raise ValueError("edge_extension_max_m must be None (off) or finite and positive")
        footprint = np.asarray(footprint, dtype=bool)
        if footprint.shape != shape[:2]:
            raise ValueError("footprint %s does not match the grid columns %s" % (footprint.shape, shape[:2]))
        # validated footprint, face / centre coordinates, lid and the column search
        self.columns = ChamberColumns(footprint, cell_faces_x_m=np.asarray(grid.cell_faces_x_m),
                                      cell_faces_y_m=np.asarray(grid.cell_faces_y_m),
                                      cell_centers_z_m=np.asarray(grid.cell_centers_z_m), lid_z_m=float(lid_z_m))
        self.fluid = fluid
        self.axes = axes
        self.points = points
        self.shape = shape
        self.penalty_factor = penalty
        self.tag_bit = int(tag_bit)
        self.lid_z_m = float(lid_z_m)
        self.rho = float(fluid.rho)
        self.authority = int(fluid._velocity_dirichlet_boundary_authority_code())
        self.footprint_columns = int(footprint.sum())
        self.air_cell = ti.field(ti.i32, shape=shape)
        self.outlet_rhs = ti.field(ti.f64, shape=shape[:2])      # this module's copy of R at k = 0
        self.outlet_half_rho = outlet_half_rho_field(self.outlet_inflow, self.rho)     # None under "static"
        # exactly once + the outlet invariant + the projection contract of the class
        self.outlet_guard = OutletShiftGuard(fluid, requirements=type(self).PROJECT_REQUIREMENTS)
        self.stats = ti.field(ti.f64, shape=STATS_SLOTS)
        self.stats.fill(0.0)
        self._physical_pressure = None
        self.last_p_air_pa = None
        self.last_t_s = None
        self.steps = 0
        if lower_bound not in LOWER_BOUND_RULES:
            raise ValueError("lower_bound must be one of %s" % (LOWER_BOUND_RULES,))
        self.lower_bound = lower_bound
        self.f1_on = lower_bound == "material"
        self.f1_body_id = int(material_body_id)
        if self.f1_on:
            if not hasattr(points, "body_id"):
                raise ValueError("lower_bound='material' needs points.body_id")
            members = int(np.count_nonzero(np.asarray(points.body_id.to_numpy()) == self.f1_body_id))
            if members == 0:
                raise ValueError("no point of material_body_id %d" % self.f1_body_id)
            self.material_top = ti.field(ti.f64, shape=shape[:2])
            self.f1_members = members
        self._setup_fallback(footprint, np.asarray(grid.cell_centers_x_m, dtype=np.float64),
                             np.asarray(grid.cell_centers_y_m, dtype=np.float64), margin, edge_extension_max_m)
        self.fallback_events_path = None if fallback_events_path is None else str(fallback_events_path)
        self._events = (open(self.fallback_events_path, "x", encoding="utf-8")
                        if self.fallback_events_path is not None else None)
        self._fb_columns_seen = set()
        self._fb_lines = 0

    # ------------------------------------------------------------------------------------------------ surface setup
    def _setup_fallback(self, footprint, centres_x, centres_y, margin, edge_max):
        points = self.points
        reference = points.X0 if hasattr(points, "X0") else points.x
        X0 = np.asarray(reference.to_numpy(), dtype=np.float64)
        tags = np.asarray(points.tag.to_numpy())
        grid = self.fluid.grid
        _, triangles, boundary, report = triangulate_chamber_surface(X0, tags, np.asarray(grid.cell_faces_x_m),
                                                          np.asarray(grid.cell_faces_y_m), self.tag_bit)
        self.n_tri, self.n_edge = int(len(triangles)), int(len(boundary))
        self.edge_max_m = 0.0 if edge_max is None else float(edge_max)
        self.candidate_margin_m = margin
        xy = X0[:, :2]
        if self.n_tri:
            tri_lo, tri_hi = xy[triangles].min(axis=1), xy[triangles].max(axis=1)
            tri_start, tri_items = build_candidate_table(tri_lo, tri_hi, centres_x, centres_y, footprint, margin)
        else:
            tri_start, tri_items = np.zeros(footprint.size + 1, np.int32), np.zeros(0, np.int32)
        if self.n_edge and self.edge_max_m > 0.0:
            e_lo, e_hi = xy[boundary].min(axis=1), xy[boundary].max(axis=1)
            edge_start, edge_items = build_candidate_table(e_lo, e_hi, centres_x, centres_y, footprint,
                                                     margin + self.edge_max_m)
        else:
            edge_start, edge_items = np.zeros(footprint.size + 1, np.int32), np.zeros(0, np.int32)
        self.tri_p = ti.Vector.field(3, ti.i32, shape=max(1, self.n_tri))
        self.edge_p = ti.Vector.field(2, ti.i32, shape=max(1, self.n_edge))
        self.col_tri_start = ti.field(ti.i32, shape=len(tri_start))
        self.col_tri = ti.field(ti.i32, shape=max(1, len(tri_items)))
        self.col_edge_start = ti.field(ti.i32, shape=len(edge_start))
        self.col_edge = ti.field(ti.i32, shape=max(1, len(edge_items)))
        self.col_cx = ti.field(ti.f64, shape=len(centres_x))
        self.col_cy = ti.field(ti.f64, shape=len(centres_y))
        self.fallback_ok = ti.field(ti.i32, shape=self.shape[:2])
        self.fallback_seen = ti.field(ti.i32, shape=self.shape[:2])
        if self.n_tri:
            self.tri_p.from_numpy(triangles.astype(np.int32))
        if self.n_edge:
            self.edge_p.from_numpy(boundary.astype(np.int32))
        self.col_tri_start.from_numpy(tri_start)
        if len(tri_items):
            self.col_tri.from_numpy(tri_items)
        self.col_edge_start.from_numpy(edge_start)
        if len(edge_items):
            self.col_edge.from_numpy(edge_items)
        self.col_cx.from_numpy(centres_x)
        self.col_cy.from_numpy(centres_y)
        self.fallback_ok.fill(0)
        self.fallback_seen.fill(0)
        per_column = np.diff(tri_start)[np.asarray(footprint, bool).reshape(-1)]
        self.fallback_setup = dict(report, candidate_margin_m=margin, candidate_entries=int(len(tri_items)),
                                   candidates_per_footprint_column=dict(
                                       min=int(per_column.min()) if per_column.size else 0,
                                       mean=float(per_column.mean()) if per_column.size else 0.0,
                                       max=int(per_column.max()) if per_column.size else 0),
                                   edge_extension_max_m=None if edge_max is None else self.edge_max_m,
                                   edge_candidate_entries=int(len(edge_items)))

    # ------------------------------------------------------------------------------------------------ kernels
    @ti.func
    def _vertex_xy(self, p):
        x = self.points.x[p]
        return ti.Vector([ti.cast(x[0], ti.f64), ti.cast(x[1], ti.f64)]), ti.cast(x[2], ti.f64)

    @ti.func
    def _triangle_test(self, t, cx, cy, tag_bit: ti.i32):
        """(state, z, nearest-vertex distance): state 0 = does not cover (or a vertex is no longer a surface point),
        1 = covers with positive current area, 2 = covers but inverted (current signed area <= 0)."""
        state = 0
        z = ti.cast(0.0, ti.f64)
        near = ti.cast(1.0e30, ti.f64)
        ids = self.tri_p[t]
        if ((self.points.tag[ids[0]] & tag_bit) != 0 and (self.points.tag[ids[1]] & tag_bit) != 0
                and (self.points.tag[ids[2]] & tag_bit) != 0):
            a, za = self._vertex_xy(ids[0])
            b, zb = self._vertex_xy(ids[1])
            c, zc = self._vertex_xy(ids[2])
            area2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if area2 != 0.0:
                w0 = (b[0] - cx) * (c[1] - cy) - (b[1] - cy) * (c[0] - cx)        # opposite a
                w1 = (c[0] - cx) * (a[1] - cy) - (c[1] - cy) * (a[0] - cx)        # opposite b
                w2 = (a[0] - cx) * (b[1] - cy) - (a[1] - cy) * (b[0] - cx)        # opposite c
                l0, l1, l2 = w0 / area2, w1 / area2, w2 / area2
                tol = BARYCENTRIC_TOLERANCE
                if l0 >= -tol and l1 >= -tol and l2 >= -tol:
                    if area2 > 0.0:
                        state = 1
                        z = (l0 * za + l1 * zb + l2 * zc) / (l0 + l1 + l2)
                    else:
                        state = 2
                    near = ti.min(ti.min(ti.sqrt((a[0] - cx) ** 2 + (a[1] - cy) ** 2),
                                         ti.sqrt((b[0] - cx) ** 2 + (b[1] - cy) ** 2)),
                                  ti.sqrt((c[0] - cx) ** 2 + (c[1] - cy) ** 2))
        return state, z, near

    @ti.func
    def _edge_test(self, e, cx, cy, tag_bit: ti.i32):
        """(valid, distance from the centre to the edge, z at the nearest edge point, nearest-vertex distance)."""
        valid = 0
        dist = ti.cast(1.0e30, ti.f64)
        z = ti.cast(0.0, ti.f64)
        near = ti.cast(1.0e30, ti.f64)
        ids = self.edge_p[e]
        if (self.points.tag[ids[0]] & tag_bit) != 0 and (self.points.tag[ids[1]] & tag_bit) != 0:
            a, za = self._vertex_xy(ids[0])
            b, zb = self._vertex_xy(ids[1])
            d = b - a
            length2 = d[0] * d[0] + d[1] * d[1]
            s = ti.cast(0.0, ti.f64)
            if length2 > 0.0:
                s = ti.min(ti.max(((cx - a[0]) * d[0] + (cy - a[1]) * d[1]) / length2, 0.0), 1.0)
            px, py = a[0] + s * d[0], a[1] + s * d[1]
            valid = 1
            dist = ti.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            z = (1.0 - s) * za + s * zb
            near = ti.min(ti.sqrt((a[0] - cx) ** 2 + (a[1] - cy) ** 2), ti.sqrt((b[0] - cx) ** 2 + (b[1] - cy) ** 2))
        return valid, dist, z, near

    @ti.kernel
    def _update_kernel(self, t_s: ti.f64, p_air: ti.f64, penalty: ti.f64, lid_z: ti.f64, tag_bit: ti.i32,
                       authority: ti.template(), edge_max: ti.f64):
        for i, j in self.columns.column_sum:
            self.columns.column_sum[i, j] = -1.0e30
            self.columns.column_count_field[i, j] = 0
            self.fallback_ok[i, j] = 0
            if ti.static(self.f1_on):                         # the highest material point per column
                self.material_top[i, j] = -1.0e30
        for p in range(self.points.n):
            if (self.points.tag[p] & tag_bit) != 0:
                position = self.points.x[p].cast(ti.f64)
                i = self.columns._locate(self.columns.face_x, self.shape[0], position.x)
                j = self.columns._locate(self.columns.face_y, self.shape[1], position.y)
                if i >= 0 and j >= 0:
                    ti.atomic_max(self.columns.column_sum[i, j], position.z)
                    ti.atomic_add(self.columns.column_count_field[i, j], 1)
            if ti.static(self.f1_on):                         # any layer of the material body
                if self.points.body_id[p] == self.f1_body_id:
                    position_m = self.points.x[p].cast(ti.f64)
                    i_m = self.columns._locate(self.columns.face_x, self.shape[0], position_m.x)
                    j_m = self.columns._locate(self.columns.face_y, self.shape[1], position_m.y)
                    if i_m >= 0 and j_m >= 0:
                        ti.atomic_max(self.material_top[i_m, j_m], position_m.z)
        self.stats[S_STEP_AIR_CELLS] = 0.0
        self.stats[S_STEP_HOLES] = 0.0
        self.stats[S_STEPS] += 1.0
        self.stats[S_FB_STEP_COLS] = 0.0
        # columns without a surface point take the surface height (SURFACE FALLBACK)
        for i, j in self.columns.footprint:
            if self.columns.footprint[i, j] != 0 and self.columns.column_count_field[i, j] == 0:
                cx, cy = self.col_cx[i], self.col_cy[j]
                column = i * self.shape[1] + j
                found, inverted, miss = 0, 0, 0
                z = ti.cast(0.0, ti.f64)
                near = ti.cast(1.0e30, ti.f64)
                for e in range(self.col_tri_start[column], self.col_tri_start[column + 1]):
                    state, zt, nt = self._triangle_test(self.col_tri[e], cx, cy, tag_bit)
                    if state == 2:
                        inverted = 1
                    elif state == 1 and found == 0:
                        found, z, near = 1, zt, nt
                if found == 0 and inverted == 0:                # exact: search every kept triangle
                    for t in range(self.n_tri):
                        state, zt, nt = self._triangle_test(t, cx, cy, tag_bit)
                        if state == 2:
                            inverted = 1
                        elif state == 1 and found == 0:
                            found, z, near = 1, zt, nt
                    if found == 1 or inverted == 1:
                        miss = 1
                kind = 0                                        # 1 = triangle, 2 = edge extension
                reason = 0                                      # hole reason (HOLE_REASONS)
                boundary_dist = ti.cast(-1.0, ti.f64)
                if inverted == 1:
                    reason = 2
                elif found == 1:
                    kind = 1
                else:
                    best = ti.cast(1.0e30, ti.f64)
                    best_z = ti.cast(0.0, ti.f64)
                    best_near = ti.cast(1.0e30, ti.f64)
                    if edge_max > 0.0:
                        for e in range(self.col_edge_start[column], self.col_edge_start[column + 1]):
                            valid, dist, ze, ne = self._edge_test(self.col_edge[e], cx, cy, tag_bit)
                            if valid == 1 and dist < best:
                                best, best_z, best_near = dist, ze, ne
                    if best > edge_max:                         # off, or no candidate within reach: exact distance
                        for e in range(self.n_edge):
                            valid, dist, ze, ne = self._edge_test(e, cx, cy, tag_bit)
                            if valid == 1 and dist < best:
                                best, best_z, best_near = dist, ze, ne
                    boundary_dist = best
                    if edge_max > 0.0 and best <= edge_max:
                        kind, z, near = 2, best_z, best_near
                    else:
                        reason = 1
                if kind > 0:
                    self.columns.column_sum[i, j] = z
                    self.fallback_ok[i, j] = kind
                    self.fallback_seen[i, j] = ti.max(self.fallback_seen[i, j], kind)
                    ti.atomic_add(self.stats[S_FB_STEP_COLS], 1.0)
                    ti.atomic_add(self.stats[S_FB_INT_COLSTEPS], 1.0)
                    total = ti.atomic_add(self.stats[S_FB_TOTAL], 1.0)
                    if total == 0.0:
                        self.stats[S_FB_FIRST_T] = t_s
                    ti.atomic_max(self.stats[S_FB_INT_MAX_DIST], near)
                    ti.atomic_max(self.stats[S_FB_MAX_DIST], near)
                    if miss == 1:
                        ti.atomic_add(self.stats[S_FB_INT_MISSES], 1.0)
                        ti.atomic_add(self.stats[S_FB_MISSES], 1.0)
                    if kind == 2:
                        ti.atomic_add(self.stats[S_FB_INT_EDGE], 1.0)
                        ti.atomic_add(self.stats[S_FB_EDGE], 1.0)
                else:
                    ti.atomic_add(self.stats[S_STEP_HOLES], 1.0)
                    first = ti.atomic_add(self.stats[S_HOLE_EVENTS], 1.0)
                    if first == 0.0:
                        self.stats[S_HOLE_T] = t_s
                        self.stats[S_HOLE_I] = ti.cast(i, ti.f64)
                        self.stats[S_HOLE_J] = ti.cast(j, ti.f64)
                        self.stats[S_HOLE_REASON] = ti.cast(reason, ti.f64)
                        self.stats[S_HOLE_DIST] = boundary_dist
        self.stats[S_FB_INT_MAX_COLS] = ti.max(self.stats[S_FB_INT_MAX_COLS], self.stats[S_FB_STEP_COLS])
        self.stats[S_FB_MAX_COLS] = ti.max(self.stats[S_FB_MAX_COLS], self.stats[S_FB_STEP_COLS])
        if ti.static(self.f1_on):                             # raise every existing bound to the material top
            self.stats[S_F1_STEP_COLS] = 0.0
            for i, j in self.columns.footprint:
                if self.columns.footprint[i, j] != 0 and (self.columns.column_count_field[i, j] > 0
                                                          or self.fallback_ok[i, j] > 0):
                    top = self.material_top[i, j]
                    if top > self.columns.column_sum[i, j]:
                        rise = top - self.columns.column_sum[i, j]
                        self.columns.column_sum[i, j] = top
                        ti.atomic_add(self.stats[S_F1_STEP_COLS], 1.0)
                        ti.atomic_add(self.stats[S_F1_INT_COLSTEPS], 1.0)
                        total_f1 = ti.atomic_add(self.stats[S_F1_TOTAL], 1.0)
                        if total_f1 == 0.0:
                            self.stats[S_F1_FIRST_T] = t_s
                        ti.atomic_max(self.stats[S_F1_INT_MAX_RAISE], rise)
                        ti.atomic_max(self.stats[S_F1_MAX_RAISE], rise)
        for i, j, k in self.air_cell:
            air = 0
            in_column = self.columns.footprint[i, j] != 0 and (self.columns.column_count_field[i, j] > 0
                                                                or self.fallback_ok[i, j] > 0)
            if in_column and self.fluid.obstacle[i, j, k] == 0:
                z = self.columns.centre_z[k]
                if z > self.columns.column_sum[i, j] and z < lid_z:      # above the column's lower bound
                    air = 1
            self.air_cell[i, j, k] = air
            diagonal = ti.cast(0.0, ti.f64)
            rhs = ti.cast(0.0, ti.f64)
            if air == 1:
                products, denominator = self.fluid._fv_pressure_face_products_and_denominator(
                    self.fluid.obstacle, self.fluid.velocity_dirichlet_boundary_active,
                    self.fluid.velocity_dirichlet_boundary_projection_weight,
                    self.fluid.velocity_dirichlet_pressure_hard_fixed_component_mask,
                    self.fluid.velocity_dirichlet_boundary_active_component_mask,
                    self.fluid.velocity_dirichlet_boundary_pressure_mobility,
                    self.fluid.velocity_dirichlet_boundary_hard_fixed_component_mask,
                    self.fluid.cell_width_x_m, self.fluid.cell_width_y_m, self.fluid.cell_width_z_m,
                    self.fluid.center_distance_x_m, self.fluid.center_distance_y_m, self.fluid.center_distance_z_m,
                    i, j, k, self.shape[0], self.shape[1], self.shape[2], 1, authority)
                diagonal = penalty * denominator
                ti.atomic_add(self.stats[S_STEP_AIR_CELLS], 1.0)
            elif k == 0 and self.fluid.obstacle[i, j, k] == 0:
                weight = self.fluid._fv_pressure_velocity_face_weight(
                    self.fluid.velocity_dirichlet_boundary_active,
                    self.fluid.velocity_dirichlet_boundary_projection_weight,
                    self.fluid.velocity_dirichlet_pressure_hard_fixed_component_mask,
                    self.fluid.velocity_dirichlet_boundary_active_component_mask,
                    self.fluid.velocity_dirichlet_boundary_pressure_mobility,
                    self.fluid.velocity_dirichlet_boundary_hard_fixed_component_mask, i, j, k, 4, authority)
                dz = ti.cast(self.fluid.cell_width_z_m[k], ti.f64)
                rhs = weight * 2.0 / (dz * dz) * outlet_ghost_pressure(     # R = T_out p~_out (module docstring, 2)
                    p_air, self.fluid.velocity, i, j, self.outlet_half_rho, ti.static(self.outlet_total_pressure))
            self.fluid.pressure_interface_matrix_diagonal[i, j, k] = diagonal
            self.fluid.pressure_interface_matrix_rhs[i, j, k] = rhs
            if k == 0:
                self.outlet_rhs[i, j] = rhs
        if self.stats[S_STEP_AIR_CELLS] == 0.0:
            if self.stats[S_NO_AIR_EVENTS] == 0.0:
                self.stats[S_NO_AIR_T] = t_s
            self.stats[S_NO_AIR_EVENTS] += 1.0

    @ti.kernel
    def _reset_interval_kernel(self):
        for s in ti.static((S_FB_INT_COLSTEPS, S_FB_INT_MAX_COLS, S_FB_INT_MAX_DIST, S_FB_INT_MISSES, S_FB_INT_EDGE)):
            self.stats[s] = 0.0
        for i, j in self.fallback_seen:
            self.fallback_seen[i, j] = 0

    @ti.kernel
    def _reset_f1_interval_kernel(self):
        """Reset the per-interval material-top counters (called by report() when lower_bound == "material")."""
        for s in ti.static((S_F1_INT_COLSTEPS, S_F1_INT_MAX_RAISE)):
            self.stats[s] = 0.0

    @ti.kernel
    def _after_project_kernel(self, dt_over_rho_f32: ti.f32, dt_over_rho: ti.f64):
        for i, j in self.outlet_rhs:
            rhs = self.outlet_rhs[i, j]
            dz = ti.cast(self.fluid.cell_width_z_m[0], ti.f64)
            velocity = self.fluid.velocity[i, j, 0]
            velocity.z += ti.cast(ti.cast(dt_over_rho_f32, ti.f64) * rhs * dz, ti.f32)
            self.fluid.velocity[i, j, 0] = velocity
            self.fluid.pressure_interface_projection_divergence_s[i, j, 0] += dt_over_rho * rhs

    @ti.kernel
    def _physical_pressure_kernel(self, out: ti.template(), p_air: ti.f64):
        for i, j, k in out:
            value = ti.cast(0.0, ti.f64)
            if self.fluid.obstacle[i, j, k] == 0:
                value = ti.cast(self.fluid.pressure[i, j, k], ti.f64) + p_air
            out[i, j, k] = value

    @ti.kernel
    def _report_kernel(self, with_divergence: ti.i32):
        self.stats[S_AIR_CELLS] = 0.0
        self.stats[S_AIR_VOLUME] = 0.0
        self.stats[S_AIR_PMAX] = 0.0
        self.stats[S_OUTLET_FLUX] = 0.0
        self.stats[S_WATER_PMIN] = 1.0e300
        self.stats[S_WATER_PMAX] = -1.0e300
        self.stats[S_DIV_WATER] = 0.0
        self.stats[S_DIV_OUTLET] = 0.0
        self.stats[S_OUTLET_CELLS] = 0.0
        for i, j, k in self.air_cell:
            if self.fluid.obstacle[i, j, k] == 0:
                volume = (ti.cast(self.fluid.cell_width_x_m[i], ti.f64) * ti.cast(self.fluid.cell_width_y_m[j], ti.f64)
                          * ti.cast(self.fluid.cell_width_z_m[k], ti.f64))
                p_tilde = ti.cast(self.fluid.pressure[i, j, k], ti.f64)
                if self.air_cell[i, j, k] != 0:
                    ti.atomic_add(self.stats[S_AIR_CELLS], 1.0)
                    ti.atomic_add(self.stats[S_AIR_VOLUME], volume)
                    ti.atomic_max(self.stats[S_AIR_PMAX], ti.abs(p_tilde))
                else:
                    ti.atomic_min(self.stats[S_WATER_PMIN], p_tilde)
                    ti.atomic_max(self.stats[S_WATER_PMAX], p_tilde)
                    if with_divergence != 0:
                        residual = ti.abs(ti.cast(self.fluid.divergence[i, j, k], ti.f64)
                                          + self.fluid.pressure_interface_projection_divergence_s[i, j, k]
                                          - ti.cast(self.fluid.volume_source_s[i, j, k], ti.f64))
                        ti.atomic_max(self.stats[S_DIV_WATER], residual)
                        if k == 0:
                            ti.atomic_max(self.stats[S_DIV_OUTLET], residual)
                    if k == 0:
                        area = ti.cast(self.fluid.cell_width_x_m[i], ti.f64) * ti.cast(self.fluid.cell_width_y_m[j], ti.f64)
                        ti.atomic_add(self.stats[S_OUTLET_FLUX], ti.cast(self.fluid.velocity[i, j, 0].z, ti.f64) * area)
                        ti.atomic_add(self.stats[S_OUTLET_CELLS], 1.0)

    # ------------------------------------------------------------------------------------------------ host API
    def update(self, t_s, p_air_pa):
        """Recompute air_cell from points.x / points.tag and write the penalty diagonal and outlet RHS (one kernel; no host
        reads).  p_air_pa is the chamber pressure for this step's solve (p_chamber at the step midpoint)."""
        p_air = float(p_air_pa)
        if not math.isfinite(p_air):
            raise ValueError("p_air must be finite")
        self._update_kernel(float(t_s), p_air, self.penalty_factor, self.lid_z_m, self.tag_bit, self.authority,
                            self.edge_max_m)
        self.last_p_air_pa = p_air
        self.last_t_s = float(t_s)
        self.steps += 1
        self.outlet_guard.arm()

    def after_project(self, dt_s, p_air_pa):
        """Outlet-face correction for the nonzero ghost value p~_out = -p_air, and -p_air - rho w^2 / 2 on an inflow face
        under outlet_inflow="total_pressure" (one kernel; no host reads)."""
        if self.last_p_air_pa is None or float(p_air_pa) != self.last_p_air_pa:
            raise ValueError("after_project must use the p_air of the preceding update() (%r vs %r)"
                             % (p_air_pa, self.last_p_air_pa))
        dt_over_rho = float(dt_s) / self.rho
        if not (math.isfinite(dt_over_rho) and dt_over_rho > 0.0):
            raise ValueError("dt_s must be finite and positive")
        self.outlet_guard.before(self.outlet_rhs, dt_over_rho)   # consumes update()'s arming
        self._after_project_kernel(dt_over_rho, dt_over_rho)
        self.outlet_guard.after(self.outlet_rhs, dt_over_rho)

    def physical_pressure_into(self, out_field):
        """out = p~ + p_air (physical pressure, Pa) in fluid cells, 0 in obstacle cells (device only)."""
        if self.last_p_air_pa is None:
            raise RuntimeError("physical pressure requested before the first update()")
        self._physical_pressure_kernel(out_field, self.last_p_air_pa)
        return out_field

    def physical_pressure_field(self):
        """A module-owned f64 field holding the current physical pressure (refreshed on every call)."""
        if self._physical_pressure is None:
            self._physical_pressure = ti.field(ti.f64, shape=self.shape)
        return self.physical_pressure_into(self._physical_pressure)

    def _log_fallback(self, s):
        """One JSONL line per report() with fallbacks since the last report; reset the interval counters."""
        if s[S_FB_INT_COLSTEPS] <= 0.0:
            return None
        seen = self.fallback_seen.to_numpy()
        ids = np.argwhere(seen > 0)
        columns = [[int(i), int(j)] for i, j in ids]
        edge_columns = [[int(i), int(j)] for i, j in np.argwhere(seen == 2)]
        self._fb_columns_seen.update(tuple(c) for c in columns)
        line = dict(t_s=self.last_t_s, step=int(s[S_STEPS]), fallback_column_steps=int(s[S_FB_INT_COLSTEPS]),
                    max_fallback_columns_in_a_step=int(s[S_FB_INT_MAX_COLS]), columns=columns,
                    max_distance_to_nearest_vertex_m=float(s[S_FB_INT_MAX_DIST]),
                    table_misses=int(s[S_FB_INT_MISSES]), edge_extension_column_steps=int(s[S_FB_INT_EDGE]),
                    edge_extension_columns=edge_columns)
        if self._events is not None:
            self._events.write(json.dumps(line) + "\n")
            self._events.flush()
            self._fb_lines += 1
        self._reset_interval_kernel()
        return line

    def fallback_summary(self, s=None):
        """Cumulative fallback summary for run_meta.json (host bookkeeping + one stats read when s is None)."""
        s = self.stats.to_numpy() if s is None else s
        return dict(setup=self.fallback_setup,
                    total_column_steps=int(s[S_FB_TOTAL]),
                    first_t_s=float(s[S_FB_FIRST_T]) if s[S_FB_TOTAL] > 0 else None,
                    max_columns_in_a_step=int(s[S_FB_MAX_COLS]), unique_columns=len(self._fb_columns_seen),
                    max_distance_to_nearest_vertex_m=float(s[S_FB_MAX_DIST]), table_misses=int(s[S_FB_MISSES]),
                    edge_extension_column_steps=int(s[S_FB_EDGE]), events_file=self.fallback_events_path,
                    event_lines=self._fb_lines)

    def report(self, *, divergence=False):
        """Air cells, air volume, max |p~| in air, outlet flux, physical water pressure range (and, with
        divergence=True, max |div + s| in water and at the outlet after fluid.compute_divergence); one device read
        (plus the fallback column flags when a fallback happened).  Writes the fallback events line, then raises the
        hard stops counted on the device since the start."""
        if divergence:
            self.fluid.compute_divergence(pressure_outlet_zmin=True)
        self._report_kernel(1 if divergence else 0)
        s = self.stats.to_numpy()
        line = self._log_fallback(s)
        if s[S_HOLE_EVENTS] > 0.0:
            reason = int(s[S_HOLE_REASON])
            raise SimulationHardStop("pressure_chamber_column_without_surface", dict(
                events=int(s[S_HOLE_EVENTS]), first_t_s=float(s[S_HOLE_T]),
                first_column_ij=[int(s[S_HOLE_I]), int(s[S_HOLE_J])], footprint_columns=self.footprint_columns,
                reason=HOLE_REASONS.get(reason, "unknown"),
                first_column_distance_to_surface_boundary_m=float(s[S_HOLE_DIST]) if reason == 1 else None,
                edge_extension_max_m=self.edge_max_m or None, fallback_column_steps_total=int(s[S_FB_TOTAL])))
        if s[S_NO_AIR_EVENTS] > 0.0:
            raise SimulationHardStop("pressure_chamber_no_chamber_cells", dict(
                events=int(s[S_NO_AIR_EVENTS]), first_t_s=float(s[S_NO_AIR_T])))
        p_air = 0.0 if self.last_p_air_pa is None else self.last_p_air_pa
        out = dict(t_s=self.last_t_s, p_air_pa=p_air, outlet_inflow=self.outlet_inflow,
                   penalty_factor=self.penalty_factor, steps=int(s[S_STEPS]),
                   air_cell_count=int(s[S_AIR_CELLS]), air_volume_m3=float(s[S_AIR_VOLUME]),
                   p_tilde_air_max_abs_pa=float(s[S_AIR_PMAX]),
                   p_tilde_air_max_abs_over_p_air=(float(s[S_AIR_PMAX]) / abs(p_air)) if p_air else 0.0,
                   outlet_flux_m3s=float(s[S_OUTLET_FLUX]), outlet_cells=int(s[S_OUTLET_CELLS]),
                   water_pressure_min_pa=float(s[S_WATER_PMIN]) + p_air,
                   water_pressure_max_pa=float(s[S_WATER_PMAX]) + p_air,
                   fallback_column_steps_interval=0 if line is None else line["fallback_column_steps"],
                   fallback_summary=self.fallback_summary(s))
        if divergence:
            out.update(divergence_residual_water_max_1ps=float(s[S_DIV_WATER]),
                       divergence_residual_outlet_max_1ps=float(s[S_DIV_OUTLET]))
        if self.f1_on:
            out["lower_bound"] = self.lower_bound
            out["material_top_bound"] = dict(
                material_body_id=self.f1_body_id, material_points=self.f1_members,
                raised_columns_last_step=int(s[S_F1_STEP_COLS]),
                raised_column_steps_interval=int(s[S_F1_INT_COLSTEPS]),
                max_raise_interval_m=float(s[S_F1_INT_MAX_RAISE]),
                raised_column_steps_total=int(s[S_F1_TOTAL]),
                first_t_s=float(s[S_F1_FIRST_T]) if s[S_F1_TOTAL] > 0 else None,
                max_raise_m=float(s[S_F1_MAX_RAISE]))
            self._reset_f1_interval_kernel()
        out.update(self.outlet_guard.report())                    # guard maxima since the last report
        return out

    def close(self):
        if self._events is not None:
            self._events.close()
            self._events = None
