"""One sealed gas body in a box of the fluid grid whose gas amount follows another region by an isothermal ledger: the
cavity of the s26 soft outer chamber (Root's spec v2 and design note r2, 2026-10-31).

BODY.  The cavity of one deformable body: its material points (a SolidOccupancy of them on the box gives the material
cells) and an embedded triangulation of the cavity's faces, closed by FIXED faces where the cavity meets an obstacle;
plus a rigid share of constant volume V_rigid (a part of the cavity held inside an obstacle).

SURFACE.  A carried vertex follows the body: x_v = X_v + sum_k w_vk u_pk (u = x - X0 of the points), with the
EMBED_POINTS nearest body points and inverse-distance weights fixed in the reference configuration; a fixed vertex (on
an obstacle face) does not move.  Coordinates are taken relative to an origin near the cavity, in f64.

VOLUME (from the boundary).  V = V_rigid + sum_t a . (b x c) / 6 over the closed, outward-oriented triangulation.  The
free volume of the flooded cells, sum (1 - coverage) V_cell, is reported beside it (raster_volume_m3).

MEMBERSHIP (no topology).  SEEDS: for every vertex v and each depth d of seed_depths_m, the cell holding x_v - d n_v
(n_v the unit area-weighted vertex normal, pointing out of the cavity) is a seed when it is free: not topology material
of the body, not a fluid obstacle.  FLOOD: the free cells connected to a seed (6-neighbours) are the cavity cells.  Every
part of a locally closed cavity touches the surface, so every part is reached, and the whole set is ONE gas body with ONE
pressure.  A cavity cell in the box's outer cell layer means the flood reached the water outside (the wall's raster
opened): the hard stop cavity_gas_leak.  No cavity cell at all (every seed in material: the walls closed the cells, or
the hysteresis still holds them) would leave the gas body's pressure acting nowhere: the hard stop cavity_gas_empty.
The number of separate flooded parts (subregions) is counted on request (min-label propagation) and only reported.

LEDGER.  The gas amount q = p_abs V (Pa m^3, isothermal, as the sealed pockets): q(t) = q_0 - (I(t) - I_0), with I the
partner region's cumulative inflow (boundary.inlet_air: inflow_pa_m3 = sum p_atm Q dt, the air the pump moved out of
this body into the partner) and q_0 = p_atm V(0): gauge 0 at the start.  p = q / V - p_atm; the mass is
rho_atm q / p_atm and the moved mass rho_atm (I - I_0) / p_atm.  GUARD: a non-positive volume or a non-finite or
non-positive absolute pressure is the hard stop cavity_gas_invalid.  THE CLOSED LOOP'S CHECK: with the partner's total
gas Q_p (its inlet region plus its sealed pockets) recorded once by set_ledger_reference(), ledger_residual() =
(q + Q_p - reappear) - (q_0 + Q_p,0), zero to round-off while the pump only moves gas between the two (the partner's
reappear gas, its fallback, is taken out).

PRESSURE.  The cavity cells are the external region of the partner's PressureRegions (attach_external_region; this
class owns the code field and the gauge): Dirichlet cells at p.  In the shared air_cell field every accept sets the
cavity cells to 1 and clears the cells that were cavity cells at the last accept and are no longer; it touches no
other cell, because the box overlaps the partner's (the cavity's annulus surrounds it).  The accept must follow the
partner's accept, whose topology refresh clears the whole field; then every consumer of air_cell sees both as gas.

STIFFNESS.  omega^2 = p_abs A^2 / (V m_free), A the surface's area and m_free the body's free (untethered) mass:
dt_limit() = 0.9 x 0.5 / omega, the pockets' bound.

PER STEP (driver): the partner's update() reads the code field and the gauge at each solve; after the partner's
accept(t_new, ...): accept(t_new, inflow_pa_m3).  After a checkpoint load (all components), write_air_cell().

CHECKPOINT: the occupancy's state (prefix occupancy.), the code field and the gauge; host: q_0, I_0, the accept count
and the last report.

No `from __future__ import annotations` (Taichi kernel module).
"""
import math

import numpy as np
import taichi as ti
from scipy.spatial import cKDTree

from fastfsi.boundary.gas_pockets import P_ATMOSPHERE_PA
from fastfsi.boundary.gas_regions_device import STIFFNESS_LIMIT, STIFFNESS_MARGIN
from fastfsi.boundary.inlet_air import AIR_DENSITY_KG_M3
from fastfsi.boundary.solid_occupancy import (DEFAULT_BOX_SAMPLES, DEFAULT_HYSTERESIS_STEPS, DEFAULT_STRETCH_FACTOR,
                                              DEFAULT_SUB, SolidOccupancy)
from fastfsi.errors import SimulationHardStop
from fastfsi.io.checkpoint import ArrayState, HostState

HARD_STOP_CODES = {"cavity_gas_leak", "cavity_gas_invalid", "cavity_gas_empty"}
EMBED_POINTS = 4                         # body points carrying one surface vertex
SEED_DEPTHS_M = (0.6e-3, 1.2e-3, 1.8e-3)  # seed offsets into the cavity: under a 4 mm wall, so a seed never crosses one
FLOOD_SWEEPS = 8                          # flood sweeps per host read
FLOOD_MAX_READS = 4096
COMPONENT_SWEEPS = 16                     # min-label sweeps per host read (subregion count)
COMPONENT_MAX_READS = 4096
# reduce slots
R_VOLUME, R_AREA, R_SEEDS, R_CHANGED, R_CELLS, R_LEAK, R_RASTER, R_ROOTS = 0, 1, 2, 3, 4, 5, 6, 7
R_SLOTS = 8


def closed_surface_report(triangles):
    """Edges of a triangulation used by exactly one / more than two triangles, and the edges whose two triangles traverse
    them in the same direction (orientation mismatch); all three are 0 for a closed, consistently oriented surface."""
    tri = np.asarray(triangles, dtype=np.int64)
    directed = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _, counts = np.unique(undirected, axis=0, return_counts=True)
    _, directed_counts = np.unique(directed, axis=0, return_counts=True)
    return dict(open_edges=int((counts == 1).sum()), multi_edges=int((counts > 2).sum()),
                orientation_mismatches=int((directed_counts > 1).sum()))


@ti.data_oriented
class CavityLedgerGas:
    """The cavity gas body of one deformable body (see the module docstring)."""

    CHECKPOINT_FIELDS = ("code", "gauge")
    CHECKPOINT_HOST = ("q0_pa_m3", "inflow0_pa_m3", "partner_q0_pa_m3", "accepts", "last")

    def __init__(self, fluid, axes, points, *, body_indices, reference_positions_m, spacing_m, box_cells,
                 surface_vertices_m, surface_triangles, fixed_vertices, rigid_volume_m3, air_cell, pressure_regions,
                 body_density_kgm3, inflow_pa_m3=0.0, origin_m=None, ambient_pressure_pa=P_ATMOSPHERE_PA,
                 seed_depths_m=SEED_DEPTHS_M, embed_points=EMBED_POINTS, sub=DEFAULT_SUB,
                 stretch_factor=DEFAULT_STRETCH_FACTOR, box_samples=DEFAULT_BOX_SAMPLES,
                 hysteresis_steps=DEFAULT_HYSTERESIS_STEPS):
        self.fluid, self.points = fluid, points
        self.box_cells = tuple((int(lo), int(hi)) for lo, hi in box_cells)
        self.offset = tuple(lo for lo, _ in self.box_cells)
        self.shape = tuple(hi - lo for lo, hi in self.box_cells)
        self.ambient_pa = float(ambient_pressure_pa)
        self.rigid_volume_m3 = float(rigid_volume_m3)
        if not (math.isfinite(self.rigid_volume_m3) and self.rigid_volume_m3 >= 0.0):
            raise ValueError("rigid_volume_m3 must be finite and non-negative")
        if tuple(int(v) for v in air_cell.shape) != tuple(int(v) for v in fluid.obstacle.shape):
            raise ValueError("air_cell must be a field of the fluid grid")
        self.air_cell = air_cell
        self.seed_depths_m = tuple(float(d) for d in seed_depths_m)
        if not self.seed_depths_m or any(not (math.isfinite(d) and d > 0.0) for d in self.seed_depths_m):
            raise ValueError("seed depths must be finite and positive")
        body = np.asarray(body_indices, dtype=np.int64)
        reference = np.asarray(reference_positions_m, dtype=np.float64)
        self.occupancy = SolidOccupancy(axes, self.box_cells, points, body, reference, spacing_m, sub=sub,
                                        stretch_factor=stretch_factor, box_samples=box_samples,
                                        hysteresis_steps=hysteresis_steps)
        faces = [np.asarray(f, dtype=np.float64)[lo:hi + 1] for (lo, hi), f in zip(self.box_cells, axes.faces_m)]
        self._faces = [ti.field(ti.f64, shape=len(f)) for f in faces]
        for field, values in zip(self._faces, faces):
            field.from_numpy(values)
        vertices = np.asarray(surface_vertices_m, dtype=np.float64)
        triangles = np.asarray(surface_triangles, dtype=np.int64)
        fixed = np.asarray(fixed_vertices, dtype=bool)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
            raise ValueError("surface_vertices_m must be a finite (n >= 4, 3) array")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.min() < 0 or triangles.max() >= len(vertices):
            raise ValueError("surface_triangles must be (m, 3) vertex numbers")
        if fixed.shape != (len(vertices),):
            raise ValueError("fixed_vertices must have one flag per vertex")
        closure = closed_surface_report(triangles)
        if any(closure.values()):
            raise ValueError("the cavity surface must be closed and consistently oriented: %s" % closure)
        low = np.array([f[0] for f in faces])
        high = np.array([f[-1] for f in faces])
        if np.any(vertices < low) or np.any(vertices > high):
            raise ValueError("every cavity surface vertex must lie inside the box")
        self.origin = (vertices.mean(axis=0) if origin_m is None else np.asarray(origin_m, dtype=np.float64))
        relative = vertices - self.origin
        a, b, c = (relative[triangles[:, q]] for q in range(3))
        volume0 = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
        if not volume0 > 0.0:
            raise ValueError("the cavity surface must be oriented outward (its volume is %r)" % volume0)
        k = int(embed_points)
        if not 1 <= k <= len(body):
            raise ValueError("embed_points must be 1 .. the number of body points")
        idx = np.zeros((len(vertices), k), dtype=np.int32)
        weights = np.zeros((len(vertices), k), dtype=np.float64)
        count = np.zeros(len(vertices), dtype=np.int32)
        carried = np.flatnonzero(~fixed)
        if carried.size:
            distance, nearest = cKDTree(reference).query(vertices[carried], k=k)
            distance = np.asarray(distance, dtype=np.float64).reshape(len(carried), k)
            nearest = np.asarray(nearest, dtype=np.int64).reshape(len(carried), k)
            inverse = 1.0 / np.maximum(distance, 1.0e-12)
            idx[carried] = body[nearest].astype(np.int32)
            weights[carried] = inverse / inverse.sum(axis=1, keepdims=True)
            count[carried] = k
        self.embed_distance_max_m = float(distance.max()) if carried.size else 0.0
        self.n_v, self.n_t, self.k = len(vertices), len(triangles), k
        self.vertex_ref = ti.Vector.field(3, ti.f64, shape=self.n_v)
        self.vertex_ref.from_numpy(relative)
        self.vertex_points = ti.field(ti.i32, shape=(self.n_v, k))
        self.vertex_points.from_numpy(idx)
        self.vertex_weights = ti.field(ti.f64, shape=(self.n_v, k))
        self.vertex_weights.from_numpy(weights)
        self.vertex_count = ti.field(ti.i32, shape=self.n_v)
        self.vertex_count.from_numpy(count)
        self.vertex_x = ti.Vector.field(3, ti.f64, shape=self.n_v)
        self.vertex_n = ti.Vector.field(3, ti.f64, shape=self.n_v)
        self.tri = ti.Vector.field(3, ti.i32, shape=self.n_t)
        self.tri.from_numpy(triangles.astype(np.int32))
        self.depths = ti.field(ti.f64, shape=len(self.seed_depths_m))
        self.depths.from_numpy(np.asarray(self.seed_depths_m, dtype=np.float64))
        self.origin_field = ti.Vector.field(3, ti.f64, shape=())
        self.origin_field[None] = self.origin
        self.code = ti.field(ti.i32, shape=self.shape)
        self.prev = ti.field(ti.i32, shape=self.shape)        # the code of the last accept (air_cell's clearing)
        self.gauge = ti.field(ti.f64, shape=())
        self.comp = ti.field(ti.i32, shape=self.shape)
        self.reduce = ti.field(ti.f64, shape=R_SLOTS)
        tether = points.tether.to_numpy()[body]
        volumes = points.vol.to_numpy().astype(np.float64)[body]
        self.free_mass_kg = float(body_density_kgm3) * float(volumes[tether == 0].sum())
        if not (math.isfinite(self.free_mass_kg) and self.free_mass_kg > 0.0):
            raise ValueError("the body needs free points and a positive density")
        pressure_regions.attach_external_region(self.box_cells, self.code, self.gauge)
        self.pressure_regions = pressure_regions
        self.surface_report = dict(vertices=self.n_v, triangles=self.n_t, fixed_vertices=int(fixed.sum()),
                                   reference_volume_m3=volume0, embed_points=k,
                                   embed_distance_max_m=self.embed_distance_max_m, **closure)
        # the start: a cold occupancy, gauge 0 at the reference volume, the first membership
        self.accepts = 0
        self.occupancy.update()
        self.occupancy.reset_topology()
        self._surface_kernel()
        volume = self.rigid_volume_m3 + float(self.reduce[R_VOLUME])
        self.q0_pa_m3 = self.ambient_pa * volume
        self.inflow0_pa_m3 = float(inflow_pa_m3)
        self.partner_q0_pa_m3 = None
        self.last = None
        self._settle(0.0, self.inflow0_pa_m3)

    # ------------------------------------------------------------------------------------------------ device helpers
    @ti.func
    def _locate(self, faces: ti.template(), value: ti.f64) -> ti.i32:
        """Cell interval of `faces` holding value, -1 outside."""
        n = faces.shape[0] - 1
        result = -1
        if faces[0] <= value and value < faces[n]:
            low, high = 0, n
            while high - low > 1:
                mid = (low + high) // 2
                if faces[mid] <= value:
                    low = mid
                else:
                    high = mid
            result = low
        return result

    @ti.func
    def _free(self, i, j, k):
        """1 on a box cell that is neither topology material of the body nor a fluid obstacle."""
        free = 0
        if self.occupancy.topo[i, j, k] == 0:
            if self.fluid.obstacle[i + self.offset[0], j + self.offset[1], k + self.offset[2]] == 0:
                free = 1
        return free

    @ti.func
    def _inside(self, i, j, k):
        return 0 <= i < self.shape[0] and 0 <= j < self.shape[1] and 0 <= k < self.shape[2]

    # ------------------------------------------------------------------------------------------------ kernels
    @ti.kernel
    def _surface_kernel(self):
        for v in range(self.n_v):
            x = self.vertex_ref[v]
            for q in range(self.vertex_count[v]):
                x += self.vertex_weights[v, q] * ti.cast(self.points.u[self.vertex_points[v, q]], ti.f64)
            self.vertex_x[v] = x
            self.vertex_n[v] = ti.Vector([0.0, 0.0, 0.0], ti.f64)
        self.reduce[R_VOLUME] = 0.0
        self.reduce[R_AREA] = 0.0
        for t in range(self.n_t):
            a = self.vertex_x[self.tri[t][0]]
            b = self.vertex_x[self.tri[t][1]]
            c = self.vertex_x[self.tri[t][2]]
            normal = (b - a).cross(c - a)
            ti.atomic_add(self.reduce[R_VOLUME], a.dot(b.cross(c)) / 6.0)
            ti.atomic_add(self.reduce[R_AREA], 0.5 * normal.norm())
            for q in ti.static(range(3)):
                for r in ti.static(range(3)):
                    ti.atomic_add(self.vertex_n[self.tri[t][q]][r], normal[r])

    @ti.kernel
    def _seed_kernel(self):
        for i, j, k in self.code:
            self.prev[i, j, k] = self.code[i, j, k]
            self.code[i, j, k] = 0
        self.reduce[R_SEEDS] = 0.0
        for v in range(self.n_v):
            n = self.vertex_n[v]
            norm = n.norm()
            if norm > 0.0:
                n = n / norm
                for d in range(self.depths.shape[0]):
                    s = self.vertex_x[v] + self.origin_field[None] - self.depths[d] * n
                    i = self._locate(self._faces[0], s[0])
                    j = self._locate(self._faces[1], s[1])
                    k = self._locate(self._faces[2], s[2])
                    if i >= 0 and j >= 0 and k >= 0:
                        if self._free(i, j, k) != 0:
                            self.code[i, j, k] = 1
                            ti.atomic_add(self.reduce[R_SEEDS], 1.0)

    @ti.kernel
    def _flood_kernel(self):
        self.reduce[R_CHANGED] = 0.0
        for i, j, k in self.code:
            if self.code[i, j, k] == 0 and self._free(i, j, k) != 0:
                hit = 0
                for q in ti.static(range(6)):
                    o = ti.Vector([0, 0, 0])
                    o[q // 2] = 1 if q % 2 == 0 else -1
                    ii, jj, kk = i + o[0], j + o[1], k + o[2]
                    if self._inside(ii, jj, kk):
                        if self.code[ii, jj, kk] != 0:
                            hit = 1
                if hit != 0:
                    self.code[i, j, k] = 1
                    ti.atomic_add(self.reduce[R_CHANGED], 1.0)

    @ti.kernel
    def _finish_kernel(self):
        self.reduce[R_CELLS] = 0.0
        self.reduce[R_LEAK] = 0.0
        self.reduce[R_RASTER] = 0.0
        # one branch per case and the outer layer as one product test: the earlier form (a second `if c != 0` after the
        # if/elif, a six-way `or`) failed Taichi 1.7.4's IR verifier at the default opt_level (the tests' 0 passed)
        for i, j, k in self.code:
            if self.code[i, j, k] != 0:
                self.air_cell[i + self.offset[0], j + self.offset[1], k + self.offset[2]] = 1
                ti.atomic_add(self.reduce[R_CELLS], 1.0)
                volume = ((self._faces[0][i + 1] - self._faces[0][i]) * (self._faces[1][j + 1] - self._faces[1][j])
                          * (self._faces[2][k + 1] - self._faces[2][k]))
                free = 1.0 - ti.cast(self.occupancy.coverage_field[i, j, k], ti.f64)
                ti.atomic_add(self.reduce[R_RASTER], free * volume)
                low = ti.min(i, ti.min(j, k))                                  # >= 0, 0 on a lower face
                high = ti.max(i - self.shape[0] + 1, ti.max(j - self.shape[1] + 1, k - self.shape[2] + 1))   # <= 0
                if low * high == 0:                                            # the box's outer layer
                    ti.atomic_add(self.reduce[R_LEAK], 1.0)
            elif self.prev[i, j, k] != 0:
                self.air_cell[i + self.offset[0], j + self.offset[1], k + self.offset[2]] = 0

    @ti.kernel
    def _air_kernel(self):
        for i, j, k in self.code:
            if self.code[i, j, k] != 0:
                self.air_cell[i + self.offset[0], j + self.offset[1], k + self.offset[2]] = 1

    @ti.kernel
    def _clear_air_kernel(self):
        for i, j, k in self.code:
            if self.code[i, j, k] != 0:
                self.air_cell[i + self.offset[0], j + self.offset[1], k + self.offset[2]] = 0

    @ti.kernel
    def _component_init_kernel(self):
        for i, j, k in self.code:
            label = -1
            if self.code[i, j, k] != 0:
                label = (i * self.shape[1] + j) * self.shape[2] + k
            self.comp[i, j, k] = label

    @ti.kernel
    def _component_sweep_kernel(self):
        self.reduce[R_CHANGED] = 0.0
        for i, j, k in self.comp:
            label = self.comp[i, j, k]
            if label >= 0:
                low = label
                for q in ti.static(range(6)):
                    o = ti.Vector([0, 0, 0])
                    o[q // 2] = 1 if q % 2 == 0 else -1
                    ii, jj, kk = i + o[0], j + o[1], k + o[2]
                    if self._inside(ii, jj, kk):
                        other = self.comp[ii, jj, kk]
                        if 0 <= other < low:
                            low = other
                if low < label:
                    self.comp[i, j, k] = low
                    ti.atomic_add(self.reduce[R_CHANGED], 1.0)

    @ti.kernel
    def _component_count_kernel(self):
        self.reduce[R_ROOTS] = 0.0
        for i, j, k in self.comp:
            if self.comp[i, j, k] == (i * self.shape[1] + j) * self.shape[2] + k:
                ti.atomic_add(self.reduce[R_ROOTS], 1.0)

    # ------------------------------------------------------------------------------------------------ host API
    def _settle(self, t_s, inflow_pa_m3):
        """Volume, ledger and pressure at the current geometry, then the membership (seeds, flood, air cells)."""
        volume = self.rigid_volume_m3 + float(self.reduce[R_VOLUME])
        area = float(self.reduce[R_AREA])
        inflow = float(inflow_pa_m3)
        q = self.q0_pa_m3 - (inflow - self.inflow0_pa_m3)
        p_abs = q / volume if volume > 0.0 else float("nan")
        if not (math.isfinite(volume) and volume > 0.0 and math.isfinite(p_abs) and p_abs > 0.0):
            raise SimulationHardStop("cavity_gas_invalid", dict(time_s=float(t_s), volume_m3=volume, q_pa_m3=q,
                                                                inflow_pa_m3=inflow))
        self.gauge[None] = p_abs - self.ambient_pa
        self._seed_kernel()
        reads = 0
        while True:
            for _ in range(FLOOD_SWEEPS):
                self._flood_kernel()
            reads += 1
            if float(self.reduce[R_CHANGED]) == 0.0:
                break
            if reads >= FLOOD_MAX_READS:
                raise SimulationHardStop("cavity_gas_leak", dict(time_s=float(t_s), reason="flood did not settle",
                                                                 reads=reads))
        self._finish_kernel()
        r = self.reduce.to_numpy()
        if r[R_LEAK] > 0.0:
            raise SimulationHardStop("cavity_gas_leak", dict(time_s=float(t_s), boundary_cells=int(r[R_LEAK]),
                                                             cavity_cells=int(r[R_CELLS]), seeds=int(r[R_SEEDS])))
        if r[R_CELLS] == 0.0:
            raise SimulationHardStop("cavity_gas_empty", dict(time_s=float(t_s), seeds=int(r[R_SEEDS]),
                                                              volume_m3=volume, p_gauge_pa=p_abs - self.ambient_pa))
        omega = math.sqrt(p_abs * area * area / (volume * self.free_mass_kg))
        self.last = dict(time_s=float(t_s), q_pa_m3=q, p_gauge_pa=p_abs - self.ambient_pa, volume_m3=volume,
                         free_volume_m3=volume - self.rigid_volume_m3, raster_volume_m3=float(r[R_RASTER]),
                         area_m2=area, mass_kg=AIR_DENSITY_KG_M3 * q / self.ambient_pa,
                         moved_mass_kg=AIR_DENSITY_KG_M3 * (inflow - self.inflow0_pa_m3) / self.ambient_pa,
                         inflow_pa_m3=inflow, cavity_cells=int(r[R_CELLS]), seeds=int(r[R_SEEDS]),
                         flood_reads=reads, omega_rad_s=omega,
                         dt_limit_s=STIFFNESS_MARGIN * STIFFNESS_LIMIT / omega if omega > 0.0 else math.inf)

    def accept(self, t_s, inflow_pa_m3):
        """The accepted geometry: occupancy, surface, volume, ledger, pressure and membership (call after the partner's
        accept).  Returns the report of this accept (without the subregion count)."""
        occupancy = self.occupancy.update()
        self._surface_kernel()
        self._settle(t_s, inflow_pa_m3)
        self.accepts += 1
        self.last["occupancy"] = {key: occupancy[key] for key in ("material_cells", "topo_cells", "flipped",
                                                                  "lattice_cells_dropped", "lattice_cells_inverted")}
        return dict(self.last)

    def subregions(self):
        """The number of separate parts of the last membership (min-label propagation; -1 when it did not settle)."""
        self._component_init_kernel()
        for _ in range(COMPONENT_MAX_READS):
            for _ in range(COMPONENT_SWEEPS):
                self._component_sweep_kernel()
            if float(self.reduce[R_CHANGED]) == 0.0:
                self._component_count_kernel()
                return int(self.reduce[R_ROOTS])
        return -1

    def report(self, count_subregions=True):
        """The last accept's values (with the subregion count when asked)."""
        out = dict(self.last)
        if count_subregions:
            out["subregions"] = self.subregions()
        return out

    def dt_limit(self):
        """The gas spring's bound 0.9 x 0.5 / omega of the last accept."""
        return float(self.last["dt_limit_s"])

    def set_ledger_reference(self, partner_q_total_pa_m3):
        """Record the partner's total gas Q_p,0 at the start (once; a restart loads it with the host state)."""
        value = float(partner_q_total_pa_m3)
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError("the partner's gas must be finite and positive")
        self.partner_q0_pa_m3 = value

    def ledger_residual(self, partner_q_total_pa_m3, partner_reappear_pa_m3=0.0):
        """(q + Q_p - reappear) - (q_0 + Q_p,0) at the last accept (module docstring); None before the reference."""
        if self.partner_q0_pa_m3 is None:
            return None
        return ((self.last["q_pa_m3"] + float(partner_q_total_pa_m3) - float(partner_reappear_pa_m3))
                - (self.q0_pa_m3 + self.partner_q0_pa_m3))

    def write_air_cell(self):
        """Set the cavity cells of the shared air_cell field from the code field (after a checkpoint load)."""
        self._air_kernel()

    # ------------------------------------------------------------------------------------------------ checkpoint
    def state_array_names(self):
        return tuple(self.CHECKPOINT_FIELDS) + tuple("occupancy." + name
                                                     for name in self.occupancy.state_array_names())

    def state_array(self, name):
        if name in self.CHECKPOINT_FIELDS:
            return getattr(self, name).to_numpy()
        if name.startswith("occupancy."):
            return self.occupancy.state_array(name[len("occupancy."):])
        raise KeyError(name)

    def load_state_arrays(self, arrays):
        self._clear_air_kernel()             # the construction's cavity cells leave air_cell before the loaded ones enter
        ArrayState({name: getattr(self, name) for name in self.CHECKPOINT_FIELDS}).load_state_arrays(
            {name: arrays[name] for name in self.CHECKPOINT_FIELDS if name in arrays})
        self.occupancy.load_state_arrays({name[len("occupancy."):]: value for name, value in arrays.items()
                                          if name.startswith("occupancy.")})
        self._surface_kernel()
        self.write_air_cell()

    def state_host(self):
        host = {"occupancy." + name: value for name, value in self.occupancy.state_host().items()}
        host.update(HostState(self, self.CHECKPOINT_HOST).state_host())
        return host

    def load_state_host(self, values):
        self.occupancy.load_state_host({name[len("occupancy."):]: value for name, value in values.items()
                                        if name.startswith("occupancy.")})
        HostState(self, self.CHECKPOINT_HOST).load_state_host({name: values[name] for name in self.CHECKPOINT_HOST})
