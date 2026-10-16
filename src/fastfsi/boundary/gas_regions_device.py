"""Air regions of a chamber box on the device: inlet-connected air, water, sealed air pockets with their gas amounts,
fractional volumes and pressures, written straight into a per-cell Dirichlet chamber.  Device-first placement:
the fields stay on the device; only counts, tables and the compact list of sealed candidate cells move.

CODES (gas_pocket_transitions):  0 material or wall, 1 INLET_GAS (inlet-connected), 2 LIQUID (floor-connected), 3 water pocket,
4 + k air pocket k.

PER REFRESH (an accepted geometry; refresh(t, p_in)):
  1. SolidOccupancy.update(): rasterisation, coverage, topology with hysteresis, flip count     [read 1: statistics]
  2. only when the topology changed (flips > 0, or the first refresh):
     a. FLOODS on the free cells (not wall, not topology material): INLET_GAS from the free inlet cells, LIQUID from the free
        cells of the box floor; one neighbour level per launch of a single-level kernel, iterated to convergence, the
        frontier count read once per batch of `flood_batch` launches;
     b. CLASSIFY: INLET_GAS, LIQUID; a cell reached by both is the hard stop gas_region_leak; a cell going directly between the
        air side (INLET_GAS, air pocket) and the water side (LIQUID, water pocket) is gas_region_gas_to_liquid /
        gas_region_liquid_to_gas (hysteresis makes a direct change impossible without a leak); the rest are SEALED
        CANDIDATES, appended to a compact device list with their previous codes; the old pockets' fate counts
        (cells now material / INLET_GAS / LIQUID / candidate)                                          [read: counters]
     c. HOST STAGE, only if candidates or old pockets exist and the sealed candidates are not exactly the old pocket
        cells (when they are -- no candidate lay outside a pocket before and the counts agree -- the cell set and so
        the components are unchanged: codes and gas stay on the device): the candidate list and the old pockets' fate
        and gas come down, gas_pocket_transitions.plan_pocket_transitions() gives the components, codes and gas plan, the
        candidates' new codes go back up, the new pockets' volumes are reduced on the device and read,
        gas_pocket_transitions.allocate_pocket_gas() gives the new gas amounts, which go back up.
  3. VOLUMES and PRESSURES on the device: per pocket the effective volume (free cells (1 - c) V plus the equal shares
     of the face-adjacent material cells' free volume among their distinct adjacent regions), the pocket's faces
     against material (area A), the body mass and kinetic energy in its footprint columns (the column sums of
     SolidOccupancy, taken at the top cell of each vertical run of the pocket); p_abs = q / V; the stiffness
     omega^2 = p_abs A^2 / (V m) and the time-step bound 0.9 x 0.5 / max omega; the chamber's pocket pressures and,
     after a topology change, its codes and the full-grid air_cell field (Dirichlet cells) are written in place;
     the noise statistics (per volume bin: max |dp| between refreshes without a topology change, sum dp^2, count;
     and the noise events: pockets >= 5 mm^3 with |dp| > 1 kPa, their number and the largest one)
                                                                                                 [read 2: statistics]
  With no flipped cell, step 2 and every transfer but the two small reads are skipped.
SEALING RULE (gas_pocket_transitions): a candidate region cut off from the inlet air seals only after
`seal_steps` occupancy updates with at least `min_cells` cells (N_SEAL = 5, M_MIN = 8); until then its cells stay
INLET_GAS (Dirichlet at p_in) and carry a pending-region number (field cand_id).  The host keeps one record per pending
region (census id, birth time, age, cells, last free volume); every occupancy update ages each region by one, and an
update without flips runs the topology stage anyway when a region has come of age (>= seal_steps updates and
>= min_cells cells).  The classification counts each old region's cells by fate (material / INLET_GAS / LIQUID /
candidate) and the volumes kernel sums each region's free volume ((1 - c) V of its cells).  Counters (created,
sealed, ended by covered / reconnected / merged, the largest region and age, the free volume of the regions that
ended unsealed, the largest such volume as a fraction of the inlet air) and the events (the first
CANDIDATE_EVENT_LOG_MAX in full) are host state, checkpointed with cand_id.  seal_steps = min_cells = 1 is the
earlier rule (every candidate sealed at once).
TYPE MEMORY (type_memory, default on): every commit records, per box cell, the class of its last free code (field
last_free: INLET_GAS, LIQUID or a water pocket; 0 when the cell was never free since the start, or last belonged to a
sealed air pocket, whose number does not survive a renumbering).  A candidate cell that was material before the
classification reports that class as its previous code, so gas_pocket_transitions types a sealed region of freed cells
by its cells' history: a void freed on the gas side is a candidate region of the inlet gas (the sealing rule; joining
the inlet gas again ends it as "reconnected"), a void with a liquid history is a water pocket, mixed histories are
the hard stop gas_pocket_origin_ambiguous; cells with no history keep the old rule (a water pocket).  Without it, a
one-cell void that a flickering rasterised surface enclosed for one update on the gas side became a water pocket and
the next update's gas contact stopped the run (gas_region_liquid_to_gas) although gas and liquid never met.  last_free
is checkpointed.
RASTER INTERVAL (raster_interval N >= 1, default 1): with N > 1 only every Nth refresh re-rasterises the body
(SolidOccupancy.update); the others reuse the last occupancy (no flips, no topology stage, no ageing) while volumes,
pressures and the step check run at every refresh.  A restart keeps the cadence: the refresh count is carried, and
SolidOccupancy's checkpoint carries what a skipped refresh reuses (coverage, column sums, the last statistics).
TRANSFERS per refresh (counted: diagnostics "transfers"): read 1 (SolidOccupancy statistics, 9 f64), read 2
(STATS_SLOTS f64); on a topology change in addition the counters (C_SLOTS i32; they include the material cells on the
box floor) after every `flood_batch` flood levels and after the classification, and with a host stage the candidate list
(2 x i32 per candidate cell), the candidates' new codes (the same), the old pockets' fate and gas, the new pockets'
volumes and their gas (TABLE_ROWS f64 per pocket each).  Lists and tables move through the smallest field of a size
ladder that holds them (STAGING_LADDER then `staging` cells; TABLE_LADDER then `max_pockets` pockets), so the bytes scale
with the counts.

GAS RULES (gas_pocket_transitions): a pocket sealing off from the inlet air starts at the inlet pressure
of the seal time, q = (p_in(t) + p_atm) V x (its share of former INLET_GAS cells); old pockets keep, split (by the volume
each child received) and merge (sum) their gas; a share joining the inlet air returns to the reservoir (exchange
counter); strict hard stops otherwise.  p_atm is the standard atmosphere, 101325 Pa (a physical constant; recorded).
Guards (hard stops): gas_pocket_zero_volume (a pocket without volume), gas_pocket_invalid_pressure (non-finite or not
positive), gas_pocket_pressure_out_of_bounds (optional absolute bounds).

The host-labelling reference path is boundary/gas_regions.GasRegions with boundary/gas_pockets.GasPocketModel
(seal_from_inlet_pressure=True).

No `from __future__ import annotations` (Taichi kernel module).
"""
import math
import time

import numpy as np
import taichi as ti

from fastfsi.boundary.gas_pocket_transitions import (M_MIN, N_SEAL, allocate_pocket_gas, continuing_pockets,
                                                     plan_pocket_transitions)
from fastfsi.errors import SimulationHardStop
from fastfsi.io.checkpoint import Checkpointable

from fastfsi.boundary.gas_pockets import P_ATMOSPHERE_PA  # noqa: E402  (the ambient of the gas law, one definition)
DEFAULT_MAX_POCKETS = 4096
DEFAULT_FLOOD_BATCH = 16
DEFAULT_STAGING = 65536
DEFAULT_VOLUME_BINS_M3 = (1.0e-9, 5.0e-9, 1.0e-8, 1.0e-7)       # 1, 5, 10, 100 mm^3: five bins
STIFFNESS_LIMIT = 0.5                    # omega dt above this is a hard stop for a step taken
STIFFNESS_MARGIN = 0.9                   # the time-step bound keeps omega dt <= 0.9 x 0.5
HARD_STOP_CODES = {"gas_region_leak", "gas_region_gas_to_liquid", "gas_region_liquid_to_gas", "gas_pocket_zero_volume",
                   "gas_pocket_invalid_pressure", "gas_pocket_pressure_out_of_bounds", "gas_pocket_stiffness",
                   "gas_pocket_origin_ambiguous", "gas_pocket_ambiguous_topology"}
NBINS = len(DEFAULT_VOLUME_BINS_M3) + 1
# integer counters (one i32 field)
C_FRONTIER, C_CANDIDATES, C_LEAK, C_AIR_TO_WATER, C_WATER_TO_AIR, C_INLET_AIR, C_FIRST_BAD = 0, 1, 2, 3, 4, 5, 6
C_FLOOR_MATERIAL, C_NEW_CANDIDATES, C_OLD_POCKET_CELLS = 7, 8, 9
C_SLOTS = 10
STAGING_LADDER = (256, 4096)             # staging sizes (cells) below `staging`
TABLE_LADDER = (64, 512)                 # table sizes (pockets) below `max_pockets`
TABLE_ROWS = 5                           # old pockets: fate (4 counts) and gas q; new pockets: volume, gas
# statistics (one f64 field, read once per refresh)
S_OMEGA_MAX, S_V_MIN, S_P_MIN, S_P_MAX, S_ZERO_VOLUME, S_BAD_PRESSURE, S_OUT_OF_BOUNDS = 0, 1, 2, 3, 4, 5, 6
S_KE, S_AIR_VOLUME, S_AIR_CELLS, S_POCKET_VOLUME, S_WORST_POCKET = 7, 8, 9, 10, 11
# the noise trigger: pockets of at least NOISE_EVENT_VOLUME_M3 whose pressure moved by more
# than NOISE_EVENT_DP_PA between refreshes without a topology change -- their number and the largest one
S_EVENTS, S_EVENT_DP, S_EVENT_POCKET, S_EVENT_VOLUME = 12, 13, 14, 15
S_INLET_AIR = 16                         # INLET_GAS cells on the inlet: 0 = the hole is covered (counted at every refresh)
S_NOISE = 17                             # NBINS x (max |dp|, sum dp^2, count)
# the stiffness check of a step covers the pockets that CONTINUED through the refresh (same cell set; new, split and
# merged pockets are exempt once: their gas spring has not acted yet): their max omega and its pocket
S_OMEGA_CHECKED = S_NOISE + 3 * NBINS
S_WORST_CHECKED = S_OMEGA_CHECKED + 1
STATS_SLOTS = S_WORST_CHECKED + 1
NOISE_EVENT_VOLUME_M3 = 5.0e-9           # 5 mm^3
NOISE_EVENT_DP_PA = 1.0e3                # 1 kPa
CANDIDATE_EVENT_LOG_MAX = 200            # the first candidate-region events kept in full (counters complete)
STAGE_ROWS = 3                           # staging rows: linear index, code, pending-region number


def _ladder(sizes, top):
    """The staging sizes: the ladder entries below `top`, then `top`."""
    return tuple(sorted({int(s) for s in sizes if s < top} | {int(top)}))


def _fit(fields, n):
    """The smallest size of a ladder (a dict size -> field, ascending) that holds n entries."""
    for size in fields:
        if n <= size:
            return size
    raise ValueError("%d entries exceed the largest staging size" % n)


def _no_transfers():
    return dict(reads=0, read_bytes=0, writes=0, write_bytes=0)


@ti.data_oriented
class DeviceGasRegions(Checkpointable):
    """Device classification and gas of a chamber box (see the module docstring)."""

    CHECKPOINT_FIELDS = ("code", "q", "p_prev", "cand_id", "last_free")
    CHECKPOINT_HOST = ("pockets", "initialized", "refreshes", "topology_changes", "host_stages", "flood_levels",
                       "exchange_pa_m3", "seal_events", "last_omega_rad_s", "last_t_s", "pocket_set_unchanged",
                       "pending_regions", "candidate_counters", "candidate_events", "raster_skips",
                       "last_air_volume_m3")
    # host_stage_s is wall-clock time per segment: never carried state

    def __init__(self, occupancy, chamber, air_cell, wall, inlet_cells, *, ambient_pressure_pa=P_ATMOSPHERE_PA,
                 pressure_bounds_abs_pa=None, max_pockets=DEFAULT_MAX_POCKETS, flood_batch=DEFAULT_FLOOD_BATCH,
                 staging=DEFAULT_STAGING, volume_bins_m3=DEFAULT_VOLUME_BINS_M3, seal_steps=N_SEAL, min_cells=M_MIN,
                 raster_interval=1, type_memory=True):
        shape = tuple(occupancy.shape)
        wall = np.asarray(wall, dtype=bool)
        inlet = np.asarray(inlet_cells, dtype=bool)
        if wall.shape != shape or inlet.shape != shape:
            raise ValueError("wall and inlet masks must have the occupancy box shape %s" % (shape,))
        if not inlet.any() or (inlet & wall).any() or inlet[:, :, :-1].any():
            raise ValueError("inlet cells must be non-wall cells of the top layer, at least one")
        if tuple(chamber.box_shape) != shape or tuple(chamber.offset) != tuple(lo for lo, _ in occupancy.box_cells):
            raise ValueError("the chamber box must be the occupancy box")
        if tuple(int(v) for v in air_cell.shape) != tuple(chamber.shape):
            raise ValueError("air_cell must be a field of the fluid grid")
        if len(volume_bins_m3) != NBINS - 1:
            raise ValueError("volume_bins_m3 must have %d edges" % (NBINS - 1))
        ambient = float(ambient_pressure_pa)
        if not (math.isfinite(ambient) and ambient > 0.0):
            raise ValueError("ambient_pressure_pa must be finite and positive")
        if pressure_bounds_abs_pa is not None:
            low, high = (float(v) for v in pressure_bounds_abs_pa)
            if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
                raise ValueError("absolute pressure bounds must be finite with 0 < low < high")
            pressure_bounds_abs_pa = (low, high)
        self.occupancy = occupancy
        self.chamber = chamber
        self.air_cell = air_cell
        self.shape = shape
        self.ambient_pressure_pa = ambient
        self.pressure_bounds_abs_pa = pressure_bounds_abs_pa
        self.max_pockets = int(max_pockets)
        self.flood_batch = int(flood_batch)
        self.staging = int(staging)
        self.volume_bins_m3 = tuple(float(v) for v in volume_bins_m3)
        self.seal_steps, self.min_cells = int(seal_steps), int(min_cells)
        self.raster_interval = int(raster_interval)
        self.type_memory = bool(type_memory)
        if self.seal_steps < 1 or self.min_cells < 1 or self.raster_interval < 1:
            raise ValueError("seal_steps, min_cells and raster_interval must be >= 1")
        cells = int(np.prod(shape))
        self.wall = ti.field(ti.i32, shape=shape)
        self.wall.from_numpy(wall.astype(np.int32))
        self.inlet = ti.field(ti.i32, shape=shape)
        self.inlet.from_numpy(inlet.astype(np.int32))
        self.vol = ti.field(ti.f64, shape=shape)
        self.vol.from_numpy(np.asarray(occupancy.cell_volume_m3, dtype=np.float64))
        self.code = ti.field(ti.i32, shape=shape)
        self.last_free = ti.field(ti.i32, shape=shape)     # type memory: the class of the last free code (0 none)
        self.new_code = ti.field(ti.i32, shape=shape)
        self.reach = ti.field(ti.i32, shape=shape)
        self.cand = ti.field(ti.i32, shape=cells)
        self.cand_prev = ti.field(ti.i32, shape=cells)
        self.cand_preg = ti.field(ti.i32, shape=cells)            # the candidates' old region numbers
        self.cand_id = ti.field(ti.i32, shape=shape)              # pending-region number per cell (0 none)
        self.new_cand = ti.field(ti.i32, shape=shape)
        self.stage = {size: ti.field(ti.i32, shape=(STAGE_ROWS, size))
                      for size in _ladder(STAGING_LADDER, self.staging)}
        self.table = {size: ti.field(ti.f64, shape=(TABLE_ROWS, size))
                      for size in _ladder(TABLE_LADDER, max(self.max_pockets, 2))}
        self.fate = ti.field(ti.i32, shape=(self.max_pockets, 4))
        self.rfate = ti.field(ti.i32, shape=(self.max_pockets, 4))   # old regions' cells by fate
        self.rV = ti.field(ti.f64, shape=self.max_pockets)           # pending regions' free volumes
        self.q = ti.field(ti.f64, shape=self.max_pockets)
        self.p_prev = ti.field(ti.f64, shape=self.max_pockets)
        self.dp = ti.field(ti.f64, shape=self.max_pockets)     # scratch of _pressure_kernel (entries < pockets)
        self.V = ti.field(ti.f64, shape=self.max_pockets)
        self.A = ti.field(ti.f64, shape=self.max_pockets)
        self.M = ti.field(ti.f64, shape=self.max_pockets)
        self.KE = ti.field(ti.f64, shape=self.max_pockets)
        # 1: the pocket continued through the last refresh (checked); written by the host stage, set to 1 after every
        # refresh, so it is 1 at every step boundary (not checkpointed)
        self.checked = ti.field(ti.i32, shape=self.max_pockets)
        self.checked.fill(1)
        self.bins = ti.field(ti.f64, shape=NBINS - 1)
        self.bins.from_numpy(np.asarray(self.volume_bins_m3, dtype=np.float64))
        self.ctr = ti.field(ti.i32, shape=C_SLOTS)
        self.badbox = ti.field(ti.i32, shape=6)
        self.stats = ti.field(ti.f64, shape=STATS_SLOTS)
        self.pockets = 0
        self.initialized = False
        self.refreshes = 0
        self.topology_changes = 0
        self.host_stages = 0
        self.flood_levels = 0
        self.exchange_pa_m3 = 0.0
        self.seal_events = 0
        self.pocket_set_unchanged = 0
        self.pending_regions = []        # one dict per pending region (number = position + 1)
        self.candidate_counters = dict(created=0, sealed=0, covered=0, reconnected=0, merged=0, cells_max=0,
                                       age_max=0, ended_volume_m3=0.0, ended_fraction_max=0.0, census_next=0)
        self.candidate_events = []       # the first CANDIDATE_EVENT_LOG_MAX events
        self.candidate_events_new = []   # the events of the last refresh (for the caller; not carried)
        self.raster_skips = 0
        self.last_air_volume_m3 = None
        self.host_stage_s = 0.0
        self.last_omega_rad_s = 0.0
        self.last_omega_checked_rad_s = 0.0
        self.last_t_s = None
        self.last = None
        self.transfers = _no_transfers()

    # ------------------------------------------------------------------------------------------------ device helpers
    @ti.func
    def _free(self, i, j, k):
        return self.wall[i, j, k] == 0 and self.occupancy.topo[i, j, k] == 0

    @ti.func
    def _face_area(self, i, j, k, axis: ti.template()):
        area = ti.cast(0.0, ti.f64)
        if ti.static(axis == 0):
            area = (self.occupancy._faces[1][j + 1] - self.occupancy._faces[1][j]) * (self.occupancy._faces[2][k + 1] - self.occupancy._faces[2][k])
        elif ti.static(axis == 1):
            area = (self.occupancy._faces[0][i + 1] - self.occupancy._faces[0][i]) * (self.occupancy._faces[2][k + 1] - self.occupancy._faces[2][k])
        else:
            area = (self.occupancy._faces[0][i + 1] - self.occupancy._faces[0][i]) * (self.occupancy._faces[1][j + 1] - self.occupancy._faces[1][j])
        return area

    # ------------------------------------------------------------------------------------------------ kernels
    @ti.kernel
    def _flood_init_kernel(self):
        for s in range(C_SLOTS):
            self.ctr[s] = 0
        self.ctr[C_FIRST_BAD] = 2147483647
        for i, j, k in self.reach:
            r = 0
            if self._free(i, j, k):
                if self.inlet[i, j, k] != 0:
                    r |= 1
                if k == 0:
                    r |= 2
            self.reach[i, j, k] = r

    @ti.kernel
    def _flood_level_kernel(self):
        for i, j, k in self.reach:
            if self._free(i, j, k):
                r = self.reach[i, j, k]
                around = 0
                if i > 0:
                    around |= self.reach[i - 1, j, k]
                if i < self.shape[0] - 1:
                    around |= self.reach[i + 1, j, k]
                if j > 0:
                    around |= self.reach[i, j - 1, k]
                if j < self.shape[1] - 1:
                    around |= self.reach[i, j + 1, k]
                if k > 0:
                    around |= self.reach[i, j, k - 1]
                if k < self.shape[2] - 1:
                    around |= self.reach[i, j, k + 1]
                gained = around & ~r & 3
                if gained != 0:
                    ti.atomic_or(self.reach[i, j, k], gained)
                    ti.atomic_add(self.ctr[C_FRONTIER], 1)

    @ti.kernel
    def _classify_kernel(self, pockets: ti.i32, regions: ti.i32, memory: ti.i32):
        for k_ in range(pockets):
            for s in ti.static(range(4)):
                self.fate[k_, s] = 0
        for r_ in range(regions):
            for s in ti.static(range(4)):
                self.rfate[r_, s] = 0
        for i, j, k in self.reach:
            old = self.code[i, j, k]
            old_region = self.cand_id[i, j, k]
            new = 0
            linear = (i * self.shape[1] + j) * self.shape[2] + k
            if self._free(i, j, k):
                r = self.reach[i, j, k]
                if r == 3:
                    ti.atomic_add(self.ctr[C_LEAK], 1)
                    ti.atomic_min(self.ctr[C_FIRST_BAD], linear)
                    new = 1
                elif r == 1:
                    new = 1
                elif r == 2:
                    new = 2
                else:
                    new = -1
                    slot = ti.atomic_add(self.ctr[C_CANDIDATES], 1)
                    self.cand[slot] = linear
                    parent = old
                    if memory != 0 and old == 0:          # type memory: a freed cell reports its last free class
                        parent = self.last_free[i, j, k]
                    self.cand_prev[slot] = parent
                    self.cand_preg[slot] = old_region
                    if old < 3:
                        ti.atomic_add(self.ctr[C_NEW_CANDIDATES], 1)
                if (old == 1 or old >= 4) and new == 2:
                    ti.atomic_add(self.ctr[C_AIR_TO_WATER], 1)
                    ti.atomic_min(self.ctr[C_FIRST_BAD], linear)
                if (old == 2 or old == 3) and new == 1:
                    ti.atomic_add(self.ctr[C_WATER_TO_AIR], 1)
                    ti.atomic_min(self.ctr[C_FIRST_BAD], linear)
                if new == 1 and self.inlet[i, j, k] != 0:
                    ti.atomic_add(self.ctr[C_INLET_AIR], 1)
            if old >= 4:
                fate = 0
                if new == 1:
                    fate = 1
                elif new == 2:
                    fate = 2
                elif new == -1:
                    fate = 3
                ti.atomic_add(self.fate[old - 4, fate], 1)
            if old_region > 0:                  # an old pending region's cells by fate
                region_fate = 0
                if new == 1:
                    region_fate = 1
                elif new == 2:
                    region_fate = 2
                elif new == -1:
                    region_fate = 3
                ti.atomic_add(self.rfate[old_region - 1, region_fate], 1)
            if old >= 3:
                ti.atomic_add(self.ctr[C_OLD_POCKET_CELLS], 1)
            if k == 0 and self.wall[i, j, k] == 0 and self.occupancy.topo[i, j, k] != 0:
                ti.atomic_add(self.ctr[C_FLOOR_MATERIAL], 1)
            self.new_code[i, j, k] = new
            self.new_cand[i, j, k] = 0

    @ti.kernel
    def _stage_out_kernel(self, stage: ti.template(), offset: ti.i32, n: ti.i32):
        for s in range(n):
            stage[0, s] = self.cand[offset + s]
            stage[1, s] = self.cand_prev[offset + s]
            stage[2, s] = self.cand_preg[offset + s]

    @ti.kernel
    def _stage_in_kernel(self, stage: ti.template(), n: ti.i32):
        for s in range(n):
            linear = stage[0, s]
            k = linear % self.shape[2]
            j = (linear // self.shape[2]) % self.shape[1]
            i = linear // (self.shape[1] * self.shape[2])
            self.new_code[i, j, k] = stage[1, s]
            self.new_cand[i, j, k] = stage[2, s]

    @ti.kernel
    def _keep_codes_kernel(self):
        for I in ti.grouped(self.code):
            if self.new_code[I] < 0:
                self.new_code[I] = self.code[I]

    @ti.kernel
    def _old_table_kernel(self, table: ti.template(), n: ti.i32):
        for p in range(n):
            for s in ti.static(range(4)):
                table[s, p] = ti.cast(self.fate[p, s], ti.f64)
            table[4, p] = self.q[p]

    @ti.kernel
    def _volume_table_kernel(self, table: ti.template(), n: ti.i32):
        for p in range(n):
            table[0, p] = self.V[p]
        table[1, 0] = self.stats[S_AIR_VOLUME]
        table[1, 1] = self.stats[S_AIR_CELLS]

    @ti.kernel
    def _gas_in_kernel(self, table: ti.template(), n: ti.i32, old: ti.i32):
        for p in range(n):
            self.q[p] = table[0, p]
            self.checked[p] = ti.cast(table[1, p], ti.i32)     # 1: continued through this change
        for p in range(n, old):
            self.q[p] = 0.0

    @ti.kernel
    def _commit_kernel(self):
        for I in ti.grouped(self.code):
            value = self.new_code[I]
            if value < 0:
                value = 3                    # a candidate that no host stage classified: water pocket (no pockets)
            self.code[I] = value
            self.cand_id[I] = self.new_cand[I]
            if value >= 4:
                self.last_free[I] = 0        # a sealed air pocket: its number does not survive a renumbering
            elif value != 0:
                self.last_free[I] = value    # INLET_GAS, LIQUID or a water pocket; material keeps the history

    @ti.kernel
    def _chamber_codes_kernel(self):
        for i, j, k in self.code:
            c = self.code[i, j, k]
            out = 0
            if c == 1:
                out = 1
            elif c >= 4:
                out = c - 2
            self.chamber.box_code[i, j, k] = out
        for I in ti.grouped(self.air_cell):
            self.air_cell[I] = 0
        for i, j, k in self.code:
            c = self.code[i, j, k]
            if c == 1 or c >= 4:
                self.air_cell[i + self.chamber.offset[0], j + self.chamber.offset[1], k + self.chamber.offset[2]] = 1

    @ti.kernel
    def _volumes_kernel(self, pockets: ti.i32, regions: ti.i32):
        for r_ in range(regions):
            self.rV[r_] = 0.0
        for k_ in range(pockets):
            self.V[k_] = 0.0
            self.A[k_] = 0.0
            self.M[k_] = 0.0
            self.KE[k_] = 0.0
        self.stats[S_AIR_VOLUME] = 0.0
        self.stats[S_AIR_CELLS] = 0.0
        self.stats[S_INLET_AIR] = 0.0
        for i, j, k in self.code:
            c = self.code[i, j, k]
            cover = ti.cast(self.occupancy.coverage_field[i, j, k], ti.f64)
            free_volume = (1.0 - cover) * self.vol[i, j, k]
            if c >= 4:
                p = c - 4
                ti.atomic_add(self.V[p], free_volume)
                area = ti.cast(0.0, ti.f64)
                if i > 0 and self.code[i - 1, j, k] == 0 and self.wall[i - 1, j, k] == 0:
                    area += self._face_area(i, j, k, 0)
                if i < self.shape[0] - 1 and self.code[i + 1, j, k] == 0 and self.wall[i + 1, j, k] == 0:
                    area += self._face_area(i, j, k, 0)
                if j > 0 and self.code[i, j - 1, k] == 0 and self.wall[i, j - 1, k] == 0:
                    area += self._face_area(i, j, k, 1)
                if j < self.shape[1] - 1 and self.code[i, j + 1, k] == 0 and self.wall[i, j + 1, k] == 0:
                    area += self._face_area(i, j, k, 1)
                if k > 0 and self.code[i, j, k - 1] == 0 and self.wall[i, j, k - 1] == 0:
                    area += self._face_area(i, j, k, 2)
                if k < self.shape[2] - 1 and self.code[i, j, k + 1] == 0 and self.wall[i, j, k + 1] == 0:
                    area += self._face_area(i, j, k, 2)
                ti.atomic_add(self.A[p], area)
                top = k == self.shape[2] - 1
                if not top:
                    top = self.code[i, j, k + 1] != c
                if top and ti.static(self.occupancy.with_columns):
                    ti.atomic_add(self.M[p], self.occupancy.column_mass_field[i, j])
                    ti.atomic_add(self.KE[p], self.occupancy.column_ke_field[i, j])
            elif c == 1:
                ti.atomic_add(self.stats[S_AIR_VOLUME], free_volume)
                ti.atomic_add(self.stats[S_AIR_CELLS], 1.0)
                region = self.cand_id[i, j, k]
                if region > 0:                  # a pending region is inlet gas with its own sum
                    ti.atomic_add(self.rV[region - 1], free_volume)
                if self.inlet[i, j, k] != 0:
                    ti.atomic_add(self.stats[S_INLET_AIR], 1.0)
            elif c == 0 and self.wall[i, j, k] == 0:
                around = ti.Vector([0, 0, 0, 0, 0, 0])
                if i > 0:
                    around[0] = self.code[i - 1, j, k]
                if i < self.shape[0] - 1:
                    around[1] = self.code[i + 1, j, k]
                if j > 0:
                    around[2] = self.code[i, j - 1, k]
                if j < self.shape[1] - 1:
                    around[3] = self.code[i, j + 1, k]
                if k > 0:
                    around[4] = self.code[i, j, k - 1]
                if k < self.shape[2] - 1:
                    around[5] = self.code[i, j, k + 1]
                distinct = 0
                for a in ti.static(range(6)):
                    first = around[a] >= 1
                    for b in ti.static(range(a)):
                        if around[b] == around[a]:
                            first = False
                    if first:
                        distinct += 1
                if distinct > 0:
                    share = free_volume / distinct
                    for a in ti.static(range(6)):
                        first = around[a] >= 1
                        for b in ti.static(range(a)):
                            if around[b] == around[a]:
                                first = False
                        if first:
                            if around[a] >= 4:
                                ti.atomic_add(self.V[around[a] - 4], share)
                            elif around[a] == 1:
                                ti.atomic_add(self.stats[S_AIR_VOLUME], share)

    @ti.kernel
    def _pressure_kernel(self, pockets: ti.i32, ambient: ti.f64, low: ti.f64, high: ti.f64, compare: ti.i32):
        for s in ti.static((S_OMEGA_MAX, S_ZERO_VOLUME, S_BAD_PRESSURE, S_OUT_OF_BOUNDS, S_KE, S_POCKET_VOLUME,
                            S_P_MAX)):
            self.stats[s] = 0.0
        self.stats[S_V_MIN] = 1.0e300
        self.stats[S_P_MIN] = 1.0e300
        self.stats[S_WORST_POCKET] = -1.0
        self.stats[S_OMEGA_CHECKED] = 0.0
        self.stats[S_WORST_CHECKED] = -1.0
        self.stats[S_EVENTS] = 0.0
        self.stats[S_EVENT_DP] = 0.0
        self.stats[S_EVENT_POCKET] = 1.0e300
        self.stats[S_EVENT_VOLUME] = 0.0
        for s in range(3 * NBINS):
            self.stats[S_NOISE + s] = 0.0
        for p in range(pockets):
            volume = self.V[p]
            self.dp[p] = 0.0
            ti.atomic_min(self.stats[S_V_MIN], volume)
            ti.atomic_add(self.stats[S_POCKET_VOLUME], volume)
            ti.atomic_add(self.stats[S_KE], self.KE[p])
            pressure = ti.cast(0.0, ti.f64)
            if volume > 0.0:
                pressure = self.q[p] / volume
                if pressure > 0.0 and pressure < 1.0e300:
                    if pressure < low or pressure > high:
                        ti.atomic_add(self.stats[S_OUT_OF_BOUNDS], 1.0)
                    ti.atomic_min(self.stats[S_P_MIN], pressure)
                    ti.atomic_max(self.stats[S_P_MAX], pressure)
                    self.chamber.pocket_gauge[p] = pressure - ambient
                    if self.M[p] > 0.0:
                        omega = ti.sqrt(pressure * self.A[p] * self.A[p] / (volume * self.M[p]))
                        old = ti.atomic_max(self.stats[S_OMEGA_MAX], omega)
                        if omega > old:
                            self.stats[S_WORST_POCKET] = ti.cast(p, ti.f64)
                        if self.checked[p] != 0:
                            old_checked = ti.atomic_max(self.stats[S_OMEGA_CHECKED], omega)
                            if omega > old_checked:
                                self.stats[S_WORST_CHECKED] = ti.cast(p, ti.f64)
                    if compare != 0:
                        bin_ = 0
                        for b in range(NBINS - 1):
                            if volume >= self.bins[b]:
                                bin_ = b + 1
                        dp = pressure - self.p_prev[p]
                        ti.atomic_max(self.stats[S_NOISE + 3 * bin_], ti.abs(dp))
                        ti.atomic_add(self.stats[S_NOISE + 3 * bin_ + 1], dp * dp)
                        ti.atomic_add(self.stats[S_NOISE + 3 * bin_ + 2], 1.0)
                        self.dp[p] = dp
                        if volume >= NOISE_EVENT_VOLUME_M3 and ti.abs(dp) > NOISE_EVENT_DP_PA:
                            ti.atomic_add(self.stats[S_EVENTS], 1.0)
                            ti.atomic_max(self.stats[S_EVENT_DP], ti.abs(dp))
                else:
                    ti.atomic_add(self.stats[S_BAD_PRESSURE], 1.0)
            else:
                ti.atomic_add(self.stats[S_ZERO_VOLUME], 1.0)
            self.p_prev[p] = pressure
        for p in range(pockets):                    # the largest event's pocket: the smallest index with that |dp|
            if self.V[p] >= NOISE_EVENT_VOLUME_M3 and ti.abs(self.dp[p]) > NOISE_EVENT_DP_PA:
                if ti.abs(self.dp[p]) == self.stats[S_EVENT_DP]:
                    ti.atomic_min(self.stats[S_EVENT_POCKET], ti.cast(p, ti.f64))
        for p in range(pockets):
            if ti.cast(p, ti.f64) == self.stats[S_EVENT_POCKET]:
                self.stats[S_EVENT_VOLUME] = self.V[p]
        for p in range(pockets):                    # every pocket continues into the next step
            self.checked[p] = 1

    @ti.kernel
    def _bad_cell_box_kernel(self, which: ti.i32):
        """min / max box indices of the cells of a failure (1: leak, 2: air -> water, 3: water -> air) into badbox."""
        for s in ti.static(range(3)):
            self.badbox[s] = 2147483647
            self.badbox[3 + s] = -1
        for i, j, k in self.reach:
            bad = False
            if self._free(i, j, k):
                old = self.code[i, j, k]
                r = self.reach[i, j, k]
                if which == 1:
                    bad = r == 3
                elif which == 2:
                    bad = (old == 1 or old >= 4) and r == 2
                else:
                    bad = (old == 2 or old == 3) and r == 1
            if bad:
                ti.atomic_min(self.badbox[0], i)
                ti.atomic_min(self.badbox[1], j)
                ti.atomic_min(self.badbox[2], k)
                ti.atomic_max(self.badbox[3], i)
                ti.atomic_max(self.badbox[4], j)
                ti.atomic_max(self.badbox[5], k)

    # ------------------------------------------------------------------------------------------------ host stages
    def _flood(self):
        self._flood_init_kernel()
        levels = 0
        while True:
            for _ in range(self.flood_batch):
                self._flood_level_kernel()
            levels += self.flood_batch
            frontier = int(self._read(self.ctr)[C_FRONTIER])
            if frontier == 0:
                break
            self.ctr[C_FRONTIER] = 0
            if levels > 4 * sum(self.shape):
                raise RuntimeError("the occupancy flood did not converge in %d levels" % levels)
        self.flood_levels += levels
        return levels

    def _fail(self, code, which, counters, t_s):
        self._bad_cell_box_kernel(which)
        box = self.badbox.to_numpy()
        first = int(counters[C_FIRST_BAD])
        raise SimulationHardStop(code, dict(
            time_s=float(t_s), cells=int(counters[{1: C_LEAK, 2: C_AIR_TO_WATER, 3: C_WATER_TO_AIR}[which]]),
            first_box_index=[int(v) for v in np.unravel_index(first, self.shape)] if 0 <= first < np.prod(self.shape)
            else None, box_index_min=[int(v) for v in box[:3]], box_index_max=[int(v) for v in box[3:]],
            box_offset=[int(v) for v in self.chamber.offset]))

    def _read(self, field):
        value = field.to_numpy()
        self.transfers["reads"] += 1
        self.transfers["read_bytes"] += int(value.nbytes)
        return value

    def _write(self, field, value):
        field.from_numpy(value)
        self.transfers["writes"] += 1
        self.transfers["write_bytes"] += int(value.nbytes)

    def _download(self, count):
        indices = np.empty(count, dtype=np.int64)
        previous = np.empty(count, dtype=np.int64)
        regions = np.empty(count, dtype=np.int64)
        for offset in range(0, count, self.staging):
            n = min(self.staging, count - offset)
            stage = self.stage[_fit(self.stage, n)]
            self._stage_out_kernel(stage, offset, n)
            block = self._read(stage)
            indices[offset:offset + n] = block[0, :n]
            previous[offset:offset + n] = block[1, :n]
            regions[offset:offset + n] = block[2, :n]
        return indices, previous, regions

    def _upload(self, indices, codes, regions):
        for offset in range(0, len(indices), self.staging):
            n = min(self.staging, len(indices) - offset)
            size = _fit(self.stage, n)
            block = np.zeros((STAGE_ROWS, size), dtype=np.int32)
            block[0, :n] = indices[offset:offset + n]
            block[1, :n] = codes[offset:offset + n]
            block[2, :n] = regions[offset:offset + n]
            self._write(self.stage[size], block)
            self._stage_in_kernel(self.stage[size], n)

    def _topology_stage(self, t_s, inlet_gauge_pa):
        self._flood()
        old_pockets = self.pockets
        old_regions = len(self.pending_regions)
        self._classify_kernel(old_pockets, old_regions, int(self.type_memory))
        counters = self._read(self.ctr)
        if counters[C_LEAK]:
            self._fail("gas_region_leak", 1, counters, t_s)
        if counters[C_AIR_TO_WATER]:
            self._fail("gas_region_gas_to_liquid", 2, counters, t_s)
        if counters[C_WATER_TO_AIR]:
            self._fail("gas_region_liquid_to_gas", 3, counters, t_s)
        candidates = int(counters[C_CANDIDATES])
        self.topology_changes += 1
        out = dict(candidates=candidates, inlet_air_cells=int(counters[C_INLET_AIR]),
                   floor_material_cells=int(counters[C_FLOOR_MATERIAL]), host_stage=False)
        if candidates == 0 and old_pockets == 0 and old_regions == 0:
            self._commit_kernel()
            self.pockets = 0
            self._chamber_codes_kernel()
            self.chamber.pocket_count = 0
            return dict(out, pockets=0)
        if (old_regions == 0 and int(counters[C_NEW_CANDIDATES]) == 0
                and candidates == int(counters[C_OLD_POCKET_CELLS])):
            self._keep_codes_kernel()                 # the same sealed cells: the same components, codes and gas
            self._commit_kernel()
            self._chamber_codes_kernel()
            self.pocket_set_unchanged += 1
            return dict(out, pockets=self.pockets)
        start = time.perf_counter()
        self.host_stages += 1
        indices, previous, previous_regions = self._download(candidates)
        fate = np.zeros((0, 4), dtype=np.int64)
        q_old = np.zeros(0, dtype=np.float64)
        if old_pockets:
            table = self.table[_fit(self.table, old_pockets)]
            self._old_table_kernel(table, old_pockets)
            block = self._read(table)
            fate = block[:4, :old_pockets].T.astype(np.int64)
            q_old = block[4, :old_pockets].copy()
        region_fate = np.zeros((0, 4), dtype=np.int64)
        if old_regions:
            region_fate = self._read(self.rfate)[:old_regions].astype(np.int64)
        plan = plan_pocket_transitions(indices, previous, fate, old_pockets, self.shape, t_s=t_s,
                                       previous_regions=previous_regions,
                                       region_ages=[r["age"] for r in self.pending_regions],
                                       region_fate=region_fate, seal_steps=self.seal_steps,
                                       min_cells=self.min_cells)
        if plan.pockets > self.max_pockets or len(plan.regions) > self.max_pockets:
            raise RuntimeError("%d air pockets / %d candidate regions exceed max_pockets %d"
                               % (plan.pockets, len(plan.regions), self.max_pockets))
        self._upload(indices, plan.candidate_codes, plan.candidate_regions)
        self._commit_kernel()
        self.pockets = plan.pockets
        self._volumes_kernel(self.pockets, len(plan.regions))
        size = _fit(self.table, max(self.pockets, 2))
        self._volume_table_kernel(self.table[size], self.pockets)
        block = self._read(self.table[size])
        volumes = block[0, :self.pockets].copy()
        q_new, exchange = allocate_pocket_gas(plan, q_old, volumes, inlet_absolute_pa=self.ambient_pressure_pa + inlet_gauge_pa,
                                       air_volume_m3=float(block[1, 0]), air_cells=float(block[1, 1]))
        gas = np.zeros((TABLE_ROWS, size), dtype=np.float64)
        gas[0, :self.pockets] = q_new
        gas[1, :self.pockets] = continuing_pockets(plan)      # row 1: the check flags, same write
        self._write(self.table[size], gas)
        self._gas_in_kernel(self.table[size], self.pockets, old_pockets)
        self.exchange_pa_m3 += float(exchange)
        self.seal_events += int((plan.air_cells_of_new > 0).sum())
        self._candidate_census(plan, t_s, inlet_gauge_pa, volumes)
        self._chamber_codes_kernel()
        self.chamber.pocket_count = self.pockets
        self.host_stage_s += time.perf_counter() - start
        return dict(out, pockets=self.pockets, host_stage=True)

    def _candidate_census(self, plan, t_s, inlet_gauge_pa, pocket_volumes):
        """Sealing-rule bookkeeping of one host stage: the new pending regions (their census id and birth time carried
        by cell overlap from the oldest parent region), the regions sealed now and the old regions that ended unsealed
        (counters and events; an ended region's free volume is its last refresh's, its fraction of the inlet air that
        refresh's inlet-air volume, its age the updates it persisted as a candidate; a sealed region's age includes the
        sealing update)."""
        old, counters, events = self.pending_regions, self.candidate_counters, []
        volumes = self._read(self.rV)[:len(plan.regions)] if plan.regions else np.zeros(0)
        new = []
        for region, volume in zip(plan.regions, volumes):
            if region["parents"]:
                first = min((old[r - 1] for r in region["parents"]), key=lambda o: (o["born_t_s"], o["census_id"]))
                census_id, born = first["census_id"], first["born_t_s"]
            else:
                census_id, born = counters["census_next"], float(t_s)
                counters["census_next"] += 1
                counters["created"] += 1
                events.append(dict(kind="created", census_id=census_id, t_s=float(t_s), cells=region["cells"],
                                   free_volume_m3=float(volume), inlet_gauge_pa=float(inlet_gauge_pa)))
            new.append(dict(census_id=census_id, born_t_s=born, age=region["age"], cells=region["cells"],
                            free_volume_m3=float(volume)))
            counters["cells_max"] = max(counters["cells_max"], region["cells"])
            counters["age_max"] = max(counters["age_max"], region["age"])
        for s in plan.sealed:
            counters["sealed"] += 1
            counters["cells_max"] = max(counters["cells_max"], s["cells"])
            counters["age_max"] = max(counters["age_max"], s["age"])
            events.append(dict(kind="sealed", census_ids=[old[r - 1]["census_id"] for r in s["parents"]],
                               t_s=float(t_s), age=s["age"], cells=s["cells"], pocket=s["pocket"],
                               volume_m3=float(pocket_volumes[s["pocket"]]), inlet_gauge_pa=float(inlet_gauge_pa)))
        air = self.last_air_volume_m3
        for e in plan.ended:
            o = old[e["region"] - 1]
            counters[e["how"]] += 1
            volume = float(o["free_volume_m3"])
            fraction = volume / air if air else None
            counters["ended_volume_m3"] += volume
            if fraction is not None:
                counters["ended_fraction_max"] = max(counters["ended_fraction_max"], fraction)
            events.append(dict(kind="ended", how=e["how"], census_id=o["census_id"], t_s=float(t_s),
                               born_t_s=o["born_t_s"], age=o["age"] - 1, cells=o["cells"], free_volume_m3=volume,
                               fraction_of_inlet_air=fraction, fate=e["fate"]))
        self.pending_regions = new
        self.candidate_events_new.extend(events)
        room = CANDIDATE_EVENT_LOG_MAX - len(self.candidate_events)
        if room > 0:
            self.candidate_events.extend(dict(e) for e in events[:room])

    # ------------------------------------------------------------------------------------------------ host API
    def refresh(self, t_s, inlet_gauge_pa):
        """One accepted geometry: occupancy, topology (when it changed), volumes, pressures and the chamber's codes and
        pocket pressures.  Raises SimulationHardStop on a leak, a direct air <-> water change, an ambiguous pocket
        origin or topology, a pocket without volume or with an invalid / out-of-bounds pressure."""
        self.transfers = _no_transfers()
        self.candidate_events_new = []
        first = not self.initialized
        skip = self.raster_interval > 1 and not first and self.refreshes % self.raster_interval != 0
        if skip:                                          # raster interval: the last occupancy, no flips, no ageing
            occupancy = dict(getattr(self.occupancy, "last", None) or {}, flipped=0, rasterised=False)
            self.raster_skips += 1
        else:
            occupancy = self.occupancy.update()
            self.transfers["reads"] += 1                      # SolidOccupancy.update(): its statistics
            self.transfers["read_bytes"] += 8 * int(self.occupancy.stats.shape[0])
            # one more occupancy update per pending region (new records: a saved host state is never aliased)
            self.pending_regions = [dict(r, age=r["age"] + 1) for r in self.pending_regions]
            if self.pending_regions:
                self.candidate_counters["age_max"] = max(self.candidate_counters["age_max"],
                                                         max(r["age"] for r in self.pending_regions))
        if first:
            self.occupancy.reset_topology()
        topology = None
        due = (not skip) and any(r["cells"] >= self.min_cells and r["age"] >= self.seal_steps
                                 for r in self.pending_regions)     # a candidate region came of age
        if first or occupancy["flipped"] > 0 or due:
            topology = self._topology_stage(t_s, float(inlet_gauge_pa))
        self.initialized = True
        self.refreshes += 1
        regions = len(self.pending_regions)
        self._volumes_kernel(self.pockets, regions)
        if regions:                                       # the pending regions' free volumes
            self.pending_regions = [dict(r, free_volume_m3=float(v))
                                    for r, v in zip(self.pending_regions, self._read(self.rV)[:regions])]
        low, high = self.pressure_bounds_abs_pa or (0.0, 1.0e300)
        self._pressure_kernel(self.pockets, self.ambient_pressure_pa, low, high, 0 if topology is not None else 1)
        s = self._read(self.stats)
        witness = dict(time_s=float(t_s), pockets=self.pockets, worst_pocket=int(s[S_WORST_POCKET]))
        self.last_air_volume_m3 = float(s[S_AIR_VOLUME])
        if s[S_ZERO_VOLUME]:
            raise SimulationHardStop("gas_pocket_zero_volume", dict(witness, count=int(s[S_ZERO_VOLUME])))
        if s[S_BAD_PRESSURE]:
            raise SimulationHardStop("gas_pocket_invalid_pressure", dict(witness, count=int(s[S_BAD_PRESSURE])))
        if s[S_OUT_OF_BOUNDS]:
            raise SimulationHardStop("gas_pocket_pressure_out_of_bounds", dict(
                witness, count=int(s[S_OUT_OF_BOUNDS]), bounds_abs_pa=list(self.pressure_bounds_abs_pa),
                p_min_abs_pa=float(s[S_P_MIN]), p_max_abs_pa=float(s[S_P_MAX])))
        noise = [dict(volume_from_m3=([0.0] + list(self.volume_bins_m3))[b], max_abs_dp_pa=float(s[S_NOISE + 3 * b]),
                      sum_dp2_pa2=float(s[S_NOISE + 3 * b + 1]), count=int(s[S_NOISE + 3 * b + 2]))
                 for b in range(NBINS)]
        events = int(s[S_EVENTS])
        event = (dict(t_s=float(t_s), count=events, pocket=int(s[S_EVENT_POCKET]), volume_m3=float(s[S_EVENT_VOLUME]),
                      abs_dp_pa=float(s[S_EVENT_DP])) if events else None)
        omega = float(s[S_OMEGA_MAX])
        self.last_omega_rad_s = omega
        self.last_omega_checked_rad_s = float(s[S_OMEGA_CHECKED])
        self.last_t_s = float(t_s)
        self.last = dict(
            t_s=float(t_s), occupancy=occupancy, topology=topology, pockets=self.pockets,
            inlet_covered=bool(s[S_INLET_AIR] == 0.0), inlet_air_cells=int(s[S_INLET_AIR]),
            omega_max_rad_s=omega, worst_pocket=int(s[S_WORST_POCKET]),
            omega_checked_max_rad_s=self.last_omega_checked_rad_s, worst_checked_pocket=int(s[S_WORST_CHECKED]),
            dt_limit_s=(STIFFNESS_MARGIN * STIFFNESS_LIMIT / omega) if omega > 0.0 else math.inf,
            pocket_volume_m3=float(s[S_POCKET_VOLUME]), pocket_v_min_m3=(float(s[S_V_MIN]) if self.pockets else None),
            pocket_p_min_abs_pa=(float(s[S_P_MIN]) if self.pockets else None),
            pocket_p_max_abs_pa=(float(s[S_P_MAX]) if self.pockets else None),
            pocket_footprint_kinetic_energy_j=float(s[S_KE]), air_volume_m3=float(s[S_AIR_VOLUME]),
            air_cells=int(s[S_AIR_CELLS]), noise=noise, exchange_pa_m3=self.exchange_pa_m3,
            seal_events=self.seal_events, transfers=dict(self.transfers), noise_events=events, noise_event=event,
            candidates_pending=len(self.pending_regions),
            candidate_cells_pending=int(sum(r["cells"] for r in self.pending_regions)),
            candidate_counters=dict(self.candidate_counters),
            candidate_events_new=[dict(e) for e in self.candidate_events_new],
            raster_skips=self.raster_skips, rasterised=not skip)
        return dict(self.last)

    def dt_limit(self):
        """The pocket stiffness bound 0.9 x 0.5 / max omega of the last refresh (inf without pockets)."""
        omega = self.last_omega_rad_s
        return STIFFNESS_MARGIN * STIFFNESS_LIMIT / omega if omega > 0.0 else math.inf

    def check_step(self, dt_s):
        """After a refresh: the step just taken must satisfy omega dt <= 0.5 for every pocket that CONTINUED through the
        refresh (hard stop otherwise).  A pocket that is new, split or merged at this refresh did not act as a gas spring
        during the step (it held the inlet pressure, or its parent's pressure: q is shared by volume); it is exempt once,
        and its omega already bounds the next step through dt_limit()."""
        omega = self.last_omega_checked_rad_s
        if omega * float(dt_s) > STIFFNESS_LIMIT:
            raise SimulationHardStop("gas_pocket_stiffness", dict(
                time_s=self.last_t_s, dt_s=float(dt_s), omega_rad_s=omega, omega_dt=omega * float(dt_s),
                pocket=(self.last or {}).get("worst_checked_pocket"), pockets=self.pockets,
                omega_all_pockets_rad_s=self.last_omega_rad_s))

    def codes(self):
        """Host copy of the accepted codes (tests, outputs)."""
        return self.code.to_numpy()

    def pocket_table(self):
        """Host copies of the pockets' gas, volume, area, mass and pressure (tests, outputs)."""
        n = self.pockets
        q, V = self.q.to_numpy()[:n], self.V.to_numpy()[:n]
        return dict(q_pa_m3=q, volume_m3=V, area_m2=self.A.to_numpy()[:n], mass_kg=self.M.to_numpy()[:n],
                    pressure_abs_pa=np.where(V > 0.0, q / np.where(V > 0.0, V, 1.0), np.nan))

    def _after_load_state(self):
        self._chamber_codes_kernel()
        self.chamber.pocket_count = self.pockets

    def load_state_host(self, values):
        """Checkpointable.load_state_host, then the region bounds (the arrays are loaded first): the pending regions fit
        the region tables and every cell's pending-region number names one of them; otherwise the fate and volume
        kernels would index out of bounds.  Raises ValueError."""
        super().load_state_host(values)
        regions = len(self.pending_regions)
        if regions > self.max_pockets:
            raise ValueError("%d pending regions in the loaded state exceed max_pockets %d" % (regions,
                                                                                            self.max_pockets))
        cand = self.cand_id.to_numpy()
        top, bottom = (int(cand.max()), int(cand.min())) if cand.size else (0, 0)
        if top > regions or bottom < 0:
            raise ValueError("cand_id holds region numbers %d .. %d but the loaded state has %d pending regions"
                             % (bottom, top, regions))
