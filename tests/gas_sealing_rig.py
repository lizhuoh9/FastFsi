"""The synthetic rig of test_gas_pocket_sealing.py: a fake occupancy and chamber on a 6 x 6 x 8 box of 1 mm^3 cells,
masks and the inlet pressure sequence (no fastfsi import: the device class under test is passed in)."""
import numpy as np
import taichi as ti

SHAPE = (6, 6, 8)
H = 1.0e-3
V = H ** 3


@ti.data_oriented
class FakeOccupancy:
    """test_gas_regions_device.py's fake (topology mask, coverage, flip count, column sums), with the flips counted per
    update() against the mask of the previous update(), as SolidOccupancy counts them (a skipped rasterisation leaves
    its change to the next update)."""

    def __init__(self):
        self.shape = SHAPE
        self.box_cells = tuple((0, n) for n in SHAPE)
        self._faces = [ti.field(ti.f64, shape=n + 1) for n in SHAPE]
        for field, n in zip(self._faces, SHAPE):
            field.from_numpy(np.arange(n + 1) * H)
        self.cell_volume_m3 = np.full(SHAPE, V)
        self.topo = ti.field(ti.i32, shape=SHAPE)
        self.coverage_field = ti.field(ti.f32, shape=SHAPE)
        self.with_columns = True
        self.column_mass_field = ti.field(ti.f64, shape=SHAPE[:2])
        self.column_ke_field = ti.field(ti.f64, shape=SHAPE[:2])
        self.column_mass_field.fill(1.0e-6)
        self.stats = ti.field(ti.f64, shape=9)
        self.updates = 0
        self.current = np.zeros(SHAPE, dtype=bool)
        self.at_update = np.zeros(SHAPE, dtype=bool)

    def set(self, material):
        self.topo.from_numpy(material.astype(np.int32))
        self.coverage_field.from_numpy(np.where(material, 1.0, 0.0).astype(np.float32))
        self.current = material.copy()

    def update(self):
        self.updates += 1
        flips = int((self.current != self.at_update).sum())
        self.at_update = self.current.copy()
        return dict(flipped=flips)

    def reset_topology(self):
        return None


@ti.data_oriented
class FakeChamber:
    def __init__(self):
        self.shape = SHAPE
        self.box_shape = SHAPE
        self.offset = (0, 0, 0)
        self.box_code = ti.field(ti.i32, shape=SHAPE)
        self.pocket_gauge = ti.field(ti.f64, shape=64)
        self.pocket_count = 0


class Rig:
    """A DeviceGasRegions-like class (passed in) on the fake occupancy and chamber: a membrane slab at k = 3 (inlet gas
    above, liquid below) and the inlet at (0, 0, 7)."""

    def __init__(self, device_class, **options):
        wall = np.zeros(SHAPE, dtype=bool)
        inlet = np.zeros(SHAPE, dtype=bool)
        inlet[0, 0, -1] = True
        self.slab = np.zeros(SHAPE, dtype=bool)
        self.slab[:, :, 3] = True
        self.occ = FakeOccupancy()
        self.chamber = FakeChamber()
        self.air_cell = ti.field(ti.i32, shape=SHAPE)
        self.device = device_class(self.occ, self.chamber, self.air_cell, wall, inlet,
                                   **dict(dict(max_pockets=64, flood_batch=4, staging=8), **options))
        self.events = []

    def step(self, material, t, p_in):
        self.occ.set(material)
        out = self.device.refresh(t, p_in)
        self.events += out.get("candidate_events_new") or []
        return out


def enclose(material, cells):
    """Material on every face neighbour of `cells` (a list of (i, j, k)) that is not one of them."""
    out = material.copy()
    inside = set(map(tuple, cells))
    for i, j, k in cells:
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            n = (i + d[0], j + d[1], k + d[2])
            if n not in inside and all(0 <= n[a] < SHAPE[a] for a in range(3)):
                out[n] = True
    return out


ONE = [(3, 3, 5)]
BLOCK = [(i, j, k) for i in (2, 3) for j in (2, 3) for k in (5, 6)]
SEVEN = [c for c in BLOCK if c != (3, 3, 6)]
CORNER = [(5, 5, 5)]                 # a 1-cell region away from BLOCK's shell


def p_in(n):
    return -4500.0 + 100.0 * n


def scenario_one(rig):
    """A 1-cell candidate born at the 2nd refresh and covered at the 4th (2 updates later)."""
    rig.step(rig.slab, 0.0, p_in(0))
    sealed = enclose(rig.slab, ONE)
    out1 = rig.step(sealed, 0.1, p_in(1))
    out2 = rig.step(sealed, 0.2, p_in(2))
    covered = sealed.copy()
    covered[ONE[0]] = True
    out3 = rig.step(covered, 0.3, p_in(3))
    return out1, out2, out3
