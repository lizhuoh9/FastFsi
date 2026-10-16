"""DeviceGasRegions (boundary/gas_regions_device.py) against the host reference path (GasRegions + GasPocketModel with
fractional volumes and seal_from_inlet_pressure), on synthetic topology masks in a 6 x 6 x 8 box of 1 mm^3 cells.

A fake occupancy supplies the topology mask, the coverage, the flip count and the column sums; a fake chamber
receives the codes and pocket pressures.  The sequence: a membrane slab (INLET_GAS above, LIQUID below), a shell above the
slab sealing a pocket off the inlet air, the same topology with less free volume (no topology stage), a split into two
pockets, the reconnection of one of them, and the hard stops (leak, air -> water).  Codes, pocket volumes and pocket
pressures must equal the reference; the staging is small (8 cells) so the chunked transfers are exercised."""
import math

import numpy as np
import pytest
import taichi as ti

from fastfsi.boundary.gas_regions import INLET_GAS as REF_AIR
from fastfsi.boundary.gas_regions import LIQUID as REF_WATER
from fastfsi.boundary.gas_regions import GasRegions
from fastfsi.boundary.gas_regions_device import STAGE_ROWS, STATS_SLOTS, P_ATMOSPHERE_PA, DeviceGasRegions
from fastfsi.boundary.gas_pockets import GasPocketModel, PressureInlet
from fastfsi.errors import SimulationHardStop

SHAPE = (6, 6, 8)
H = 1.0e-3
V = H ** 3
P_IN = -4500.0


@pytest.fixture(scope="module", autouse=True)
def cpu():
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1, opt_level=0, offline_cache=False)


@ti.data_oriented
class FakeOccupancy:
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
        self.stats = ti.field(ti.f64, shape=9)               # as SolidOccupancy: read once per update
        self.flips = 0
        self.previous = np.zeros(SHAPE, dtype=bool)

    def set(self, material, coverage=None):
        self.topo.from_numpy(material.astype(np.int32))
        cover = np.where(material, 1.0, 0.0) if coverage is None else coverage
        self.coverage_field.from_numpy(cover.astype(np.float32))
        self.flips = int((material != self.previous).sum())
        self.previous = material.copy()

    def update(self):
        return dict(flipped=self.flips)

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


def masks():
    wall = np.zeros(SHAPE, dtype=bool)
    inlet = np.zeros(SHAPE, dtype=bool)
    inlet[0, 0, -1] = True
    slab = np.zeros(SHAPE, dtype=bool)
    slab[:, :, 3] = True
    return wall, inlet, slab


def shell(material, k0, k1):
    out = material.copy()
    out[1:5, 1:5, k0:k1 + 1] = True
    out[2:4, 2:4, k0 + 1:k1] = False
    return out


class Pair:
    """The device path and the reference path driven with the same masks."""

    def __init__(self, **options):
        wall, inlet, slab = masks()
        self.occ = FakeOccupancy()
        self.chamber = FakeChamber()
        self.air_cell = ti.field(ti.i32, shape=SHAPE)
        # seal_steps = min_cells = 1: the earlier rule (every candidate seals at once), as the host reference path
        options = dict(dict(max_pockets=64, flood_batch=4, staging=8, seal_steps=1, min_cells=1), **options)
        self.device = DeviceGasRegions(self.occ, self.chamber, self.air_cell, wall, inlet, **options)
        self.regions = GasRegions(np.full(SHAPE, V), wall, inlet)
        self.gas = GasPocketModel(np.full(SHAPE, V), [PressureInlet(inlet, P_IN)], ambient_pressure_pa=P_ATMOSPHERE_PA,
                                  fractional_volumes=True, seal_from_inlet_pressure=True)
        self.slab = slab

    def step(self, material, t, coverage=None):
        self.occ.set(material, coverage)
        out = self.device.refresh(t, P_IN)
        cover = np.where(material, 1.0, 0.0) if coverage is None else coverage
        c = self.regions.classify(material, cover, t_s=t)
        ref = self.gas.update(c.air_mask, t_s=t, cell_volumes_m3=c.effective_volume_m3)
        self.regions.accept(c)
        return out, c, ref

    def compare(self, c, ref):
        codes = self.device.codes()
        air_component = np.isin(c.labels, np.flatnonzero(c.component_class == REF_AIR))
        water_component = np.isin(c.labels, np.flatnonzero(c.component_class == REF_WATER))
        assert np.array_equal(codes == 1, air_component)
        assert np.array_equal(codes == 2, water_component)
        assert np.array_equal(codes >= 4, c.air_mask & ~air_component)
        assert np.array_equal(codes == 3, c.water_mask & ~water_component)
        table = self.device.pocket_table()
        for k in range(self.device.pockets):
            labels = np.unique(ref.labels[codes == 4 + k])
            assert len(labels) == 1 and labels[0] > 0 and not ref.connected_to_inlet[labels[0]]
            assert table["volume_m3"][k] == pytest.approx(ref.volume_m3[labels[0]], rel=1e-12)
            assert table["pressure_abs_pa"][k] == pytest.approx(ref.pressure_abs_pa[labels[0]], rel=1e-12)
            assert self.chamber.pocket_gauge[k] == pytest.approx(ref.pressure_abs_pa[labels[0]] - P_ATMOSPHERE_PA,
                                                                 rel=1e-12, abs=1e-9)
        assert len(ref.sealed_labels) == self.device.pockets
        chamber_codes = self.chamber.box_code.to_numpy()
        assert np.array_equal(chamber_codes == 1, codes == 1)
        assert np.array_equal(chamber_codes >= 2, codes >= 4)
        assert np.array_equal(self.air_cell.to_numpy() != 0, (codes == 1) | (codes >= 4))


def test_sequence_matches_the_reference_path():
    pair = Pair()
    out, c, ref = pair.step(pair.slab, 0.0)
    pair.compare(c, ref)
    assert out["pockets"] == 0 and not out["inlet_covered"] and out["dt_limit_s"] == math.inf
    sealed = shell(pair.slab, 3, 6)
    out, c, ref = pair.step(sealed, 0.1)
    pair.compare(c, ref)
    assert out["pockets"] == 1 and out["topology"]["candidates"] == 8 and pair.device.seal_events == 1
    p_seal = P_ATMOSPHERE_PA + P_IN
    assert pair.device.pocket_table()["pressure_abs_pa"][0] == pytest.approx(p_seal, rel=1e-12)
    # the same topology with 12.5 % less free volume in the pocket (f32-exact coverage): no topology stage, p = q / V
    stages = pair.device.host_stages
    cover = np.where(sealed, 1.0, 0.0)
    cover[2:4, 2:4, 4:6] = 0.125
    out, c, ref = pair.step(sealed, 0.2, coverage=cover)
    pair.compare(c, ref)
    assert out["topology"] is None and pair.device.host_stages == stages
    assert pair.device.pocket_table()["pressure_abs_pa"][0] == pytest.approx(p_seal / 0.875, rel=1e-12)
    noise = out["noise"][2]                                         # 7 mm^3: the 5 .. 10 mm^3 bin
    assert noise["count"] == 1 and noise["max_abs_dp_pa"] == pytest.approx(p_seal / 0.875 - p_seal, rel=1e-9)
    # split into two diagonal halves (4 interior cells become material)
    split = sealed.copy()
    split[2, 3, 4:6] = True
    split[3, 2, 4:6] = True
    out, c, ref = pair.step(split, 0.3)
    pair.compare(c, ref)
    assert out["pockets"] == 2
    q = pair.device.pocket_table()["q_pa_m3"]
    assert q[0] == pytest.approx(q[1], rel=1e-12)
    # reconnect pocket of the cells (2, 2, 4..5) to the inlet air through the shell's top
    open_ = split.copy()
    open_[2, 2, 6] = False
    exchange = pair.device.exchange_pa_m3
    out, c, ref = pair.step(open_, 0.4)
    pair.compare(c, ref)
    assert out["pockets"] == 1 and pair.device.exchange_pa_m3 < exchange


def test_leak_and_air_to_water_are_hard_stops():
    pair = Pair()
    pair.step(pair.slab, 0.0)
    holed = pair.slab.copy()
    holed[4, 4, 3] = False
    pair.occ.set(holed)
    with pytest.raises(SimulationHardStop) as stop:
        pair.device.refresh(0.1, P_IN)
    assert stop.value.code == "gas_region_leak" and stop.value.witness["box_index_min"] == [0, 0, 0]
    # a slab moved up by one cell in one refresh: the old INLET_GAS layer k = 4 is now LIQUID directly
    pair2 = Pair()
    pair2.step(pair2.slab, 0.0)
    moved = np.zeros(SHAPE, dtype=bool)
    moved[:, :, 5] = True
    pair2.occ.set(moved)
    with pytest.raises(SimulationHardStop) as stop:
        pair2.device.refresh(0.1, P_IN)
    assert stop.value.code == "gas_region_gas_to_liquid" and stop.value.witness["cells"] == 36


def pocket_omega(device, k=0):
    table = device.pocket_table()
    return math.sqrt(table["pressure_abs_pa"][k] * table["area_m2"][k] ** 2 / (table["volume_m3"][k]
                                                                              * table["mass_kg"][k]))


def test_stiffness_bound_and_step_check():
    """The seal's refresh bounds the next step with the new pocket's omega but does not check the step that sealed it;
    from the next refresh on the pocket is checked: omega dt <= 0.5 passes, above is the hard stop."""
    pair = Pair()
    pair.step(pair.slab, 0.0)
    sealed = shell(pair.slab, 3, 6)
    out, _, _ = pair.step(sealed, 0.1)
    omega = pocket_omega(pair.device)
    assert out["omega_max_rad_s"] == pytest.approx(omega, rel=1e-12)
    assert out["dt_limit_s"] == pytest.approx(0.45 / omega, rel=1e-12)
    assert out["omega_checked_max_rad_s"] == 0.0 and out["worst_checked_pocket"] == -1
    pair.device.check_step(10.0 / omega)                    # the sealing step: exempt
    out, _, _ = pair.step(sealed, 0.2)                      # no flip: the pocket continued through the step
    assert out["omega_checked_max_rad_s"] == pytest.approx(omega, rel=1e-12) and out["worst_checked_pocket"] == 0
    pair.device.check_step(0.5 / omega * 0.999)
    with pytest.raises(SimulationHardStop) as stop:
        pair.device.check_step(0.5 / omega * 1.001)
    assert stop.value.code == "gas_pocket_stiffness" and stop.value.witness["pocket"] == 0


def test_the_s1g6_witness_is_exempt_at_the_seal_and_bounded_after():
    """The S1+G6 stop (omega dt = 0.593 on the step that sealed the first pocket): no stop at the seal; the next step's
    bound is 0.45 / omega; the same 0.593 on a step the pocket took part in is still the hard stop."""
    pair = Pair()
    pair.step(pair.slab, 0.0)
    sealed = shell(pair.slab, 3, 6)
    out, _, _ = pair.step(sealed, 0.1)
    omega = pocket_omega(pair.device)
    pair.device.check_step(0.593 / omega)
    assert pair.device.dt_limit() == pytest.approx(0.9 * 0.5 / omega, rel=1e-12)
    pair.step(sealed, 0.2)
    with pytest.raises(SimulationHardStop):
        pair.device.check_step(0.593 / omega)


def test_split_and_merge_children_are_exempt_once():
    """A split gives two new pockets (exempt at the split, checked after); a reconnection leaves the other pocket
    continuing (checked at once); two pockets merging give one new pocket (exempt once)."""
    pair = Pair()
    pair.step(pair.slab, 0.0)
    sealed = shell(pair.slab, 3, 6)
    pair.step(sealed, 0.1)
    pair.step(sealed, 0.2)
    split = sealed.copy()
    split[2, 3, 4:6] = True
    split[3, 2, 4:6] = True
    out, _, _ = pair.step(split, 0.3)
    assert out["pockets"] == 2 and out["omega_checked_max_rad_s"] == 0.0 and out["omega_max_rad_s"] > 0.0
    pair.device.check_step(10.0 / out["omega_max_rad_s"])
    out, _, _ = pair.step(split, 0.4)
    assert out["omega_checked_max_rad_s"] == pytest.approx(out["omega_max_rad_s"], rel=1e-12)
    open_ = split.copy()
    open_[2, 2, 6] = False                                  # one half reconnects to the inlet air
    out, _, _ = pair.step(open_, 0.5)
    assert out["pockets"] == 1 and out["omega_checked_max_rad_s"] == pytest.approx(out["omega_max_rad_s"], rel=1e-12)
    merged = split.copy()                                   # the two halves of the earlier split join again
    merged[2, 3, 4:6] = False
    merged[3, 2, 4:6] = False
    pair2 = Pair()
    pair2.step(pair2.slab, 0.0)
    pair2.step(sealed, 0.1)
    pair2.step(split, 0.2)
    pair2.step(split, 0.3)
    out, _, _ = pair2.step(merged, 0.4)
    assert out["pockets"] == 1 and out["omega_checked_max_rad_s"] == 0.0
    out, _, _ = pair2.step(merged, 0.5)
    assert out["omega_checked_max_rad_s"] == pytest.approx(out["omega_max_rad_s"], rel=1e-12)


def test_a_pocket_shrunk_by_material_continues_and_is_checked():
    """Cells of a pocket becoming material change the candidates (a host stage) but not the pocket's identity: it is
    checked at once, with its new omega."""
    pair = Pair()
    pair.step(pair.slab, 0.0)
    sealed = shell(pair.slab, 3, 6)
    pair.step(sealed, 0.1)
    pair.step(sealed, 0.2)
    shrunk = sealed.copy()
    shrunk[2, 2, 4] = True
    out, _, _ = pair.step(shrunk, 0.3)
    assert out["pockets"] == 1 and out["topology"]["host_stage"]
    assert out["omega_checked_max_rad_s"] == pytest.approx(pocket_omega(pair.device), rel=1e-12)


def test_checkpoint_round_trip_restores_codes_and_gas():
    pair = Pair()
    pair.step(pair.slab, 0.0)
    pair.step(shell(pair.slab, 3, 6), 0.1)
    arrays = {name: pair.device.state_array(name) for name in pair.device.state_array_names()}
    host = pair.device.state_host()
    other = Pair()
    other.device.load_state_arrays(arrays)
    other.device.load_state_host(host)
    assert np.array_equal(other.device.codes(), pair.device.codes())
    assert other.device.pockets == 1 and np.array_equal(other.chamber.box_code.to_numpy(),
                                                        pair.chamber.box_code.to_numpy())


def test_transfers_scale_with_the_counts():
    """Default staging (65536 cells) and tables (4096 pockets): a refresh without flips reads the two statistics only;
    the seal of an 8-cell pocket moves the smallest ladder fields (256 cells, 64 pockets), not the whole buffers."""
    pair = Pair(max_pockets=4096, flood_batch=16, staging=65536)
    pair.step(pair.slab, 0.0)
    sealed = shell(pair.slab, 3, 6)
    out, c, ref = pair.step(sealed, 0.1)
    pair.compare(c, ref)
    x = out["transfers"]
    assert out["topology"]["host_stage"] and x["writes"] == 2           # the candidates' codes and the new gas
    assert x["write_bytes"] == STAGE_ROWS * 256 * 4 + 5 * 64 * 8         # index, code, region rows
    assert x["read_bytes"] < 8 * 1024
    out, c, ref = pair.step(sealed, 0.2)                                # the same mask: no flip
    pair.compare(c, ref)
    assert out["topology"] is None
    assert out["transfers"] == dict(reads=2, read_bytes=9 * 8 + STATS_SLOTS * 8, writes=0, write_bytes=0)


def test_unchanged_pocket_set_skips_the_host_stage():
    """A flip in the inlet air far from the pocket changes the topology but not the sealed cells: no host stage, the
    pocket keeps its code and gas, and everything equals the reference path."""
    pair = Pair()
    pair.step(pair.slab, 0.0)
    sealed = shell(pair.slab, 3, 6)
    pair.step(sealed, 0.1)
    stages, q = pair.device.host_stages, pair.device.pocket_table()["q_pa_m3"].copy()
    moved = sealed.copy()
    moved[5, 5, 5] = True                                               # inlet air, not next to the pocket
    out, c, ref = pair.step(moved, 0.2)
    pair.compare(c, ref)
    assert out["topology"] is not None and not out["topology"]["host_stage"]
    assert pair.device.host_stages == stages and pair.device.pocket_set_unchanged == 1
    assert np.array_equal(pair.device.pocket_table()["q_pa_m3"], q)
    assert out["transfers"]["writes"] == 0
    # a cell of the pocket turning into material changes the sealed cell set: host stage
    shrunk = moved.copy()
    shrunk[2, 2, 4] = True
    out, c, ref = pair.step(shrunk, 0.3)
    pair.compare(c, ref)
    assert out["topology"]["host_stage"] and pair.device.host_stages == stages + 1


def test_floor_material_is_counted_by_the_classification():
    pair = Pair()
    pair.step(pair.slab, 0.0)
    floor = pair.slab.copy()
    floor[0, 0, 0] = True
    floor[1, 0, 0] = True
    pair.occ.set(floor)
    out = pair.device.refresh(0.1, P_IN)          # the device path only: the reference stops on its own floor check
    assert out["topology"]["floor_material_cells"] == 2


def test_noise_events_record_the_pocket_volume_and_time():
    """The noise trigger: a pocket >= 5 mm^3 whose pressure moves by > 1 kPa between refreshes without a topology change is
    an event (count, |dp|, pocket, volume); the seal itself (a topology change) is none."""
    pair = Pair()
    pair.step(pair.slab, 0.0)
    sealed = shell(pair.slab, 3, 6)
    out, _, _ = pair.step(sealed, 0.1)
    assert out["noise_events"] == 0 and out["noise_event"] is None
    p_seal = P_ATMOSPHERE_PA + P_IN
    cover = np.where(sealed, 1.0, 0.0)
    cover[2:4, 2:4, 4:6] = 0.125                                    # 8 -> 7 mm^3: p rises by p_seal / 7
    out, c, ref = pair.step(sealed, 0.2, coverage=cover)
    pair.compare(c, ref)
    event = out["noise_event"]
    assert out["noise_events"] == 1 and event["count"] == 1 and event["pocket"] == 0 and event["t_s"] == 0.2
    assert event["volume_m3"] == pytest.approx(7.0e-9, rel=1e-6)
    assert event["abs_dp_pa"] == pytest.approx(p_seal / 0.875 - p_seal, rel=1e-9)
    out, _, _ = pair.step(sealed, 0.3, coverage=cover)              # unchanged: no event
    assert out["noise_events"] == 0 and out["noise_event"] is None


def test_restart_while_covered_keeps_the_cover():
    """The cover is device state: a restart while the hole is covered (the inlet cell turned material, all air sealed)
    and a first refresh without a flip still report the hole covered, with the same codes and gas."""
    pair = Pair()
    pair.step(pair.slab, 0.0)
    covered = pair.slab.copy()
    covered[0, 0, 7] = True                                            # the only inlet cell
    out, _, _ = pair.step(covered, 0.1)
    assert out["inlet_covered"] and out["inlet_air_cells"] == 0 and out["pockets"] == 1
    arrays = {name: pair.device.state_array(name) for name in pair.device.state_array_names()}
    host = pair.device.state_host()
    assert "host_stage_s" not in host
    other = Pair()
    other.device.load_state_arrays(arrays)
    other.device.load_state_host(host)
    other.occ.previous = covered.copy()                                # the restored topology: no flip
    other.occ.set(covered)
    again = other.device.refresh(0.2, P_IN)
    assert again["topology"] is None and again["inlet_covered"] and again["inlet_air_cells"] == 0
    assert np.array_equal(other.device.codes(), pair.device.codes())
    assert np.array_equal(other.device.pocket_table()["q_pa_m3"], pair.device.pocket_table()["q_pa_m3"])
