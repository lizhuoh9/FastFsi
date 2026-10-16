"""The sealing rule of DeviceGasRegions (boundary/gas_pocket_transitions + gas_regions_device): a region cut off from
the inlet gas seals only after N_SEAL occupancy updates with at least M_MIN cells; until then it is inlet gas (code 1,
Dirichlet at p_in) and a pending candidate region.  Synthetic topology masks in a 6 x 6 x 8 box of 1 mm^3 cells
(gas_sealing_rig.py): a membrane slab at k = 3 (inlet gas above, liquid below), the inlet at (0, 0, 7).
  - a 1-cell candidate covered by material after 2 updates: no pocket, no hard stop, an "ended covered" event;
  - a 2 x 2 x 2 candidate persisting 5 updates: sealed at the 5th (a forced topology stage: the mask did not change)
    with q = (p_in of the seal update + p_atm) V; until then its cells are inlet gas;
  - the same candidate reconnecting to the inlet gas at its 3rd update: never sealed, "ended reconnected";
  - a 7-cell candidate persisting 20 updates: never sealed, stays inlet gas at p_in;
  - the earlier rule (seal_steps = min_cells = 1) on the 1-cell candidate is the hard stop gas_pocket_zero_volume;
  - a checkpoint round trip while a region is pending: the restored run seals at the same update with the same q;
  - raster interval 2: only every 2nd refresh rasterises and ages the region, so it seals 10 refreshes after the change;
  - the defaults with two regions (one seals, one is covered) and the sealed pocket reconnecting: gas conservation and
    the region counters at every refresh.  The host reference path (GasRegions + GasPocketModel) implements the earlier
    rule only, so this is an invariant test, not a device-vs-host comparison."""
import copy

import numpy as np
import pytest
import taichi as ti

from fastfsi.boundary.gas_pocket_transitions import M_MIN, N_SEAL
from fastfsi.boundary.gas_pockets import P_ATMOSPHERE_PA
from fastfsi.boundary.gas_regions_device import DeviceGasRegions
from fastfsi.errors import SimulationHardStop
from gas_sealing_rig import BLOCK, CORNER, ONE, SEVEN, V, Rig, enclose, p_in, scenario_one


@pytest.fixture(scope="module", autouse=True)
def cpu():
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1, opt_level=0, offline_cache=False)


def rig(**options):
    return Rig(DeviceGasRegions, **options)


def block_cells(device):
    return device.codes()[tuple(np.array(BLOCK).T)].copy()


def run_block(r, updates, *, reconnect_at=None, codes=None):
    r.step(r.slab, 0.0, p_in(0))
    sealed = enclose(r.slab, BLOCK)
    outs = []
    for n in range(1, updates + 1):
        mask = sealed
        if reconnect_at is not None and n >= reconnect_at:
            mask = sealed.copy()
            mask[1, 2, 5] = False                                    # a hole in the shell: the block meets the gas
        outs.append(r.step(mask, 0.1 * n, p_in(n)))
        if codes is not None:
            codes.append(block_cells(r.device))
    return outs


def test_defaults_are_five_updates_and_eight_cells():
    assert (N_SEAL, M_MIN) == (5, 8)
    r = rig()
    assert (r.device.seal_steps, r.device.min_cells, r.device.raster_interval) == (N_SEAL, M_MIN, 1)


def test_one_cell_candidate_covered_ends_without_a_pocket():
    r = rig()
    out1, out2, out3 = scenario_one(r)
    assert out1["pockets"] == 0 and out1["candidates_pending"] == 1 and out1["candidate_cells_pending"] == 1
    assert out2["candidates_pending"] == 1
    assert out3["pockets"] == 0 and out3["candidates_pending"] == 0
    c = out3["candidate_counters"]
    assert (c["created"], c["sealed"], c["covered"], c["reconnected"], c["merged"]) == (1, 0, 1, 0, 0)
    ended = [e for e in r.events if e["kind"] == "ended"]
    assert len(ended) == 1 and ended[0]["how"] == "covered" and ended[0]["cells"] == 1 and ended[0]["age"] == 2
    assert ended[0]["free_volume_m3"] == pytest.approx(V, rel=1e-12)
    assert ended[0]["fraction_of_inlet_air"] == pytest.approx(V / r.device.last_air_volume_m3, rel=0.05)
    assert c["ended_volume_m3"] == pytest.approx(V, rel=1e-12)


def test_pending_cell_is_inlet_gas_at_p_in():
    r = rig()
    r.step(r.slab, 0.0, p_in(0))
    out = r.step(enclose(r.slab, ONE), 0.1, p_in(1))
    assert out["candidates_pending"] == 1
    assert r.device.codes()[ONE[0]] == 1 and r.air_cell.to_numpy()[ONE[0]] == 1
    assert r.chamber.box_code.to_numpy()[ONE[0]] == 1                # the chamber's code 1: Dirichlet at p_in


def test_eight_cells_persisting_five_updates_seal_at_the_seal_time_pressure():
    r = rig()
    codes = []
    outs = run_block(r, 5, codes=codes)
    for n, (out, block) in enumerate(zip(outs[:4], codes[:4]), start=1):
        assert out["pockets"] == 0 and out["candidates_pending"] == 1, n
        assert np.all(block == 1), n                                 # inlet gas until it seals
    assert outs[3]["topology"] is None                               # update 4: no flip, not yet of age
    last = outs[4]
    assert last["topology"] is not None and last["pockets"] == 1     # update 5: the forced stage seals it
    assert np.all(codes[4] == 4)
    q = r.device.pocket_table()["q_pa_m3"][0]
    assert q == pytest.approx((p_in(5) + P_ATMOSPHERE_PA) * 8 * V, rel=1e-12)
    sealed = [e for e in r.events if e["kind"] == "sealed"]
    assert len(sealed) == 1 and sealed[0]["age"] == N_SEAL and sealed[0]["cells"] == 8
    assert last["candidate_counters"]["sealed"] == 1 and last["candidates_pending"] == 0


def test_reconnecting_before_five_updates_never_seals():
    r = rig()
    outs = run_block(r, 8, reconnect_at=3)
    assert all(o["pockets"] == 0 for o in outs)
    c = outs[-1]["candidate_counters"]
    assert (c["created"], c["sealed"], c["reconnected"]) == (1, 0, 1) and outs[-1]["candidates_pending"] == 0
    assert np.all(block_cells(r.device) == 1)


def test_seven_cells_never_seal_and_stay_inlet_gas():
    r = rig()
    r.step(r.slab, 0.0, p_in(0))
    sealed = enclose(r.slab, SEVEN)
    for n in range(1, 21):
        out = r.step(sealed, 0.1 * n, p_in(n))
        assert out["pockets"] == 0 and out["candidates_pending"] == 1
    cells = tuple(np.array(SEVEN).T)
    assert np.all(r.device.codes()[cells] == 1) and np.all(r.air_cell.to_numpy()[cells] == 1)
    assert np.all(r.chamber.box_code.to_numpy()[cells] == 1)
    assert r.device.pending_regions[0]["age"] == 20 and out["candidate_counters"]["age_max"] == 20
    assert out["candidate_counters"]["sealed"] == 0 and out["candidate_cells_pending"] == 7


def test_earlier_rule_stops_on_the_one_cell_candidate_the_sealing_rule_does_not():
    with pytest.raises(SimulationHardStop) as stop:
        scenario_one(rig(seal_steps=1, min_cells=1))
    assert stop.value.code == "gas_pocket_zero_volume"
    scenario_one(rig())                                              # the sealing rule: no stop


def test_checkpoint_round_trip_while_a_region_is_pending():
    r = rig()
    outs = run_block(r, 3)
    assert outs[-1]["candidates_pending"] == 1
    arrays = {name: r.device.state_array(name) for name in r.device.state_array_names()}
    host = copy.deepcopy(r.device.state_host())                      # a checkpoint holds a copy, never the live state
    other = rig()
    other.device.load_state_arrays(arrays)
    other.device.load_state_host(host)
    sealed = enclose(r.slab, BLOCK)
    other.occ.at_update = sealed.copy()                              # the restored run continues the same masks
    for n in (4, 5):
        a = r.step(sealed, 0.1 * n, p_in(n))
        b = other.step(sealed, 0.1 * n, p_in(n))
        assert a["pockets"] == b["pockets"] and np.array_equal(r.device.codes(), other.device.codes()), n
    assert b["pockets"] == 1
    assert other.device.pocket_table()["q_pa_m3"][0] == r.device.pocket_table()["q_pa_m3"][0]


def test_real_checkpoint_round_trip_while_a_region_is_pending(tmp_path):
    """The same continuation through fastfsi.io.checkpoint (NPZ arrays + the encoded host JSON), not an in-memory copy:
    the pending region, its counters and the seal update and gas survive the file."""
    from fastfsi.io.checkpoint import load_components, read_checkpoint, write_checkpoint
    r = rig()
    outs = run_block(r, 3)
    assert outs[-1]["candidates_pending"] == 1
    path = tmp_path / "pending.npz"
    write_checkpoint(path, dict(regions=r.device), dict(refresh=3), dict(rig="sealing"))
    arrays, host, loop_state, manifest, _ = read_checkpoint(path)
    other = rig()
    load_components(dict(regions=other.device), arrays, host)
    assert loop_state == dict(refresh=3) and manifest == dict(rig="sealing")
    assert other.device.pending_regions == r.device.pending_regions
    assert other.device.candidate_counters == r.device.candidate_counters
    sealed = enclose(r.slab, BLOCK)
    other.occ.at_update = sealed.copy()
    for n in (4, 5):
        a = r.step(sealed, 0.1 * n, p_in(n))
        b = other.step(sealed, 0.1 * n, p_in(n))
        assert a["pockets"] == b["pockets"] and np.array_equal(r.device.codes(), other.device.codes()), n
    assert b["pockets"] == 1
    assert other.device.pocket_table()["q_pa_m3"][0] == r.device.pocket_table()["q_pa_m3"][0]


def test_loading_impossible_regions_is_refused():
    """load_state_host checks the region bounds: more pending regions than the region tables hold, or a cell naming a
    region beyond the pending ones, raise ValueError (the fate and volume kernels would index out of bounds)."""
    r = rig()
    run_block(r, 3)
    arrays = {name: r.device.state_array(name) for name in r.device.state_array_names()}
    host = copy.deepcopy(r.device.state_host())
    assert len(host["pending_regions"]) == 1
    too_many = copy.deepcopy(host)
    too_many["pending_regions"] = [dict(host["pending_regions"][0]) for _ in range(r.device.max_pockets + 1)]
    other = rig()
    other.device.load_state_arrays(arrays)
    with pytest.raises(ValueError, match="pending regions"):
        other.device.load_state_host(too_many)
    beyond = dict(arrays)
    beyond["cand_id"] = arrays["cand_id"].copy()
    beyond["cand_id"][tuple(BLOCK[0])] = 2                           # region 2 while only 1 is pending
    third = rig()
    third.device.load_state_arrays(beyond)
    with pytest.raises(ValueError, match="cand_id"):
        third.device.load_state_host(host)


def test_raster_interval_two_ages_on_rasterised_refreshes_only():
    r = rig(raster_interval=2)
    r.step(r.slab, 0.0, p_in(0))                                     # refresh 1 rasterises (the first)
    sealed = enclose(r.slab, BLOCK)
    pockets = [r.step(sealed, 0.1 * n, p_in(n))["pockets"] for n in range(1, 11)]
    # skipped at n 1, 3, 5, 7, 9; the region is seen at n 2 and ages at n 2, 4, 6, 8, 10: of age at n 10
    assert pockets == [0] * 9 + [1], pockets
    assert r.device.raster_skips == 5 and r.occ.updates == 6


def test_defaults_conserve_gas_and_count_regions():
    """BLOCK (8 cells) and CORNER (1 cell) cut off at refresh 1; BLOCK seals at refresh 5; CORNER is covered at refresh
    8; BLOCK's pocket reconnects at refresh 10.  No region splits or merges, so every created region is pending,
    sealed or ended."""
    r = rig()
    assert (r.device.seal_steps, r.device.min_cells) == (N_SEAL, M_MIN)
    shells = enclose(enclose(r.slab, BLOCK), CORNER)
    covered = shells.copy()
    covered[CORNER[0]] = True
    opened = covered.copy()
    opened[1, 2, 5] = False                                          # a hole in BLOCK's shell
    masks = [r.slab] + [shells] * 7 + [covered, covered, opened]
    pockets, q_seal = [], None
    for n, mask in enumerate(masks):
        out = r.step(mask, 0.1 * n, p_in(n))
        table = r.device.pocket_table()
        pockets.append(out["pockets"])
        # gas: the pockets hold exactly the net gas taken from the inlet reservoir (q is conserved between topology
        # changes; a seal takes (p_in + p_atm) V, a reconnection returns the pocket's q)
        assert float(table["q_pa_m3"].sum()) == pytest.approx(r.device.exchange_pa_m3, rel=1e-12, abs=1e-18), n
        c = out["candidate_counters"]
        assert c["created"] == c["sealed"] + c["covered"] + c["reconnected"] + c["merged"] + out["candidates_pending"], n
        pending = r.device.cand_id.to_numpy() > 0
        assert np.all(r.device.codes()[pending] == 1), n             # pending cells are inlet gas
        assert int(pending.sum()) == out["candidate_cells_pending"], n
        if n == 5:
            assert table["volume_m3"][0] == pytest.approx(8 * V, rel=1e-12)
            q_seal = table["q_pa_m3"][0]
            assert q_seal == pytest.approx((p_in(5) + P_ATMOSPHERE_PA) * 8 * V, rel=1e-12)
        elif n in (6, 7):
            assert table["q_pa_m3"][0] == q_seal                     # no topology stage: the same gas, bit for bit
        elif n in (8, 9):                                            # CORNER's stage re-plans BLOCK: q V / V
            assert table["q_pa_m3"][0] == pytest.approx(q_seal, rel=1e-14)
    assert pockets == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], pockets
    c = out["candidate_counters"]
    assert (c["created"], c["sealed"], c["covered"], c["reconnected"], c["merged"]) == (2, 1, 1, 0, 0)
    assert out["candidates_pending"] == 0 and r.device.exchange_pa_m3 == pytest.approx(0.0, abs=1e-15)
    for e in (e for e in r.events if e["kind"] == "sealed"):
        assert e["age"] >= N_SEAL and e["cells"] >= M_MIN
    ended = [e for e in r.events if e["kind"] == "ended"]
    assert len(ended) == 1 and ended[0]["how"] == "covered" and ended[0]["cells"] == 1
    assert ended[0]["free_volume_m3"] == pytest.approx(V, rel=1e-12)


# ------------------------------------------------------------------------------------------------- type memory
# The local configuration of a run's stop (a gas-side void at a wall, enclosed by a flickering rasterised surface for one
# update, then joined to the gas by its neighbour's release): W is the void, N the neighbour released one update later,
# G the gas cell above N.
W, N, G = (3, 3, 5), (4, 3, 5), (4, 3, 6)


def gas_side_void(r):
    """Masks: the slab (W is inlet gas); W and its face neighbours material; W freed while enclosed; N freed."""
    m0 = r.slab
    m1 = enclose(r.slab, [W])
    m1[W] = True
    m2 = m1.copy()
    m2[W] = False
    m3 = m2.copy()
    m3[N] = False
    return m0, m1, m2, m3


def test_a_gas_side_void_freed_while_enclosed_rejoins_the_gas():
    r = rig()
    m0, m1, m2, m3 = gas_side_void(r)
    r.step(m0, 0.0, p_in(0))
    assert r.device.last_free.to_numpy()[W] == 1                     # its last free code: inlet gas
    r.step(m1, 0.1, p_in(1))
    assert r.device.codes()[W] == 0 and r.device.last_free.to_numpy()[W] == 1
    out2 = r.step(m2, 0.2, p_in(2))
    assert r.device.codes()[W] == 1 and r.device.cand_id.to_numpy()[W] > 0       # a pending region of the inlet gas
    assert out2["pockets"] == 0 and out2["candidates_pending"] == 1
    assert out2["candidate_counters"]["history_gas"] == 1
    out3 = r.step(m3, 0.3, p_in(3))                                  # N's release joins W to the gas: no stop
    assert r.device.codes()[W] == 1 and out3["pockets"] == 0 and out3["candidates_pending"] == 0
    c = out3["candidate_counters"]
    assert (c["created"], c["sealed"], c["reconnected"]) == (1, 0, 1)
    assert (c["history_gas"], c["history_liquid"], c["history_mixed"], c["history_none"]) == (1, 0, 0, 0)
    created, = [e for e in r.events if e["kind"] == "created"]
    history, = [e for e in r.events if e["kind"] == "history"]
    ended, = [e for e in r.events if e["kind"] == "ended"]
    assert created["typed_by_history"] and ended["typed_by_history"] and ended["how"] == "reconnected"
    assert (history["how"], history["typed_as"], history["cells"], history["census_id"]) == (
        "gas", "inlet gas", 1, created["census_id"])
    assert history["first_box_index"] == list(W)


def test_the_gas_side_void_stops_the_run_without_type_memory():
    r = rig(type_memory=False)
    m0, m1, m2, m3 = gas_side_void(r)
    r.step(m0, 0.0, p_in(0))
    r.step(m1, 0.1, p_in(1))
    out2 = r.step(m2, 0.2, p_in(2))
    assert r.device.codes()[W] == 3                                  # the old rule: freed material is a water pocket
    assert out2["candidate_counters"]["history_none"] == 1           # no history without the memory
    with pytest.raises(SimulationHardStop) as stop:
        r.step(m3, 0.3, p_in(3))
    assert stop.value.code == "gas_region_liquid_to_gas"


L, S = (3, 3, 2), (3, 3, 3)          # a liquid cell under the slab, and the slab cell above it


def liquid_void(r):
    """Masks: the slab (L is liquid); L and its face neighbours material; L freed while enclosed; S freed, so the gas
    above the slab reaches L."""
    m0 = r.slab
    m1 = enclose(r.slab, [L])
    m1[L] = True
    m2 = m1.copy()
    m2[L] = False
    m3 = m2.copy()
    m3[S] = False
    return m0, m1, m2, m3


def test_a_freed_void_with_a_liquid_history_is_a_water_pocket_and_a_true_flip_still_stops():
    r = rig()
    m0, m1, m2, m3 = liquid_void(r)
    r.step(m0, 0.0, p_in(0))
    assert r.device.last_free.to_numpy()[L] == 2
    r.step(m1, 0.1, p_in(1))
    out2 = r.step(m2, 0.2, p_in(2))
    assert r.device.codes()[L] == 3                                  # liquid history: a water pocket, as before
    assert out2["candidate_counters"]["history_liquid"] == 1
    history, = [e for e in r.events if e["kind"] == "history"]
    assert (history["how"], history["typed_as"], history["cells"]) == ("liquid", "water pocket", 1)
    with pytest.raises(SimulationHardStop) as stop:                  # gas reaches the water: still the hard stop
        r.step(m3, 0.3, p_in(3))
    assert stop.value.code == "gas_region_liquid_to_gas"


A = (3, 3, 4)                        # the gas cell above the slab cell S
COLUMN = [A, S, L]


def test_freed_only_mixed_histories_fall_back_to_a_water_pocket_and_are_counted():
    """A gas-history cell above the slab, the slab cell (never free) and a liquid-history cell below, freed together and
    enclosed: a region of freed cells alone with mixed histories keeps the old rule (a water pocket, no stop), counted
    as history_mixed with a "history" event; without the memory it is a region without history (history_none)."""
    for memory in (True, False):
        r = rig(type_memory=memory)
        r.step(r.slab, 0.0, p_in(0))
        closed = enclose(r.slab, COLUMN)
        for cell in COLUMN:
            closed[cell] = True
        r.step(closed, 0.1, p_in(1))
        opened = closed.copy()
        for cell in COLUMN:
            opened[cell] = False
        out = r.step(opened, 0.2, p_in(2))
        assert all(r.device.codes()[cell] == 3 for cell in COLUMN), memory
        c = out["candidate_counters"]
        assert (c["history_mixed"], c["history_none"]) == ((1, 0) if memory else (0, 1)), memory
        assert [e["how"] for e in r.events if e["kind"] == "history"] == (["mixed"] if memory else []), memory


def test_a_real_air_parent_decides_over_a_freed_liquid_history():
    """A gas cell enclosed alone (a pending region of the inlet gas), then the slab cell below it and a liquid-history
    cell below that freed into the same region: its real parent (the inlet gas) decides, the freed cells' histories
    are ignored -- the region stays a pending inlet-gas region (the same census id), not gas_pocket_origin_ambiguous,
    and no freed-only region is counted."""
    r = rig()
    r.step(r.slab, 0.0, p_in(0))
    m1 = enclose(r.slab, COLUMN)
    m1[L] = True                                                     # L covered: its liquid history in last_free
    out1 = r.step(m1, 0.1, p_in(1))
    assert r.device.codes()[A] == 1 and out1["candidates_pending"] == 1
    m2 = m1.copy()
    m2[S] = m2[L] = False
    out2 = r.step(m2, 0.2, p_in(2))
    assert all(r.device.codes()[cell] == 1 for cell in COLUMN)
    assert out2["candidates_pending"] == 1 and out2["candidate_cells_pending"] == 3 and out2["pockets"] == 0
    c = out2["candidate_counters"]
    assert c["created"] == 1 and sum(c["history_" + k] for k in ("gas", "liquid", "mixed", "none")) == 0
    assert r.device.pending_regions[0]["age"] == 2 and not r.device.pending_regions[0]["typed_by_history"]


def test_the_plan_types_only_freed_only_components_by_history():
    """plan_pocket_transitions on a row of cells (shape 1 x 1 x 20; components are runs of indices): real parents
    decide, history types freed-only components only, cells without a history do not vote; with the earlier rule a
    gas-history cell seals at the inlet pressure (counted as inlet gas)."""
    from fastfsi.boundary.gas_pocket_transitions import plan_pocket_transitions
    shape = (1, 1, 20)
    runs = [([0, 1], [1, -2]),        # a real air parent and a liquid history: a pending region (not ambiguous)
            ([3], [-1]),              # gas history: a pending region
            ([5], [-2]),              # liquid history: a water pocket
            ([7, 8], [-1, -3]),       # mixed: a water pocket
            ([10], [0]),              # no history: a water pocket
            ([12, 13], [-1, 0]),      # gas history and a cell without one: a pending region
            ([15, 16], [2, -1])]      # a real liquid parent and a gas history: a water pocket (not ambiguous)
    idx = [i for cells, _ in runs for i in cells]
    prev = [p for _, codes in runs for p in codes]
    none = np.zeros((0, 4), dtype=np.int64)
    plan = plan_pocket_transitions(idx, prev, none, 0, shape, t_s=0.0, seal_steps=N_SEAL, min_cells=M_MIN)
    assert plan.candidate_codes.tolist() == [1, 1, 1, 3, 3, 3, 3, 1, 1, 3, 3]
    assert [(h["how"], h["cells"]) for h in plan.history] == [("gas", 1), ("liquid", 1), ("mixed", 2), ("none", 1),
                                                              ("gas", 2)]
    assert [(h["region"], h["pocket"]) for h in plan.history] == [(2, -1), (0, -1), (0, -1), (0, -1), (3, -1)]
    assert [r["history"] for r in plan.regions] == [False, True, True] and plan.pockets == 0
    at_once = plan_pocket_transitions([3], [-1], none, 0, shape, t_s=0.0)          # the earlier rule (1, 1)
    assert (at_once.pockets, at_once.candidate_codes.tolist(), at_once.air_cells_of_new.tolist()) == (1, [4], [1])
    assert at_once.sealed[0]["history"] and at_once.history[0]["pocket"] == 0
    with pytest.raises(SimulationHardStop) as stop:                                 # real parents: unchanged
        plan_pocket_transitions([0, 1], [1, 2], none, 0, shape, t_s=0.0)
    assert stop.value.code == "gas_pocket_origin_ambiguous"
    with pytest.raises(ValueError):
        plan_pocket_transitions([0], [-4], none, 0, shape, t_s=0.0)


def test_type_memory_survives_a_checkpoint_round_trip():
    r = rig()
    m0, m1, m2, m3 = gas_side_void(r)
    r.step(m0, 0.0, p_in(0))
    r.step(m1, 0.1, p_in(1))                                         # W covered, its gas history only in last_free
    arrays = {name: r.device.state_array(name) for name in r.device.state_array_names()}
    assert "last_free" in arrays
    host = copy.deepcopy(r.device.state_host())
    other = rig()
    other.device.load_state_arrays(arrays)
    other.device.load_state_host(host)
    other.occ.at_update = m1.copy()
    for n, mask in ((2, m2), (3, m3)):
        a = r.step(mask, 0.1 * n, p_in(n))
        b = other.step(mask, 0.1 * n, p_in(n))
        assert np.array_equal(r.device.codes(), other.device.codes()), n
        assert np.array_equal(r.device.last_free.to_numpy(), other.device.last_free.to_numpy()), n
        assert a["candidate_counters"] == b["candidate_counters"], n
    assert b["candidate_counters"]["reconnected"] == 1 and b["candidate_counters"]["history_gas"] == 1


def test_a_state_saved_without_history_counters_loads_them_at_zero():
    r = rig()
    r.step(r.slab, 0.0, p_in(0))
    arrays = {name: r.device.state_array(name) for name in r.device.state_array_names()}
    host = copy.deepcopy(r.device.state_host())
    for key in ("history_gas", "history_liquid", "history_mixed", "history_none"):
        del host["candidate_counters"][key]                          # a state of the code before the counters
    other = rig()
    other.device.load_state_arrays(arrays)
    other.device.load_state_host(host)
    assert other.device.candidate_counters == r.device.candidate_counters
