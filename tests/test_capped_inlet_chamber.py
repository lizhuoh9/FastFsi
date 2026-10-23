"""The capped inlet in PneumaticChamber on the real solver (the duct rig of test_pneumatic_chamber: a membrane slab
under the lid covers the 2 mm inlet, moves 1.5 mm down, and the inlet reconnects), Root 2026-10-23, stage 1:
(c) an unlimited hole (a = b = 0, no cap) reproduces the ideal inlet's regions and pressure to round-off; the ideal
    inlet's report and summary carry no capped keys;
the region's gas spring: area and footprint mass > 0 once the inlet is open, omega = sqrt(p A^2 / (V m)) and
    dt_limit() = min(the pocket bound, 0.9 x 0.5 / omega);
the covered start with no inlet air falls back to the line pressure (counted), and the state survives a checkpoint (with
    a connector's bore before the lid hole: the chamber composes the series line; line elements need the capped
    inlet)."""
import json
import math

import numpy as np
import pytest

from fastfsi.boundary.inlet_air import FlowElement, SeriesRestriction
from test_pneumatic_chamber import build, move

CONNECTOR = FlowElement("connector bore", "fitting", 1.5e-3)

STEP = 1.0e-4


def line(t_s):
    return 4.0e3 * math.sin(2.0 * math.pi * 50.0 * t_s)


def run(s, accepts=8):
    """Accept 0 at rest (inlet covered), then the slab 1.5 mm down; per accept: covered, pockets, air cells, codes and
    the step's inlet pressure (inlet_gauge_pa at the next midpoint)."""
    points, chamber, positions = s["points"], s["chamber"], s["positions"]
    out = []
    for n in range(accepts):
        move(points, positions, 0.0 if n == 0 else -1.5e-3)
        t = STEP * n
        r = chamber.accept(t, line(t), dt_taken_s=None if n == 0 else STEP)
        out.append((r["inlet_covered"], r["pockets"], chamber.air_cell.to_numpy().copy(),
                    chamber.regions.codes().copy(), chamber.inlet_gauge_pa(t + 0.5 * STEP, line), r))
    return out


def test_c_an_unlimited_hole_reproduces_the_ideal_inlet_and_the_ideal_report_is_unchanged():
    ideal = run(build())
    capped = run(build(inlet_model="capped", inlet_hole_coefficients=(0.0, 0.0), inlet_flow_cap_m3_s=math.inf))
    assert [x[0] for x in ideal] != [ideal[0][0]] * len(ideal)            # the inlet reconnects in the window
    for a, b, n in zip(ideal, capped, range(len(ideal))):
        assert a[0] == b[0] and a[1] == b[1]
        assert np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3])
        if not b[0]:                                                       # open: the chamber sits at the line
            assert b[5]["p_chamber_gauge_pa"] == pytest.approx(line(STEP * n), abs=1e-6)
        assert a[4] == line(STEP * n + 0.5 * STEP)                         # ideal: the line pressure itself
    chamber = build()["chamber"]
    assert not any(k.startswith(("p_chamber", "inlet_q", "inlet_air_", "inlet_omega")) for k in chamber.report())
    assert "inlet_air" not in chamber.run_summary() and "inlet_air" not in chamber.state_host()


def test_the_inlet_region_is_a_gas_spring_in_dt_limit():
    s = build(inlet_model="capped", inlet_hole_length_m=7.0e-3, inlet_flow_cap_m3_s=8.0e-3 / 60.0)
    out = run(s)
    chamber = s["chamber"]
    first = out[0][5]                                     # the start: covered, no inlet air, at the line pressure
    assert first["inlet_covered"] and first["inlet_air_volume_m3"] == 0.0 and first["inlet_fallbacks"] == 0
    assert first["p_chamber_gauge_pa"] == line(0.0) and first["inlet_omega_rad_s"] == 0.0
    assert out[1][5]["inlet_covered"] and out[1][5]["inlet_fallback"]      # covered with no volume: the fallback
    open_ = [x[5] for x in out if not x[0]]
    assert open_, "the inlet must reconnect in the window"
    r = open_[-1]
    assert r["inlet_area_m2"] > 0.0 and r["inlet_column_mass_kg"] > 0.0 and r["inlet_air_volume_m3"] > 0.0
    omega = math.sqrt((r["p_chamber_gauge_pa"] + 101325.0) * r["inlet_area_m2"] ** 2
                      / (r["inlet_air_volume_m3"] * r["inlet_column_mass_kg"]))
    assert r["inlet_omega_rad_s"] == pytest.approx(omega, rel=1e-12)
    assert chamber.dt_limit() == pytest.approx(min(chamber.regions.dt_limit(), 0.45 / omega), rel=1e-12)
    assert r["inlet_dt_limit_s"] == pytest.approx(0.45 / omega, rel=1e-12)
    assert abs(r["inlet_ledger_residual_pa_m3"]) < 1e-15
    summary = chamber.run_summary()["inlet_air"]
    assert summary["kind"] == "capped" and summary["hole_length_m"] == 7.0e-3
    assert summary["fallbacks"] >= 1                                       # the covered accepts before reconnection


def test_the_capped_state_survives_a_checkpoint(tmp_path):
    from fastfsi.io.checkpoint import load_components, read_checkpoint, write_checkpoint
    options = dict(inlet_model="capped", inlet_hole_length_m=7.0e-3, inlet_flow_cap_m3_s=8.0e-3 / 60.0,
                   inlet_line_elements=(CONNECTOR,))
    straight = build(**options)
    restriction = straight["chamber"].inlet_air.restriction
    composed = SeriesRestriction([CONNECTOR, FlowElement("lid hole", "hole", straight["chamber"].inlet_diameter_m,
                                                         7.0e-3)])
    assert [e["name"] for e in restriction.describe()["elements"]] == ["connector bore", "lid hole"]
    assert (restriction.a, restriction.b) == (composed.a, composed.b)
    with pytest.raises(ValueError):
        build(inlet_line_elements=(CONNECTOR,))                            # line elements need the capped inlet
    reference = run(straight, accepts=10)
    first = build(**options)
    run(first, accepts=6)
    host = first["chamber"].state_host()
    json.dumps(host, allow_nan=False)                                      # the checkpoint header's rule
    path = tmp_path / "capped.npz"
    write_checkpoint(path, dict(chamber=first["chamber"]), dict(accept=5), dict(rig="capped duct"))
    restored = build(**options)
    arrays, host, _, _, _ = read_checkpoint(path)
    load_components(dict(chamber=restored["chamber"]), arrays, host)
    points, chamber, positions = restored["points"], restored["chamber"], restored["positions"]
    for n in range(6, 10):
        move(points, positions, -1.5e-3)
        r = chamber.accept(STEP * n, line(STEP * n), dt_taken_s=STEP)
        ref = reference[n][5]
        for key in ("p_chamber_gauge_pa", "inlet_q_m3s", "inlet_air_mass_kg", "inlet_air_volume_m3",
                    "inlet_omega_rad_s"):
            assert r[key] == ref[key], key
    assert chamber.inlet_air.state_host() == straight["chamber"].inlet_air.state_host()
