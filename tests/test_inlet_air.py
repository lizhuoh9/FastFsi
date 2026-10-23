"""The capped inlet's lumped model (boundary/inlet_air.py), host only: the hole law from the geometry, its
backward-Euler step against the closed forms (with the density factor p_atm / p_mean), the sealed and ideal limits, the
cap, the mass ledger, the fallback, the guard and the checkpoint round trip (Root 2026-10-23: tests (b)-(f) of the
capped inlet, stage 1); the series line (Root 2026-10-23, the USER's layout: Y fitting -> hose -> straight connector ->
lid hole): the hole alone is the hole law bit for bit, the joints follow the area ratios, the hose is Poiseuille below
Re 2300 and Blasius above, and no element is no loss."""
import json
import math

import numpy as np
import pytest

from fastfsi.boundary.gas_pockets import P_ATMOSPHERE_PA
from fastfsi.boundary.inlet_air import (AIR_DENSITY_KG_M3, AIR_VISCOSITY_PA_S, SOUND_SPEED_M_S, FlowElement,
                                        InletAirModel, LPM_TO_M3_S, SeriesRestriction, hole_law,
                                        hole_pressure_drop_pa, solve_hole_flow)
from fastfsi.errors import SimulationHardStop

P0 = P_ATMOSPHERE_PA
LID_HOLE = hole_law(2.0e-3, 7.0e-3)                 # the robot's 2 mm hole through the 7 mm lid
HOLE = FlowElement("lid hole", "hole", 2.0e-3, 7.0e-3)
CAP = 8.0 * LPM_TO_M3_S


def test_f_the_quadratic_law_at_35_ml_s_gives_122_pa():
    a, b = LID_HOLE
    drop = hole_pressure_drop_pa(35.0e-6, a, b)
    assert drop == pytest.approx(122.0, abs=5.0)
    assert b * 35.0e-6 ** 2 > 0.85 * drop                         # the dynamic heads dominate (111 of 122 Pa)
    assert hole_pressure_drop_pa(-35.0e-6, a, b) == -drop         # odd in Q
    for q in (1.0e-7, 3.0e-5, CAP, 1.0e-3):                       # the inverse of the law
        assert solve_hole_flow(hole_pressure_drop_pa(q, a, b), a, b) == pytest.approx(q, rel=1e-12)


def linear_step(q_gas, dt, p_line, volume, k):
    """The linear law's backward-Euler step in closed form:
    q* = (q + p_atm dt K (p_line + p_atm)) / (1 + dt K p_atm / V)."""
    q_new = (q_gas + P0 * dt * k * (p_line + P0)) / (1.0 + dt * k * P0 / volume)
    return q_new, q_new / volume - P0, (q_new - q_gas) / (P0 * dt)


def density_scale(model, p_line):
    """The step's density factor p_atm / p_mean, p_mean = p_atm + (p_line + p_chamber) / 2 (the state's p_chamber)."""
    return P0 / (P0 + 0.5 * (p_line + model.p_gauge_pa))


def test_b_zero_reproduces_the_linear_law():
    a = LID_HOLE[0]
    model = InletAirModel("capped", hole_a_pa_s_m3=a, hole_b_pa_s2_m6=0.0, flow_cap_m3_s=math.inf)
    model.start(0.0, 2.0e-5)
    q = model.q_pa_m3
    rng = np.random.default_rng(1)
    for _ in range(200):
        dt, p_line, volume = 3.0e-5, float(rng.uniform(-8e3, 8e3)), float(rng.uniform(1e-6, 3e-5))
        scale = density_scale(model, p_line)
        p = model.advance(dt, p_line, volume, False, 0.0)
        q, p_ref, flow_ref = linear_step(q, dt, p_line, volume, 1.0 / (scale * a))
        assert model.q_pa_m3 == pytest.approx(q, rel=1e-13)
        assert p == pytest.approx(p_ref, abs=1e-8)
        assert model.q_m3_s == pytest.approx(flow_ref, rel=1e-9, abs=1e-16)
        assert model.loss_scale == scale


def test_the_quadratic_step_satisfies_the_hole_law_at_the_new_state():
    a, b = LID_HOLE
    model = InletAirModel("capped", restriction=SeriesRestriction([HOLE]), flow_cap_m3_s=math.inf)
    model.start(0.0, 2.0e-5)
    for p_line in (2.0e3, 9.0e3, -4.4e3, 0.0):
        scale = density_scale(model, p_line)                      # the gas at p_abs: the losses x p_atm / p_mean
        p = model.advance(3.0e-5, p_line, 2.0e-5, False, 0.0)
        assert p_line - p == pytest.approx(scale * hole_pressure_drop_pa(model.q_m3_s, a, b), rel=1e-9, abs=1e-9)
        assert model.mach == pytest.approx(abs(model.q_m3_s) * scale / (HOLE.area_m2 * SOUND_SPEED_M_S), rel=1e-14)
    assert model.mach_max == max(model.mach_max, model.mach) > 0.0


def test_b_a_sealed_chamber_under_a_prescribed_volume_follows_pv_const():
    model = InletAirModel("capped", hole_a_pa_s_m3=LID_HOLE[0], hole_b_pa_s2_m6=LID_HOLE[1], flow_cap_m3_s=CAP)
    model.start(3.0e3, 2.0e-5)
    pv = (3.0e3 + P0) * 2.0e-5
    for n in range(100):
        volume = 2.0e-5 * (1.0 + 0.3 * math.sin(0.1 * n))
        p = model.advance(3.0e-5, 9.0e3 * math.cos(0.2 * n), volume, True, 0.0)     # covered: no path, Q = 0
        assert model.q_m3_s == 0.0
        assert (p + P0) * volume == pytest.approx(pv, rel=1e-14)
    assert model.covered_steps == 100


def test_c_an_unlimited_hole_reproduces_the_ideal_inlet():
    model = InletAirModel("capped", hole_a_pa_s_m3=0.0, hole_b_pa_s2_m6=0.0, flow_cap_m3_s=math.inf)
    ideal = InletAirModel("ideal")
    model.start(0.0, 1.0e-5)
    rng = np.random.default_rng(2)
    for _ in range(300):
        p_line, volume = float(rng.uniform(-9e3, 9e3)), float(rng.uniform(1e-7, 4e-5))
        p = model.advance(float(rng.uniform(1e-6, 1e-4)), p_line, volume, False, float(rng.uniform(-1e-3, 1e-3)))
        assert p == pytest.approx(p_line, abs=1e-9 * P0)
        assert model.pressure_gauge_pa(-1.0) == p and ideal.pressure_gauge_pa(p_line) == p_line
        assert (p + P0) * volume == pytest.approx(model.q_pa_m3, rel=1e-12)
    assert ideal.advance(1e-4, 5.0, 1e-5, False, 0.0) == 5.0 and ideal.q_pa_m3 is None


def test_d_a_demand_above_the_cap_is_capped_and_the_chamber_leaves_the_line():
    model = InletAirModel("capped", hole_a_pa_s_m3=LID_HOLE[0], hole_b_pa_s2_m6=LID_HOLE[1], flow_cap_m3_s=CAP)
    model.start(0.0, 2.0e-5)
    volume = 2.0e-5
    pressures = [model.advance(3.0e-5, 20.0e3, volume, False, 0.0) for _ in range(40)]
    assert model.capped and model.q_m3_s == CAP and model.capped_steps == 40
    assert pressures[-1] < 20.0e3 - 1.0e3                         # the chamber lags the line
    expected = (P0 * volume + 40 * P0 * 3.0e-5 * CAP) / volume - P0          # the gas rises at the cap
    assert pressures[-1] == pytest.approx(expected, rel=1e-12)
    for _ in range(4000):                                         # then the line is reached (uncapped, Q -> 0)
        p = model.advance(3.0e-5, 20.0e3, volume, False, 0.0)
    assert not model.capped and p == pytest.approx(20.0e3, abs=1.0)
    model.advance(3.0e-5, -40.0e3, volume, False, 0.0)          # suction: capped the other way
    assert model.capped and model.q_m3_s == -CAP


def test_e_the_air_mass_is_conserved():
    model = InletAirModel("capped", hole_a_pa_s_m3=LID_HOLE[0], hole_b_pa_s2_m6=LID_HOLE[1], flow_cap_m3_s=CAP,
                          line_volume_m3=2.0e-6)
    model.start(1.0e3, 1.5e-5)
    rng = np.random.default_rng(3)
    for n in range(2000):
        volume = float(rng.uniform(5e-6, 3e-5))
        exchange = float(rng.uniform(-1e-3, 1e-3)) if n % 7 == 0 else 0.0   # pockets seal (+) or reopen (-)
        model.advance(float(rng.uniform(1e-5, 5e-5)), float(rng.uniform(-8e3, 12e3)), volume, n % 11 == 0,
                      exchange)
    ledger = model.ledger()
    assert abs(ledger["residual_pa_m3"]) <= 1e-12 * max(abs(model.q_pa_m3), 1.0)
    assert model.capped_steps > 0 and model.covered_steps > 0
    mass = model.mass_kg()
    assert mass == pytest.approx(1.204 * model.q_pa_m3 / P0)


def test_no_volume_falls_back_to_the_line_pressure_and_is_logged():
    model = InletAirModel("capped", hole_a_pa_s_m3=LID_HOLE[0], hole_b_pa_s2_m6=LID_HOLE[1], flow_cap_m3_s=CAP)
    model.start(0.0, 1.0e-6)
    q = model.q_pa_m3
    p = model.advance(3.0e-5, -2.5e3, 0.0, True, 0.0, t_s=0.1)
    assert p == -2.5e3 and model.fallback and model.fallbacks == 1 and model.q_m3_s == 0.0
    assert model.fallback_events == [dict(time_s=0.1, p_line_gauge_pa=-2.5e3, covered=True, q_pa_m3=q)]
    assert model.ledger()["residual_pa_m3"] == 0.0
    model.advance(3.0e-5, -2.5e3, 1.0e-6, True, 0.0)
    assert not model.fallback and model.fallbacks == 1


def test_a_region_back_from_no_volume_restarts_at_the_line_pressure():
    model = InletAirModel("capped", restriction=SeriesRestriction([HOLE]), flow_cap_m3_s=CAP)
    model.start(1.0e3, 0.0)                                       # a covered start without inlet air
    model.advance(3.0e-5, 2.5e3, 0.0, True, 0.0, t_s=0.1)          # still none: the fallback
    assert model.fallback and model.reappearances == 0 and model.q_pa_m3 == 0.0
    p = model.advance(3.0e-5, 3.0e3, 1.0e-5, False, 0.0, t_s=0.2)  # back: at the line, not near vacuum
    assert p == pytest.approx(3.0e3, abs=1e-9 * P0) and abs(model.q_m3_s) < 1e-12
    assert model.reappearances == 1 and model.reappear_events == [dict(
        time_s=0.2, p_line_gauge_pa=3.0e3, volume_m3=1.0e-5, q_before_pa_m3=0.0, covered=False)]
    model.advance(3.0e-5, 3.5e3, 0.0, True, 0.0)                  # gone again, then back while covered
    p = model.advance(3.0e-5, -1.5e3, 2.0e-5, True, 0.0)
    assert p == pytest.approx(-1.5e3, abs=1e-9 * P0) and model.reappearances == 2
    for n in range(50):                                           # an ordinary run on: no more reappearances
        model.advance(3.0e-5, 4.0e3, 2.0e-5, n < 10, 0.0)
    ledger = model.ledger()
    assert model.reappearances == 2 and ledger["reappear_pa_m3"] > 0.0
    assert abs(ledger["residual_pa_m3"]) <= 1e-12 * model.q_pa_m3


def test_a_checkpoint_of_another_line_is_refused():
    model = InletAirModel("capped", restriction=SeriesRestriction([HOLE]), flow_cap_m3_s=CAP)
    model.start(0.0, 2.0e-5)
    model.advance(3.0e-5, 5.0e3, 2.0e-5, False, 0.0)
    host = json.loads(json.dumps(model.state_host(), allow_nan=False))
    assert host["line_config"]["elements"] == [["lid hole", "hole", 2.0e-3, 7.0e-3]]
    connector = FlowElement("connector bore", "fitting", 1.5e-3)
    for other in (InletAirModel("capped", restriction=SeriesRestriction([connector, HOLE]), flow_cap_m3_s=CAP),
                  InletAirModel("capped", restriction=SeriesRestriction([HOLE]), flow_cap_m3_s=2.0 * CAP),
                  InletAirModel("capped", restriction=SeriesRestriction([HOLE]), flow_cap_m3_s=CAP,
                                line_volume_m3=1.0e-6)):
        with pytest.raises(ValueError, match="inlet line"):
            other.load_state_host(host)
        assert other.q_pa_m3 is None                              # refused before anything was loaded
    same = InletAirModel("capped", restriction=SeriesRestriction([HOLE]), flow_cap_m3_s=CAP)
    same.load_state_host(host)
    assert same.state_host() == model.state_host()
    with pytest.raises(ValueError, match="inlet line"):          # a state without its line
        same.load_state_host({k: v for k, v in host.items() if k != "line_config"})


def test_a_non_positive_absolute_pressure_is_a_hard_stop():
    model = InletAirModel("capped", hole_a_pa_s_m3=LID_HOLE[0], hole_b_pa_s2_m6=LID_HOLE[1], flow_cap_m3_s=CAP)
    model.start(0.0, 1.0e-5)
    with pytest.raises(SimulationHardStop) as stop:
        model.advance(3.0e-5, 0.0, 1.0e-5, True, 2.0 * model.q_pa_m3)    # a seal took more gas than there was
    assert stop.value.code == "inlet_air_invalid_pressure"


def test_the_state_round_trips_through_json():
    model = InletAirModel("capped", hole_a_pa_s_m3=LID_HOLE[0], hole_b_pa_s2_m6=LID_HOLE[1], flow_cap_m3_s=CAP)
    model.start(0.0, 2.0e-5)
    for n in range(30):
        model.advance(3.0e-5, 15.0e3, 2.0e-5, False, 0.0, t_s=n * 3.0e-5)
    model.advance(3.0e-5, 15.0e3, 0.0, True, 0.0, t_s=1.0)              # one logged fallback
    host = json.loads(json.dumps(model.state_host(), allow_nan=False))   # the checkpoint header's rule: no inf / nan
    other = InletAirModel("capped", hole_a_pa_s_m3=LID_HOLE[0], hole_b_pa_s2_m6=LID_HOLE[1], flow_cap_m3_s=CAP)
    other.load_state_host(host)
    for m in (model, other):
        for n in range(20):
            m.advance(3.0e-5, 15.0e3 - 500.0 * n, 2.0e-5, False, 0.0)
    assert model.state_host() == other.state_host()


def test_invalid_parameters_are_refused():
    with pytest.raises(ValueError):
        InletAirModel("leaky")
    with pytest.raises(ValueError):
        InletAirModel("capped", hole_a_pa_s_m3=-1.0)
    with pytest.raises(ValueError):
        InletAirModel("capped", flow_cap_m3_s=0.0)
    with pytest.raises(ValueError):
        hole_law(0.0, 7.0e-3)
    with pytest.raises(ValueError):
        FlowElement("hose", "pipe", 2.0e-3, 0.1)
    with pytest.raises(ValueError):
        FlowElement("hose", "tube", 0.0, 0.1)
    with pytest.raises(ValueError):
        FlowElement("hose", "tube", 2.0e-3, -0.1)
    with pytest.raises(TypeError):
        SeriesRestriction([(2.0e-3, 7.0e-3)])
    with pytest.raises(ValueError):
        SeriesRestriction([HOLE], coefficients=LID_HOLE)
    with pytest.raises(ValueError):
        InletAirModel("capped", hole_a_pa_s_m3=1.0, restriction=SeriesRestriction([HOLE]))


def test_the_lid_hole_alone_as_a_series_line_is_the_hole_law_bit_for_bit():
    line = SeriesRestriction([HOLE])
    assert (line.a, line.b) == LID_HOLE
    assert [(j["kind"], j["heads"]) for j in line.joints] == [("entrance", 0.5), ("exit", 1.0)]
    model = InletAirModel("capped", restriction=line, flow_cap_m3_s=CAP)
    reference = InletAirModel("capped", hole_a_pa_s_m3=LID_HOLE[0], hole_b_pa_s2_m6=LID_HOLE[1], flow_cap_m3_s=CAP)
    for m in (model, reference):
        m.start(0.0, 2.0e-5)
    def physics(m):                                  # the Mach log and the line's elements need the geometry: the
        return {k: v for k, v in m.state_host().items() if not k.startswith(("mach", "line_config"))}   # coefficients
    lines = [m.line_config() for m in (model, reference)]                                                # have none
    assert [(c["a_pa_s_m3"], c["b_pa_s2_m6"]) for c in lines] == [LID_HOLE, LID_HOLE]
    for n in range(60):
        for m in (model, reference):
            m.advance(3.0e-5, 25.0e3 * math.sin(0.2 * n), 2.0e-5, n % 13 == 0, 0.0)
        assert physics(model) == physics(reference)
    assert model.capped_steps > 0 and model.turbulent_steps == model.transition_steps == 0
    assert model.mach_max > 0.0 == reference.mach_max


def test_the_joints_follow_the_area_ratios():
    bore, hose = 1.1e-3, 2.0e-3
    line = SeriesRestriction([FlowElement("Y bore", "fitting", bore), FlowElement("hose", "tube", hose, 0.3),
                              FlowElement("connector bore", "fitting", bore), HOLE])
    shrink = (bore / hose) ** 2
    assert [j["kind"] for j in line.joints] == ["entrance", "expansion", "contraction", "expansion", "exit"]
    bores = 0.5 + (1.0 - shrink) ** 2 + 0.5 * (1.0 - shrink) + (1.0 - (bore / 2.0e-3) ** 2) ** 2
    area = 0.25 * math.pi * bore ** 2
    b = bores * AIR_DENSITY_KG_M3 / (2.0 * area * area) + 1.0 * AIR_DENSITY_KG_M3 / (2.0 * HOLE.area_m2 ** 2)
    assert line.b == pytest.approx(b, rel=1e-14)                  # no hose heads: the hose has the bores' joints
    assert line.a == pytest.approx(LID_HOLE[0] + 128.0 * AIR_VISCOSITY_PA_S * 0.3 / (math.pi * hose ** 4), rel=1e-14)
    wide = SeriesRestriction([FlowElement("connector bore", "fitting", 3.0e-3), HOLE])   # a wider bore feeds the hole
    assert [j["kind"] for j in wide.joints] == ["entrance", "contraction", "exit"]
    assert wide.joints[1]["heads"] == pytest.approx(0.5 * (1.0 - (2.0 / 3.0) ** 2), rel=1e-14)
    assert line.describe()["elements"][1] == dict(name="hose", kind="tube", diameter_m=hose, length_m=0.3)


def test_the_hose_is_poiseuille_below_re_2300_and_blasius_above():
    hose = FlowElement("hose", "tube", 2.0e-3, 0.3)
    line = SeriesRestriction([hose])
    joints = line.b
    q_laminar = 2300.0 * AIR_VISCOSITY_PA_S * hose.area_m2 / (AIR_DENSITY_KG_M3 * hose.diameter_m)
    poiseuille = 128.0 * AIR_VISCOSITY_PA_S * 0.3 / (math.pi * 2.0e-3 ** 4)
    assert poiseuille * 35.0e-6 == pytest.approx(484.0, abs=1.0)  # Root: a 2 mm hose of 0.3 m, about 0.5 kPa
    assert line.pressure_drop_pa(35.0e-6) == pytest.approx(poiseuille * 35.0e-6 + joints * 35.0e-6 ** 2, rel=1e-12)
    q = 1.0e-4                                                     # Re = 4240
    u = q / hose.area_m2
    reynolds = AIR_DENSITY_KG_M3 * u * hose.diameter_m / AIR_VISCOSITY_PA_S
    blasius = 0.3164 * reynolds ** -0.25 * (0.3 / hose.diameter_m) * 0.5 * AIR_DENSITY_KG_M3 * u * u
    assert line.pressure_drop_pa(q) == pytest.approx(blasius + joints * q * q, rel=1e-12)
    assert line.pressure_drop_pa(-q) == -line.pressure_drop_pa(q)
    for d in (500.0, 2.0e4, -2.0e4):                               # closed form below, bisected above
        flow, regime = line.solve(d)
        assert line.pressure_drop_pa(flow) == pytest.approx(d, rel=1e-12)
        assert regime == ("laminar" if abs(flow) <= q_laminar else "turbulent")
    low, high = line.pressure_drop_pa(q_laminar), line.pressure_drop_pa(math.nextafter(q_laminar, 1.0))
    assert high - low == pytest.approx((0.3164 * 2300.0 ** -0.25 / (64.0 / 2300.0) - 1.0) * poiseuille * q_laminar,
                                       rel=1e-9)                   # Blasius is 1.64 x Poiseuille at Re 2300
    flow, regime = line.solve(0.5 * (low + high))                  # inside the jump: the flow stays at Re 2300
    assert regime == "transition" and flow == pytest.approx(q_laminar, rel=1e-12)
    model = InletAirModel("capped", restriction=line, flow_cap_m3_s=CAP)
    model.start(0.0, 2.0e-5)
    for n in range(200):
        model.advance(3.0e-5, 20.0e3, 2.0e-5, False, 0.0)
    assert model.turbulent_steps > 0 and abs(model.ledger()["residual_pa_m3"]) <= 1e-12 * model.q_pa_m3


def test_c_a_line_without_elements_reproduces_the_ideal_inlet():
    line = SeriesRestriction(())
    assert (line.a, line.b, line.joints, line.tubes) == (0.0, 0.0, [], [])
    model = InletAirModel("capped", restriction=line)
    model.start(0.0, 1.0e-5)
    for p_line in (3.0e3, -2.0e3, 8.0e3):
        assert model.advance(3.0e-5, p_line, 1.2e-5, False, 0.0) == pytest.approx(p_line, abs=1e-9 * P0)
