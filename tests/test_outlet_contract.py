"""The shifted outlet's projection contract and its guard (boundary/outlet_guard.py).

A chamber that solves a shifted pressure needs the solver's outlet backflow ALLOWED: the solver's ghost-0 half of the
outlet correction leaves the outlet face temporarily inflow, and pressure_outlet_backflow_policy="clamp" (the solver's
default) zeroes it before the chamber's after_project() adds its half -- the water near the outlet then settles near
-p_ref.  On the duct rig of test_pressure_regions.py (8 x 8 x 32, z graded, z-min outlet, the real solver on the CPU,
PressureRegions over the whole box), OLD = project() kwargs without the contract and NEW =
merge_project_requirements(chamber, the same kwargs):
  1. contract: the chamber classes carry PROJECT_REQUIREMENTS (backflow 'allow', no accumulation); the merge raises
     without one and on a conflict; pneumatic_chamber re-exports it;
  2. still tank (closed obstacle membrane, inlet air above at p_in = +-4500 Pa): NEW keeps the outlet layer and the
     water at 0 within 1e-6 |p_in| and the air at p_in; OLD puts the tank near -p_in;
  3. guard: OLD's ratio is about 1 and strict mode raises "outlet_shift_invariant"; NEW's ratio is below 1e-4
     (the OLD runs switch the guard's contract check off: they reproduce the defect on purpose);
  3b. the contract check: after_project() after a project() without the contract raises ValueError (the solver records
     the contract arguments of every call), with the contract it passes;
  4. exactly once: a second after_project() for one update() raises;
  5. idempotent projections: two consecutive update -> project -> after_project with nothing changed leave the outlet
     faces and pressure unchanged (NEW); OLD's outlet pressure changes;
  6. forced auto-cleanup (loose CG tolerance; the pressure-outlet cleanup passes are counted and must run): NEW keeps
     the outlet at 0 and the guard ratio below the production limit;
  7. flowing duct (air directly on water): NEW's outlet stays small and decays;
  8. pressurised-chamber jet (a 2 x 2-cell hole in an obstacle plate): NEW's mean hole speed <= the ideal Bernoulli
     speed sqrt(2 p_in / rho) (the discharge coefficient is printed); OLD's is at least 20 % faster than NEW's;
  9. the shift floor: ratio = pre / max(shift, shift_floor) (the report arithmetic; a strict sweep of p_in through zero
     with a pressure-independent divergence source does not stop, although the plain pre / shift exceeds the limit)."""
import math

import numpy as np
import pytest
import taichi as ti

import test_pressure_regions as duct
from fastfsi.boundary import pneumatic_chamber, pressure_chamber, pressure_regions
from fastfsi.boundary.pneumatic_chamber import PneumaticChamber
from fastfsi.boundary.pressure_regions import PressureRegions
from fastfsi.boundary.outlet_guard import (INVARIANT_RATIO_MAX, PROJECT_REQUIREMENTS, merge_project_requirements)
from fastfsi.boundary.pressure_chamber import ColumnPressureChamber
from fastfsi.errors import SimulationHardStop

P_JET = 4500.0
DRIVER_KWARGS = dict(iterations=400, pressure_outlet_zmin=True, pressure_reachability_prepared=True,
                     reset_pressure=False, pressure_solver="fv_cg", cg_tolerance=duct.CG_TOLERANCE,
                     cg_preconditioner="fv_multigrid", pressure_solve_failure_policy="raise", read_report=False)


@pytest.fixture(scope="module", autouse=True)
def own_duct_rig():
    """This module builds its own duct rig: test_pressure_regions caches its solver in duct._STATE, and a module that runs
    in between may re-initialise Taichi (cpu_runtime.initialize()), which would leave a solver cached here dead there."""
    duct._STATE.clear()
    yield
    duct._STATE.clear()


def kwargs_for(chamber, mode, **override):
    base = dict(DRIVER_KWARGS, **override)
    # OLD reproduces the defect on purpose: the guard's contract check (on by default) would refuse its projections
    chamber.outlet_guard.verify_contract = mode != "old"
    return base if mode == "old" else merge_project_requirements(chamber, base)


def reset(fluid, chamber, obstacle, codes):
    duct.set_obstacle(fluid, obstacle)
    chamber.set_regions(codes, [])
    zero_u = np.zeros((duct.NX, duct.NY, duct.NZ, 3), np.float32)
    fluid.velocity.from_numpy(zero_u)
    fluid.velocity_prev.from_numpy(zero_u)
    fluid.pressure.from_numpy(np.zeros((duct.NX, duct.NY, duct.NZ), np.float64))
    chamber.outlet_guard.report()                         # clear the interval maxima of earlier tests


def step(fluid, chamber, p_in, kwargs, t, predict=True):
    chamber.update(t + 0.5 * duct.DT, p_in)
    if predict:
        fluid.predict(dt_s=duct.DT, advection_scheme="muscl_tvd", pressure_outlet_zmin=True)
    fluid.project(dt_s=duct.DT, **kwargs)
    chamber.after_project(duct.DT, p_in)
    return t + duct.DT


def membrane_tank():
    obstacle = np.zeros((duct.NX, duct.NY, duct.NZ), np.int32)
    obstacle[:, :, duct.K_AIR - 1] = 1
    codes = np.zeros((duct.NX, duct.NY, duct.NZ), np.int32)
    codes[:, :, duct.K_AIR:] = 1
    return obstacle, codes


def layer_stats(fluid, chamber, obstacle, codes):
    p = chamber.physical_pressure_field().to_numpy()
    water = (obstacle == 0) & (codes == 0)
    return dict(outlet=float(p[:, :, 0][water[:, :, 0]].mean()), water=float(p[water].mean()),
                water_abs_max=float(np.abs(p[water]).max()), air=float(p[codes == 1].mean()),
                uz0=fluid.velocity.to_numpy()[:, :, 0, 2].copy())


def run_tank(mode, p_in, steps=30):
    fluid, _, chamber = duct.rig()
    obstacle, codes = membrane_tank()
    reset(fluid, chamber, obstacle, codes)
    kwargs = kwargs_for(chamber, mode)
    t = 0.0
    for _ in range(steps):
        t = step(fluid, chamber, p_in, kwargs, t)
    return fluid, chamber, obstacle, codes, kwargs, t


# ------------------------------------------------------------------------------------------------------- 1. contract
def test_chamber_classes_carry_the_projection_contract():
    for cls in (ColumnPressureChamber, PressureRegions, PneumaticChamber):
        assert cls.PROJECT_REQUIREMENTS is PROJECT_REQUIREMENTS
    assert PROJECT_REQUIREMENTS["pressure_outlet_backflow_policy"] == "allow"
    assert PROJECT_REQUIREMENTS["accumulate_pressure_into_previous"] is False
    for module in (pressure_chamber, pressure_regions, pneumatic_chamber):   # a module lookup finds it too
        assert getattr(module, "PROJECT_REQUIREMENTS") is PROJECT_REQUIREMENTS


def test_merge_raises_without_a_contract_and_on_a_conflict():
    class NoContract:
        pass

    with pytest.raises(ValueError, match="declares no PROJECT_REQUIREMENTS"):
        merge_project_requirements(NoContract(), DRIVER_KWARGS)
    _, _, chamber = duct.rig()
    with pytest.raises(ValueError, match="pressure_outlet_backflow_policy"):
        merge_project_requirements(chamber, dict(DRIVER_KWARGS, pressure_outlet_backflow_policy="clamp"))
    merged = merge_project_requirements(chamber, dict(DRIVER_KWARGS, cg_preconditioner="fv_multigrid_light"))
    assert merged["cg_preconditioner"] == "fv_multigrid_light"            # the multigrid family is one value
    assert merged["pressure_outlet_backflow_policy"] == "allow"
    assert "pressure_outlet_backflow_policy" not in DRIVER_KWARGS          # the caller's dict is not modified


# --------------------------------------------------------------------------------------------- 2.-3. tank and guard
@pytest.mark.parametrize("p_in", [P_JET, -P_JET])
def test_still_tank_outlet_at_zero_with_the_contract(p_in):
    fluid, chamber, obstacle, codes, _, _ = run_tank("new", p_in)
    s = layer_stats(fluid, chamber, obstacle, codes)
    print("OUTLET still tank NEW p_in %+.0f: outlet %.3e Pa, water |p| max %.3e Pa, air %.3f Pa"
          % (p_in, s["outlet"], s["water_abs_max"], s["air"]))
    assert abs(s["outlet"]) <= 1.0e-6 * abs(p_in)
    assert s["water_abs_max"] <= 1.0e-6 * abs(p_in)
    assert abs(s["air"] - p_in) <= 1.0e-6 * abs(p_in)
    guard = chamber.outlet_guard.report()
    print("OUTLET still tank NEW p_in %+.0f: guard %s" % (p_in, guard))
    assert guard["outlet_guard_calls"] > 0 and guard["outlet_guard_ratio"] < 1.0e-4


def test_still_tank_old_kwargs_reproduce_the_minus_p_in_tank_and_trip_the_guard():
    fluid, chamber, obstacle, codes, _, _ = run_tank("old", P_JET)
    s = layer_stats(fluid, chamber, obstacle, codes)
    print("OUTLET still tank OLD p_in %+.0f: outlet %.1f Pa (%.3f p_in), water mean %.1f Pa, air %.1f Pa"
          % (P_JET, s["outlet"], s["outlet"] / P_JET, s["water"], s["air"]))
    assert s["outlet"] < -0.5 * P_JET                                      # the tank near -p_in
    chamber.outlet_guard.strict = True
    try:
        with pytest.raises(SimulationHardStop) as stop:
            chamber.report()
        assert stop.value.code == "outlet_shift_invariant"
        print("OUTLET still tank OLD guard witness %s" % (stop.value.witness,))
        assert stop.value.witness["outlet_guard_ratio"] > 0.5
    finally:
        chamber.outlet_guard.strict = False


# ------------------------------------------------------------------------------------------- 3b. the contract check
def test_after_project_refuses_a_projection_without_the_contract():
    """The guard's contract check is on by default: after_project() following a project() with the solver's defaults
    (backflow 'clamp') raises and names the broken key; the same step with the merged contract passes; a conflicting
    preconditioner is refused and the multigrid family accepted; with no project() since update() there is nothing to
    check."""
    fluid, _, chamber = duct.rig()
    obstacle, codes = membrane_tank()
    reset(fluid, chamber, obstacle, codes)
    assert chamber.outlet_guard.requirements == PROJECT_REQUIREMENTS
    chamber.outlet_guard.verify_contract = True
    chamber.update(0.5 * duct.DT, P_JET)
    fluid.project(dt_s=duct.DT, **DRIVER_KWARGS)                          # the caller dropped the merge
    with pytest.raises(ValueError, match="pressure_outlet_backflow_policy='clamp' \\(required 'allow'\\)"):
        chamber.after_project(duct.DT, P_JET)
    reset(fluid, chamber, obstacle, codes)
    merged = merge_project_requirements(chamber, DRIVER_KWARGS)
    chamber.update(0.5 * duct.DT, P_JET)
    fluid.project(dt_s=duct.DT, **merged)
    chamber.after_project(duct.DT, P_JET)                                 # the contract: no error
    assert fluid.last_project_contract["pressure_outlet_backflow_policy"] == "allow"
    # a conflicting value and the multigrid family, through the record the solver keeps of a call (this graded duct
    # admits only the FV solvers and no pressure accumulation, so the record stands in for those projections)
    for preconditioner, refused in (("jacobi", True), ("fv_multigrid_light", False)):
        reset(fluid, chamber, obstacle, codes)
        chamber.update(0.5 * duct.DT, P_JET)
        fluid.project_calls += 1
        fluid.last_project_contract = dict(PROJECT_REQUIREMENTS, cg_preconditioner=preconditioner)
        if refused:
            with pytest.raises(ValueError, match="cg_preconditioner='jacobi' \\(required 'fv_multigrid'\\)"):
                chamber.after_project(duct.DT, P_JET)
        else:
            chamber.after_project(duct.DT, P_JET)                         # the multigrid family is one value
    reset(fluid, chamber, obstacle, codes)
    chamber.update(0.5 * duct.DT, P_JET)
    chamber.after_project(duct.DT, P_JET)                                 # no projection since update()


def test_the_solver_records_the_contract_arguments_of_every_projection():
    fluid, _, chamber = duct.rig()
    obstacle, codes = membrane_tank()
    reset(fluid, chamber, obstacle, codes)
    calls = fluid.project_calls
    step(fluid, chamber, P_JET, kwargs_for(chamber, "new"), 0.0)
    assert fluid.project_calls == calls + 1
    assert fluid.last_project_contract == PROJECT_REQUIREMENTS


# ------------------------------------------------------------------------------------------------- 4. exactly once
def test_second_after_project_for_one_update_raises():
    fluid, chamber, obstacle, codes, kwargs, t = run_tank("new", P_JET, steps=2)
    chamber.update(t + 0.5 * duct.DT, P_JET)
    fluid.project(dt_s=duct.DT, **kwargs)
    chamber.after_project(duct.DT, P_JET)
    with pytest.raises(RuntimeError, match="applied twice"):
        chamber.after_project(duct.DT, P_JET)


# ------------------------------------------------------------------------------------------------- 5. idempotence
@pytest.mark.parametrize("mode", ["new", "old"])
def test_two_projections_with_nothing_changed_leave_the_outlet(mode):
    """From the fresh still tank (u = 0, p~ = 0): two update -> project -> after_project without a predict.  NEW: the
    second leaves the outlet faces and the outlet-layer pressure unchanged.  OLD: the faces end at the same wrong value
    after each projection, but the outlet pressure doubles (-p_in, then -2 p_in in p~): the clamp's fixed point."""
    fluid, _, chamber = duct.rig()
    obstacle, codes = membrane_tank()
    reset(fluid, chamber, obstacle, codes)
    kwargs = kwargs_for(chamber, mode)
    t = step(fluid, chamber, P_JET, kwargs, 0.0, predict=False)
    faces = [fluid.velocity.to_numpy()[:, :, 0, 2].astype(np.float64)]
    layer = [fluid.pressure.to_numpy()[:, :, 0].astype(np.float64)]
    step(fluid, chamber, P_JET, kwargs, t, predict=False)
    faces.append(fluid.velocity.to_numpy()[:, :, 0, 2].astype(np.float64))
    layer.append(fluid.pressure.to_numpy()[:, :, 0].astype(np.float64))
    shift = duct.DT / duct.RHO * 2.0 * P_JET / float(fluid.cell_width_z_m.to_numpy()[0])
    du = float(np.abs(faces[1] - faces[0]).max())
    dp = float(np.abs(layer[1] - layer[0]).max())
    print("OUTLET idempotence %s: max |du_z| at the outlet %.3e m/s (outlet shift %.4f m/s); max |dp~| outlet layer "
          "%.3e Pa; p~ outlet mean %.1f -> %.1f Pa" % (mode, du, shift, dp, layer[0].mean(), layer[1].mean()))
    if mode == "new":
        assert du <= 1.0e-6 * shift and dp <= 1.0e-6 * P_JET
    else:
        assert dp > 0.5 * P_JET


# ------------------------------------------------------------------------------------------------ 6. forced cleanup
def test_forced_auto_cleanup_keeps_the_outlet():
    from fastfsi.fluid.pressure_projection import PressureProjection as S
    fluid, _, chamber = duct.rig()
    obstacle, codes = membrane_tank()
    reset(fluid, chamber, obstacle, codes)
    rng = np.random.default_rng(11)
    u = rng.normal(0.0, 0.05, (duct.NX, duct.NY, duct.NZ, 3)).astype(np.float32)
    u[obstacle != 0] = 0.0
    fluid.velocity.from_numpy(u)
    fluid.velocity_prev.from_numpy(u)
    name = "_clear_pressure_interface_matrix_rhs_kernel"        # runs once per auto-cleanup pass, nowhere else
    original = getattr(S, name)
    count = dict(passes=0)
    setattr(S, "_counted_orig_" + name, original)

    def counted(self, *args):
        count["passes"] += 1
        return getattr(self, "_counted_orig_" + name)(*args)
    setattr(S, name, counted)
    try:
        kwargs = kwargs_for(chamber, "new", cg_tolerance=0.3)   # the main solve stops at 30 %: the cleanup runs
        t = 0.0
        for _ in range(12):
            t = step(fluid, chamber, P_JET, kwargs, t)
    finally:
        setattr(S, name, original)
        delattr(S, "_counted_orig_" + name)
    guard = chamber.outlet_guard.report()
    s = layer_stats(fluid, chamber, obstacle, codes)
    print("OUTLET forced cleanup: %d cleanup passes in 12 steps; outlet %.3e Pa (%.2e p_in); guard %s"
          % (count["passes"], s["outlet"], s["outlet"] / P_JET, guard))
    assert count["passes"] > 0
    assert abs(s["outlet"]) <= 1.0e-3 * P_JET
    # the 30 % solver tolerance leaves its residual in the guard's pre measure (3e-4 measured): the production limit
    # applies, not the tight bound of the converged tests
    assert guard["outlet_guard_ratio"] < INVARIANT_RATIO_MAX


# ---------------------------------------------------------------------------------------------- 7. flowing duct
def test_flowing_duct_outlet_stays_small():
    fluid, _, chamber = duct.rig()
    obstacle = np.zeros((duct.NX, duct.NY, duct.NZ), np.int32)
    codes = np.zeros((duct.NX, duct.NY, duct.NZ), np.int32)
    codes[:, :, duct.K_AIR:] = 1
    reset(fluid, chamber, obstacle, codes)
    kwargs = kwargs_for(chamber, "new")
    t, outlet = 0.0, []
    for n in range(1, 121):
        t = step(fluid, chamber, P_JET, kwargs, t)
        if n % 40 == 0:
            outlet.append(layer_stats(fluid, chamber, obstacle, codes)["outlet"] / P_JET)
    guard = chamber.outlet_guard.report()
    print("OUTLET flowing duct NEW: outlet / p_in at steps 40, 80, 120: %s; guard ratio %.2e"
          % (["%.4f" % v for v in outlet], guard["outlet_guard_ratio"]))
    assert all(abs(v) < 0.05 for v in outlet) and abs(outlet[-1]) <= abs(outlet[0])
    assert guard["outlet_guard_ratio"] < 1.0e-4


# ------------------------------------------------------------------------------------------------------ 8. the jet
def jet_speed(mode, steps=200):
    fluid, _, chamber = duct.rig()
    k_plate = 12
    obstacle = np.zeros((duct.NX, duct.NY, duct.NZ), np.int32)
    obstacle[:, :, k_plate] = 1
    obstacle[3:5, 3:5, k_plate] = 0                                       # the 2 x 2-cell hole
    codes = np.zeros((duct.NX, duct.NY, duct.NZ), np.int32)
    codes[:, :, duct.K_AIR:] = 1
    reset(fluid, chamber, obstacle, codes)
    kwargs = kwargs_for(chamber, mode)
    wx, wy = fluid.cell_width_x_m.to_numpy(), fluid.cell_width_y_m.to_numpy()
    area = float(wx[3:5].sum() * wy[3:5].sum())
    t, speeds = 0.0, []
    for n in range(1, steps + 1):
        t = step(fluid, chamber, P_JET, kwargs, t)
        if n > steps - 20:                                                # the mean over the last 20 steps
            uz = fluid.velocity.to_numpy()[3:5, 3:5, k_plate, 2].astype(np.float64)   # the hole's lower face
            flux = float((uz * wx[3:5, None] * wy[None, 3:5]).sum())
            speeds.append(abs(flux) / area)
    p = chamber.physical_pressure_field().to_numpy()
    return float(np.mean(speeds)), float(p[:, :, 0][obstacle[:, :, 0] == 0].mean())


def test_jet_mean_speed_at_most_ideal_bernoulli():
    ideal = math.sqrt(2.0 * P_JET / duct.RHO)
    new_speed, new_outlet = jet_speed("new")
    old_speed, old_outlet = jet_speed("old")
    print("OUTLET jet: ideal Bernoulli %.4f m/s; NEW mean hole speed %.4f m/s (discharge coefficient %.3f, outlet %.1f "
          "Pa); OLD %.4f m/s (ratio %.3f, outlet %.1f Pa)" % (ideal, new_speed, new_speed / ideal, new_outlet,
                                                              old_speed, old_speed / ideal, old_outlet))
    assert new_speed <= ideal
    assert abs(new_outlet) <= 0.05 * P_JET
    assert old_speed >= 1.2 * new_speed


# ---------------------------------------------------------------------------------- 9. the shift floor (zero crossing)
def test_guard_ratio_uses_the_shift_floor():
    """report(): ratio = pre / max(shift, shift_floor).  Near a zero crossing of p_ref the shift is tiny and the plain
    pre / shift would exceed the limit; a real defect (pre about the shift at a large p_ref) still gives about 1."""
    from fastfsi.boundary.outlet_guard import G_CALLS, G_FLOOR, G_PRE, G_POST, G_SHIFT, OutletShiftGuard
    fluid, _, _ = duct.rig()
    guard = OutletShiftGuard(fluid, strict=True)

    def report_with(pre, shift, floor):
        values = np.zeros(guard.stats.shape, np.float64)
        values[G_PRE], values[G_POST], values[G_SHIFT], values[G_CALLS], values[G_FLOOR] = pre, 0.0, shift, 1.0, floor
        guard.stats.from_numpy(values)
        return guard.report()

    near_zero = report_with(pre=1.0e-3, shift=0.04, floor=1.0)              # p_ref about 0: pre / shift = 2.5e-2
    assert near_zero["outlet_guard_ratio"] == pytest.approx(1.0e-3)
    assert near_zero["outlet_guard_shift_floor_1ps"] == pytest.approx(1.0)
    at_zero = report_with(pre=1.0e-3, shift=0.0, floor=1.0)                 # p_ref exactly 0: still checked
    assert at_zero["outlet_guard_ratio"] == pytest.approx(1.0e-3)
    with pytest.raises(SimulationHardStop) as stop:                         # the defect: pre about the shift
        report_with(pre=0.5, shift=1.0, floor=0.1)
    assert stop.value.code == "outlet_shift_invariant"


def test_strict_guard_through_zero_inlet_pressure():
    """p_in swept through zero in strict mode while a pressure-independent divergence source keeps the solver's
    residual (pre) at its tolerance level, as a moving body or a decelerating jet does in a run (here: a random
    velocity perturbation before every step and a loose CG tolerance).  Near zero the shift shrinks with |p_in| while
    pre does not, so the plain pre / shift exceeds the limit (strict mode would stop); the floored ratio stays at the
    solver level, and the ratio is defined at p_in = 0 too."""
    fluid, _, chamber = duct.rig()
    obstacle, codes = membrane_tank()
    reset(fluid, chamber, obstacle, codes)
    rng = np.random.default_rng(23)
    kwargs = kwargs_for(chamber, "new", cg_tolerance=0.3)
    sweep = (100.0, 1.0, 0.01, 0.001, 0.0, -0.001, -0.01, -1.0, -100.0)
    chamber.outlet_guard.strict = True
    plain, floored = [], []
    try:
        t = 0.0
        for p_in in sweep:
            u = fluid.velocity.to_numpy()
            u += rng.normal(0.0, 0.01, u.shape).astype(np.float32)
            u[obstacle != 0] = 0.0
            fluid.velocity.from_numpy(u)
            t = step(fluid, chamber, p_in, kwargs, t)
            g = chamber.outlet_guard.report()                                  # strict: raises on a violation
            assert g["outlet_guard_ratio"] is not None and g["outlet_guard_ratio"] <= INVARIANT_RATIO_MAX
            floored.append(g["outlet_guard_ratio"])
            if g["outlet_guard_shift_max_1ps"] > 0.0:
                plain.append(g["outlet_guard_pre_max_1ps"] / g["outlet_guard_shift_max_1ps"])
    finally:
        chamber.outlet_guard.strict = False
    print("OUTLET zero crossing: plain pre / shift max %.3e, floored ratio max %.3e (limit %.0e)"
          % (max(plain), max(floored), INVARIANT_RATIO_MAX))
    assert max(plain) > INVARIANT_RATIO_MAX                                   # without the floor strict mode would stop
