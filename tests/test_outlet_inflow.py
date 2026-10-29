"""The total-pressure outlet inflow: the chambers' option outlet_inflow (outlet_guard.outlet_ghost_pressure).

On the duct rig of test_pressure_regions.py (8 x 8 x 32, z graded, z-min outlet, the real solver on the CPU):
  1. rows: a hand-set outlet layer with inflow, outflow and still faces.  "total_pressure" writes
     outlet_rhs = w_f 2 / dz0^2 (-p_ref - rho w^2 / 2) on the inflow faces and w_f 2 / dz0^2 (-p_ref) elsewhere (rel.
     1e-12), in PressureRegions and in ColumnPressureChamber; "static" (the default) writes the rows the chambers
     wrote before the option (the formula's bits, whatever the velocity); another mode is refused;
  2. the split stays closed: strict mode from the environment (FASTFSI_STRICT_INVARIANTS=1), "total_pressure", the
     still tank with an exchange through the outlet (inflow on half of it): no outlet_shift_invariant stop and the
     guard ratio <= 1e-4; the outlet monitors split the outlet flux into inflow and outflow;
  3. mechanism: from a divergence-free exchange through the outlet (half the bottom up, half down, zero net), no
     forcing, 1000 steps under "static" and under "total_pressure" (the up-flow and the kinetic energy every 50 steps
     are printed): "total_pressure" does not run away (its KE at the end <= KE(0)) and ends with up-flow and KE <= those
     of "static"."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

import test_pressure_regions as duct
from fastfsi.boundary.outlet_guard import OUTLET_INFLOW_MODES, merge_project_requirements
from fastfsi.boundary.pressure_chamber import ColumnPressureChamber
from fastfsi.boundary.pressure_regions import PressureRegions
from fastfsi.point_tags import TAG_CHAMBER_SURFACE

P_IN = 4500.0
W_IN, W_OUT = 0.30, -0.20                 # the hand-set outlet faces (m/s)
EXCHANGE_W = 0.20                         # the exchange's largest outlet speed (m/s)
MECHANISM_STEPS, MECHANISM_EVERY = 1000, 50
DRIVER_KWARGS = dict(iterations=400, pressure_outlet_zmin=True, pressure_reachability_prepared=True,
                     reset_pressure=False, pressure_solver="fv_cg", cg_tolerance=duct.CG_TOLERANCE,
                     cg_preconditioner="fv_multigrid", pressure_solve_failure_policy="raise", read_report=False)
BOX = ((0, duct.NX), (0, duct.NY), (0, duct.NZ))


@pytest.fixture(scope="module", autouse=True)
def own_duct_rig():
    """This module builds its own duct rig (as test_outlet_contract does)."""
    duct._STATE.clear()
    yield
    duct._STATE.clear()


def outlet_layer():
    """The outlet faces' w: inflow on x < 3, outflow on x >= 5, still between (f32, as the solver stores it)."""
    w = np.zeros((duct.NX, duct.NY), np.float32)
    w[:3, :] = W_IN
    w[5:, :] = W_OUT
    return w


def still_rig(w0=None):
    fluid, grid, _ = duct.rig()
    duct.set_obstacle(fluid, np.zeros((duct.NX, duct.NY, duct.NZ), np.int32))
    u = np.zeros((duct.NX, duct.NY, duct.NZ, 3), np.float32)
    if w0 is not None:
        u[:, :, 0, 2] = w0
    fluid.velocity.from_numpy(u)
    fluid.pressure.from_numpy(np.zeros((duct.NX, duct.NY, duct.NZ), np.float64))
    return fluid, grid


def expected_rows(fluid, mode, p_ref):
    """w_f 2 / dz0^2 p~_out in the kernels' operation order (free outlet faces: w_f = 1)."""
    dz = np.float64(fluid.cell_width_z_m.to_numpy()[0])
    w = outlet_layer().astype(np.float64)
    ghost = np.full(w.shape, -p_ref)
    if mode == "total_pressure":
        ghost = np.where(w > 0.0, ghost - 0.5 * float(fluid.rho) * w * w, ghost)
    return np.float64(1.0) * 2.0 / (dz * dz) * ghost


def regions_rows(fluid, mode, **kwargs):
    chamber = PressureRegions(fluid, BOX, **kwargs)
    codes = np.zeros((duct.NX, duct.NY, duct.NZ), np.int32)
    codes[:, :, duct.K_AIR:] = 1
    chamber.set_regions(codes, [])
    chamber.update(0.0, P_IN)                         # p_ref = p_in in the water (no pocket)
    assert chamber.outlet_inflow == mode
    return chamber.outlet_rhs.to_numpy()


def surface_points(grid):
    """One chamber-surface point per column centre at the face z of layer K_AIR: the chamber is k >= K_AIR."""
    cx, cy = np.asarray(grid.cell_centers_x_m), np.asarray(grid.cell_centers_y_m)
    z = float(np.asarray(grid.cell_faces_z_m)[duct.K_AIR])
    xyz = np.stack(np.broadcast_arrays(cx[:, None], cy[None, :], z), axis=-1).reshape(-1, 3).astype(np.float32)
    points = SimpleNamespace(n=len(xyz), x=ti.Vector.field(3, ti.f32, shape=len(xyz)),
                             tag=ti.field(ti.i32, shape=len(xyz)))
    points.x.from_numpy(xyz)
    points.tag.from_numpy(np.full(len(xyz), TAG_CHAMBER_SURFACE, np.int32))
    return points


def column_rows(fluid, grid, points, mode, **kwargs):
    chamber = ColumnPressureChamber(fluid, None, np.ones((duct.NX, duct.NY), bool),
                                    float(np.asarray(grid.cell_faces_z_m)[-1]), points, **kwargs)
    chamber.update(0.0, P_IN)
    assert chamber.outlet_inflow == mode and chamber.report()["air_cell_count"] == duct.NX * duct.NY * (
        duct.NZ - duct.K_AIR)
    return chamber.outlet_rhs.to_numpy()


# ------------------------------------------------------------------------------------------------------------ 1. rows
def test_rows_of_both_chambers_static_and_total_pressure():
    fluid, grid = still_rig(outlet_layer())
    points = surface_points(grid)
    assert OUTLET_INFLOW_MODES == ("static", "total_pressure")
    for name, rows_of in (("PressureRegions", lambda mode, **kw: regions_rows(fluid, mode, **kw)),
                          ("ColumnPressureChamber", lambda mode, **kw: column_rows(fluid, grid, points, mode, **kw))):
        default, static = rows_of("static"), rows_of("static", outlet_inflow="static")
        total = rows_of("total_pressure", outlet_inflow="total_pressure")
        static_formula, total_formula = expected_rows(fluid, "static", P_IN), expected_rows(fluid, "total_pressure", P_IN)
        inflow = outlet_layer() > 0.0
        error = float(np.abs(total / total_formula - 1.0).max())
        print("OUTLET_INFLOW rows %s: static == formula bits %s; total_pressure rel. error %.2e; inflow row %.6e vs "
              "static %.6e" % (name, np.array_equal(static, static_formula), error, total[inflow][0], static[0, 0]))
        assert np.array_equal(default, static) and np.array_equal(static, static_formula)
        assert error <= 1.0e-12
        assert np.array_equal(total[~inflow], static[~inflow]) and (total[inflow] < static[inflow]).all()
    fluid.velocity.from_numpy(np.zeros((duct.NX, duct.NY, duct.NZ, 3), np.float32))
    assert np.array_equal(regions_rows(fluid, "static"), static_formula)      # the static rows ignore the velocity
    for chamber_type, args in ((PressureRegions, (fluid, BOX)),
                               (ColumnPressureChamber, (fluid, None, np.ones((duct.NX, duct.NY), bool),
                                                        float(np.asarray(grid.cell_faces_z_m)[-1]), points))):
        with pytest.raises(ValueError, match="outlet_inflow must be one of"):
            chamber_type(*args, outlet_inflow="total-pressure")


# ---------------------------------------------------------------------------------------------- 2. the split closed
def exchange_velocity(grid, amplitude, top_k=duct.NZ):
    """A discrete streamfunction psi(x face, z face) = A sin(pi x / L) cos(pi / 2 (z - z0) / H), zero on the side walls
    and from the z face top_k up (H = its height): u = d psi / dz, w = -d psi / dx on the MAC faces, divergence-free
    cell by cell and zero from cell top_k up; at the outlet (z0) w > 0 on the low-x half and < 0 on the high-x half
    with zero net.  Scaled to max |w at the outlet| = amplitude."""
    fx, fz = np.asarray(grid.cell_faces_x_m, np.float64), np.asarray(grid.cell_faces_z_m, np.float64)
    length, height = fx[-1] - fx[0], fz[top_k] - fz[0]
    psi = np.sin(math.pi * (fx[:, None] - fx[0]) / length) * np.cos(0.5 * math.pi * (fz[None, :] - fz[0]) / height)
    psi[:, top_k:] = 0.0
    u_x = np.diff(psi, axis=1) / np.diff(fz)[None, :]              # x faces 0 .. NX, z cells
    w = -np.diff(psi, axis=0) / np.diff(fx)[:, None]               # x cells, z faces 0 .. NZ
    scale = amplitude / np.abs(w[:, 0]).max()
    u = np.zeros((duct.NX, duct.NY, duct.NZ, 3), np.float64)
    u[:, :, :, 0] = scale * u_x[:-1, None, :]
    u[:, :, :, 2] = scale * w[:, None, :-1]
    return u.astype(np.float32)


def step(fluid, chamber, p_in, kwargs, t):
    chamber.update(t + 0.5 * duct.DT, p_in)
    fluid.predict(dt_s=duct.DT, advection_scheme="muscl_tvd", pressure_outlet_zmin=True)
    fluid.project(dt_s=duct.DT, **kwargs)
    chamber.after_project(duct.DT, p_in)
    return t + duct.DT


def test_the_split_stays_closed_in_strict_mode(monkeypatch):
    monkeypatch.setenv("FASTFSI_STRICT_INVARIANTS", "1")
    fluid, grid = still_rig()
    obstacle = np.zeros((duct.NX, duct.NY, duct.NZ), np.int32)
    obstacle[:, :, duct.K_AIR - 1] = 1                              # the still tank's membrane
    duct.set_obstacle(fluid, obstacle)
    u = exchange_velocity(grid, EXCHANGE_W, top_k=duct.K_AIR - 1)   # the water below the membrane only
    fluid.velocity.from_numpy(u)
    fluid.velocity_prev.from_numpy(u)
    chamber = PressureRegions(fluid, BOX, outlet_inflow="total_pressure")
    assert chamber.outlet_guard.strict
    codes = np.zeros((duct.NX, duct.NY, duct.NZ), np.int32)
    codes[:, :, duct.K_AIR:] = 1
    chamber.set_regions(codes, [])
    kwargs = merge_project_requirements(chamber, DRIVER_KWARGS)
    t, ratios, first = 0.0, [], None
    for n in range(1, 31):
        t = step(fluid, chamber, P_IN, kwargs, t)
        if n % 10 == 0:
            report = chamber.report()                                 # strict: raises on a violation
            ratios.append(report["outlet_guard_ratio"])
            first = first or report
    print("OUTLET_INFLOW strict split: guard ratios %s; first report inflow %.4e outflow %.4e net %.4e m3/s, max "
          "inflow w %.4f m/s" % (["%.2e" % r for r in ratios], first["outlet_inflow_m3s"], first["outlet_outflow_m3s"],
                                 first["outlet_flux_m3s"], first["outlet_inflow_w_max_mps"]))
    assert max(ratios) <= 1.0e-4
    assert first["outlet_inflow"] == "total_pressure"
    assert first["outlet_inflow_m3s"] > 0.0 > first["outlet_outflow_m3s"] and first["outlet_inflow_w_max_mps"] > 0.0
    assert math.isclose(first["outlet_inflow_m3s"] + first["outlet_outflow_m3s"], first["outlet_flux_m3s"],
                        rel_tol=1.0e-9, abs_tol=1.0e-15)


# ------------------------------------------------------------------------------------------------------ 3. mechanism
def kinetic_energy_and_upflow(fluid, grid):
    """KE = rho / 2 sum |u_c|^2 V_c with u_c the mean of each cell's two faces (walls 0, the outlet face as stored);
    up-flow = sum max(w, 0) A over the outlet faces."""
    u = fluid.velocity.to_numpy().astype(np.float64)
    wx, wy, wz = (np.asarray(w, np.float64) for w in (grid.cell_widths_x_m, grid.cell_widths_y_m, grid.cell_widths_z_m))
    centre = np.zeros(u.shape)
    for axis in range(3):
        low = u[..., axis]
        high = np.zeros_like(low)
        index, source = [slice(None)] * 3, [slice(None)] * 3
        index[axis], source[axis] = slice(0, -1), slice(1, None)
        high[tuple(index)] = low[tuple(source)]
        if axis < 2:                                                  # the low face of cell 0 is a side wall
            low = low.copy()
            first = [slice(None)] * 3
            first[axis] = 0
            low[tuple(first)] = 0.0
        centre[..., axis] = 0.5 * (low + high)
    volume = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    ke = 0.5 * float(fluid.rho) * float(((centre ** 2).sum(axis=-1) * volume).sum())
    area = wx[:, None] * wy[None, :]
    return ke, float((np.maximum(u[:, :, 0, 2], 0.0) * area).sum())


def run_exchange(mode):
    fluid, grid = still_rig()
    u = exchange_velocity(grid, EXCHANGE_W)
    fluid.velocity.from_numpy(u)
    fluid.velocity_prev.from_numpy(u)
    chamber = PressureRegions(fluid, BOX, outlet_inflow=mode)
    chamber.set_regions(np.zeros((duct.NX, duct.NY, duct.NZ), np.int32), [])
    kwargs = merge_project_requirements(chamber, DRIVER_KWARGS)
    series, t = [(0,) + kinetic_energy_and_upflow(fluid, grid)], 0.0
    for n in range(1, MECHANISM_STEPS + 1):
        t = step(fluid, chamber, 0.0, kwargs, t)
        if n % MECHANISM_EVERY == 0:
            series.append((n,) + kinetic_energy_and_upflow(fluid, grid))
    guard = chamber.report()["outlet_guard_ratio"]
    return series, guard


def test_total_pressure_does_not_feed_the_outlet_exchange():
    runs = {mode: run_exchange(mode) for mode in OUTLET_INFLOW_MODES}
    print("OUTLET_INFLOW mechanism, every %d steps: step | static KE (J) up-flow (m3/s) | total_pressure KE up-flow"
          % MECHANISM_EVERY)
    for (n, ke_s, up_s), (_, ke_t, up_t) in zip(runs["static"][0], runs["total_pressure"][0]):
        print("  %5d | %.4e %.4e | %.4e %.4e" % (n, ke_s, up_s, ke_t, up_t))
    print("OUTLET_INFLOW mechanism guard ratios: static %.2e, total_pressure %.2e"
          % (runs["static"][1], runs["total_pressure"][1]))
    (_, ke_s, up_s), (_, ke_t, up_t) = runs["static"][0][-1], runs["total_pressure"][0][-1]
    ke_0 = runs["total_pressure"][0][0][1]
    assert runs["static"][0][0] == runs["total_pressure"][0][0]
    assert ke_t <= ke_0
    assert up_t <= up_s and ke_t <= ke_s
