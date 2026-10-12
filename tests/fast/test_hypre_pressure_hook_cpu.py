"""REQ_407 fast mode, HyprePressureHook (simulation_core/fast/hypre_pressure.py): fluid.project() with the linear solve
replaced by hypre Struct PCG + PFMG, against the production FV-CG on the SAME state (CPU, WSL hypre build; skipped if
absent).  Rig: real CartesianFluidSolver 8 x 8 x 24 (z graded, stretch 1.2), obstacle block, z-min outlet, Helper 1's
AirChamberFast (shifted penalty, air on top), random predicted velocity; driven like cases/squid_fast.py.

1. production setting (cg_tolerance 1e-5, PFMG relax 1 / RAP 0 / skip 0 / 1 sweep, tol factor 0.4): the solver's own
   criterion holds on the hypre solution (its CG returns with 0 iterations, converged), the solution agrees with the
   CG's and with a direct solve to the tolerance level, the post-projection divergence obeys the identity
   ||div + s - vs||_V = (dt / rho) ||b - A p||_V (fluid cells) for both, outlet flux / air-cell pressure as the CG;
2. matched residual (cg_tolerance = the CG's achieved relative residual): pressure, velocity, outlet flux and the
   post-projection divergence (L2 and max over water cells) equal the CG's (no worse);
3. data path: upload "matrix" on the first call, nothing + reused PFMG setup when nothing changed, "diagonal" when the
   air cells change, "matrix" again after an obstacle change (integrator's replace_obstacle_cells sequence);
4. fallbacks: a closed domain (no z-min outlet) runs the original CG unchanged (bit-identical to the unhooked run);
5. pointer data path (Taichi ndarray pointers handed to hypre) gives bit-identical results to the host path;
6. u* = 0 duct (triggers one pressure-outlet cleanup pass): both solves of project() go through the hook.
Prints HYPRE_HOOK rows (iterations, residuals, differences, copy bytes / times).
"""
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

hp = pytest.importorskip("simulation_core.fast.hypre_pressure")
if not os.path.exists(os.path.join(os.environ.get(hp.PREFIX_ENV, hp.DEFAULT_PREFIX), "lib", "libHYPRE.so")):
    pytest.skip("hypre build not found", allow_module_level=True)

from simulation_core.diagnostics.runtime import TaichiRuntimeConfig  # noqa: E402
from simulation_core.fast.air_chamber_fast import PROJECT_REQUIREMENTS, AirChamberFast  # noqa: E402
from simulation_core.fluids.grid import CartesianGrid  # noqa: E402
from simulation_core.fluids.solver import CartesianFluidSolver  # noqa: E402
from simulation_core.fluids.spec import FluidDomainSpec  # noqa: E402

RHO, MU, DT, P_AIR, TOL = 1000.0, 1.0e-3, 1.0e-4, 8000.0, 1.0e-5
NX, NY, NZ = 8, 8, 24
K_AIR = 17
_STATE = {}


def grid():
    widths_z = tuple(6.0e-4 * 1.2 ** max(0, abs(k - 12) - 3) for k in range(NZ))
    return CartesianGrid(bounds_min_m=(0.0, 0.0, 0.98), cell_widths_x_m=(1.0e-3,) * NX,
                         cell_widths_y_m=(1.0e-3,) * NY, cell_widths_z_m=widths_z)


def obstacle_mask(extra=False):
    obstacle = np.zeros((NX, NY, NZ), np.int32)
    obstacle[2:5, 3:6, 6:9] = 1
    if extra:
        obstacle[5:7, 1:3, 3:5] = 1
    return obstacle


def set_obstacle(fluid, obstacle):
    """The integrator's replace_obstacle_cells sequence (fluid_hooks.py)."""
    fluid.obstacle.from_numpy(np.ascontiguousarray(obstacle, dtype=np.int32))
    fluid._invalidate_pressure_nullspace_component_graph()
    fluid._invalidate_external_obstacle_topology_derived_state()
    fluid.prepare_and_seal_velocity_dirichlet_component_ledger()
    fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(pressure_outlet_zmin=True)


def make_body(top_cell, g):
    fx, fy, fz = (np.asarray(f) for f in (g.cell_faces_x_m, g.cell_faces_y_m, g.cell_faces_z_m))
    cx, cy = 0.5 * (fx[1:] + fx[:-1]), 0.5 * (fy[1:] + fy[:-1])
    rows = [[cx[i], cy[j], fz[top_cell[i, j] + 1] - 0.25 * (fz[top_cell[i, j] + 1] - fz[top_cell[i, j]])]
            for i in range(NX) for j in range(NY)]
    body = SimpleNamespace(n=len(rows), x=ti.Vector.field(3, ti.f32, shape=len(rows)),
                           tag=ti.field(ti.i32, shape=len(rows)))
    body.x.from_numpy(np.asarray(rows, np.float32))
    body.tag.from_numpy(np.ones(len(rows), np.int32))
    return body


def rig():
    if "fluid" not in _STATE:
        g = grid()
        spec = FluidDomainSpec(bounds_min_m=g.bounds_min_m, bounds_max_m=g.bounds_max_m, grid_nodes=g.grid_nodes,
                               density_kgm3=RHO, viscosity_pa_s=MU, dt_s=DT, cartesian_grid=g)
        fluid = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cpu", default_fp="f32"),
                                     multigrid_semi_coarsening=True)
        shape = g.grid_nodes
        fluid.velocity.from_numpy(np.zeros(shape + (3,), np.float32))
        fluid.velocity_prev.from_numpy(np.zeros(shape + (3,), np.float32))
        fluid.pressure.from_numpy(np.zeros(shape, np.float64))
        fluid.obstacle.from_numpy(obstacle_mask())
        fluid.clear_volume_source()
        fluid.set_velocity_dirichlet_boundary_authority("canonical")
        fluid.prepare_and_seal_velocity_dirichlet_component_ledger()
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(pressure_outlet_zmin=True)
        top = np.full((NX, NY), K_AIR - 1)
        top[2:6, 3:7] = K_AIR - 2
        body = make_body(top, g)
        chamber = SimpleNamespace(axis_xy_m=(0.004, 0.004), inner_radius_m=1.0,
                                  lid_z_m=float(g.bounds_max_m[2]) + 1.0e-3)
        _STATE.update(fluid=fluid, grid=g, body=body, air=AirChamberFast(fluid, None, chamber, body))
    return _STATE["fluid"], _STATE["grid"], _STATE["body"], _STATE["air"]


def u_star(seed, obstacle):
    rng = np.random.default_rng(seed)
    u = (rng.normal(size=(NX, NY, NZ, 3)) * 0.01).astype(np.float32)
    u[0, :, :, 0] = 0.0
    u[:, 0, :, 1] = 0.0
    u[:, :, 0, 2] = 0.0
    solid = obstacle != 0
    for axis in range(3):
        touch = solid.copy()
        index = [slice(None)] * 3
        index[axis] = slice(1, None)
        source = [slice(None)] * 3
        source[axis] = slice(0, -1)
        shifted = np.zeros_like(solid)
        shifted[tuple(index)] = solid[tuple(source)]
        u[..., axis][touch | shifted] = 0.0
    return u


def projection(fluid, air, velocity, pressure, *, tol=TOL, outlet=True, p_air=P_AIR):
    fluid.velocity.from_numpy(velocity)
    fluid.pressure.from_numpy(pressure)
    air.update(0.5, p_air)
    kwargs = dict(PROJECT_REQUIREMENTS)
    if not outlet:
        kwargs.update(pressure_outlet_zmin=False, pressure_outlet_backflow_policy="clamp")
    fluid.project(iterations=400, hibm_pressure_reachability_prepared=outlet, dt_s=DT, reset_pressure=False,
                  cg_tolerance=tol, pressure_solve_failure_policy="raise", read_report=False, **kwargs)
    if outlet:
        air.after_project(DT, p_air)
    p = fluid.pressure.to_numpy()
    u = fluid.velocity.to_numpy()
    fluid.compute_divergence(pressure_outlet_zmin=outlet)
    divergence = (fluid.divergence.to_numpy().astype(np.float64)
                  + fluid.pressure_interface_projection_divergence_s.to_numpy()
                  - fluid.volume_source_s.to_numpy().astype(np.float64))
    return dict(p=p, u=u, divergence=divergence, cg_iterations=int(fluid.last_project_cg_iterations_total),
                cg_relres=float(fluid.last_project_cg_relative_residual_max),
                cg_converged=bool(fluid.last_project_cg_converged_all), solves=int(fluid.last_project_cg_project_calls),
                air=air.air_cell.to_numpy(), outlet_flux=float(u[:, :, 0, 2].astype(np.float64).sum() * 1e-6))


def water_stats(result, obstacle, volume):
    water = (obstacle == 0) & (result["air"] == 0)
    fluid_cells = obstacle == 0
    d = result["divergence"]
    return dict(l2_V_water=float(np.sqrt(np.sum(volume[water] * d[water] ** 2))),
                max_water=float(np.abs(d[water]).max()),
                l2_V_fluid=float(np.sqrt(np.sum(volume[fluid_cells] * d[fluid_cells] ** 2))))


def cell_volume(g):
    w = [np.asarray(g.cell_widths_x_m, np.float32).astype(np.float64),
         np.asarray(g.cell_widths_y_m, np.float32).astype(np.float64),
         np.asarray(g.cell_widths_z_m, np.float32).astype(np.float64)]
    return w[0][:, None, None] * w[1][None, :, None] * w[2][None, None, :]


def residual_norms(fluid, p):
    """b and the residual ||b - A p||_V / ||b||_V of the CURRENT system (cg_rhs of the last solve) for pressure p."""
    export = hp.export_pressure_system(fluid)
    export.update(b=fluid.cg_rhs.to_numpy())
    return export, hp.relative_residual_v(export, p)


def direct(export):
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    fluid = export["obstacle"] == 0
    index = -np.ones(fluid.shape, np.int64)
    index[fluid] = np.arange(int(fluid.sum()))
    rows, cols, vals = [], [], []
    for (i, j, k) in zip(*np.nonzero(fluid)):
        c = index[i, j, k]
        rows.append(c)
        cols.append(c)
        vals.append(export["diag"][i, j, k])
        for axis in range(3):
            for side, slot in ((-1, 2 * axis), (1, 2 * axis + 1)):
                nb = [i, j, k]
                nb[axis] += side
                if 0 <= nb[axis] < fluid.shape[axis] and fluid[tuple(nb)] and export["products"][i, j, k, slot]:
                    rows.append(c)
                    cols.append(index[tuple(nb)])
                    vals.append(-export["products"][i, j, k, slot])
    a = coo_matrix((vals, (rows, cols)), shape=(int(fluid.sum()),) * 2).tocsr()
    p = np.zeros(fluid.shape)
    p[fluid] = spsolve(a, export["b"][fluid])
    return p


def test_production_setting_and_matched_residual_equivalence():
    fluid, g, body, air = rig()
    obstacle = obstacle_mask()
    volume = cell_volume(g)
    velocity = u_star(5, obstacle)
    p0 = np.zeros((NX, NY, NZ))
    cg = projection(fluid, air, velocity, p0)
    export, cg_relres = residual_norms(fluid, cg["p"])
    exact = direct(export)
    hook = hp.HyprePressureHook(fluid).install()
    try:
        hy = projection(fluid, air, velocity, p0)
        last = dict(hook.last)
        _, hy_relres = residual_norms(fluid, hy["p"])
        matched_tol = max(cg_relres, 1e-13) * 1.0001
        hm = projection(fluid, air, velocity, p0, tol=matched_tol)
        _, hm_relres = residual_norms(fluid, hm["p"])
        report = hook.report()
    finally:
        hook.close()
    fluid_cells = obstacle == 0
    scale = np.abs(exact[fluid_cells]).max()
    rows = dict(cg_iterations=cg["cg_iterations"], cg_relres_V=cg_relres, hypre_iterations=last["hypre_iterations"],
                hypre_relres_V=hy_relres, solver_cg_after_hook=hy["cg_iterations"],
                err_cg_vs_exact=float(np.abs(cg["p"] - exact)[fluid_cells].max() / scale),
                err_hypre_vs_exact=float(np.abs(hy["p"] - exact)[fluid_cells].max() / scale),
                diff_hypre_vs_cg=float(np.abs(hy["p"] - cg["p"])[fluid_cells].max() / scale),
                matched_tol=matched_tol, matched_relres_V=hm_relres,
                matched_diff_vs_cg=float(np.abs(hm["p"] - cg["p"])[fluid_cells].max() / scale),
                div_cg=water_stats(cg, obstacle, volume), div_hypre=water_stats(hy, obstacle, volume),
                div_matched=water_stats(hm, obstacle, volume),
                outlet_flux=dict(cg=cg["outlet_flux"], hypre=hy["outlet_flux"], matched=hm["outlet_flux"]),
                last=last, totals=report["totals"])
    print("HYPRE_HOOK_EQUIVALENCE", json.dumps(rows, default=float))
    # 1. production setting: the solver's own criterion confirmed on the hypre solution, no CG iterations needed
    assert hy["cg_converged"] and hy["cg_iterations"] == 0 and last["hypre_iterations"] > 0
    assert last["cg_polish_iterations"] == 0 and last["solver_converged"]
    assert hy_relres <= TOL and cg_relres <= TOL
    assert np.array_equal(hy["air"], cg["air"])
    # both solutions are within the tolerance-level error of the direct solution, and of each other
    assert rows["err_hypre_vs_exact"] <= 1e-3 and rows["err_cg_vs_exact"] <= 1e-3
    assert rows["diff_hypre_vs_cg"] <= 1e-3
    # divergence identity: ||div + s - vs||_V over fluid cells = (dt / rho) ||r||_V  (f32 velocity storage floor)
    rhs_scale = RHO / DT
    b_norm = math.sqrt(float(np.sum(volume[fluid_cells] * export["b"][fluid_cells] ** 2)))
    f32_floor = 1e-6 * float(np.abs(velocity).max()) / 6e-4 * math.sqrt(float(volume[fluid_cells].sum()))
    for result, relres in ((cg, cg_relres), (hy, hy_relres)):
        expected = relres * b_norm / rhs_scale
        assert abs(water_stats(result, obstacle, volume)["l2_V_fluid"] - expected) <= 0.05 * expected + f32_floor
    # 2. matched residual: pressure, outlet flux and divergence as the CG (no worse)
    assert hm_relres <= matched_tol
    assert rows["matched_diff_vs_cg"] <= 10.0 * matched_tol + 1e-9
    assert rows["div_matched"]["l2_V_water"] <= rows["div_cg"]["l2_V_water"] * 1.05 + f32_floor
    assert rows["div_matched"]["max_water"] <= rows["div_cg"]["max_water"] * 1.05 + 1e-6 * f32_floor
    assert abs(hm["outlet_flux"] - cg["outlet_flux"]) <= 1e-5 * abs(cg["outlet_flux"]) + 1e-12
    assert abs(hy["outlet_flux"] - cg["outlet_flux"]) <= 1e-3 * abs(cg["outlet_flux"]) + 1e-12
    # air cells pinned identically (|p~| tiny, shifted-penalty boundary unchanged)
    air_cells = cg["air"] != 0
    assert np.abs(hy["p"][air_cells]).max() <= 1e-6 * P_AIR and np.abs(cg["p"][air_cells]).max() <= 1e-6 * P_AIR


def test_upload_logic_setup_reuse_and_obstacle_change():
    fluid, g, body, air = rig()
    obstacle = obstacle_mask()
    velocity = u_star(7, obstacle)
    hook = hp.HyprePressureHook(fluid).install()
    events = []
    try:
        projection(fluid, air, velocity, np.zeros((NX, NY, NZ)))
        events.append(dict(hook.last))
        projection(fluid, air, velocity * 0.5, np.zeros((NX, NY, NZ)))     # same matrix, new RHS
        events.append(dict(hook.last))
        x = body.x.to_numpy()
        x[:, 2] -= float(g.cell_widths_z_m[K_AIR - 2])                     # membrane down: air cells change
        body.x.from_numpy(x)
        projection(fluid, air, velocity, np.zeros((NX, NY, NZ)))
        events.append(dict(hook.last))
        set_obstacle(fluid, obstacle_mask(extra=True))                     # mask change (valve-like)
        projection(fluid, air, u_star(7, obstacle_mask(extra=True)), np.zeros((NX, NY, NZ)))
        events.append(dict(hook.last))
        report = hook.report()
    finally:
        hook.close()
        x = body.x.to_numpy()
        x[:, 2] += float(g.cell_widths_z_m[K_AIR - 2])
        body.x.from_numpy(x)
        set_obstacle(fluid, obstacle_mask())
    print("HYPRE_HOOK_UPLOADS", json.dumps(dict(events=events, totals=report["totals"]), default=float))
    assert [e["upload"] for e in events] == ["matrix", "none", "diagonal", "matrix"]
    assert [e["setup"] for e in events] == ["new", "reused", "new", "new"]
    assert all(e["solver_converged"] and e["cg_polish_iterations"] == 0 for e in events)
    # the diagonal upload covers only the box of changed (air) cells: the membrane rows moved by one cell
    (x0, y0, z0), (x1, y1, z1) = events[2]["diagonal_box"]
    assert K_AIR - 3 <= z0 <= z1 <= K_AIR
    cells = NX * NY * NZ

    def expected_h2d(event, extra=0):          # rhs + x0 into hypre, the solution into Taichi per PCG run, + matrix
        return 16 * cells + 8 * cells * (1 + event["restarts"]) + extra

    assert events[1]["h2d_bytes"] == expected_h2d(events[1])
    assert events[2]["h2d_bytes"] == expected_h2d(events[2], 8 * (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1))
    assert events[0]["h2d_bytes"] == expected_h2d(events[0], 32 * cells)


def test_closed_domain_falls_back_to_the_original_cg_bit_identically():
    fluid, g, body, air = rig()
    obstacle = obstacle_mask()
    velocity = u_star(11, obstacle)
    plain = projection(fluid, air, velocity, np.zeros((NX, NY, NZ)), outlet=False)
    hook = hp.HyprePressureHook(fluid).install()
    try:
        hooked = projection(fluid, air, velocity, np.zeros((NX, NY, NZ)), outlet=False)
        fallbacks = dict(hook.totals["fallbacks"])
    finally:
        hook.close()
        set_obstacle(fluid, obstacle_mask())
    assert fallbacks.get("no_zmin_pressure_outlet", 0) >= 1 and hook.totals["hypre_calls"] == 0
    np.testing.assert_array_equal(hooked["p"], plain["p"])
    np.testing.assert_array_equal(hooked["u"], plain["u"])


def test_hypre_error_falls_back_to_the_cg_bit_identically_and_recovers():
    fluid, g, body, air = rig()
    velocity = u_star(17, obstacle_mask())
    plain = projection(fluid, air, velocity, np.zeros((NX, NY, NZ)))
    hook = hp.HyprePressureHook(fluid).install()
    original_run = hook.solver.run
    calls = []

    def failing_run(tol=None):
        calls.append(tol)
        if len(calls) == 1:
            raise hp.FastModeHardStop("hypre_error", dict(call="HYPRE_StructPCGSolve", injected=True))
        return original_run(tol)

    hook.solver.run = failing_run
    try:
        failed = projection(fluid, air, velocity, np.zeros((NX, NY, NZ)))
        first = dict(hook.last)
        recovered = projection(fluid, air, velocity, np.zeros((NX, NY, NZ)))
        second = dict(hook.last)
    finally:
        hook.close()
    assert first["fallback"] == "hypre_error" and first["cg_polish_iterations"] > 0
    assert hook.totals["fallbacks"] == {"hypre_error": 1}
    np.testing.assert_array_equal(failed["p"], plain["p"])               # the CG did the solve from the same guess
    np.testing.assert_array_equal(failed["u"], plain["u"])
    assert second["fallback"] is None and second["upload"] == "matrix" and second["setup"] == "new"
    assert second["cg_polish_iterations"] == 0 and recovered["cg_converged"]


def test_pointer_path_equals_host_path():
    fluid, g, body, air = rig()
    velocity = u_star(13, obstacle_mask())
    results = {}
    for path in ("host", "pointer"):
        hook = hp.HyprePressureHook(fluid, data_path=path).install()
        try:
            results[path] = projection(fluid, air, velocity, np.zeros((NX, NY, NZ)))
            results[path + "_report"] = hook.report()
        finally:
            hook.close()
    assert results["pointer_report"]["data_path"] == "pointer"
    assert results["pointer_report"]["totals"]["h2d_bytes"] == 0
    np.testing.assert_array_equal(results["pointer"]["p"], results["host"]["p"])
    np.testing.assert_array_equal(results["pointer"]["u"], results["host"]["u"])


def test_zero_velocity_duct_cleanup_pass_goes_through_the_hook():
    fluid, g, body, air = rig()
    obstacle = obstacle_mask()
    zero = np.zeros((NX, NY, NZ, 3), np.float32)
    cg = projection(fluid, air, zero, np.zeros((NX, NY, NZ)))
    hook = hp.HyprePressureHook(fluid).install()
    try:
        hy = projection(fluid, air, zero, np.zeros((NX, NY, NZ)))
        totals = dict(hook.totals)
    finally:
        hook.close()
    print("HYPRE_HOOK_DUCT", json.dumps(dict(cg_solves=cg["solves"], hook_solves=hy["solves"], totals=totals,
                                             flux=dict(cg=cg["outlet_flux"], hypre=hy["outlet_flux"])), default=float))
    assert totals["calls"] == hy["solves"] >= 1
    fluid_cells = obstacle == 0
    scale = np.abs(cg["p"][fluid_cells]).max()
    assert np.abs(hy["p"] - cg["p"])[fluid_cells].max() <= 1e-4 * scale
    assert abs(hy["outlet_flux"] - cg["outlet_flux"]) <= 1e-4 * abs(cg["outlet_flux"])
