"""REQ_401 fast mode, simulation_core/fast/air_chamber_fast.py (contract 01 §3, Root acceptance conditions 2026-10-12,
REQ_404 per-step path): CPU tests on a real CartesianFluidSolver (8 x 8 x 32, z graded with ratio 1.05, z near 1 m),
driven like cases/turek_fast.py (fv_cg + fv_multigrid, pressure_outlet_zmin, reset_pressure=False, read_report=False)
plus pressure_outlet_backflow_policy="allow" (PROJECT_REQUIREMENTS).

1. Flat duct, air on top, water below, z-min outlet, u* = 0 (the explicit-Dirichlet discrete solution is exactly linear
   in z): physical p = p_air (z - z0) / (z_a - z0) in water, uniform velocity -dt/rho p_air / (z_a - z0) on every water
   face incl. the outlet face, divergence-free in every water cell incl. k = 0, outlet flux, |p~| <= 1e-6 p_air in air.
2. Root condition 2: random u*, non-flat air interface and an obstacle block against an independent explicit-Dirichlet
   reference (scipy sparse direct solve of the same FV system, Dirichlet p = p_air at air-cell centres and p = 0 at the
   outlet faces): physical water pressure and corrected velocities to solver tolerance; the solver's own divergence
   residual (div + s) is ~0 in water and at the outlet and no auto-cleanup pass runs.
3. Root condition 4: CG iterations and MG preconditioner applications with the penalty (c = 1e2 .. 1e8) vs the same
   u* without any penalty (air cells as obstacles: water only; air cells as plain fluid); printed as AIR_PENALTY rows.
4. Air-cell rule (partial footprint, per-column top air-face particle, lid, obstacle) against a numpy rule; the
   column-without-membrane hard stop; p_air(t) from the config equals the sharp path's pressure_schedule_pa.
5. REQ_404: update() + after_project() wall time per step (AIR_CHAMBER_TIMING row).
"""
import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from simulation_core.diagnostics.runtime import TaichiRuntimeConfig
from simulation_core.fast.air_chamber_fast import (DEFAULT_PENALTY_FACTOR, HARD_STOP_CODES, PROJECT_REQUIREMENTS,
                                                   AirChamberFast, AirPressureWaveform, FastModeHardStop)
from simulation_core.fluids.grid import CartesianGrid
from simulation_core.fluids.solver import CartesianFluidSolver
from simulation_core.fluids.spec import FluidDomainSpec

RHO, MU, DT, P_AIR = 1000.0, 1.0e-3, 1.0e-4, 8000.0
NX, NY, NZ = 8, 8, 32
K_AIR = 20                             # flat interface: water k < 20, air k >= 20
CG_TOLERANCE = 1.0e-10
SQUID_CONFIG = os.environ.get("SQUID_SOURCE_CONFIG", "")     # FastFsi export: not distributed
CONFIG_LOAD = {"id": "bc-5", "type": "Pressure",                 # loads[0] of SQUID_CONFIG (copied 2026-10-12)
               "params": {"Pressure": 8000.0, "Direction": "-z", "Direction Switch Period s": 0.0,
                          "Pressure Schedule Times s": [0.0, 1.0, 2.0],
                          "Pressure Schedule Values Pa": [0.0, 8000.0, -8000.0]}}


def duct_grid():
    widths_z = tuple(6.0e-4 * 1.05 ** abs(k - 16) for k in range(NZ))
    return CartesianGrid(bounds_min_m=(0.0, 0.0, 0.98), cell_widths_x_m=(1.0e-3,) * NX,
                         cell_widths_y_m=(1.0e-3,) * NY, cell_widths_z_m=widths_z)


def set_obstacle(fluid, obstacle):
    """The obstacle-change sequence of the integrator's replace_obstacle_cells (fluid_hooks.py)."""
    fluid.obstacle.from_numpy(np.ascontiguousarray(obstacle, dtype=np.int32))
    fluid._invalidate_pressure_nullspace_component_graph()
    fluid._invalidate_external_obstacle_topology_derived_state()
    fluid.prepare_and_seal_velocity_dirichlet_component_ledger()
    fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(pressure_outlet_zmin=True)


_STATE = {}


def rig():
    """One solver and one full-footprint air chamber per session (Taichi compiles per instance)."""
    if "fluid" not in _STATE:
        grid = duct_grid()
        spec = FluidDomainSpec(bounds_min_m=grid.bounds_min_m, bounds_max_m=grid.bounds_max_m,
                               grid_nodes=grid.grid_nodes, density_kgm3=RHO, viscosity_pa_s=MU, dt_s=DT,
                               cartesian_grid=grid)
        fluid = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cpu", default_fp="f32"),
                                     multigrid_semi_coarsening=True)
        shape = grid.grid_nodes
        fluid.velocity.from_numpy(np.zeros(shape + (3,), np.float32))
        fluid.velocity_prev.from_numpy(np.zeros(shape + (3,), np.float32))
        fluid.pressure.from_numpy(np.zeros(shape, np.float64))
        fluid.obstacle.from_numpy(np.zeros(shape, np.int32))
        fluid.clear_volume_source()
        fluid.set_velocity_dirichlet_boundary_authority("canonical")
        fluid.prepare_and_seal_velocity_dirichlet_component_ledger()
        fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(pressure_outlet_zmin=True)
        body = make_body(np.full((NX, NY), K_AIR - 1))
        chamber = SimpleNamespace(axis_xy_m=(0.004, 0.004), inner_radius_m=1.0,
                                  lid_z_m=float(grid.bounds_max_m[2]) + 1.0e-3)
        _STATE.update(fluid=fluid, grid=grid, body=body, air=AirChamberFast(fluid, None, chamber, body))
    return _STATE["fluid"], _STATE["grid"], _STATE["body"], _STATE["air"]


def make_body(top_cell, spare=4):
    """One tagged (air-face) particle per column inside cell top_cell[i, j], above its centre, plus untagged
    particles below (they must not count); `spare` extra slots for later edits."""
    grid = duct_grid()
    fx, fy, fz = (np.asarray(f) for f in (grid.cell_faces_x_m, grid.cell_faces_y_m, grid.cell_faces_z_m))
    cx, cy = 0.5 * (fx[1:] + fx[:-1]), 0.5 * (fy[1:] + fy[:-1])
    rows, tags = [], []
    for i in range(NX):
        for j in range(NY):
            k = int(top_cell[i, j])
            rows.append([cx[i], cy[j], fz[k + 1] - 0.25 * (fz[k + 1] - fz[k])])
            tags.append(1)
            rows.append([cx[i] + 2e-4, cy[j], fz[k + 1] + 3.0 * (fz[k + 1] - fz[k])])   # high but untagged
            tags.append(2)
    rows += [[cx[0], cy[0], fz[0]]] * spare
    tags += [0] * spare
    n = len(rows)
    body = SimpleNamespace(n=n, x=ti.Vector.field(3, ti.f32, shape=n), tag=ti.field(ti.i32, shape=n))
    body.x.from_numpy(np.asarray(rows, np.float32))
    body.tag.from_numpy(np.asarray(tags, np.int32))
    return body


def project(fluid, *, tolerance=CG_TOLERANCE, iterations=400, policy="raise"):
    fluid.project(iterations=iterations, hibm_pressure_reachability_prepared=True, dt_s=DT, reset_pressure=False,
                  cg_tolerance=tolerance, pressure_solve_failure_policy=policy, read_report=False,
                  **{k: v for k, v in PROJECT_REQUIREMENTS.items()})
    return dict(cg_iterations=int(fluid.last_project_cg_iterations_total),
                mg_applies=int(fluid.last_project_cg_multigrid_apply_count),
                solves=int(fluid.last_project_cg_project_calls),
                preconditioner=str(fluid.last_project_cg_preconditioner_effective),
                residual=float(fluid.last_project_cg_relative_residual_max))


def residual_after_16(fluid, u_star):
    """The solver checks the CG residual every 16 iterations (solver.py 28932), so converged counts are multiples of
    16; the relative residual after exactly 16 PCG iterations from p = 0 resolves smaller differences."""
    fluid.velocity.from_numpy(u_star)
    fluid.pressure.from_numpy(np.zeros((NX, NY, NZ), np.float64))
    counts = project(fluid, tolerance=1.0e-14, iterations=16, policy="report")
    return counts["residual"] if counts["solves"] == 1 else float("nan")


def random_u_star(seed, obstacle):
    """Random predicted velocity, zero on the closed-wall faces, the outlet faces and every face touching an
    obstacle (the solver clamps those)."""
    rng = np.random.default_rng(seed)
    u = (rng.normal(size=(NX, NY, NZ, 3)) * 0.01).astype(np.float32)
    u[0, :, :, 0] = 0.0
    u[:, 0, :, 1] = 0.0
    u[:, :, 0, 2] = 0.0
    solid = obstacle != 0
    for axis in range(3):
        touch = solid.copy()
        shifted = np.zeros_like(solid)
        index = [slice(None)] * 3
        index[axis] = slice(1, None)
        source = [slice(None)] * 3
        source[axis] = slice(0, -1)
        shifted[tuple(index)] = solid[tuple(source)]
        touch |= shifted
        u[..., axis][touch] = 0.0
    return u


def dirichlet_reference(grid, obstacle, air, u_star, p_air):
    """Explicit-Dirichlet FV reference (scipy, float64): water cells unknown, p = p_air at air-cell centres, p = 0 at
    the outlet faces (2 / dz0^2), closed walls; returns physical p (water + air) and the projected face velocities."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    widths = [np.asarray(grid.cell_widths_x_m), np.asarray(grid.cell_widths_y_m), np.asarray(grid.cell_widths_z_m)]
    distance = [np.asarray(grid.center_distances_x_m), np.asarray(grid.center_distances_y_m),
                np.asarray(grid.center_distances_z_m)]
    water = (obstacle == 0) & (air == 0)
    index = -np.ones(obstacle.shape, np.int64)
    index[water] = np.arange(int(water.sum()))
    u = u_star.astype(np.float64)
    rows, cols, vals = [], [], []
    b = np.zeros(int(water.sum()))
    scale = RHO / DT
    for (i, j, k) in zip(*np.nonzero(water)):
        cell = (i, j, k)
        c = index[cell]
        divergence = 0.0
        diagonal = 0.0
        for axis in range(3):
            lo = u[i, j, k, axis] if cell[axis] > 0 else 0.0
            hi_cell = list(cell)
            hi_cell[axis] += 1
            hi = u[tuple(hi_cell) + (axis,)] if hi_cell[axis] < obstacle.shape[axis] else 0.0
            divergence += (hi - lo) / widths[axis][cell[axis]]
            for side in (-1, 1):
                nb = list(cell)
                nb[axis] += side
                if not 0 <= nb[axis] < obstacle.shape[axis] or obstacle[tuple(nb)] != 0:
                    continue
                face = cell[axis] if side < 0 else cell[axis] + 1
                coefficient = 1.0 / (widths[axis][cell[axis]] * distance[axis][face])
                diagonal += coefficient
                if water[tuple(nb)]:
                    rows.append(c)
                    cols.append(index[tuple(nb)])
                    vals.append(coefficient)
                else:
                    b[c] -= coefficient * p_air
        if k == 0:
            diagonal += 2.0 / widths[2][0] ** 2
        rows.append(c)
        cols.append(c)
        vals.append(-diagonal)
        b[c] += scale * divergence
    a = coo_matrix((vals, (rows, cols)), shape=(len(b), len(b))).tocsr()
    p = np.where(air != 0, p_air, 0.0)
    p[water] = spsolve(a, b)
    velocity = u.copy()
    for axis in range(3):
        for (i, j, k) in zip(*np.nonzero(obstacle == 0)):
            cell = (i, j, k)
            if cell[axis] == 0:
                if axis == 2 and water[cell]:
                    velocity[i, j, k, 2] = u[i, j, k, 2] - DT / RHO * 2.0 * p[cell] / widths[2][0]
                continue
            nb = list(cell)
            nb[axis] -= 1
            if obstacle[tuple(nb)] != 0:
                continue
            velocity[i, j, k, axis] = u[i, j, k, axis] - DT / RHO * (p[cell] - p[tuple(nb)]) / distance[axis][cell[axis]]
    return p, velocity, water


def water_faces(obstacle, water):
    """Faces (component, i, j, k) with at least one water neighbour and no obstacle neighbour, incl. the outlet."""
    mask = np.zeros((NX, NY, NZ, 3), bool)
    for axis in range(3):
        here = water.copy()
        below = np.zeros_like(water)
        solid_below = np.zeros_like(water)
        index = [slice(None)] * 3
        index[axis] = slice(1, None)
        source = [slice(None)] * 3
        source[axis] = slice(0, -1)
        below[tuple(index)] = water[tuple(source)]
        solid_below[tuple(index)] = obstacle[tuple(source)] != 0
        interior = np.zeros_like(water)
        interior[tuple(index)] = True
        ok = (here | below) & (obstacle == 0) & ~solid_below & interior
        if axis == 2:
            ok[:, :, 0] = water[:, :, 0]                                       # the outlet faces
        mask[..., axis] = ok
    return mask


def cell_divergence(velocity, grid):
    widths = [np.asarray(grid.cell_widths_x_m), np.asarray(grid.cell_widths_y_m), np.asarray(grid.cell_widths_z_m)]
    u = velocity.astype(np.float64)
    div = np.zeros((NX, NY, NZ))
    for axis in range(3):
        lo = u[..., axis].copy()
        if axis != 2:                                    # closed walls at x-min / y-min; z-min is the outlet
            index = [slice(None)] * 3
            index[axis] = 0
            lo[tuple(index)] = 0.0
        hi = np.zeros_like(lo)
        index = [slice(None)] * 3
        index[axis] = slice(0, -1)
        source = [slice(None)] * 3
        source[axis] = slice(1, None)
        hi[tuple(index)] = u[..., axis][tuple(source)]
        shape = [1, 1, 1]
        shape[axis] = -1
        div += (hi - lo) / widths[axis].reshape(shape)
    return div


# ---------------------------------------------------------------------------------------------------- tests
def test_waveform_from_config_equals_the_sharp_path_schedule():
    from cases.squid_setup.schedules import pressure_schedule_pa
    wave = AirPressureWaveform.from_config({"loads": [CONFIG_LOAD]})
    for t in np.linspace(-0.5, 2.5, 301):
        assert wave(t) == pytest.approx(pressure_schedule_pa(float(t)), abs=1e-9)
    assert wave(0.5) == pytest.approx(4000.0) and wave(1.5) == pytest.approx(0.0, abs=1e-9)
    assert wave(3.0) == -8000.0 and wave(-1.0) == 0.0
    if os.path.exists(SQUID_CONFIG):
        with open(SQUID_CONFIG) as stream:
            real = AirPressureWaveform.from_config(json.load(stream))
        np.testing.assert_array_equal(real.times_s, wave.times_s)
        np.testing.assert_array_equal(real.values_pa, wave.values_pa)
    bad = json.loads(json.dumps(CONFIG_LOAD))
    bad["params"]["Direction"] = "+z"
    with pytest.raises(ValueError):
        AirPressureWaveform.from_config({"loads": [bad]})
    with pytest.raises(ValueError):
        AirPressureWaveform([0.0, 1.0, 1.0], [0.0, 1.0, 2.0])


def test_flat_duct_linear_pressure_uniform_velocity_and_outlet_flux():
    fluid, grid, body, air = rig()
    set_obstacle(fluid, np.zeros((NX, NY, NZ), np.int32))
    fluid.velocity.from_numpy(np.zeros((NX, NY, NZ, 3), np.float32))
    fluid.pressure.from_numpy(np.zeros((NX, NY, NZ), np.float64))
    air.update(0.5, P_AIR)
    counts = project(fluid)
    air.after_project(DT, P_AIR)
    report = air.report(divergence=True)
    faces_z = np.asarray(grid.cell_faces_z_m)
    centres_z = 0.5 * (faces_z[1:] + faces_z[:-1])
    z0, z_air = faces_z[0], centres_z[K_AIR]
    p_tilde = fluid.pressure.to_numpy()
    physical = p_tilde + P_AIR
    expected = P_AIR * (centres_z - z0) / (z_air - z0)
    water = np.arange(NZ) < K_AIR
    error = np.abs(physical[:, :, water] - expected[None, None, water]).max()
    air_max = np.abs(p_tilde[:, :, ~water]).max()
    velocity = fluid.velocity.to_numpy().astype(np.float64)
    u_expected = -DT / RHO * P_AIR / (z_air - z0)
    uz_error = np.abs(velocity[:, :, :K_AIR + 1, 2] - u_expected).max()
    uxy = np.abs(velocity[1:, :, :K_AIR, 0]).max() + np.abs(velocity[:, 1:, :K_AIR, 1]).max()
    div = cell_divergence(velocity, grid)[:, :, :K_AIR]
    flux = velocity[:, :, 0, 2].sum() * 1.0e-6
    print("AIR_DUCT_FLAT c=%.0e cg=%d mg=%d solves=%d max|p-p_ref|/p_air=%.3e max|p~_air|/p_air=%.3e "
          "max|uz-u_ref|/|u_ref|=%.3e max|uxy|=%.3e max|div|*dz/|u|=%.3e outlet_flux=%.6e expected=%.6e report=%s"
          % (air.penalty_factor, counts["cg_iterations"], counts["mg_applies"], counts["solves"], error / P_AIR,
             air_max / P_AIR, uz_error / abs(u_expected), uxy, np.abs(div).max() * 6e-4 / abs(u_expected),
             flux, u_expected * 64e-6, json.dumps(report)))
    # u* = 0: the pre-projection divergence is 0, so the solver's auto-cleanup target is its absolute floor
    # (PRESSURE_OUTLET_AUTO_CLEANUP_MIN_L2 = 1e-8) and the f32 velocity roundoff alone triggers one cleanup pass; with
    # a nonzero u* (next test) no cleanup runs.  The cleanup increment is homogeneous (ghost 0), so it is harmless.
    assert counts["solves"] <= 2 and counts["preconditioner"] == "fv_multigrid"
    assert air_max <= 1.0e-6 * P_AIR                                   # Root condition 1
    assert report["p_tilde_air_max_abs_over_p_air"] <= 1.0e-6
    assert error <= 1.0e-6 * P_AIR
    assert uz_error <= 1.0e-5 * abs(u_expected) and uxy <= 1.0e-5 * abs(u_expected)
    assert np.abs(div).max() <= 1.0e-4 * abs(u_expected) / 6e-4        # f32 velocity storage
    assert flux == pytest.approx(u_expected * 64e-6, rel=1e-5)
    assert report["outlet_flux_m3s"] == pytest.approx(flux, rel=1e-6)
    assert report["air_cell_count"] == NX * NY * (NZ - K_AIR)
    assert report["divergence_residual_water_max_1ps"] <= 1.0e-4 * abs(u_expected) / 6e-4
    assert report["divergence_residual_outlet_max_1ps"] <= 1.0e-4 * abs(u_expected) / 6e-4
    assert report["water_pressure_max_pa"] == pytest.approx(expected[K_AIR - 1], rel=1e-6)
    pressure = air.physical_pressure_field().to_numpy()
    np.testing.assert_allclose(pressure[:, :, water], physical[:, :, water], rtol=0, atol=1e-9 * P_AIR)


def nonflat_case():
    top = np.full((NX, NY), K_AIR - 1)
    top[2:6, 3:7] = K_AIR - 2                          # a lower membrane patch
    obstacle = np.zeros((NX, NY, NZ), np.int32)
    obstacle[2:5, 2:5, 6:9] = 1                        # a block in the water
    return top, obstacle


def test_random_u_star_matches_the_explicit_dirichlet_reference_and_penalty_sweep():
    fluid, grid, body, air = rig()
    top, obstacle = nonflat_case()
    nonflat = make_body(top)
    body.x.from_numpy(nonflat.x.to_numpy())
    body.tag.from_numpy(nonflat.tag.to_numpy())
    set_obstacle(fluid, obstacle)
    u_star = random_u_star(3, obstacle)
    rows = []
    try:
        for penalty in (1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8):
            air.penalty_factor = penalty
            fluid.velocity.from_numpy(u_star)
            fluid.pressure.from_numpy(np.zeros((NX, NY, NZ), np.float64))
            air.update(1.0, P_AIR)
            counts = project(fluid)
            air.after_project(DT, P_AIR)
            report = air.report(divergence=True)
            air_mask = air.air_cell.to_numpy()
            p_ref, u_ref, water = dirichlet_reference(grid, obstacle, air_mask, u_star, P_AIR)
            p_fast = fluid.pressure.to_numpy() + P_AIR
            u_fast = fluid.velocity.to_numpy().astype(np.float64)
            faces = water_faces(obstacle, water)
            p_error = np.abs(p_fast[water] - p_ref[water]).max() / P_AIR
            u_scale = np.abs(u_ref[faces]).max()
            u_error = np.abs(u_fast[faces] - u_ref[faces]).max() / u_scale
            p_air_error = np.abs(fluid.pressure.to_numpy()[air_mask != 0]).max() / P_AIR
            probe = residual_after_16(fluid, u_star)
            rows.append(dict(c=penalty, p_error=p_error, u_error=u_error, p_air=p_air_error, **counts,
                             div_water=report["divergence_residual_water_max_1ps"],
                             div_outlet=report["divergence_residual_outlet_max_1ps"], u_scale=u_scale,
                             residual16=probe))
            print("AIR_PENALTY c=%.0e cg=%d mg=%d solves=%d residual_after_16=%.3e max|p-p_ref|/p_air=%.3e "
                  "max|u-u_ref|/max|u_ref|=%.3e max|p~_air|/p_air=%.3e max|div+s|_water=%.3e "
                  "max|div+s|_outlet=%.3e (1/s; |u|/dz=%.3e)"
                  % (penalty, counts["cg_iterations"], counts["mg_applies"], counts["solves"], probe, p_error,
                     u_error, p_air_error, rows[-1]["div_water"], rows[-1]["div_outlet"], u_scale / 6e-4))
        # baselines without any penalty: the same u*, (a) air cells as obstacles (water only, closed top),
        # (b) air cells as plain fluid (closed top); no outlet shift
        air_mask = air.air_cell.to_numpy()
        for name, extra_obstacle in (("air_as_obstacle", air_mask), ("air_as_fluid", np.zeros_like(air_mask))):
            set_obstacle(fluid, np.maximum(obstacle, extra_obstacle))
            fluid.pressure_interface_matrix_diagonal.from_numpy(np.zeros((NX, NY, NZ), np.float64))
            fluid.pressure_interface_matrix_rhs.from_numpy(np.zeros((NX, NY, NZ), np.float64))
            baseline_u = random_u_star(3, np.maximum(obstacle, extra_obstacle))
            fluid.velocity.from_numpy(baseline_u)
            fluid.pressure.from_numpy(np.zeros((NX, NY, NZ), np.float64))
            counts = project(fluid)
            probe = residual_after_16(fluid, baseline_u)
            rows.append(dict(c=name, residual16=probe, **counts))
            print("AIR_PENALTY baseline=%s cg=%d mg=%d solves=%d residual_after_16=%.3e (no interface diagonal / rhs)"
                  % (name, counts["cg_iterations"], counts["mg_applies"], counts["solves"], probe))
    finally:
        flat = make_body(np.full((NX, NY), K_AIR - 1))
        body.x.from_numpy(flat.x.to_numpy())
        body.tag.from_numpy(flat.tag.to_numpy())
        air.penalty_factor = DEFAULT_PENALTY_FACTOR
        set_obstacle(fluid, np.zeros((NX, NY, NZ), np.int32))
    chosen = [r for r in rows if r["c"] == DEFAULT_PENALTY_FACTOR][0]
    baseline = [r for r in rows if r["c"] == "air_as_obstacle"][0]
    assert chosen["p_air"] <= 1.0e-6                                   # Root condition 1
    assert chosen["p_error"] <= 1.0e-6 and chosen["u_error"] <= 1.0e-5   # Root condition 2
    assert chosen["solves"] == 1                                       # no auto-cleanup pass
    assert chosen["div_water"] <= 1.0e-4 * chosen["u_scale"] / 6e-4
    assert chosen["div_outlet"] <= 1.0e-4 * chosen["u_scale"] / 6e-4
    assert chosen["cg_iterations"] <= baseline["cg_iterations"]        # Root condition 4 (16-iteration granularity)
    assert chosen["residual16"] <= 10.0 * baseline["residual16"]       # and the residual after 16 iterations


def test_air_cell_rule_partial_footprint_and_hard_stops():
    fluid, grid, _, _ = rig()
    set_obstacle(fluid, np.zeros((NX, NY, NZ), np.int32))
    top = np.full((NX, NY), K_AIR - 1)
    top[1:4, 5:8] = K_AIR - 4
    top[6, 1] = K_AIR + 3
    body = make_body(top)
    lid = float(np.asarray(grid.cell_faces_z_m)[NZ - 2]) + 1e-5          # the last full cell row is above the lid
    chamber = SimpleNamespace(axis_xy_m=(0.0045, 0.0035), inner_radius_m=0.0032, lid_z_m=lid)
    obstacle = np.zeros((NX, NY, NZ), np.int32)
    obstacle[4, 4, 25:27] = 1
    set_obstacle(fluid, obstacle)
    try:
        air = AirChamberFast(fluid, None, chamber, body, penalty_factor=1.0e6)
        air.update(0.25, 2000.0)
        mask = air.air_cell.to_numpy()
        faces = [np.asarray(f) for f in (grid.cell_faces_x_m, grid.cell_faces_y_m, grid.cell_faces_z_m)]
        cx, cy, cz = (0.5 * (f[1:] + f[:-1]) for f in faces)
        footprint = np.hypot(cx[:, None] - 0.0045, cy[None, :] - 0.0035) < 0.0032
        top_z = faces[2][top + 1] - 0.25 * (faces[2][top + 1] - faces[2][top])
        expected = (footprint[:, :, None] & (cz[None, None, :] > top_z[:, :, None]) & (cz[None, None, :] < lid)
                    & (obstacle == 0)).astype(np.int32)
        assert 0 < footprint.sum() < NX * NY and expected.sum() > 0
        np.testing.assert_array_equal(mask, expected)
        diagonal = fluid.pressure_interface_matrix_diagonal.to_numpy()
        rhs = fluid.pressure_interface_matrix_rhs.to_numpy()
        assert np.all((diagonal > 0.0) == (expected != 0)) and np.all(rhs[:, :, 1:] == 0.0)
        dz0 = float(grid.cell_widths_z_m[0])
        dz0_f32 = float(np.float32(dz0))                                # the solver's width fields are f32
        np.testing.assert_allclose(rhs[:, :, 0], 2.0 / dz0_f32 ** 2 * (-2000.0), rtol=1e-12)
        report = air.report()
        assert report["air_cell_count"] == int(expected.sum()) and report["p_air_pa"] == 2000.0
        with pytest.raises(ValueError):
            air.after_project(DT, 2500.0)                              # p_air must match the update
        # a footprint column whose air-face particle is gone: device event, hard stop at report()
        hole = int(np.flatnonzero(footprint.ravel())[0])
        tags = body.tag.to_numpy()
        tags[2 * hole] = 0
        body.tag.from_numpy(tags)
        air.update(0.5, 2000.0)
        with pytest.raises(FastModeHardStop) as stop:
            air.report()
        assert stop.value.code == "air_chamber_column_without_membrane"
        assert stop.value.witness["first_column_ij"] == [hole // NY, hole % NY]
        assert stop.value.witness["first_t_s"] == 0.5
        assert {"air_chamber_column_without_membrane", "air_chamber_no_air_cells"} <= HARD_STOP_CODES
    finally:
        set_obstacle(fluid, np.zeros((NX, NY, NZ), np.int32))


def test_per_step_device_path_timing():
    fluid, grid, body, air = rig()
    fluid.velocity.from_numpy(np.zeros((NX, NY, NZ, 3), np.float32))
    air.update(0.1, 800.0)
    air.after_project(DT, 800.0)
    ti.sync()
    steps = 50
    start = time.perf_counter()
    for step in range(steps):
        air.update(0.1 + step * DT, 800.0 + step)
        air.after_project(DT, 800.0 + step)
    ti.sync()
    per_step_ms = (time.perf_counter() - start) / steps * 1e3
    print("AIR_CHAMBER_TIMING cells=%d particles=%d update+after_project=%.3f ms/step (CPU, 1 thread, opt_level 0)"
          % (NX * NY * NZ, body.n, per_step_ms))
    assert math.isfinite(per_step_ms)
