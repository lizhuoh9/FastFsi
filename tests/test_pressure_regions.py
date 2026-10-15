"""PressureRegions (boundary/pressure_regions.py): per-cell Dirichlet pressures (inlet air at p_in, an air
pocket at its own p_k) on the real solver, against an explicit-Dirichlet reference.

Duct 8 x 8 x 32 (z graded 1.05), z-min outlet, driven like a soft-actuator chamber (fv_cg + fv_multigrid, PROJECT_REQUIREMENTS):
inlet air in the top layers k >= 26, an air pocket block surrounded by water, an obstacle block, random u*.
  1. physical water pressure and corrected water-face velocities equal the scipy explicit-Dirichlet solution;
  2. |p~| in Dirichlet cells is tiny, the water divergence bookkeeping (div + s) closes;
  3. the conjugate-gradient count does not grow with the pocket's pressure jump (1 Pa vs 10 kPa);
  4. without pockets the rows equal ColumnPressureChamber's for the same air cells (R = outlet only)."""
import numpy as np
import pytest
import taichi as ti

from fastfsi.boundary.pressure_regions import PressureRegions
from fastfsi.boundary.pressure_chamber import PROJECT_REQUIREMENTS
from fastfsi.fluid.solver import IncompressibleFluidSolver
from fastfsi.grid.cartesian_grid import CartesianGrid
from fastfsi.grid.specification import FluidDomainSpecification
from fastfsi.simulation.taichi_runtime import TaichiRuntimeConfig

RHO, MU, DT = 1000.0, 1.0e-3, 1.0e-4
NX, NY, NZ = 8, 8, 32
K_AIR = 26
P_IN = -4500.0
CG_TOLERANCE = 1.0e-10
_STATE = {}


def duct_grid():
    widths_z = tuple(6.0e-4 * 1.05 ** abs(k - 16) for k in range(NZ))
    return CartesianGrid(bounds_min_m=(0.0, 0.0, 0.98), cell_widths_x_m=(1.0e-3,) * NX,
                         cell_widths_y_m=(1.0e-3,) * NY, cell_widths_z_m=widths_z)


def set_obstacle(fluid, obstacle):
    fluid.obstacle.from_numpy(np.ascontiguousarray(obstacle, dtype=np.int32))
    fluid._invalidate_pressure_nullspace_component_graph()
    fluid._invalidate_external_obstacle_topology_derived_state()
    fluid.prepare_and_seal_velocity_dirichlet_component_registry()
    fluid.mark_pressure_outlet_disconnected_nonprojectable_cells(pressure_outlet_zmin=True)


def rig():
    if "fluid" not in _STATE:
        from cpu_runtime import initialize          # the tests' deterministic CPU runtime
        initialize()
        grid = duct_grid()
        spec = FluidDomainSpecification(bounds_min_m=grid.bounds_min_m, bounds_max_m=grid.bounds_max_m,
                                        grid_nodes=grid.grid_nodes, density_kgm3=RHO, viscosity_pa_s=MU, dt_s=DT,
                                        cartesian_grid=grid)
        fluid = IncompressibleFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cpu", default_fp="f32"),
                                          multigrid_semi_coarsening=True)
        shape = grid.grid_nodes
        fluid.velocity.from_numpy(np.zeros(shape + (3,), np.float32))
        fluid.velocity_prev.from_numpy(np.zeros(shape + (3,), np.float32))
        fluid.pressure.from_numpy(np.zeros(shape, np.float64))
        fluid.obstacle.from_numpy(np.zeros(shape, np.int32))
        fluid.clear_volume_source()
        fluid.set_velocity_dirichlet_boundary_authority("canonical")
        fluid.prepare_and_seal_velocity_dirichlet_component_registry()
        fluid.mark_pressure_outlet_disconnected_nonprojectable_cells(pressure_outlet_zmin=True)
        chamber = PressureRegions(fluid, ((0, NX), (0, NY), (0, NZ)))
        _STATE.update(fluid=fluid, grid=grid, chamber=chamber)
    return _STATE["fluid"], _STATE["grid"], _STATE["chamber"]


def project(fluid):
    fluid.project(iterations=400, pressure_reachability_prepared=True, dt_s=DT, reset_pressure=False,
                  cg_tolerance=CG_TOLERANCE, pressure_solve_failure_policy="raise", read_report=False,
                  **PROJECT_REQUIREMENTS)
    return int(fluid.last_project_cg_iterations_total), int(fluid.last_project_cg_project_calls)


def obstacle_block():
    obstacle = np.zeros((NX, NY, NZ), np.int32)
    obstacle[5:7, 1:3, 6:9] = 1
    return obstacle


def codes_with_pocket():
    codes = np.zeros((NX, NY, NZ), np.int32)
    codes[:, :, K_AIR:] = 1
    codes[2:5, 2:5, 15:18] = 2                        # pocket 0, surrounded by water
    return codes


def random_u_star(seed, obstacle):
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


def dirichlet_reference(grid, obstacle, values, u_star):
    """Explicit-Dirichlet FV reference (scipy, f64): cells with a finite `values` entry are Dirichlet cells at that
    pressure (cell centre), the outlet faces are at p = 0 (2 / dz0^2), closed walls.  Returns physical p and the face
    velocities corrected with the gradient of p."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    widths = [np.asarray(grid.cell_widths_x_m), np.asarray(grid.cell_widths_y_m), np.asarray(grid.cell_widths_z_m)]
    distance = [np.asarray(grid.center_distances_x_m), np.asarray(grid.center_distances_y_m),
                np.asarray(grid.center_distances_z_m)]
    dirichlet = np.isfinite(values)
    water = (obstacle == 0) & ~dirichlet
    index = -np.ones(obstacle.shape, np.int64)
    index[water] = np.arange(int(water.sum()))
    u = u_star.astype(np.float64)
    rows, cols, vals = [], [], []
    b = np.zeros(int(water.sum()))
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
                    b[c] -= coefficient * values[tuple(nb)]
        if k == 0:
            diagonal += 2.0 / widths[2][0] ** 2
        rows.append(c)
        cols.append(c)
        vals.append(-diagonal)
        b[c] += RHO / DT * divergence
    a = coo_matrix((vals, (rows, cols)), shape=(len(b), len(b))).tocsr()
    p = np.where(dirichlet, values, 0.0)
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
    mask = np.zeros((NX, NY, NZ, 3), bool)
    for axis in range(3):
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
        mask[..., axis] = (water | below) & (obstacle == 0) & ~solid_below & interior
        if axis == 2:
            mask[:, :, 0, 2] = water[:, :, 0]
    return mask


def solve_case(fluid, grid, chamber, codes, pockets_pa, u_star, obstacle):
    chamber.set_regions(codes, pockets_pa)
    fluid.velocity.from_numpy(u_star)
    fluid.pressure.from_numpy(np.zeros((NX, NY, NZ), np.float64))
    chamber.update(0.5, P_IN)
    cg, solves = project(fluid)
    chamber.after_project(DT, P_IN)
    values = np.full((NX, NY, NZ), np.nan)
    values[codes == 1] = P_IN
    for n, p in enumerate(pockets_pa):
        values[codes == 2 + n] = p
    p_ref, u_ref, water = dirichlet_reference(grid, obstacle, values, u_star)
    physical = chamber.physical_pressure_field().to_numpy()
    faces = water_faces(obstacle, water)
    u_fast = fluid.velocity.to_numpy().astype(np.float64)
    scale = max(abs(P_IN), max(abs(p) for p in pockets_pa) if pockets_pa else 0.0)
    return dict(cg=cg, solves=solves, report=chamber.report(),
                p_error=float(np.abs(physical[water] - p_ref[water]).max() / scale),
                u_error=float(np.abs(u_fast[faces] - u_ref[faces]).max() / np.abs(u_ref[faces]).max()),
                p_tilde_dirichlet=float(np.abs(fluid.pressure.to_numpy()[codes > 0]).max() / scale))


def test_inlet_air_and_a_pocket_match_the_explicit_dirichlet_reference():
    fluid, grid, chamber = rig()
    obstacle = obstacle_block()
    set_obstacle(fluid, obstacle)
    u_star = random_u_star(5, obstacle)
    rows = {}
    for dp in (1.0, 3000.0, 10000.0):
        rows[dp] = solve_case(fluid, grid, chamber, codes_with_pocket(), [P_IN + dp], u_star, obstacle)
        print("OCC_CHAMBER dp=%g cg=%d solves=%d p_err=%.3e u_err=%.3e p~_dir=%.3e report=%s"
              % (dp, rows[dp]["cg"], rows[dp]["solves"], rows[dp]["p_error"], rows[dp]["u_error"],
                 rows[dp]["p_tilde_dirichlet"], rows[dp]["report"]))
    for dp, row in rows.items():
        assert row["solves"] == 1
        assert row["p_error"] <= 1.0e-6 and row["u_error"] <= 1.0e-5
        assert row["p_tilde_dirichlet"] <= 1.0e-6
        assert row["report"]["pocket_cells"] == 27 and row["report"]["inlet_cells"] == NX * NY * (NZ - K_AIR)
        assert row["report"]["water_divergence_max_1ps"] <= 1.0e-3
    assert rows[10000.0]["cg"] <= rows[1.0]["cg"] + 16                  # the jump size does not cost iterations


def test_without_pockets_the_rows_are_the_pressure_chamber_rows():
    fluid, grid, chamber = rig()
    obstacle = np.zeros((NX, NY, NZ), np.int32)
    set_obstacle(fluid, obstacle)
    codes = np.zeros((NX, NY, NZ), np.int32)
    codes[:, :, K_AIR:] = 1
    chamber.set_regions(codes, [])
    chamber.update(0.25, P_IN)
    diagonal = fluid.pressure_interface_matrix_diagonal.to_numpy()
    rhs = fluid.pressure_interface_matrix_rhs.to_numpy()
    assert (diagonal[:, :, :K_AIR] == 0.0).all() and (diagonal[:, :, K_AIR:] > 0.0).all()
    assert (rhs[:, :, 1:] == 0.0).all()                                # only the outlet row carries R
    widths_z = np.asarray(duct_grid().cell_widths_z_m)
    assert np.allclose(rhs[:, :, 0], 2.0 / widths_z[0] ** 2 * (-P_IN))
    assert chamber.report()["jump_rows"] == 0


def test_set_regions_validates_codes_and_pressures():
    fluid, grid, chamber = rig()
    codes = np.zeros((NX, NY, NZ), np.int32)
    codes[0, 0, 0] = 3
    with pytest.raises(ValueError):
        chamber.set_regions(codes, [0.0])                              # code 3 = pocket 1, only one pressure
    with pytest.raises(ValueError):
        chamber.set_regions(np.zeros((NX, NY, NZ), np.int32), [np.nan])
    with pytest.raises(ValueError):
        chamber.after_project(DT, P_IN + 1.0) if chamber.last_p_in_pa is not None else chamber.update(0.0, np.inf)
