"""REQ_407 fast mode, simulation_core/fast/hypre_pressure.py on the WSL CPU build of hypre 3.2.0 (skipped when the
library is absent): correctness of the operator mapping and of the stopping criterion.

1. export_pressure_system() reproduces the production solver's CG operator: fv_apply(export, p) equals the solver's own
   _apply_fv_pressure_operator on a real CartesianFluidSolver (graded z, obstacle block, air-chamber penalty diagonal,
   z-min outlet) to roundoff, and V A is symmetric.
2. Mapping: hypre (V form, tight tolerance) agrees with a scipy direct solve of OUR operator on a graded synthetic system
   with obstacle, outlet and penalty cells; the sqrtV form too.
3. Criterion: the V form + solve_to_solver_tolerance always ends with ||r||_V <= tol ||b||_V (the solver's norm); the
   sqrtV form's hypre residual equals ||r||_V / ||b||_V; symmetric and full stencil storage agree.
4. All 16 PFMG sweep combinations and SMG meet the solver's criterion on the graded synthetic system (V form).
5. Raw-pointer path (Taichi ndarray data pointers, the zero-host-copy interop) gives the same solution as numpy arrays.
"""
import os

import numpy as np
import pytest

hp = pytest.importorskip("simulation_core.fast.hypre_pressure")
PREFIX = os.environ.get(hp.PREFIX_ENV, hp.DEFAULT_PREFIX)
if not os.path.exists(os.path.join(PREFIX, "lib", "libHYPRE.so")):
    pytest.skip("hypre build not found at %s" % PREFIX, allow_module_level=True)


def graded_widths(n, h0, ratio, core):
    left = (n - core) // 2
    right = n - core - left
    return np.array([h0 * ratio ** (left - i) for i in range(left)] + [h0] * core
                    + [h0 * ratio ** (i + 1) for i in range(right)])


def synthetic(nx=16, ny=14, nz=24, ratio=1.2, penalty=1.0e7, seed=3):
    """Solver-structured FV system: T(i, i-1) = 1 / (w_i d_i), outlet 2 / dz0^2 at k = 0, penalty diag in an air box."""
    rng = np.random.default_rng(seed)
    widths = [graded_widths(nx, 4e-4, ratio, 4), graded_widths(ny, 4e-4, ratio, 4), graded_widths(nz, 5e-4, ratio, 8)]
    centres = [np.cumsum(w) - 0.5 * w for w in widths]
    distance = [np.concatenate([[w[0]], np.diff(c)]) for w, c in zip(widths, centres)]
    obstacle = np.zeros((nx, ny, nz), np.int8)
    obstacle[5:9, 4:8, 8:12] = 1
    fluid = obstacle == 0
    air = np.zeros_like(fluid)
    air[3:12, 3:11, 18:] = True
    air &= fluid
    volume = widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]
    products = np.zeros((nx, ny, nz, 6))
    diag = np.zeros((nx, ny, nz))
    for axis in range(3):
        shape = [-1 if a == axis else 1 for a in range(3)]
        w = widths[axis].reshape(shape)
        d = distance[axis]
        index = [slice(None)] * 3
        index[axis] = slice(1, None)
        source = [slice(None)] * 3
        source[axis] = slice(0, -1)
        has_minus = np.zeros_like(fluid)
        has_minus[tuple(index)] = fluid[tuple(source)]
        has_plus = np.zeros_like(fluid)
        has_plus[tuple(source)] = fluid[tuple(index)]
        minus = np.where(has_minus & fluid, np.broadcast_to(1.0 / (w * d.reshape(shape)), fluid.shape), 0.0)
        plus = np.where(has_plus & fluid,
                        np.broadcast_to(1.0 / (w * np.concatenate([d[1:], [1.0]]).reshape(shape)), fluid.shape), 0.0)
        products[..., 2 * axis], products[..., 2 * axis + 1] = minus, plus
        diag += minus + plus
    diag[:, :, 0] += np.where(fluid[:, :, 0], 2.0 / widths[2][0] ** 2, 0.0)
    diag = np.where(fluid, np.where(air, diag * (1.0 + penalty), diag), 0.0)
    b = np.where(fluid & ~air, rng.normal(size=fluid.shape) * 1e6, 0.0)
    p0 = np.where(fluid, rng.normal(size=fluid.shape), 0.0)
    return dict(diag=diag, products=products, volume_m3=np.where(fluid, volume, 0.0), obstacle=obstacle, b=b, p0=p0,
                cell_widths_x_m=widths[0], cell_widths_y_m=widths[1], cell_widths_z_m=widths[2])


def direct_solution(export):
    """scipy sparse direct solve of OUR operator (fluid cells)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    fluid = export["obstacle"] == 0
    index = -np.ones(fluid.shape, np.int64)
    index[fluid] = np.arange(int(fluid.sum()))
    rows, cols, vals = [], [], []
    products = export["products"]
    for (i, j, k) in zip(*np.nonzero(fluid)):
        c = index[i, j, k]
        rows.append(c)
        cols.append(c)
        vals.append(export["diag"][i, j, k])
        for axis in range(3):
            for side, slot in ((-1, 2 * axis), (1, 2 * axis + 1)):
                nb = [i, j, k]
                nb[axis] += side
                if 0 <= nb[axis] < fluid.shape[axis] and fluid[tuple(nb)] and products[i, j, k, slot] != 0.0:
                    rows.append(c)
                    cols.append(index[tuple(nb)])
                    vals.append(-products[i, j, k, slot])
    a = coo_matrix((vals, (rows, cols)), shape=(int(fluid.sum()),) * 2).tocsr()
    p = np.zeros(fluid.shape)
    p[fluid] = spsolve(a, export["b"][fluid])
    return p


def solve(export, *, form="V", tol=1e-5, exact=False, **options):
    system = hp.scaled_system(export, form=form)
    solver = hp.StructPressureSolver(system["shape"], prefix=PREFIX, tol=tol, max_iter=2000, **options)
    try:
        solver.set_system(system["values"], system["rhs"], system["x0"])
        unscale = lambda x: np.where(system["fluid"], hp.from_hypre_order(x, system["shape"]) / system["scale"], 0.0)
        if exact or form == "sqrtV":
            result = solver.solve()
        else:
            result = solver.solve_to_solver_tolerance(lambda x: hp.relative_residual_v(export, unscale(x)), tol)
        p = unscale(solver.get_solution())
    finally:
        solver.close()
    return p, result, system


def test_mapping_against_a_direct_solve():
    export = synthetic()
    reference = direct_solution(export)
    scale = np.abs(reference).max()
    for form in ("V", "sqrtV"):
        p, result, system = solve(export, form=form, tol=1e-12, exact=True)
        assert system["asymmetry"] < 1e-13
        assert np.abs(p - reference).max() <= 1e-5 * scale, (form, np.abs(p - reference).max() / scale)
    assert hp.relative_residual_v(export, reference) < 1e-9


def test_v_form_meets_the_solver_criterion_and_sqrtv_norm_is_exact():
    export = synthetic()
    p, result, _ = solve(export, form="V", tol=1e-5)
    assert result["converged"] and hp.relative_residual_v(export, p) <= 1e-5
    assert result["relres_V"] == pytest.approx(hp.relative_residual_v(export, p), rel=1e-9)
    p, result, _ = solve(export, form="sqrtV", tol=1e-5, relax_type=1)
    # same norm by construction; the only difference is PCG's recursively updated residual vs the true residual
    # (measured 2e-5 relative on this penalty-stiff system), far below a norm mismatch (O(1) on a graded grid)
    assert result["final_relative_residual"] == pytest.approx(hp.relative_residual_v(export, p), rel=1e-3)


def test_symmetric_and_full_storage_agree():
    export = synthetic()
    p_sym, _, _ = solve(export, tol=1e-12, exact=True)
    p_full, _, _ = solve(export, tol=1e-12, exact=True, symmetric_storage=False)
    assert np.abs(p_sym - p_full).max() <= 1e-9 * np.abs(p_sym).max()


@pytest.mark.parametrize("combo", list(hp.PFMG_SWEEP) + [dict(solver="smg")])
def test_every_sweep_configuration_meets_the_solver_criterion(combo):
    export = synthetic()
    p, result, _ = solve(export, tol=1e-5, **combo)
    assert result["converged"], result
    assert hp.relative_residual_v(export, p) <= 1e-5


def test_raw_pointer_path_matches_the_array_path():
    import taichi as ti
    export = synthetic()
    system = hp.scaled_system(export)
    n = int(np.prod(system["shape"]))
    values = ti.ndarray(ti.f64, shape=(n, 4))
    rhs = ti.ndarray(ti.f64, shape=n)
    x = ti.ndarray(ti.f64, shape=n)
    values.from_numpy(system["values"])
    rhs.from_numpy(system["rhs"])
    x.from_numpy(system["x0"])
    program = ti.lang.impl.get_runtime().prog
    pointers = [program.get_ndarray_data_ptr_as_int(array.arr) for array in (values, rhs, x)]
    solver = hp.StructPressureSolver(system["shape"], prefix=PREFIX, tol=1e-10)
    try:
        solver.set_system_pointers(*pointers)
        solver.solve()
        solver.get_solution_pointer(pointers[2])
        via_pointers = x.to_numpy()
        solver.set_system(system["values"], system["rhs"], system["x0"])
        solver.solve()
        via_arrays = solver.get_solution()
    finally:
        solver.close()
    np.testing.assert_array_equal(via_pointers, via_arrays)


def test_export_reproduces_the_solver_operator():
    import taichi as ti
    from simulation_core.diagnostics.runtime import TaichiRuntimeConfig
    from simulation_core.fluids.grid import CartesianGrid
    from simulation_core.fluids.solver import CartesianFluidSolver
    from simulation_core.fluids.spec import FluidDomainSpec
    nx, ny, nz = 8, 8, 20
    grid = CartesianGrid(bounds_min_m=(0.0, 0.0, 0.98), cell_widths_x_m=(1e-3,) * nx, cell_widths_y_m=(1.2e-3,) * ny,
                         cell_widths_z_m=tuple(6e-4 * 1.2 ** abs(k - 10) for k in range(nz)))
    spec = FluidDomainSpec(bounds_min_m=grid.bounds_min_m, bounds_max_m=grid.bounds_max_m,
                           grid_nodes=grid.grid_nodes, density_kgm3=1000.0, viscosity_pa_s=1e-3, dt_s=1e-4,
                           cartesian_grid=grid)
    fluid = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cpu", default_fp="f32"))
    shape = grid.grid_nodes
    fluid.velocity.from_numpy(np.zeros(shape + (3,), np.float32))
    fluid.velocity_prev.from_numpy(np.zeros(shape + (3,), np.float32))
    fluid.pressure.from_numpy(np.zeros(shape, np.float64))
    obstacle = np.zeros(shape, np.int32)
    obstacle[2:5, 3:6, 6:9] = 1
    fluid.obstacle.from_numpy(obstacle)
    fluid.clear_volume_source()
    fluid.set_velocity_dirichlet_boundary_authority("canonical")
    fluid.prepare_and_seal_velocity_dirichlet_component_ledger()
    penalty = np.zeros(shape)
    penalty[1:7, 1:7, 15:] = 1.0e9
    fluid.pressure_interface_matrix_diagonal.from_numpy(np.where(obstacle == 0, penalty, 0.0))
    export = hp.export_pressure_system(fluid)
    rng = np.random.default_rng(9)
    p = np.where(obstacle == 0, rng.normal(size=shape), 0.0)
    p_field = ti.field(ti.f64, shape=shape)
    out_field = ti.field(ti.f64, shape=shape)
    p_field.from_numpy(p)
    fluid._apply_fv_pressure_operator(p_field, out_field, 1, fluid._velocity_dirichlet_boundary_authority_code())
    solver_out = out_field.to_numpy()
    ours = hp.fv_apply(export, p)
    scale = np.abs(solver_out).max()
    assert np.abs(ours - solver_out).max() <= 1e-12 * scale
    export.update(b=np.zeros(shape), p0=p, cell_widths_x_m=np.asarray(grid.cell_widths_x_m),
                  cell_widths_y_m=np.asarray(grid.cell_widths_y_m), cell_widths_z_m=np.asarray(grid.cell_widths_z_m))
    assert hp.scaled_system(export)["asymmetry"] < 1e-12
