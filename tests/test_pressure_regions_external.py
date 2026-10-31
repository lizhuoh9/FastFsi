"""PressureRegions' external region (attach_external_region; the s26 outer chamber's cavity, Root 2026-10-31), on the
real solver of test_pressure_regions (duct 8 x 8 x 32, z-min outlet, an obstacle block, random u*): the region box holds
the inlet air (k >= 26), a disjoint external box holds a 3 x 3 x 3 gas block at the caller's gauge pressure.
  1. physical water pressure and corrected water faces equal the explicit-Dirichlet reference, the block's cells are
     Dirichlet cells counted as external_cells (not pockets), the water divergence bookkeeping closes;
  2. an attached region whose code is all zero leaves every row bit for bit as without it;
  3. attaching after the first update or with wrong fields is refused;
  4. an external box over the region box is allowed: on a cell both mark the region's code wins and is counted."""
import numpy as np
import pytest
import taichi as ti

from fastfsi.boundary.pressure_regions import PressureRegions
from fastfsi.fluid.solver import IncompressibleFluidSolver
from fastfsi.grid.specification import FluidDomainSpecification
from fastfsi.simulation.taichi_runtime import TaichiRuntimeConfig
from test_pressure_regions import (DT, K_AIR, MU, NX, NY, NZ, P_IN, RHO, dirichlet_reference, duct_grid,
                                   obstacle_block, project, random_u_star, set_obstacle, water_faces)

REGION_BOX = ((0, NX), (0, NY), (K_AIR - 2, NZ))
EXTERNAL_BOX = ((1, 6), (1, 6), (13, 20))
BLOCK = (slice(2, 5), slice(2, 5), slice(15, 18))          # grid cells of the gas block
_STATE = {}


def fluid_rig():
    if "fluid" not in _STATE:
        from cpu_runtime import initialize
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
        _STATE.update(fluid=fluid, grid=grid)
    return _STATE["fluid"], _STATE["grid"]


def region_codes():
    codes = np.zeros(tuple(hi - lo for lo, hi in REGION_BOX), np.int32)
    codes[:, :, K_AIR - REGION_BOX[2][0]:] = 1
    return codes


def external_fields(on=True):
    shape = tuple(hi - lo for lo, hi in EXTERNAL_BOX)
    code = ti.field(ti.i32, shape=shape)
    values = np.zeros(shape, np.int32)
    if on:
        values[tuple(slice(s.start - lo, s.stop - lo) for s, (lo, _) in zip(BLOCK, EXTERNAL_BOX))] = 1
    code.from_numpy(values)
    gauge = ti.field(ti.f64, shape=())
    return code, gauge


def test_an_external_block_matches_the_explicit_dirichlet_reference():
    fluid, grid = fluid_rig()
    obstacle = obstacle_block()
    set_obstacle(fluid, obstacle)
    u_star = random_u_star(7, obstacle)
    regions = PressureRegions(fluid, REGION_BOX)
    code, gauge = external_fields()
    regions.attach_external_region(EXTERNAL_BOX, code, gauge)
    regions.set_regions(region_codes(), [])
    for dp in (1.0, 6000.0):
        gauge[None] = P_IN + dp
        fluid.velocity.from_numpy(u_star)
        fluid.pressure.from_numpy(np.zeros((NX, NY, NZ), np.float64))
        regions.update(0.5, P_IN)
        project(fluid)
        regions.after_project(DT, P_IN)
        values = np.full((NX, NY, NZ), np.nan)
        values[:, :, K_AIR:] = P_IN
        values[BLOCK] = P_IN + dp
        p_ref, u_ref, water = dirichlet_reference(grid, obstacle, values, u_star)
        physical = regions.physical_pressure_field().to_numpy()
        faces = water_faces(obstacle, water)
        u_fast = fluid.velocity.to_numpy().astype(np.float64)
        scale = abs(P_IN) + dp
        report = regions.report()
        assert np.abs(physical[water] - p_ref[water]).max() / scale <= 1.0e-6
        assert np.abs(u_fast[faces] - u_ref[faces]).max() / np.abs(u_ref[faces]).max() <= 1.0e-5
        assert np.abs(fluid.pressure.to_numpy()[BLOCK]).max() / scale <= 1.0e-6        # p~ ~ 0: p = the gauge
        assert np.allclose(physical[BLOCK], P_IN + dp, rtol=0, atol=1.0e-6 * scale)
        assert report["external_cells"] == 27 and report["pocket_cells"] == 0
        assert report["inlet_cells"] == NX * NY * (NZ - K_AIR)
        assert report["water_divergence_max_1ps"] <= 1.0e-3


def test_an_empty_external_region_leaves_the_rows_bit_for_bit():
    fluid, _ = fluid_rig()
    set_obstacle(fluid, obstacle_block())
    rows = []
    for attach in (False, True):
        regions = PressureRegions(fluid, REGION_BOX)
        if attach:
            code, gauge = external_fields(on=False)
            gauge[None] = 12345.0
            regions.attach_external_region(EXTERNAL_BOX, code, gauge)
        regions.set_regions(region_codes(), [])
        regions.update(0.25, P_IN)
        rows.append((fluid.pressure_interface_matrix_diagonal.to_numpy().tobytes(),
                     fluid.pressure_interface_matrix_rhs.to_numpy().tobytes(), regions.p_ref.to_numpy().tobytes()))
    assert rows[0] == rows[1]


def test_bad_attachments_are_refused():
    fluid, _ = fluid_rig()
    code, gauge = external_fields()
    regions = PressureRegions(fluid, REGION_BOX)
    with pytest.raises(ValueError, match="i32 field"):
        regions.attach_external_region(EXTERNAL_BOX, ti.field(ti.f32, shape=code.shape), gauge)
    with pytest.raises(ValueError, match="0-d f64"):
        regions.attach_external_region(EXTERNAL_BOX, code, ti.field(ti.f64, shape=1))
    with pytest.raises(ValueError, match="inside the grid"):
        regions.attach_external_region(((1, 6), (1, 6), (30, 40)), code, gauge)
    regions.set_regions(region_codes(), [])
    regions.update(0.0, P_IN)
    with pytest.raises(RuntimeError, match="before the first update"):
        regions.attach_external_region(EXTERNAL_BOX, code, gauge)
    reported = PressureRegions(fluid, REGION_BOX)
    reported.set_regions(region_codes(), [])
    reported.report()                                       # report() compiled _report_kernel without the region
    with pytest.raises(RuntimeError, match="or report"):
        reported.attach_external_region(EXTERNAL_BOX, code, gauge)


def test_an_overlapping_external_box_lets_the_region_win_and_counts_conflicts():
    fluid, _ = fluid_rig()
    set_obstacle(fluid, np.zeros((NX, NY, NZ), np.int32))
    regions = PressureRegions(fluid, REGION_BOX)
    box = ((2, 5), (2, 5), (K_AIR - 1, K_AIR + 2))            # 3 x 3 x 3 over the inlet air's lowest two layers
    code = ti.field(ti.i32, shape=(3, 3, 3))
    code.fill(1)
    gauge = ti.field(ti.f64, shape=())
    gauge[None] = 777.0
    regions.attach_external_region(box, code, gauge)
    regions.set_regions(region_codes(), [])
    regions.update(0.1, P_IN)
    report = regions.report()
    p_ref = regions.p_ref.to_numpy()
    assert report["external_conflicts"] == 18 and report["external_cells"] == 9      # the region's air won 18 cells
    assert np.all(p_ref[2:5, 2:5, K_AIR:K_AIR + 2] == P_IN) and np.all(p_ref[2:5, 2:5, K_AIR - 1] == 777.0)
