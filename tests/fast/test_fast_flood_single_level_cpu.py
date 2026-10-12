"""Fast-mode reachability flood on the production single-level path (fluid_hooks.single_level_reachability_flood,
2026-10-12: the F2(a) 16-level batch kernel's CUDA compile exceeded the 16 GB host cap on the RTX 3080): the same
reachable set, the same unreached-cell count and the same exact BFS level count as the batched flood; the module
constants are restored afterwards."""
import numpy as np

from simulation_core.diagnostics.runtime import TaichiRuntimeConfig
from simulation_core.fast import fluid_hooks
from simulation_core.fluids import solver as solver_module
from simulation_core.fluids.grid import CartesianGrid
from simulation_core.fluids.solver import CartesianFluidSolver
from simulation_core.fluids.spec import FluidDomainSpec

NX, NY, NZ = 8, 8, 32


def obstacle_with_a_sealed_pocket():
    obstacle = np.zeros((NX, NY, NZ), np.int32)
    obstacle[2:6, 2:6, 20:26] = 1                 # a closed shell ...
    obstacle[3:5, 3:5, 21:25] = 0                 # ... around a sealed fluid pocket (unreachable from the outlet)
    obstacle[:, :, 10] = 1                        # a floor across the duct with one hole: a long BFS path
    obstacle[7, 7, 10] = 0
    return obstacle


def test_single_level_flood_matches_the_batched_flood():
    grid = CartesianGrid(bounds_min_m=(0.0, 0.0, 0.0), cell_widths_x_m=(1.0e-3,) * NX, cell_widths_y_m=(1.0e-3,) * NY,
                         cell_widths_z_m=(1.0e-3,) * NZ)
    spec = FluidDomainSpec(bounds_min_m=grid.bounds_min_m, bounds_max_m=grid.bounds_max_m, grid_nodes=grid.grid_nodes,
                           density_kgm3=1000.0, viscosity_pa_s=1.0e-3, dt_s=1.0e-4, cartesian_grid=grid)
    fluid = CartesianFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cpu", default_fp="f32"))
    shape = grid.grid_nodes
    fluid.velocity.from_numpy(np.zeros(shape + (3,), np.float32))
    fluid.velocity_prev.from_numpy(np.zeros(shape + (3,), np.float32))
    fluid.pressure.from_numpy(np.zeros(shape, np.float64))
    fluid.obstacle.from_numpy(obstacle_with_a_sealed_pocket())
    fluid.clear_volume_source()
    fluid.set_velocity_dirichlet_boundary_authority("canonical")
    fluid.prepare_and_seal_velocity_dirichlet_component_ledger()
    batched_levels_constant = solver_module.HIBM_REACHABILITY_FLOOD_BATCH_LEVELS
    assert batched_levels_constant >= 2
    unreached_batched = fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(pressure_outlet_zmin=True)
    reach_batched = fluid.hibm_pressure_outlet_reachable.to_numpy().copy()
    levels_batched = int(fluid.last_hibm_pressure_reachability_sweeps)
    # the same obstacle again through the fast-mode obstacle path (single-level flood inside)
    unreached_single = fluid_hooks.replace_obstacle_cells(fluid, obstacle_with_a_sealed_pocket())
    reach_single = fluid.hibm_pressure_outlet_reachable.to_numpy()
    levels_single = int(fluid.last_hibm_pressure_reachability_sweeps)
    assert levels_batched > batched_levels_constant           # the batched path really batched several launches
    assert int(unreached_single) == int(unreached_batched) == 2 * 2 * 4
    assert np.array_equal(reach_single, reach_batched)
    assert levels_single == levels_batched
    assert solver_module.HIBM_REACHABILITY_FLOOD_BATCH_LEVELS == batched_levels_constant   # restored
