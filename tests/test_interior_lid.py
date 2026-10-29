"""A chamber lid inside the grid (water beyond the lid, an open domain): pneumatic_chamber.chamber_box, lid_plate_gaps
and PneumaticChamber on the duct of test_pneumatic_chamber.py (8 x 8 x 32, z graded, the real solver on the CPU).

1. chamber_box: a lid on an interior z face ends the box there; a lid on no face, or a box floor at or above the lid,
   is refused; with the lid on the top face the ranges are exactly those of the lid-at-top rule (a frozen copy of the
   rule before the option).
2. The plate: the lid on face K_LID with a one-layer plate (the footprint cells of layer K_LID obstacle) and water
   above it (layers above K_LID, and the plate layer's cells outside the footprint).  The box, the inlet cells (layer
   K_LID - 1 only), the wall mask and the first accept's region codes are those of the same duct cut at K_LID with the
   lid on its top face; one open footprint cell of the plate is refused ("water above the chamber lid").
3. One coupled CPU step with the inlet open (update, predict, project, after_project, accept): the air cells and the
   Dirichlet rows stay under the lid, every pressure is finite and the inlet air holds p_in."""
import numpy as np
import pytest

from fastfsi.boundary.outlet_guard import merge_project_requirements
from fastfsi.boundary.pneumatic_chamber import PneumaticChamber, chamber_box, lid_plate_gaps
from fastfsi.fluid.solver import IncompressibleFluidSolver
from fastfsi.grid.axes import GradedAxes
from fastfsi.grid.cartesian_grid import CartesianGrid
from fastfsi.grid.specification import FluidDomainSpecification
from fastfsi.point_tags import TAG_CHAMBER_SURFACE
from fastfsi.simulation.taichi_runtime import TaichiRuntimeConfig
from fastfsi.solid.material_points import MaterialPoints
from fastfsi.solid.materials import NeoHookeanMaterial

RHO, MU, DT = 1000.0, 1.0e-3, 1.0e-4
NX, NY, NZ = 8, 8, 32
K_LID = 28                                  # the interior lid's z face: the plate is layer K_LID, water above it
S_XY, S_Z = 0.5e-3, 0.3e-3
AXIS_XY, RADIUS, INLET_D, BOX_Z_LO = (4.0e-3, 4.0e-3), 3.5e-3, 2.0e-3, 0.99
P_IN = 1000.0
PRESSURE_RTOL = 1.0e-3                      # the inlet air's physical pressure against p_in (penalised Dirichlet)
DRIVER_KWARGS = dict(iterations=400, pressure_outlet_zmin=True, pressure_reachability_prepared=True,
                     reset_pressure=False, pressure_solver="fv_cg", cg_tolerance=1.0e-10,
                     cg_preconditioner="fv_multigrid", pressure_solve_failure_policy="raise", read_report=False)
_STATE = {}


def widths_z(nz):
    return tuple(6.0e-4 * 1.05 ** abs(k - 16) for k in range(nz))


def lid_top_rule(faces, axis_xy_m, radius_m, z_lo_m, lid_z_m):
    """chamber_box before the interior lid (frozen copy): the lid must be the grid's top face."""
    fx, fy, fz = (np.asarray(f, dtype=np.float64) for f in faces)
    if abs(fz[-1] - float(lid_z_m)) > 1.0e-6:
        raise ValueError("the lid is not the top face")
    ranges = []
    for f, centre in ((fx, axis_xy_m[0]), (fy, axis_xy_m[1])):
        lo = max(int(np.searchsorted(f, centre - radius_m, side="right")) - 2, 0)
        hi = min(int(np.searchsorted(f, centre + radius_m, side="left")) + 1, len(f) - 1)
        ranges.append((lo, hi))
    k0 = int(np.searchsorted(fz, z_lo_m, side="right")) - 1
    if not 0 <= k0 < len(fz) - 2:
        raise ValueError("box_z_lo_m is outside the grid")
    ranges.append((k0, len(fz) - 1))
    return tuple(ranges)


def footprint():
    centres = (np.arange(NX) + 0.5) * 1.0e-3
    return np.hypot(centres[:, None] - AXIS_XY[0], centres[None, :] - AXIS_XY[1]) <= RADIUS


def plate_obstacle(nz, open_cell=None):
    """The plate: the footprint cells of layer K_LID (none when the grid ends at K_LID); one left open on request."""
    obstacle = np.zeros((NX, NY, nz), np.int32)
    if nz > K_LID:
        obstacle[:, :, K_LID][footprint()] = 1
        if open_cell is not None:
            obstacle[open_cell[0], open_cell[1], K_LID] = 0
    return obstacle


def set_obstacle(fluid, obstacle):
    fluid.obstacle.from_numpy(np.ascontiguousarray(obstacle, dtype=np.int32))
    fluid._invalidate_pressure_nullspace_component_graph()
    fluid._invalidate_external_obstacle_topology_derived_state()
    fluid.prepare_and_seal_velocity_dirichlet_component_registry()
    fluid.mark_pressure_outlet_disconnected_nonprojectable_cells(pressure_outlet_zmin=True)


def duct(nz, obstacle):
    """A still duct of nz z cells (the first nz of the graded widths) with `obstacle` (CPU runtime of the tests)."""
    if not _STATE.get("runtime"):
        from cpu_runtime import initialize
        initialize()
        _STATE["runtime"] = True
    grid = CartesianGrid(bounds_min_m=(0.0, 0.0, 0.98), cell_widths_x_m=(1.0e-3,) * NX,
                         cell_widths_y_m=(1.0e-3,) * NY, cell_widths_z_m=widths_z(nz))
    spec = FluidDomainSpecification(bounds_min_m=grid.bounds_min_m, bounds_max_m=grid.bounds_max_m,
                                    grid_nodes=grid.grid_nodes, density_kgm3=RHO, viscosity_pa_s=MU, dt_s=DT,
                                    cartesian_grid=grid)
    fluid = IncompressibleFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cpu", default_fp="f32"),
                                      multigrid_semi_coarsening=True)
    shape = grid.grid_nodes
    fluid.velocity.from_numpy(np.zeros(shape + (3,), np.float32))
    fluid.velocity_prev.from_numpy(np.zeros(shape + (3,), np.float32))
    fluid.pressure.from_numpy(np.zeros(shape, np.float64))
    fluid.clear_volume_source()
    fluid.set_velocity_dirichlet_boundary_authority("canonical")
    set_obstacle(fluid, obstacle)
    return fluid, grid


def chamber_under_lid(fluid, grid):
    """The membrane slab of test_pneumatic_chamber.py under the lid on face K_LID, and its PneumaticChamber."""
    faces = [np.asarray(f) for f in (grid.cell_faces_x_m, grid.cell_faces_y_m, grid.cell_faces_z_m)]
    lid = faces[2][K_LID]
    i, j, k = np.meshgrid(np.arange(16), np.arange(16), np.arange(2), indexing="ij")
    positions = np.stack(((i.ravel() + 0.5) * S_XY, (j.ravel() + 0.5) * S_XY,
                          lid - (1.5 - k.ravel()) * S_Z), axis=1)          # layer k = 1's boxes touch the lid
    tags = np.where(k.ravel() == 1, TAG_CHAMBER_SURFACE, 0).astype(np.int32)
    volumes = np.full(len(positions), S_XY * S_XY * S_Z)
    points = MaterialPoints(positions_m=positions.astype(np.float32), volumes_m3=volumes,
                            material=NeoHookeanMaterial(1000.0, 1.0, 10.0), tether_mask=np.zeros(len(positions)),
                            tag=tags)
    chamber = PneumaticChamber(fluid, GradedAxes(*faces), points, body_indices=np.arange(len(positions)),
                               reference_positions_m=positions,
                               spacing_m=np.tile([S_XY, S_XY, S_Z], (len(positions), 1)),
                               point_mass_kg=volumes * 1040.0, axis_xy_m=AXIS_XY, radius_m=RADIUS, lid_z_m=lid,
                               box_z_lo_m=BOX_Z_LO, inlet_centre_xy_m=AXIS_XY, inlet_diameter_m=INLET_D,
                               hysteresis_steps=3, staging=1024)
    return chamber, points, positions


def move(points, positions, dz):
    u = np.zeros_like(positions)
    u[:, 2] = dz
    points.u.from_numpy(u.astype(np.float32))
    points.x.from_numpy((positions + u).astype(np.float32))
    points.F.from_numpy(np.tile(np.eye(3, dtype=np.float32), (len(positions), 1, 1)))


def test_chamber_box_interior_lid_and_the_top_face_rule():
    faces = [np.cumsum((0.0,) + (1.0e-3,) * NX), np.cumsum((0.0,) + (1.0e-3,) * NY),
             0.98 + np.cumsum((0.0,) + widths_z(NZ))]
    interior = chamber_box(faces, AXIS_XY, RADIUS, BOX_Z_LO, faces[2][K_LID])
    assert interior[2] == (int(np.searchsorted(faces[2], BOX_Z_LO, side="right")) - 1, K_LID)
    assert interior[:2] == lid_top_rule(faces, AXIS_XY, RADIUS, BOX_Z_LO, faces[2][-1])[:2]
    with pytest.raises(ValueError, match="not a z face"):
        chamber_box(faces, AXIS_XY, RADIUS, BOX_Z_LO, 0.5 * (faces[2][K_LID] + faces[2][K_LID + 1]))
    with pytest.raises(ValueError, match="below the lid"):
        chamber_box(faces, AXIS_XY, RADIUS, faces[2][K_LID] + 1.0e-5, faces[2][K_LID])
    for radius, z_lo, shift in ((RADIUS, BOX_Z_LO, 0.0), (2.2e-3, 0.995, 3.0e-4), (5.0e-3, 0.98, -7.0e-4)):
        axis = (AXIS_XY[0] + shift, AXIS_XY[1] - shift)
        assert chamber_box(faces, axis, radius, z_lo, faces[2][-1]) == lid_top_rule(faces, axis, radius, z_lo,
                                                                                    faces[2][-1])
    assert lid_plate_gaps(np.zeros((NX, NY, NZ), np.int32), interior[:2] + ((0, NZ),), footprint()) == 0


def test_plate_cells_and_the_box_below_match_the_lid_at_top():
    fluid, grid = duct(NZ, plate_obstacle(NZ))
    chamber, points, positions = chamber_under_lid(fluid, grid)
    top_fluid, top_grid = duct(K_LID, plate_obstacle(K_LID))
    top, top_points, top_positions = chamber_under_lid(top_fluid, top_grid)
    assert np.array_equal(np.asarray(grid.cell_faces_z_m)[:K_LID + 1], np.asarray(top_grid.cell_faces_z_m))
    assert chamber.box_cells == top.box_cells and chamber.box_cells[2][1] == K_LID
    inlet = chamber.regions.inlet.to_numpy()
    assert np.array_equal(inlet, top.regions.inlet.to_numpy()) and chamber.inlet_cells == top.inlet_cells == 4
    assert inlet[:, :, -1].sum() == 4 and not inlet[:, :, :-1].any()      # the box's top layer: K_LID - 1
    assert np.array_equal(chamber.regions.wall.to_numpy(), top.regions.wall.to_numpy())
    move(points, positions, -1.5e-3)
    move(top_points, top_positions, -1.5e-3)
    first, first_top = chamber.accept(0.0, 0.0), top.accept(0.0, 0.0)
    assert not first["inlet_covered"] and first["air_cells"] == first_top["air_cells"] > 0
    assert np.array_equal(chamber.regions.codes(), top.regions.codes())
    air = chamber.air_cell.to_numpy()
    assert np.array_equal(air[:, :, :K_LID], top.air_cell.to_numpy()) and not air[:, :, K_LID:].any()
    set_obstacle(fluid, plate_obstacle(NZ, open_cell=(3, 1)))
    with pytest.raises(ValueError, match="water above the chamber lid: 1 footprint cells"):
        chamber_under_lid(fluid, grid)


def test_one_coupled_step_under_an_interior_lid():
    fluid, grid = duct(NZ, plate_obstacle(NZ))
    chamber, points, positions = chamber_under_lid(fluid, grid)
    move(points, positions, 0.0)
    assert chamber.accept(0.0, 0.0)["inlet_covered"]
    move(points, positions, -1.5e-3)
    covered = [chamber.accept(1.0e-3 * (n + 1), 0.0, dt_taken_s=DT)["inlet_covered"] for n in range(4)]
    assert covered == [True, True, False, False]
    air = chamber.air_cell.to_numpy() != 0                            # the Dirichlet cells of the step's rows
    kwargs = merge_project_requirements(chamber, DRIVER_KWARGS)
    t = 4.0e-3
    chamber.update(t + 0.5 * DT, P_IN)
    fluid.predict(dt_s=DT, advection_scheme="muscl_tvd", pressure_outlet_zmin=True)
    fluid.project(dt_s=DT, **kwargs)
    chamber.after_project(DT, P_IN)
    out = chamber.accept(t + DT, P_IN, dt_taken_s=DT)
    assert np.array_equal(chamber.air_cell.to_numpy() != 0, air)       # the slab is still: no flip
    diagonal = fluid.pressure_interface_matrix_diagonal.to_numpy()
    pressure = chamber.physical_pressure_field().to_numpy()
    print("INTERIOR_LID one step: air cells %d (layers %s), Dirichlet rows %d, inlet air p %.3f..%.3f Pa, water above "
          "the plate p %.3f..%.3f Pa" % (air.sum(), sorted(set(np.nonzero(air)[2].tolist())),
                                         int((diagonal != 0.0).sum()), pressure[air].min(), pressure[air].max(),
                                         pressure[:, :, K_LID + 1:].min(), pressure[:, :, K_LID + 1:].max()))
    assert not out["inlet_covered"] and air[:, :, K_LID - 1].any() and not air[:, :, K_LID:].any()
    assert np.array_equal(diagonal != 0.0, air)                       # Dirichlet rows: the air cells, under the lid
    assert np.isfinite(pressure).all()
    assert np.abs(pressure[air] - P_IN).max() <= PRESSURE_RTOL * P_IN
