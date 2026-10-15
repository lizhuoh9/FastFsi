"""SolidOccupancy (boundary/solid_occupancy.py): own-F boxes united with the untangled lattice cells, coverage, hysteresis.

A 0.5 mm grid (20^3 cells) and a lattice block of 8 x 8 x 4 points at 0.25 mm spacing (x, y 3..5 mm, z 4..5 mm): at rest
the block's faces are cell faces, so the material, coverage and union volume are exact.  CPU, one thread, f32."""
import numpy as np
import pytest
import taichi as ti

from fastfsi.boundary.solid_occupancy import TETS, SolidOccupancy, kuhn_tetrahedra, reference_lattice_cells
from fastfsi.grid.axes import GradedAxes
from fastfsi.solid.material_points import MaterialPoints
from fastfsi.solid.materials import NeoHookeanMaterial

H = 0.5e-3                 # grid cell
S = 0.25e-3                # lattice spacing
SHAPE = (8, 8, 4)


@pytest.fixture(scope="module", autouse=True)
def cpu():
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1, opt_level=0, offline_cache=False)


def block_positions():
    i, j, k = np.meshgrid(*(np.arange(n) for n in SHAPE), indexing="ij")
    return np.stack((3.0e-3 + (i.ravel() + 0.5) * S, 3.0e-3 + (j.ravel() + 0.5) * S,
                     4.0e-3 + (k.ravel() + 0.5) * S), axis=1)


@pytest.fixture(scope="module")
def setup():
    faces = np.arange(21) * H
    axes = GradedAxes(faces, faces, faces)
    positions = block_positions()
    points = MaterialPoints(positions_m=positions.astype(np.float32), volumes_m3=np.full(len(positions), S ** 3),
                            material=NeoHookeanMaterial(1000.0, 1.0, 10.0), tether_mask=np.zeros(len(positions)))
    spacing = np.full_like(positions, S)
    occupancy = SolidOccupancy(axes, ((0, 20), (0, 20), (0, 20)), points, np.arange(len(positions)), positions, spacing,
                              hysteresis_steps=3)
    return occupancy, points, positions


def place(points, positions, displacement=None, F=None):
    n = len(positions)
    u = np.zeros((n, 3)) if displacement is None else np.broadcast_to(displacement, (n, 3))
    points.u.from_numpy(np.ascontiguousarray(u, dtype=np.float32))
    points.x.from_numpy((positions + u).astype(np.float32))
    Fs = np.tile(np.eye(3), (n, 1, 1)) if F is None else F
    points.F.from_numpy(np.ascontiguousarray(Fs, dtype=np.float32))


def test_kuhn_tetrahedra_tile_the_unit_cube():
    corners = np.array([[di, dj, dk] for di in (0, 1) for dj in (0, 1) for dk in (0, 1)], dtype=float)
    volumes = []
    for tet in kuhn_tetrahedra():
        v = corners[list(tet)]
        volumes.append(abs(np.linalg.det(np.stack((v[1] - v[0], v[2] - v[0], v[3] - v[0])))) / 6.0)
        assert tet[0] == 0 and tet[3] == 7
    assert len(TETS) == 6 and np.isclose(sum(volumes), 1.0)


def test_lattice_cells_of_a_block():
    positions = block_positions()
    cells, refused = reference_lattice_cells(positions, np.full_like(positions, S))
    assert refused == 0 and len(cells) == 7 * 7 * 3
    for c, (di, dj, dk) in enumerate([(a, b, d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]):
        assert np.allclose(positions[cells[:, c]] - positions[cells[:, 0]], np.array([di, dj, dk]) * S)


def test_lattice_refuses_a_non_adjacent_gap():
    positions = block_positions()
    spacing = np.full_like(positions, S)
    spacing[:, 0] = 0.5 * S                         # declared spacing disagrees with the coordinate gaps along x
    cells, refused = reference_lattice_cells(positions, spacing)
    assert refused > 0 and all(np.ptp(positions[c][:, 0]) == 0.0 for c in cells)


def test_rest_block_is_exact(setup):
    occupancy, points, positions = setup
    place(points, positions)
    stats = occupancy.update()
    occupancy.reset_topology()
    material = occupancy.material_mask()
    coverage = occupancy.coverage()
    expected = np.zeros((20, 20, 20), dtype=bool)
    expected[6:10, 6:10, 8:10] = True
    assert np.array_equal(material, expected)
    assert np.allclose(coverage[expected], 1.0) and np.all(coverage[~expected] == 0.0)
    assert np.isclose(stats["union_volume_m3"], 4.0e-9, rtol=1e-6)
    assert np.isclose(stats["sum_v0_j_m3"], 4.0e-9, rtol=1e-6) and np.isclose(stats["overfill"], 1.0, rtol=1e-6)
    assert stats["lattice_cells"] == 147 and stats["lattice_cells_dropped"] == 0
    assert stats["lattice_cells_inverted"] == 0 and stats["tets_degenerate"] == 0 and stats["boxes_singular"] == 0


def test_union_volume_under_small_moves(setup):
    occupancy, points, positions = setup
    volumes = []
    for dx in np.linspace(0.0, 0.2e-3, 41):
        place(points, positions, displacement=np.array([dx, 0.37 * dx, -0.21 * dx]))
        volumes.append(occupancy.update()["union_volume_m3"])
    volumes = np.array(volumes)
    assert np.all(np.abs(volumes / 4.0e-9 - 1.0) < 0.03)
    quanta = volumes / (H / 3.0) ** 3                                      # centre sampling: whole sub-cells
    assert np.allclose(quanta, np.round(quanta), atol=1e-6)


def test_a_thin_box_marks_its_own_cell_through_the_samples(setup):
    occupancy, points, positions = setup
    F = np.tile(np.diag([1.0, 1.0, 0.02]), (len(positions), 1, 1))      # 5 um thick boxes, no lattice cell between
    far = positions.copy()
    far[:, 2] = 4.0e-3 + (np.arange(len(positions)) % 4 + 0.5) * 1.0e-3   # layers 1 mm apart: cells too stretched
    place(points, positions, displacement=far - positions, F=F)
    stats = occupancy.update()
    material = occupancy.material_mask()
    cells = np.floor(far / H).astype(int)
    assert material[cells[:, 0], cells[:, 1], cells[:, 2]].all()
    assert stats["lattice_cells_dropped"] == stats["lattice_cells"]      # every cell 4x stretched, F = 0.02 J


def test_stretched_cells_are_dropped_and_inverted_cells_kept(setup):
    occupancy, points, positions = setup
    displaced = np.zeros_like(positions)
    corner = np.flatnonzero((positions[:, 0] > 4.8e-3) & (positions[:, 1] > 4.8e-3) & (positions[:, 2] > 4.8e-3))
    displaced[corner] = [2.0e-3, 2.0e-3, 0.0]                              # one corner point pulled 2.8 mm away
    place(points, positions, displacement=displaced)
    stats = occupancy.update()
    assert stats["lattice_cells_dropped"] >= 1
    folded = np.zeros_like(positions)
    a = np.flatnonzero(np.isclose(positions[:, 0], 3.0e-3 + 3.5 * S) & np.isclose(positions[:, 1], 3.0e-3 + 3.5 * S)
                       & np.isclose(positions[:, 2], 4.0e-3 + 1.5 * S))
    folded[a] = [2.0 * S, 0.0, 0.0]                                       # an interior point pushed past its neighbour
    place(points, positions, displacement=folded)
    stats = occupancy.update()
    assert stats["lattice_cells_inverted"] >= 1 and stats["lattice_cells_dropped"] == 0
    assert abs(stats["overfill"] - 1.0) < 0.05


def test_singular_box_is_counted_not_fatal(setup):
    occupancy, points, positions = setup
    F = np.tile(np.eye(3), (len(positions), 1, 1))
    F[0] = 0.0
    place(points, positions, F=F)
    stats = occupancy.update()
    assert stats["boxes_singular"] == 1


def test_topology_hysteresis_and_flip_count(setup):
    occupancy, points, positions = setup
    place(points, positions)
    occupancy.update()
    occupancy.reset_topology()
    assert occupancy.update()["flipped"] == 0
    place(points, positions, displacement=np.array([1.0e-3, 0.0, 0.0]))        # the block moves two cells along x
    first = occupancy.update()
    topo = occupancy.topology_mask()
    assert topo[6:12, 6:10, 8:10].all()                                    # old cells held, new cells at once
    assert first["flipped"] == 2 * 4 * 2
    assert occupancy.update()["flipped"] == 0
    third = occupancy.update()                                             # third non-material update: released
    assert third["flipped"] == 2 * 4 * 2
    topo = occupancy.topology_mask()
    assert not topo[6:8, 6:10, 8:10].any() and topo[8:12, 6:10, 8:10].all()


def test_checkpoint_state_round_trip(setup):
    occupancy, points, positions = setup
    place(points, positions)
    occupancy.update()
    saved = {name: occupancy.state_array(name) for name in occupancy.state_array_names()}
    place(points, positions, displacement=np.array([1.0e-3, 0.0, 0.0]))
    occupancy.update()
    occupancy.load_state_arrays(saved)
    assert np.array_equal(occupancy.topo.to_numpy(), saved["topo"]) and np.array_equal(occupancy.age.to_numpy(),
                                                                                     saved["age"])
