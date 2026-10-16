"""Union-find labels of the exact pressure-outlet operator graph (fluid/pressure_nullspace.py).

The graph build labels each operator component by its minimum linear cell index with a lock-free union-find (one union
kernel, one compression kernel) and certifies the result with one min-label sweep triple.  Contracts:

- labels bitwise equal to the fixed point of the min-label sweeps, and to an independent host union-find, on random
  masks with face-mobility cuts, long non-local interface-row chains (with rejected rows) and legacy coupling slots;
- the same component counts as the host union-find;
- the graph build (counts, compact labels, raw-component ledgers) identical to the iterative build;
- a label state that is not the fixed point fails the graph build closed (the certificate's failure path).
"""
from types import MethodType

import numpy as np
import pytest

from fastfsi.fluid.common import PressureSolveConvergenceError
from fastfsi.fluid.interface_constants import PRESSURE_INTERFACE_COUPLING_SLOT_COUNT
from fastfsi.fluid.solver import IncompressibleFluidSolver
from fastfsi.grid.specification import FluidDomainSpecification
from fastfsi.simulation.taichi_runtime import TaichiRuntimeConfig

NOT_FLUID = 1 << 30


@pytest.fixture(scope="module", autouse=True)
def cpu():
    from cpu_runtime import initialize
    initialize()


def solver(grid_nodes):
    return IncompressibleFluidSolver(FluidDomainSpecification.unit_box(grid_nodes=grid_nodes, dt_s=1.0e-3),
                                     runtime=TaichiRuntimeConfig(arch="cpu", default_fp="f32"))


def clear_interface(fluid):
    fluid.pressure_interface_row_count[None] = 0
    fluid.pressure_interface_coupling_active.fill(0)


def random_masks(fluid, rng, obstacle_fraction, cut_fraction):
    shape = (fluid.nx, fluid.ny, fluid.nz)
    obstacle = (rng.random(shape) < obstacle_fraction).astype(np.int32)
    cuts = np.zeros(shape, dtype=np.int32)
    for bit in (1, 2, 4):
        cuts |= np.where(rng.random(shape) < cut_fraction, bit, 0).astype(np.int32)
    fluid.obstacle.from_numpy(obstacle)
    fluid.velocity_dirichlet_pressure_hard_fixed_component_mask.from_numpy(cuts)
    fluid.velocity_dirichlet_boundary_authority_code_device[None] = 0
    return obstacle, cuts


def install_rows(fluid, rows):
    fluid.pressure_interface_row_count[None] = len(rows)
    for row, (owner, neighbor, transmissibility) in enumerate(rows):
        fluid.pressure_interface_row_owner[row] = owner
        fluid.pressure_interface_row_neighbor[row] = neighbor
        fluid.pressure_interface_row_transmissibility[row] = transmissibility


def random_fluid_cells(obstacle, rng, count):
    cells = np.argwhere(obstacle == 0)
    picks = rng.choice(len(cells), size=min(count, len(cells)), replace=False)
    return [tuple(int(v) for v in cells[p]) for p in picks]


def chain_rows(obstacle, rng, length, obstacle_cell):
    """A long non-local chain through random fluid cells plus rows every filter must reject."""
    cells = random_fluid_cells(obstacle, rng, length)
    rows = [(cells[n], cells[n + 1], float(rng.uniform(0.1, 2.0))) for n in range(len(cells) - 1)]
    a, b = cells[0], cells[-1]
    rows += [(a, b, float("nan")), (a, b, float("inf")), (a, b, 0.0), (a, b, -1.0), (a, a, 1.0),
             (a, obstacle_cell, 1.0)]
    order = rng.permutation(len(rows))
    return [rows[n] for n in order]


def install_slots(fluid, obstacle, rng, cell_count, obstacle_cell):
    """Legacy slots on random fluid cells (active counts up to the slot count + 2: the extra ones are ignored)."""
    shape = obstacle.shape
    active = np.zeros(shape, dtype=np.int32)
    neighbor = np.zeros(shape + (3,), dtype=np.int32)
    coefficient = np.zeros(shape, dtype=np.float64)
    extra_slots = PRESSURE_INTERFACE_COUPLING_SLOT_COUNT - 1
    extra_neighbor = np.zeros(shape + (extra_slots, 3), dtype=np.int32)
    extra_coefficient = np.zeros(shape + (extra_slots,), dtype=np.float64)
    fluid_cells = np.argwhere(obstacle == 0)
    invalid = (float("nan"), float("inf"), 0.0, -1.0)
    for owner in random_fluid_cells(obstacle, rng, cell_count):
        active[owner] = int(rng.integers(1, PRESSURE_INTERFACE_COUPLING_SLOT_COUNT + 3))
        for slot in range(PRESSURE_INTERFACE_COUPLING_SLOT_COUNT):
            target = tuple(int(v) for v in fluid_cells[rng.integers(len(fluid_cells))])
            value = float(rng.uniform(0.1, 2.0))
            roll = rng.random()
            if roll < 0.1:
                value = invalid[int(rng.integers(len(invalid)))]
            elif roll < 0.15:
                target = owner
            elif roll < 0.2:
                target = obstacle_cell
            if slot == 0:
                neighbor[owner] = target
                coefficient[owner] = value
            else:
                extra_neighbor[owner + (slot - 1,)] = target
                extra_coefficient[owner + (slot - 1,)] = value
    fluid.pressure_interface_row_count[None] = 0
    fluid.pressure_interface_coupling_active.from_numpy(active)
    fluid.pressure_interface_coupling_neighbor.from_numpy(neighbor)
    fluid.pressure_interface_coupling_coefficient.from_numpy(coefficient)
    fluid.pressure_interface_coupling_extra_neighbor.from_numpy(extra_neighbor)
    fluid.pressure_interface_coupling_extra_coefficient.from_numpy(extra_coefficient)
    return active, neighbor, coefficient, extra_neighbor, extra_coefficient


def host_labels(obstacle, cuts, rows=(), slots=None):
    """Independent host union-find of the operator graph: the minimum linear index per component."""
    nx, ny, nz = obstacle.shape
    parent = list(range(nx * ny * nz))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    def index(c):
        return (c[0] * ny + c[1]) * nz + c[2]

    def fluid(c):
        return all(0 <= c[d] < obstacle.shape[d] for d in range(3)) and obstacle[c] == 0

    def valid(a, b, t):
        return fluid(a) and fluid(b) and np.isfinite(t) and t > 0.0 and tuple(a) != tuple(b)

    for c in map(tuple, np.argwhere(obstacle == 0)):
        for axis, bit in ((0, 1), (1, 2), (2, 4)):
            n = list(c)
            n[axis] += 1
            n = tuple(n)
            if n[axis] < obstacle.shape[axis] and obstacle[n] == 0 and not (cuts[n] & bit):
                union(index(c), index(n))
    for owner, neighbor, t in rows:
        if valid(tuple(owner), tuple(neighbor), t):
            union(index(owner), index(neighbor))
    if not rows and slots is not None:
        active, neighbor, coefficient, extra_neighbor, extra_coefficient = slots
        for c in map(tuple, np.argwhere((active > 0) & (obstacle == 0))):
            for slot in range(min(int(active[c]), PRESSURE_INTERFACE_COUPLING_SLOT_COUNT)):
                if slot == 0:
                    n, t = tuple(int(v) for v in neighbor[c]), coefficient[c]
                else:
                    n, t = tuple(int(v) for v in extra_neighbor[c + (slot - 1,)]), extra_coefficient[c + (slot - 1,)]
                if valid(c, n, t):
                    union(index(c), index(n))
    labels = np.full(obstacle.shape, NOT_FLUID, dtype=np.int32)
    for c in map(tuple, np.argwhere(obstacle == 0)):
        labels[c] = find(index(c))
    return labels


def sweep_labels(fluid):
    """The iterative reference: init, then min-label sweep batches until one changes nothing."""
    fluid._init_pressure_outlet_operator_component_labels_kernel()
    for _ in range(4096):
        if int(fluid._propagate_pressure_outlet_operator_labels_batch()) == 0:
            return fluid.pressure_outlet_operator_component_label.to_numpy()
    raise AssertionError("min-label sweeps did not converge")


def union_find_labels(fluid):
    fluid._init_pressure_outlet_operator_component_labels_kernel()
    fluid._union_pressure_outlet_operator_component_labels_kernel()
    fluid._compress_pressure_outlet_operator_component_labels_kernel()
    return fluid.pressure_outlet_operator_component_label.to_numpy()


def component_count(labels, obstacle):
    return int(np.unique(labels[obstacle == 0]).size)


def check(fluid, obstacle, host):
    reference = sweep_labels(fluid)
    labels = union_find_labels(fluid)
    np.testing.assert_array_equal(labels, reference)
    np.testing.assert_array_equal(labels, host)
    assert int(fluid._certify_pressure_outlet_operator_component_labels()) == 0
    np.testing.assert_array_equal(fluid.pressure_outlet_operator_component_label.to_numpy(), reference)
    assert component_count(labels, obstacle) == component_count(host, obstacle)
    return component_count(labels, obstacle)


def test_grid_faces_with_mobility_cuts():
    fluid = solver((6, 9, 23))
    counts = []
    for seed in range(6):
        rng = np.random.default_rng(500 + seed)
        clear_interface(fluid)
        obstacle, cuts = random_masks(fluid, rng, 0.25 + 0.05 * seed, 0.15)
        counts.append(check(fluid, obstacle, host_labels(obstacle, cuts)))
    assert max(counts) > 1


def test_long_nonlocal_row_chains_with_rejected_rows():
    fluid = solver((5, 8, 31))
    for seed in range(5):
        rng = np.random.default_rng(600 + seed)
        clear_interface(fluid)
        obstacle, cuts = random_masks(fluid, rng, 0.55, 0.3)
        obstacle_cell = tuple(int(v) for v in np.argwhere(obstacle != 0)[0])
        rows = chain_rows(obstacle, rng, 60, obstacle_cell)
        install_rows(fluid, rows)
        count = check(fluid, obstacle, host_labels(obstacle, cuts, rows=rows))
        assert count < component_count(host_labels(obstacle, cuts), obstacle)


def test_legacy_coupling_slots_without_rows():
    fluid = solver((4, 7, 19))
    for seed in range(5):
        rng = np.random.default_rng(700 + seed)
        clear_interface(fluid)
        obstacle, cuts = random_masks(fluid, rng, 0.5, 0.3)
        obstacle_cell = tuple(int(v) for v in np.argwhere(obstacle != 0)[0])
        slots = install_slots(fluid, obstacle, rng, 12, obstacle_cell)
        check(fluid, obstacle, host_labels(obstacle, cuts, slots=slots))
        assert int(fluid.pressure_outlet_operator_interface_slot_cell_count[None]) > 0


def test_rows_take_precedence_over_slots():
    fluid = solver((4, 7, 19))
    rng = np.random.default_rng(800)
    clear_interface(fluid)
    obstacle, cuts = random_masks(fluid, rng, 0.5, 0.3)
    obstacle_cell = tuple(int(v) for v in np.argwhere(obstacle != 0)[0])
    install_slots(fluid, obstacle, rng, 12, obstacle_cell)
    rows = chain_rows(obstacle, rng, 20, obstacle_cell)
    install_rows(fluid, rows)
    check(fluid, obstacle, host_labels(obstacle, cuts, rows=rows))


def test_open_box_is_one_component_labelled_zero():
    fluid = solver((4, 8, 96))
    clear_interface(fluid)
    fluid.obstacle.fill(0)
    fluid.velocity_dirichlet_pressure_hard_fixed_component_mask.fill(0)
    fluid.velocity_dirichlet_boundary_authority_code_device[None] = 0
    np.testing.assert_array_equal(union_find_labels(fluid), 0)
    assert int(fluid._certify_pressure_outlet_operator_component_labels()) == 0


def pocket_solver(seed):
    """Outlet layer k = 0 open, random isolated two-cell pockets above it (cells (i, j, k) and (i, j, k + 1): a cell
    without a free face would stop the reachability marking, fluid_isolated_cell_without_free_face), interface rows
    joining some pockets."""
    grid_nodes = (5, 6, 9)
    fluid = solver(grid_nodes)
    rng = np.random.default_rng(seed)
    obstacle = np.ones(grid_nodes, dtype=np.int32)
    obstacle[:, :, 0] = 0
    pockets = []
    for i in range(0, grid_nodes[0], 2):
        for j in range(0, grid_nodes[1], 2):
            for k in range(2, grid_nodes[2] - 1, 3):
                if rng.random() < 0.6:
                    obstacle[i, j, k] = obstacle[i, j, k + 1] = 0
                    pockets.append((i, j, k))
    fluid.obstacle.from_numpy(obstacle)
    unreached = fluid.mark_pressure_outlet_disconnected_nonprojectable_cells(pressure_outlet_zmin=True)
    assert unreached == 2 * len(pockets), "fixture expected %d isolated cells, got %d" % (2 * len(pockets), unreached)
    cell_volume_m3 = float(fluid.dx * fluid.dy * fluid.dz)
    diagonal = fluid.pressure_interface_matrix_diagonal.to_numpy()
    order = rng.permutation(len(pockets))
    rows = [(pockets[order[n]], pockets[order[n + 1]], float(rng.uniform(0.1, 1.0)))
            for n in range(0, len(pockets) - 1, 3)]
    install_rows(fluid, rows)
    for owner, neighbor, transmissibility in rows:
        diagonal[owner] += transmissibility / cell_volume_m3
        diagonal[neighbor] += transmissibility / cell_volume_m3
    fluid.pressure_interface_matrix_diagonal.from_numpy(diagonal)
    return fluid


def ledger(fluid, result):
    names = ("pressure_outlet_operator_component_label", "pressure_outlet_operator_raw_component_cell_count",
             "pressure_outlet_operator_raw_component_to_compact", "pressure_outlet_operator_raw_component_anchored",
             "pressure_outlet_operator_raw_component_diagonal_excess_integral",
             "pressure_outlet_operator_raw_component_diagonal_scale_integral",
             "pressure_outlet_operator_raw_component_diagonal_term_count")
    scalars = ("pressure_solve_component_count", "pressure_outlet_operator_raw_component_count",
               "pressure_outlet_operator_empty_singleton_component_count",
               "pressure_outlet_operator_active_unanchored_component_count")
    return (result, {n: getattr(fluid, n).to_numpy() for n in names},
            {n: int(getattr(fluid, n)[None]) for n in scalars})


def test_graph_build_matches_the_iterative_build():
    for seed in range(3):
        fluid = pocket_solver(900 + seed)
        union_find = ledger(fluid, fluid._prepare_pressure_outlet_nullspace_component_graph())

        def sweeps_instead(this):
            for _ in range(4096):
                if int(this._propagate_pressure_outlet_operator_labels_batch()) == 0:
                    return
            raise AssertionError("min-label sweeps did not converge")

        fluid._union_pressure_outlet_operator_component_labels_kernel = MethodType(sweeps_instead, fluid)
        fluid._compress_pressure_outlet_operator_component_labels_kernel = MethodType(lambda this: None, fluid)
        iterative = ledger(fluid, fluid._prepare_pressure_outlet_nullspace_component_graph())
        assert union_find[0] == iterative[0] and union_find[0][1] > 1
        for name, values in union_find[1].items():
            np.testing.assert_array_equal(values, iterative[1][name], err_msg=name)
        assert union_find[2] == iterative[2]


def test_certificate_failure_fails_the_graph_closed():
    fluid = pocket_solver(950)
    real_compress = fluid._compress_pressure_outlet_operator_component_labels_kernel

    def detach_one_cell(this):
        real_compress()
        labels = this.pressure_outlet_operator_component_label.to_numpy()
        fluid_cells = np.argwhere(labels < NOT_FLUID)
        index = (fluid_cells[:, 0] * this.ny + fluid_cells[:, 1]) * this.nz + fluid_cells[:, 2]
        offset = index - labels[tuple(fluid_cells.T)]
        assert int(offset.max()) > 0
        detached = fluid_cells[np.argmax(offset)]
        labels[tuple(detached)] = (detached[0] * this.ny + detached[1]) * this.nz + detached[2]
        this.pressure_outlet_operator_component_label.from_numpy(labels)

    fluid._compress_pressure_outlet_operator_component_labels_kernel = MethodType(detach_one_cell, fluid)
    with pytest.raises(PressureSolveConvergenceError) as caught:
        fluid._prepare_pressure_outlet_nullspace_component_graph()
    diagnostics = caught.value.diagnostics
    assert diagnostics["stage"] == "pressure_nullspace_component_graph"
    assert diagnostics["reason"] == "component_labels_not_converged" and not diagnostics["labels_converged"]
    assert diagnostics["label_sweep_count"] == 1 and diagnostics["certificate_label_changes"] > 0
    assert not bool(fluid._pressure_outlet_nullspace_graph_valid)


def test_graph_build_runs_one_certificate_sweep():
    fluid = pocket_solver(960)
    calls = []
    real_certify = fluid._certify_pressure_outlet_operator_component_labels

    def counted(this):
        calls.append(1)
        return real_certify()

    def no_batches(this):
        raise AssertionError("the graph build may not run min-label batches")

    fluid._certify_pressure_outlet_operator_component_labels = MethodType(counted, fluid)
    fluid._propagate_pressure_outlet_operator_labels_batch = MethodType(no_batches, fluid)
    fluid._prepare_pressure_outlet_nullspace_component_graph()
    assert calls == [1] and bool(fluid._pressure_outlet_nullspace_graph_valid)
