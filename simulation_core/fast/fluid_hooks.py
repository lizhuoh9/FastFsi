"""Fast-mode integrator hooks (Claude 2026-10-12; contract 01 §1.1, §1.2, §5).

UniformAxes: the uniform-grid axes object (axis_weights / face_volume / n / min_spacing_m) for tests and uniform rigs;
GradedAxes (graded_kernel.py) has the same API.  AllValidMask: every face valid (no obstacle / air handling), for
tests; FaceMask (graded_kernel.py) is the real one.  replace_obstacle_cells: the ONE place where the fast mode changes
the fluid obstacle topology at run time (valve seal ring), with the same invalidation / ledger-seal / reachability
sequence that cases/turek_hron_fsi._write_channel_external_velocity_faces uses for a boundary-mask change.

Reuse check (REQ_400): the IB_4 index map is IBAMR v0.19.0 ibtk/src/lagrangian/LEInteractor.cpp (side-centred data as
cell-centred data shifted by half a cell along its own axis), BSD-3-Clause; the obstacle-change sequence reuses the
production CartesianFluidSolver's own invalidation API (no new solver code).
"""
import contextlib

import numpy as np
import taichi as ti

from simulation_core.fast.ib_mpm_fast import ib4_dphi, ib4_phi


@ti.data_oriented
class UniformAxes:
    """Axes object of a uniform MAC grid with origin `origin_m`, spacing `spacing_m` and `nodes` cells."""

    def __init__(self, origin_m, spacing_m, nodes):
        self.origin = tuple(float(v) for v in origin_m)
        self.h = tuple(float(v) for v in spacing_m)
        self.n = tuple(int(v) for v in nodes)
        if min(self.h) <= 0.0 or min(self.n) <= 0:
            raise ValueError("uniform axes need positive spacing and node counts")
        self.min_spacing_m = min(self.h)
        self.cell_volume_m3 = self.h[0] * self.h[1] * self.h[2]

    @classmethod
    def from_fluid(cls, fluid):
        grid = fluid.grid
        if not grid.is_uniform:
            raise ValueError("UniformAxes needs a uniform fluid grid; use GradedAxes")
        return cls(grid.bounds_min_m, grid.uniform_spacing_m, grid.grid_nodes)

    @ti.func
    def axis_weights(self, a: ti.template(), staggered: ti.template(), coord):
        s = (coord - self.origin[a]) / self.h[a] - (0.0 if staggered else 0.5)
        i0 = ti.cast(ti.floor(s), ti.i32) - 1
        w = ti.Vector([0.0, 0.0, 0.0, 0.0])
        dw = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for q in ti.static(range(4)):
            r = s - ti.cast(i0 + q, ti.f32)
            w[q] = ib4_phi(r)
            dw[q] = ib4_dphi(r) / self.h[a]
        return i0, w, dw

    @ti.func
    def face_volume(self, c: ti.template(), i, j, k):
        return self.cell_volume_m3


@ti.data_oriented
class AllValidMask:
    """Every MAC face valid (tests and obstacle-free rigs)."""

    def refresh(self, air_cell_field=None):
        return None

    @ti.func
    def face_valid(self, c: ti.template(), i, j, k):
        return 1


@ti.kernel
def _write_obstacle_kernel(obstacle: ti.template(), velocity: ti.template(), velocity_prev: ti.template(),
                           mask: ti.types.ndarray()):
    for i, j, k in obstacle:
        new = mask[i, j, k]
        if new != 0 and obstacle[i, j, k] == 0:
            velocity[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            velocity_prev[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        obstacle[i, j, k] = 1 if new != 0 else 0


FAST_FLOOD_BATCH_LEVELS = 0      # outlet-reachability flood levels per launch in the fast mode (< 2: single-level path)


@contextlib.contextmanager
def single_level_reachability_flood():
    """Scoped: the production outlet-reachability flood takes its single-level path (one BFS level per launch and one
    3-scalar read) instead of the F2(a) batch kernel (16 levels unrolled into one kernel).  The BFS is the same (same
    reachable set, same exact level count); only how many levels go into one launch differs.  Reason: the batch
    kernel's CUDA compile took ~110 s and more than 16 GB host RSS on the RTX 3080 (smoke 2026-10-12, py-spy at
    gpu_fluid_reachability.flood_frontier:60), over the REQ_381 cap; the fast mode floods once per setup / mask change."""
    from simulation_core.fluids import gpu_fluid_reachability as gpu_module
    from simulation_core.fluids import solver as solver_module
    saved = (solver_module.HIBM_REACHABILITY_FLOOD_BATCH_LEVELS, gpu_module.HIBM_REACHABILITY_FLOOD_BATCH_LEVELS)
    solver_module.HIBM_REACHABILITY_FLOOD_BATCH_LEVELS = FAST_FLOOD_BATCH_LEVELS
    gpu_module.HIBM_REACHABILITY_FLOOD_BATCH_LEVELS = FAST_FLOOD_BATCH_LEVELS
    try:
        yield
    finally:
        solver_module.HIBM_REACHABILITY_FLOOD_BATCH_LEVELS, gpu_module.HIBM_REACHABILITY_FLOOD_BATCH_LEVELS = saved


def replace_obstacle_cells(fluid, mask, *, pressure_outlet_zmin=True):
    """Set fluid.obstacle to `mask` (i32 0/1, grid shape), zeroing the velocity of newly blocked cells, then retire and
    rebuild every obstacle-topology-derived contract in the order the production solver requires: nullspace graph and
    external-topology state invalidated, velocity-Dirichlet ledger prepared and sealed, z-min outlet reachability
    re-marked.  Returns the reachability report (unreached cell count)."""
    mask = np.ascontiguousarray(np.asarray(mask, dtype=np.int32))
    if mask.shape != tuple(fluid.grid.grid_nodes):
        raise ValueError("obstacle mask shape %s != grid %s" % (mask.shape, tuple(fluid.grid.grid_nodes)))
    _write_obstacle_kernel(fluid.obstacle, fluid.velocity, fluid.velocity_prev, mask)
    fluid._invalidate_pressure_nullspace_component_graph()
    fluid._invalidate_external_obstacle_topology_derived_state()
    fluid.prepare_and_seal_velocity_dirichlet_component_ledger()
    with single_level_reachability_flood():
        return fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=bool(pressure_outlet_zmin))
