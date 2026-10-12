"""Opt-in complete component storage; native graphs and gates are unchanged."""
import math
import taichi as ti

from .constants import HIBM_PRESSURE_COMPONENT_CAPACITY
from .solver import CartesianFluidSolver


def component_capacity(grid_nodes):
    # Every component owns at least one cell. Retain the legacy floor for tiny
    # grids. All fields are allocated once, before any native kernel compiles.
    return max(HIBM_PRESSURE_COMPONENT_CAPACITY, math.prod(grid_nodes))


def policy_identity(grid_nodes):
    return dict(revision="F1-complete-pressure-component-linear-compaction-v4-no-free-face-all-sites",
        capacity_rule="max_legacy_capacity_grid_cell_count",
        capacity=component_capacity(grid_nodes),
        scope="physical_air_cleanup_closed_and_outlet_operator_components",
        overflow_gate="unchanged",
        no_free_face_cells="solid_for_projection",
        no_free_face_scope="isolated_unanchored_singleton_after_interface_merge",
        no_free_face_authority="REQ_369 user 2026-10-11 无自由面格子计为固体",
        no_free_face_sites="cpu+cuda: outlet operator graph (both contexts), unreached flood (interface-coupled kept), "
                           "closed-Neumann physical budget (removed components skipped); runtime coverage guard")


@ti.data_oriented
class CompletePressureComponentFluid(CartesianFluidSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pressure_component_raw_to_compact = ti.field(
            ti.i32, shape=self.nx*self.ny*self.nz)

    @property
    def pressure_component_capacity(self):
        return component_capacity((self.nx, self.ny, self.nz))

    @ti.kernel
    def _assign_hibm_unreached_compact_component_labels_kernel(self, component_count: ti.i32):
        # Raw roots are flattened grid-cell indices. Separate offloaded loops
        # give a barrier before consumers and O(N+C) work even at full capacity.
        for root in self.pressure_component_raw_to_compact:
            self.pressure_component_raw_to_compact[root] = 0
        for component in range(ti.min(component_count, self.pressure_component_capacity)):
            root = self.hibm_unreached_component_compact_raw_label[component]
            if 0 <= root < self.nx*self.ny*self.nz:
                self.pressure_component_raw_to_compact[root] = -(component+1)
        for i, j, k in self.obstacle:
            if (self.obstacle[i,j,k] == 0
                    and self.hibm_pressure_reachability_barrier[i,j,k] == 0
                    and self.hibm_pressure_outlet_reachable[i,j,k] == 0):
                root = self.hibm_pressure_unreached_component_label[i,j,k]
                if 0 <= root < self.nx*self.ny*self.nz:
                    compact = self.pressure_component_raw_to_compact[root]
                    if compact < 0:
                        self.hibm_pressure_unreached_component_label[i,j,k] = compact
