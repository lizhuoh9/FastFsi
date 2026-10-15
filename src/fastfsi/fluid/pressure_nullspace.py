"""The pressure operator's component labels, nullspace graph and pocket means (mixin)."""

import taichi as ti

from fastfsi.fluid.common import COMPONENT_LABEL_BATCH_SWEEPS, PressureSolveConvergenceError
from fastfsi.fluid.interface_constants import PRESSURE_INTERFACE_COUPLING_SLOT_COUNT


class PressureNullspace:
    """Mixin of IncompressibleFluidSolver; not used on its own.  Contract (solver attributes):
      reads:  _pressure_outlet_nullspace_graph_context, _pressure_outlet_nullspace_graph_valid,
    _pressure_outlet_nullspace_source_component_count, _pressure_outlet_operator_component_count,
    _subtract_pressure_outlet_nullspace_component_means_kernel,
    _subtract_pressure_outlet_operator_component_means_kernel, _subtract_unreached_component_means_kernel,
    cg_unreached_component_sum, cg_unreached_component_volume, cg_weighted_mean, pressure_outlet_reachable,
    pressure_reachability_barrier, pressure_unreached_component_label,
    last_cg_closed_neumann_physical_component_budgets, last_cg_closed_neumann_physical_rhs_integral,
    last_cg_closed_neumann_physical_rhs_mean, last_cg_closed_neumann_physical_rhs_measured,
    last_pressure_component_labels_converged, last_pressure_unreached_component_overflow, nx, ny, nz,
    obstacle, pressure_interface_coupling_active, pressure_interface_coupling_coefficient,
    pressure_interface_coupling_extra_coefficient, pressure_interface_coupling_extra_neighbor,
    pressure_interface_coupling_neighbor, pressure_interface_matrix_diagonal, pressure_interface_row_capacity,
    pressure_interface_row_count, pressure_interface_row_neighbor, pressure_interface_row_owner,
    pressure_interface_row_support_count, pressure_interface_row_transmissibility,
    pressure_outlet_operator_active_unanchored_component_count, pressure_outlet_operator_component_label,
    pressure_outlet_operator_empty_singleton_component_count, pressure_outlet_operator_raw_component_anchored,
    pressure_outlet_operator_raw_component_cell_count, pressure_outlet_operator_raw_component_count,
    pressure_outlet_operator_raw_component_diagonal_excess_integral,
    pressure_outlet_operator_raw_component_diagonal_scale_integral,
    pressure_outlet_operator_raw_component_diagonal_term_count, pressure_outlet_operator_raw_component_to_compact,
    pressure_solve_component_anchored, pressure_solve_component_count,
    pressure_solve_component_diagonal_excess_integral, pressure_solve_component_diagonal_scale_integral,
    pressure_solve_component_diagonal_term_count, pressure_solve_component_invalid, pressure_solve_component_parent,
    reduction_count
      writes: _pressure_outlet_nullspace_component_count, _pressure_outlet_nullspace_graph_context,
    _pressure_outlet_nullspace_graph_valid, _pressure_outlet_nullspace_source_component_count,
    _pressure_outlet_operator_component_count, last_unreached_component_rhs_integral_max_abs,
    last_unreached_component_rhs_mean_max_abs, last_unreached_incompatible_component_count"""

    @ti.kernel
    def _weighted_unreached_set_mean_to_cg_field_kernel(self, field: ti.template()):
        total = ti.cast(0.0, ti.f64)
        volume_total = ti.cast(0.0, ti.f64)
        for i, j, k in field:
            if (
                self.obstacle[i, j, k] == 0
                and self.pressure_reachability_barrier[i, j, k] == 0
                and self.pressure_outlet_reachable[i, j, k] == 0
            ):
                volume_m3 = self._cell_volume_f64_m3(i, j, k)
                total += ti.cast(field[i, j, k], ti.f64) * volume_m3
                volume_total += volume_m3
        self.cg_weighted_mean[None] = 0.0
        if volume_total > 0.0:
            self.cg_weighted_mean[None] = total / volume_total

    @ti.kernel
    def _subtract_cg_weighted_mean_from_unreached_set_kernel(
        self,
        field: ti.template(),
    ):
        mean = self.cg_weighted_mean[None]
        for i, j, k in field:
            if (
                self.obstacle[i, j, k] == 0
                and self.pressure_reachability_barrier[i, j, k] == 0
                and not self._velocity_dirichlet_pressure_barrier(i, j, k)
                and self.pressure_outlet_reachable[i, j, k] == 0
            ):
                field[i, j, k] -= mean

    @ti.kernel
    def _init_pressure_outlet_operator_component_labels_kernel(self):
        self.pressure_outlet_operator_raw_component_count[None] = 0
        self.pressure_solve_component_invalid[None] = 0
        for i, j, k in self.obstacle:
            label = 1 << 30
            if self.obstacle[i, j, k] == 0:
                label = (i * self.ny + j) * self.nz + k
            self.pressure_outlet_operator_component_label[i, j, k] = label
            self.pressure_outlet_operator_raw_component_cell_count[i, j, k] = 0
            self.pressure_outlet_operator_raw_component_to_compact[i, j, k] = -1
            self.pressure_outlet_operator_raw_component_anchored[i, j, k] = 0
            self.pressure_outlet_operator_raw_component_diagonal_excess_integral[
                i, j, k
            ] = 0.0
            self.pressure_outlet_operator_raw_component_diagonal_scale_integral[
                i, j, k
            ] = 0.0
            self.pressure_outlet_operator_raw_component_diagonal_term_count[
                i, j, k
            ] = 0

    @ti.func
    def _relax_pressure_outlet_operator_grid_label_cell(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
    ) -> ti.i32:
        """One min-label relaxation over grid faces; returns 1 if lowered."""
        changed = 0
        if self.obstacle[i, j, k] == 0:
            label = self.pressure_outlet_operator_component_label[i, j, k]
            best = label
            if i > 0:
                if (
                    self.obstacle[i - 1, j, k] == 0
                    and self._pressure_cells_connected(i, j, k, i - 1, j, k)
                ):
                    best = ti.min(
                        best,
                        self.pressure_outlet_operator_component_label[i - 1, j, k],
                    )
            if i < self.nx - 1:
                if (
                    self.obstacle[i + 1, j, k] == 0
                    and self._pressure_cells_connected(i, j, k, i + 1, j, k)
                ):
                    best = ti.min(
                        best,
                        self.pressure_outlet_operator_component_label[i + 1, j, k],
                    )
            if j > 0:
                if (
                    self.obstacle[i, j - 1, k] == 0
                    and self._pressure_cells_connected(i, j, k, i, j - 1, k)
                ):
                    best = ti.min(
                        best,
                        self.pressure_outlet_operator_component_label[i, j - 1, k],
                    )
            if j < self.ny - 1:
                if (
                    self.obstacle[i, j + 1, k] == 0
                    and self._pressure_cells_connected(i, j, k, i, j + 1, k)
                ):
                    best = ti.min(
                        best,
                        self.pressure_outlet_operator_component_label[i, j + 1, k],
                    )
            if k > 0:
                if (
                    self.obstacle[i, j, k - 1] == 0
                    and self._pressure_cells_connected(i, j, k, i, j, k - 1)
                ):
                    best = ti.min(
                        best,
                        self.pressure_outlet_operator_component_label[i, j, k - 1],
                    )
            if k < self.nz - 1:
                if (
                    self.obstacle[i, j, k + 1] == 0
                    and self._pressure_cells_connected(i, j, k, i, j, k + 1)
                ):
                    best = ti.min(
                        best,
                        self.pressure_outlet_operator_component_label[i, j, k + 1],
                    )
            if best < label:
                self.pressure_outlet_operator_component_label[i, j, k] = best
                changed = 1
        return changed

    @ti.kernel
    def _propagate_pressure_outlet_operator_grid_labels_sweep_kernel(self):
        changed = 0
        for i, j, k in self.obstacle:
            changed += self._relax_pressure_outlet_operator_grid_label_cell(i, j, k)
        ti.atomic_add(self.reduction_count[None], changed)

    @ti.func
    def _relax_pressure_outlet_operator_interface_edge(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        ni: ti.i32,
        nj: ti.i32,
        nk: ti.i32,
    ) -> ti.i32:
        owner_label = self.pressure_outlet_operator_component_label[i, j, k]
        neighbor_label = self.pressure_outlet_operator_component_label[ni, nj, nk]
        best = ti.min(owner_label, neighbor_label)
        changed = 0
        old_owner = ti.atomic_min(
            self.pressure_outlet_operator_component_label[i, j, k], best
        )
        old_neighbor = ti.atomic_min(
            self.pressure_outlet_operator_component_label[ni, nj, nk], best
        )
        if old_owner > best:
            changed += 1
        if old_neighbor > best:
            changed += 1
        return changed

    @ti.func
    def _relax_pressure_outlet_operator_interface_row(
        self,
        row: ti.i32,
        row_limit: ti.i32,
    ):
        """Relax the representative edge of one explicit interface row.

        For a normal-ray row the producer verifies a connected donor support;
        this edge encodes its only possible base-component bridge.
        """
        if row < row_limit:
            owner = self.pressure_interface_row_owner[row]
            neighbor = self.pressure_interface_row_neighbor[row]
            i, j, k = owner.x, owner.y, owner.z
            ni, nj, nk = neighbor.x, neighbor.y, neighbor.z
            transmissibility = self.pressure_interface_row_transmissibility[row]
            if (
                0 <= i < self.nx
                and 0 <= j < self.ny
                and 0 <= k < self.nz
                and 0 <= ni < self.nx
                and 0 <= nj < self.ny
                and 0 <= nk < self.nz
                and self.obstacle[i, j, k] == 0
                and self.obstacle[ni, nj, nk] == 0
                and not ti.math.isnan(transmissibility)
                and not ti.math.isinf(transmissibility)
                and transmissibility > 0.0
                and (i != ni or j != nj or k != nk)
            ):
                ti.atomic_add(
                    self.reduction_count[None],
                    self._relax_pressure_outlet_operator_interface_edge(
                        i, j, k, ni, nj, nk
                    ),
                )

    @ti.func
    def _relax_pressure_outlet_operator_interface_cell(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
    ):
        """Relax the per-cell coupling-slot edges (legacy slot storage)."""
        active_count = self.pressure_interface_coupling_active[i, j, k]
        if active_count > 0 and self.obstacle[i, j, k] == 0:
            for coupling_slot in ti.static(
                range(PRESSURE_INTERFACE_COUPLING_SLOT_COUNT)
            ):
                if coupling_slot < active_count:
                    neighbor = self.pressure_interface_coupling_neighbor[
                        i, j, k
                    ]
                    transmissibility = (
                        self.pressure_interface_coupling_coefficient[i, j, k]
                    )
                    if ti.static(coupling_slot > 0):
                        extra_slot = coupling_slot - 1
                        neighbor = (
                            self.pressure_interface_coupling_extra_neighbor[
                                i, j, k, extra_slot
                            ]
                        )
                        transmissibility = (
                            self.pressure_interface_coupling_extra_coefficient[
                                i, j, k, extra_slot
                            ]
                        )
                    ni, nj, nk = neighbor.x, neighbor.y, neighbor.z
                    if (
                        0 <= ni < self.nx
                        and 0 <= nj < self.ny
                        and 0 <= nk < self.nz
                        and self.obstacle[ni, nj, nk] == 0
                        and not ti.math.isnan(transmissibility)
                        and not ti.math.isinf(transmissibility)
                        and transmissibility > 0.0
                        and (i != ni or j != nj or k != nk)
                    ):
                        ti.atomic_add(
                            self.reduction_count[None],
                            self._relax_pressure_outlet_operator_interface_edge(
                                i, j, k, ni, nj, nk
                            ),
                        )

    @ti.kernel
    def _propagate_pressure_outlet_operator_interface_labels_kernel(self):
        row_count = self.pressure_interface_row_count[None]
        row_limit = ti.max(
            0,
            ti.min(row_count, self.pressure_interface_row_capacity),
        )
        for row in range(ti.min(row_limit, self.pressure_interface_row_capacity)):
            self._relax_pressure_outlet_operator_interface_row(row, row_limit)
        for i, j, k in self.obstacle:
            if row_count == 0:
                self._relax_pressure_outlet_operator_interface_cell(i, j, k)

    @ti.kernel
    def _jump_pressure_outlet_operator_component_labels_kernel(self):
        """Pointer jumping: label <- label[cell(label)].

        Every fluid label is the linear index of a cell of the same operator
        component and labels only decrease, so the jump keeps that invariant
        and the fixed point (each component's minimum index) while shortening
        the propagation distance from ~the component diameter to its log.
        """

        changed = 0
        for i, j, k in self.obstacle:
            if self.obstacle[i, j, k] == 0:
                label = self.pressure_outlet_operator_component_label[i, j, k]
                root_i = label // (self.ny * self.nz)
                remainder = label - root_i * self.ny * self.nz
                root_j = remainder // self.nz
                root_k = remainder - root_j * self.nz
                if (
                    0 <= label
                    and 0 <= root_i < self.nx
                    and 0 <= root_j < self.ny
                    and 0 <= root_k < self.nz
                ):
                    root_label = self.pressure_outlet_operator_component_label[
                        root_i, root_j, root_k
                    ]
                    if root_label < label:
                        previous = ti.atomic_min(
                            self.pressure_outlet_operator_component_label[i, j, k],
                            root_label,
                        )
                        if previous > root_label:
                            changed += 1
        ti.atomic_add(self.reduction_count[None], changed)

    def _propagate_pressure_outlet_operator_labels_batch(self) -> int:
        self.reduction_count[None] = 0
        for _ in range(COMPONENT_LABEL_BATCH_SWEEPS):
            self._propagate_pressure_outlet_operator_grid_labels_sweep_kernel()
            self._propagate_pressure_outlet_operator_interface_labels_kernel()
            self._jump_pressure_outlet_operator_component_labels_kernel()
        return int(self.reduction_count[None])

    @ti.kernel
    def _accumulate_pressure_outlet_operator_raw_components_kernel(self):
        for i, j, k in self.obstacle:
            if self.obstacle[i, j, k] == 0:
                label = self.pressure_outlet_operator_component_label[i, j, k]
                root_i = label // (self.ny * self.nz)
                remainder = label - root_i * self.ny * self.nz
                root_j = remainder // self.nz
                root_k = remainder - root_j * self.nz
                if (
                    0 <= root_i < self.nx
                    and 0 <= root_j < self.ny
                    and 0 <= root_k < self.nz
                ):
                    ti.atomic_add(
                        self.pressure_outlet_operator_raw_component_cell_count[
                            root_i, root_j, root_k
                        ],
                        1,
                    )
                    if k == 0 and not self._pressure_face_has_zero_mobility(
                        i, j, k, 4
                    ):
                        ti.atomic_or(
                            self.pressure_outlet_operator_raw_component_anchored[
                                root_i, root_j, root_k
                            ],
                            1,
                        )
                    diagonal_density = ti.cast(
                        self.pressure_interface_matrix_diagonal[i, j, k], ti.f64
                    )
                    cell_volume_m3 = self._cell_volume_f64_m3(i, j, k)
                    diagonal_integral = diagonal_density * cell_volume_m3
                    invalid = (
                        ti.math.isnan(diagonal_density)
                        or ti.math.isinf(diagonal_density)
                        or ti.math.isnan(cell_volume_m3)
                        or ti.math.isinf(cell_volume_m3)
                        or ti.math.isnan(diagonal_integral)
                        or ti.math.isinf(diagonal_integral)
                        or diagonal_density < 0.0
                        or cell_volume_m3 <= 0.0
                    )
                    if invalid:
                        ti.atomic_add(
                            self.pressure_solve_component_invalid[None], 1
                        )
                    else:
                        ti.atomic_add(
                            self.pressure_outlet_operator_raw_component_diagonal_excess_integral[
                                root_i, root_j, root_k
                            ],
                            diagonal_integral,
                        )
                        ti.atomic_add(
                            self.pressure_outlet_operator_raw_component_diagonal_scale_integral[
                                root_i, root_j, root_k
                            ],
                            ti.abs(diagonal_integral),
                        )
                        if diagonal_integral != 0.0:
                            ti.atomic_add(
                                self.pressure_outlet_operator_raw_component_diagonal_term_count[
                                    root_i, root_j, root_k
                                ],
                                1,
                            )
                else:
                    ti.atomic_add(self.pressure_solve_component_invalid[None], 1)

    @ti.func
    def _subtract_pressure_outlet_operator_edge_from_raw_roots(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        ni: ti.i32,
        nj: ti.i32,
        nk: ti.i32,
        transmissibility: ti.f64,
    ):
        edge_integral = ti.cast(transmissibility, ti.f64)
        owner_label = self.pressure_outlet_operator_component_label[i, j, k]
        owner_root_i = owner_label // (self.ny * self.nz)
        owner_remainder = owner_label - owner_root_i * self.ny * self.nz
        owner_root_j = owner_remainder // self.nz
        owner_root_k = owner_remainder - owner_root_j * self.nz
        neighbor_label = self.pressure_outlet_operator_component_label[ni, nj, nk]
        neighbor_root_i = neighbor_label // (self.ny * self.nz)
        neighbor_remainder = neighbor_label - neighbor_root_i * self.ny * self.nz
        neighbor_root_j = neighbor_remainder // self.nz
        neighbor_root_k = neighbor_remainder - neighbor_root_j * self.nz
        ti.atomic_add(
            self.pressure_outlet_operator_raw_component_diagonal_excess_integral[
                owner_root_i, owner_root_j, owner_root_k
            ],
            -edge_integral,
        )
        ti.atomic_add(
            self.pressure_outlet_operator_raw_component_diagonal_scale_integral[
                owner_root_i, owner_root_j, owner_root_k
            ],
            ti.abs(edge_integral),
        )
        ti.atomic_add(
            self.pressure_outlet_operator_raw_component_diagonal_term_count[
                owner_root_i, owner_root_j, owner_root_k
            ],
            1,
        )
        ti.atomic_add(
            self.pressure_outlet_operator_raw_component_diagonal_excess_integral[
                neighbor_root_i, neighbor_root_j, neighbor_root_k
            ],
            -edge_integral,
        )
        ti.atomic_add(
            self.pressure_outlet_operator_raw_component_diagonal_scale_integral[
                neighbor_root_i, neighbor_root_j, neighbor_root_k
            ],
            ti.abs(edge_integral),
        )
        ti.atomic_add(
            self.pressure_outlet_operator_raw_component_diagonal_term_count[
                neighbor_root_i, neighbor_root_j, neighbor_root_k
            ],
            1,
        )

    @ti.kernel
    def _subtract_pressure_outlet_operator_interface_edges_kernel(self):
        row_count = self.pressure_interface_row_count[None]
        row_limit = ti.max(
            0,
            ti.min(row_count, self.pressure_interface_row_capacity),
        )
        for row in range(ti.min(row_limit, self.pressure_interface_row_capacity)):
            if row < row_limit:
                owner = self.pressure_interface_row_owner[row]
                neighbor = self.pressure_interface_row_neighbor[row]
                i, j, k = owner.x, owner.y, owner.z
                ni, nj, nk = neighbor.x, neighbor.y, neighbor.z
                transmissibility = self.pressure_interface_row_transmissibility[row]
                if (
                    0 <= i < self.nx
                    and 0 <= j < self.ny
                    and 0 <= k < self.nz
                    and 0 <= ni < self.nx
                    and 0 <= nj < self.ny
                    and 0 <= nk < self.nz
                    and self.obstacle[i, j, k] == 0
                    and self.obstacle[ni, nj, nk] == 0
                    and not ti.math.isnan(transmissibility)
                    and not ti.math.isinf(transmissibility)
                    and transmissibility > 0.0
                    and (i != ni or j != nj or k != nk)
                ):
                    if self.pressure_interface_row_support_count[row] == 0:
                        self._subtract_pressure_outlet_operator_edge_from_raw_roots(
                            i, j, k, ni, nj, nk, transmissibility
                        )
                    else:
                        # The published diagonal contains every T * e_i**2.
                        # Representatives encode connectivity, not incidence.
                        for slot in range(ti.min(
                            self.pressure_interface_row_support_count[row], 9
                        )):
                            cell, coefficient = self._pressure_interface_row_support(row, slot)
                            diagonal_integral = transmissibility * coefficient * coefficient
                            label = self.pressure_outlet_operator_component_label[cell]
                            root_i = label // (self.ny * self.nz)
                            remainder = label - root_i * self.ny * self.nz
                            root_j = remainder // self.nz
                            root_k = remainder - root_j * self.nz
                            ti.atomic_add(
                                self.pressure_outlet_operator_raw_component_diagonal_excess_integral[
                                    root_i, root_j, root_k
                                ],
                                -diagonal_integral,
                            )
                            ti.atomic_add(
                                self.pressure_outlet_operator_raw_component_diagonal_scale_integral[
                                    root_i, root_j, root_k
                                ],
                                ti.abs(diagonal_integral),
                            )
                            ti.atomic_add(
                                self.pressure_outlet_operator_raw_component_diagonal_term_count[
                                    root_i, root_j, root_k
                                ],
                                1,
                            )
        for i, j, k in self.obstacle:
            if row_count == 0:
                active_count = self.pressure_interface_coupling_active[i, j, k]
                if active_count > 0 and self.obstacle[i, j, k] == 0:
                    for coupling_slot in ti.static(
                        range(PRESSURE_INTERFACE_COUPLING_SLOT_COUNT)
                    ):
                        if coupling_slot < active_count:
                            neighbor = self.pressure_interface_coupling_neighbor[
                                i, j, k
                            ]
                            transmissibility = (
                                self.pressure_interface_coupling_coefficient[i, j, k]
                            )
                            if ti.static(coupling_slot > 0):
                                extra_slot = coupling_slot - 1
                                neighbor = (
                                    self.pressure_interface_coupling_extra_neighbor[
                                        i, j, k, extra_slot
                                    ]
                                )
                                transmissibility = (
                                    self.pressure_interface_coupling_extra_coefficient[
                                        i, j, k, extra_slot
                                    ]
                                )
                            ni, nj, nk = neighbor.x, neighbor.y, neighbor.z
                            if (
                                0 <= ni < self.nx
                                and 0 <= nj < self.ny
                                and 0 <= nk < self.nz
                                and self.obstacle[ni, nj, nk] == 0
                                and not ti.math.isnan(transmissibility)
                                and not ti.math.isinf(transmissibility)
                                and transmissibility > 0.0
                                and (i != ni or j != nj or k != nk)
                            ):
                                self._subtract_pressure_outlet_operator_edge_from_raw_roots(
                                    i, j, k, ni, nj, nk, transmissibility
                                )

    @ti.kernel
    def _finalize_pressure_outlet_operator_raw_roots_kernel(self):
        self.pressure_outlet_operator_raw_component_count[None] = 0
        self.pressure_outlet_operator_empty_singleton_component_count[None] = 0
        self.pressure_outlet_operator_active_unanchored_component_count[None] = 0
        self.pressure_solve_component_count[None] = 0
        for component in range(self.pressure_component_capacity):
            self.pressure_solve_component_parent[component] = -1
            self.pressure_solve_component_anchored[component] = 0
            self.pressure_solve_component_diagonal_excess_integral[component] = 0.0
            self.pressure_solve_component_diagonal_scale_integral[component] = 0.0
            self.pressure_solve_component_diagonal_term_count[component] = 0
        for i, j, k in self.pressure_outlet_operator_raw_component_cell_count:
            cell_count = (
                self.pressure_outlet_operator_raw_component_cell_count[i, j, k]
            )
            if cell_count > 0:
                ti.atomic_add(
                    self.pressure_outlet_operator_raw_component_count[None], 1
                )
                excess = (
                    self.pressure_outlet_operator_raw_component_diagonal_excess_integral[
                        i, j, k
                    ]
                )
                scale = (
                    self.pressure_outlet_operator_raw_component_diagonal_scale_integral[
                        i, j, k
                    ]
                )
                raw_term_count_i32 = (
                    self.pressure_outlet_operator_raw_component_diagonal_term_count[
                        i, j, k
                    ]
                )
                term_count_i32 = ti.max(
                    raw_term_count_i32,
                    1,
                )
                term_count = ti.cast(term_count_i32, ti.f64)
                unit_roundoff = ti.cast(1.1102230246251565e-16, ti.f64)
                operation_count = 4.0 * term_count + 4.0
                rounding_product = operation_count * unit_roundoff
                gamma_n = rounding_product / ti.max(
                    1.0 - rounding_product,
                    0.5,
                )
                tolerance = gamma_n * ti.max(scale, 1.0e-300)
                invalid_numeric = (
                    ti.math.isnan(excess)
                    or ti.math.isinf(excess)
                    or ti.math.isnan(scale)
                    or ti.math.isinf(scale)
                    or scale < 0.0
                )
                if invalid_numeric or excess < -tolerance:
                    ti.atomic_add(self.pressure_solve_component_invalid[None], 1)
                elif (
                    self.pressure_outlet_operator_raw_component_anchored[i, j, k]
                    == 0
                    and excess <= tolerance
                ):
                    operator_empty_singleton = (
                        cell_count == 1
                        and raw_term_count_i32 == 0
                        and scale == 0.0
                        and excess == 0.0
                    )
                    if operator_empty_singleton:
                        ti.atomic_add(
                            self.pressure_outlet_operator_empty_singleton_component_count[
                                None
                            ],
                            1,
                        )
                    else:
                        ti.atomic_add(
                            self.pressure_outlet_operator_active_unanchored_component_count[
                                None
                            ],
                            1,
                        )
                    compact = ti.atomic_add(
                        self.pressure_solve_component_count[None], 1
                    )
                    if compact < self.pressure_component_capacity:
                        self.pressure_outlet_operator_raw_component_to_compact[
                            i, j, k
                        ] = compact
                        self.pressure_solve_component_parent[compact] = compact
                        self.pressure_solve_component_diagonal_excess_integral[
                            compact
                        ] = excess
                        self.pressure_solve_component_diagonal_scale_integral[
                            compact
                        ] = scale
                        self.pressure_solve_component_diagonal_term_count[
                            compact
                        ] = term_count_i32

    @ti.kernel
    def _publish_pressure_outlet_operator_compact_labels_kernel(
        self,
        component_count: ti.i32,
    ):
        for i, j, k in self.obstacle:
            compact = -1
            if self.obstacle[i, j, k] == 0:
                label = self.pressure_outlet_operator_component_label[i, j, k]
                root_i = label // (self.ny * self.nz)
                remainder = label - root_i * self.ny * self.nz
                root_j = remainder // self.nz
                root_k = remainder - root_j * self.nz
                if (
                    0 <= root_i < self.nx
                    and 0 <= root_j < self.ny
                    and 0 <= root_k < self.nz
                ):
                    candidate = (
                        self.pressure_outlet_operator_raw_component_to_compact[
                            root_i, root_j, root_k
                        ]
                    )
                    if 0 <= candidate < component_count:
                        compact = candidate
            self.pressure_outlet_operator_component_label[i, j, k] = compact

    @ti.func
    def _pressure_outlet_physical_component(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        component_count: ti.i32,
    ) -> ti.i32:
        component = -1
        if (
            self.obstacle[i, j, k] == 0
            and self.pressure_reachability_barrier[i, j, k] == 0
            and self.pressure_outlet_reachable[i, j, k] == 0
        ):
            label = self.pressure_unreached_component_label[i, j, k]
            candidate = -label - 1
            if label < 0 and 0 <= candidate < component_count:
                component = candidate
        return component

    def _invalidate_pressure_nullspace_component_graph(self) -> None:
        self._pressure_outlet_nullspace_graph_valid = False
        self._pressure_outlet_nullspace_source_component_count = 0
        self._pressure_outlet_operator_component_count = 0
        self._pressure_outlet_nullspace_component_count = 0
        self._pressure_outlet_nullspace_graph_context = ""

    def _prepare_pressure_outlet_nullspace_component_graph(self) -> tuple[int, int]:
        """Build the outlet-disconnected exact-operator nullspace graph."""

        context = "outlet-disconnected"
        physical_compact_count = int(
            getattr(self, "_pressure_unreached_component_count", 0)
        )
        source_physical_count = max(
            0,
            int(getattr(self, "last_pressure_unreached_component_raw_count", 0)),
            physical_compact_count,
        )
        self._invalidate_pressure_nullspace_component_graph()

        def fail_graph(
            message: str,
            *,
            reason: str,
            count: int,
            labels_converged: bool,
            component_overflow: bool = False,
            **extra: object,
        ) -> None:
            self._invalidate_pressure_nullspace_component_graph()
            diagnostics: dict[str, object] = {
                "stage": "pressure_nullspace_component_graph",
                "context": context,
                "reason": str(reason),
                "count": int(count),
                "capacity": int(self.pressure_component_capacity),
                "labels_converged": bool(labels_converged),
                "component_overflow": bool(component_overflow),
                "source_physical_component_count": int(source_physical_count),
            }
            diagnostics.update(extra)
            raise PressureSolveConvergenceError(
                message,
                diagnostics=diagnostics,
            )

        if not bool(self.last_pressure_component_labels_converged):
            fail_graph(
                "outlet physical pressure component labels are not converged",
                reason="component_labels_not_converged",
                count=source_physical_count,
                labels_converged=False,
                component_overflow=bool(
                    self.last_pressure_unreached_component_overflow
                ),
            )
        physical_overflow = bool(
            self.last_pressure_unreached_component_overflow
        )
        if physical_compact_count < 0 or (
            physical_compact_count > self.pressure_component_capacity
            and not physical_overflow
        ):
            fail_graph(
                "outlet physical pressure component count is outside device "
                "capacity without a matching overflow registry",
                reason="component_count_out_of_range",
                count=physical_compact_count,
                labels_converged=True,
                component_overflow=physical_overflow,
            )
        self._preflight_pressure_interface_operator_storage(
            context=f"{context}-graph",
        )
        self._init_pressure_outlet_operator_component_labels_kernel()
        # Interface rows are non-local graph edges and can form an arbitrary
        # chain, so a geometric grid-diameter bound is not sufficient.  The
        # number of active cells is the finite monotone-label upper bound; the
        # normal path still exits after the first unchanged batch.
        propagation_sweep_budget = max(1, int(self.nx * self.ny * self.nz))
        # A batch reports the sum of changes across all of its sweeps.  Keep
        # one extra batch so labels that stabilize inside the last propagation
        # batch can still produce the all-zero confirmation required here.
        max_sweeps = (
            propagation_sweep_budget + COMPONENT_LABEL_BATCH_SWEEPS
        )
        labels_converged = False
        label_sweeps_done = 0
        while label_sweeps_done < max_sweeps:
            batch_changed = self._propagate_pressure_outlet_operator_labels_batch()
            label_sweeps_done += COMPONENT_LABEL_BATCH_SWEEPS
            if batch_changed == 0:
                labels_converged = True
                break
        if not labels_converged:
            fail_graph(
                "outlet exact pressure-operator component labels are not converged",
                reason="component_labels_not_converged",
                count=source_physical_count,
                labels_converged=False,
                label_sweep_count=label_sweeps_done,
            )

        self._accumulate_pressure_outlet_operator_raw_components_kernel()
        self._subtract_pressure_outlet_operator_interface_edges_kernel()
        invalid_count = int(self.pressure_solve_component_invalid[None])
        if invalid_count != 0:
            fail_graph(
                "pressure-interface graph is invalid for outlet nullspace classification",
                reason="interface_graph_invalid",
                count=source_physical_count,
                labels_converged=True,
                invalid_count=invalid_count,
            )

        self._finalize_pressure_outlet_operator_raw_roots_kernel()
        operator_raw_count = int(
            self.pressure_outlet_operator_raw_component_count[None]
        )
        final_unanchored_count = int(self.pressure_solve_component_count[None])
        invalid_count = int(self.pressure_solve_component_invalid[None])
        if invalid_count != 0:
            fail_graph(
                "pressure-interface diagonal provenance does not match its edge graph",
                reason="diagonal_provenance_mismatch",
                count=final_unanchored_count,
                labels_converged=True,
                invalid_count=invalid_count,
                operator_raw_component_count=operator_raw_count,
            )
        if final_unanchored_count > self.pressure_component_capacity:
            fail_graph(
                "outlet exact pressure-operator nullspace component count exceeds "
                "device capacity",
                reason="component_capacity_overflow",
                count=final_unanchored_count,
                labels_converged=True,
                component_overflow=True,
                operator_raw_component_count=operator_raw_count,
                operator_empty_singleton_component_count=int(
                    self.pressure_outlet_operator_empty_singleton_component_count[
                        None
                    ]
                ),
                operator_active_unanchored_component_count=int(
                    self.pressure_outlet_operator_active_unanchored_component_count[
                        None
                    ]
                ),
            )

        self._publish_pressure_outlet_operator_compact_labels_kernel(
            final_unanchored_count
        )
        self._pressure_outlet_nullspace_source_component_count = (
            source_physical_count
        )
        self._pressure_outlet_operator_component_count = final_unanchored_count
        self._pressure_outlet_nullspace_component_count = final_unanchored_count
        self._pressure_outlet_nullspace_graph_context = context
        self._pressure_outlet_nullspace_graph_valid = True
        return source_physical_count, final_unanchored_count

    @ti.kernel
    def _accumulate_unreached_component_means_kernel(
        self,
        field: ti.template(),
        component_count: ti.i32,
    ):
        for c in range(self.pressure_component_capacity):
            self.cg_unreached_component_sum[c] = 0.0
            self.cg_unreached_component_volume[c] = 0.0
        for i, j, k in field:
            if (
                self.obstacle[i, j, k] == 0
                and self.pressure_reachability_barrier[i, j, k] == 0
                and self.pressure_outlet_reachable[i, j, k] == 0
            ):
                component = (
                    -self.pressure_unreached_component_label[i, j, k] - 1
                )
                if 0 <= component < component_count:
                    volume_m3 = self._cell_volume_f64_m3(i, j, k)
                    ti.atomic_add(
                        self.cg_unreached_component_sum[component],
                        ti.cast(field[i, j, k], ti.f64) * volume_m3,
                    )
                    ti.atomic_add(
                        self.cg_unreached_component_volume[component],
                        volume_m3,
                    )

    @ti.kernel
    def _accumulate_pressure_outlet_operator_component_means_kernel(
        self,
        field: ti.template(),
        component_count: ti.i32,
    ):
        for component in range(self.pressure_component_capacity):
            self.cg_unreached_component_sum[component] = 0.0
            self.cg_unreached_component_volume[component] = 0.0
        for i, j, k in field:
            component = self.pressure_outlet_operator_component_label[i, j, k]
            if (
                self.obstacle[i, j, k] == 0
                and 0 <= component < component_count
            ):
                volume_m3 = self._cell_volume_f64_m3(i, j, k)
                ti.atomic_add(
                    self.cg_unreached_component_sum[component],
                    ti.cast(field[i, j, k], ti.f64) * volume_m3,
                )
                ti.atomic_add(
                    self.cg_unreached_component_volume[component],
                    volume_m3,
                )

    @ti.kernel
    def _accumulate_pressure_outlet_nullspace_component_means_kernel(
        self,
        field: ti.template(),
        physical_component_count: ti.i32,
    ):
        for component in range(self.pressure_component_capacity):
            self.cg_unreached_component_sum[component] = 0.0
            self.cg_unreached_component_volume[component] = 0.0
        for i, j, k in field:
            component = self._pressure_outlet_physical_component(
                i,
                j,
                k,
                physical_component_count,
            )
            if component >= 0:
                root = self.pressure_solve_component_parent[component]
                if self.pressure_solve_component_anchored[root] == 0:
                    volume_m3 = self._cell_volume_f64_m3(i, j, k)
                    ti.atomic_add(
                        self.cg_unreached_component_sum[root],
                        ti.cast(field[i, j, k], ti.f64) * volume_m3,
                    )
                    ti.atomic_add(
                        self.cg_unreached_component_volume[root],
                        volume_m3,
                    )

    def _subtract_pressure_outlet_nullspace_component_means_device(
        self,
        field: ti.template(),
    ) -> None:
        if not bool(self._pressure_outlet_nullspace_graph_valid):
            raise self._pressure_operator_graph_missing_or_stale_error(
                context="pressure-outlet-nullspace-mean-subtraction",
            )
        if self._pressure_outlet_nullspace_graph_context == "outlet-disconnected":
            operator_component_count = int(
                self._pressure_outlet_operator_component_count
            )
            if operator_component_count <= 0:
                return
            self._accumulate_pressure_outlet_operator_component_means_kernel(
                field,
                operator_component_count,
            )
            self._subtract_pressure_outlet_operator_component_means_kernel(
                field,
                operator_component_count,
            )
            return
        physical_count = int(
            self._pressure_outlet_nullspace_source_component_count
        )
        if physical_count <= 0:
            return
        self._accumulate_pressure_outlet_nullspace_component_means_kernel(
            field,
            physical_count,
        )
        self._subtract_pressure_outlet_nullspace_component_means_kernel(
            field,
            physical_count,
        )

    def _subtract_pressure_component_means_device(
        self,
        field: ti.template(),
        component_count: int,
    ) -> None:
        component_count = max(0, int(component_count))
        if component_count > 0:
            self._accumulate_unreached_component_means_kernel(
                field,
                component_count,
            )
            self._subtract_unreached_component_means_kernel(
                field,
                component_count,
            )
        else:
            self._weighted_unreached_set_mean_to_cg_field_kernel(field)
            self._subtract_cg_weighted_mean_from_unreached_set_kernel(field)

    def _closed_neumann_rhs_compatibility_diagnostics(self) -> dict[str, object]:
        if not self.last_cg_closed_neumann_physical_rhs_measured:
            return {}
        return {
            "pressure_nullspace_compatibility_basis": "physical_boundary_flux_source_affine_f64",
            "pressure_nullspace_physical_rhs_mean": float(
                self.last_cg_closed_neumann_physical_rhs_mean
            ),
            "pressure_nullspace_physical_rhs_integral": float(
                self.last_cg_closed_neumann_physical_rhs_integral
            ),
            "pressure_nullspace_physical_component_budgets": [
                dict(budget)
                for budget in self.last_cg_closed_neumann_physical_component_budgets
            ],
        }

    def _measure_unreached_component_rhs_incompatibility(
        self,
        field: ti.template(),
        *,
        component_count: int | None = None,
        mean_abs_tolerance: float = 1.0e-12,
    ) -> int:
        if component_count is None:
            component_count = int(
                getattr(self, "_pressure_unreached_component_count", 0)
            )
        component_count = max(0, int(component_count))
        self.last_unreached_incompatible_component_count = 0
        self.last_unreached_component_rhs_mean_max_abs = 0.0
        self.last_unreached_component_rhs_integral_max_abs = 0.0
        if component_count <= 0:
            return 0
        self._accumulate_unreached_component_means_kernel(
            field,
            component_count,
        )
        incompatible_count = 0
        max_abs_mean = 0.0
        max_abs_integral = 0.0
        tolerance = max(0.0, float(mean_abs_tolerance))
        volumes = self.cg_unreached_component_volume.to_numpy()[:component_count]
        integrals = self.cg_unreached_component_sum.to_numpy()[:component_count]
        removed_roots = frozenset()
        for component, (volume, integral) in enumerate(zip(volumes.tolist(), integrals.tolist())):
            if component in removed_roots:
                continue
            abs_integral = abs(integral)
            max_abs_integral = max(max_abs_integral, abs_integral)
            if volume <= 0.0:
                continue
            abs_mean = abs(integral / volume)
            max_abs_mean = max(max_abs_mean, abs_mean)
            if abs_mean > tolerance:
                incompatible_count += 1
        self.last_unreached_incompatible_component_count = incompatible_count
        self.last_unreached_component_rhs_mean_max_abs = max_abs_mean
        self.last_unreached_component_rhs_integral_max_abs = max_abs_integral
        return incompatible_count

    def _measure_pressure_outlet_nullspace_rhs_incompatibility(
        self,
        field: ti.template(),
        *,
        mean_abs_tolerance: float = 1.0e-12,
    ) -> int:
        if not bool(self._pressure_outlet_nullspace_graph_valid):
            raise self._pressure_operator_graph_missing_or_stale_error(
                context="pressure-outlet-nullspace-rhs-compatibility",
            )
        self.last_unreached_incompatible_component_count = 0
        self.last_unreached_component_rhs_mean_max_abs = 0.0
        self.last_unreached_component_rhs_integral_max_abs = 0.0
        removed_roots = frozenset()
        if self._pressure_outlet_nullspace_graph_context == "outlet-disconnected":
            operator_component_count = int(
                self._pressure_outlet_operator_component_count
            )
            if operator_component_count <= 0:
                return 0
            self._accumulate_pressure_outlet_operator_component_means_kernel(
                field,
                operator_component_count,
            )
            volumes = self.cg_unreached_component_volume.to_numpy()[
                :operator_component_count
            ]
            integrals = self.cg_unreached_component_sum.to_numpy()[
                :operator_component_count
            ]
            active_components = range(operator_component_count)
            parents = None
            anchored = None
        else:
            physical_count = int(
                self._pressure_outlet_nullspace_source_component_count
            )
            if physical_count <= 0:
                return 0
            self._accumulate_pressure_outlet_nullspace_component_means_kernel(
                field,
                physical_count,
            )
            parents = self.pressure_solve_component_parent.to_numpy()[:physical_count]
            anchored = self.pressure_solve_component_anchored.to_numpy()[
                :physical_count
            ]
            volumes = self.cg_unreached_component_volume.to_numpy()[:physical_count]
            integrals = self.cg_unreached_component_sum.to_numpy()[:physical_count]
            active_components = range(physical_count)
        incompatible_count = 0
        max_abs_mean = 0.0
        max_abs_integral = 0.0
        tolerance = max(0.0, float(mean_abs_tolerance))
        for component in active_components:
            if (
                parents is not None
                and anchored is not None
                and (
                    int(parents[component]) != component
                    or int(anchored[component]) != 0
                )
            ):
                continue
            if component in removed_roots:
                continue
            volume = float(volumes[component])
            integral = float(integrals[component])
            abs_integral = abs(integral)
            max_abs_integral = max(max_abs_integral, abs_integral)
            if volume <= 0.0:
                continue
            abs_mean = abs(integral / volume)
            max_abs_mean = max(max_abs_mean, abs_mean)
            if abs_mean > tolerance:
                incompatible_count += 1
        self.last_unreached_incompatible_component_count = incompatible_count
        self.last_unreached_component_rhs_mean_max_abs = max_abs_mean
        self.last_unreached_component_rhs_integral_max_abs = max_abs_integral
        return incompatible_count
