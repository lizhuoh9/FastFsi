"""Pressure projection: the finite-volume operator, the velocity correction, the outlet and the divergence (mixin)."""

from collections.abc import Mapping
import math

import taichi as ti

from fastfsi.fluid.common import PressureSolveConvergenceError, _diagnostic_int, _diagnostic_str
from fastfsi.fluid.device_scalars import read_scalar_fields
from fastfsi.fluid.interface_constants import PRESSURE_INTERFACE_COUPLING_SLOT_COUNT


class PressureProjection:
    """Mixin of IncompressibleFluidSolver; not used on its own.  Contract (solver attributes):
      reads:  _mg_shapes, cell_width_x_m, cell_width_y_m, cell_width_z_m, center_distance_x_m, center_distance_y_m,
    center_distance_z_m, cleanup_acceptance_l2_sq, cleanup_rejected, cleanup_required, cleanup_saved_outlet_flux,
    cleanup_saved_pressure, cleanup_saved_pressure_interface_matrix_rhs,
    cleanup_saved_pressure_interface_projection_divergence_s, cleanup_saved_velocity, cleanup_target_l2_sq,
    divergence, divergence_report_snapshot, external_velocity_boundary_z_face_active_component_mask,
    pressure_outlet_reachable, last_cg_bicgstab_iterations, last_cg_breakdown, last_cg_exact_relative_residual,
    last_cg_exact_residual_confirmation_count, last_cg_iterations, last_cg_multigrid_apply_count,
    last_cg_multigrid_coarse_smooth_iterations, last_cg_multigrid_post_smooth_iterations,
    last_cg_multigrid_pre_smooth_iterations, last_cg_multigrid_to_jacobi_fallback_count, last_cg_pcg_iterations,
    last_cg_preconditioner_effective, last_cg_preconditioner_requested, last_cg_recursive_relative_residual,
    last_cg_restart_count, last_pressure_component_labels_converged, last_pressure_reachability_converged,
    last_pressure_reachability_frontier_overflow, last_pressure_reachability_sweeps,
    last_pressure_unreached_cell_count, last_pressure_unreached_component_count,
    last_pressure_unreached_component_overflow, last_pressure_unreached_component_raw_count,
    last_pressure_unreached_raw_cell_count, nx, ny, nz, obstacle, pressure, pressure_accum,
    pressure_interface_coupling_active, pressure_interface_coupling_coefficient,
    pressure_interface_coupling_extra_coefficient, pressure_interface_coupling_extra_neighbor,
    pressure_interface_coupling_neighbor, pressure_interface_matrix_diagonal, pressure_interface_matrix_rhs,
    pressure_interface_preflight_incident_diagonal_integral, pressure_interface_preflight_incident_term_count,
    pressure_interface_preflight_invalid_counts, pressure_interface_projection_divergence_s,
    pressure_interface_row_capacity, pressure_interface_row_count, pressure_interface_row_diagonal_density,
    pressure_interface_row_neighbor, pressure_interface_row_owner, pressure_interface_row_support_cell,
    pressure_interface_row_support_coefficient, pressure_interface_row_support_count,
    pressure_interface_row_transmissibility, pressure_tmp, reduction_count, reduction_max, reduction_sum,
    report_pressure_interface_matrix_active_cells, report_pressure_interface_matrix_active_row_count,
    report_pressure_interface_matrix_diagonal_integral, report_pressure_interface_matrix_invalid_row_count,
    report_pressure_interface_matrix_max_abs_diagonal, report_pressure_interface_matrix_overflow_row_count,
    report_pressure_interface_matrix_rhs_integral, report_pressure_interface_matrix_row_count,
    report_pressure_interface_matrix_row_diagonal_abs_mismatch,
    report_pressure_interface_matrix_row_diagonal_integral,
    report_pressure_interface_matrix_row_diagonal_max_abs_density_mismatch,
    report_pressure_interface_matrix_row_max_transmissibility, report_zmin_pressure_outlet_flux_m3s,
    report_zmin_pressure_step_pre_velocity_outlet_flux_m3s,
    report_zmin_projection_post_boundary_velocity_outlet_flux_m3s,
    report_zmin_projection_post_pressure_velocity_outlet_flux_m3s,
    report_zmin_projection_pre_velocity_outlet_flux_m3s, velocity, velocity_dirichlet_boundary_active,
    velocity_dirichlet_boundary_active_component_mask, velocity_dirichlet_boundary_hard_fixed_component_mask,
    velocity_dirichlet_boundary_pressure_mobility, velocity_dirichlet_boundary_projection_weight,
    velocity_dirichlet_pressure_hard_fixed_component_mask, volume_source_s
      writes: last_cg_operator_apply_count"""

    @ti.kernel
    def _snapshot_projection_cleanup_state_kernel(self):
        self.cleanup_saved_outlet_flux[None] = ti.Vector([
            self.report_zmin_pressure_outlet_flux_m3s[None],
            self.report_zmin_pressure_step_pre_velocity_outlet_flux_m3s[None],
            self.report_zmin_projection_post_pressure_velocity_outlet_flux_m3s[None],
        ])
        for i, j, k in self.velocity:
            self.cleanup_saved_velocity[i, j, k] = self.velocity[i, j, k]
            self.cleanup_saved_pressure[i, j, k] = self.pressure[i, j, k]
            self.cleanup_saved_pressure_interface_matrix_rhs[i, j, k] = (
                self.pressure_interface_matrix_rhs[i, j, k]
            )
            self.cleanup_saved_pressure_interface_projection_divergence_s[i, j, k] = (
                self.pressure_interface_projection_divergence_s[i, j, k]
            )

    @ti.kernel
    def _restore_projection_cleanup_state_kernel(self):
        self.report_zmin_pressure_outlet_flux_m3s[None] = self.cleanup_saved_outlet_flux[None][0]
        self.report_zmin_pressure_step_pre_velocity_outlet_flux_m3s[None] = self.cleanup_saved_outlet_flux[None][1]
        self.report_zmin_projection_post_pressure_velocity_outlet_flux_m3s[None] = self.cleanup_saved_outlet_flux[None][2]
        for i, j, k in self.velocity:
            self.velocity[i, j, k] = self.cleanup_saved_velocity[i, j, k]
            self.pressure[i, j, k] = self.cleanup_saved_pressure[i, j, k]
            self.pressure_tmp[i, j, k] = self.cleanup_saved_pressure[i, j, k]
            self.pressure_accum[i, j, k] = self.cleanup_saved_pressure[i, j, k]
            self.pressure_interface_matrix_rhs[i, j, k] = (
                self.cleanup_saved_pressure_interface_matrix_rhs[i, j, k]
            )
            self.pressure_interface_projection_divergence_s[i, j, k] = (
                self.cleanup_saved_pressure_interface_projection_divergence_s[i, j, k]
            )

    @ti.kernel
    def _clear_pressure_interface_matrix_rhs_kernel(self):
        for i, j, k in self.pressure_interface_matrix_rhs:
            self.pressure_interface_matrix_rhs[i, j, k] = 0.0

    @ti.kernel
    def _clear_pressure_interface_projection_divergence_kernel(self):
        for i, j, k in self.pressure_interface_projection_divergence_s:
            self.pressure_interface_projection_divergence_s[i, j, k] = 0.0

    @ti.func
    def _cell_volume_m3(self, i, j, k):
        return self.cell_width_x_m[i] * self.cell_width_y_m[j] * self.cell_width_z_m[k]

    @ti.func
    def _cell_volume_f64_m3(self, i, j, k):
        """Authoritative f64 FV volume used by pressure operators and CG."""

        return (
            ti.cast(self.cell_width_x_m[i], ti.f64)
            * ti.cast(self.cell_width_y_m[j], ti.f64)
            * ti.cast(self.cell_width_z_m[k], ti.f64)
        )

    @ti.func
    def _z_face_area_f64_m2(self, i, j):
        """Authoritative promoted area for z-normal flux diagnostics."""

        return (
            ti.cast(self.cell_width_x_m[i], ti.f64)
            * ti.cast(self.cell_width_y_m[j], ti.f64)
        )

    @ti.func
    def _pressure_interface_row_support(self, row: ti.i32, slot: ti.i32):
        cell = self.pressure_interface_row_owner[row]
        coefficient = ti.cast(1.0, ti.f64)
        if self.pressure_interface_row_support_count[row] == 0:
            if slot == 1:
                cell = self.pressure_interface_row_neighbor[row]
                coefficient = ti.cast(-1.0, ti.f64)
        else:
            cell = self.pressure_interface_row_support_cell[row, slot]
            coefficient = self.pressure_interface_row_support_coefficient[row, slot]
        return cell, coefficient

    @ti.func
    def _pressure_interface_row_support_is_valid(self, row: ti.i32) -> ti.i32:
        """Validate compact support without dereferencing an invalid cell."""
        owner = self.pressure_interface_row_owner[row]
        neighbor = self.pressure_interface_row_neighbor[row]
        support_invalid = False
        support_count = self.pressure_interface_row_support_count[row]
        if support_count != 0:
            support_invalid = support_count < 2 or support_count > 9
            coefficient_sum = ti.cast(0.0, ti.f64)
            coefficient_l1 = ti.cast(0.0, ti.f64)
            owner_seen = 0
            neighbor_seen = 0
            for slot in range(ti.max(0, ti.min(support_count, 9))):
                cell = self.pressure_interface_row_support_cell[row, slot]
                coefficient = self.pressure_interface_row_support_coefficient[row, slot]
                cell_valid = (
                    0 <= cell.x < self.nx and 0 <= cell.y < self.ny
                    and 0 <= cell.z < self.nz
                )
                if cell_valid:
                    cell_valid = self.obstacle[cell] == 0
                if (
                    not cell_valid or ti.math.isnan(coefficient)
                    or ti.math.isinf(coefficient) or coefficient == 0.0
                ):
                    support_invalid = True
                coefficient_sum += coefficient
                coefficient_l1 += ti.abs(coefficient)
                if cell.x == owner.x and cell.y == owner.y and cell.z == owner.z:
                    owner_seen = 1
                if cell.x == neighbor.x and cell.y == neighbor.y and cell.z == neighbor.z:
                    neighbor_seen = 1
                for previous in range(slot):
                    other = self.pressure_interface_row_support_cell[row, previous]
                    if cell.x == other.x and cell.y == other.y and cell.z == other.z:
                        support_invalid = True
            if (
                owner_seen == 0 or neighbor_seen == 0
                or ti.math.isnan(coefficient_sum) or ti.math.isinf(coefficient_sum)
                or ti.math.isnan(coefficient_l1) or ti.math.isinf(coefficient_l1)
                or ti.abs(coefficient_sum) > (
                    32.0 * 1.1102230246251565e-16
                    * ti.max(coefficient_l1, 1.0e-300)
                )
            ):
                support_invalid = True
        return ti.cast(not support_invalid, ti.i32)

    @ti.func
    def _accumulate_pressure_interface_rank_one_offdiagonal(
        self, row: ti.i32, pressure: ti.template(), output: ti.template(),
        scale: ti.f64,
    ):
        count = self.pressure_interface_row_support_count[row]
        row_value = ti.cast(0.0, ti.f64)
        for slot in range(ti.min(count, 9)):
            cell, coefficient = self._pressure_interface_row_support(row, slot)
            row_value += coefficient * pressure[cell]
        transmissibility = self.pressure_interface_row_transmissibility[row]
        for slot in range(ti.min(count, 9)):
            cell, coefficient = self._pressure_interface_row_support(row, slot)
            ti.atomic_add(
                output[cell],
                scale * (transmissibility / self._cell_volume_f64_m3(
                    cell.x, cell.y, cell.z
                )) * coefficient * (row_value - coefficient * pressure[cell]),
            )

    @ti.kernel
    def _pressure_interface_matrix_terms_report_kernel(self):
        self.report_pressure_interface_matrix_diagonal_integral[None] = 0.0
        self.report_pressure_interface_matrix_rhs_integral[None] = 0.0
        self.report_pressure_interface_matrix_max_abs_diagonal[None] = 0.0
        self.report_pressure_interface_matrix_active_cells[None] = 0
        self.report_pressure_interface_matrix_row_count[None] = (
            self.pressure_interface_row_count[None]
        )
        self.report_pressure_interface_matrix_active_row_count[None] = 0
        self.report_pressure_interface_matrix_invalid_row_count[None] = 0
        self.report_pressure_interface_matrix_overflow_row_count[None] = ti.max(
            self.pressure_interface_row_count[None]
            - self.pressure_interface_row_capacity,
            0,
        )
        self.report_pressure_interface_matrix_row_diagonal_integral[None] = 0.0
        self.report_pressure_interface_matrix_row_diagonal_abs_mismatch[None] = 0.0
        self.report_pressure_interface_matrix_row_diagonal_max_abs_density_mismatch[
            None
        ] = 0.0
        self.report_pressure_interface_matrix_row_max_transmissibility[None] = 0.0
        for i, j, k in self.pressure_interface_row_diagonal_density:
            self.pressure_interface_row_diagonal_density[i, j, k] = 0.0
        row_limit = ti.min(
            self.pressure_interface_row_count[None],
            self.pressure_interface_row_capacity,
        )
        for row in range(ti.min(row_limit, self.pressure_interface_row_capacity)):
            if row < row_limit:
                owner = self.pressure_interface_row_owner[row]
                neighbor = self.pressure_interface_row_neighbor[row]
                i = owner.x
                j = owner.y
                k = owner.z
                ni = neighbor.x
                nj = neighbor.y
                nk = neighbor.z
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
                    and transmissibility > 0.0
                    and (i != ni or j != nj or k != nk)
                    and self._pressure_interface_row_support_is_valid(row) != 0
                ):
                    if self.pressure_interface_row_support_count[row] == 0:
                        active_volume_m3 = self._cell_volume_f64_m3(i, j, k)
                        neighbor_volume_m3 = self._cell_volume_f64_m3(ni, nj, nk)
                        ti.atomic_add(
                            self.pressure_interface_row_diagonal_density[i, j, k],
                            transmissibility / active_volume_m3,
                        )
                        ti.atomic_add(
                            self.pressure_interface_row_diagonal_density[ni, nj, nk],
                            transmissibility / neighbor_volume_m3,
                        )
                    else:
                        for slot in range(ti.min(
                            self.pressure_interface_row_support_count[row], 9
                        )):
                            cell, coefficient = self._pressure_interface_row_support(row, slot)
                            ti.atomic_add(
                                self.pressure_interface_row_diagonal_density[cell],
                                transmissibility * coefficient * coefficient
                                / self._cell_volume_f64_m3(cell.x, cell.y, cell.z),
                            )
                    row_diagonal_integral = 2.0 * transmissibility
                    if self.pressure_interface_row_support_count[row] > 0:
                        row_diagonal_integral = ti.cast(0.0, ti.f64)
                        for slot in range(ti.min(
                            self.pressure_interface_row_support_count[row], 9
                        )):
                            coefficient = self.pressure_interface_row_support_coefficient[row, slot]
                            row_diagonal_integral += transmissibility * coefficient * coefficient
                    ti.atomic_add(
                        self.report_pressure_interface_matrix_active_row_count[None],
                        1,
                    )
                    ti.atomic_add(
                        self.report_pressure_interface_matrix_row_diagonal_integral[
                            None
                        ],
                        ti.cast(row_diagonal_integral, ti.f64),
                    )
                    ti.atomic_max(
                        self.report_pressure_interface_matrix_row_max_transmissibility[
                            None
                        ],
                        transmissibility,
                    )
                else:
                    ti.atomic_add(
                        self.report_pressure_interface_matrix_invalid_row_count[None],
                        1,
                    )
        for i, j, k in self.pressure_interface_coupling_active:
            if (
                self.pressure_interface_row_count[None] == 0
                and self.pressure_interface_coupling_active[i, j, k] > 0
                and self.obstacle[i, j, k] == 0
            ):
                slot_count = ti.min(
                    self.pressure_interface_coupling_active[i, j, k],
                    PRESSURE_INTERFACE_COUPLING_SLOT_COUNT,
                )
                for slot in ti.static(range(PRESSURE_INTERFACE_COUPLING_SLOT_COUNT)):
                    if slot < slot_count:
                        neighbor = self.pressure_interface_coupling_neighbor[i, j, k]
                        transmissibility = (
                            self.pressure_interface_coupling_coefficient[i, j, k]
                        )
                        if ti.static(slot > 0):
                            extra_slot = slot - 1
                            neighbor = (
                                self.pressure_interface_coupling_extra_neighbor[
                                    i,
                                    j,
                                    k,
                                    extra_slot,
                                ]
                            )
                            transmissibility = (
                                self.pressure_interface_coupling_extra_coefficient[
                                    i,
                                    j,
                                    k,
                                    extra_slot,
                                ]
                            )
                        ni = neighbor.x
                        nj = neighbor.y
                        nk = neighbor.z
                        if (
                            0 <= ni < self.nx
                            and 0 <= nj < self.ny
                            and 0 <= nk < self.nz
                            and self.obstacle[ni, nj, nk] == 0
                            and transmissibility > 0.0
                            and (i != ni or j != nj or k != nk)
                        ):
                            active_volume_m3 = self._cell_volume_f64_m3(i, j, k)
                            neighbor_volume_m3 = self._cell_volume_f64_m3(
                                ni, nj, nk
                            )
                            ti.atomic_add(
                                self.pressure_interface_row_diagonal_density[
                                    i,
                                    j,
                                    k,
                                ],
                                transmissibility / active_volume_m3,
                            )
                            ti.atomic_add(
                                self.pressure_interface_row_diagonal_density[
                                    ni,
                                    nj,
                                    nk,
                                ],
                                transmissibility / neighbor_volume_m3,
                            )
                            ti.atomic_add(
                                self.report_pressure_interface_matrix_active_row_count[
                                    None
                                ],
                                1,
                            )
                            ti.atomic_add(
                                self.report_pressure_interface_matrix_row_diagonal_integral[
                                    None
                                ],
                                ti.cast(2.0 * transmissibility, ti.f64),
                            )
                            ti.atomic_max(
                                self.report_pressure_interface_matrix_row_max_transmissibility[
                                    None
                                ],
                                transmissibility,
                            )
                        else:
                            ti.atomic_add(
                                self.report_pressure_interface_matrix_invalid_row_count[
                                    None
                                ],
                                1,
                            )
        for i, j, k in self.pressure_interface_matrix_diagonal:
            diagonal = self.pressure_interface_matrix_diagonal[i, j, k]
            rhs = self.pressure_interface_matrix_rhs[i, j, k]
            if ti.abs(diagonal) > 0.0 or ti.abs(rhs) > 0.0:
                ti.atomic_add(self.report_pressure_interface_matrix_active_cells[None], 1)
            cell_volume_m3 = self._cell_volume_f64_m3(i, j, k)
            diagonal_density_mismatch = (
                diagonal - self.pressure_interface_row_diagonal_density[i, j, k]
            )
            ti.atomic_add(
                self.report_pressure_interface_matrix_diagonal_integral[None],
                ti.cast(diagonal * cell_volume_m3, ti.f64),
            )
            ti.atomic_add(
                self.report_pressure_interface_matrix_rhs_integral[None],
                ti.cast(rhs * cell_volume_m3, ti.f64),
            )
            ti.atomic_max(
                self.report_pressure_interface_matrix_max_abs_diagonal[None],
                ti.abs(diagonal),
            )
            ti.atomic_add(
                self.report_pressure_interface_matrix_row_diagonal_abs_mismatch[
                    None
                ],
                ti.cast(ti.abs(diagonal_density_mismatch) * cell_volume_m3, ti.f64),
            )
            ti.atomic_max(
                self.report_pressure_interface_matrix_row_diagonal_max_abs_density_mismatch[
                    None
                ],
                ti.abs(diagonal_density_mismatch),
            )

    def pressure_interface_matrix_terms_report(self) -> dict[str, float | int]:
        self._pressure_interface_matrix_terms_report_kernel()
        counters = read_scalar_fields(
            self,
            (
                self.report_pressure_interface_matrix_diagonal_integral,
                self.report_pressure_interface_matrix_rhs_integral,
                self.report_pressure_interface_matrix_max_abs_diagonal,
                self.report_pressure_interface_matrix_active_cells,
                self.report_pressure_interface_matrix_row_count,
                self.report_pressure_interface_matrix_active_row_count,
                self.report_pressure_interface_matrix_invalid_row_count,
                self.report_pressure_interface_matrix_overflow_row_count,
                self.report_pressure_interface_matrix_row_diagonal_integral,
                self.report_pressure_interface_matrix_row_diagonal_abs_mismatch,
                self.report_pressure_interface_matrix_row_diagonal_max_abs_density_mismatch,
                self.report_pressure_interface_matrix_row_max_transmissibility,
            ),
        )
        return {
            "diagonal_integral": float(
                counters[self.report_pressure_interface_matrix_diagonal_integral]
            ),
            "rhs_integral": float(counters[self.report_pressure_interface_matrix_rhs_integral]),
            "max_abs_diagonal": float(
                counters[self.report_pressure_interface_matrix_max_abs_diagonal]
            ),
            "active_cells": int(counters[self.report_pressure_interface_matrix_active_cells]),
            "row_count": int(counters[self.report_pressure_interface_matrix_row_count]),
            "row_active_count": int(
                counters[self.report_pressure_interface_matrix_active_row_count]
            ),
            "row_invalid_count": int(
                counters[self.report_pressure_interface_matrix_invalid_row_count]
            ),
            "row_overflow_count": int(
                counters[self.report_pressure_interface_matrix_overflow_row_count]
            ),
            "row_diagonal_integral": float(
                counters[self.report_pressure_interface_matrix_row_diagonal_integral]
            ),
            "row_diagonal_integral_abs_mismatch": float(
                counters[self.report_pressure_interface_matrix_row_diagonal_abs_mismatch]
            ),
            "row_diagonal_max_abs_density_mismatch": float(
                counters[self.report_pressure_interface_matrix_row_diagonal_max_abs_density_mismatch]
            ),
            "row_max_transmissibility": float(
                counters[self.report_pressure_interface_matrix_row_max_transmissibility]
            ),
        }

    @ti.kernel
    def _pressure_interface_matrix_policy_probe_kernel(self):
        self.report_pressure_interface_matrix_max_abs_diagonal[None] = 0.0
        self.report_pressure_interface_matrix_active_cells[None] = 0
        self.report_pressure_interface_matrix_row_count[None] = (
            self.pressure_interface_row_count[None]
        )
        self.report_pressure_interface_matrix_active_row_count[None] = ti.min(
            self.pressure_interface_row_count[None],
            self.pressure_interface_row_capacity,
        )
        for i, j, k in self.pressure_interface_matrix_diagonal:
            diagonal = self.pressure_interface_matrix_diagonal[i, j, k]
            rhs = self.pressure_interface_matrix_rhs[i, j, k]
            if ti.abs(diagonal) > 0.0 or ti.abs(rhs) > 0.0:
                ti.atomic_add(self.report_pressure_interface_matrix_active_cells[None], 1)
            ti.atomic_max(
                self.report_pressure_interface_matrix_max_abs_diagonal[None],
                ti.abs(diagonal),
            )

    @ti.kernel
    def _validate_pressure_interface_operator_storage_kernel(
        self,
        row_count: ti.i32,
    ):
        self.pressure_interface_preflight_invalid_counts[None] = ti.Vector(
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        for i, j, k in self.obstacle:
            diagonal = self.pressure_interface_matrix_diagonal[i, j, k]
            rhs = self.pressure_interface_matrix_rhs[i, j, k]
            if (
                ti.math.isnan(diagonal)
                or ti.math.isinf(diagonal)
                or diagonal < 0.0
            ):
                ti.atomic_add(
                    self.pressure_interface_preflight_invalid_counts[None][0],
                    1,
                )
            if ti.math.isnan(rhs) or ti.math.isinf(rhs):
                ti.atomic_add(
                    self.pressure_interface_preflight_invalid_counts[None][1],
                    1,
                )

            if row_count == 0:
                active_count = self.pressure_interface_coupling_active[i, j, k]
                if (
                    active_count < 0
                    or active_count > PRESSURE_INTERFACE_COUPLING_SLOT_COUNT
                ):
                    ti.atomic_add(
                        self.pressure_interface_preflight_invalid_counts[None][4],
                        1,
                    )
                elif active_count > 0:
                    owner_valid = self.obstacle[i, j, k] == 0
                    if not owner_valid:
                        ti.atomic_add(
                            self.pressure_interface_preflight_invalid_counts[None][5],
                            1,
                        )
                    else:
                        for slot in ti.static(
                            range(PRESSURE_INTERFACE_COUPLING_SLOT_COUNT)
                        ):
                            if slot < active_count:
                                neighbor = (
                                    self.pressure_interface_coupling_neighbor[i, j, k]
                                )
                                transmissibility = (
                                    self.pressure_interface_coupling_coefficient[i, j, k]
                                )
                                if ti.static(slot > 0):
                                    extra_slot = slot - 1
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
                                endpoint_valid = (
                                    0 <= ni < self.nx
                                    and 0 <= nj < self.ny
                                    and 0 <= nk < self.nz
                                    and (i != ni or j != nj or k != nk)
                                )
                                if endpoint_valid:
                                    endpoint_valid = self.obstacle[ni, nj, nk] == 0
                                if not endpoint_valid:
                                    ti.atomic_add(
                                        self.pressure_interface_preflight_invalid_counts[
                                            None
                                        ][5],
                                        1,
                                    )
                                if (
                                    ti.math.isnan(transmissibility)
                                    or ti.math.isinf(transmissibility)
                                    or transmissibility <= 0.0
                                ):
                                    ti.atomic_add(
                                        self.pressure_interface_preflight_invalid_counts[
                                            None
                                        ][6],
                                        1,
                                    )

        for row in range(ti.min(row_count, self.pressure_interface_row_capacity)):
            if row < row_count:
                owner = self.pressure_interface_row_owner[row]
                neighbor = self.pressure_interface_row_neighbor[row]
                i, j, k = owner.x, owner.y, owner.z
                ni, nj, nk = neighbor.x, neighbor.y, neighbor.z
                endpoint_valid = (
                    0 <= i < self.nx
                    and 0 <= j < self.ny
                    and 0 <= k < self.nz
                    and 0 <= ni < self.nx
                    and 0 <= nj < self.ny
                    and 0 <= nk < self.nz
                    and (i != ni or j != nj or k != nk)
                )
                if endpoint_valid:
                    endpoint_valid = (
                        self.obstacle[i, j, k] == 0
                        and self.obstacle[ni, nj, nk] == 0
                    )
                if not endpoint_valid:
                    ti.atomic_add(
                        self.pressure_interface_preflight_invalid_counts[None][2],
                        1,
                    )
                transmissibility = self.pressure_interface_row_transmissibility[row]
                if (
                    ti.math.isnan(transmissibility)
                    or ti.math.isinf(transmissibility)
                    or transmissibility <= 0.0
                ):
                    ti.atomic_add(
                        self.pressure_interface_preflight_invalid_counts[None][3],
                        1,
                    )

                if self._pressure_interface_row_support_is_valid(row) == 0:
                    ti.atomic_add(
                        self.pressure_interface_preflight_invalid_counts[None][8], 1
                    )

    @ti.kernel
    def _clear_pressure_interface_preflight_incident_diagonal_kernel(self):
        for i, j, k in self.pressure_interface_preflight_incident_diagonal_integral:
            self.pressure_interface_preflight_incident_diagonal_integral[i, j, k] = 0.0
            self.pressure_interface_preflight_incident_term_count[i, j, k] = 0

    @ti.kernel
    def _accumulate_pressure_interface_preflight_incident_diagonal_kernel(
        self,
        row_count: ti.i32,
    ):
        for row in range(ti.min(row_count, self.pressure_interface_row_capacity)):
            if row < row_count:
                owner = self.pressure_interface_row_owner[row]
                neighbor = self.pressure_interface_row_neighbor[row]
                i, j, k = owner.x, owner.y, owner.z
                ni, nj, nk = neighbor.x, neighbor.y, neighbor.z
                transmissibility = ti.cast(
                    self.pressure_interface_row_transmissibility[row],
                    ti.f64,
                )
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
                        ti.atomic_add(
                            self.pressure_interface_preflight_incident_diagonal_integral[
                                i, j, k
                            ],
                            transmissibility,
                        )
                        ti.atomic_add(
                            self.pressure_interface_preflight_incident_diagonal_integral[
                                ni, nj, nk
                            ],
                            transmissibility,
                        )
                        ti.atomic_add(
                            self.pressure_interface_preflight_incident_term_count[i, j, k],
                            1,
                        )
                        ti.atomic_add(
                            self.pressure_interface_preflight_incident_term_count[
                                ni, nj, nk
                            ],
                            1,
                        )
                    else:
                        for slot in range(ti.min(
                            self.pressure_interface_row_support_count[row], 9
                        )):
                            cell, coefficient = self._pressure_interface_row_support(row, slot)
                            ti.atomic_add(
                                self.pressure_interface_preflight_incident_diagonal_integral[cell],
                                transmissibility * coefficient * coefficient,
                            )
                            ti.atomic_add(
                                self.pressure_interface_preflight_incident_term_count[cell], 1
                            )

        for i, j, k in self.pressure_interface_coupling_active:
            if row_count == 0:
                active_count = self.pressure_interface_coupling_active[i, j, k]
                if (
                    0 < active_count <= PRESSURE_INTERFACE_COUPLING_SLOT_COUNT
                    and self.obstacle[i, j, k] == 0
                ):
                    for slot in ti.static(
                        range(PRESSURE_INTERFACE_COUPLING_SLOT_COUNT)
                    ):
                        if slot < active_count:
                            neighbor = self.pressure_interface_coupling_neighbor[
                                i, j, k
                            ]
                            transmissibility = ti.cast(
                                self.pressure_interface_coupling_coefficient[i, j, k],
                                ti.f64,
                            )
                            if ti.static(slot > 0):
                                extra_slot = slot - 1
                                neighbor = (
                                    self.pressure_interface_coupling_extra_neighbor[
                                        i, j, k, extra_slot
                                    ]
                                )
                                transmissibility = ti.cast(
                                    self.pressure_interface_coupling_extra_coefficient[
                                        i, j, k, extra_slot
                                    ],
                                    ti.f64,
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
                                    self.pressure_interface_preflight_incident_diagonal_integral[
                                        i, j, k
                                    ],
                                    transmissibility,
                                )
                                ti.atomic_add(
                                    self.pressure_interface_preflight_incident_diagonal_integral[
                                        ni, nj, nk
                                    ],
                                    transmissibility,
                                )
                                ti.atomic_add(
                                    self.pressure_interface_preflight_incident_term_count[
                                        i, j, k
                                    ],
                                    1,
                                )
                                ti.atomic_add(
                                    self.pressure_interface_preflight_incident_term_count[
                                        ni, nj, nk
                                    ],
                                    1,
                                )

    @ti.kernel
    def _validate_pressure_interface_local_diagonal_provenance_kernel(self):
        for i, j, k in self.obstacle:
            if self.obstacle[i, j, k] == 0:
                diagonal_integral = (
                    ti.cast(self.pressure_interface_matrix_diagonal[i, j, k], ti.f64)
                    * self._cell_volume_f64_m3(i, j, k)
                )
                incident_integral = (
                    self.pressure_interface_preflight_incident_diagonal_integral[
                        i, j, k
                    ]
                )
                local_excess = diagonal_integral - incident_integral
                scale = ti.abs(diagonal_integral) + ti.abs(incident_integral)
                term_count = ti.cast(
                    ti.max(
                        self.pressure_interface_preflight_incident_term_count[
                            i, j, k
                        ]
                        + ti.cast(diagonal_integral != 0.0, ti.i32),
                        1,
                    ),
                    ti.f64,
                )
                unit_roundoff = ti.cast(1.1102230246251565e-16, ti.f64)
                operation_count = 4.0 * term_count + 4.0
                rounding_product = operation_count * unit_roundoff
                gamma_n = rounding_product / ti.max(
                    1.0 - rounding_product,
                    0.5,
                )
                tolerance = gamma_n * ti.max(scale, 1.0e-300)
                if (
                    ti.math.isnan(local_excess)
                    or ti.math.isinf(local_excess)
                    or local_excess < -tolerance
                ):
                    ti.atomic_add(
                        self.pressure_interface_preflight_invalid_counts[None][7],
                        1,
                    )

    def _preflight_pressure_interface_operator_storage(
        self,
        *,
        context: str,
    ) -> None:
        """Fail before any pressure matvec sees malformed interface storage."""

        row_count = int(self.pressure_interface_row_count[None])
        capacity = int(self.pressure_interface_row_capacity)
        if not 0 <= row_count <= capacity:
            raise PressureSolveConvergenceError(
                "pressure-interface row count is outside device capacity",
                diagnostics={
                    "stage": "pressure_interface_operator_preflight",
                    "context": str(context),
                    "row_count": row_count,
                    "row_capacity": capacity,
                    "invalid_counts": {"row_count": 1},
                },
            )
        self._validate_pressure_interface_operator_storage_kernel(row_count)
        # One transfer: indexing the 9-vector field syncs once per component.
        counts_device = self.pressure_interface_preflight_invalid_counts.to_numpy()
        names = (
            "diagonal",
            "rhs",
            "row_endpoint",
            "row_transmissibility",
            "legacy_active_count",
            "legacy_endpoint",
            "legacy_transmissibility",
            "local_diagonal_provenance",
            "row_support",
        )
        counts = {
            name: int(counts_device[index])
            for index, name in enumerate(names)
        }
        invalid_total = sum(counts.values())
        if invalid_total != 0:
            raise PressureSolveConvergenceError(
                "pressure-interface operator storage is invalid",
                diagnostics={
                    "stage": "pressure_interface_operator_preflight",
                    "context": str(context),
                    "row_count": row_count,
                    "row_capacity": capacity,
                    "invalid_total": invalid_total,
                    "invalid_counts": counts,
                },
            )
        self._clear_pressure_interface_preflight_incident_diagonal_kernel()
        self._accumulate_pressure_interface_preflight_incident_diagonal_kernel(
            row_count
        )
        self._validate_pressure_interface_local_diagonal_provenance_kernel()
        local_invalid_count = int(
            self.pressure_interface_preflight_invalid_counts.to_numpy()[7]
        )
        if local_invalid_count != 0:
            counts["local_diagonal_provenance"] = local_invalid_count
            raise PressureSolveConvergenceError(
                "pressure-interface local diagonal provenance is invalid",
                diagnostics={
                    "stage": "pressure_interface_operator_preflight",
                    "context": str(context),
                    "row_count": row_count,
                    "row_capacity": capacity,
                    "invalid_total": local_invalid_count,
                    "invalid_counts": counts,
                },
            )

    def _pressure_interface_matrix_policy_report(self) -> dict[str, float | int]:
        row_count = int(self.pressure_interface_row_count[None])
        if row_count > 0:
            active_rows = min(row_count, self.pressure_interface_row_capacity)
            # The row list is the canonical production representation. Its
            # non-empty scalar count proves matrix activity without an O(N)
            # diagonal/RHS scan or four diagnostic scalar readbacks. Exact
            # magnitudes are still computed by the final report when requested.
            return {
                "max_abs_diagonal": 1.0,
                "active_cells": active_rows,
                "row_count": row_count,
                "row_active_count": active_rows,
            }
        self._pressure_interface_matrix_policy_probe_kernel()
        return {
            "max_abs_diagonal": float(
                self.report_pressure_interface_matrix_max_abs_diagonal[None]
            ),
            "active_cells": int(self.report_pressure_interface_matrix_active_cells[None]),
            "row_count": int(self.report_pressure_interface_matrix_row_count[None]),
            "row_active_count": int(
                self.report_pressure_interface_matrix_active_row_count[None]
            ),
        }

    @ti.func
    def _atomic_add_report_vector(self, field, value):
        ti.atomic_add(field[None].x, value.x)
        ti.atomic_add(field[None].y, value.y)
        ti.atomic_add(field[None].z, value.z)

    @ti.kernel
    def _divergence_residual_stats_kernel(
        self,
        interior_only: ti.i32,
        exclude_unreached: ti.i32,
    ):
        self.reduction_sum[None] = 0.0
        self.reduction_max[None] = 0.0
        self.reduction_count[None] = 0
        for i, j, k in self.divergence:
            is_interior = i > 0 and i < self.nx - 1
            is_interior = (
                is_interior
                and j > 0
                and j < self.ny - 1
                and k > 0
                and k < self.nz - 1
            )
            is_unreached = (
                exclude_unreached == 1
                and self.pressure_outlet_reachable[i, j, k] == 0
            )
            if (
                self.obstacle[i, j, k] == 0
                and not is_unreached
                and (interior_only == 0 or is_interior)
            ):
                total_divergence = (
                    self.divergence[i, j, k]
                    + self.pressure_interface_projection_divergence_s[i, j, k]
                )
                value = ti.abs(total_divergence - self.volume_source_s[i, j, k])
                ti.atomic_max(self.reduction_max[None], value)
                ti.atomic_add(self.reduction_sum[None], value * value)
                ti.atomic_add(self.reduction_count[None], 1)
        self.divergence_report_snapshot[None] = ti.Vector(
            [
                ti.cast(self.reduction_max[None], ti.f64),
                ti.cast(self.reduction_sum[None], ti.f64),
                ti.cast(self.reduction_count[None], ti.f64),
            ]
        )

    @ti.func
    def _cleanup_metric_is_finite(self, value):
        # Check the stored binary64 exponent. Floating comparisons alone can
        # accept NaN (or be simplified under the qualified fast-math runtime).
        exponent = ti.bit_cast(value, ti.u64) & ti.u64(0x7ff0000000000000)
        return exponent != ti.u64(0x7ff0000000000000)

    @ti.kernel
    def _store_cleanup_target_l2_sq_from_reduction_kernel(
        self,
        tolerance_scale_sq: ti.f64,
        min_l2_sq: ti.f64,
    ):
        count = ti.max(self.reduction_count[None], 1)
        l2_sq = ti.cast(self.reduction_sum[None], ti.f64) / ti.cast(count, ti.f64)
        scaled = l2_sq * tolerance_scale_sq
        self.cleanup_target_l2_sq[None] = ti.max(scaled, min_l2_sq)
        # min/max may hide NaN. Retain the invalid metric for the host gate,
        # without changing the expression or ordering for finite inputs.
        if not self._cleanup_metric_is_finite(l2_sq):
            self.cleanup_target_l2_sq[None] = l2_sq
        elif not self._cleanup_metric_is_finite(scaled):
            self.cleanup_target_l2_sq[None] = scaled
        elif not self._cleanup_metric_is_finite(min_l2_sq):
            self.cleanup_target_l2_sq[None] = min_l2_sq

    @ti.kernel
    def _tighten_cleanup_target_l2_sq_from_reduction_kernel(
        self,
        tolerance_scale_sq: ti.f64,
        min_l2_sq: ti.f64,
    ):
        count = ti.max(self.reduction_count[None], 1)
        l2_sq = ti.cast(self.reduction_sum[None], ti.f64) / ti.cast(count, ti.f64)
        scaled = l2_sq * tolerance_scale_sq
        candidate = ti.max(scaled, min_l2_sq)
        previous = self.cleanup_target_l2_sq[None]
        if not self._cleanup_metric_is_finite(previous):
            self.cleanup_target_l2_sq[None] = previous
        elif not self._cleanup_metric_is_finite(l2_sq):
            self.cleanup_target_l2_sq[None] = l2_sq
        elif not self._cleanup_metric_is_finite(scaled):
            self.cleanup_target_l2_sq[None] = scaled
        elif not self._cleanup_metric_is_finite(min_l2_sq):
            self.cleanup_target_l2_sq[None] = min_l2_sq
        else:
            self.cleanup_target_l2_sq[None] = ti.min(previous, candidate)

    @ti.kernel
    def _store_cleanup_acceptance_l2_sq_from_reduction_kernel(
        self,
        acceptance_scale_sq: ti.f64,
    ):
        count = ti.max(self.reduction_count[None], 1)
        l2_sq = ti.cast(self.reduction_sum[None], ti.f64) / ti.cast(count, ti.f64)
        self.cleanup_acceptance_l2_sq[None] = l2_sq * acceptance_scale_sq
        if not self._cleanup_metric_is_finite(l2_sq):
            self.cleanup_acceptance_l2_sq[None] = l2_sq

    @ti.kernel
    def _update_cleanup_rejected_from_reduction_kernel(self):
        count = ti.max(self.reduction_count[None], 1)
        l2_sq = ti.cast(self.reduction_sum[None], ti.f64) / ti.cast(count, ti.f64)
        self.cleanup_rejected[None] = 0
        if (not self._cleanup_metric_is_finite(l2_sq)
                or not self._cleanup_metric_is_finite(self.cleanup_acceptance_l2_sq[None])):
            self.cleanup_rejected[None] = 2
        elif l2_sq > self.cleanup_acceptance_l2_sq[None]:
            self.cleanup_rejected[None] = 1

    @ti.kernel
    def _update_cleanup_required_from_reduction_kernel(self):
        count = ti.max(self.reduction_count[None], 1)
        l2_sq = ti.cast(self.reduction_sum[None], ti.f64) / ti.cast(count, ti.f64)
        self.cleanup_required[None] = 0
        if (not self._cleanup_metric_is_finite(l2_sq)
                or not self._cleanup_metric_is_finite(self.cleanup_target_l2_sq[None])):
            self.cleanup_required[None] = 2
        elif l2_sq > self.cleanup_target_l2_sq[None]:
            self.cleanup_required[None] = 1

    @ti.kernel
    def _clear_pressure_kernel(self):
        for i, j, k in self.pressure:
            self.pressure[i, j, k] = 0.0
            self.pressure_tmp[i, j, k] = 0.0
            self.pressure_accum[i, j, k] = 0.0

    @ti.kernel
    def _clear_pressure_correction_kernel(self):
        for i, j, k in self.pressure:
            self.pressure[i, j, k] = 0.0
            self.pressure_tmp[i, j, k] = 0.0

    @ti.func
    def _fv_pressure_velocity_face_weight(
        self,
        velocity_dirichlet_boundary_active: ti.template(),
        velocity_dirichlet_boundary_projection_weight: ti.template(),
        velocity_dirichlet_pressure_hard_fixed_component_mask: ti.template(),
        canonical_active_component_mask: ti.template(),
        canonical_pressure_mobility: ti.template(),
        canonical_hard_fixed_component_mask: ti.template(),
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        component_bit: ti.i32,
        canonical_authority: ti.template(),
    ):
        component_axis = 0
        if component_bit == 2:
            component_axis = 1
        elif component_bit == 4:
            component_axis = 2
        return self._velocity_dirichlet_pressure_face_weight(
            velocity_dirichlet_boundary_active,
            velocity_dirichlet_boundary_projection_weight,
            velocity_dirichlet_pressure_hard_fixed_component_mask,
            canonical_active_component_mask,
            canonical_pressure_mobility,
            canonical_hard_fixed_component_mask,
            i,
            j,
            k,
            component_axis,
            canonical_authority,
        )

    @ti.func
    def _fv_pressure_neighbor_sum_denominator(
        self,
        pressure: ti.template(),
        obstacle: ti.template(),
        velocity_dirichlet_boundary_active: ti.template(),
        velocity_dirichlet_boundary_projection_weight: ti.template(),
        velocity_dirichlet_pressure_hard_fixed_component_mask: ti.template(),
        canonical_active_component_mask: ti.template(),
        canonical_pressure_mobility: ti.template(),
        canonical_hard_fixed_component_mask: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        center_distance_x_m: ti.template(),
        center_distance_y_m: ti.template(),
        center_distance_z_m: ti.template(),
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        pressure_outlet_zmin: ti.i32,
        canonical_authority: ti.template(),
    ):
        neighbor_sum = ti.cast(0.0, ti.f64)
        denominator = ti.cast(0.0, ti.f64)
        if i > 0 and obstacle[i - 1, j, k] == 0:
            face_weight = self._fv_pressure_velocity_face_weight(
                velocity_dirichlet_boundary_active,
                velocity_dirichlet_boundary_projection_weight,
                velocity_dirichlet_pressure_hard_fixed_component_mask,
                canonical_active_component_mask,
                canonical_pressure_mobility,
                canonical_hard_fixed_component_mask,
                i,
                j,
                k,
                1,
                canonical_authority,
            )
            coeff = ti.cast(1.0, ti.f64) / (
                ti.cast(cell_width_x_m[i], ti.f64)
                * ti.cast(center_distance_x_m[i], ti.f64)
            )
            neighbor_sum += coeff * face_weight * pressure[i - 1, j, k]
            denominator += coeff * face_weight
        if i < nx - 1 and obstacle[i + 1, j, k] == 0:
            face_weight = self._fv_pressure_velocity_face_weight(
                velocity_dirichlet_boundary_active,
                velocity_dirichlet_boundary_projection_weight,
                velocity_dirichlet_pressure_hard_fixed_component_mask,
                canonical_active_component_mask,
                canonical_pressure_mobility,
                canonical_hard_fixed_component_mask,
                i + 1,
                j,
                k,
                1,
                canonical_authority,
            )
            coeff = ti.cast(1.0, ti.f64) / (
                ti.cast(cell_width_x_m[i], ti.f64)
                * ti.cast(center_distance_x_m[i + 1], ti.f64)
            )
            neighbor_sum += coeff * face_weight * pressure[i + 1, j, k]
            denominator += coeff * face_weight
        if j > 0 and obstacle[i, j - 1, k] == 0:
            face_weight = self._fv_pressure_velocity_face_weight(
                velocity_dirichlet_boundary_active,
                velocity_dirichlet_boundary_projection_weight,
                velocity_dirichlet_pressure_hard_fixed_component_mask,
                canonical_active_component_mask,
                canonical_pressure_mobility,
                canonical_hard_fixed_component_mask,
                i,
                j,
                k,
                2,
                canonical_authority,
            )
            coeff = ti.cast(1.0, ti.f64) / (
                ti.cast(cell_width_y_m[j], ti.f64)
                * ti.cast(center_distance_y_m[j], ti.f64)
            )
            neighbor_sum += coeff * face_weight * pressure[i, j - 1, k]
            denominator += coeff * face_weight
        if j < ny - 1 and obstacle[i, j + 1, k] == 0:
            face_weight = self._fv_pressure_velocity_face_weight(
                velocity_dirichlet_boundary_active,
                velocity_dirichlet_boundary_projection_weight,
                velocity_dirichlet_pressure_hard_fixed_component_mask,
                canonical_active_component_mask,
                canonical_pressure_mobility,
                canonical_hard_fixed_component_mask,
                i,
                j + 1,
                k,
                2,
                canonical_authority,
            )
            coeff = ti.cast(1.0, ti.f64) / (
                ti.cast(cell_width_y_m[j], ti.f64)
                * ti.cast(center_distance_y_m[j + 1], ti.f64)
            )
            neighbor_sum += coeff * face_weight * pressure[i, j + 1, k]
            denominator += coeff * face_weight
        if k > 0 and obstacle[i, j, k - 1] == 0:
            face_weight = self._fv_pressure_velocity_face_weight(
                velocity_dirichlet_boundary_active,
                velocity_dirichlet_boundary_projection_weight,
                velocity_dirichlet_pressure_hard_fixed_component_mask,
                canonical_active_component_mask,
                canonical_pressure_mobility,
                canonical_hard_fixed_component_mask,
                i,
                j,
                k,
                4,
                canonical_authority,
            )
            coeff = ti.cast(1.0, ti.f64) / (
                ti.cast(cell_width_z_m[k], ti.f64)
                * ti.cast(center_distance_z_m[k], ti.f64)
            )
            neighbor_sum += coeff * face_weight * pressure[i, j, k - 1]
            denominator += coeff * face_weight
        if k < nz - 1 and obstacle[i, j, k + 1] == 0:
            face_weight = self._fv_pressure_velocity_face_weight(
                velocity_dirichlet_boundary_active,
                velocity_dirichlet_boundary_projection_weight,
                velocity_dirichlet_pressure_hard_fixed_component_mask,
                canonical_active_component_mask,
                canonical_pressure_mobility,
                canonical_hard_fixed_component_mask,
                i,
                j,
                k + 1,
                4,
                canonical_authority,
            )
            coeff = ti.cast(1.0, ti.f64) / (
                ti.cast(cell_width_z_m[k], ti.f64)
                * ti.cast(center_distance_z_m[k + 1], ti.f64)
            )
            neighbor_sum += coeff * face_weight * pressure[i, j, k + 1]
            denominator += coeff * face_weight
        if pressure_outlet_zmin == 1 and k == 0:
            outlet_face_weight = self._fv_pressure_velocity_face_weight(
                velocity_dirichlet_boundary_active,
                velocity_dirichlet_boundary_projection_weight,
                velocity_dirichlet_pressure_hard_fixed_component_mask,
                canonical_active_component_mask,
                canonical_pressure_mobility,
                canonical_hard_fixed_component_mask,
                i,
                j,
                k,
                4,
                canonical_authority,
            )
            denominator += outlet_face_weight * ti.cast(2.0, ti.f64) / (
                ti.cast(cell_width_z_m[k], ti.f64)
                * ti.cast(cell_width_z_m[k], ti.f64)
            )
        return neighbor_sum, denominator

    @ti.func
    def _fv_pressure_face_products_and_denominator(
        self,
        obstacle: ti.template(),
        velocity_dirichlet_boundary_active: ti.template(),
        velocity_dirichlet_boundary_projection_weight: ti.template(),
        velocity_dirichlet_pressure_hard_fixed_component_mask: ti.template(),
        canonical_active_component_mask: ti.template(),
        canonical_pressure_mobility: ti.template(),
        canonical_hard_fixed_component_mask: ti.template(),
        cell_width_x_m: ti.template(),
        cell_width_y_m: ti.template(),
        cell_width_z_m: ti.template(),
        center_distance_x_m: ti.template(),
        center_distance_y_m: ti.template(),
        center_distance_z_m: ti.template(),
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        pressure_outlet_zmin: ti.i32,
        canonical_authority: ti.template(),
    ):
        """Pressure-independent part of ``_fv_pressure_neighbor_sum_denominator``.

        Returns the six face products ``coeff * face_weight`` (x-, x+, y-, y+,
        z-, z+; zero where that function skips the face) and the denominator
        accumulated in the same order, so a sweep using them reproduces its
        neighbor sum term by term.
        """

        products = ti.Vector.zero(ti.f64, 6)
        denominator = ti.cast(0.0, ti.f64)
        if i > 0 and obstacle[i - 1, j, k] == 0:
            face_weight = self._fv_pressure_velocity_face_weight(
                velocity_dirichlet_boundary_active,
                velocity_dirichlet_boundary_projection_weight,
                velocity_dirichlet_pressure_hard_fixed_component_mask,
                canonical_active_component_mask,
                canonical_pressure_mobility,
                canonical_hard_fixed_component_mask,
                i,
                j,
                k,
                1,
                canonical_authority,
            )
            coeff = ti.cast(1.0, ti.f64) / (
                ti.cast(cell_width_x_m[i], ti.f64)
                * ti.cast(center_distance_x_m[i], ti.f64)
            )
            products[0] = coeff * face_weight
            denominator += coeff * face_weight
        if i < nx - 1 and obstacle[i + 1, j, k] == 0:
            face_weight = self._fv_pressure_velocity_face_weight(
                velocity_dirichlet_boundary_active,
                velocity_dirichlet_boundary_projection_weight,
                velocity_dirichlet_pressure_hard_fixed_component_mask,
                canonical_active_component_mask,
                canonical_pressure_mobility,
                canonical_hard_fixed_component_mask,
                i + 1,
                j,
                k,
                1,
                canonical_authority,
            )
            coeff = ti.cast(1.0, ti.f64) / (
                ti.cast(cell_width_x_m[i], ti.f64)
                * ti.cast(center_distance_x_m[i + 1], ti.f64)
            )
            products[1] = coeff * face_weight
            denominator += coeff * face_weight
        if j > 0 and obstacle[i, j - 1, k] == 0:
            face_weight = self._fv_pressure_velocity_face_weight(
                velocity_dirichlet_boundary_active,
                velocity_dirichlet_boundary_projection_weight,
                velocity_dirichlet_pressure_hard_fixed_component_mask,
                canonical_active_component_mask,
                canonical_pressure_mobility,
                canonical_hard_fixed_component_mask,
                i,
                j,
                k,
                2,
                canonical_authority,
            )
            coeff = ti.cast(1.0, ti.f64) / (
                ti.cast(cell_width_y_m[j], ti.f64)
                * ti.cast(center_distance_y_m[j], ti.f64)
            )
            products[2] = coeff * face_weight
            denominator += coeff * face_weight
        if j < ny - 1 and obstacle[i, j + 1, k] == 0:
            face_weight = self._fv_pressure_velocity_face_weight(
                velocity_dirichlet_boundary_active,
                velocity_dirichlet_boundary_projection_weight,
                velocity_dirichlet_pressure_hard_fixed_component_mask,
                canonical_active_component_mask,
                canonical_pressure_mobility,
                canonical_hard_fixed_component_mask,
                i,
                j + 1,
                k,
                2,
                canonical_authority,
            )
            coeff = ti.cast(1.0, ti.f64) / (
                ti.cast(cell_width_y_m[j], ti.f64)
                * ti.cast(center_distance_y_m[j + 1], ti.f64)
            )
            products[3] = coeff * face_weight
            denominator += coeff * face_weight
        if k > 0 and obstacle[i, j, k - 1] == 0:
            face_weight = self._fv_pressure_velocity_face_weight(
                velocity_dirichlet_boundary_active,
                velocity_dirichlet_boundary_projection_weight,
                velocity_dirichlet_pressure_hard_fixed_component_mask,
                canonical_active_component_mask,
                canonical_pressure_mobility,
                canonical_hard_fixed_component_mask,
                i,
                j,
                k,
                4,
                canonical_authority,
            )
            coeff = ti.cast(1.0, ti.f64) / (
                ti.cast(cell_width_z_m[k], ti.f64)
                * ti.cast(center_distance_z_m[k], ti.f64)
            )
            products[4] = coeff * face_weight
            denominator += coeff * face_weight
        if k < nz - 1 and obstacle[i, j, k + 1] == 0:
            face_weight = self._fv_pressure_velocity_face_weight(
                velocity_dirichlet_boundary_active,
                velocity_dirichlet_boundary_projection_weight,
                velocity_dirichlet_pressure_hard_fixed_component_mask,
                canonical_active_component_mask,
                canonical_pressure_mobility,
                canonical_hard_fixed_component_mask,
                i,
                j,
                k + 1,
                4,
                canonical_authority,
            )
            coeff = ti.cast(1.0, ti.f64) / (
                ti.cast(cell_width_z_m[k], ti.f64)
                * ti.cast(center_distance_z_m[k + 1], ti.f64)
            )
            products[5] = coeff * face_weight
            denominator += coeff * face_weight
        if pressure_outlet_zmin == 1 and k == 0:
            outlet_face_weight = self._fv_pressure_velocity_face_weight(
                velocity_dirichlet_boundary_active,
                velocity_dirichlet_boundary_projection_weight,
                velocity_dirichlet_pressure_hard_fixed_component_mask,
                canonical_active_component_mask,
                canonical_pressure_mobility,
                canonical_hard_fixed_component_mask,
                i,
                j,
                k,
                4,
                canonical_authority,
            )
            denominator += outlet_face_weight * ti.cast(2.0, ti.f64) / (
                ti.cast(cell_width_z_m[k], ti.f64)
                * ti.cast(cell_width_z_m[k], ti.f64)
            )
        return products, denominator

    @ti.kernel
    def _copy_scalar_field_kernel(self, destination: ti.template(), source: ti.template()):
        for i, j, k in destination:
            destination[i, j, k] = source[i, j, k]

    @ti.kernel
    def _fv_diagonal_kernel(
        self,
        diagonal: ti.template(),
        pressure_outlet_zmin: ti.i32,
        canonical_authority: ti.template(),
    ):
        for i, j, k in diagonal:
            if self.obstacle[i, j, k] == 1:
                diagonal[i, j, k] = 0.0
            else:
                _, denominator = self._fv_pressure_neighbor_sum_denominator(
                    self.pressure,
                    self.obstacle,
                    self.velocity_dirichlet_boundary_active,
                    self.velocity_dirichlet_boundary_projection_weight,
                    self.velocity_dirichlet_pressure_hard_fixed_component_mask,
                    self.velocity_dirichlet_boundary_active_component_mask,
                    self.velocity_dirichlet_boundary_pressure_mobility,
                    self.velocity_dirichlet_boundary_hard_fixed_component_mask,
                    self.cell_width_x_m,
                    self.cell_width_y_m,
                    self.cell_width_z_m,
                    self.center_distance_x_m,
                    self.center_distance_y_m,
                    self.center_distance_z_m,
                    i,
                    j,
                    k,
                    self.nx,
                    self.ny,
                    self.nz,
                    pressure_outlet_zmin,
                    canonical_authority,
                )
                diagonal[i, j, k] = denominator + self.pressure_interface_matrix_diagonal[i, j, k]

    @ti.kernel
    def _fv_laplacian_apply_kernel(
        self,
        pressure: ti.template(),
        output: ti.template(),
        pressure_outlet_zmin: ti.i32,
        canonical_authority: ti.template(),
    ):
        for i, j, k in output:
            if self.obstacle[i, j, k] == 1:
                output[i, j, k] = 0.0
            else:
                neighbor_sum, denominator = self._fv_pressure_neighbor_sum_denominator(
                    pressure,
                    self.obstacle,
                    self.velocity_dirichlet_boundary_active,
                    self.velocity_dirichlet_boundary_projection_weight,
                    self.velocity_dirichlet_pressure_hard_fixed_component_mask,
                    self.velocity_dirichlet_boundary_active_component_mask,
                    self.velocity_dirichlet_boundary_pressure_mobility,
                    self.velocity_dirichlet_boundary_hard_fixed_component_mask,
                    self.cell_width_x_m,
                    self.cell_width_y_m,
                    self.cell_width_z_m,
                    self.center_distance_x_m,
                    self.center_distance_y_m,
                    self.center_distance_z_m,
                    i,
                    j,
                    k,
                    self.nx,
                    self.ny,
                    self.nz,
                    pressure_outlet_zmin,
                    canonical_authority,
                )
                output[i, j, k] = (
                    (denominator + self.pressure_interface_matrix_diagonal[i, j, k])
                    * pressure[i, j, k]
                    - neighbor_sum
                )
        row_limit = ti.min(
            self.pressure_interface_row_count[None],
            self.pressure_interface_row_capacity,
        )
        for row in range(ti.min(row_limit, self.pressure_interface_row_capacity)):
            if row < row_limit:
                owner = self.pressure_interface_row_owner[row]
                neighbor = self.pressure_interface_row_neighbor[row]
                i = owner.x
                j = owner.y
                k = owner.z
                ni = neighbor.x
                nj = neighbor.y
                nk = neighbor.z
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
                    and transmissibility > 0.0
                ):
                    if self.pressure_interface_row_support_count[row] == 0:
                        active_volume_m3 = self._cell_volume_f64_m3(i, j, k)
                        neighbor_volume_m3 = self._cell_volume_f64_m3(ni, nj, nk)
                        ti.atomic_add(
                            output[i, j, k],
                            -(transmissibility / active_volume_m3)
                            * pressure[ni, nj, nk],
                        )
                        ti.atomic_add(
                            output[ni, nj, nk],
                            -(transmissibility / neighbor_volume_m3) * pressure[i, j, k],
                        )
                    else:
                        self._accumulate_pressure_interface_rank_one_offdiagonal(
                            row, pressure, output, 1.0,
                        )
        for i, j, k in output:
            if (
                self.pressure_interface_row_count[None] == 0
                and (
                    self.pressure_interface_coupling_active[i, j, k] > 0
                    and self.obstacle[i, j, k] == 0
                )
            ):
                slot_count = ti.min(
                    self.pressure_interface_coupling_active[i, j, k],
                    PRESSURE_INTERFACE_COUPLING_SLOT_COUNT,
                )
                for slot in ti.static(range(PRESSURE_INTERFACE_COUPLING_SLOT_COUNT)):
                    if slot < slot_count:
                        neighbor = self.pressure_interface_coupling_neighbor[
                            i,
                            j,
                            k,
                        ]
                        transmissibility = (
                            self.pressure_interface_coupling_coefficient[i, j, k]
                        )
                        if ti.static(slot > 0):
                            extra_slot = slot - 1
                            neighbor = (
                                self.pressure_interface_coupling_extra_neighbor[
                                    i,
                                    j,
                                    k,
                                    extra_slot,
                                ]
                            )
                            transmissibility = (
                                self.pressure_interface_coupling_extra_coefficient[
                                    i,
                                    j,
                                    k,
                                    extra_slot,
                                ]
                            )
                        ni = neighbor.x
                        nj = neighbor.y
                        nk = neighbor.z
                        if (
                            0 <= ni < self.nx
                            and 0 <= nj < self.ny
                            and 0 <= nk < self.nz
                            and self.obstacle[ni, nj, nk] == 0
                            and transmissibility > 0.0
                        ):
                            active_volume_m3 = self._cell_volume_f64_m3(i, j, k)
                            neighbor_volume_m3 = self._cell_volume_f64_m3(
                                ni, nj, nk
                            )
                            ti.atomic_add(
                                output[i, j, k],
                                -(transmissibility / active_volume_m3)
                                * pressure[ni, nj, nk],
                            )
                            ti.atomic_add(
                                output[ni, nj, nk],
                                -(transmissibility / neighbor_volume_m3)
                                * pressure[i, j, k],
                            )

    def _apply_fv_pressure_operator(
        self,
        pressure: object,
        output: object,
        pressure_outlet_zmin: int,
        canonical_authority: int,
        pressure_velocity_nullspace_projector: object | None = None,
    ) -> None:
        """Apply the exact fine FV operator, optionally preserving marker Q."""

        self.last_cg_operator_apply_count += 1
        self._fv_laplacian_apply_kernel(
            pressure,
            output,
            pressure_outlet_zmin,
            canonical_authority,
        )
        if pressure_velocity_nullspace_projector is None:
            return
        raise NotImplementedError("a pressure_velocity_nullspace_projector is not supported")

    def _apply_fv_pressure_velocity_correction(
        self,
        dt_over_rho: float,
        pressure_outlet_zmin: int,
        canonical_authority: int,
        pressure_velocity_nullspace_projector: object | None = None,
    ) -> dict[str, object]:
        """Commit the same pressure correction used by the fine operator."""

        if pressure_velocity_nullspace_projector is None:
            self._subtract_pressure_gradient_kernel(
                dt_over_rho,
                pressure_outlet_zmin,
                canonical_authority,
            )
            return {}
        raise NotImplementedError("a pressure_velocity_nullspace_projector is not supported")

    @ti.kernel
    def _reset_zmin_projection_flux_report_kernel(self):
        self.report_zmin_pressure_outlet_flux_m3s[None] = 0.0
        self.report_zmin_projection_pre_velocity_outlet_flux_m3s[None] = 0.0
        self.report_zmin_pressure_step_pre_velocity_outlet_flux_m3s[None] = 0.0
        self.report_zmin_projection_post_pressure_velocity_outlet_flux_m3s[None] = 0.0
        self.report_zmin_projection_post_boundary_velocity_outlet_flux_m3s[None] = 0.0

    @ti.kernel
    def _record_zmin_velocity_flux_kernel(self, destination: ti.template()):
        destination[None] = 0.0
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.obstacle[i, j, 0] == 0:
                outlet_cell_area_m2 = self._z_face_area_f64_m2(i, j)
                ti.atomic_add(
                    destination[None],
                    -ti.cast(self.velocity[i, j, 0].z, ti.f64)
                    * outlet_cell_area_m2,
                )

    @ti.kernel
    def _accumulate_zmin_pressure_correction_flux_kernel(self):
        self.report_zmin_projection_post_pressure_velocity_outlet_flux_m3s[None] = 0.0
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.obstacle[i, j, 0] == 0:
                outlet_cell_area_m2 = self._z_face_area_f64_m2(i, j)
                ti.atomic_add(
                    self.report_zmin_projection_post_pressure_velocity_outlet_flux_m3s[None],
                    -ti.cast(self.velocity[i, j, 0].z, ti.f64)
                    * outlet_cell_area_m2,
                )
        self.report_zmin_pressure_outlet_flux_m3s[None] += (
            self.report_zmin_projection_post_pressure_velocity_outlet_flux_m3s[None]
            - self.report_zmin_pressure_step_pre_velocity_outlet_flux_m3s[None]
        )

    @ti.kernel
    def _copy_pressure_to_accum_kernel(self):
        for i, j, k in self.pressure:
            self.pressure_accum[i, j, k] = self.pressure[i, j, k]

    @ti.kernel
    def _accumulate_pressure_correction_kernel(self):
        for i, j, k in self.pressure:
            corrected = self.pressure_accum[i, j, k] + self.pressure[i, j, k]
            self.pressure_accum[i, j, k] = corrected
            self.pressure[i, j, k] = corrected
            self.pressure_tmp[i, j, k] = corrected

    @ti.kernel
    def _apply_closed_boundary_no_normal_flow_kernel(self, pressure_outlet_zmin: ti.i32):
        # Only the domain min-side faces (index 0) are actual outward wall
        # faces in this backward-face velocity layout: velocity[i,j,k]
        # stores the face between cell (..k-1) and cell (..k), so index 0
        # is the wall face of cell 0 (no cell -1 exists) while index
        # nx-1/ny-1/nz-1 is an INTERIOR face between the last two cells,
        # already governed by the pressure projection. The Poisson matrix
        # and divergence stencil already enforce no-flux at the max walls
        # via the missing-neighbor convention, so clamping those interior
        # faces here would overwrite a correctly projected value.
        for j, k in ti.ndrange(self.ny, self.nz):
            if self.velocity_dirichlet_boundary_active[0, j, k] == 0:
                self.velocity[0, j, k].x = 0.0
        for i, k in ti.ndrange(self.nx, self.nz):
            if self.velocity_dirichlet_boundary_active[i, 0, k] == 0:
                self.velocity[i, 0, k].y = 0.0
        for i, j in ti.ndrange(self.nx, self.ny):
            if (
                pressure_outlet_zmin == 0
                and self.velocity_dirichlet_boundary_active[i, j, 0] == 0
            ):
                self.velocity[i, j, 0].z = 0.0

    @ti.kernel
    def _apply_obstacle_no_normal_flow_kernel(
        self,
        canonical_authority: ti.template(),
    ):
        for i, j, k in self.velocity:
            if (
                self.obstacle[i, j, k] != 0
                or (i > 0 and self.obstacle[i - 1, j, k] != 0)
            ) and not self._obstacle_interface_component_is_prescribed(
                i, j, k, 0, canonical_authority
            ):
                self.velocity[i, j, k].x = 0.0
            if (
                self.obstacle[i, j, k] != 0
                or (j > 0 and self.obstacle[i, j - 1, k] != 0)
            ) and not self._obstacle_interface_component_is_prescribed(
                i, j, k, 1, canonical_authority
            ):
                self.velocity[i, j, k].y = 0.0
            if (
                self.obstacle[i, j, k] != 0
                or (k > 0 and self.obstacle[i, j, k - 1] != 0)
            ) and not self._obstacle_interface_component_is_prescribed(
                i, j, k, 2, canonical_authority
            ):
                self.velocity[i, j, k].z = 0.0

    @ti.kernel
    def _subtract_pressure_gradient_kernel(
        self,
        dt_over_rho: ti.f32,
        pressure_outlet_zmin: ti.i32,
        canonical_authority: ti.template(),
    ):
        for i, j, k in self.velocity:
            im = ti.max(i - 1, 0)
            jm = ti.max(j - 1, 0)
            km = ti.max(k - 1, 0)
            center = self.pressure[i, j, k]
            if self.obstacle[i, j, k] == 1:
                cleaned_velocity = self.velocity[i, j, k]
                for axis in ti.static(range(3)):
                    if not self._obstacle_interface_component_is_prescribed(
                        i, j, k, axis, canonical_authority
                    ):
                        cleaned_velocity[axis] = 0.0
                self.velocity[i, j, k] = cleaned_velocity
            else:
                grad = ti.Vector(
                    [
                        ti.cast(0.0, ti.f64),
                        ti.cast(0.0, ti.f64),
                        ti.cast(0.0, ti.f64),
                    ]
                )
                if i > 0 and self.obstacle[im, j, k] == 0:
                    grad.x = (center - self.pressure[im, j, k]) / self.center_distance_x_m[i]
                if j > 0 and self.obstacle[i, jm, k] == 0:
                    grad.y = (center - self.pressure[i, jm, k]) / self.center_distance_y_m[j]
                if k > 0 and self.obstacle[i, j, km] == 0:
                    grad.z = (center - self.pressure[i, j, km]) / self.center_distance_z_m[k]
                elif pressure_outlet_zmin == 1 and k == 0:
                    grad.z = 2.0 * center / self.cell_width_z_m[k]
                correction_weight = ti.Vector.zero(ti.f64, 3)
                for axis in ti.static(range(3)):
                    correction_weight[axis] = (
                        self._velocity_dirichlet_pressure_face_weight(
                            self.velocity_dirichlet_boundary_active,
                            self.velocity_dirichlet_boundary_projection_weight,
                            self.velocity_dirichlet_pressure_hard_fixed_component_mask,
                            self.velocity_dirichlet_boundary_active_component_mask,
                            self.velocity_dirichlet_boundary_pressure_mobility,
                            self.velocity_dirichlet_boundary_hard_fixed_component_mask,
                            i,
                            j,
                            k,
                            axis,
                            canonical_authority,
                        )
                    )
                correction = (
                    correction_weight * ti.cast(dt_over_rho, ti.f64) * grad
                )
                self.velocity[i, j, k] -= ti.cast(correction, ti.f32)

    @ti.func
    def _accumulate_pressure_interface_row_projection_divergence(
        self,
        owner: ti.types.vector(3, ti.i32),
        neighbor: ti.types.vector(3, ti.i32),
        transmissibility: ti.f64,
        dt_over_rho: ti.f64,
    ):
        i = owner.x
        j = owner.y
        k = owner.z
        ni = neighbor.x
        nj = neighbor.y
        nk = neighbor.z
        if (
            0 <= i < self.nx
            and 0 <= j < self.ny
            and 0 <= k < self.nz
            and 0 <= ni < self.nx
            and 0 <= nj < self.ny
            and 0 <= nk < self.nz
            and self.obstacle[i, j, k] == 0
            and self.obstacle[ni, nj, nk] == 0
            and transmissibility > 0.0
        ):
            active_volume_m3 = self._cell_volume_f64_m3(i, j, k)
            neighbor_volume_m3 = self._cell_volume_f64_m3(ni, nj, nk)
            ti.atomic_add(
                self.pressure_interface_projection_divergence_s[i, j, k],
                -dt_over_rho
                * (transmissibility / active_volume_m3)
                * self.pressure[ni, nj, nk],
            )
            ti.atomic_add(
                self.pressure_interface_projection_divergence_s[ni, nj, nk],
                -dt_over_rho
                * (transmissibility / neighbor_volume_m3)
                * self.pressure[i, j, k],
            )

    @ti.kernel
    def _update_pressure_interface_projection_divergence_kernel(
        self,
        dt_over_rho: ti.f64,
    ):
        for i, j, k in self.pressure_interface_projection_divergence_s:
            if self.obstacle[i, j, k] == 1:
                self.pressure_interface_projection_divergence_s[i, j, k] = 0.0
            else:
                self.pressure_interface_projection_divergence_s[i, j, k] = (
                    dt_over_rho
                    * (
                        self.pressure_interface_matrix_diagonal[i, j, k]
                        * self.pressure[i, j, k]
                        - self.pressure_interface_matrix_rhs[i, j, k]
                    )
                )
        row_limit = ti.min(
            self.pressure_interface_row_count[None],
            self.pressure_interface_row_capacity,
        )
        for row in range(ti.min(row_limit, self.pressure_interface_row_capacity)):
            if row < row_limit:
                if self.pressure_interface_row_support_count[row] > 0:
                    self._accumulate_pressure_interface_rank_one_offdiagonal(
                        row, self.pressure,
                        self.pressure_interface_projection_divergence_s, dt_over_rho,
                    )
                else:
                    self._accumulate_pressure_interface_row_projection_divergence(
                        self.pressure_interface_row_owner[row],
                        self.pressure_interface_row_neighbor[row],
                        self.pressure_interface_row_transmissibility[row],
                        dt_over_rho,
                    )
        for i, j, k in self.pressure_interface_projection_divergence_s:
            if (
                self.pressure_interface_row_count[None] == 0
                and self.pressure_interface_coupling_active[i, j, k] > 0
                and self.obstacle[i, j, k] == 0
            ):
                slot_count = ti.min(
                    self.pressure_interface_coupling_active[i, j, k],
                    PRESSURE_INTERFACE_COUPLING_SLOT_COUNT,
                )
                for slot in ti.static(range(PRESSURE_INTERFACE_COUPLING_SLOT_COUNT)):
                    if slot < slot_count:
                        neighbor = self.pressure_interface_coupling_neighbor[i, j, k]
                        transmissibility = self.pressure_interface_coupling_coefficient[
                            i,
                            j,
                            k,
                        ]
                        if ti.static(slot > 0):
                            extra_slot = slot - 1
                            neighbor = self.pressure_interface_coupling_extra_neighbor[
                                i,
                                j,
                                k,
                                extra_slot,
                            ]
                            transmissibility = (
                                self.pressure_interface_coupling_extra_coefficient[
                                    i,
                                    j,
                                    k,
                                    extra_slot,
                                ]
                            )
                        self._accumulate_pressure_interface_row_projection_divergence(
                            ti.Vector([i, j, k]),
                            neighbor,
                            transmissibility,
                            dt_over_rho,
                        )

    @ti.kernel
    def _accumulate_pressure_interface_projection_divergence_kernel(
        self,
        dt_over_rho: ti.f64,
    ):
        for i, j, k in self.pressure_interface_projection_divergence_s:
            if self.obstacle[i, j, k] == 0:
                self.pressure_interface_projection_divergence_s[i, j, k] += (
                    dt_over_rho
                    * (
                        self.pressure_interface_matrix_diagonal[i, j, k]
                        * self.pressure[i, j, k]
                        - self.pressure_interface_matrix_rhs[i, j, k]
                    )
                )
            else:
                self.pressure_interface_projection_divergence_s[i, j, k] = 0.0
        row_limit = ti.min(
            self.pressure_interface_row_count[None],
            self.pressure_interface_row_capacity,
        )
        for row in range(ti.min(row_limit, self.pressure_interface_row_capacity)):
            if row < row_limit:
                if self.pressure_interface_row_support_count[row] > 0:
                    self._accumulate_pressure_interface_rank_one_offdiagonal(
                        row, self.pressure,
                        self.pressure_interface_projection_divergence_s, dt_over_rho,
                    )
                else:
                    self._accumulate_pressure_interface_row_projection_divergence(
                        self.pressure_interface_row_owner[row],
                        self.pressure_interface_row_neighbor[row],
                        self.pressure_interface_row_transmissibility[row],
                        dt_over_rho,
                    )
        for i, j, k in self.pressure_interface_projection_divergence_s:
            if (
                self.pressure_interface_row_count[None] == 0
                and self.pressure_interface_coupling_active[i, j, k] > 0
                and self.obstacle[i, j, k] == 0
            ):
                slot_count = ti.min(
                    self.pressure_interface_coupling_active[i, j, k],
                    PRESSURE_INTERFACE_COUPLING_SLOT_COUNT,
                )
                for slot in ti.static(range(PRESSURE_INTERFACE_COUPLING_SLOT_COUNT)):
                    if slot < slot_count:
                        neighbor = self.pressure_interface_coupling_neighbor[i, j, k]
                        transmissibility = self.pressure_interface_coupling_coefficient[
                            i,
                            j,
                            k,
                        ]
                        if ti.static(slot > 0):
                            extra_slot = slot - 1
                            neighbor = self.pressure_interface_coupling_extra_neighbor[
                                i,
                                j,
                                k,
                                extra_slot,
                            ]
                            transmissibility = (
                                self.pressure_interface_coupling_extra_coefficient[
                                    i,
                                    j,
                                    k,
                                    extra_slot,
                                ]
                            )
                        self._accumulate_pressure_interface_row_projection_divergence(
                            ti.Vector([i, j, k]),
                            neighbor,
                            transmissibility,
                            dt_over_rho,
                        )

    @ti.kernel
    def _apply_zmin_no_backflow_kernel(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.obstacle[i, j, 0] == 0 and self.velocity[i, j, 0].z > 0.0:
                self.velocity[i, j, 0].z = 0.0

    def _solve_pressure_poisson_with_solver(
        self,
        *,
        iterations: int,
        rhs_scale: float,
        pressure_outlet_zmin: bool,
        pressure_solver: str,
        multigrid_cycles: int | None,
        cg_tolerance: float,
        cg_preconditioner: str = "auto",
        remove_nullspace_mean: bool = True,
        pressure_nullspace_component_count: int = 0,
        pressure_interface_matrix_active: bool = False,
        pressure_components_use_operator_graph: bool | None = None,
        report_requested: bool = True,
        pressure_velocity_nullspace_projector: object | None = None,
        closed_neumann_velocity_inlet_zmax_mode: int | None = None,
    ) -> None:
        if pressure_solver in {"jacobi", "compact_jacobi"}:
            raise NotImplementedError("pressure_solver must be 'fv_cg' (conjugate gradient with a multigrid preconditioner; hypre can replace its linear solve)")
        if pressure_solver == "fv_jacobi":
            raise NotImplementedError("pressure_solver must be 'fv_cg' (conjugate gradient with a multigrid preconditioner; hypre can replace its linear solve)")
        if pressure_solver == "fv_multigrid":
            raise NotImplementedError("pressure_solver must be 'fv_cg' (conjugate gradient with a multigrid preconditioner; hypre can replace its linear solve)")
        if pressure_solver == "fv_cg":
            resolved_operator_graph = pressure_components_use_operator_graph
            if resolved_operator_graph is None:
                resolved_operator_graph = bool(
                    pressure_outlet_zmin
                    and pressure_interface_matrix_active
                )
            self._solve_pressure_poisson_fv_cg(
                closed_neumann_velocity_inlet_zmax_mode=(
                    closed_neumann_velocity_inlet_zmax_mode
                ),
                iterations=int(iterations),
                rhs_scale=float(rhs_scale),
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                tolerance=float(cg_tolerance),
                preconditioner=str(cg_preconditioner),
                remove_nullspace_mean=bool(remove_nullspace_mean),
                pressure_nullspace_component_count=int(
                    pressure_nullspace_component_count
                ),
                pressure_interface_matrix_active=bool(
                    pressure_interface_matrix_active
                ),
                pressure_components_use_operator_graph=bool(
                    resolved_operator_graph
                ),
                report_requested=bool(report_requested),
                pressure_velocity_nullspace_projector=(
                    pressure_velocity_nullspace_projector
                ),
            )
            self._confirm_pressure_poisson_fv_cg_exact_residual(
                pressure_outlet_zmin=bool(pressure_outlet_zmin),
                tolerance=float(cg_tolerance),
                remove_nullspace_mean=bool(remove_nullspace_mean),
                pressure_nullspace_component_count=int(
                    pressure_nullspace_component_count
                ),
                pressure_components_use_operator_graph=bool(
                    resolved_operator_graph
                ),
                pressure_velocity_nullspace_projector=(
                    pressure_velocity_nullspace_projector
                ),
            )
            return
        raise ValueError(f"unsupported pressure_solver: {pressure_solver!r}")

    def _pressure_solve_failure_diagnostics(
        self,
        *,
        tolerance: float,
        pressure_interface_matrix_active: bool,
        context: Mapping[str, object],
    ) -> dict[str, object]:
        interface_report = self.pressure_interface_matrix_terms_report()
        row_diagonal_integral = float(interface_report["row_diagonal_integral"])
        row_diagonal_abs_mismatch = float(
            interface_report["row_diagonal_integral_abs_mismatch"]
        )
        return {
            "preconditioner_requested": str(self.last_cg_preconditioner_requested),
            "preconditioner_effective": str(self.last_cg_preconditioner_effective),
            "multigrid_smoothing": {
                "pre": int(self.last_cg_multigrid_pre_smooth_iterations),
                "coarse": int(self.last_cg_multigrid_coarse_smooth_iterations),
                "post": int(self.last_cg_multigrid_post_smooth_iterations),
            },
            "multigrid_levels": [list(shape) for shape in self._mg_shapes],
            "multigrid_level_count": len(self._mg_shapes),
            "multigrid_apply_count": int(self.last_cg_multigrid_apply_count),
            "multigrid_to_jacobi_fallback_count": int(
                self.last_cg_multigrid_to_jacobi_fallback_count
            ),
            "iterations": int(self.last_cg_iterations),
            "pcg_iterations": int(self.last_cg_pcg_iterations),
            "bicgstab_iterations": int(self.last_cg_bicgstab_iterations),
            "recursive_relative_residual": float(
                self.last_cg_recursive_relative_residual
            ),
            "exact_relative_residual": float(self.last_cg_exact_relative_residual),
            "exact_residual_confirmation_count": int(
                self.last_cg_exact_residual_confirmation_count
            ),
            "tolerance": float(tolerance),
            "breakdown": str(self.last_cg_breakdown),
            "restart_count": int(self.last_cg_restart_count),
            "pressure_interface_matrix_active": bool(
                pressure_interface_matrix_active
            ),
            "interface_row_count": int(interface_report["row_count"]),
            "interface_row_active_count": int(interface_report["row_active_count"]),
            "interface_row_invalid_count": int(interface_report["row_invalid_count"]),
            "interface_row_overflow_count": int(interface_report["row_overflow_count"]),
            "interface_row_diagonal_integral_abs_mismatch": (
                row_diagonal_abs_mismatch
            ),
            "interface_row_diagonal_integral_rel_mismatch": (
                row_diagonal_abs_mismatch
                / max(abs(row_diagonal_integral), 1.0e-30)
            ),
            "correction_applied": False,
            "phase": _diagnostic_str(context.get("phase", "unknown")),
            "step_index_local": _diagnostic_int(
                context.get("step_index_local", -1)
            ),
            "step_index_global": _diagnostic_int(
                context.get("step_index_global", -1)
            ),
            "context": dict(context),
        }

    def _pressure_projection_physical_failure_diagnostics(
        self,
        *,
        tolerance: float,
        pressure_interface_matrix_active: bool,
        context: Mapping[str, object],
        reason: str,
        action: str,
        compatibility_measured: bool,
        incompatible_component_count: int,
        rhs_mean_max_abs: float,
        rhs_integral_max_abs: float,
    ) -> dict[str, object]:
        diagnostics = self._pressure_solve_failure_diagnostics(
            tolerance=float(tolerance),
            pressure_interface_matrix_active=bool(
                pressure_interface_matrix_active
            ),
            context=context,
        )
        for residual_key in (
            "recursive_relative_residual",
            "exact_relative_residual",
        ):
            residual = float(diagnostics[residual_key])
            diagnostics[residual_key] = (
                residual if math.isfinite(residual) else None
            )
        component_count = int(
            self.last_pressure_unreached_component_count
        )
        raw_component_count = int(
            self.last_pressure_unreached_component_raw_count
        )
        component_overflow = bool(
            self.last_pressure_unreached_component_overflow
        )
        diagnostics.update(
            {
                "stage": "pressure_projection_physical_gate",
                "reason": str(reason),
                "count": component_count,
                "capacity": int(self.pressure_component_capacity),
                "failure_kind": "pressure_projection_physical_gate",
                "pressure_projection_physical_failure": True,
                "pressure_projection_physical_failure_reason": str(reason),
                "pressure_projection_physical_failure_action": str(action),
                "correction_applied": False,
                "pressure_unreached_cell_count": int(
                    self.last_pressure_unreached_cell_count
                ),
                "pressure_unreached_raw_cell_count": int(
                    self.last_pressure_unreached_raw_cell_count
                ),
                "pressure_unreached_component_count": component_count,
                "pressure_unreached_component_raw_count": (
                    raw_component_count
                ),
                "pressure_unreached_component_capacity": int(
                    self.pressure_component_capacity
                ),
                "pressure_unreached_component_overflow": (
                    component_overflow
                ),
                "cg_unreached_component_count": component_count,
                "cg_unreached_component_raw_count": raw_component_count,
                "cg_unreached_component_capacity": int(
                    self.pressure_component_capacity
                ),
                "cg_unreached_component_overflow": component_overflow,
                "pressure_component_labels_converged": bool(
                    self.last_pressure_component_labels_converged
                ),
                "pressure_reachability_converged": bool(
                    self.last_pressure_reachability_converged
                ),
                "pressure_reachability_frontier_overflow": bool(
                    self.last_pressure_reachability_frontier_overflow
                ),
                "pressure_reachability_frontier_levels": int(
                    self.last_pressure_reachability_sweeps
                ),
                "unreached_rhs_compatibility_measured": bool(
                    compatibility_measured
                ),
                "unreached_incompatible_component_count": int(
                    incompatible_component_count
                ),
                "unreached_component_rhs_mean_max_abs": float(
                    rhs_mean_max_abs
                ),
                "unreached_component_rhs_integral_max_abs": float(
                    rhs_integral_max_abs
                ),
            }
        )
        return diagnostics

    @ti.kernel
    def _count_external_exact_zmin_normal_rows_kernel(self):
        self.reduction_count[None] = 0
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.obstacle[i, j, 0] == 0 and (self.external_velocity_boundary_z_face_active_component_mask[0, i, j] & 4) != 0:
                ti.atomic_add(self.reduction_count[None], 1)

    def _validate_pressure_outlet_zmin_boundary(self) -> None:
        self._count_external_exact_zmin_normal_rows_kernel()
        if int(self.reduction_count[None]) > 0:
            raise ValueError(
                "pressure_outlet_zmin=True cannot be combined with an external exact "
                "zmin normal velocity on a fluid face; mixed pressure/velocity zmin "
                "boundaries are not supported by the pressure hierarchy"
            )
