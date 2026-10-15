"""Momentum advection: MUSCL-TVD fluxes on the MAC grid (mixin of IncompressibleFluidSolver)."""

import numpy as np
import taichi as ti


class MomentumAdvection:
    """Mixin of IncompressibleFluidSolver; not used on its own.  Contract (solver attributes):
      reads:  cell_center_x_m, cell_center_y_m, cell_center_z_m, cell_face_x_m, cell_face_y_m, cell_face_z_m,
    cell_width_x_m, cell_width_y_m, cell_width_z_m, center_distance_x_m, center_distance_y_m, center_distance_z_m,
    external_velocity_boundary_x_face_active_component_mask, external_velocity_boundary_x_face_value_mps,
    external_velocity_boundary_y_face_active_component_mask, external_velocity_boundary_y_face_value_mps,
    external_velocity_boundary_z_face_active_component_mask, external_velocity_boundary_z_face_value_mps,
    muscl_momentum_dual_volume_m3, muscl_momentum_flux_x, muscl_momentum_flux_y, muscl_momentum_flux_z,
    muscl_momentum_volume_flux_x_half_m3_s, muscl_momentum_volume_flux_y_half_m3_s,
    muscl_momentum_volume_flux_z_half_m3_s, muscl_normal_velocity_x, muscl_normal_velocity_y,
    muscl_normal_velocity_z, muscl_velocity_extension_slope, muscl_velocity_slope, muscl_velocity_smooth_mask, nx,
    ny, nz, obstacle, reduction_max, velocity, velocity_dirichlet_boundary_active,
    velocity_dirichlet_boundary_active_component_mask, velocity_dirichlet_boundary_component_enforcement_weight,
    velocity_dirichlet_boundary_enforcement_weight, velocity_dirichlet_boundary_external_exact_component_mask,
    velocity_dirichlet_boundary_hard_fixed_component_mask, velocity_dirichlet_boundary_owned_component_mask,
    velocity_dirichlet_boundary_owned_row, velocity_dirichlet_boundary_pressure_mobility,
    velocity_dirichlet_boundary_value_mps, velocity_prev, velocity_transport_base
      writes: -"""

    @ti.kernel
    def _copy_velocity_to_prev_kernel(self):
        for i, j, k in self.velocity:
            self.velocity_prev[i, j, k] = self.velocity[i, j, k]

    @ti.kernel
    def _copy_velocity_to_transport_base_kernel(self):
        for i, j, k in self.velocity:
            self.velocity_transport_base[i, j, k] = self.velocity[i, j, k]

    @ti.kernel
    def _restore_velocity_from_transport_base_kernel(self):
        for i, j, k in self.velocity:
            self.velocity[i, j, k] = self.velocity_transport_base[i, j, k]

    @ti.func
    def _muscl_minmod3(self, first, centered, second):
        value = 0.0
        if first * centered > 0.0 and first * second > 0.0:
            magnitude = ti.min(
                ti.abs(first), ti.min(ti.abs(centered), ti.abs(second))
            )
            value = magnitude if first > 0.0 else -magnitude
        return value

    @ti.func
    def _muscl_smooth_extremum_from_slopes(
        self,
        backward,
        forward,
        outer_backward,
        outer_forward,
        outer_minus_distance,
        minus_distance,
        plus_distance,
        outer_plus_distance,
        value_scale,
        stencil_span,
    ):
        """Recognize a resolved smooth extremum without relaxing shock limiting."""

        d2_minus = 2.0 * (backward - outer_backward) / (
            outer_minus_distance + minus_distance
        )
        d2_center = 2.0 * (forward - backward) / (
            minus_distance + plus_distance
        )
        d2_plus = 2.0 * (outer_forward - forward) / (
            plus_distance + outer_plus_distance
        )
        minimum_curvature = ti.min(
            ti.abs(d2_minus), ti.min(ti.abs(d2_center), ti.abs(d2_plus))
        )
        maximum_curvature = ti.max(
            ti.abs(d2_minus), ti.max(ti.abs(d2_center), ti.abs(d2_plus))
        )
        curvature_epsilon = (
            64.0
            * 1.1920929e-7
            * ti.max(value_scale, 1.0e-30)
            / ti.max(stencil_span * stencil_span, 1.0e-30)
        )
        coherent_curvature = (
            d2_minus * d2_center > 0.0
            and d2_center * d2_plus > 0.0
            and minimum_curvature >= 0.25 * maximum_curvature
        )
        local_spacing = ti.max(
            outer_minus_distance,
            ti.max(
                minus_distance,
                ti.max(plus_distance, outer_plus_distance),
            ),
        )
        resolved_curvature = (
            maximum_curvature * local_spacing * local_spacing
            <= 0.10 * ti.max(value_scale, 1.0e-30)
        )
        smooth = 0.0
        limited_slope = 0.0
        extension_slope = 0.0
        if (
            backward * forward <= 0.0
            and maximum_curvature > curvature_epsilon
            and coherent_curvature
            and resolved_curvature
        ):
            d2_limited_magnitude = ti.min(
                ti.abs(d2_center),
                ti.min(1.25 * ti.abs(d2_minus), 1.25 * ti.abs(d2_plus)),
            )
            d2_limited = (
                d2_limited_magnitude if d2_center > 0.0 else -d2_limited_magnitude
            )
            centered = (
                plus_distance * backward + minus_distance * forward
            ) / (minus_distance + plus_distance)
            extension_slope = (
                1.25 * ti.abs(d2_limited) * ti.max(minus_distance, plus_distance)
            )
            limited_slope = ti.min(
                ti.max(centered, -extension_slope), extension_slope
            )
            smooth = 1.0
        return ti.Vector([smooth, limited_slope, extension_slope])

    @ti.func
    def _muscl_axis_neighbor_is_open(
        self,
        i,
        j,
        k,
        axis: ti.template(),
        side: ti.template(),
    ):
        """Return whether one cell neighbour is connected for reconstruction."""

        neighbor_is_open = 0
        if ti.static(axis == 0):
            if ti.static(side < 0):
                if i > 0:
                    if (
                        self.obstacle[i - 1, j, k] == 0
                        and self._canonical_exact_component_owner(i, j, k, 0) == 0
                    ):
                        neighbor_is_open = 1
            else:
                if i + 1 < self.nx:
                    if (
                        self.obstacle[i + 1, j, k] == 0
                        and self._canonical_exact_component_owner(i + 1, j, k, 0) == 0
                    ):
                        neighbor_is_open = 1
        elif ti.static(axis == 1):
            if ti.static(side < 0):
                if j > 0:
                    if (
                        self.obstacle[i, j - 1, k] == 0
                        and self._canonical_exact_component_owner(i, j, k, 1) == 0
                    ):
                        neighbor_is_open = 1
            else:
                if j + 1 < self.ny:
                    if (
                        self.obstacle[i, j + 1, k] == 0
                        and self._canonical_exact_component_owner(i, j + 1, k, 1) == 0
                    ):
                        neighbor_is_open = 1
        else:
            if ti.static(side < 0):
                if k > 0:
                    if (
                        self.obstacle[i, j, k - 1] == 0
                        and self._canonical_exact_component_owner(i, j, k, 2) == 0
                    ):
                        neighbor_is_open = 1
            else:
                if k + 1 < self.nz:
                    if (
                        self.obstacle[i, j, k + 1] == 0
                        and self._canonical_exact_component_owner(i, j, k + 1, 2) == 0
                    ):
                        neighbor_is_open = 1
        return neighbor_is_open

    @ti.func
    def _muscl_extremum_stencil_is_unconstrained(
        self,
        i,
        j,
        k,
        axis: ti.template(),
        component_mask: ti.template(),
    ):
        unconstrained = 1
        for offset in ti.static(range(-2, 3)):
            ii = i
            jj = j
            kk = k
            if ti.static(axis == 0):
                ii += offset
            elif ti.static(axis == 1):
                jj += offset
            else:
                kk += offset
            if not (0 <= ii < self.nx and 0 <= jj < self.ny and 0 <= kk < self.nz):
                unconstrained = 0
            elif (
                self.velocity_dirichlet_boundary_active[ii, jj, kk] != 0
                or (
                    self.velocity_dirichlet_boundary_active_component_mask[ii, jj, kk]
                    & component_mask
                )
                != 0
            ):
                unconstrained = 0
        # Five sampled cells span four primal faces.  A canonical fluid-fluid
        # wall is not visible in ``obstacle`` and must truncate the stencil at
        # the owned face just like a phase-changing obstacle interface.
        for face_offset in ti.static(range(-1, 3)):
            ii = i
            jj = j
            kk = k
            if ti.static(axis == 0):
                ii += face_offset
            elif ti.static(axis == 1):
                jj += face_offset
            else:
                kk += face_offset
            if 0 <= ii < self.nx and 0 <= jj < self.ny and 0 <= kk < self.nz:
                if (
                    self._canonical_exact_component_owner(
                        ii, jj, kk, axis
                    )
                    != 0
                ):
                    unconstrained = 0
        return unconstrained

    @ti.func
    def _muscl_vector_component_has_smooth_extremum(
        self,
        field: ti.template(),
        i,
        j,
        k,
        axis: ti.template(),
        component: ti.template(),
    ):
        center = field[i, j, k][component]
        minus = center
        plus = center
        outer_minus = center
        outer_plus = center
        minus_distance = 1.0
        plus_distance = 1.0
        outer_minus_distance = 1.0
        outer_plus_distance = 1.0
        complete_stencil = 0
        if ti.static(axis == 0):
            if (
                i > 1
                and i + 2 < self.nx
                and self.obstacle[i - 2, j, k] == 0
                and self.obstacle[i - 1, j, k] == 0
                and self.obstacle[i + 1, j, k] == 0
                and self.obstacle[i + 2, j, k] == 0
                and self._muscl_extremum_stencil_is_unconstrained(
                    i, j, k, axis, 1 << component
                )
                != 0
            ):
                outer_minus = field[i - 2, j, k][component]
                minus = field[i - 1, j, k][component]
                plus = field[i + 1, j, k][component]
                outer_plus = field[i + 2, j, k][component]
                if ti.static(component == 0):
                    outer_minus_distance = self.cell_width_x_m[i - 2]
                    minus_distance = self.cell_width_x_m[i - 1]
                    plus_distance = self.cell_width_x_m[i]
                    outer_plus_distance = self.cell_width_x_m[i + 1]
                else:
                    outer_minus_distance = self.center_distance_x_m[i - 1]
                    minus_distance = self.center_distance_x_m[i]
                    plus_distance = self.center_distance_x_m[i + 1]
                    outer_plus_distance = self.center_distance_x_m[i + 2]
                complete_stencil = 1
        elif ti.static(axis == 1):
            if (
                j > 1
                and j + 2 < self.ny
                and self.obstacle[i, j - 2, k] == 0
                and self.obstacle[i, j - 1, k] == 0
                and self.obstacle[i, j + 1, k] == 0
                and self.obstacle[i, j + 2, k] == 0
                and self._muscl_extremum_stencil_is_unconstrained(
                    i, j, k, axis, 1 << component
                )
                != 0
            ):
                outer_minus = field[i, j - 2, k][component]
                minus = field[i, j - 1, k][component]
                plus = field[i, j + 1, k][component]
                outer_plus = field[i, j + 2, k][component]
                if ti.static(component == 1):
                    outer_minus_distance = self.cell_width_y_m[j - 2]
                    minus_distance = self.cell_width_y_m[j - 1]
                    plus_distance = self.cell_width_y_m[j]
                    outer_plus_distance = self.cell_width_y_m[j + 1]
                else:
                    outer_minus_distance = self.center_distance_y_m[j - 1]
                    minus_distance = self.center_distance_y_m[j]
                    plus_distance = self.center_distance_y_m[j + 1]
                    outer_plus_distance = self.center_distance_y_m[j + 2]
                complete_stencil = 1
        else:
            if (
                k > 1
                and k + 2 < self.nz
                and self.obstacle[i, j, k - 2] == 0
                and self.obstacle[i, j, k - 1] == 0
                and self.obstacle[i, j, k + 1] == 0
                and self.obstacle[i, j, k + 2] == 0
                and self._muscl_extremum_stencil_is_unconstrained(
                    i, j, k, axis, 1 << component
                )
                != 0
            ):
                outer_minus = field[i, j, k - 2][component]
                minus = field[i, j, k - 1][component]
                plus = field[i, j, k + 1][component]
                outer_plus = field[i, j, k + 2][component]
                if ti.static(component == 2):
                    outer_minus_distance = self.cell_width_z_m[k - 2]
                    minus_distance = self.cell_width_z_m[k - 1]
                    plus_distance = self.cell_width_z_m[k]
                    outer_plus_distance = self.cell_width_z_m[k + 1]
                else:
                    outer_minus_distance = self.center_distance_z_m[k - 1]
                    minus_distance = self.center_distance_z_m[k]
                    plus_distance = self.center_distance_z_m[k + 1]
                    outer_plus_distance = self.center_distance_z_m[k + 2]
                complete_stencil = 1
        smooth_info = ti.Vector([0.0, 0.0, 0.0])
        if complete_stencil != 0:
            backward = (center - minus) / minus_distance
            forward = (plus - center) / plus_distance
            outer_backward = (minus - outer_minus) / outer_minus_distance
            outer_forward = (outer_plus - plus) / outer_plus_distance
            value_scale = ti.max(
                ti.abs(center),
                ti.max(
                    ti.abs(minus),
                    ti.max(
                        ti.abs(plus),
                        ti.max(ti.abs(outer_minus), ti.abs(outer_plus)),
                    ),
                ),
            )
            smooth_info = self._muscl_smooth_extremum_from_slopes(
                backward,
                forward,
                outer_backward,
                outer_forward,
                outer_minus_distance,
                minus_distance,
                plus_distance,
                outer_plus_distance,
                value_scale,
                outer_minus_distance
                + minus_distance
                + plus_distance
                + outer_plus_distance,
            )
        return smooth_info

    @ti.func
    def _muscl_vector_component_slope(
        self, field: ti.template(), i, j, k, axis: ti.template(), component: ti.template()
    ):
        center = field[i, j, k][component]
        minus = center
        plus = center
        minus_distance = 1.0
        plus_distance = 1.0
        complete = 0
        if ti.static(axis == 0):
            if (
                i > 0
                and i + 1 < self.nx
                and self.obstacle[i - 1, j, k] == 0
                and self.obstacle[i + 1, j, k] == 0
                and self._muscl_axis_neighbor_is_open(i, j, k, axis, -1) != 0
                and self._muscl_axis_neighbor_is_open(i, j, k, axis, 1) != 0
            ):
                minus = field[i - 1, j, k][component]
                plus = field[i + 1, j, k][component]
                if ti.static(component == 0):
                    minus_distance = self.cell_width_x_m[i - 1]
                    plus_distance = self.cell_width_x_m[i]
                else:
                    minus_distance = self.center_distance_x_m[i]
                    plus_distance = self.center_distance_x_m[i + 1]
                complete = 1
        elif ti.static(axis == 1):
            if (
                j > 0
                and j + 1 < self.ny
                and self.obstacle[i, j - 1, k] == 0
                and self.obstacle[i, j + 1, k] == 0
                and self._muscl_axis_neighbor_is_open(i, j, k, axis, -1) != 0
                and self._muscl_axis_neighbor_is_open(i, j, k, axis, 1) != 0
            ):
                minus = field[i, j - 1, k][component]
                plus = field[i, j + 1, k][component]
                if ti.static(component == 1):
                    minus_distance = self.cell_width_y_m[j - 1]
                    plus_distance = self.cell_width_y_m[j]
                else:
                    minus_distance = self.center_distance_y_m[j]
                    plus_distance = self.center_distance_y_m[j + 1]
                complete = 1
        else:
            if (
                k > 0
                and k + 1 < self.nz
                and self.obstacle[i, j, k - 1] == 0
                and self.obstacle[i, j, k + 1] == 0
                and self._muscl_axis_neighbor_is_open(i, j, k, axis, -1) != 0
                and self._muscl_axis_neighbor_is_open(i, j, k, axis, 1) != 0
            ):
                minus = field[i, j, k - 1][component]
                plus = field[i, j, k + 1][component]
                if ti.static(component == 2):
                    minus_distance = self.cell_width_z_m[k - 1]
                    plus_distance = self.cell_width_z_m[k]
                else:
                    minus_distance = self.center_distance_z_m[k]
                    plus_distance = self.center_distance_z_m[k + 1]
                complete = 1
        slope = 0.0
        if complete != 0:
            backward = (center - minus) / minus_distance
            forward = (plus - center) / plus_distance
            centered = (
                plus_distance * backward + minus_distance * forward
            ) / (minus_distance + plus_distance)
            slope = self._muscl_minmod3(
                2.0 * backward, centered, 2.0 * forward
            )
        return slope

    @ti.kernel
    def _prepare_muscl_velocity_axis_reconstruction_kernel(
        self,
        field: ti.template(),
        axis: ti.template(),
    ):
        for i, j, k in field:
            for component in ti.static(range(3)):
                slope = 0.0
                extension = 0.0
                smooth = 0
                if self.obstacle[i, j, k] == 0:
                    slope = self._muscl_vector_component_slope(
                        field, i, j, k, axis, component
                    )
                    smooth_info = (
                        self._muscl_vector_component_has_smooth_extremum(
                            field, i, j, k, axis, component
                        )
                    )
                    if smooth_info[0] != 0.0:
                        slope = smooth_info[1]
                        extension = smooth_info[2]
                        smooth = 1
                self.muscl_velocity_slope[i, j, k][axis, component] = slope
                self.muscl_velocity_smooth_mask[i, j, k][axis, component] = smooth
                self.muscl_velocity_extension_slope[i, j, k][
                    axis, component
                ] = extension

    @ti.func
    def _muscl_vector_face_state(
        self,
        field: ti.template(),
        i,
        j,
        k,
        axis: ti.template(),
        side: ti.template(),
    ):
        center = field[i, j, k]
        neighbor = center
        neighbor_valid = 0
        if ti.static(axis == 0):
            if ti.static(side < 0):
                if self._muscl_axis_neighbor_is_open(i, j, k, axis, side) != 0:
                    neighbor = field[i - 1, j, k]
                    neighbor_valid = 1
            else:
                if self._muscl_axis_neighbor_is_open(i, j, k, axis, side) != 0:
                    neighbor = field[i + 1, j, k]
                    neighbor_valid = 1
        elif ti.static(axis == 1):
            if ti.static(side < 0):
                if self._muscl_axis_neighbor_is_open(i, j, k, axis, side) != 0:
                    neighbor = field[i, j - 1, k]
                    neighbor_valid = 1
            else:
                if self._muscl_axis_neighbor_is_open(i, j, k, axis, side) != 0:
                    neighbor = field[i, j + 1, k]
                    neighbor_valid = 1
        else:
            if ti.static(side < 0):
                if self._muscl_axis_neighbor_is_open(i, j, k, axis, side) != 0:
                    neighbor = field[i, j, k - 1]
                    neighbor_valid = 1
            else:
                if self._muscl_axis_neighbor_is_open(i, j, k, axis, side) != 0:
                    neighbor = field[i, j, k + 1]
                    neighbor_valid = 1
        value = center
        for component in ti.static(range(3)):
            # Packed backward-MAC components do not share one collocated
            # reconstruction coordinate.  The component normal to ``axis``
            # lives at F_axis and reconstructs to the adjacent scalar centres
            # (the faces of its dual control volume).  Transverse components
            # live at C_axis and reconstruct to the primal scalar faces.
            face_position = 0.0
            center_position = 0.0
            if ti.static(axis == 0):
                if ti.static(component == 0):
                    center_position = self.cell_face_x_m[i]
                    face_position = (
                        self.cell_face_x_m[0]
                        if side < 0 and i == 0
                        else self.cell_center_x_m[i - 1 if side < 0 else i]
                    )
                else:
                    center_position = self.cell_center_x_m[i]
                    face_position = self.cell_face_x_m[i if side < 0 else i + 1]
            elif ti.static(axis == 1):
                if ti.static(component == 1):
                    center_position = self.cell_face_y_m[j]
                    face_position = (
                        self.cell_face_y_m[0]
                        if side < 0 and j == 0
                        else self.cell_center_y_m[j - 1 if side < 0 else j]
                    )
                else:
                    center_position = self.cell_center_y_m[j]
                    face_position = self.cell_face_y_m[j if side < 0 else j + 1]
            else:
                if ti.static(component == 2):
                    center_position = self.cell_face_z_m[k]
                    face_position = (
                        self.cell_face_z_m[0]
                        if side < 0 and k == 0
                        else self.cell_center_z_m[k - 1 if side < 0 else k]
                    )
                else:
                    center_position = self.cell_center_z_m[k]
                    face_position = self.cell_face_z_m[k if side < 0 else k + 1]
            distance = face_position - center_position
            value[component] = center[component] + (
                self.muscl_velocity_slope[i, j, k][axis, component] * distance
            )
            if neighbor_valid != 0:
                smooth = self.muscl_velocity_smooth_mask[i, j, k][
                    axis, component
                ] != 0
                if not smooth:
                    value[component] = ti.min(
                        ti.max(
                            value[component],
                            ti.min(center[component], neighbor[component]),
                        ),
                        ti.max(center[component], neighbor[component]),
                    )
                else:
                    extension = (
                        self.muscl_velocity_extension_slope[i, j, k][axis, component]
                        * ti.abs(distance)
                    )
                    value[component] = ti.min(
                        ti.max(
                            value[component],
                            ti.min(center[component], neighbor[component]) - extension,
                        ),
                        ti.max(center[component], neighbor[component]) + extension,
                    )
        return value

    @ti.kernel
    def _synchronize_muscl_minimum_normal_boundary_rows_kernel(
        self,
        source: ti.template(),
        pressure_outlet_zmin: ti.i32,
        velocity_inlet_zmax_mode: ti.i32,
    ):
        """Make the colocated minimum-side MAC normals match physical faces.

        A compact backward-MAC row owns three different faces.  Only row0's
        component normal to a minimum domain side is colocated with that
        physical face; the two tangential components must remain untouched.
        Use the same exterior contract as Q, including default closure and
        the explicit pressure-correctable zmin row.  No maximum-side packed
        row is modified: those rows own internal faces.
        """

        for i, j, k in source:
            value = source[i, j, k]
            if self.obstacle[i, j, k] == 0:
                if i == 0:
                    value.x = self._physical_exterior_normal_velocity(
                        source, 0, 0, i, j, k,
                        pressure_outlet_zmin, velocity_inlet_zmax_mode,
                    )
                if j == 0:
                    value.y = self._physical_exterior_normal_velocity(
                        source, 1, 0, i, j, k,
                        pressure_outlet_zmin, velocity_inlet_zmax_mode,
                    )
                if k == 0:
                    value.z = self._physical_exterior_normal_velocity(
                        source, 2, 0, i, j, k,
                        pressure_outlet_zmin, velocity_inlet_zmax_mode,
                    )
            source[i, j, k] = value

    @ti.kernel
    def _compute_muscl_momentum_dual_geometry_kernel(self):
        for i, j, k in self.velocity:
            for component in ti.static(range(3)):
                minus_fluid = 0
                plus_fluid = 0
                minus_width = 0.0
                plus_width = 0.0
                transverse_area = 0.0
                if ti.static(component == 0):
                    if i > 0:
                        minus_fluid = 1 if self.obstacle[i - 1, j, k] == 0 else 0
                        minus_width = 0.5 * self.cell_width_x_m[i - 1]
                    plus_fluid = 1 if self.obstacle[i, j, k] == 0 else 0
                    plus_width = 0.5 * self.cell_width_x_m[i]
                    transverse_area = (
                        self.cell_width_y_m[j] * self.cell_width_z_m[k]
                    )
                elif ti.static(component == 1):
                    if j > 0:
                        minus_fluid = 1 if self.obstacle[i, j - 1, k] == 0 else 0
                        minus_width = 0.5 * self.cell_width_y_m[j - 1]
                    plus_fluid = 1 if self.obstacle[i, j, k] == 0 else 0
                    plus_width = 0.5 * self.cell_width_y_m[j]
                    transverse_area = (
                        self.cell_width_x_m[i] * self.cell_width_z_m[k]
                    )
                else:
                    if k > 0:
                        minus_fluid = 1 if self.obstacle[i, j, k - 1] == 0 else 0
                        minus_width = 0.5 * self.cell_width_z_m[k - 1]
                    plus_fluid = 1 if self.obstacle[i, j, k] == 0 else 0
                    plus_width = 0.5 * self.cell_width_z_m[k]
                    transverse_area = (
                        self.cell_width_x_m[i] * self.cell_width_y_m[j]
                    )

                self.muscl_momentum_dual_volume_m3[i, j, k, component] = (
                    minus_width * ti.cast(minus_fluid, ti.f32)
                    + plus_width * ti.cast(plus_fluid, ti.f32)
                ) * transverse_area

    @ti.kernel
    def _compute_muscl_primal_normal_velocity_x_kernel(
        self,
        source: ti.template(),
        pressure_outlet_zmin: ti.i32,
        velocity_inlet_zmax_mode: ti.i32,
    ):
        for face, j, k in self.muscl_normal_velocity_x:
            normal_velocity = 0.0
            if face == 0:
                normal_velocity = self._physical_exterior_normal_velocity(
                    source, 0, 0, 0, j, k,
                    pressure_outlet_zmin, velocity_inlet_zmax_mode,
                )
            elif face == self.nx:
                normal_velocity = self._physical_exterior_normal_velocity(
                    source, 0, 1, self.nx - 1, j, k,
                    pressure_outlet_zmin, velocity_inlet_zmax_mode,
                )
            else:
                minus_fluid = self.obstacle[face - 1, j, k] == 0
                plus_fluid = self.obstacle[face, j, k] == 0
                canonical_wall = (
                    self._canonical_exact_component_owner(
                        face, j, k, 0
                    )
                    != 0
                )
                if minus_fluid and plus_fluid and not canonical_wall:
                    normal_velocity = source[face, j, k].x
                # Any immersed primal face is impermeable in the wall-relative
                # transport registry.  In particular, never admit a stale raw A
                # or B owner value into Q: an exact moving-wall target also has
                # zero relative normal volume flux.
            self.muscl_normal_velocity_x[face, j, k] = normal_velocity

    @ti.kernel
    def _compute_muscl_primal_normal_velocity_y_kernel(
        self,
        source: ti.template(),
        pressure_outlet_zmin: ti.i32,
        velocity_inlet_zmax_mode: ti.i32,
    ):
        for i, face, k in self.muscl_normal_velocity_y:
            normal_velocity = 0.0
            if face == 0:
                normal_velocity = self._physical_exterior_normal_velocity(
                    source, 1, 0, i, 0, k,
                    pressure_outlet_zmin, velocity_inlet_zmax_mode,
                )
            elif face == self.ny:
                normal_velocity = self._physical_exterior_normal_velocity(
                    source, 1, 1, i, self.ny - 1, k,
                    pressure_outlet_zmin, velocity_inlet_zmax_mode,
                )
            else:
                minus_fluid = self.obstacle[i, face - 1, k] == 0
                plus_fluid = self.obstacle[i, face, k] == 0
                canonical_wall = (
                    self._canonical_exact_component_owner(
                        i, face, k, 1
                    )
                    != 0
                )
                if minus_fluid and plus_fluid and not canonical_wall:
                    normal_velocity = source[i, face, k].y
            self.muscl_normal_velocity_y[i, face, k] = normal_velocity

    @ti.kernel
    def _compute_muscl_primal_normal_velocity_z_kernel(
        self,
        source: ti.template(),
        pressure_outlet_zmin: ti.i32,
        velocity_inlet_zmax_mode: ti.i32,
    ):
        for i, j, face in self.muscl_normal_velocity_z:
            normal_velocity = 0.0
            if face == 0:
                normal_velocity = self._physical_exterior_normal_velocity(
                    source, 2, 0, i, j, 0,
                    pressure_outlet_zmin, velocity_inlet_zmax_mode,
                )
            elif face == self.nz:
                normal_velocity = self._physical_exterior_normal_velocity(
                    source, 2, 1, i, j, self.nz - 1,
                    pressure_outlet_zmin, velocity_inlet_zmax_mode,
                )
            else:
                minus_fluid = self.obstacle[i, j, face - 1] == 0
                plus_fluid = self.obstacle[i, j, face] == 0
                canonical_wall = (
                    self._canonical_exact_component_owner(
                        i, j, face, 2
                    )
                    != 0
                )
                if minus_fluid and plus_fluid and not canonical_wall:
                    normal_velocity = source[i, j, face].z
            self.muscl_normal_velocity_z[i, j, face] = normal_velocity

    @ti.kernel
    def _compute_muscl_momentum_flux_x_kernel(
        self,
        source: ti.template(),
        no_slip_xmin: ti.i32,
        no_slip_xmax: ti.i32,
        pressure_outlet_zmin: ti.i32,
        velocity_inlet_zmax_mode: ti.i32,
    ):
        for face, j, k in self.muscl_momentum_flux_x:
            left = ti.Vector([0.0, 0.0, 0.0])
            right = ti.Vector([0.0, 0.0, 0.0])
            if face == 0:
                right = self._muscl_vector_face_state(source, 0, j, k, 0, -1)
                left = right
                if no_slip_xmin != 0:
                    left = ti.Vector([0.0, 0.0, 0.0])
                mask = self.external_velocity_boundary_x_face_active_component_mask[
                    0, j, k
                ]
                target = self.external_velocity_boundary_x_face_value_mps[0, j, k]
                for component in ti.static(range(3)):
                    if (mask & (1 << component)) != 0:
                        left[component] = target[component]
                left.x = self._physical_exterior_normal_velocity(
                    source, 0, 0, 0, j, k,
                    pressure_outlet_zmin, velocity_inlet_zmax_mode,
                )
            elif face == self.nx:
                left = self._muscl_vector_face_state(
                    source, self.nx - 1, j, k, 0, 1
                )
                right = left
                if no_slip_xmax != 0:
                    right = ti.Vector([0.0, 0.0, 0.0])
                mask = self.external_velocity_boundary_x_face_active_component_mask[
                    1, j, k
                ]
                target = self.external_velocity_boundary_x_face_value_mps[1, j, k]
                for component in ti.static(range(3)):
                    if (mask & (1 << component)) != 0:
                        right[component] = target[component]
                right.x = self._physical_exterior_normal_velocity(
                    source, 0, 1, self.nx - 1, j, k,
                    pressure_outlet_zmin, velocity_inlet_zmax_mode,
                )
            else:
                left = self._muscl_vector_face_state(
                    source, face - 1, j, k, 0, 1
                )
                right = self._muscl_vector_face_state(
                    source, face, j, k, 0, -1
                )

            flux = ti.Vector([0.0, 0.0, 0.0])
            for component in ti.static(range(3)):
                q_minus = 0.0
                q_plus = 0.0
                if ti.static(component == 0):
                    area = self.cell_width_y_m[j] * self.cell_width_z_m[k]
                    if face == 0:
                        q_minus = self.muscl_normal_velocity_x[0, j, k] * area
                    elif self.obstacle[face - 1, j, k] == 0:
                        q_minus = 0.5 * area * (
                            self.muscl_normal_velocity_x[face - 1, j, k]
                            + self.muscl_normal_velocity_x[face, j, k]
                        )
                elif ti.static(component == 1):
                    if j > 0:
                        q_minus = (
                            0.5
                            * self.cell_width_y_m[j - 1]
                            * self.cell_width_z_m[k]
                            * self.muscl_normal_velocity_x[face, j - 1, k]
                        )
                    q_plus = (
                        0.5
                        * self.cell_width_y_m[j]
                        * self.cell_width_z_m[k]
                        * self.muscl_normal_velocity_x[face, j, k]
                    )
                else:
                    if k > 0:
                        q_minus = (
                            0.5
                            * self.cell_width_z_m[k - 1]
                            * self.cell_width_y_m[j]
                            * self.muscl_normal_velocity_x[face, j, k - 1]
                        )
                    q_plus = (
                        0.5
                        * self.cell_width_z_m[k]
                        * self.cell_width_y_m[j]
                        * self.muscl_normal_velocity_x[face, j, k]
                    )
                self.muscl_momentum_volume_flux_x_half_m3_s[
                    face, j, k, component, 0
                ] = q_minus
                self.muscl_momentum_volume_flux_x_half_m3_s[
                    face, j, k, component, 1
                ] = q_plus
                flux[component] = q_minus * (
                    left[component] if q_minus >= 0.0 else right[component]
                ) + q_plus * (
                    left[component] if q_plus >= 0.0 else right[component]
                )
            self.muscl_momentum_flux_x[face, j, k] = flux

    @ti.kernel
    def _compute_muscl_momentum_flux_y_kernel(
        self,
        source: ti.template(),
        no_slip_ymin: ti.i32,
        no_slip_ymax: ti.i32,
        pressure_outlet_zmin: ti.i32,
        velocity_inlet_zmax_mode: ti.i32,
    ):
        for i, face, k in self.muscl_momentum_flux_y:
            left = ti.Vector([0.0, 0.0, 0.0])
            right = ti.Vector([0.0, 0.0, 0.0])
            if face == 0:
                right = self._muscl_vector_face_state(source, i, 0, k, 1, -1)
                left = right
                if no_slip_ymin != 0:
                    left = ti.Vector([0.0, 0.0, 0.0])
                mask = self.external_velocity_boundary_y_face_active_component_mask[
                    0, i, k
                ]
                target = self.external_velocity_boundary_y_face_value_mps[0, i, k]
                for component in ti.static(range(3)):
                    if (mask & (1 << component)) != 0:
                        left[component] = target[component]
                left.y = self._physical_exterior_normal_velocity(
                    source, 1, 0, i, 0, k,
                    pressure_outlet_zmin, velocity_inlet_zmax_mode,
                )
            elif face == self.ny:
                left = self._muscl_vector_face_state(
                    source, i, self.ny - 1, k, 1, 1
                )
                right = left
                if no_slip_ymax != 0:
                    right = ti.Vector([0.0, 0.0, 0.0])
                mask = self.external_velocity_boundary_y_face_active_component_mask[
                    1, i, k
                ]
                target = self.external_velocity_boundary_y_face_value_mps[1, i, k]
                for component in ti.static(range(3)):
                    if (mask & (1 << component)) != 0:
                        right[component] = target[component]
                right.y = self._physical_exterior_normal_velocity(
                    source, 1, 1, i, self.ny - 1, k,
                    pressure_outlet_zmin, velocity_inlet_zmax_mode,
                )
            else:
                left = self._muscl_vector_face_state(
                    source, i, face - 1, k, 1, 1
                )
                right = self._muscl_vector_face_state(
                    source, i, face, k, 1, -1
                )

            flux = ti.Vector([0.0, 0.0, 0.0])
            for component in ti.static(range(3)):
                q_minus = 0.0
                q_plus = 0.0
                if ti.static(component == 0):
                    if i > 0:
                        q_minus = (
                            0.5
                            * self.cell_width_x_m[i - 1]
                            * self.cell_width_z_m[k]
                            * self.muscl_normal_velocity_y[i - 1, face, k]
                        )
                    q_plus = (
                        0.5
                        * self.cell_width_x_m[i]
                        * self.cell_width_z_m[k]
                        * self.muscl_normal_velocity_y[i, face, k]
                    )
                elif ti.static(component == 1):
                    area = self.cell_width_x_m[i] * self.cell_width_z_m[k]
                    if face == 0:
                        q_minus = self.muscl_normal_velocity_y[i, 0, k] * area
                    elif self.obstacle[i, face - 1, k] == 0:
                        q_minus = 0.5 * area * (
                            self.muscl_normal_velocity_y[i, face - 1, k]
                            + self.muscl_normal_velocity_y[i, face, k]
                        )
                else:
                    if k > 0:
                        q_minus = (
                            0.5
                            * self.cell_width_z_m[k - 1]
                            * self.cell_width_x_m[i]
                            * self.muscl_normal_velocity_y[i, face, k - 1]
                        )
                    q_plus = (
                        0.5
                        * self.cell_width_z_m[k]
                        * self.cell_width_x_m[i]
                        * self.muscl_normal_velocity_y[i, face, k]
                    )
                self.muscl_momentum_volume_flux_y_half_m3_s[
                    i, face, k, component, 0
                ] = q_minus
                self.muscl_momentum_volume_flux_y_half_m3_s[
                    i, face, k, component, 1
                ] = q_plus
                flux[component] = q_minus * (
                    left[component] if q_minus >= 0.0 else right[component]
                ) + q_plus * (
                    left[component] if q_plus >= 0.0 else right[component]
                )
            self.muscl_momentum_flux_y[i, face, k] = flux

    @ti.kernel
    def _compute_muscl_momentum_flux_z_kernel(
        self,
        source: ti.template(),
        no_slip_zmin: ti.i32,
        no_slip_zmax: ti.i32,
        pressure_outlet_zmin: ti.i32,
        velocity_inlet_zmax_mode: ti.i32,
    ):
        for i, j, face in self.muscl_momentum_flux_z:
            left = ti.Vector([0.0, 0.0, 0.0])
            right = ti.Vector([0.0, 0.0, 0.0])
            if face == 0:
                right = self._muscl_vector_face_state(source, i, j, 0, 2, -1)
                left = right
                if no_slip_zmin != 0:
                    left = ti.Vector([0.0, 0.0, 0.0])
                mask = self.external_velocity_boundary_z_face_active_component_mask[
                    0, i, j
                ]
                target = self.external_velocity_boundary_z_face_value_mps[0, i, j]
                for component in ti.static(range(3)):
                    if (mask & (1 << component)) != 0:
                        left[component] = target[component]
                left.z = self._physical_exterior_normal_velocity(
                    source, 2, 0, i, j, 0,
                    pressure_outlet_zmin, velocity_inlet_zmax_mode,
                )
            elif face == self.nz:
                left = self._muscl_vector_face_state(
                    source, i, j, self.nz - 1, 2, 1
                )
                right = left
                if no_slip_zmax != 0:
                    right = ti.Vector([0.0, 0.0, 0.0])
                mask = self.external_velocity_boundary_z_face_active_component_mask[
                    1, i, j
                ]
                target = self.external_velocity_boundary_z_face_value_mps[1, i, j]
                for component in ti.static(range(3)):
                    if (mask & (1 << component)) != 0:
                        right[component] = target[component]
                right.z = self._physical_exterior_normal_velocity(
                    source, 2, 1, i, j, self.nz - 1,
                    pressure_outlet_zmin, velocity_inlet_zmax_mode,
                )
            else:
                left = self._muscl_vector_face_state(
                    source, i, j, face - 1, 2, 1
                )
                right = self._muscl_vector_face_state(
                    source, i, j, face, 2, -1
                )

            flux = ti.Vector([0.0, 0.0, 0.0])
            for component in ti.static(range(3)):
                q_minus = 0.0
                q_plus = 0.0
                if ti.static(component == 0):
                    if i > 0:
                        q_minus = (
                            0.5
                            * self.cell_width_x_m[i - 1]
                            * self.cell_width_y_m[j]
                            * self.muscl_normal_velocity_z[i - 1, j, face]
                        )
                    q_plus = (
                        0.5
                        * self.cell_width_x_m[i]
                        * self.cell_width_y_m[j]
                        * self.muscl_normal_velocity_z[i, j, face]
                    )
                elif ti.static(component == 1):
                    if j > 0:
                        q_minus = (
                            0.5
                            * self.cell_width_y_m[j - 1]
                            * self.cell_width_x_m[i]
                            * self.muscl_normal_velocity_z[i, j - 1, face]
                        )
                    q_plus = (
                        0.5
                        * self.cell_width_y_m[j]
                        * self.cell_width_x_m[i]
                        * self.muscl_normal_velocity_z[i, j, face]
                    )
                else:
                    area = self.cell_width_x_m[i] * self.cell_width_y_m[j]
                    if face == 0:
                        q_minus = self.muscl_normal_velocity_z[i, j, 0] * area
                    elif self.obstacle[i, j, face - 1] == 0:
                        q_minus = 0.5 * area * (
                            self.muscl_normal_velocity_z[i, j, face - 1]
                            + self.muscl_normal_velocity_z[i, j, face]
                        )
                self.muscl_momentum_volume_flux_z_half_m3_s[
                    i, j, face, component, 0
                ] = q_minus
                self.muscl_momentum_volume_flux_z_half_m3_s[
                    i, j, face, component, 1
                ] = q_plus
                flux[component] = q_minus * (
                    left[component] if q_minus >= 0.0 else right[component]
                ) + q_plus * (
                    left[component] if q_plus >= 0.0 else right[component]
                )
            self.muscl_momentum_flux_z[i, j, face] = flux

    def _compute_muscl_primal_normal_velocity_registry(
        self,
        source: object,
        wall_flag_codes: tuple[int, int, int, int, int, int],
        *,
        pressure_outlet_zmin: bool = False,
        velocity_inlet_zmax: bool | None = None,
    ) -> None:
        """Build the authoritative scalar primal-face relative-Q registry."""
        mode = self._resolve_physical_face_topology(
            pressure_outlet_zmin=pressure_outlet_zmin,
            velocity_inlet_zmax=velocity_inlet_zmax,
            wall_flag_codes=wall_flag_codes,
        )
        outlet = 1 if pressure_outlet_zmin else 0
        self._compute_muscl_primal_normal_velocity_x_kernel(source, outlet, mode)
        self._compute_muscl_primal_normal_velocity_y_kernel(source, outlet, mode)
        self._compute_muscl_primal_normal_velocity_z_kernel(source, outlet, mode)

    def _compute_muscl_momentum_fluxes(
        self,
        source: object,
        wall_flag_codes: tuple[int, int, int, int, int, int],
        *,
        dual_geometry_prepared: bool = False,
        pressure_outlet_zmin: bool = False,
        velocity_inlet_zmax: bool | None = None,
    ) -> None:
        # Reject an inconsistent topology before synchronizing source rows.
        mode = self._resolve_physical_face_topology(
            pressure_outlet_zmin=pressure_outlet_zmin,
            velocity_inlet_zmax=velocity_inlet_zmax,
            wall_flag_codes=wall_flag_codes,
        )
        # The minimum-side normal compact rows are colocated with physical
        # boundary faces.  Synchronize them before slope/state construction so
        # stale packed values cannot enter the first RK flux.  Plus-side faces
        # have no compact owner and remain registry-only by design.
        self._synchronize_muscl_minimum_normal_boundary_rows_kernel(
            source, int(pressure_outlet_zmin), mode,
        )
        # An immersed A/B owner is also a native MAC face state.  Restore its
        # strict target before slopes and upwind states are formed; clamping
        # only after a stage cannot undo momentum already sent to a neighbour.
        self._synchronize_muscl_exact_interface_owner_kernel(
            source,
            self._velocity_dirichlet_boundary_authority_code(),
        )
        # Phase 1: build the one authoritative scalar primal-face relative
        # volume-velocity registry.  SST scalar transport consumes these same
        # face velocities; immersed moving-wall faces are closed in relative
        # coordinates and exact external faces retain their directed target.
        self._compute_muscl_primal_normal_velocity_registry(
            source, wall_flag_codes,
            pressure_outlet_zmin=pressure_outlet_zmin,
            velocity_inlet_zmax=velocity_inlet_zmax,
        )
        if not dual_geometry_prepared:
            self._compute_muscl_momentum_dual_geometry_kernel()
        for axis in range(3):
            self._prepare_muscl_velocity_axis_reconstruction_kernel(source, axis)
        # Phase 2: lift each scalar primal flux into the three component-native
        # MAC dual registries.  Transverse faces preserve both graded half areas
        # and upwind each signed half independently before summation.
        self._compute_muscl_momentum_flux_x_kernel(
            source, wall_flag_codes[0], wall_flag_codes[1],
            int(pressure_outlet_zmin), mode,
        )
        self._compute_muscl_momentum_flux_y_kernel(
            source, wall_flag_codes[2], wall_flag_codes[3],
            int(pressure_outlet_zmin), mode,
        )
        self._compute_muscl_momentum_flux_z_kernel(
            source, wall_flag_codes[4], wall_flag_codes[5],
            int(pressure_outlet_zmin), mode,
        )

    @ti.kernel
    def _muscl_momentum_ssp_stage_device_kernel(
        self,
        source: ti.template(),
        dt_s: ti.f32,
        final_stage: ti.i32,
        canonical_authority: ti.template(),
    ):
        for i, j, k in self.velocity:
            updated = ti.Vector([0.0, 0.0, 0.0])
            for component in ti.static(range(3)):
                minus_fluid = 0
                plus_fluid = 1 if self.obstacle[i, j, k] == 0 else 0
                has_minus_neighbor = 0
                if ti.static(component == 0):
                    if i > 0:
                        has_minus_neighbor = 1
                        minus_fluid = 1 if self.obstacle[i - 1, j, k] == 0 else 0
                elif ti.static(component == 1):
                    if j > 0:
                        has_minus_neighbor = 1
                        minus_fluid = 1 if self.obstacle[i, j - 1, k] == 0 else 0
                else:
                    if k > 0:
                        has_minus_neighbor = 1
                        minus_fluid = 1 if self.obstacle[i, j, k - 1] == 0 else 0

                physical_interface = (
                    has_minus_neighbor != 0 and minus_fluid != plus_fluid
                )
                component_bit = 1 << component
                canonical_exact_owner = 0
                if ti.static(canonical_authority == 1):
                    if (
                        self._canonical_exact_component_owner(
                            i, j, k, component
                        )
                    ) != 0:
                        canonical_exact_owner = 1
                if canonical_exact_owner != 0:
                    updated[component] = (
                        self.velocity_dirichlet_boundary_value_mps[i, j, k][
                            component
                        ]
                    )
                elif physical_interface:
                    # Strict exact owners were synchronized to their target
                    # before flux construction.  A soft active owner, however,
                    # must retain the already-applied source value: the generic
                    # post-stage dispatch deliberately preserves projected soft
                    # rows and cannot recreate a value erased here.  Only a
                    # truly unowned impermeable face is fail-closed to zero.
                    active_owner = 0
                    if ti.static(canonical_authority == 1):
                        if (
                            self.velocity_dirichlet_boundary_active_component_mask[
                                i, j, k
                            ]
                            & component_bit
                        ) != 0:
                            active_owner = 1
                    else:
                        if self.velocity_dirichlet_boundary_active[i, j, k] != 0:
                            active_owner = 1
                    updated[component] = (
                        source[i, j, k][component]
                        if active_owner != 0
                        else 0.0
                    )
                else:
                    dual_volume = self.muscl_momentum_dual_volume_m3[
                        i, j, k, component
                    ]
                    if dual_volume > 1.0e-30:
                        integrated_momentum_divergence = (
                            self.muscl_momentum_flux_x[i + 1, j, k][component]
                            - self.muscl_momentum_flux_x[i, j, k][component]
                            + self.muscl_momentum_flux_y[i, j + 1, k][component]
                            - self.muscl_momentum_flux_y[i, j, k][component]
                            + self.muscl_momentum_flux_z[i, j, k + 1][component]
                            - self.muscl_momentum_flux_z[i, j, k][component]
                        )
                        integrated_volume_divergence = 0.0
                        for half in ti.static(range(2)):
                            integrated_volume_divergence += (
                                self.muscl_momentum_volume_flux_x_half_m3_s[
                                    i + 1, j, k, component, half
                                ]
                                - self.muscl_momentum_volume_flux_x_half_m3_s[
                                    i, j, k, component, half
                                ]
                                + self.muscl_momentum_volume_flux_y_half_m3_s[
                                    i, j + 1, k, component, half
                                ]
                                - self.muscl_momentum_volume_flux_y_half_m3_s[
                                    i, j, k, component, half
                                ]
                                + self.muscl_momentum_volume_flux_z_half_m3_s[
                                    i, j, k + 1, component, half
                                ]
                                - self.muscl_momentum_volume_flux_z_half_m3_s[
                                    i, j, k, component, half
                                ]
                            )
                        # Use exactly the same integrated dual Q for both
                        # div(Q*u) and the finite-continuity correction.  This
                        # keeps a constant component invariant on graded MAC
                        # volumes even when the projected Q registry has residual
                        # divergence.
                        advective_derivative = (
                            integrated_momentum_divergence
                            - source[i, j, k][component]
                            * integrated_volume_divergence
                        ) / dual_volume
                        candidate = (
                            source[i, j, k][component]
                            - dt_s * advective_derivative
                        )
                        if final_stage != 0:
                            updated[component] = 0.5 * (
                                self.velocity_transport_base[i, j, k][component]
                                + candidate
                            )
                        else:
                            updated[component] = candidate
            self.velocity[i, j, k] = updated

    @ti.kernel
    def _synchronize_muscl_exact_interface_owner_kernel(
        self,
        field: ti.template(),
        canonical_authority: ti.template(),
    ):
        for i, j, k in field:
            value = field[i, j, k]
            for component in ti.static(range(3)):
                has_minus_neighbor = 0
                minus_fluid = 0
                if ti.static(component == 0):
                    if i > 0:
                        has_minus_neighbor = 1
                        minus_fluid = 1 if self.obstacle[i - 1, j, k] == 0 else 0
                elif ti.static(component == 1):
                    if j > 0:
                        has_minus_neighbor = 1
                        minus_fluid = 1 if self.obstacle[i, j - 1, k] == 0 else 0
                else:
                    if k > 0:
                        has_minus_neighbor = 1
                        minus_fluid = 1 if self.obstacle[i, j, k - 1] == 0 else 0
                plus_fluid = 1 if self.obstacle[i, j, k] == 0 else 0
                physical_interface = (
                    has_minus_neighbor != 0 and minus_fluid != plus_fluid
                )
                component_bit = 1 << component
                target = self.velocity_dirichlet_boundary_value_mps[i, j, k][
                    component
                ]
                contract_tolerance = ti.cast(1.0e-6, ti.f32)
                target_is_finite = not (
                    ti.math.isnan(target) or ti.math.isinf(target)
                )
                exact_owner = 0
                if ti.static(canonical_authority == 1):
                    if (
                        self._canonical_exact_component_owner(
                            i, j, k, component
                        )
                    ) != 0:
                        exact_owner = 1
                    pressure_mobility = (
                        self.velocity_dirichlet_boundary_pressure_mobility[
                            i, j, k
                        ][component]
                    )
                    enforcement_weight = (
                        self.velocity_dirichlet_boundary_component_enforcement_weight[
                            i, j, k
                        ][component]
                    )
                    mobility_is_closed = (
                        not ti.math.isnan(pressure_mobility)
                        and not ti.math.isinf(pressure_mobility)
                        and ti.abs(pressure_mobility) <= contract_tolerance
                    )
                    enforcement_is_exact = (
                        not ti.math.isnan(enforcement_weight)
                        and not ti.math.isinf(enforcement_weight)
                        and ti.abs(enforcement_weight - 1.0) <= contract_tolerance
                    )
                    if (
                        physical_interface
                        and (
                            self.velocity_dirichlet_boundary_active_component_mask[
                                i, j, k
                            ]
                            & component_bit
                        )
                        != 0
                        and (
                            self.velocity_dirichlet_boundary_hard_fixed_component_mask[
                                i, j, k
                            ]
                            & component_bit
                        )
                        != 0
                        and (
                            self.velocity_dirichlet_boundary_owned_component_mask[
                                i, j, k
                            ]
                            & component_bit
                        )
                        != 0
                        and (
                            self.velocity_dirichlet_boundary_external_exact_component_mask[
                                i, j, k
                            ]
                            & component_bit
                        )
                        == 0
                        and mobility_is_closed
                        and enforcement_is_exact
                        and target_is_finite
                    ):
                        exact_owner = 1
                else:
                    enforcement_weight = (
                        self.velocity_dirichlet_boundary_enforcement_weight[i, j, k]
                    )
                    enforcement_is_exact = (
                        not ti.math.isnan(enforcement_weight)
                        and not ti.math.isinf(enforcement_weight)
                        and ti.abs(enforcement_weight - 1.0) <= contract_tolerance
                    )
                    if (
                        physical_interface
                        and self.velocity_dirichlet_boundary_active[i, j, k] != 0
                        and (
                            self.velocity_dirichlet_boundary_hard_fixed_component_mask[
                                i, j, k
                            ]
                            & component_bit
                        )
                        != 0
                        and self.velocity_dirichlet_boundary_owned_row[i, j, k] != 0
                        and enforcement_is_exact
                        and target_is_finite
                    ):
                        exact_owner = 1
                if exact_owner != 0:
                    # Canonical owner orientation is independent of the scalar
                    # cell phase.  A B owner is stored on the obstacle row and
                    # therefore must be synchronized before reconstruction as
                    # well as restored after the generic stage.
                    value[component] = target
            field[i, j, k] = value

    def _muscl_momentum_ssp_stage_kernel(
        self,
        source: object,
        dt_s: float,
        final_stage: int,
        wall_flag_codes: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
        *,
        pressure_outlet_zmin: bool = False,
        velocity_inlet_zmax: bool | None = None,
    ) -> None:
        """Launch one dual-CV stage with compile-time boundary authority."""

        mode = self._resolve_physical_face_topology(
            pressure_outlet_zmin=pressure_outlet_zmin,
            velocity_inlet_zmax=velocity_inlet_zmax,
            wall_flag_codes=wall_flag_codes,
        )
        self._muscl_momentum_ssp_stage_device_kernel(
            source,
            float(dt_s),
            int(final_stage),
            self._velocity_dirichlet_boundary_authority_code(),
        )
        # Restore the physical minimum-side normals before the next RK source
        # copy/rate evaluation.  This complements the pre-flux synchronization:
        # either one alone leaves one stage exposed to stale packed data.
        self._synchronize_muscl_minimum_normal_boundary_rows_kernel(
            self.velocity, int(pressure_outlet_zmin), mode,
        )
        self._synchronize_muscl_exact_interface_owner_kernel(
            self.velocity,
            self._velocity_dirichlet_boundary_authority_code()
        )

    @ti.kernel
    def _muscl_momentum_advection_rate_kernel(self) -> ti.f64:
        self.reduction_max[None] = 0.0
        for i, j, k in self.velocity:
            for component in ti.static(range(3)):
                dual_volume = self.muscl_momentum_dual_volume_m3[
                    i, j, k, component
                ]
                minus_fluid = 0
                plus_fluid = 1 if self.obstacle[i, j, k] == 0 else 0
                has_minus_neighbor = 0
                if ti.static(component == 0):
                    if i > 0:
                        has_minus_neighbor = 1
                        minus_fluid = 1 if self.obstacle[i - 1, j, k] == 0 else 0
                elif ti.static(component == 1):
                    if j > 0:
                        has_minus_neighbor = 1
                        minus_fluid = 1 if self.obstacle[i, j - 1, k] == 0 else 0
                else:
                    if k > 0:
                        has_minus_neighbor = 1
                        minus_fluid = 1 if self.obstacle[i, j, k - 1] == 0 else 0
                physical_interface = (
                    has_minus_neighbor != 0 and minus_fluid != plus_fluid
                )
                canonical_exact_owner = (
                    self._canonical_exact_component_owner(
                        i, j, k, component
                    )
                ) != 0
                if (
                    dual_volume > 1.0e-30
                    and not physical_interface
                    and not canonical_exact_owner
                ):
                    x_out = 0.0
                    x_in = 0.0
                    y_out = 0.0
                    y_in = 0.0
                    z_out = 0.0
                    z_in = 0.0
                    for half in ti.static(range(2)):
                        x_minus = self.muscl_momentum_volume_flux_x_half_m3_s[
                            i, j, k, component, half
                        ]
                        x_plus = self.muscl_momentum_volume_flux_x_half_m3_s[
                            i + 1, j, k, component, half
                        ]
                        y_minus = self.muscl_momentum_volume_flux_y_half_m3_s[
                            i, j, k, component, half
                        ]
                        y_plus = self.muscl_momentum_volume_flux_y_half_m3_s[
                            i, j + 1, k, component, half
                        ]
                        z_minus = self.muscl_momentum_volume_flux_z_half_m3_s[
                            i, j, k, component, half
                        ]
                        z_plus = self.muscl_momentum_volume_flux_z_half_m3_s[
                            i, j, k + 1, component, half
                        ]
                        # Never collapse signed halves before the CFL bound:
                        # counterflow can have zero net Q while carrying both
                        # inward and outward characteristics at finite speed.
                        x_out += ti.max(-x_minus, 0.0) + ti.max(x_plus, 0.0)
                        x_in += ti.max(x_minus, 0.0) + ti.max(-x_plus, 0.0)
                        y_out += ti.max(-y_minus, 0.0) + ti.max(y_plus, 0.0)
                        y_in += ti.max(y_minus, 0.0) + ti.max(-y_plus, 0.0)
                        z_out += ti.max(-z_minus, 0.0) + ti.max(z_plus, 0.0)
                        z_in += ti.max(z_minus, 0.0) + ti.max(-z_plus, 0.0)
                    local_rate = (
                        ti.max(x_out, x_in)
                        + ti.max(y_out, y_in)
                        + ti.max(z_out, z_in)
                    ) / dual_volume
                    ti.atomic_max(
                        self.reduction_max[None], ti.cast(local_rate, ti.f64)
                    )
        return self.reduction_max[None]

    def _muscl_momentum_advection_rate_s(self) -> float:
        """Return the current physical-face transport rate to the host controller."""

        return float(self._muscl_momentum_advection_rate_kernel())

    def _muscl_invalid_rate_diagnostics(self) -> str:
        """Describe a failed transport rate without adding normal-path transfers."""

        def _array_summary(name: str, values: np.ndarray) -> str:
            array = np.asarray(values)
            finite_mask = np.isfinite(array)
            nonfinite_count = int(array.size - np.count_nonzero(finite_mask))
            if np.any(finite_mask):
                finite_abs = np.where(finite_mask, np.abs(array), -1.0)
                flat_index = int(np.argmax(finite_abs))
                maximum_index = tuple(
                    int(index) for index in np.unravel_index(flat_index, array.shape)
                )
                maximum_abs = float(finite_abs.reshape(-1)[flat_index])
            else:
                maximum_index = ()
                maximum_abs = float("nan")
            first_nonfinite_index: tuple[int, ...] = ()
            if nonfinite_count:
                first_nonfinite_index = tuple(
                    int(index)
                    for index in np.argwhere(~finite_mask)[0]
                )
            return (
                f"{name}_nonfinite_count={nonfinite_count}, "
                f"{name}_max_abs={maximum_abs:g}, "
                f"{name}_max_abs_index={maximum_index}, "
                f"{name}_first_nonfinite_index={first_nonfinite_index}"
            )

        return "; ".join(
            (
                _array_summary("velocity", self.velocity.to_numpy()),
                _array_summary(
                    "normal_face_x", self.muscl_normal_velocity_x.to_numpy()
                ),
                _array_summary(
                    "normal_face_y", self.muscl_normal_velocity_y.to_numpy()
                ),
                _array_summary(
                    "normal_face_z", self.muscl_normal_velocity_z.to_numpy()
                ),
            )
        )
