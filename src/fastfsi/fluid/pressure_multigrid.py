"""The multigrid preconditioner: levels, restriction, prolongation, smoothers (mixin)."""

import math

import taichi as ti

from fastfsi.grid.cartesian_grid import CartesianGrid


class MultigridPreconditioner:
    """Mixin of IncompressibleFluidSolver; not used on its own.  Contract (solver attributes):
      reads:  MG_JACOBI_SWEEP_PAIRS_PER_LAUNCH, BATCH_MG_PROLONGATION,
    _VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES, _mg_axis_widths_m, _mg_cell_width_x_m,
    _mg_cell_width_y_m, _mg_cell_width_z_m, _mg_center_distance_x_m, _mg_center_distance_y_m,
    _mg_center_distance_z_m, _mg_level_factors, _mg_obstacle, _mg_pressure, _mg_pressure_interface_matrix_diagonal,
    _mg_residual, _mg_rhs, _mg_shapes, _mg_tmp, _mg_velocity_dirichlet_boundary_active,
    _mg_velocity_dirichlet_boundary_active_component_mask,
    _mg_velocity_dirichlet_boundary_external_exact_component_mask,
    _mg_velocity_dirichlet_boundary_hard_fixed_component_mask, _mg_velocity_dirichlet_boundary_marker_region_id,
    _mg_velocity_dirichlet_boundary_owned_component_mask, _mg_velocity_dirichlet_boundary_pressure_mobility,
    _mg_velocity_dirichlet_boundary_projection_weight, _mg_velocity_dirichlet_pressure_hard_fixed_component_mask,
    _pcg_mg_pressure, _pcg_mg_residual, _pcg_mg_rhs, _pcg_mg_tmp, bounds_min, divergence, grid, obstacle,
    pressure_interface_matrix_rhs, pressure_interface_projection_divergence_s, reduction_count,
    velocity_dirichlet_boundary_authority, volume_source_s
      writes: _mg_jacobi_coefficients"""

    @staticmethod
    def _build_multigrid_shapes(
        shape: tuple[int, int, int], *, semi_coarsening: bool = False
    ) -> tuple[tuple[int, int, int], ...]:
        shapes = [tuple(int(n) for n in shape)]
        current = shapes[0]
        if not semi_coarsening:
            while min(current) > 3:
                next_shape = tuple(max(2, (n + 1) // 2) for n in current)
                shapes.append(next_shape)
                current = next_shape
            return tuple(shapes)
        # Semi-coarsening: halve every axis that still has >= 4 cells and keep the
        # others, until no axis can be halved or the level is tiny.
        while len(shapes) < 16:
            next_shape = tuple((n + 1) // 2 if n >= 4 else n for n in current)
            if next_shape == current or math.prod(next_shape) <= 16:
                break
            shapes.append(next_shape)
            current = next_shape
        return tuple(shapes)

    @staticmethod
    def _build_multigrid_level_factors(
        shapes: tuple[tuple[int, int, int], ...],
    ) -> tuple[tuple[int, int, int], ...]:
        """Per-axis coarsening factor (1 or 2) of every fine -> coarse transition."""
        factors: list[tuple[int, int, int]] = []
        for fine, coarse in zip(shapes, shapes[1:]):
            level: list[int] = []
            for n_fine, n_coarse in zip(fine, coarse):
                if int(n_coarse) == int(n_fine):
                    level.append(1)
                elif int(n_coarse) in (max(2, (int(n_fine) + 1) // 2), (int(n_fine) + 1) // 2):
                    level.append(2)
                else:
                    raise ValueError("multigrid level shapes are not a 1x/2x coarsening")
            factors.append((level[0], level[1], level[2]))
        return tuple(factors)

    @staticmethod
    def _coarsen_axis_widths(widths: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(
            sum(widths[index : index + 2])
            for index in range(0, len(widths), 2)
        )

    @staticmethod
    def _build_multigrid_axis_widths(
        grid: CartesianGrid,
        shapes: tuple[tuple[int, int, int], ...],
    ) -> tuple[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]], ...]:
        levels: list[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]] = []
        current = (
            grid.cell_widths_x_m,
            grid.cell_widths_y_m,
            grid.cell_widths_z_m,
        )
        for level, shape in enumerate(shapes):
            if tuple(len(axis) for axis in current) != shape:
                raise ValueError("multigrid axis widths do not match multigrid shapes")
            levels.append(current)
            if level + 1 < len(shapes):
                next_shape = shapes[level + 1]
                current = tuple(
                    axis_widths
                    if len(axis_widths) == int(next_shape[axis])
                    else MultigridPreconditioner._coarsen_axis_widths(axis_widths)
                    for axis, axis_widths in enumerate(current)
                )
        return tuple(levels)

    def _load_multigrid_axis_fields(self) -> None:
        for level, axis_widths in enumerate(self._mg_axis_widths_m[1:], start=1):
            widths_x, widths_y, widths_z = axis_widths
            centers_x = CartesianGrid._axis_centers(self.bounds_min[0], widths_x)
            centers_y = CartesianGrid._axis_centers(self.bounds_min[1], widths_y)
            centers_z = CartesianGrid._axis_centers(self.bounds_min[2], widths_z)
            self._load_axis_field(self._mg_cell_width_x_m[level], widths_x)
            self._load_axis_field(self._mg_cell_width_y_m[level], widths_y)
            self._load_axis_field(self._mg_cell_width_z_m[level], widths_z)
            self._load_axis_field(
                self._mg_center_distance_x_m[level],
                CartesianGrid._axis_center_distances(centers_x, widths_x),
            )
            self._load_axis_field(
                self._mg_center_distance_y_m[level],
                CartesianGrid._axis_center_distances(centers_y, widths_y),
            )
            self._load_axis_field(
                self._mg_center_distance_z_m[level],
                CartesianGrid._axis_center_distances(centers_z, widths_z),
            )

    @ti.func
    def _canonical_multigrid_fine_face_index(
        self,
        coarse_i: ti.i32,
        coarse_j: ti.i32,
        coarse_k: ti.i32,
        component_axis: ti.i32,
        transverse_offset_a: ti.i32,
        transverse_offset_b: ti.i32,
        factor_x: ti.i32,
        factor_y: ti.i32,
        factor_z: ti.i32,
    ):
        """Map a coarse MAC face to one coplanar fine subface."""

        fine_i = factor_x * coarse_i
        fine_j = factor_y * coarse_j
        fine_k = factor_z * coarse_k
        if component_axis == 0:
            fine_j += transverse_offset_a
            fine_k += transverse_offset_b
        elif component_axis == 1:
            fine_i += transverse_offset_a
            fine_k += transverse_offset_b
        else:
            fine_i += transverse_offset_a
            fine_j += transverse_offset_b
        return ti.Vector([fine_i, fine_j, fine_k])

    @ti.kernel
    def _build_canonical_velocity_dirichlet_multigrid_level_kernel(
        self,
        fine_active_component_mask: ti.template(),
        fine_pressure_mobility: ti.template(),
        fine_hard_fixed_component_mask: ti.template(),
        fine_owned_component_mask: ti.template(),
        fine_external_exact_component_mask: ti.template(),
        fine_cell_width_x_m: ti.template(),
        fine_cell_width_y_m: ti.template(),
        fine_cell_width_z_m: ti.template(),
        coarse_active_component_mask: ti.template(),
        coarse_pressure_mobility: ti.template(),
        coarse_hard_fixed_component_mask: ti.template(),
        coarse_owned_component_mask: ti.template(),
        coarse_external_exact_component_mask: ti.template(),
        coarse_pressure_hard_fixed_component_mask: ti.template(),
        fine_nx: ti.i32,
        fine_ny: ti.i32,
        fine_nz: ti.i32,
        factor_x: ti.i32,
        factor_y: ti.i32,
        factor_z: ti.i32,
    ):
        for i, j, k in coarse_active_component_mask:
            active_mask = 0
            hard_mask = 0
            owned_mask = 0
            external_mask = 0
            coarse_mobility = ti.Vector([1.0, 1.0, 1.0])
            invalid_contract = 0
            for component_axis in ti.static(range(3)):
                component_bit = 1 << component_axis
                # transverse coarsening factors of this face orientation
                transverse_factor_a = factor_y if component_axis == 0 else factor_x
                transverse_factor_b = factor_z if component_axis != 2 else factor_y
                total_area = ti.cast(0.0, ti.f64)
                weighted_pressure_mobility = ti.cast(0.0, ti.f64)
                valid_count = 0
                active_count = 0
                hard_count = 0
                owned_count = 0
                external_count = 0
                for transverse_offset_a, transverse_offset_b in ti.static(
                    ti.ndrange(2, 2)
                ):
                    fine_index = self._canonical_multigrid_fine_face_index(
                        i,
                        j,
                        k,
                        component_axis,
                        transverse_offset_a,
                        transverse_offset_b,
                        factor_x,
                        factor_y,
                        factor_z,
                    )
                    fine_i = fine_index.x
                    fine_j = fine_index.y
                    fine_k = fine_index.z
                    if (
                        transverse_offset_a < transverse_factor_a
                        and transverse_offset_b < transverse_factor_b
                        and fine_i < fine_nx
                        and fine_j < fine_ny
                        and fine_k < fine_nz
                    ):
                        valid_count += 1
                        area = ti.cast(1.0, ti.f64)
                        if component_axis == 0:
                            area = ti.cast(
                                fine_cell_width_y_m[fine_j], ti.f64
                            ) * ti.cast(fine_cell_width_z_m[fine_k], ti.f64)
                        elif component_axis == 1:
                            area = ti.cast(
                                fine_cell_width_x_m[fine_i], ti.f64
                            ) * ti.cast(fine_cell_width_z_m[fine_k], ti.f64)
                        else:
                            area = ti.cast(
                                fine_cell_width_x_m[fine_i], ti.f64
                            ) * ti.cast(fine_cell_width_y_m[fine_j], ti.f64)
                        if (
                            area <= 0.0
                            or ti.math.isnan(area)
                            or ti.math.isinf(area)
                        ):
                            invalid_contract = 1
                            area = ti.cast(0.0, ti.f64)
                        is_active = (
                            fine_active_component_mask[fine_i, fine_j, fine_k]
                            & component_bit
                        ) != 0
                        is_hard = (
                            fine_hard_fixed_component_mask[
                                fine_i, fine_j, fine_k
                            ]
                            & component_bit
                        ) != 0
                        is_owned = (
                            fine_owned_component_mask[fine_i, fine_j, fine_k]
                            & component_bit
                        ) != 0
                        is_external = (
                            fine_external_exact_component_mask[
                                fine_i, fine_j, fine_k
                            ]
                            & component_bit
                        ) != 0
                        mobility = fine_pressure_mobility[
                            fine_i, fine_j, fine_k
                        ][component_axis]
                        if ti.math.isnan(mobility) or ti.math.isinf(mobility):
                            invalid_contract = 1
                            mobility = 0.0
                        pressure_weight = ti.cast(1.0, ti.f64)
                        if is_active:
                            active_count += 1
                            if is_hard:
                                hard_count += 1
                                pressure_weight = ti.cast(0.0, ti.f64)
                            else:
                                pressure_weight = ti.cast(
                                    ti.min(ti.max(mobility, 0.0), 1.0),
                                    ti.f64,
                                )
                            if is_owned:
                                owned_count += 1
                            if is_external:
                                external_count += 1
                            if is_owned and is_external:
                                invalid_contract = 1
                        elif is_hard or is_owned or is_external:
                            invalid_contract = 1
                        total_area += area
                        weighted_pressure_mobility += area * pressure_weight
                if valid_count <= 0 or total_area <= 0.0:
                    invalid_contract = 1
                else:
                    coarse_mobility[component_axis] = ti.cast(
                        weighted_pressure_mobility / total_area,
                        ti.f32,
                    )
                    if active_count > 0:
                        active_mask |= component_bit
                    if hard_count == valid_count:
                        hard_mask |= component_bit
                    if owned_count == valid_count:
                        owned_mask |= component_bit
                    if external_count == valid_count:
                        external_mask |= component_bit
                    elif external_count > 0:
                        invalid_contract = 1
            if invalid_contract != 0:
                # Canonical masks reserve only bits 0..2.  A sentinel bit lets
                # the later validation pass fail closed without host copies.
                active_mask |= 8
            coarse_active_component_mask[i, j, k] = active_mask
            coarse_pressure_mobility[i, j, k] = coarse_mobility
            coarse_hard_fixed_component_mask[i, j, k] = hard_mask
            coarse_owned_component_mask[i, j, k] = owned_mask
            coarse_external_exact_component_mask[i, j, k] = external_mask
            pressure_hard_mask = hard_mask
            if external_mask != 0:
                pressure_hard_mask = hard_mask & (owned_mask | external_mask)
            coarse_pressure_hard_fixed_component_mask[i, j, k] = (
                pressure_hard_mask
            )

    def _build_canonical_velocity_dirichlet_multigrid_level(
        self,
        fine_level: int,
        coarse_level: int,
    ) -> None:
        if coarse_level != fine_level + 1:
            raise ValueError("canonical multigrid levels must be adjacent")
        fine_shape = self._mg_shapes[fine_level]
        self._build_canonical_velocity_dirichlet_multigrid_level_kernel(
            self._mg_velocity_dirichlet_boundary_active_component_mask[fine_level],
            self._mg_velocity_dirichlet_boundary_pressure_mobility[fine_level],
            self._mg_velocity_dirichlet_boundary_hard_fixed_component_mask[
                fine_level
            ],
            self._mg_velocity_dirichlet_boundary_owned_component_mask[fine_level],
            self._mg_velocity_dirichlet_boundary_external_exact_component_mask[
                fine_level
            ],
            self._mg_cell_width_x_m[fine_level],
            self._mg_cell_width_y_m[fine_level],
            self._mg_cell_width_z_m[fine_level],
            self._mg_velocity_dirichlet_boundary_active_component_mask[coarse_level],
            self._mg_velocity_dirichlet_boundary_pressure_mobility[coarse_level],
            self._mg_velocity_dirichlet_boundary_hard_fixed_component_mask[
                coarse_level
            ],
            self._mg_velocity_dirichlet_boundary_owned_component_mask[coarse_level],
            self._mg_velocity_dirichlet_boundary_external_exact_component_mask[
                coarse_level
            ],
            self._mg_velocity_dirichlet_pressure_hard_fixed_component_mask[
                coarse_level
            ],
            int(fine_shape[0]),
            int(fine_shape[1]),
            int(fine_shape[2]),
            int(self._mg_level_factors[fine_level][0]),
            int(self._mg_level_factors[fine_level][1]),
            int(self._mg_level_factors[fine_level][2]),
        )

    @ti.kernel
    def _validate_canonical_velocity_dirichlet_multigrid_level_kernel(
        self,
        active_component_mask: ti.template(),
        pressure_mobility: ti.template(),
        hard_fixed_component_mask: ti.template(),
        owned_component_mask: ti.template(),
        external_exact_component_mask: ti.template(),
    ):
        self.reduction_count[None] = 0
        for i, j, k in active_component_mask:
            active = active_component_mask[i, j, k]
            hard = hard_fixed_component_mask[i, j, k]
            owned = owned_component_mask[i, j, k]
            external = external_exact_component_mask[i, j, k]
            invalid = (
                ((active | hard | owned | external) & ~7) != 0
                or (hard & ~active) != 0
                or (owned & ~active) != 0
                or (external & ~active) != 0
                or (owned & external) != 0
            )
            for component_axis in ti.static(range(3)):
                mobility = pressure_mobility[i, j, k][component_axis]
                if ti.math.isnan(mobility) or ti.math.isinf(mobility):
                    invalid = True
                if mobility < 0.0 or mobility > 1.0:
                    invalid = True
            if invalid:
                ti.atomic_add(self.reduction_count[None], 1)

    def _validate_canonical_velocity_dirichlet_multigrid_level(
        self,
        level: int,
    ) -> int:
        self._validate_canonical_velocity_dirichlet_multigrid_level_kernel(
            self._mg_velocity_dirichlet_boundary_active_component_mask[level],
            self._mg_velocity_dirichlet_boundary_pressure_mobility[level],
            self._mg_velocity_dirichlet_boundary_hard_fixed_component_mask[level],
            self._mg_velocity_dirichlet_boundary_owned_component_mask[level],
            self._mg_velocity_dirichlet_boundary_external_exact_component_mask[level],
        )
        return int(self.reduction_count[None])

    def prepare_velocity_dirichlet_component_registry_multigrid(self) -> None:
        """Build, validate and register the component-face MG hierarchy."""

        if self.velocity_dirichlet_boundary_authority != "canonical":
            return
        for fine_level in range(len(self._mg_shapes) - 1):
            self._build_canonical_velocity_dirichlet_multigrid_level(
                fine_level,
                fine_level + 1,
            )
        for level in range(len(self._mg_shapes)):
            invalid_count = (
                self._validate_canonical_velocity_dirichlet_multigrid_level(level)
            )
            if invalid_count != 0:
                raise RuntimeError(
                    "canonical velocity Dirichlet multigrid level "
                    f"{level} is invalid: invalid_cells={invalid_count}"
                )
        self._register_velocity_dirichlet_component_registry_consumer_generation(
            "multigrid",
            capability=self._VELOCITY_DIRICHLET_COMPONENT_REGISTRY_CONSUMER_CAPABILITIES[
                "multigrid"
            ],
        )

    @ti.kernel
    def _mg_prepare_jacobi_coefficients_kernel(
        self,
        coefficients: ti.template(),
        pressure_interface_matrix_diagonal: ti.template(),
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
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        pressure_outlet_zmin: ti.i32,
        canonical_authority: ti.template(),
    ):
        """Store the level operator once per solve: six face products + diagonal."""

        for i, j, k in coefficients:
            value = ti.Vector.zero(ti.f64, 7)
            # Same cell test as the smoother/residual kernels (``== 1`` is solid).
            if obstacle[i, j, k] != 1:
                products, denominator = self._fv_pressure_face_products_and_denominator(
                    obstacle,
                    velocity_dirichlet_boundary_active,
                    velocity_dirichlet_boundary_projection_weight,
                    velocity_dirichlet_pressure_hard_fixed_component_mask,
                    canonical_active_component_mask,
                    canonical_pressure_mobility,
                    canonical_hard_fixed_component_mask,
                    cell_width_x_m,
                    cell_width_y_m,
                    cell_width_z_m,
                    center_distance_x_m,
                    center_distance_y_m,
                    center_distance_z_m,
                    i,
                    j,
                    k,
                    nx,
                    ny,
                    nz,
                    pressure_outlet_zmin,
                    canonical_authority,
                )
                denominator += pressure_interface_matrix_diagonal[i, j, k]
                for face in ti.static(range(6)):
                    value[face] = products[face]
                value[6] = denominator
            coefficients[i, j, k] = value

    @ti.func
    def _mg_precomputed_neighbor_sum(
        self,
        pressure: ti.template(),
        obstacle: ti.template(),
        coefficients: ti.template(),
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        # Same faces, order and products as _fv_pressure_neighbor_sum_denominator.
        c = coefficients[i, j, k]
        neighbor_sum = ti.cast(0.0, ti.f64)
        if i > 0 and obstacle[i - 1, j, k] == 0:
            neighbor_sum += c[0] * pressure[i - 1, j, k]
        if i < nx - 1 and obstacle[i + 1, j, k] == 0:
            neighbor_sum += c[1] * pressure[i + 1, j, k]
        if j > 0 and obstacle[i, j - 1, k] == 0:
            neighbor_sum += c[2] * pressure[i, j - 1, k]
        if j < ny - 1 and obstacle[i, j + 1, k] == 0:
            neighbor_sum += c[3] * pressure[i, j + 1, k]
        if k > 0 and obstacle[i, j, k - 1] == 0:
            neighbor_sum += c[4] * pressure[i, j, k - 1]
        if k < nz - 1 and obstacle[i, j, k + 1] == 0:
            neighbor_sum += c[5] * pressure[i, j, k + 1]
        return neighbor_sum, c[6]

    @ti.func
    def _mg_precomputed_jacobi_update(
        self,
        pressure: ti.template(),
        tmp: ti.template(),
        rhs: ti.template(),
        obstacle: ti.template(),
        coefficients: ti.template(),
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        omega: ti.f32,
    ):
        # One cell of _pressure_fv_jacobi_kernel with the stored operator.
        if obstacle[i, j, k] == 1:
            tmp[i, j, k] = 0.0
        else:
            neighbor_sum, denominator = self._mg_precomputed_neighbor_sum(
                pressure, obstacle, coefficients, i, j, k, nx, ny, nz,
            )
            if denominator <= 0.0:
                tmp[i, j, k] = 0.0
            else:
                jacobi_value = (neighbor_sum - rhs[i, j, k]) / denominator
                tmp[i, j, k] = pressure[i, j, k] + omega * (jacobi_value - pressure[i, j, k])

    @ti.kernel
    def _mg_precomputed_jacobi_sweep_kernel(
        self,
        pressure: ti.template(),
        tmp: ti.template(),
        rhs: ti.template(),
        obstacle: ti.template(),
        coefficients: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        omega: ti.f32,
    ):
        for i, j, k in pressure:
            self._mg_precomputed_jacobi_update(
                pressure, tmp, rhs, obstacle, coefficients, i, j, k, nx, ny, nz, omega,
            )

    @ti.kernel
    def _mg_precomputed_jacobi_sweep_pairs_kernel(
        self,
        pressure: ti.template(),
        tmp: ti.template(),
        rhs: ti.template(),
        obstacle: ti.template(),
        coefficients: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        omega: ti.f32,
        sweep_pairs: ti.template(),
    ):
        # Ping-pong pressure -> tmp -> pressure; each sweep reads exactly the
        # iterate the previous one wrote, as the one-sweep launches do.
        for _pair in ti.static(range(sweep_pairs)):
            for i, j, k in pressure:
                self._mg_precomputed_jacobi_update(
                    pressure, tmp, rhs, obstacle, coefficients, i, j, k, nx, ny, nz, omega,
                )
            for i, j, k in pressure:
                self._mg_precomputed_jacobi_update(
                    tmp, pressure, rhs, obstacle, coefficients, i, j, k, nx, ny, nz, omega,
                )

    @ti.kernel
    def _mg_precomputed_residual_kernel(
        self,
        pressure: ti.template(),
        rhs: ti.template(),
        obstacle: ti.template(),
        coefficients: ti.template(),
        residual: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        # _mg_compute_residual_kernel with the stored operator.
        for i, j, k in pressure:
            if obstacle[i, j, k] == 1:
                residual[i, j, k] = 0.0
            else:
                neighbor_sum, denominator = self._mg_precomputed_neighbor_sum(
                    pressure, obstacle, coefficients, i, j, k, nx, ny, nz,
                )
                applied = neighbor_sum - denominator * pressure[i, j, k]
                residual[i, j, k] = rhs[i, j, k] - applied

    @ti.kernel
    def _mg_prepare_level0_kernel(
        self,
        rhs: ti.template(),
        residual: ti.template(),
        rhs_scale: ti.f32,
    ):
        for i, j, k in rhs:
            if self.obstacle[i, j, k] == 1:
                rhs[i, j, k] = 0.0
                residual[i, j, k] = 0.0
            else:
                rhs[i, j, k] = (
                    rhs_scale * (
                        self.divergence[i, j, k]
                        + self.pressure_interface_projection_divergence_s[i, j, k]
                        - self.volume_source_s[i, j, k]
                    )
                    - self.pressure_interface_matrix_rhs[i, j, k]
                )
                residual[i, j, k] = 0.0

    @ti.kernel
    def _mg_clear_level_kernel(
        self,
        pressure: ti.template(),
        tmp: ti.template(),
        rhs: ti.template(),
        residual: ti.template(),
        pressure_interface_matrix_diagonal: ti.template(),
    ):
        for i, j, k in pressure:
            pressure[i, j, k] = 0.0
            tmp[i, j, k] = 0.0
            rhs[i, j, k] = 0.0
            residual[i, j, k] = 0.0
            pressure_interface_matrix_diagonal[i, j, k] = 0.0

    @ti.kernel
    def _mg_restrict_residual_kernel(
        self,
        fine_residual: ti.template(),
        fine_pressure_interface_matrix_diagonal: ti.template(),
        fine_obstacle: ti.template(),
        fine_velocity_dirichlet_boundary_active: ti.template(),
        fine_velocity_dirichlet_boundary_projection_weight: ti.template(),
        fine_cell_width_x_m: ti.template(),
        fine_cell_width_y_m: ti.template(),
        fine_cell_width_z_m: ti.template(),
        coarse_rhs: ti.template(),
        coarse_pressure_interface_matrix_diagonal: ti.template(),
        coarse_pressure: ti.template(),
        coarse_tmp: ti.template(),
        coarse_residual: ti.template(),
        coarse_obstacle: ti.template(),
        coarse_velocity_dirichlet_boundary_active: ti.template(),
        coarse_velocity_dirichlet_boundary_projection_weight: ti.template(),
        coarse_velocity_dirichlet_boundary_marker_region_id: ti.template(),
        coarse_velocity_dirichlet_pressure_hard_fixed_component_mask: ti.template(),
        fine_nx: ti.i32,
        fine_ny: ti.i32,
        fine_nz: ti.i32,
        canonical_authority: ti.template(),
        factor_x: ti.i32,
        factor_y: ti.i32,
        factor_z: ti.i32,
    ):
        for i, j, k in coarse_rhs:
            weighted_residual_sum = ti.cast(0.0, ti.f64)
            weighted_interface_diagonal_sum = ti.cast(0.0, ti.f64)
            free_volume_m3 = ti.cast(0.0, ti.f64)
            obstacle_count = 0
            soft_boundary_active = 0
            soft_boundary_projection_weight = ti.cast(0.0, ti.f32)
            for oi, oj, ok in ti.static(ti.ndrange(2, 2, 2)):
                fi = factor_x * i + oi
                fj = factor_y * j + oj
                fk = factor_z * k + ok
                if (
                    oi < factor_x
                    and oj < factor_y
                    and ok < factor_z
                    and fi < fine_nx
                    and fj < fine_ny
                    and fk < fine_nz
                ):
                    if fine_obstacle[fi, fj, fk] == 1:
                        obstacle_count += 1
                    else:
                        if ti.static(canonical_authority == 0):
                            if (
                                fine_velocity_dirichlet_boundary_active[
                                    fi, fj, fk
                                ]
                                != 0
                            ):
                                soft_boundary_active = 1
                                soft_boundary_projection_weight = ti.max(
                                    soft_boundary_projection_weight,
                                    fine_velocity_dirichlet_boundary_projection_weight[
                                        fi, fj, fk
                                    ],
                                )
                        volume_m3 = (
                            ti.cast(fine_cell_width_x_m[fi], ti.f64)
                            * ti.cast(fine_cell_width_y_m[fj], ti.f64)
                            * ti.cast(fine_cell_width_z_m[fk], ti.f64)
                        )
                        weighted_residual_sum = weighted_residual_sum + (
                            ti.cast(fine_residual[fi, fj, fk], ti.f64) * volume_m3
                        )
                        weighted_interface_diagonal_sum = (
                            weighted_interface_diagonal_sum
                            + ti.cast(
                                fine_pressure_interface_matrix_diagonal[fi, fj, fk],
                                ti.f64,
                            )
                            * volume_m3
                        )
                        free_volume_m3 = free_volume_m3 + volume_m3
            if obstacle_count > 0:
                coarse_obstacle[i, j, k] = 1
                coarse_rhs[i, j, k] = 0.0
                coarse_pressure_interface_matrix_diagonal[i, j, k] = 0.0
            else:
                coarse_obstacle[i, j, k] = 0
                inverse_free_volume_m3 = 1.0 / free_volume_m3
                coarse_rhs[i, j, k] = ti.cast(
                    weighted_residual_sum * inverse_free_volume_m3,
                    ti.f32,
                )
                coarse_pressure_interface_matrix_diagonal[i, j, k] = ti.cast(
                    weighted_interface_diagonal_sum * inverse_free_volume_m3,
                    ti.f32,
                )
            # Legacy scalar metadata is restricted only for legacy authority.
            # Canonical authority consumes the separately prepared coplanar
            # component-face hierarchy and must never use this 8-child OR/max.
            if ti.static(canonical_authority == 0):
                coarse_velocity_dirichlet_boundary_active[
                    i, j, k
                ] = soft_boundary_active
                coarse_velocity_dirichlet_boundary_projection_weight[i, j, k] = (
                    soft_boundary_projection_weight
                )
            else:
                coarse_velocity_dirichlet_boundary_active[i, j, k] = 0
                coarse_velocity_dirichlet_boundary_projection_weight[i, j, k] = 0.0
            coarse_velocity_dirichlet_boundary_marker_region_id[i, j, k] = -1
            if ti.static(canonical_authority == 0):
                coarse_velocity_dirichlet_pressure_hard_fixed_component_mask[
                    i, j, k
                ] = 0
            coarse_pressure[i, j, k] = 0.0
            coarse_tmp[i, j, k] = 0.0
            coarse_residual[i, j, k] = 0.0

    @ti.kernel
    def _mg_prolongate_add_kernel(
        self,
        fine_pressure: ti.template(),
        fine_obstacle: ti.template(),
        coarse_pressure: ti.template(),
        coarse_nx: ti.i32,
        coarse_ny: ti.i32,
        coarse_nz: ti.i32,
        correction_scale: ti.f32,
        factor_x: ti.i32,
        factor_y: ti.i32,
        factor_z: ti.i32,
    ):
        for i, j, k in fine_pressure:
            if fine_obstacle[i, j, k] == 0:
                ci = ti.min(i // factor_x, coarse_nx - 1)
                cj = ti.min(j // factor_y, coarse_ny - 1)
                ck = ti.min(k // factor_z, coarse_nz - 1)
                fine_pressure[i, j, k] += correction_scale * coarse_pressure[ci, cj, ck]

    @ti.kernel
    def _mg_prolongate_and_smooth_pair_kernel(
        self,
        pressure: ti.template(),
        obstacle: ti.template(),
        coarse_pressure: ti.template(),
        tmp: ti.template(),
        rhs: ti.template(),
        coefficients: ti.template(),
        coarse_nx: ti.i32,
        coarse_ny: ti.i32,
        coarse_nz: ti.i32,
        correction_scale: ti.f32,
        factor_x: ti.i32,
        factor_y: ti.i32,
        factor_z: ti.i32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
        omega: ti.f32,
    ):
        # batched Python dispatch only. Three separate top-level loops keep
        # the global barriers and the exact stored f32 iterates, including tmp.
        # No in-place Jacobi fusion and no reduction-order change are allowed.
        for i, j, k in pressure:
            if obstacle[i, j, k] == 0:
                ci = ti.min(i // factor_x, coarse_nx - 1)
                cj = ti.min(j // factor_y, coarse_ny - 1)
                ck = ti.min(k // factor_z, coarse_nz - 1)
                pressure[i, j, k] += correction_scale * coarse_pressure[ci, cj, ck]
        for i, j, k in pressure:
            self._mg_precomputed_jacobi_update(
                pressure, tmp, rhs, obstacle, coefficients, i, j, k, nx, ny, nz, omega,
            )
        for i, j, k in pressure:
            self._mg_precomputed_jacobi_update(
                tmp, pressure, rhs, obstacle, coefficients, i, j, k, nx, ny, nz, omega,
            )

    def _mg_jacobi_coefficient_fields(self) -> list:
        """Per-level stored operator (six face products + diagonal), lazily allocated."""

        fields = getattr(self, "_mg_jacobi_coefficients", None)
        if fields is None:
            fields = [
                ti.Vector.field(7, dtype=ti.f64, shape=level_shape)
                for level_shape in self._mg_shapes
            ]
            self._mg_jacobi_coefficients = fields
        return fields

    def _smooth_fv_pressure_fields_precomputed(
        self,
        *,
        pressure: object,
        tmp: object,
        rhs: object,
        obstacle: object,
        coefficients: object,
        shape: tuple[int, int, int],
        iterations: int,
        omega: float,
    ) -> None:
        """``_smooth_fv_pressure_fields`` on a stored operator, several sweeps per launch.

        Every iterate equals the one-sweep ping-pong: pairs end in
        ``pressure``, and an odd count finishes with one sweep into ``tmp``
        followed by the same copy back.
        """

        nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])
        pairs, odd = divmod(max(0, int(iterations)), 2)
        per_launch = int(self.MG_JACOBI_SWEEP_PAIRS_PER_LAUNCH)
        while pairs > 0:
            launch_pairs = min(pairs, per_launch)
            self._mg_precomputed_jacobi_sweep_pairs_kernel(
                pressure, tmp, rhs, obstacle, coefficients,
                nx, ny, nz, float(omega), launch_pairs,
            )
            pairs -= launch_pairs
        if odd:
            self._mg_precomputed_jacobi_sweep_kernel(
                pressure, tmp, rhs, obstacle, coefficients, nx, ny, nz, float(omega),
            )
            self._copy_scalar_field_kernel(pressure, tmp)

    def _prepare_fv_multigrid_rhs(self, rhs_scale: float) -> None:
        self._mg_prepare_level0_kernel(
            self._mg_rhs[0],
            self._mg_residual[0],
            float(rhs_scale),
        )
        for level in range(1, len(self._mg_shapes)):
            self._mg_clear_level_kernel(
                self._mg_pressure[level],
                self._mg_tmp[level],
                self._mg_rhs[level],
                self._mg_residual[level],
                self._mg_pressure_interface_matrix_diagonal[level],
            )

    def _apply_fv_multigrid_preconditioner(
        self,
        residual: object,
        *,
        pressure_outlet_zmin: bool,
        pre_smooth_iterations: int = 2,
        coarse_smooth_iterations: int = 24,
        post_smooth_iterations: int = 2,
        prepare_level_operators: bool = True,
    ) -> None:
        """One V-cycle on stored per-level operators.

        The smoother and residual use each level's face products and diagonal
        stored by ``_mg_prepare_jacobi_coefficients_kernel`` (term by term the
        values the on-the-fly kernels compute).  They depend only on the
        level masks, which are fixed within one pressure solve (coarse masks
        are rebuilt identically by every restriction), so a solve prepares
        them on its first V-cycle and passes ``prepare_level_operators=False``
        afterwards.
        """

        pre_iterations = max(1, int(pre_smooth_iterations))
        coarse_iterations = max(1, int(coarse_smooth_iterations))
        post_iterations = max(1, int(post_smooth_iterations))
        coefficients = self._mg_jacobi_coefficient_fields()
        outlet = 1 if pressure_outlet_zmin else 0

        def prepare_level_operator(level: int) -> None:
            shape = self._mg_shapes[level]
            self._mg_prepare_jacobi_coefficients_kernel(
                coefficients[level],
                self._mg_pressure_interface_matrix_diagonal[level],
                self._mg_obstacle[level],
                self._mg_velocity_dirichlet_boundary_active[level],
                self._mg_velocity_dirichlet_boundary_projection_weight[level],
                self._mg_velocity_dirichlet_pressure_hard_fixed_component_mask[level],
                self._mg_velocity_dirichlet_boundary_active_component_mask[level],
                self._mg_velocity_dirichlet_boundary_pressure_mobility[level],
                self._mg_velocity_dirichlet_boundary_hard_fixed_component_mask[level],
                self._mg_cell_width_x_m[level],
                self._mg_cell_width_y_m[level],
                self._mg_cell_width_z_m[level],
                self._mg_center_distance_x_m[level],
                self._mg_center_distance_y_m[level],
                self._mg_center_distance_z_m[level],
                int(shape[0]),
                int(shape[1]),
                int(shape[2]),
                outlet,
                self._velocity_dirichlet_boundary_authority_code(),
            )

        def smooth_pcg_level(level: int, iterations: int) -> None:
            self._smooth_fv_pressure_fields_precomputed(
                pressure=self._pcg_mg_pressure[level],
                tmp=self._pcg_mg_tmp[level],
                rhs=self._pcg_mg_rhs[level],
                obstacle=self._mg_obstacle[level],
                coefficients=coefficients[level],
                shape=self._mg_shapes[level],
                iterations=iterations,
                omega=0.8,
            )

        self._pcg_prepare_mg_level0_kernel(
            residual,
            self._pcg_mg_rhs[0],
            self._pcg_mg_pressure[0],
            self._pcg_mg_tmp[0],
            self._pcg_mg_residual[0],
        )
        if prepare_level_operators:
            prepare_level_operator(0)
        level_count = len(self._mg_shapes)
        if level_count <= 1:
            smooth_pcg_level(0, max(pre_iterations + post_iterations, coarse_iterations))
            return

        coarse_correction_scale = 1.0 if self.grid.is_uniform else 0.7
        for level in range(level_count - 1):
            smooth_pcg_level(level, pre_iterations)
            shape = self._mg_shapes[level]
            self._mg_precomputed_residual_kernel(
                self._pcg_mg_pressure[level],
                self._pcg_mg_rhs[level],
                self._mg_obstacle[level],
                coefficients[level],
                self._pcg_mg_residual[level],
                int(shape[0]),
                int(shape[1]),
                int(shape[2]),
            )
            self._mg_restrict_residual_kernel(
                self._pcg_mg_residual[level],
                self._mg_pressure_interface_matrix_diagonal[level],
                self._mg_obstacle[level],
                self._mg_velocity_dirichlet_boundary_active[level],
                self._mg_velocity_dirichlet_boundary_projection_weight[level],
                self._mg_cell_width_x_m[level],
                self._mg_cell_width_y_m[level],
                self._mg_cell_width_z_m[level],
                self._pcg_mg_rhs[level + 1],
                self._mg_pressure_interface_matrix_diagonal[level + 1],
                self._pcg_mg_pressure[level + 1],
                self._pcg_mg_tmp[level + 1],
                self._pcg_mg_residual[level + 1],
                self._mg_obstacle[level + 1],
                self._mg_velocity_dirichlet_boundary_active[level + 1],
                self._mg_velocity_dirichlet_boundary_projection_weight[level + 1],
                self._mg_velocity_dirichlet_boundary_marker_region_id[level + 1],
                self._mg_velocity_dirichlet_pressure_hard_fixed_component_mask[
                    level + 1
                ],
                int(shape[0]),
                int(shape[1]),
                int(shape[2]),
                self._velocity_dirichlet_boundary_authority_code(),
                int(self._mg_level_factors[level][0]),
                int(self._mg_level_factors[level][1]),
                int(self._mg_level_factors[level][2]),
            )
            if prepare_level_operators:
                # The restriction has just (re)built this level's masks.
                prepare_level_operator(level + 1)

        smooth_pcg_level(level_count - 1, coarse_iterations)
        for level in range(level_count - 2, -1, -1):
            coarse_shape = self._mg_shapes[level + 1]
            if self.BATCH_MG_PROLONGATION and post_iterations == 2:
                # The production fv_multigrid path has exactly two post
                # sweeps. Other counts retain their original batching/layout.
                shape = self._mg_shapes[level]
                self._mg_prolongate_and_smooth_pair_kernel(
                    self._pcg_mg_pressure[level],
                    self._mg_obstacle[level],
                    self._pcg_mg_pressure[level + 1],
                    self._pcg_mg_tmp[level],
                    self._pcg_mg_rhs[level],
                    coefficients[level],
                    int(coarse_shape[0]), int(coarse_shape[1]), int(coarse_shape[2]),
                    float(coarse_correction_scale),
                    int(self._mg_level_factors[level][0]),
                    int(self._mg_level_factors[level][1]),
                    int(self._mg_level_factors[level][2]),
                    int(shape[0]), int(shape[1]), int(shape[2]), 0.8,
                )
                continue
            self._mg_prolongate_add_kernel(
                self._pcg_mg_pressure[level],
                self._mg_obstacle[level],
                self._pcg_mg_pressure[level + 1],
                int(coarse_shape[0]),
                int(coarse_shape[1]),
                int(coarse_shape[2]),
                float(coarse_correction_scale),
                int(self._mg_level_factors[level][0]),
                int(self._mg_level_factors[level][1]),
                int(self._mg_level_factors[level][2]),
            )
            smooth_pcg_level(level, post_iterations)
