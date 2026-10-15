"""Outlet reachability: the breadth-first marking of cells connected to the outlet, unreached pockets (mixin)."""

import contextlib

import taichi as ti

from fastfsi.fluid.common import COMPONENT_LABEL_BATCH_SWEEPS, REACHABILITY_FLOOD_BATCH_LEVELS, _i32_int
from fastfsi.fluid.constants import UNREACHED_COMPONENT_SMALL_THRESHOLD_CELLS
from fastfsi.errors import SimulationHardStop

SINGLE_LEVEL_BATCH = 0         # outlet-reachability levels per launch (< 2: the single-level path)


@contextlib.contextmanager
def single_level_reachability_flood():
    """Scoped: the outlet-reachability marking takes its single-level path (one breadth-first level per launch).
    The search and the reachable set are the same; only the number of levels per launch differs (the unrolled
    kernel takes minutes and more than 16 GB of host memory to compile on CUDA)."""
    global REACHABILITY_FLOOD_BATCH_LEVELS
    saved = REACHABILITY_FLOOD_BATCH_LEVELS
    REACHABILITY_FLOOD_BATCH_LEVELS = SINGLE_LEVEL_BATCH
    try:
        yield
    finally:
        REACHABILITY_FLOOD_BATCH_LEVELS = saved


class OutletReachability:
    """Mixin of IncompressibleFluidSolver; not used on its own.  Contract (solver attributes):
      reads:  _reachability_checksum, pressure_outlet_reachable, pressure_outlet_reachable_next,
    pressure_reachability_barrier, pressure_reachability_frontier_a,
    pressure_reachability_frontier_a_count, pressure_reachability_frontier_b,
    pressure_reachability_frontier_b_count, pressure_reachability_frontier_capacity,
    pressure_reachability_frontier_levels, pressure_reachability_frontier_overflow,
    pressure_unreached_component_label, reachability_checksum_snapshot, reachability_revision,
    unreached_component_compact_raw_label, unreached_raw_component_cell_count,
    last_pressure_component_labels_converged, last_pressure_unreached_component_overflow,
    last_pressure_unreached_component_small_threshold_cells, last_reachability_revision,
    last_reachability_valid, nx, ny, nz, obstacle, reduction_count,
    report_pressure_disconnected_nonprojectable_cells, report_pressure_disconnected_unreached_raw_cells,
    report_unreached_component_largest_cell_count, report_unreached_component_raw_count,
    report_unreached_component_singleton_count, report_unreached_component_small_cell_count,
    report_unreached_component_small_count, velocity_dirichlet_boundary_active_component_mask,
    velocity_dirichlet_boundary_authority, velocity_dirichlet_boundary_authority_code_device,
    velocity_dirichlet_boundary_external_exact_component_mask,
    velocity_dirichlet_boundary_hard_fixed_component_mask, velocity_dirichlet_boundary_owned_component_mask,
    velocity_dirichlet_boundary_pressure_mobility, velocity_dirichlet_pressure_hard_fixed_component_mask
      writes: _pressure_unreached_component_count, _pressure_unreached_count, _reachability_checksum,
    reachability_revision, last_pressure_component_labels_converged,
    last_pressure_reachability_converged, last_pressure_reachability_frontier_overflow,
    last_pressure_reachability_reused, last_pressure_reachability_sweeps,
    last_pressure_unreached_cell_count, last_pressure_unreached_component_count,
    last_pressure_unreached_component_largest_cell_count, last_pressure_unreached_component_overflow,
    last_pressure_unreached_component_raw_count, last_pressure_unreached_component_singleton_count,
    last_pressure_unreached_component_small_cell_count, last_pressure_unreached_component_small_count,
    last_pressure_unreached_component_small_threshold_cells, last_pressure_unreached_raw_cell_count,
    last_reachability_revision, last_reachability_valid, last_row_cloud_orphan_cell_count,
    last_row_cloud_orphan_component_count, last_unreached_cells_with_interface_coupling,
    last_unreached_cells_with_interface_diagonal, last_unreached_components_with_interface_hits"""

    @ti.kernel
    def _clear_pressure_reachability_barrier_kernel(self):
        for i, j, k in self.pressure_reachability_barrier:
            self.pressure_reachability_barrier[i, j, k] = 0

    def _prepare_pressure_reachability_barrier(
        self,
        node_kind_code,
        *,
        barrier_node_code: int,
    ) -> None:
        if node_kind_code is None:
            self._clear_pressure_reachability_barrier_kernel()
            return
        raise NotImplementedError("a reachability barrier from node kinds (node_kind_code) is not supported")

    @ti.func
    def _pressure_face_has_zero_mobility(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        component_bit: ti.i32,
    ):
        zero_mobility = (
            self.velocity_dirichlet_pressure_hard_fixed_component_mask[i, j, k]
            & component_bit
        ) != 0
        canonical_active = False
        if self.velocity_dirichlet_boundary_authority_code_device[None] != 0:
            canonical_active = (
                self.velocity_dirichlet_boundary_active_component_mask[i, j, k]
                & component_bit
            ) != 0
        if canonical_active:
            component_axis = 0
            if component_bit == 2:
                component_axis = 1
            elif component_bit == 4:
                component_axis = 2
            raw_hard = (
                self.velocity_dirichlet_boundary_hard_fixed_component_mask[i, j, k]
                & component_bit
            ) != 0
            mobility = self.velocity_dirichlet_boundary_pressure_mobility[
                i, j, k
            ][component_axis]
            if (
                raw_hard
                or ti.math.isnan(mobility)
                or ti.math.isinf(mobility)
                or mobility <= 0.0
            ):
                zero_mobility = True
        return zero_mobility

    @ti.func
    def _pressure_cells_connected(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        ni: ti.i32,
        nj: ti.i32,
        nk: ti.i32,
    ):
        connected = False
        if ni == i - 1 and nj == j and nk == k:
            connected = not self._pressure_face_has_zero_mobility(i, j, k, 1)
        elif ni == i + 1 and nj == j and nk == k:
            connected = not self._pressure_face_has_zero_mobility(ni, nj, nk, 1)
        elif nj == j - 1 and ni == i and nk == k:
            connected = not self._pressure_face_has_zero_mobility(i, j, k, 2)
        elif nj == j + 1 and ni == i and nk == k:
            connected = not self._pressure_face_has_zero_mobility(ni, nj, nk, 2)
        elif nk == k - 1 and ni == i and nj == j:
            connected = not self._pressure_face_has_zero_mobility(i, j, k, 4)
        elif nk == k + 1 and ni == i and nj == j:
            connected = not self._pressure_face_has_zero_mobility(ni, nj, nk, 4)
        return connected

    @ti.kernel
    def _reachability_pattern_checksum_kernel(self):
        self.reachability_checksum_snapshot[None] = ti.Vector(
            [0, 0, 0, 0]
        )
        for i, j, k in self.obstacle:
            linear = ti.cast((i * self.ny + j) * self.nz + k + 1, ti.i64)
            hard_mask = ti.cast(0, ti.i64)
            if self._pressure_face_has_zero_mobility(i, j, k, 1):
                hard_mask |= ti.cast(1, ti.i64)
            if self._pressure_face_has_zero_mobility(i, j, k, 2):
                hard_mask |= ti.cast(2, ti.i64)
            if self._pressure_face_has_zero_mobility(i, j, k, 4):
                hard_mask |= ti.cast(4, ti.i64)
            authority_code = ti.cast(
                self.velocity_dirichlet_boundary_authority_code_device[None],
                ti.i64,
            )
            canonical_token = ti.cast(0, ti.i64)
            if authority_code != 0:
                active_mask = ti.cast(
                    self.velocity_dirichlet_boundary_active_component_mask[i, j, k]
                    & 7,
                    ti.i64,
                )
                raw_hard_mask = ti.cast(
                    self.velocity_dirichlet_boundary_hard_fixed_component_mask[
                        i, j, k
                    ]
                    & 7,
                    ti.i64,
                )
                external_mask = ti.cast(
                    self.velocity_dirichlet_boundary_external_exact_component_mask[
                        i, j, k
                    ]
                    & 7,
                    ti.i64,
                )
                owned_mask = ti.cast(
                    self.velocity_dirichlet_boundary_owned_component_mask[i, j, k]
                    & 7,
                    ti.i64,
                )
                mobility = self.velocity_dirichlet_boundary_pressure_mobility[
                    i, j, k
                ]
                mobility_x_bits = ti.cast(
                    ti.bit_cast(mobility.x, ti.i32),
                    ti.i64,
                )
                mobility_y_bits = ti.cast(
                    ti.bit_cast(mobility.y, ti.i32),
                    ti.i64,
                )
                mobility_z_bits = ti.cast(
                    ti.bit_cast(mobility.z, ti.i32),
                    ti.i64,
                )
                mobility_x_fold = (
                    mobility_x_bits ^ (mobility_x_bits >> 16)
                ) & 65535
                mobility_y_fold = (
                    mobility_y_bits ^ (mobility_y_bits >> 16)
                ) & 65535
                mobility_z_fold = (
                    mobility_z_bits ^ (mobility_z_bits >> 16)
                ) & 65535
                canonical_token = (
                    active_mask * 13
                    + raw_hard_mask * 17
                    + external_mask * 19
                    + owned_mask * 23
                    + mobility_x_fold * 29
                    + mobility_y_fold * 31
                    + mobility_z_fold * 37
                )
            obstacle = ti.cast(self.obstacle[i, j, k], ti.i64)
            barrier = ti.cast(
                self.pressure_reachability_barrier[i, j, k],
                ti.i64,
            )
            token = (
                hard_mask * 3
                + obstacle * 5
                + barrier * 7
                + authority_code * 11
                + canonical_token
            )
            ti.atomic_add(self.reachability_checksum_snapshot[None][0], token)
            ti.atomic_add(
                self.reachability_checksum_snapshot[None][1],
                token * linear,
            )
            ti.atomic_add(
                self.reachability_checksum_snapshot[None][2],
                token * (linear * linear + 3 * linear + 7),
            )
            ti.atomic_add(
                self.reachability_checksum_snapshot[None][3],
                token * token + linear * 104729,
            )

    def _reachability_pattern_checksum(self) -> tuple[int, int, int, int]:
        # Keep the device mirror honest even if an internal test deliberately
        # bypasses the public authority setter; the checksum must still expose
        # that identity change and fail closed.
        self.velocity_dirichlet_boundary_authority_code_device[None] = (
            self._velocity_dirichlet_boundary_authority_code()
        )
        self._reachability_pattern_checksum_kernel()
        # One transfer of the four words (indexing syncs once per word).
        checksum = self.reachability_checksum_snapshot.to_numpy()
        return tuple(int(checksum[index]) for index in range(4))

    def _record_pressure_reachability_identity(
        self,
        pattern_checksum: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int]:
        """Record the exact device topology consumed by a completed flood."""

        if pattern_checksum is None:
            pattern_checksum = self._reachability_pattern_checksum()
        recorded = tuple(int(value) for value in pattern_checksum)
        if len(recorded) != 4:
            raise RuntimeError("interface reachability topology identity must have four words")
        self._reachability_checksum = recorded
        return recorded

    def _require_prepared_pressure_reachability_current(self) -> None:
        """Fail closed unless a prepared flood matches the current device topology."""

        if not bool(self.last_reachability_valid):
            raise RuntimeError(
                "prepared interface pressure reachability is not valid for projection"
            )
        if int(self.last_reachability_revision) != int(
            self.reachability_revision
        ):
            self._invalidate_pressure_reachability()
            raise RuntimeError(
                "prepared interface pressure reachability revision mismatch"
            )
        expected = self._reachability_checksum
        if expected is None:
            self._invalidate_pressure_reachability()
            raise RuntimeError(
                "prepared interface pressure reachability has no topology identity"
            )
        current = self._reachability_pattern_checksum()
        if current != expected:
            self._invalidate_pressure_reachability()
            raise RuntimeError(
                "prepared interface pressure reachability topology identity mismatch"
            )

    @ti.kernel
    def _init_pressure_outlet_reachable_frontier_kernel(
        self,
        frontier_capacity: ti.i32,
    ):
        self.pressure_reachability_frontier_a_count[None] = 0
        self.pressure_reachability_frontier_b_count[None] = 0
        self.pressure_reachability_frontier_overflow[None] = 0
        for i, j, k in self.obstacle:
            self.pressure_outlet_reachable[i, j, k] = 0
            self.pressure_outlet_reachable_next[i, j, k] = 0
            if (
                self.obstacle[i, j, k] == 0
                and self.pressure_reachability_barrier[i, j, k] == 0
                and k == 0
                and not self._pressure_face_has_zero_mobility(i, j, k, 4)
            ):
                self.pressure_outlet_reachable[i, j, k] = 1
                self.pressure_outlet_reachable_next[i, j, k] = 1
                slot = ti.atomic_add(
                    self.pressure_reachability_frontier_a_count[None],
                    1,
                )
                if slot < frontier_capacity:
                    self.pressure_reachability_frontier_a[slot] = ti.Vector(
                        [i, j, k]
                    )
                else:
                    ti.atomic_or(
                        self.pressure_reachability_frontier_overflow[None],
                        1,
                    )

    @ti.kernel
    def _expand_pressure_outlet_reachable_frontier_kernel(
        self,
        current_frontier: ti.template(),
        next_frontier: ti.template(),
        next_frontier_count: ti.template(),
        current_count: ti.i32,
        frontier_capacity: ti.i32,
    ):
        for slot in range(current_count):
            self._expand_pressure_outlet_reachable_frontier_cell(
                current_frontier[slot],
                next_frontier,
                next_frontier_count,
                frontier_capacity,
            )

    @ti.func
    def _expand_pressure_outlet_reachable_frontier_cell(
        self,
        cell,
        next_frontier: ti.template(),
        next_frontier_count: ti.template(),
        frontier_capacity: ti.i32,
    ):
        """Expand one frontier cell into the next level (shared by both floods)."""
        i = cell[0]
        j = cell[1]
        k = cell[2]
        for direction in ti.static(range(6)):
            ni = i
            nj = j
            nk = k
            if ti.static(direction == 0):
                ni -= 1
            elif ti.static(direction == 1):
                ni += 1
            elif ti.static(direction == 2):
                nj -= 1
            elif ti.static(direction == 3):
                nj += 1
            elif ti.static(direction == 4):
                nk -= 1
            else:
                nk += 1
            if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                if (
                    self.obstacle[ni, nj, nk] == 0
                    and self.pressure_reachability_barrier[ni, nj, nk] == 0
                    and self._pressure_cells_connected(i, j, k, ni, nj, nk)
                ):
                    was_reachable = ti.atomic_or(
                        self.pressure_outlet_reachable[ni, nj, nk],
                        1,
                    )
                    if was_reachable == 0:
                        self.pressure_outlet_reachable_next[ni, nj, nk] = 1
                        next_slot = ti.atomic_add(next_frontier_count[None], 1)
                        if next_slot < frontier_capacity:
                            next_frontier[next_slot] = ti.Vector([ni, nj, nk])
                        else:
                            ti.atomic_or(
                                self.pressure_reachability_frontier_overflow[
                                    None
                                ],
                                1,
                            )

    @ti.kernel
    def _expand_pressure_outlet_reachable_frontier_level_pair_kernel(
        self,
        frontier_capacity: ti.i32,
    ):
        """Two consecutive BFS levels (a -> b, then b -> a) driven by device counts.

        The host enqueues several pairs back to back and reads the frontier
        count once per batch.  Each level resets its target count and
        increments the level counter exactly when its source frontier is
        non-empty (so the count equals the host-driven loop); kernel tasks run
        in order, so the second level reads the count the first one wrote.
        """
        self.pressure_reachability_frontier_b_count[None] = 0
        if self.pressure_reachability_frontier_a_count[None] > 0:
            self.pressure_reachability_frontier_levels[None] += 1
        for slot in range(self.pressure_reachability_frontier_a_count[None]):
            self._expand_pressure_outlet_reachable_frontier_cell(
                self.pressure_reachability_frontier_a[slot],
                self.pressure_reachability_frontier_b,
                self.pressure_reachability_frontier_b_count,
                frontier_capacity,
            )
        self.pressure_reachability_frontier_a_count[None] = 0
        if self.pressure_reachability_frontier_b_count[None] > 0:
            self.pressure_reachability_frontier_levels[None] += 1
        for slot in range(self.pressure_reachability_frontier_b_count[None]):
            self._expand_pressure_outlet_reachable_frontier_cell(
                self.pressure_reachability_frontier_b[slot],
                self.pressure_reachability_frontier_a,
                self.pressure_reachability_frontier_a_count,
                frontier_capacity,
            )

    def _fail_pressure_reachability_frontier_overflow(self) -> None:
        self.last_pressure_reachability_frontier_overflow = True
        self.last_pressure_reachability_converged = False
        self._reachability_checksum = None
        self._invalidate_pressure_reachability()
        raise RuntimeError(
            "interface pressure reachability frontier overflow; "
            "the outlet-connected set is not certified"
        )

    def _flood_pressure_outlet_reachable_frontier(self) -> int:
        self.velocity_dirichlet_boundary_authority_code_device[None] = (
            self._velocity_dirichlet_boundary_authority_code()
        )
        capacity = int(self.pressure_reachability_frontier_capacity)
        self.last_pressure_reachability_frontier_overflow = False
        self._init_pressure_outlet_reachable_frontier_kernel(capacity)
        if int(self.pressure_reachability_frontier_overflow[None]) != 0:
            self._fail_pressure_reachability_frontier_overflow()

        current_frontier = self.pressure_reachability_frontier_a
        next_frontier = self.pressure_reachability_frontier_b
        current_count_field = self.pressure_reachability_frontier_a_count
        next_count_field = self.pressure_reachability_frontier_b_count
        levels_field = self.pressure_reachability_frontier_levels
        levels_field[None] = 0
        batch_levels = int(REACHABILITY_FLOOD_BATCH_LEVELS)
        current_count = int(current_count_field[None])
        frontier_levels = 0
        host_passes = 0
        def fail_progress(reason):
            evidence = dict(stage="pressure_reachability", reason=reason,
                host_passes=host_passes, capacity=capacity, frontier_levels=frontier_levels,
                current_count=current_count, cells=[
                    tuple(int(v) for v in current_frontier[slot])
                    for slot in range(min(max(current_count, 0), capacity, 8))])
            self.last_pressure_reachability_converged = False
            self._reachability_checksum = None
            self._invalidate_pressure_reachability()
            error = RuntimeError("S30_NONTERMINATION " + repr(evidence))
            error.termination_witness = evidence
            raise error

        while current_count > 0:
            # A nonempty BFS level consumes at least one of the finite cells.
            # Count on the host as well: a stuck device level counter must not
            # make the batched branch loop forever.
            host_passes += 1
            if host_passes > capacity or not 0 < current_count <= capacity:
                fail_progress("frontier_progress_bound")
            if (
                batch_levels >= 2
                and current_frontier is self.pressure_reachability_frontier_a
                and frontier_levels + batch_levels <= capacity
            ):
                # Device batch: ``batch_levels`` levels enqueued back to back
                # with device-side counts, so overflow, the exact level count
                # and the frontier size are read once per batch instead of
                # once per level.  Parity is even, so the current frontier is
                # ``a`` again afterwards.
                for _level in range(batch_levels // 2):
                    self._expand_pressure_outlet_reachable_frontier_level_pair_kernel(
                        capacity,
                    )
                if int(self.pressure_reachability_frontier_overflow[None]) != 0:
                    self._fail_pressure_reachability_frontier_overflow()
                previous_levels = frontier_levels
                frontier_levels = int(levels_field[None])
                current_count = int(current_count_field[None])
                if not previous_levels < frontier_levels <= previous_levels + batch_levels:
                    fail_progress("device_frontier_level_did_not_advance_within_batch")
                continue
            # Single-level path (also the exact fail-closed tail near capacity).
            if frontier_levels >= capacity:
                self._fail_pressure_reachability_frontier_overflow()
            frontier_levels += 1
            next_count_field[None] = 0
            self._expand_pressure_outlet_reachable_frontier_kernel(
                current_frontier,
                next_frontier,
                next_count_field,
                current_count,
                capacity,
            )
            if int(self.pressure_reachability_frontier_overflow[None]) != 0:
                self._fail_pressure_reachability_frontier_overflow()
            next_count = int(next_count_field[None])
            current_frontier, next_frontier = next_frontier, current_frontier
            current_count_field, next_count_field = (
                next_count_field,
                current_count_field,
            )
            current_count = next_count
        self.last_pressure_reachability_sweeps = int(frontier_levels)
        return int(frontier_levels)

    @ti.kernel
    def _count_pressure_outlet_unreached_cells_kernel(self):
        self.report_pressure_disconnected_unreached_raw_cells[None] = 0
        for i, j, k in self.obstacle:
            if (
                self.obstacle[i, j, k] == 0
                and self.pressure_reachability_barrier[i, j, k] == 0
                and self.pressure_outlet_reachable[i, j, k] == 0
            ):
                ti.atomic_add(
                    self.report_pressure_disconnected_unreached_raw_cells[None],
                    1,
                )
        self.report_pressure_disconnected_nonprojectable_cells[None] = (
            self.report_pressure_disconnected_unreached_raw_cells[None]
        )

    def _reset_pressure_unreached_component_distribution_stats(self) -> None:
        self.last_pressure_unreached_component_raw_count = 0
        self.last_pressure_unreached_component_largest_cell_count = 0
        self.last_pressure_unreached_component_singleton_count = 0
        self.last_pressure_unreached_component_small_threshold_cells = (
            UNREACHED_COMPONENT_SMALL_THRESHOLD_CELLS
        )
        self.last_pressure_unreached_component_small_count = 0
        self.last_pressure_unreached_component_small_cell_count = 0

    def _invalidate_pressure_reachability(self) -> None:
        self._invalidate_pressure_nullspace_component_graph()
        if bool(self.last_reachability_valid):
            self.reachability_revision += 1
        self.last_reachability_valid = False
        self.last_reachability_revision = int(self.reachability_revision)
        self._reachability_checksum = None

    @ti.kernel
    def _count_no_free_face_unreached_cells_kernel(self):
        self.no_free_face_stats[0] = 0
        self.no_free_face_stats[1] = 2147483647
        for i, j, k in self.obstacle:
            if self.obstacle[i, j, k] == 0 and self.pressure_reachability_barrier[i, j, k] == 0:
                if self.pressure_outlet_reachable[i, j, k] == 0:
                    free = 0
                    if i > 0:
                        if self.obstacle[i - 1, j, k] == 0 and self._pressure_cells_connected(i, j, k, i - 1, j, k):
                            free += 1
                    if i < self.nx - 1:
                        if self.obstacle[i + 1, j, k] == 0 and self._pressure_cells_connected(i, j, k, i + 1, j, k):
                            free += 1
                    if j > 0:
                        if self.obstacle[i, j - 1, k] == 0 and self._pressure_cells_connected(i, j, k, i, j - 1, k):
                            free += 1
                    if j < self.ny - 1:
                        if self.obstacle[i, j + 1, k] == 0 and self._pressure_cells_connected(i, j, k, i, j + 1, k):
                            free += 1
                    if k > 0:
                        if self.obstacle[i, j, k - 1] == 0 and self._pressure_cells_connected(i, j, k, i, j, k - 1):
                            free += 1
                    if k < self.nz - 1:
                        if self.obstacle[i, j, k + 1] == 0 and self._pressure_cells_connected(i, j, k, i, j, k + 1):
                            free += 1
                    coupled = 0
                    if self.pressure_interface_matrix_diagonal[i, j, k] > 0.0:
                        coupled = 1
                    if self.pressure_interface_coupling_active[i, j, k] != 0:
                        coupled = 1
                    if free == 0 and coupled == 0:
                        ti.atomic_add(self.no_free_face_stats[0], 1)
                        ti.atomic_min(self.no_free_face_stats[1], (i * self.ny + j) * self.nz + k)

    def _check_no_free_face_unreached_cells(self):
        """Safeguard of the removed no-free-face rule: an outlet-unreached water cell with no free
        face and no interface term would be a singular pressure row, which the removed rule zeroed; it is now a hard
        stop with the count and the first cell (one device read, only when unreached cells exist)."""
        self._count_no_free_face_unreached_cells_kernel()
        stats = self.no_free_face_stats.to_numpy()
        self.no_free_face_checks += 1
        if int(stats[0]) > 0:
            first = int(stats[1])
            cell = [first // (self.ny * self.nz), (first // self.nz) % self.ny, first % self.nz]
            raise SimulationHardStop("fluid_isolated_cell_without_free_face", dict(cells=int(stats[0]), first_cell=cell))

    def _mark_pressure_reachability_valid(
        self,
        *,
        fresh_flood: bool,
        valid: bool = True,
    ) -> None:
        if bool(fresh_flood):
            self.reachability_revision += 1
        self.last_reachability_revision = int(self.reachability_revision)
        self.last_reachability_valid = bool(valid)

    def mark_pressure_outlet_disconnected_nonprojectable_cells(
        self,
        *,
        pressure_outlet_zmin: bool = False,
        node_kind_code=None,
        barrier_node_code: int = 0,
        use_existing_reachability_barrier: bool = False,
    ) -> int:
        """Compute z-min outlet pressure reachability and report unreached cells.

        Unreached active fluid cells are NOT converted into obstacle cells and
        their velocity is preserved. The pressure solve anchors the unreached
        set's nullspace through a set-restricted zero-mean projection instead
        of freezing the cells out of the flow solution.
        """
        authority = self._validated_velocity_dirichlet_boundary_authority(
            self.velocity_dirichlet_boundary_authority
        )
        if authority == "canonical":
            # The canonical prepare path refreshes the component-face hard mask,
            # invalidates stale reachability state and registers this generation
            # before the global registry seal.  No barrier, checksum or flood
            # kernel may observe an unsealed/stale canonical registry.
            self._require_velocity_dirichlet_component_registry_sealed()
        if not pressure_outlet_zmin:
            raise NotImplementedError("closed domains without an outlet are not supported yet")
        if not bool(use_existing_reachability_barrier):
            self._prepare_pressure_reachability_barrier(
                node_kind_code,
                barrier_node_code=int(barrier_node_code),
            )
        # Legacy rows have no generation-scoped prepare/seal phase, so retain
        # their historical just-in-time refresh.  Canonical reachability must
        # consume the exact mask certified by its prepare path and global seal.
        if authority == "legacy":
            self._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask()
        # Always recompute reachability. The shared profile rejects the
        # retired checksum-only reuse switch. Record a fresh checksum
        # below for the separate prepared-projection validity audit.
        pattern_checksum: tuple[int, int, int, int] | None = None
        self._reachability_checksum = None
        self.last_pressure_unreached_cell_count = 0
        self.last_pressure_unreached_raw_cell_count = 0
        self.last_pressure_reachability_converged = True
        self.last_pressure_reachability_sweeps = 0
        self.last_pressure_reachability_frontier_overflow = False
        self.last_pressure_reachability_reused = False
        self.last_pressure_component_labels_converged = True
        self.last_unreached_cells_with_interface_diagonal = 0
        self.last_unreached_cells_with_interface_coupling = 0
        self.last_unreached_components_with_interface_hits = 0
        self._pressure_unreached_count = 0
        # The legacy _init_pressure_outlet_reachable_kernel and
        # _expand_and_commit_pressure_outlet_reachable_kernel are retained
        # for narrow compatibility tests only; production uses the sparse
        # frontier flood below and never calls either full-grid expand kernel.
        self._flood_pressure_outlet_reachable_frontier()
        max_sweeps = max(1, 32 * int(self.nx + self.ny + self.nz))
        converged = True
        self.last_pressure_reachability_converged = converged
        self._count_pressure_outlet_unreached_cells_kernel()
        unreached = int(
            self.report_pressure_disconnected_nonprojectable_cells[None]
        )
        self.last_pressure_unreached_cell_count = unreached
        self._pressure_unreached_count = unreached
        self.last_pressure_unreached_raw_cell_count = unreached
        self.last_pressure_unreached_component_count = 0
        self._pressure_unreached_component_count = 0
        self.last_pressure_unreached_component_overflow = False
        self._reset_pressure_unreached_component_distribution_stats()
        if unreached > 0:
            self._check_no_free_face_unreached_cells()
            self._init_unreached_component_labels_kernel()
            labels_converged = False
            label_sweeps_done = 0
            while label_sweeps_done < max_sweeps:
                batch_changed = (
                    self._propagate_unreached_component_labels_batch_kernel()
                )
                label_sweeps_done += COMPONENT_LABEL_BATCH_SWEEPS
                if batch_changed == 0:
                    labels_converged = True
                    break
            # R2-M5: False means the block budget ran out before the label
            # propagation reached a fixed point, so one physical component may
            # still carry several labels (partial merge).
            self.last_pressure_component_labels_converged = labels_converged
            component_count = self._compact_unreached_component_labels_device()
            self.last_pressure_unreached_component_count = component_count
            self._pressure_unreached_component_count = component_count
        self._record_pressure_reachability_identity(pattern_checksum)
        self._mark_pressure_reachability_valid(
            fresh_flood=True,
            valid=bool(converged),
        )
        return unreached

    def convert_row_cloud_orphan_components(
        self,
        *,
        max_component_cells: int,
        overflow_singletons_only: bool = False,
        protect_velocity_dirichlet_radius_cells: int = 0,
        protect_solid_band_mask: bool = False,
        convert_unstamped_small_components: bool = False,
    ) -> int:
        """Convert small disconnected row-cloud components to obstacle cells.

        Outlet-unreachable components are eligible when the component is small
        and its divergence stencil touches a region-stamped velocity
        Dirichlet row. If the previous component labeling overflowed, isolated
        singleton components from the already-computed pressure component labels
        are also cleaned so they cannot exhaust the diagnostic label budget.
        ``overflow_singletons_only`` restricts the cleanup to that latter
        no-row-reload case. ``protect_velocity_dirichlet_radius_cells`` can
        keep singleton cleanup away from no-slip row neighborhoods so residual
        sampling remains measured. ``convert_unstamped_small_components`` is an
        opt-in sharp-interface sliver cleanup for tiny disconnected fragments that
        are not row-stamped; it must be used with a small threshold so larger
        disconnected water bodies remain active for the explicit air-backed or
        pressure compatibility paths.
        """
        protect_radius = _i32_int(
            protect_velocity_dirichlet_radius_cells,
            name="protect_velocity_dirichlet_radius_cells",
        )
        if not -1 <= protect_radius <= 4:
            raise ValueError("protect_velocity_dirichlet_radius_cells must be in [-1, 4]")
        max_cells = int(max_component_cells)
        self.last_row_cloud_orphan_cell_count = 0
        self.last_row_cloud_orphan_component_count = 0
        if max_cells <= 0:
            return 0
        overflow_singleton_cleanup = bool(
            self.last_pressure_unreached_component_overflow
        )
        if bool(overflow_singletons_only) and (
            not overflow_singleton_cleanup
            or not bool(self.last_pressure_component_labels_converged)
        ):
            return 0
        if bool(convert_unstamped_small_components) and not bool(
            self.last_pressure_component_labels_converged
        ):
            return 0
        raise NotImplementedError("turning small disconnected water pockets into obstacles is not supported (tiny_unreached_cleanup_component_cells must be 0; after a pocket-label overflow, increase pressure_component_capacity)")

    @ti.kernel
    def _init_unreached_component_labels_kernel(self):
        for i, j, k in self.obstacle:
            label = 1 << 30
            if (
                self.obstacle[i, j, k] == 0
                and self.pressure_reachability_barrier[i, j, k] == 0
                and self.pressure_outlet_reachable[i, j, k] == 0
            ):
                label = (i * self.ny + j) * self.nz + k
            self.pressure_unreached_component_label[i, j, k] = label

    @ti.func
    def _relax_unreached_component_label_cell(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
    ) -> ti.i32:
        """One min-label relaxation of an unreached cell; returns 1 if lowered."""
        changed = 0
        if (
            self.obstacle[i, j, k] == 0
            and self.pressure_reachability_barrier[i, j, k] == 0
            and self.pressure_outlet_reachable[i, j, k] == 0
        ):
            label = self.pressure_unreached_component_label[i, j, k]
            best = label
            if i > 0:
                if (
                    self.obstacle[i - 1, j, k] == 0
                    and self.pressure_reachability_barrier[i - 1, j, k] == 0
                    and self.pressure_outlet_reachable[i - 1, j, k] == 0
                    and self._pressure_cells_connected(i, j, k, i - 1, j, k)
                ):
                    best = ti.min(
                        best,
                        self.pressure_unreached_component_label[i - 1, j, k],
                    )
            if i < self.nx - 1:
                if (
                    self.obstacle[i + 1, j, k] == 0
                    and self.pressure_reachability_barrier[i + 1, j, k] == 0
                    and self.pressure_outlet_reachable[i + 1, j, k] == 0
                    and self._pressure_cells_connected(i, j, k, i + 1, j, k)
                ):
                    best = ti.min(
                        best,
                        self.pressure_unreached_component_label[i + 1, j, k],
                    )
            if j > 0:
                if (
                    self.obstacle[i, j - 1, k] == 0
                    and self.pressure_reachability_barrier[i, j - 1, k] == 0
                    and self.pressure_outlet_reachable[i, j - 1, k] == 0
                    and self._pressure_cells_connected(i, j, k, i, j - 1, k)
                ):
                    best = ti.min(
                        best,
                        self.pressure_unreached_component_label[i, j - 1, k],
                    )
            if j < self.ny - 1:
                if (
                    self.obstacle[i, j + 1, k] == 0
                    and self.pressure_reachability_barrier[i, j + 1, k] == 0
                    and self.pressure_outlet_reachable[i, j + 1, k] == 0
                    and self._pressure_cells_connected(i, j, k, i, j + 1, k)
                ):
                    best = ti.min(
                        best,
                        self.pressure_unreached_component_label[i, j + 1, k],
                    )
            if k > 0:
                if (
                    self.obstacle[i, j, k - 1] == 0
                    and self.pressure_reachability_barrier[i, j, k - 1] == 0
                    and self.pressure_outlet_reachable[i, j, k - 1] == 0
                    and self._pressure_cells_connected(i, j, k, i, j, k - 1)
                ):
                    best = ti.min(
                        best,
                        self.pressure_unreached_component_label[i, j, k - 1],
                    )
            if k < self.nz - 1:
                if (
                    self.obstacle[i, j, k + 1] == 0
                    and self.pressure_reachability_barrier[i, j, k + 1] == 0
                    and self.pressure_outlet_reachable[i, j, k + 1] == 0
                    and self._pressure_cells_connected(i, j, k, i, j, k + 1)
                ):
                    best = ti.min(
                        best,
                        self.pressure_unreached_component_label[i, j, k + 1],
                    )
            if best < label:
                self.pressure_unreached_component_label[i, j, k] = best
                changed = 1
        return changed

    @ti.kernel
    def _propagate_unreached_component_labels_sweep_kernel(self):
        changed = 0
        for i, j, k in self.obstacle:
            changed += self._relax_unreached_component_label_cell(i, j, k)
        ti.atomic_add(self.reduction_count[None], changed)

    def _propagate_unreached_component_labels_batch_kernel(self) -> int:
        self.reduction_count[None] = 0
        for _ in range(COMPONENT_LABEL_BATCH_SWEEPS):
            self._propagate_unreached_component_labels_sweep_kernel()
        return int(self.reduction_count[None])

    @ti.kernel
    def _accumulate_unreached_raw_component_counts_kernel(self):
        for i, j, k in self.unreached_raw_component_cell_count:
            self.unreached_raw_component_cell_count[i, j, k] = 0
        for component in range(self.pressure_component_capacity):
            self.unreached_component_compact_raw_label[component] = 1 << 30
        for i, j, k in self.obstacle:
            if (
                self.obstacle[i, j, k] == 0
                and self.pressure_reachability_barrier[i, j, k] == 0
                and self.pressure_outlet_reachable[i, j, k] == 0
            ):
                label = self.pressure_unreached_component_label[i, j, k]
                if label >= 0 and label < (1 << 30):
                    seed_i = label // (self.ny * self.nz)
                    remainder = label - seed_i * self.ny * self.nz
                    seed_j = remainder // self.nz
                    seed_k = remainder - seed_j * self.nz
                    if (
                        0 <= seed_i < self.nx
                        and 0 <= seed_j < self.ny
                        and 0 <= seed_k < self.nz
                    ):
                        ti.atomic_add(
                            self.unreached_raw_component_cell_count[
                                seed_i,
                                seed_j,
                                seed_k,
                            ],
                            1,
                        )

    @ti.kernel
    def _scan_unreached_raw_component_counts_kernel(
        self,
        small_threshold_cells: ti.i32,
    ):
        self.report_unreached_component_raw_count[None] = 0
        self.report_unreached_component_largest_cell_count[None] = 0
        self.report_unreached_component_singleton_count[None] = 0
        self.report_unreached_component_small_count[None] = 0
        self.report_unreached_component_small_cell_count[None] = 0
        for i, j, k in self.unreached_raw_component_cell_count:
            cell_count = self.unreached_raw_component_cell_count[i, j, k]
            if cell_count > 0:
                rank = ti.atomic_add(
                    self.report_unreached_component_raw_count[None],
                    1,
                )
                ti.atomic_max(
                    self.report_unreached_component_largest_cell_count[None],
                    cell_count,
                )
                if cell_count == 1:
                    ti.atomic_add(
                        self.report_unreached_component_singleton_count[None],
                        1,
                    )
                if cell_count <= small_threshold_cells:
                    ti.atomic_add(
                        self.report_unreached_component_small_count[None],
                        1,
                    )
                    ti.atomic_add(
                        self.report_unreached_component_small_cell_count[None],
                        cell_count,
                    )
                if rank < self.pressure_component_capacity:
                    self.unreached_component_compact_raw_label[rank] = (
                        (i * self.ny + j) * self.nz + k
                    )

    @ti.kernel
    def _assign_unreached_compact_component_labels_kernel(
        self,
        component_count: ti.i32,
    ):
        for i, j, k in self.obstacle:
            if (
                self.obstacle[i, j, k] == 0
                and self.pressure_reachability_barrier[i, j, k] == 0
                and self.pressure_outlet_reachable[i, j, k] == 0
            ):
                label = self.pressure_unreached_component_label[i, j, k]
                if label >= 0 and label < (1 << 30):
                    compact_label = label
                    for component in range(ti.min(component_count, self.pressure_component_capacity)):
                        if (
                            component < component_count
                            and label
                            == self.unreached_component_compact_raw_label[
                                component
                            ]
                        ):
                            compact_label = -(component + 1)
                    self.pressure_unreached_component_label[i, j, k] = (
                        compact_label
                    )

    def _compact_unreached_component_labels_device(self) -> int:
        threshold = max(
            0,
            int(self.last_pressure_unreached_component_small_threshold_cells),
        )
        self._accumulate_unreached_raw_component_counts_kernel()
        self._scan_unreached_raw_component_counts_kernel(int(threshold))
        raw_component_count = int(self.report_unreached_component_raw_count[None])
        self.last_pressure_unreached_component_raw_count = raw_component_count
        if raw_component_count > 0:
            self.last_pressure_unreached_component_largest_cell_count = int(
                self.report_unreached_component_largest_cell_count[None]
            )
            self.last_pressure_unreached_component_singleton_count = int(
                self.report_unreached_component_singleton_count[None]
            )
            self.last_pressure_unreached_component_small_count = int(
                self.report_unreached_component_small_count[None]
            )
            self.last_pressure_unreached_component_small_cell_count = int(
                self.report_unreached_component_small_cell_count[None]
            )
        else:
            self._reset_pressure_unreached_component_distribution_stats()
        component_count = min(
            raw_component_count,
            self.pressure_component_capacity,
        )
        if component_count > 0:
            self._assign_unreached_compact_component_labels_kernel(
                int(component_count),
            )
        self.last_pressure_unreached_component_overflow = (
            raw_component_count > self.pressure_component_capacity
        )
        return component_count
