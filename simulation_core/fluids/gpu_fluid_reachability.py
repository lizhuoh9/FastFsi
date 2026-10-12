"""CUDA-only copy: original flood16/tail/first-error/progress checks unchanged."""
from .solver import HIBM_REACHABILITY_FLOOD_BATCH_LEVELS, HIBM_REACHABILITY_FLOOD_BATCH_PAIRS

def flood_frontier(self) -> int:
    self.velocity_dirichlet_boundary_authority_code_device[None] = (
        self._velocity_dirichlet_boundary_authority_code()
    )
    capacity = int(self.hibm_pressure_reachability_frontier_capacity)
    self.last_hibm_pressure_reachability_frontier_overflow = False
    self._init_hibm_pressure_outlet_reachable_frontier_kernel(capacity)
    initial_packet = self._h2_fluid_scalars.frontier(
        self, self.hibm_pressure_reachability_frontier_a_count,
        self.hibm_pressure_reachability_frontier_levels)
    if int(initial_packet[0]) != 0:
        self._fail_hibm_pressure_reachability_frontier_overflow()

    current_frontier = self.hibm_pressure_reachability_frontier_a
    next_frontier = self.hibm_pressure_reachability_frontier_b
    current_count_field = self.hibm_pressure_reachability_frontier_a_count
    next_count_field = self.hibm_pressure_reachability_frontier_b_count
    levels_field = self.hibm_pressure_reachability_frontier_levels
    levels_field[None] = 0
    batch_levels = int(HIBM_REACHABILITY_FLOOD_BATCH_LEVELS)
    current_count = int(current_count_field[None])
    frontier_levels = 0
    host_passes = 0
    def fail_progress(reason):
        evidence = dict(stage="pressure_reachability", reason=reason,
            host_passes=host_passes, capacity=capacity, frontier_levels=frontier_levels,
            current_count=current_count, cells=[
                tuple(int(v) for v in current_frontier[slot])
                for slot in range(min(max(current_count, 0), capacity, 8))])
        self.last_hibm_pressure_reachability_converged = False
        self._hibm_reachability_checksum = None
        self._invalidate_hibm_pressure_reachability()
        error = RuntimeError("S30_NONTERMINATION " + repr(evidence))
        error.hibm_termination_witness = evidence
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
            and current_frontier is self.hibm_pressure_reachability_frontier_a
            and frontier_levels + batch_levels <= capacity
        ):
            # Device batch: ``batch_levels`` levels enqueued back to back
            # with device-side counts, so overflow, the exact level count
            # and the frontier size are read once per batch instead of
            # once per level.  Parity is even, so the current frontier is
            # ``a`` again afterwards.
            if batch_levels == 2 * HIBM_REACHABILITY_FLOOD_BATCH_PAIRS:
                # F2(a) (REQ_343): the same level pairs in one launch call (solver.py batch kernel: the
                # pair kernel body unrolled HIBM_REACHABILITY_FLOOD_BATCH_PAIRS times; same tasks, same order).
                self._expand_hibm_pressure_outlet_reachable_frontier_level_batch_kernel(capacity)
            else:
                for _level in range(batch_levels // 2):
                    self._expand_hibm_pressure_outlet_reachable_frontier_level_pair_kernel(
                        capacity,
                    )
            batch_packet = self._h2_fluid_scalars.frontier(self, current_count_field, levels_field)
            if int(batch_packet[0]) != 0:
                self._fail_hibm_pressure_reachability_frontier_overflow()
            previous_levels = frontier_levels
            frontier_levels = int(batch_packet[2])
            current_count = int(batch_packet[1])
            if not previous_levels < frontier_levels <= previous_levels + batch_levels:
                fail_progress("device_frontier_level_did_not_advance_within_batch")
            continue
        # Single-level path (also the exact fail-closed tail near capacity).
        if frontier_levels >= capacity:
            self._fail_hibm_pressure_reachability_frontier_overflow()
        frontier_levels += 1
        next_count_field[None] = 0
        self._expand_hibm_pressure_outlet_reachable_frontier_kernel(
            current_frontier,
            next_frontier,
            next_count_field,
            current_count,
            capacity,
        )
        tail_packet = self._h2_fluid_scalars.frontier(self, next_count_field, levels_field)
        if int(tail_packet[0]) != 0:
            self._fail_hibm_pressure_reachability_frontier_overflow()
        next_count = int(tail_packet[1])
        current_frontier, next_frontier = next_frontier, current_frontier
        current_count_field, next_count_field = (
            next_count_field,
            current_count_field,
        )
        current_count = next_count
    self.last_hibm_pressure_reachability_sweeps = int(frontier_levels)
    return int(frontier_levels)

