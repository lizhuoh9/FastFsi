from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral


def minimum_admissible_physical_substep_s(*, requested_time_s: float) -> float:
    """Return the shared floor below which a physical slice must fail closed."""

    requested = float(requested_time_s)
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("requested_time_s must be finite and positive")
    return max(requested * 1.0e-12, 1.0e-15)


def require_zero_remaining_physical_time_s(
    *,
    remaining_time_s: float,
    component: str,
) -> float:
    """Reject a loop that exited before consuming its requested physical time."""

    remaining = float(remaining_time_s)
    if not math.isfinite(remaining) or remaining != 0.0:
        raise FloatingPointError(
            f"{component} left unadvanced physical time before accepted ledger "
            f"reconstruction: remaining_time_s={remaining:g}"
        )
    return remaining


def physical_time_roundoff_tolerance_s(
    *,
    requested_time_s: float,
    accepted_time_s: float,
    accepted_substep_count: int,
) -> float:
    """Bound accepted-time ledger roundoff without accepting a missing slice."""

    requested = float(requested_time_s)
    accepted = float(accepted_time_s)
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("requested_time_s must be finite and positive")
    if not math.isfinite(accepted) or accepted < 0.0:
        raise ValueError("accepted_time_s must be finite and non-negative")
    if isinstance(accepted_substep_count, bool) or not isinstance(
        accepted_substep_count,
        Integral,
    ):
        raise TypeError("accepted_substep_count must be a non-boolean integer")
    operation_count = int(accepted_substep_count)
    if operation_count <= 0:
        raise ValueError("accepted_substep_count must be positive")

    # The adaptive ledgers perform at most one rounded remaining-time
    # subtraction per accepted slice.  Reserve two more ULPs for the final
    # compensated sum and closure subtraction.  Unlike a fixed ULP multiplier,
    # this bound therefore grows with the operations that can create the tail.
    ulp_scale_s = max(math.ulp(requested), math.ulp(accepted))
    operation_roundoff_bound_s = (
        0.5 * float(operation_count) + 2.0
    ) * ulp_scale_s

    # Both adaptive fluid loops refuse physical slices at or below this
    # existing representability floor.  Keep the accounting tolerance below
    # half that floor, so even the smallest admissible omitted slice remains a
    # fail-closed error rather than being reclassified as roundoff.
    minimum_admissible_substep_s = minimum_admissible_physical_substep_s(
        requested_time_s=requested
    )
    missing_substep_guard_s = 0.5 * min(
        requested,
        minimum_admissible_substep_s,
    )
    return min(operation_roundoff_bound_s, missing_substep_guard_s)


@dataclass(frozen=True)
class CflSubstepController:
    """Choose substeps from previously computed CFL diagnostics."""

    base_substeps: int = 1
    target_cfl: float = 0.25
    max_substeps: int = 16
    growth_safety: float = 1.25

    def __post_init__(self) -> None:
        for name, value in (
            ("base_substeps", self.base_substeps),
            ("max_substeps", self.max_substeps),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive non-boolean integer")
            object.__setattr__(self, name, int(value))
        if self.max_substeps < self.base_substeps:
            raise ValueError("max_substeps must be >= base_substeps")
        if not math.isfinite(self.target_cfl) or self.target_cfl <= 0.0:
            raise ValueError("target_cfl must be finite and positive")
        if not math.isfinite(self.growth_safety) or self.growth_safety < 1.0:
            raise ValueError("growth_safety must be finite and >= 1")

    def substeps_for_next_step(
        self,
        *,
        previous_cfl: float | None,
        previous_substeps: int | None = None,
    ) -> int:
        if previous_cfl is None:
            return self.base_substeps
        cfl = float(previous_cfl)
        if not math.isfinite(cfl):
            # A NaN/Inf CFL diagnostic means the upstream velocity field has
            # already diverged. The pre-2026-07 behavior lumped this with the
            # "no load" branch below and fell back to the MINIMUM substep
            # count -- the unsafe direction: it feeds a blown-up state
            # forward with the least stabilization. Physics-first: refuse to
            # guess a substep count from a non-physical diagnostic.
            raise ValueError(
                f"previous_cfl is non-finite ({cfl!r}): the upstream fluid "
                "state has diverged; refusing to choose a substep count from "
                "a non-physical CFL diagnostic (the old fallback silently "
                "continued at minimum substeps)"
            )
        if cfl < 0.0:
            raise ValueError("previous_cfl must be non-negative")
        if cfl == 0.0:
            return self.base_substeps
        if previous_substeps is not None and (
            isinstance(previous_substeps, bool)
            or not isinstance(previous_substeps, Integral)
            or previous_substeps <= 0
        ):
            raise ValueError(
                "previous_substeps must be a positive non-boolean integer"
            )
        reference_substeps = max(
            self.base_substeps,
            self.base_substeps
            if previous_substeps is None
            else int(previous_substeps),
        )
        requested = math.ceil(
            reference_substeps * cfl / self.target_cfl * self.growth_safety
        )
        return min(self.max_substeps, max(self.base_substeps, int(requested)))
