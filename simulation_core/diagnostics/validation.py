from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from collections.abc import Sequence
from typing import Any


@dataclass(frozen=True)
class FieldDiagnostic:
    step: Any
    field: str
    value: Any
    reason: str


@dataclass(frozen=True)
class BoundaryDriveComplianceReport:
    prescribed_velocity_boundary: bool
    prescribed_pressure_or_flow_boundary: bool
    nonzero_fluid_traction_scale: float
    compliant: bool


@dataclass(frozen=True)
class ReferenceCurve:
    name: str
    units: str
    points: tuple[tuple[float, float], ...]
    source: str

    def __init__(
        self,
        *,
        name: str,
        units: str,
        points: Sequence[tuple[float, float]],
        source: str,
    ) -> None:
        normalized = tuple(
            (float(time_s), float(value)) for time_s, value in points
        )
        if not name:
            raise ValueError("name must be non-empty")
        if not units:
            raise ValueError("units must be non-empty")
        if not source:
            raise ValueError("source must be non-empty")
        if len(normalized) < 2:
            raise ValueError("reference curve requires at least two points")
        previous_time = -math.inf
        for time_s, value in normalized:
            if not math.isfinite(time_s) or not math.isfinite(value):
                raise ValueError("reference curve points must be finite")
            if time_s <= previous_time:
                raise ValueError("reference curve times must be strictly increasing")
            previous_time = time_s
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "points", normalized)
        object.__setattr__(self, "source", source)

    def value_at(self, time_s: float) -> float:
        query = float(time_s)
        if not math.isfinite(query):
            raise ValueError("time_s must be finite")
        first_time = self.points[0][0]
        last_time = self.points[-1][0]
        if query < first_time or query > last_time:
            raise ValueError("time_s is outside the reference curve range")
        for (left_time, left_value), (right_time, right_value) in zip(
            self.points,
            self.points[1:],
        ):
            if query <= right_time:
                span = right_time - left_time
                fraction = 0.0 if span == 0.0 else (query - left_time) / span
                return left_value + fraction * (right_value - left_value)
        return self.points[-1][1]

    def relative_error_at(self, *, time_s: float, computed_value: float) -> float:
        reference = self.value_at(time_s)
        computed = float(computed_value)
        if not math.isfinite(computed):
            raise ValueError("computed_value must be finite")
        if abs(reference) <= 1.0e-30:
            return abs(computed - reference)
        return abs(computed - reference) / abs(reference)


def vector_norm(values: Any) -> float:
    components = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in components):
        raise ValueError("values must contain only finite numbers")
    return math.hypot(*components)


def finite_field_diagnostics(
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        for field in fields:
            if field not in row:
                diagnostics.append(
                    asdict(
                        FieldDiagnostic(
                            step=row.get("step"),
                            field=field,
                            value=None,
                            reason="missing",
                        )
                    )
                )
                continue
            value = row[field]
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                diagnostics.append(
                    asdict(
                        FieldDiagnostic(
                            step=row.get("step"),
                            field=field,
                            value=value,
                            reason="not_numeric",
                        )
                    )
                )
                continue
            if not math.isfinite(numeric):
                diagnostics.append(
                    asdict(
                        FieldDiagnostic(
                            step=row.get("step"),
                            field=field,
                            value=numeric,
                            reason="nonfinite",
                        )
                    )
                )
    return diagnostics


def force_nonzero_when_loaded(
    *,
    force_components_n: Any,
    load_value: float,
    force_required: bool,
    tolerance_n: float = 0.0,
) -> bool:
    load = float(load_value)
    tolerance = float(tolerance_n)
    if not math.isfinite(load):
        raise ValueError("load_value must be finite")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_n must be a finite non-negative number")
    if not force_required:
        return True
    if load == 0.0:
        return True
    return vector_norm(force_components_n) > tolerance


def boundary_drive_compliance_report(
    *,
    prescribed_velocity_boundary: bool,
    prescribed_pressure_or_flow_boundary: bool,
    nonzero_fluid_traction_scale: float,
    force_scale_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    force_scale = float(nonzero_fluid_traction_scale)
    report = BoundaryDriveComplianceReport(
        prescribed_velocity_boundary=bool(prescribed_velocity_boundary),
        prescribed_pressure_or_flow_boundary=bool(prescribed_pressure_or_flow_boundary),
        nonzero_fluid_traction_scale=force_scale,
        compliant=(
            not bool(prescribed_velocity_boundary)
            and not bool(prescribed_pressure_or_flow_boundary)
            and abs(force_scale) <= float(force_scale_tolerance)
        ),
    )
    return asdict(report)


def checks_passed(checks: dict[str, Any]) -> bool:
    """Require a nonempty set of explicit Python boolean acceptance checks.

    Callers own their required check names. Numeric metrics and nested
    diagnostic objects are not boolean acceptance results.
    """
    return bool(checks) and all(value is True for value in checks.values())
