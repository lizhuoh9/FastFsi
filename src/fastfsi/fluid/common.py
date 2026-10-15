"""Module-level helpers and constants shared by the fluid mixins."""

from collections.abc import Mapping
from collections.abc import Sequence
from numbers import Integral
from typing import NamedTuple
import math

import numpy as np


PREDICTOR_ADVECTION_CFL_FAIL_FAST_LIMIT = 10.0


MUSCL_MOMENTUM_INITIAL_CFL_TARGET = 0.45


MUSCL_MOMENTUM_STAGE_CFL_LIMIT = 0.90


MUSCL_MOMENTUM_STAGE_RETRY_SAFETY = 0.90


MUSCL_MOMENTUM_MAX_CONSECUTIVE_STAGE_RETRIES = 16


COMPONENT_LABEL_BATCH_SWEEPS = 8


REACHABILITY_FLOOD_BATCH_LEVELS = 16


CG_DOT_LANES = 8192


CG_DOT_GROUPS = 64


_F32_MAX = float(np.finfo(np.float32).max)


PRESSURE_OUTLET_BACKFLOW_POLICIES = {"clamp", "allow"}


OBSTACLE_NORMAL_VELOCITY_POLICIES = {"face_clamp", "cell_zero_only"}


PRESSURE_INTERFACE_MATRIX_FV_CG_REQUIRED_SOLVERS = frozenset(
    (
        "jacobi",
        "compact_jacobi",
        "fv_jacobi",
        "fv_multigrid",
    )
)


PRESSURE_INTERFACE_MATRIX_FV_CG_FORCE_REASON = (
    "pressure_interface_matrix_requires_rowlist_fv_cg"
)


def _f32_float(value: object, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and f32-representable") from exc
    if (
        not math.isfinite(numeric)
        or abs(numeric) > _F32_MAX
        or (numeric != 0.0 and np.float32(numeric) == 0.0)
    ):
        raise ValueError(f"{name} must be finite and f32-representable")
    return numeric


def _finite_float(
    value: object,
    *,
    name: str,
    minimum: float,
    inclusive: bool,
) -> float:
    numeric = _f32_float(value, name=name)
    below_minimum = numeric < minimum if inclusive else numeric <= minimum
    if below_minimum:
        relation = "at least" if inclusive else "greater than"
        raise ValueError(
            f"{name} must be finite, f32-representable, and {relation} {minimum:g}"
        )
    return numeric


def _pressure_poisson_coefficients(
    *,
    density_kgm3: float,
    dt_s: float,
    spacing_m: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    dx, dy, dz = spacing_m
    return (
        _finite_float(
            density_kgm3 / dt_s,
            name="pressure rhs scale",
            minimum=0.0,
            inclusive=False,
        ),
        *(
            _finite_float(
                1.0 / (spacing * spacing),
                name=f"inverse {axis}-spacing squared",
                minimum=0.0,
                inclusive=False,
            )
            for axis, spacing in zip("xyz", (dx, dy, dz), strict=True)
        ),
    )


def _validated_count(
    value: object,
    *,
    name: str,
    allow_zero: bool,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Integral, np.integer),
    ):
        raise TypeError(f"{name} must be an integer")
    count = int(value)
    if count < 0 or (count == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return count


def _i32_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Integral, np.integer),
    ):
        raise TypeError(f"{name} must be a non-bool integer")
    numeric = int(value)
    if numeric < np.iinfo(np.int32).min or numeric > np.iinfo(np.int32).max:
        raise ValueError(f"{name} must fit signed int32 storage")
    return numeric


class _SSTWallDistanceCacheKey(NamedTuple):
    wall_flags: tuple[bool, bool, bool, bool, bool, bool]
    obstacle_topology_revision: int
    inactive_axis: int
    marker_count: int
    segment_count: int
    marker_digest: str
    segment_digest: str


class _SSTWallDistanceBaseCacheKey(NamedTuple):
    wall_flags: tuple[bool, bool, bool, bool, bool, bool]
    obstacle_topology_revision: int
    propagate_obstacle_distance: bool


def _diagnostic_int(value: object, default: int = -1) -> int:
    """Return an integer diagnostic without executing opaque conversions."""

    fallback = default if type(default) is int else -1
    if isinstance(value, (bool, np.bool_)):
        return fallback
    if isinstance(value, (Integral, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    return fallback


def _diagnostic_str(value: object, default: str = "unknown") -> str:
    """Return a string diagnostic without executing opaque conversions."""

    fallback = default if isinstance(default, str) else "unknown"
    return value if isinstance(value, str) else fallback


def _validated_pressure_solve_context(
    context: Mapping[str, object] | None,
) -> dict[str, object]:
    """Copy pressure-solve context while rejecting opaque mapping keys."""

    if context is None:
        return {}
    if not isinstance(context, Mapping):
        raise TypeError("pressure_solve_context must be a mapping or None")
    copied_context: dict[str, object] = {}
    for key, value in context.items():
        if not isinstance(key, str):
            raise TypeError("pressure_solve_context keys must be strings")
        copied_context[key] = value
    return copied_context


def _diagnostic_type_name(value: object) -> str:
    """Describe a value by type without invoking value-level user code."""

    value_type = type(value)
    try:
        module = type.__getattribute__(value_type, "__module__")
        qualname = type.__getattribute__(value_type, "__qualname__")
    except (AttributeError, TypeError):
        return "opaque"
    if not isinstance(module, str) or not isinstance(qualname, str):
        return "opaque"
    return f"{module}.{qualname}"


def _json_safe_diagnostic_mapping(
    value: Mapping[object, object],
    *,
    _active_ids: set[int],
) -> dict[str, object]:
    """Copy a mapping without stringifying non-string user-controlled keys."""

    reserved_string_keys = {
        key for key in value.keys() if isinstance(key, str)
    }
    safe_mapping: dict[str, object] = {}
    used_keys: set[str] = set()
    for ordinal, (key, nested_value) in enumerate(value.items()):
        if isinstance(key, str):
            safe_key = key
        else:
            key_type_name = _diagnostic_type_name(key)
            safe_key = f"__diagnostic_key__[{ordinal}:{key_type_name}]"
            collision_index = 0
            while safe_key in reserved_string_keys or safe_key in used_keys:
                collision_index += 1
                safe_key = (
                    f"__diagnostic_key__[{ordinal}:{key_type_name}:"
                    f"{collision_index}]"
                )
        used_keys.add(safe_key)
        safe_mapping[safe_key] = _json_safe_diagnostic_value(
            nested_value,
            _active_ids=_active_ids,
        )
    return safe_mapping


def _json_safe_diagnostic_value(
    value: object,
    *,
    _active_ids: set[int] | None = None,
) -> object:
    """Return a detached, strictly JSON-safe diagnostic value.

    Pressure failures are written to artifacts with ``allow_nan=False``.  This
    conversion therefore treats non-finite real values as unavailable instead
    of leaking JSON extensions, recursively copies containers, and represents
    opaque values by type only.  The active-id stack also keeps a malformed
    recursive diagnostics container from recursing forever.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.generic):
        return _json_safe_diagnostic_value(
            value.item(),
            _active_ids=_active_ids,
        )

    active_ids = set() if _active_ids is None else _active_ids
    value_id = id(value)
    if value_id in active_ids:
        return {"unsupported_type": "recursive_reference"}

    if isinstance(value, np.ndarray):
        active_ids.add(value_id)
        try:
            raw_value = value.item() if value.ndim == 0 else value.tolist()
            return _json_safe_diagnostic_value(
                raw_value,
                _active_ids=active_ids,
            )
        finally:
            active_ids.remove(value_id)
    if isinstance(value, Mapping):
        active_ids.add(value_id)
        try:
            return _json_safe_diagnostic_mapping(
                value,
                _active_ids=active_ids,
            )
        finally:
            active_ids.remove(value_id)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        active_ids.add(value_id)
        try:
            return [
                _json_safe_diagnostic_value(
                    nested_value,
                    _active_ids=active_ids,
                )
                for nested_value in value
            ]
        finally:
            active_ids.remove(value_id)

    return {
        "unsupported_type": _diagnostic_type_name(value)
    }


class PressureSolveConvergenceError(RuntimeError):
    """Fail-closed pressure convergence error with JSON-safe diagnostics."""

    def __init__(self, message: str, *, diagnostics: Mapping[str, object]) -> None:
        super().__init__(message)
        json_safe = _json_safe_diagnostic_value(diagnostics)
        if not isinstance(json_safe, dict):
            raise TypeError("pressure diagnostics must convert to a JSON object")
        self.diagnostics = json_safe


def _run_pre_projection_velocity_transaction(
    *,
    pre_projection_velocity_projector: object | None,
    fluid: object,
    pressure_solve_context: Mapping[str, object],
) -> dict[str, object]:
    """Run one optional velocity projection transaction before pressure.

    The protocol is deliberately generic: the fluid solver neither imports nor
    identifies a particular immersed-boundary implementation.  The caller owns
    the concrete projector and its cached geometry.  Keeping the three phases
    explicit also makes the transaction fail closed: an exception from prepare,
    solve, or commit propagates before pressure projection can begin.
    """

    if pre_projection_velocity_projector is None:
        return {
            "pre_projection_velocity_projector_prepared": False,
            "pre_projection_velocity_projector_converged": False,
            "pre_projection_velocity_projector_committed": False,
        }

    pre_projection_velocity_projector.prepare_projection_transaction(
        fluid=fluid,
        pressure_solve_context=pressure_solve_context,
    )
    pre_projection_velocity_projector.solve_projection_transaction()
    commit_diagnostics = (
        pre_projection_velocity_projector.commit_projection_transaction()
    )
    if not isinstance(commit_diagnostics, Mapping):
        raise TypeError(
            "commit_projection_transaction must return a diagnostics mapping"
        )
    diagnostics = dict(commit_diagnostics)
    required_status = ("prepared", "converged", "committed")
    missing_status = tuple(
        key for key in required_status if key not in diagnostics
    )
    if missing_status:
        raise RuntimeError(
            "pre-projection velocity transaction diagnostics are missing "
            f"required status keys: {missing_status}"
        )
    invalid_status_types = tuple(
        key
        for key in required_status
        if not isinstance(diagnostics[key], (bool, np.bool_))
    )
    if invalid_status_types:
        raise TypeError(
            "pre-projection velocity transaction status values must be "
            f"boolean: {invalid_status_types}"
        )
    failed_status = tuple(
        key for key in required_status if not bool(diagnostics[key])
    )
    if failed_status:
        raise RuntimeError(
            "pre-projection velocity transaction failed before pressure "
            f"projection: {failed_status}"
        )
    return {
        **diagnostics,
        "pre_projection_velocity_projector_prepared": bool(
            diagnostics["prepared"]
        ),
        "pre_projection_velocity_projector_converged": bool(
            diagnostics["converged"]
        ),
        "pre_projection_velocity_projector_committed": bool(
            diagnostics["committed"]
        ),
    }
