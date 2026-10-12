"""Portable, fail-closed snapshots for reusable steady preflow states.

This module is intentionally host-only.  It defines the NumPy persistence
contract; callers remain responsible for transferring arrays to or from their
runtime fields.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


PREFLOW_SNAPSHOT_SCHEMA_VERSION = 8
_DIRECTED_BOUNDARY_PREFLOW_SNAPSHOT_SCHEMA_VERSION = 7
_CANONICAL_LEDGER_PREFLOW_SNAPSHOT_SCHEMA_VERSION = 6
_LEGACY_PREFLOW_SNAPSHOT_SCHEMA_VERSION = 5
PREFLOW_SNAPSHOT_FORMAT = "simulation_core.preflow_snapshot"

_VELOCITY_DIRICHLET_BOUNDARY_AUTHORITIES = frozenset({"legacy", "canonical"})
_CANONICAL_LEDGER_ONLY_FIELD_NAMES = (
    "velocity_dirichlet_boundary_active_component_mask",
    "velocity_dirichlet_boundary_pressure_mobility",
    "velocity_dirichlet_boundary_component_enforcement_weight",
    "velocity_dirichlet_boundary_component_region_id",
    "velocity_dirichlet_boundary_owned_component_mask",
)
_DIRECTED_EXTERNAL_BOUNDARY_MASK_FIELD_NAMES = (
    "external_velocity_boundary_x_face_active_component_mask",
    "external_velocity_boundary_y_face_active_component_mask",
    "external_velocity_boundary_z_face_active_component_mask",
)
_DIRECTED_EXTERNAL_BOUNDARY_VALUE_FIELD_NAMES = (
    "external_velocity_boundary_x_face_value_mps",
    "external_velocity_boundary_y_face_value_mps",
    "external_velocity_boundary_z_face_value_mps",
)
_DIRECTED_EXTERNAL_BOUNDARY_FIELD_NAMES = (
    *_DIRECTED_EXTERNAL_BOUNDARY_MASK_FIELD_NAMES,
    *_DIRECTED_EXTERNAL_BOUNDARY_VALUE_FIELD_NAMES,
)
_SST_FIELD_NAMES = (
    "sst_turbulent_kinetic_energy",
    "sst_specific_dissipation_rate",
    "sst_eddy_viscosity_pa_s",
    "sst_wall_distance_m",
)

_FIELD_DTYPES = {
    "velocity": np.dtype(np.float32),
    "velocity_prev": np.dtype(np.float32),
    "pressure": np.dtype(np.float64),
    "fsi_pressure": np.dtype(np.float64),
    "obstacle": np.dtype(np.int32),
    "hibm_base_obstacle": np.dtype(np.int32),
    "hibm_dynamic_solid_volume_obstacle": np.dtype(np.int32),
    "hibm_dynamic_solid_volume_external_carve": np.dtype(np.int32),
    "velocity_dirichlet_boundary_active": np.dtype(np.int32),
    "velocity_dirichlet_boundary_value_mps": np.dtype(np.float32),
    "velocity_dirichlet_boundary_projection_weight": np.dtype(np.float32),
    "velocity_dirichlet_boundary_enforcement_weight": np.dtype(np.float32),
    "velocity_dirichlet_boundary_marker_region_id": np.dtype(np.int32),
    "velocity_dirichlet_boundary_hard_fixed_component_mask": np.dtype(
        np.int32
    ),
    "velocity_dirichlet_boundary_external_exact_component_mask": np.dtype(
        np.int32
    ),
    "velocity_dirichlet_boundary_owned_row": np.dtype(np.int32),
    "velocity_dirichlet_boundary_active_component_mask": np.dtype(np.int32),
    "velocity_dirichlet_boundary_pressure_mobility": np.dtype(np.float32),
    "velocity_dirichlet_boundary_component_enforcement_weight": np.dtype(
        np.float32
    ),
    "velocity_dirichlet_boundary_component_region_id": np.dtype(np.int32),
    "velocity_dirichlet_boundary_owned_component_mask": np.dtype(np.int32),
    "external_velocity_boundary_x_face_active_component_mask": np.dtype(
        np.int32
    ),
    "external_velocity_boundary_x_face_value_mps": np.dtype(np.float32),
    "external_velocity_boundary_y_face_active_component_mask": np.dtype(
        np.int32
    ),
    "external_velocity_boundary_y_face_value_mps": np.dtype(np.float32),
    "external_velocity_boundary_z_face_active_component_mask": np.dtype(
        np.int32
    ),
    "external_velocity_boundary_z_face_value_mps": np.dtype(np.float32),
    "sst_turbulent_kinetic_energy": np.dtype(np.float32),
    "sst_specific_dissipation_rate": np.dtype(np.float32),
    "sst_eddy_viscosity_pa_s": np.dtype(np.float32),
    "sst_wall_distance_m": np.dtype(np.float32),
}

PREFLOW_SNAPSHOT_FIELD_NAMES = tuple(_FIELD_DTYPES)
_SCHEMA_V7_FIELD_NAMES = tuple(
    name for name in PREFLOW_SNAPSHOT_FIELD_NAMES if name not in _SST_FIELD_NAMES
)
_SCHEMA_V6_FIELD_NAMES = tuple(
    name
    for name in _SCHEMA_V7_FIELD_NAMES
    if name not in _DIRECTED_EXTERNAL_BOUNDARY_FIELD_NAMES
)
_LEGACY_SCHEMA_V5_FIELD_NAMES = tuple(
    name
    for name in _SCHEMA_V6_FIELD_NAMES
    if name not in _CANONICAL_LEDGER_ONLY_FIELD_NAMES
)

_VECTOR_FIELD_NAMES = frozenset(
    {
        "velocity",
        "velocity_prev",
        "velocity_dirichlet_boundary_value_mps",
        "velocity_dirichlet_boundary_pressure_mobility",
        "velocity_dirichlet_boundary_component_enforcement_weight",
        "velocity_dirichlet_boundary_component_region_id",
    }
)
_IDENTITY_FIELD_NAMES = (
    "config_sha256",
    "source_sha256",
    "geometry_sha256",
)
_ARTIFACT_IDENTITY_FIELD_NAMES = frozenset(
    {
        "metadata_file_sha256",
        "manifest_sha256",
        "npz_file",
        "npz_sha256",
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_GENERATION_NPZ_PATTERN = re.compile(r".+\.[0-9a-f]{32}\.npz\Z")
_MANIFEST_FIELD_NAMES = frozenset(
    {
        "format",
        "schema_version",
        "grid_shape",
        "identity",
        "fields",
        "history",
        "velocity_dirichlet_boundary_authority",
        "velocity_dirichlet_component_ledger_generation",
        "npz_file",
        "npz_sha256",
        "manifest_sha256",
    }
)
_LEGACY_SCHEMA_V5_MANIFEST_FIELD_NAMES = frozenset(
    _MANIFEST_FIELD_NAMES
    - {
        "velocity_dirichlet_boundary_authority",
        "velocity_dirichlet_component_ledger_generation",
    }
)


class PreflowSnapshotError(ValueError):
    """Base class for preflow snapshot contract failures."""


class PreflowSnapshotValidationError(PreflowSnapshotError):
    """Raised before persistence when a proposed snapshot is invalid."""


class PreflowSnapshotIntegrityError(PreflowSnapshotError):
    """Raised when persisted snapshot content cannot be trusted."""


class PreflowSnapshotMismatchError(PreflowSnapshotError):
    """Raised when a valid snapshot belongs to a different current model."""


def _validated_velocity_dirichlet_boundary_authority(
    authority: object,
    *,
    error_type: type[PreflowSnapshotError],
) -> str:
    if not isinstance(authority, str):
        raise error_type(
            "snapshot velocity Dirichlet boundary authority must be 'legacy' "
            "or 'canonical'"
        )
    if authority not in _VELOCITY_DIRICHLET_BOUNDARY_AUTHORITIES:
        raise error_type(
            "snapshot velocity Dirichlet boundary authority must be 'legacy' "
            f"or 'canonical', got {authority!r}"
        )
    return authority


def _validated_velocity_dirichlet_component_ledger_generation(
    generation: object,
    *,
    error_type: type[PreflowSnapshotError],
) -> int:
    if (
        not isinstance(generation, Integral)
        or isinstance(generation, bool)
        or int(generation) < 0
    ):
        raise error_type(
            "snapshot velocity Dirichlet component ledger generation must be "
            "a non-negative integer"
        )
    return int(generation)


def _neutral_canonical_ledger_fields(
    grid_shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    vector_shape = grid_shape + (3,)
    return {
        "velocity_dirichlet_boundary_active_component_mask": np.zeros(
            grid_shape,
            dtype=np.int32,
        ),
        "velocity_dirichlet_boundary_pressure_mobility": np.ones(
            vector_shape,
            dtype=np.float32,
        ),
        "velocity_dirichlet_boundary_component_enforcement_weight": np.zeros(
            vector_shape,
            dtype=np.float32,
        ),
        "velocity_dirichlet_boundary_component_region_id": np.full(
            vector_shape,
            -1,
            dtype=np.int32,
        ),
        "velocity_dirichlet_boundary_owned_component_mask": np.zeros(
            grid_shape,
            dtype=np.int32,
        ),
    }


def _neutral_directed_external_boundary_fields(
    grid_shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    nx, ny, nz = grid_shape
    plane_shapes = {
        "x": (2, ny, nz),
        "y": (2, nx, nz),
        "z": (2, nx, ny),
    }
    fields: dict[str, np.ndarray] = {}
    for axis_name, plane_shape in plane_shapes.items():
        fields[
            f"external_velocity_boundary_{axis_name}_face_active_component_mask"
        ] = np.zeros(plane_shape, dtype=np.int32)
        fields[f"external_velocity_boundary_{axis_name}_face_value_mps"] = (
            np.zeros(plane_shape + (3,), dtype=np.float32)
        )
    return fields


def _neutral_laminar_sst_fields(
    grid_shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    return {
        "sst_turbulent_kinetic_energy": np.zeros(
            grid_shape,
            dtype=np.float32,
        ),
        "sst_specific_dissipation_rate": np.ones(
            grid_shape,
            dtype=np.float32,
        ),
        "sst_eddy_viscosity_pa_s": np.zeros(
            grid_shape,
            dtype=np.float32,
        ),
        "sst_wall_distance_m": np.full(
            grid_shape,
            np.float32(1.0e20),
            dtype=np.float32,
        ),
    }


def _sha256_bytes(payload: bytes, *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def _canonical_json_value(value: Any, *, context: str) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} mapping keys must be strings")
            normalized[key] = _canonical_json_value(
                item,
                context=f"{context}.{key}",
            )
        return normalized
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray, memoryview),
    ):
        return [
            _canonical_json_value(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(
        f"{context} contains unsupported value type {type(value).__name__!r}"
    )


def _canonical_json_bytes(value: Any, *, context: str) -> bytes:
    normalized = _canonical_json_value(value, context=context)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_config_sha256(config: Any) -> str:
    """Hash a JSON-compatible configuration with stable key ordering."""

    return _sha256_bytes(
        _canonical_json_bytes(config, context="config"),
        domain=b"preflow-config-v1",
    )


def canonical_source_sha256(sources: Mapping[str, str | bytes]) -> str:
    """Hash logical source names and exact source bytes, independent of map order."""

    if not isinstance(sources, Mapping):
        raise ValueError("source payload must be a mapping of logical names to content")
    entries: list[dict[str, Any]] = []
    normalized_names: set[str] = set()
    for logical_name, content in sources.items():
        if not isinstance(logical_name, str) or not logical_name:
            raise ValueError("source logical names must be non-empty strings")
        normalized_name = logical_name.replace("\\", "/")
        if normalized_name in normalized_names:
            raise ValueError(f"source contains duplicate logical name {normalized_name!r}")
        normalized_names.add(normalized_name)
        if isinstance(content, str):
            source_bytes = content.encode("utf-8")
        elif isinstance(content, bytes):
            source_bytes = content
        else:
            raise ValueError(
                "source content must be str or bytes: "
                f"{normalized_name!r} has {type(content).__name__!r}"
            )
        entries.append(
            {
                "name": normalized_name,
                "size_bytes": len(source_bytes),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
        )
    entries.sort(key=lambda item: item["name"])
    return _sha256_bytes(
        _canonical_json_bytes(entries, context="source"),
        domain=b"preflow-source-v1",
    )


def _array_descriptor(name: str, value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{context} array names must be non-empty strings")
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "biuf":
        raise ValueError(
            f"{context} array {name!r} must have a real numeric or boolean dtype"
        )
    canonical_dtype = (
        array.dtype.newbyteorder("<")
        if array.dtype.itemsize > 1
        else array.dtype
    )
    contiguous = np.asarray(array, dtype=canonical_dtype, order="C")
    if not bool(np.all(np.isfinite(contiguous))):
        raise ValueError(f"{context} array {name!r} must contain only finite values")
    content_digest = hashlib.sha256()
    if contiguous.size > 0:
        content_digest.update(memoryview(contiguous).cast("B"))
    return {
        "name": name,
        "dtype": contiguous.dtype.str,
        "shape": [int(size) for size in contiguous.shape],
        "data_sha256": content_digest.hexdigest(),
    }


def canonical_geometry_sha256(geometry: Mapping[str, Any]) -> str:
    """Hash named numeric arrays independent of mapping and memory layout."""

    if not isinstance(geometry, Mapping):
        raise ValueError("geometry payload must be a mapping of names to arrays")
    descriptors = [
        _array_descriptor(name, value, context="geometry")
        for name, value in geometry.items()
    ]
    names = [item["name"] for item in descriptors]
    if len(names) != len(set(names)):
        raise ValueError("geometry contains duplicate array names")
    descriptors.sort(key=lambda item: item["name"])
    return _sha256_bytes(
        _canonical_json_bytes(descriptors, context="geometry"),
        domain=b"preflow-geometry-v1",
    )


def _require_sha256(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hexadecimal digest")
    return value


@dataclass(frozen=True)
class PreflowSnapshotIdentity:
    """Identity of the configuration, source, and geometry that own a snapshot."""

    config_sha256: str
    source_sha256: str
    geometry_sha256: str

    def __post_init__(self) -> None:
        for field_name in _IDENTITY_FIELD_NAMES:
            _require_sha256(getattr(self, field_name), field_name=field_name)

    @classmethod
    def from_inputs(
        cls,
        *,
        config: Any,
        sources: Mapping[str, str | bytes],
        geometry: Mapping[str, Any],
    ) -> "PreflowSnapshotIdentity":
        return cls(
            config_sha256=canonical_config_sha256(config),
            source_sha256=canonical_source_sha256(sources),
            geometry_sha256=canonical_geometry_sha256(geometry),
        )


def _expected_shape(name: str, grid_shape: tuple[int, int, int]) -> tuple[int, ...]:
    nx, ny, nz = grid_shape
    directed_plane_shapes = {
        "external_velocity_boundary_x_face_active_component_mask": (2, ny, nz),
        "external_velocity_boundary_x_face_value_mps": (2, ny, nz, 3),
        "external_velocity_boundary_y_face_active_component_mask": (2, nx, nz),
        "external_velocity_boundary_y_face_value_mps": (2, nx, nz, 3),
        "external_velocity_boundary_z_face_active_component_mask": (2, nx, ny),
        "external_velocity_boundary_z_face_value_mps": (2, nx, ny, 3),
    }
    if name in directed_plane_shapes:
        return directed_plane_shapes[name]
    return grid_shape + ((3,) if name in _VECTOR_FIELD_NAMES else ())


def _validated_field_copies(
    fields: Mapping[str, Any],
    *,
    authority: str,
    error_type: type[PreflowSnapshotError],
) -> dict[str, np.ndarray]:
    if not isinstance(fields, Mapping):
        raise error_type("snapshot fields must be a mapping")
    provided = set(fields)
    required = set(PREFLOW_SNAPSHOT_FIELD_NAMES)
    missing = sorted(required - provided)
    unexpected = sorted(provided - required)
    canonical_only = set(_CANONICAL_LEDGER_ONLY_FIELD_NAMES)
    directed_external_only = set(_DIRECTED_EXTERNAL_BOUNDARY_FIELD_NAMES)
    missing_set = set(missing)
    legacy_neutral_upgrade = authority == "legacy" and missing_set in (
        canonical_only,
        directed_external_only,
        canonical_only | directed_external_only,
    )
    if legacy_neutral_upgrade:
        pressure_for_shape = np.asarray(fields["pressure"])
        if pressure_for_shape.ndim != 3 or any(
            int(size) <= 0 for size in pressure_for_shape.shape
        ):
            raise error_type(
                "snapshot field 'pressure' shape must be a non-empty 3-D grid"
            )
        grid_shape_for_upgrade = tuple(
            int(size) for size in pressure_for_shape.shape
        )
        neutral_fields: dict[str, np.ndarray] = {}
        if canonical_only <= missing_set:
            neutral_fields.update(
                _neutral_canonical_ledger_fields(grid_shape_for_upgrade)
            )
        if directed_external_only <= missing_set:
            neutral_fields.update(
                _neutral_directed_external_boundary_fields(
                    grid_shape_for_upgrade
                )
            )
        working_fields: Mapping[str, Any] = {**fields, **neutral_fields}
        missing = []
    else:
        working_fields = fields
    if missing:
        raise error_type(f"snapshot fields are missing required fields: {missing}")
    if unexpected:
        raise error_type(f"snapshot fields contain unexpected fields: {unexpected}")

    pressure = np.asarray(working_fields["pressure"])
    if pressure.ndim != 3 or any(int(size) <= 0 for size in pressure.shape):
        raise error_type(
            "snapshot field 'pressure' shape must be a non-empty 3-D grid"
        )
    grid_shape = tuple(int(size) for size in pressure.shape)
    validated: dict[str, np.ndarray] = {}
    for name in PREFLOW_SNAPSHOT_FIELD_NAMES:
        array = np.asarray(working_fields[name])
        expected_dtype = _FIELD_DTYPES[name]
        expected_shape = _expected_shape(name, grid_shape)
        if array.dtype != expected_dtype:
            raise error_type(
                f"snapshot field {name!r} dtype {array.dtype.str!r} does not match "
                f"required dtype {expected_dtype.str!r}"
            )
        if tuple(array.shape) != expected_shape:
            raise error_type(
                f"snapshot field {name!r} shape {tuple(array.shape)} does not match "
                f"required shape {expected_shape}"
            )
        if not bool(np.all(np.isfinite(array))):
            raise error_type(f"snapshot field {name!r} must contain only finite values")
        contiguous = np.asarray(array, dtype=expected_dtype, order="C")
        copied = np.frombuffer(
            contiguous.tobytes(order="C"),
            dtype=expected_dtype,
        ).reshape(expected_shape)
        validated[name] = copied

    binary_fields = (
        "obstacle",
        "hibm_base_obstacle",
        "hibm_dynamic_solid_volume_obstacle",
        "hibm_dynamic_solid_volume_external_carve",
        "velocity_dirichlet_boundary_active",
        "velocity_dirichlet_boundary_owned_row",
    )
    for name in binary_fields:
        array = validated[name]
        if not bool(np.all((array == 0) | (array == 1))):
            raise error_type(
                f"snapshot field {name!r} must contain only binary 0/1 values"
            )

    nonnegative_sst_fields = (
        "sst_turbulent_kinetic_energy",
        "sst_eddy_viscosity_pa_s",
    )
    for name in nonnegative_sst_fields:
        if bool(np.any(validated[name] < 0.0)):
            raise error_type(
                f"snapshot field {name!r} must contain only non-negative values"
            )
    positive_sst_fields = (
        "sst_specific_dissipation_rate",
        "sst_wall_distance_m",
    )
    for name in positive_sst_fields:
        if bool(np.any(validated[name] <= 0.0)):
            raise error_type(
                f"snapshot field {name!r} must contain only positive values"
            )

    hard_mask_name = "velocity_dirichlet_boundary_hard_fixed_component_mask"
    hard_mask = validated[hard_mask_name]
    if not bool(np.all((hard_mask >= 0) & (hard_mask <= 7))):
        raise error_type(
            f"snapshot field {hard_mask_name!r} must contain only three-bit "
            "hard_fixed_component_mask values in [0, 7]"
        )

    external_exact_mask_name = (
        "velocity_dirichlet_boundary_external_exact_component_mask"
    )
    external_exact_mask = validated[external_exact_mask_name]
    if not bool(np.all((external_exact_mask >= 0) & (external_exact_mask <= 7))):
        raise error_type(
            f"snapshot field {external_exact_mask_name!r} must contain only "
            "three-bit external_exact component mask values in [0, 7]"
        )
    if not bool(np.all((external_exact_mask & hard_mask) == external_exact_mask)):
        raise error_type(
            f"snapshot field {external_exact_mask_name!r} must be a bitwise "
            f"subset of {hard_mask_name!r}"
        )

    projection_weight_name = "velocity_dirichlet_boundary_projection_weight"
    projection_weight = validated[projection_weight_name]
    if not bool(np.all((projection_weight >= 0.0) & (projection_weight <= 1.0))):
        raise error_type(
            f"snapshot field {projection_weight_name!r} projection_weight values "
            "must lie in [0, 1]"
        )
    enforcement_weight_name = "velocity_dirichlet_boundary_enforcement_weight"
    enforcement_weight = validated[enforcement_weight_name]
    if not bool(
        np.all((enforcement_weight >= 0.0) & (enforcement_weight <= 1.0))
    ):
        raise error_type(
            f"snapshot field {enforcement_weight_name!r} enforcement_weight "
            "values must lie in [0, 1]"
        )

    active_component_mask_name = (
        "velocity_dirichlet_boundary_active_component_mask"
    )
    active_component_mask = validated[active_component_mask_name]
    owned_component_mask_name = "velocity_dirichlet_boundary_owned_component_mask"
    owned_component_mask = validated[owned_component_mask_name]
    for name, mask in (
        (active_component_mask_name, active_component_mask),
        (owned_component_mask_name, owned_component_mask),
        *(
            (name, validated[name])
            for name in _DIRECTED_EXTERNAL_BOUNDARY_MASK_FIELD_NAMES
        ),
    ):
        if not bool(np.all((mask >= 0) & (mask <= 7))):
            raise error_type(
                f"snapshot field {name!r} must contain only three-bit mask "
                "values in [0, 7]"
            )

    component_pressure_mobility_name = (
        "velocity_dirichlet_boundary_pressure_mobility"
    )
    component_pressure_mobility = validated[component_pressure_mobility_name]
    component_enforcement_weight_name = (
        "velocity_dirichlet_boundary_component_enforcement_weight"
    )
    component_enforcement_weight = validated[component_enforcement_weight_name]
    for name, weight in (
        (component_pressure_mobility_name, component_pressure_mobility),
        (component_enforcement_weight_name, component_enforcement_weight),
    ):
        if not bool(np.all((weight >= 0.0) & (weight <= 1.0))):
            raise error_type(
                f"snapshot field {name!r} values must lie in [0, 1]"
            )

    if authority == "canonical":
        if bool(np.any((hard_mask & active_component_mask) != hard_mask)):
            raise error_type(
                "snapshot canonical hard component masks must be a bitwise "
                "subset of active component masks"
            )
        if bool(np.any((owned_component_mask & active_component_mask) != owned_component_mask)):
            raise error_type(
                "snapshot canonical owned component masks must be a bitwise "
                "subset of active component masks"
            )
        if bool(np.any((external_exact_mask & owned_component_mask) != 0)):
            raise error_type(
                "snapshot canonical external exact and owned component masks "
                "must not overlap"
            )
        component_region_id = validated[
            "velocity_dirichlet_boundary_component_region_id"
        ]
        component_value = validated["velocity_dirichlet_boundary_value_mps"]
        obstacle = validated["obstacle"] != 0
        # Match the device precommit and obstacle-interface consumer exactly.
        # NPZ persists float32 bit patterns losslessly, so restore needs no
        # wider serialization tolerance here.
        component_tolerance = 1.0e-6
        for axis in range(3):
            axis_bit = 1 << axis
            active_axis = (active_component_mask & axis_bit) != 0
            hard_axis = (hard_mask & axis_bit) != 0
            owned_axis = (owned_component_mask & axis_bit) != 0
            external_axis = (external_exact_mask & axis_bit) != 0
            inactive = (active_component_mask & (1 << axis)) == 0
            if bool(
                np.any(
                    hard_axis
                    & (
                        np.abs(component_pressure_mobility[..., axis])
                        > component_tolerance
                    )
                )
            ):
                raise error_type(
                    "snapshot canonical hard components require zero pressure "
                    "mobility"
                )
            if bool(
                np.any(
                    hard_axis
                    & (
                        np.abs(component_enforcement_weight[..., axis] - 1.0)
                        > component_tolerance
                    )
                )
            ):
                raise error_type(
                    "snapshot canonical hard components require unit enforcement "
                    "weight"
                )
            if bool(np.any(active_axis & ~hard_axis & ~owned_axis)):
                raise error_type(
                    "snapshot canonical active soft components require ownership"
                )
            if bool(
                np.any(
                    inactive
                    & (np.abs(component_value[..., axis]) > component_tolerance)
                )
            ):
                raise error_type(
                    "snapshot canonical inactive components require zero velocity "
                    "value"
                )
            if bool(
                np.any(
                    inactive
                    & (
                        np.abs(component_pressure_mobility[..., axis] - 1.0)
                        > component_tolerance
                    )
                )
            ):
                raise error_type(
                    "snapshot canonical inactive components require unit "
                    "pressure mobility"
                )
            if bool(
                np.any(
                    inactive
                    & (
                        np.abs(component_enforcement_weight[..., axis])
                        > component_tolerance
                    )
                )
            ):
                raise error_type(
                    "snapshot canonical inactive components require zero "
                    "enforcement weight"
                )
            if bool(np.any(inactive & (component_region_id[..., axis] != -1))):
                raise error_type(
                    "snapshot canonical inactive components require region id -1"
                )

            minus_neighbor_is_fluid = np.zeros_like(obstacle, dtype=np.bool_)
            if axis == 0:
                minus_neighbor_is_fluid[1:, :, :] = ~obstacle[:-1, :, :]
            elif axis == 1:
                minus_neighbor_is_fluid[:, 1:, :] = ~obstacle[:, :-1, :]
            else:
                minus_neighbor_is_fluid[:, :, 1:] = ~obstacle[:, :, :-1]
            legal_obstacle_storage = (
                minus_neighbor_is_fluid
                & hard_axis
                & owned_axis
                & ~external_axis
                & (
                    np.abs(component_pressure_mobility[..., axis])
                    <= component_tolerance
                )
                & (
                    np.abs(component_enforcement_weight[..., axis] - 1.0)
                    <= component_tolerance
                )
            )
            if bool(np.any(active_axis & obstacle & ~legal_obstacle_storage)):
                raise error_type(
                    "snapshot canonical active components use illegal obstacle "
                    "storage"
                )

    # Scalar compatibility rows remain live inputs to auxiliary reconstruction,
    # row-cloud cleanup, and symmetry paths even under canonical authority, so
    # their own ledger must stay coherent.  Canonical hard/external component
    # provenance is intentionally independent of the legacy scalar active bit.
    active = validated["velocity_dirichlet_boundary_active"] != 0
    owned = validated["velocity_dirichlet_boundary_owned_row"] != 0
    obstacle = validated["obstacle"] != 0
    marker_region_id = validated["velocity_dirichlet_boundary_marker_region_id"]
    if bool(np.any(owned & ~active)):
        raise error_type(
            "snapshot velocity boundary owned rows must also be active"
        )
    if bool(np.any((marker_region_id >= 0) & ~active)):
        raise error_type(
            "snapshot velocity boundary marker region provenance requires an "
            "active row"
        )
    if bool(np.any(owned & (external_exact_mask != 0))):
        raise error_type(
            "snapshot velocity boundary dynamic owned rows must not overlap "
            "external exact component provenance"
        )
    owned_partial_hard = owned & (hard_mask != 0) & (hard_mask != 7)
    if bool(np.any(owned_partial_hard)):
        raise error_type(
            "snapshot velocity boundary dynamic owned rows require either no "
            "hard components or the full hard mask"
        )
    owned_soft = owned & (hard_mask == 0)
    owned_soft_complement_error = np.abs(
        projection_weight + enforcement_weight - 1.0
    )
    if bool(np.any(owned_soft & (owned_soft_complement_error > 2.0e-6))):
        raise error_type(
            "snapshot velocity boundary dynamic owned soft rows require "
            "projection/enforcement complement weights"
        )
    owned_exact = owned & (hard_mask == 7)
    if bool(
        np.any(owned_exact & (np.abs(enforcement_weight - 1.0) > 2.0e-6))
    ):
        raise error_type(
            "snapshot velocity boundary dynamic owned exact rows require "
            "unit enforcement weight"
        )
    direct = active & ~owned
    direct_split_error = np.abs(projection_weight - enforcement_weight)
    if bool(np.any(direct & (direct_split_error > 2.0e-6))):
        raise error_type(
            "snapshot velocity boundary direct rows require matching "
            "projection/enforcement weights"
        )
    if authority == "legacy":
        if bool(np.any((external_exact_mask != 0) & ~active)):
            raise error_type(
                "snapshot velocity boundary external exact component masks must be "
                "zero on inactive rows"
            )
        if bool(np.any((hard_mask != 0) & ~active)):
            raise error_type(
                "snapshot velocity boundary hard component masks require active rows"
            )
    if bool(np.any((projection_weight != 0.0) & ~active)):
        raise error_type(
            "snapshot velocity boundary projection weight must be zero on "
            "inactive rows"
        )
    if bool(np.any((enforcement_weight != 0.0) & ~active)):
        raise error_type(
            "snapshot velocity boundary enforcement weight must be zero on "
            "inactive rows"
        )
    if bool(np.any(active & obstacle)):
        raise error_type(
            "snapshot active velocity boundary rows cannot occupy obstacle cells"
        )
    return validated


def validate_preflow_snapshot_fields(
    fields: Mapping[str, Any],
    *,
    velocity_dirichlet_boundary_authority: object,
) -> dict[str, np.ndarray]:
    """Validate and copy a field payload before any runtime state is mutated.

    Loading from disk already constructs :class:`PreflowSnapshot`, but the
    runner restore helper is also exercised directly.  Keeping this host-only
    validation entry point makes both paths share schema, dtype, shape,
    finiteness, and canonical-ledger invariants before the first ``from_numpy``.
    """

    authority = _validated_velocity_dirichlet_boundary_authority(
        velocity_dirichlet_boundary_authority,
        error_type=PreflowSnapshotValidationError,
    )
    return _validated_field_copies(
        fields,
        authority=authority,
        error_type=PreflowSnapshotValidationError,
    )


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _validated_artifact_identity(
    value: Mapping[str, str] | None,
    *,
    error_type: type[PreflowSnapshotError],
) -> Mapping[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _ARTIFACT_IDENTITY_FIELD_NAMES:
        raise error_type(
            "snapshot artifact identity must contain exactly metadata_file_sha256, "
            "manifest_sha256, npz_file, and npz_sha256"
        )
    normalized = {name: value[name] for name in _ARTIFACT_IDENTITY_FIELD_NAMES}
    try:
        for name in ("metadata_file_sha256", "manifest_sha256", "npz_sha256"):
            _require_sha256(normalized[name], field_name=name)
    except (TypeError, ValueError) as exc:
        raise error_type(f"snapshot artifact identity is invalid: {exc}") from exc
    npz_file = normalized["npz_file"]
    if (
        not isinstance(npz_file, str)
        or Path(npz_file).name != npz_file
        or _GENERATION_NPZ_PATTERN.fullmatch(npz_file) is None
    ):
        raise error_type("snapshot artifact identity npz_file is invalid")
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class PreflowSnapshot:
    """Validated immutable host representation of a reusable preflow state."""

    fields: Mapping[str, np.ndarray]
    identity: PreflowSnapshotIdentity
    history: Any = None
    velocity_dirichlet_boundary_authority: str = "legacy"
    velocity_dirichlet_component_ledger_generation: int = 0
    artifact_identity: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PreflowSnapshotIdentity):
            raise PreflowSnapshotValidationError(
                "snapshot identity must be a PreflowSnapshotIdentity"
            )
        authority = _validated_velocity_dirichlet_boundary_authority(
            self.velocity_dirichlet_boundary_authority,
            error_type=PreflowSnapshotValidationError,
        )
        generation = _validated_velocity_dirichlet_component_ledger_generation(
            self.velocity_dirichlet_component_ledger_generation,
            error_type=PreflowSnapshotValidationError,
        )
        artifact_identity = _validated_artifact_identity(
            self.artifact_identity,
            error_type=PreflowSnapshotValidationError,
        )
        validated = _validated_field_copies(
            self.fields,
            authority=authority,
            error_type=PreflowSnapshotValidationError,
        )
        try:
            history_bytes = _canonical_json_bytes(self.history, context="history")
            normalized_history = _freeze_json_value(
                json.loads(history_bytes.decode("utf-8"))
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PreflowSnapshotValidationError(f"invalid snapshot history: {exc}") from exc
        object.__setattr__(self, "fields", MappingProxyType(validated))
        object.__setattr__(self, "history", normalized_history)
        object.__setattr__(self, "velocity_dirichlet_boundary_authority", authority)
        object.__setattr__(
            self,
            "velocity_dirichlet_component_ledger_generation",
            generation,
        )
        object.__setattr__(self, "artifact_identity", artifact_identity)


@dataclass(frozen=True)
class PreflowSnapshotFiles:
    """Stable load handle plus the manifest and generation it committed."""

    snapshot_path: Path
    npz_path: Path
    metadata_path: Path

    def __fspath__(self) -> str:
        return os.fspath(self.snapshot_path)


def _snapshot_files(path: str | os.PathLike[str]) -> PreflowSnapshotFiles:
    requested = Path(path)
    if _GENERATION_NPZ_PATTERN.fullmatch(requested.name) is not None:
        raise ValueError(
            "generation NPZ paths are payload artifacts, not load handles; "
            "pass PreflowSnapshotFiles or its snapshot_path"
        )
    if requested.suffix.lower() == ".npz":
        npz_path = requested
    elif requested.suffix.lower() == ".json":
        raise ValueError("snapshot path must be a prefix or an .npz path, not .json")
    else:
        npz_path = Path(f"{requested}.npz")
    return PreflowSnapshotFiles(
        snapshot_path=requested,
        npz_path=npz_path,
        metadata_path=npz_path.with_suffix(".json"),
    )


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")


def _generation_npz_path(base_path: Path) -> Path:
    return base_path.with_name(f"{base_path.stem}.{uuid.uuid4().hex}.npz")


def _best_effort_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _write_npz(path: Path, fields: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **fields)
        handle.flush()
        os.fsync(handle.fileno())


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_payload(identity: PreflowSnapshotIdentity) -> dict[str, str]:
    return {name: getattr(identity, name) for name in _IDENTITY_FIELD_NAMES}


def _field_manifest(fields: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": [int(size) for size in fields[name].shape],
            "dtype": fields[name].dtype.str,
            "sha256": canonical_geometry_sha256({name: fields[name]}),
        }
        for name in PREFLOW_SNAPSHOT_FIELD_NAMES
    }


def save_preflow_snapshot(
    path: str | os.PathLike[str],
    snapshot: PreflowSnapshot,
) -> PreflowSnapshotFiles:
    """Commit a unique NPZ generation via one atomic manifest pointer swap.

    Existing generations are never overwritten.  The fixed JSON manifest is
    the sole live pointer, so interrupted or concurrent writers leave either
    the old complete generation or one new complete generation loadable.
    Old and interrupted-writer generations are retained deliberately; callers
    may prune unreferenced generations only after all possible readers of an
    older manifest have quiesced.
    """

    if not isinstance(snapshot, PreflowSnapshot):
        raise TypeError("snapshot must be a PreflowSnapshot")
    stable_snapshot = snapshot
    base_files = _snapshot_files(path)
    files = PreflowSnapshotFiles(
        snapshot_path=base_files.snapshot_path,
        npz_path=_generation_npz_path(base_files.npz_path),
        metadata_path=base_files.metadata_path,
    )
    files.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = _temporary_sibling(files.npz_path)
    temporary_metadata = _temporary_sibling(files.metadata_path)
    pending_paths = {temporary_npz, temporary_metadata}
    try:
        _write_npz(temporary_npz, stable_snapshot.fields)
        manifest: dict[str, Any] = {
            "format": PREFLOW_SNAPSHOT_FORMAT,
            "schema_version": PREFLOW_SNAPSHOT_SCHEMA_VERSION,
            "grid_shape": [
                int(size) for size in stable_snapshot.fields["pressure"].shape
            ],
            "identity": _identity_payload(stable_snapshot.identity),
            "fields": _field_manifest(stable_snapshot.fields),
            "history": stable_snapshot.history,
            "velocity_dirichlet_boundary_authority": (
                stable_snapshot.velocity_dirichlet_boundary_authority
            ),
            "velocity_dirichlet_component_ledger_generation": int(
                stable_snapshot.velocity_dirichlet_component_ledger_generation
            ),
            "npz_file": files.npz_path.name,
            "npz_sha256": _sha256_file(temporary_npz),
        }
        manifest["manifest_sha256"] = canonical_config_sha256(manifest)
        metadata_bytes = _canonical_json_bytes(manifest, context="manifest")
        _write_bytes(temporary_metadata, metadata_bytes)

        os.replace(temporary_npz, files.npz_path)
        pending_paths.discard(temporary_npz)
        os.replace(temporary_metadata, files.metadata_path)
        pending_paths.discard(temporary_metadata)
    finally:
        for temporary_path in pending_paths:
            _best_effort_unlink(temporary_path)
    return files


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON key {key!r}")
        parsed[key] = value
    return parsed


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        metadata_bytes = path.read_bytes()
        parsed = json.loads(
            metadata_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except FileNotFoundError as exc:
        raise PreflowSnapshotIntegrityError(
            f"snapshot metadata file is missing: {path}"
        ) from exc
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise PreflowSnapshotIntegrityError(
            f"snapshot metadata is not valid UTF-8 JSON: {path}: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise PreflowSnapshotIntegrityError("snapshot metadata root must be an object")
    schema_version = parsed.get("schema_version")
    if schema_version == _LEGACY_PREFLOW_SNAPSHOT_SCHEMA_VERSION:
        required_manifest_fields = _LEGACY_SCHEMA_V5_MANIFEST_FIELD_NAMES
    elif schema_version in {
        _CANONICAL_LEDGER_PREFLOW_SNAPSHOT_SCHEMA_VERSION,
        _DIRECTED_BOUNDARY_PREFLOW_SNAPSHOT_SCHEMA_VERSION,
        PREFLOW_SNAPSHOT_SCHEMA_VERSION,
    }:
        required_manifest_fields = _MANIFEST_FIELD_NAMES
    elif set(parsed) == _LEGACY_SCHEMA_V5_MANIFEST_FIELD_NAMES:
        required_manifest_fields = _LEGACY_SCHEMA_V5_MANIFEST_FIELD_NAMES
    else:
        required_manifest_fields = _MANIFEST_FIELD_NAMES
    provided_keys = set(parsed)
    if provided_keys != required_manifest_fields:
        missing = sorted(required_manifest_fields - provided_keys)
        unexpected = sorted(provided_keys - required_manifest_fields)
        raise PreflowSnapshotIntegrityError(
            "snapshot metadata fields do not match the schema: "
            f"missing={missing}, unexpected={unexpected}"
        )
    claimed_hash = parsed.get("manifest_sha256")
    try:
        _require_sha256(claimed_hash, field_name="manifest_sha256")
        actual_hash = canonical_config_sha256(
            {key: value for key, value in parsed.items() if key != "manifest_sha256"}
        )
    except (TypeError, ValueError) as exc:
        raise PreflowSnapshotIntegrityError(f"invalid snapshot manifest: {exc}") from exc
    if claimed_hash != actual_hash:
        raise PreflowSnapshotIntegrityError(
            "snapshot manifest content hash does not match its metadata"
        )
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise PreflowSnapshotIntegrityError(
            "snapshot schema version must be an integer"
        )
    return parsed, hashlib.sha256(metadata_bytes).hexdigest()


def _manifest_identity(manifest: Mapping[str, Any]) -> PreflowSnapshotIdentity:
    payload = manifest.get("identity")
    if not isinstance(payload, Mapping) or set(payload) != set(_IDENTITY_FIELD_NAMES):
        raise PreflowSnapshotIntegrityError("snapshot identity fields are invalid")
    try:
        return PreflowSnapshotIdentity(
            **{name: payload[name] for name in _IDENTITY_FIELD_NAMES}
        )
    except (TypeError, ValueError) as exc:
        raise PreflowSnapshotIntegrityError(f"snapshot identity is invalid: {exc}") from exc


def _require_identity_match(
    stored: PreflowSnapshotIdentity,
    expected: PreflowSnapshotIdentity,
) -> None:
    for field_name in _IDENTITY_FIELD_NAMES:
        stored_value = getattr(stored, field_name)
        expected_value = getattr(expected, field_name)
        if stored_value != expected_value:
            raise PreflowSnapshotMismatchError(
                f"snapshot {field_name} does not match the current model: "
                f"stored={stored_value}, expected={expected_value}"
            )


def _manifest_grid_shape(manifest: Mapping[str, Any]) -> tuple[int, int, int]:
    value = manifest.get("grid_shape")
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in value)
    ):
        raise PreflowSnapshotIntegrityError(
            "snapshot grid shape must contain three positive integers"
        )
    return tuple(value)


def _manifest_npz_path(
    base_files: PreflowSnapshotFiles,
    manifest: Mapping[str, Any],
) -> Path:
    filename = manifest.get("npz_file")
    if not isinstance(filename, str) or not filename:
        raise PreflowSnapshotIntegrityError("snapshot npz_file pointer is invalid")
    relative_path = Path(filename)
    expected_prefix = f"{base_files.npz_path.stem}."
    if (
        relative_path.name != filename
        or relative_path.suffix.lower() != ".npz"
        or not filename.startswith(expected_prefix)
    ):
        raise PreflowSnapshotIntegrityError(
            f"snapshot npz_file pointer is outside the expected generation: {filename!r}"
        )
    return base_files.metadata_path.parent / filename


def _validate_manifest_fields(
    manifest: Mapping[str, Any],
    *,
    grid_shape: tuple[int, int, int],
    field_names: Sequence[str],
) -> Mapping[str, Any]:
    fields = manifest.get("fields")
    if not isinstance(fields, Mapping):
        raise PreflowSnapshotIntegrityError("snapshot field metadata must be an object")
    provided = set(fields)
    required = set(field_names)
    if provided != required:
        raise PreflowSnapshotIntegrityError(
            "snapshot field metadata does not match required fields: "
            f"missing={sorted(required - provided)}, unexpected={sorted(provided - required)}"
        )
    for name in field_names:
        metadata = fields[name]
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "shape",
            "dtype",
            "sha256",
        }:
            raise PreflowSnapshotIntegrityError(
                f"snapshot field {name!r} metadata is invalid"
            )
        expected_shape = list(_expected_shape(name, grid_shape))
        if metadata.get("shape") != expected_shape:
            raise PreflowSnapshotIntegrityError(
                f"snapshot field {name!r} shape metadata does not match schema: "
                f"{metadata.get('shape')!r} != {expected_shape!r}"
            )
        expected_dtype = _FIELD_DTYPES[name].str
        if metadata.get("dtype") != expected_dtype:
            raise PreflowSnapshotIntegrityError(
                f"snapshot field {name!r} dtype metadata does not match schema: "
                f"{metadata.get('dtype')!r} != {expected_dtype!r}"
            )
        try:
            _require_sha256(metadata.get("sha256"), field_name=f"fields.{name}.sha256")
        except (TypeError, ValueError) as exc:
            raise PreflowSnapshotIntegrityError(str(exc)) from exc
    return fields


def _read_npz_fields(
    path: Path,
    *,
    expected_file_sha256: str,
    field_names: Sequence[str],
) -> dict[str, np.ndarray]:
    try:
        _require_sha256(expected_file_sha256, field_name="npz_sha256")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            actual_file_sha256 = digest.hexdigest()
            if actual_file_sha256 != expected_file_sha256:
                raise PreflowSnapshotIntegrityError(
                    "snapshot NPZ content hash does not match its manifest"
                )
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as archive:
                provided = set(archive.files)
                required = set(field_names)
                if provided != required or len(archive.files) != len(provided):
                    raise PreflowSnapshotIntegrityError(
                        "snapshot NPZ fields do not match the schema: "
                        f"missing={sorted(required - provided)}, "
                        f"unexpected={sorted(provided - required)}"
                    )
                return {
                    name: archive[name]
                    for name in field_names
                }
    except PreflowSnapshotIntegrityError:
        raise
    except FileNotFoundError as exc:
        raise PreflowSnapshotIntegrityError(f"snapshot NPZ file is missing: {path}") from exc
    except (OSError, TypeError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise PreflowSnapshotIntegrityError(f"snapshot NPZ cannot be read safely: {exc}") from exc


def load_preflow_snapshot(
    path: str | os.PathLike[str],
    *,
    expected_identity: PreflowSnapshotIdentity,
    expected_velocity_dirichlet_boundary_authority: str = "legacy",
) -> PreflowSnapshot:
    """Load a snapshot only when integrity and current-model identity both match."""

    if not isinstance(expected_identity, PreflowSnapshotIdentity):
        raise TypeError("expected_identity must be a PreflowSnapshotIdentity")
    expected_authority = _validated_velocity_dirichlet_boundary_authority(
        expected_velocity_dirichlet_boundary_authority,
        error_type=PreflowSnapshotValidationError,
    )
    files = _snapshot_files(path)
    manifest, metadata_file_sha256 = _load_manifest(files.metadata_path)
    if manifest.get("format") != PREFLOW_SNAPSHOT_FORMAT:
        raise PreflowSnapshotIntegrityError(
            f"unsupported snapshot format: {manifest.get('format')!r}"
        )
    schema_version = manifest.get("schema_version")
    if schema_version not in {
        _LEGACY_PREFLOW_SNAPSHOT_SCHEMA_VERSION,
        _CANONICAL_LEDGER_PREFLOW_SNAPSHOT_SCHEMA_VERSION,
        _DIRECTED_BOUNDARY_PREFLOW_SNAPSHOT_SCHEMA_VERSION,
        PREFLOW_SNAPSHOT_SCHEMA_VERSION,
    }:
        raise PreflowSnapshotIntegrityError(
            f"unsupported snapshot schema version: {schema_version!r}"
        )
    if schema_version == _LEGACY_PREFLOW_SNAPSHOT_SCHEMA_VERSION:
        if expected_authority != "legacy":
            raise PreflowSnapshotMismatchError(
                "schema-5 preflow snapshots are legacy-authority only; old "
                "scalar rows are never broadcast into the canonical ledger"
            )
        stored_authority = "legacy"
        stored_generation = 0
        field_names = _LEGACY_SCHEMA_V5_FIELD_NAMES
    else:
        stored_authority = _validated_velocity_dirichlet_boundary_authority(
            manifest.get("velocity_dirichlet_boundary_authority"),
            error_type=PreflowSnapshotIntegrityError,
        )
        stored_generation = (
            _validated_velocity_dirichlet_component_ledger_generation(
                manifest.get("velocity_dirichlet_component_ledger_generation"),
                error_type=PreflowSnapshotIntegrityError,
            )
        )
        if schema_version == _CANONICAL_LEDGER_PREFLOW_SNAPSHOT_SCHEMA_VERSION:
            field_names = _SCHEMA_V6_FIELD_NAMES
            if stored_authority == "canonical":
                raise PreflowSnapshotMismatchError(
                    "schema-6 canonical preflow snapshots cannot recover the "
                    "directed external boundary planes without aliasing the "
                    "compact backward-MAC state"
                )
        elif schema_version == _DIRECTED_BOUNDARY_PREFLOW_SNAPSHOT_SCHEMA_VERSION:
            field_names = _SCHEMA_V7_FIELD_NAMES
        else:
            field_names = PREFLOW_SNAPSHOT_FIELD_NAMES
        if stored_authority != expected_authority:
            raise PreflowSnapshotMismatchError(
                "snapshot velocity Dirichlet boundary authority does not match "
                f"the current solver: stored={stored_authority!r}, "
                f"expected={expected_authority!r}"
            )

    stored_identity = _manifest_identity(manifest)
    _require_identity_match(stored_identity, expected_identity)
    grid_shape = _manifest_grid_shape(manifest)
    field_metadata = _validate_manifest_fields(
        manifest,
        grid_shape=grid_shape,
        field_names=field_names,
    )
    generation_npz_path = _manifest_npz_path(files, manifest)
    arrays = _read_npz_fields(
        generation_npz_path,
        expected_file_sha256=manifest.get("npz_sha256"),
        field_names=field_names,
    )

    for name in field_names:
        array = arrays[name]
        metadata = field_metadata[name]
        if tuple(array.shape) != tuple(metadata["shape"]):
            raise PreflowSnapshotIntegrityError(
                f"snapshot field {name!r} shape does not match its manifest"
            )
        if array.dtype.str != metadata["dtype"]:
            raise PreflowSnapshotIntegrityError(
                f"snapshot field {name!r} dtype does not match its manifest"
            )
        if not bool(np.all(np.isfinite(array))):
            raise PreflowSnapshotIntegrityError(
                f"snapshot field {name!r} must contain only finite values"
            )
        actual_content_sha256 = canonical_geometry_sha256({name: array})
        if actual_content_sha256 != metadata["sha256"]:
            raise PreflowSnapshotIntegrityError(
                f"snapshot field {name!r} content hash does not match its manifest"
            )
    if schema_version == _LEGACY_PREFLOW_SNAPSHOT_SCHEMA_VERSION:
        arrays = {
            **arrays,
            **_neutral_canonical_ledger_fields(grid_shape),
            **_neutral_directed_external_boundary_fields(grid_shape),
            **_neutral_laminar_sst_fields(grid_shape),
        }
    elif schema_version == _CANONICAL_LEDGER_PREFLOW_SNAPSHOT_SCHEMA_VERSION:
        arrays = {
            **arrays,
            **_neutral_directed_external_boundary_fields(grid_shape),
            **_neutral_laminar_sst_fields(grid_shape),
        }
    elif schema_version == _DIRECTED_BOUNDARY_PREFLOW_SNAPSHOT_SCHEMA_VERSION:
        arrays = {
            **arrays,
            **_neutral_laminar_sst_fields(grid_shape),
        }

    try:
        return PreflowSnapshot(
            fields=arrays,
            identity=stored_identity,
            history=manifest.get("history"),
            velocity_dirichlet_boundary_authority=stored_authority,
            velocity_dirichlet_component_ledger_generation=stored_generation,
            artifact_identity={
                "metadata_file_sha256": metadata_file_sha256,
                "manifest_sha256": manifest["manifest_sha256"],
                "npz_file": generation_npz_path.name,
                "npz_sha256": manifest["npz_sha256"],
            },
        )
    except PreflowSnapshotValidationError as exc:
        raise PreflowSnapshotIntegrityError(
            f"snapshot payload failed schema validation: {exc}"
        ) from exc


def inspect_preflow_snapshot(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate a snapshot through the production loader and return its byte identity."""

    files = _snapshot_files(path)
    manifest, _metadata_file_sha256 = _load_manifest(files.metadata_path)
    stored_identity = _manifest_identity(manifest)
    schema_version = manifest.get("schema_version")
    stored_authority = (
        "legacy"
        if schema_version == _LEGACY_PREFLOW_SNAPSHOT_SCHEMA_VERSION
        else manifest.get("velocity_dirichlet_boundary_authority")
    )
    snapshot = load_preflow_snapshot(
        files,
        expected_identity=stored_identity,
        expected_velocity_dirichlet_boundary_authority=stored_authority,
    )
    if snapshot.artifact_identity is None:  # pragma: no cover - loader invariant
        raise PreflowSnapshotIntegrityError(
            "loaded snapshot is missing its artifact identity"
        )
    artifact_identity = dict(snapshot.artifact_identity)
    return {
        "prefix": str(Path(files.snapshot_path).expanduser().resolve()),
        "manifest_sha256": artifact_identity["metadata_file_sha256"],
        "npz_file": artifact_identity["npz_file"],
        "npz_sha256": artifact_identity["npz_sha256"],
        "identity": _identity_payload(snapshot.identity),
        "artifact_identity": artifact_identity,
    }


__all__ = [
    "PREFLOW_SNAPSHOT_FIELD_NAMES",
    "PREFLOW_SNAPSHOT_FORMAT",
    "PREFLOW_SNAPSHOT_SCHEMA_VERSION",
    "PreflowSnapshot",
    "PreflowSnapshotError",
    "PreflowSnapshotFiles",
    "PreflowSnapshotIdentity",
    "PreflowSnapshotIntegrityError",
    "PreflowSnapshotMismatchError",
    "PreflowSnapshotValidationError",
    "canonical_config_sha256",
    "canonical_geometry_sha256",
    "canonical_source_sha256",
    "inspect_preflow_snapshot",
    "load_preflow_snapshot",
    "save_preflow_snapshot",
]
