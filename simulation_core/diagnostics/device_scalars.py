"""Batched host reads of 0-D Taichi report fields.

Reading ``field[None]`` launches a reader kernel and synchronizes the device,
once per field.  Report and precommit blocks read dozens of counters after the
same kernels; :func:`read_scalar_fields` gathers them with one launch and one
transfer.  Integer counters and f32/f64 values are exact in f64, so callers
keep their ``int``/``float`` conversions and see identical values.
"""
# No ``from __future__ import annotations``: Taichi evaluates the kernel
# annotations below at decoration time.
from collections.abc import Sequence

import taichi as ti

SCALAR_GATHER_CAPACITY = 128


@ti.kernel
def _gather_scalar_fields_kernel(fields: ti.template(), buffer: ti.template()):
    for index in ti.static(range(len(fields))):
        buffer[index] = ti.cast(fields[index][None], ti.f64)


class HostScalarSnapshot:
    """Host values of 0-D Taichi fields, indexed by the field object."""

    __slots__ = ("_values",)

    def __init__(self, fields: Sequence[object], values: Sequence[float]) -> None:
        self._values = {id(field): value for field, value in zip(fields, values)}

    def __getitem__(self, field: object) -> float:
        return self._values[id(field)]


def read_scalar_fields(owner: object, fields: Sequence[object]) -> HostScalarSnapshot:
    """Snapshot ``fields`` (0-D scalar fields) with one launch and one transfer.

    ``owner`` keeps the gather buffer, so its lifetime follows the Taichi
    objects that own the fields.
    """

    fields = tuple(fields)
    if len(fields) > SCALAR_GATHER_CAPACITY:
        raise ValueError(
            f"{len(fields)} scalar fields exceed the gather capacity "
            f"{SCALAR_GATHER_CAPACITY}"
        )
    buffer = getattr(owner, "_scalar_gather_buffer", None)
    if buffer is None:
        buffer = ti.field(dtype=ti.f64, shape=SCALAR_GATHER_CAPACITY)
        owner._scalar_gather_buffer = buffer
    _gather_scalar_fields_kernel(fields, buffer)
    values = buffer.to_numpy()[: len(fields)]
    return HostScalarSnapshot(fields, values.tolist())
