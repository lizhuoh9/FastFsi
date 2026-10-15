"""The device maximum of two f64 scalars by IEEE bit order: Python max(a, b) semantics (ordered comparison, NaN and both signed zeros are ties, the first argument
kept on a tie) evaluated on the device, so the CUDA conjugate-gradient controllers take their tolerances on the same
path as every other device scalar.

No `from __future__ import annotations` (Taichi kernel module).
"""
from functools import lru_cache
import struct

import taichi as ti


@ti.func
def greater_binary64(left, right):
    """IEEE ordered comparison by bits; NaNs and both signed zeros are ties."""
    a = left & ti.u64(0x7fffffffffffffff)
    b = right & ti.u64(0x7fffffffffffffff)
    result = False
    if a <= ti.u64(0x7ff0000000000000) and b <= ti.u64(0x7ff0000000000000):
        if not (a == 0 and b == 0):
            sa, sb = left >> 63, right >> 63
            if sa != sb:
                result = sa == 0
            elif sa == 0:
                result = a > b
            else:
                result = a < b
    return result


@ti.data_oriented
class _SelectKernels:
    @ti.kernel
    def select_scalar(self, first: ti.u64, second: ti.u64, minimum: ti.i32,
                      output: ti.types.ndarray(ti.u64, ndim=1)):
        selected = first
        replace = greater_binary64(second, first)
        if minimum != 0:
            replace = greater_binary64(first, second)
        if replace:
            selected = second
        output[0] = selected


@lru_cache(maxsize=1)
def _select_kernels():
    # One holder per process; a fresh data_oriented instance per call would recompile the kernel.
    return _SelectKernels()


def _scalar_raw(value):
    # IO only: no float operation, rounding, normalization or NaN rewriting.
    return struct.unpack('<Q', struct.pack('<d', float(value)))[0]


def _scalar_value(value):
    return struct.unpack('<d', struct.pack('<Q', int(value)))[0]


def scalar_maximum(first, second):
    output = ti.ndarray(ti.u64, shape=1)
    _select_kernels().select_scalar(_scalar_raw(first), _scalar_raw(second), 0, output)
    return _scalar_value(output.to_numpy()[0])
