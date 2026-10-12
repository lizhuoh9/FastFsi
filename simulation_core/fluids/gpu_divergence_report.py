"""CUDA arithmetic for the existing divergence report snapshots.

The original reduction, every raw report slot, decoding order and error remain
authority. Private output never overwrites the original saved report fields.
CPU owners continue to use the original host methods.
"""
import math
import taichi as ti


class DeviceSnapshot(tuple):
    def __new__(cls, raw, norms, negative):
        result = super().__new__(cls, raw)
        result.norms = tuple(norms)
        result.negative = tuple(negative)
        return result


@ti.data_oriented
class DivergenceReport:
    def __init__(self):
        self.packet = ti.field(ti.f64, shape=120)

    @ti.func
    def finite(self, value):
        return (ti.bit_cast(value, ti.u64) & ti.u64(0x7ff0000000000000)
                ) != ti.u64(0x7ff0000000000000)

    @ti.kernel
    def collect(self, source: ti.template(), groups: ti.template(),
                flat_vector: ti.template()):
        for i in range(120):
            self.packet[i] = 0.0
        for group in range(groups):
            offset = group * 3
            peak = ti.cast(0.0, ti.f64)
            square = ti.cast(0.0, ti.f64)
            count = ti.cast(0.0, ti.f64)
            if ti.static(flat_vector):
                peak = source[None][offset]
                square = source[None][offset + 1]
                count = source[None][offset + 2]
            else:
                peak = source[offset // 24][offset % 24]
                square = source[(offset + 1) // 24][(offset + 1) % 24]
                count = source[(offset + 2) // 24][(offset + 2) % 24]
            self.packet[offset] = peak
            self.packet[offset + 1] = square
            self.packet[offset + 2] = count
            denominator = ti.cast(1, ti.i64)
            # Real reduction counts are i32 values stored in f64. Invalid
            # injected counts are kept raw for Python int's original error.
            if self.finite(count) and ti.abs(count) < 9223372036854775808.0:
                denominator = ti.cast(count, ti.i64)
                if denominator < 1:
                    denominator = 1
            quotient = square / ti.cast(denominator, ti.f64)
            self.packet[72 + group] = ti.sqrt(quotient)
            self.packet[96 + group] = ti.cast(quotient < 0.0, ti.f64)

    def snapshot(self, source, groups, flat_vector):
        self.collect(source, int(groups), bool(flat_vector))
        packet = self.packet.to_numpy()
        return DeviceSnapshot(tuple(float(v) for v in packet[:groups * 3]),
                              packet[72:72 + groups], packet[96:96 + groups])


def decode(self, snapshot, offset):
    if not isinstance(snapshot, DeviceSnapshot):
        from .solver import CartesianFluidSolver
        return CartesianFluidSolver._decode_divergence_report_snapshot(snapshot, offset)
    actual_count = int(snapshot[offset + 2])
    group = int(offset) // 3
    # Preserve the original first decoder which raises math-domain error.
    if snapshot.negative[group]:
        raise ValueError('math domain error')
    if abs(actual_count) >= 2**63:
        # Outside the original i32 count representation: retain the old
        # diagnostic behavior rather than silently narrowing the count.
        from .solver import CartesianFluidSolver
        return CartesianFluidSolver._decode_divergence_report_snapshot(snapshot, offset)
    return {'max_abs': float(snapshot[offset]), 'l2': float(snapshot.norms[group]),
            'count': actual_count}


def read_final(self):
    return self._h2_divergence_report.snapshot(self.divergence_final_report_snapshot,
                                              24, False)


def raw_stats(self, *, interior_only=False):
    self._divergence_stats_kernel(1 if interior_only else 0)
    self.last_divergence_report_host_reads += 1
    values = self._h2_divergence_report.snapshot(self.divergence_report_snapshot, 1, True)
    result = decode(self, values, 0)
    return {'max_abs': result['max_abs'], 'l2': result['l2']}


def residual_stats(self, *, interior_only=False):
    self._divergence_residual_stats_kernel(
        1 if interior_only else 0,
        1 if int(self._hibm_pressure_unreached_count) > 0 else 0)
    self.last_divergence_report_host_reads += 1
    values = self._h2_divergence_report.snapshot(self.divergence_report_snapshot, 1, True)
    result = decode(self, values, 0)
    return {'max_abs': result['max_abs'], 'l2': result['l2']}


def partition_stats(self):
    self._divergence_dirichlet_partition_report_kernel()
    self.last_divergence_report_host_reads += 1
    values = self._h2_divergence_report.snapshot(self.divergence_dirichlet_partition_snapshot,
                                              4, True)
    return tuple(decode(self, values, offset) for offset in (0, 3, 6, 9))
