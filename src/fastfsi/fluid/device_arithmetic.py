"""Private f64 device arithmetic of the CUDA pressure-solve and flood controllers: the CG right-hand-side norm,
the residual poll, the rz packet, the exact-residual packet and the flood packet.  Each read observes one complete f64 / i64 packet
after the same solver kernels; the arithmetic matches the CPU host expressions (NaN-preserving max, IEEE finiteness by
bits).

No `from __future__ import annotations` (Taichi kernel module).
"""
import math

import taichi as ti


@ti.data_oriented
class DeviceArithmetic:
    def __init__(self):
        self.values = ti.field(ti.f64, shape=12)
        self.integer_values = ti.field(ti.i64, shape=4)
        self.rhs_norm_value = ti.field(ti.f64, shape=())
        self.dot1 = ti.field(ti.f64, shape=())
        self.dot2 = ti.field(ti.f64, shape=())

    @ti.func
    def finite(self, value):
        return (ti.bit_cast(ti.cast(value, ti.f64), ti.u64)
                & ti.u64(0x7ff0000000000000)) != ti.u64(0x7ff0000000000000)

    @ti.func
    def nonnegative_sqrt(self, value):
        # Python max(value, 0.0) leaves a leading NaN unchanged.
        if value < 0.0:
            value = 0.0
        return ti.sqrt(value)

    @ti.kernel
    def _rhs(self, owner: ti.template()):
        self.rhs_norm_value[None] = self.nonnegative_sqrt(self.dot1[None])
        self.values[0] = self.rhs_norm_value[None]

    @ti.kernel
    def _poll(self, owner: ti.template(), tolerance: ti.f64):
        norm = self.nonnegative_sqrt(owner.cg_rr[None])
        rhs_norm = self.rhs_norm_value[None]
        self.values[0] = norm
        self.values[1] = norm / rhs_norm
        self.values[2] = ti.cast(owner.cg_breakdown_code[None], ti.f64)
        self.values[3] = owner.cg_breakdown_dAd[None]
        self.values[4] = ti.cast(norm <= tolerance * rhs_norm, ti.f64)
        self.values[5] = ti.cast(self.values[1] <= tolerance, ti.f64)

    @ti.kernel
    def _rz(self, owner: ti.template()):
        value = owner.cg_rz[None]
        self.values[0] = value
        self.values[6] = ti.cast(self.finite(value), ti.f64)

    @ti.kernel
    def _frontier(self, owner: ti.template(), current_count: ti.template(),
                  levels: ti.template()):
        self.integer_values[0] = owner.pressure_reachability_frontier_overflow[None]
        self.integer_values[1] = current_count[None]
        self.integer_values[2] = levels[None]

    def frontier(self, owner, current_count, levels):
        self._frontier(owner, current_count, levels)
        return self.integer_values.to_numpy()

    @ti.kernel
    def _exact(self, owner: ti.template(), tolerance: ti.f64):
        residual = self.nonnegative_sqrt(self.dot1[None])
        rhs = self.nonnegative_sqrt(self.dot2[None])
        ratio = ti.cast(0.0, ti.f64)
        if rhs <= 1.0e-30:
            if not residual <= 1.0e-30:
                ratio = math.inf
        else:
            ratio = residual / rhs
        self.values[0] = residual
        self.values[1] = rhs
        self.values[2] = ratio
        self.values[6] = ti.cast(self.finite(ratio), ti.f64)
        self.values[7] = ti.cast(self.finite(ratio) and ratio <= tolerance, ti.f64)

    def rhs(self, owner):
        owner._weighted_dot_to_field_kernel(owner.cg_z, owner.cg_z, self.dot1)
        self._rhs(owner)
        return float(self.values.to_numpy()[0])

    def poll(self, owner, tolerance):
        self._poll(owner, float(tolerance))
        owner._device_cg_packet = self.values.to_numpy()
        return float(owner._device_cg_packet[0])

    def rz(self, owner):
        self._rz(owner)
        owner._device_cg_packet = self.values.to_numpy()
        return float(owner._device_cg_packet[0])

    def exact(self, owner, tolerance):
        owner._weighted_dot_to_field_kernel(owner.cg_r, owner.cg_r, self.dot1)
        owner._weighted_dot_to_field_kernel(owner.cg_rhs, owner.cg_rhs, self.dot2)
        self._exact(owner, float(tolerance))
        return self.values.to_numpy()
