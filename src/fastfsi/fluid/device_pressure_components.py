"""CUDA measures of the pressure-component right-hand-side compatibility: the per-component means of the unreached / outlet components are judged in one serialized
device kernel instead of host loops over downloaded arrays.  Components beyond the device capacity fall back to the
CPU methods.  As in the CPU methods, an isolated cell without a free face is never made solid here: the outlet
reachability marking stops the run on such a cell (fastfsi.fluid.outlet_reachability).

No `from __future__ import annotations` (Taichi kernel module).
"""
import taichi as ti


@ti.data_oriented
class DevicePressureComponents:
    def __init__(self):
        self.values = ti.field(ti.f64, shape=8)

    @ti.kernel
    def _compatibility(self, owner: ti.template(), count: ti.i32,
                       mapped: ti.i32, tolerance: ti.f64):
        incompatible = 0
        max_mean = ti.cast(0.0, ti.f64)
        max_integral = ti.cast(0.0, ti.f64)
        ti.loop_config(serialize=True)
        for component in range(count):
            active = 1
            if mapped != 0:
                if owner.pressure_solve_component_parent[component] != component or owner.pressure_solve_component_anchored[component] != 0:
                    active = 0
            if active != 0:
                volume = owner.cg_unreached_component_volume[component]
                integral = owner.cg_unreached_component_sum[component]
                abs_integral = ti.abs(integral)
                # Explicit comparisons preserve Python max(prior, NaN).
                if abs_integral > max_integral:
                    max_integral = abs_integral
                if not volume <= 0.0:
                    mean = ti.abs(integral / volume)
                    if mean > max_mean:
                        max_mean = mean
                    if mean > tolerance:
                        incompatible += 1
        self.values[0] = ti.cast(incompatible, ti.f64)
        self.values[1] = max_mean
        self.values[2] = max_integral

    def compatibility(self, owner, count, mapped, tolerance):
        self._compatibility(owner, int(count), int(mapped), float(tolerance))
        packet = self.values.to_numpy()
        owner.last_unreached_incompatible_component_count = int(packet[0])
        owner.last_unreached_component_rhs_mean_max_abs = float(packet[1])
        owner.last_unreached_component_rhs_integral_max_abs = float(packet[2])
        return int(packet[0])


def measure_unreached(self, field, *, component_count=None, mean_abs_tolerance=1.0e-12):
    if component_count is None:
        component_count = int(getattr(self, '_pressure_unreached_component_count', 0))
    count = max(0, int(component_count))
    if count > int(self.pressure_component_capacity):
        from fastfsi.fluid.solver import IncompressibleFluidSolver
        return IncompressibleFluidSolver._measure_unreached_component_rhs_incompatibility(
            self, field, component_count=component_count,
            mean_abs_tolerance=mean_abs_tolerance)
    self.last_unreached_incompatible_component_count = 0
    self.last_unreached_component_rhs_mean_max_abs = 0.0
    self.last_unreached_component_rhs_integral_max_abs = 0.0
    if count <= 0:
        return 0
    self._accumulate_unreached_component_means_kernel(field, count)
    return self._device_pressure_components.compatibility(self, count, False, max(0.0, float(mean_abs_tolerance)))


def measure_outlet(self, field, *, mean_abs_tolerance=1.0e-12):
    if not bool(self._pressure_outlet_nullspace_graph_valid):
        raise self._pressure_operator_graph_missing_or_stale_error(context='pressure-outlet-nullspace-rhs-compatibility')
    self.last_unreached_incompatible_component_count = 0
    self.last_unreached_component_rhs_mean_max_abs = 0.0
    self.last_unreached_component_rhs_integral_max_abs = 0.0
    mapped = self._pressure_outlet_nullspace_graph_context != 'outlet-disconnected'
    requested = (int(self._pressure_outlet_nullspace_source_component_count) if mapped
                 else int(self._pressure_outlet_operator_component_count))
    if requested > int(self.pressure_component_capacity):
        from fastfsi.fluid.solver import IncompressibleFluidSolver
        return IncompressibleFluidSolver._measure_pressure_outlet_nullspace_rhs_incompatibility(
            self, field, mean_abs_tolerance=mean_abs_tolerance)
    if not mapped:
        count = int(self._pressure_outlet_operator_component_count)
        if count <= 0:
            return 0
        self._accumulate_pressure_outlet_operator_component_means_kernel(field, count)
    else:
        count = int(self._pressure_outlet_nullspace_source_component_count)
        if count <= 0:
            return 0
        self._accumulate_pressure_outlet_nullspace_component_means_kernel(field, count)
    return self._device_pressure_components.compatibility(self, count, mapped, max(0.0, float(mean_abs_tolerance)))
