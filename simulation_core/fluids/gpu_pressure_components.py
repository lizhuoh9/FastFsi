"""Ordered CUDA interpretation of the original component budget arrays.

Original device accumulators and physical component labels are retained.
Only the host array arithmetic is moved; dictionary construction remains IO.
"""
import math
import taichi as ti


@ti.data_oriented
class PressureComponents:
    def __init__(self):
        self.values = ti.field(ti.f64, shape=8)
        self.raw = None
        self.rows = None
        self.capacity = 0

    @ti.func
    def finite(self, value):
        return (ti.bit_cast(ti.cast(value, ti.f64), ti.u64)
                & ti.u64(0x7ff0000000000000)) != ti.u64(0x7ff0000000000000)

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

    @ti.kernel
    def _save_raw(self, owner: ti.template(), count: ti.i32):
        for component in range(count):
            self.raw[component] = owner.cg_unreached_component_sum[component]

    @ti.kernel
    def _physical(self, owner: ti.template(), count: ti.i32,
                  tolerance: ti.f64):
        incompatible = 0
        volume_total = ti.cast(0.0, ti.f64)
        integral_total = ti.cast(0.0, ti.f64)
        ti.loop_config(serialize=True)
        for component in range(count):
            volume = owner.cg_unreached_component_volume[component]
            integral = owner.cg_unreached_component_sum[component]
            self.rows[component] = ti.Vector([volume, integral, 0.0, 0.0])
            if not volume <= 0.0:
                mean = integral / volume
                self.rows[component][2] = mean
                self.rows[component][3] = 1.0
                if not self.finite(self.raw[component]) or not self.finite(mean) or ti.abs(mean) > tolerance:
                    incompatible += 1
                volume_total += volume
                integral_total += integral
        self.values[0] = ti.cast(incompatible, ti.f64)
        self.values[1] = integral_total
        self.values[2] = 0.0
        if volume_total > 0.0:
            self.values[2] = integral_total / volume_total

    def compatibility(self, owner, count, mapped, tolerance):
        self._compatibility(owner, int(count), int(mapped), float(tolerance))
        packet = self.values.to_numpy()
        owner.last_hibm_unreached_incompatible_component_count = int(packet[0])
        owner.last_hibm_unreached_component_rhs_mean_max_abs = float(packet[1])
        owner.last_hibm_unreached_component_rhs_integral_max_abs = float(packet[2])
        return int(packet[0])

    def physical_prepare(self, owner, count):
        capacity = int(owner.pressure_component_capacity)
        if self.raw is None:
            # Same immutable per-solver capacity as the existing component arrays.
            self.raw = ti.field(ti.f64, shape=max(1, capacity))
            self.rows = ti.Vector.field(4, ti.f64, shape=max(1, capacity))
            self.capacity = capacity
        if capacity != self.capacity or not 0 <= count <= capacity:
            raise RuntimeError('pressure component snapshot capacity changed')
        self._save_raw(owner, int(count))

    def physical_publish(self, owner, count, tolerance, excluded=frozenset()):
        self._physical(owner, int(count), float(tolerance))
        packet = self.values.to_numpy()
        data = self.rows.to_numpy()[:count]
        if excluded:
            # REQ_373: components holding R1b-removed cells are solid for the projection (host recount, same tests).
            raw = self.raw.to_numpy()[:count]
            kept = [(c, row) for c, row in enumerate(data) if row[3] != 0.0 and c not in excluded]
            incompatible = sum(1 for c, row in kept if not math.isfinite(float(raw[c]))
                               or not math.isfinite(float(row[2])) or abs(float(row[2])) > tolerance)
            volume = sum(float(row[0]) for c, row in kept)
            integral = sum(float(row[1]) for c, row in kept)
            packet = packet.copy()
            packet[0], packet[1], packet[2] = incompatible, integral, (integral / volume if volume > 0.0 else 0.0)
            data = [row * 0.0 if c in excluded else row for c, row in enumerate(data)]
        budgets = tuple(dict(component=c, volume_m3=float(row[0]),
                             rhs_integral=float(row[1]), rhs_mean=float(row[2]))
                        for c, row in enumerate(data) if row[3] != 0.0)
        owner.last_cg_closed_neumann_physical_rhs_measured = True
        owner.last_cg_closed_neumann_physical_rhs_integral = float(packet[1])
        owner.last_cg_closed_neumann_physical_rhs_mean = float(packet[2])
        owner.last_cg_closed_neumann_physical_component_budgets = budgets
        owner.last_hibm_unreached_incompatible_component_count = int(packet[0])
        return int(packet[0])


def measure_unreached(self, field, *, component_count=None, mean_abs_tolerance=1.0e-12):
    if component_count is None:
        component_count = int(getattr(self, '_hibm_pressure_unreached_component_count', 0))
    count = max(0, int(component_count))
    if count > int(self.pressure_component_capacity):
        from .solver import CartesianFluidSolver
        return CartesianFluidSolver._measure_unreached_component_rhs_incompatibility(
            self, field, component_count=component_count,
            mean_abs_tolerance=mean_abs_tolerance)
    self.last_hibm_unreached_incompatible_component_count = 0
    self.last_hibm_unreached_component_rhs_mean_max_abs = 0.0
    self.last_hibm_unreached_component_rhs_integral_max_abs = 0.0
    if count <= 0:
        return 0
    self._accumulate_unreached_component_means_kernel(field, count)
    if bool(getattr(self, 'no_free_face_isolated_cells_solid', False)):
        # REQ_373 R1b-U on the CUDA flood branch: the same host rule; removed singletons re-accumulate to zero.
        volumes = self.cg_unreached_component_volume.to_numpy()[:count]
        integrals = self.cg_unreached_component_sum.to_numpy()[:count]
        if self._remove_no_free_face_isolated_singletons(field, count, integrals, volumes, flood_only=True):
            self._accumulate_unreached_component_means_kernel(field, count)
    return self._h2_pressure_components.compatibility(self, count, False, max(0.0, float(mean_abs_tolerance)))


def measure_outlet(self, field, *, mean_abs_tolerance=1.0e-12):
    if not bool(self._pressure_outlet_nullspace_graph_valid):
        raise self._pressure_operator_graph_missing_or_stale_error(context='pressure-outlet-nullspace-rhs-compatibility')
    self.last_hibm_unreached_incompatible_component_count = 0
    self.last_hibm_unreached_component_rhs_mean_max_abs = 0.0
    self.last_hibm_unreached_component_rhs_integral_max_abs = 0.0
    mapped = self._pressure_outlet_nullspace_graph_context != 'outlet-disconnected'
    requested = (int(self._pressure_outlet_nullspace_source_component_count) if mapped
                 else int(self._pressure_outlet_operator_component_count))
    if requested > int(self.pressure_component_capacity):
        from .solver import CartesianFluidSolver
        return CartesianFluidSolver._measure_pressure_outlet_nullspace_rhs_incompatibility(
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
    if bool(getattr(self, 'no_free_face_isolated_cells_solid', False)):
        # REQ_373: R1b on the CUDA operator branch, both graph contexts, exactly as the CPU method.
        volumes = self.cg_unreached_component_volume.to_numpy()[:count]
        integrals = self.cg_unreached_component_sum.to_numpy()[:count]
        if not mapped:
            removed = self._remove_no_free_face_isolated_singletons(field, count, integrals, volumes)
        else:
            removed = self._remove_no_free_face_isolated_singletons(
                field, count, integrals, volumes,
                parents=self.pressure_solve_component_parent.to_numpy()[:count],
                anchored=self.pressure_solve_component_anchored.to_numpy()[:count])
        if removed:
            if not mapped:
                self._accumulate_pressure_outlet_operator_component_means_kernel(field, count)
            else:
                self._accumulate_pressure_outlet_nullspace_component_means_kernel(field, count)
    return self._h2_pressure_components.compatibility(self, count, mapped, max(0.0, float(mean_abs_tolerance)))


def measure_physical(self, *, component_count, mapped_components, rhs_scale,
                     velocity_inlet_zmax_mode, mean_abs_tolerance=1.0e-12):
    count = int(component_count)
    if not 0 <= count <= int(self.pressure_component_capacity):
        from .solver import CartesianFluidSolver
        return CartesianFluidSolver._measure_closed_component_physical_rhs_incompatibility(
            self, component_count=component_count, mapped_components=mapped_components,
            rhs_scale=rhs_scale, velocity_inlet_zmax_mode=velocity_inlet_zmax_mode,
            mean_abs_tolerance=mean_abs_tolerance)
    self._h2_pressure_components.physical_prepare(self, count)
    self._accumulate_closed_component_physical_rhs_kernel(count, bool(mapped_components), float(rhs_scale), int(velocity_inlet_zmax_mode), self._velocity_dirichlet_boundary_authority_code())
    excluded = self._no_free_face_closed_components(count, bool(mapped_components))
    return self._h2_pressure_components.physical_publish(self, count, max(0.0, float(mean_abs_tolerance)),
                                                         excluded=excluded)


# REQ_373 runtime coverage guard marker (= simulation_core.fluids.solver.NO_FREE_FACE_RULE_REVISION).
for _function in (measure_unreached, measure_outlet, measure_physical):
    _function.no_free_face_rule = "REQ369-R1b-v2-all-sites"
del _function
