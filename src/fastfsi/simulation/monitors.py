"""Run monitors: hard stops, phase timing, volume-error thresholds and a control-volume force balance.

SimulationHardStop (re-exported from fastfsi.errors) ends a run whose state can no longer be trusted.
PhaseTimer measures the wall time and the kernel launches of the phases of a few steps.
volume_error_thresholds parses a comma-separated list of increasing thresholds in (0, 1).
ControlVolumeForce is the hydrodynamic force on everything inside a fixed (y, z) box of a quasi-2-D channel by a
momentum balance: F = surface integral of (sigma n - rho u u.n) - d/dt (momentum inside the box) + d/dt (solid momentum
inside); measure() returns the surface integrals and the fluid momentum inside, the caller differentiates in time.
PressureRegionMonitor reports the largest |pressure| in the free fluid and, separately, in the region covered by the
immersed bodies (where the single-velocity IB pressure is the body's hydrostatic stress).

No `from __future__ import annotations` (Taichi kernel module).
"""
import math
import time

import taichi as ti

from fastfsi.errors import SimulationHardStop  # noqa: F401  (re-export)


class PhaseTimer:
    """Diagnostic instrument (off unless steps > 0): for steps start+1 .. start+steps every phase is wrapped by
    ti.sync() + perf_counter, and Taichi kernel calls are counted per phase (a wrapper on Kernel.__call__, removed
    afterwards), a proxy for the kernel launches per step.  It perturbs the timing of the instrumented steps only (the
    syncs), never the physics."""

    def __init__(self, start, steps):
        self.start, self.steps = int(start), int(steps)
        self.active = False
        self.times, self.calls, self.counted_steps = {}, {}, 0
        self._count, self._restore = None, None

    def begin(self, step):
        want = self.steps > 0 and self.start < step <= self.start + self.steps
        if want and not self.active:
            from taichi.lang import kernel_impl
            original = kernel_impl.Kernel.__call__
            count = {'n': 0}

            def counted(kernel, *args, **kwargs):
                count['n'] += 1
                return original(kernel, *args, **kwargs)
            kernel_impl.Kernel.__call__ = counted
            self._count = count
            self._restore = lambda: setattr(kernel_impl.Kernel, '__call__', original)
            self.active = True
        elif not want and self.active:
            self._restore()
            self.active = False
        if self.active:
            self.counted_steps += 1

    def phase(self, name, fn, *args, **kwargs):
        if not self.active:
            return fn(*args, **kwargs)
        import taichi as ti
        ti.sync()
        n0, t0 = self._count['n'], time.perf_counter()
        result = fn(*args, **kwargs)
        ti.sync()
        self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - t0
        self.calls[name] = self.calls.get(name, 0) + self._count['n'] - n0
        return result

    def report(self):
        k = max(self.counted_steps, 1)
        rows = {name: dict(ms_per_step=1e3 * self.times[name] / k, kernel_calls_per_step=self.calls[name] / k)
                for name in self.times}
        return dict(steps=self.counted_steps, phases=rows,
                    total_ms_per_step=sum(r['ms_per_step'] for r in rows.values()),
                    total_kernel_calls_per_step=sum(r['kernel_calls_per_step'] for r in rows.values()))


def volume_error_thresholds(text):
    """Volume-error event thresholds from a comma-separated list (finite, positive, strictly increasing)."""
    values = [float(v) for v in str(text).split(',') if v.strip()]
    if not values or not all(math.isfinite(v) and v > 0.0 for v in values) or values != sorted(set(values)):
        raise ValueError("volume-error thresholds must be finite, positive and increasing, got %r" % text)
    return values


@ti.data_oriented
class ControlVolumeForce:
    """Hydrodynamic force on everything inside a fixed (y, z) box (e.g. a cylinder and its beam) by a momentum balance:
    F = surface integral of (sigma n - rho u u.n) - d/dt (momentum inside the box) + d/dt (solid momentum inside)."""

    def __init__(self, fluid, *, j_range, k_range, origin_m, spacing_m, grid_nodes, span_m):
        self.fluid = fluid
        self.j0, self.j1 = (int(v) for v in j_range)
        self.k0, self.k1 = (int(v) for v in k_range)
        self.nx, self.ny, self.nz = (int(v) for v in grid_nodes)
        self.h = tuple(float(v) for v in spacing_m)
        self.span_m = float(span_m)
        self.out = ti.field(ti.f64, shape=4)

    @ti.func
    def _centre_u(self, velocity: ti.template(), j, k):
        """Cell-centre (uy, uz) of cell (0, j, k) from its backward and forward MAC faces."""
        uy = 0.5 * (velocity[0, j, k][1] + velocity[0, ti.min(j + 1, self.ny - 1), k][1])
        uz = 0.5 * (velocity[0, j, k][2] + velocity[0, j, ti.min(k + 1, self.nz - 1)][2])
        return ti.Vector([ti.cast(uy, ti.f64), ti.cast(uz, ti.f64)])

    @ti.kernel
    def _measure_kernel(self, velocity: ti.template(), pressure: ti.template(), obstacle: ti.template(),
                        rho: ti.f64, mu: ti.f64):
        for s in range(4):
            self.out[s] = 0.0
        hy, hz = self.h[1], self.h[2]
        # momentum inside the box (per unit span)
        for j, k in ti.ndrange((self.j0, self.j1), (self.k0, self.k1)):
            if obstacle[0, j, k] == 0:
                u = self._centre_u(velocity, j, k)
                ti.atomic_add(self.out[2], rho * u[0] * hy * hz)
                ti.atomic_add(self.out[3], rho * u[1] * hz * hy)
        # surface integrals: faces z = k0 (normal -z), z = k1 (normal +z), y = j0 (normal -y), y = j1 (normal +y)
        for side, j in ti.ndrange(2, (self.j0, self.j1)):
            k = self.k0 if side == 0 else self.k1 - 1
            n = -1.0 if side == 0 else 1.0
            u = self._centre_u(velocity, j, k)
            kk = ti.max(0, ti.min(k + (1 if side == 1 else -1), self.nz - 1))
            un = self._centre_u(velocity, j, kk)
            dz = n * hz
            duy_dz = (un[0] - u[0]) / dz
            duz_dz = (un[1] - u[1]) / dz
            jp, jm = ti.min(j + 1, self.ny - 1), ti.max(j - 1, 0)
            duz_dy = (self._centre_u(velocity, jp, k)[1] - self._centre_u(velocity, jm, k)[1]) / ((jp - jm) * hy)
            p = ti.cast(pressure[0, j, k], ti.f64)
            # traction sigma.n with n = (0, n): sigma_yz, sigma_zz
            t_y = mu * (duy_dz + duz_dy) * n
            t_z = (-p + 2.0 * mu * duz_dz) * n
            flux = rho * u * (u[1] * n)
            ti.atomic_add(self.out[0], (t_y - flux[0]) * hy)
            ti.atomic_add(self.out[1], (t_z - flux[1]) * hy)
        for side, k in ti.ndrange(2, (self.k0, self.k1)):
            j = self.j0 if side == 0 else self.j1 - 1
            n = -1.0 if side == 0 else 1.0
            u = self._centre_u(velocity, j, k)
            jj = ti.max(0, ti.min(j + (1 if side == 1 else -1), self.ny - 1))
            un = self._centre_u(velocity, jj, k)
            dy = n * hy
            duy_dy = (un[0] - u[0]) / dy
            duz_dy = (un[1] - u[1]) / dy
            kp, km = ti.min(k + 1, self.nz - 1), ti.max(k - 1, 0)
            duy_dz = (self._centre_u(velocity, j, kp)[0] - self._centre_u(velocity, j, km)[0]) / ((kp - km) * hz)
            p = ti.cast(pressure[0, j, k], ti.f64)
            t_y = (-p + 2.0 * mu * duy_dy) * n
            t_z = mu * (duz_dy + duy_dz) * n
            flux = rho * u * (u[0] * n)
            ti.atomic_add(self.out[0], (t_y - flux[0]) * hz)
            ti.atomic_add(self.out[1], (t_z - flux[1]) * hz)

    def measure(self):
        fluid = self.fluid
        self._measure_kernel(fluid.velocity, fluid.pressure, fluid.obstacle, float(fluid.rho), float(fluid.mu))
        r = self.out.to_numpy()
        return dict(surface_y=float(r[0]), surface_z=float(r[1]), momentum_y=float(r[2]), momentum_z=float(r[3]))


@ti.data_oriented
class PressureRegionMonitor:
    """Largest |pressure| in the free fluid and, separately, in the region covered by immersed bodies.

    In the single-velocity immersed-boundary model the Eulerian pressure inside an incompressible body is the body's
    hydrostatic stress, so one pressure field holds two physical quantities: the fluid pressure, and inside and around
    the bodies the solid's isotropic stress (largest at clamped corners, where it is singular in the continuum and grows
    with resolution).  The monitor marks every cell within `reach_cells` cells of the cell holding a material point
    (default 2: the half-width of the IB_4 support) as the body region; the other cells that are neither obstacle nor
    excluded (e.g. a pressure chamber's cells) are the free fluid.  measure() returns max |p + p_shift| over each set
    (non-finite values are counted, not compared), at the check interval only.

    axes: the grid axes with index_coordinate (grid.axes.GradedAxes); points: an object with the point positions `x`
    (n, 3) and the count `n` (solid.material_points.MaterialPoints)."""

    def __init__(self, fluid, axes, points, *, reach_cells=2):
        self.fluid = fluid
        self.axes = axes
        self.points = points
        self.n = int(points.n)
        self.reach = int(reach_cells)
        if self.reach < 0:
            raise ValueError("reach_cells must be >= 0")
        self.shape = tuple(int(v) for v in fluid.grid.grid_nodes)
        if tuple(int(v) for v in axes.n) != self.shape:
            raise ValueError("axes %s do not match the fluid grid %s" % (tuple(axes.n), self.shape))
        self.region = ti.field(ti.i32, shape=self.shape)
        self.no_exclusion = ti.field(ti.i32, shape=self.shape)   # all zero: nothing excluded
        self.out = ti.field(ti.f64, shape=4)

    @ti.kernel
    def _mark_kernel(self, x: ti.template()):
        for i, j, k in self.region:
            self.region[i, j, k] = 0
        for p in range(self.n):
            cell = ti.Vector([0, 0, 0])
            for a in ti.static(range(3)):
                xi, local = self.axes.index_coordinate(a, 1, x[p][a])
                cell[a] = ti.cast(ti.floor(xi), ti.i32)
            for a, b, c in ti.ndrange((-self.reach, self.reach + 1), (-self.reach, self.reach + 1),
                                      (-self.reach, self.reach + 1)):
                i, j, k = cell[0] + a, cell[1] + b, cell[2] + c
                if 0 <= i < self.shape[0] and 0 <= j < self.shape[1] and 0 <= k < self.shape[2]:
                    self.region[i, j, k] = 1

    @ti.kernel
    def _reduce_kernel(self, pressure: ti.template(), obstacle: ti.template(), exclude: ti.template(),
                       p_shift: ti.f32):
        for s in range(4):
            self.out[s] = 0.0
        for i, j, k in pressure:
            if obstacle[i, j, k] == 0 and exclude[i, j, k] == 0:
                value = ti.cast(pressure[i, j, k] + p_shift, ti.f64)
                if ti.math.isnan(value) or ti.math.isinf(value):
                    ti.atomic_add(self.out[3], 1.0)
                elif self.region[i, j, k] != 0:
                    ti.atomic_max(self.out[1], ti.abs(value))
                    ti.atomic_add(self.out[2], 1.0)
                else:
                    ti.atomic_max(self.out[0], ti.abs(value))

    def measure(self, p_shift_pa=0.0, exclude=None):
        """max |p + p_shift| (Pa) over the free fluid and over the body region; exclude: an i32 field of the grid shape
        whose nonzero cells are left out of both (e.g. pressure-chamber cells)."""
        self._mark_kernel(self.points.x)
        self._reduce_kernel(self.fluid.pressure, self.fluid.obstacle,
                            self.no_exclusion if exclude is None else exclude, float(p_shift_pa))
        r = self.out.to_numpy()
        return dict(fluid_pressure_max_pa=float(r[0]), solid_region_pressure_max_pa=float(r[1]),
                    solid_region_cells=int(r[2]), nonfinite_cells=int(r[3]))
