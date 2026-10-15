"""PneumaticChamber (boundary/pneumatic_chamber.py) on the real solver: a membrane slab of lattice points under the lid of
an 8 x 8 x 32 duct covers a 2 mm inlet disc, then moves down and the inlet reconnects; plus the
gas-spring oscillator behind the pocket stiffness bound.

1. Covered inlet: the traction p_in A_hole acts downward on the top-surface points over the disc (total force exact,
   only tagged points near the footprint carry it); the inlet is not connected, no cell is Dirichlet air.
2. The slab moves 1.5 mm down: the top cells stay material for N_h - 1 accepts (hysteresis), then the inlet reconnects:
   the cells between the lid and the slab are Dirichlet air (air_cell) and the traction stops.
3. A lagged (explicit) gas spring, p from the accepted volume: frequency sqrt(p A^2 / (V m)) to second order in
   omega dt, bounded energy at omega dt = 0.45 (the bound with its margin), growth at omega dt > 2."""
import math

import numpy as np
import pytest

from fastfsi.boundary.pneumatic_chamber import PneumaticChamber
from fastfsi.fluid.solver import IncompressibleFluidSolver
from fastfsi.grid.cartesian_grid import CartesianGrid
from fastfsi.grid.specification import FluidDomainSpecification
from fastfsi.simulation.taichi_runtime import TaichiRuntimeConfig
from fastfsi.grid.axes import GradedAxes
from fastfsi.point_tags import TAG_CHAMBER_SURFACE
from fastfsi.solid.material_points import MaterialPoints
from fastfsi.solid.materials import NeoHookeanMaterial

RHO, MU, DT = 1000.0, 1.0e-3, 1.0e-4
NX, NY, NZ = 8, 8, 32
S_XY, S_Z = 0.5e-3, 0.3e-3
_STATE = {}


def duct_grid():
    widths_z = tuple(6.0e-4 * 1.05 ** abs(k - 16) for k in range(NZ))
    return CartesianGrid(bounds_min_m=(0.0, 0.0, 0.98), cell_widths_x_m=(1.0e-3,) * NX,
                         cell_widths_y_m=(1.0e-3,) * NY, cell_widths_z_m=widths_z)


def rig():
    if "chamber" not in _STATE:
        from cpu_runtime import initialize
        initialize()
        grid = duct_grid()
        spec = FluidDomainSpecification(bounds_min_m=grid.bounds_min_m, bounds_max_m=grid.bounds_max_m,
                                        grid_nodes=grid.grid_nodes, density_kgm3=RHO, viscosity_pa_s=MU, dt_s=DT,
                                        cartesian_grid=grid)
        fluid = IncompressibleFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cpu", default_fp="f32"))
        fluid.obstacle.from_numpy(np.zeros(grid.grid_nodes, np.int32))
        faces = [np.asarray(f) for f in (grid.cell_faces_x_m, grid.cell_faces_y_m, grid.cell_faces_z_m)]
        top = faces[2][-1]
        i, j, k = np.meshgrid(np.arange(16), np.arange(16), np.arange(2), indexing="ij")
        positions = np.stack(((i.ravel() + 0.5) * S_XY, (j.ravel() + 0.5) * S_XY,
                              top - (1.5 - k.ravel()) * S_Z), axis=1)          # layer k = 1's boxes touch the lid
        tags = np.where(k.ravel() == 1, TAG_CHAMBER_SURFACE, 0).astype(np.int32)
        volumes = np.full(len(positions), S_XY * S_XY * S_Z)
        points = MaterialPoints(positions_m=positions.astype(np.float32), volumes_m3=volumes,
                                material=NeoHookeanMaterial(1000.0, 1.0, 10.0), tether_mask=np.zeros(len(positions)),
                                tag=tags)
        chamber = PneumaticChamber(fluid, GradedAxes(*faces), points, body_indices=np.arange(len(positions)),
                                     reference_positions_m=positions,
                                     spacing_m=np.tile([S_XY, S_XY, S_Z], (len(positions), 1)),
                                     point_mass_kg=volumes * 1040.0, axis_xy_m=(4.0e-3, 4.0e-3), radius_m=3.5e-3,
                                     lid_z_m=top, box_z_lo_m=0.99, inlet_centre_xy_m=(4.0e-3, 4.0e-3),
                                     inlet_diameter_m=2.0e-3, hysteresis_steps=3, staging=1024)
        _STATE.update(points=points, chamber=chamber, positions=positions)
    return _STATE


def move(points, positions, dz):
    u = np.zeros_like(positions)
    u[:, 2] = dz
    points.u.from_numpy(u.astype(np.float32))
    points.x.from_numpy((positions + u).astype(np.float32))
    points.F.from_numpy(np.tile(np.eye(3, dtype=np.float32), (len(positions), 1, 1)))


def test_covered_inlet_traction_and_reconnection():
    s = rig()
    points, chamber, positions = s["points"], s["chamber"], s["positions"]
    move(points, positions, 0.0)
    out = chamber.accept(0.0, 0.0)
    assert out["inlet_covered"] and out["pockets"] == 0
    assert not chamber.air_cell.to_numpy().any()
    points.f_extra.fill(0.0)
    assert chamber.apply_inlet_traction(10.0e3)
    force = points.f_extra.to_numpy().astype(np.float64)
    assert force[:, 2].sum() == pytest.approx(-10.0e3 * math.pi * 1.0e-6, rel=1e-5)
    assert np.all(force[:, :2] == 0.0)
    carriers = np.flatnonzero(force[:, 2] != 0.0)
    xy = positions[carriers, :2] - 4.0e-3
    assert np.all(points.tag.to_numpy()[carriers] == TAG_CHAMBER_SURFACE)
    assert np.all(np.hypot(xy[:, 0], xy[:, 1]) <= 1.0e-3 + 0.5 * math.sqrt(2.0) * S_XY)
    move(points, positions, -1.5e-3)
    covered = [chamber.accept(1.0e-3 * (n + 1), 0.0, dt_taken_s=1.0e-4)["inlet_covered"] for n in range(4)]
    assert covered == [True, True, False, False]
    air = chamber.air_cell.to_numpy()
    assert air[:, :, -1].any() and not air[:, :, :20].any()
    points.f_extra.fill(0.0)
    assert not chamber.apply_inlet_traction(10.0e3)
    assert not points.f_extra.to_numpy().any()


def lagged_gas_spring(omega_dt, steps):
    """A plate of mass m on a sealed pocket (area A, height h): x' = v, m v' = (p(V) - p_out) A with p from the
    accepted volume of the step start (explicit, symplectic Euler, as the coupled step); x compresses the pocket.  Returns (measured period /
    analytic period, energy envelope max / min, largest |x| / x0)."""
    p0, A, h, m = 96825.0, 1.0e-6, 3.0e-4, 1.0e-6
    q = p0 * A * h
    omega = math.sqrt(p0 * A * A / (A * h * m))
    dt = omega_dt / omega
    x0 = 1.0e-7
    x, v = x0, 0.0
    crossings, energies, previous, peak = [], [], x, abs(x)
    for n in range(steps):
        if not abs(x) < 0.5 * h:
            break
        p = q / (A * (h - x))
        v -= dt * (p - p0) * A / m                                # x compresses the pocket: the gas pushes back
        x += dt * v
        peak = max(peak, abs(x))
        energies.append(0.5 * m * v * v + p0 * A * h * (math.log(h / (h - x)) - x / h) if abs(x) < h else math.inf)
        if previous < 0.0 <= x:
            crossings.append(n * dt)
        previous = x
    period = (crossings[-1] - crossings[0]) / (len(crossings) - 1) if len(crossings) > 2 else float("nan")
    return period * omega / (2.0 * math.pi), max(energies) / min(e for e in energies if e > 0.0), peak / x0


def test_lagged_gas_spring_frequency_and_stability():
    ratio, _, _ = lagged_gas_spring(0.05, 40000)
    assert ratio == pytest.approx(1.0, abs=2.0e-3)
    ratio, envelope, growth = lagged_gas_spring(0.45, 200000)
    assert abs(ratio - 1.0) < 0.02 and growth < 1.2                  # (omega dt)^2 / 24: under 1 %; no growth
    _, _, growth = lagged_gas_spring(2.2, 2000)
    assert growth > 1.0e3                                            # beyond omega dt = 2: unstable
