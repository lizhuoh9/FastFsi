"""REQ_379 fast-mode prototype (simulation_core/fast/ib_mpm_fast.py): CPU unit tests.

IB_4 partition of unity, zero first moment and derivative sum; spreading and interpolation are adjoints (the discrete
power of a tether force equals the power of its spread density); a uniform velocity translates the points and keeps
F = I; a uniform shear gives F = I + dt G; the neo-Hookean Kirchhoff stress is zero at F = I and mu * gamma in small
shear; the control-volume force of a quiescent fluid is zero; (v2, Root REQ_382) the volume-drift statistics of the free body.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from simulation_core.fast.ib_mpm_fast import (ControlVolumeForce, FastIbMpmQuasi2D, ib4_dphi, ib4_phi,
                                              neo_hookean_kirchhoff_2d)

NODES = (2, 16, 16)
H = (0.1, 0.1, 0.1)


@ti.kernel
def _phi_sums(r: ti.f32, out: ti.types.ndarray()):
    s0, s1, sd = 0.0, 0.0, 0.0
    for j in range(-3, 4):
        x = r - j
        s0 += ib4_phi(x)
        s1 += x * ib4_phi(x)
        sd += ib4_dphi(x)
    out[0], out[1], out[2] = s0, s1, sd


@pytest.mark.parametrize("r", [0.0, 0.1, 0.37, 0.5, 0.83, 0.999])
def test_ib4_partition_of_unity_and_moments(r):
    out = np.zeros(3, dtype=np.float32)
    _phi_sums(r, out)
    assert out[0] == pytest.approx(1.0, abs=1e-6)
    assert out[1] == pytest.approx(0.0, abs=1e-6)
    assert out[2] == pytest.approx(0.0, abs=1e-5)


def fluid():
    return SimpleNamespace(force=ti.Vector.field(3, ti.f32, shape=NODES),
                           velocity=ti.Vector.field(3, ti.f32, shape=NODES),
                           pressure=ti.field(ti.f64, shape=NODES), obstacle=ti.field(ti.i32, shape=NODES),
                           rho=1000.0, mu=1.0)


def body(f, points, tether=None, mu=1.0e3, kappa=0.0, k=1.0e4):
    points = np.asarray(points, dtype=np.float32)
    return FastIbMpmQuasi2D(fluid=f, positions_yz_m=points, areas_m2=np.full(len(points), 1e-3),
                            tether_mask=np.ones(len(points), np.int32) if tether is None else tether,
                            density_kgm3=1000.0, shear_modulus_pa=mu, kappa_pa=kappa,
                            tether_stiffness_pa_per_m=k, grid_origin_m=(0.0, 0.0, 0.0), grid_spacing_m=H,
                            grid_nodes=NODES, span_m=0.2)


def test_spread_and_interpolate_are_adjoints():
    f = fluid()
    rng = np.random.default_rng(3)
    points = 0.5 + rng.random((5, 2)) * 0.6
    b = body(f, points, mu=0.0)
    # pull the tethered points: displaced positions give known per-point forces k A (X0 - x)
    shift = rng.normal(size=(5, 2)).astype(np.float32) * 1e-3
    b.x.from_numpy((points + shift).astype(np.float32))
    b.spread_elastic_force()
    force = f.force.to_numpy()
    velocity = rng.normal(size=NODES + (3,)).astype(np.float32)
    velocity[:] = velocity[0:1]                       # quasi-2-D: uniform along x
    f.velocity.from_numpy(velocity)
    b.move_with_fluid(0.0)                            # interpolation only (dt = 0)
    v = b.v.to_numpy()
    # the device pulls use the stored f32 positions: k A (X0 - x)
    pulls = 1.0e4 * 1e-3 * (b.X0.to_numpy().astype(np.float64) - b.x.to_numpy().astype(np.float64))
    particle_power = float(np.sum(pulls * v))
    face_volume = H[1] * H[2]
    grid_power = float(np.sum(force[0, :, :, 1:] * velocity[0, :, :, 1:]) * face_volume)
    assert grid_power == pytest.approx(particle_power, rel=1e-4, abs=1e-9)


def test_uniform_velocity_translates_and_keeps_F_identity():
    f = fluid()
    b = body(f, [[0.8, 0.8], [0.75, 0.9]])
    velocity = np.zeros(NODES + (3,), np.float32)
    velocity[..., 1], velocity[..., 2] = 0.3, -0.2
    f.velocity.from_numpy(velocity)
    b.move_with_fluid(0.1)
    np.testing.assert_allclose(b.x.to_numpy(), [[0.83, 0.78], [0.78, 0.88]], atol=1e-6)
    np.testing.assert_allclose(b.F.to_numpy(), np.tile(np.eye(2), (2, 1, 1)), atol=1e-6)


def test_uniform_shear_updates_F():
    f = fluid()
    b = body(f, [[0.8, 0.8]])
    velocity = np.zeros(NODES + (3,), np.float32)
    zc = (np.arange(16) + 0.5) * H[2]            # y faces sit at cell-centre z
    velocity[..., 1] = 2.0 * zc[None, None, :]   # u_y = 2 z  -> du_y/dz = 2
    f.velocity.from_numpy(velocity)
    b.move_with_fluid(0.01)
    np.testing.assert_allclose(b.F.to_numpy()[0], [[1.0, 0.02], [0.0, 1.0]], atol=1e-5)


@ti.kernel
def _stress(gamma: ti.f32, mu: ti.f32, kappa: ti.f32, out: ti.types.ndarray()):
    F = ti.Matrix([[1.0, gamma], [0.0, 1.0]])
    tau = neo_hookean_kirchhoff_2d(F, mu, kappa)
    for a, b in ti.static(ti.ndrange(2, 2)):
        out[a, b] = tau[a, b]


def test_neo_hookean_stress_zero_at_rest_and_mu_gamma_in_shear():
    out = np.zeros((2, 2), dtype=np.float32)
    _stress(0.0, 2.0e6, 2.0e7, out)
    np.testing.assert_allclose(out, 0.0, atol=1e-3)
    _stress(1.0e-3, 2.0e6, 2.0e7, out)
    assert out[0, 1] == pytest.approx(2.0e6 * 1.0e-3, rel=1e-3)
    assert out[1, 0] == pytest.approx(out[0, 1], rel=1e-6)


def test_control_volume_force_of_quiescent_fluid_is_zero():
    f = fluid()
    f.pressure.fill(5.0)                          # uniform pressure: no net surface force
    cv = ControlVolumeForce(f, j_range=(3, 12), k_range=(3, 12), origin_m=(0, 0, 0), spacing_m=H, grid_nodes=NODES,
                            span_m=0.2)
    r = cv.measure()
    assert abs(r["surface_y"]) < 1e-9 and abs(r["surface_z"]) < 1e-9
    assert r["momentum_y"] == 0.0 and r["momentum_z"] == 0.0


def test_elastic_dt_uses_the_shear_wave_on_the_smallest_cell():
    b = body(fluid(), [[0.8, 0.8]], mu=2.0e6)
    assert b.elastic_dt_limit(0.5) == pytest.approx(0.5 * 0.1 / math.sqrt(2.0e6 / 1000.0))


def test_midpoint_rule_matches_euler_for_a_steady_uniform_shear():
    f = fluid()
    euler, mid = body(f, [[0.8, 0.8]]), body(f, [[0.8, 0.8]])
    velocity = np.zeros(NODES + (3,), np.float32)
    zc = (np.arange(16) + 0.5) * H[2]
    velocity[..., 1] = 2.0 * zc[None, None, :]
    f.velocity.from_numpy(velocity)
    old = ti.Vector.field(3, ti.f32, shape=NODES)
    old.copy_from(f.velocity)
    euler.move_with_fluid(0.01)
    mid.midpoint_predict(0.01)
    mid.midpoint_correct(old, 0.01)
    # steady linear shear: the point moves along y by u_y(z) dt, F = I + dt G (G constant) for both rules
    np.testing.assert_allclose(mid.x.to_numpy(), euler.x.to_numpy(), atol=1e-6)
    np.testing.assert_allclose(mid.F.to_numpy(), euler.F.to_numpy(), atol=1e-5)


def test_volume_drift_statistics_of_the_free_body():
    f = fluid()
    points = [[0.5, 0.5], [0.6, 0.5], [0.7, 0.5], [0.8, 0.5]]
    b = FastIbMpmQuasi2D(fluid=f, positions_yz_m=np.asarray(points, np.float32), areas_m2=np.full(4, 1e-3),
                         tether_mask=np.array([0, 0, 0, 1], np.int32), density_kgm3=1000.0, shear_modulus_pa=1.0e3,
                         kappa_pa=0.0, tether_stiffness_pa_per_m=1.0e4, grid_origin_m=(0.0, 0.0, 0.0),
                         grid_spacing_m=H, grid_nodes=NODES, span_m=0.2,
                         near_surface_mask=np.array([1, 0, 0, 0], np.int32))
    s = 1.004                                   # J = s^2 = 1.008016 on the first two free points, 1 on the third
    F = np.tile(np.eye(2, dtype=np.float32), (4, 1, 1))
    F[0] *= s
    F[1] *= s
    b.F.from_numpy(F)
    d = b.diagnostics()
    J = float(np.float32(s) * np.float32(s))
    assert d["volume_error"] == pytest.approx((2 * J + 1) / 3 - 1, rel=1e-5)   # the tethered point is excluded
    assert d["J_deviated_fraction"] == pytest.approx(2 / 3)
    assert d["J_deviated_near_surface_share"] == pytest.approx(0.5)
