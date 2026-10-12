"""Fast-mode 3-D body (simulation_core/fast/ib_mpm_fast3d.py; Root REQ_401 step 1): CPU unit tests.

Force conservation of the spread point force (also next to invalid faces: one-sided normalisation); the power identity
between spreading and interpolation (adjoint pair, with stress, point forces and invalid faces); the internal-stress
force balance; a uniform translation and a rigid rotation are reproduced exactly (IB_4 reproduces linear fields); the 3-D
neo-Hookean stress; checkpoint state round trip; the empty-support hard stop.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from simulation_core.fast.fluid_hooks import AllValidMask, UniformAxes
from simulation_core.fast.ib_mpm_fast3d import FastIbMpm3D, FastModeHardStop, neo_hookean_kirchhoff_3d

NODES = (20, 12, 12)
H = (0.1, 0.1, 0.1)
ORIGIN = (0.0, 0.0, 0.0)


@ti.data_oriented
class WallMask:
    """Faces of every component with cell index i < cut along x are invalid (a wall on the x-min side)."""

    def __init__(self, cut):
        self.cut = int(cut)

    def refresh(self, air_cell_field=None):
        return None

    @ti.func
    def face_valid(self, c: ti.template(), i, j, k):
        return 1 if i >= self.cut else 0


@ti.data_oriented
class NoneValidMask:
    def refresh(self, air_cell_field=None):
        return None

    @ti.func
    def face_valid(self, c: ti.template(), i, j, k):
        return 0


def fluid():
    return SimpleNamespace(force=ti.Vector.field(3, ti.f32, shape=NODES),
                           velocity=ti.Vector.field(3, ti.f32, shape=NODES))


def make_body(f, points, *, mask=None, mu=0.0, kappa=0.0, ext=None, tether=None, k=0.0, volume=1.0e-3):
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    return FastIbMpm3D(fluid=f, axes=UniformAxes(ORIGIN, H, NODES), mask=mask or AllValidMask(), positions_m=points,
                       volumes_m3=np.full(n, volume, np.float32),
                       tether_mask=np.zeros(n, np.int32) if tether is None else tether, density_kgm3=1000.0,
                       shear_modulus_pa=mu, kappa_pa=kappa, tether_stiffness_pa_per_m=k, external_force_n=ext)


def face_positions(c):
    """(nx, ny, nz, 3) positions of the backward MAC faces of component c."""
    idx = np.indices(NODES).astype(np.float64)
    pos = np.empty(NODES + (3,))
    for d in range(3):
        pos[..., d] = ORIGIN[d] + (idx[d] + (0.0 if d == c else 0.5)) * H[d]
    return pos


def total_force(f):
    return f.force.to_numpy().reshape(-1, 3).sum(axis=0) * (H[0] * H[1] * H[2])


def test_spread_conserves_the_point_force():
    rng = np.random.default_rng(1)
    f = fluid()
    F = np.array([[0.3, -1.2, 0.7]], np.float32)
    b = make_body(f, rng.uniform(0.45, 0.75, size=(1, 3)), ext=F)
    b.spread_force()
    np.testing.assert_allclose(total_force(f), F[0], rtol=2e-6, atol=1e-7)
    assert b.one_sided_report()['min_D'] == pytest.approx(1.0, abs=2e-6)


def test_one_sided_spread_conserves_the_force_and_skips_invalid_faces():
    f = fluid()
    F = np.array([[1.0, 2.0, -0.5]], np.float32)
    # point 0.13 from the wall at x = 0.5 (cut = 5 cells): part of every support is invalid
    b = make_body(f, [[0.52, 0.61, 0.58]], mask=WallMask(5), ext=F)
    b.spread_force()
    force = f.force.to_numpy()
    assert np.all(force[:5] == 0.0)
    np.testing.assert_allclose(total_force(f), F[0], rtol=2e-6, atol=1e-7)
    report = b.one_sided_report()
    assert 0.0 < report['min_D'] < 1.0


@pytest.mark.parametrize("mask", [None, WallMask(5)])
def test_spread_and_interpolation_are_adjoint(mask):
    rng = np.random.default_rng(7)
    f = fluid()
    n = 6
    points = rng.uniform(0.5, 0.7, size=(n, 3))
    ext = rng.normal(size=(n, 3)).astype(np.float32)
    b = make_body(f, points, mask=mask, mu=2.0, kappa=5.0, ext=ext)
    Fdef = (np.eye(3) + 0.05 * rng.normal(size=(n, 3, 3))).astype(np.float32)
    b.F.from_numpy(Fdef)
    u = rng.normal(size=NODES + (3,)).astype(np.float32)
    f.velocity.from_numpy(u)
    b.spread_force()
    grid_power = float(np.sum(f.force.to_numpy().astype(np.float64) * u) * (H[0] * H[1] * H[2]))
    b.move_with_fluid(0.0)                       # v and grad_v with the same one-sided weights
    v = b.v.to_numpy().astype(np.float64)
    G = b.grad_v.to_numpy().astype(np.float64)
    tau = b.tau.to_numpy().astype(np.float64)
    point_power = float(np.sum(-1.0e-3 * np.einsum('pij,pij->p', tau, G)) + np.sum(ext * v))
    assert grid_power == pytest.approx(point_power, rel=2e-5, abs=1e-7)


@pytest.mark.parametrize("mask", [None, WallMask(5)])
def test_internal_stress_forces_balance(mask):
    f = fluid()
    g = np.linspace(0.52, 0.68, 4)
    points = np.array(np.meshgrid(g, g, g, indexing='ij')).reshape(3, -1).T
    b = make_body(f, points, mask=mask, mu=3.0, kappa=7.0)
    b.F.from_numpy(np.tile(np.diag([1.05, 0.98, 1.0]).astype(np.float32), (len(points), 1, 1)))
    b.spread_force()
    force = f.force.to_numpy().reshape(-1, 3) * (H[0] * H[1] * H[2])
    scale = np.abs(force).sum(axis=0)
    assert np.all(scale > 0)
    np.testing.assert_allclose(force.sum(axis=0) / scale, 0.0, atol=2e-6)


@pytest.mark.parametrize("mask", [None, WallMask(5)])
def test_uniform_translation_is_exact(mask):
    f = fluid()
    u0 = np.array([0.2, -0.1, 0.05], np.float32)
    f.velocity.from_numpy(np.broadcast_to(u0, NODES + (3,)).copy())
    points = np.array([[0.53, 0.6, 0.6], [0.66, 0.55, 0.71]])
    b = make_body(f, points, mask=mask)
    b.move_with_fluid(0.01)
    np.testing.assert_allclose(b.positions(), points + 0.01 * u0, atol=2e-7)
    np.testing.assert_allclose(b.grad_v.to_numpy(), 0.0, atol=2e-5)
    np.testing.assert_allclose(b.F.to_numpy(), np.tile(np.eye(3), (2, 1, 1)), atol=2e-7)


def test_rigid_rotation_is_reproduced():
    f = fluid()
    omega = np.array([0.3, -0.2, 0.5])
    centre = np.array([0.6, 0.6, 0.6])
    u = np.zeros(NODES + (3,), np.float32)
    for c in range(3):
        pos = face_positions(c) - centre
        u[..., c] = np.cross(omega, pos)[..., c]
    f.velocity.from_numpy(u)
    points = np.array([[0.57, 0.62, 0.6], [0.63, 0.58, 0.64]])
    b = make_body(f, points)
    dt = 1.0e-3
    b.move_with_fluid(dt)
    W = np.array([[0.0, -omega[2], omega[1]], [omega[2], 0.0, -omega[0]], [-omega[1], omega[0], 0.0]])
    np.testing.assert_allclose(b.velocities(), np.cross(omega, points - centre), atol=2e-6)
    np.testing.assert_allclose(b.grad_v.to_numpy(), np.tile(W, (2, 1, 1)), atol=2e-5)
    np.testing.assert_allclose(b.F.to_numpy(), np.tile(np.eye(3) + dt * W, (2, 1, 1)), atol=2e-7)


@ti.kernel
def _stress(F: ti.types.ndarray(), mu: ti.f32, kappa: ti.f32, out: ti.types.ndarray()):
    for _ in range(1):
        Fm = ti.Matrix([[F[0, 0], F[0, 1], F[0, 2]], [F[1, 0], F[1, 1], F[1, 2]], [F[2, 0], F[2, 1], F[2, 2]]])
        t = neo_hookean_kirchhoff_3d(Fm, mu, kappa)
        for i, j in ti.static(ti.ndrange(3, 3)):
            out[i, j] = t[i, j]


def stress(F, mu, kappa):
    out = np.zeros((3, 3), np.float32)
    _stress(np.asarray(F, np.float32), mu, kappa, out)
    return out


def test_neo_hookean_3d_stress():
    np.testing.assert_allclose(stress(np.eye(3), 13.3e3, 133e3), 0.0, atol=1e-3)
    gamma = 1.0e-3
    shear = stress([[1.0, gamma, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], 13.3e3, 0.0)
    assert shear[0, 1] == pytest.approx(13.3e3 * gamma, rel=1e-3)
    J = 1.01
    vol = stress(np.eye(3) * J ** (1.0 / 3.0), 0.0, 1.0e5)
    np.testing.assert_allclose(np.diag(vol), 1.0e5 * (J - 1.0) * J, rtol=1e-3)


def test_checkpoint_state_round_trip():
    f = fluid()
    points = np.array([[0.6, 0.6, 0.6], [0.62, 0.6, 0.61]], np.float32)
    b = make_body(f, points)
    b.u.from_numpy(np.full((2, 3), 1.0e-4, np.float32))
    state = b.state_arrays()
    b.u.from_numpy(np.zeros((2, 3), np.float32))
    b.load_state_arrays(state)
    np.testing.assert_array_equal(b.u.to_numpy(), state['u'])
    np.testing.assert_allclose(b.positions(), points.astype(np.float64) + 1.0e-4, atol=1e-9)
    np.testing.assert_allclose(b.x.to_numpy(), points + np.float32(1.0e-4), atol=1e-7)   # derived x refreshed
    bad = dict(state, u=np.zeros((3, 3), np.float32))
    with pytest.raises(ValueError):
        b.load_state_arrays(bad)


def test_displacement_integration_keeps_small_steps_near_one_metre():
    """f32 absolute positions near 1 m have an ulp of 1.2e-7 m; the body integrates u, so 1000 steps of 1e-8 m move
    the point by 1e-5 m (an absolute f32 x would not move at all)."""
    f = fluid()
    origin = (0.0, 0.0, 1.0)
    axes = UniformAxes(origin, H, NODES)
    point = np.array([[0.6, 0.6, 1.6]], np.float32)
    b = FastIbMpm3D(fluid=f, axes=axes, mask=AllValidMask(), positions_m=point, volumes_m3=np.full(1, 1e-3, np.float32),
                    tether_mask=np.zeros(1, np.int32), density_kgm3=1000.0, shear_modulus_pa=0.0, kappa_pa=0.0,
                    tether_stiffness_pa_per_m=0.0)
    f.velocity.from_numpy(np.broadcast_to(np.array([0.0, 0.0, 1.0e-3], np.float32), NODES + (3,)).copy())
    for _ in range(1000):
        b.move_with_fluid(1.0e-5)
    assert b.u.to_numpy()[0, 2] == pytest.approx(1.0e-5, rel=1e-3)
    assert b.positions()[0, 2] - float(np.float32(1.6)) == pytest.approx(1.0e-5, rel=1e-3)


def _spread_and_gather(fast_path, unroll_static=False, refresh=True):
    rng = np.random.default_rng(11)
    f = fluid()
    interior = np.column_stack([rng.uniform(0.86, 0.93, 4), rng.uniform(0.5, 0.7, 4), rng.uniform(0.5, 0.7, 4)])
    points = np.vstack([interior,                                         # host cells 8-9: unflagged (fast path)
                        [[0.52, 0.61, 0.58], [0.535, 0.7, 0.66]]])       # next to the wall at x = 0.5 (full path)
    ext = rng.normal(size=(6, 3)).astype(np.float32)
    b = FastIbMpm3D(fluid=f, axes=UniformAxes(ORIGIN, H, NODES), mask=WallMask(5), positions_m=points,
                    volumes_m3=np.full(6, 1e-3, np.float32), tether_mask=np.zeros(6, np.int32), density_kgm3=1000.0,
                    shear_modulus_pa=2.0, kappa_pa=5.0, tether_stiffness_pa_per_m=0.0, external_force_n=ext,
                    fast_path=fast_path, unroll_static=unroll_static)
    b.F.from_numpy((np.eye(3) + 0.05 * rng.normal(size=(6, 3, 3))).astype(np.float32))
    f.velocity.from_numpy(rng.normal(size=NODES + (3,)).astype(np.float32))
    if refresh:
        b.refresh_support_flags()
    b.spread_force()
    b.move_with_fluid(0.0)
    flags = b.support_flag.to_numpy()
    return f.force.to_numpy(), b.v.to_numpy(), b.grad_v.to_numpy(), flags


def test_interior_fast_path_matches_the_full_one_sided_path():
    force_full, v_full, g_full, _ = _spread_and_gather(False)
    force_fast, v_fast, g_fast, flags = _spread_and_gather(True)
    scale = np.abs(force_full).max()
    np.testing.assert_allclose(force_fast, force_full, atol=1e-6 * scale)
    np.testing.assert_allclose(v_fast, v_full, atol=1e-6 * np.abs(v_full).max())
    np.testing.assert_allclose(g_fast, g_full, atol=1e-5 * np.abs(g_full).max())


def test_support_flags_cover_the_wall_band_the_domain_edges_and_leave_the_interior():
    """WallMask(5) invalidates every face with i < 5: base flags on cells 0..4, dilated by 3 -> cells 0..7.  The
    outermost cell layer of every axis is base-flagged too (supports reach unstored nodes), dilated by 3: with
    NODES = (20, 12, 12) that flags x >= 16 and y, z <= 3 or >= 8, leaving the box x 8..15, y 4..7, z 4..7."""
    _, _, _, flags = _spread_and_gather(True)
    assert np.all(flags[:8] == 1)
    assert np.all(flags[16:] == 1) and np.all(flags[:, :4] == 1) and np.all(flags[:, 8:] == 1)
    assert np.all(flags[:, :, :4] == 1) and np.all(flags[:, :, 8:] == 1)
    assert np.all(flags[8:16, 4:8, 4:8] == 0)


def test_fast_path_next_to_the_far_domain_faces():
    """A point within 2 cells of x-max (face index n unstored) must take the full path: spread force conserved."""
    f = fluid()
    F = np.array([[1.0e-3, 2.0e-3, 3.0e-3]], np.float32)
    b = make_body(f, [[NODES[0] * H[0] - 0.04, 0.61, 0.58]], ext=F)
    b.refresh_support_flags()
    b.spread_force()
    total = total_force(f)
    assert np.all(np.abs(total) <= np.abs(F[0]) * (1.0 + 1e-6))      # nothing created
    full = fluid()
    b_full = FastIbMpm3D(fluid=full, axes=UniformAxes(ORIGIN, H, NODES), mask=AllValidMask(),
                         positions_m=np.array([[NODES[0] * H[0] - 0.04, 0.61, 0.58]], np.float32),
                         volumes_m3=np.full(1, 1e-3, np.float32), tether_mask=np.zeros(1, np.int32), density_kgm3=1000.0,
                         shear_modulus_pa=0.0, kappa_pa=0.0, tether_stiffness_pa_per_m=0.0, external_force_n=F,
                         fast_path=False)
    b_full.spread_force()
    np.testing.assert_allclose(f.force.to_numpy(), full.force.to_numpy(), atol=1e-7 * np.abs(F).max())


def test_static_unroll_matches_the_dynamic_inner_loops():
    a = _spread_and_gather(True, unroll_static=False)
    b = _spread_and_gather(True, unroll_static=True)
    for x, y in zip(a[:3], b[:3]):
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-7 * max(np.abs(x).max(), 1e-30))


def test_empty_support_is_a_hard_stop():
    f = fluid()
    b = make_body(f, [[0.6, 0.6, 0.6]], mask=NoneValidMask(), ext=np.ones((1, 3), np.float32))
    b.spread_force()
    with pytest.raises(FastModeHardStop) as stop:
        b.one_sided_report()
    assert stop.value.code == 'one_sided_empty_support'
    assert stop.value.witness['count'] == 3


def test_per_body_volume_error_rim_min_D_and_flap_band_lift():
    """Root 2026-10-12 items 2 and 4: volume error per body (1 = flap, 2 = membrane), the flap's bonded-band lift and
    min D over the free valve-rim points only."""
    f = fluid()
    points = np.array([[0.90, 0.6, 0.6], [0.62, 0.6, 0.6], [0.75, 0.6, 0.6], [0.55, 0.6, 0.6],
                       [0.90, 0.7, 0.6], [1.00, 0.6, 0.6], [1.10, 0.6, 0.6], [1.20, 0.6, 0.6]], np.float32)
    body_id = np.array([1, 1, 1, 1, 2, 2, 2, 2], np.int32)
    tether = np.array([1, 0, 0, 0, 1, 0, 0, 0], np.int32)
    tag = np.array([0, 2, 2, 0, 0, 0, 0, 0], np.int32)       # TAG_VALVE_RIM on two free flap points
    b = FastIbMpm3D(fluid=f, axes=UniformAxes(ORIGIN, H, NODES), mask=WallMask(5), positions_m=points,
                    volumes_m3=np.full(8, 1.0e-3, np.float32), tether_mask=tether, density_kgm3=1000.0,
                    shear_modulus_pa=0.0, kappa_pa=0.0, tether_stiffness_pa_per_m=0.0, body_id=body_id, tag=tag)
    F = np.tile(np.eye(3, dtype=np.float32), (8, 1, 1))
    F[1:4] *= np.float32(1.01)                                # free flap points: J = 1.01^3
    b.F.from_numpy(F)
    u = np.zeros((8, 3), np.float32)
    u[0] = (0.0, 0.0, 2.0e-4)                                 # tethered flap point lifted 0.2 mm
    u[4] = (3.0e-4, 0.0, 0.0)                                 # tethered membrane point
    b.u.from_numpy(u)
    d = b.diagnostics()
    J = float(np.float32(1.01)) ** 3
    assert d['volume_error_flap'] == pytest.approx(J - 1.0, rel=1e-5)
    assert d['volume_error_membrane'] == pytest.approx(0.0, abs=1e-12)
    assert d['volume_error'] == pytest.approx(0.5 * (J - 1.0), rel=1e-5)
    assert d['tether_max_displacement_flap_m'] == pytest.approx(2.0e-4, rel=1e-6)
    assert d['tether_max_uz_flap_m'] == pytest.approx(2.0e-4, rel=1e-6)
    assert d['tether_max_displacement_membrane_m'] == pytest.approx(3.0e-4, rel=1e-6)
    report = b.one_sided_report()
    only_rim = make_body(fluid(), points[1:3], mask=WallMask(5))
    expected = only_rim.one_sided_report()['min_D']
    assert 0.0 < expected < 1.0
    assert report['min_D_rim_free'] == pytest.approx(expected, rel=1e-6)
    assert report['min_D_free'] < report['min_D_rim_free']   # the non-rim free point at x = 0.55 is nearer the wall
