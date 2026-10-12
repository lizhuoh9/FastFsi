"""Integration check (Helper 1, 2026-10-12): the integrator's body (simulation_core/fast/ib_mpm_fast3d.py, FastIbMpm3D)
on GradedAxes with the RKPM moment correction and a real FaceMask (obstacle wall), production stretch 1.2.

- the interior fast path (support flags, no one-sided normalisation) equals the full one-sided path: this needs
  sum w~ = 1 and sum dw~ = 0 per axis, which the moment correction keeps (graded_kernel tests);
- a rigid rotation sampled on the MAC faces gives grad_v = W (zero strain) at interior points across the transitions.
Skipped when the body module is not importable (it lives in the integrator's tree until proto_05).
"""
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

body_module = pytest.importorskip("simulation_core.fast.ib_mpm_fast3d")

from simulation_core.fast.graded_kernel import FaceMask, GradedAxes  # noqa: E402


def graded_faces(origin, n, h0, ratio, core):
    left = (n - core) // 2
    right = n - core - left
    widths = ([h0 * ratio ** (left - i) for i in range(left)] + [h0] * core
              + [h0 * ratio ** (i + 1) for i in range(right)])
    return origin + np.concatenate([[0.0], np.cumsum(widths)])


AXES = GradedAxes(graded_faces(-0.01, 16, 1.0e-3, 1.2, 4), 0.004 + 1.2e-3 * np.arange(13),
                  graded_faces(0.98, 16, 6.0e-4, 1.2, 4))
SHAPE = tuple(AXES.n)


def stub_fluid(wall_cells=3):
    fluid = SimpleNamespace(force=ti.Vector.field(3, ti.f32, shape=SHAPE),
                            velocity=ti.Vector.field(3, ti.f32, shape=SHAPE),
                            obstacle=ti.field(ti.i32, shape=SHAPE))
    obstacle = np.zeros(SHAPE, np.int32)
    obstacle[:wall_cells] = 1                                   # a solid wall on the x-min side
    fluid.obstacle.from_numpy(obstacle)
    return fluid


def points_interior_and_near_wall(rng):
    centres = AXES.centres_m
    interior = np.column_stack([rng.uniform(centres[0][8], centres[0][11], 5),
                                rng.uniform(centres[1][4], centres[1][8], 5),
                                rng.uniform(centres[2][5], centres[2][10], 5)])
    near_wall = np.array([[AXES.faces_m[0][3] + 0.3e-3, centres[1][6], centres[2][7]],
                          [AXES.faces_m[0][4] + 0.2e-3, centres[1][5], centres[2][8]]])
    return np.vstack([interior, near_wall]).astype(np.float32)


def run(fast_path):
    rng = np.random.default_rng(23)
    fluid = stub_fluid()
    mask = FaceMask(fluid, AXES)
    points = points_interior_and_near_wall(rng)
    n = len(points)
    ext = (rng.normal(size=(n, 3)) * 1e-3).astype(np.float32)
    body = body_module.FastIbMpm3D(fluid=fluid, axes=AXES, mask=mask, positions_m=points,
                                   volumes_m3=np.full(n, 1e-9, np.float32), tether_mask=np.zeros(n, np.int32),
                                   density_kgm3=1000.0, shear_modulus_pa=200.0, kappa_pa=500.0,
                                   tether_stiffness_pa_per_m=0.0, external_force_n=ext, fast_path=fast_path)
    body.F.from_numpy((np.eye(3) + 0.05 * rng.normal(size=(n, 3, 3))).astype(np.float32))
    fluid.velocity.from_numpy(rng.normal(size=SHAPE + (3,)).astype(np.float32))
    body.refresh_support_flags()
    body.spread_force()
    body.move_with_fluid(0.0)
    return fluid.force.to_numpy(), body.v.to_numpy(), body.grad_v.to_numpy(), body, fluid


def test_interior_fast_path_matches_the_full_one_sided_path_on_rkpm_graded_axes():
    force_full, v_full, g_full, _, _ = run(False)
    force_fast, v_fast, g_fast, body, _ = run(True)
    flags = body.support_flag.to_numpy()
    # wall band flagged (obstacle cells 0..2 + the x-face of cell 3, dilated by SUPPORT_REACH = 3); the interior
    # host cells of the five interior points are not (the domain-edge faces at j = 0 / k = 0 flag their bands too).
    # Integrator 2026-10-12 (edge-flag fix for the xfail below): the outermost cell layer of every axis is flagged as
    # well, dilated by 3, so the interior box excludes the far-edge bands too (x >= 12, y >= 8, z >= 12).
    assert flags[:7].all() and not flags[8:12, 4:8, 4:12].any()
    np.testing.assert_allclose(force_fast, force_full, atol=1e-6 * np.abs(force_full).max())
    np.testing.assert_allclose(v_fast, v_full, atol=1e-6 * np.abs(v_full).max())
    np.testing.assert_allclose(g_fast, g_full, atol=1e-5 * np.abs(g_full).max())


def test_body_rigid_rotation_on_rkpm_graded_axes():
    _, _, _, body, fluid = run(True)
    omega = np.array([0.3, -0.5, 0.7])
    spin = np.array([[0.0, -omega[2], omega[1]], [omega[2], 0.0, -omega[0]], [-omega[1], omega[0], 0.0]])
    centre = np.array([0.5 * (AXES.faces_m[a][0] + AXES.faces_m[a][-1]) for a in range(3)])
    velocity = np.zeros(SHAPE + (3,))
    for c in range(3):
        coordinates = [AXES.centres_m[a] for a in range(3)]
        coordinates[c] = AXES.faces_m[c][:-1]
        X, Y, Z = np.meshgrid(*coordinates, indexing="ij")
        r = np.stack([X - centre[0], Y - centre[1], Z - centre[2]], axis=-1)
        velocity[..., c] = np.cross(omega, r)[..., c]
    fluid.velocity.from_numpy(velocity.astype(np.float32))
    body.move_with_fluid(0.0)
    g = body.grad_v.to_numpy().astype(np.float64)[:5]           # the five interior points
    strain = 0.5 * (g + np.transpose(g, (0, 2, 1)))
    print("GRADED_BODY_ROTATION max|G-W|=%.3e max|sym G|=%.3e" % (np.abs(g - spin).max(), np.abs(strain).max()))
    assert np.abs(g - spin).max() <= 2e-4 * np.linalg.norm(omega)
    assert np.abs(strain).max() <= 2e-4 * np.linalg.norm(omega)


# Integrator 2026-10-12: the xfail mark is removed — the body now base-flags the outermost cell layer of every axis
# (far domain faces, index n_a, are unstored), so this test must pass as a regular test.
def test_fast_path_next_to_the_far_domain_faces():
    rng = np.random.default_rng(5)
    results = []
    for fast_path in (False, True):
        fluid = stub_fluid(wall_cells=0)
        mask = FaceMask(fluid, AXES)
        point = np.array([[AXES.faces_m[0][-1] - 0.4e-3, AXES.centres_m[1][6], AXES.centres_m[2][8]]], np.float32)
        body = body_module.FastIbMpm3D(fluid=fluid, axes=AXES, mask=mask, positions_m=point,
                                       volumes_m3=np.full(1, 1e-9, np.float32), tether_mask=np.zeros(1, np.int32),
                                       density_kgm3=1000.0, shear_modulus_pa=0.0, kappa_pa=0.0,
                                       tether_stiffness_pa_per_m=0.0,
                                       external_force_n=np.array([[1e-3, 2e-3, 3e-3]], np.float32),
                                       fast_path=fast_path)
        fluid.velocity.from_numpy(rng.normal(size=SHAPE + (3,)).astype(np.float32))
        body.refresh_support_flags()
        body.spread_force()
        volumes = np.zeros((3,) + SHAPE, np.float32)
        _face_volumes(AXES, volumes)
        total = (np.moveaxis(fluid.force.to_numpy(), -1, 0).astype(np.float64) * volumes).reshape(3, -1).sum(axis=1)
        results.append(total)
    print("GRADED_BODY_FAR_EDGE total spread force full=%s fast=%s (point force 1e-3, 2e-3, 3e-3 N)"
          % (np.array2string(results[0], precision=6), np.array2string(results[1], precision=6)))
    np.testing.assert_allclose(results[1], results[0], rtol=1e-5)


@ti.kernel
def _face_volumes(axes: ti.template(), out: ti.types.ndarray()):
    for c in ti.static(range(3)):
        for i, j, k in ti.ndrange(axes.n[0], axes.n[1], axes.n[2]):
            out[c, i, j, k] = axes.face_volume(c, i, j, k)
