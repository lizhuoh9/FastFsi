"""f_update="isochoric" (Root 2026-10-24, the membrane gates' drift): F updated with the deviatoric gradient and
renormalised to det F = 1.  A linear shear-plus-dilation flow u = L x, L = [[D, 0, g], [0, D, 0], [0, 0, D]], on a
uniform grid (IB_4 interpolates it exactly), four free points, 1000 midpoint steps: the default ("full") is bit for bit
the update without the argument and its det F follows exp(3 D t); "isochoric" keeps det F = 1 to round-off while its
F equals the full F / det(F)^(1/3) (L and its trace part commute, so that is the exact deviatoric motion); there the
deviatoric part is a nilpotent shear, so nothing is removed, while a traceless extension's second-order volume change
is removed and reported per large step, the report resetting on reading; with held_f="frozen" the held F stay I; a
wrong mode is refused."""
import numpy as np
import pytest
import taichi as ti
from types import SimpleNamespace

from fastfsi.coupling.face_mask import AllValidFaceMask
from fastfsi.coupling.immersed_boundary import ImmersedBoundaryCoupler
from fastfsi.coupling.kernel_weights import KernelWeights
from fastfsi.grid.axes import UniformAxes
from fastfsi.solid.material_points import MaterialPoints
from fastfsi.solid.materials import NeoHookeanMaterial

SHEAR = 0.1                     # 1/s
DILATION = 0.05                 # 1/s per axis: tr L = 0.15 /s
DT = 1.0e-3
STEPS = 1000
POSITIONS = np.array([[3.5, 3.5, 3.5], [4.0, 3.5, 4.0], [3.5, 4.5, 3.5], [4.5, 4.0, 4.0]], np.float32)


@pytest.fixture(scope="module", autouse=True)
def cpu():
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1, opt_level=0, offline_cache=False)


@pytest.fixture(scope="module")
def flow():
    velocity = ti.Vector.field(3, ti.f32, shape=(8, 8, 8))
    i, j, k = np.meshgrid(*(np.arange(8, dtype=np.float32),) * 3, indexing="ij")
    values = np.zeros((8, 8, 8, 3), np.float32)
    values[..., 0] = SHEAR * (k + 0.5) + DILATION * i               # u_x on the x faces (x = i, z = k + 1/2)
    values[..., 1] = DILATION * j                                    # u_y on the y faces (y = j)
    values[..., 2] = DILATION * k                                    # u_z on the z faces (z = k)
    velocity.from_numpy(values)
    return velocity


def coupler(velocity, held=(0, 0, 0, 0), **options):
    points = MaterialPoints(positions_m=POSITIONS, volumes_m3=np.ones(4), material=NeoHookeanMaterial(1000., 1., 10.),
                            tether_mask=np.array(held, np.int32))
    ib = ImmersedBoundaryCoupler(fluid=SimpleNamespace(velocity=velocity, rho=1000.),
                                 weights=KernelWeights(UniformAxes((0., 0., 0.), (1., 1., 1.), (8, 8, 8))),
                                 mask=AllValidFaceMask(), points=points, tether_stiffness_pa_per_m=0.,
                                 interior_shortcut=False, **options)
    return points, ib


def run(points, ib, velocity, steps=STEPS):
    for _ in range(steps):
        ib.midpoint_predict(DT)
        ib.midpoint_correct(velocity, DT)
    return points.F.to_numpy().astype(np.float64)


def test_isochoric_keeps_det_one_and_follows_the_deviatoric_motion(flow):
    default = run(*coupler(flow), flow)
    full = run(*coupler(flow, f_update="full"), flow)
    assert full.tobytes() == default.tobytes()                                      # the default, bit for bit
    t = STEPS * DT
    assert np.allclose(np.linalg.det(full), np.exp(3 * DILATION * t), rtol=1e-4)     # full: det F = exp(tr L t)
    points, ib = coupler(flow, f_update="isochoric")
    iso = run(points, ib, flow)
    assert np.abs(np.linalg.det(iso) - 1.0).max() < 5e-6                            # det F = 1 to round-off
    expected = full / np.cbrt(np.linalg.det(full))[:, None, None]
    assert np.allclose(iso, expected, rtol=0, atol=2e-5)                             # the deviatoric part follows
    report = ib.f_update_report()                    # the deviatoric part is a nilpotent shear: nothing to remove
    assert report["f_update"] == "isochoric" and 0.0 <= report["max_abs_det_f_minus_1_removed"] < 1e-6
    assert coupler(flow)[1].f_update_report() == dict(f_update="full", max_abs_det_f_minus_1_removed=None)


def test_the_renormalisation_removes_and_reports_the_second_order_volume_change():
    velocity = ti.Vector.field(3, ti.f32, shape=(8, 8, 8))
    i, j, _ = np.meshgrid(*(np.arange(8, dtype=np.float32),) * 3, indexing="ij")
    values = np.zeros((8, 8, 8, 3), np.float32)
    values[..., 0], values[..., 1] = 2.0 * i, -2.0 * j               # traceless extension: det(I + dt L) = 1 - (2 dt)^2
    velocity.from_numpy(values)
    points, ib = coupler(velocity, f_update="isochoric")
    F = run(points, ib, velocity, steps=1)
    step = 0.05
    ib.midpoint_predict(step)
    ib.midpoint_correct(velocity, step)
    removed = ib.f_update_report()["max_abs_det_f_minus_1_removed"]
    assert 1e-3 < removed < 0.1                                       # O((2 dt)^2) = 0.01 per large step
    assert np.abs(np.linalg.det(points.F.to_numpy().astype(np.float64)) - 1.0).max() < 5e-6 and F.shape == (4, 3, 3)
    assert ib.f_update_report()["max_abs_det_f_minus_1_removed"] == 0.0              # reset by the read


def test_held_points_follow_held_f_under_the_isochoric_update(flow):
    points, ib = coupler(flow, held=(1, 1, 0, 0), held_f="frozen", f_update="isochoric")
    F = run(points, ib, flow, steps=200)
    assert all(np.array_equal(F[p], np.eye(3)) for p in (0, 1))                     # held: exactly I
    assert np.abs(np.linalg.det(F[2:]) - 1.0).max() < 5e-6 and abs(F[2, 0, 2]) > 1e-3   # free: isochoric shear


def test_a_wrong_mode_is_refused(flow):
    with pytest.raises(ValueError, match="f_update"):
        coupler(flow, f_update="volumetric")
