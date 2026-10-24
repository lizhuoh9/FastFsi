"""held_f="frozen" (Root 2026-10-24, the held-point F diagnostic): a tethered point's F stays I.  A linear shear
u_x = g z on a uniform grid (IB_4 interpolates it exactly), two held and two free points, 1000 midpoint steps:
frozen keeps every held F == I exactly while the free F follow the shear, bit for bit as with the default "evolve",
where the held F drift like the free ones; F of held points is set to I at construction and after a checkpoint load,
with the largest |F - I| removed recorded; a wrong mode is refused."""
import numpy as np
import pytest
import taichi as ti
from types import SimpleNamespace

from fastfsi.coupling.face_mask import AllValidFaceMask
from fastfsi.coupling.immersed_boundary import ImmersedBoundaryCoupler
from fastfsi.coupling.kernel_weights import KernelWeights
from fastfsi.grid.axes import UniformAxes
from fastfsi.io.checkpoint import load_components, read_checkpoint, write_checkpoint
from fastfsi.solid.material_points import MaterialPoints
from fastfsi.solid.materials import NeoHookeanMaterial

SHEAR = 0.1                     # 1/s
DT = 1.0e-3
STEPS = 1000
POSITIONS = np.array([[3.5, 3.5, 3.5], [4.0, 3.5, 4.0], [3.5, 4.5, 3.5], [4.5, 4.0, 4.0]], np.float32)
HELD = np.array([1, 1, 0, 0], np.int32)


@pytest.fixture(scope="module", autouse=True)
def cpu():
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1, opt_level=0, offline_cache=False)


@pytest.fixture(scope="module")
def shear():
    velocity = ti.Vector.field(3, ti.f32, shape=(8, 8, 8))
    k = np.arange(8, dtype=np.float32)
    values = np.zeros((8, 8, 8, 3), np.float32)
    values[..., 0] = SHEAR * (k + 0.5)[None, None, :]              # u_x on the x faces at z = k + 1/2
    velocity.from_numpy(values)
    return velocity


def coupler(velocity, held_f, F0=None):
    points = MaterialPoints(positions_m=POSITIONS, volumes_m3=np.ones(4), material=NeoHookeanMaterial(1000., 1., 10.),
                            tether_mask=HELD)
    if F0 is not None:
        points.F.from_numpy(F0)
    ib = ImmersedBoundaryCoupler(fluid=SimpleNamespace(velocity=velocity, rho=1000.),
                                 weights=KernelWeights(UniformAxes((0., 0., 0.), (1., 1., 1.), (8, 8, 8))),
                                 mask=AllValidFaceMask(), points=points, tether_stiffness_pa_per_m=0.,
                                 interior_shortcut=False, held_f=held_f)
    return points, ib


def run(points, ib, velocity, steps=STEPS):
    for _ in range(steps):
        ib.midpoint_predict(DT)
        ib.midpoint_correct(velocity, DT)
    return points.F.to_numpy()


def test_frozen_keeps_held_f_identity_and_the_free_f_bit_for_bit(shear):
    frozen = run(*coupler(shear, "frozen"), shear)
    evolve = run(*coupler(shear, "evolve"), shear)
    eye = np.eye(3, dtype=np.float32)
    assert all(np.array_equal(frozen[p], eye) for p in (0, 1))                     # held: exactly I
    assert frozen[2:].tobytes() == evolve[2:].tobytes()                             # free: unchanged
    assert np.allclose(evolve[2:, 0, 2], SHEAR * STEPS * DT, rtol=1e-3)             # F_xz = g t
    assert np.allclose(evolve[:2, 0, 2], SHEAR * STEPS * DT, rtol=1e-3)             # evolve: the held F drift too


def test_held_f_is_reset_at_construction_and_after_a_checkpoint_load(shear, tmp_path):
    drifted = np.tile(np.eye(3, dtype=np.float32), (4, 1, 1))
    drifted[:, 0, 2] = [0.2, 0.05, 0.3, 0.3]
    points, ib = coupler(shear, "frozen", F0=drifted)
    assert ib.held_f_reset == dict(max_abs_f_minus_i=pytest.approx(0.2), held_points=2)
    assert np.array_equal(points.F.to_numpy()[:2], np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)))
    assert np.array_equal(points.F.to_numpy()[2:], drifted[2:])
    points_e, ib_e = coupler(shear, "evolve")
    saved = run(points_e, ib_e, shear, steps=100)
    path = tmp_path / "held.npz"
    write_checkpoint(path, dict(points=points_e, body=ib_e), dict(step=100), dict(test="held_f"))
    arrays, host, _, _, _ = read_checkpoint(path)
    points_f, ib_f = coupler(shear, "frozen")
    load_components(dict(points=points_f, body=ib_f), arrays, host)
    loaded = points_f.F.to_numpy()
    assert np.array_equal(loaded[:2], np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)))
    assert loaded[2:].tobytes() == saved[2:].tobytes()
    removed = max(np.linalg.norm(saved[p] - np.eye(3)) for p in (0, 1))
    assert ib_f.held_f_reset["held_points"] == 2
    assert ib_f.held_f_reset["max_abs_f_minus_i"] == pytest.approx(removed, rel=1e-5)
    load_components(dict(points=points_e, body=ib_e), arrays, host)                  # evolve keeps the drift
    assert points_e.F.to_numpy().tobytes() == saved.tobytes() and ib_e.held_f_reset is None


def test_a_wrong_mode_is_refused(shear):
    with pytest.raises(ValueError, match="held_f"):
        coupler(shear, "frozen_at_load")
    _, ib = coupler(shear, "evolve")
    with pytest.raises(ValueError, match="frozen"):
        ib.reset_held_deformation()
