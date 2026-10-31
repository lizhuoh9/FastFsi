"""Per-body materials (the s26 outer chamber, Root 2026-10-31): MaterialPoints(body_materials={body id: material}).
A linear shear u_x = g z on a uniform grid (IB_4 interpolates it exactly) deforms four points of bodies 1, 1, 2, 2 over
200 midpoint steps; F does not depend on the material, so the stress compares point by point.  A table holding only
the main material is bit for bit the default; body 2's own mu and kappa give its stress exactly as a run made of that
material; the elastic limits take the stiffest body; the excess mass takes each point's own density and switches on
when any body is heavier than the fluid; bad tables are refused; the fingerprint carries the table."""
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

SHEAR = 0.5                     # 1/s
DT = 1.0e-3
STEPS = 200
POSITIONS = np.array([[3.5, 3.5, 3.5], [4.0, 3.5, 4.0], [3.5, 4.5, 3.5], [4.5, 4.0, 4.0]], np.float32)
BODIES = np.array([1, 1, 2, 2], np.int32)
MAIN = NeoHookeanMaterial(1000., 1., 10.)
STIFF = NeoHookeanMaterial(1000., 5., 50.)


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


def coupler(velocity, material=MAIN, table=None, rho_f=1000., excess_mass_mode="auto"):
    points = MaterialPoints(positions_m=POSITIONS, volumes_m3=np.full(4, 2.0), material=material,
                            tether_mask=np.zeros(4, np.int32), body_id=BODIES, body_materials=table)
    fluid = SimpleNamespace(velocity=velocity, force=ti.Vector.field(3, ti.f32, shape=(8, 8, 8)), rho=rho_f)
    ib = ImmersedBoundaryCoupler(fluid=fluid, weights=KernelWeights(UniformAxes((0., 0., 0.), (1., 1., 1.), (8, 8, 8))),
                                 mask=AllValidFaceMask(), points=points, tether_stiffness_pa_per_m=0.,
                                 interior_shortcut=False, excess_mass_mode=excess_mass_mode)
    return points, ib, fluid


def deform_and_spread(points, ib, fluid, velocity):
    for _ in range(STEPS):
        ib.midpoint_predict(DT)
        ib.midpoint_correct(velocity, DT)
    ib.spread_force(dt_s=DT)
    return points.F.to_numpy(), points.tau.to_numpy(), fluid.force.to_numpy()


def neo_hookean_tau(F, mu, kappa):
    J = np.linalg.det(F)
    b = F @ F.T
    return mu * J ** (-2.0 / 3.0) * (b - np.trace(b) / 3.0 * np.eye(3)) + kappa * (J - 1.0) * J * np.eye(3)


def test_a_table_of_the_main_material_is_bit_for_bit_the_default(shear):
    F0, tau0, force0 = deform_and_spread(*coupler(shear), shear)
    F1, tau1, force1 = deform_and_spread(*coupler(shear, table={1: MAIN, 2: MAIN}), shear)
    assert F1.tobytes() == F0.tobytes() and tau1.tobytes() == tau0.tobytes() and force1.tobytes() == force0.tobytes()
    assert np.allclose(F0[:, 0, 2], SHEAR * STEPS * DT, rtol=1e-3)                  # F_xz = g t: the points deform


def test_each_body_takes_its_own_mu_and_kappa(shear):
    _, tau_main, _ = deform_and_spread(*coupler(shear), shear)
    _, tau_stiff, _ = deform_and_spread(*coupler(shear, material=STIFF), shear)
    F, tau, _ = deform_and_spread(*coupler(shear, table={2: STIFF}), shear)
    assert tau[:2].tobytes() == tau_main[:2].tobytes()                               # body 1 keeps the main material
    assert tau[2:].tobytes() == tau_stiff[2:].tobytes()                              # body 2 is its own material
    for p in (2, 3):
        assert np.allclose(tau[p], neo_hookean_tau(F[p].astype(np.float64), 5., 50.), rtol=1e-5, atol=1e-7)
    assert not np.allclose(tau[2:], tau_main[2:])


def test_the_elastic_limits_take_the_stiffest_body(shear):
    points, ib, _ = coupler(shear, table={2: STIFF})
    assert points.shear_modulus_max_pa() == 5.0
    assert ib.elastic_dt_limit(0.3) == pytest.approx(0.3 * 1.0 / np.sqrt(5.0 / 1000.))
    assert points.elastic_dt_limit(0.3, 1.0) == pytest.approx(0.3 / np.sqrt(5.0 / 1000.))
    points_d, ib_d, _ = coupler(shear)                                               # the default: the material's own
    assert points_d.shear_modulus_max_pa() == 1.0 and points_d.body_material_report() is None
    assert ib_d.elastic_dt_limit(0.3) == pytest.approx(0.3 / np.sqrt(1.0 / 1000.))
    assert points.body_material_report() == {1: (1000., 1., 10.), 2: (1000., 5., 50.)}


def test_the_excess_mass_takes_each_point_density(shear):
    membrane, sleeve = NeoHookeanMaterial(1040., 1., 50.), NeoHookeanMaterial(1070., 3., 150.)
    points, ib, _ = coupler(shear, material=membrane, table={2: sleeve}, rho_f=1025.)
    assert ib.excess_mass is not None
    assert np.allclose(ib.excess_mass.mass.to_numpy(), [30., 30., 90., 90.])          # (rho_b - rho_f) x 2 m^3
    assert ib.excess_mass.reference_mass_sum == pytest.approx(240.)
    assert ib.excess_mass.neutral_elastic_dt == pytest.approx(0.3 * 1.0 / np.sqrt(3.0 / 1025.))
    report = ib.density_report()
    assert report["body_densities_kgm3"] == {1: 1040., 2: 1070.}
    assert report["target_reference_density_ratio"] == pytest.approx(1040. / 1025.)  # the main material's
    _, light, _ = coupler(shear, material=NeoHookeanMaterial(1025., 1., 50.), table={2: sleeve}, rho_f=1025.)
    assert light.excess_mass is not None                                             # one heavier body switches it on
    assert np.allclose(light.excess_mass.mass.to_numpy(), [0., 0., 90., 90.])
    _, plain, _ = coupler(shear, material=membrane, rho_f=1025.)
    assert "body_densities_kgm3" not in plain.density_report()
    assert np.allclose(plain.excess_mass.mass.to_numpy(), 30.)


def test_bad_tables_are_refused(shear):
    for key in (4, -1, "2", True, 2.0):
        with pytest.raises(ValueError, match="per-body material ids"):
            coupler(shear, table={key: STIFF})
    with pytest.raises(TypeError, match="NeoHookeanMaterial"):
        coupler(shear, table={2: (1000., 5., 50.)})
    with pytest.raises(ValueError, match="rho_s < rho_f"):
        coupler(shear, material=NeoHookeanMaterial(1040., 1., 50.), table={2: NeoHookeanMaterial(1000., 1., 50.)},
                rho_f=1025.)
    with pytest.raises(ValueError, match="every body id"):
        MaterialPoints(positions_m=POSITIONS, volumes_m3=np.ones(4), material=MAIN, tether_mask=np.zeros(4, np.int32),
                       body_id=np.array([1, 1, 2, 7], np.int32), body_materials={2: STIFF})


def test_the_fingerprint_carries_the_table(shear):
    _, default, _ = coupler(shear)
    _, again, _ = coupler(shear)
    _, main_table, _ = coupler(shear, table={1: MAIN, 2: MAIN})
    _, stiff_table, _ = coupler(shear, table={2: STIFF})
    assert default.fingerprint() == again.fingerprint()
    assert len({default.fingerprint(), main_table.fingerprint(), stiff_table.fingerprint()}) == 3
