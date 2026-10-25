"""PenaltyContact(rest_cap=True) (the USER's decision 4.3 (a), 2026-10-25): each free point's wall contact distance is
capped at its own rest wall distance, d_c,p = min(d_c,p(n), phi_p(X0) - REST_CAP_MARGIN_M), with phi_p(X0) the step
kernel's own wall distance at the reference position.  A rigid slab z < 2 mm under an 8 mm cube of 1 mm cells (the
obstacle: the cells below it), four points of volume (0.5 mm)^3 (d_c along z = a_z = 0.25 mm), k_c given:
  p0 free, body 2, phi(X0) = +0.1 mm (inside its contact distance at rest: pushed today);
  p1 free, body 1, phi(X0) = 1.5 mm (out of contact);
  p2 free, body 1, phi(X0) = -0.1 mm (inside the wall at rest);
  p3 held, body 2, phi(X0) = +0.1 mm (never in contact).
Off (the default) is bit for bit the call without the argument and today's law F = k V0 (d_c - phi) n.  On: no force
at rest (also on the selected body's obstacle-face wall distance), the capped and phi <= 0 points reported; a point
moved from rest toward the wall is pushed with the same k (slope k V0) once it is closer than at rest by the margin;
a point moved away is not pushed."""
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from fastfsi.coupling.face_mask import AllValidFaceMask
from fastfsi.coupling.immersed_boundary import ImmersedBoundaryCoupler
from fastfsi.coupling.kernel_weights import KernelWeights
from fastfsi.grid.axes import UniformAxes
from fastfsi.grid.cartesian_grid import CartesianGrid
from fastfsi.solid.contact import REST_CAP_MARGIN_M, PenaltyContact
from fastfsi.solid.material_points import MaterialPoints
from fastfsi.solid.materials import NeoHookeanMaterial

MM = 1.0e-3
N = 8
TOP = 2.0 * MM                        # the slab's top face
K_C = 1.0e9                           # N/m^4
V0 = (0.5 * MM) ** 3                  # half extents 0.25 mm: d_c along z = 0.25 mm
D_C = 0.25 * MM
POSITIONS = np.array([[3.5, 3.5, 2.1], [4.5, 4.5, 3.5], [5.5, 3.5, 1.9], [3.5, 5.5, 2.1]]) * MM
BODY = np.array([2, 1, 1, 2], np.int32)
HELD = np.array([0, 0, 0, 1], np.int32)
DT = 1.0e-6


@pytest.fixture(scope="module", autouse=True)
def cpu():
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1, opt_level=0, offline_cache=False)


def slab_triangles(lo, hi):
    """A closed box (12 triangles) from corner lo to corner hi (m)."""
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([[c[a], c[b], c[d]] for a, b, _, d in quads] + [[c[b], c[e], c[d]] for _, b, e, d in quads])


def build(**options):
    grid = CartesianGrid.uniform(bounds_min_m=(0.0, 0.0, 0.0), bounds_max_m=(N * MM,) * 3, grid_nodes=(N,) * 3)
    obstacle = ti.field(ti.i32, shape=(N,) * 3)
    mask = np.zeros((N,) * 3, np.int32)
    mask[:, :, :2] = 1                                           # cell centres z = 0.5, 1.5 mm: inside the slab
    obstacle.from_numpy(mask)
    axes = UniformAxes((0.0, 0.0, 0.0), (MM,) * 3, (N,) * 3)
    points = MaterialPoints(positions_m=POSITIONS, volumes_m3=np.full(4, V0), material=NeoHookeanMaterial(1000., 1e4, 1e5),
                            tether_mask=HELD, body_id=BODY)
    coupler = ImmersedBoundaryCoupler(fluid=SimpleNamespace(velocity=ti.Vector.field(3, ti.f32, shape=(N,) * 3), rho=1000.),
                                      weights=KernelWeights(axes), mask=AllValidFaceMask(), points=points,
                                      tether_stiffness_pa_per_m=0., interior_shortcut=False)
    contact = PenaltyContact(SimpleNamespace(grid=grid, obstacle=obstacle), axes, coupler,
                             slab_triangles((-MM, -MM, -MM), (9 * MM, 9 * MM, TOP)), k_c=K_C, **options)
    return points, contact


def forces(points, contact, z=None):
    """f_extra after one contact evaluation at the positions (the reference ones, or with the given z per point)."""
    x = POSITIONS.copy()
    if z is not None:
        x[:, 2] = z
    points.x.from_numpy(x.astype(np.float32))
    points.f_extra.fill(0.0)
    contact.apply_point_forces(DT)
    return points.f_extra.to_numpy().astype(np.float64)


def test_off_is_the_default_law_bit_for_bit():
    default = forces(*build())
    off = forces(*build(rest_cap=False))
    assert default.tobytes() == off.tobytes()
    expected = np.zeros((4, 3))
    expected[0, 2] = K_C * V0 * (D_C - 0.1 * MM)                 # today's law: p0 and p2 pushed at rest
    expected[2, 2] = K_C * V0 * (D_C + 0.1 * MM)
    assert np.allclose(default, expected, rtol=1e-5, atol=1e-12)
    assert "rest_cap" not in build()[1].setup_report()


@pytest.mark.parametrize("options", [{}, dict(selected_wall_on_obstacle_faces=True, selected_body_id=2)])
def test_the_cap_gives_no_force_at_rest_and_reports_the_capped_points(options):
    points, contact = build(rest_cap=True, **options)
    assert not forces(points, contact).any()                     # exactly zero at rest
    state = contact.report(0.0, DT)
    assert state["contact_points"] == 0 and state["contact_force_n"] == 0.0
    report = contact.setup_report()["rest_cap"]
    assert report["margin_m"] == REST_CAP_MARGIN_M
    assert report["capped_points"] == 2 and report["rest_phi_le_0_points"] == 1    # p0, p2; p2; the held p3 not
    lo, hi = report["capped_rest_phi_m"]
    assert lo == pytest.approx(-0.1 * MM, rel=1e-4) and hi == pytest.approx(0.1 * MM, rel=1e-4)
    assert report["by_body"] == {"1": dict(capped_points=1, rest_phi_le_0_points=1),
                                 "2": dict(capped_points=1, rest_phi_le_0_points=0)}
    assert report["rest_preload_removed_max_m"] == pytest.approx(D_C + 0.1 * MM, rel=1e-4)


def test_a_point_moved_toward_the_wall_is_pushed_with_the_same_k():
    points, contact = build(rest_cap=True)
    z0 = POSITIONS[:, 2]
    slope = K_C * V0
    for delta in (0.02 * MM, 0.06 * MM):                         # p0 and p2 toward the wall, p1 into the box rule
        f = forces(points, contact, z0 - np.array([delta, 1.4 * MM - delta, delta, delta]))
        assert f[0, 2] == pytest.approx(slope * (delta - REST_CAP_MARGIN_M), rel=1e-4)
        assert f[2, 2] == pytest.approx(slope * (delta - REST_CAP_MARGIN_M), rel=1e-4)
        assert f[1, 2] == pytest.approx(slope * (D_C - (0.1 * MM + delta)), rel=1e-4)   # uncapped: today's law
        assert not f[3].any() and not f[:, :2].any()
    away = forces(points, contact, z0 + 0.1 * MM)                 # moved away from rest: not pushed
    assert not away.any()
