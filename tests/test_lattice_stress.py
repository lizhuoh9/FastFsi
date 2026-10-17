"""The lattice correction of the solid stress (coupling.lattice_stress) against the point stress of the coupler: an affine
patch, the zero-energy mode of the point stress coupled through a small incompressible grid, and a free elastic bar's
axial oscillation (no volumetric locking)."""
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from fastfsi.coupling.face_mask import AllValidFaceMask
from fastfsi.coupling.immersed_boundary import ImmersedBoundaryCoupler
from fastfsi.coupling.kernel_weights import KernelWeights
from fastfsi.coupling.lattice_stress import LatticeStress, lattice_energy, validate_stress_source
from fastfsi.grid.axes import UniformAxes
from fastfsi.solid.material_points import MaterialPoints
from fastfsi.solid.materials import NeoHookeanMaterial
from fastfsi.solid.reference_lattice import TETS, reference_lattice_cells

MU, KAPPA = 1.0, 50.0                   # the robot's kappa / mu
A_AFFINE = np.array([[1.04, 0.02, -0.01], [0.015, 0.97, 0.03], [-0.02, 0.01, 1.02]])
OFFSET = np.array([0.01, -0.02, 0.005])


@pytest.fixture(scope="module", autouse=True)
def cpu():
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1, opt_level=0, offline_cache=False)


def block(n=(4, 4, 4), spacing=0.5, start=4.25):
    """Points on a lattice and its cells."""
    axes = [start + spacing * np.arange(k) for k in n]
    positions = np.array(list(itertools.product(*axes)), dtype=np.float64)
    cells, refused = reference_lattice_cells(positions, np.full_like(positions, spacing))
    assert refused == 0
    return positions, cells


def make(positions, cells, source, volumes=None, grid=(14, 14, 14)):
    """A coupler on a uniform unit grid (all faces valid, equal densities: no excess mass)."""
    fluid = SimpleNamespace(rho=1.0, velocity=ti.Vector.field(3, ti.f32, shape=grid),
                            force=ti.Vector.field(3, ti.f32, shape=grid))
    volumes = np.full(len(positions), 0.125) if volumes is None else volumes
    points = MaterialPoints(positions_m=positions, volumes_m3=volumes, material=NeoHookeanMaterial(1.0, MU, KAPPA),
                            tether_mask=np.zeros(len(positions), np.int32))
    lattice = LatticeStress(points, cells, positions) if source == "lattice_correction" else None
    body = ImmersedBoundaryCoupler(fluid=fluid, weights=KernelWeights(UniformAxes((0., 0., 0.), (1., 1., 1.), grid)),
                                   mask=AllValidFaceMask(), points=points, tether_stiffness_pa_per_m=0.,
                                   interior_shortcut=False, excess_mass_mode="off", lattice_stress=lattice)
    return body, points, fluid


def place(points, positions, A, point_F):
    """Positions moved by the affine map A about the block centre; the points' F set to point_F."""
    u = (positions - positions.mean(axis=0)) @ (A - np.eye(3)).T + OFFSET
    points.u.from_numpy(u.astype(np.float32))
    points.refresh_positions()
    points.F.from_numpy(np.broadcast_to(point_F.astype(np.float32), (len(positions), 3, 3)).copy())


def grid_moments(fluid):
    """Total force and first moment sum_faces f_c (x_face)_d of the spread force density (unit faces)."""
    f = fluid.force.to_numpy().astype(np.float64)
    idx = np.indices(f.shape[:3]).astype(np.float64)
    dipole = np.zeros((3, 3))
    for c in range(3):
        pos = idx + 0.5
        pos[c] -= 0.5                                                   # the face of component c
        for d in range(3):
            dipole[c, d] = (f[..., c] * pos[d]).sum()
    return f.reshape(-1, 3).sum(axis=0), dipole


def lumped_volumes(lattice, n):
    vol = np.zeros(n)
    for t in range(6):
        np.add.at(vol, lattice.cells_host[:, list(TETS[t])], np.repeat(lattice.tet_volume_m3[:, t:t + 1] / 4.0, 4, axis=1))
    return vol


def kirchhoff(A):
    J = np.linalg.det(A)
    b = A @ A.T
    return MU * J ** (-2 / 3) * (b - np.trace(b) / 3 * np.eye(3)) + KAPPA * (J - 1) * J * np.eye(3)


def test_unknown_source_is_refused():
    with pytest.raises(ValueError):
        validate_stress_source("lattice")
    assert validate_stress_source("point_F") == "point_F"


def test_affine_patch_correction_is_zero_when_F_follows():
    positions, cells = block()
    body, points, _ = make(positions, cells, "lattice_correction")
    place(points, positions, A_AFFINE, A_AFFINE)
    body.lattice_stress.compute(MU, KAPPA)
    f = body.lattice_stress.force.to_numpy().astype(np.float64)
    scale = np.abs(kirchhoff(A_AFFINE)).max() * body.lattice_stress.tet_volume_m3.max() / 0.5   # a tetrahedron's force
    np.testing.assert_allclose(f, 0.0, atol=1e-4 * scale)
    assert body.lattice_stress.report()["max_F_minus_point_F"] < 1e-5


def test_affine_patch_with_F_left_behind_matches_the_point_stress():
    """Positions moved by A while the points' F stays I: the correction is then the whole lattice force, internal
    (zero total force and torque, balanced interior nodes) with the force dipole -V tau(A); spread on the grid it carries
    the same total force (0) and first moment as the point stress of F = A on the same volume."""
    positions, cells = block()
    body, points, fluid = make(positions, cells, "lattice_correction")
    lattice = body.lattice_stress
    place(points, positions, A_AFFINE, np.eye(3))
    lattice.compute(MU, KAPPA)
    f = lattice.force.to_numpy().astype(np.float64)
    x = points.positions()
    scale = np.abs(f).max()
    assert scale > 0.0
    np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-5 * scale)
    np.testing.assert_allclose(np.cross(x, f).sum(axis=0), 0.0, atol=1e-4 * scale)
    interior = np.all((positions > 4.3) & (positions < 5.7), axis=1)
    assert interior.sum() == 8
    np.testing.assert_allclose(f[interior], 0.0, atol=1e-4 * scale)
    hull = lattice.tet_volume_m3.sum()
    tau = kirchhoff(A_AFFINE)
    np.testing.assert_allclose(np.einsum("pi,pj->ij", f, x), -hull * tau, atol=2e-4 * np.abs(tau).max() * hull)
    ref_body, ref_points, ref_fluid = make(positions, cells, "point_F", volumes=lumped_volumes(lattice, len(positions)))
    place(ref_points, positions, A_AFFINE, A_AFFINE)
    ref_body.spread_force()
    body.spread_force()                     # the point stress of F = I is zero: the grid holds the correction alone
    total, dipole = grid_moments(fluid)
    ref_total, ref_dipole = grid_moments(ref_fluid)
    size = np.abs(ref_dipole).max()
    np.testing.assert_allclose(total, 0.0, atol=1e-5 * size)
    np.testing.assert_allclose(ref_total, 0.0, atol=1e-5 * size)
    np.testing.assert_allclose(dipole, ref_dipole, atol=1e-3 * size)


def project(u):
    """The exact discrete projection of a periodic MAC field u (3, N, N, N) (face c at index i: cells i - 1 and i)."""
    n = u.shape[1]
    k = 2 * np.pi * np.fft.fftfreq(n)
    th = np.meshgrid(k, k, k, indexing="ij")
    uh = [np.fft.fftn(u[c]) for c in range(3)]
    div = sum((np.exp(1j * th[c]) - 1.0) * uh[c] for c in range(3))
    lam = sum(2.0 * np.cos(th[c]) - 2.0 for c in range(3))
    lam[0, 0, 0] = 1.0
    phi = div / lam
    phi[0, 0, 0] = 0.0
    return np.stack([np.real(np.fft.ifftn(uh[c] - (1.0 - np.exp(-1j * th[c])) * phi)) for c in range(3)])


@pytest.mark.parametrize("source", ("point_F", "lattice_correction"))
def test_zero_energy_stretch_through_an_incompressible_grid(source):
    """A 6^3 block (spacing 0.5 cell) stretched isochorically by 10 % along x while its F stays I -- a deformation the
    point stress does not see -- coupled through a periodic 16^3 grid (spread, u += dt f / rho, exact projection, the
    points move with the interpolated velocity) at the robot's explicit step dt = 0.3 h / c_s.  point_F: nothing moves.
    lattice_correction: the stretch is pulled back below half within 40 steps and stays bounded over 200."""
    n = 16
    positions, cells = block(n=(6, 6, 6), spacing=0.5, start=6.0)
    body, points, fluid = make(positions, cells, source, grid=(n, n, n))
    u = 0.1 * (positions - positions.mean(axis=0)) * np.array([1.0, -0.5, -0.5])
    points.u.from_numpy(u.astype(np.float32))
    points.refresh_positions()
    left, right = np.argmin(positions[:, 0]), np.argmax(positions[:, 0])
    start = u[right, 0] - u[left, 0]
    dt = 0.3 * 1.0 / math.sqrt(MU / 1.0)
    velocity = np.zeros((3, n, n, n))
    stretch = []
    for _ in range(200):
        body.spread_force()
        velocity = project(velocity + dt * np.moveaxis(fluid.force.to_numpy().astype(np.float64), -1, 0))
        fluid.velocity.from_numpy(np.moveaxis(velocity, 0, -1).astype(np.float32))
        body.move_with_fluid(dt)
        x = points.positions()
        stretch.append((x[right, 0] - x[left, 0]) - (positions[right, 0] - positions[left, 0]))
    stretch = np.array(stretch)
    if source == "point_F":
        np.testing.assert_allclose(stretch, start, rtol=0, atol=1e-6)
        assert np.abs(velocity).max() == 0.0
    else:
        assert np.abs(stretch[:40]).min() < 0.5 * start, stretch[:40]
        assert np.abs(stretch).max() <= 1.05 * start
        assert np.isfinite(velocity).all() and np.abs(velocity).max() < 1.0


def test_free_bar_axial_oscillation_has_the_youngs_modulus_frequency_and_bounded_energy():
    """Solid-only dynamics of the lattice correction with the points' F held at I (the correction is then the whole
    lattice force; lumped masses, velocity Verlet): the first free-free axial mode of a slender bar oscillates at
    c / (2 L), c = sqrt(E / rho), E = 9 K mu / (3 K + mu) (lateral contraction free); a locking volumetric term would
    give the constrained modulus K + 4/3 mu (a frequency 4x higher)."""
    n_long, h = 25, 0.5
    positions, cells = block(n=(n_long, 2, 2), spacing=h, start=0.25)
    body, points, _ = make(positions, cells, "lattice_correction", grid=(16, 4, 4))
    lattice = body.lattice_stress
    rho = 1.0
    mass = rho * lumped_volumes(lattice, len(positions))
    L = (n_long - 1) * h                                    # the lattice bar (the hull between the end points)
    period = 2 * L / math.sqrt(9 * KAPPA * MU / (3 * KAPPA + MU) / rho)
    dt = 0.05 * h / math.sqrt((KAPPA + 4 / 3 * MU) / rho)
    xs = positions[:, 0] - positions[:, 0].mean()
    u = np.zeros_like(positions)
    u[:, 0] = 1e-3 * np.sin(math.pi * xs / L)               # the first free-free axial mode shape (odd about the middle)
    v = np.zeros_like(positions)
    end = positions[:, 0] == positions[:, 0].max()

    def force(u):
        points.u.from_numpy(u.astype(np.float32))
        lattice.compute(MU, KAPPA)
        return lattice.force.to_numpy().astype(np.float64)

    signal, energies = [], []
    f = force(u)
    for step in range(int(2.2 * period / dt)):              # velocity Verlet: u, v and the energy at the same time
        v += 0.5 * dt * f / mass[:, None]
        u += dt * v
        f = force(u)
        v += 0.5 * dt * f / mass[:, None]
        signal.append(u[end, 0].mean())
        if step % 10 == 0:
            energies.append(0.5 * (mass * (v ** 2).sum(axis=1)).sum()
                            + lattice_energy(lattice.tet_deformation_gradients(), lattice.tet_volume_m3, MU, KAPPA))
    s = np.array(signal)
    crossings = np.flatnonzero((s[:-1] > 0) & (s[1:] <= 0))
    measured = (crossings[-1] - crossings[0]) * dt / (len(crossings) - 1)
    assert abs(measured / period - 1.0) < 0.05, (measured, period)
    e = np.array(energies)
    assert e.max() / e.min() < 1.02, (e.min(), e.max())
