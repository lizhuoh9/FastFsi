"""Physical excess inertia through real IB kernels, including a filled periodic MAC known solution."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from fastfsi.coupling.face_mask import AllValidFaceMask
from fastfsi.coupling.immersed_boundary import ImmersedBoundaryCoupler, PlanarImmersedBoundaryCoupler
from fastfsi.coupling.kernel_weights import KernelWeights
from fastfsi.errors import SimulationHardStop
from fastfsi.grid.axes import UniformAxes
from fastfsi.io.checkpoint import load_components, read_checkpoint, write_checkpoint
from fastfsi.solid.material_points import MaterialPoints, PlanarMaterialPoints
from fastfsi.solid.materials import NeoHookeanMaterial


@pytest.fixture(scope="module", autouse=True)
def cpu():
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1, opt_level=0, offline_cache=False)


def make_body(dim=3, rho_s=10., rho_f=1., *, tether=(0,), clamp="penalty", omega=20.,
              positions=None, volumes=None, initial_v=(.2, -.1, .05), origin=None, grid=None):
    grid = grid or ((1, 8, 8) if dim == 2 else (8, 8, 8))
    origin = origin or (0., 0., 0.)
    positions = np.asarray(positions if positions is not None else [[3.5] * dim], dtype=np.float32)
    volumes = np.asarray(volumes if volumes is not None else np.ones(len(positions)), dtype=np.float32)
    fluid = SimpleNamespace(rho=rho_f, velocity=ti.Vector.field(3, ti.f32, shape=grid),
                            force=ti.Vector.field(3, ti.f32, shape=grid))
    fluid.velocity.from_numpy(np.broadcast_to(np.array(initial_v, np.float32), grid + (3,)).copy())
    mat = NeoHookeanMaterial(rho_s, 1., 10.)
    if dim == 2:
        points = PlanarMaterialPoints(positions_yz_m=positions, areas_m2=volumes, material=mat,
                                      tether_mask=np.asarray(tether, np.int32))
        body = PlanarImmersedBoundaryCoupler(fluid=fluid, points=points, tether_stiffness_pa_per_m=0.,
            grid_origin_m=origin, grid_spacing_m=(1., 1., 1.), grid_nodes=grid, span_m=1., clamp=clamp,
            excess_mass_omega_rad_s=omega)
    else:
        points = MaterialPoints(positions_m=positions, volumes_m3=volumes, material=mat,
                                tether_mask=np.asarray(tether, np.int32))
        body = ImmersedBoundaryCoupler(fluid=fluid, points=points, tether_stiffness_pa_per_m=0.,
            weights=KernelWeights(UniformAxes(origin, (1., 1., 1.), grid)), mask=AllValidFaceMask(),
            interior_shortcut=False, clamp=clamp, excess_mass_omega_rad_s=omega)
    return body, fluid, points


@pytest.mark.parametrize("dim", (2, 3))
def test_positive_reference_mass_and_real_nonzero_initial_velocity(dim):
    body, fluid, points = make_body(dim)
    ex = body.excess_mass
    assert ex is not None
    np.testing.assert_array_equal(ex.mass.to_numpy(), [9.])
    np.testing.assert_array_equal(ex.displacement.to_numpy(), np.zeros((1, dim)))
    np.testing.assert_allclose(ex.velocity.to_numpy(), np.array([.2, -.1, .05])[-dim:][None, :], atol=2e-7, rtol=0.)
    assert ex.omega_rad_s == 20.
    points.F.from_numpy((np.eye(dim, dtype=np.float32) * 1.01)[None])
    assert ex.mass.to_numpy()[0] == 9.  # reference mass is never multiplied by current J


@pytest.mark.parametrize("dim", (2, 3))
def test_same_saved_spring_impulse_is_equal_and_opposite(dim):
    body, fluid, points = make_body(dim, initial_v=(0., 0., 0.))
    ex = body.excess_mass
    offset = np.array([[.01, -.005, .003]])[:, -dim:]
    ex.displacement.from_numpy(offset)
    dt = min(.001, body.excess_mass_dt_limit())
    k = ex.mass.to_numpy()[0] * ex.omega_rad_s ** 2
    if dim == 2:
        body.spread_elastic_force(dt_s=dt)
        grid_impulse = fluid.force.to_numpy()[0, :, :, 1:].astype(np.float64).sum(axis=(0, 1)) * dt
    else:
        body.spread_force(dt_s=dt)
        grid_impulse = fluid.force.to_numpy().astype(np.float64).sum(axis=(0, 1, 2)) * dt
    saved_force = ex.spring_force.to_numpy().copy()
    np.testing.assert_array_equal(saved_force, (k * offset).astype(np.float32).astype(np.float64))
    assert ex.mass.to_numpy()[0] == 9.
    # Update real grid velocity from the real spread; move consumes precisely that saved force.
    fluid.velocity.from_numpy(fluid.force.to_numpy() * np.float32(dt / fluid.rho))
    body.move_with_fluid(dt)
    particle_impulse = (ex.mass.to_numpy()[:, None] * ex.velocity.to_numpy()).sum(axis=0)
    roundoff = 8. * np.finfo(np.float32).eps * np.linalg.norm(particle_impulse)
    np.testing.assert_allclose(grid_impulse + particle_impulse, 0., atol=roundoff, rtol=0.)
    np.testing.assert_allclose(ex.displacement.to_numpy(), offset - dt * dt * saved_force / 9., atol=1e-14)


@pytest.mark.parametrize("dim", (2, 3))
def test_fixed_stiffness_and_overlap_dt_limit(dim):
    positions = np.full((4, dim), 3.5, np.float32)
    body, fluid, points = make_body(dim, positions=positions, volumes=[1.] * 4, tether=[0] * 4, initial_v=(0., 0., 0.))
    ex = body.excess_mass
    dt = body.excess_mass_dt_limit()
    # Collocated point mobility must include all four marker masses, not only a single-point diagonal.
    known_diagonal = 9. * (3. / 8.) ** dim
    assert ex.last_beta >= 4. * known_diagonal * (1. - 3e-7)
    assert dt * ex.omega_rad_s * math.sqrt(1. + ex.last_beta) <= .5 * (1. + 1e-12)
    ex.displacement.from_numpy(np.full((4, dim), .001, np.float64))
    for scale in (1., .5):
        step = dt * scale
        if dim == 2: body.spread_elastic_force(dt_s=step)
        else: body.spread_force(dt_s=step)
        np.testing.assert_array_equal(ex.spring_force.to_numpy(), np.full((4, dim), np.float64(np.float32(9. * 20. ** 2 * .001))))
        body.move_with_fluid(step)
        # Restore displacement so both spreads probe the same geometry and spring extension.
        ex.displacement.from_numpy(np.full((4, dim), .001, np.float64))
    with pytest.raises(SimulationHardStop, match="excess_mass_dt_limit"):
        if dim == 2: body.spread_elastic_force(dt_s=dt * 2.)
        else: body.spread_force(dt_s=dt * 2.)


@pytest.mark.parametrize("rho_s,rho_f", ((.5, 1.), (1., -1.), (1., float("nan"))))
def test_unsupported_or_bad_density_fails_closed(rho_s, rho_f):
    with pytest.raises(ValueError): make_body(rho_s=rho_s, rho_f=rho_f)


@pytest.mark.parametrize("dim", (2, 3))
def test_both_existing_clamps_move_tethers_and_keep_their_reference_inertia(dim):
    direct, _, _ = make_body(dim, tether=[1], clamp="direct")
    penalty, _, _ = make_body(dim, tether=[1], clamp="penalty")
    assert direct.excess_mass.mass.to_numpy()[0] == 9.
    assert penalty.excess_mass.mass.to_numpy()[0] == 9.


@pytest.mark.parametrize("dim", (2, 3))
def test_midpoint_and_euler_steps_must_not_be_mixed_or_reordered(dim):
    body, fluid, points = make_body(dim)
    ex = body.excess_mass
    before = points.x.to_numpy().tobytes(), ex.displacement.to_numpy().tobytes()
    with pytest.raises(ValueError, match="finish_midpoint"): body.midpoint_correct(fluid.velocity, .001)
    assert (points.x.to_numpy().tobytes(), ex.displacement.to_numpy().tobytes()) == before
    body.midpoint_predict(.001)
    with pytest.raises(ValueError, match="in progress"): body.midpoint_predict(.001)
    with pytest.raises(ValueError, match="predictor"):
        if dim == 2: body.spread_elastic_force(dt_s=.002)
        else: body.spread_force(dt_s=.002)
    if dim == 2: body.spread_elastic_force(dt_s=.001)
    else: body.spread_force(dt_s=.001)
    with pytest.raises(ValueError, match="midpoint step is in progress"): body.move_with_fluid(.001)
    body.midpoint_correct(fluid.velocity, .001)
    assert ex.last_integrator == "stormer_verlet" and ex.last_accepted_dt == .001
    if dim == 2: body.spread_elastic_force(dt_s=.001)
    else: body.spread_force(dt_s=.001)
    with pytest.raises(ValueError, match="finish_midpoint"): body.midpoint_correct(fluid.velocity, .001)
    body.move_with_fluid(.001)
    assert ex.last_integrator == "symplectic_euler"


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("field", ("displacement", "velocity"))
def test_nonfinite_partner_is_rejected_without_mutation(dim, field):
    body, _, _ = make_body(dim)
    ex = body.excess_mass
    getattr(ex, field).fill(float("nan"))
    before = {name: getattr(ex, name).to_numpy().tobytes() for name in ex.CHECKPOINT_FIELDS}
    with pytest.raises(SimulationHardStop) as caught: ex.validate_state()
    assert caught.value.witness["component"] == "excess_mass"
    json.dumps(caught.value.witness, allow_nan=False)
    assert before == {name: getattr(ex, name).to_numpy().tobytes() for name in ex.CHECKPOINT_FIELDS}


@pytest.mark.parametrize("dim", (2, 3))
def test_partner_checkpoint_and_nan_scratch_are_rewritten_before_use(dim, tmp_path):
    body, fluid, points = make_body(dim)
    ex = body.excess_mass
    ex.displacement.fill(.002)
    path = tmp_path / "ex.npz"
    components = {"excess_mass": ex}
    write_checkpoint(path, components, {"step": 0}, {"fingerprint": ex.fingerprint()})
    arrays, host, loop, manifest, _ = read_checkpoint(path)
    expected = {name: arr.tobytes() for name, arr in arrays["excess_mass"].items()}
    ex.displacement.fill(4.)
    ex.velocity.fill(8.)
    ex.spring_force.fill(float("nan"))
    ex.face_mass_density.fill(float("nan"))
    load_components(components, arrays, host)
    assert expected == {name: ex.state_array(name).tobytes() for name in expected}
    ex.validate_state()
    dt = min(.001, body.excess_mass_dt_limit())
    if dim == 2: body.spread_elastic_force(dt_s=dt)
    else: body.spread_force(dt_s=dt)
    assert np.isfinite(ex.spring_force.to_numpy()).all()
    assert np.isfinite(ex.face_mass_density.to_numpy()).all()
    with pytest.raises(ValueError, match="accepted"): ex.state_array_names()
    body.move_with_fluid(dt)
    assert tuple(ex.state_array_names()) == (("displacement", "velocity", "material_displacement") if dim == 2
                                            else ("displacement", "velocity"))


def test_planar_spring_uses_relative_f64_displacement_at_large_absolute_offset():
    body, _, _ = make_body(2, positions=[[1003.5, 1003.5]], origin=(0., 1000., 1000.), initial_v=(0., 0., 0.))
    ex = body.excess_mass
    ex.displacement.from_numpy(np.array([[1e-6, -2e-6]]))
    body.spread_elastic_force(dt_s=.0001)
    np.testing.assert_array_equal(ex.spring_force.to_numpy(), np.array([[.0036, -.0072]], dtype=np.float32).astype(np.float64))


def test_equal_density_has_no_new_fields_and_matches_original_neutral_kernels(tmp_path):
    # Save the production methods before replacing them with original source at the same runtime state.
    import importlib.util
    from pathlib import Path
    original = Path(__file__).resolve().parents[1] / "dev/density_ratio_20261014_01/implementation/original/fastfsi_dev/src/fastfsi/coupling/immersed_boundary.py.original"
    spec = importlib.util.spec_from_file_location("neutral_ib_before_density", original.with_suffix(".py"))
    # SourceFileLoader supports the preserved .original extension without copying or changing the source.
    from importlib.machinery import SourceFileLoader
    spec = importlib.util.spec_from_loader("neutral_ib_before_density", SourceFileLoader("neutral_ib_before_density", str(original)))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    body, fluid, points = make_body(2, rho_s=1.)
    assert body.excess_mass is None
    before_names = {name for name, value in vars(body).items() if hasattr(value, "to_numpy")}
    oldbody = old.PlanarImmersedBoundaryCoupler(fluid=fluid, points=points, tether_stiffness_pa_per_m=0.,
        grid_origin_m=(0., 0., 0.), grid_spacing_m=(1., 1., 1.), grid_nodes=(1, 8, 8), span_m=1.)
    assert before_names == {name for name, value in vars(oldbody).items() if hasattr(value, "to_numpy")}
    originals = {name: getattr(points, name).to_numpy().copy() for name in ("x", "v", "F")}
    body.spread_elastic_force(dt_s=.001)
    force = fluid.force.to_numpy().tobytes()
    body.move_with_fluid(.001)
    new = {name: getattr(points, name).to_numpy().tobytes() for name in originals}
    for name, value in originals.items(): getattr(points, name).from_numpy(value)
    oldbody.spread_elastic_force(dt_s=.001)
    assert force == fluid.force.to_numpy().tobytes()
    oldbody.move_with_fluid(.001)
    assert new == {name: getattr(points, name).to_numpy().tobytes() for name in originals}
class PeriodicMacOracle:
    """Actual FV periodic projection oracle around production IB, not CartesianFluidSolver acceptance.

    Existing production IB has no periodic indexing. A two-cell halo stores the periodic velocity copies; force
    spread into the halo is folded back to the physical MAC domain before its real discrete pressure projection.
    The uniform filled state has exactly one reference-area point per physical cell and the known total density.
    """
    def __init__(self, n=4, halo=2):
        self.n, self.halo = n, halo
        self.grid_n = n + 2 * halo
        self.indices = (np.arange(self.grid_n) - halo) % n
        freq = 2. * np.pi * np.fft.fftfreq(n)
        self.laplacian = -4. * (np.sin(freq[:, None] / 2.) ** 2 + np.sin(freq[None, :] / 2.) ** 2)
        self.laplacian[0, 0] = 1.
        self.u = np.zeros((n, n, 2), dtype=np.float64)

    def fill_velocity(self, fluid):
        ghost = self.u[self.indices[:, None], self.indices[None, :]]
        field = np.zeros((1, self.grid_n, self.grid_n, 3), np.float32)
        field[0, :, :, 1:] = ghost
        fluid.velocity.from_numpy(field)

    def advance(self, fluid, dt, forcing):
        field = fluid.force.to_numpy()[0, :, :, 1:].astype(np.float64)
        force = np.zeros_like(self.u)
        for j in range(self.grid_n):
            for k in range(self.grid_n):
                force[self.indices[j], self.indices[k]] += field[j, k]
        force[:, :, 0] += forcing
        self.u += dt * force / fluid.rho
        div = np.roll(self.u[:, :, 0], -1, axis=0) - self.u[:, :, 0]
        div += np.roll(self.u[:, :, 1], -1, axis=1) - self.u[:, :, 1]
        rhs = np.fft.fft2(div * fluid.rho / dt)
        pressure_hat = rhs / self.laplacian
        pressure_hat[0, 0] = 0.
        pressure = np.fft.ifft2(pressure_hat).real
        self.u[:, :, 0] -= dt / fluid.rho * (pressure - np.roll(pressure, 1, axis=0))
        self.u[:, :, 1] -= dt / fluid.rho * (pressure - np.roll(pressure, 1, axis=1))
        self.fill_velocity(fluid)


def filled_periodic_response(rho_s, omega, dt, end=.5, integrator="euler"):
    oracle = PeriodicMacOracle()
    positions = np.array([[oracle.halo + j + .5, oracle.halo + k + .5]
                          for j in range(oracle.n) for k in range(oracle.n)], np.float32)
    body, fluid, points = make_body(2, rho_s=rho_s, positions=positions, volumes=np.ones(len(positions)),
                                    tether=np.zeros(len(positions), np.int32), omega=omega,
                                    initial_v=(0., 0., 0.), grid=(1, oracle.grid_n, oracle.grid_n))
    velocity_old = (ti.Vector.field(3, ti.f32, shape=(1, oracle.grid_n, oracle.grid_n)) if integrator == "midpoint"
                    else None)
    frequency = 2. * np.pi
    times, body_v, fluid_v, partner_v, slip, beta, max_j_error = [], [], [], [], [], [], 0.
    for step in range(round(end / dt)):
        t = (step + 1) * dt
        if velocity_old is not None:
            velocity_old.copy_from(fluid.velocity)
            body.midpoint_predict(dt)
        body.spread_elastic_force(dt_s=dt)
        oracle.advance(fluid, dt, np.sin(frequency * (t - .5 * dt)))
        if velocity_old is not None:
            body.midpoint_correct(velocity_old, dt)
        else:
            body.move_with_fluid(dt)
        points.validate_state()
        v = points.v.to_numpy().astype(np.float64)
        j_error = float(np.max(np.abs(np.linalg.det(points.F.to_numpy().astype(np.float64)) - 1.)))
        max_j_error = max(max_j_error, j_error)
        times.append(t)
        body_v.append(float(v[:, 0].mean()))
        fluid_v.append(float(oracle.u[:, :, 0].mean()))    # u^{n+1}: the midpoint's points.v is (u^n + u^{n+1}) / 2
        if body.excess_mass is None:
            partner_v.append(None)
            slip.append(0.)
            beta.append(0.)
        else:
            ex = body.excess_mass
            partner_v.append(float(ex.velocity.to_numpy()[:, 0].mean()))
            slip.append(float(np.max(np.linalg.norm(ex.displacement.to_numpy() -
                ex.material_displacement.to_numpy(), axis=1))))
            beta.append(ex.last_beta)
    t = np.array(times)
    ideal = (1. - np.cos(frequency * t)) / (rho_s * frequency)
    exact = ideal.copy()
    exact_partner = ideal.copy()
    if rho_s > 1.:
        fast = omega * math.sqrt(rho_s)
        correction = frequency / (fast ** 2 - frequency ** 2) * (np.cos(frequency * t) - np.cos(fast * t))
        exact += (rho_s - 1.) / rho_s * correction
        exact_partner -= correction / rho_s
    body_v = np.array(body_v)
    fluid_v = np.array(fluid_v)
    return dict(rho_s=rho_s, rho_f=1., omega_rad_s=omega, dt_s=dt, end_time_s=end, points=len(positions),
        integrator=integrator, finite_K_fluid_error_l2=float(np.linalg.norm(fluid_v - exact) / np.linalg.norm(exact)),
        backend="production planar Taichi IB kernels + periodic FV/FFT projection oracle; not production CartesianFluidSolver",
        max_J_error=max_j_error, max_position_slip_m=max(slip), max_beta=max(beta),
        actual_body_v=body_v.tolist(), actual_partner_v=partner_v, time_s=times,
        analytic_finite_K_body_v=exact.tolist(), analytic_finite_K_partner_v=exact_partner.tolist(),
        ideal_density_response=ideal.tolist(),
        finite_K_body_error_l2=float(np.linalg.norm(body_v - exact) / np.linalg.norm(exact)),
        finite_K_partner_error_l2=(None if rho_s == 1. else float(np.linalg.norm(np.array(partner_v) - exact_partner) /
                                                               np.linalg.norm(exact_partner))),
        ideal_density_error_l2=float(np.linalg.norm(body_v - ideal) / np.linalg.norm(ideal)))


def test_filled_periodic_actual_inertia_and_fixed_K_convergence(tmp_path):
    runs = [filled_periodic_response(1., 40., .001), filled_periodic_response(10., 40., .001),
            filled_periodic_response(10., 80., .001), filled_periodic_response(10., 80., .0005)]
    receipt = dict(runs=runs, acceptance=dict(body_finite_K_relative_l2_max=.01,
        J_error_max=2e-5, density10_vs1_response_ratio_tolerance=.01,
        frequency_doubling_ideal_error_ratio_max=.4, dt_halving_finite_K_error_ratio_max=.8),
        scope="reference density gate under fully filled periodic low-frequency forcing; no official FSI acceptance")
    path = tmp_path / "PERIODIC_MAC_RECEIPT.json"
    path.write_text(json.dumps(receipt, indent=1, allow_nan=False))
    print("PERIODIC_MAC_RECEIPT", path)
    for run in runs:
        assert run["finite_K_body_error_l2"] < .01, (run["omega_rad_s"], run["dt_s"], run["finite_K_body_error_l2"])
        if run["finite_K_partner_error_l2"] is not None:
            assert run["finite_K_partner_error_l2"] < .01
        assert run["max_J_error"] < 2e-5
        assert run["dt_s"] * run["omega_rad_s"] * math.sqrt(10.) <= .5
    neutral, coarse, stiff, refined = runs
    # At half a forcing cycle the ideal density10 fluid/body velocity is 1/10 of density1; account for its finite K.
    measured_ratio = neutral["actual_body_v"][-1] / refined["actual_body_v"][-1]
    assert abs(measured_ratio / 10. - 1.) < .01
    assert stiff["ideal_density_error_l2"] < .4 * coarse["ideal_density_error_l2"]
    assert refined["max_position_slip_m"] < .4 * coarse["max_position_slip_m"]
    assert refined["finite_K_body_error_l2"] < .8 * stiff["finite_K_body_error_l2"]

@ti.kernel
def write_real_initial_gather(body: ti.template(), target: ti.template(), dim: ti.template()):
    for p in range(body.n):
        if ti.static(dim == 3):
            v, G = body._gather(body.fluid.velocity, body.fluid.velocity, 0., p, body.points.x[p])
            target[p] = ti.cast(v, ti.f64)
        else:
            v = body._gather_face_vector(body.fluid.velocity, body.points.x[p][0], body.points.x[p][1])
            target[p] = ti.cast(v, ti.f64)


@pytest.mark.parametrize("dim", (2, 3))
def test_initial_partner_velocity_bitwise_matches_real_gather_and_ignores_scratch(dim):
    body, _, points = make_body(dim)
    expected = ti.Vector.field(dim, ti.f64, shape=points.n)
    write_real_initial_gather(body, expected, dim)
    np.testing.assert_array_equal(body.excess_mass.velocity.to_numpy(), expected.to_numpy())
    points.v.fill(float("nan"))
    body.excess_mass._initialise_kernel()
    np.testing.assert_array_equal(body.excess_mass.velocity.to_numpy(), expected.to_numpy())


def test_density_override_must_match_actual_fluid_momentum_density():
    from fastfsi.coupling.excess_mass import validate_densities
    with pytest.raises(ValueError, match="momentum density"):
        validate_densities(SimpleNamespace(rho_s=10.), SimpleNamespace(rho=1.), 2.)


def test_bad_planar_material_displacement_and_position_rounding_are_both_rejected():
    body, _, points = make_body(2)
    ex = body.excess_mass
    good = ex.save_state()
    ex.material_displacement.fill(float("nan"))
    with pytest.raises(SimulationHardStop): ex.validate_state()
    with pytest.raises(SimulationHardStop): ex.state_array_names()
    ex.restore_state(good)
    altered = points.x.to_numpy()
    altered[0, 0] += .25
    points.x.from_numpy(altered)
    with pytest.raises(SimulationHardStop) as caught: ex.validate_state()
    witness = caught.value.witness
    assert witness["position_m"][0] == pytest.approx(witness["expected_position_m"][0] + .25)
    json.dumps(witness, allow_nan=False)
    ex.restore_state(good)
    ex.validate_state()


@pytest.mark.parametrize("dim", (2, 3))
def test_failed_material_move_cannot_accept_partner_or_write_checkpoint(dim):
    body, fluid, points = make_body(dim)
    ex = body.excess_mass
    saved = ex.save_state()
    saved_f = points.F.to_numpy().copy()
    saved_u = None if dim == 2 else points.u.to_numpy().copy()
    dt = .001
    if dim == 2: body.spread_elastic_force(dt_s=dt)
    else: body.spread_force(dt_s=dt)
    fluid.velocity.fill(float("nan"))
    with pytest.raises(SimulationHardStop): body.move_with_fluid(dt)
    assert ex.last_accepted_dt is None
    assert ex._pending_dt == dt
    with pytest.raises(ValueError, match="accepted"): ex.state_array_names()
    points.F.from_numpy(saved_f)
    if dim == 3:
        points.u.from_numpy(saved_u)
        points._after_load_state()
    ex.restore_state(saved)
    assert ex.state_host() == saved["host"]
    for name, array in saved["arrays"].items():
        np.testing.assert_array_equal(ex.state_array(name), array)


@pytest.mark.parametrize("dim", (2, 3))
def test_large_offset_microincrements_survive_split_restore_with_nan_scratch(dim):
    from fastfsi.io.checkpoint import same_host_value
    origin = (1000., 1000., 1000.)
    body, fluid, points = make_body(dim, positions=[[1003.5] * dim], origin=origin,
                                    initial_v=(1e-4, 1e-4, 1e-4))
    ex = body.excess_mass
    dt = 1e-5
    def advance():
        if dim == 2: body.spread_elastic_force(dt_s=dt)
        else: body.spread_force(dt_s=dt)
        body.move_with_fluid(dt)
    for _ in range(10): advance()
    saved = ex.save_state()
    point_arrays = {name: getattr(points, name).to_numpy().copy()
                    for name in (("F",) if dim == 2 else ("u", "F", "f_extra"))}
    for _ in range(10): advance()
    expected = ex.save_state()
    final_points = {name: getattr(points, name).to_numpy().tobytes() for name in ("x", "v", "F")}
    actual_u = ex.material_displacement.to_numpy() if dim == 2 else points.u.to_numpy().astype(np.float64)
    expected_u = 20. * float(np.float32(dt)) * points.v.to_numpy().astype(np.float64)
    np.testing.assert_allclose(actual_u, expected_u, atol=1e-14, rtol=1e-6)
    assert np.linalg.norm(actual_u) > 1e-8
    # The f32 spatial mirror is correctly unchanged at this absolute offset; the canonical motion is retained.
    np.testing.assert_array_equal(points.x.to_numpy(), points.X0.to_numpy())
    for name in ("spring_force", "face_mass_density", "bound_beta", "reduce", "displacement_n", "velocity_n"):
        getattr(ex, name).fill(float("nan"))
    for name in ("bound_empty", "status"):
        getattr(ex, name).fill(0x5A5A5A5A)
    points.v.fill(float("nan"))
    for name, array in point_arrays.items(): getattr(points, name).from_numpy(array)
    if dim == 3: points._after_load_state()
    ex.restore_state(saved)
    for _ in range(10): advance()
    result = ex.save_state()
    assert same_host_value(result["host"], expected["host"])
    for name, array in expected["arrays"].items():
        assert ex.state_array(name).tobytes() == array.tobytes()
    assert final_points == {name: getattr(points, name).to_numpy().tobytes() for name in final_points}


@ti.data_oriented
class SignedWeights:
    """A signed four-node corrected stencil, testing a bound beyond the accepted r<=1.2 nonnegative grids."""
    n = (8, 8, 8)
    min_spacing_m = 1.
    @ti.func
    def axis_weights(self, a: ti.template(), staggered: ti.template(), coordinate):
        start = ti.cast(ti.floor(coordinate - (.5 if staggered == 0 else 0.)), ti.i32) - 1
        return start, ti.Vector([-.125, .625, .625, -.125]), ti.Vector([0., 0., 0., 0.])
    @ti.func
    def face_volume(self, c, i, j, k):
        return 1.


@ti.data_oriented
class ChangingMask:
    def __init__(self):
        self.bits = ti.Vector.field(3, ti.i32, shape=(8, 8, 8))
        self.bits.fill(1)
    @ti.func
    def face_valid(self, c: ti.template(), i, j, k):
        return self.bits[i, j, k][c]


def test_signed_dense_markers_and_changed_one_sided_mask_use_full_overlap_bound():
    positions = np.array([[3.5, 3.5, 3.5], [3.5, 3.5, 3.5], [4.5, 3.5, 3.5], [4.5, 3.5, 3.5]], np.float32)
    points = MaterialPoints(positions_m=positions, volumes_m3=np.ones(4), material=NeoHookeanMaterial(10., 1., 10.),
                            tether_mask=np.zeros(4, np.int32))
    fluid = SimpleNamespace(rho=1., velocity=ti.Vector.field(3, ti.f32, shape=(8,8,8)),
                            force=ti.Vector.field(3, ti.f32, shape=(8,8,8)))
    mask = ChangingMask()
    body = ImmersedBoundaryCoupler(fluid=fluid, points=points, weights=SignedWeights(), mask=mask,
        tether_stiffness_pa_per_m=0., interior_shortcut=False, excess_mass_omega_rad_s=20.)
    raw = np.array([-.125, .625, .625, -.125])
    before = None
    for cut in (False, True):
        valid = np.ones((8,8,8,3), np.int32)
        if cut: valid[:,:,3,0] = 0  # remove a positive weight, making the global bound strictly tighter
        mask.bits.from_numpy(valid)
        body.excess_mass_dt_limit()
        observed = body.excess_mass.last_beta
        row_bounds, eigenvalues = [], []
        for c in range(3):
            w = np.zeros((4,8,8,8))
            for p, xp in enumerate(positions):
                starts = np.floor(xp - np.array([0. if a == c else .5 for a in range(3)])).astype(int)-1
                for a,b,e in np.ndindex(4,4,4):
                    i,j,k = starts + [a,b,e]
                    w[p,i,j,k] = raw[a]*raw[b]*raw[e]*valid[i,j,k,c]
                w[p] /= w[p].sum()
            matrix = w.reshape(4,-1)
            gram = 9.*matrix @ matrix.T
            absolute = 9.*np.abs(matrix) @ np.abs(matrix).T
            row_bounds.append(float(absolute.sum(axis=1).max()))
            eigenvalues.append(float(np.linalg.eigvalsh(gram).max()))
        assert observed >= max(eigenvalues)*(1.-1e-6)
        assert observed == pytest.approx(max(row_bounds),rel=1e-6)
        if before is not None: assert observed != before
        before = observed


def test_finite_unrepresentable_planar_restore_is_atomic_for_all_owned_state():
    body, _, points = make_body(2)
    ex = body.excess_mass
    saved = ex.save_state()
    bad = {name: array.copy() for name, array in saved["arrays"].items()}
    bad["material_displacement"][0,0] = 1e100
    body.spread_elastic_force(dt_s=.001)
    before = {name:getattr(ex,name).to_numpy().tobytes() for name in ex.CHECKPOINT_FIELDS}
    before_x, before_host, before_pending = points.x.to_numpy().tobytes(), ex.state_host(), ex._pending_dt
    with pytest.raises(ValueError,match="finite f32 position"):
        ex.load_state_arrays(bad)
    assert before == {name:getattr(ex,name).to_numpy().tobytes() for name in ex.CHECKPOINT_FIELDS}
    assert before_x == points.x.to_numpy().tobytes()
    assert before_host == ex.state_host() and before_pending == ex._pending_dt
    ex.restore_state(saved)
    ex.material_displacement.from_numpy(np.full((1,2),1e100))
    points.x.fill(float("inf"))
    with pytest.raises(SimulationHardStop):ex.validate_state()


@pytest.mark.parametrize("key,value", (("last_beta",float("nan")), ("last_beta",-1.),
    ("last_accepted_dt",float("inf")), ("last_accepted_dt",0.)))
def test_bad_carried_candidate_bound_host_cannot_change_owned_state(key,value):
    body,_,points=make_body(2)
    ex=body.excess_mass
    saved=ex.save_state()
    bad=dict(saved["host"]);bad[key]=value
    before={name:ex.state_array(name).tobytes() for name in ex.state_array_names()}
    with pytest.raises(ValueError):ex.restore_state(dict(arrays=saved["arrays"],host=bad))
    assert before=={name:ex.state_array(name).tobytes() for name in ex.state_array_names()}
    assert ex.state_host()==saved["host"]


def test_structured_spring_stop_is_registered_from_one_error_catalogue():
    from fastfsi.errors import HARD_STOP_CODES as registered
    from fastfsi.coupling.immersed_boundary import HARD_STOP_CODES as coupled
    assert "excess_mass_dt_limit" in registered
    assert coupled == set(registered)
    error = SimulationHardStop("excess_mass_dt_limit", dict(dt_s=.1,limit_s=.01))
    assert error.code == "excess_mass_dt_limit"
    json.dumps(error.witness, allow_nan=False)


@pytest.mark.parametrize("dim", (2,3))
def test_combined_reference_body_momentum_uses_distinct_material_and_partner_velocities(dim):
    body,_,points=make_body(dim)
    material_v=np.array([[.2,-.1,.05]],np.float32)[:,-dim:]
    partner_v=np.array([[.75,-.5,.25]],np.float64)[:,-dim:]
    points.v.from_numpy(material_v)
    body.excess_mass.velocity.from_numpy(partner_v)
    d=body.diagnostics()
    expected=(material_v.astype(np.float64)+9.*partner_v)[0]
    np.testing.assert_allclose(d["combined_reference_body_momentum"],expected,atol=1e-12,rtol=0.)
    nominal=np.array([d["momentum_y"],d["momentum_z"]]) if dim==2 else np.array(d["momentum"])
    np.testing.assert_array_equal(nominal,10.*material_v.astype(np.float64)[0])
    # prose about the quantities lives in the docstrings, not in returned data
    assert not [k for k in list(d)+list(d["excess_mass"])+list(body.density_report()) if "scope" in k]


@pytest.mark.parametrize("shortcut", (False, True))
def test_overlap_bound_uses_actual_spread_stencil_for_each_interior_mode_and_phase(shortcut):
    positions=np.array([[6.15,6.3,6.45],[6.8,6.6,6.7]],np.float32)
    points=MaterialPoints(positions_m=positions,volumes_m3=np.ones(2),material=NeoHookeanMaterial(10.,1.,10.),
                          tether_mask=np.zeros(2,np.int32))
    fluid=SimpleNamespace(rho=1.,velocity=ti.Vector.field(3,ti.f32,shape=(12,12,12)),
                          force=ti.Vector.field(3,ti.f32,shape=(12,12,12)))
    body=ImmersedBoundaryCoupler(fluid=fluid,points=points,
         weights=KernelWeights(UniformAxes((0.,0.,0.),(1.,1.,1.),(12,12,12))),mask=AllValidFaceMask(),
         tether_stiffness_pa_per_m=0.,interior_shortcut=shortcut,excess_mass_omega_rad_s=20.)
    body.refresh_support_flags()
    if shortcut:
        assert np.any(body.support_flag.to_numpy()==0)
    ex=body.excess_mass
    for phase in (0.,.125,.25):
        points.u.from_numpy(np.full((2,3),phase,np.float32))
        points._after_load_state()
        ex._initialise_kernel()
        saved=ex.save_state()
        # Extract the actual production force stencil using a unit point force and zero identity-F stress.
        # No reimplementation of kernel weights/normalisation enters this oracle.
        matrices=[np.zeros((2,12**3),np.float64) for _ in range(3)]
        for p in range(2):
            for c in range(3):
                force=np.zeros((2,3),np.float32);force[p,c]=1.
                points.f_extra.from_numpy(force)
                body.spread_force(dt_s=.0001)
                matrices[c][p]=fluid.force.to_numpy()[:,:,:,c].astype(np.float64).reshape(-1)
                ex.restore_state(saved)
        body.excess_mass_dt_limit()
        row_bound=max(float((9.*np.abs(w)@np.abs(w).T).sum(axis=1).max()) for w in matrices)
        eigenvalue=max(float(np.linalg.eigvalsh(9.*w@w.T).max()) for w in matrices)
        assert ex.last_beta == pytest.approx(row_bound,rel=2e-12,abs=1e-14)
        assert ex.last_beta >= eigenvalue*(1.-2e-12)


def test_planar_actual_unit_spring_stencil_matches_overlap_gram_across_phases():
    body,fluid,points=make_body(2,positions=[[3.15,3.3],[3.8,3.6]],volumes=[1.,1.],tether=[0,0],
                               initial_v=(0.,0.,0.))
    ex=body.excess_mass
    for phase in (0.,.125,.25):
        material=np.full((2,2),phase,np.float64)
        arrays=ex.save_state()["arrays"]
        arrays["material_displacement"]=material
        arrays["displacement"]=material.copy()
        ex.load_state_arrays(arrays)
        saved=ex.save_state()
        matrices=[np.zeros((2,8**2),np.float64) for _ in range(2)]
        for p in range(2):
            for c in range(2):
                displacement=material.copy();displacement[p,c]+=1./(9.*20.**2)
                ex.displacement.from_numpy(displacement)
                body.spread_elastic_force(dt_s=.0001)
                assert ex.spring_force.to_numpy()[p,c]==1.
                matrices[c][p]=fluid.force.to_numpy()[0,:,:,c+1].astype(np.float64).reshape(-1)
                ex.restore_state(saved)
        body.excess_mass_dt_limit()
        row_bound=max(float((9*np.abs(w)@np.abs(w).T).sum(1).max()) for w in matrices)
        eigenvalue=max(float(np.linalg.eigvalsh(9*w@w.T).max()) for w in matrices)
        assert ex.last_beta==pytest.approx(row_bound,rel=2e-12,abs=1e-14)
        assert ex.last_beta>=eigenvalue*(1.-2e-12)


# ---------------------------------------------------------------------------------------- midpoint partner and step checks
def spread(body, dt):
    if isinstance(body, PlanarImmersedBoundaryCoupler): body.spread_elastic_force(dt_s=dt)
    else: body.spread_force(dt_s=dt)


def test_midpoint_partner_is_second_order_on_the_filled_periodic_oracle(tmp_path):
    # The forced response is dominated by the slow mode and the midpoint-in-time forcing, which both integrators treat
    # to second order (explicit Euler measured 0.249 here too); the free spring test below separates the two.
    dts = (.001, .0005, .00025)
    midpoint = [filled_periodic_response(10., 80., dt, integrator="midpoint") for dt in dts]
    receipt = dict(dt_s=dts, midpoint_fluid=[r["finite_K_fluid_error_l2"] for r in midpoint],
                   midpoint_partner=[r["finite_K_partner_error_l2"] for r in midpoint])
    (tmp_path / "MIDPOINT_ORDER_RECEIPT.json").write_text(json.dumps(receipt, indent=1))
    print("MIDPOINT_ORDER_RECEIPT", json.dumps(receipt))
    for errors in (receipt["midpoint_fluid"], receipt["midpoint_partner"]):
        assert errors[1] / errors[0] < .3 and errors[2] / errors[1] < .3     # second order: 1/4 per halving
    assert all(r["max_J_error"] < 2e-5 for r in midpoint)


def free_spring_oscillation_error(integrator, dt, end=.1):
    """Filled periodic body (rho_s / rho_f = 10) at rest, springs stretched by s0, no forcing: the relative mode
    s(t) = s0 cos(omega_r t), omega_r = omega0 sqrt(10).  Relative L2 error of the slip Y - X at every step."""
    n, halo, omega = 4, 2, 80.
    grid_n = n + 2 * halo
    positions = np.array([[halo + j + .5, halo + k + .5] for j in range(n) for k in range(n)], np.float32)
    body, fluid, points = make_body(2, rho_s=10., positions=positions, volumes=np.ones(len(positions)),
                                    tether=np.zeros(len(positions), np.int32), omega=omega, initial_v=(0., 0., 0.),
                                    grid=(1, grid_n, grid_n))
    ex = body.excess_mass
    s0 = np.array([.002, -.001])
    ex.displacement.from_numpy(np.tile(s0, (len(positions), 1)))
    mode = UniformFluidMode(fluid, 1. * n * n)
    velocity_old = ti.Vector.field(3, ti.f32, shape=(1, grid_n, grid_n))
    omega_r = omega * math.sqrt(10.)
    error, norm = 0., 0.
    for step in range(round(end / dt)):
        if integrator == "midpoint":
            velocity_old.copy_from(fluid.velocity)
            body.midpoint_predict(dt)
        body.spread_elastic_force(dt_s=dt)
        mode.advance(dt)
        if integrator == "midpoint":
            body.midpoint_correct(velocity_old, dt)
        else:
            body.move_with_fluid(dt)
        slip = (ex.displacement.to_numpy() - ex.material_displacement.to_numpy()).mean(axis=0)
        exact = s0 * math.cos(omega_r * (step + 1) * dt)
        error += float((slip - exact) @ (slip - exact))
        norm += float(exact @ exact)
    return math.sqrt(error / norm)


def test_free_spring_oscillation_is_second_order_with_midpoint_and_first_order_with_euler():
    dts = (.001, .0005, .00025)
    midpoint = [free_spring_oscillation_error("midpoint", dt) for dt in dts]
    euler = [free_spring_oscillation_error("euler", dt) for dt in dts]
    print("FREE_SPRING_ORDER", dict(dt_s=dts, midpoint=midpoint, euler=euler))
    assert midpoint[1] / midpoint[0] < .3 and midpoint[2] / midpoint[1] < .3      # Stormer-Verlet: 1/4
    assert euler[1] / euler[0] > .4 and euler[2] / euler[1] > .4                  # symplectic Euler: 1/2
    assert midpoint[2] < .25 * euler[2]


@ti.data_oriented
class UniformFluidMode:
    """Fluid of a completely filled periodic channel in its uniform mode (test only): the summed spread force moves one
    f64 velocity, copied to every face; the periodic projection leaves a uniform field unchanged."""
    def __init__(self, fluid, mass_per_span):
        self.fluid, self.mass = fluid, float(mass_per_span)
        self.u = ti.field(ti.f64, shape=2)

    @ti.kernel
    def advance(self, step_s: ti.f64):
        total = ti.Vector([0., 0.], dt=ti.f64)
        ti.loop_config(serialize=True)
        for j, k in ti.ndrange(self.fluid.force.shape[1], self.fluid.force.shape[2]):
            total += ti.Vector([ti.cast(self.fluid.force[0, j, k][1], ti.f64), ti.cast(self.fluid.force[0, j, k][2], ti.f64)])
        for q in ti.static(range(2)):
            self.u[q] += step_s * total[q] / self.mass
        for i, j, k in self.fluid.velocity:
            self.fluid.velocity[i, j, k] = ti.Vector([0., ti.cast(self.u[0], ti.f32), ti.cast(self.u[1], ti.f32)])


def test_midpoint_energy_of_fluid_partners_and_springs_stays_bounded_over_1e5_steps():
    """omega0 dt = 0.1; a filled periodic body (rho_s / rho_f = 10, 16 points) without forcing, springs stretched at
    t = 0.  Fluid, partners and springs exchange energy in the relative mode omega_r = omega0 sqrt(10) (omega_r dt =
    0.32).  Stormer-Verlet conserves E~ = mu (1 - (omega_r dt / 2)^2) v_r^2 / 2 + K s^2 / 2 exactly, so the true energy
    stays in [E0, E0 / (1 - (omega_r dt / 2)^2)] with no secular trend.  Contrast: RK2 (Heun) on the same mode multiplies
    the energy by 1 + (omega_r dt)^4 / 4 every step."""
    n, halo, omega = 4, 2, 80.
    dt = .1 / omega
    grid_n = n + 2 * halo
    positions = np.array([[halo + j + .5, halo + k + .5] for j in range(n) for k in range(n)], np.float32)
    body, fluid, points = make_body(2, rho_s=10., positions=positions, volumes=np.ones(len(positions)),
                                    tether=np.zeros(len(positions), np.int32), omega=omega, initial_v=(0., 0., 0.),
                                    grid=(1, grid_n, grid_n))
    ex = body.excess_mass
    ex.displacement.from_numpy(np.tile([[.002, -.001]], (len(positions), 1)))
    fluid_mass, partner_mass = 1. * n * n, 9. * n * n
    mu = fluid_mass * partner_mass / (fluid_mass + partner_mass)
    omega_r = omega * math.sqrt(1. + partner_mass / fluid_mass)
    band = (omega_r * dt / 2.) ** 2
    mode = UniformFluidMode(fluid, fluid_mass)
    velocity_old = ti.Vector.field(3, ti.f32, shape=(1, grid_n, grid_n))

    def energies():
        d = ex.diagnostics()
        u = mode.u.to_numpy()
        v_rel = ex.velocity.to_numpy().mean(axis=0) - u
        total = .5 * fluid_mass * float(u @ u) + d["kinetic_energy"] + d["spring_energy"]
        modified = .5 * mu * (1. - band) * float(v_rel @ v_rel) + d["spring_energy"]
        return total, modified
    e0, m0 = energies()
    total, modified = [], []
    for step in range(100_000):
        velocity_old.copy_from(fluid.velocity)
        body.midpoint_predict(dt)
        body.spread_elastic_force(dt_s=dt)
        mode.advance(dt)
        body.midpoint_correct(velocity_old, dt)
        if (step + 1) % 500 == 0:
            e, m = energies()
            total.append(e / e0)
            modified.append(m / m0)
    total, modified = np.array(total), np.array(modified)
    print("ENERGY_1E5", dict(total_min=float(total.min()), total_max=float(total.max()), band=band,
                             modified_max_dev=float(np.abs(modified - 1.).max())))
    assert total.min() > 1. - 1e-4 and total.max() < 1. / (1. - band) + 1e-4
    assert np.abs(modified - 1.).max() < 1e-3                 # no secular growth or decay over 1e5 steps
    assert total.max() - 1. > .5 * band                       # the bounded oscillation is really sampled
    # contrast: RK2 on the same relative mode (q'' = -omega_r^2 q), the same check sees the growth
    q, v = 1., 0.
    for _ in range(10_000):
        k1q, k1v = v, -omega_r ** 2 * q
        k2q, k2v = v + dt * k1v, -omega_r ** 2 * (q + dt * k1q)
        q, v = q + .5 * dt * (k1q + k2q), v + .5 * dt * (k1v + k2v)
    assert (v * v + omega_r ** 2 * q * q) / omega_r ** 2 > 1e3


def _point_state(points, ex, dim):
    names = ("x", "F", "v") if dim == 2 else ("x", "u", "F", "v")
    state = {name: getattr(points, name).to_numpy().tobytes() for name in names}
    state.update({name: getattr(ex, name).to_numpy().tobytes() for name in ex.CHECKPOINT_FIELDS})
    state["host"] = (ex.last_beta, ex.last_accepted_dt)
    return state


@pytest.mark.parametrize("dim", (2, 3))
def test_midpoint_split_restart_at_accepted_steps_is_bitwise(dim, tmp_path):
    positions = np.array([[3.5, 3.5, 3.5], [4.2, 3.7, 3.4]])[:, -dim:]
    body, fluid, points = make_body(dim, positions=positions, volumes=[1., 1.], tether=[0, 0])
    grid = tuple(fluid.velocity.shape)
    shear = np.zeros(grid + (3,), np.float32)
    shear[..., -dim] = .002 * (np.arange(grid[-1], dtype=np.float32) - 4.)        # a shear: F evolves
    shear[..., -1] = .01
    fluid.velocity.from_numpy(shear)
    ex = body.excess_mass
    ex._initialise_kernel()
    offset = np.array([[.003, -.002, .001], [-.001, .002, .0005]])[:, -dim:]
    ex.displacement.from_numpy(ex.displacement.to_numpy() + offset)
    dt = .2 * body.excess_mass_dt_limit()
    velocity_old = ti.Vector.field(3, ti.f32, shape=grid)

    def step():
        velocity_old.copy_from(fluid.velocity)
        body.midpoint_predict(dt)
        with pytest.raises(ValueError, match="accepted"): ex.state_array_names()
        spread(body, dt)
        with pytest.raises(ValueError, match="accepted"): ex.state_array_names()
        body.midpoint_correct(velocity_old, dt)
    for _ in range(3): step()
    if dim == 3:
        components = dict(points=points, body=body, excess_mass=ex)
        path = tmp_path / "midpoint.npz"
        write_checkpoint(path, components, {"step": 3}, {"fingerprint": body.fingerprint()})
    else:
        saved, saved_f = ex.save_state(), points.F.to_numpy().copy()
    for _ in range(4): step()
    expected = _point_state(points, ex, dim)
    for name in ("spring_force", "face_mass_density", "bound_beta", "displacement_n", "velocity_n", "reduce"):
        getattr(ex, name).fill(float("nan"))
    points.v.fill(float("nan"))
    if dim == 3:
        arrays, host, loop, manifest, _ = read_checkpoint(path)
        load_components(components, arrays, host)
    else:
        points.F.from_numpy(saved_f)
        ex.restore_state(saved)
    for _ in range(4): step()
    assert _point_state(points, ex, dim) == expected


@pytest.mark.parametrize("dim", (2, 3))
def test_midpoint_stops_restore_the_partner_from_its_exact_copy(dim):
    body, fluid, points = make_body(dim, initial_v=(.02, -.01, .005))
    ex = body.excess_mass
    ex.displacement.from_numpy(ex.displacement.to_numpy() + np.array([[.003, -.002, .001]])[:, -dim:])
    accepted = (ex.displacement.to_numpy().tobytes(), ex.velocity.to_numpy().tobytes())
    material = (ex.material_displacement if dim == 2 else points.u).to_numpy().tobytes()
    limit = body.excess_mass_dt_limit()
    velocity_old = ti.Vector.field(3, ti.f32, shape=tuple(fluid.velocity.shape))

    def partner():
        return ex.displacement.to_numpy().tobytes(), ex.velocity.to_numpy().tobytes()

    def back_to_step_start():
        ex.abort_step()                                  # planar: also the material displacement and points.x
        if dim == 3:
            points.u.copy_from(points.u_n)
            points._after_load_state()
        points.F.copy_from(points.F_n)
        body.invalidate_overlap_bound()
        assert partner() == accepted
        assert (ex.material_displacement if dim == 2 else points.u).to_numpy().tobytes() == material
        ex.state_array_names()                           # an accepted boundary again

    # 1. the bound at the half-step positions is known before the spread: the stop is raised there
    velocity_old.copy_from(fluid.velocity)
    body.midpoint_predict(3. * limit)
    body.excess_mass_dt_limit()
    with pytest.raises(SimulationHardStop, match="excess_mass_dt_limit"): spread(body, 3. * limit)
    assert partner() == accepted
    with pytest.raises(ValueError, match="accepted"): ex.state_array_names()
    with pytest.raises(ValueError, match="in progress"): body.midpoint_predict(limit)
    back_to_step_start()
    # 2. not known: evaluated with the spring force, checked at the step's end
    velocity_old.copy_from(fluid.velocity)
    body.midpoint_predict(3. * limit)
    spread(body, 3. * limit)
    with pytest.raises(SimulationHardStop, match="excess_mass_dt_limit") as caught:
        body.midpoint_correct(velocity_old, 3. * limit)
    assert {"dt_s", "limit_s", "beta", "omega_rad_s"} <= set(caught.value.witness)
    assert partner() == accepted
    back_to_step_start()
    # 3. a non-finite material state at the step's end
    velocity_old.copy_from(fluid.velocity)
    body.midpoint_predict(.5 * limit)
    spread(body, .5 * limit)
    velocity_old.fill(float("nan"))
    with pytest.raises(SimulationHardStop) as caught: body.midpoint_correct(velocity_old, .5 * limit)
    assert caught.value.code == "nonfinite_point_state"
    assert partner() == accepted
    assert ex.last_accepted_dt is None


@pytest.mark.parametrize("dim", (2, 3))
def test_healthy_explicit_steps_read_the_device_once_and_evaluate_the_bound_once(dim):
    body, fluid, points = make_body(dim, initial_v=(.02, -.01, .005))
    ex = body.excess_mass
    calls = {}

    def counted(name):
        original = getattr(ex, name)

        def wrapper(*args):
            calls[(name,) + args] = calls.get((name,) + args, 0) + 1
            return original(*args)
        setattr(ex, name, wrapper)
    for name in ("_dt_limit_kernel", "_prepare_kernel", "_accept_kernel", "_first_invalid_kernel"):
        counted(name)

    def detailed(*args, **kwargs):
        raise AssertionError("a detailed check ran in a healthy step")
    points.validate_state = detailed
    dt = .5 * body.excess_mass_dt_limit()               # the only explicit evaluation: one launch, one read
    for _ in range(5):
        body.excess_mass_dt_limit()                      # the bound where this step spreads: already known
        spread(body, dt)
        body.move_with_fluid(dt)
    assert calls == {("_dt_limit_kernel",): 1, ("_prepare_kernel", False): 5, ("_accept_kernel", True): 5}


@pytest.mark.parametrize("dim", (2, 3))
def test_cached_overlap_bound_is_bitwise_a_fresh_evaluation(dim):
    positions = np.array([[3.4, 3.6, 3.5], [3.9, 3.3, 3.7]])[:, -dim:]
    body, fluid, points = make_body(dim, positions=positions, volumes=[1., 1.], tether=[0, 0],
                                    initial_v=(.05, -.02, .01))
    ex = body.excess_mass
    dt = .5 * body.excess_mass_dt_limit()
    for _ in range(3):
        spread(body, dt)
        body.move_with_fluid(dt)
    key, cached = ex._bound_cache
    assert key == body.overlap_bound_key()
    body.invalidate_overlap_bound()
    body.excess_mass_dt_limit()
    assert ex.last_beta == cached


def test_face_mask_refresh_and_support_flags_invalidate_the_cached_bound():
    from fastfsi.coupling.face_mask import FaceValidityMask
    axes = UniformAxes((0., 0., 0.), (1., 1., 1.), (8, 8, 8))
    fluid = SimpleNamespace(rho=1., velocity=ti.Vector.field(3, ti.f32, shape=(8, 8, 8)),
                            force=ti.Vector.field(3, ti.f32, shape=(8, 8, 8)), obstacle=ti.field(ti.i32, shape=(8, 8, 8)))
    mask = FaceValidityMask(fluid, axes)

    def coupler():
        points = MaterialPoints(positions_m=[[3.6, 3.5, 3.5]], volumes_m3=[1.], material=NeoHookeanMaterial(10., 1., 10.),
                                tether_mask=[0])
        body = ImmersedBoundaryCoupler(fluid=fluid, weights=KernelWeights(axes), mask=mask, points=points,
                                       tether_stiffness_pa_per_m=0., excess_mass_omega_rad_s=20.)
        body.refresh_support_flags()
        return body
    body = coupler()
    ex = body.excess_mass
    launches = []
    original = ex._dt_limit_kernel

    def counted():
        launches.append(1)
        return original()
    ex._dt_limit_kernel = counted
    first = body.excess_mass_dt_limit()
    assert body.excess_mass_dt_limit() == first and len(launches) == 1
    obstacle = np.zeros((8, 8, 8), np.int32)
    obstacle[5] = 1                                       # a wall inside the point's support
    fluid.obstacle.from_numpy(obstacle)
    assert body.excess_mass_dt_limit() == first and len(launches) == 1    # the mask itself has not changed yet
    mask.refresh()
    body.refresh_support_flags()
    changed = body.excess_mass_dt_limit()
    assert len(launches) == 2 and changed != first
    assert coupler().excess_mass_dt_limit() == changed


@pytest.mark.parametrize("dim", (2, 3))
def test_failed_step_check_reports_the_spring_force_witness(dim):
    body, fluid, points = make_body(dim, initial_v=(0., 0., 0.))
    ex = body.excess_mass
    ex.displacement.from_numpy(np.full((1, dim), 1e36))   # K (Y - X) is beyond f32: the saved force is infinite
    spread(body, 1e-6)
    with pytest.raises(SimulationHardStop) as caught: body.move_with_fluid(1e-6)
    witness = caught.value.witness
    assert caught.value.code == "nonfinite_point_state" and witness["component"] == "excess_mass"
    assert "spring_force" in witness
    json.dumps(witness, allow_nan=False)


def test_failed_step_check_reports_an_empty_kernel_support():
    points = MaterialPoints(positions_m=[[3.5, 3.5, 3.5]], volumes_m3=[1.], material=NeoHookeanMaterial(10., 1., 10.),
                            tether_mask=[0])
    fluid = SimpleNamespace(rho=1., velocity=ti.Vector.field(3, ti.f32, shape=(8, 8, 8)),
                            force=ti.Vector.field(3, ti.f32, shape=(8, 8, 8)))
    mask = ChangingMask()
    body = ImmersedBoundaryCoupler(fluid=fluid, weights=KernelWeights(UniformAxes((0., 0., 0.), (1., 1., 1.), (8, 8, 8))),
                                   mask=mask, points=points, tether_stiffness_pa_per_m=0., interior_shortcut=False,
                                   excess_mass_omega_rad_s=20.)
    mask.bits.fill(0)                                      # no valid face anywhere
    body.spread_force(dt_s=1e-3)
    with pytest.raises(SimulationHardStop, match="one_sided_empty_support"): body.move_with_fluid(1e-3)
