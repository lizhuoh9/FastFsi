"""Solid-state guards: real CPU point fields and a cheap injected Turek time loop."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from fastfsi.errors import SimulationHardStop
from fastfsi.solid.material_points import MaterialPoints, PlanarMaterialPoints
from fastfsi.solid.materials import NeoHookeanMaterial


@pytest.fixture(scope="module", autouse=True)
def cpu():
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1, opt_level=0, offline_cache=False)


@pytest.fixture(scope="module", params=(2, 3), ids=("planar", "3d"))
def point_set(request):
    dim = request.param
    positions = np.arange(3 * dim, dtype=np.float32).reshape(3, dim) * 0.01
    material = NeoHookeanMaterial(1000.0, 1.0, 10.0)
    if dim == 2:
        points = PlanarMaterialPoints(positions_yz_m=positions, areas_m2=np.ones(3), material=material,
                                      tether_mask=np.array([1, 0, 0]))
    else:
        points = MaterialPoints(positions_m=positions, volumes_m3=np.ones(3), material=material,
                                tether_mask=np.array([1, 0, 0]))
    return points, dim


@pytest.fixture
def points(point_set):
    points, dim = point_set
    points.x.copy_from(points.X0)
    points.v.fill(0.0)
    points.F.from_numpy(np.tile(np.eye(dim, dtype=np.float32), (points.n, 1, 1)))
    if dim == 3:
        points.u.fill(0.0)
    return points, dim


def corrupt(points, dim, field, p=1):
    if field in ("x", "u", "v"):
        value = getattr(points, field).to_numpy()
        value[p, 0] = np.inf if field == "v" else np.nan
    else:
        value = points.F.to_numpy()
        if field == "F":
            value[p, 0, 1] = np.nan
        elif field == "J_overflow":
            value[p] = np.eye(dim, dtype=np.float32) * np.float32(1.0e30)
        elif field == "J_zero":
            value[p, 0, 0] = 0.0
        elif field == "J_negative":
            value[p, 0, 0] = -1.0
        else:
            raise AssertionError(field)
        field = "F"
    getattr(points, field).from_numpy(value)


@pytest.mark.parametrize("field", ("x", "v", "F", "J_overflow"))
def test_diagnostics_count_nonfinite_point_state(points, field):
    points, dim = points
    corrupt(points, dim, field)
    assert points.diagnostics()["nonfinite_points"] == 1


@pytest.mark.parametrize("field", ("J_zero", "J_negative"))
def test_diagnostics_count_nonpositive_jacobians(points, field):
    points, dim = points
    corrupt(points, dim, field)
    d = points.diagnostics()
    assert d["nonfinite_points"] == 0
    assert d["nonpositive_J_points"] == 1


@pytest.mark.parametrize("field", ("x", "v", "F", "J_overflow", "J_zero", "J_negative"))
def test_guard_raises_with_first_point_and_strict_json_witness(points, field):
    points, dim = points
    corrupt(points, dim, field)
    corrupt(points, dim, "v", p=2)
    before = {key: getattr(points, key).to_numpy().tobytes() for key in ("x", "v", "F")}
    with pytest.raises(SimulationHardStop) as caught:
        points.validate_state()
    error = caught.value
    expected = "invalid_deformation_gradient" if field in ("J_zero", "J_negative") else "nonfinite_point_state"
    assert error.code == expected
    assert error.witness["point"] == 1
    assert error.witness["invalid_fields"]
    assert len(error.witness["F"]) == dim
    json.dumps(error.witness, allow_nan=False)
    assert before == {key: getattr(points, key).to_numpy().tobytes() for key in before}


def test_guard_and_diagnostics_cover_saved_3d_displacement(points):
    points, dim = points
    if dim != 3:
        pytest.skip("planar state integrates x directly")
    corrupt(points, dim, "u")
    assert np.isfinite(points.x.to_numpy()).all()  # deliberately stale derived x
    assert points.diagnostics()["nonfinite_points"] == 1
    with pytest.raises(SimulationHardStop) as caught:
        points.validate_state()
    assert "u" in caught.value.witness["invalid_fields"]


def test_healthy_guard_downloads_no_point_arrays_and_preserves_state(points, monkeypatch):
    points, dim = points
    names = [name for name in ("x", "X0", "u", "u_n", "x_n", "v", "F", "F_n", "tau", "f_extra")
             if hasattr(points, name)]
    before = {name: getattr(points, name).to_numpy().tobytes() for name in names}
    fields = points.state_array_names() if dim == 3 else None
    with monkeypatch.context() as patch:
        def refuse_download():
            raise AssertionError("the healthy guard must return only a device scalar")
        for name in names:
            patch.setattr(getattr(points, name), "to_numpy", refuse_download)
        assert points.validate_state() is None
    assert before == {name: getattr(points, name).to_numpy().tobytes() for name in names}
    if dim == 3:
        assert fields == points.state_array_names() == ("u", "F", "f_extra")


def make_module(name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    return module


@pytest.fixture
def runner(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/turek_hron_fsi/run_turek_hron_fsi.py"
    spec = importlib.util.spec_from_file_location("turek_state_guard_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    fluid = SimpleNamespace(last_cg_iterations=1, velocity=ti.Vector.field(3, ti.f32, shape=(4, 2, 2)),
                            apply_body_force=lambda *args, **kwargs: None,
                            predict=lambda *args, **kwargs: None,
                            project=lambda *args, **kwargs: None,
                            mark_pressure_outlet_disconnected_nonprojectable_cells=lambda **kwargs: None)
    cfg = SimpleNamespace(young_modulus_pa=2.8, poisson_ratio=0.4, solid_density_kgm3=1.0,
                          fluid_density_kgm3=1.0, fluid_viscosity_pa_s=1.0, span_m=1.0,
                          inlet_ramp_time_s=2.0, mean_inlet_velocity_mps=1.0)
    channel = make_module("turek_hron_channel", preset=lambda *args, **kwargs: cfg,
                          build_fluid=lambda *args: fluid, full_bounds=lambda cfg: ((0., 0., 0.), (1., 1., 1.)),
                          cell_spacing_m=lambda cfg: (0.1, 0.1, 0.1), solver_z=lambda x, cfg: x,
                          write_channel_velocity_faces=lambda *args: None, update_inlet_face_values=lambda *args: None)
    import fastfsi.fluid.outlet_reachability as reachability_module
    import fastfsi.fluid.solver as solver_module
    import fastfsi.simulation.taichi_runtime as runtime_module
    monkeypatch.setattr(runtime_module, "TaichiRuntimeConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(runtime_module, "init_taichi", lambda config: None)
    monkeypatch.setattr(solver_module, "enforce_planar_velocity", lambda fluid: None)
    monkeypatch.setattr(reachability_module, "single_level_reachability_flood", contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, "turek_hron_channel", channel)
    position = np.array([[0., 0.], [0.1, 0.1], [0.2, 0.2]], dtype=np.float32)
    monkeypatch.setattr(runner, "beam_points", lambda *args: (
        position, np.ones(3), np.array([1, 0, 0]), 2, np.zeros(3, np.int32), {}))
    return runner, monkeypatch


@pytest.mark.parametrize("integrator,phase,field,fail_step,max_steps", (
    ("euler", "after_move", "v", 3, 0),
    ("euler", "after_move", "J_negative", 2, 0),
    ("euler", "after_move", "J_zero", 2, 2),
    ("midpoint", "midpoint_predict", "F", 2, 0),
    ("euler", "initial", "F", 0, 0),
    ("euler", "none", None, 0, 0),
    ("midpoint", "none", None, 0, 2),
))
def test_runner_guards_every_phase_and_terminal_boundary(
        runner, tmp_path, integrator, phase, field, fail_step, max_steps):
    runner, patch = runner
    events = []

    class Coupler:
        def __init__(self, *, points, **kwargs):
            self.points = points
            self.step = 0
            self.excess_mass = None
            if phase == "initial":
                corrupt(self.points, 2, field)

        def density_report(self):
            return dict(effective_reference_density_kgm3=1., effective_reference_density_ratio=1.)

        def excess_mass_dt_limit(self):
            return float("inf")

        def midpoint_predict(self, dt):
            if phase == "midpoint_predict" and self.step + 1 == fail_step:
                corrupt(self.points, 2, field)

        def spread_elastic_force(self, dt_s):
            events.append("spread")

        def move_with_fluid(self, dt):
            self.step += 1
            if phase == "after_move" and self.step == fail_step:
                corrupt(self.points, 2, field)

        def midpoint_correct(self, velocity_old, dt):
            self.move_with_fluid(dt)

        def diagnostics(self):
            return self.points.diagnostics()

    class CV:
        def __init__(self, *args, **kwargs):
            pass

        def measure(self):
            return dict(momentum_y=0., momentum_z=0., surface_y=0., surface_z=0.)

    patch.setitem(sys.modules, "fastfsi.coupling.immersed_boundary", make_module(
        "fastfsi.coupling.immersed_boundary", PlanarImmersedBoundaryCoupler=Coupler))
    patch.setitem(sys.modules, "fastfsi.simulation.monitors", make_module(
        "fastfsi.simulation.monitors", ControlVolumeForce=CV))
    out = tmp_path / "run"
    args = ["--arch", "cpu", "--preset", "fsi3", "--grid", "4,2,2", "--end-time", "0.003",
            "--row-interval", "10", "--dt-max", "0.001", "--time-integration", integrator,
            "--max-steps", str(max_steps), "--output", str(out)]
    if field is None:
        runner.main(args)
    else:
        with pytest.raises(SimulationHardStop):
            runner.main(args)
    final = json.loads((out / "final.json").read_text())
    assert len((out / "history.csv").read_text().splitlines()) == 1
    if field is None:
        assert final["status"] == "COMPLETED"
        assert "hard_stop" not in final
        assert final["steps"] == (max_steps or 3)
        assert final["t_s"] == pytest.approx(0.001 * final["steps"])
        assert len(events) == final["steps"]
    else:
        assert final["status"] != "COMPLETED"
        witness = final["hard_stop"]["witness"]
        assert witness["point"] == 1
        assert witness["phase"] == phase
        assert witness["step"] == fail_step
        assert final["steps"] == max(0, fail_step - 1)
        assert final["t_s"] == pytest.approx(0.001 * final["steps"])
        assert len(events) == max(0, fail_step - (phase == "midpoint_predict"))


@pytest.mark.parametrize("field", ("u", "F", "J_negative"))
def test_canonical_restore_guard_ignores_only_scratch_velocity(field):
    points = MaterialPoints(positions_m=np.array([[3.5, 3.5, 3.5], [4.5, 3.5, 3.5]], np.float32),
                            volumes_m3=np.ones(2), material=NeoHookeanMaterial(1000., 1., 10.),
                            tether_mask=np.zeros(2, np.int32))
    points.v.fill(float("nan"))
    corrupt(points, 3, field)
    velocity_before = points.v.to_numpy().tobytes()
    with pytest.raises(SimulationHardStop) as caught:
        points.validate_state(check_velocity=False)
    error = caught.value
    assert "v" not in error.witness["invalid_fields"]
    assert "velocity_mps" not in error.witness
    assert error.code == ("invalid_deformation_gradient" if field == "J_negative" else "nonfinite_point_state")
    assert points.v.to_numpy().tobytes() == velocity_before


def test_nan_scratch_restore_is_rewritten_by_real_3d_gather():
    from fastfsi.coupling.face_mask import AllValidFaceMask
    from fastfsi.coupling.immersed_boundary import ImmersedBoundaryCoupler
    from fastfsi.coupling.kernel_weights import KernelWeights
    from fastfsi.grid.axes import UniformAxes
    points = MaterialPoints(positions_m=np.array([[3.5, 3.5, 3.5], [4.5, 3.5, 3.5]], np.float32),
                            volumes_m3=np.ones(2), material=NeoHookeanMaterial(1000., 1., 10.),
                            tether_mask=np.zeros(2, np.int32))
    fields = points.state_array_names()
    saved = {name: points.state_array(name).copy() for name in fields}
    velocity = ti.Vector.field(3, ti.f32, shape=(8, 8, 8))
    expected_v = np.array([0.2, -0.1, 0.05], dtype=np.float32)
    velocity.from_numpy(np.broadcast_to(expected_v, (8, 8, 8, 3)).copy())
    ib = ImmersedBoundaryCoupler(fluid=SimpleNamespace(velocity=velocity, rho=1000.),
        weights=KernelWeights(UniformAxes((0., 0., 0.), (1., 1., 1.), (8, 8, 8))),
        mask=AllValidFaceMask(), points=points, tether_stiffness_pa_per_m=0., interior_shortcut=False)
    for method in (ib.move_with_fluid, ib.midpoint_predict):
        points.load_state_arrays(saved)
        points.v.fill(0.)
        method(0.001)
        clean = {name: getattr(points, name).to_numpy().tobytes() for name in ("x", "u", "v", "F", "grad_v")}
        points.v.fill(float("nan"))
        points.load_state_arrays(saved)  # the real restore refreshes x, and correctly leaves v as scratch
        poisoned_v = points.v.to_numpy().tobytes()
        points.validate_state(check_velocity=False)
        assert points.v.to_numpy().tobytes() == poisoned_v
        with pytest.raises(SimulationHardStop) as caught:
            points.validate_state()
        assert "v" in caught.value.witness["invalid_fields"]
        method(0.001)  # actual production gather and F/x update; no stub and no sanitisation
        points.validate_state()
        assert np.allclose(points.v.to_numpy(), expected_v, atol=2e-7, rtol=0.)
        assert clean == {name: getattr(points, name).to_numpy().tobytes() for name in clean}
        assert points.state_array_names() == fields == ("u", "F", "f_extra")
