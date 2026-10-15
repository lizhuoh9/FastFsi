"""Known momentum balances through the real example loop, without a costly fluid solver."""
import csv
import json
import sys

import numpy as np
import pytest

from test_material_point_state import cpu, make_module, runner  # shared tiny real-point/mock-fluid fixture


DT = 1.0 / 1024.0  # exact in binary: the equal-density identity check makes this limited assumption explicit


def run_balance(runner_fixture, tmp_path, *, rho_s=10., rho_f=1., acceleration=(0., 0.),
                box_rate=(0., 0.), surface=(2., 3.), clipped=False):
    runner, patch = runner_fixture
    cfg = sys.modules["turek_hron_channel"].preset("fsi2")
    cfg.solid_density_kgm3, cfg.fluid_density_kgm3 = rho_s, rho_f
    time_state = [0.]
    advances, raw_samples, beam_samples = [], [], []
    acceleration = np.array(acceleration)
    box_rate, surface = np.array(box_rate), np.array(surface)

    class Coupler:
        def __init__(self, *, points, **kwargs):
            self.points = points
            self.excess_mass = None  # this CV test explicitly selects diagnostic neutral inertia

        def density_report(self):
            return dict(effective_reference_density_kgm3=rho_f, effective_reference_density_ratio=1.)

        def excess_mass_dt_limit(self):
            return float("inf")

        def spread_elastic_force(self, dt_s):
            pass

        def move_with_fluid(self, dt):
            advances.append(dt)
            time_state[0] += dt
            v = np.zeros((3, 2), np.float32)
            v[1:] = np.array([0.25, 0.5]) + acceleration * time_state[0]
            self.points.v.from_numpy(v)  # two unit-area free points; clamp's velocity remains zero

        def diagnostics(self):
            d = self.points.diagnostics()
            beam_samples.append(np.array([d["momentum_y"], d["momentum_z"]]))
            return d

    class CV:
        def __init__(self, *args, **kwargs):
            pass

        def measure(self):
            box = np.array([0.5, 0.75]) + box_rate * time_state[0]
            raw_samples.append((time_state[0], box.copy(), surface.copy()))
            return dict(momentum_y=box[0], momentum_z=box[1], surface_y=surface[0], surface_z=surface[1])

    patch.setitem(sys.modules, "fastfsi.coupling.immersed_boundary", make_module(
        "fastfsi.coupling.immersed_boundary", PlanarImmersedBoundaryCoupler=Coupler))
    patch.setitem(sys.modules, "fastfsi.simulation.monitors", make_module(
        "fastfsi.simulation.monitors", ControlVolumeForce=CV))
    out = tmp_path / "run"
    end = 3.25 * DT if clipped else 6. * DT
    runner.main(["--arch", "cpu", "--preset", "fsi2" if rho_s != rho_f else "fsi3", "--grid", "4,2,2",
                 "--end-time", str(end), "--dt-max", str(DT), "--excess-mass-mode", "off", "--row-interval", "2", "--output", str(out)])
    with (out / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    final = json.loads((out / "final.json").read_text())
    assert final["status"] == "COMPLETED"
    assert final["steps"] == (4 if clipped else 6)
    assert final["t_s"] == end
    assert advances == ([DT, DT, DT, DT / 4.] if clipped else [DT] * 6)
    return rows, json.loads((out / "run_meta.json").read_text()), raw_samples, beam_samples


@pytest.mark.parametrize("acceleration,box_rate,surface,clipped", (
    ((0., 0.), (0., 0.), (2., 3.), False),                         # constant momentum
    ((1., -2.), (2., -4.), (0., 0.), False),                      # embedded-fluid cancellation
    ((0.75, -0.25), (5., 1.), (7., -3.), False),                  # mixed known acceleration
    ((0.75, -0.25), (5., 1.), (7., -3.), True),                   # final clipped dt, row_interval > 1
))
def test_analytic_control_volume_force(runner, tmp_path, acceleration, box_rate, surface, clipped):
    rows, meta, _, _ = run_balance(runner, tmp_path, acceleration=acceleration, box_rate=box_rate,
                                  surface=surface, clipped=clipped)
    expected = np.array(surface) - np.array(box_rate) + 2. * np.array(acceleration)  # rho_f * total free area = 2
    assert rows[0]["lift_n_per_m"] == format(surface[0], ".9g")
    assert rows[0]["drag_n_per_m"] == format(-surface[1], ".9g")  # first row keeps the surface-only convention
    for row in rows[1:]:
        observed = np.array([float(row["lift_n_per_m"]), -float(row["drag_n_per_m"])])
        assert np.allclose(observed, expected, atol=1e-10, rtol=0.)
    assert meta["effective_solid_density_kgm3"] == 1.
    assert meta["effective_density_ratio"] == 1.
    assert meta["rho_s"] == 10. and meta["rho_f"] == 1.
    assert meta["control_volume"]["beam_momentum_scale"] == 0.1
    assert meta["control_volume"]["first_row"] == "surface_only_no_previous_momentum_sample"
    # the method's limits are documented in the example's docstring, not repeated as prose in every run_meta
    assert "scope" not in meta["control_volume"]
    doc = " ".join(runner[0].__doc__.split())
    assert "same-particle-sum inertia consistency" in doc and "not clipped" in doc


def test_equal_density_fixed_binary_interval_keeps_old_csv_values(runner, tmp_path):
    rows, meta, samples, beam = run_balance(runner, tmp_path, rho_s=1., rho_f=1.,
        acceleration=(0.75, -0.25), box_rate=(5., 1.), surface=(7., -3.))
    for i, row in enumerate(rows):
        t, box, surface = samples[i]
        force = surface
        if i:
            actual_interval = t - samples[i - 1][0]
            old_interval = DT * 2
            assert np.float64(actual_interval).tobytes() == np.float64(old_interval).tobytes()
            assert np.array(beam[i] * 1.).tobytes() == beam[i].tobytes()
            force = force - (box - samples[i - 1][1]) / old_interval + (beam[i] - beam[i - 1]) / old_interval
        assert row["lift_n_per_m"] == format(force[0], ".9g")
        assert row["drag_n_per_m"] == format(-force[1], ".9g")
    assert meta["control_volume"]["beam_momentum_scale"] == 1.
