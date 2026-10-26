"""fastfsi.io.restart: the checkpoint schedule, the restart decision on manifests, the run-record helpers, and the Turek
example's restart arguments, argument files and reference volumetric modulus.  Pure Python (no solver)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from fastfsi.io.checkpoint import read_checkpoint
from fastfsi.io.restart import (CheckpointPlan, check_restart, code_manifest, device_changed, grid_manifest,
                                restart_record)

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "turek_hron_fsi"


class Counter:
    """A tiny checkpointable component: one array and one host value."""

    def __init__(self):
        self.values, self.steps = np.arange(4.0), 0

    def state_arrays(self):
        return {"values": self.values}

    def state_host(self):
        return {"steps": self.steps}


def manifest(**changes):
    base = dict(config={"preset": "fsi3", "grid": "4,24,144"}, grid=dict(nodes=[4, 24, 144], faces_sha256="f"),
                coupler="c", code={"fastfsi.fluid.solver": "s"}, libraries=dict(taichi="1.7.3"),
                device=dict(arch="cuda", gpus=["GPU A"]), inputs=dict(paths={"/a": "x"}))
    for key, value in changes.items():
        base[key] = value
    return base


def test_schedule_every_interval_and_explicit_times(tmp_path):
    component = Counter()
    plan = CheckpointPlan(tmp_path, 0.25, [0.1, 0.6], dict(counter=component), dict(m=1), 0.0, label="TEST")
    assert plan.planned_count(1.0, 0.0) == 4 + 2
    due = []
    for step, t in ((5, 0.05), (10, 0.1), (20, 0.2), (26, 0.26), (40, 0.4), (50, 0.5), (61, 0.61), (74, 0.74),
                    (76, 0.76)):
        if plan.due(t):
            plan.write(step, t, dict(t=t))
            due.append(t)
    assert due == [0.1, 0.26, 0.5, 0.61, 0.76]
    rows = [json.loads(line) for line in (tmp_path / "checkpoints" / "checkpoints.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [10, 26, 50, 61, 76]
    arrays, host, loop_state, saved_manifest, _ = read_checkpoint(rows[0]["file"])
    assert arrays["counter"]["values"].tolist() == [0.0, 1.0, 2.0, 3.0] and host == {"counter": {"steps": 0}}
    assert loop_state == dict(t=0.1) and saved_manifest == dict(m=1)


def test_schedule_after_a_restart_skips_past_times(tmp_path):
    plan = CheckpointPlan(tmp_path, 0.25, [0.1, 0.6], dict(counter=Counter()), {}, 0.3)
    assert plan.next_every == pytest.approx(0.5) and plan.pending == [0.6]
    assert plan.planned_count(1.0, 0.3) == 3 + 1
    assert CheckpointPlan(tmp_path, 0.0, [], {}, {}, 0.0).planned_count(1.0, 0.0) == 0


@pytest.mark.parametrize("every,at", ((-1.0, []), (float("nan"), []), (0.0, [0.0]), (0.0, [float("inf")])))
def test_schedule_rejects_invalid_times(tmp_path, every, at):
    with pytest.raises(ValueError):
        CheckpointPlan(tmp_path, every, at, {}, {}, 0.0)


def test_restart_accepts_an_identical_manifest_and_ignores_the_device_and_input_paths():
    other_device = manifest(device=dict(arch="cuda", gpus=["GPU B"]), inputs=dict(paths={"/b": "x"}))
    assert check_restart(manifest(), other_device, diagnostic=False, output_name="run_2") == []
    assert device_changed(manifest()["device"], other_device["device"])


@pytest.mark.parametrize("change", (dict(config={"preset": "fsi2", "grid": "4,24,144"}), dict(coupler="other"),
                                    dict(grid=dict(nodes=[4, 48, 288], faces_sha256="f"))))
def test_restart_refuses_a_different_set_up(change):
    with pytest.raises(ValueError, match="restart refused"):
        check_restart(manifest(), manifest(**change), diagnostic=True, output_name="run_DIAGNOSTIC")


def test_a_code_difference_needs_a_diagnostic_restart_into_a_marked_directory():
    changed = manifest(code={"fastfsi.fluid.solver": "t"})
    with pytest.raises(ValueError, match="restart refused"):
        check_restart(manifest(), changed, diagnostic=False, output_name="run_2")
    with pytest.raises(ValueError, match="DIAGNOSTIC"):
        check_restart(manifest(), changed, diagnostic=True, output_name="run_2")
    diffs = check_restart(manifest(), changed, diagnostic=True, output_name="run_2_DIAGNOSTIC")
    assert diffs == [("code/fastfsi.fluid.solver", "s", "t")]


def test_restart_record_and_manifest_helpers(tmp_path):
    checkpoint = tmp_path / "out" / "checkpoints" / "checkpoint_s000000012.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"data")
    record = restart_record(checkpoint, dict(t=0.125, step=12), manifest(), manifest(), dict(created_utc="x"), [])
    assert record["step"] == 12 and record["t_text"] == "0.125" and record["diagnostic"] is False
    assert record["source_output"] == str((tmp_path / "out").resolve()) and record["device_changed"] is False
    faces = (np.linspace(0.0, 1.0, 5), np.linspace(0.0, 2.0, 3), np.linspace(0.0, 3.0, 4))
    moved = (faces[0], faces[1], faces[2] + 1.0e-9)
    assert grid_manifest((4, 2, 3), faces) == grid_manifest([4, 2, 3], [list(f) for f in faces])
    assert grid_manifest((4, 2, 3), faces)["faces_sha256"] != grid_manifest((4, 2, 3), moved)["faces_sha256"]
    assert grid_manifest((4, 2, 3), faces)["nodes"] == [4, 2, 3]
    script = tmp_path / "script.py"
    script.write_text("x = 1\n")
    code = code_manifest(("fastfsi",), {"script.py": script})
    assert "fastfsi.io.restart" in code and len(code["script.py"]) == 64


def load_example():
    spec = importlib.util.spec_from_file_location("turek_example_arguments", EXAMPLE_DIR / "run_turek_hron_fsi.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_reference_volumetric_modulus():
    example = load_example()
    assert example.reference_kappa_over_mu(0.4) == 14.0 / 3.0 == 4.666666666666667
    assert example.reference_kappa_over_mu(0.25) == pytest.approx(5.0 / 3.0)
    with pytest.raises(ValueError):
        example.reference_kappa_over_mu(0.5)


@pytest.mark.parametrize("name,preset,grid", (("fsi1", "fsi1", "4,96,576"), ("fsi2", "fsi2", "4,144,864"),
                                              ("fsi3", "fsi3", "4,96,576")))
def test_example_argument_files_hold_the_reference_parameters(tmp_path, name, preset, grid):
    example = load_example()
    a = example.parse(["@%s" % (EXAMPLE_DIR / "configs" / ("%s.args" % name)), "--output", str(tmp_path / "x"),
                       "--end-time", "4.0"])
    assert (a.preset, a.grid, a.ib_kernel, a.kappa_over_mu, a.end_time) == (preset, grid, "ib4", 14.0 / 3.0, 4.0)
    assert a.equivalent_bending_nu is None and a.excess_mass_mode == "auto" and a.dt_max == 1.0e-3
    assert a.cfl_elastic == 0.3 and a.ramp_time is None


def test_example_restart_arguments(tmp_path):
    example = load_example()
    a = example.parse(["--end-time", "1", "--output", str(tmp_path), "--checkpoint-every-s", "0.25",
                       "--checkpoint-at", "0.1", "--checkpoint-at", "0.6"])
    assert a.checkpoint_every_s == 0.25 and a.checkpoint_at == [0.1, 0.6] and a.restart_from is None
    assert a.kappa_over_mu is None                  # resolved from the preset's Poisson ratio at the start of main()
    for bad in (["--diagnostic-restart"], ["--checkpoint-every-s", "-1"], ["--end-time", "nan"], ["--dt-max", "0"],
                ["--cfl-elastic", "-0.3"], ["--row-interval", "0"], ["--vtk-interval-s", "nan"]):
        with pytest.raises(SystemExit):
            example.parse(["--end-time", "1", "--output", str(tmp_path)] + bad)
    assert set(example.RESTART_FREE_ARGS) <= set(vars(a))
