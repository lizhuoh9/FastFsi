"""Checkpoint / restart of the Turek-Hron example on a tiny grid (CPU, one thread, separate processes): a run continued
from its middle checkpoint reproduces the straight run bit for bit -- the history rows after the restart (the
wall-clock column aside), every array and host value of the final checkpoint, and the loop variables.  Two cases:
FSI3 with the explicit Euler update, FSI2 with the midpoint rule and the excess-mass partner points."""
import csv
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from fastfsi.io.checkpoint import decode_host

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "turek_hron_fsi" / "run_turek_hron_fsi.py"
RUNNER = Path(__file__).resolve().parent / "example_runner.py"
COMMON = ["--grid", "4,24,144", "--end-time", "2e-3", "--row-interval", "1", "--arch", "cpu"]
CASES = {
    "fsi3_euler": ["--preset", "fsi3"],
    "fsi2_midpoint_excess_mass": ["--preset", "fsi2", "--time-integration", "midpoint"],
}
WALL_COLUMNS = ("wall_s_per_step",)


def run_example(args, cwd):
    path = os.pathsep.join([str(ROOT / "src"), str(EXAMPLE.parent)])
    env = dict(os.environ, PYTHONPATH=path, PYTHONHASHSEED="1", PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, "-B", str(RUNNER), str(EXAMPLE), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]


def history(directory):
    with open(directory / "history.csv", newline="") as stream:
        rows = list(csv.reader(stream))
    keep = [i for i, name in enumerate(rows[0]) if name not in WALL_COLUMNS]
    return [rows[0][i] for i in keep], [[row[i] for i in keep] for row in rows[1:]]


def header(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("__checkpoint__.json"))


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_continuation_reproduces_the_straight_run(tmp_path, case):
    args = CASES[case] + COMMON
    straight, continued = tmp_path / "straight", tmp_path / "continued"
    run_example(args + ["--checkpoint-at", "1e-3", "--checkpoint-at", "2e-3", "--output", str(straight)], tmp_path)
    middle, final = sorted((straight / "checkpoints").glob("checkpoint_s*.npz"))
    run_example(args + ["--checkpoint-at", "2e-3", "--restart-from", str(middle), "--output", str(continued)],
                tmp_path)

    for directory in (straight, continued):
        assert json.loads((directory / "final.json").read_text())["status"] == "COMPLETED"
    restart = json.loads((continued / "restart.json").read_text())
    assert restart["diagnostic"] is False and restart["step"] == decode_host(header(middle)["loop_state"])["step"]
    columns_a, rows_a = history(straight)
    columns_b, rows_b = history(continued)
    assert columns_a == columns_b
    assert rows_b and rows_b == [row for row in rows_a if int(row[0]) > restart["step"]]

    (final_b,) = sorted((continued / "checkpoints").glob("checkpoint_s*.npz"))
    a, b = header(final), header(final_b)
    assert final.name == final_b.name
    assert [(e["component"], e["name"], e["sha256"]) for e in a["arrays"]] == \
        [(e["component"], e["name"], e["sha256"]) for e in b["arrays"]]
    assert a["host_state"] == b["host_state"]
    assert a["loop_state"] == b["loop_state"]
