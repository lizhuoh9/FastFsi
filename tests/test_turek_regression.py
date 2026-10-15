"""Regression of the Turek-Hron example on a tiny grid (CPU, one thread, a separate process): the history of a short
FSI3 run and of a short FSI2 run with the excess-mass partner points must stay at the stored values (tests/data).
Integer columns (step, conjugate-gradient iterations, counts) must be equal, text columns too, every other column
within a relative 1e-6 (an absolute 1e-12 near zero); the wall-clock column is not compared.  When a change of the
numerics is intended, regenerate the data with REGENERATE=1 and review the difference."""
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "turek_hron_fsi" / "run_turek_hron_fsi.py"
RUNNER = Path(__file__).resolve().parent / "example_runner.py"
DATA = Path(__file__).resolve().parent / "data"
CASES = {
    "turek_fsi3_4x24x144_20_steps": ["--preset", "fsi3", "--max-steps", "20"],
    "turek_fsi2_excess_mass_4x24x144_16_steps": ["--preset", "fsi2", "--max-steps", "16"],
}
COMMON = ["--grid", "4,24,144", "--end-time", "1.0", "--row-interval", "1", "--arch", "cpu"]
WALL_COLUMNS = ("wall_s_per_step",)


def run_history(args, cwd):
    path = os.pathsep.join([str(ROOT / "src"), str(EXAMPLE.parent)])
    env = dict(os.environ, PYTHONPATH=path, PYTHONHASHSEED="1", PYTHONDONTWRITEBYTECODE="1")
    out = cwd / "run"
    done = subprocess.run([sys.executable, "-B", str(RUNNER), str(EXAMPLE), *args, "--output", str(out)], cwd=cwd,
                          env=env, capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return out / "history.csv"


def read(path):
    with open(path, newline="") as stream:
        rows = list(csv.reader(stream))
    keep = [i for i, name in enumerate(rows[0]) if name not in WALL_COLUMNS]
    return [rows[0][i] for i in keep], [[row[i] for i in keep] for row in rows[1:]]


def same_value(expected, actual):
    try:
        a, b = float(expected), float(actual)
    except ValueError:
        return expected == actual
    if a.is_integer() and b.is_integer() and "." not in expected + actual and "e" not in (expected + actual).lower():
        return a == b
    return math.isclose(a, b, rel_tol=1.0e-6, abs_tol=1.0e-12)


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_history_matches_the_stored_values(tmp_path, case):
    columns, rows = read(run_history(CASES[case] + COMMON, tmp_path))
    stored = DATA / ("%s.csv" % case)
    if os.environ.get("REGENERATE") == "1":
        DATA.mkdir(exist_ok=True)
        with open(stored, "w", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows([columns] + rows)
        pytest.skip("regenerated %s" % stored)
    stored_columns, stored_rows = read(stored)
    assert columns == stored_columns
    assert len(rows) == len(stored_rows)
    differing = [(row[0], name, s, a) for row, stored_row in zip(rows, stored_rows)
                 for name, s, a in zip(columns, stored_row, row) if not same_value(s, a)]
    assert differing == []
