"""Run an example script on the tests' deterministic CPU runtime (cpu_runtime.py), as a separate process.
usage: python -B example_runner.py SCRIPT [script arguments ...]"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_runtime import initialize  # noqa: E402

if __name__ == "__main__":
    script = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(script.parent))
    initialize()
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")
