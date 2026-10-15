"""fastfsi.simulation.taichi_runtime: the CPU backend is an explicit opt-in (a RuntimeWarning says it is slow), its
thread count is configurable, and invalid thread settings are refused.  Each case runs in a separate process because a
Taichi runtime is initialized once per process."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def run(code):
    env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    return subprocess.run([sys.executable, "-B", "-W", "always", "-c", code], env=env, capture_output=True, text=True,
                          timeout=600)


def test_cpu_opt_in_initializes_the_cpu_with_a_warning():
    done = run("from fastfsi.simulation.taichi_runtime import TaichiRuntimeConfig, init_taichi, taichi_runtime_identity\n"
               "import taichi as ti\n"
               "init_taichi(TaichiRuntimeConfig(arch='cpu', cpu_max_num_threads=1, offline_cache=False))\n"
               "print('ARCH_IS_CPU', ti.cfg.arch == ti.cpu, ti.cfg.cpu_max_num_threads)\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert "ARCH_IS_CPU True 1" in done.stdout
    assert "RuntimeWarning" in done.stderr and "CPU backend" in done.stderr


@pytest.mark.parametrize("config", ("arch='cuda', cpu_max_num_threads=2", "arch='cpu', cpu_max_num_threads=0",
                                    "arch='cpu', cpu_max_num_threads=True"))
def test_invalid_thread_settings_are_refused(config):
    done = run("from fastfsi.simulation.taichi_runtime import TaichiRuntimeConfig, init_taichi\n"
               "init_taichi(TaichiRuntimeConfig(%s))\n" % config)
    assert done.returncode != 0 and "cpu_max_num_threads" in done.stderr
