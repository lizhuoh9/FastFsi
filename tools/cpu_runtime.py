"""Serial CPU harness for the FastFsi tests and CPU runs of the drivers: select the CPU before importing cases;
never probe CUDA.

The production runtime (simulation_core/diagnostics/runtime.init_taichi) is GPU-only and rejects arch='cpu'.  This
harness initialises Taichi on the CPU (f32, one thread, opt_level 0, offline cache off, no fallback) and rebinds
init_taichi in the modules that call it to a check that the runtime is the CPU.  Only backend selection is adapted;
numerical code and assertions are unchanged.  Each job uses a fresh process.

Source: UDF/u1_cpu.py of the source repository (with the CPU aliases of UDF/ki2_entry.py); the rebinding of the
sharp-interface modules, which are not part of FastFsi, is removed.

usage:
  python -B -m tools.cpu_runtime tests/fast [pytest args]                       # the CPU test suite
  python -B -m tools.cpu_runtime --module cases.turek_fast --arch cpu [args]     # a driver on the CPU
"""
import argparse
from pathlib import Path
import runpy
import sys


def initialize():
    import taichi as ti
    from simulation_core.diagnostics import runtime
    original_init = ti.init

    def init(*args, **kwargs):
        kwargs.update(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1,
                      opt_level=0, offline_cache=False, enable_fallback=False,
                      offline_cache_file_path=str(Path.cwd() / 'tmp/u1/cache'))
        original_init(**kwargs)
        runtime._INITIALIZED = True
        runtime._INITIALIZED_ARCH = 'cpu'
        runtime._INITIALIZED_FP = 'f32'
        runtime._INITIALIZED_RANDOM_SEED = 0
        runtime._INITIALIZED_OFFLINE_CACHE = False
        runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH = str(Path.cwd() / 'tmp/u1/cache')
        runtime._INITIALIZED_NUM_COMPILE_THREADS = 1

    ti.init = init
    init()
    # The production selector rejects requested CPU. Existing numerical test
    # fixtures use this same preinitialization convention; no GPU is queried.
    def ensure_cpu(_configuration=None):
        if ti.cfg.arch != ti.cpu:
            raise RuntimeError('U1 requires the actual CPU runtime')
    runtime.init_taichi = ensure_cpu
    from simulation_core.fluids import solver
    solver.init_taichi = ensure_cpu
    # UDF/ki2_entry.py (--backend cpu): the package-level aliases point at the same CPU check.
    import simulation_core
    from simulation_core import diagnostics
    simulation_core.init_taichi = diagnostics.init_taichi = runtime.init_taichi
    print('U1_CPU_RUNTIME_ADAPTATION: numerical assertions unchanged; no CUDA', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--module')
    args, rest = parser.parse_known_args()
    sys.path.insert(0, str(args.repo.resolve()))
    initialize()
    if args.module:
        sys.argv = [args.module, *rest]
        runpy.run_module(args.module, run_name='__main__')
    else:
        import pytest
        raise SystemExit(pytest.main(['-q', '-s', '-p', 'no:cacheprovider', *rest]))
