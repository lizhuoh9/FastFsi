"""The deterministic CPU runtime of the tests: Taichi on the CPU with one thread (f32 default, opt_level 0, no offline
cache, no fallback), so that two runs give the same bits, and fastfsi.simulation.taichi_runtime adapted so that the
library's own init_taichi only checks that this CPU runtime is active.  Numerical code is unchanged.
Usage in a test: from cpu_runtime import initialize; initialize()."""
import importlib
from pathlib import Path

ADAPTED_MODULES = ("fastfsi.fluid.solver",)     # modules that imported init_taichi by name


def initialize(cache_dir=None):
    import taichi as ti
    from fastfsi.simulation import taichi_runtime as runtime
    cache = str(Path(cache_dir) if cache_dir is not None else Path.cwd() / "tmp" / "taichi_cache")
    original_init = ti.init

    def init(*args, **kwargs):
        kwargs.update(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1, opt_level=0, offline_cache=False,
                      enable_fallback=False, offline_cache_file_path=cache)
        original_init(**kwargs)
        runtime._INITIALIZED = True
        runtime._INITIALIZED_ARCH = "cpu"
        runtime._INITIALIZED_FP = "f32"
        runtime._INITIALIZED_RANDOM_SEED = 0
        runtime._INITIALIZED_OFFLINE_CACHE = False
        runtime._INITIALIZED_OFFLINE_CACHE_FILE_PATH = cache
        runtime._INITIALIZED_NUM_COMPILE_THREADS = 1

    ti.init = init
    init()

    def ensure_cpu(_configuration=None):
        if ti.cfg.arch != ti.cpu:
            raise RuntimeError("the tests require the CPU runtime")
    runtime.init_taichi = ensure_cpu
    for name in ADAPTED_MODULES:
        module = importlib.import_module(name)
        if hasattr(module, "init_taichi"):
            module.init_taichi = ensure_cpu
