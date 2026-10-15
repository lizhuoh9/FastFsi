"""Taichi runtime set-up."""
from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass
from numbers import Integral

import taichi as ti


_SUPPORTED_ARCHS = ("cuda", "gpu", "cpu")      # cpu: an explicit opt-in only (slow; for testing and small cases)
_SUPPORTED_FPS = ("f32", "f64")
_INITIALIZATION_ENVIRONMENT_NAMES = (
    "SIMULATION_TAICHI_OFFLINE_CACHE",
    "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH",
    "TI_OFFLINE_CACHE",
    "TI_OFFLINE_CACHE_FILE_PATH",
    "TI_OFFLINE_CACHE_MAX_SIZE_OF_FILES",
    "TI_NUM_COMPILE_THREADS",
)
_COMPILER_BOOLEAN_FIELDS = (
    "cfg_optimization",
    "advanced_optimization",
    "fast_math",
    "debug",
)


@dataclass(frozen=True)
class TaichiRuntimeConfig:
    arch: str = "cuda"
    default_fp: str = "f32"
    random_seed: int = 0
    offline_cache: bool | None = None
    offline_cache_file_path: str | None = None
    strict_arch: bool = False
    default_ip: str | None = None
    cfg_optimization: bool | None = None
    opt_level: int | None = None
    advanced_optimization: bool | None = None
    fast_math: bool | None = None
    debug: bool | None = None
    cpu_max_num_threads: int | None = None       # arch="cpu" only; 1 gives the same bits run to run


_INITIALIZED = False
_INITIALIZED_ARCH: str | None = None
_INITIALIZED_FP: str | None = None
_INITIALIZED_RANDOM_SEED: int | None = None
_INITIALIZED_OFFLINE_CACHE: bool | None = None
_INITIALIZED_OFFLINE_CACHE_FILE_PATH: str | None = None
_INITIALIZED_OFFLINE_CACHE_MAX_SIZE_OF_FILES: int | None = None
_INITIALIZED_NUM_COMPILE_THREADS: int | None = None
_INITIALIZED_ENVIRONMENT: dict[str, str | None] = {}
_INITIALIZED_STRICT_ARCH_VERIFIED = False
_INIT_LOCK = threading.RLock()


def _environment_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off"
    )


def _normalized_cache_path(value: object, source: str) -> str:
    try:
        path = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"{source} must be a filesystem path") from exc
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{source} must be a non-blank filesystem path")
    return os.path.normcase(os.path.abspath(os.path.expanduser(path)))


def _requested_offline_cache(
    config: TaichiRuntimeConfig,
) -> tuple[bool, str | None]:
    offline_cache = config.offline_cache
    if offline_cache is not None and not isinstance(offline_cache, bool):
        raise ValueError("offline_cache must be a bool or None")
    if offline_cache is None:
        offline_cache = _environment_flag("SIMULATION_TAICHI_OFFLINE_CACHE")
    if offline_cache is None:
        offline_cache = _environment_flag("TI_OFFLINE_CACHE")
    if offline_cache is None:
        # Match Taichi's frontend default and make it explicit to ti.init.
        offline_cache = True

    if config.offline_cache_file_path is not None:
        normalized_path = _normalized_cache_path(
            config.offline_cache_file_path,
            "offline_cache_file_path",
        )
    else:
        offline_cache_file_path = os.environ.get(
            "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH"
        )
        path_source = "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH"
        if offline_cache_file_path in (None, ""):
            offline_cache_file_path = os.environ.get(
                "TI_OFFLINE_CACHE_FILE_PATH"
            )
            path_source = "TI_OFFLINE_CACHE_FILE_PATH"
        normalized_path = None
        if offline_cache_file_path not in (None, ""):
            normalized_path = _normalized_cache_path(
                offline_cache_file_path,
                path_source,
            )
    return offline_cache, normalized_path


def _assert_actual_runtime_arch(requested_arch: str) -> None:
    actual_arch = getattr(ti.cfg, "arch", None)
    if requested_arch == "cuda":
        matches_request = actual_arch == ti.cuda
    elif requested_arch == "cpu":
        matches_request = actual_arch == ti.cpu
    else:
        matches_request = actual_arch in ti.gpu
    if not matches_request:
        raise RuntimeError(
            "Taichi actual runtime arch="
            f"{actual_arch!r} does not match requested arch="
            f"{requested_arch!r}; TI_ARCH may have overridden the request"
        )


def _actual_runtime_arch_name() -> str:
    actual_arch = getattr(ti.cfg, "arch", None)
    for name in (
        "cuda",
        "cpu",
        "x64",
        "arm64",
        "metal",
        "opengl",
        "vulkan",
        "dx11",
        "amdgpu",
        "gpu",
    ):
        candidate = getattr(ti, name, None)
        if candidate is not None and actual_arch == candidate:
            return name
    raise RuntimeError(
        "Taichi runtime reports an unrecognized actual arch: "
        f"{actual_arch!r}"
    )


def _actual_runtime_default_fp_name() -> str:
    actual_default_fp = getattr(ti.cfg, "default_fp", None)
    if actual_default_fp == ti.f32:
        return "f32"
    if actual_default_fp == ti.f64:
        return "f64"
    raise RuntimeError(
        "Taichi runtime reports an unrecognized actual default_fp: "
        f"{actual_default_fp!r}"
    )


def _requested_compiler_configuration(
    config: TaichiRuntimeConfig,
) -> dict[str, object]:
    requested: dict[str, object] = {}
    if config.default_ip is not None:
        if (
            not isinstance(config.default_ip, str)
            or config.default_ip not in {"i32", "i64"}
        ):
            raise ValueError("default_ip must be 'i32', 'i64', or None")
        requested["default_ip"] = config.default_ip

    if config.opt_level is not None:
        if isinstance(config.opt_level, bool) or not isinstance(
            config.opt_level, Integral
        ):
            raise ValueError("opt_level must be an integer or None")
        requested["opt_level"] = int(config.opt_level)

    for field in _COMPILER_BOOLEAN_FIELDS:
        value = getattr(config, field)
        if value is not None:
            if not isinstance(value, bool):
                raise ValueError(f"{field} must be a bool or None")
            requested[field] = value
    return requested


def _actual_compiler_configuration() -> dict[str, object]:
    actual_default_ip = ti.cfg.default_ip
    if actual_default_ip == ti.i32:
        default_ip = "i32"
    elif actual_default_ip == ti.i64:
        default_ip = "i64"
    else:
        raise RuntimeError(
            "Taichi runtime reports an unrecognized actual default_ip: "
            f"{actual_default_ip!r}"
        )
    return {
        "default_ip": default_ip,
        "cfg_optimization": bool(ti.cfg.cfg_optimization),
        "opt_level": int(ti.cfg.opt_level),
        "advanced_optimization": bool(ti.cfg.advanced_optimization),
        "fast_math": bool(ti.cfg.fast_math),
        "debug": bool(ti.cfg.debug),
    }


def _assert_actual_compiler_configuration(
    requested: dict[str, object],
) -> None:
    if not requested:
        return
    actual = _actual_compiler_configuration()
    mismatches = "; ".join(
        f"{field}: actual={actual[field]!r}, requested={value!r}"
        for field, value in requested.items()
        if actual[field] != value
    )
    if mismatches:
        raise RuntimeError(
            "Taichi actual compiler configuration does not match the "
            f"explicit request: {mismatches}"
        )


def _compiler_init_kwargs(
    requested: dict[str, object],
) -> dict[str, object]:
    kwargs = dict(requested)
    default_ip = kwargs.get("default_ip")
    if default_ip is not None:
        kwargs["default_ip"] = ti.i32 if default_ip == "i32" else ti.i64
    return kwargs


def taichi_runtime_identity() -> dict[str, object]:
    """Return the initialized Taichi runtime identity for persisted evidence.

    This is deliberately unavailable before ``init_taichi`` has published a
    successful initialization: an inferred or requested identity is not valid
    execution provenance.
    """

    with _INIT_LOCK:
        if (
            not _INITIALIZED
            or _INITIALIZED_ARCH is None
            or _INITIALIZED_RANDOM_SEED is None
            or _INITIALIZED_OFFLINE_CACHE is None
        ):
            raise RuntimeError(
                "Taichi runtime identity is unavailable because Taichi is not initialized"
            )
        if _INITIALIZED_STRICT_ARCH_VERIFIED:
            _assert_actual_runtime_arch(_INITIALIZED_ARCH)
        return {
            "requested_arch": _INITIALIZED_ARCH,
            "actual_arch": _actual_runtime_arch_name(),
            "default_fp": _actual_runtime_default_fp_name(),
            "random_seed": _INITIALIZED_RANDOM_SEED,
            "compiler_configuration": {
                "taichi_version": ".".join(map(str, ti.__version__)),
                **_actual_compiler_configuration(),
                "num_compile_threads": _INITIALIZED_NUM_COMPILE_THREADS,
            },
            "offline_cache_identity": {
                "enabled": _INITIALIZED_OFFLINE_CACHE,
                "file_path": _INITIALIZED_OFFLINE_CACHE_FILE_PATH,
                "max_size_of_files": _INITIALIZED_OFFLINE_CACHE_MAX_SIZE_OF_FILES,
            },
            "initialization_environment": dict(_INITIALIZED_ENVIRONMENT),
            "strict_arch_verified": _INITIALIZED_STRICT_ARCH_VERIFIED,
        }


def init_taichi(config: TaichiRuntimeConfig | None = None) -> None:
    """Initialize Taichi once for the simulation core.

    The first call wins. Legacy implicit calls (``config is None``) reuse an
    existing runtime when its floating-point mode matches. Explicit configs
    must match the existing architecture, floating-point mode, random seed,
    and offline-cache identity. Conflicting explicit requests fail instead
    of producing misleading provenance.
    """

    implicit_default_request = config is None
    cfg = config or TaichiRuntimeConfig()
    requested_arch = cfg.arch.lower()
    if requested_arch not in _SUPPORTED_ARCHS:
        raise ValueError(f"unsupported Taichi arch: {cfg.arch!r}")
    if not isinstance(cfg.strict_arch, bool):
        raise ValueError("strict_arch must be a bool")
    requested_fp = str(cfg.default_fp)
    if requested_fp not in _SUPPORTED_FPS:
        raise ValueError(
            f"unsupported Taichi default_fp: {cfg.default_fp!r}; expected one of {_SUPPORTED_FPS}"
        )
    if isinstance(cfg.random_seed, bool) or not isinstance(
        cfg.random_seed, Integral
    ):
        raise ValueError("random_seed must be an integer")
    requested_random_seed = int(cfg.random_seed)
    threads = cfg.cpu_max_num_threads
    if threads is not None and (requested_arch != "cpu" or isinstance(threads, bool)
                                or not isinstance(threads, Integral) or threads < 1):
        raise ValueError("cpu_max_num_threads must be a positive integer, and only with arch='cpu'")
    requested_compiler = _requested_compiler_configuration(cfg)

    global _INITIALIZED, _INITIALIZED_ARCH, _INITIALIZED_FP
    global _INITIALIZED_RANDOM_SEED
    global _INITIALIZED_OFFLINE_CACHE
    global _INITIALIZED_OFFLINE_CACHE_FILE_PATH
    global _INITIALIZED_OFFLINE_CACHE_MAX_SIZE_OF_FILES
    global _INITIALIZED_NUM_COMPILE_THREADS, _INITIALIZED_ENVIRONMENT
    global _INITIALIZED_STRICT_ARCH_VERIFIED
    with _INIT_LOCK:
        if _INITIALIZED:
            if cfg.strict_arch:
                _assert_actual_runtime_arch(requested_arch)
                _INITIALIZED_STRICT_ARCH_VERIFIED = True
            if (
                not implicit_default_request
                and _INITIALIZED_ARCH is not None
                and requested_arch != _INITIALIZED_ARCH
            ):
                raise ValueError(
                    "Taichi is already initialized with "
                    f"arch={_INITIALIZED_ARCH!r}; cannot re-initialize with "
                    f"arch={requested_arch!r}"
                )
            if _INITIALIZED_FP is not None and requested_fp != _INITIALIZED_FP:
                raise ValueError(
                    "Taichi is already initialized with "
                    f"default_fp={_INITIALIZED_FP!r}; cannot re-initialize with "
                    f"default_fp={requested_fp!r}"
                )
            if implicit_default_request:
                return
            offline_cache, offline_cache_file_path = (
                _requested_offline_cache(cfg)
            )
            if requested_random_seed != _INITIALIZED_RANDOM_SEED:
                raise ValueError(
                    "Taichi is already initialized with "
                    f"random_seed={_INITIALIZED_RANDOM_SEED!r}; cannot "
                    f"re-initialize with random_seed={requested_random_seed!r}"
                )
            initialized_cache_identity = (
                _INITIALIZED_OFFLINE_CACHE,
                _INITIALIZED_OFFLINE_CACHE_FILE_PATH,
            )
            requested_cache_identity = (
                offline_cache,
                offline_cache_file_path,
            )
            if requested_cache_identity != initialized_cache_identity:
                raise RuntimeError(
                    "Taichi is already initialized with offline-cache "
                    f"configuration={initialized_cache_identity!r}; cannot "
                    f"reconfigure it as {requested_cache_identity!r}"
                )
            _assert_actual_compiler_configuration(requested_compiler)
            return

        offline_cache, offline_cache_file_path = _requested_offline_cache(cfg)
        arch = {"cuda": ti.cuda, "cpu": ti.cpu}.get(requested_arch, ti.gpu)
        if requested_arch == "cpu":
            warnings.warn("fastfsi on the CPU backend: the same code as on the GPU, but slow; meant for testing and "
                          "small cases", RuntimeWarning, stacklevel=2)
        default_fp = ti.f32 if requested_fp == "f32" else ti.f64
        taichi_kwargs: dict[str, object] = {
            "offline_cache": bool(offline_cache),
            **_compiler_init_kwargs(requested_compiler),
        }
        if offline_cache_file_path:
            taichi_kwargs["offline_cache_file_path"] = offline_cache_file_path
        if cfg.strict_arch:
            taichi_kwargs["enable_fallback"] = False
        if threads is not None:
            taichi_kwargs["cpu_max_num_threads"] = int(threads)
        initialization_environment = {
            name: os.environ.get(name) for name in _INITIALIZATION_ENVIRONMENT_NAMES
        }
        ti.init(
            arch=arch,
            default_fp=default_fp,
            random_seed=requested_random_seed,
            **taichi_kwargs,
        )
        if cfg.strict_arch:
            _assert_actual_runtime_arch(requested_arch)
        _assert_actual_compiler_configuration(requested_compiler)
        if not bool(ti.lang.impl.current_cfg().debug):
            # Initialized non-debug runtime: ndarray allocation call-stack strings empty; debug/uninit: original (ndarray_debug_info).
            from fastfsi.simulation.ndarray_debug_info import use_empty_ndarray_debug_info
            use_empty_ndarray_debug_info()
        # These supported CompileConfig fields describe the initialized runtime.
        # Missing API values remain unavailable; ambient environment is only
        # provenance and must never be substituted for an actual setting.
        actual_cache_limit = getattr(ti.cfg, "offline_cache_max_size_of_files", None)
        actual_compile_threads = getattr(ti.cfg, "num_compile_threads", None)
        cache_limit = None if actual_cache_limit is None else int(actual_cache_limit)
        compile_threads = None if actual_compile_threads is None else int(actual_compile_threads)
        _INITIALIZED = True
        _INITIALIZED_ARCH = requested_arch
        _INITIALIZED_FP = requested_fp
        _INITIALIZED_RANDOM_SEED = requested_random_seed
        _INITIALIZED_OFFLINE_CACHE = offline_cache
        _INITIALIZED_OFFLINE_CACHE_FILE_PATH = offline_cache_file_path
        _INITIALIZED_OFFLINE_CACHE_MAX_SIZE_OF_FILES = cache_limit
        _INITIALIZED_NUM_COMPILE_THREADS = compile_threads
        _INITIALIZED_ENVIRONMENT = initialization_environment
        _INITIALIZED_STRICT_ARCH_VERIFIED = bool(cfg.strict_arch)
