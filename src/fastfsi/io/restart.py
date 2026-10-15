"""Checkpoint schedules and restart checks for drivers built on fastfsi.io.checkpoint.

CheckpointPlan(out, every_s, at_s, components, manifest, start_t_s, label)
    a checkpoint at the first history row at or after every multiple of every_s (0 = off) and at or after every time
    in at_s, written to out/checkpoints/checkpoint_s{step:09d}.npz (create-only) and listed in checkpoints.jsonl with
    its size, write time and sha256.  Checkpoints are never deleted.
restart_manifest(...) / the helpers below
    what a restart must match: the configuration (every argument a restart may not change), the grid, a fingerprint of
    the static setup, the source code of the loaded modules (sha256 per module), the library versions, and the device
    (logged, not compared: a run may continue on another device).
restart_policy(saved, current, *, diagnostic, output_name) / check_restart(...)
    the restart decision (classify_restart sorts every manifest difference):
      dropped      Taichi's version-check file (<Taichi repo dir>/version_info, a date stamp that Taichi's background
                   thread reads and rewrites): never an input, dropped from both sides and listed;
      allowed      only together with a recorded device/host change (the arch, the GPU UUIDs -- the GPU names where a
                   side records none -- or the host id from /etc/machine-id, else the host name; a manifest without a
                   host id proves no host change): a module loaded in this run only, a module loaded in the
                   checkpoint's run only whose source file here (found without importing it) has the checkpoint's
                   sha256 ("not loaded here; same source"), the Python patch version, hypre's build flags, the libHYPRE
                   binary, and the environment files (recorded inputs under the Taichi repo dir or the hypre prefix);
      overridable  code / library differences otherwise (a module loaded on both sides with another sha256, a module
                   loaded in the checkpoint's run only whose source here differs or is not found, the numpy / taichi /
                   hypre versions, the items above without a device/host change): only a diagnostic restart accepts
                   them, into a directory whose name contains DIAGNOSTIC (never a result);
      refused      always: a Python major / minor difference, the physics inputs, inconsistent input records, and
                   every other difference (configuration, grid, setup fingerprint, anything unknown).
    restart_policy returns the classification (with one log line per difference); check_restart returns its
    overridden differences (the earlier interface).
"""
import datetime
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from fastfsi.io.checkpoint import ABSENT, compare_manifests, write_checkpoint

CODE_MANIFEST_KEYS = ("code", "libraries")       # only a diagnostic restart may differ here
LOGGED_ONLY = ("device", "inputs/paths")          # recorded, never compared
DIAGNOSTIC_MARK = "DIAGNOSTIC"
RESTART_POLICY = "fastfsi restart policy v1"
TAICHI_VERSION_FILE = "version_info"              # Taichi's version-check date stamp: never an input
HYPRE_HEADER = "include/HYPRE_config.h"
SAME_SOURCE_NOTE = "not loaded here; same source"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path, data):
    """Write a JSON file that must not exist yet."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=1, default=str)
        stream.write("\n")


def code_manifest(packages, extra_files=()):
    """sha256 of the source of every loaded module whose top-level package is in `packages`, plus `extra_files`
    ({label: path}, e.g. the running script, which is loaded as __main__)."""
    code = {}
    for name, module in sorted(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if source and name.split(".")[0] in packages and source.endswith(".py"):
            code[name] = file_sha256(source)
    for label, path in dict(extra_files).items():
        code[label] = file_sha256(path)
    return code


def library_versions():
    import taichi as ti
    return dict(python=sys.version.split()[0], numpy=np.__version__,
                taichi=".".join(str(v) for v in ti.__version__))


def module_source_sha256(name):
    """sha256 of the source file that importing `name` would load here, found WITHOUT importing it (find_spec, which
    needs the parent package already imported); None when there is none: an unimported parent package, no spec, no .py
    source file, or a lookup error."""
    parent = name.rpartition(".")[0]
    if parent and parent not in sys.modules:
        return None
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        return None
    origin = getattr(spec, "origin", None)
    if not origin or not str(origin).endswith(".py") or not os.path.isfile(origin):
        return None
    return file_sha256(origin)


def taichi_repo_dir():
    """Taichi's own repo directory (its version-check file and default caches live there); None without Taichi."""
    try:
        from taichi._lib import core as ti_core
    except ImportError:
        return None
    return os.path.normpath(ti_core.get_repo_dir())


def is_taichi_version_file(path):
    """Taichi's version-check file (read and rewritten by a background thread at ti.init) is never an input."""
    if os.path.basename(path) != TAICHI_VERSION_FILE:
        return False
    repo = taichi_repo_dir()
    return repo is not None and os.path.normpath(path) == os.path.join(repo, TAICHI_VERSION_FILE)


def host_id():
    """/etc/machine-id if readable (read through pathlib), else the host name."""
    try:
        value = Path("/etc/machine-id").read_text().strip()
    except OSError:
        value = ""
    return ("machine-id:%s" % value) if value else ("hostname:%s" % (os.uname().nodename if hasattr(os, "uname")
                                                                      else "unknown"))


def device_identity(arch):
    """The arch, the host id, Taichi's repo dir and, on CUDA, the visible GPUs (name and compute capability; their
    uuids) from nvidia-smi."""
    out = dict(arch=str(arch), host=os.uname().nodename if hasattr(os, "uname") else None,
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), gpus=None, gpu_uuids=None,
               host_id=host_id(), taichi_repo_dir=taichi_repo_dir())
    if arch == "cuda":
        try:
            text = subprocess.run(["nvidia-smi", "--query-gpu=index,name,compute_cap,uuid", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError) as error:
            out["gpus"] = "unknown: %r" % (error,)
            return out
        rows = [[v.strip() for v in line.split(",")] for line in text.splitlines() if line.count(",") >= 3]
        visible = out["cuda_visible_devices"]
        if visible:
            wanted = [v.strip() for v in visible.split(",")]
            rows = [r for r in rows if r[0] in wanted or r[3] in wanted]
        out["gpus"] = ["%s (compute %s)" % (r[1], r[2]) for r in rows]
        out["gpu_uuids"] = [r[3] for r in rows]
    return out


def device_host_change(saved, now):
    """A device/host change: the arch differs, or the GPU UUID list differs (the GPU name list where a side records no
    UUIDs), or the host id differs; a manifest without a host id cannot prove a host change (counted as none)."""
    su, nu = saved.get("gpu_uuids"), now.get("gpu_uuids")
    gpus = su != nu if su is not None and nu is not None else saved.get("gpus") != now.get("gpus")
    sh, nh = saved.get("host_id"), now.get("host_id")
    host = None if sh is None or nh is None else sh != nh
    arch = saved.get("arch") != now.get("arch")
    return dict(arch=bool(arch), gpus=bool(gpus), host=host, changed=bool(arch or gpus or host))


def device_changed(saved, now):
    return device_host_change(saved, now)["changed"]


def manifest_roots(manifest):
    """The Taichi repo dir and the hypre prefix of a manifest: recorded in 'device', or for older manifests recognised
    from the recorded version-check file (Taichi's default <home>/.cache/taichi layout) and hypre build header."""
    device = manifest.get("device") or {}
    paths = list((manifest.get("inputs") or {}).get("paths") or {})
    repo, prefix = device.get("taichi_repo_dir"), device.get("hypre_prefix")
    if repo is None:
        found = [os.path.dirname(p) for p in paths
                 if os.path.basename(p) == TAICHI_VERSION_FILE and p.split("/")[-3:-1] == [".cache", "taichi"]]
        repo = found[0] if len(found) == 1 else None
    if prefix is None:
        found = [p[:-len("/" + HYPRE_HEADER)] for p in paths if p.endswith("/" + HYPRE_HEADER)]
        prefix = found[0] if len(found) == 1 else None
    return (os.path.normpath(repo) if repo else None), (os.path.normpath(prefix) if prefix else None)


def split_inputs(manifest):
    """(recorded, consistent, dropped [path, sha], environment [path, sha], physics shas) of a manifest's input
    record.  A manifest without inputs/sha256 records no compared inputs (its inputs/paths, if any, are logged only)."""
    inputs = manifest.get("inputs") or {}
    if "sha256" not in inputs:
        return False, True, [], [], []
    paths = dict(inputs.get("paths") or {})
    consistent = sorted(inputs.get("sha256") or []) == sorted(set(paths.values()))
    repo, prefix = manifest_roots(manifest)
    roots = tuple(r + os.sep for r in (repo, prefix) if r)
    version_file = os.path.join(repo, TAICHI_VERSION_FILE) if repo else None
    dropped, environment, physics = [], [], []
    for p, sha in sorted(paths.items()):
        norm = os.path.normpath(p)
        if version_file is not None and norm == version_file:
            dropped.append([p, sha])
        elif roots and norm.startswith(roots):
            environment.append([p, sha])
        else:
            physics.append(sha)
    return True, consistent, dropped, environment, sorted(physics)


def classify_restart(saved, current):
    """Every difference between a checkpoint's manifest and this run's, sorted into dropped / allowed / overridable /
    refused (module docstring), with the device/host change and one log line per difference."""
    change = device_host_change(saved.get("device") or {}, current.get("device") or {})
    out = dict(change=change, dropped=[], allowed=[], overridable=[], refused=[], log=[], policy=RESTART_POLICY)
    device_only = out["allowed"] if change["changed"] else out["overridable"]
    s_rec, s_ok, s_drop, s_env, s_phys = split_inputs(saved)
    n_rec, n_ok, n_drop, n_env, n_phys = split_inputs(current)
    out["dropped"] = [("checkpoint", p, sha) for p, sha in s_drop] + [("this run", p, sha) for p, sha in n_drop]
    if s_rec != n_rec:
        out["refused"].append(("inputs/sha256 (recorded on one side only)", s_rec, n_rec))
    if not (s_ok and n_ok):
        out["refused"].append(("inputs/sha256 (inconsistent with inputs/paths)", s_ok, n_ok))
    if s_phys != n_phys:
        out["refused"].append(("inputs/sha256 (physics inputs)", s_phys, n_phys))
    if s_env != n_env:
        device_only.append(("inputs/environment_files", s_env, n_env))
    s_lib = (saved.get("device") or {}).get("libHYPRE_sha256")
    n_lib = (current.get("device") or {}).get("libHYPRE_sha256")
    if s_lib != n_lib:
        device_only.append(("device/libHYPRE_sha256", s_lib, n_lib))
    for d in compare_manifests(saved, current, ignore=LOGGED_ONLY + ("inputs/sha256",)):
        key = d[0]
        if key.startswith("code/"):
            if d[2] == ABSENT and d[1] != ABSENT:        # loaded in the checkpoint's run only: is its source here the same?
                if change["changed"] and module_source_sha256(key[len("code/"):]) == d[1]:
                    out["allowed"].append(tuple(d) + (SAME_SOURCE_NOTE,))
                else:
                    out["overridable"].append(d)
            else:                                         # loaded in this run only, or on both sides with another sha256
                (device_only if d[1] == ABSENT else out["overridable"]).append(d)
        elif key == "libraries/python":
            minor = [tuple(str(v).split(".")[:2]) for v in (d[1], d[2])]
            (out["refused"] if minor[0] != minor[1] else device_only).append(d)
        elif key.startswith("libraries/hypre/flags"):
            device_only.append(d)
        elif key == "libraries" or key.startswith("libraries/"):
            out["overridable"].append(d)
        else:
            out["refused"].append(d)
    short = lambda d: json.dumps([str(v) for v in d])[:600]
    out["log"].append("policy %s; device/host change %s" % (RESTART_POLICY, json.dumps(change)))
    out["log"] += ["dropped (Taichi version-check file): %s" % short(d) for d in out["dropped"]]
    out["log"] += ["allowed with the device/host change: %s" % short(d) for d in out["allowed"]]
    out["log"] += ["code/library difference (diagnostic restart only): %s" % short(d) for d in out["overridable"]]
    out["log"] += ["REFUSED: %s" % short(d) for d in out["refused"]]
    return out


def grid_manifest(grid_nodes, faces):
    """Grid nodes and the sha256 of the cell-face coordinates (f64)."""
    data = b"".join(np.ascontiguousarray(np.asarray(f, np.float64)).tobytes() for f in faces)
    return dict(nodes=[int(v) for v in grid_nodes], faces_sha256=hashlib.sha256(data).hexdigest())


def restart_policy(saved, current, *, diagnostic, output_name):
    """The restart decision (classify_restart).  Raises ValueError when a difference is refused, when code / library
    differences remain without `diagnostic`, or when a diagnostic continuation writes into a directory whose name lacks
    DIAGNOSTIC.  Returns the classification; its 'overridable' differences are the ones the diagnostic flag accepted."""
    policy = classify_restart(saved, current)
    blocking = policy["refused"] + ([] if diagnostic else policy["overridable"])
    if blocking:
        raise ValueError("restart refused: the checkpoint manifest differs: %s"
                         % json.dumps([list(map(str, d)) for d in blocking][:40]))
    if policy["overridable"] and DIAGNOSTIC_MARK not in str(output_name):
        raise ValueError("a diagnostic restart writes into a directory whose name contains %s" % DIAGNOSTIC_MARK)
    return policy


def check_restart(saved, current, *, diagnostic, output_name):
    """restart_policy's decision; returns the code / library differences a diagnostic restart accepted."""
    return restart_policy(saved, current, diagnostic=diagnostic, output_name=output_name)["overridable"]


def restart_record(checkpoint_path, loop_state, saved_manifest, current_manifest, header, code_diffs, policy=None):
    """The content of restart.json (fastfsi.io.run_segments reads step, t_s, t_text and diagnostic); with the policy of
    restart_policy also its allowed / dropped differences and the device/host change."""
    checkpoint = Path(checkpoint_path).resolve()
    record = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                  step=int(loop_state["step"]), t_s=float(loop_state["t"]), t_text="%.9g" % loop_state["t"],
                  source_output=str(checkpoint.parent.parent), diagnostic=bool(code_diffs),
                  code_differences=[list(map(str, d)) for d in code_diffs],
                  device_before=saved_manifest.get("device"), device_now=current_manifest.get("device"),
                  device_changed=device_changed(saved_manifest.get("device", {}), current_manifest.get("device", {})),
                  checkpoint_created_utc=header.get("created_utc"),
                  restarted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if policy is not None:
        record.update(allowed_differences=[list(map(str, d)) for d in policy["allowed"]],
                      dropped_inputs=[list(map(str, d)) for d in policy["dropped"]],
                      device_host_change=policy["change"], restart_policy=policy["policy"])
    return record


def checkpoint_bytes(components):
    """Bytes of one checkpoint's arrays (from the components' host copies, one array at a time)."""
    total = 0
    for component in components.values():
        names = component.state_array_names() if hasattr(component, "state_array_names") else ()
        for name in names:
            total += int(np.asarray(component.state_array(name)).nbytes)
    return total


class CheckpointPlan:
    """Checkpoints at the first history row at or after every multiple of every_s and at or after every time in at_s."""

    def __init__(self, out, every_s, at_s, components, manifest, start_t_s, label="FASTFSI"):
        every = float(every_s)
        if not (math.isfinite(every) and every >= 0.0):
            raise ValueError("the checkpoint interval must be finite and >= 0")
        if any(not (math.isfinite(float(t)) and float(t) > 0.0) for t in at_s):
            raise ValueError("checkpoint times must be finite and positive")
        self.dir = Path(out) / "checkpoints"
        self.every = every
        self.components, self.manifest, self.label = components, manifest, label
        self.next_every = self._next_multiple(start_t_s) if self.every > 0.0 else math.inf
        self.pending = sorted(float(t) for t in at_s if float(t) > start_t_s + 1.0e-12)
        self.written = []

    def _next_multiple(self, t_s):
        return (math.floor(t_s / self.every + 1.0e-9) + 1) * self.every

    def planned_count(self, end_s, start_s):
        every = 0 if self.every <= 0.0 else max(0, int(math.floor(end_s / self.every + 1.0e-9))
                                                - int(math.floor(start_s / self.every + 1.0e-9)))
        return every + len([t for t in self.pending if t <= end_s + 1.0e-12])

    def due(self, t_s):
        return t_s >= self.next_every - 1.0e-12 or bool(self.pending and t_s >= self.pending[0] - 1.0e-12)

    def write(self, step, t_s, loop_state):
        self.dir.mkdir(exist_ok=True)
        path = self.dir / ("checkpoint_s%09d.npz" % step)
        started = time.perf_counter()
        header = write_checkpoint(path, self.components, loop_state, self.manifest)
        seconds = time.perf_counter() - started
        size = path.stat().st_size
        row = dict(step=int(step), t_s=float(t_s), file=str(path), bytes=size, write_s=seconds,
                   file_sha256=file_sha256(path), arrays=len(header["arrays"]), unix_s=time.time())
        with open(self.dir / "checkpoints.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        self.written.append(row)
        if self.every > 0.0:
            self.next_every = self._next_multiple(t_s)
        self.pending = [t for t in self.pending if t > t_s + 1.0e-12]
        print("%s checkpoint step %d t %.6f -> %s (%.1f MB, %.2f s)"
              % (self.label, step, t_s, path.name, size / 1.0e6, seconds), flush=True)
        return row
