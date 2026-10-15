"""Checkpoint schedules and restart checks for drivers built on fastfsi.io.checkpoint.

CheckpointPlan(out, every_s, at_s, components, manifest, start_t_s, label)
    a checkpoint at the first history row at or after every multiple of every_s (0 = off) and at or after every time
    in at_s, written to out/checkpoints/checkpoint_s{step:09d}.npz (create-only) and listed in checkpoints.jsonl with
    its size, write time and sha256.  Checkpoints are never deleted.
restart_manifest(...) / the helpers below
    what a restart must match: the configuration (every argument a restart may not change), the grid, a fingerprint of
    the static setup, the source code of the loaded modules (sha256 per module), the library versions, and the device
    (logged, not compared: a run may continue on another device).
check_restart(saved, current, *, diagnostic, output_name)
    the restart decision: a configuration / grid / setup difference always refuses; a code or library difference
    refuses unless diagnostic is set, and then the output directory name must contain DIAGNOSTIC (such a continuation
    is never a result).  Returns the code / library differences.
"""
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from fastfsi.io.checkpoint import compare_manifests, write_checkpoint

CODE_MANIFEST_KEYS = ("code", "libraries")       # only a diagnostic restart may differ here
LOGGED_ONLY = ("device", "inputs/paths")          # recorded, never compared
DIAGNOSTIC_MARK = "DIAGNOSTIC"


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


def device_identity(arch):
    """The arch and, on CUDA, the visible GPUs (name and compute capability; their uuids logged) from nvidia-smi."""
    out = dict(arch=str(arch), host=os.uname().nodename if hasattr(os, "uname") else None,
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), gpus=None, gpu_uuids=None)
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


def device_changed(saved, now):
    return (saved.get("arch"), saved.get("gpus")) != (now.get("arch"), now.get("gpus"))


def grid_manifest(grid_nodes, faces):
    """Grid nodes and the sha256 of the cell-face coordinates (f64)."""
    data = b"".join(np.ascontiguousarray(np.asarray(f, np.float64)).tobytes() for f in faces)
    return dict(nodes=[int(v) for v in grid_nodes], faces_sha256=hashlib.sha256(data).hexdigest())


def check_restart(saved, current, *, diagnostic, output_name):
    """The code / library differences between two manifests (LOGGED_ONLY ignored).  Raises ValueError when any other
    key differs, when code / libraries differ without `diagnostic`, or when a diagnostic continuation writes into a
    directory whose name lacks DIAGNOSTIC."""
    diffs = compare_manifests(saved, current, ignore=LOGGED_ONLY)
    code_diffs = [d for d in diffs if d[0].split("/")[0] in CODE_MANIFEST_KEYS]
    other = [d for d in diffs if d not in code_diffs]
    if other or (code_diffs and not diagnostic):
        raise ValueError("restart refused: the checkpoint manifest differs: %s"
                         % json.dumps([list(map(str, d)) for d in diffs][:40]))
    if code_diffs and DIAGNOSTIC_MARK not in str(output_name):
        raise ValueError("a diagnostic restart writes into a directory whose name contains %s" % DIAGNOSTIC_MARK)
    return code_diffs


def restart_record(checkpoint_path, loop_state, saved_manifest, current_manifest, header, code_diffs):
    """The content of restart.json (fastfsi.io.run_segments reads step, t_s, t_text and diagnostic)."""
    checkpoint = Path(checkpoint_path).resolve()
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                step=int(loop_state["step"]), t_s=float(loop_state["t"]), t_text="%.9g" % loop_state["t"],
                source_output=str(checkpoint.parent.parent), diagnostic=bool(code_diffs),
                code_differences=[list(map(str, d)) for d in code_diffs],
                device_before=saved_manifest.get("device"), device_now=current_manifest.get("device"),
                device_changed=device_changed(saved_manifest.get("device", {}), current_manifest.get("device", {})),
                checkpoint_created_utc=header.get("created_utc"),
                restarted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())


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
