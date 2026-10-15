"""fastfsi.io.restart: the restart policy (classify_restart / restart_policy).  A checkpoint written on one machine or
device continues on another only for differences a device/host change explains (modules loaded on one side only, the
Python patch version, hypre's build flags and binary, files under Taichi's repo dir or the hypre prefix); physics inputs
and the set-up never differ; code / library differences otherwise need a diagnostic restart; Taichi's version-check
file is never an input.  Pure Python (no solver)."""
import copy
import sys

import pytest

from fastfsi.io import restart
from fastfsi.io.restart import (classify_restart, device_host_change, restart_policy, restart_record,
                                split_inputs)

REAL_MODULE_SOURCE_SHA256 = restart.module_source_sha256
GPU_ONLY = "fastfsi.fluid.device_kernels"          # the synthetic manifests' module loaded in the CUDA run only
REPO = "/opt/user/.cache/taichi"
HYPRE = "/opt/hypre-cuda"


def manifest(arch="cuda", **changes):
    """A two-input manifest in the drivers' format (inputs/sha256 = the set of inputs/paths values)."""
    paths = {"/srv/case/config.json": "c" * 64, "/srv/case/waveform.csv": "w" * 64,
             HYPRE + "/include/HYPRE_config.h": "h" * 64, REPO + "/version_info": "v" * 64}
    base = dict(config={"preset": "fsi3"}, grid=dict(nodes=[4, 24, 144], faces_sha256="f"), coupler="c",
                code={"fastfsi": "1", "fastfsi.fluid.solver": "2", "fastfsi.fluid.device_kernels": "3"},
                libraries=dict(python="3.10.21", numpy="2.1.2", taichi="1.7.4",
                               hypre=dict(version="3.2.0", flags=["HYPRE_USING_CUDA"])),
                device=dict(arch=arch, gpus=["GPU A (compute 8.6)"], gpu_uuids=["GPU-a"], libHYPRE_sha256="lib-a",
                            hypre_prefix=HYPRE, taichi_repo_dir=REPO, host_id="machine-id:aa"),
                inputs=dict(paths=paths, sha256=sorted(set(paths.values()))))
    for key, value in changes.items():
        base[key] = value
    return base


@pytest.fixture(autouse=True)
def gpu_only_module_has_the_same_source_here(monkeypatch):
    """The synthetic CUDA run's GPU-only module (sha256 "3") has the same source file in this environment."""
    monkeypatch.setattr(restart, "module_source_sha256", lambda name: "3" if name == GPU_ONLY else None)


def real_module_file(tmp_path, monkeypatch, name, text):
    """A real module file `name`.py on sys.path (never imported) and the real source lookup; its sha256."""
    path = tmp_path / ("%s.py" % name)
    path.write_text(text)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(restart, "module_source_sha256", REAL_MODULE_SOURCE_SHA256)
    return restart.file_sha256(path)


def cpu_run_of(saved):
    """The same run restarted on this machine's CPU: GPU-only module not loaded, other Python patch and hypre build."""
    out = copy.deepcopy(saved)
    out["device"] = dict(arch="cpu", gpus=None, gpu_uuids=None, libHYPRE_sha256="lib-cpu", hypre_prefix="/opt/hypre-cpu",
                         taichi_repo_dir="/opt/other/.cache/taichi", host_id="machine-id:bb")
    out["code"].pop(GPU_ONLY, None)
    out["libraries"]["python"] = "3.10.12"
    out["libraries"]["hypre"]["flags"] = ["HYPRE_USING_HOST_MEMORY"]
    paths = {k: v for k, v in out["inputs"]["paths"].items() if not k.startswith((HYPRE, REPO))}
    paths["/opt/hypre-cpu/include/HYPRE_config.h"] = "k" * 64
    out["inputs"] = dict(paths=paths, sha256=sorted(set(paths.values())))
    return out


def set_input(m, path, sha):
    out = copy.deepcopy(m)
    out["inputs"]["paths"][path] = sha
    out["inputs"]["sha256"] = sorted(set(out["inputs"]["paths"].values()))
    return out


def keys(items):
    return [d[0] for d in items]


def test_a_gpu_checkpoint_continues_on_a_cpu_with_the_device_change_listed():
    saved = manifest()
    policy = restart_policy(saved, cpu_run_of(saved), diagnostic=False, output_name="run_2")
    assert policy["change"]["changed"] and policy["refused"] == [] and policy["overridable"] == []
    allowed = keys(policy["allowed"])
    assert "code/fastfsi.fluid.device_kernels" in allowed and "libraries/python" in allowed
    assert "inputs/environment_files" in allowed and "device/libHYPRE_sha256" in allowed
    assert any(k.startswith("libraries/hypre/flags") for k in allowed)
    assert [(side, p) for side, p, _ in policy["dropped"]] == [("checkpoint", REPO + "/version_info")]


def test_a_physics_input_difference_is_always_refused():
    saved = manifest()
    other = set_input(cpu_run_of(saved), "/srv/case/waveform.csv", "x" * 64)
    with pytest.raises(ValueError, match="restart refused"):
        restart_policy(saved, other, diagnostic=True, output_name="run_DIAGNOSTIC")


def test_device_explained_differences_need_a_diagnostic_restart_without_a_device_change():
    saved = manifest()
    same_device = set_input(saved, HYPRE + "/include/HYPRE_config.h", "z" * 64)
    same_device["libraries"]["python"] = "3.10.22"
    with pytest.raises(ValueError, match="restart refused"):
        restart_policy(saved, same_device, diagnostic=False, output_name="run_2")
    policy = restart_policy(saved, same_device, diagnostic=True, output_name="run_2_DIAGNOSTIC")
    assert keys(policy["overridable"]) == ["inputs/environment_files", "libraries/python"]


def test_a_python_minor_difference_and_a_changed_set_up_are_refused_even_for_a_diagnostic_restart():
    saved = manifest()
    newer = cpu_run_of(saved)
    newer["libraries"]["python"] = "3.11.4"
    with pytest.raises(ValueError, match="restart refused"):
        restart_policy(saved, newer, diagnostic=True, output_name="run_DIAGNOSTIC")
    with pytest.raises(ValueError, match="restart refused"):
        restart_policy(saved, manifest(coupler="other"), diagnostic=True, output_name="run_DIAGNOSTIC")


def test_a_module_with_another_sha256_needs_a_diagnostic_restart_even_with_a_device_change():
    saved = manifest()
    other = cpu_run_of(saved)
    other["code"]["fastfsi.fluid.solver"] = "9"
    policy = classify_restart(saved, other)
    assert keys(policy["overridable"]) == ["code/fastfsi.fluid.solver"] and policy["refused"] == []


def test_library_versions_get_no_device_allowance():
    saved = manifest()
    other = cpu_run_of(saved)
    other["libraries"]["numpy"] = "2.2.0"
    assert keys(classify_restart(saved, other)["overridable"]) == ["libraries/numpy"]


def test_the_device_host_change_rules():
    base = manifest()["device"]
    assert device_host_change(base, dict(base))["changed"] is False
    assert device_host_change(base, dict(base, gpu_uuids=["GPU-b"]))["gpus"] is True
    assert device_host_change(base, dict(base, host_id="machine-id:cc")) == dict(arch=False, gpus=False, host=True,
                                                                                changed=True)
    old = {k: v for k, v in base.items() if k != "host_id"}
    assert device_host_change(old, dict(base, host_id="machine-id:cc"))["host"] is None
    names_only = dict(arch="cuda", gpus=["GPU A"], gpu_uuids=None)
    assert device_host_change(names_only, dict(names_only, gpus=["GPU B"]))["gpus"] is True


def test_environment_files_differ_only_with_a_host_change_and_not_on_an_unproven_host():
    saved = manifest()
    other_host = set_input(manifest(), HYPRE + "/include/HYPRE_config.h", "z" * 64)
    other_host["device"]["host_id"] = "machine-id:cc"
    assert keys(classify_restart(saved, other_host)["allowed"]) == ["inputs/environment_files"]
    unproven = copy.deepcopy(other_host)
    old_saved = copy.deepcopy(saved)
    del old_saved["device"]["host_id"]
    assert keys(classify_restart(old_saved, unproven)["overridable"]) == ["inputs/environment_files"]


def test_the_version_check_file_is_dropped_and_older_manifests_are_recognised():
    saved = manifest()
    other = set_input(saved, REPO + "/version_info", "u" * 64)
    policy = classify_restart(saved, other)
    assert policy["allowed"] == policy["overridable"] == policy["refused"] == []
    assert [side for side, _, _ in policy["dropped"]] == ["checkpoint", "this run"]
    older = copy.deepcopy(saved)                     # no recorded roots: recognised from the recorded files
    for key in ("taichi_repo_dir", "hypre_prefix"):
        del older["device"][key]
    recorded, consistent, dropped, environment, physics = split_inputs(older)
    assert recorded and consistent and [p for p, _ in dropped] == [REPO + "/version_info"]
    assert [p for p, _ in environment] == [HYPRE + "/include/HYPRE_config.h"] and len(physics) == 2


def test_inconsistent_or_one_sided_input_records_are_refused():
    saved = manifest()
    broken = copy.deepcopy(saved)
    broken["inputs"]["sha256"] = broken["inputs"]["sha256"] + ["e" * 64]
    assert "inputs/sha256 (inconsistent with inputs/paths)" in keys(classify_restart(saved, broken)["refused"])
    unrecorded = copy.deepcopy(saved)
    del unrecorded["inputs"]
    assert "inputs/sha256 (recorded on one side only)" in keys(classify_restart(saved, unrecorded)["refused"])


def test_the_restart_record_lists_the_policy(tmp_path):
    checkpoint = tmp_path / "out" / "checkpoints" / "checkpoint_s000000010.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"data")
    saved = manifest()
    current = cpu_run_of(saved)
    policy = restart_policy(saved, current, diagnostic=False, output_name="run_2")
    record = restart_record(checkpoint, dict(t=0.5, step=10), saved, current, dict(created_utc="x"), [], policy=policy)
    assert record["restart_policy"] == restart.RESTART_POLICY and record["device_changed"] is True
    assert record["device_host_change"]["arch"] is True and record["diagnostic"] is False
    assert len(record["allowed_differences"]) == len(policy["allowed"]) and len(record["dropped_inputs"]) == 1


def test_the_version_check_file_is_recognised_through_taichi(tmp_path, monkeypatch):
    monkeypatch.setattr(restart, "taichi_repo_dir", lambda: str(tmp_path))
    assert restart.is_taichi_version_file(str(tmp_path / "version_info"))
    assert not restart.is_taichi_version_file(str(tmp_path / "other"))
    assert not restart.is_taichi_version_file(str(tmp_path / "sub" / "version_info"))


def test_a_module_loaded_only_in_the_checkpoint_run_is_allowed_when_its_source_here_is_the_same(tmp_path, monkeypatch):
    saved = manifest()
    del saved["code"][GPU_ONLY]
    saved["code"]["gpu_only_kernels_same"] = real_module_file(tmp_path, monkeypatch, "gpu_only_kernels_same", "X = 1\n")
    current = cpu_run_of(saved)
    del current["code"]["gpu_only_kernels_same"]
    current["code"]["cpu_only_helper"] = "a" * 64          # loaded in this run only: allowed with the change, as before
    policy = classify_restart(saved, current)
    assert policy["overridable"] == [] and policy["refused"] == []
    entry = [d for d in policy["allowed"] if d[0] == "code/gpu_only_kernels_same"]
    assert len(entry) == 1 and entry[0][-1] == "not loaded here; same source"
    assert "code/cpu_only_helper" in keys(policy["allowed"])
    assert "gpu_only_kernels_same" not in sys.modules     # found, never imported
    same_device = copy.deepcopy(saved)                     # without a device/host change: diagnostic only, as before
    del same_device["code"]["gpu_only_kernels_same"]
    assert keys(classify_restart(saved, same_device)["overridable"]) == ["code/gpu_only_kernels_same"]


def test_a_module_loaded_only_in_the_checkpoint_run_needs_a_diagnostic_restart_when_its_source_here_differs(
        tmp_path, monkeypatch):
    saved = manifest()
    del saved["code"][GPU_ONLY]
    real_module_file(tmp_path, monkeypatch, "gpu_only_kernels_other", "X = 2\n")
    saved["code"]["gpu_only_kernels_other"] = "f" * 64     # the checkpoint run loaded another source
    saved["code"]["gpu_only_kernels_missing"] = "e" * 64   # and one that does not exist here
    current = cpu_run_of(saved)
    for name in ("gpu_only_kernels_other", "gpu_only_kernels_missing"):
        del current["code"][name]
    policy = classify_restart(saved, current)
    assert policy["change"]["changed"] and policy["refused"] == []
    assert keys(policy["overridable"]) == ["code/gpu_only_kernels_missing", "code/gpu_only_kernels_other"]
    with pytest.raises(ValueError, match="restart refused"):
        restart_policy(saved, current, diagnostic=False, output_name="run_2")
    assert restart_policy(saved, current, diagnostic=True, output_name="run_2_DIAGNOSTIC")["overridable"]


def test_the_source_lookup_imports_nothing(tmp_path, monkeypatch):
    package = tmp_path / "gpu_pkg_not_imported"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('imported')\n")
    (package / "kernels.py").write_text("X = 3\n")
    sha = real_module_file(tmp_path, monkeypatch, "gpu_top_level", "X = 4\n")
    assert restart.module_source_sha256("gpu_top_level") == sha
    assert restart.module_source_sha256("gpu_pkg_not_imported.kernels") is None      # its parent is not imported
    assert restart.module_source_sha256("no_such_module_here") is None
    assert "gpu_pkg_not_imported" not in sys.modules and "gpu_top_level" not in sys.modules
    assert restart.ABSENT is __import__("fastfsi.io.checkpoint", fromlist=["ABSENT"]).ABSENT


def test_the_host_id_comes_from_the_system():
    value = restart.host_id()
    assert value.startswith("machine-id:") or value.startswith("hostname:")
