"""CUDA volume report gather; original erosion and acceptance stay unchanged."""
import json
from types import MethodType
import taichi as ti
from simulation_core.diagnostics.device_scalars import read_scalar_fields
from .consistent_volume_refresh import policy_identity

def retain_unchanged_pockets(self):
    self._s28_refresh_boundary_contracts()
    self._s28_candidates(self.s28_accepted_support, self.s28_accepted_kind, True, True)
    candidates = int(self.s28_count[None])
    # Monotone erosion: each productive sweep removes >=1 candidate.
    # At the fixed point no retained component touches any surviving fluid.
    for _ in range(candidates + 1):
        self._s28_prune_open_components()
        if int(self.s28_count[None]) == 0:
            break
    else:
        raise RuntimeError("S28 numerical-pocket certificate did not converge")
    self._s28_retain_and_count()
    snapshot = read_scalar_fields(self, (self.s28_count, self.s28_volume, self.report_hibm_fresh_fluid_cells))
    count, volume = int(snapshot[self.s28_count]), float(snapshot[self.s28_volume])
    report = dict(revision=policy_identity()["revision"], stage=self._s28_transaction,
        exclusion_version=self._s28_exclusion_version,
        boundary_provenance=self._s28_boundary_provenance(),
        candidate_cells=candidates, retained_cells=count, retained_excluded_volume_m3=volume,
        accounting_scope="retained_numerical_exclusions_only; native physical transitions excluded",
        native_fresh_cells_after_retention=int(snapshot[self.report_hibm_fresh_fluid_cells]),
        created_fluid_mass_kg=0.0, created_fluid_momentum_kg_mps=[0.0, 0.0, 0.0])
    if count:
        report["witness"] = self._s28_witness(retained=True)
        print("S28_VOLUME_REFRESH " + json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    self.last_consistent_volume_refresh = report


def install_volume_device_control(owner):
    if ti.lang.impl.current_cfg().arch == ti.cuda:
        owner._s28_retain_unchanged_pockets = MethodType(retain_unchanged_pockets, owner)
