from __future__ import annotations

from simulation_core.diagnostics.runtime import TaichiRuntimeConfig, init_taichi
from simulation_core.diagnostics.time_stepping import CflSubstepController
from simulation_core.diagnostics.validation import (
    BoundaryDriveComplianceReport,
    FieldDiagnostic,
    ReferenceCurve,
    boundary_drive_compliance_report,
    checks_passed,
    finite_field_diagnostics,
    force_nonzero_when_loaded,
    vector_norm,
)

__all__ = [
    "BoundaryDriveComplianceReport",
    "CflSubstepController",
    "FieldDiagnostic",
    "ReferenceCurve",
    "TaichiRuntimeConfig",
    "boundary_drive_compliance_report",
    "checks_passed",
    "finite_field_diagnostics",
    "force_nonzero_when_loaded",
    "init_taichi",
    "vector_norm",
]
