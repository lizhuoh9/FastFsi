from __future__ import annotations

# K9 (REQ_334): the exactly equivalent fast path for Taichi 1.7.4's data_oriented attribute hook, installed
# before any of our classes is decorated (k9_fast_hook.py; HIBM_K9_FAST_HOOK=0 keeps Taichi's own hooks).
from . import k9_fast_hook as _k9_fast_hook
_k9_fast_hook.install()

from .fluids import (
    CG_PRECONDITIONER_CHOICES,
    CartesianFluidSolver,
    CartesianGrid,
    FluidDomainSpec,
    GradedGridSpec,
    RefinementRegion,
    build_graded_grid,
)

__all__ = [
    "CartesianFluidSolver",
    "CartesianGrid",
    "CG_PRECONDITIONER_CHOICES",
    "FluidDomainSpec",
    "GradedGridSpec",
    "RefinementRegion",
    "build_graded_grid",
]

__version__ = "0.2.0"
