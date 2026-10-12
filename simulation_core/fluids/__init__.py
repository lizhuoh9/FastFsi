from __future__ import annotations

from simulation_core.fluids.constants import (
    CG_PRECONDITIONER_CHOICES,
    HIBM_PRESSURE_COMPONENT_CAPACITY,
    HIBM_TINY_UNREACHED_COMPONENT_CLEANUP_THRESHOLD_CELLS,
    HIBM_UNREACHED_COMPONENT_SMALL_THRESHOLD_CELLS,
    PRESSURE_OUTLET_AUTO_CLEANUP_ACCEPTANCE_MARGIN,
    PRESSURE_OUTLET_AUTO_CLEANUP_MAX_PASSES,
    PRESSURE_OUTLET_AUTO_CLEANUP_MIN_L2,
    PRESSURE_OUTLET_AUTO_CLEANUP_NO_REGRESSION_MARGIN,
    PRESSURE_OUTLET_AUTO_CLEANUP_TARGET_REDUCTION,
)
from simulation_core.fluids.grid import (
    CartesianGrid,
    GradedGridSpec,
    RefinementRegion,
    build_graded_grid,
)
from simulation_core.fluids.pressure_outlet import (
    pressure_outlet_cleanup_iteration_budget,
)
from simulation_core.fluids.reports import (
    FluidImpulseReport,
    ForceSpreadingReport,
    VelocityConstraintReport,
    VelocityDirichletBoundaryReport,
)
from simulation_core.fluids.solver import CartesianFluidSolver
from simulation_core.fluids.spec import FluidDomainSpec

__all__ = [
    "CG_PRECONDITIONER_CHOICES",
    "CartesianFluidSolver",
    "CartesianGrid",
    "FluidDomainSpec",
    "FluidImpulseReport",
    "ForceSpreadingReport",
    "GradedGridSpec",
    "HIBM_PRESSURE_COMPONENT_CAPACITY",
    "HIBM_TINY_UNREACHED_COMPONENT_CLEANUP_THRESHOLD_CELLS",
    "HIBM_UNREACHED_COMPONENT_SMALL_THRESHOLD_CELLS",
    "PRESSURE_OUTLET_AUTO_CLEANUP_ACCEPTANCE_MARGIN",
    "PRESSURE_OUTLET_AUTO_CLEANUP_MAX_PASSES",
    "PRESSURE_OUTLET_AUTO_CLEANUP_MIN_L2",
    "PRESSURE_OUTLET_AUTO_CLEANUP_NO_REGRESSION_MARGIN",
    "PRESSURE_OUTLET_AUTO_CLEANUP_TARGET_REDUCTION",
    "RefinementRegion",
    "VelocityConstraintReport",
    "VelocityDirichletBoundaryReport",
    "build_graded_grid",
    "pressure_outlet_cleanup_iteration_budget",
]
