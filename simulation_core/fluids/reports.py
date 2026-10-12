from dataclasses import dataclass


@dataclass(frozen=True)
class ForceSpreadingReport:
    surface_force_n: tuple[float, float, float]
    grid_force_n: tuple[float, float, float]
    action_reaction_relative_error: float
    active_grid_cells: int


@dataclass(frozen=True)
class FluidImpulseReport:
    grid_impulse_n_s: tuple[float, float, float]
    momentum_delta_n_s: tuple[float, float, float]
    impulse_relative_error: float
    active_velocity_cells: int


@dataclass(frozen=True)
class VelocityConstraintReport:
    active_cells: int
    max_delta_mps: float
    mean_delta_mps: float
    momentum_delta_n_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    primary_momentum_delta_n_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    secondary_momentum_delta_n_s: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class VelocityDirichletBoundaryReport:
    active_cells: int
    max_delta_mps: float
    mean_delta_mps: float
    momentum_delta_n_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
