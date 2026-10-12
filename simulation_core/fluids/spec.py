import math
from dataclasses import dataclass
from numbers import Integral

from simulation_core.fluids.grid import CartesianGrid, GradedGridSpec, build_graded_grid


class _GradedCartesianGrid(CartesianGrid):
    """Materialized grid tagged so dataclasses.replace preserves provenance."""


def _materialize_graded_grid(spec: GradedGridSpec) -> _GradedCartesianGrid:
    grid = build_graded_grid(spec)
    return _GradedCartesianGrid(
        bounds_min_m=grid.bounds_min_m,
        cell_widths_x_m=grid.cell_widths_x_m,
        cell_widths_y_m=grid.cell_widths_y_m,
        cell_widths_z_m=grid.cell_widths_z_m,
    )


def _finite_triplet(value: object, *, name: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain three finite values")
    return values


def _finite_scalar(value: object, *, name: str, allow_zero: bool) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    invalid_sign = numeric < 0.0 if allow_zero else numeric <= 0.0
    if not math.isfinite(numeric) or invalid_sign:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return numeric


def _validated_grid_nodes(value: object) -> tuple[int, int, int]:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError("grid_nodes must contain three integers") from exc
    if len(values) != 3 or any(
        isinstance(item, bool) or not isinstance(item, Integral) for item in values
    ):
        raise ValueError("grid_nodes must contain three integers")
    nodes = tuple(int(item) for item in values)
    if any(node < 4 for node in nodes):
        raise ValueError("grid_nodes must be at least 4 in every dimension")
    return nodes


@dataclass(frozen=True)
class FluidDomainSpec:
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    grid_nodes: tuple[int, int, int] | None
    density_kgm3: float
    viscosity_pa_s: float
    dt_s: float
    cartesian_grid: CartesianGrid | None = None
    graded_grid: GradedGridSpec | None = None

    @classmethod
    def unit_box(
        cls,
        grid_nodes: tuple[int, int, int] = (32, 32, 32),
        density_kgm3: float = 1000.0,
        viscosity_pa_s: float = 1.0e-3,
        dt_s: float = 1.0e-3,
    ):
        return cls(
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=grid_nodes,
            density_kgm3=density_kgm3,
            viscosity_pa_s=viscosity_pa_s,
            dt_s=dt_s,
        )

    def __post_init__(self) -> None:
        bounds_min = _finite_triplet(self.bounds_min_m, name="bounds_min_m")
        bounds_max = _finite_triplet(self.bounds_max_m, name="bounds_max_m")
        density = _finite_scalar(
            self.density_kgm3,
            name="density_kgm3",
            allow_zero=False,
        )
        viscosity = _finite_scalar(
            self.viscosity_pa_s,
            name="viscosity_pa_s",
            allow_zero=True,
        )
        dt_s = _finite_scalar(self.dt_s, name="dt_s", allow_zero=False)
        object.__setattr__(self, "bounds_min_m", bounds_min)
        object.__setattr__(self, "bounds_max_m", bounds_max)
        object.__setattr__(self, "density_kgm3", density)
        object.__setattr__(self, "viscosity_pa_s", viscosity)
        object.__setattr__(self, "dt_s", dt_s)
        if any(hi <= lo for lo, hi in zip(bounds_min, bounds_max, strict=True)):
            raise ValueError("bounds_max_m must be greater than bounds_min_m")
        specified_grid_nodes = (
            None
            if self.grid_nodes is None
            else _validated_grid_nodes(self.grid_nodes)
        )
        if self.graded_grid is not None:
            grid = _materialize_graded_grid(self.graded_grid)
            # dataclasses.replace() copies the materialized public
            # cartesian_grid from an existing graded spec. Accept exactly
            # that derived grid, but reject a genuinely distinct pair.
            if self.cartesian_grid is not None and (
                not isinstance(self.cartesian_grid, _GradedCartesianGrid)
                or self.cartesian_grid != grid
            ):
                raise ValueError("cartesian_grid and graded_grid are mutually exclusive")
        elif self.cartesian_grid is not None:
            grid = self.cartesian_grid
        else:
            if specified_grid_nodes is None:
                raise ValueError("grid_nodes is required when no cartesian_grid or graded_grid is provided")
            grid = CartesianGrid.uniform(
                bounds_min_m=bounds_min,
                bounds_max_m=bounds_max,
                grid_nodes=specified_grid_nodes,
            )

        grid_nodes = (
            grid.grid_nodes
            if specified_grid_nodes is None
            else specified_grid_nodes
        )
        if any(n < 4 for n in grid_nodes):
            raise ValueError("grid_nodes must be at least 4 in every dimension")
        if grid.grid_nodes != grid_nodes:
            raise ValueError("cartesian_grid grid_nodes must match FluidDomainSpec.grid_nodes")
        for actual, expected in zip(grid.bounds_min_m, self.bounds_min_m, strict=True):
            if abs(actual - expected) > max(abs(expected), 1.0) * 1.0e-12:
                raise ValueError("cartesian_grid bounds_min_m must match FluidDomainSpec.bounds_min_m")
        for actual, expected in zip(grid.bounds_max_m, self.bounds_max_m, strict=True):
            if abs(actual - expected) > max(abs(expected), 1.0) * 1.0e-12:
                raise ValueError("cartesian_grid bounds_max_m must match FluidDomainSpec.bounds_max_m")
        object.__setattr__(self, "grid_nodes", grid_nodes)
        object.__setattr__(self, "cartesian_grid", grid)

    @property
    def spacing_m(self) -> tuple[float, float, float]:
        return self.cartesian_grid.uniform_spacing_m

    @property
    def cell_volume_m3(self) -> float:
        dx, dy, dz = self.spacing_m
        return dx * dy * dz
