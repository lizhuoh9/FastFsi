"""The Cartesian (graded) fluid grid."""
import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class CartesianGrid:
    bounds_min_m: tuple[float, float, float]
    cell_widths_x_m: tuple[float, ...]
    cell_widths_y_m: tuple[float, ...]
    cell_widths_z_m: tuple[float, ...]

    @classmethod
    def uniform(
        cls,
        *,
        bounds_min_m: tuple[float, float, float],
        bounds_max_m: tuple[float, float, float],
        grid_nodes: tuple[int, int, int],
    ) -> "CartesianGrid":
        if len(grid_nodes) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value <= 0
            for value in grid_nodes
        ):
            raise ValueError(
                "grid_nodes must contain three positive non-boolean integers"
            )
        counts = tuple(int(value) for value in grid_nodes)
        widths = tuple(
            (hi - lo) / n
            for lo, hi, n in zip(bounds_min_m, bounds_max_m, counts, strict=True)
        )
        return cls(
            bounds_min_m=tuple(float(v) for v in bounds_min_m),
            cell_widths_x_m=(float(widths[0]),) * counts[0],
            cell_widths_y_m=(float(widths[1]),) * counts[1],
            cell_widths_z_m=(float(widths[2]),) * counts[2],
        )

    def __post_init__(self) -> None:
        bounds_min = tuple(float(v) for v in self.bounds_min_m)
        widths_x = tuple(float(v) for v in self.cell_widths_x_m)
        widths_y = tuple(float(v) for v in self.cell_widths_y_m)
        widths_z = tuple(float(v) for v in self.cell_widths_z_m)
        if len(bounds_min) != 3:
            raise ValueError("bounds_min_m must have three coordinates")
        if not widths_x or not widths_y or not widths_z:
            raise ValueError("CartesianGrid axes must contain at least one cell")
        if any(
            not math.isfinite(width) or width <= 0.0
            for width in widths_x + widths_y + widths_z
        ):
            raise ValueError(
                "CartesianGrid cell widths must be finite and positive"
            )
        if any(not math.isfinite(coordinate) for coordinate in bounds_min):
            raise ValueError("CartesianGrid bounds_min_m must be finite")
        object.__setattr__(self, "bounds_min_m", bounds_min)
        object.__setattr__(self, "cell_widths_x_m", widths_x)
        object.__setattr__(self, "cell_widths_y_m", widths_y)
        object.__setattr__(self, "cell_widths_z_m", widths_z)

    @staticmethod
    def _axis_centers(origin: float, widths: tuple[float, ...]) -> tuple[float, ...]:
        centers: list[float] = []
        face = float(origin)
        for width in widths:
            centers.append(face + 0.5 * width)
            face += width
        return tuple(centers)

    @staticmethod
    def _axis_faces(origin: float, widths: tuple[float, ...]) -> tuple[float, ...]:
        faces = [float(origin)]
        for width in widths:
            faces.append(faces[-1] + width)
        return tuple(faces)

    @staticmethod
    def _axis_center_distances(centers: tuple[float, ...], widths: tuple[float, ...]) -> tuple[float, ...]:
        distances = [float(widths[0])]
        for index in range(1, len(centers)):
            distances.append(float(centers[index] - centers[index - 1]))
        return tuple(distances)

    @staticmethod
    def _axis_grid_coordinate(value: float, faces: tuple[float, ...], centers: tuple[float, ...]) -> float:
        coordinate = 0.0
        if value <= centers[0]:
            half_width = max(centers[0] - faces[0], 1.0e-18)
            return -0.5 * (centers[0] - value) / half_width
        if value >= centers[-1]:
            half_width = max(faces[-1] - centers[-1], 1.0e-18)
            return float(len(centers) - 1) + 0.5 * (value - centers[-1]) / half_width
        for index in range(len(centers) - 1):
            left = centers[index]
            right = centers[index + 1]
            if left <= value <= right:
                coordinate = float(index) + (value - left) / max(right - left, 1.0e-18)
                break
        return coordinate

    @property
    def grid_nodes(self) -> tuple[int, int, int]:
        return (len(self.cell_widths_x_m), len(self.cell_widths_y_m), len(self.cell_widths_z_m))

    @property
    def bounds_max_m(self) -> tuple[float, float, float]:
        return (
            self.bounds_min_m[0] + sum(self.cell_widths_x_m),
            self.bounds_min_m[1] + sum(self.cell_widths_y_m),
            self.bounds_min_m[2] + sum(self.cell_widths_z_m),
        )

    @property
    def cell_centers_x_m(self) -> tuple[float, ...]:
        return self._axis_centers(self.bounds_min_m[0], self.cell_widths_x_m)

    @property
    def cell_centers_y_m(self) -> tuple[float, ...]:
        return self._axis_centers(self.bounds_min_m[1], self.cell_widths_y_m)

    @property
    def cell_centers_z_m(self) -> tuple[float, ...]:
        return self._axis_centers(self.bounds_min_m[2], self.cell_widths_z_m)

    @property
    def cell_faces_x_m(self) -> tuple[float, ...]:
        return self._axis_faces(self.bounds_min_m[0], self.cell_widths_x_m)

    @property
    def cell_faces_y_m(self) -> tuple[float, ...]:
        return self._axis_faces(self.bounds_min_m[1], self.cell_widths_y_m)

    @property
    def cell_faces_z_m(self) -> tuple[float, ...]:
        return self._axis_faces(self.bounds_min_m[2], self.cell_widths_z_m)

    @property
    def center_distances_x_m(self) -> tuple[float, ...]:
        return self._axis_center_distances(self.cell_centers_x_m, self.cell_widths_x_m)

    @property
    def center_distances_y_m(self) -> tuple[float, ...]:
        return self._axis_center_distances(self.cell_centers_y_m, self.cell_widths_y_m)

    @property
    def center_distances_z_m(self) -> tuple[float, ...]:
        return self._axis_center_distances(self.cell_centers_z_m, self.cell_widths_z_m)

    @staticmethod
    def _axis_is_uniform(widths: tuple[float, ...]) -> bool:
        first = widths[0]
        return all(math.isclose(width, first, rel_tol=1.0e-12, abs_tol=0.0) for width in widths)

    @functools.cached_property
    def is_uniform(self) -> bool:
        # Cached: the widths are immutable and solvers query this per CG iteration.
        return (
            self._axis_is_uniform(self.cell_widths_x_m)
            and self._axis_is_uniform(self.cell_widths_y_m)
            and self._axis_is_uniform(self.cell_widths_z_m)
        )

    @property
    def uniform_spacing_m(self) -> tuple[float, float, float]:
        if not self.is_uniform:
            raise ValueError("uniform_spacing_m is only available for a uniform CartesianGrid")
        return (
            self.cell_widths_x_m[0],
            self.cell_widths_y_m[0],
            self.cell_widths_z_m[0],
        )

    def grid_coordinate_x(self, x_m: float) -> float:
        return self._axis_grid_coordinate(float(x_m), self.cell_faces_x_m, self.cell_centers_x_m)

@dataclass(frozen=True)
class RefinementRegion:
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    target_spacing_m: float | tuple[float, float, float]

    def __post_init__(self) -> None:
        bounds_min = tuple(float(v) for v in self.bounds_min_m)
        bounds_max = tuple(float(v) for v in self.bounds_max_m)
        if len(bounds_min) != 3 or len(bounds_max) != 3:
            raise ValueError("RefinementRegion bounds must have three coordinates")
        if any(not math.isfinite(value) for value in bounds_min + bounds_max):
            raise ValueError("RefinementRegion bounds must be finite")
        if any(hi <= lo for lo, hi in zip(bounds_min, bounds_max, strict=True)):
            raise ValueError("RefinementRegion bounds_max_m must be greater than bounds_min_m")
        spacing = self.target_spacing_m
        if isinstance(spacing, tuple):
            target = tuple(float(v) for v in spacing)
            if len(target) != 3:
                raise ValueError("target_spacing_m tuple must have three values")
        else:
            target = (float(spacing),) * 3
        if any(not math.isfinite(value) or value <= 0.0 for value in target):
            raise ValueError("target_spacing_m must be finite and positive")
        object.__setattr__(self, "bounds_min_m", bounds_min)
        object.__setattr__(self, "bounds_max_m", bounds_max)
        object.__setattr__(self, "target_spacing_m", target)


@dataclass(frozen=True)
class GradedGridSpecification:
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    farfield_spacing_m: float | tuple[float, float, float]
    max_growth_ratio: float
    refinement_regions: tuple[RefinementRegion, ...] = ()
    max_cells: int | None = None

    def __post_init__(self) -> None:
        bounds_min = tuple(float(v) for v in self.bounds_min_m)
        bounds_max = tuple(float(v) for v in self.bounds_max_m)
        if len(bounds_min) != 3 or len(bounds_max) != 3:
            raise ValueError("GradedGridSpecification bounds must have three coordinates")
        if any(not math.isfinite(value) for value in bounds_min + bounds_max):
            raise ValueError("GradedGridSpecification bounds must be finite")
        if any(
            not math.isfinite(hi - lo)
            for lo, hi in zip(bounds_min, bounds_max, strict=True)
        ):
            raise ValueError("GradedGridSpecification axis lengths must be finite")
        if any(hi <= lo for lo, hi in zip(bounds_min, bounds_max, strict=True)):
            raise ValueError("bounds_max_m must be greater than bounds_min_m")
        spacing = self.farfield_spacing_m
        if isinstance(spacing, tuple):
            farfield = tuple(float(v) for v in spacing)
            if len(farfield) != 3:
                raise ValueError("farfield_spacing_m tuple must have three values")
        else:
            farfield = (float(spacing),) * 3
        if any(not math.isfinite(value) or value <= 0.0 for value in farfield):
            raise ValueError("farfield_spacing_m must be finite and positive")
        growth_ratio = float(self.max_growth_ratio)
        if not math.isfinite(growth_ratio) or growth_ratio <= 1.0:
            raise ValueError("max_growth_ratio must be finite and greater than 1")
        try:
            max_cells = None if self.max_cells is None else int(self.max_cells)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("max_cells must be finite and positive when provided") from exc
        if max_cells is not None and max_cells <= 0:
            raise ValueError("max_cells must be finite and positive when provided")
        object.__setattr__(self, "bounds_min_m", bounds_min)
        object.__setattr__(self, "bounds_max_m", bounds_max)
        object.__setattr__(self, "farfield_spacing_m", farfield)
        object.__setattr__(self, "max_growth_ratio", growth_ratio)
        object.__setattr__(self, "refinement_regions", tuple(self.refinement_regions))
        object.__setattr__(self, "max_cells", max_cells)


class _GradedGridCellBudgetExceeded(ValueError):
    pass


def _axis_gap_cell_count(
    length: float,
    spacing: float,
    *,
    cell_limit: int | None = None,
    width_tolerance: float = 1.0e-10,
) -> int:
    if cell_limit is not None:
        # The count loop divides in floating point; keep this early bound
        # conservative at its rounding boundary (for example, spacing=1/3).
        largest_width = math.nextafter(spacing * (1.0 + width_tolerance), math.inf)
        if math.isfinite(largest_width):
            length_num, length_den = length.as_integer_ratio()
            width_num, width_den = largest_width.as_integer_ratio()
            if length_num * width_den > cell_limit * width_num * length_den:
                raise _GradedGridCellBudgetExceeded("graded axis exceeds its cell budget")
    cells = max(1, int(round(length / spacing)))
    while length / cells > spacing * (1.0 + width_tolerance):
        cells += 1
    if cell_limit is not None and cells > cell_limit:
        raise _GradedGridCellBudgetExceeded("graded axis exceeds its cell budget")
    return cells


def _fill_axis_gap(
    left: float,
    right: float,
    spacing: float,
    *,
    cell_limit: int | None = None,
) -> list[float]:
    length = right - left
    if length <= 0.0:
        return []
    cells = _axis_gap_cell_count(length, spacing, cell_limit=cell_limit)
    width = length / cells
    return [left + width * index for index in range(1, cells)]


_WIDTH_TOLERANCE = 1.0e-10


def _side_width_bounds(
    cells: int,
    *,
    inner_width: float,
    farfield_spacing: float,
    max_growth_ratio: float,
) -> tuple[list[float], list[float]]:
    lower: list[float] = []
    upper: list[float] = []
    lower_width = upper_width = float(inner_width)
    for _ in range(cells):
        lower_width /= max_growth_ratio
        upper_width = min(farfield_spacing, upper_width * max_growth_ratio)
        lower.append(lower_width)
        upper.append(upper_width)
    return lower, upper


def _bridge_width_bounds(
    cells: int,
    *,
    left_width: float,
    right_width: float,
    farfield_spacing: float,
    max_growth_ratio: float,
) -> tuple[list[float], list[float]]:
    left_lower, left_upper = _side_width_bounds(
        cells,
        inner_width=left_width,
        farfield_spacing=farfield_spacing,
        max_growth_ratio=max_growth_ratio,
    )
    right_lower, right_upper = _side_width_bounds(
        cells,
        inner_width=right_width,
        farfield_spacing=farfield_spacing,
        max_growth_ratio=max_growth_ratio,
    )
    lower = [
        max(left_lower[index], right_lower[cells - index - 1])
        for index in range(cells)
    ]
    upper = [
        min(left_upper[index], right_upper[cells - index - 1])
        for index in range(cells)
    ]
    return lower, upper


def _minimum_feasible_width_bounds(
    *,
    distance: float,
    max_cells: int,
    width_bounds: Callable[[int], tuple[list[float], list[float]]],
    cell_limit: int | None = None,
) -> tuple[list[float], list[float]]:
    search_limit = max_cells if cell_limit is None else min(max_cells, cell_limit)
    if search_limit <= 0:
        raise _GradedGridCellBudgetExceeded("graded transition exceeds its cell budget")

    def can_span_distance(cells: int) -> bool:
        lower, upper = width_bounds(cells)
        bounds_are_feasible = all(
            low <= high * (1.0 + _WIDTH_TOLERANCE)
            for low, high in zip(lower, upper, strict=True)
        )
        return bounds_are_feasible and (
            distance <= math.fsum(upper) * (1.0 + _WIDTH_TOLERANCE)
        )

    if not can_span_distance(search_limit):
        if search_limit < max_cells:
            raise _GradedGridCellBudgetExceeded("graded transition exceeds its cell budget")
        raise ValueError("graded transition cannot span the requested distance")
    # Adding cells only relaxes endpoint envelopes and increases span capacity.
    low = 1
    high = search_limit
    while low < high:
        middle = (low + high) // 2
        if can_span_distance(middle):
            high = middle
        else:
            low = middle + 1
    lower, upper = width_bounds(low)
    if distance < math.fsum(lower) * (1.0 - _WIDTH_TOLERANCE):
        raise ValueError("graded transition is shorter than its minimum width")
    return lower, upper


def _interpolate_width_bounds(
    *,
    distance: float,
    lower: list[float],
    upper: list[float],
) -> tuple[float, ...]:
    lower_sum = math.fsum(lower)
    upper_sum = math.fsum(upper)
    if upper_sum <= lower_sum:
        return tuple(lower)
    alpha = min(1.0, max(0.0, (distance - lower_sum) / (upper_sum - lower_sum)))
    return tuple(
        lower_width + alpha * (upper_width - lower_width)
        for lower_width, upper_width in zip(lower, upper, strict=True)
    )


def _graded_side_widths(
    *,
    distance: float,
    inner_width: float,
    farfield_spacing: float,
    max_growth_ratio: float,
    cell_limit: int | None = None,
) -> tuple[float, ...]:
    if distance <= 0.0:
        return ()
    smallest_anchor = max(min(inner_width, farfield_spacing), 1.0e-12)
    max_cells = max(64, int(distance / smallest_anchor) * 8 + 64)
    growth = float(max_growth_ratio)

    def width_bounds(cells: int) -> tuple[list[float], list[float]]:
        return _side_width_bounds(
            cells,
            inner_width=inner_width,
            farfield_spacing=farfield_spacing,
            max_growth_ratio=growth,
        )

    lower, upper = _minimum_feasible_width_bounds(
        distance=distance,
        max_cells=max_cells,
        width_bounds=width_bounds,
        cell_limit=cell_limit,
    )
    widths = _interpolate_width_bounds(
        distance=distance,
        lower=lower,
        upper=upper,
    )
    if not _axis_widths_respect_growth(
        widths,
        max_growth_ratio=max_growth_ratio,
        left_boundary_width=inner_width,
    ):
        raise ValueError("could not build graded side satisfying max_growth_ratio")
    return widths


@dataclass(frozen=True)
class _AxisRefinementSegment:
    left: float
    right: float
    fine_width: float
    fine_cells: int


def _axis_widths_respect_growth(
    widths: tuple[float, ...],
    *,
    max_growth_ratio: float,
    left_boundary_width: float | None = None,
    right_boundary_width: float | None = None,
) -> bool:
    values: list[float] = []
    if left_boundary_width is not None:
        values.append(float(left_boundary_width))
    values.extend(float(width) for width in widths)
    if right_boundary_width is not None:
        values.append(float(right_boundary_width))
    tolerance = 1.0 + 1.0e-10
    for left, right in zip(values, values[1:]):
        if left <= 0.0 or right <= 0.0:
            return False
        ratio = max(left / right, right / left)
        if ratio > float(max_growth_ratio) * tolerance:
            return False
    return True


def _graded_bridge_widths(
    *,
    distance: float,
    left_width: float,
    right_width: float,
    farfield_spacing: float,
    max_growth_ratio: float,
    cell_limit: int | None = None,
) -> tuple[float, ...]:
    if distance <= 0.0:
        return ()
    smallest_anchor = max(min(left_width, right_width, farfield_spacing), 1.0e-12)
    max_cells = max(64, int(distance / smallest_anchor) * 8 + 64)
    growth = float(max_growth_ratio)

    def width_bounds(cells: int) -> tuple[list[float], list[float]]:
        return _bridge_width_bounds(
            cells,
            left_width=left_width,
            right_width=right_width,
            farfield_spacing=farfield_spacing,
            max_growth_ratio=growth,
        )

    lower, upper = _minimum_feasible_width_bounds(
        distance=distance,
        max_cells=max_cells,
        width_bounds=width_bounds,
        cell_limit=cell_limit,
    )
    widths = _interpolate_width_bounds(
        distance=distance,
        lower=lower,
        upper=upper,
    )
    if not _axis_widths_respect_growth(
        widths,
        max_growth_ratio=max_growth_ratio,
        left_boundary_width=left_width,
        right_boundary_width=right_width,
    ):
        raise ValueError("could not build graded bridge satisfying max_growth_ratio")
    return widths


def _bridge_is_geometrically_feasible(
    *,
    distance: float,
    left_width: float,
    right_width: float,
    farfield_spacing: float,
    max_growth_ratio: float,
) -> bool:
    # A merge probe must distinguish an impossible bridge from an expensive one.
    # Sum its two endpoint envelopes as geometric series without cell arrays.
    growth = float(max_growth_ratio)
    log_growth = math.log(growth)

    def logarithmic_ratio(numerator: float, denominator: float) -> float:
        relative_difference = (numerator - denominator) / denominator
        if math.isfinite(relative_difference) and relative_difference > -1.0:
            return math.log1p(relative_difference)
        return math.log(numerator) - math.log(denominator)

    log_ratio = logarithmic_ratio(left_width, right_width) / log_growth
    tolerance = 1.0 + _WIDTH_TOLERANCE
    if max(left_width, right_width) / growth > farfield_spacing * tolerance:
        return False

    def lower_sum(width: float, cells: int) -> float:
        return width * (-math.expm1(-cells * log_growth) / (growth - 1.0))

    def upper_sum(width: float, cells: int) -> float:
        if cells == 0:
            return 0.0
        growing_cells = min(
            cells,
            max(0, int(logarithmic_ratio(farfield_spacing, width) / log_growth)),
        )
        if growing_cells == 0:
            return cells * farfield_spacing
        last_width = min(
            farfield_spacing, math.exp(math.log(width) + growing_cells * log_growth)
        )
        growing_sum = last_width * (
            -math.expm1(-growing_cells * log_growth) / ((growth - 1.0) / growth)
        )
        return growing_sum + (cells - growing_cells) * farfield_spacing

    def can_span(cells: int) -> bool:
        if abs(log_ratio) > cells + 1 + math.log(tolerance) / log_growth:
            return False
        split = min(cells, max(0, math.floor((cells + 1 - log_ratio) / 2.0)))
        capacity = upper_sum(left_width, split) + upper_sum(right_width, cells - split)
        return distance <= capacity * tolerance

    smallest_anchor = max(min(left_width, right_width, farfield_spacing), 1.0e-12)
    high = max(64, int(distance / smallest_anchor) * 8 + 64)
    if not can_span(high):
        return False
    low = 1
    while low < high:
        middle = (low + high) // 2
        if can_span(middle):
            high = middle
        else:
            low = middle + 1
    split = min(low, max(0, math.floor((low + 1 + log_ratio) / 2.0)))
    minimum_width = lower_sum(left_width, split) + lower_sum(right_width, low - split)
    return distance >= minimum_width * (1.0 - _WIDTH_TOLERANCE)


def _axis_refinement_segments(
    *,
    bounds_min: float,
    bounds_max: float,
    farfield_spacing: float,
    max_growth_ratio: float,
    intervals: tuple[tuple[float, float, float], ...],
    tolerance: float,
    cell_limit: int | None = None,
) -> tuple[_AxisRefinementSegment, ...]:
    clipped: list[tuple[float, float, float]] = []
    for interval_min, interval_max, target_spacing in intervals:
        left = max(bounds_min, float(interval_min))
        right = min(bounds_max, float(interval_max))
        target = float(target_spacing)
        if right <= bounds_min or left >= bounds_max or right <= left:
            continue
        min_side_width = target / float(max_growth_ratio)
        if left - bounds_min < min_side_width:
            left = bounds_min
        if bounds_max - right < min_side_width:
            right = bounds_max
        clipped.append((left, right, target))
    if not clipped:
        return ()

    clipped.sort(key=lambda item: (item[0], item[1]))
    merged: list[list[float]] = []
    for left, right, target in clipped:
        if not merged:
            merged.append([left, right, target])
            continue
        previous = merged[-1]
        gap = left - previous[1]
        merge_gap = min(previous[2], target) / float(max_growth_ratio)
        should_merge = left <= previous[1] + max(tolerance, merge_gap)
        if not should_merge and gap > 0.0:
            try:
                _graded_bridge_widths(
                    distance=gap,
                    left_width=previous[2],
                    right_width=target,
                    farfield_spacing=farfield_spacing,
                    max_growth_ratio=max_growth_ratio,
                    cell_limit=cell_limit,
                )
            except _GradedGridCellBudgetExceeded:
                if _bridge_is_geometrically_feasible(
                    distance=gap,
                    left_width=previous[2],
                    right_width=target,
                    farfield_spacing=farfield_spacing,
                    max_growth_ratio=max_growth_ratio,
                ):
                    raise
                should_merge = True
            except ValueError:
                should_merge = True
        if should_merge:
            previous[1] = max(previous[1], right)
            previous[2] = min(previous[2], target)
        else:
            merged.append([left, right, target])

    segments: list[_AxisRefinementSegment] = []
    remaining_cells = cell_limit
    for left, right, target in merged:
        fine_cells = _axis_gap_cell_count(
            right - left, target, cell_limit=remaining_cells, width_tolerance=0.0
        )
        if remaining_cells is not None:
            remaining_cells -= fine_cells
        fine_width = (right - left) / fine_cells
        segments.append(
            _AxisRefinementSegment(
                left=left,
                right=right,
                fine_width=fine_width,
                fine_cells=fine_cells,
            )
        )
    return tuple(segments)


def _build_graded_axis(
    *,
    bounds_min: float,
    bounds_max: float,
    farfield_spacing: float,
    max_growth_ratio: float,
    intervals: tuple[tuple[float, float, float], ...],
    cell_limit: int | None = None,
) -> tuple[float, ...]:
    tolerance = max(abs(bounds_max - bounds_min), 1.0) * 1.0e-12
    if not intervals:
        faces = [bounds_min, bounds_max]
        faces[1:1] = _fill_axis_gap(
            bounds_min, bounds_max, farfield_spacing, cell_limit=cell_limit
        )
        return tuple(faces[index + 1] - faces[index] for index in range(len(faces) - 1))

    segments = _axis_refinement_segments(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        farfield_spacing=farfield_spacing,
        max_growth_ratio=max_growth_ratio,
        intervals=intervals,
        tolerance=tolerance,
        cell_limit=cell_limit,
    )
    if not segments:
        faces = [bounds_min, bounds_max]
        faces[1:1] = _fill_axis_gap(
            bounds_min, bounds_max, farfield_spacing, cell_limit=cell_limit
        )
        return tuple(faces[index + 1] - faces[index] for index in range(len(faces) - 1))

    widths: list[float] = []
    # Reserve every refined block before allocating any transition cells.
    remaining_cells = None if cell_limit is None else cell_limit - sum(
        segment.fine_cells for segment in segments
    )
    first = segments[0]
    widths.extend(
        reversed(
            _graded_side_widths(
                distance=first.left - bounds_min,
                inner_width=first.fine_width,
                farfield_spacing=farfield_spacing,
                max_growth_ratio=max_growth_ratio,
                cell_limit=remaining_cells,
            )
        )
    )
    if remaining_cells is not None:
        remaining_cells -= len(widths)
    widths.extend([first.fine_width] * first.fine_cells)
    previous = first
    for segment in segments[1:]:
        previous_count = len(widths)
        widths.extend(
            _graded_bridge_widths(
                distance=segment.left - previous.right,
                left_width=previous.fine_width,
                right_width=segment.fine_width,
                farfield_spacing=farfield_spacing,
                max_growth_ratio=max_growth_ratio,
                cell_limit=remaining_cells,
            )
        )
        if remaining_cells is not None:
            remaining_cells -= len(widths) - previous_count
        widths.extend([segment.fine_width] * segment.fine_cells)
        previous = segment
    widths.extend(
        _graded_side_widths(
            distance=bounds_max - previous.right,
            inner_width=previous.fine_width,
            farfield_spacing=farfield_spacing,
            max_growth_ratio=max_growth_ratio,
            cell_limit=remaining_cells,
        )
    )
    return tuple(widths)


def build_graded_grid(spec: GradedGridSpecification) -> CartesianGrid:
    axis_intervals: list[list[tuple[float, float, float]]] = [[], [], []]
    for region in spec.refinement_regions:
        for axis in range(3):
            axis_intervals[axis].append(
                (
                    region.bounds_min_m[axis],
                    region.bounds_max_m[axis],
                    region.target_spacing_m[axis],
                )
            )
    widths: list[tuple[float, ...]] = []
    cell_count = 1
    try:
        minimum_counts = tuple(
            _axis_gap_cell_count(
                spec.bounds_max_m[axis] - spec.bounds_min_m[axis],
                spec.farfield_spacing_m[axis],
                cell_limit=spec.max_cells,
            ) if not axis_intervals[axis] else 1
            for axis in range(3)
        )
        minimum_count = math.prod(minimum_counts)
        if spec.max_cells is not None and minimum_count > spec.max_cells:
            raise ValueError(
                f"graded grid cell count {minimum_count} exceeds max_cells {spec.max_cells}"
            )
        for axis in range(3):
            axis_limit = None if spec.max_cells is None else spec.max_cells // (
                cell_count * math.prod(minimum_counts[axis + 1:])
            )
            axis_widths = _build_graded_axis(
                bounds_min=spec.bounds_min_m[axis],
                bounds_max=spec.bounds_max_m[axis],
                farfield_spacing=spec.farfield_spacing_m[axis],
                max_growth_ratio=float(spec.max_growth_ratio),
                intervals=tuple(axis_intervals[axis]),
                cell_limit=axis_limit,
            )
            widths.append(axis_widths)
            cell_count *= len(axis_widths)
    except _GradedGridCellBudgetExceeded as exc:
        raise ValueError(f"graded grid cell count exceeds max_cells {spec.max_cells}") from exc
    if spec.max_cells is not None and cell_count > int(spec.max_cells):
        raise ValueError(
            f"graded grid cell count {cell_count} exceeds max_cells {int(spec.max_cells)}"
        )
    return CartesianGrid(
        bounds_min_m=spec.bounds_min_m,
        cell_widths_x_m=widths[0],
        cell_widths_y_m=widths[1],
        cell_widths_z_m=widths[2],
    )
