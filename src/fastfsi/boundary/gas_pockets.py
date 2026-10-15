"""Host-only ideal pressure inlets and isothermal gas accounting on trusted air masks.

This module is opt-in: it neither imports the device runtime nor modifies a pressure
projection. An accepted component carries q = p_abs V, including while connected
so that closing its inlet has a tracked gas source. Mask overlap establishes simple
split/merge correspondence; untracked or ambiguous topology is a hard stop.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import ndimage

from fastfsi.errors import SimulationHardStop

P_ATMOSPHERE_PA = 101325.0               # standard atmosphere: the ambient absolute pressure of the gas law


HARD_STOP_CODES = {
    "gas_pocket_invalid_pressure", "gas_pocket_pressure_out_of_bounds",
    "gas_pocket_inlet_pressure_conflict", "gas_pocket_missing_initial_pressure",
    "gas_pocket_untracked_component", "gas_pocket_zero_volume",
    "gas_pocket_ambiguous_topology", "gas_pocket_invalid_gas_amount",
}
_FACE_NEIGHBOURS = ndimage.generate_binary_structure(3, 1)
_STATE_NAMES = {"labels", "gas_amount_pa_m3", "time_s", "initialized"}


def _json_witness(value):
    """Preserve non-finite diagnostic values as explicit strict-JSON strings."""
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("+inf" if value > 0.0 else "-inf")
    if isinstance(value, dict):
        return {key: _json_witness(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_witness(item) for item in value]
    return value


def _hard_stop(code, witness):
    return SimulationHardStop(code, _json_witness(witness))


def _mask(value, name):
    array = np.asarray(value)
    if array.ndim != 3 or min(array.shape) == 0:
        raise ValueError("%s must be a non-empty three-dimensional mask" % name)
    if array.dtype.kind not in "biu" or not np.all((array == 0) | (array == 1)):
        raise ValueError("%s must contain boolean or integer 0/1 values" % name)
    return np.array(array, dtype=bool, copy=True)


def _readonly_copy(value):
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def circular_wall_patch(cell_faces_m, *, wall_axis, wall_side, center_m, radius_m):
    """Boundary-cell mask of a circular wall patch, sampled at cell-face centres.

The three physical face arrays define a Cartesian, possibly graded, grid. The
normal coordinate of center_m must lie on the chosen min/max wall. No cell centre
inside the patch is a setup error; it must not masquerade as a covered inlet.
"""
    if wall_axis not in (0, 1, 2) or wall_side not in ("min", "max"):
        raise ValueError("the patch needs wall_axis 0/1/2 and wall_side min/max")
    if len(cell_faces_m) != 3:
        raise ValueError("the grid needs three cell-face arrays")
    faces = tuple(np.asarray(axis, dtype=np.float64) for axis in cell_faces_m)
    if any(axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
           or not np.all(np.diff(axis) > 0.0) for axis in faces):
        raise ValueError("cell faces must be finite and strictly increasing")
    centre = np.asarray(center_m, dtype=np.float64)
    radius = float(radius_m)
    if centre.shape != (3,) or not np.isfinite(centre).all() or not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("the patch needs a finite 3D centre and positive radius")
    face_index = 0 if wall_side == "min" else -1
    wall_position = float(faces[wall_axis][face_index])
    roundoff = 8 * np.finfo(np.float64).eps * max(1.0, abs(wall_position), abs(float(centre[wall_axis])))
    if abs(float(centre[wall_axis]) - wall_position) > roundoff:
        raise ValueError("the patch centre is not on the selected wall")
    tangential_axes = [axis for axis in range(3) if axis != wall_axis]
    offsets = [0.5 * faces[axis][:-1] + 0.5 * faces[axis][1:] - centre[axis] for axis in tangential_axes]
    on_wall = np.hypot(offsets[0][:, None], offsets[1][None, :]) <= radius
    cells = np.zeros(tuple(len(axis) - 1 for axis in faces), dtype=bool)
    selection = [slice(None)] * 3
    selection[wall_axis] = face_index
    cells[tuple(selection)] = on_wall
    if not cells.any():
        raise ValueError("the circular inlet patch misses every boundary cell centre")
    return cells


@dataclass(frozen=True)
class PressureInlet:
    """A caller-resolved wall-cell mask and a gauge-pressure value or source p(t)."""

    cells: np.ndarray
    pressure_source_pa: float | Callable[[float], float]

    def __post_init__(self):
        cells = _mask(self.cells, "inlet cells")
        if not cells.any():
            raise ValueError("an inlet patch must contain at least one wall cell")
        cells.setflags(write=False)
        object.__setattr__(self, "cells", cells)


@dataclass(frozen=True)
class GasPocketUpdate:
    """Copied output arrays; label zero is non-air and has no pressure assignment."""

    labels: np.ndarray
    gauge_pressure_pa: np.ndarray
    volume_m3: np.ndarray
    pressure_abs_pa: np.ndarray
    gas_amount_pa_m3: np.ndarray
    connected_to_inlet: np.ndarray
    inlet_covered: tuple[bool, ...]
    reservoir_exchange_pa_m3: float
    initial_sealed_gas_pa_m3: float
    ambient_pressure_pa: float

    @property
    def sealed_labels(self):
        return tuple(int(label) for label in np.flatnonzero(~self.connected_to_inlet[1:]) + 1)

    def diagnostics(self):
        sealed = self.sealed_labels
        return {
            "inlet_covered": list(self.inlet_covered), "pocket_count": len(sealed),
            "pockets": [
                {"label": label, "volume_m3": float(self.volume_m3[label]),
                 "pressure_abs_pa": float(self.pressure_abs_pa[label]),
                 "pressure_gauge_pa": float(self.pressure_abs_pa[label] - self.ambient_pressure_pa),
                 "gas_amount_pa_m3": float(self.gas_amount_pa_m3[label])} for label in sealed
            ],
            "total_sealed_gas_pa_m3": math.fsum(float(self.gas_amount_pa_m3[label]) for label in sealed),
            "total_gas_pa_m3": math.fsum(float(q) for q in self.gas_amount_pa_m3[1:]),
            "connected_volume_m3": math.fsum(float(self.volume_m3[label])
                                               for label in np.flatnonzero(self.connected_to_inlet)),
            "min_pocket_pressure_abs_pa": min((float(self.pressure_abs_pa[label]) for label in sealed), default=None),
            "max_pocket_pressure_abs_pa": max((float(self.pressure_abs_pa[label]) for label in sealed), default=None),
            "reservoir_exchange_pa_m3": self.reservoir_exchange_pa_m3,
            "initial_sealed_gas_pa_m3": self.initial_sealed_gas_pa_m3,
        }


class GasPocketModel:
    """Transactional host bookkeeping; update only at an accepted geometry state.

The cell grid is fixed. Every old component must retain at least one overlapping
new air cell. Removed cells compress that component, without removing gas. Direct
mask correspondence cannot resolve many-to-many mixing and therefore rejects it.
Use state_arrays/load_state_arrays for rollback; coupled stability is a separate
requirement when this component is eventually attached to a pressure projection.
"""

    def __init__(self, cell_volumes_m3, inlets: Sequence[PressureInlet], *, ambient_pressure_pa,
                 initial_pressure_gauge_pa=None, pressure_bounds_abs_pa=None, fractional_volumes=False,
                 seal_from_inlet_pressure=False):
        volumes = np.array(cell_volumes_m3, dtype=np.float64, copy=True)
        if volumes.ndim != 3 or min(volumes.shape) == 0 or not np.isfinite(volumes).all() or not (volumes > 0.0).all():
            raise ValueError("cell volumes must be a positive finite three-dimensional array")
        ambient = float(ambient_pressure_pa)
        if not math.isfinite(ambient) or ambient <= 0.0:
            raise ValueError("ambient_pressure_pa must be explicit, finite and positive")
        if pressure_bounds_abs_pa is not None:
            low, high = (float(value) for value in pressure_bounds_abs_pa)
            if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
                raise ValueError("absolute-pressure bounds must be positive finite min < max")
            pressure_bounds_abs_pa = (low, high)
        self._cell_volumes = volumes
        self.fractional_volumes = bool(fractional_volumes)
        self.seal_from_inlet_pressure = bool(seal_from_inlet_pressure)
        self._accepted_volumes = volumes.copy() if self.fractional_volumes else None
        self._inlets = tuple(PressureInlet(inlet.cells, inlet.pressure_source_pa) for inlet in inlets)
        if any(inlet.cells.shape != volumes.shape for inlet in self._inlets):
            raise ValueError("inlet cells and cell volumes must use the same grid")
        self.ambient_pressure_pa = ambient
        self.initial_pressure_gauge_pa = initial_pressure_gauge_pa
        self.pressure_bounds_abs_pa = pressure_bounds_abs_pa
        self._labels = np.zeros(volumes.shape, dtype=np.int32)
        self._gas_amounts = np.zeros(1, dtype=np.float64)
        self._time_s = 0.0
        self._initialized = False

    def _check_pressure(self, pressure_abs_pa, witness):
        pressure = float(pressure_abs_pa)
        witness = dict(witness, pressure_abs_pa=pressure)
        if not math.isfinite(pressure) or pressure <= 0.0:
            raise _hard_stop("gas_pocket_invalid_pressure", witness)
        if self.pressure_bounds_abs_pa is not None:
            low, high = self.pressure_bounds_abs_pa
            if not low <= pressure <= high:
                raise _hard_stop("gas_pocket_pressure_out_of_bounds", dict(witness, bounds_abs_pa=[low, high]))
        return pressure

    def _gas_sum(self, amounts, witness):
        try:
            total = math.fsum(float(value) for value in amounts)
        except OverflowError:
            total = math.inf
        if not math.isfinite(total) or total < 0.0:
            raise _hard_stop("gas_pocket_invalid_gas_amount", dict(witness, gas_amount_pa_m3=total))
        return total

    def update(self, air_mask, *, t_s, cell_volumes_m3=None):
        """Commit a fully checked mask transition and return copied pressures/diagnostics.

On any exception the prior accepted arrays stay intact. Positive reservoir
exchange adds gas; negative exchange releases gas. Previously open components
carry their last accepted gas amount into closure before reservoir resets.
"""
        time = float(t_s)
        if not math.isfinite(time) or (self._initialized and time < self._time_s):
            raise ValueError("accepted update time must be finite and nondecreasing")
        air = _mask(air_mask, "air mask")
        if air.shape != self._cell_volumes.shape:
            raise ValueError("air mask and cell volumes must use the same grid")
        if self.fractional_volumes:
            cell_volumes = np.asarray(cell_volumes_m3, dtype=np.float64) if cell_volumes_m3 is not None else None
            if cell_volumes is None or cell_volumes.shape != air.shape or not np.isfinite(cell_volumes).all() \
                    or (cell_volumes < 0.0).any():
                raise ValueError("fractional volumes need finite, nonnegative per-update cell volumes of the grid shape")
        elif cell_volumes_m3 is not None:
            raise ValueError("cell_volumes_m3 is only accepted with fractional_volumes=True")
        else:
            cell_volumes = self._cell_volumes
        same_mask = self._initialized and np.array_equal(air, self._labels > 0)
        if same_mask:              # identical air cells: the same canonical labels, every component its own child
            labels, count = self._labels.copy(), len(self._gas_amounts) - 1
        else:
            labels, count = ndimage.label(air, structure=_FACE_NEIGHBOURS)
            labels = labels.astype(np.int32, copy=False)
        volumes = np.bincount(labels[air], weights=cell_volumes[air], minlength=count + 1)
        if not np.isfinite(volumes).all() or (count and not (volumes[1:] > 0.0).all()):
            raise _hard_stop("gas_pocket_zero_volume", {"time_s": time, "volumes_m3": volumes.tolist()})
        connected = np.zeros(count + 1, dtype=bool)
        source_gauge = np.full(count + 1, np.nan, dtype=np.float64)
        covered = []
        for inlet_index, inlet in enumerate(self._inlets):
            inlet_labels = np.unique(labels[inlet.cells & air])
            covered.append(len(inlet_labels) == 0)
            if len(inlet_labels) == 0:
                continue
            source = inlet.pressure_source_pa
            gauge = float(source(time) if callable(source) else source)
            self._check_pressure(self.ambient_pressure_pa + gauge, {"time_s": time, "inlet": inlet_index})
            for label in inlet_labels:
                if connected[label] and source_gauge[label] != gauge:
                    raise _hard_stop("gas_pocket_inlet_pressure_conflict",
                                             {"time_s": time, "label": int(label), "inlet": inlet_index,
                                              "gauge_pressures_pa": [float(source_gauge[label]), gauge]})
                connected[label] = True
                source_gauge[label] = gauge
        allocated = np.zeros(count + 1, dtype=np.float64)
        initial_sealed_gas = 0.0
        if same_mask:
            allocated[1:] = self._gas_amounts[1:]
        elif self._initialized:
            children = [[] for _ in range(len(self._gas_amounts))]
            parents = [[] for _ in range(count + 1)]
            shared = (self._labels > 0) & air
            pairs, pair_cells = np.unique(np.stack((self._labels[shared], labels[shared]), axis=1), axis=0,
                                          return_counts=True)
            for old, new in pairs:
                children[int(old)].append(int(new))
                parents[int(new)].append(int(old))
            seal_rule = self.seal_from_inlet_pressure
            if seal_rule:
                received = {(int(o), int(n)): int(c) for (o, n), c in zip(pairs, pair_cells)}
                new_cells = np.bincount(labels[air], minlength=count + 1)
                old_connected = np.zeros(len(self._gas_amounts), dtype=bool)
                old_source = np.full(len(self._gas_amounts), np.nan, dtype=np.float64)
                for inlet in self._inlets:
                    for label in np.unique(self._labels[inlet.cells & (self._labels > 0)]):
                        source = inlet.pressure_source_pa
                        old_connected[label] = True
                        old_source[label] = float(source(time) if callable(source) else source)
                parents = [[o for o in plist if not old_connected[o]] for plist in parents]
            for old, gas in enumerate(self._gas_amounts[1:], start=1):
                if seal_rule and old_connected[old]:
                    for new in children[old]:
                        if not connected[new]:
                            allocated[new] += ((self.ambient_pressure_pa + old_source[old]) * volumes[new]
                                               * received[(old, new)] / new_cells[new])
                    continue
                if not children[old]:
                    raise _hard_stop("gas_pocket_zero_volume",
                                             {"time_s": time, "old_label": old, "volume_m3": 0.0,
                                              "gas_amount_pa_m3": float(gas), "reason": "no_air_overlap"})
                destinations = children[old]
                if len(destinations) > 1:
                    if any(len(parents[new]) > 1 for new in destinations):
                        raise _hard_stop("gas_pocket_ambiguous_topology",
                                                 {"time_s": time, "old_label": old,
                                                  "new_labels": destinations,
                                                  "new_parents": [parents[new] for new in destinations]})
                    weights = volumes[destinations]
                    if seal_rule:
                        weights = np.array([volumes[new] * received[(old, new)] / new_cells[new]
                                            for new in destinations])
                    total_volume = self._gas_sum(weights, {"time_s": time, "old_label": old})
                    shares = float(gas) * (weights / total_volume)
                    shares[-1] = float(gas) - self._gas_sum(shares[:-1], {"time_s": time, "old_label": old})
                    allocated[destinations] += shares
                else:
                    allocated[destinations[0]] += gas
            for label in range(1, count + 1):
                if not parents[label] and not connected[label] and not (
                        seal_rule and any(old_connected[o] for o, n in received if n == label)):
                    raise _hard_stop("gas_pocket_untracked_component", {"time_s": time, "label": label})
        else:
            sealed = np.flatnonzero(~connected[1:]) + 1
            if len(sealed):
                if self.initial_pressure_gauge_pa is None:
                    raise _hard_stop("gas_pocket_missing_initial_pressure",
                                             {"time_s": time, "labels": sealed.tolist()})
                initial_pressure = self._check_pressure(self.ambient_pressure_pa + float(self.initial_pressure_gauge_pa),
                                                        {"time_s": time, "reason": "initial_sealed_air"})
                with np.errstate(over="ignore", invalid="ignore"):
                    allocated[sealed] = initial_pressure * volumes[sealed]
                initial_sealed_gas = self._gas_sum(allocated[sealed], {"time_s": time})
        pressures = np.full(count + 1, np.nan, dtype=np.float64)
        inherited_connected = self._gas_sum(allocated[connected], {"time_s": time})
        for label in range(1, count + 1):
            if connected[label]:
                pressure = self.ambient_pressure_pa + source_gauge[label]
                with np.errstate(over="ignore", invalid="ignore"):
                    allocated[label] = pressure * volumes[label]
            else:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    pressure = allocated[label] / volumes[label]
            if not math.isfinite(float(allocated[label])) or allocated[label] <= 0.0:
                raise _hard_stop("gas_pocket_invalid_gas_amount",
                                         {"time_s": time, "label": label, "gas_amount_pa_m3": float(allocated[label])})
            pressures[label] = self._check_pressure(pressure, {"time_s": time, "label": label})
        self._gas_sum(allocated[1:], {"time_s": time})
        reservoir_exchange = self._gas_sum(allocated[connected], {"time_s": time}) - inherited_connected
        gauge_cells = pressures[labels] - self.ambient_pressure_pa
        result = GasPocketUpdate(
            *(_readonly_copy(array) for array in (labels, gauge_cells, volumes, pressures, allocated, connected)),
            tuple(covered), reservoir_exchange, initial_sealed_gas, self.ambient_pressure_pa,
        )
        self._labels = labels.copy()
        self._gas_amounts = allocated.copy()
        if self.fractional_volumes:
            self._accepted_volumes = np.array(cell_volumes, dtype=np.float64, copy=True)
        self._time_s = time
        self._initialized = True
        return result

    def state_arrays(self):
        """Independent copies of the carried state for checkpoint/rollback."""
        state = {
            "labels": self._labels.copy(), "gas_amount_pa_m3": self._gas_amounts.copy(),
            "time_s": np.asarray(self._time_s, dtype=np.float64),
            "initialized": np.asarray(self._initialized, dtype=np.uint8),
        }
        if self.fractional_volumes:
            state["cell_volumes_m3"] = self._accepted_volumes.copy()
        return state

    def load_state_arrays(self, arrays: Mapping[str, np.ndarray]):
        """Validate a complete accepted-state snapshot before replacing any state.

The caller's checkpoint manifest must also match grid, inlet masks, sources,
ambient pressure and pressure limits; this method validates the numerical state.
"""
        if set(arrays) != (_STATE_NAMES | {"cell_volumes_m3"} if self.fractional_volumes else _STATE_NAMES):
            raise ValueError("gas state arrays do not match the state inventory")
        copied = {name: np.array(value, copy=True) for name, value in arrays.items()}
        labels, gas = copied["labels"], copied["gas_amount_pa_m3"]
        time, initialized = copied["time_s"], copied["initialized"]
        if labels.dtype != np.int32 or labels.shape != self._cell_volumes.shape or (labels < 0).any():
            raise ValueError("gas component labels are invalid")
        if gas.dtype != np.float64 or gas.ndim != 1 or gas.size == 0 or gas[0] != 0.0 or not np.isfinite(gas).all():
            raise ValueError("gas amounts are invalid")
        if time.dtype != np.float64 or time.shape != () or not np.isfinite(time):
            raise ValueError("gas state time is invalid")
        if initialized.dtype != np.uint8 or initialized.shape != () or int(initialized) not in (0, 1):
            raise ValueError("gas state initialized marker is invalid")
        canonical, count = ndimage.label(labels > 0, structure=_FACE_NEIGHBOURS)
        if not np.array_equal(labels, canonical) or len(gas) != count + 1 or (gas[1:] <= 0.0).any():
            raise ValueError("gas labels and amounts are inconsistent")
        try:
            total_gas = math.fsum(float(value) for value in gas[1:])
        except OverflowError as error:
            raise ValueError("restored total gas amount is nonfinite") from error
        if not math.isfinite(total_gas):
            raise ValueError("restored total gas amount is nonfinite")
        if not bool(initialized) and (count or len(gas) != 1 or float(time) != 0.0):
            raise ValueError("uninitialized gas state must be empty")
        cell_volumes = self._cell_volumes
        if self.fractional_volumes:
            cell_volumes = copied["cell_volumes_m3"]
            if cell_volumes.dtype != np.float64 or cell_volumes.shape != self._cell_volumes.shape \
                    or not np.isfinite(cell_volumes).all() or (cell_volumes < 0.0).any():
                raise ValueError("gas state cell volumes are invalid")
        volumes = np.bincount(labels[labels > 0], weights=cell_volumes[labels > 0], minlength=count + 1)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pressures = gas[1:] / volumes[1:]
        if not np.isfinite(pressures).all() or (pressures <= 0.0).any():
            raise ValueError("restored gas pressures are invalid")
        if self.pressure_bounds_abs_pa is not None:
            low, high = self.pressure_bounds_abs_pa
            if not ((pressures >= low) & (pressures <= high)).all():
                raise ValueError("restored gas pressure is outside the configured bounds")
        self._labels = labels
        self._gas_amounts = gas
        if self.fractional_volumes:
            self._accepted_volumes = cell_volumes
        self._time_s = float(time)
        self._initialized = bool(initialized)
