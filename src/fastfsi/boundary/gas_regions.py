"""Air regions of a chamber box from a body's topology mask: which free cells are air connected to an inlet, which are
water, which are sealed air pockets, and their fractional volumes.  Host only (NumPy / SciPy).

LABELS.  free = not wall and not topology material (SolidOccupancy.topology_mask(), with its time hysteresis).  The free
cells form 6-connected components (faces only).  A component is
  INLET_GAS     when it holds an inlet cell (the lid-layer cells of the inlet disc);
  LIQUID   when it holds a cell of the box floor (layer 0);
  LEAK    when it holds both: air would reach the water -> hard stop gas_region_leak;
  SEALED  when it holds neither.
A sealed component takes its ORIGIN from the previous accepted classification, by the cells it shares with it:
previous air only -> an GAS POCKET (gas bookkeeping: boundary.gas_pockets); previous water only, or cells freed from
material only -> a LIQUID POCKET (trapped water, solved by the projection like any water; its volume is reported);
previous air AND previous water -> hard stop gas_pocket_origin_ambiguous.  On the first classification every
sealed component is a water pocket (the chamber starts with its air connected to the inlet).
FLOOR.  The box floor must stay free of material: the box is chosen below anything the body can reach; material there
is the hard stop pneumatic_chamber_floor_material.

FRACTIONAL VOLUMES (design note section 2).  A free cell holds (1 - c) V of free volume (c = its coverage); a material
cell's free volume (1 - c) V is shared equally among the distinct free components face-adjacent to it, and within one
component equally among its adjacent cells.  effective_volume_m3 is that per-cell field (zero outside the free cells):
the component sums change continuously while the coverage changes, except at real topology changes.

DIRICHLET CODES for the pressure chamber: 0 = no Dirichlet value (water, material, water pockets), 1 = air connected
to an inlet (the inlet pressure at the solve), 2 + n = air pocket n (its own gas pressure, a table from the gas model).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from fastfsi.errors import SimulationHardStop

HARD_STOP_CODES = {"gas_region_leak", "gas_pocket_origin_ambiguous", "pneumatic_chamber_floor_material"}
INLET_GAS, LIQUID, LEAK, SEALED = 0, 1, 2, 3
_SIX = ndimage.generate_binary_structure(3, 1)
_DIRECTIONS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _shifted(array, direction, fill):
    """out[c] = array[c + direction] (fill outside)."""
    out = np.full_like(array, fill)
    src = tuple(slice(max(d, 0), array.shape[a] + min(d, 0)) for a, d in enumerate(direction))
    dst = tuple(slice(max(-d, 0), array.shape[a] + min(-d, 0)) for a, d in enumerate(direction))
    out[dst] = array[src]
    return out


@dataclass(frozen=True)
class GasRegionClassification:
    """One classification: labels of the free cells (0 elsewhere), each label's class and origin, the masks and the
    effective (fractional) cell volumes."""

    labels: np.ndarray
    component_class: np.ndarray          # per label: INLET_GAS / LIQUID / SEALED (LEAK never survives)
    air_pocket: np.ndarray               # per label: True for an air pocket
    air_mask: np.ndarray                 # INLET_GAS components and air pockets
    water_mask: np.ndarray               # LIQUID components and water pockets
    effective_volume_m3: np.ndarray
    time_s: float
    material: np.ndarray                 # the topology mask it was built from (for update_volumes)

    def diagnostics(self):
        volumes = np.bincount(self.labels.ravel(), weights=self.effective_volume_m3.ravel(),
                              minlength=len(self.component_class))
        sealed = np.flatnonzero(self.component_class == SEALED)
        sealed = sealed[sealed > 0]
        air_pockets = [int(c) for c in sealed if self.air_pocket[c]]
        water_pockets = [int(c) for c in sealed if not self.air_pocket[c]]
        return dict(time_s=self.time_s, components=int(len(self.component_class) - 1),
                    air_volume_m3=float(volumes[self.component_class == INLET_GAS].sum()),
                    air_pockets=len(air_pockets), air_pocket_volume_m3=float(volumes[air_pockets].sum()),
                    water_pockets=len(water_pockets), water_pocket_volume_m3=float(volumes[water_pockets].sum()),
                    inlet_connected=bool((self.component_class[1:] == INLET_GAS).any()))


class GasRegions:
    """Classifier of a chamber box (see the module docstring).  classify() is pure; accept() stores the accepted
    classification as the origin reference of the next one (and is what a checkpoint saves)."""

    def __init__(self, cell_volumes_m3, wall, inlet_cells):
        volumes = np.array(cell_volumes_m3, dtype=np.float64, copy=True)
        wall = np.array(wall, dtype=bool, copy=True)
        inlet = np.array(inlet_cells, dtype=bool, copy=True)
        if volumes.ndim != 3 or not (np.isfinite(volumes).all() and (volumes > 0.0).all()):
            raise ValueError("cell volumes must be a positive finite 3-D array")
        if wall.shape != volumes.shape or inlet.shape != volumes.shape:
            raise ValueError("wall and inlet masks must have the box shape")
        if not inlet.any() or (inlet & wall).any() or inlet[:, :, :-1].any():
            raise ValueError("inlet cells must be non-wall cells of the top layer, at least one")
        self.cell_volumes_m3 = volumes
        self.wall = wall
        self.inlet_cells = inlet
        self.previous_air = None
        self.previous_water = None

    def classify(self, topology_material, coverage, *, t_s):
        """Classify one accepted geometry (no state change).  Raises SimulationHardStop for a leak, an ambiguous pocket
        origin or material on the box floor."""
        material = np.asarray(topology_material, dtype=bool)
        cover = np.asarray(coverage, dtype=np.float64)
        if material.shape != self.wall.shape or cover.shape != self.wall.shape:
            raise ValueError("topology mask and coverage must have the box shape")
        floor = material[:, :, 0] & ~self.wall[:, :, 0]
        if floor.any():
            i, j = np.argwhere(floor)[0]
            raise SimulationHardStop("pneumatic_chamber_floor_material", dict(time_s=float(t_s), cells=int(floor.sum()),
                                                                             first_cell=[int(i), int(j), 0]))
        free = ~self.wall & ~material
        labels, count = ndimage.label(free, structure=_SIX)
        has_inlet = np.zeros(count + 1, dtype=bool)
        has_floor = np.zeros(count + 1, dtype=bool)
        has_inlet[np.unique(labels[self.inlet_cells])] = True
        has_floor[np.unique(labels[:, :, 0][free[:, :, 0]])] = True
        has_inlet[0] = has_floor[0] = False
        leak = np.flatnonzero(has_inlet & has_floor)
        if leak.size:
            cells = np.argwhere(np.isin(labels, leak))
            raise SimulationHardStop("gas_region_leak", dict(
                time_s=float(t_s), components=leak.tolist(), cells=int(len(cells)),
                box_index_min=cells.min(axis=0).tolist(), box_index_max=cells.max(axis=0).tolist(),
                inlet_cells_free=int((self.inlet_cells & free).sum())))
        component = np.full(count + 1, SEALED, dtype=np.int8)
        component[has_inlet] = INLET_GAS
        component[has_floor] = LIQUID
        component[0] = -1
        air_pocket = np.zeros(count + 1, dtype=bool)
        sealed = np.flatnonzero(component == SEALED)
        if sealed.size and self.previous_air is not None:
            index = labels.ravel()
            over_air = np.bincount(index, weights=self.previous_air.ravel(), minlength=count + 1) > 0
            over_water = np.bincount(index, weights=self.previous_water.ravel(), minlength=count + 1) > 0
            ambiguous = sealed[over_air[sealed] & over_water[sealed]]
            if ambiguous.size:
                cells = np.argwhere(labels == ambiguous[0])
                raise SimulationHardStop("gas_pocket_origin_ambiguous", dict(
                    time_s=float(t_s), components=ambiguous.tolist(), cells=int(len(cells)),
                    box_index_min=cells.min(axis=0).tolist(), box_index_max=cells.max(axis=0).tolist()))
            air_pocket[sealed] = over_air[sealed]
        air_mask = np.isin(labels, np.flatnonzero((component == INLET_GAS) | air_pocket))
        water_mask = free & ~air_mask
        effective = self._effective_volumes(labels, free, material, cover)
        return GasRegionClassification(labels=labels.astype(np.int32), component_class=component, air_pocket=air_pocket,
                                 air_mask=air_mask, water_mask=water_mask, effective_volume_m3=effective,
                                 time_s=float(t_s), material=material.copy())

    def update_volumes(self, classification, coverage, *, t_s):
        """The same topology (labels, classes, masks) with the effective volumes of a new coverage: the path for an
        update whose topology mask did not change (SolidOccupancy flipped == 0); no labelling, no hard-stop checks
        (they are topology properties and were made when the classification was built)."""
        cover = np.asarray(coverage, dtype=np.float64)
        if cover.shape != self.wall.shape:
            raise ValueError("coverage must have the box shape")
        free = classification.labels > 0
        effective = self._effective_volumes(classification.labels, free, classification.material, cover)
        return GasRegionClassification(labels=classification.labels, component_class=classification.component_class,
                                 air_pocket=classification.air_pocket, air_mask=classification.air_mask,
                                 water_mask=classification.water_mask, effective_volume_m3=effective,
                                 time_s=float(t_s), material=classification.material)

    def _effective_volumes(self, labels, free, material, cover):
        effective = np.where(free, (1.0 - cover) * self.cell_volumes_m3, 0.0)
        shared = material & ~self.wall & ndimage.binary_dilation(free, structure=_SIX)
        if not shared.any():
            return effective
        neighbours = np.stack([_shifted(labels, d, 0) for d in _DIRECTIONS])          # (6, nx, ny, nz)
        at = np.nonzero(shared)
        around = neighbours[:, at[0], at[1], at[2]]                                    # (6, m)
        ordered = np.sort(around, axis=0)
        distinct = ((ordered[1:] != ordered[:-1]) & (ordered[1:] > 0)).sum(axis=0) + (ordered[0] > 0)
        free_volume = (1.0 - cover[at]) * self.cell_volumes_m3[at]
        share = np.where(distinct > 0, free_volume / np.maximum(distinct, 1), 0.0)
        same = (around[:, None, :] == around[None, :, :]).sum(axis=1)                 # copies of each label among 6
        for d, direction in enumerate(_DIRECTIONS):
            label = around[d]
            use = label > 0
            target = tuple(a[use] + direction[axis] for axis, a in enumerate(at))
            np.add.at(effective, target, share[use] / same[d][use])
        return effective

    def accept(self, classification):
        """Make an accepted classification the origin reference of the next classify()."""
        self.previous_air = np.array(classification.air_mask, dtype=bool, copy=True)
        self.previous_water = np.array(classification.water_mask, dtype=bool, copy=True)

    def dirichlet_codes(self, classification, gas_labels):
        """Per-cell codes for the pressure chamber: 1 = inlet-connected air, 2 + n = cells of the gas model's sealed
        label n (n numbering the gas labels: gas_labels is the gas model's label field over the box), 0 otherwise."""
        codes = np.zeros(self.wall.shape, dtype=np.int32)
        connected = classification.air_mask & np.isin(classification.labels,
                                                      np.flatnonzero(classification.component_class == INLET_GAS))
        codes[connected] = 1
        pockets = classification.air_mask & ~connected
        codes[pockets] = 2 + np.asarray(gas_labels)[pockets]
        return codes

    # checkpoint protocol (small component)
    def state_arrays(self):
        if self.previous_air is None:
            return {"previous_air": np.zeros(self.wall.shape, np.uint8), "previous_water": np.zeros(self.wall.shape, np.uint8),
                    "initialized": np.asarray(0, np.uint8)}
        return {"previous_air": self.previous_air.astype(np.uint8), "previous_water": self.previous_water.astype(np.uint8),
                "initialized": np.asarray(1, np.uint8)}

    def load_state_arrays(self, arrays):
        if set(arrays) != {"previous_air", "previous_water", "initialized"}:
            raise ValueError("air region state arrays do not match the state inventory")
        air, water, initialized = (np.asarray(arrays[k]) for k in ("previous_air", "previous_water", "initialized"))
        if air.shape != self.wall.shape or water.shape != self.wall.shape or initialized.shape != ():
            raise ValueError("air region state arrays have the wrong shape")
        if int(initialized) not in (0, 1) or ((air != 0) & (water != 0)).any():
            raise ValueError("air region state arrays are inconsistent")
        if int(initialized) == 0:
            self.previous_air = self.previous_water = None
        else:
            self.previous_air, self.previous_water = air != 0, water != 0
