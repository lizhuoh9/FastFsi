"""Host rules for sealed air pockets when the topology of a chamber box changes: which sealed candidate cells form which
component, which components are air pockets, pending candidate regions or water pockets, and how the pockets' gas
amounts carry over.  Pure NumPy / SciPy on compact data (the candidate cells' linear indices, previous codes and
previous candidate-region numbers, the old pockets' and old regions' fate counts), so only small arrays cross between
the device and the host.

CODES (also the device classification of boundary/gas_regions_device.py):
    0 material or wall, 1 INLET_GAS (inlet-connected air, and pending candidate regions), 2 LIQUID (floor-connected),
    3 water pocket, 4 + k air pocket k.

COMPONENTS.  The candidate cells (free cells reached by neither flood) form 6-connected components.  Each component's
PARENTS are the previous codes of its cells.  A component with an air parent (INLET_GAS or an air pocket) and a water parent
(LIQUID or a water pocket) is the hard stop gas_pocket_origin_ambiguous; one without air parents is a water pocket
(code 3; freed material counts as no parent, unless the device's type memory reports the class of the cell's last
free code -- INLET_GAS, LIQUID or a water pocket -- as its previous code).  A component with an old air pocket among
its parents continues that
pocket lineage: it is an air pocket (new number, in the order of the components' smallest cell index, so the numbering
is deterministic).

SEALING RULE.  A component whose air parents are inlet air only (no old pocket) is a CANDIDATE REGION cut off from
the inlet.  It becomes a sealed air pocket only when it has persisted `seal_steps` consecutive occupancy updates (the
same region by cell overlap: its age is the largest age of the old candidate regions whose cells it holds, or 1 for
a new region; the caller ages every region by one per update) AND has at least `min_cells` cells.  Until then it
stays with its parent, the inlet air: its cells keep code 1 (Dirichlet at p_in) and carry a pending-region number.
A pending region that is covered by material, reconnected to the inlet air or merged into a sealed pocket before
sealing simply ENDS: no pocket ever existed, so there is no hard stop.  N_SEAL = 5 and M_MIN = 8 (2 x 2 x 2) are the
defaults; seal_steps = min_cells = 1 is the earlier rule (every candidate sealed at once).

GAS (q = p_abs V, Pa m^3, isothermal).  Old pocket k sends its gas to its children: the new air pockets holding cells
that were k, and INLET_GAS when some of its cells joined the inlet-connected air (that share returns to the reservoir).  The
shares follow the volume each child received from k: a new pocket c takes weight V_c x (cells of c that were k) /
(cells of c), the inlet air takes V_air x (cells of k now INLET_GAS) / (cells of INLET_GAS), with V the effective volumes.  A new pocket's gas is the sum of its old pockets'
shares plus, when part of it was INLET_GAS, (p_in + p_ambient) V_c x (cells of c that were INLET_GAS) / (cells of c): air that
seals off from the inlet starts at the inlet pressure of the seal time (it was connected to the pump until then; under
the sealing rule a candidate region is held at p_in until it seals, so the seal time is the update at which it seals).
Hard stops (strict, as boundary/gas_pockets.py; sealed pockets only, unchanged by the sealing rule): an old pocket
with no child (gas_pocket_zero_volume), an old pocket with several children of which one has several old-pocket
parents (gas_pocket_ambiguous_topology), cells of an old pocket that became LIQUID (gas_region_gas_to_liquid: air
reached the water).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from fastfsi.errors import SimulationHardStop

MATERIAL, INLET_GAS, LIQUID, LIQUID_POCKET, FIRST_POCKET = 0, 1, 2, 3, 4
FATE_MATERIAL, FATE_AIR, FATE_WATER, FATE_CANDIDATE = 0, 1, 2, 3
HARD_STOP_CODES = {"gas_pocket_origin_ambiguous", "gas_pocket_zero_volume", "gas_pocket_ambiguous_topology",
                   "gas_region_gas_to_liquid"}
N_SEAL = 5        # the sealing rule: consecutive occupancy updates a candidate region persists before it seals
M_MIN = 8         # the sealing rule: the fewest cells a candidate region seals with (2 x 2 x 2)
REGION_ENDS = ("covered", "reconnected", "merged")


def sealed_components(indices, shape):
    """6-connected components of the cells with the given sorted, distinct linear indices (C order) in `shape`.
    Returns one component number per cell (0 .. n-1), numbered by each component's smallest index."""
    idx = np.asarray(indices, dtype=np.int64)
    n = len(idx)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if np.any(np.diff(idx) <= 0):
        raise ValueError("candidate indices must be sorted and distinct")
    nx, ny, nz = shape
    k = idx % nz
    j = (idx // nz) % ny
    i = idx // (ny * nz)
    rows, cols = [], []
    for stride, ok in ((1, k < nz - 1), (nz, j < ny - 1), (ny * nz, i < nx - 1)):
        neighbour = idx + stride
        pos = np.searchsorted(idx, neighbour)
        hit = ok & (pos < n)
        hit[hit] &= idx[pos[hit]] == neighbour[hit]
        rows.append(np.flatnonzero(hit))
        cols.append(pos[hit])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    first = np.full(labels.max() + 1, n, dtype=np.int64)
    np.minimum.at(first, labels, np.arange(n))
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[labels]


@dataclass(frozen=True)
class PocketTransitionPlan:
    """The classification of the candidates and the gas transfer plan, before the new volumes are known."""

    candidate_codes: np.ndarray        # new code per candidate cell (1 pending, 3, or 4 + new pocket number)
    pockets: int                       # number of new air pockets
    old_to_new: dict                   # old pocket -> {new pocket: cells of the new pocket that were the old one}
    old_to_air_cells: np.ndarray       # per old pocket: its cells now in INLET_GAS
    air_cells_of_new: np.ndarray       # per new pocket: its cells that were INLET_GAS
    cells_of_new: np.ndarray           # per new pocket: its cell count
    candidate_regions: np.ndarray = None   # per candidate cell: its new pending-region number (1-based), 0 = none
    regions: tuple = ()                # per new pending region (number = position + 1): dict(age, cells, parents)
    sealed: tuple = ()                 # candidate regions sealed now: dict(pocket, age, cells, parents)
    ended: tuple = ()                  # old pending regions ended unsealed: dict(region, how, fate)


def continuing_pockets(plan):
    """Per new pocket: 1.0 when it CONTINUES an old pocket through this topology change -- exactly one old pocket is its
    parent, that parent has no other child and sent no cells to the inlet air, and none of its cells were inlet air
    (cells freed from material may join) -- else 0.0 (new, split or merged: exempt once from the step's stiffness
    check)."""
    flags = np.zeros(plan.pockets, dtype=np.float64)
    parents = {}
    for old, children in plan.old_to_new.items():
        for child in children:
            parents.setdefault(child, []).append(old)
    for child in range(plan.pockets):
        owners = parents.get(child, [])
        if len(owners) != 1 or plan.air_cells_of_new[child] != 0:
            continue
        old = owners[0]
        if set(plan.old_to_new[old]) == {child} and plan.old_to_air_cells[old] == 0:
            flags[child] = 1.0
    return flags


def _parents_by_component(component, region, count):
    """Per component: the sorted distinct nonzero old-region numbers of its cells."""
    out = [[] for _ in range(count)]
    held = region > 0
    if held.any():
        pairs = np.unique(np.stack((component[held], region[held])), axis=1)
        for c, r in pairs.T:
            out[int(c)].append(int(r))
    return out


def plan_pocket_transitions(indices, previous_codes, fate, old_pockets, shape, *, t_s, previous_regions=None,
                            region_ages=(), region_fate=None, seal_steps=1, min_cells=1):
    """Components, codes, the gas plan and the candidate regions of one topology change (see the module docstring).
    indices / previous_codes: the candidate cells (any order); fate: (old_pockets, 4) cell counts per old pocket
    (material, INLET_GAS, LIQUID, candidate); previous_regions: per candidate cell, its old pending-region number
    (0 = none); region_ages: per old pending region r (r = 1 .. R), its age INCLUDING this update; region_fate:
    (R, 4) cell counts per old region; seal_steps / min_cells: the sealing rule (1 / 1 = every candidate seals at
    once)."""
    seal_steps, min_cells = int(seal_steps), int(min_cells)
    if seal_steps < 1 or min_cells < 1:
        raise ValueError("seal_steps and min_cells must be >= 1")
    idx = np.asarray(indices, dtype=np.int64)
    prev = np.asarray(previous_codes, dtype=np.int64)
    if prev.shape != idx.shape:
        raise ValueError("one previous code per candidate cell")
    order = np.argsort(idx, kind="stable")
    idx, prev = idx[order], prev[order]
    ages = np.asarray(region_ages, dtype=np.int64).reshape(-1)
    old_regions = len(ages)
    if previous_regions is None:
        preg = np.zeros(len(idx), dtype=np.int64)
    else:
        preg = np.asarray(previous_regions, dtype=np.int64)
        if preg.shape != idx.shape:
            raise ValueError("one previous region number per candidate cell")
        preg = preg[order]
    if preg.size and (preg.min() < 0 or preg.max() > old_regions):
        raise ValueError("previous region numbers must lie in 0 .. %d" % old_regions)
    rfate = (np.zeros((old_regions, 4), dtype=np.int64) if region_fate is None
             else np.asarray(region_fate, dtype=np.int64).reshape(old_regions, 4))
    fate = np.asarray(fate, dtype=np.int64).reshape(old_pockets, 4)
    water_hit = np.flatnonzero(fate[:, FATE_WATER] > 0)
    if water_hit.size:
        raise SimulationHardStop("gas_region_gas_to_liquid", dict(time_s=float(t_s), old_pockets=water_hit.tolist(),
                                                                cells=fate[water_hit, FATE_WATER].tolist()))
    component = sealed_components(idx, shape)
    count = int(component.max()) + 1 if len(component) else 0
    air_parent = np.zeros(count, dtype=bool)
    water_parent = np.zeros(count, dtype=bool)
    pocket_parent = np.zeros(count, dtype=bool)
    np.logical_or.at(air_parent, component, (prev == INLET_GAS) | (prev >= FIRST_POCKET))
    np.logical_or.at(water_parent, component, (prev == LIQUID) | (prev == LIQUID_POCKET))
    np.logical_or.at(pocket_parent, component, prev >= FIRST_POCKET)
    ambiguous = np.flatnonzero(air_parent & water_parent)
    if ambiguous.size:
        cells = idx[component == ambiguous[0]]
        raise SimulationHardStop("gas_pocket_origin_ambiguous", dict(
            time_s=float(t_s), components=ambiguous.tolist(), cells=int(len(cells)),
            first_index=np.unravel_index(int(cells[0]), shape)))
    # the sealing rule: an inlet-gas-only component is a candidate region; it seals when old and large enough
    size = np.bincount(component, minlength=count)
    age = np.ones(count, dtype=np.int64)                   # a new region: this update is its first
    held = preg > 0
    if held.any():
        np.maximum.at(age, component[held], ages[preg[held] - 1])
    candidate = air_parent & ~pocket_parent
    seal = candidate & (size >= min_cells) & (age >= seal_steps)
    pending = candidate & ~seal
    pocket = pocket_parent | seal
    new_number = np.full(count, -1, dtype=np.int64)
    new_number[pocket] = np.arange(int(pocket.sum()))
    pockets = int(pocket.sum())
    region_number = np.zeros(count, dtype=np.int64)
    region_number[pending] = np.arange(1, int(pending.sum()) + 1)
    codes = np.where(pocket[component], FIRST_POCKET + new_number[component],
                     np.where(pending[component], INLET_GAS, LIQUID_POCKET)).astype(np.int32)
    cand_region = np.where(pending[component], region_number[component], 0).astype(np.int32)
    in_pocket = pocket[component]
    pocket_of = new_number[component]
    cells_of_new = np.bincount(pocket_of[in_pocket], minlength=pockets)
    air_cells_of_new = np.bincount(pocket_of[in_pocket & (prev == INLET_GAS)], minlength=pockets)
    old_to_new = {}
    from_old = in_pocket & (prev >= FIRST_POCKET)
    if from_old.any():
        pairs, counts = np.unique(np.stack((prev[from_old] - FIRST_POCKET, pocket_of[from_old])), axis=1,
                                  return_counts=True)
        for (old, new), n in zip(pairs.T, counts):
            old_to_new.setdefault(int(old), {})[int(new)] = int(n)
    parents_of_new = {}
    for old, children in old_to_new.items():
        for new in children:
            parents_of_new.setdefault(new, set()).add(old)
    for old in range(old_pockets):
        children = old_to_new.get(old, {})
        if not children and fate[old, FATE_AIR] == 0:
            raise SimulationHardStop("gas_pocket_zero_volume", dict(time_s=float(t_s), old_pocket=old,
                                                                    reason="no_air_overlap",
                                                                    fate=fate[old].tolist()))
        if len(children) + (1 if fate[old, FATE_AIR] > 0 else 0) > 1:
            shared = [new for new in children if len(parents_of_new.get(new, ())) > 1]
            if shared:
                raise SimulationHardStop("gas_pocket_ambiguous_topology", dict(
                    time_s=float(t_s), old_pocket=old, new_pockets=sorted(children),
                    new_with_several_parents=shared))
    # the candidate regions: new pending ones, sealed ones, and the old ones that ended unsealed
    parents = _parents_by_component(component, preg, count)
    regions = tuple(dict(age=int(age[c]), cells=int(size[c]), parents=parents[c]) for c in np.flatnonzero(pending))
    sealed = tuple(dict(pocket=int(new_number[c]), age=int(age[c]), cells=int(size[c]), parents=parents[c])
                   for c in np.flatnonzero(seal))
    kept, merged = set(), set()
    for c in range(count):
        for r in parents[c]:
            (kept if candidate[c] else merged).add(r)
    ended = []
    for r in range(1, old_regions + 1):
        if r in kept:
            continue
        if r in merged:
            how = "merged"
        elif rfate[r - 1, FATE_AIR] > 0:
            how = "reconnected"
        else:
            how = "covered"
        ended.append(dict(region=r, how=how, fate=rfate[r - 1].tolist()))
    back = np.argsort(order, kind="stable")
    return PocketTransitionPlan(candidate_codes=codes[back], pockets=pockets, old_to_new=old_to_new,
                      old_to_air_cells=fate[:, FATE_AIR].copy(), air_cells_of_new=air_cells_of_new,
                      cells_of_new=cells_of_new, candidate_regions=cand_region[back], regions=regions,
                      sealed=sealed, ended=tuple(ended))


def allocate_pocket_gas(plan, old_gas_pa_m3, new_volumes_m3, *, inlet_absolute_pa, air_volume_m3, air_cells):
    """New pockets' gas amounts from the plan, the old pockets' gas and the new volumes.  Returns (q_new, exchange)
    with exchange = gas taken from the inlet reservoir minus gas returned to it (Pa m^3)."""
    q_old = np.asarray(old_gas_pa_m3, dtype=np.float64)
    volumes = np.asarray(new_volumes_m3, dtype=np.float64)
    q_new = np.zeros(plan.pockets, dtype=np.float64)
    exchange = 0.0
    for old, q in enumerate(q_old):
        children = plan.old_to_new.get(old, {})
        weights = {new: volumes[new] * cells / plan.cells_of_new[new] for new, cells in children.items()}
        to_air = air_volume_m3 * plan.old_to_air_cells[old] / air_cells if air_cells > 0 else 0.0
        total = sum(weights.values()) + to_air
        if total <= 0.0:
            continue
        for new, weight in weights.items():
            q_new[new] += q * weight / total
        exchange -= q * to_air / total
    sealed = plan.air_cells_of_new > 0
    taken = inlet_absolute_pa * volumes[sealed] * plan.air_cells_of_new[sealed] / plan.cells_of_new[sealed]
    q_new[sealed] += taken
    exchange += float(taken.sum())
    return q_new, exchange
