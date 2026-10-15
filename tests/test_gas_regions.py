"""GasRegions (boundary/gas_regions.py) and the fractional-volume mode of GasPocketModel (boundary/gas_pockets.py).

Synthetic masks on a 6 x 6 x 8 box of 1 mm^3 cells: an inlet cell in the lid layer, a floor, a horizontal 'membrane'
slab of material cells at k = 3, and closed shells of material that seal pockets above, below or across it."""
import numpy as np
import pytest

from fastfsi.boundary.gas_regions import INLET_GAS, SEALED, LIQUID, GasRegions
from fastfsi.boundary.gas_pockets import GasPocketModel, PressureInlet
from fastfsi.errors import SimulationHardStop

SHAPE = (6, 6, 8)
V = 1.0e-9


def box():
    wall = np.zeros(SHAPE, dtype=bool)
    inlet = np.zeros(SHAPE, dtype=bool)
    inlet[0, 0, -1] = True
    slab = np.zeros(SHAPE, dtype=bool)
    slab[:, :, 3] = True
    return wall, inlet, slab


def shell(material, k0, k1):
    """A closed material shell over i, j in 1..4 and k in k0..k1 whose inside (i, j in 2..3, k0 < k < k1) is free."""
    out = material.copy()
    out[1:5, 1:5, k0:k1 + 1] = True
    out[2:4, 2:4, k0 + 1:k1] = False
    return out


def cover_of(material):
    return np.where(material, 1.0, 0.0)


def classified_rest():
    wall, inlet, slab = box()
    r = GasRegions(np.full(SHAPE, V), wall, inlet)
    first = r.classify(slab, cover_of(slab), t_s=0.0)
    r.accept(first)
    return r, first, wall, inlet, slab


def sealed_labels(c):
    return [int(label) for label in np.flatnonzero(c.component_class == SEALED) if label > 0]


def test_slab_separates_air_and_water():
    r, first, wall, inlet, slab = classified_rest()
    above = np.zeros(SHAPE, dtype=bool)
    above[:, :, 4:] = True
    below = np.zeros(SHAPE, dtype=bool)
    below[:, :, :3] = True
    assert np.array_equal(first.air_mask, above) and np.array_equal(first.water_mask, below)
    assert sorted(first.component_class[1:].tolist()) == [INLET_GAS, LIQUID]
    assert np.isclose(first.diagnostics()["air_volume_m3"], 6 * 6 * 4 * V)


def test_leak_through_a_hole_in_the_slab_is_a_hard_stop():
    r, first, wall, inlet, slab = classified_rest()
    holed = slab.copy()
    holed[4, 4, 3] = False
    with pytest.raises(SimulationHardStop) as stop:
        r.classify(holed, cover_of(holed), t_s=0.1)
    assert stop.value.code == "gas_region_leak"


def test_floor_material_is_a_hard_stop():
    r, first, wall, inlet, slab = classified_rest()
    floor = slab.copy()
    floor[2, 2, 0] = True
    with pytest.raises(SimulationHardStop) as stop:
        r.classify(floor, cover_of(floor), t_s=0.1)
    assert stop.value.code == "pneumatic_chamber_floor_material"


def test_pocket_origins():
    r, first, wall, inlet, slab = classified_rest()
    air = r.classify(shell(slab, 3, 6), cover_of(shell(slab, 3, 6)), t_s=0.1)     # above the slab: previous air
    assert len(sealed_labels(air)) == 1 and air.air_pocket[sealed_labels(air)[0]]
    water = r.classify(shell(slab, 0 + 1, 3), cover_of(shell(slab, 1, 3)), t_s=0.1)  # below: previous water
    assert len(sealed_labels(water)) == 1 and not water.air_pocket[sealed_labels(water)[0]]
    across = shell(slab, 1, 5)                                                       # across the old slab: both
    with pytest.raises(SimulationHardStop) as stop:
        r.classify(across, cover_of(across), t_s=0.1)
    assert stop.value.code == "gas_pocket_origin_ambiguous"


def test_first_classification_treats_sealed_regions_as_water():
    wall, inlet, slab = box()
    r = GasRegions(np.full(SHAPE, V), wall, inlet)
    c = r.classify(shell(slab, 3, 6), cover_of(shell(slab, 3, 6)), t_s=0.0)
    assert len(sealed_labels(c)) == 1 and not c.air_pocket[sealed_labels(c)[0]]


def test_fractional_volumes_share_material_free_space():
    r, first, wall, inlet, slab = classified_rest()
    cover = cover_of(slab)
    cover[:, :, 3] = 0.75                             # every slab cell is 25 % free
    c = r.classify(slab, cover, t_s=0.0)
    # each slab cell touches two distinct free components (air above, water below): 0.125 V to each
    assert np.isclose(c.effective_volume_m3[c.air_mask].sum(), 6 * 6 * 4 * V + 6 * 6 * 0.125 * V)
    assert np.isclose(c.effective_volume_m3.sum(), 6 * 6 * 7 * V + 6 * 6 * 0.25 * V)


def test_fractional_volume_is_continuous_when_a_cell_turns_material():
    r, first, wall, inlet, slab = classified_rest()
    a = r.classify(slab, cover_of(slab), t_s=0.0)
    after = slab.copy()
    after[3, 3, 5] = True                            # an air cell becomes material with a tiny coverage
    cover = cover_of(slab)
    cover[3, 3, 5] = 1.0e-6
    b = r.classify(after, cover, t_s=0.0)
    va = a.effective_volume_m3[a.air_mask].sum()
    vb = b.effective_volume_m3[b.air_mask].sum()
    assert abs(vb - va) <= 2.0e-6 * V


def test_gas_model_fractional_mode():
    r, first, wall, inlet, slab = classified_rest()
    gas = GasPocketModel(np.full(SHAPE, V), [PressureInlet(inlet, -4500.0)], ambient_pressure_pa=101325.0,
                         fractional_volumes=True)
    gas.update(first.air_mask, t_s=0.0, cell_volumes_m3=first.effective_volume_m3)
    material = shell(slab, 3, 6)
    c = r.classify(material, cover_of(material), t_s=0.1)
    seal = gas.update(c.air_mask, t_s=0.1, cell_volumes_m3=c.effective_volume_m3)
    pocket = seal.sealed_labels[0]
    # gas_pockets' split rule: the old connected gas (144 cells at the inlet pressure) goes to the children in
    # proportion to their volumes (96 connected + 8 sealed cells; the 40 cells the shell took are compression)
    p_seal = (101325.0 - 4500.0) * 144.0 / 104.0
    assert np.isclose(seal.pressure_abs_pa[pocket], p_seal, rtol=1e-12)
    # the same topology with 10 % less free volume in the pocket (coverage only): the identical-mask fast path
    cover = cover_of(material)
    cover[2:4, 2:4, 4:6] = 0.1
    c2 = r.classify(material, cover, t_s=0.2)
    fast = gas.update(c2.air_mask, t_s=0.2, cell_volumes_m3=c2.effective_volume_m3)
    assert np.array_equal(fast.labels, seal.labels)
    assert np.isclose(fast.pressure_abs_pa[pocket], p_seal / 0.9, rtol=1e-12)
    again = GasPocketModel(np.full(SHAPE, V), [PressureInlet(inlet, -4500.0)], ambient_pressure_pa=101325.0,
                           fractional_volumes=True)
    again.load_state_arrays(gas.state_arrays())
    assert np.array_equal(again.state_arrays()["cell_volumes_m3"], c2.effective_volume_m3)
    with pytest.raises(ValueError):
        gas.update(c2.air_mask, t_s=0.3)                                    # fractional mode needs the volumes


def test_gas_model_default_mode_refuses_per_update_volumes():
    wall, inlet, slab = box()
    gas = GasPocketModel(np.full(SHAPE, V), [PressureInlet(inlet, 0.0)], ambient_pressure_pa=101325.0)
    with pytest.raises(ValueError):
        gas.update(~slab, t_s=0.0, cell_volumes_m3=np.full(SHAPE, V))


def test_dirichlet_codes_and_state_round_trip():
    r, first, wall, inlet, slab = classified_rest()
    gas = GasPocketModel(np.full(SHAPE, V), [PressureInlet(inlet, -4500.0)], ambient_pressure_pa=101325.0,
                         fractional_volumes=True)
    gas.update(first.air_mask, t_s=0.0, cell_volumes_m3=first.effective_volume_m3)
    material = shell(slab, 3, 6)
    c = r.classify(material, cover_of(material), t_s=0.1)
    update = gas.update(c.air_mask, t_s=0.1, cell_volumes_m3=c.effective_volume_m3)
    codes = r.dirichlet_codes(c, update.labels)
    assert set(np.unique(codes).tolist()) == {0, 1, 2 + update.sealed_labels[0]}
    assert (codes[c.water_mask] == 0).all() and (codes[material] == 0).all()
    other = GasRegions(np.full(SHAPE, V), wall, inlet)
    other.load_state_arrays(r.state_arrays())
    assert np.array_equal(other.previous_air, r.previous_air)
