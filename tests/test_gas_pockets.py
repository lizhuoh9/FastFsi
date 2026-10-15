"""Isothermal gas accounting on synthetic masks; no device initialization."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fastfsi.boundary.gas_pockets import GasPocketModel, PressureInlet, circular_wall_patch
from fastfsi.errors import SimulationHardStop


AMBIENT_PA = 100_000.0
CELL_VOLUME_M3 = 1.0e-9


def line_mask(length, indices):
    mask = np.zeros((1, 1, length), dtype=bool)
    mask[0, 0, indices] = True
    return mask


def model_for_line(length, *, source=0.0, initial=0.0, volumes=None, bounds=None):
    inlet = PressureInlet(line_mask(length, [0]), source)
    return GasPocketModel(
        np.full((1, 1, length), CELL_VOLUME_M3) if volumes is None else volumes,
        (inlet,), ambient_pressure_pa=AMBIENT_PA,
        initial_pressure_gauge_pa=initial, pressure_bounds_abs_pa=bounds,
    )


def pressure_at(update, index):
    return update.pressure_abs_pa[update.labels[0, 0, index]]


def assert_same_arrays(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert left[name].dtype == right[name].dtype, name
        assert left[name].shape == right[name].shape, name
        assert left[name].tobytes() == right[name].tobytes(), name


def test_open_air_uses_gauge_source_and_explicit_ambient():
    model = model_for_line(5, source=lambda t: -4_000.0 + 2_000.0 * t)
    update = model.update(line_mask(5, range(5)), t_s=0.5)
    np.testing.assert_array_equal(update.gauge_pressure_pa, np.full((1, 1, 5), -3_000.0))
    assert update.inlet_covered == (False,)
    assert update.sealed_labels == ()
    assert update.diagnostics()["pocket_count"] == 0
    assert update.gas_amount_pa_m3.sum() == pytest.approx(97_000.0 * 5 * CELL_VOLUME_M3)


def test_covering_inlet_preserves_last_open_gas_and_ignores_new_source():
    def source(t):
        if t > 0.0:
            raise AssertionError("a covered inlet must not set pocket pressure")
        return -4_000.0

    model = model_for_line(5, source=source)
    old = model.update(line_mask(5, range(5)), t_s=0.0)
    sealed = model.update(line_mask(5, range(1, 5)), t_s=1.0)
    assert sealed.inlet_covered == (True,)
    assert len(sealed.sealed_labels) == 1
    assert sealed.gas_amount_pa_m3.sum() == old.gas_amount_pa_m3.sum()
    assert pressure_at(sealed, 1) == pytest.approx(96_000.0 * 5 / 4)
    assert sealed.reservoir_exchange_pa_m3 == 0.0


def test_only_face_neighbours_connect_air():
    air = np.zeros((2, 2, 2), dtype=bool)
    air[0, 0, 0] = air[1, 1, 1] = True
    inlet_mask = np.zeros_like(air)
    inlet_mask[0, 0, 0] = True
    model = GasPocketModel(np.full(air.shape, CELL_VOLUME_M3), (PressureInlet(inlet_mask, 500.0),),
                           ambient_pressure_pa=AMBIENT_PA, initial_pressure_gauge_pa=-500.0)
    update = model.update(air, t_s=0.0)
    assert update.labels.max() == 2
    assert update.connected_to_inlet.tolist() == [False, True, False]
    assert update.pressure_abs_pa[1:].tolist() == [100_500.0, 99_500.0]


def test_initial_sealed_air_requires_an_explicit_pressure():
    model = model_for_line(5, initial=None)
    with pytest.raises(SimulationHardStop) as caught:
        model.update(line_mask(5, [2, 3]), t_s=0.0)
    assert caught.value.code == "gas_pocket_missing_initial_pressure"


def test_analytic_compression_and_repeated_small_pocket_stiffness():
    model = model_for_line(101)
    full = line_mask(101, range(1, 101))
    compressed = line_mask(101, range(1, 100))
    initial = model.update(full, t_s=0.0)
    initial_gas = initial.gas_amount_pa_m3.sum()
    for step in range(1, 41):
        mask = compressed if step % 2 else full
        update = model.update(mask, t_s=float(step))
        assert update.gas_amount_pa_m3.sum() == initial_gas
        expected = AMBIENT_PA * (100 / 99 if step % 2 else 1.0)
        assert pressure_at(update, 1) == pytest.approx(expected, rel=1e-14)
    compressed_update = model.update(compressed, t_s=41.0)
    assert pressure_at(compressed_update, 1) - AMBIENT_PA == pytest.approx(1_010.10101010101)


def test_split_uses_child_physical_volumes_on_a_graded_grid():
    volumes = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]).reshape(1, 1, 7) * CELL_VOLUME_M3
    model = model_for_line(7, volumes=volumes)
    before = model.update(line_mask(7, range(1, 7)), t_s=0.0)
    split = model.update(line_mask(7, [1, 3, 4, 5, 6]), t_s=1.0)
    assert len(split.sealed_labels) == 2
    np.testing.assert_allclose(split.volume_m3[1:], [1 * CELL_VOLUME_M3, 8 * CELL_VOLUME_M3])
    np.testing.assert_allclose(split.gas_amount_pa_m3[1:], before.gas_amount_pa_m3.sum() * np.array([1, 8]) / 9)
    assert split.gas_amount_pa_m3.sum() == pytest.approx(before.gas_amount_pa_m3.sum(), rel=1e-15)
    np.testing.assert_allclose(split.pressure_abs_pa[1:], np.full(2, AMBIENT_PA * 10 / 9))


def test_merge_sums_gas_even_when_parent_pressures_differ():
    model = model_for_line(9)
    before = model.update(line_mask(9, [1, 2, 3, 4, 6, 7]), t_s=0.0)
    compressed = model.update(line_mask(9, [1, 2, 6, 7]), t_s=1.0)
    np.testing.assert_allclose(compressed.pressure_abs_pa[1:], [2 * AMBIENT_PA, AMBIENT_PA])
    merged = model.update(line_mask(9, range(1, 8)), t_s=2.0)
    assert len(merged.sealed_labels) == 1
    assert merged.gas_amount_pa_m3.sum() == before.gas_amount_pa_m3.sum()
    assert pressure_at(merged, 1) == pytest.approx(AMBIENT_PA * 6 / 7)


def test_reconnection_and_reopening_record_reservoir_exchange():
    model = model_for_line(5, source=lambda t: -5_000.0 if t < 2.0 else 8_000.0)
    sealed = model.update(line_mask(5, [2, 3, 4]), t_s=0.0)
    old_total = sealed.gas_amount_pa_m3.sum()
    opened = model.update(line_mask(5, range(5)), t_s=2.0)
    assert opened.inlet_covered == (False,)
    assert opened.sealed_labels == ()
    np.testing.assert_array_equal(opened.gauge_pressure_pa, np.full((1, 1, 5), 8_000.0))
    assert opened.reservoir_exchange_pa_m3 == pytest.approx(opened.gas_amount_pa_m3.sum() - old_total)
    closed = model.update(line_mask(5, [1, 2, 3, 4]), t_s=3.0)
    reopened = model.update(line_mask(5, range(5)), t_s=4.0)
    assert reopened.reservoir_exchange_pa_m3 == pytest.approx(
        reopened.gas_amount_pa_m3.sum() - closed.gas_amount_pa_m3.sum())


def test_new_connected_air_can_be_supplied_but_new_sealed_air_is_refused():
    model = model_for_line(6)
    before = model.update(line_mask(6, [4, 5]), t_s=0.0)
    supplied = model.update(line_mask(6, [0, 4, 5]), t_s=1.0)
    assert supplied.reservoir_exchange_pa_m3 == pytest.approx(AMBIENT_PA * CELL_VOLUME_M3)
    assert supplied.gas_amount_pa_m3.sum() == pytest.approx(before.gas_amount_pa_m3.sum()
                                                           + supplied.reservoir_exchange_pa_m3)
    snapshot = model.state_arrays()
    with pytest.raises(SimulationHardStop) as caught:
        model.update(line_mask(6, [0, 2, 4, 5]), t_s=2.0)
    assert caught.value.code == "gas_pocket_untracked_component"
    assert_same_arrays(snapshot, model.state_arrays())


def test_disappearing_pocket_refuses_zero_volume_with_gas_and_preserves_state():
    model = model_for_line(5)
    model.update(line_mask(5, [2, 3]), t_s=0.0)
    before = model.state_arrays()
    with pytest.raises(SimulationHardStop) as caught:
        model.update(line_mask(5, []), t_s=1.0)
    assert caught.value.code == "gas_pocket_zero_volume"
    assert caught.value.witness["gas_amount_pa_m3"] > 0.0
    assert_same_arrays(before, model.state_arrays())


def test_ambiguous_many_to_many_topology_fails_closed():
    air = np.zeros((1, 3, 3), dtype=bool)
    air[0, 0, :] = air[0, 2, :] = True
    model = GasPocketModel(np.full(air.shape, CELL_VOLUME_M3), (),
                           ambient_pressure_pa=AMBIENT_PA, initial_pressure_gauge_pa=0.0)
    model.update(air, t_s=0.0)
    before = model.state_arrays()
    crossed = np.zeros_like(air)
    crossed[0, :, 0] = crossed[0, :, 2] = True
    with pytest.raises(SimulationHardStop) as caught:
        model.update(crossed, t_s=1.0)
    assert caught.value.code == "gas_pocket_ambiguous_topology"
    assert_same_arrays(before, model.state_arrays())


@pytest.mark.parametrize("source", [-AMBIENT_PA, -2 * AMBIENT_PA, np.nan, np.inf])
def test_nonpositive_or_nonfinite_absolute_pressure_is_refused(source):
    model = model_for_line(2, source=source)
    with pytest.raises(SimulationHardStop) as caught:
        model.update(line_mask(2, [0, 1]), t_s=0.0)
    assert caught.value.code == "gas_pocket_invalid_pressure"


def test_caller_pressure_bound_stops_compression_without_clipping():
    model = model_for_line(5, bounds=(50_000.0, 150_000.0))
    before = model.update(line_mask(5, [1, 2, 3, 4]), t_s=0.0)
    accepted = model.state_arrays()
    with pytest.raises(SimulationHardStop) as caught:
        model.update(line_mask(5, [1]), t_s=1.0)
    assert caught.value.code == "gas_pocket_pressure_out_of_bounds"
    assert caught.value.witness["pressure_abs_pa"] == pytest.approx(4 * AMBIENT_PA)
    assert_same_arrays(accepted, model.state_arrays())
    assert before.pressure_abs_pa[1] == AMBIENT_PA


def test_different_pressure_sources_in_one_air_component_are_refused():
    shape = (1, 1, 5)
    inlets = (PressureInlet(line_mask(5, [0]), 0.0), PressureInlet(line_mask(5, [4]), 100.0))
    model = GasPocketModel(np.full(shape, CELL_VOLUME_M3), inlets, ambient_pressure_pa=AMBIENT_PA)
    with pytest.raises(SimulationHardStop) as caught:
        model.update(line_mask(5, range(5)), t_s=0.0)
    assert caught.value.code == "gas_pocket_inlet_pressure_conflict"


def test_matching_pressure_sources_share_a_component():
    inlets = (PressureInlet(line_mask(5, [0]), 250.0), PressureInlet(line_mask(5, [4]), lambda t: 250.0))
    model = GasPocketModel(np.full((1, 1, 5), CELL_VOLUME_M3), inlets, ambient_pressure_pa=AMBIENT_PA)
    update = model.update(line_mask(5, range(5)), t_s=0.0)
    assert update.inlet_covered == (False, False)
    assert update.pressure_abs_pa[1] == AMBIENT_PA + 250.0


def test_nonair_cells_have_no_pressure_assignment():
    update = model_for_line(5).update(line_mask(5, [1, 2]), t_s=0.0)
    assert np.isnan(update.gauge_pressure_pa[0, 0, [0, 3, 4]]).all()


def test_snapshot_restore_is_exact_and_failed_restore_is_atomic():
    model = model_for_line(5)
    model.update(line_mask(5, [1, 2, 3, 4]), t_s=0.0)
    saved = model.state_arrays()
    continuous = model.update(line_mask(5, [1, 2]), t_s=1.0)
    model.load_state_arrays(saved)
    repeated = model.update(line_mask(5, [1, 2]), t_s=1.0)
    np.testing.assert_array_equal(continuous.gas_amount_pa_m3, repeated.gas_amount_pa_m3)
    np.testing.assert_array_equal(continuous.pressure_abs_pa, repeated.pressure_abs_pa)
    accepted = model.state_arrays()
    corrupt = {name: value.copy() for name, value in accepted.items()}
    corrupt["gas_amount_pa_m3"][1] = np.nan
    with pytest.raises(ValueError):
        model.load_state_arrays(corrupt)
    assert_same_arrays(accepted, model.state_arrays())


def test_caller_arrays_and_results_cannot_mutate_accepted_state():
    volumes = np.full((1, 1, 5), CELL_VOLUME_M3)
    inlet_mask = line_mask(5, [0])
    model = GasPocketModel(volumes, (PressureInlet(inlet_mask, 0.0),),
                           ambient_pressure_pa=AMBIENT_PA, initial_pressure_gauge_pa=0.0)
    volumes[...] = 1.0
    inlet_mask[...] = True
    air = line_mask(5, [1, 2, 3, 4])
    update = model.update(air, t_s=0.0)
    air[...] = False
    snapshot = model.state_arrays()
    snapshot["gas_amount_pa_m3"][1] = 0.0
    update.labels.setflags(write=True)
    update.labels[...] = 0
    compressed = model.update(line_mask(5, [1, 2]), t_s=1.0)
    assert pressure_at(compressed, 1) == pytest.approx(2 * AMBIENT_PA)
    assert compressed.inlet_covered == (True,)


def test_circular_patch_is_on_the_selected_wall_and_uses_physical_position():
    faces = (np.array([0.0, 1.0, 2.0]), np.array([10.0, 11.0, 12.0]), np.array([20.0, 21.0, 22.0]))
    patch = circular_wall_patch(faces, wall_axis=2, wall_side="max", center_m=(0.5, 10.5, 22.0), radius_m=0.4)
    expected = np.zeros((2, 2, 2), dtype=bool)
    expected[0, 0, -1] = True
    np.testing.assert_array_equal(patch, expected)
    with pytest.raises(ValueError):
        circular_wall_patch(faces, wall_axis=2, wall_side="max", center_m=(0.5, 10.5, 21.0), radius_m=0.4)
    with pytest.raises(ValueError):
        circular_wall_patch(faces, wall_axis=2, wall_side="max", center_m=(0.0, 10.0, 22.0), radius_m=0.1)


@pytest.mark.parametrize("bad_volume", [0.0, -1.0, np.nan, np.inf])
def test_invalid_grid_volume_is_refused(bad_volume):
    with pytest.raises(ValueError):
        model_for_line(2, volumes=np.full((1, 1, 2), bad_volume))


def test_passive_import_does_not_load_or_initialize_device_or_existing_chamber():
    source = (
        "import sys; import fastfsi.boundary.gas_pockets; "
        "assert 'taichi' not in sys.modules; "
        "assert 'fastfsi.boundary.pressure_chamber' not in sys.modules"
    )
    src = str(Path(__file__).resolve().parents[1] / "src")       # the package without an installation
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, (src, os.environ.get("PYTHONPATH")))))
    completed = subprocess.run([sys.executable, "-c", source], env=env, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_open_component_can_split_into_open_and_sealed_children():
    model = model_for_line(7, source=lambda t: 0.0 if t == 0.0 else -10_000.0)
    original = model.update(line_mask(7, range(7)), t_s=0.0)
    split = model.update(line_mask(7, [0, 1, 3, 4, 5, 6]), t_s=1.0)
    assert split.inlet_covered == (False,)
    assert split.sealed_labels == (2,)
    assert split.pressure_abs_pa[1] == 90_000.0
    assert split.pressure_abs_pa[2] == pytest.approx(AMBIENT_PA * 7 / 6)
    assert split.gas_amount_pa_m3[2] == pytest.approx(original.gas_amount_pa_m3[1] * 4 / 6)
    assert split.gas_amount_pa_m3.sum() - original.gas_amount_pa_m3.sum() == pytest.approx(
        split.reservoir_exchange_pa_m3)
    merged = model.update(line_mask(7, range(7)), t_s=2.0)
    assert merged.sealed_labels == ()
    assert merged.gas_amount_pa_m3.sum() - split.gas_amount_pa_m3.sum() == pytest.approx(
        merged.reservoir_exchange_pa_m3)


def test_partial_inlet_coverage_keeps_the_surviving_inlet_cell_open():
    inlet = PressureInlet(line_mask(5, [0, 1]), -4_000.0)
    model = GasPocketModel(np.full((1, 1, 5), CELL_VOLUME_M3), (inlet,), ambient_pressure_pa=AMBIENT_PA)
    model.update(line_mask(5, range(5)), t_s=0.0)
    partial = model.update(line_mask(5, [1, 2, 3, 4]), t_s=1.0)
    assert partial.inlet_covered == (False,)
    assert partial.pressure_abs_pa[1] == 96_000.0


def test_extreme_compression_refuses_nonfinite_pressure_without_losing_gas():
    # Normal-range volumes keep this pressure-overflow case independent of CPU FTZ mode.
    volumes = np.array([1.0e308, 1.0e-8]).reshape(1, 1, 2)
    model = GasPocketModel(volumes, (), ambient_pressure_pa=1.0, initial_pressure_gauge_pa=0.0)
    model.update(line_mask(2, [0, 1]), t_s=0.0)
    saved = model.state_arrays()
    with pytest.raises(SimulationHardStop) as caught:
        model.update(line_mask(2, [1]), t_s=1.0)
    assert caught.value.code == "gas_pocket_invalid_pressure"
    assert_same_arrays(saved, model.state_arrays())


def test_checkpoint_round_trip_of_open_state_can_close_identically(tmp_path):
    from fastfsi.io.checkpoint import read_checkpoint, write_checkpoint

    model = model_for_line(5, source=-4_000.0)
    model.update(line_mask(5, range(5)), t_s=0.0)
    checkpoint = tmp_path / "open_state.npz"
    write_checkpoint(checkpoint, {"gas": model}, {"time_s": 0.0}, {"ambient_pressure_pa": AMBIENT_PA})
    continuous = model.update(line_mask(5, [1, 2, 3, 4]), t_s=1.0)
    arrays, _, _, _, _ = read_checkpoint(checkpoint)
    restarted = model_for_line(5, source=-4_000.0)
    restarted.load_state_arrays(arrays["gas"])
    resumed = restarted.update(line_mask(5, [1, 2, 3, 4]), t_s=1.0)
    np.testing.assert_array_equal(continuous.gas_amount_pa_m3, resumed.gas_amount_pa_m3)
    np.testing.assert_array_equal(continuous.pressure_abs_pa, resumed.pressure_abs_pa)
    assert_same_arrays(model.state_arrays(), restarted.state_arrays())


def test_initial_sealed_gas_is_distinguished_from_reservoir_supply():
    import json

    model = model_for_line(5)
    update = model.update(line_mask(5, [0, 2, 3, 4]), t_s=0.0)
    assert update.initial_sealed_gas_pa_m3 == pytest.approx(AMBIENT_PA * 3 * CELL_VOLUME_M3)
    assert update.reservoir_exchange_pa_m3 == pytest.approx(AMBIENT_PA * CELL_VOLUME_M3)
    assert update.gas_amount_pa_m3.sum() == pytest.approx(
        update.initial_sealed_gas_pa_m3 + update.reservoir_exchange_pa_m3)
    json.dumps(update.diagnostics(), allow_nan=False)


@pytest.mark.parametrize("bad_source, token", [(np.nan, "nan"), (np.inf, "+inf"), (-np.inf, "-inf")])
def test_nonfinite_source_failure_witness_is_strict_json_and_atomic(bad_source, token):
    import json

    model = model_for_line(5, source=lambda t: 0.0 if t == 0.0 else bad_source)
    model.update(line_mask(5, range(5)), t_s=0.0)
    accepted = model.state_arrays()
    with pytest.raises(SimulationHardStop) as caught:
        model.update(line_mask(5, range(5)), t_s=1.0)
    assert caught.value.code == "gas_pocket_invalid_pressure"
    assert_same_arrays(accepted, model.state_arrays())
    json.dumps(caught.value.witness, allow_nan=False)
    assert caught.value.witness["pressure_abs_pa"] == token


def test_compression_overflow_failure_witness_is_strict_json_and_atomic():
    import json

    # Normal-range volumes keep this pressure-overflow case independent of CPU FTZ mode.
    volumes = np.array([1.0e308, 1.0e-8]).reshape(1, 1, 2)
    model = GasPocketModel(volumes, (), ambient_pressure_pa=1.0, initial_pressure_gauge_pa=0.0)
    model.update(line_mask(2, [0, 1]), t_s=0.0)
    accepted = model.state_arrays()
    with pytest.raises(SimulationHardStop) as caught:
        model.update(line_mask(2, [1]), t_s=1.0)
    assert caught.value.code == "gas_pocket_invalid_pressure"
    assert_same_arrays(accepted, model.state_arrays())
    json.dumps(caught.value.witness, allow_nan=False)
    assert caught.value.witness["pressure_abs_pa"] == "+inf"


@pytest.mark.parametrize("source", [0.0, 1.0])
def test_gas_sum_or_product_overflow_failure_witness_is_strict_json_and_atomic(source):
    import json

    volumes = np.array([1.0e308, CELL_VOLUME_M3, 1.0e308]).reshape(1, 1, 3)
    inlet = PressureInlet(line_mask(3, [2]), source)
    model = GasPocketModel(volumes, (inlet,), ambient_pressure_pa=1.0, initial_pressure_gauge_pa=0.0)
    model.update(line_mask(3, [0]), t_s=0.0)
    accepted = model.state_arrays()
    with pytest.raises(SimulationHardStop) as caught:
        model.update(line_mask(3, [0, 2]), t_s=1.0)
    assert caught.value.code == "gas_pocket_invalid_gas_amount"
    assert_same_arrays(accepted, model.state_arrays())
    json.dumps(caught.value.witness, allow_nan=False)
    assert caught.value.witness["gas_amount_pa_m3"] == "+inf"


def test_nested_volume_overflow_failure_witness_is_strict_json_and_atomic():
    import json

    model = GasPocketModel(np.full((1, 1, 2), 1.0e308), (),
                           ambient_pressure_pa=1.0, initial_pressure_gauge_pa=0.0)
    model.update(line_mask(2, [0]), t_s=0.0)
    accepted = model.state_arrays()
    with pytest.raises(SimulationHardStop) as caught:
        model.update(line_mask(2, [0, 1]), t_s=1.0)
    assert caught.value.code == "gas_pocket_zero_volume"
    assert_same_arrays(accepted, model.state_arrays())
    json.dumps(caught.value.witness, allow_nan=False)
    assert caught.value.witness["volumes_m3"] == [0.0, "+inf"]


def test_restore_total_gas_overflow_is_refused_before_any_state_changes():
    volumes = np.array([1.0e308, 1.0, 1.0e308]).reshape(1, 1, 3)
    model = GasPocketModel(volumes, (), ambient_pressure_pa=1.0, initial_pressure_gauge_pa=0.0)
    model.update(line_mask(3, [0]), t_s=0.0)
    accepted = model.state_arrays()
    corrupt = {name: value.copy() for name, value in accepted.items()}
    corrupt["labels"] = np.array([1, 0, 2], dtype=np.int32).reshape(1, 1, 3)
    corrupt["gas_amount_pa_m3"] = np.array([0.0, 1.0e308, 1.0e308], dtype=np.float64)
    with pytest.raises(ValueError, match="total gas"):
        model.load_state_arrays(corrupt)
    assert_same_arrays(accepted, model.state_arrays())


def test_restore_accepts_a_large_finite_total_gas_amount():
    import math

    volumes = np.array([1.0e308, 1.0, 1.0e308]).reshape(1, 1, 3)
    model = GasPocketModel(volumes, (), ambient_pressure_pa=1.0, initial_pressure_gauge_pa=0.0)
    valid = model.state_arrays()
    valid["labels"] = np.array([1, 0, 2], dtype=np.int32).reshape(1, 1, 3)
    valid["gas_amount_pa_m3"] = np.array([0.0, 8.0e307, 8.0e307], dtype=np.float64)
    valid["initialized"] = np.asarray(1, dtype=np.uint8)
    model.load_state_arrays(valid)
    assert_same_arrays(valid, model.state_arrays())
    update = model.update(line_mask(3, [0, 2]), t_s=1.0)
    assert math.fsum(update.gas_amount_pa_m3[1:]) == 1.6e308
