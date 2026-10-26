"""Checkpoint file format: type-exact host state, array round trip, create-only, no pickle (no Taichi needed)."""
import json
import struct
import zipfile
from collections import defaultdict, namedtuple

import numpy as np
import pytest

from fastfsi.io.checkpoint import (HOST_ARRAY_MAX_BYTES, ArrayState, HostState, decode_host, encode_host,
                                   load_components, read_checkpoint, same_host_value, write_checkpoint)

NAN_PAYLOAD = struct.unpack("<d", bytes.fromhex("0100000000f8ff7f"))[0]       # a quiet NaN with payload bits


def host_values():
    nested = {"b": [1, 2.0], "a": (np.float32(0.5), None)}
    return {
        "none": None, "true": True, "false": False, "zero": 0, "negative": -1, "big_int": 2 ** 70,
        "float": 1.5, "negative_zero": -0.0, "nan": float("nan"), "nan_payload": NAN_PAYLOAD,
        "inf": float("inf"), "negative_inf": float("-inf"), "tiny": 5e-324,
        "float32": np.float32(1.1), "float32_nan": np.float32("nan"), "float32_negative_zero": np.float32(-0.0),
        "float64": np.float64(2.5), "int32": np.int32(-3), "int64": np.int64(7), "uint8": np.uint8(255),
        "numpy_bool": np.bool_(True), "string": "s", "bytes": b"\x00\xff",
        "tuple": (1, 2.0), "list": [1, (2, 3)], "empty_list": [], "empty_tuple": (), "set": {3, 1, 2},
        "frozenset": frozenset({(0, 1), (2, 3)}), "empty_set": set(),
        "dict_key_types": {1: "a", "1": "b", (2, 3): 4.0, 2.5: None, False: 1},
        "nested": nested, "array_float32": np.array([1.0, np.nan, -0.0], dtype=np.float32),
        "array_int16": np.arange(6, dtype=np.int16).reshape(2, 3),
    }


def json_round_trip(value):
    return decode_host(json.loads(json.dumps(encode_host(value), allow_nan=False)))


def test_every_host_value_comes_back_with_the_same_type_and_bits():
    for name, value in host_values().items():
        back = json_round_trip(value)
        assert type(back) is type(value), name
        assert same_host_value(back, value), name


def test_float_bits_are_exact():
    for value in (NAN_PAYLOAD, -0.0, float("inf"), 5e-324, 0.1):
        assert struct.pack("<d", json_round_trip(value)) == struct.pack("<d", value)
    assert json_round_trip(np.float32("nan")).tobytes() == np.float32("nan").tobytes()


def test_dict_order_and_key_types_survive():
    value = {"z": 1, 1: 2, "1": 3, (0, 0): 4}
    back = json_round_trip(value)
    assert list(back) == list(value)
    assert [type(k) for k in back] == [type(k) for k in value]


def test_same_host_value_tells_types_apart():
    assert not same_host_value(1, 1.0)
    assert not same_host_value(np.float32(1.0), 1.0)
    assert not same_host_value(np.float32(1.0), np.float64(1.0))
    assert not same_host_value((1, 2), [1, 2])
    assert not same_host_value(True, 1)
    assert not same_host_value({1: 1, 2: 2}, {2: 2, 1: 1})              # dict order is state
    assert same_host_value(float("nan"), float("nan"))
    assert not same_host_value(0.0, -0.0)


@pytest.mark.parametrize("value", [object(), defaultdict(int), namedtuple("P", "x")(1),
                                   np.zeros(HOST_ARRAY_MAX_BYTES // 8 + 1), np.array(["a"]), 1j * np.ones(1, object)])
def test_values_without_an_exact_encoding_are_refused(value):
    with pytest.raises(TypeError):
        encode_host(value)


class FakeField:
    def __init__(self, array):
        self.array = array

    def to_numpy(self):
        return self.array.copy()

    def from_numpy(self, array):
        self.array[...] = array


class Counters:
    def __init__(self):
        self.stats = {"events": 0, "first_t_s": None}
        self.seen = set()
        self.last_dt = 0.0


def make_components():
    fields = ArrayState(dict(u=FakeField(np.zeros((4, 3), np.float32)), flag=FakeField(np.zeros(4, np.int32))))
    counters = Counters()

    class Both:
        def __init__(self):
            self.arrays, self.host = fields, HostState(counters, ("stats", "seen", "last_dt"))

        def state_array_names(self):
            return self.arrays.state_array_names()

        def state_array(self, name):
            return self.arrays.state_array(name)

        def load_state_arrays(self, arrays):
            self.arrays.load_state_arrays(arrays)

        def state_host(self):
            return self.host.state_host()

        def load_state_host(self, values):
            self.host.load_state_host(values)

    return Both(), fields, counters


def test_checkpoint_round_trip_restores_arrays_host_state_and_loop_state(tmp_path):
    component, fields, counters = make_components()
    fields.fields["u"].array[...] = np.arange(12, dtype=np.float32).reshape(4, 3)
    fields.fields["u"].array[0, 0] = np.nan
    fields.fields["flag"].array[...] = [1, -2, 3, 0]
    counters.stats.update(events=3, first_t_s=np.float32(0.25))
    counters.seen.update({(1, 2), (0, 5)})
    counters.last_dt = 1.0e-4
    loop_state = dict(step=17, t_s=0.0017, dt_s=1.0e-4, limiter="cfl", values=host_values())
    path = tmp_path / "state.npz"
    write_checkpoint(path, dict(points=component), loop_state, dict(grid=[92, 92, 186]))
    assert not (tmp_path / "state.npz.partial").exists()
    with pytest.raises(FileExistsError):
        write_checkpoint(path, dict(points=component), loop_state, {})
    with np.load(path, allow_pickle=False) as npz:                  # plain NPZ, no pickle needed
        assert sorted(npz.files) == ["__checkpoint__.json", "points/flag", "points/u"]
        assert npz["points/u"].tobytes() == fields.fields["u"].array.tobytes()

    fresh, fresh_fields, fresh_counters = make_components()
    alias = fresh_counters.stats
    arrays, host, loop_back, manifest, header = read_checkpoint(path)
    load_components(dict(points=fresh), arrays, host)
    assert fresh_fields.fields["u"].array.tobytes() == fields.fields["u"].array.tobytes()
    assert np.array_equal(fresh_fields.fields["flag"].array, fields.fields["flag"].array)
    assert fresh_counters.stats is alias and same_host_value(fresh_counters.stats, counters.stats)
    assert same_host_value(fresh_counters.seen, counters.seen)
    assert same_host_value(fresh_counters.last_dt, counters.last_dt)
    assert same_host_value(loop_back, loop_state)
    assert manifest == dict(grid=[92, 92, 186])


def test_a_corrupted_array_is_refused(tmp_path):
    component, fields, _ = make_components()
    path = tmp_path / "state.npz"
    write_checkpoint(path, dict(points=component), {}, {})
    damaged = tmp_path / "damaged.npz"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(damaged, "x") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "points/u.npy":
                data = data[:-1] + bytes([data[-1] ^ 1])
            target.writestr(item, data)
    with pytest.raises(ValueError, match="sha256"):
        read_checkpoint(damaged)


def test_a_checkpoint_never_replaces_a_file_that_appears_while_it_is_written(tmp_path):
    """Codex review 2026-10-26 (checked by Root): the start's existence check left a window, and os.replace overwrote a
    file another writer created in it.  The hard-link publish refuses it and keeps the new checkpoint as .partial."""
    path = tmp_path / "state.npz"

    class Racer:                                       # another writer takes the name while this one writes
        def state_arrays(self):
            path.write_bytes(b"the other writer")
            return {"u": np.zeros(3, np.float32)}

    with pytest.raises(FileExistsError, match="appeared while it was written"):
        write_checkpoint(path, dict(points=Racer()), {}, {})
    assert path.read_bytes() == b"the other writer"
    assert read_checkpoint(tmp_path / "state.npz.partial")[0]["points"]["u"].tolist() == [0.0, 0.0, 0.0]


def test_components_must_match_exactly(tmp_path):
    component, _, _ = make_components()
    path = tmp_path / "state.npz"
    write_checkpoint(path, dict(points=component), {}, {})
    arrays, host, _, _, _ = read_checkpoint(path)
    with pytest.raises(ValueError):
        load_components(dict(other=make_components()[0]), arrays, host)
    fewer = {"points": {k: v for k, v in arrays["points"].items() if k != "flag"}}
    with pytest.raises(ValueError):
        load_components(dict(points=make_components()[0]), fewer, host)


class GrowingHost:
    """A component whose host names grow during its first steps (like the fluid's diagnostics)."""

    def __init__(self, names):
        self.values = {name: 0 for name in names}

    def state_host(self):
        return dict(self.values)

    def load_state_host(self, values):
        missing = set(self.values) - set(values)
        if missing:
            raise ValueError("missing %s" % sorted(missing))
        self.values = dict(values)


def test_a_checkpoint_may_hold_host_names_a_fresh_component_creates_later(tmp_path):
    path = tmp_path / "state.npz"
    write_checkpoint(path, dict(fluid=GrowingHost(["a", "b", "late"])), {}, {})
    arrays, host, _, _, _ = read_checkpoint(path)
    fresh = GrowingHost(["a", "b"])
    load_components(dict(fluid=fresh), arrays, host)
    assert set(fresh.values) == {"a", "b", "late"}
    with pytest.raises(ValueError):                              # a name the run has but the checkpoint lacks
        load_components(dict(fluid=GrowingHost(["a", "b", "other"])), arrays, host)


def test_a_fixed_host_component_refuses_extra_saved_names(tmp_path):
    from fastfsi.io.checkpoint import Checkpointable

    class Fixed(Checkpointable):
        CHECKPOINT_HOST = ("a",)

        def __init__(self):
            self.a = 1

    path = tmp_path / "state.npz"
    write_checkpoint(path, dict(c=GrowingHost(["a", "b"])), {}, {})
    arrays, host, _, _, _ = read_checkpoint(path)
    with pytest.raises(ValueError):
        load_components(dict(c=Fixed()), arrays, host)


def test_zero_dimensional_arrays_keep_their_shape(tmp_path):
    scalar = FakeField(np.array(7, dtype=np.int32))
    vector = FakeField(np.array([[1.5, -0.0, np.nan]], dtype=np.float32))
    component = ArrayState(dict(count=scalar, rows=vector))
    path = tmp_path / "state.npz"
    write_checkpoint(path, dict(c=component), {}, {})
    arrays, host, _, _, header = read_checkpoint(path)
    assert arrays["c"]["count"].shape == () and arrays["c"]["count"].dtype == np.int32
    assert [e["shape"] for e in header["arrays"]] == [[], [1, 3]]
    fresh = ArrayState(dict(count=FakeField(np.array(0, dtype=np.int32)), rows=FakeField(np.zeros((1, 3), np.float32))))
    load_components(dict(c=fresh), arrays, host)
    assert int(fresh.fields["count"].array) == 7
    assert fresh.fields["rows"].array.tobytes() == vector.array.tobytes()


def test_host_state_sets_attributes_the_owner_does_not_have_yet(tmp_path):
    class Owner:
        def __init__(self):
            self.early = 1

    saved = Owner()
    saved.late = [1, 2]
    saved.late_float = 0.25
    path = tmp_path / "state.npz"
    write_checkpoint(path, dict(c=HostState(saved, ("early", "late", "late_float"))), {}, {})
    _, host, _, _, _ = read_checkpoint(path)
    fresh = Owner()
    HostState(fresh, tuple(host["c"])).load_state_host(host["c"])
    assert fresh.late == [1, 2] and type(fresh.late_float) is float and fresh.early == 1
