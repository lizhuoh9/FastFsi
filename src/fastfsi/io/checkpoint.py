"""Create-only checkpoints of a simulation's carried state (one NPZ file).

A component is CHECKPOINTABLE when it has
    state_array_names() -> tuple of names      and   state_array(name) -> numpy array      (lazy, one array at a time)
or, for small components,
    state_arrays() -> {name: numpy array}
and, to restore,
    load_state_arrays(arrays)                  (all-or-nothing: every array checked before anything is written).
A component with host-side state (counters, first times, running maxima: Python / NumPy scalars and small containers)
also has
    state_host() -> {name: value}              and   load_state_host(values)
whose values are stored in the JSON header with a type-exact encoding (encode_host / decode_host): a value comes back
with the same type and the same bits (Python float vs numpy.float32, int vs float, bool, None, tuple vs list, set,
dict key types and order, NaN payloads, signed zeros, infinities).  Nothing is ever pickled.

write_checkpoint(path, components, loop_state, manifest)
    components: {component name: checkpointable object}; loop_state: {name: host value} (the time loop's own
    variables, taken inside the loop at the checkpoint, encoded like host state); manifest: JSON-serialisable dict (what
    a restart must match).  Each array is fetched, written into the zip and released before the next one (peak host
    memory = the largest array), its sha256 recorded.  The JSON header (format, creation time, encoded loop state, the
    encoded host state per component, manifest, the array table with dtype / shape / sha256) is the last member,
    "__checkpoint__.json".  The file is written as PATH.partial and renamed to PATH; an existing PATH or PATH.partial
    is refused (create-only).  Returns the header.
read_checkpoint(path) -> (arrays {component: {name: array}}, host {component: {name: value}}, loop_state, manifest,
                          header)
    verifies every array's sha256, dtype and shape against the header and decodes the host values.
compare_manifests(saved, current, ignore=()) -> list of (key path, saved value, current value) differences.

The format is plain NPZ (numpy.load(allow_pickle=False) reads every array); the header is a UTF-8 JSON member.
"""
import datetime
import hashlib
import io
import json
import os
import struct
import zipfile
from pathlib import Path

import numpy as np

FORMAT = "fastfsi-checkpoint-v1"
HEADER = "__checkpoint__.json"
HOST_ARRAY_MAX_BYTES = 1 << 16          # a NumPy array kept as host state goes in the header only up to this size
_SCALAR_KINDS = "biufc"


def encode_host(value, _where="value"):
    """Type-exact JSON encoding of a host value: {"t": type tag, ...}; floats and NumPy scalars by their raw bits."""
    kind = type(value)
    if value is None:
        return {"t": "none"}
    if kind is bool:
        return {"t": "bool", "v": value}
    if kind is int:
        return {"t": "int", "v": value}
    if kind is float:
        return {"t": "float", "bits": struct.pack("<d", value).hex(), "v": repr(value)}
    if kind is str:
        return {"t": "str", "v": value}
    if kind is bytes:
        return {"t": "bytes", "hex": value.hex()}
    if isinstance(value, np.generic) and kind is value.dtype.type and value.dtype.kind in _SCALAR_KINDS:
        return {"t": "numpy_scalar", "dtype": value.dtype.str, "bits": value.tobytes().hex(), "v": repr(value)}
    if kind is np.ndarray and value.dtype.kind in _SCALAR_KINDS:
        if value.nbytes > HOST_ARRAY_MAX_BYTES:
            raise TypeError("%s: a %d-byte array is not host state; save it as a state array" % (_where, value.nbytes))
        data = _c_order(value)
        return {"t": "numpy_array", "dtype": value.dtype.str, "shape": list(value.shape), "bits": data.tobytes().hex()}
    if kind in (tuple, list, set, frozenset):
        items = list(value)
        if kind in (set, frozenset):
            items = sorted(items, key=_order_key)          # a set has no order; store a canonical one
        return {"t": kind.__name__, "items": [encode_host(item, "%s[%d]" % (_where, i)) for i, item in enumerate(items)]}
    if kind is dict:
        return {"t": "dict", "items": [[encode_host(k, "%s key" % _where), encode_host(v, "%s[%r]" % (_where, k))]
                                       for k, v in value.items()]}
    raise TypeError("%s: type %s.%s has no type-exact checkpoint encoding"
                    % (_where, kind.__module__, kind.__qualname__))


def _order_key(item):
    return (type(item).__name__, repr(item))


def decode_host(obj):
    """Inverse of encode_host: the same type and the same bits."""
    tag = obj["t"]
    if tag == "none":
        return None
    if tag in ("bool", "int", "str"):
        value = obj["v"]
        if type(value) is not {"bool": bool, "int": int, "str": str}[tag]:
            raise ValueError("host value tagged %s holds %r" % (tag, value))
        return value
    if tag == "float":
        return struct.unpack("<d", bytes.fromhex(obj["bits"]))[0]
    if tag == "bytes":
        return bytes.fromhex(obj["hex"])
    if tag == "numpy_scalar":
        return np.frombuffer(bytes.fromhex(obj["bits"]), dtype=np.dtype(obj["dtype"]))[0]
    if tag == "numpy_array":
        flat = np.frombuffer(bytes.fromhex(obj["bits"]), dtype=np.dtype(obj["dtype"]))
        return flat.reshape(obj["shape"]).copy()
    if tag in ("tuple", "list", "set", "frozenset"):
        return {"tuple": tuple, "list": list, "set": set, "frozenset": frozenset}[tag](decode_host(i) for i in obj["items"])
    if tag == "dict":
        return {decode_host(k): decode_host(v) for k, v in obj["items"]}
    raise ValueError("unknown host value tag %r" % tag)


def same_host_value(a, b):
    """True when two host values have the same types and the same bits, recursively (NaN equals NaN by bits)."""
    if type(a) is not type(b):
        return False
    if a is None:
        return True
    if isinstance(a, float):
        return struct.pack("<d", a) == struct.pack("<d", b)
    if isinstance(a, np.generic):
        return a.dtype == b.dtype and a.tobytes() == b.tobytes()
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same_host_value(x, y) for x, y in zip(a, b))
    if isinstance(a, (set, frozenset)):
        return encode_host(a) == encode_host(b)
    if isinstance(a, dict):
        return list(a) == list(b) and all(same_host_value(k, l) for k, l in zip(a, b)) \
            and all(same_host_value(a[k], b[k]) for k in a)
    return a == b


_MISSING = object()


class HostState:
    """A checkpointable view of named host attributes of one object.  load_state_host requires exactly the same names,
    then restores each value with its saved type; a dict / list / set attribute whose current value has the saved type
    is refilled in place (other references to it stay valid)."""

    def __init__(self, owner, names):
        self.owner, self.names = owner, tuple(names)

    def state_host(self):
        return {name: getattr(self.owner, name) for name in self.names}

    def load_state_host(self, values):
        if set(values) != set(self.names):
            raise ValueError("host state names %s, expected %s" % (sorted(values), sorted(self.names)))
        for name in self.names:
            value, current = values[name], getattr(self.owner, name, _MISSING)   # may not exist yet
            if type(current) is type(value) and type(value) is list:
                current[:] = value
            elif type(current) is type(value) and type(value) in (dict, set):
                current.clear()
                current.update(value)
            else:
                setattr(self.owner, name, value)


def _c_order(array):
    """The array in C order WITHOUT changing its shape (np.ascontiguousarray turns a 0-d array into shape (1,))."""
    array = np.asarray(array)
    return array if array.flags.c_contiguous else np.array(array, order="C")


def _sha256(array):
    data = _c_order(array)
    return hashlib.sha256(data.tobytes()).hexdigest()


def _array_names(component):
    """(names, eager dict or None) of a component's state arrays; ((), None) for a host-only component."""
    if hasattr(component, "state_array_names"):
        return tuple(component.state_array_names()), None
    if hasattr(component, "state_arrays"):
        arrays = component.state_arrays()
        return tuple(arrays), arrays
    if hasattr(component, "state_host"):
        return (), None
    raise TypeError("%r has neither state arrays nor host state" % (component,))


def write_checkpoint(path, components, loop_state, manifest):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError("checkpoint %s (or its .partial) exists; checkpoints are create-only" % path)
    # host values are encoded first: an unencodable value fails before any file is created
    # (whole dicts: their key order survives the sorted JSON header)
    encoded_loop = encode_host(dict(loop_state), "loop_state")
    encoded_host = {name: encode_host(dict(component.state_host()), name)
                    for name, component in components.items() if hasattr(component, "state_host")}
    table = []
    with zipfile.ZipFile(partial, mode="x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for component_name, component in components.items():
            names, eager = _array_names(component)
            for name in names:
                array = eager[name] if eager is not None else component.state_array(name)
                array = np.asarray(array)
                member = "%s/%s.npy" % (component_name, name)
                with archive.open(member, mode="w", force_zip64=True) as stream:
                    np.lib.format.write_array(stream, _c_order(array), allow_pickle=False)
                table.append(dict(component=component_name, name=name, member=member, dtype=array.dtype.str,
                                  shape=list(array.shape), sha256=_sha256(array), bytes=int(array.nbytes)))
                del array
            del eager
        header = dict(format=FORMAT, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      loop_state=encoded_loop, host_state=encoded_host, manifest=manifest, arrays=table)
        archive.writestr(HEADER, json.dumps(header, indent=1, sort_keys=True, allow_nan=False))
    _publish_create_only(partial, path)
    return header


def _publish_create_only(partial, path):
    """Give the finished .partial its final name without ever replacing a file: a hard link fails if `path` appeared
    while the checkpoint was written (FileExistsError; the .partial stays for inspection), where os.replace would have
    overwritten it.  A file system without hard links falls back to the re-checked rename."""
    try:
        os.link(partial, path)
    except FileExistsError:
        raise FileExistsError("checkpoint %s appeared while it was written; checkpoints are create-only (the new one "
                              "is kept as %s)" % (path, partial)) from None
    except OSError:
        if path.exists():
            raise FileExistsError("checkpoint %s appeared while it was written; checkpoints are create-only (the new "
                                  "one is kept as %s)" % (path, partial)) from None
        os.replace(partial, path)
        return
    os.unlink(partial)


def read_checkpoint(path):
    path = Path(path)
    with zipfile.ZipFile(path, mode="r") as archive:
        header = json.loads(archive.read(HEADER).decode("utf-8"))
        if header.get("format") != FORMAT:
            raise ValueError("unknown checkpoint format %r" % header.get("format"))
        arrays = {}
        for entry in header["arrays"]:
            with archive.open(entry["member"], mode="r") as stream:
                array = np.lib.format.read_array(io.BufferedReader(stream), allow_pickle=False)
            if array.dtype.str != entry["dtype"] or list(array.shape) != entry["shape"]:
                raise ValueError("checkpoint array %s: %s %s, header says %s %s"
                                 % (entry["member"], array.dtype.str, list(array.shape), entry["dtype"], entry["shape"]))
            if _sha256(array) != entry["sha256"]:
                raise ValueError("checkpoint array %s: sha256 mismatch" % entry["member"])
            arrays.setdefault(entry["component"], {})[entry["name"]] = array
    host = {component: decode_host(values) for component, values in header["host_state"].items()}
    loop_state = decode_host(header["loop_state"])
    return arrays, host, loop_state, header["manifest"], header


def load_components(components, arrays, host):
    """Restore every component from read_checkpoint's arrays and host values.  The checkpoint must hold exactly the
    run's components, each with exactly its array names, and every host name the component has now (a component may
    create host attributes during its first steps, so a checkpoint may hold more; the component's load_state_host
    decides about those).  All of this is checked before anything is loaded; shapes and dtypes are checked inside each
    component's load_state_arrays before it writes."""
    problems = []
    for name in sorted(set(arrays) | set(host)):
        if name not in components:
            problems.append("the checkpoint holds component %s, the run has none" % name)
    for name, component in components.items():
        expected = set(_array_names(component)[0])
        if expected != set(arrays.get(name, {})):
            problems.append("%s arrays: checkpoint %s, run %s" % (name, sorted(arrays.get(name, {})), sorted(expected)))
        has_host = hasattr(component, "state_host")
        missing = (set(component.state_host()) if has_host else set()) - set(host.get(name, {}))
        if missing or has_host != (name in host):
            problems.append("%s host state: checkpoint %s, run has %s" % (
                name, sorted(host.get(name, {})) if name in host else None, sorted(missing) if has_host else None))
    if problems:
        raise ValueError("checkpoint does not match the run: " + "; ".join(problems))
    for name, component in components.items():
        if hasattr(component, "load_state_arrays"):
            component.load_state_arrays(arrays.get(name, {}))
        if name in host:
            component.load_state_host(host[name])


ABSENT = "<absent>"                    # compare_manifests' value for a key present on one side only


def compare_manifests(saved, current, ignore=(), _path=""):
    """Differences between two manifests (nested dicts / lists / scalars), skipping the key paths in `ignore`; a key
    present on one side only is reported with ABSENT on the other side."""
    diffs = []
    if _path in ignore:
        return diffs
    if isinstance(saved, dict) and isinstance(current, dict):
        for key in sorted(set(saved) | set(current)):
            sub = "%s/%s" % (_path, key) if _path else str(key)
            if sub in ignore:
                continue
            if key not in saved or key not in current:
                diffs.append((sub, saved.get(key, ABSENT), current.get(key, ABSENT)))
            else:
                diffs.extend(compare_manifests(saved[key], current[key], ignore, sub))
    elif isinstance(saved, list) and isinstance(current, list) and len(saved) == len(current):
        for index, (a, b) in enumerate(zip(saved, current)):
            diffs.extend(compare_manifests(a, b, ignore, "%s/%d" % (_path, index)))
    elif saved != current:
        diffs.append((_path, saved, current))
    return diffs


class ArrayState:
    """A checkpointable view of named Taichi fields of one object (lazy: one field read at a time).
    fields: {name: field}; load_state_arrays checks every shape and dtype first, then writes all fields."""

    def __init__(self, fields):
        self.fields = dict(fields)

    def state_array_names(self):
        return tuple(self.fields)

    def state_array(self, name):
        return self.fields[name].to_numpy()

    def load_state_arrays(self, arrays):
        staged = []
        for name, field in self.fields.items():
            if name not in arrays:
                raise ValueError("checkpoint is missing %s" % name)
            value = np.asarray(arrays[name])
            current = field.to_numpy()                 # one host copy per field, only to check shape and dtype
            if value.dtype != current.dtype or value.shape != current.shape:
                raise ValueError("checkpoint %s is %s %s, expected %s %s"
                                 % (name, value.shape, value.dtype, current.shape, current.dtype))
            del current
            staged.append((field, value))
        for field, value in staged:
            field.from_numpy(value)


class Checkpointable:
    """Mixin for a component whose carried state is a set of its own Taichi fields (CHECKPOINT_FIELDS, attribute
    names) and host attributes (CHECKPOINT_HOST); provides the checkpoint protocol over them.  _after_load_state() (a
    no-op here) runs after the fields are loaded, for derived fields recomputed from the loaded ones."""
    CHECKPOINT_FIELDS = ()
    CHECKPOINT_HOST = ()

    def state_array_names(self):
        return tuple(self.CHECKPOINT_FIELDS)

    def state_array(self, name):
        if name not in self.CHECKPOINT_FIELDS:
            raise KeyError("%s is not a checkpoint field of %s" % (name, type(self).__name__))
        return getattr(self, name).to_numpy()

    def load_state_arrays(self, arrays):
        ArrayState({name: getattr(self, name) for name in self.CHECKPOINT_FIELDS}).load_state_arrays(arrays)
        self._after_load_state()

    def _after_load_state(self):
        return None

    def state_host(self):
        return HostState(self, self.CHECKPOINT_HOST).state_host()

    def load_state_host(self, values):
        HostState(self, self.CHECKPOINT_HOST).load_state_host(values)
