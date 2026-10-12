"""Fast-mode VTK export (Helper 2, 2026-10-12; Root REQ_409; USER 「为什么要写渲染程序啊，不能直接复用别人的吗？」).

VTK XML files for ParaView / PyVista, written by the squid fast-mode driver at its diagnostics points from host numpy
arrays (nothing on the per-step hot path).

REUSED CONVENTIONS (cases/squid_soft_robot/snapshots.py): structured data are flattened in VTK's x-fastest order
(_vtk_point_data_values: transpose(2, 1, 0[, 3])); the graded grid is a RectilinearGrid (.vtr) whose coordinates are
the CELL CENTRES, with the cell values as point data (velocity_mps, speed_mps, obstacle, ...).  No new format: VTR for
the fluid (full 3-D and the two single-layer axis planes), VTP (PolyData, one vertex cell per point) for the material
points, PVD collections for the time series.

ENCODING: VTK XML version 1.0, byte_order="LittleEndian", header_type="UInt64", AppendedData encoding="raw" (no base64).
compress=True: each array is zlib-compressed in 32768-byte blocks with VTK's standard block header
(compressor="vtkZLibDataCompressor"): [number of blocks, block size, size of the last block if partial else 0,
compressed size of each block] as UInt64, then the compressed blocks.  compress=False: [byte count] (UInt64) + raw data.

FIELDS
  fluid (3-D and planes), point data on the cell centres:
    velocity_mps (Float32 x3) = the solver's MAC velocity CENTRED on the cell: the mean of the two faces of the cell
      along each component (the solver stores component c on the backward face of each cell along c; the far-boundary
      face, which it does not store, is a closed wall: 0).  The existing debug writer wrote the raw backward-face values;
      speed_mps = |velocity_mps|; pressure_pa = the PHYSICAL pressure given by the caller (p~ + p_shift); obstacle
      (rigid regions + the user's chamber-wall groove, as fluid.obstacle) and air (the air-chamber cells) as two
      separate Int32 0/1 masks: air is never merged into obstacle (Root REQ_410).  A plane is the layer j = plane_j
      (x-z) or i = plane_i (y-z) of the same centred 3-D fields (its normal component uses the faces j and j + 1).
  points: Points = X0 + u summed in float64 on the host; point data X0_m (x3), displacement_m (x3), J, body_id,
    tag_air_face (tag bit 1, value 1), tag_valve_rim (tag bit 2, value 2), tether.
  every file: FieldData TimeValue (s), step, label_utf8 and geometry_note_utf8 (Root REQ_410: the run label and the
    driver's geometry note as UInt8 arrays of their UTF-8 bytes; decode with bytes(array).decode('utf-8')); the label
    is also an XML comment.  vtk_series_meta.json (create-only) holds both strings, the encoding and the field list;
    the .pvd files carry them as an XML comment.

CREATE-ONLY: every data file is written once (mode 'x', a per-kind counter in the name, so a repeated step never
collides); vtk_series.jsonl is an append-only index (one line per data file, appended after the file is complete); the
.pvd collections are written once by finalize(); after a kill, make_pvd(index_path, out_dir) (CLI:
python -m simulation_core.fast.vtk_export_fast make-pvd INDEX OUT_DIR) rebuilds them from the index.

REUSE CHECK (REQ_400): the codebase's own writer (above) for the ordering, grid type and names; the VTK XML file-format
specification (Kitware) for header_type, AppendedData and the vtkZLibDataCompressor block layout, implemented from the
specification (no VTK code copied).  vtk / pyvista are not installed in WSL (no installs), so the CPU test reads the
files back with an independent minimal reader; the integrator has Root confirm with real VTK on Windows.
"""
import hashlib
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

VTK_EXPORT_REVISION = "REQ409-vtk-export-fast-v1"
BLOCK_SIZE = 32768                      # VTK's default compressed block size (bytes, uncompressed)
TAG_AIR_FACE = 1                        # simulation_core.fast.ib_mpm_fast3d.TAG_AIR_FACE (bit 1)
TAG_VALVE_RIM = 2                       # simulation_core.fast.ib_mpm_fast3d.TAG_VALVE_RIM (bit 2)
DEFAULT_LABEL = "UNVALIDATED engineering run (fast mode, diffuse IB-MPM)"
SUBDIRS = {"fluid_3d": "fluid_3d", "plane_xz": "planes_xz", "plane_yz": "planes_yz", "points": "points"}
PVD_NAMES = {"fluid_3d": "fluid_3d.pvd", "plane_xz": "planes_xz.pvd", "plane_yz": "planes_yz.pvd",
             "points": "points.pvd"}
_VTK_TYPES = {np.dtype(np.float32): "Float32", np.dtype(np.float64): "Float64", np.dtype(np.int8): "Int8",
              np.dtype(np.uint8): "UInt8", np.dtype(np.int16): "Int16", np.dtype(np.uint16): "UInt16",
              np.dtype(np.int32): "Int32", np.dtype(np.uint32): "UInt32", np.dtype(np.int64): "Int64",
              np.dtype(np.uint64): "UInt64"}


# ---------------------------------------------------------------------------------------------------- encoding
def vtk_flat(values):
    """VTK x-fastest flattening (cases/squid_soft_robot/snapshots._vtk_point_data_values; 2-D = (n, components))."""
    array = np.asarray(values)
    if array.ndim == 3:
        return array.transpose(2, 1, 0).reshape(-1)
    if array.ndim == 4:
        return array.transpose(2, 1, 0, 3).reshape(-1)
    if array.ndim in (1, 2):
        return array.reshape(-1)
    raise ValueError("VTK data must have 1, 2, 3 or 4 dimensions")


def encode_array(values, *, compress, level):
    """(VTK type name, appended-raw payload: header + data) of one DataArray, little-endian."""
    flat = vtk_flat(values)
    dtype = np.dtype(flat.dtype)
    if dtype not in _VTK_TYPES:
        raise ValueError("no VTK type for dtype %s" % dtype)
    raw = np.ascontiguousarray(flat, dtype=dtype.newbyteorder("<")).tobytes()
    if not compress:
        return _VTK_TYPES[dtype], np.array([len(raw)], dtype="<u8").tobytes() + raw
    view = memoryview(raw)
    blocks = [zlib.compress(view[i:i + BLOCK_SIZE], level) for i in range(0, len(raw), BLOCK_SIZE)]
    header = np.array([len(blocks), BLOCK_SIZE, len(raw) % BLOCK_SIZE] + [len(b) for b in blocks], dtype="<u8")
    return _VTK_TYPES[dtype], header.tobytes() + b"".join(blocks)


class _Appended:
    """Collects DataArray tags and their appended-raw payloads (offsets relative to the byte after '_')."""

    def __init__(self, compress, level):
        self.compress, self.level = compress, level
        self.chunks, self.offset = [], 0

    def add_encoded(self, vtk_type, payload, name, components=1, tuples=None):
        attrs = ' NumberOfComponents="%d"' % components if components != 1 else ""
        if tuples is not None:
            attrs += ' NumberOfTuples="%d"' % tuples
        tag = '<DataArray type="%s" Name="%s"%s format="appended" offset="%d"/>' % (vtk_type, name, attrs,
                                                                                   self.offset)
        self.chunks.append(payload)
        self.offset += len(payload)
        return tag

    def add(self, name, values, components=1, tuples=None):
        vtk_type, payload = encode_array(values, compress=self.compress, level=self.level)
        return self.add_encoded(vtk_type, payload, name, components, tuples)


def _file_bytes(vtk_type, label, body_xml, appended, compress):
    compressor = ' compressor="vtkZLibDataCompressor"' if compress else ""
    head = ('<?xml version="1.0"?>\n<!-- %s -->\n<VTKFile type="%s" version="1.0" byte_order="LittleEndian" '
            'header_type="UInt64"%s>\n%s  <AppendedData encoding="raw">\n   _'
            % (label.replace("--", "- -"), vtk_type, compressor, body_xml))
    return head.encode("utf-8") + b"".join(appended.chunks) + b"\n  </AppendedData>\n</VTKFile>\n"


def utf8_array(text):
    """A string as a UInt8 array of its UTF-8 bytes (FieldData; read back with bytes(array).decode('utf-8'))."""
    data = str(text).encode("utf-8") or b"(none)"
    return np.frombuffer(data, dtype=np.uint8).copy()


# ---------------------------------------------------------------------------------------------------- fields
def centred_velocity(velocity_mac):
    """Cell-centred Float32 velocity from the solver's MAC storage (component c on the backward face of each cell
    along c): mean of the two faces along c; the far-boundary face (not stored) is a closed wall (0)."""
    v = np.asarray(velocity_mac)
    if v.ndim != 4 or v.shape[-1] != 3:
        raise ValueError("velocity must be (nx, ny, nz, 3), got %s" % (v.shape,))
    out = np.empty(v.shape, dtype=np.float32)
    for c in range(3):
        a = v[..., c].astype(np.float64)
        following = np.zeros_like(a)
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        src[c], dst[c] = slice(1, None), slice(0, -1)
        following[tuple(dst)] = a[tuple(src)]
        out[..., c] = (0.5 * (a + following)).astype(np.float32)
    return out


def speed_of(velocity_centred):
    return np.sqrt(np.sum(np.asarray(velocity_centred, dtype=np.float64) ** 2, axis=-1)).astype(np.float32)


def _pressure_array(pressure):
    p = np.asarray(pressure)
    return p if p.dtype in (np.dtype(np.float32), np.dtype(np.float64)) else p.astype(np.float64)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------------------------------- writer
class VtkSeriesWriter:
    """Create-only VTK series of the squid fast-mode run (fluid 3-D, the two axis planes, the material points)."""

    def __init__(self, out_dir, *, faces_x, faces_y, faces_z, plane_i, plane_j, X0, body_id, tag, tether,
                 compress=True, compress_level=1, label=DEFAULT_LABEL, geometry_note=""):
        self.out_dir = Path(out_dir)
        self.compress, self.level = bool(compress), int(compress_level)
        self.label, self.geometry_note = str(label), str(geometry_note)
        faces = [np.asarray(f, dtype=np.float64).reshape(-1) for f in (faces_x, faces_y, faces_z)]
        for name, f in zip("xyz", faces):
            if f.size < 2 or not np.all(np.isfinite(f)) or np.any(np.diff(f) <= 0.0):
                raise ValueError("faces_%s must be finite and strictly increasing" % name)
        self.centres = [0.5 * (f[:-1] + f[1:]) for f in faces]
        self.shape = tuple(len(c) for c in self.centres)
        self.plane_i, self.plane_j = int(plane_i), int(plane_j)
        if not (0 <= self.plane_i < self.shape[0] and 0 <= self.plane_j < self.shape[1]):
            raise ValueError("plane indices (%d, %d) outside the grid %s" % (self.plane_i, self.plane_j, self.shape))
        X0 = np.asarray(X0)
        n = X0.shape[0]
        if X0.ndim != 2 or X0.shape[1] != 3 or n == 0:
            raise ValueError("X0 must be (n, 3)")
        body_id, tag, tether = (np.asarray(a).reshape(-1) for a in (body_id, tag, tether))
        if not (body_id.shape == tag.shape == tether.shape == (n,)):
            raise ValueError("body_id, tag and tether must have one entry per point")
        self.n_points = n
        self._X0_64 = X0.astype(np.float64)
        self.out_dir.mkdir(parents=True, exist_ok=False)
        for sub in SUBDIRS.values():
            (self.out_dir / sub).mkdir()
        enc = dict(compress=self.compress, level=self.level)
        # static payloads, encoded once and reused in every file
        self._coords = {
            "fluid_3d": [encode_array(c, **enc) for c in self.centres],
            "plane_xz": [encode_array(self.centres[0], **enc),
                         encode_array(self.centres[1][self.plane_j:self.plane_j + 1], **enc),
                         encode_array(self.centres[2], **enc)],
            "plane_yz": [encode_array(self.centres[0][self.plane_i:self.plane_i + 1], **enc),
                         encode_array(self.centres[1], **enc), encode_array(self.centres[2], **enc)]}
        tag = tag.astype(np.int64)
        self._points_static = [
            ("X0_m", 3, encode_array(X0, **enc)),
            ("body_id", 1, encode_array(body_id.astype(np.int32), **enc)),
            ("tag_air_face", 1, encode_array(((tag & TAG_AIR_FACE) != 0).astype(np.int32), **enc)),
            ("tag_valve_rim", 1, encode_array(((tag & TAG_VALVE_RIM) != 0).astype(np.int32), **enc)),
            ("tether", 1, encode_array(tether.astype(np.int32), **enc))]
        self._verts = [("connectivity", encode_array(np.arange(n, dtype=np.int64), **enc)),
                       ("offsets", encode_array(np.arange(1, n + 1, dtype=np.int64), **enc))]
        label_bytes, note_bytes = utf8_array(self.label), utf8_array(self.geometry_note)
        self._field_static = [("label_utf8", len(label_bytes), encode_array(label_bytes, **enc)),
                              ("geometry_note_utf8", len(note_bytes), encode_array(note_bytes, **enc))]
        meta = dict(revision=VTK_EXPORT_REVISION, label=self.label, geometry_note=self.geometry_note,
                    encoding=dict(vtk_xml_version="1.0", header_type="UInt64", byte_order="LittleEndian",
                                  data="AppendedData raw",
                                  compressor="vtkZLibDataCompressor" if self.compress else None,
                                  block_size_bytes=BLOCK_SIZE, zlib_level=self.level),
                    grid=dict(cells=list(self.shape), plane_i=self.plane_i, plane_j=self.plane_j,
                              coordinates="cell centres of the graded grid (RectilinearGrid)"),
                    points=n, field_data=["TimeValue", "step", "label_utf8", "geometry_note_utf8"],
                    fluid_fields=["velocity_mps (cell-centred from the MAC faces)", "speed_mps",
                                  "pressure_pa (physical)", "obstacle (rigid regions + groove)",
                                  "air (air-chamber cells, own mask, never merged into obstacle)"],
                    point_fields=["Points = X0 + u (float64)", "X0_m", "displacement_m", "J", "body_id",
                                  "tag_air_face", "tag_valve_rim", "tether"])
        with open(self.out_dir / "vtk_series_meta.json", "x", encoding="utf-8") as stream:
            stream.write(json.dumps(meta, indent=1) + "\n")
        self._index_path = self.out_dir / "vtk_series.jsonl"
        self._index = open(self._index_path, "x", encoding="utf-8")
        self._entries = []
        self._counters = {kind: 0 for kind in SUBDIRS}
        self._finalized = None

    # -------------------------------------------------------------------------------------------- internals
    def _field_data(self, appended, t_s, step):
        rows = [appended.add("TimeValue", np.array([float(t_s)], dtype=np.float64), tuples=1),
                appended.add("step", np.array([int(step)], dtype=np.int64), tuples=1)]
        rows += [appended.add_encoded(vtk_type, payload, name, tuples=count)
                 for name, count, (vtk_type, payload) in self._field_static]
        return "    <FieldData>\n%s    </FieldData>\n" % "".join("      %s\n" % r for r in rows)

    def _emit(self, kind, step, t_s, suffix, data, started):
        number = self._counters[kind]
        self._counters[kind] += 1
        relative = "%s/%s_%05d_s%07d.%s" % (SUBDIRS[kind], kind, number, int(step), suffix)
        path = self.out_dir / relative
        with open(path, "xb") as stream:
            stream.write(data)
        entry = dict(kind=kind, step=int(step), t_s=float(t_s), file=relative, bytes=len(data), sha256=_sha256(data),
                     write_s=time.perf_counter() - started, unix_s=time.time())
        self._index.write(json.dumps(entry) + "\n")
        self._index.flush()
        self._entries.append(entry)
        return path, len(data)

    def _grid_file(self, kind, step, t_s, block, extent, started):
        """block: dict of the five point-data arrays shaped (nx', ny', nz'[, 3])."""
        appended = _Appended(self.compress, self.level)
        field = self._field_data(appended, t_s, step)
        rows = [appended.add("velocity_mps", block["velocity_mps"], components=3),
                appended.add("speed_mps", block["speed_mps"]),
                appended.add("pressure_pa", block["pressure_pa"]),
                appended.add("obstacle", block["obstacle"]),
                appended.add("air", block["air"])]
        coords = [appended.add_encoded(vtk_type, payload, "%s_coordinates_m" % axis)
                  for axis, (vtk_type, payload) in zip("xyz", self._coords[kind])]
        body = ('  <RectilinearGrid WholeExtent="%s">\n%s    <Piece Extent="%s">\n'
                '      <PointData Scalars="speed_mps" Vectors="velocity_mps">\n%s      </PointData>\n'
                '      <CellData>\n      </CellData>\n      <Coordinates>\n%s      </Coordinates>\n'
                '    </Piece>\n  </RectilinearGrid>\n'
                % (extent, field, extent, "".join("        %s\n" % r for r in rows),
                   "".join("        %s\n" % c for c in coords)))
        data = _file_bytes("RectilinearGrid", self.label, body, appended, self.compress)
        return self._emit(kind, step, t_s, "vtr", data, started)

    def _check_fluid(self, velocity, pressure, obstacle, air):
        shape = self.shape
        if np.shape(velocity) != shape + (3,):
            raise ValueError("velocity shape %s != %s" % (np.shape(velocity), shape + (3,)))
        for name, a in (("pressure", pressure), ("obstacle", obstacle), ("air", air)):
            if np.shape(a) != shape:
                raise ValueError("%s shape %s != %s" % (name, np.shape(a), shape))

    @staticmethod
    def _block(velocity_mac, pressure, obstacle, air):
        vc = centred_velocity(velocity_mac)
        return dict(velocity_mps=vc, speed_mps=speed_of(vc), pressure_pa=_pressure_array(pressure),
                    obstacle=(np.asarray(obstacle) != 0).astype(np.int32), air=(np.asarray(air) != 0).astype(np.int32))

    # -------------------------------------------------------------------------------------------- public API
    def write_fluid_3d(self, step, t_s, velocity, pressure_phys, obstacle, air):
        """Full 3-D fluid (VTR).  velocity: the solver's MAC storage (nx, ny, nz, 3); pressure_phys: PHYSICAL pressure
        (p~ + p_shift); obstacle, air: (nx, ny, nz) 0/1.  Returns (path, bytes)."""
        started = time.perf_counter()
        self._check_fluid(velocity, pressure_phys, obstacle, air)
        nx, ny, nz = self.shape
        extent = "0 %d 0 %d 0 %d" % (nx - 1, ny - 1, nz - 1)
        return self._grid_file("fluid_3d", step, t_s, self._block(velocity, pressure_phys, obstacle, air), extent,
                               started)

    def write_planes(self, step, t_s, velocity, pressure_phys, obstacle, air):
        """The x-z layer j = plane_j and the y-z layer i = plane_i of the same fields (single-layer VTRs) from the full
        3-D host arrays.  Returns [(path_xz, bytes), (path_yz, bytes)]."""
        self._check_fluid(velocity, pressure_phys, obstacle, air)
        nx, ny, nz = self.shape
        i, j = self.plane_i, self.plane_j
        out = []
        started = time.perf_counter()
        v2 = np.asarray(velocity)[:, j:j + 2]                       # faces j and j + 1 for the y component
        block = self._block(v2, np.asarray(pressure_phys)[:, j:j + 2], np.asarray(obstacle)[:, j:j + 2],
                            np.asarray(air)[:, j:j + 2])
        block = {k: v[:, 0:1] for k, v in block.items()}
        out.append(self._grid_file("plane_xz", step, t_s, block, "0 %d %d %d 0 %d" % (nx - 1, j, j, nz - 1),
                                   started))
        started = time.perf_counter()
        v2 = np.asarray(velocity)[i:i + 2]
        block = self._block(v2, np.asarray(pressure_phys)[i:i + 2], np.asarray(obstacle)[i:i + 2],
                            np.asarray(air)[i:i + 2])
        block = {k: v[0:1] for k, v in block.items()}
        out.append(self._grid_file("plane_yz", step, t_s, block, "%d %d 0 %d 0 %d" % (i, i, ny - 1, nz - 1),
                                   started))
        return out

    def write_points(self, step, t_s, u, J):
        """Material points (VTP): Points = X0 + u (float64), point data X0_m, displacement_m, J, body_id, tag bits,
        tether.  Returns (path, bytes)."""
        started = time.perf_counter()
        u = np.asarray(u)
        J = np.asarray(J).reshape(-1)
        n = self.n_points
        if u.shape != (n, 3) or J.shape != (n,):
            raise ValueError("u must be (%d, 3) and J (%d,), got %s and %s" % (n, n, u.shape, J.shape))
        appended = _Appended(self.compress, self.level)
        field = self._field_data(appended, t_s, step)
        rows = [appended.add_encoded(self._points_static[0][2][0], self._points_static[0][2][1], "X0_m", 3),
                appended.add("displacement_m", u, components=3), appended.add("J", J)]
        rows += [appended.add_encoded(vtk_type, payload, name, comps)
                 for name, comps, (vtk_type, payload) in self._points_static[1:]]
        points = appended.add("Points", self._X0_64 + u.astype(np.float64), components=3)
        verts = [appended.add_encoded(vtk_type, payload, name) for name, (vtk_type, payload) in self._verts]
        body = ('  <PolyData>\n%s    <Piece NumberOfPoints="%d" NumberOfVerts="%d" NumberOfLines="0" '
                'NumberOfStrips="0" NumberOfPolys="0">\n'
                '      <PointData Scalars="J" Vectors="displacement_m">\n%s      </PointData>\n'
                '      <CellData>\n      </CellData>\n      <Points>\n        %s\n      </Points>\n'
                '      <Verts>\n%s      </Verts>\n    </Piece>\n  </PolyData>\n'
                % (field, n, n, "".join("        %s\n" % r for r in rows), points,
                   "".join("        %s\n" % v for v in verts)))
        data = _file_bytes("PolyData", self.label, body, appended, self.compress)
        return self._emit("points", step, t_s, "vtp", data, started)

    def entries(self):
        return [dict(e) for e in self._entries]

    def finalize(self):
        """Write the .pvd collections once (idempotent: a second call returns the same paths, nothing rewritten)."""
        if self._finalized is None:
            self._index.close()
            self._finalized = write_pvds(self._entries, self.out_dir, self.out_dir,
                                         comment="%s | %s" % (self.label, self.geometry_note))
        return list(self._finalized)


# ---------------------------------------------------------------------------------------------------- PVD
def write_pvds(entries, data_dir, out_dir, comment=""):
    """One .pvd per kind present, create-only; DataSet file paths relative to out_dir; comment = an XML comment
    (the run label and geometry note)."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    written = []
    for kind, name in PVD_NAMES.items():
        rows = sorted((e for e in entries if e["kind"] == kind), key=lambda e: (e["t_s"], e["step"], e["file"]))
        if not rows:
            continue
        lines = ['<?xml version="1.0"?>']
        if comment:
            lines.append("<!-- %s -->" % comment.replace("--", "- -"))
        lines += ['<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">', "  <Collection>"]
        for e in rows:
            relative = os.path.relpath(data_dir / e["file"], out_dir).replace(os.sep, "/")
            lines.append('    <DataSet timestep="%s" group="" part="0" file="%s"/>' % (repr(float(e["t_s"])),
                                                                                       relative))
        lines += ["  </Collection>", "</VTKFile>", ""]
        path = out_dir / name
        with open(path, "x", encoding="utf-8") as stream:
            stream.write("\n".join(lines))
        written.append(path)
    return written


def read_index(index_path):
    """Entries of vtk_series.jsonl whose file exists; a truncated or malformed line (a kill while appending) is
    skipped and counted."""
    index_path = Path(index_path)
    entries, skipped = [], 0
    for line in index_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
            if not isinstance(entry, dict) or not (index_path.parent / entry["file"]).is_file():
                raise ValueError
        except (ValueError, KeyError, TypeError):
            skipped += 1
            continue
        entries.append(entry)
    return entries, skipped


def make_pvd(index_path, out_dir):
    """Rebuild the .pvd collections from the index after a kill (create-only: out_dir must not hold them yet).  The
    label and geometry note come from vtk_series_meta.json next to the index when present."""
    entries, skipped = read_index(index_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comment = ""
    meta_path = Path(index_path).parent / "vtk_series_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        comment = "%s | %s" % (meta.get("label", ""), meta.get("geometry_note", ""))
    return write_pvds(entries, Path(index_path).parent, out_dir, comment=comment), skipped


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3 or argv[0] != "make-pvd":
        print("usage: python -m simulation_core.fast.vtk_export_fast make-pvd INDEX_JSONL OUT_DIR", file=sys.stderr)
        return 2
    paths, skipped = make_pvd(argv[1], argv[2])
    print("MAKE_PVD %s (skipped index lines: %d)" % (" ".join(str(p) for p in paths), skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
