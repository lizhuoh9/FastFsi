"""Fast-mode VTK export (simulation_core/fast/vtk_export_fast.py; Root REQ_409): CPU tests.

vtk / pyvista are not installed in WSL, so every file is read back by an INDEPENDENT minimal reader written here from
the VTK XML format specification (parse the XML header, follow the appended-raw offsets, check the UInt64 block header,
zlib-decode the blocks) and compared BITWISE with the inputs: x-fastest ordering, graded cell-centre coordinates, the
MAC-to-cell-centre velocity (expected values computed independently with np.pad), physical pressure, masks, the
single-layer planes, the points (X0 + u in float64), the vertex cells, FieldData, the PVD collections, the create-only
rules and the rebuild of the PVDs from the append-only index after a kill.  The last test writes one file of each kind
on the squid shapes (92 x 92 x 186 cells, 332 k points) and prints the write time and size per file.
"""
import json
import time
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pytest

from simulation_core.fast.vtk_export_fast import (BLOCK_SIZE, VtkSeriesWriter, encode_array, make_pvd, read_index)

TYPES = {"Float32": "<f4", "Float64": "<f8", "Int8": "<i1", "UInt8": "<u1", "Int32": "<i4", "Int64": "<i8",
         "UInt32": "<u4", "UInt64": "<u8"}


# ---------------------------------------------------------------------------------------------------- reader
def decode_payload(data, pos, compressed):
    """Bytes of one appended-raw DataArray at absolute position pos (UInt64 headers)."""
    if not compressed:
        n = int(np.frombuffer(data, "<u8", 1, pos)[0])
        return data[pos + 8:pos + 8 + n]
    nblocks, block, last = (int(v) for v in np.frombuffer(data, "<u8", 3, pos))
    sizes = np.frombuffer(data, "<u8", nblocks, pos + 24).astype(np.int64)
    p = pos + 24 + 8 * nblocks
    out = []
    for k, size in enumerate(sizes):
        chunk = zlib.decompress(data[p:p + int(size)])
        p += int(size)
        expected = last if (k == nblocks - 1 and last) else block
        assert len(chunk) == expected, (k, len(chunk), expected)
        out.append(chunk)
    return b"".join(out)


def read_vtk(path):
    """(root element, {name: array (tuples, components) or (tuples,)}) of a VTK XML file with appended raw data."""
    data = path.read_bytes()
    marker = b'<AppendedData encoding="raw">'
    a = data.index(marker)
    blob = data.index(b"_", a + len(marker)) + 1
    root = ET.fromstring(data[:a].decode("utf-8") + "</VTKFile>")
    assert root.get("version") == "1.0" and root.get("header_type") == "UInt64"
    assert root.get("byte_order") == "LittleEndian"
    compressed = root.get("compressor") == "vtkZLibDataCompressor"
    arrays = {}
    for element in root.iter("DataArray"):
        assert element.get("format") == "appended"
        raw = decode_payload(data, blob + int(element.get("offset")), compressed)
        values = np.frombuffer(raw, dtype=TYPES[element.get("type")])
        comps = int(element.get("NumberOfComponents", "1"))
        name = element.get("Name")
        assert name not in arrays
        arrays[name] = values.reshape(-1, comps) if comps > 1 else values
    assert data.endswith(b"\n  </AppendedData>\n</VTKFile>\n")
    return root, arrays


def grid_array(arrays, name, shape, comps=1):
    """Undo the x-fastest flattening: flat -> (nx, ny, nz[, comps])."""
    nx, ny, nz = shape
    flat = arrays[name]
    if comps == 1:
        return flat.reshape(nz, ny, nx).transpose(2, 1, 0)
    return flat.reshape(nz, ny, nx, comps).transpose(2, 1, 0, 3)


# ---------------------------------------------------------------------------------------------------- data
def graded_faces(n, lo, rng):
    widths = rng.uniform(0.5e-3, 2.0e-3, size=n)
    return lo + np.concatenate([[0.0], np.cumsum(widths)])


def fixture(seed=0, shape=(5, 4, 6), n_points=50, pressure_dtype=np.float32):
    rng = np.random.default_rng(seed)
    faces = [graded_faces(n, lo, rng) for n, lo in zip(shape, (-0.08, -0.03, 0.9))]
    velocity = rng.normal(size=shape + (3,)).astype(np.float32)
    pressure = rng.normal(scale=100.0, size=shape).astype(pressure_dtype)
    obstacle = (rng.random(shape) < 0.3).astype(np.int32)
    air = (rng.random(shape) < 0.2).astype(np.int8)
    X0 = (rng.normal(size=(n_points, 3)) * 0.01 + [-0.03, 0.016, 1.01]).astype(np.float32)
    u = (rng.normal(size=(n_points, 3)) * 1e-4).astype(np.float32)
    J = (1.0 + rng.normal(size=n_points) * 1e-3).astype(np.float32)
    body_id = rng.integers(1, 3, size=n_points).astype(np.int32)
    tag = rng.integers(0, 4, size=n_points).astype(np.int32)
    tether = (rng.random(n_points) < 0.1).astype(np.int32)
    return dict(faces=faces, velocity=velocity, pressure=pressure, obstacle=obstacle, air=air, X0=X0, u=u, J=J,
                body_id=body_id, tag=tag, tether=tether)


LABEL = "UNVALIDATED engineering run, proto_test, grid synthetic, solver none"
NOTE = "chamber-wall groove widens the refill annulus 2.0 -> 5.0 mm (user 2026-10-03), 槽"


def writer_for(out, d, compress=True, plane=(2, 1)):
    return VtkSeriesWriter(out, faces_x=d["faces"][0], faces_y=d["faces"][1], faces_z=d["faces"][2], plane_i=plane[0],
                           plane_j=plane[1], X0=d["X0"], body_id=d["body_id"], tag=d["tag"], tether=d["tether"],
                           compress=compress, label=LABEL, geometry_note=NOTE)


def assert_strings(arrays):
    assert bytes(arrays["label_utf8"]).decode("utf-8") == LABEL
    assert bytes(arrays["geometry_note_utf8"]).decode("utf-8") == NOTE


def expected_centred(v):
    """Independent MAC -> cell-centre velocity: mean with the next face, the missing far face = 0 (np.pad)."""
    v = v.astype(np.float64)
    out = np.empty(v.shape, np.float32)
    nxt = np.pad(v[1:, :, :, 0], ((0, 1), (0, 0), (0, 0)))
    out[..., 0] = (0.5 * (v[..., 0] + nxt)).astype(np.float32)
    nxt = np.pad(v[:, 1:, :, 1], ((0, 0), (0, 1), (0, 0)))
    out[..., 1] = (0.5 * (v[..., 1] + nxt)).astype(np.float32)
    nxt = np.pad(v[:, :, 1:, 2], ((0, 0), (0, 0), (0, 1)))
    out[..., 2] = (0.5 * (v[..., 2] + nxt)).astype(np.float32)
    return out


# ---------------------------------------------------------------------------------------------------- tests
def test_block_header_follows_the_vtk_zlib_layout():
    for count, blocks, last in ((BLOCK_SIZE // 4, 1, 0), (BLOCK_SIZE // 4 + 1, 2, 4), (10, 1, 40)):
        values = np.arange(count, dtype=np.float32)
        vtk_type, payload = encode_array(values, compress=True, level=1)
        assert vtk_type == "Float32"
        header = np.frombuffer(payload, "<u8", 3, 0)
        assert tuple(int(v) for v in header) == (blocks, BLOCK_SIZE, last)
        assert np.array_equal(np.frombuffer(decode_payload(payload, 0, True), "<f4"), values)
        vtk_type, plain = encode_array(values, compress=False, level=1)
        assert int(np.frombuffer(plain, "<u8", 1, 0)[0]) == 4 * count and plain[8:] == values.tobytes()


@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("pressure_dtype", [np.float32, np.float64])
def test_fluid_3d_round_trip_is_bitwise(tmp_path, compress, pressure_dtype):
    d = fixture(pressure_dtype=pressure_dtype)
    w = writer_for(tmp_path / "vtk", d, compress=compress)
    path, nbytes = w.write_fluid_3d(7, 1.25e-3, d["velocity"], d["pressure"], d["obstacle"], d["air"])
    assert path.stat().st_size == nbytes
    root, arrays = read_vtk(path)
    nx, ny, nz = shape = d["velocity"].shape[:3]
    assert root.get("type") == "RectilinearGrid"
    assert root.find("RectilinearGrid").get("WholeExtent") == "0 %d 0 %d 0 %d" % (nx - 1, ny - 1, nz - 1)
    assert (root.get("compressor") == "vtkZLibDataCompressor") == compress
    for axis, faces in zip("xyz", d["faces"]):                    # coordinates = graded cell centres, bitwise
        assert np.array_equal(arrays["%s_coordinates_m" % axis], 0.5 * (faces[:-1] + faces[1:]))
    vc = expected_centred(d["velocity"])
    assert np.array_equal(grid_array(arrays, "velocity_mps", shape, 3), vc)
    speed = np.sqrt(np.sum(vc.astype(np.float64) ** 2, axis=-1)).astype(np.float32)
    assert np.array_equal(grid_array(arrays, "speed_mps", shape), speed)
    p = grid_array(arrays, "pressure_pa", shape)
    assert p.dtype == np.dtype(pressure_dtype).newbyteorder("<") and np.array_equal(p, d["pressure"])
    assert np.array_equal(grid_array(arrays, "obstacle", shape), d["obstacle"])
    assert np.array_equal(grid_array(arrays, "air", shape), d["air"].astype(np.int32))
    assert arrays["TimeValue"].tolist() == [1.25e-3] and arrays["step"].tolist() == [7]
    assert_strings(arrays)
    # x-fastest order, explicitly: flat index i + nx (j + ny k)
    flat = arrays["obstacle"]
    for i, j, k in ((1, 2, 3), (4, 0, 5), (0, 3, 1)):
        assert flat[i + nx * (j + ny * k)] == d["obstacle"][i, j, k]


def test_planes_are_layers_of_the_3d_fields(tmp_path):
    d = fixture(seed=3, shape=(6, 5, 7))
    for plane in ((2, 1), (5, 4)):                                # interior layer and the last layer (far wall)
        out = tmp_path / ("vtk_%d_%d" % plane)
        w = writer_for(out, d, plane=plane)
        full_path, _ = w.write_fluid_3d(3, 2e-3, d["velocity"], d["pressure"], d["obstacle"], d["air"])
        (xz_path, _), (yz_path, _) = w.write_planes(3, 2e-3, d["velocity"], d["pressure"], d["obstacle"], d["air"])
        _, full = read_vtk(full_path)
        shape = d["velocity"].shape[:3]
        i, j = plane
        root, xz = read_vtk(xz_path)
        assert root.find("RectilinearGrid").get("WholeExtent") == "0 %d %d %d 0 %d" % (shape[0] - 1, j, j,
                                                                                         shape[2] - 1)
        assert xz["y_coordinates_m"].tolist() == [0.5 * (d["faces"][1][j] + d["faces"][1][j + 1])]
        root, yz = read_vtk(yz_path)
        assert root.find("RectilinearGrid").get("WholeExtent") == "%d %d 0 %d 0 %d" % (i, i, shape[1] - 1,
                                                                                         shape[2] - 1)
        assert yz["x_coordinates_m"].tolist() == [0.5 * (d["faces"][0][i] + d["faces"][0][i + 1])]
        assert_strings(xz)
        assert_strings(yz)
        for name, comps in (("velocity_mps", 3), ("speed_mps", 1), ("pressure_pa", 1), ("obstacle", 1),
                            ("air", 1)):
            whole = grid_array(full, name, shape, comps)
            assert np.array_equal(grid_array(xz, name, (shape[0], 1, shape[2]), comps), whole[:, j:j + 1])
            assert np.array_equal(grid_array(yz, name, (1, shape[1], shape[2]), comps), whole[i:i + 1])


def test_points_round_trip_is_bitwise(tmp_path):
    d = fixture(seed=5, n_points=1000)
    w = writer_for(tmp_path / "vtk", d)
    path, _ = w.write_points(11, 3e-3, d["u"], d["J"])
    root, arrays = read_vtk(path)
    n = len(d["X0"])
    piece = root.find("PolyData/Piece")
    assert piece.get("NumberOfPoints") == piece.get("NumberOfVerts") == str(n)
    assert np.array_equal(arrays["Points"], d["X0"].astype(np.float64) + d["u"].astype(np.float64))
    assert arrays["Points"].dtype == np.dtype("<f8")
    assert np.array_equal(arrays["X0_m"], d["X0"]) and np.array_equal(arrays["displacement_m"], d["u"])
    assert np.array_equal(arrays["J"], d["J"]) and np.array_equal(arrays["body_id"], d["body_id"])
    assert np.array_equal(arrays["tag_air_face"], (d["tag"] & 1 != 0).astype(np.int32))
    assert np.array_equal(arrays["tag_valve_rim"], (d["tag"] & 2 != 0).astype(np.int32))
    assert np.array_equal(arrays["tether"], d["tether"])
    assert np.array_equal(arrays["connectivity"], np.arange(n)) and np.array_equal(arrays["offsets"],
                                                                                   np.arange(1, n + 1))
    assert arrays["TimeValue"].tolist() == [3e-3] and arrays["step"].tolist() == [11]
    assert_strings(arrays)


def test_series_index_pvd_and_create_only(tmp_path):
    d = fixture(seed=7)
    out = tmp_path / "vtk"
    w = writer_for(out, d)
    times = [(20, 1e-3), (40, 2e-3), (40, 2e-3), (60, 3.5e-3)]   # a repeated step must not collide
    for step, t in times:
        w.write_points(step, t, d["u"], d["J"])
        w.write_planes(step, t, d["velocity"], d["pressure"], d["obstacle"], d["air"])
    w.write_fluid_3d(60, 3.5e-3, d["velocity"], d["pressure"], d["obstacle"], d["air"])
    pvds = w.finalize()
    assert sorted(p.name for p in pvds) == ["fluid_3d.pvd", "planes_xz.pvd", "planes_yz.pvd", "points.pvd"]
    meta = json.loads((out / "vtk_series_meta.json").read_text(encoding="utf-8"))
    assert meta["label"] == LABEL and meta["geometry_note"] == NOTE
    assert LABEL in pvds[0].read_text(encoding="utf-8")
    assert w.finalize() == pvds                                   # idempotent, nothing rewritten
    for pvd in pvds:
        root = ET.parse(pvd).getroot()
        assert root.get("type") == "Collection"
        rows = root.findall("Collection/DataSet")
        expected = 1 if pvd.name == "fluid_3d.pvd" else len(times)
        assert len(rows) == expected
        stamps = [float(r.get("timestep")) for r in rows]
        assert stamps == sorted(stamps)
        assert all((out / r.get("file")).is_file() for r in rows)
    entries, skipped = read_index(out / "vtk_series.jsonl")
    assert skipped == 0 and len(entries) == 3 * len(times) + 1
    for e in entries:
        data = (out / e["file"]).read_bytes()
        assert len(data) == e["bytes"]
    with pytest.raises(FileExistsError):
        VtkSeriesWriter(out, faces_x=d["faces"][0], faces_y=d["faces"][1], faces_z=d["faces"][2], plane_i=0,
                        plane_j=0, X0=d["X0"], body_id=d["body_id"], tag=d["tag"], tether=d["tether"])


def test_make_pvd_rebuilds_the_collections_after_a_kill(tmp_path):
    d = fixture(seed=9)
    out = tmp_path / "vtk"
    w = writer_for(out, d)
    for step, t in ((20, 1e-3), (40, 2e-3)):
        w.write_points(step, t, d["u"], d["J"])
        w.write_planes(step, t, d["velocity"], d["pressure"], d["obstacle"], d["air"])
    w._index.flush()
    with open(out / "vtk_series.jsonl", "a", encoding="utf-8") as stream:   # killed while appending a line
        stream.write('{"kind": "points", "step": 60, "t_s": 0.003, "fi')
    pvds, skipped = make_pvd(out / "vtk_series.jsonl", out)
    assert skipped == 1 and sorted(p.name for p in pvds) == ["planes_xz.pvd", "planes_yz.pvd", "points.pvd"]
    rows = ET.parse(out / "points.pvd").getroot().findall("Collection/DataSet")
    assert [float(r.get("timestep")) for r in rows] == [1e-3, 2e-3]
    with pytest.raises(FileExistsError):                          # create-only: never rewritten
        make_pvd(out / "vtk_series.jsonl", out)
    elsewhere, _ = make_pvd(out / "vtk_series.jsonl", tmp_path / "pvd_copy")    # relative paths from another dir
    for row in ET.parse(elsewhere[-1]).getroot().findall("Collection/DataSet"):
        assert (tmp_path / "pvd_copy" / row.get("file")).is_file()


def test_squid_shapes_write_time_and_size(tmp_path):
    """One file of each kind on the squid shapes (92 x 92 x 186 cells, 332,000 points), synthetic smooth fields with
    an obstacle and an air block; prints the write time and size per file (report, no bound asserted)."""
    shape, n = (92, 92, 186), 332000
    rng = np.random.default_rng(1)
    faces = [np.concatenate([[0.0], np.cumsum(rng.uniform(0.375e-3, 2.9e-3, size=s))]) for s in shape]
    x, y, z = np.meshgrid(*[np.linspace(0.0, 1.0, s) for s in shape], indexing="ij")
    velocity = np.stack([np.sin(3 * x) * np.cos(2 * z), np.cos(4 * y) * z, np.sin(5 * z) * x], -1).astype(np.float32)
    velocity *= 0.05
    obstacle = ((x - 0.5) ** 2 + (y - 0.5) ** 2 > 0.16).astype(np.int32)
    velocity[obstacle != 0] = 0.0
    air = (z > 0.8).astype(np.int8)
    pressure = (100.0 * np.cos(3 * z) + 0.5).astype(np.float32)
    X0 = (rng.normal(size=(n, 3)) * 0.01).astype(np.float32)
    w = VtkSeriesWriter(tmp_path / "vtk", faces_x=faces[0], faces_y=faces[1], faces_z=faces[2], plane_i=46,
                        plane_j=46, X0=X0, body_id=np.ones(n, np.int32), tag=np.zeros(n, np.int32),
                        tether=np.zeros(n, np.int32))
    u = (rng.normal(size=(n, 3)) * 1e-6).astype(np.float32)
    J = np.ones(n, np.float32)
    rows = []
    t0 = time.perf_counter()
    rows.append(("fluid_3d",) + w.write_fluid_3d(20, 1e-3, velocity, pressure, obstacle, air))
    t1 = time.perf_counter()
    planes = w.write_planes(20, 1e-3, velocity, pressure, obstacle, air)
    t2 = time.perf_counter()
    rows.append(("points",) + w.write_points(20, 1e-3, u, J))
    t3 = time.perf_counter()
    w.finalize()
    report = dict(fluid_3d=dict(seconds=t1 - t0, mbytes=rows[0][2] / 1e6),
                  planes_xz_plus_yz=dict(seconds=t2 - t1, mbytes=(planes[0][1] + planes[1][1]) / 1e6),
                  points=dict(seconds=t3 - t2, mbytes=rows[1][2] / 1e6))
    print("VTK_SQUID_SHAPES", json.dumps(report))
    _, arrays = read_vtk(rows[0][1])
    assert np.array_equal(grid_array(arrays, "obstacle", shape), obstacle)
