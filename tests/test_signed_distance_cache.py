"""The rigid-contact SDF cache (geometry/signed_distance.py; Codex review 2026-10-26, checked by Root): the key separates
the inputs the build separates (a fractional band; a sign mask's nonzero values), and a write that fails half-way
leaves nothing under the cache name (before, a half-written file there broke every later run's load)."""
from pathlib import Path

import numpy as np
import pytest

from fastfsi.geometry import signed_distance
from fastfsi.geometry.signed_distance import load_or_build_signed_distance, signed_distance_cache_key
from fastfsi.grid.cartesian_grid import CartesianGrid

H = 1.0e-3
N = 8


def box_triangles(lo, hi):
    """The 12 triangles of an axis-aligned box."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    corners = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float64)
    faces = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
    return corners[np.array(faces)]


def case():
    grid = CartesianGrid(bounds_min_m=(0.0, 0.0, 0.0), cell_widths_x_m=(H,) * N, cell_widths_y_m=(H,) * N,
                         cell_widths_z_m=(H,) * N)
    return box_triangles((2.5 * H,) * 3, (5.5 * H,) * 3), grid


def test_the_key_separates_what_the_build_separates():
    triangles, grid = case()

    def key(**options):
        return signed_distance_cache_key(triangles, grid, **options)

    assert key(band_cells=0.1) != key(band_cells=0.9)                 # int() made one key of them
    shape = grid.grid_nodes
    assert key(sign_mask=np.zeros(shape, np.int32)) != key(sign_mask=np.full(shape, 256, np.int32))   # uint8: 256 -> 0
    assert key(sign_mask=np.full(shape, 256, np.int32)) == key(sign_mask=np.ones(shape, bool))         # nonzero = inside


def test_a_write_that_fails_half_way_leaves_no_cache_file(tmp_path, monkeypatch):
    triangles, grid = case()
    savez = np.savez

    def failing_savez(stream, **arrays):
        stream.write(b"PK\x03\x04 half an archive")
        raise OSError("disk full")

    monkeypatch.setattr(signed_distance.np, "savez", failing_savez)
    with pytest.raises(OSError, match="disk full"):
        load_or_build_signed_distance(tmp_path, triangles, grid)
    assert list(tmp_path.iterdir()) == []                              # neither the cache name nor a temporary
    monkeypatch.setattr(signed_distance.np, "savez", savez)
    phi, written = load_or_build_signed_distance(tmp_path, triangles, grid)
    assert written["cache"] == "miss (written)"
    assert [p.name for p in tmp_path.iterdir()] == [Path(written["cache_path"]).name]
    phi_hit, hit = load_or_build_signed_distance(tmp_path, triangles, grid)
    assert hit["cache"] == "hit" and np.array_equal(phi_hit, phi)
