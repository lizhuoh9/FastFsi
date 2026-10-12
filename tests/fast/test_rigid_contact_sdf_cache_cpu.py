"""Integrator's rigid_contact v5 (Root 2026-10-12, after the smoke RSS block): the exact-distance candidate loop in
sub-batches of at most PAIR_BUDGET pairs gives bitwise the same signed distance; the on-disk SDF cache is create-only,
keyed by every input, and a hit returns the stored field."""
import numpy as np

from simulation_core.fast import rigid_contact as rc
from simulation_core.fluids.grid import CartesianGrid


def box_triangles(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float64)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return np.asarray([v[list(q)] for a, b, c, d in quads for q in ((a, b, c), (a, c, d))])


GRID = CartesianGrid.uniform(bounds_min_m=(0.0, 0.0, 0.0), bounds_max_m=(0.012, 0.010, 0.008), grid_nodes=(12, 10, 8))
TRI = {"box": box_triangles((0.0031, 0.0024, 0.0018), (0.0083, 0.0072, 0.0046))}


def test_pair_budget_batches_give_bitwise_the_same_distance(monkeypatch):
    phi_ref, rep_ref = rc.build_signed_distance(TRI, GRID)
    monkeypatch.setattr(rc, "PAIR_BUDGET", 97)          # forces many small batches and several k rounds per batch
    phi_small, rep_small = rc.build_signed_distance(TRI, GRID)
    assert np.array_equal(phi_ref, phi_small)
    assert rep_ref["exact_pairs"] == rep_small["exact_pairs"] and rep_ref["exact_cells"] == rep_small["exact_cells"]


def test_sdf_cache_is_create_only_keyed_and_reused(tmp_path):
    sign = np.zeros((12, 10, 8), bool)
    sign[4:8, 3:7, 2:4] = True
    phi1, rep1 = rc.cached_signed_distance(tmp_path, TRI, GRID, sign_mask=sign)
    assert rep1["cache"] == "miss (written)"
    files = sorted(tmp_path.glob("sdf_*.npz"))
    assert len(files) == 1
    stamp = files[0].stat().st_mtime_ns
    phi2, rep2 = rc.cached_signed_distance(tmp_path, TRI, GRID, sign_mask=sign)
    assert rep2["cache"] == "hit" and rep2["cache_key"] == rep1["cache_key"]
    assert np.array_equal(phi1, phi2) and files[0].stat().st_mtime_ns == stamp     # read, never rewritten
    direct, _ = rc.build_signed_distance(TRI, GRID, sign_mask=sign)
    assert np.array_equal(direct, phi2)
    # any input change is a different key (here: the sign mask, then the reach margin)
    sign2 = sign.copy()
    sign2[0, 0, 0] = True
    _, rep3 = rc.cached_signed_distance(tmp_path, TRI, GRID, sign_mask=sign2)
    pts = np.array([[0.002, 0.005, 0.004]])
    _, rep4 = rc.cached_signed_distance(tmp_path, TRI, GRID, sign_mask=sign, reach_points_m=pts, reach_margin_m=0.003)
    keys = {rep1["cache_key"], rep3["cache_key"], rep4["cache_key"]}
    assert len(keys) == 3 and len(sorted(tmp_path.glob("sdf_*.npz"))) == 3
