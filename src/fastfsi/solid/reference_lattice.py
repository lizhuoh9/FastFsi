"""The reference lattice of a body's material points: the cells whose 8 corners are points, and their Kuhn split.

Material points that sit on a tensor-product lattice (one point per lattice index, consecutive coordinates (s_a + s_b) / 2
apart for the local spacings s) define hexahedral cells between neighbouring points.  Each cell is split into the 6 Kuhn
tetrahedra 000 -> e_a -> e_a + e_b -> 111 (one per axis permutation); every cell uses the same split, so neighbouring
cells share their face triangles.  Used by the occupancy (boundary.solid_occupancy) and by the lattice stress
(coupling.lattice_stress).  Host-side setup only (NumPy).
"""
import itertools

import numpy as np

KEY_QUANTUM_M = 1.0e-8             # reference lattice coordinates are identified to 10 nm
ADJACENCY_RTOL = 1.0e-6            # consecutive lattice coordinates must be (s_a + s_b) / 2 apart to this tolerance


def kuhn_tetrahedra():
    """The 6 Kuhn tetrahedra of a cell as corner numbers c = 4 di + 2 dj + dk (paths 000 -> e_a -> e_a + e_b -> 111)."""
    tets = []
    for perm in itertools.permutations(range(3)):
        corner, path = [0, 0, 0], [0]
        for axis in perm:
            corner[axis] = 1
            path.append(4 * corner[0] + 2 * corner[1] + corner[2])
        tets.append(tuple(path))
    return tuple(tets)


TETS = kuhn_tetrahedra()


def reference_lattice_cells(reference_positions_m, spacing_m):
    """Host setup: the reference-lattice cells whose 8 corners are points.  Returns (cells (H, 8) int64 of point numbers,
    corner c = 4 di + 2 dj + dk, and the count of candidate cells refused because two consecutive lattice coordinates
    are not (s_a + s_b) / 2 apart).  Points must sit on a tensor-product lattice (one point per lattice index)."""
    positions = np.asarray(reference_positions_m, dtype=np.float64)
    spacing = np.asarray(spacing_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or spacing.shape != positions.shape:
        raise ValueError("reference positions and spacing must both be (n, 3)")
    if not (np.isfinite(positions).all() and np.isfinite(spacing).all() and (spacing > 0.0).all()):
        raise ValueError("reference positions must be finite and spacings finite and positive")
    keys = np.round(positions / KEY_QUANTUM_M).astype(np.int64)
    unique = [np.unique(keys[:, axis]) for axis in range(3)]
    index = np.stack([np.searchsorted(unique[axis], keys[:, axis]) for axis in range(3)], axis=1)
    table = np.full(tuple(len(u) for u in unique), -1, dtype=np.int64)
    table[index[:, 0], index[:, 1], index[:, 2]] = np.arange(len(positions))
    if int((table >= 0).sum()) != len(positions):
        raise ValueError("two points share a reference lattice index")
    nx, ny, nz = table.shape
    corners = np.stack([table[di:di + nx - 1, dj:dj + ny - 1, dk:dk + nz - 1].ravel()
                        for di, dj, dk in itertools.product((0, 1), repeat=3)], axis=1)
    corners = corners[np.all(corners >= 0, axis=1)]
    adjacent = np.ones(len(corners), dtype=bool)
    for axis, (low, high) in enumerate(((0, 4), (0, 2), (0, 1))):
        gap = positions[corners[:, high], axis] - positions[corners[:, low], axis]
        want = 0.5 * (spacing[corners[:, low], axis] + spacing[corners[:, high], axis])
        adjacent &= np.abs(gap - want) <= ADJACENCY_RTOL * want
    return corners[adjacent], int((~adjacent).sum())
