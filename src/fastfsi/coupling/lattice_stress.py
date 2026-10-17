"""Position-based elastic force of the reference lattice (the coupler's `solid_stress_source`).

The point stress of the coupler (tau from each point's own F, spread as -V0 tau grad w) does not see two neighbouring
points drifting apart while their F stays unchanged: that separation costs no elastic energy.  The reference-lattice
cells (fastfsi.solid.reference_lattice: 8 corner points per cell, 6 Kuhn tetrahedra) measure the deformation from the
points' actual positions instead.  Per tetrahedron t of cell h, with corners a, b, c, d:

    F_t = Ds(x) Ds(X)^-1 = I + Ds(u) Dm^-1,   Dm = [X_b - X_a, X_c - X_a, X_d - X_a]  (reference, float64 at setup),
    J_t = det F_t,   J_h = sum_t V0_t J_t / sum_t V0_t  (the cell's volume ratio: mean dilatation),

and the energy of the points' neo-Hookean law W = mu/2 (I1b - 3) + kappa/2 (J - 1)^2 with its volumetric part taken per
cell (the mean-dilatation form of nearly incompressible constant-strain tetrahedra: per-tetrahedron volume terms lock
the bending of the thin plates at kappa / mu = 50; a cell has one volume constraint for its 8 corners):

    E = sum_t V0_t mu/2 (I1b(F_t) - 3) + sum_h V0_h kappa/2 (J_h - 1)^2,

whose exact nodal forces are f_i = -sum_t V0_t P_t grad_X N_i with P_t = tau_t F_t^-T,
tau_t = mu J_t^(-2/3) dev(F_t F_t^T) + kappa (J_h - 1) J_t I  (rows of Dm^-1 for corners b, c, d; f_a = -(f_b + f_c + f_d)).
The current edges are formed from the f32 displacements (u_i - u_a), so the f32 rounding of the absolute positions never
enters F_t.  The nodal forces are spread through the coupler's point-force path (point_force w, the same weights as the
gather), at the positions of the spread.

The point stress stays for every point; the lattice adds only the INCOMPATIBLE part (source "lattice_correction"):
the nodal forces above with tau_t - taubar_t, where taubar_t is the same expression with Fbar_t = the mean of the
tetrahedron's 4 corner point F (and the cell's mean of det Fbar_t).  It is zero for any motion the point F follows
(affine motion exactly), and gives the lattice restoring force for a separation that F misses.  The points keep their own
volumes and F for everything else (the lattice hull, which joins the point centres, would miss half a spacing at every
surface).
A cell with some det F_t <= 0 or det Fbar_t <= 0 contributes nothing and is counted.

No `from __future__ import annotations` (Taichi kernel module).
"""
import numpy as np
import taichi as ti

from fastfsi.solid.reference_lattice import TETS

SOURCES = ("point_F", "lattice_correction")       # "point_F": no LatticeStress
# statistics slots (one f64 field, read on request)
S_SKIPPED, S_MIN_J, S_MAX_J, S_MAX_DF = 0, 1, 2, 3
STATS_SLOTS = 4


def validate_stress_source(name):
    """The stress source name; anything outside SOURCES is refused."""
    if name not in SOURCES:
        raise ValueError("solid stress source must be one of %s, got %r" % (SOURCES, name))
    return name


@ti.func
def mean_dilatation_kirchhoff(F, J, J_cell, mu, kappa):
    """tau = mu J^(-2/3) dev(F F^T) + kappa (J_cell - 1) J I (the volumetric part from the cell's volume ratio)."""
    b = F @ F.transpose()
    identity = ti.Matrix.identity(ti.f32, 3)
    return mu * ti.pow(J, -2.0 / 3.0) * (b - (b.trace() / 3.0) * identity) + kappa * (J_cell - 1.0) * J * identity


def lattice_energy(F_tets, v0_tets, mu, kappa):
    """Host: E of the module docstring for (H, 6, 3, 3) tetrahedron F and (H, 6) V0 (tests and diagnostics)."""
    F = np.asarray(F_tets, dtype=np.float64)
    v0 = np.asarray(v0_tets, dtype=np.float64)
    J = np.linalg.det(F)
    i1b = np.einsum("htij,htij->ht", F, F) * J ** (-2.0 / 3.0)
    J_cell = (v0 * J).sum(axis=1) / v0.sum(axis=1)
    return float((v0 * 0.5 * mu * (i1b - 3.0)).sum() + (v0.sum(axis=1) * 0.5 * kappa * (J_cell - 1.0) ** 2).sum())


@ti.data_oriented
class LatticeStress:
    """The incompatible-part nodal forces of a set of reference-lattice cells (global point numbers) of `points`
    (MaterialPoints)."""

    mode = "lattice_correction"

    def __init__(self, points, cells, reference_positions_m):
        cells = np.asarray(cells, dtype=np.int64)
        reference = np.asarray(reference_positions_m, dtype=np.float64)
        if cells.ndim != 2 or cells.shape[1] != 8 or len(cells) == 0:
            raise ValueError("cells must be a non-empty (H, 8) array of point numbers")
        if cells.min() < 0 or cells.max() >= points.n or reference.shape != (points.n, 3):
            raise ValueError("cell corners must be point numbers and the reference positions one per point")
        tets = cells[:, np.asarray(TETS, dtype=np.int64)]                                   # (H, 6, 4)
        dm = np.stack([reference[tets[..., q]] - reference[tets[..., 0]] for q in (1, 2, 3)], axis=-1)   # columns
        det = np.linalg.det(dm)
        if np.any(det == 0.0):
            raise ValueError("a reference lattice tetrahedron has zero volume")
        self.points = points
        self.n_cells = int(len(cells))
        self.cells_host = cells
        self.tet_volume_m3 = np.abs(det) / 6.0                                            # (H, 6)
        covered = np.zeros(points.n, dtype=bool)
        covered[np.unique(cells)] = True
        self.covered_host = covered
        self.cell = ti.field(ti.i32, shape=(self.n_cells, 8))
        self.cell.from_numpy(cells.astype(np.int32))
        self.dm_inv = ti.Matrix.field(3, 3, ti.f32, shape=(self.n_cells, 6))
        self.dm_inv.from_numpy(np.linalg.inv(dm).astype(np.float32))
        self.v0 = ti.field(ti.f32, shape=(self.n_cells, 6))
        self.v0.from_numpy(self.tet_volume_m3.astype(np.float32))
        self.force = ti.Vector.field(3, ti.f32, shape=points.n)          # nodal forces of the last compute (N)
        self.stats = ti.field(ti.f64, shape=STATS_SLOTS)

    @ti.kernel
    def _clear_kernel(self):
        for p in self.force:
            self.force[p] = ti.Vector([0.0, 0.0, 0.0])
        self.stats[S_SKIPPED] = 0.0
        self.stats[S_MIN_J] = 1.0e30
        self.stats[S_MAX_J] = -1.0e30
        self.stats[S_MAX_DF] = 0.0

    @ti.func
    def _tet_F(self, h, t: ti.template()):
        a = self.cell[h, TETS[t][0]]
        u0 = self.points.u[a]
        Du = ti.Matrix.cols([self.points.u[self.cell[h, TETS[t][1]]] - u0, self.points.u[self.cell[h, TETS[t][2]]] - u0,
                             self.points.u[self.cell[h, TETS[t][3]]] - u0])
        return ti.Matrix.identity(ti.f32, 3) + Du @ self.dm_inv[h, t]

    @ti.func
    def _corner_mean_F(self, h, t: ti.template()):
        return 0.25 * (self.points.F[self.cell[h, TETS[t][0]]] + self.points.F[self.cell[h, TETS[t][1]]]
                       + self.points.F[self.cell[h, TETS[t][2]]] + self.points.F[self.cell[h, TETS[t][3]]])

    @ti.kernel
    def _force_kernel(self, mu: ti.f32, kappa: ti.f32):
        for h in range(self.n_cells):
            Fs = [self._tet_F(h, t) for t in ti.static(range(6))]
            Js = [Fs[t].determinant() for t in ti.static(range(6))]
            weight = 0.0
            J_cell = 0.0
            valid = 1
            for t in ti.static(range(6)):
                weight += self.v0[h, t]
                J_cell += self.v0[h, t] * Js[t]
                ti.atomic_min(self.stats[S_MIN_J], ti.cast(Js[t], ti.f64))
                ti.atomic_max(self.stats[S_MAX_J], ti.cast(Js[t], ti.f64))
                if Js[t] <= 0.0:
                    valid = 0
            J_cell /= weight
            Fb = [self._corner_mean_F(h, t) for t in ti.static(range(6))]
            Jb = [Fb[t].determinant() for t in ti.static(range(6))]
            Jb_cell = 0.0
            for t in ti.static(range(6)):
                Jb_cell += self.v0[h, t] * Jb[t]
                ti.atomic_max(self.stats[S_MAX_DF], ti.cast((Fs[t] - Fb[t]).norm(), ti.f64))
                if Jb[t] <= 0.0:
                    valid = 0
            Jb_cell /= weight
            if valid == 0:
                ti.atomic_add(self.stats[S_SKIPPED], 1.0)
            else:
                for t in ti.static(range(6)):
                    P = (mean_dilatation_kirchhoff(Fs[t], Js[t], J_cell, mu, kappa) @ Fs[t].inverse().transpose()
                         - mean_dilatation_kirchhoff(Fb[t], Jb[t], Jb_cell, mu, kappa) @ Fb[t].inverse().transpose())
                    H = -self.v0[h, t] * P @ self.dm_inv[h, t].transpose()   # column q: the force on corner q + 1
                    f1 = ti.Vector([H[0, 0], H[1, 0], H[2, 0]])
                    f2 = ti.Vector([H[0, 1], H[1, 1], H[2, 1]])
                    f3 = ti.Vector([H[0, 2], H[1, 2], H[2, 2]])
                    ti.atomic_add(self.force[self.cell[h, TETS[t][1]]], f1)
                    ti.atomic_add(self.force[self.cell[h, TETS[t][2]]], f2)
                    ti.atomic_add(self.force[self.cell[h, TETS[t][3]]], f3)
                    ti.atomic_add(self.force[self.cell[h, TETS[t][0]]], -(f1 + f2 + f3))

    def compute(self, mu, kappa):
        """The nodal forces at the current displacements and point F into `force`."""
        self._clear_kernel()
        self._force_kernel(float(mu), float(kappa))

    def tet_deformation_gradients(self):
        """Host: (H, 6, 3, 3) F_t at the current displacements (float64; tests and diagnostics)."""
        u = self.points.u.to_numpy().astype(np.float64)
        tets = self.cells_host[:, np.asarray(TETS, dtype=np.int64)]
        du = np.stack([u[tets[..., q]] - u[tets[..., 0]] for q in (1, 2, 3)], axis=-1)
        return np.eye(3) + du @ self.dm_inv.to_numpy().astype(np.float64)

    def report(self):
        """Statistics of the last compute: cells skipped (some det <= 0), the J range of the lattice tetrahedra and the
        largest |F_t - Fbar_t| (Frobenius; the amplitude of the incompatible deformation)."""
        s = self.stats.to_numpy()
        return dict(mode=self.mode, cells_skipped=int(s[S_SKIPPED]), lattice_J_min=float(s[S_MIN_J]),
                    lattice_J_max=float(s[S_MAX_J]),
                    max_F_minus_point_F=float(s[S_MAX_DF]))

    def setup_report(self, body_id=None, tether=None, point_volumes_m3=None):
        """Coverage and volume partition (per body id when given): cells, covered and uncovered points (uncovered
        tethered), the sum of the tetrahedra's V0 and of the body's point V0."""
        n = self.points.n
        body = np.zeros(n, np.int64) if body_id is None else np.asarray(body_id, np.int64)
        tether = np.zeros(n, bool) if tether is None else np.asarray(tether) != 0
        volumes = None if point_volumes_m3 is None else np.asarray(point_volumes_m3, np.float64)
        out = {}
        for b in sorted(set(body.tolist())):
            mine = body == b
            cell_mine = mine[self.cells_host[:, 0]]
            entry = dict(points=int(mine.sum()), cells=int(cell_mine.sum()),
                         uncovered=int((mine & ~self.covered_host).sum()),
                         uncovered_tethered=int((mine & ~self.covered_host & tether).sum()),
                         tet_volume_m3=float(self.tet_volume_m3[cell_mine].sum()))
            if volumes is not None:
                entry.update(point_volume_m3=float(volumes[mine].sum()),
                             tet_to_point_volume=entry["tet_volume_m3"] / float(volumes[mine].sum()))
            out[int(b)] = entry
        return dict(mode=self.mode, cells=self.n_cells, bodies=out)
