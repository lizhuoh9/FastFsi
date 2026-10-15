"""Immersed-boundary kernel weights on the axes of a MAC grid (grid.axes).

KernelWeights(axes, ib_kernel, moment_correction) evaluates, along axis a and for one node family (staggered = 1: the
faces of the MAC component along its own axis; staggered = 0: the cell centres), the weights of the four nodes around a
physical coordinate and their physical derivatives:

    w_q = phi(xi - (i0 + q)),   dw_q = phi'(xi - (i0 + q)) * dxi/dx,   dxi/dx = 1 / (local node spacing)   [1/m]

with xi the node-index coordinate of the axes, i0 = floor(xi) - 1, and phi = IB_4 or IB_3 (coupling.kernels).  It also
forwards the axes' n, min_spacing_m and face_volume, so a coupler sees one object.

First-order moment correction (graded axes, on by default).  On a stretched axis the index-space weights do not
reproduce linear fields in physical space: the interpolated gradient of u = a x has a relative error of up to
~(r - 1) / 2 for an adjacent-cell stretch ratio r (measured without the correction: 0.025 / 0.050 / 0.100 at
r = 1.05 / 1.1 / 1.2).  Per axis and node family the four weights are therefore corrected, reproducing-kernel style, as
w~_q = w_q (alpha + beta d_q) with d_q = (x_q - x) / s (s the local node spacing), where alpha, beta solve the 2 x 2
moment system

    sum_q w~_q = 1,   sum_q w~_q (x_q - x) = 0,

and dw~_q is the exact physical derivative d w~_q / dx (alpha, beta and d_q differentiated), which then satisfies
sum_q dw~_q = 0 and sum_q dw~_q (x_q - x) = 1 identically.  The interpolated value and gradient of a linear field are
therefore exact on any grading (per axis; tensor products are exact in 3-D away from invalid faces), the derivative
weights are the gradient of the corrected weights (the spread stress force and the interpolated velocity gradient are
variationally consistent), and force conservation, the power identity and the internal-force balance (sum dw~ = 0)
hold because the same w~, dw~ are used to spread and to interpolate.  The correction uses all four nodes at their
physical (ghost) positions before out-of-range nodes are zeroed; the domain edge and invalid faces are left to the
one-sided normalisation of the coupler.  On a uniform axis of GradedAxes the moments already hold, no correction is
applied and xi = (coord - origin) / h - (0 if staggered else 0.5), dw = phi' / h.  For UniformAxes (IB_4 only) the
weights of all four nodes are returned without an out-of-range check, as for an unbounded uniform grid.

Measured properties (stretch ratio r <= 1.2): linear value and gradient exact to f32 roundoff; a rigid rotation
sampled on a grid with 1.2 band transitions gives max |sym grad u| = 5.2e-7 for |omega| = 0.91 (raw kernel: 2.5e-2);
min w~ >= 0 (bounded, sum |w~| = 1); w~ is Lipschitz-continuous in x (no grid locking); dw~ has a kink where the node
spacing changes, max |dw(x+) - dw(x-)| / max |dw| = 0.010 / 0.017 / 0.039 at r = 1.05 / 1.1 / 1.2 (raw kernel:
0.048 / 0.091 / 0.166); the IB_4 identity sum w^2 = 3/8 deviates by up to 0.8 % at r = 1.2.

References: the side-centred index map follows IBAMR's LEInteractor (MAC components treated as cell-centred data
shifted by half a cell along their own axis; IBAMR, BSD-3-Clause, algorithm only, no source copied).  The moment
correction is the algorithm of A. Pinelli, I. Z. Naqavi, U. Piomelli, J. Favier, "Immersed-boundary methods for
general finite-difference and finite-volume Navier-Stokes solvers", J. Comput. Phys. 229 (2010) 9073-9091; the
exact-derivative weights are this library's construction.

No `from __future__ import annotations` (Taichi kernel module).
"""
import taichi as ti

from fastfsi.coupling.kernels import (evaluate_ib3_weight, evaluate_ib3_weight_derivative, evaluate_ib4_weight,
                                      evaluate_ib4_weight_derivative, validate_kernel_name)
from fastfsi.grid.axes import FINITE_INDEX_BOUND, NONFINITE_INDEX_COORDINATE, UniformAxes


@ti.func
def correct_first_moment(w, dw, d, spacing):
    """Reproducing-kernel correction of four 1-D weights (Pinelli et al. 2010): w~ = w (alpha + beta d) with
    sum w~ = 1 and sum w~ d = 0, d_q = (x_q - x) / spacing; dw~ is the exact x-derivative (d d_q / dx = -1 / spacing),
    so sum dw~ = 0 and sum dw~ (x_q - x) = 1.  w, dw, d: Vector(4); dw in 1/m."""
    inv_s = 1.0 / spacing
    m0 = w.sum()
    m1 = (w * d).sum()
    m2 = (w * d * d).sum()
    dm0 = dw.sum()
    dm1 = (dw * d).sum() - m0 * inv_s
    dm2 = (dw * d * d).sum() - 2.0 * m1 * inv_s
    det = m0 * m2 - m1 * m1
    ddet = dm0 * m2 + m0 * dm2 - 2.0 * m1 * dm1
    alpha = m2 / det
    beta = -m1 / det
    dalpha = (dm2 - alpha * ddet) / det
    dbeta = (-dm1 - beta * ddet) / det
    shape = alpha + beta * d
    return w * shape, dw * shape + w * (dalpha + dbeta * d - beta * inv_s)


@ti.data_oriented
class KernelWeights:
    """IB_4 / IB_3 kernel weights on the axes of a MAC grid (GradedAxes or UniformAxes)."""

    def __init__(self, axes, *, ib_kernel="ib4", moment_correction=True):
        self.axes = axes
        self.ib_kernel = validate_kernel_name(ib_kernel)   # compile-time IB_4 / IB_3 choice
        self.ib3 = self.ib_kernel == "ib3"
        self.simple_uniform = isinstance(axes, UniformAxes)
        if self.simple_uniform and self.ib3:
            raise ValueError("UniformAxes supports the IB_4 kernel only; use GradedAxes for IB_3")
        self.moment_correction = bool(moment_correction) and not self.simple_uniform
        self.n = tuple(axes.n)
        self.min_spacing_m = axes.min_spacing_m

    def identity(self):
        """JSON-able description for run metadata."""
        return dict(ib_kernel=self.ib_kernel, moment_correction=self.moment_correction,
                    axes=self.axes.identity() if hasattr(self.axes, "identity") else None)

    @ti.func
    def axis_weights(self, a: ti.template(), staggered: ti.template(), coord):
        """Kernel weights of the 4 nodes along axis a around physical coordinate `coord`.
        Returns (i0: i32, w: Vector(4, f32), dw: Vector(4, f32)) for nodes i0 + 0..3, dw = dw/dcoord in 1/m (the
        physical derivative); moment-corrected on graded axes; on GradedAxes nodes outside 0 <= i < n_a are returned
        with w = dw = 0."""
        i0 = 0
        w = ti.Vector([0.0, 0.0, 0.0, 0.0])
        dw = ti.Vector([0.0, 0.0, 0.0, 0.0])
        if ti.static(self.simple_uniform):
            s = (coord - self.axes.origin[a]) / self.axes.h[a] - (0.0 if staggered else 0.5)
            i0 = ti.cast(ti.floor(s), ti.i32) - 1
            for q in ti.static(range(4)):
                r = s - ti.cast(i0 + q, ti.f32)
                w[q] = evaluate_ib4_weight(r)
                dw[q] = evaluate_ib4_weight_derivative(r) / self.axes.h[a]
        else:
            if ti.static(self.axes.uniform[a]):
                # the uniform reduction (compile-time h)
                s = (coord - self.axes.origin[a]) / self.axes.h[a] - (0.0 if staggered else 0.5)
                if not (ti.abs(s) < FINITE_INDEX_BOUND):
                    s = NONFINITE_INDEX_COORDINATE
                i0 = ti.cast(ti.floor(s), ti.i32) - 1
                for q in ti.static(range(4)):
                    i = i0 + q
                    if 0 <= i < self.axes.n[a]:
                        r = s - ti.cast(i, ti.f32)
                        if ti.static(self.ib3):            # IB_3 on the same 4-node frame
                            w[q] = evaluate_ib3_weight(r)
                            dw[q] = evaluate_ib3_weight_derivative(r) / self.axes.h[a]
                        else:                              # IB_4 (default)
                            w[q] = evaluate_ib4_weight(r)
                            dw[q] = evaluate_ib4_weight_derivative(r) / self.axes.h[a]
            else:
                family = ti.static(1 if staggered else 0)
                relative = coord - self.axes.origin[a]
                xi, local = self.axes._graded_index_coordinate(a, family, relative)
                if not (ti.abs(xi) < FINITE_INDEX_BOUND):
                    xi = NONFINITE_INDEX_COORDINATE
                i0 = ti.cast(ti.floor(xi), ti.i32) - 1
                for q in ti.static(range(4)):
                    r = xi - ti.cast(i0 + q, ti.f32)
                    if ti.static(self.ib3):                # IB_3, then the same moment correction
                        w[q] = evaluate_ib3_weight(r)
                        dw[q] = evaluate_ib3_weight_derivative(r) / local
                    else:                                  # IB_4 (default)
                        w[q] = evaluate_ib4_weight(r)
                        dw[q] = evaluate_ib4_weight_derivative(r) / local
                if ti.static(self.moment_correction):
                    d = ti.Vector([0.0, 0.0, 0.0, 0.0])
                    for q in ti.static(range(4)):
                        d[q] = self.axes._node_offset(a, family, i0 + q, relative) / local
                    w, dw = correct_first_moment(w, dw, d, local)
                for q in ti.static(range(4)):
                    i = i0 + q
                    if i < 0 or i >= self.axes.n[a]:
                        w[q] = 0.0
                        dw[q] = 0.0
        return i0, w, dw

    @ti.func
    def face_volume(self, c: ti.template(), i, j, k):
        """The MAC face control volume of the axes (m^3)."""
        return self.axes.face_volume(c, i, j, k)
