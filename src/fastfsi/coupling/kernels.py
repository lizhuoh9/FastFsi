"""Regularized delta functions of the immersed-boundary method.

IB_4 is Peskin's four-point kernel (C. S. Peskin, "The immersed boundary method", Acta Numerica 11 (2002) 479-517):
support |r| < 2, unit sum, zero first moment, sum of squares 3/8 on a uniform grid.

IB_3 is the three-point kernel of A. M. Roma, C. S. Peskin and M. J. Berger, "An adaptive version of the immersed
boundary method", J. Comput. Phys. 153 (1999) 509-534: support |r| < 1.5, unit sum, zero first moment, sum of
squares 1/2.  It is evaluated on the same four-node frame as IB_4 (nodes floor(s) - 1 .. floor(s) + 2), which always
contains its three nodes with |r| < 1.5; the fourth weight is zero.

Both are the kernels of IBAMR's IBTK library (IB_4, IB_3; BSD-3-Clause); the formulas are implemented here, no source
is copied.  r is the signed distance in units of the local node spacing; the derivatives are d/dr.
"""
import taichi as ti

KERNEL_NAMES = ("ib4", "ib3")


@ti.func
def evaluate_ib4_weight(r):
    """Peskin 4-point regularized delta (IBAMR IBTK IB_4), support |r| < 2, unit sum and zero first moment."""
    a = ti.abs(r)
    value = 0.0
    if a < 1.0:
        value = (3.0 - 2.0 * a + ti.sqrt(ti.max(1.0 + 4.0 * a - 4.0 * a * a, 0.0))) / 8.0
    elif a < 2.0:
        value = (5.0 - 2.0 * a - ti.sqrt(ti.max(-7.0 + 12.0 * a - 4.0 * a * a, 0.0))) / 8.0
    return value


@ti.func
def evaluate_ib3_weight(r):
    """Roma, Peskin & Berger (1999) 3-point regularized delta (IBAMR IBTK 'IB_3'): support |r| < 1.5, unit sum, zero
    first moment and sum of squares 1/2 on a uniform grid.  Evaluated on the IB_4 4-node frame, which always contains
    the 3 nodes with |r| < 1.5 (the 4th weight is 0)."""
    a = ti.abs(r)
    value = 0.0
    if a < 0.5:
        value = (1.0 + ti.sqrt(ti.max(1.0 - 3.0 * a * a, 0.0))) / 3.0
    elif a < 1.5:
        b = 1.0 - a
        value = (5.0 - 3.0 * a - ti.sqrt(ti.max(1.0 - 3.0 * b * b, 0.0))) / 6.0
    return value


@ti.func
def evaluate_ib3_weight_derivative(r):
    """d phi / d r of IB_3 (continuous, -1 at |r| = 0.5 from both sides, zero at |r| = 1.5)."""
    a = ti.abs(r)
    slope = 0.0
    if a < 0.5:
        root = ti.sqrt(ti.max(1.0 - 3.0 * a * a, 1.0e-30))
        slope = -a / root
    elif a < 1.5:
        b = 1.0 - a
        root = ti.sqrt(ti.max(1.0 - 3.0 * b * b, 1.0e-30))
        slope = -0.5 * (1.0 + b / root)
    sign = 1.0
    if r < 0.0:
        sign = -1.0
    return sign * slope


def validate_kernel_name(name):
    """The kernel name, 'ib4' (default) or 'ib3'; anything else is refused."""
    if name not in KERNEL_NAMES:
        raise ValueError("ib_kernel must be one of %s, got %r" % (KERNEL_NAMES, name))
    return name


@ti.func
def evaluate_ib4_weight_derivative(r):
    """d phi / d r of IB_4 (continuous, zero at |r| = 2)."""
    a = ti.abs(r)
    slope = 0.0
    if a < 1.0:
        root = ti.sqrt(ti.max(1.0 + 4.0 * a - 4.0 * a * a, 1.0e-30))
        slope = (-2.0 + (2.0 - 4.0 * a) / root) / 8.0
    elif a < 2.0:
        root = ti.sqrt(ti.max(-7.0 + 12.0 * a - 4.0 * a * a, 1.0e-30))
        slope = (-2.0 - (6.0 - 4.0 * a) / root) / 8.0
    sign = 1.0
    if r < 0.0:
        sign = -1.0
    return sign * slope
