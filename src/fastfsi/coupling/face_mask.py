"""Validity of MAC faces for the one-sided immersed-boundary support.

The coupler normalises its kernel weights over the valid faces only (w~ = w m / D with D = sum w m), so a point next to
an obstacle, an air region or the domain edge interpolates and spreads one-sidedly instead of through faces that carry
no physical velocity.

FaceValidityMask: a MAC face of component c at (i, j, k) (between cells (i, j, k) - e_c and (i, j, k)) is valid iff
neither adjacent cell is an obstacle (fluid.obstacle != 0), the face is not the domain-edge face (index 0 along c), and,
when air cells are given, the two cells are not both air cells.  With exclude_faces_touching_air=True a face next to
ANY air cell is invalid (the air acts on a body through its pressure and carries no velocity for the solid to follow).
The bits are stored as one u8 per cell (bit c).

AllValidFaceMask: every face valid (tests and obstacle-free set-ups).

`version` changes whenever the validity may have changed (at every refresh); a coupler reuses quantities computed from
the mask while it stays the same.  Change the validity only through refresh().

No `from __future__ import annotations` (Taichi kernel module).
"""
import taichi as ti


@ti.data_oriented
class FaceValidityMask:
    """One-sided normalisation validity of every MAC face, from fluid.obstacle and the air cells."""

    def __init__(self, fluid, axes, exclude_faces_touching_air=False):
        shape = tuple(int(v) for v in fluid.obstacle.shape)
        if shape != tuple(int(v) for v in axes.n):
            raise ValueError("fluid grid %s does not match the axes %s" % (shape, tuple(axes.n)))
        self.fluid = fluid
        self.n = shape
        self.bits = ti.field(ti.u8, shape=shape)          # bit c set: the face of component c at (i, j, k) is valid
        self.valid_count = ti.field(ti.i32, shape=3)
        self.with_air = False
        # with air cells: True = a face next to ANY air cell is invalid; False = a face between TWO air cells is invalid
        self.exclude_faces_touching_air = bool(exclude_faces_touching_air)
        self.version = 0
        self.refresh()

    @ti.kernel
    def _refresh_kernel(self, obstacle: ti.template(), air: ti.template(), use_air: ti.template(),
                        touching: ti.template()):
        for c in ti.static(range(3)):
            self.valid_count[c] = 0
        for i, j, k in obstacle:
            index = ti.Vector([i, j, k])
            bits = 0
            if obstacle[i, j, k] == 0:
                for c in ti.static(range(3)):
                    if index[c] >= 1:                               # index 0 along c: the domain-edge face
                        neighbour = index - ti.Vector.unit(3, c, ti.i32)
                        if obstacle[neighbour] == 0:
                            valid = 1
                            if ti.static(use_air):
                                if ti.static(touching):             # next to ANY air cell
                                    if air[i, j, k] != 0 or air[neighbour] != 0:
                                        valid = 0
                                else:                               # between two air cells
                                    if air[i, j, k] != 0 and air[neighbour] != 0:
                                        valid = 0
                            if valid != 0:
                                bits |= 1 << c
                                ti.atomic_add(self.valid_count[c], 1)
            self.bits[i, j, k] = ti.cast(bits, ti.u8)

    def refresh(self, air_cell_field=None):
        """Recompute the validity bits; call after any obstacle or air-cell change."""
        self.version += 1
        if air_cell_field is None:
            self.with_air = False
            self._refresh_kernel(self.fluid.obstacle, self.fluid.obstacle, False, False)
            return
        if tuple(int(v) for v in air_cell_field.shape) != self.n:
            raise ValueError("air cell field %s does not match the grid %s" % (tuple(air_cell_field.shape), self.n))
        self.with_air = True
        self._refresh_kernel(self.fluid.obstacle, air_cell_field, True, self.exclude_faces_touching_air)

    @ti.func
    def face_valid(self, c: ti.template(), i, j, k):
        """1 if the MAC face of component c at (i, j, k) may carry kernel weight, else 0 (i32)."""
        return (ti.cast(self.bits[i, j, k], ti.i32) >> c) & 1

    def counts(self):
        """Valid faces per component after the last refresh (diagnostics)."""
        values = self.valid_count.to_numpy()
        return dict(valid_faces=[int(v) for v in values], with_air=bool(self.with_air))


@ti.data_oriented
class AllValidFaceMask:
    """Every MAC face valid (tests and obstacle-free set-ups)."""
    version = 0                     # the validity never changes

    def refresh(self, air_cell_field=None):
        return None

    @ti.func
    def face_valid(self, c: ti.template(), i, j, k):
        return 1
