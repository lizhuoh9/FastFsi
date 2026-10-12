"""REQ_401 fast mode, simulation_core/fast/graded_kernel.py (contract 01 §1.1, §1.2, T7 items 1-2): CPU tests.

- uniform axes reproduce phi(xi - i) bit for bit (and fluid_hooks.UniformAxes when it is importable);
- the graded index map hits every node, is monotone and equals the piecewise-linear reference map;
- raw G1: partition of unity, zero first moment in index space, sum dw = 0, sum (i0 + q) dw = 1 / local spacing;
- moment-corrected (default): sum w = 1, sum w (x_q - x) = 0, sum dw = 0, sum dw (x_q - x) = 1 in physical space;
- dw is the physical derivative d w / d coord (1/m), checked by central differences inside a segment (raw, corrected);
- face_volume is the MAC momentum control volume (half cell at index 0);
- T7 item 1 (force conservation) and item 2 (spread / interpolation power identity) to f32 roundoff, for a point force
  and the stress term, on a uniform and on a graded grid, and with the one-sided normalisation next to an obstacle,
  air cells and the domain edge (FaceMask);
- Root's linear-field table at stretch ratio 1.0 / 1.05 / 1.1 / 1.2: raw G1 within the summation-by-parts bound,
  corrected weights exact (value and gradient) to f32 roundoff; min w~ and sum |w~| (boundedness) are reported;
- FaceMask validity cases (obstacle neighbour, both-air, domain edge) against an independent numpy rule;
- a non-finite coordinate gets zero weights (the body's D = 0 hard stop), FastModeHardStop codes.

Roundoff bounds: an identity holds to f32 roundoff when |lhs - rhs| <= 1e-5 * (sum of the magnitudes of its terms);
a modelling inconsistency (a wrong volume, weight or derivative) gives errors of order 1e-2 .. 1 of that scale.
"""
import functools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from simulation_core.fast.graded_kernel import (HARD_STOP_CODES, FaceMask, FastModeHardStop, GradedAxes)
from simulation_core.fast.ib_mpm_fast import ib4_dphi, ib4_phi
from simulation_core.fluids.grid import CartesianGrid

ROUNDOFF = 1.0e-5


def graded_faces(origin, n, h0, ratio, core):
    """Faces of an axis with a uniform core of `core` cells of width h0 and geometric growth `ratio` on both sides."""
    left = (n - core) // 2
    right = n - core - left
    widths = ([h0 * ratio ** (left - i) for i in range(left)] + [h0] * core
              + [h0 * ratio ** (i + 1) for i in range(right)])
    return origin + np.concatenate([[0.0], np.cumsum(widths)])


GRADED_FACES = (graded_faces(-0.012, 14, 1.0e-3, 1.1, 4), graded_faces(0.004, 12, 1.4e-3, 1.08, 2),
                graded_faces(0.98, 16, 6.0e-4, 1.1, 6))


@functools.lru_cache(maxsize=None)
def graded_axes():
    """Moment-corrected (default); one instance per session (template-argument kernels compile once)."""
    return GradedAxes(*GRADED_FACES)


@functools.lru_cache(maxsize=None)
def graded_axes_raw():
    return GradedAxes(*GRADED_FACES, moment_correction=False)


@functools.lru_cache(maxsize=None)
def graded_axes_r12(corrected=True):
    """Production maximum adjacent stretch 1.2 along x and z (band transitions at faces 6 / 10), uniform y."""
    return GradedAxes(graded_faces(-0.01, 16, 1.0e-3, 1.2, 4), 0.004 + 1.2e-3 * np.arange(13),
                      graded_faces(0.98, 16, 6.0e-4, 1.2, 4), moment_correction=corrected)


UNIFORM_ORIGIN = (-0.02, 0.01, 0.98)
UNIFORM_H = (1.0e-3, 1.25e-3, 6.0e-4)
UNIFORM_N = (10, 12, 14)


@functools.lru_cache(maxsize=None)
def uniform_axes():
    return GradedAxes(*[UNIFORM_ORIGIN[a] + UNIFORM_H[a] * np.arange(UNIFORM_N[a] + 1) for a in range(3)])


@ti.data_oriented
class _AllValid:
    @ti.func
    def face_valid(self, c: ti.template(), i, j, k):
        return 1


ALL_VALID = _AllValid()


# ---------------------------------------------------------------------------------------------------- test kernels
@ti.kernel
def _weights(axes: ti.template(), a: ti.template(), staggered: ti.template(), coords: ti.types.ndarray(),
             i0_out: ti.types.ndarray(), w_out: ti.types.ndarray(), dw_out: ti.types.ndarray(),
             xi_out: ti.types.ndarray(), local_out: ti.types.ndarray()):
    for p in range(coords.shape[0]):
        i0, w, dw = axes.axis_weights(a, staggered, coords[p])
        xi, local = axes.index_coordinate(a, staggered, coords[p])
        i0_out[p] = i0
        xi_out[p] = xi
        local_out[p] = local
        for q in ti.static(range(4)):
            w_out[p, q] = w[q]
            dw_out[p, q] = dw[q]


@ti.kernel
def _weights_uniform(axes: ti.template(), a: ti.template(), staggered: ti.template(), coords: ti.types.ndarray(),
                     i0_out: ti.types.ndarray(), w_out: ti.types.ndarray(), dw_out: ti.types.ndarray()):
    for p in range(coords.shape[0]):
        i0, w, dw = axes.axis_weights(a, staggered, coords[p])
        i0_out[p] = i0
        for q in ti.static(range(4)):
            w_out[p, q] = w[q]
            dw_out[p, q] = dw[q]


@ti.kernel
def _uniform_reference(origin: ti.template(), h: ti.template(), shift: ti.template(), coords: ti.types.ndarray(),
                       i0_out: ti.types.ndarray(), w_out: ti.types.ndarray(), dw_out: ti.types.ndarray()):
    """The contract's uniform reduction: w_q = phi(xi - (i0 + q)), xi = (coord - origin) / h - shift."""
    for p in range(coords.shape[0]):
        s = (coords[p] - origin) / h - shift
        i0 = ti.cast(ti.floor(s), ti.i32) - 1
        i0_out[p] = i0
        for q in ti.static(range(4)):
            r = s - ti.cast(i0 + q, ti.f32)
            w_out[p, q] = ib4_phi(r)
            dw_out[p, q] = ib4_dphi(r) / h


@ti.kernel
def _face_volumes(axes: ti.template(), out: ti.types.ndarray()):
    for c in ti.static(range(3)):
        for i, j, k in ti.ndrange(axes.n[0], axes.n[1], axes.n[2]):
            out[c, i, j, k] = axes.face_volume(c, i, j, k)


@ti.kernel
def _face_valid_dump(mask: ti.template(), out: ti.types.ndarray()):
    for c in ti.static(range(3)):
        for i, j, k in ti.ndrange(out.shape[1], out.shape[2], out.shape[3]):
            out[c, i, j, k] = mask.face_valid(c, i, j, k)


@ti.func
def _support(axes: ti.template(), mask: ti.template(), c: ti.template(), xp):
    """Per-axis weights of MAC component c at xp, the one-sided denominator D and its gradient (the body's rule)."""
    ix, wx, dwx = axes.axis_weights(0, 1 if c == 0 else 0, xp[0])
    iy, wy, dwy = axes.axis_weights(1, 1 if c == 1 else 0, xp[1])
    iz, wz, dwz = axes.axis_weights(2, 1 if c == 2 else 0, xp[2])
    D = 0.0
    gD = ti.Vector([0.0, 0.0, 0.0])
    for a, b, e in ti.static(ti.ndrange(4, 4, 4)):
        i, j, k = ix + a, iy + b, iz + e
        if 0 <= i < axes.n[0] and 0 <= j < axes.n[1] and 0 <= k < axes.n[2]:
            m = ti.cast(mask.face_valid(c, i, j, k), ti.f32)
            D += m * wx[a] * wy[b] * wz[e]
            gD += m * ti.Vector([dwx[a] * wy[b] * wz[e], wx[a] * dwy[b] * wz[e], wx[a] * wy[b] * dwz[e]])
    return ix, iy, iz, wx, wy, wz, dwx, dwy, dwz, D, gD


@ti.kernel
def _spread(axes: ti.template(), mask: ti.template(), x: ti.types.ndarray(), force_n: ti.types.ndarray(),
            tau_v0: ti.types.ndarray(), out: ti.template(), denominator: ti.types.ndarray()):
    """f_density = (F_c w~ - V0 tau_c . grad w~) / dV_face with w~ = w m / D (one-sided normalisation)."""
    for p in range(x.shape[0]):
        xp = ti.Vector([x[p, 0], x[p, 1], x[p, 2]])
        for c in ti.static(range(3)):
            ix, iy, iz, wx, wy, wz, dwx, dwy, dwz, D, gD = _support(axes, mask, c, xp)
            denominator[p, c] = D
            if D > 0.0:
                for a, b, e in ti.static(ti.ndrange(4, 4, 4)):
                    i, j, k = ix + a, iy + b, iz + e
                    if 0 <= i < axes.n[0] and 0 <= j < axes.n[1] and 0 <= k < axes.n[2]:
                        if mask.face_valid(c, i, j, k) != 0:
                            w = wx[a] * wy[b] * wz[e] / D
                            g = (ti.Vector([dwx[a] * wy[b] * wz[e], wx[a] * dwy[b] * wz[e],
                                            wx[a] * wy[b] * dwz[e]]) - w * gD) / D
                            f = force_n[p, c] * w - (tau_v0[p, c, 0] * g[0] + tau_v0[p, c, 1] * g[1]
                                                     + tau_v0[p, c, 2] * g[2])
                            out[i, j, k][c] += f / axes.face_volume(c, i, j, k)


@ti.kernel
def _interpolate(axes: ti.template(), mask: ti.template(), x: ti.types.ndarray(), u: ti.template(),
                 v_out: ti.types.ndarray(), g_out: ti.types.ndarray()):
    for p in range(x.shape[0]):
        xp = ti.Vector([x[p, 0], x[p, 1], x[p, 2]])
        for c in ti.static(range(3)):
            ix, iy, iz, wx, wy, wz, dwx, dwy, dwz, D, gD = _support(axes, mask, c, xp)
            v = 0.0
            G = ti.Vector([0.0, 0.0, 0.0])
            if D > 0.0:
                for a, b, e in ti.static(ti.ndrange(4, 4, 4)):
                    i, j, k = ix + a, iy + b, iz + e
                    if 0 <= i < axes.n[0] and 0 <= j < axes.n[1] and 0 <= k < axes.n[2]:
                        if mask.face_valid(c, i, j, k) != 0:
                            w = wx[a] * wy[b] * wz[e] / D
                            g = (ti.Vector([dwx[a] * wy[b] * wz[e], wx[a] * dwy[b] * wz[e],
                                            wx[a] * wy[b] * dwz[e]]) - w * gD) / D
                            v += u[i, j, k][c] * w
                            G += u[i, j, k][c] * g
            v_out[p, c] = v
            for d in ti.static(range(3)):
                g_out[p, c, d] = G[d]


# ---------------------------------------------------------------------------------------------------- helpers
def axis_weights(axes, a, staggered, coords):
    coords = np.ascontiguousarray(np.asarray(coords, dtype=np.float32))
    n = len(coords)
    i0 = np.zeros(n, np.int32)
    w = np.zeros((n, 4), np.float32)
    dw = np.zeros((n, 4), np.float32)
    xi = np.zeros(n, np.float32)
    local = np.zeros(n, np.float32)
    _weights(axes, a, staggered, coords, i0, w, dw, xi, local)
    return i0, w, dw, xi, local


def node_family(axes, a, staggered):
    """Physical node coordinates (float64) of a family with one mirrored ghost node on each side; index i = g - 1."""
    faces, widths, centres = axes.faces_m[a], axes.widths_m[a], axes.centres_m[a]
    base = faces if staggered else centres
    return np.concatenate([[base[0] - widths[0]], base, [base[-1] + widths[-1]]])


def away_from_nodes(coords, family):
    """Drop coordinates within a few f32 ulps of a node (either adjacent segment may hold them)."""
    gap = np.abs(coords.astype(np.float64)[:, None] - family[None, :]).min(axis=1)
    return coords[gap > 8.0 * np.spacing(np.abs(coords)).astype(np.float64)]


def interior_points(axes, rng, count, margin_cells=3):
    """Random points at least `margin_cells` cells away from every domain edge (all 4 nodes stored, every axis)."""
    lo = [axes.faces_m[a][margin_cells] for a in range(3)]
    hi = [axes.faces_m[a][-1 - margin_cells] for a in range(3)]
    return np.column_stack([rng.uniform(lo[a], hi[a], count) for a in range(3)]).astype(np.float32)


def spread_and_interpolate(axes, mask, points, forces, tau_v0, velocity):
    n = axes.n
    out = ti.Vector.field(3, ti.f32, shape=n)
    u = ti.Vector.field(3, ti.f32, shape=n)
    out.fill(0.0)
    u.from_numpy(velocity.astype(np.float32))
    denominator = np.zeros((len(points), 3), np.float32)
    _spread(axes, mask, points, forces.astype(np.float32), tau_v0.astype(np.float32), out, denominator)
    v = np.zeros((len(points), 3), np.float32)
    g = np.zeros((len(points), 3, 3), np.float32)
    _interpolate(axes, mask, points, u, v, g)
    volumes = np.zeros((3,) + tuple(n), np.float32)
    _face_volumes(axes, volumes)
    density = np.moveaxis(out.to_numpy().astype(np.float64), -1, 0)          # (3, nx, ny, nz)
    return density, volumes.astype(np.float64), v.astype(np.float64), g.astype(np.float64), denominator


def check_conservation_and_power(density, volumes, velocity, forces, tau_v0, v, g):
    u = np.moveaxis(velocity.astype(np.float32).astype(np.float64), -1, 0)
    total = (density * volumes).reshape(3, -1).sum(axis=1)
    expected = forces.astype(np.float32).astype(np.float64).sum(axis=0)
    scale = (np.abs(density) * volumes).reshape(3, -1).sum(axis=1)
    assert np.all(np.abs(total - expected) <= ROUNDOFF * scale), (total, expected, scale)
    grid_power_terms = density * u * volumes
    grid_power = grid_power_terms.sum()
    f = forces.astype(np.float32).astype(np.float64)
    t = tau_v0.astype(np.float32).astype(np.float64)
    point_power_terms = np.concatenate([(f * v).ravel(), -(t * g).ravel()])
    point_power = point_power_terms.sum()
    bound = ROUNDOFF * (np.abs(grid_power_terms).sum() + np.abs(point_power_terms).sum())
    assert abs(grid_power - point_power) <= bound, (grid_power, point_power, bound)


# ---------------------------------------------------------------------------------------------------- tests
@pytest.mark.parametrize("staggered", [0, 1])
@pytest.mark.parametrize("a", [0, 1, 2])
def test_uniform_axes_reproduce_phi_exactly(a, staggered):
    axes = uniform_axes()
    assert axes.uniform == (True, True, True)
    rng = np.random.default_rng(10 + 3 * a + staggered)
    lo, hi = UNIFORM_ORIGIN[a], UNIFORM_ORIGIN[a] + UNIFORM_H[a] * UNIFORM_N[a]
    coords = np.concatenate([rng.uniform(lo, hi, 400), lo + UNIFORM_H[a] * np.arange(UNIFORM_N[a] + 1),
                             [lo + 0.25 * UNIFORM_H[a], hi - 0.25 * UNIFORM_H[a]]]).astype(np.float32)
    i0, w, dw, _, _ = axis_weights(axes, a, staggered, coords)
    ref_i0 = np.zeros_like(i0)
    ref_w = np.zeros_like(w)
    ref_dw = np.zeros_like(dw)
    _uniform_reference(UNIFORM_ORIGIN[a], axes.h[a], 0.0 if staggered else 0.5, coords, ref_i0, ref_w, ref_dw)
    assert np.array_equal(i0, ref_i0)
    stored = (ref_i0[:, None] + np.arange(4)[None, :] >= 0) & (ref_i0[:, None] + np.arange(4)[None, :] < UNIFORM_N[a])
    assert np.array_equal(w, np.where(stored, ref_w, 0.0))         # bit for bit on the stored nodes, zero outside
    assert np.array_equal(dw, np.where(stored, ref_dw, 0.0))
    assert stored.all(axis=1).sum() > 0.4 * len(coords)              # most samples have all four nodes stored


def test_uniform_axes_equal_the_integrator_uniform_axes_bit_for_bit():
    hooks = pytest.importorskip("simulation_core.fast.fluid_hooks")
    graded = uniform_axes()
    uniform = hooks.UniformAxes(UNIFORM_ORIGIN, graded.h, UNIFORM_N)
    rng = np.random.default_rng(7)
    for a in range(3):
        for staggered in (0, 1):
            lo, hi = UNIFORM_ORIGIN[a], UNIFORM_ORIGIN[a] + UNIFORM_H[a] * UNIFORM_N[a]
            coords = rng.uniform(lo, hi, 200).astype(np.float32)
            i0, w, dw, _, _ = axis_weights(graded, a, staggered, coords)
            j0, v, dv = np.zeros_like(i0), np.zeros_like(w), np.zeros_like(dw)
            _weights_uniform(uniform, a, staggered, coords, j0, v, dv)
            stored = (j0[:, None] + np.arange(4) >= 0) & (j0[:, None] + np.arange(4) < UNIFORM_N[a])
            assert np.array_equal(i0, j0)
            assert np.array_equal(w, np.where(stored, v, 0.0))
            assert np.array_equal(dw, np.where(stored, dv, 0.0))


@pytest.mark.parametrize("staggered", [0, 1])
@pytest.mark.parametrize("a", [0, 1, 2])
def test_graded_index_map_hits_the_nodes_and_is_the_piecewise_linear_map(a, staggered):
    axes = graded_axes()
    assert not axes.uniform[a]
    assert max(axes.max_adjacent_stretch) == pytest.approx(1.1, rel=1e-9)
    family = node_family(axes, a, staggered)
    index = np.arange(len(family)) - 1.0
    nodes32 = family.astype(np.float32)
    _, _, _, xi_nodes, _ = axis_weights(axes, a, staggered, nodes32)
    reference = np.interp(nodes32.astype(np.float64), family, index)
    np.testing.assert_allclose(xi_nodes, reference, atol=2e-5)
    np.testing.assert_allclose(xi_nodes[1:-1], index[1:-1], atol=2e-3)   # node coordinates themselves are f32
    sweep = np.linspace(family[0], family[-1], 3001).astype(np.float32)
    _, _, _, xi, _ = axis_weights(axes, a, staggered, sweep)
    assert np.all(np.diff(xi.astype(np.float64)) >= 0.0)
    np.testing.assert_allclose(xi, np.interp(sweep.astype(np.float64), family, index), atol=2e-5)


def _full_support_samples(axes, a, staggered, seed, count=2000):
    rng = np.random.default_rng(seed)
    family = node_family(axes, a, staggered)
    coords = away_from_nodes(rng.uniform(axes.faces_m[a][0], axes.faces_m[a][-1], count).astype(np.float32), family)
    i0, w, dw, xi, local = axis_weights(axes, a, staggered, coords)
    full = (i0 >= 0) & (i0 + 3 < axes.n[a])
    assert full.sum() > count // 2
    nodes = i0[full, None] + np.arange(4)[None, :]
    offsets = family[nodes + 1] - coords[full].astype(np.float64)[:, None]     # x_q - x (m)
    return (w[full].astype(np.float64), dw[full].astype(np.float64), nodes, xi[full].astype(np.float64),
            local[full].astype(np.float64), offsets, family, coords[full])


@pytest.mark.parametrize("staggered", [0, 1])
@pytest.mark.parametrize("a", [0, 1, 2])
def test_raw_g1_moments_in_index_space(a, staggered):
    axes = graded_axes_raw()
    w, dw, nodes, xi, local, _, family, coords = _full_support_samples(axes, a, staggered, 100 + 3 * a + staggered)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose((w * nodes).sum(axis=1), xi, atol=1e-5)
    assert np.all(np.abs(dw.sum(axis=1)) <= ROUNDOFF * np.abs(dw).sum(axis=1))
    segment = np.clip(np.searchsorted(family, coords.astype(np.float64), side="right") - 1, 0, len(family) - 2)
    np.testing.assert_allclose((dw * nodes).sum(axis=1), 1.0 / np.diff(family)[segment], rtol=1e-4)


@pytest.mark.parametrize("staggered", [0, 1])
@pytest.mark.parametrize("a", [0, 1, 2])
def test_corrected_moments_in_physical_space(a, staggered):
    """sum w = 1, sum w (x_q - x) = 0, sum dw = 0, sum dw (x_q - x) = 1 (to f32 roundoff); nodes outside the stored
    range carry nothing."""
    axes = graded_axes()
    w, dw, _, _, local, offsets, _, _ = _full_support_samples(axes, a, staggered, 200 + 3 * a + staggered)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(np.abs((w * offsets).sum(axis=1)) <= ROUNDOFF * (np.abs(w * offsets).sum(axis=1) + 1e-3 * local))
    assert np.all(np.abs(dw.sum(axis=1)) <= ROUNDOFF * np.abs(dw).sum(axis=1))
    np.testing.assert_allclose((dw * offsets).sum(axis=1), 1.0, atol=ROUNDOFF)
    coords = np.linspace(axes.faces_m[a][0], axes.faces_m[a][-1], 997).astype(np.float32)
    i0, w32, dw32, _, _ = axis_weights(axes, a, staggered, coords)
    outside = (i0[:, None] + np.arange(4)[None, :] < 0) | (i0[:, None] + np.arange(4)[None, :] >= axes.n[a])
    assert outside.any() and np.all(w32[outside] == 0.0) and np.all(dw32[outside] == 0.0)


@pytest.mark.parametrize("corrected", [False, True])
@pytest.mark.parametrize("a", [0, 2])
def test_dw_is_the_physical_derivative_of_w(a, corrected):
    axes = graded_axes() if corrected else graded_axes_raw()
    for staggered in (0, 1):
        family = node_family(axes, a, staggered)
        mids = 0.5 * (family[1:-2] + family[2:-1])
        step = 0.02 * np.diff(family)[1:-1]
        plus = (mids + step).astype(np.float32)
        minus = (mids - step).astype(np.float32)
        centre = mids.astype(np.float32)
        i_p, w_p, _, _, _ = axis_weights(axes, a, staggered, plus)
        i_m, w_m, _, _, _ = axis_weights(axes, a, staggered, minus)
        i_c, _, dw_c, _, _ = axis_weights(axes, a, staggered, centre)
        same = (i_p == i_c) & (i_m == i_c)
        assert same.sum() >= len(mids) - 2
        h_eff = plus.astype(np.float64) - minus.astype(np.float64)          # the f32 coordinates actually used
        fd = (w_p.astype(np.float64) - w_m.astype(np.float64)) / h_eff[:, None]
        scale = np.abs(dw_c).max()
        assert np.all(np.abs(fd[same] - dw_c[same]) <= 2e-3 * scale)


def test_face_volume_is_the_mac_control_volume():
    for axes in (graded_axes(), uniform_axes()):
        volumes = np.zeros((3,) + tuple(axes.n), np.float32)
        _face_volumes(axes, volumes)
        for c in range(3):
            distance = np.concatenate([[0.5 * axes.widths_m[c][0]], np.diff(axes.centres_m[c])])
            factors = [axes.widths_m[a] for a in range(3)]
            factors[c] = distance
            expected = factors[0][:, None, None] * factors[1][None, :, None] * factors[2][None, None, :]
            np.testing.assert_allclose(volumes[c], expected, rtol=1e-6)
    axes = uniform_axes()
    volumes = np.zeros((3,) + tuple(axes.n), np.float32)
    _face_volumes(axes, volumes)
    np.testing.assert_allclose(volumes[:, 1:, 1:, 1:], np.prod(UNIFORM_H), rtol=1e-6)


@pytest.mark.parametrize("kind", ["uniform", "graded", "graded_raw", "graded_r12"])
def test_force_conservation_and_power_identity_T7(kind):
    axes = {"uniform": uniform_axes, "graded": graded_axes, "graded_raw": graded_axes_raw,
            "graded_r12": graded_axes_r12}[kind]()
    rng = np.random.default_rng(42)
    points = interior_points(axes, rng, 24)
    forces = rng.normal(size=(24, 3)) * 1e-3
    tau_v0 = rng.normal(size=(24, 3, 3)) * 1e-5
    velocity = rng.normal(size=tuple(axes.n) + (3,))
    density, volumes, v, g, denominator = spread_and_interpolate(axes, ALL_VALID, points, forces, tau_v0, velocity)
    np.testing.assert_allclose(denominator, 1.0, atol=1e-6)            # unmasked interior: D = sum w = 1
    check_conservation_and_power(density, volumes, velocity, forces, tau_v0, v, g)
    # the stress term alone carries no net force (sum grad w = 0) and its power is -V0 tau : G
    zero = np.zeros_like(forces)
    density, volumes, v, g, _ = spread_and_interpolate(axes, ALL_VALID, points, zero, tau_v0, velocity)
    check_conservation_and_power(density, volumes, velocity, zero, tau_v0, v, g)


def test_interpolation_of_constant_and_linear_fields_in_3d():
    """Corrected graded grid: a constant field is reproduced with zero gradient; u_z = z - z0 (linear in physical
    space) gives the exact value and G_zz = 1, G_zx = G_zy = 0 with the body's product weights (f32 roundoff)."""
    axes = graded_axes()
    rng = np.random.default_rng(5)
    points = interior_points(axes, rng, 16)
    velocity = np.zeros(tuple(axes.n) + (3,))
    velocity[..., 0], velocity[..., 1], velocity[..., 2] = 0.3, -0.2, 0.7
    _, _, v, g, _ = spread_and_interpolate(axes, ALL_VALID, points, np.zeros((16, 3)), np.zeros((16, 3, 3)),
                                           velocity)
    np.testing.assert_allclose(v, np.tile([0.3, -0.2, 0.7], (16, 1)), atol=2e-6)
    np.testing.assert_allclose(g, 0.0, atol=1e-2)                      # scale |u| / h ~ 1e3 1/s
    faces_z = axes.faces_m[2][:-1]
    velocity = np.zeros(tuple(axes.n) + (3,))
    velocity[..., 2] = (faces_z - axes.faces_m[2][0])[None, None, :]
    _, _, v, g, _ = spread_and_interpolate(axes, ALL_VALID, points, np.zeros((16, 3)), np.zeros((16, 3, 3)),
                                           velocity)
    exact = points[:, 2].astype(np.float64) - axes.faces_m[2][0]
    assert np.all(np.abs(v[:, 2] - exact) <= 1e-5 * axes.widths_m[2].min())
    np.testing.assert_allclose(g[:, 2, 2], 1.0, atol=1e-4)
    assert np.all(np.abs(g[:, 2, :2]) <= 1e-4)


GRADIENT_RATIOS = (1.0, 1.05, 1.1, 1.2)


@functools.lru_cache(maxsize=None)
def ratio_axes(ratio, corrected):
    """A 40-cell axis: uniform core of 8 cells (h0 = 1 mm) and geometric growth `ratio` on both sides (all axes)."""
    faces = graded_faces(0.97, 40, 1.0e-3, ratio, 8)
    return GradedAxes(faces, faces, faces, moment_correction=corrected)


@pytest.mark.parametrize("ratio", GRADIENT_RATIOS)
def test_linear_field_value_and_gradient_vs_stretch_ratio(ratio):
    """Root (2026-10-12): for u = a x, G = sum_q dw_q u(x_q) and U = sum_q w_q u(x_q).
    Raw G1: exact where the index map is linear over the 4-node support; elsewhere, with s_m the support spacings and
    s_seg the spacing of the segment holding x, summation by parts gives sum_q dw_q X_q = sum_m Lambda_m s_m / s_seg,
    Lambda_m = sum_{q > m} phi'_q >= 0, sum Lambda = 1, so G / a - 1 lies in [min s_m / s_seg - 1, max s_m / s_seg - 1].
    Moment-corrected: G = a and U = u(x) to f32 roundoff for every ratio.  Printed: GRADED_KERNEL_GRADIENT rows with the
    maximum relative gradient error (raw, corrected), the corrected value error and min w~ / max sum |w~|."""
    slope = 2.5
    for a in (0, 2):
        for staggered in (0, 1):
            results = {}
            for corrected in (False, True):
                axes = ratio_axes(ratio, corrected)
                family = node_family(axes, a, staggered)                      # node index i sits at family[i + 1]
                coords = away_from_nodes(
                    np.linspace(axes.faces_m[a][0], axes.faces_m[a][-1], 8001)[1:-1].astype(np.float32), family)
                i0, w, dw, _, local = axis_weights(axes, a, staggered, coords)
                full = (i0 >= 0) & (i0 + 3 < axes.n[a])
                nodes = i0[full, None] + np.arange(4)[None, :]
                w64, dw64 = w[full].astype(np.float64), dw[full].astype(np.float64)
                np.testing.assert_allclose(w64.sum(axis=1), 1.0, atol=1e-6)
                assert np.all(np.abs(dw64.sum(axis=1)) <= ROUNDOFF * np.abs(dw64).sum(axis=1))
                # u = slope (x - x_ref) with x_ref the node i0 + 1 of each point (offset-free: no f32 cancellation)
                X = family[nodes + 1] - family[nodes[:, 1:2] + 1]
                x_rel = coords[full].astype(np.float64) - family[nodes[:, 1] + 1]
                rel = (dw64 * slope * X).sum(axis=1) / slope - 1.0
                value_error = np.abs((w64 * X).sum(axis=1) - x_rel) / local[full].astype(np.float64)
                spacings = np.diff(family)[nodes[:, :3] + 1]                # between nodes q and q + 1, q = 0, 1, 2
                linear = np.isclose(spacings.min(axis=1), spacings.max(axis=1), rtol=1e-9, atol=0.0)
                assert linear.sum() > 100
                assert np.all(np.abs(rel[linear]) <= 1e-5)                  # exact where the map is linear (both)
                if corrected:
                    assert np.all(np.abs(rel) <= 1e-5)                      # exact gradient for every ratio
                    # exact value for every ratio, to the f32 resolution of the absolute coordinate (on a uniform
                    # axis the contract's formula subtracts the f32-rounded origin: <= 1/2 ulp of the origin)
                    ulp = np.spacing(np.abs(coords[full])).astype(np.float64) / local[full].astype(np.float64)
                    assert np.all(value_error <= 1e-5 + 2.0 * ulp)
                else:
                    seg = local[full].astype(np.float64)
                    lower = spacings.min(axis=1) / seg - 1.0
                    upper = spacings.max(axis=1) / seg - 1.0
                    assert np.all(rel >= lower - 1e-5) and np.all(rel <= upper + 1e-5)
                if ratio == 1.0:
                    assert linear.all()
                results[corrected] = dict(rel=float(np.abs(rel).max()), value=float(value_error.max()),
                                          w_min=float(w64.min()), abs_sum=float(np.abs(w64).sum(axis=1).max()),
                                          points=int(full.sum()), linear=int(linear.sum()))
            # positivity (hence sum |w~| = 1, bounded) of the corrected weights for r <= 1.2
            assert results[True]["w_min"] >= -1e-7 and results[True]["abs_sum"] <= 1.0 + 1e-6
            print("GRADED_KERNEL_GRADIENT ratio=%.2f axis=%d family=%s points=%d linear=%d raw_max_rel_grad_error=%.4e "
                  "corrected_max_rel_grad_error=%.2e corrected_max_value_error_over_h=%.2e corrected_min_w=%.4e "
                  "corrected_max_sum_abs_w=%.6f raw_min_w=%.4e"
                  % (ratio, a, "faces" if staggered else "centres", results[True]["points"], results[True]["linear"],
                     results[False]["rel"], results[True]["rel"], results[True]["value"], results[True]["w_min"],
                     results[True]["abs_sum"], results[False]["w_min"]))


def face_positions(axes, c):
    """Meshgrid of the MAC face positions of component c (faces along c, centres along the other axes)."""
    coordinates = [axes.centres_m[a] for a in range(3)]
    coordinates[c] = axes.faces_m[c][:-1]
    return np.meshgrid(*coordinates, indexing="ij")


def test_rigid_rotation_gives_zero_strain_across_band_transitions():
    """Root (2026-10-12) item 1: u = omega x (x - c) sampled on the MAC faces of a grid with 1.2 stretch transitions;
    the body's product weights must give G = W (the constant spin tensor), i.e. zero strain sym(G), and the exact
    velocity, for points across the transitions.  Raw G1 is printed for comparison."""
    omega = np.array([0.3, -0.5, 0.7])
    spin = np.array([[0.0, -omega[2], omega[1]], [omega[2], 0.0, -omega[0]], [-omega[1], omega[0], 0.0]])
    results = {}
    for corrected in (False, True):
        axes = graded_axes_r12(corrected)
        centre = np.array([0.5 * (axes.faces_m[a][0] + axes.faces_m[a][-1]) for a in range(3)])
        velocity = np.zeros(tuple(axes.n) + (3,))
        for c in range(3):
            X, Y, Z = face_positions(axes, c)
            r = np.stack([X - centre[0], Y - centre[1], Z - centre[2]], axis=-1)
            velocity[..., c] = np.cross(omega, r)[..., c]
        points = interior_points(axes, np.random.default_rng(17), 48)
        _, _, v, g, _ = spread_and_interpolate(axes, ALL_VALID, points, np.zeros((48, 3)), np.zeros((48, 3, 3)),
                                               velocity)
        strain = 0.5 * (g + np.transpose(g, (0, 2, 1)))
        exact = np.cross(omega, points.astype(np.float64) - centre)
        results[corrected] = (float(np.abs(g - spin).max()), float(np.abs(strain).max()),
                              float(np.abs(v - exact).max()))
    print("GRADED_KERNEL_ROTATION stretch=1.2 |omega|=%.3f raw: max|G-W|=%.3e max|sym G|=%.3e max|v-v_exact|=%.3e "
          "corrected: max|G-W|=%.3e max|sym G|=%.3e max|v-v_exact|=%.3e"
          % ((float(np.linalg.norm(omega)),) + results[False] + results[True]))
    norm = float(np.linalg.norm(omega))
    assert results[True][0] <= 2e-4 * norm and results[True][1] <= 2e-4 * norm
    assert results[True][2] <= 1e-6 * norm * 0.02                      # |r| <= ~0.02 m
    assert results[False][1] > 100 * results[True][1]                  # the raw kernel strains a rigid rotation


def _path_profile(axes, a, staggered, count):
    """Nodal weights (dense) and the interpolated value of u = sin(2 pi x / 8 mm) along `count` positions spanning
    faces 10..30 (the stretch-to-uniform transitions at faces 16 and 24)."""
    faces = axes.faces_m[a]
    path = np.linspace(faces[10], faces[30], count).astype(np.float32)
    i0, w, dw, _, local = axis_weights(axes, a, staggered, path)
    nodes = i0[:, None] + np.arange(4)[None, :]
    assert np.all((nodes >= 0) & (nodes < axes.n[a]))
    dense = np.zeros((count, axes.n[a]))
    np.put_along_axis(dense, nodes, w.astype(np.float64), axis=1)
    family = node_family(axes, a, staggered)
    u_nodes = np.sin(2.0 * np.pi * (family[nodes + 1] - faces[0]) / 8.0e-3)
    value = (w.astype(np.float64) * u_nodes).sum(axis=1)
    exact = np.sin(2.0 * np.pi * (path.astype(np.float64) - faces[0]) / 8.0e-3)
    return path.astype(np.float64), local.astype(np.float64), dense, value, exact, w


@pytest.mark.parametrize("staggered", [0, 1])
def test_grid_locking_smooth_translation_across_band_transitions(staggered):
    """Root (2026-10-12) item 4: a point moved through 200 positions spanning faces 10..30 of the 40-cell axis (the
    stretch-to-uniform transitions at faces 16 and 24).  Per step: the change of every nodal weight (= the spread force
    of a constant unit point force) and of the interpolated value of u = sin(2 pi x / 8 mm), per local cell travelled.
    No locking jump (continuity): refining the path 10x (2000 positions) shrinks the largest step change ~10x for a
    Lipschitz kernel, while a jump would not shrink.  Printed: GRADED_KERNEL_LOCKING rows (raw / corrected, r = 1.0 ..
    1.2): the largest weight change per local cell travelled, the largest value change / max|u'|, the variation of
    sum w^2 (3/8 for IB_4 on a uniform grid), the interpolation error and the minimum weight.  3-D weights are tensor
    products of these 1-D weights, so the 3-D nodal force change is bounded by the sum of the per-axis changes."""
    a = 0
    k_wave = 2.0 * np.pi / 8.0e-3
    rows = {}
    for ratio in GRADIENT_RATIOS:
        for corrected in (False, True):
            axes = ratio_axes(ratio, corrected)
            jumps = {}
            for count in (200, 2000):
                path, local, dense, value, exact, w = _path_profile(axes, a, staggered, count)
                step = np.diff(path)
                weight_change = np.abs(np.diff(dense, axis=0)).max(axis=1)
                jumps[count] = dict(weight=float(weight_change.max()), value=float(np.abs(np.diff(value)).max()),
                                    weight_per_cell=float((weight_change / (step / local[:-1])).max()),
                                    value_per_du=float((np.abs(np.diff(value)) / step).max() / k_wave),
                                    sum_sq=(w.astype(np.float64) ** 2).sum(axis=1),
                                    interpolation_error=float(np.abs(value - exact).max()), w_min=float(w.min()))
            coarse, fine = jumps[200], jumps[2000]
            rows[(ratio, corrected)] = dict(
                weight_per_cell=coarse["weight_per_cell"], value_per_du=coarse["value_per_du"],
                weight_refinement=coarse["weight"] / fine["weight"], value_refinement=coarse["value"] / fine["value"],
                sum_sq_variation=float((coarse["sum_sq"].max() - coarse["sum_sq"].min()) / 0.375),
                interpolation_error=coarse["interpolation_error"], w_min=min(coarse["w_min"], fine["w_min"]))
    for (ratio, corrected), row in sorted(rows.items()):
        print("GRADED_KERNEL_LOCKING family=%s ratio=%.2f %s max_weight_change_per_cell=%.4f "
              "max_value_change_per_max_du=%.4f refinement_shrink(weight,value)=(%.2f,%.2f) "
              "sum_w2_variation/(3/8)=%.3e interp_error=%.3e min_w=%.3e"
              % ("faces" if staggered else "centres", ratio, "corrected" if corrected else "raw      ",
                 row["weight_per_cell"], row["value_per_du"], row["weight_refinement"], row["value_refinement"],
                 row["sum_sq_variation"], row["interpolation_error"], row["w_min"]))
    uniform = rows[(1.0, True)]
    for (ratio, corrected), row in rows.items():
        assert row["weight_refinement"] > 5.0 and row["value_refinement"] > 5.0      # continuous: no locking jump
    for ratio in GRADIENT_RATIOS:
        corrected_row = rows[(ratio, True)]
        assert corrected_row["weight_per_cell"] <= 1.5 * uniform["weight_per_cell"]  # no locking amplification
        assert corrected_row["value_per_du"] <= 1.05                                 # smooth value (|U'| ~ |u'|)
        assert corrected_row["w_min"] >= -1e-7


def _mask_rig():
    axes = graded_axes()
    n = axes.n
    obstacle = np.zeros(n, np.int32)
    obstacle[6:9, 5:8, 7:10] = 1                       # a solid block
    air = np.zeros(n, np.int32)
    air[:, :, 12:] = 1                                 # an air layer on top
    air[2:4, 2:4, 11] = 1                              # and a few air cells below it
    fluid = SimpleNamespace(obstacle=ti.field(ti.i32, shape=n))
    fluid.obstacle.from_numpy(obstacle)
    air_field = ti.field(ti.i32, shape=n)
    air_field.from_numpy(air)
    return axes, fluid, obstacle, air, air_field


def reference_validity(obstacle, air):
    """Independent numpy rule: valid iff index >= 1 along c, neither cell an obstacle, not both cells air."""
    valid = np.zeros((3,) + obstacle.shape, np.int32)
    for c in range(3):
        here = [slice(None)] * 3
        here[c] = slice(1, None)
        there = [slice(None)] * 3
        there[c] = slice(0, -1)
        ok = (obstacle[tuple(here)] == 0) & (obstacle[tuple(there)] == 0) & ~((air[tuple(here)] != 0)
                                                                                & (air[tuple(there)] != 0))
        valid[c][tuple(here)] = ok.astype(np.int32)
    return valid


def test_face_mask_validity_cases():
    axes, fluid, obstacle, air, air_field = _mask_rig()
    mask = FaceMask(fluid, axes)
    dump = np.zeros((3,) + tuple(axes.n), np.int32)
    _face_valid_dump(mask, dump)
    np.testing.assert_array_equal(dump, reference_validity(obstacle, np.zeros_like(air)))
    mask.refresh(air_field)
    _face_valid_dump(mask, dump)
    np.testing.assert_array_equal(dump, reference_validity(obstacle, air))
    assert mask.counts()["valid_faces"] == [int(v) for v in dump.reshape(3, -1).sum(axis=1)]
    # obstacle neighbour: the x face between fluid cell (9, 6, 8) and the block cell (8, 6, 8)
    assert dump[0, 9, 6, 8] == 0 and dump[0, 10, 6, 8] == 1 and dump[1, 9, 6, 8] == 1
    # both air: z face between (5, 5, 12) and (5, 5, 13) invalid; the air-water face (5, 5, 11) | (5, 5, 12) valid
    assert dump[2, 5, 5, 13] == 0 and dump[2, 5, 5, 12] == 1 and dump[0, 5, 5, 13] == 0
    assert dump[2, 2, 2, 12] == 0 and dump[2, 2, 2, 11] == 1          # the air pocket joins the air layer
    # domain edge: index 0 along c is invalid, the other components of that cell are not
    assert dump[0, 0, 5, 5] == 0 and dump[1, 0, 5, 5] == 1 and dump[2, 0, 5, 5] == 1
    assert dump[2, 5, 5, 0] == 0 and dump[0, 5, 5, 0] == 1 and dump[1, 5, 0, 5] == 0


def test_one_sided_spread_next_to_obstacle_air_and_edge_conserves_force_and_power():
    axes, fluid, obstacle, air, air_field = _mask_rig()
    mask = FaceMask(fluid, axes)
    mask.refresh(air_field)
    centres = axes.centres_m
    points = np.array([
        [centres[0][9] + 0.1e-3, centres[1][6], centres[2][8]],          # touching the block (x side)
        [centres[0][7], centres[1][6], centres[2][10] + 0.1e-3],          # above the block
        [centres[0][5], centres[1][5], centres[2][11] + 0.2e-3],          # just below the air layer
        [centres[0][5], centres[1][6], axes.faces_m[2][0] + 0.3e-3],      # at the z-min domain edge
        [axes.faces_m[0][0] + 0.2e-3, centres[1][1], centres[2][1]],      # in the x-min, y-min, z-min corner
    ], dtype=np.float32)
    rng = np.random.default_rng(11)
    forces = rng.normal(size=(len(points), 3)) * 1e-3
    tau_v0 = rng.normal(size=(len(points), 3, 3)) * 1e-5
    velocity = rng.normal(size=tuple(axes.n) + (3,))
    density, volumes, v, g, denominator = spread_and_interpolate(axes, mask, points, forces, tau_v0, velocity)
    assert np.all(denominator > 0.0)
    assert np.all(denominator.min(axis=1) < 0.999)                   # every point exercises the normalisation
    dump = np.zeros((3,) + tuple(axes.n), np.int32)
    _face_valid_dump(mask, dump)
    assert np.all(density[dump == 0] == 0.0)                         # nothing lands on an invalid face
    check_conservation_and_power(density, volumes, velocity, forces, tau_v0, v, g)


def test_nonfinite_coordinate_gets_zero_weights():
    for axes in (graded_axes(), graded_axes_raw()):
        for a in range(3):
            for staggered in (0, 1):
                i0, w, dw, _, _ = axis_weights(axes, a, staggered,
                                               np.array([np.nan, np.inf, -np.inf, 1.0e9], np.float32))
                assert np.all(w == 0.0) and np.all(dw == 0.0)
                assert np.all(i0 + 3 < 0) or np.all(i0 >= axes.n[a])


def test_stretch_in_box_reports_the_adjacent_ratio_around_a_body():
    axes = graded_axes()
    faces_z = axes.faces_m[2]
    core = (faces_z[7] + 1e-5, faces_z[9] - 1e-5)               # cells 7..8 of the uniform core 5..10 (+2 margin)
    edge = (faces_z[1] + 1e-5, faces_z[2] - 1e-5)
    lo = [axes.faces_m[0][6], axes.faces_m[1][5], core[0]]
    hi = [axes.faces_m[0][8], axes.faces_m[1][6], core[1]]
    assert axes.stretch_in_box(lo, hi)[2] == pytest.approx(1.0, abs=1e-12)
    assert axes.stretch_in_box([lo[0], lo[1], edge[0]], [hi[0], hi[1], edge[1]])[2] == pytest.approx(1.1, rel=1e-9)
    assert axes.stretch_in_box(lo, hi, margin_cells=3)[2] == pytest.approx(1.1, rel=1e-9)


def test_from_fluid_and_hard_stop_codes():
    grid = CartesianGrid(bounds_min_m=(-0.012, 0.004, 0.98),
                         cell_widths_x_m=tuple(np.diff(graded_faces(0, 14, 1e-3, 1.1, 4))),
                         cell_widths_y_m=(1.4e-3,) * 12,
                         cell_widths_z_m=tuple(np.diff(graded_faces(0, 16, 6e-4, 1.1, 6))))
    axes = GradedAxes.from_fluid(SimpleNamespace(grid=grid))
    assert axes.n == grid.grid_nodes
    assert axes.uniform == (False, True, False)
    assert axes.moment_correction is True
    assert axes.min_spacing_m == pytest.approx(6e-4, rel=1e-9)
    np.testing.assert_allclose(axes.faces_m[2], grid.cell_faces_z_m, rtol=0, atol=1e-15)
    assert "one_sided_empty_support" in HARD_STOP_CODES
    stop = FastModeHardStop("one_sided_empty_support", dict(point=3, component=2, position_m=[0.0, 0.0, 1.0]))
    assert isinstance(stop, RuntimeError) and stop.code == "one_sided_empty_support" and stop.witness["point"] == 3
    with pytest.raises(ValueError):
        FastModeHardStop("not_a_code", {})
    with pytest.raises(ValueError):
        GradedAxes([0.0, 1.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    identity = axes.identity()
    assert identity["moment_correction"] is True and math.isfinite(identity["max_adjacent_stretch"][0])
