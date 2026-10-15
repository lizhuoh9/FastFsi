"""Material points of deformable solid bodies, for the immersed-boundary coupling (coupling.immersed_boundary).

MaterialPoints (3-D) and PlanarMaterialPoints (one body in the (y, z) plane of a quasi-2-D channel) hold the point
state: reference position X0, current position x, velocity v, deformation gradient F and Kirchhoff stress tau, the
reference volume (3-D) or area per unit span (planar), the tether flag (points held by a clamp) and the near-surface flag
(diagnostics); in 3-D also the displacement u = x - X0, the last interpolated velocity gradient, the body id, the point
tags, the external load and an extra point force consumed by the next spread.  They carry the material
(fastfsi.solid.materials.NeoHookeanMaterial), the checkpoint state and the solid's own diagnostics.  The points do not
move by themselves: the coupler moves them with the interpolated fluid velocity and updates F with the interpolated
velocity gradient, and it evaluates the stress.

3-D positions are integrated as the displacement u = x - X0 (f32): an absolute f32 coordinate near 1 m has an ulp of
1.2e-7 m, so slow motions would be lost in x += dt v; x = X0 + u is kept as the current-position field and positions()
sums X0 + u in float64.

Point tags (bits of `tag`, fastfsi.point_tags): TAG_CHAMBER_SURFACE marks the points of a surface that faces a pressure
chamber (boundary.pressure_chamber uses them as the surface); TAG_CONTACT_RIM marks points whose height above a
reference plane is reported (rim_min_gap_m).

Solid diagnostics: J range, momentum, non-finite x/u/v/F/J points, nonpositive J points, the largest displacement of
the tethered points, the volume error sum(V J) / sum(V) - 1 of the free (non-tethered) points, the fraction of free
points with |J - 1| > 0.5 % and
the share of those near the reference surface; in 3-D also the kinetic energy, the largest free speed and displacement,
the lowest TAG_CONTACT_RIM point above a reference plane, and per body id 0 .. MAX_BODIES - 1 the free volume and its
error, the vertical displacement range of the free points and the tethered-band drift and lift.

No `from __future__ import annotations` (Taichi kernel module).
"""
import math

import numpy as np
import taichi as ti

from fastfsi.errors import SimulationHardStop
from fastfsi.io.checkpoint import Checkpointable
from fastfsi.point_tags import TAG_CONTACT_RIM

REDUCE_SLOTS = 32
PLANAR_REDUCE_SLOTS = 16
MAX_BODIES = 4             # per-body diagnostics for body ids 0 .. MAX_BODIES - 1
BODY_SLOTS = 6             # free volume, free volume x J, free min / max uz, tethered max |u|, tethered max uz


# Validation is read-only: no carried fields or checkpoint scratch. A device reduction returns the first invalid
# point; only a failing point's witness is read on the host. Flags also feed the existing diagnostic counters.
_STATE_X = 1
_STATE_V = 2
_STATE_F = 4
_STATE_J_NONFINITE = 8
_STATE_J_NONPOSITIVE = 16
_STATE_U = 32
_STATE_NONFINITE = _STATE_X | _STATE_V | _STATE_F | _STATE_J_NONFINITE | _STATE_U


@ti.func
def point_state_flags(points: ti.template(), p, dim: ti.template(), has_u: ti.template(),
                      check_velocity: ti.template()):
    """(_STATE_* bits, J) of point p: non-finite x, u, v, F or J, or J <= 0 (also the coupling's per-step check)."""
    flags = 0
    for q in ti.static(range(dim)):
        if ti.math.isnan(points.x[p][q]) or ti.math.isinf(points.x[p][q]):
            flags |= _STATE_X
        if ti.static(check_velocity):
            if ti.math.isnan(points.v[p][q]) or ti.math.isinf(points.v[p][q]):
                flags |= _STATE_V
        if ti.static(has_u):
            if ti.math.isnan(points.u[p][q]) or ti.math.isinf(points.u[p][q]):
                flags |= _STATE_U
        for r in ti.static(range(dim)):
            if ti.math.isnan(points.F[p][q, r]) or ti.math.isinf(points.F[p][q, r]):
                flags |= _STATE_F
    J = points.F[p].determinant()
    if ti.math.isnan(J) or ti.math.isinf(J):
        flags |= _STATE_J_NONFINITE
    elif J <= 0.0:
        flags |= _STATE_J_NONPOSITIVE
    return flags, J


@ti.kernel
def _first_invalid_point_kernel(points: ti.template(), dim: ti.template(), has_u: ti.template(),
                                check_velocity: ti.template()) -> ti.i32:
    first = points.n
    for p in range(points.n):
        flags, J = point_state_flags(points, p, dim, has_u, check_velocity)
        if flags != 0:
            ti.atomic_min(first, p)
    return first


@ti.kernel
def _invalid_point_flags_kernel(points: ti.template(), p: ti.i32, dim: ti.template(), has_u: ti.template(),
                                check_velocity: ti.template()) -> ti.types.vector(2, ti.f64):
    flags, J = point_state_flags(points, p, dim, has_u, check_velocity)
    return ti.Vector([ti.cast(flags, ti.f64), ti.cast(J, ti.f64)])


def _witness_values(value):
    """Finite numbers or explicit strings, including nested lists, for strict JSON failure reports."""
    if isinstance(value, list):
        return [_witness_values(item) for item in value]
    value = float(value)
    return value if math.isfinite(value) else repr(value)


def _validate_point_state(points, dim, has_u, check_velocity):
    p = int(_first_invalid_point_kernel(points, dim, has_u, check_velocity))
    if p == points.n:
        return
    flags_j = _invalid_point_flags_kernel(points, p, dim, has_u, check_velocity)
    flags = int(flags_j[0])
    nonfinite = bool(flags & _STATE_NONFINITE)
    witness = dict(point=p, invalid_fields=[name for bit, name in (
        (_STATE_X, "x"), (_STATE_U, "u"), (_STATE_V, "v"), (_STATE_F, "F"),
        (_STATE_J_NONFINITE | _STATE_J_NONPOSITIVE, "J")) if flags & bit],
        reason="nonfinite" if nonfinite else "nonpositive_J",
        position_m=_witness_values(points.x[p].to_numpy().tolist()),
        F=_witness_values(points.F[p].to_numpy().tolist()), J=_witness_values(flags_j[1]))
    if check_velocity:
        witness["velocity_mps"] = _witness_values(points.v[p].to_numpy().tolist())
    if has_u:
        witness["displacement_m"] = _witness_values(points.u[p].to_numpy().tolist())
    raise SimulationHardStop("nonfinite_point_state" if nonfinite else "invalid_deformation_gradient", witness)


@ti.data_oriented
class MaterialPoints(Checkpointable):
    """Material points of the deformable bodies in 3-D (see the module docstring)."""

    # checkpoint (state census): displacement, deformation gradient and the contact force accumulator (zero at a
    # checkpoint: _spread_kernel consumes it); x = X0 + u is recomputed on load; u_n, F_n, v, grad_v and tau are
    # rewritten before use in every step
    CHECKPOINT_FIELDS = ("u", "F", "f_extra")

    def __init__(self, *, positions_m, volumes_m3, material, tether_mask, body_id=None, tag=None,
                 external_force_n=None, near_surface_mask=None, probe_indices=()):
        positions = np.asarray(positions_m, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
            raise ValueError("MaterialPoints needs an (n, 3) array of positions")
        n = len(positions)
        volumes = np.asarray(volumes_m3, dtype=np.float32)
        tether = np.asarray(tether_mask, dtype=np.int32)
        if volumes.shape != (n,) or tether.shape != (n,):
            raise ValueError("volumes and tether mask must have one entry per point")
        if not np.all(volumes > 0.0):
            raise ValueError("point volumes must be positive")
        self.material = material
        self.n = n
        self.rho_s = float(material.density_kgm3)
        self.mu = float(material.shear_modulus_pa)
        self.kappa = float(material.volumetric_modulus_pa)
        self.x = ti.Vector.field(3, ti.f32, shape=n)            # current position X0 + u (derived)
        self.X0 = ti.Vector.field(3, ti.f32, shape=n)
        self.u = ti.Vector.field(3, ti.f32, shape=n)            # displacement x - X0 (integrated: f32 precision)
        self.u_n = ti.Vector.field(3, ti.f32, shape=n)          # MIDPOINT_RULE: u^n and F^n
        self.v = ti.Vector.field(3, ti.f32, shape=n)
        self.F = ti.Matrix.field(3, 3, ti.f32, shape=n)
        self.F_n = ti.Matrix.field(3, 3, ti.f32, shape=n)
        self.tau = ti.Matrix.field(3, 3, ti.f32, shape=n)
        self.grad_v = ti.Matrix.field(3, 3, ti.f32, shape=n)    # last interpolated velocity gradient (diagnostics)
        self.vol = ti.field(ti.f32, shape=n)
        self.tether = ti.field(ti.i32, shape=n)
        self.body_id = ti.field(ti.i32, shape=n)
        self.tag = ti.field(ti.i32, shape=n)
        self.near_surface = ti.field(ti.i32, shape=n)
        self.f_ext = ti.Vector.field(3, ti.f32, shape=n)        # external load (N) at load scale 1
        self.f_extra = ti.Vector.field(3, ti.f32, shape=n)      # extra point force (N), consumed and zeroed per spread
        self.reduce = ti.field(ti.f64, shape=REDUCE_SLOTS)
        self.reduce_body = ti.field(ti.f64, shape=(MAX_BODIES, BODY_SLOTS))   # per-body diagnostics
        probes = np.asarray(probe_indices, dtype=np.int32).reshape(-1)
        if np.any((probes < 0) | (probes >= n)):
            raise ValueError("probe index out of range")
        self.n_probe = max(1, len(probes))
        self.probe_index = ti.field(ti.i32, shape=self.n_probe)
        self.probe_x = ti.Vector.field(3, ti.f32, shape=self.n_probe)
        self.probe_index.from_numpy(probes if len(probes) else np.zeros(1, np.int32))
        self.x.from_numpy(positions)
        self.X0.from_numpy(positions)
        self.u.fill(0.0)
        self.u_n.fill(0.0)
        self.v.fill(0.0)
        self.grad_v.fill(0.0)
        eye = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        self.F.from_numpy(eye)
        self.F_n.from_numpy(eye)
        self.vol.from_numpy(volumes)
        self.tether.from_numpy(tether)
        self.body_id.from_numpy(np.zeros(n, np.int32) if body_id is None else np.asarray(body_id, np.int32))
        self.body_ids = tuple(sorted(int(b) for b in np.unique(self.body_id.to_numpy())))
        self.tag.from_numpy(np.zeros(n, np.int32) if tag is None else np.asarray(tag, np.int32))
        self.near_surface.from_numpy(np.zeros(n, np.int32) if near_surface_mask is None
                                     else np.asarray(near_surface_mask, dtype=np.int32))
        ext = np.zeros((n, 3), np.float32) if external_force_n is None else np.asarray(external_force_n, np.float32)
        if ext.shape != (n, 3):
            raise ValueError("external force must be (n, 3)")
        self.f_ext.from_numpy(ext)
        self.f_extra.fill(0.0)

    # ------------------------------------------------------------------------------------------------ kernels
    @ti.kernel
    def _refresh_x_kernel(self):
        for p in range(self.n):
            self.x[p] = self.X0[p] + self.u[p]

    @ti.kernel
    def _save_state_kernel(self):
        for p in range(self.n):
            self.u_n[p] = self.u[p]
            self.F_n[p] = self.F[p]

    @ti.kernel
    def _probe_kernel(self):
        for q in range(self.n_probe):
            self.probe_x[q] = self.x[self.probe_index[q]]

    @ti.kernel
    def _diagnostics_kernel(self, reference_plane_z: ti.f32):
        for s in range(REDUCE_SLOTS):
            self.reduce[s] = 0.0
        self.reduce[0] = 1.0e30
        self.reduce[1] = -1.0e30
        self.reduce[19] = 1.0e30
        for b, s in ti.ndrange(MAX_BODIES, BODY_SLOTS):
            self.reduce_body[b, s] = 0.0
        for b in range(MAX_BODIES):
            self.reduce_body[b, 2] = 1.0e30                # free min uz
            self.reduce_body[b, 3] = -1.0e30               # free max uz
            self.reduce_body[b, 5] = -1.0e30               # tethered max uz
        for p in range(self.n):
            flags, point_J = point_state_flags(self, p, 3, True, True)
            J = ti.cast(point_J, ti.f64)
            ti.atomic_min(self.reduce[0], J)
            ti.atomic_max(self.reduce[1], J)
            mass = ti.cast(self.rho_s * self.vol[p], ti.f64)
            for q in ti.static(range(3)):
                ti.atomic_add(self.reduce[2 + q], mass * ti.cast(self.v[p][q], ti.f64))
            bad = 1 if flags & _STATE_NONFINITE else 0
            ti.atomic_add(self.reduce[5], ti.cast(bad, ti.f64))
            if flags & _STATE_J_NONPOSITIVE:
                ti.atomic_add(self.reduce[18], 1.0)
            d = ti.cast(self.u[p].norm(), ti.f64)
            body = self.body_id[p]
            known = 0 <= body < MAX_BODIES
            if self.tether[p] != 0:
                ti.atomic_max(self.reduce[6], d)
                if known:                                  # tethered (clamped) band of the body: drift and lift
                    ti.atomic_max(self.reduce_body[body, 4], d)
                    ti.atomic_max(self.reduce_body[body, 5], ti.cast(self.u[p][2], ti.f64))
            else:
                v0 = ti.cast(self.vol[p], ti.f64)
                ti.atomic_add(self.reduce[10], v0)
                ti.atomic_add(self.reduce[11], v0 * J)
                if known:                                  # per-body volume
                    ti.atomic_add(self.reduce_body[body, 0], v0)
                    ti.atomic_add(self.reduce_body[body, 1], v0 * J)
                ti.atomic_add(self.reduce[12], 1.0)
                ti.atomic_max(self.reduce[13], d)
                if ti.abs(J - 1.0) > 5.0e-3:
                    ti.atomic_add(self.reduce[14], 1.0)
                    if self.near_surface[p] != 0:
                        ti.atomic_add(self.reduce[15], 1.0)
                ti.atomic_add(self.reduce[16], 0.5 * mass * ti.cast(self.v[p].norm_sqr(), ti.f64))
                ti.atomic_max(self.reduce[17], ti.cast(self.v[p].norm(), ti.f64))
                if known:                                  # vertical displacement range of the body's free points
                    ti.atomic_min(self.reduce_body[body, 2], ti.cast(self.u[p][2], ti.f64))
                    ti.atomic_max(self.reduce_body[body, 3], ti.cast(self.u[p][2], ti.f64))
            if (self.tag[p] & TAG_CONTACT_RIM) != 0:
                ti.atomic_min(self.reduce[19], ti.cast(self.X0[p][2] + self.u[p][2] - reference_plane_z, ti.f64))

    # ------------------------------------------------------------------------------------------------ host API
    def refresh_positions(self):
        """x = X0 + u (after the state was loaded)."""
        self._refresh_x_kernel()

    def save_state(self):
        """u^n, F^n <- u, F (the first half of the midpoint rule)."""
        self._save_state_kernel()

    def probes(self):
        self._probe_kernel()
        return self.probe_x.to_numpy()

    def positions(self):
        """Current positions X0 + u, summed in float64 on the host."""
        return self.X0.to_numpy().astype(np.float64) + self.u.to_numpy().astype(np.float64)

    def velocities(self):
        return self.v.to_numpy()

    def validate_state(self, *, check_velocity=True):
        """Reject invalid x/u/F/J and, by default, v. Only the initial/restored state may skip v: it is scratch,
        rewritten by the first gather before use. Skipping the check never clears or changes the scratch field."""
        _validate_point_state(self, 3, True, bool(check_velocity))

    def diagnostics(self, reference_plane_z_m=0.0):
        """The solid diagnostics (module docstring); rim_min_gap_m is the lowest TAG_CONTACT_RIM point above the plane
        z = reference_plane_z_m."""
        self._diagnostics_kernel(float(reference_plane_z_m))
        r = self.reduce.to_numpy()
        free = max(float(r[12]), 1.0)
        deviated = float(r[14])
        return dict(J_min=float(r[0]), J_max=float(r[1]), momentum=[float(r[2]), float(r[3]), float(r[4])],
                    nonfinite_points=int(r[5]), nonpositive_J_points=int(r[18]),
                    tether_max_displacement_m=float(r[6]),
                    volume_error=(float(r[11]) / float(r[10]) - 1.0) if r[10] > 0 else 0.0,
                    free_max_displacement_m=float(r[13]), J_deviated_fraction=deviated / free,
                    J_deviated_near_surface_share=(float(r[15]) / deviated) if deviated > 0 else 0.0,
                    kinetic_energy_j=float(r[16]), free_max_speed_mps=float(r[17]),
                    rim_min_gap_m=float(r[19]) if r[19] < 1.0e29 else float('nan'),
                    bodies=self._body_diagnostics())

    def _body_diagnostics(self):
        """Per body id present: free volume and its relative error, vertical displacement range of the free points,
        tethered-band drift and lift (read after _diagnostics_kernel)."""
        rb = self.reduce_body.to_numpy()
        out = {}
        for b in self.body_ids:
            if not 0 <= b < MAX_BODIES:
                continue
            s = rb[b]
            out[int(b)] = dict(
                free_volume_m3=float(s[0]),
                volume_error=(float(s[1]) / float(s[0]) - 1.0) if s[0] > 0 else float('nan'),
                free_min_uz_m=float(s[2]) if s[2] < 1.0e29 else float('nan'),
                free_max_uz_m=float(s[3]) if s[3] > -1.0e29 else float('nan'),
                tether_max_displacement_m=float(s[4]),
                tether_max_uz_m=float(s[5]) if s[5] > -1.0e29 else float('nan'))
        return out

    def elastic_dt_limit(self, cfl, min_spacing_m):
        """CFL on the shear wave over the smallest cell edge min_spacing_m."""
        c_s = math.sqrt(self.mu / self.rho_s)
        return float(cfl) * float(min_spacing_m) / c_s

    def _after_load_state(self):
        """x = X0 + u, the expression of the step kernels."""
        self._refresh_x_kernel()


@ti.data_oriented
class PlanarMaterialPoints(Checkpointable):
    """Material points of one elastic body in the (y, z) plane of a quasi-2-D channel (fields uniform along x);
    areas are per unit span (m^2)."""

    # checkpoint: the position and the deformation gradient; v, tau, x_n and F_n are rewritten before use in every step
    CHECKPOINT_FIELDS = ("x", "F")

    def __init__(self, *, positions_yz_m, areas_m2, material, tether_mask, near_surface_mask=None):
        positions = np.asarray(positions_yz_m, dtype=np.float32)
        n = len(positions)
        if n == 0:
            raise ValueError("PlanarMaterialPoints needs material points")
        self.material = material
        self.n = n
        self.rho_s = float(material.density_kgm3)
        self.mu = float(material.shear_modulus_pa)
        self.kappa = float(material.volumetric_modulus_pa)
        self.x = ti.Vector.field(2, ti.f32, shape=n)
        self.x_n = ti.Vector.field(2, ti.f32, shape=n)          # MIDPOINT_RULE: X^n and F^n
        self.F_n = ti.Matrix.field(2, 2, ti.f32, shape=n)
        self.X0 = ti.Vector.field(2, ti.f32, shape=n)
        self.v = ti.Vector.field(2, ti.f32, shape=n)
        self.F = ti.Matrix.field(2, 2, ti.f32, shape=n)
        self.tau = ti.Matrix.field(2, 2, ti.f32, shape=n)
        self.area = ti.field(ti.f32, shape=n)
        self.tether = ti.field(ti.i32, shape=n)
        # volume-drift statistics: reference points within 2 cells of the body surface
        self.near_surface = ti.field(ti.i32, shape=n)
        self.reduce = ti.field(ti.f64, shape=PLANAR_REDUCE_SLOTS)
        self.x.from_numpy(positions)
        self.X0.from_numpy(positions)
        self.v.fill(0.0)
        self.F.from_numpy(np.tile(np.eye(2, dtype=np.float32), (n, 1, 1)))
        self.area.from_numpy(np.asarray(areas_m2, dtype=np.float32))
        self.tether.from_numpy(np.asarray(tether_mask, dtype=np.int32))
        self.near_surface.from_numpy(np.zeros(n, np.int32) if near_surface_mask is None
                                     else np.asarray(near_surface_mask, dtype=np.int32))

    @ti.kernel
    def _save_state_kernel(self):
        for p in range(self.n):
            self.x_n[p] = self.x[p]
            self.F_n[p] = self.F[p]

    @ti.kernel
    def _diagnostics_kernel(self):
        for s in range(PLANAR_REDUCE_SLOTS):
            self.reduce[s] = 0.0
        self.reduce[0] = 1.0e30
        self.reduce[1] = -1.0e30
        for p in range(self.n):
            flags, point_J = point_state_flags(self, p, 2, False, True)
            J = ti.cast(point_J, ti.f64)
            ti.atomic_min(self.reduce[0], J)
            ti.atomic_max(self.reduce[1], J)
            mass = ti.cast(self.rho_s * self.area[p], ti.f64)
            ti.atomic_add(self.reduce[2], mass * self.v[p][0])
            ti.atomic_add(self.reduce[3], mass * self.v[p][1])
            if self.tether[p] != 0:
                d = (self.x[p] - self.X0[p]).norm()
                ti.atomic_max(self.reduce[4], ti.cast(d, ti.f64))
            bad = 1 if flags & _STATE_NONFINITE else 0
            ti.atomic_add(self.reduce[7], ti.cast(bad, ti.f64))
            if flags & _STATE_J_NONPOSITIVE:
                ti.atomic_add(self.reduce[13], 1.0)
            if self.tether[p] == 0:
                # volume drift of the free (non-tethered) body
                a0 = ti.cast(self.area[p], ti.f64)
                ti.atomic_add(self.reduce[8], a0)
                ti.atomic_add(self.reduce[9], a0 * J)
                ti.atomic_add(self.reduce[10], 1.0)
                if ti.abs(J - 1.0) > 5.0e-3:
                    ti.atomic_add(self.reduce[11], 1.0)
                    if self.near_surface[p] != 0:
                        ti.atomic_add(self.reduce[12], 1.0)

    def save_state(self):
        """X^n, F^n <- x, F (the first half of the midpoint rule)."""
        self._save_state_kernel()

    def positions(self):
        return self.x.to_numpy()

    def validate_state(self):
        """Raise SimulationHardStop with the first non-finite x/v/F/J or nonpositive J."""
        _validate_point_state(self, 2, False, True)

    def diagnostics(self):
        """The solid diagnostics (module docstring) of the planar body."""
        self._diagnostics_kernel()
        r = self.reduce.to_numpy()
        free = max(float(r[10]), 1.0)
        deviated = float(r[11])
        return dict(J_min=float(r[0]), J_max=float(r[1]), momentum_y=float(r[2]), momentum_z=float(r[3]),
                    tether_max_displacement_m=float(r[4]), nonfinite_points=int(r[7]),
                    nonpositive_J_points=int(r[13]),
                    volume_error=(float(r[9]) / float(r[8]) - 1.0) if r[8] > 0 else 0.0,
                    J_deviated_fraction=deviated / free,
                    J_deviated_near_surface_share=(float(r[12]) / deviated) if deviated > 0 else 0.0)

    def elastic_dt_limit(self, cfl, min_spacing_m):
        """CFL on the shear wave over the smallest cell edge min_spacing_m."""
        c_s = math.sqrt(self.mu / self.rho_s)
        return float(cfl) * float(min_spacing_m) / c_s
