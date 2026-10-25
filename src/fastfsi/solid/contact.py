"""Penalty contact between the deformable material points and rigid solids.

MODEL.  No seal assumption, no obstacle change, no switching.  The rigid solids are given by their triangles; a signed
distance field phi (m; < 0 inside a rigid solid) is built once on the host at the fluid cell centres
(geometry.signed_distance) and interpolated trilinearly, with its exact gradient, at the points.  Every non-tethered
point p (tether == 0) whose distance is below its contact distance gets the repulsive penalty force

    F_p = k_c V0_p (d_c,p(n_p) - phi_p) n_p,   n_p = grad phi / |grad phi|,   only while phi_p < d_c,p(n_p),

ADDED to the points' f_extra (the coupler spreads it with the other point forces and zeroes f_extra).  The contact is
normal only, FRICTIONLESS (no tangential force) and NON-ADHESIVE (no force for phi >= d_c, never attractive), with NO
damping (it adds no dissipation).  Whether and how hard a body touches a rigid surface follows from the computed flow;
nothing is prescribed.

PARAMETERS (nothing is tuned):
  * k_c (N/m^4: force per unit point volume and unit penetration) = rho_s (CONTACT_OMEGA_DT / dt)^2 with
    omega dt = 0.25 by default: the contact period 8 pi dt (~25 steps) is resolved by the explicit step (omega dt << 2).
  * d_c,p(n) = a_p . |n|: a point stands for a material box centred on it, of volume V0_p and with the aspect ratio of
    the fluid cell holding its reference position (half extents a_p = 0.5 dx_cell (V0 / V_cell)^(1/3) per axis; on a
    split-2 point lattice a_p is a quarter of the local cell widths).  d_c(n) is the box's support distance along n:
    the repulsion starts exactly when the material box touches the rigid surface.  A fixed d_c_m may be given instead.
  * the signed distance is exact within BAND_CELLS = 2 local cells of the rigid surfaces (the trilinear stencil of any
    point with phi < d_c, d_c <= a quarter cell, lies within 2 cells of the surface) and within reach_margin_m of a point
    at its reference position; elsewhere it is an upper bound used only where no force can act.
  * SIGN = the run's static obstacle mask (fluid.obstacle at construction) by default, so that the contact geometry and
    the fluid's rigid geometry agree cell by cell; MAGNITUDE = distance to the triangulated surface.
    sign_from_obstacle=False takes the sign from the closed-solid classification instead.

OPTIONS (each off by default; off = the base law above):
  * LID (lid_z_m): an exact analytic rigid plane z = lid_z (the top of the fluid domain cannot be represented by the
    grid signed distance): phi_lid = lid_z - z, n = -e_z, the same law with d_c = a_z, on every non-tethered point.
  * WALL ON OBSTACLE FACES (selected_wall_on_obstacle_faces): for the points of body selected_body_id the wall
    distance is geometry.signed_distance.wall_distance_on_obstacle_faces, whose trilinear zero level lies on the
    obstacle faces the fluid uses (a coarse staircase cell protruding inside the triangulated wall otherwise puts the
    zero level inside the obstacle cell, and a point can enter it).
  * STIFFNESS OF ONE BODY (selected_omega_dt): the contact of the points of body selected_body_id (walls and lid) uses
    k = rho_s (selected_omega_dt / dt)^2 (explicit stability needs omega dt < 2).
  * REST CAP (rest_cap): a point may start inside its own contact distance of a wall (a coarse wall face next to a
    clamped edge), and the base law pushes it from t = 0 with no load.  With the cap, every non-tethered point's contact
    distance against the walls is d_c,p = min(d_c,p(n), phi_p(X0) - REST_CAP_MARGIN_M), phi_p(X0) its wall distance at
    its reference position by the step kernel's own functions (phi_wall for the selected body when
    selected_wall_on_obstacle_faces, phi otherwise), evaluated once at construction.  No point is in contact at rest;
    a point moved from rest toward the wall is pushed with the same k once it is REST_CAP_MARGIN_M closer than at
    rest; a point farther than d_c at rest is unchanged.  The margin (1 nm) keeps a rounding difference between the
    construction kernel and the step kernel from putting a point in contact at rest.  A point inside a wall at rest
    (phi(X0) <= 0) is pushed only when it goes deeper: its rest position is taken as stress-free.  The lid is not capped
    (no material box crosses it at rest: checked at construction).  setup_report()['rest_cap'] gives the capped points
    and those with phi(X0) <= 0.
  Diagnostics when on: lid force / points / gap / penetration past the plane; the selected body's contact force /
  points, min wall distance, penetration past the wall, and min one-sided interpolation support D of its points in
  contact.

PER-STEP PATH.  apply_point_forces(dt) is ONE kernel launch with no host access: distance lookup, penalty force into
f_extra and on-device accumulation of the interval statistics (contact impulse, max penetration).  It evaluates the
state it is given (call it right before the coupler spreads the forces).  report() runs one measure kernel, reads one
small statistics field and resets the interval accumulators.

References (algorithms; no source copied): IBAMR (BSD-3-Clause) IBStandardForceGen target points and the IBFE penalty
tethers (a penalty spring on the IB points whose force is spread with the IB kernel; here one-sided along the distance
normal, without the damper); taichi_mpm (MIT) MLS-CPIC normal-only penetration penalty against rigid level sets.

No `from __future__ import annotations` (Taichi kernel module).
"""
import math
import time

import numpy as np
import taichi as ti

from fastfsi.errors import SimulationHardStop
from fastfsi.io.checkpoint import Checkpointable
from fastfsi.geometry.signed_distance import (BAND_CELLS, REACH_MARGIN_M, _axis_faces, build_signed_distance,
                                              load_or_build_signed_distance, wall_distance_on_obstacle_faces)

CONTACT_OMEGA_DT = 0.25        # tether convention k = rho_s (0.25 / dt)^2
REST_CAP_MARGIN_M = 1.0e-9     # rest_cap: the cap sits 1 nm under the rest wall distance (rounding between kernels)
HARD_STOP_CODES = {
    "contact_setup_invalid",      # SDF / geometry inconsistent (construction)
    "contact_nonfinite_state",    # nonfinite phi at a point
}

# statistics slots (f64): per-step scratch 3-6, interval accumulators 10-22 (reset by report), measurement 30-40; the
# other slots below STATS_SLOTS are unused, so that the option slots keep their positions
S_FORCE, S_POINTS, S_UNDEF, S_NONFINITE = 3, 4, 5, 6
A_STEPS, A_TIME = 10, 11
A_IMPULSE, A_MAX_PEN, A_NONFINITE, A_UNDEF = 17, 18, 20, 22
M_FORCE, M_POINTS, M_AREA, M_ENERGY, M_MIN_PHI, M_MAX_PEN, M_FX, M_FY, M_FZ = range(30, 39)
M_UNDEF, M_NONFINITE = 39, 40
EV_BASE = 64                   # report() checks the statistics below this index for finiteness
STATS_SLOTS = EV_BASE + 16 * 6
# option slots (allocated only when an option is on): step scratch, interval accumulators, measurement
V_S_LID_FORCE, V_S_LID_POINTS, V_S_SEL_FORCE = STATS_SLOTS, STATS_SLOTS + 1, STATS_SLOTS + 2
V_A_LID_IMPULSE, V_A_LID_MAX_PEN, V_A_SEL_IMPULSE, V_A_SEL_MAX_PEN = range(STATS_SLOTS + 3, STATS_SLOTS + 7)
V_M_LID_FORCE, V_M_LID_POINTS, V_M_LID_ENERGY, V_M_LID_GAP_MIN, V_M_LID_MAX_PEN = range(STATS_SLOTS + 7, STATS_SLOTS + 12)
V_M_SEL_FORCE, V_M_SEL_POINTS, V_M_SEL_MIN_PHI, V_M_SEL_MAX_PEN, V_M_SEL_MIN_D = range(STATS_SLOTS + 12, STATS_SLOTS + 17)
OPTION_SLOTS = 17



# ---------------------------------------------------------------------------------------------------- contact
@ti.data_oriented
class PenaltyContact(Checkpointable):
    """Repulsive frictionless penalty contact of the non-tethered material points against rigid triangulated solids,
    an optional analytic lid plane, and per-body options (module docstring)."""

    # checkpoint (state census): the statistics (all-time maxima, interval sums) and the last report time
    CHECKPOINT_FIELDS = ("stats",)
    CHECKPOINT_HOST = ("_last_report_t",)

    def __init__(self, fluid, axes, coupler, rigid_triangles_m, *, d_c_m=None, k_c=None, solid_density_kgm3=None,
                 band_cells=BAND_CELLS, sample_spacing_m=None, reach_margin_m=REACH_MARGIN_M, sign_from_obstacle=True,
                 sdf_cache_dir=None, lid_z_m=None, selected_wall_on_obstacle_faces=False, selected_body_id=2,
                 selected_omega_dt=None, rest_cap=False):
        """fluid: the fluid solver (grid, obstacle); axes: the grid axes (.n checked); coupler: the
        ImmersedBoundaryCoupler (its points carry x, X0, vol, tether, body_id, f_extra; its weights and mask give the
        one-sided D of the contact diagnostics); rigid_triangles_m: dict region -> (n, 3, 3) triangles in
        metres, or one array.  k_c None -> rho_s (CONTACT_OMEGA_DT / dt)^2 at each apply_point_forces(dt).  d_c_m None
        -> the support distance of each point's material box along the contact normal.  reach_margin_m None -> exact
        signed distance in the whole band.  lid_z_m: analytic lid plane; selected_wall_on_obstacle_faces / selected_omega_dt: the
        wall distance on the obstacle faces and the contact stiffness of the points of body selected_body_id.
        rest_cap: the wall contact distance capped at each point's rest wall distance (module docstring)."""
        self.fluid, self.axes, self.coupler, self.points = fluid, axes, coupler, coupler.points
        grid = fluid.grid
        self.faces = [_axis_faces(grid.bounds_min_m[a], w) for a, w in
                      enumerate((grid.cell_widths_x_m, grid.cell_widths_y_m, grid.cell_widths_z_m))]
        self.nodes = tuple(len(f) - 1 for f in self.faces)
        if tuple(int(v) for v in axes.n) != self.nodes:
            raise ValueError("axes.n %s != fluid grid %s" % (tuple(axes.n), self.nodes))
        if k_c is not None and not (math.isfinite(float(k_c)) and float(k_c) > 0.0):
            raise ValueError("k_c must be finite and positive")
        self.k_c = None if k_c is None else float(k_c)
        rho_s = solid_density_kgm3 if solid_density_kgm3 is not None else getattr(self.points, "rho_s", None)
        if self.k_c is None and (rho_s is None or not float(rho_s) > 0.0):
            raise ValueError("k_c or a solid density (solid_density_kgm3 / points.rho_s) is needed")
        self.rho_s = 0.0 if rho_s is None else float(rho_s)
        # options (compile-time switches: off = the base kernels)
        self.lid_on = lid_z_m is not None
        self.lid_z_m = float(lid_z_m) if self.lid_on else 0.0
        self.lid_z = self.lid_z_m                                  # kernel compile-time constant
        if self.lid_on and not math.isfinite(self.lid_z_m):
            raise ValueError("lid_z_m must be finite")
        self.selected_wall = bool(selected_wall_on_obstacle_faces)
        self.selected_body_id = int(selected_body_id)
        if selected_omega_dt is not None and not (math.isfinite(float(selected_omega_dt))
                                                  and 0.0 < float(selected_omega_dt) < 2.0):
            raise ValueError("selected_omega_dt must lie in (0, 2) (explicit penalty)")
        self.selected_omega_dt = None if selected_omega_dt is None else float(selected_omega_dt)
        self.selected_stiff = self.selected_omega_dt is not None
        self.options_on = self.lid_on or self.selected_wall or self.selected_stiff
        reach = None
        if reach_margin_m is not None:
            reach = np.asarray(self.points.X0.to_numpy(), dtype=np.float64)
        sign_mask = (np.asarray(fluid.obstacle.to_numpy()) != 0) if sign_from_obstacle else None
        sdf_kwargs = dict(band_cells=band_cells, sample_spacing_m=sample_spacing_m, reach_points_m=reach,
                          reach_margin_m=reach_margin_m, sign_mask=sign_mask)
        try:
            if sdf_cache_dir is None:
                phi, self.sdf_report = build_signed_distance(rigid_triangles_m, grid, **sdf_kwargs)
            else:
                phi, self.sdf_report = load_or_build_signed_distance(sdf_cache_dir, rigid_triangles_m, grid, **sdf_kwargs)
        except ValueError as error:
            raise SimulationHardStop("contact_setup_invalid", dict(reason=str(error))) from None
        self._setup_points(phi, d_c_m)
        self._setup_options(phi, grid, sign_mask)
        self._setup_rest_cap(rest_cap)
        self.stats = ti.field(ti.f64, shape=STATS_SLOTS + (OPTION_SLOTS if self.options_on else 0))
        self._init_stats_kernel()
        self._last_report_t = None

    # ------------------------------------------------------------------------------------------------ setup
    def _setup_points(self, phi, d_c_m):
        points = self.points
        self.n = int(points.n)
        nx, ny, nz = self.nodes
        self.phi = ti.field(ti.f32, shape=self.nodes)
        self.phi.from_numpy(phi.astype(np.float32))
        self.centre_x = ti.field(ti.f32, shape=nx)
        self.centre_y = ti.field(ti.f32, shape=ny)
        self.centre_z = ti.field(ti.f32, shape=nz)
        for field, f in zip((self.centre_x, self.centre_y, self.centre_z), self.faces):
            field.from_numpy((0.5 * (f[:-1] + f[1:])).astype(np.float32))
        vol = np.asarray(points.vol.to_numpy(), dtype=np.float64)
        if vol.shape != (self.n,) or not (vol > 0.0).all():
            raise SimulationHardStop("contact_setup_invalid", dict(reason="point volumes must be positive"))
        # material box of each point: volume V0, aspect ratio of the fluid cell holding its reference position
        x0 = np.asarray(points.X0.to_numpy(), dtype=np.float64)
        widths = []
        for a, f in enumerate(self.faces):
            i = np.clip(np.searchsorted(f, x0[:, a], side="right") - 1, 0, len(f) - 2)
            widths.append(np.diff(f)[i])
        widths = np.column_stack(widths)
        half = 0.5 * widths * np.cbrt(vol / np.prod(widths, axis=1))[:, None]
        self.half_extent = ti.Vector.field(3, ti.f32, shape=self.n)
        self.half_extent.from_numpy(half.astype(np.float32))
        if d_c_m is None:
            self.fixed_dc = 0.0
            self.d_c_rule = ("support distance of the point's material box along the normal, d_c = a . |n| (box of "
                             "volume V0 with the aspect ratio of the fluid cell holding X0)")
            self.d_c_range_m = (float(half.min()), float(half.sum(axis=1).max()))
        else:
            if not (math.isfinite(float(d_c_m)) and float(d_c_m) > 0.0):
                raise ValueError("d_c_m must be finite and positive")
            self.fixed_dc = float(d_c_m)
            self.d_c_rule = "fixed %.6g m" % self.fixed_dc
            self.d_c_range_m = (self.fixed_dc, self.fixed_dc)
        self.half_extent_z_range_m = (float(half[:, 2].min()), float(half[:, 2].max()))
        self._half_np = half

    def _setup_options(self, phi, grid, sign_mask):
        """Per-point flag of the selected body, its wall distance field (selected_wall_on_obstacle_faces) and the build
        report."""
        self.options_report = dict(lid_z_m=self.lid_z_m if self.lid_on else None,
                                   selected_wall_on_obstacle_faces=self.selected_wall,
                                   selected_body_id=self.selected_body_id, selected_omega_dt=self.selected_omega_dt)
        if not self.options_on:
            return
        body_id = np.asarray(self.points.body_id.to_numpy(), dtype=np.int32)
        self.is_selected = ti.field(ti.i32, shape=self.n)
        self.is_selected.from_numpy((body_id == self.selected_body_id).astype(np.int32))
        self.options_report["selected_points"] = int((body_id == self.selected_body_id).sum())
        if self.lid_on:
            x0 = np.asarray(self.points.X0.to_numpy(), dtype=np.float64)
            top = float(x0[:, 2].max())
            if not top + float(self._half_np[:, 2].max()) <= self.lid_z_m:
                raise SimulationHardStop("contact_setup_invalid", dict(reason="a material box crosses the lid at "
                                                                     "the reference state", lid_z_m=self.lid_z_m,
                                                                     top_point_z_m=top))
            self.options_report["reference_top_gap_m"] = self.lid_z_m - top
        if self.selected_wall:
            if sign_mask is None:
                raise ValueError("selected_wall_on_obstacle_faces needs the obstacle sign mask (sign_from_obstacle=True)")
            started = time.perf_counter()
            wall, rep = wall_distance_on_obstacle_faces(phi, grid, sign_mask)
            rep["build_s"] = time.perf_counter() - started
            self.phi_wall = ti.field(ti.f32, shape=self.nodes)
            self.phi_wall.from_numpy(wall.astype(np.float32))
            self.options_report["wall_distance_on_obstacle_faces"] = rep

    def _setup_rest_cap(self, on):
        """rest_cap: every point's cap of its wall contact distance, phi_p(X0) - REST_CAP_MARGIN_M, with phi_p(X0) from
        the step kernel's own functions at the reference positions (_rest_phi_kernel), and the report (None when off)."""
        self.rest_cap_on = bool(on)
        self.rest_cap_report = None
        if not self.rest_cap_on:
            return
        self.rest_cap = ti.field(ti.f32, shape=self.n)
        rest = ti.field(ti.f32, shape=(self.n, 2))            # phi(X0) and d_c(n(X0)) (-1: the normal is undefined)
        self._rest_phi_kernel(self.points.X0, rest)
        values = rest.to_numpy().astype(np.float64)
        phi0, dc0 = values[:, 0], values[:, 1]
        if not np.isfinite(phi0).all():
            raise SimulationHardStop("contact_setup_invalid", dict(reason="a nonfinite wall distance at rest"))
        cap = (phi0 - REST_CAP_MARGIN_M).astype(np.float32)
        self.rest_cap.from_numpy(cap)
        free = np.asarray(self.points.tether.to_numpy()) == 0
        body = np.asarray(self.points.body_id.to_numpy())
        capped = free & (dc0 >= 0.0) & (cap.astype(np.float64) < dc0)           # the cap acts at rest
        inside = free & (phi0 <= 0.0)
        self.d_c_rule += "; capped at the rest wall distance phi(X0) - %.0e m (rest_cap)" % REST_CAP_MARGIN_M
        self.rest_cap_report = dict(
            margin_m=REST_CAP_MARGIN_M, free_points=int(free.sum()), capped_points=int(capped.sum()),
            rest_phi_le_0_points=int(inside.sum()),
            capped_rest_phi_m=[float(phi0[capped].min()), float(phi0[capped].max())] if capped.any() else None,
            rest_phi_le_0_m=[float(phi0[inside].min()), float(phi0[inside].max())] if inside.any() else None,
            rest_preload_removed_max_m=float((dc0 - phi0)[capped].max()) if capped.any() else 0.0,
            by_body={str(int(b)): dict(capped_points=int((capped & (body == b)).sum()),
                                       rest_phi_le_0_points=int((inside & (body == b)).sum()))
                     for b in np.unique(body[free])})

    # ------------------------------------------------------------------------------------------------ kernels
    @ti.func
    def _locate(self, centres: ti.template(), n: ti.template(), coord):
        """Centre interval i (centres[i] <= coord < centres[i+1], clamped) and the fraction t in [0, 1]."""
        i = 0
        t = 0.0
        if coord <= centres[0]:
            i, t = 0, 0.0
        elif coord >= centres[n - 1]:
            i, t = n - 2, 1.0
        else:
            lo, hi = 0, n - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if centres[mid] <= coord:
                    lo = mid
                else:
                    hi = mid
            i = lo
            t = (coord - centres[lo]) / (centres[lo + 1] - centres[lo])
        return i, t

    @ti.func
    def _phi_grad(self, xp):
        """Trilinear phi and its exact gradient at xp."""
        i, tx = self._locate(self.centre_x, self.nodes[0], xp[0])
        j, ty = self._locate(self.centre_y, self.nodes[1], xp[1])
        k, tz = self._locate(self.centre_z, self.nodes[2], xp[2])
        hx = self.centre_x[i + 1] - self.centre_x[i]
        hy = self.centre_y[j + 1] - self.centre_y[j]
        hz = self.centre_z[k + 1] - self.centre_z[k]
        phi = 0.0
        g = ti.Vector([0.0, 0.0, 0.0])
        for a, b, c in ti.static(ti.ndrange(2, 2, 2)):
            wx = tx if a == 1 else 1.0 - tx
            wy = ty if b == 1 else 1.0 - ty
            wz = tz if c == 1 else 1.0 - tz
            sx = 1.0 if a == 1 else -1.0
            sy = 1.0 if b == 1 else -1.0
            sz = 1.0 if c == 1 else -1.0
            v = self.phi[i + a, j + b, k + c]
            phi += wx * wy * wz * v
            g += ti.Vector([sx * wy * wz / hx, wx * sy * wz / hy, wx * wy * sz / hz]) * v
        return phi, g

    @ti.func
    def _phi_grad_of(self, field: ti.template(), xp):
        """trilinear value and exact gradient of a cell-centre field (same arithmetic as _phi_grad)."""
        i, tx = self._locate(self.centre_x, self.nodes[0], xp[0])
        j, ty = self._locate(self.centre_y, self.nodes[1], xp[1])
        k, tz = self._locate(self.centre_z, self.nodes[2], xp[2])
        hx = self.centre_x[i + 1] - self.centre_x[i]
        hy = self.centre_y[j + 1] - self.centre_y[j]
        hz = self.centre_z[k + 1] - self.centre_z[k]
        phi = 0.0
        g = ti.Vector([0.0, 0.0, 0.0])
        for a, b, c in ti.static(ti.ndrange(2, 2, 2)):
            wx = tx if a == 1 else 1.0 - tx
            wy = ty if b == 1 else 1.0 - ty
            wz = tz if c == 1 else 1.0 - tz
            sx = 1.0 if a == 1 else -1.0
            sy = 1.0 if b == 1 else -1.0
            sz = 1.0 if c == 1 else -1.0
            v = field[i + a, j + b, k + c]
            phi += wx * wy * wz * v
            g += ti.Vector([sx * wy * wz / hx, wx * sy * wz / hy, wx * wy * sz / hz]) * v
        return phi, g

    @ti.func
    def _wall_phi_grad(self, p, xp):
        """The wall distance of point p: phi_wall for the selected body's points when selected_wall_on_obstacle_faces, else
        phi."""
        phi = 0.0
        g = ti.Vector([0.0, 0.0, 0.0])
        if ti.static(self.selected_wall):
            if self.is_selected[p] != 0:
                a, b = self._phi_grad_of(self.phi_wall, xp)
                phi = a
                g = b
            else:
                a, b = self._phi_grad_of(self.phi, xp)
                phi = a
                g = b
        else:
            a, b = self._phi_grad_of(self.phi, xp)
            phi = a
            g = b
        return phi, g

    @ti.func
    def _point_k(self, p, k_c, k_selected):
        """The contact stiffness of point p (the selected body's points: k_selected when selected_omega_dt is set)."""
        k = k_c
        if ti.static(self.selected_stiff):
            if self.is_selected[p] != 0:
                k = k_selected
        return k

    @ti.func
    def _support_D(self, p, xp):
        """Diagnostics: the smallest one-sided denominator D of point p over the three MAC components (the coupler's
        own weights and mask)."""
        xq = xp
        local = 1.0e30
        for c in ti.static(range(3)):
            ix, iy, iz, wx, wy, wz, dwx, dwy, dwz = self.coupler._axis(c, xq)
            D, gD = self.coupler._denominator(c, ix, iy, iz, wx, wy, wz, dwx, dwy, dwz)
            local = ti.min(local, D)
        return local

    @ti.func
    def _contact_distance(self, p, nrm):
        """Contact distance of point p along the unit normal: the support distance a . |n| of its material box, or
        the fixed d_c_m."""
        d = 0.0
        if ti.static(self.fixed_dc > 0.0):
            d = self.fixed_dc
        else:
            a = self.half_extent[p]
            d = a[0] * ti.abs(nrm[0]) + a[1] * ti.abs(nrm[1]) + a[2] * ti.abs(nrm[2])
        return d

    @ti.func
    def _wall_contact_distance(self, p, nrm):
        """Contact distance of point p against the walls: _contact_distance, capped at the point's rest wall distance
        (less REST_CAP_MARGIN_M) when rest_cap is on.  The lid keeps _contact_distance."""
        d = self._contact_distance(p, nrm)
        if ti.static(self.rest_cap_on):
            d = ti.min(d, self.rest_cap[p])
        return d

    @ti.kernel
    def _rest_phi_kernel(self, X0: ti.template(), out: ti.template()):
        """rest_cap: each point's wall distance and contact distance at its reference position, by the step kernel's
        own functions (the base phi, or the per-point wall phi when an option is on)."""
        for p in range(self.n):
            phi = 0.0
            g = ti.Vector([0.0, 0.0, 0.0])
            if ti.static(not self.options_on):
                a, b = self._phi_grad(X0[p])
                phi = a
                g = b
            else:
                a, b = self._wall_phi_grad(p, X0[p])
                phi = a
                g = b
            dc = -1.0
            gn = g.norm()
            if gn > 1.0e-6:
                nrm = g / gn
                dc = self._contact_distance(p, nrm)
            out[p, 0] = phi
            out[p, 1] = dc

    @ti.func
    def _contact_area(self, p, nrm):
        """Area of the material box of point p projected along the unit normal (its touching face for an
        axis-aligned contact)."""
        a = self.half_extent[p]
        return 4.0 * (a[1] * a[2] * ti.abs(nrm[0]) + a[0] * a[2] * ti.abs(nrm[1]) + a[0] * a[1] * ti.abs(nrm[2]))

    @ti.kernel
    def _init_stats_kernel(self):
        for s in range(STATS_SLOTS):
            self.stats[s] = 0.0

    @ti.kernel
    def _reset_interval_kernel(self):
        for s in range(A_STEPS, A_UNDEF + 1):
            self.stats[s] = 0.0
        if ti.static(self.options_on):
            for s in ti.static((V_A_LID_IMPULSE, V_A_LID_MAX_PEN, V_A_SEL_IMPULSE, V_A_SEL_MAX_PEN)):
                self.stats[s] = 0.0

    @ti.kernel
    def _step_kernel(self, x: ti.template(), vol: ti.template(), tether: ti.template(), f_extra: ti.template(),
                     dt: ti.f32, k_fixed: ti.f32, rho_s: ti.f32, k_selected: ti.f32):
        k_c = k_fixed
        if k_c <= 0.0:
            omega = CONTACT_OMEGA_DT / dt
            k_c = rho_s * omega * omega
        for s in range(S_NONFINITE + 1):
            self.stats[s] = 0.0
        if ti.static(self.options_on):
            for s in ti.static((V_S_LID_FORCE, V_S_LID_POINTS, V_S_SEL_FORCE)):
                self.stats[s] = 0.0
        for p in range(self.n):
            if tether[p] == 0:
                if ti.static(not self.options_on):
                    phi, g = self._phi_grad(x[p])
                    gn = g.norm()
                    if gn > 1.0e-6:
                        nrm = g / gn
                        dc = self._wall_contact_distance(p, nrm)
                        if phi < dc:
                            mag = k_c * vol[p] * (dc - phi)
                            f = f_extra[p]
                            f += mag * nrm
                            f_extra[p] = f
                            ti.atomic_add(self.stats[S_FORCE], ti.cast(mag, ti.f64))
                            ti.atomic_add(self.stats[S_POINTS], 1.0)
                    else:
                        ti.atomic_add(self.stats[S_UNDEF], 1.0)
                    ti.atomic_max(self.stats[A_MAX_PEN], ti.cast(ti.max(-phi, 0.0), ti.f64))
                    if ti.math.isnan(phi) or ti.math.isinf(phi):
                        ti.atomic_add(self.stats[S_NONFINITE], 1.0)
                else:                                       # wall phi per point, own stiffness, the lid
                    k_p = self._point_k(p, k_c, k_selected)
                    phi, g = self._wall_phi_grad(p, x[p])
                    gn = g.norm()
                    if gn > 1.0e-6:
                        nrm = g / gn
                        dc = self._wall_contact_distance(p, nrm)
                        if phi < dc:
                            mag = k_p * vol[p] * (dc - phi)
                            f = f_extra[p]
                            f += mag * nrm
                            f_extra[p] = f
                            ti.atomic_add(self.stats[S_FORCE], ti.cast(mag, ti.f64))
                            ti.atomic_add(self.stats[S_POINTS], 1.0)
                            if self.is_selected[p] != 0:
                                ti.atomic_add(self.stats[V_S_SEL_FORCE], ti.cast(mag, ti.f64))
                    else:
                        ti.atomic_add(self.stats[S_UNDEF], 1.0)
                    ti.atomic_max(self.stats[A_MAX_PEN], ti.cast(ti.max(-phi, 0.0), ti.f64))
                    if self.is_selected[p] != 0:
                        ti.atomic_max(self.stats[V_A_SEL_MAX_PEN], ti.cast(ti.max(-phi, 0.0), ti.f64))
                    if ti.math.isnan(phi) or ti.math.isinf(phi):
                        ti.atomic_add(self.stats[S_NONFINITE], 1.0)
                    if ti.static(self.lid_on):                 # the analytic lid plane z = lid_z
                        down = ti.Vector([0.0, 0.0, -1.0])
                        phi_lid = self.lid_z - x[p][2]
                        dc_lid = self._contact_distance(p, down)
                        if phi_lid < dc_lid:
                            mag = k_p * vol[p] * (dc_lid - phi_lid)
                            f = f_extra[p]
                            f += mag * down
                            f_extra[p] = f
                            ti.atomic_add(self.stats[V_S_LID_FORCE], ti.cast(mag, ti.f64))
                            ti.atomic_add(self.stats[V_S_LID_POINTS], 1.0)
                        ti.atomic_max(self.stats[V_A_LID_MAX_PEN], ti.cast(ti.max(-phi_lid, 0.0), ti.f64))
                        if ti.math.isnan(phi_lid) or ti.math.isinf(phi_lid):
                            ti.atomic_add(self.stats[S_NONFINITE], 1.0)
        # serial accumulation of the interval statistics
        dt64 = ti.cast(dt, ti.f64)
        self.stats[A_STEPS] += 1.0
        self.stats[A_TIME] += dt64
        self.stats[A_IMPULSE] += self.stats[S_FORCE] * dt64
        if ti.static(self.options_on):
            self.stats[V_A_LID_IMPULSE] += self.stats[V_S_LID_FORCE] * dt64
            self.stats[V_A_SEL_IMPULSE] += self.stats[V_S_SEL_FORCE] * dt64
        self.stats[A_NONFINITE] += self.stats[S_NONFINITE]
        self.stats[A_UNDEF] += self.stats[S_UNDEF]

    @ti.kernel
    def _measure_kernel(self, x: ti.template(), vol: ti.template(), tether: ti.template(), k_c: ti.f32,
                        k_selected: ti.f32):
        for s in range(M_FORCE, M_NONFINITE + 1):
            self.stats[s] = 0.0
        self.stats[M_MIN_PHI] = 1.0e30
        if ti.static(self.options_on):
            for s in ti.static((V_M_LID_FORCE, V_M_LID_POINTS, V_M_LID_ENERGY, V_M_LID_MAX_PEN, V_M_SEL_FORCE,
                                V_M_SEL_POINTS, V_M_SEL_MAX_PEN)):
                self.stats[s] = 0.0
            for s in ti.static((V_M_LID_GAP_MIN, V_M_SEL_MIN_PHI, V_M_SEL_MIN_D)):
                self.stats[s] = 1.0e30
        for p in range(self.n):
            if tether[p] == 0:
                if ti.static(self.options_on):
                    self._measure_option_point(p, x, vol, k_c, k_selected)
                phi, g = self._phi_grad(x[p])
                if ti.static(self.selected_wall):
                    a, b = self._wall_phi_grad(p, x[p])
                    phi = a
                    g = b
                k_p = k_c
                if ti.static(self.selected_stiff):
                    k_p = self._point_k(p, k_c, k_selected)
                phi64 = ti.cast(phi, ti.f64)
                ti.atomic_min(self.stats[M_MIN_PHI], phi64)
                ti.atomic_max(self.stats[M_MAX_PEN], ti.max(-phi64, 0.0))
                gn = g.norm()
                if gn > 1.0e-6:
                    nrm = g / gn
                    dc = self._wall_contact_distance(p, nrm)
                    if phi < dc:
                        depth = ti.cast(dc - phi, ti.f64)
                        mag = ti.cast(k_p * vol[p], ti.f64) * depth
                        ti.atomic_add(self.stats[M_FORCE], mag)
                        ti.atomic_add(self.stats[M_POINTS], 1.0)
                        ti.atomic_add(self.stats[M_AREA], ti.cast(self._contact_area(p, nrm), ti.f64))
                        ti.atomic_add(self.stats[M_ENERGY], 0.5 * mag * depth)
                        ti.atomic_add(self.stats[M_FX], mag * ti.cast(nrm[0], ti.f64))
                        ti.atomic_add(self.stats[M_FY], mag * ti.cast(nrm[1], ti.f64))
                        ti.atomic_add(self.stats[M_FZ], mag * ti.cast(nrm[2], ti.f64))
                else:
                    ti.atomic_add(self.stats[M_UNDEF], 1.0)
                if ti.math.isnan(phi) or ti.math.isinf(phi):
                    ti.atomic_add(self.stats[M_NONFINITE], 1.0)

    @ti.func
    def _measure_option_point(self, p, x: ti.template(), vol: ti.template(), k_c, k_selected):
        """Option diagnostics of one non-tethered point: lid force / energy / gap / penetration, the selected body's wall
        contact force, min phi_wall, penetration past phi_wall = 0, and the support D of its points in contact."""
        k_p = self._point_k(p, k_c, k_selected)
        in_contact = 0
        if ti.static(self.lid_on):
            down = ti.Vector([0.0, 0.0, -1.0])
            phi_lid = self.lid_z - x[p][2]
            dc_lid = self._contact_distance(p, down)
            ti.atomic_min(self.stats[V_M_LID_GAP_MIN], ti.cast(phi_lid - dc_lid, ti.f64))
            ti.atomic_max(self.stats[V_M_LID_MAX_PEN], ti.cast(ti.max(-phi_lid, 0.0), ti.f64))
            if phi_lid < dc_lid:
                depth = ti.cast(dc_lid - phi_lid, ti.f64)
                mag = ti.cast(k_p * vol[p], ti.f64) * depth
                ti.atomic_add(self.stats[V_M_LID_FORCE], mag)
                ti.atomic_add(self.stats[V_M_LID_POINTS], 1.0)
                ti.atomic_add(self.stats[V_M_LID_ENERGY], 0.5 * mag * depth)
                in_contact = 1
        if self.is_selected[p] != 0:
            phi, g = self._wall_phi_grad(p, x[p])
            ti.atomic_min(self.stats[V_M_SEL_MIN_PHI], ti.cast(phi, ti.f64))
            ti.atomic_max(self.stats[V_M_SEL_MAX_PEN], ti.cast(ti.max(-phi, 0.0), ti.f64))
            gn = g.norm()
            if gn > 1.0e-6:
                dc = self._wall_contact_distance(p, g / gn)
                if phi < dc:
                    ti.atomic_add(self.stats[V_M_SEL_FORCE], ti.cast(k_p * vol[p], ti.f64) * ti.cast(dc - phi, ti.f64))
                    ti.atomic_add(self.stats[V_M_SEL_POINTS], 1.0)
                    in_contact = 1
            if in_contact != 0:
                ti.atomic_min(self.stats[V_M_SEL_MIN_D], ti.cast(self._support_D(p, x[p]), ti.f64))

    def selected_contact_stiffness(self, dt_s):
        """k of the selected body's points (N/m^4): rho_s (selected_omega_dt / dt)^2, or the base k_c."""
        if not self.selected_stiff:
            return self.contact_stiffness(dt_s)
        omega = self.selected_omega_dt / float(dt_s)
        return self.rho_s * omega * omega

    # ------------------------------------------------------------------------------------------------ host API
    def contact_stiffness(self, dt_s):
        """k_c in N/m^4 (force per unit point volume and unit penetration)."""
        if self.k_c is not None:
            return self.k_c
        omega = CONTACT_OMEGA_DT / float(dt_s)
        return self.rho_s * omega * omega

    def apply_point_forces(self, dt_s):
        """ONE kernel launch: add the contact force of every non-tethered point with phi < d_c to points.f_extra and
        accumulate the interval statistics on the device.  Call right before coupler.spread_force()."""
        self._step_kernel(self.points.x, self.points.vol, self.points.tether, self.points.f_extra,
                          dt_s, 0.0 if self.k_c is None else self.k_c, self.rho_s,
                          self.selected_contact_stiffness(dt_s) if self.options_on else 0.0)

    def report(self, t_s, dt_s):
        """Contact state now plus the interval since the previous report (one device read); resets the interval."""
        t, dt = float(t_s), float(dt_s)
        k = self.contact_stiffness(dt)
        self._measure_kernel(self.points.x, self.points.vol, self.points.tether, k,
                             self.selected_contact_stiffness(dt) if self.options_on else 0.0)
        s = self.stats.to_numpy()                                          # the one device read of a report
        self._reset_interval_kernel()
        if s[M_NONFINITE] > 0 or s[A_NONFINITE] > 0 or not np.isfinite(s[:EV_BASE]).all():
            raise SimulationHardStop("contact_nonfinite_state", dict(nonfinite_phi_points=int(s[M_NONFINITE]),
                                                                     interval_nonfinite=int(s[A_NONFINITE]), t_s=t))
        span = float(s[A_TIME])
        out = dict(t_s=t, k_c=k, interval_steps=int(s[A_STEPS]), interval_time_s=span,
                   contact_force_n=float(s[M_FORCE]), contact_points=int(s[M_POINTS]),
                   contact_area_m2=float(s[M_AREA]), contact_energy_j=float(s[M_ENERGY]),
                   contact_force_vector_n=[float(s[M_FX]), float(s[M_FY]), float(s[M_FZ])],
                   contact_force_interval_mean_n=float(s[A_IMPULSE]) / span if span > 0 else None,
                   normal_undefined_points=int(s[M_UNDEF]),
                   min_phi_m=float(s[M_MIN_PHI]) if s[M_MIN_PHI] < 1.0e29 else None,
                   max_penetration_m=max(float(s[M_MAX_PEN]), float(s[A_MAX_PEN])),
                   max_penetration_now_m=float(s[M_MAX_PEN]), dissipated_energy_j=0.0, d_c_rule=self.d_c_rule,
                   normal_undefined_steps_interval=int(s[A_UNDEF]))
        if self.options_on:                                         # option diagnostics
            big = 1.0e29
            out.update(selected_k_c=self.selected_contact_stiffness(dt),
                       selected_omega_dt=self.selected_omega_dt,
                       selected_contact_force_n=float(s[V_M_SEL_FORCE]),
                       selected_contact_points=int(s[V_M_SEL_POINTS]),
                       selected_contact_force_interval_mean_n=float(s[V_A_SEL_IMPULSE]) / span if span > 0 else None,
                       selected_wall_min_phi_m=float(s[V_M_SEL_MIN_PHI]) if s[V_M_SEL_MIN_PHI] < big else None,
                       selected_wall_penetration_max_m=max(float(s[V_M_SEL_MAX_PEN]), float(s[V_A_SEL_MAX_PEN])),
                       min_D_selected_contact=float(s[V_M_SEL_MIN_D]) if s[V_M_SEL_MIN_D] < big else None)
            if self.lid_on:
                out.update(lid_z_m=self.lid_z_m, lid_contact_force_n=float(s[V_M_LID_FORCE]),
                           lid_contact_points=int(s[V_M_LID_POINTS]), lid_contact_energy_j=float(s[V_M_LID_ENERGY]),
                           lid_contact_force_interval_mean_n=float(s[V_A_LID_IMPULSE]) / span if span > 0 else None,
                           lid_gap_min_m=float(s[V_M_LID_GAP_MIN]) if s[V_M_LID_GAP_MIN] < big else None,
                           lid_penetration_max_m=max(float(s[V_M_LID_MAX_PEN]), float(s[V_A_LID_MAX_PEN])))
        self._last_report_t = t
        return out

    def setup_report(self):
        out = dict(sdf=self.sdf_report, d_c_rule=self.d_c_rule,
                   d_c_range_m=list(self.d_c_range_m), half_extent_z_range_m=list(self.half_extent_z_range_m),
                   k_c=self.k_c, friction="none (normal only)", adhesion="none (repulsive only)", damping="none",
                   options=self.options_report)
        if self.rest_cap_on:
            out["rest_cap"] = self.rest_cap_report
        return out

    def close(self):
        return None
