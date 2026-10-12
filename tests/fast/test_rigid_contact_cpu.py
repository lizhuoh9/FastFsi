"""Physical rigid contact (simulation_core/fast/rigid_contact.py; USER decision 01:35Z, Root REQ_403): CPU tests.

Unit tests (cheap): exact point-triangle distance; the SDF of a closed box equals the analytic box distance in the band
and bounds it beyond; trilinear phi and normal are exact for a plane; the contact law is repulsive, normal-only,
frictionless, skips tethered points and adds to f_extra; report diagnostics (rim gap, contact fraction, refill flow,
leakage split under-resolved / resolved, events on the 0.5 crossing to a create-only log set as an attribute); hard stop
on an open surface.

Rig (contract section 4 successor): the production CartesianFluidSolver (MUSCL predictor, fv_cg + fv_multigrid, z-min
pressure outlet) with the integrator's FastIbMpm3D flap (one-sided weights through an obstacle-aware B2 face mask)
above a rigid plate with one refill port, the flap centre bonded on a rigid post; a pump in an always-open nozzle duct
steps the pressure difference that pushes the flap down.  The rigid plate and post are closed CAD-like triangle
solids; the static obstacle is their closed-solid classification (as the squid's) and the SDF uses the same triangles.
"""
import json
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from simulation_core.fast.rigid_contact import (FastModeHardStop, HARD_STOP_CODES, RigidContact, TAG_VALVE_RIM,
                                                build_signed_distance, point_triangle_distance)
from simulation_core.fluids.grid import CartesianGrid


def box_triangles(lo, hi):
    """12 outward-wound triangles of the closed box [lo, hi] (m)."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float64)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    tri = []
    for a, b, c, d in quads:
        tri.append(v[[a, b, c]])
        tri.append(v[[a, c, d]])
    return np.asarray(tri)


def box_sdf(points, lo, hi):
    lo, hi = np.asarray(lo), np.asarray(hi)
    q = np.abs(points - 0.5 * (lo + hi)) - 0.5 * (hi - lo)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


# ---------------------------------------------------------------------------------------------------- unit tests
def test_point_triangle_distance_is_exact():
    a, b, c = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    p = np.array([[0.2, 0.2, 0.5], [2.0, 0.0, 0.0], [-1.0, -1.0, 0.0], [0.6, 0.6, -0.3], [0.5, -2.0, 1.0]])
    expected = [0.5, 1.0, math.sqrt(2.0), math.sqrt(0.1 ** 2 * 2 + 0.3 ** 2), math.sqrt(4.0 + 1.0)]
    got = point_triangle_distance(p, np.tile(a, (5, 1)), np.tile(b, (5, 1)), np.tile(c, (5, 1)))
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    rng = np.random.default_rng(2)
    tri = rng.normal(size=(40, 3, 3))
    pts = rng.normal(size=(40, 3))
    u, v = np.meshgrid(np.linspace(0, 1, 301), np.linspace(0, 1, 301))
    keep = u + v <= 1.0
    u, v = u[keep], v[keep]
    dense = tri[:, 0][:, None] + u[None, :, None] * (tri[:, 1] - tri[:, 0])[:, None] \
        + v[None, :, None] * (tri[:, 2] - tri[:, 0])[:, None]
    brute = np.linalg.norm(dense - pts[:, None], axis=2).min(axis=1)
    exact = point_triangle_distance(pts, tri[:, 0], tri[:, 1], tri[:, 2])
    assert np.all(exact <= brute + 1e-12) and np.all(brute - exact < 2e-2 * np.maximum(1.0, exact))


def test_box_signed_distance_is_exact_in_the_band():
    grid = CartesianGrid.uniform(bounds_min_m=(0.0, 0.0, 0.0), bounds_max_m=(0.012, 0.010, 0.008),
                                 grid_nodes=(12, 10, 8))
    lo, hi = (0.0031, 0.0024, 0.0018), (0.0083, 0.0072, 0.0046)
    phi, report = build_signed_distance({"box": box_triangles(lo, hi)}, grid)
    c = [np.asarray(v) for v in (grid.cell_centers_x_m, grid.cell_centers_y_m, grid.cell_centers_z_m)]
    X, Y, Z = np.meshgrid(*c, indexing="ij")
    exact = box_sdf(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]), lo, hi).reshape(phi.shape)
    band = np.abs(exact) < 2.0e-3                                     # 2 local cells of 1 mm
    np.testing.assert_allclose(phi[band], exact[band], atol=1e-12)
    assert np.all(np.abs(phi[~band]) >= np.abs(exact[~band]) - 1e-12)  # nearest-sample distance: upper bound
    assert np.all(np.abs(phi[~band]) - np.abs(exact[~band]) <= 2.0 * report["sample_spacing_m"])
    # a reach restricts the exact part to cells near the given points; elsewhere the sample distance remains
    reach_pts = np.array([[0.002, y, z] for y in np.linspace(0.0, 0.010, 11) for z in np.linspace(0.0, 0.008, 9)])
    phi_r, rep_r = build_signed_distance({"box": box_triangles(lo, hi)}, grid, reach_points_m=reach_pts,
                                         reach_margin_m=0.003)
    near_pts = X <= 0.0045                                            # within 3 mm of a reach point
    np.testing.assert_allclose(phi_r[band & near_pts], exact[band & near_pts], atol=1e-12)
    assert rep_r["exact_cells"] < report["exact_cells"]
    assert np.all(np.sign(phi_r) == np.sign(phi)) and np.all(np.abs(phi_r) >= np.abs(exact) - 1e-12)
    assert report["inside_cells"] == int((exact < 0).sum())


@ti.kernel
def _probe_phi(contact: ti.template(), pts: ti.types.ndarray(), out: ti.types.ndarray()):
    for q in range(pts.shape[0]):
        phi, g = contact._phi_grad(ti.Vector([pts[q, 0], pts[q, 1], pts[q, 2]]))
        out[q, 0] = phi
        out[q, 1] = g[0]
        out[q, 2] = g[1]
        out[q, 3] = g[2]


NODES_S = (16, 16, 12)
H_S = 1.0e-3
AXIS_S = (8.0e-3, 8.0e-3)
LAND_Z_S = 5.0e-3                     # plate z 2-5 mm (cells k 2..4), land = plate top
PORT_S = (10, 7)                      # port column (r = 2.55 mm)
SP = 0.5e-3


def stub_rigid():
    """The plate with one port as closed boxes (one region per box, like separate closed CAD solids)."""
    i, j = PORT_S
    boxes = {"west": ((0.0, 0.0, 0.002), (i * H_S, 0.016, 0.005)),
             "east": (((i + 1) * H_S, 0.0, 0.002), (0.016, 0.016, 0.005)),
             "south": ((i * H_S, 0.0, 0.002), ((i + 1) * H_S, j * H_S, 0.005)),
             "north": ((i * H_S, (j + 1) * H_S, 0.002), ((i + 1) * H_S, 0.016, 0.005))}
    return {name: box_triangles(lo, hi) for name, (lo, hi) in boxes.items()}


def stub_fluid(rigid):
    from cases.squid_setup.closed_solid_cells import classify_closed_solid_cells
    grid = CartesianGrid.uniform(bounds_min_m=(0.0, 0.0, 0.0), bounds_max_m=tuple(n * H_S for n in NODES_S),
                                 grid_nodes=NODES_S)
    obstacle = np.zeros(NODES_S, bool)
    for tri in rigid.values():
        obstacle |= classify_closed_solid_cells(tri, grid)[0]
    f = SimpleNamespace(grid=grid, obstacle=ti.field(ti.i32, shape=NODES_S),
                        velocity=ti.Vector.field(3, ti.f32, shape=NODES_S), pressure=ti.field(ti.f64, shape=NODES_S))
    f.obstacle.from_numpy(obstacle.astype(np.int32))
    return f


def stub_body(points, tags=None, tether=None):
    pts = np.asarray(points, np.float32)
    n = len(pts)
    b = SimpleNamespace(x=ti.Vector.field(3, ti.f32, shape=n), vol=ti.field(ti.f32, shape=n),
                        tether=ti.field(ti.i32, shape=n), tag=ti.field(ti.i32, shape=n),
                        f_extra=ti.Vector.field(3, ti.f32, shape=n), rho_s=1000.0, n=n)
    b.x.from_numpy(pts)
    b.vol.fill(SP ** 3)
    b.tether.from_numpy(np.zeros(n, np.int32) if tether is None else np.asarray(tether, np.int32))
    b.tag.from_numpy(np.zeros(n, np.int32) if tags is None else np.asarray(tags, np.int32))
    return b


def ring_points(gap_m, rim=True):
    pts, tags = [], []
    for r in (4.75e-3, 5.25e-3, 5.75e-3):
        for a in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
            pts.append((AXIS_S[0] + r * np.cos(a), AXIS_S[1] + r * np.sin(a), LAND_Z_S + 0.5 * SP + gap_m))
            tags.append(TAG_VALVE_RIM if rim else 0)
    return pts, tags


def stub_contact(f, b, **kw):
    return RigidContact(f, SimpleNamespace(n=NODES_S), b, stub_rigid(), axis_xy_m=AXIS_S, land_z_m=LAND_Z_S,
                        land_r_m=(4.5e-3, 6.0e-3), port_r_m=(2.0e-3, 3.5e-3), channel_r_m=None,
                        refill_plane_z_m=3.0e-3, flap_thickness_m=1.0e-3, **kw)


def test_trilinear_phi_and_normal_are_exact_for_a_plane():
    rigid = stub_rigid()
    f = stub_fluid(rigid)
    b = stub_body([(0.004, 0.004, 0.0065)])
    contact = stub_contact(f, b, k_c=1.0e8, reach_margin_m=None)      # exact SDF in the whole band
    rng = np.random.default_rng(4)
    # points within a cell of the plate top (where contact acts): their trilinear stencil lies in the exact band
    pts = np.column_stack([rng.uniform(0.0, 0.004, 20), rng.uniform(0.0, 0.016, 20),
                           rng.uniform(0.0051, 0.0064, 20)]).astype(np.float32)
    out = np.zeros((20, 4), np.float32)
    _probe_phi(contact, pts, out)
    np.testing.assert_allclose(out[:, 0], pts[:, 2] - LAND_Z_S, atol=2e-7)   # above the plate top, away from the port
    np.testing.assert_allclose(out[:, 1:], np.tile([0.0, 0.0, 1.0], (20, 1)), atol=1e-4)


def test_sign_follows_the_run_obstacle_mask():
    rigid = stub_rigid()
    f = stub_fluid(rigid)
    obstacle = f.obstacle.to_numpy()
    obstacle[2, 2, 3] = 0                                  # carved: inside the CAD plate, fluid in the run (groove)
    obstacle[2, 2, 7] = 1                                  # sealed: outside the CAD, obstacle in the run (thin wall)
    f.obstacle.from_numpy(obstacle)
    b = stub_body([(0.004, 0.004, 0.0065)])
    contact = stub_contact(f, b, k_c=1.0e8)
    phi = contact.phi.to_numpy().astype(np.float64)
    np.testing.assert_array_equal(phi < 0.0, obstacle != 0)          # sign = run obstacle mask, cell by cell
    assert phi[2, 2, 3] == pytest.approx(1.5e-3, abs=1e-7)           # magnitude: distance to the CAD (plate bottom)
    assert phi[2, 2, 7] <= -2.5e-3 + 1e-7                            # 2.5 mm above the CAD plate top
    rep = contact.setup_report()["sdf"]
    assert rep["sign_source"].startswith("run static obstacle mask")
    assert rep["cad_inside_fluid_cells"] == 1 and rep["cad_outside_obstacle_cells"] == 1
    cad = stub_contact(f, b, k_c=1.0e8, sign_from_obstacle=False).phi.to_numpy()
    assert cad[2, 2, 3] < 0.0 < cad[2, 2, 7]                          # the CAD sign, for comparison


def test_contact_law_is_repulsive_normal_only_frictionless_and_skips_tethers():
    rigid = stub_rigid()
    f = stub_fluid(rigid)
    z = LAND_Z_S + np.array([0.10e-3, 0.20e-3, 0.30e-3, -0.05e-3, 0.10e-3])
    pts = np.column_stack([np.full(5, 0.0035), np.linspace(0.003, 0.013, 5), z])
    tether = [0, 0, 0, 0, 1]
    b = stub_body(pts, tether=tether)
    preset = np.random.default_rng(1).normal(size=(5, 3)).astype(np.float32) * 1e-7
    b.f_extra.from_numpy(preset)
    k = 2.0e8
    contact = stub_contact(f, b, k_c=k)
    contact.apply_point_forces(1.0e-4)
    out = b.f_extra.to_numpy().astype(np.float64)
    d_c = 0.5 * SP
    phi = b.x.to_numpy()[:, 2].astype(np.float64) - LAND_Z_S
    expected = preset.astype(np.float64).copy()
    for p in range(5):
        if tether[p] == 0 and phi[p] < d_c:
            expected[p, 2] += k * SP ** 3 * (d_c - phi[p])
    np.testing.assert_allclose(out, expected, rtol=2e-4, atol=1e-13)      # normal (+z) only, nothing tangential
    assert out[2, 2] == pytest.approx(preset[2, 2])                       # phi = 0.3 mm > d_c: no force
    assert out[4, 2] == pytest.approx(preset[4, 2])                       # tethered point: no contact force
    rep = contact.report(1.0e-4, 1.0e-4)
    applied = sum(k * SP ** 3 * (d_c - phi[p]) for p in range(3) if phi[p] < d_c) + k * SP ** 3 * (d_c - phi[3])
    assert rep["interval_steps"] == 1 and rep["contact_points"] == 3
    assert rep["contact_force_interval_mean_n"] == pytest.approx(applied, rel=2e-4)
    assert rep["contact_force_n"] == pytest.approx(applied, rel=2e-4)       # same state measured again
    assert rep["max_penetration_m"] == pytest.approx(0.05e-3, rel=1e-3)    # the point 0.05 mm inside the plate
    assert rep["dissipated_energy_j"] == 0.0


def test_report_diagnostics_leakage_split_and_events(tmp_path):
    rigid = stub_rigid()
    f = stub_fluid(rigid)
    pts, tags = ring_points(2.0e-3)
    b = stub_body(pts, tags=tags)
    contact = stub_contact(f, b, k_c=1.0e8)
    path = tmp_path / "contact_events.jsonl"
    path.write_text("")
    with pytest.raises(FileExistsError):                   # create-only: an existing log fails at the assignment
        contact.event_log_path = str(path)
    path = tmp_path / "contact_events_2.jsonl"
    contact.event_log_path = str(path)                    # the driver sets the path after construction
    dt = 1.0e-4
    contact.apply_point_forces(dt)                        # step 1: open (gap 2 mm), no flow
    rep = contact.report(dt, dt, p_shift_pa=0.0)
    assert rep["rim_contact_fraction"] == 0.0 and not rep["closed"] and rep["interval_steps"] == 1
    assert rep["rim_gap_min_m"] == pytest.approx(2.0e-3, abs=1e-6)
    assert rep["kernel_half_width_m"] == pytest.approx(2.0 * H_S)
    vel = np.zeros(NODES_S + (3,), np.float32)
    vel[PORT_S[0], PORT_S[1], 3, 2] = 0.02                # refill face at z = 3 mm (k = 3): flux = u h^2
    f.velocity.from_numpy(vel)
    leak = 0.02 * H_S * H_S
    p = np.zeros(NODES_S)
    p[:, :, 8:10] = 40.0                                   # 2.5 cells above the rim's upper surface on the land
    p[PORT_S[0], PORT_S[1], 3] = 5.0                       # port cell above the refill face
    f.pressure.from_numpy(p - 100.0)                       # shifted field ...
    x = b.x.to_numpy()
    x[:, 2] = LAND_Z_S + 0.1e-3                            # point centres 0.1 mm above the plate: phi < d_c = 0.25 mm
    b.x.from_numpy(x)
    contact.apply_point_forces(dt)                        # step 2: closed, gap under-resolved
    rep = contact.report(2 * dt, dt, p_shift_pa=100.0)     # ... physical pressure = field + shift
    assert rep["rim_contact_fraction"] == 1.0 and rep["closed"]
    assert rep["rim_gap_min_m"] == pytest.approx(0.1e-3 - 0.5 * SP, abs=1e-6)   # material face below the land
    assert rep["refill_flow_m3s"] == pytest.approx(leak, rel=1e-6)
    assert rep["refill_flow_interval_mean_m3s"] == pytest.approx(leak, rel=1e-6)
    assert rep["p_above_land_pa"] == pytest.approx(40.0) and rep["p_ports_pa"] == pytest.approx(5.0)
    assert rep["dp_rim_pa"] == pytest.approx(35.0)
    assert rep["closed_leakage_m3s"] == pytest.approx(leak, rel=1e-6)
    assert rep["closed_leakage_underresolved_m3s"] == pytest.approx(leak, rel=1e-6)
    assert rep["closed_leakage_resolved_m3s"] == 0.0
    assert rep["contact_points"] == len(pts) and rep["max_penetration_m"] == 0.0
    assert rep["contact_force_n"] == pytest.approx(len(pts) * 1.0e8 * SP ** 3 * 0.15e-3, rel=1e-3)
    assert rep["contact_area_m2"] == pytest.approx(len(pts) * SP ** 2, rel=1e-5)
    assert rep["refill_flow_rim_underresolved_m3s"] == pytest.approx(leak, rel=1e-6)   # rim shut at grid resolution
    x[:24, 2] = LAND_Z_S + 0.5 * SP + 2.5e-3               # one ring of three lifted by 2.5 mm: resolved gap
    b.x.from_numpy(x)
    contact.apply_point_forces(dt)                        # step 3: closed (2/3 in contact), gap resolved
    rep = contact.report(3 * dt, dt, p_shift_pa=100.0)
    assert rep["closed"] and rep["closed_leakage_resolved_m3s"] == pytest.approx(leak, rel=1e-6)
    assert rep["closed_leakage_underresolved_m3s"] == 0.0
    assert rep["refill_flow_rim_underresolved_m3s"] == 0.0
    x[:48, 2] = LAND_Z_S + 0.5 * SP + 2.5e-3               # two of three lifted: contact ends
    b.x.from_numpy(x)
    contact.apply_point_forces(dt)                        # step 4: open
    contact.apply_point_forces(dt)                        # step 5: open (two steps in one report interval)
    rep = contact.report(5 * dt, dt, p_shift_pa=100.0)
    assert not rep["closed"] and rep["closed_leakage_m3s"] == 0.0 and rep["interval_steps"] == 2
    assert rep["refill_flow_interval_mean_m3s"] == pytest.approx(leak, rel=1e-6)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["contact_begin", "contact_end"] == [e["kind"] for e in contact.events()]
    assert rows[0]["t_s"] == pytest.approx(1.5 * dt) and rows[1]["t_s"] == pytest.approx(3.5 * dt)
    assert rep["closed_leakage_volume_m3"] == pytest.approx(2 * leak * dt, rel=1e-6)
    assert rep["closed_leakage_underresolved_volume_m3"] == pytest.approx(leak * dt, rel=1e-6)
    assert rep["closed_leakage_resolved_volume_m3"] == pytest.approx(leak * dt, rel=1e-6)
    assert rep["closed_time_s"] == pytest.approx(2 * dt)
    contact.close()


def test_hard_stop_on_an_open_rigid_surface():
    assert HARD_STOP_CODES == {"contact_setup_invalid", "contact_nonfinite_state"}
    rigid = stub_rigid()
    f = stub_fluid(rigid)
    b = stub_body([(0.004, 0.004, 0.0065)])
    broken = dict(rigid)
    broken["west"] = broken["west"][:-1]                   # one face missing: not a closed solid
    with pytest.raises(FastModeHardStop) as stop:
        RigidContact(f, SimpleNamespace(n=NODES_S), b, broken, k_c=1.0e8, axis_xy_m=AXIS_S, land_z_m=LAND_Z_S,
                     port_r_m=(2.0e-3, 3.5e-3), channel_r_m=None, refill_plane_z_m=3.0e-3)
    assert stop.value.code == "contact_setup_invalid"



# ---------------------------------------------------------------------------------------------------- rig
RIG = dict(
    nodes=(26, 22, 22), h=1.0e-3,
    plate_z=(5.0e-3, 9.0e-3),        # rigid plate z 5-9 mm; land = plate top
    axis=(10.0e-3, 11.0e-3),
    flap_r=8.0e-3, rim_r=6.5e-3,     # rim points: flap lower layer r >= rim_r (land rim_r .. flap_r)
    tether_r=2.0e-3,                 # bonded centre, on a rigid post (like the squid flap on the collar)
    post_xy=((8.0e-3, 9.0e-3), (12.0e-3, 13.0e-3)),
    gap0=2.0e-3, thick=2.0e-3,       # rest gap flap - plate, flap thickness
    ports=(((13, 15), (10, 12)),),   # refill ports (cell boxes) through the plate under the flap
    nozzle=((21, 23), (10, 12)),     # always-open nozzle duct outside the flap
    rho_f=1000.0, mu_f=0.02,         # viscous rig fluid (a calm rig, not the squid)
    rho_s=1000.0, mu_s=500.0, kappa_over_mu=10.0,
    f_pump=4.0e4,                    # N/m^3 on the nozzle z faces through the plate
    cfl=0.4, steps=200, refill_plane_z=7.0e-3)


def rig_rigid_triangles(cfg):
    """Plate with two through holes as closed boxes (one region per box) plus the post."""
    h, nodes = cfg["h"], cfg["nodes"]
    holes = list(cfg["ports"]) + [cfg["nozzle"]]
    xs = sorted({0, nodes[0]} | {v for (xi, _) in holes for v in xi})
    ys = sorted({0, nodes[1]} | {v for (_, yj) in holes for v in yj})
    out = {}
    for a, b in zip(xs[:-1], xs[1:]):
        for c, d in zip(ys[:-1], ys[1:]):
            if not any(i0 <= a < i1 and j0 <= c < j1 for (i0, i1), (j0, j1) in holes):
                out["plate_%d_%d" % (a, c)] = box_triangles((a * h, c * h, cfg["plate_z"][0]),
                                                            (b * h, d * h, cfg["plate_z"][1]))
    (x0, y0), (x1, y1) = cfg["post_xy"]
    top = cfg["plate_z"][1]
    out["post"] = box_triangles((x0, y0, top), (x1, y1, top + cfg["gap0"]))
    return out


@ti.data_oriented
class ObstacleFaceMask:
    """Contract 1.2 face validity without air cells: valid iff neither adjacent cell is an obstacle and not the
    domain-edge face (index 0 along its component); reads fluid.obstacle live."""

    def __init__(self, fluid):
        self.obstacle = fluid.obstacle

    def refresh(self, air_cell_field=None):
        return None

    @ti.func
    def face_valid(self, c: ti.template(), i, j, k):
        ii, jj, kk = i, j, k
        if ti.static(c == 0):
            ii = i - 1
        if ti.static(c == 1):
            jj = j - 1
        if ti.static(c == 2):
            kk = k - 1
        valid = 0
        if ii >= 0 and jj >= 0 and kk >= 0:
            if self.obstacle[i, j, k] == 0 and self.obstacle[ii, jj, kk] == 0:
                valid = 1
        return valid


@ti.kernel
def _add_box_force(force: ti.template(), i0: ti.i32, i1: ti.i32, j0: ti.i32, j1: ti.i32, k0: ti.i32, k1: ti.i32,
                   fz: ti.f32):
    for i, j, k in force:
        if i0 <= i < i1 and j0 <= j < j1 and k0 <= k < k1:
            f = force[i, j, k]
            f[2] = f[2] + fz
            force[i, j, k] = f


def flap_points(cfg, spacing):
    ax, ay = cfg["axis"]
    z_low = cfg["plate_z"][1] + cfg["gap0"]
    g = np.arange(-cfg["flap_r"], cfg["flap_r"], spacing) + 0.5 * spacing
    X, Y = np.meshgrid(ax + g, ay + g, indexing="ij")
    r = np.hypot(X - ax, Y - ay)
    keep = r < cfg["flap_r"]
    pts, tags, tether = [], [], []
    for layer in range(int(round(cfg["thick"] / spacing))):
        z = z_low + (layer + 0.5) * spacing
        for x, y, rr in zip(X[keep], Y[keep], r[keep]):
            pts.append((x, y, z))
            tags.append(TAG_VALVE_RIM if (layer == 0 and rr >= cfg["rim_r"]) else 0)
            tether.append(1 if rr < cfg["tether_r"] else 0)
    return np.asarray(pts), np.asarray(tags, np.int32), np.asarray(tether, np.int32)


def contact_log_path(tmp_path, label):
    return tmp_path / ("contact_events_%s.jsonl" % label)


def run_rig(tmp_path, *, log=None, label="a", **overrides):
    """One rig run; returns the per-step rows (contact report + independent checks), events and setup."""
    from cases.squid_setup.closed_solid_cells import classify_closed_solid_cells
    from simulation_core.diagnostics.runtime import TaichiRuntimeConfig
    from simulation_core.fast.fluid_hooks import UniformAxes, replace_obstacle_cells
    from simulation_core.fast.ib_mpm_fast3d import FastIbMpm3D
    from simulation_core.fluids import CartesianFluidSolver, FluidDomainSpec

    cfg = dict(RIG, **overrides)
    h, nodes = cfg["h"], tuple(cfg["nodes"])
    runtime = TaichiRuntimeConfig(arch="cpu", default_fp="f32", random_seed=0, offline_cache=False)
    fluid = CartesianFluidSolver(FluidDomainSpec(bounds_min_m=(0.0, 0.0, 0.0),
                                                 bounds_max_m=tuple(n * h for n in nodes), grid_nodes=nodes,
                                                 density_kgm3=cfg["rho_f"], viscosity_pa_s=cfg["mu_f"], dt_s=1.0e-4),
                                 runtime=runtime, multigrid_semi_coarsening=False)
    fluid.velocity.from_numpy(np.zeros(nodes + (3,), np.float32))
    fluid.velocity_prev.from_numpy(np.zeros(nodes + (3,), np.float32))
    fluid.pressure.from_numpy(np.zeros(nodes, np.float32))
    fluid.clear_volume_source()
    fluid.set_velocity_dirichlet_boundary_authority("canonical")
    rigid = rig_rigid_triangles(cfg)
    obstacle = np.zeros(nodes, bool)
    for tri in rigid.values():                      # the static obstacle from the same triangles (thin walls closed)
        obstacle |= classify_closed_solid_cells(tri, fluid.grid)[0]
    replace_obstacle_cells(fluid, obstacle.astype(np.int32))
    axes = UniformAxes.from_fluid(fluid)
    mask = ObstacleFaceMask(fluid)
    spacing = 0.5 * h
    pts, tags, tether = flap_points(cfg, spacing)
    dt = cfg["cfl"] * h / math.sqrt(cfg["mu_s"] / cfg["rho_s"])
    k_tether = cfg["rho_s"] * (0.25 / dt) ** 2       # the tether convention; the contact uses the same k
    body = FastIbMpm3D(fluid=fluid, axes=axes, mask=mask, positions_m=pts,
                       volumes_m3=np.full(len(pts), spacing ** 3, np.float32), tether_mask=tether,
                       density_kgm3=cfg["rho_s"], shear_modulus_pa=cfg["mu_s"],
                       kappa_pa=cfg["kappa_over_mu"] * cfg["mu_s"], tether_stiffness_pa_per_m=k_tether,
                       body_id=np.ones(len(pts), np.int32), tag=tags)
    (ni0, ni1), (nj0, nj1) = cfg["nozzle"]
    ax, ay = cfg["axis"]
    near, far = [], []
    for (pi0, pi1), (pj0, pj1) in cfg["ports"]:
        far += [math.hypot(i * h - ax, j * h - ay) for i in (pi0, pi1) for j in (pj0, pj1)]
        near.append(math.hypot(min(max(ax, pi0 * h), pi1 * h) - ax, min(max(ay, pj0 * h), pj1 * h) - ay))
    port_r = (min(near), max(far))
    contact = RigidContact(fluid, axes, body, rigid, k_c=k_tether, axis_xy_m=cfg["axis"],
                           land_z_m=cfg["plate_z"][1], land_r_m=(cfg["rim_r"], cfg["flap_r"]), port_r_m=port_r,
                           channel_r_m=None, refill_plane_z_m=cfg["refill_plane_z"], flap_thickness_m=cfg["thick"])
    contact.event_log_path = str(contact_log_path(tmp_path, label))    # as the driver does, after construction
    velocity_old = ti.Vector.field(3, ti.f32, shape=nodes)
    k0, k1 = int(round(cfg["plate_z"][0] / h)), int(round(cfg["plate_z"][1] / h)) + 1
    k_mid = int(round(0.5 * (cfg["plate_z"][0] + cfg["plate_z"][1]) / h)) - 1     # a port face level != refill plane
    cell = h ** 3
    rows, work, t, apply_s = [], 0.0, 0.0, []
    for step in range(1, int(cfg["steps"]) + 1):
        velocity_old.copy_from(fluid.velocity)
        u_old = fluid.velocity.to_numpy().astype(np.float64)
        body.midpoint_predict(dt)
        ti.sync()
        tick = time.perf_counter()
        contact.apply_point_forces(dt)
        ti.sync()
        apply_s.append(time.perf_counter() - tick)
        body.spread_force()
        _add_box_force(fluid.force, ni0, ni1, nj0, nj1, k0, k1, cfg["f_pump"])
        fluid.apply_body_force(dt, read_report=False)
        fluid.predict(dt_s=dt, advection_scheme="muscl_tvd", pressure_outlet_zmin=True, velocity_inlet_zmax=False)
        # 'allow': the z-min outlet passes local in- and outflow (the default 'clamp' zeroes inflow faces after the
        # projection and breaks the volume balance of the k = 0 cells)
        fluid.project(iterations=400, pressure_outlet_zmin=True, hibm_pressure_reachability_prepared=True,
                      velocity_inlet_zmax=False, dt_s=dt, reset_pressure=False, pressure_solver="fv_cg",
                      cg_tolerance=1.0e-6, cg_preconditioner="fv_multigrid", pressure_solve_failure_policy="raise",
                      read_report=False, pressure_outlet_backflow_policy="allow")
        body.midpoint_correct(velocity_old, dt)
        min_d = body.one_sided_report()["min_D"]
        body.reset_one_sided_stats()
        t += dt
        u = fluid.velocity.to_numpy().astype(np.float64)
        pump_faces = (slice(ni0, ni1), slice(nj0, nj1), slice(k0, k1), 2)
        work += dt * cfg["f_pump"] * 0.5 * float(u_old[pump_faces].sum() + u[pump_faces].sum()) * cell
        rep = contact.report(t, dt, p_shift_pa=0.0)
        diag = body.diagnostics()
        row = dict(step=step, t=t, kinetic_j=0.5 * cfg["rho_f"] * float((u * u).sum()) * cell, pump_work_j=work,
                   port_mid=sum(float(u[pi0:pi1, pj0:pj1, k_mid, 2].sum())
                                for (pi0, pi1), (pj0, pj1) in cfg["ports"]) * h * h,
                   nozzle=float(u[ni0:ni1, nj0:nj1, k_mid, 2].sum()) * h * h,
                   outlet=float(u[:, :, 0, 2].sum()) * h * h, min_D=min_d, volume_error=diag["volume_error"],
                   J_min=diag["J_min"], J_max=diag["J_max"], umax=float(np.abs(u).max()))
        row.update({key: rep[key] for key in (
            "contact_force_n", "contact_points", "contact_area_m2", "contact_energy_j", "min_phi_m",
            "max_penetration_m", "rim_gap_min_m", "rim_gap_max_m", "rim_contact_fraction", "refill_flow_m3s",
            "p_above_land_pa", "p_ports_pa", "dp_rim_pa", "closed", "closed_leakage_m3s",
            "closed_leakage_underresolved_m3s", "closed_leakage_resolved_m3s", "events",
            "refill_flow_interval_mean_m3s", "contact_force_interval_mean_n", "interval_steps",
            "refill_flow_rim_underresolved_m3s")})
        row["contact_force_z_n"] = rep["contact_force_vector_n"][2]
        rows.append(row)
        if log is not None:
            log(row)
        if not all(math.isfinite(row[key]) for key in ("kinetic_j", "contact_force_n", "volume_error")):
            break
    contact.close()
    return dict(rows=rows, events=contact.events(), dt=dt, k_c=k_tether, setup=contact.setup_report(), cfg=cfg,
                apply_s_per_step=float(np.median(apply_s[1:])) if len(apply_s) > 1 else None,
                pressure=fluid.pressure.to_numpy().astype(np.float64), obstacle=obstacle,
                positions=body.positions().astype(np.float64), tether=tether, tags=tags, phi=contact.phi.to_numpy(),
                log_path=contact_log_path(tmp_path, label))


# The T8 rig of the tests: four refill ports just inside the land (like the squid's ring of ports), a 1 mm rest gap,
# a 2 mm flap.  The pump step shuts the refill path within ~60 steps; at this load the flap shuts it on a sub-kernel
# fluid film without the material touching (dev runs h2_11_v10; at 4-8x the load the contact engages but carries
# < 1 % of the pressure load on its area, the film the rest: dev runs h2_13_v12/v13).
T8 = dict(ports=(((14, 16), (10, 12)), ((4, 6), (10, 12)), ((9, 11), (15, 17)), ((9, 11), (5, 7))),
          thick=2.0e-3, mu_s=150.0, f_pump=8.0e4, gap0=1.0e-3, steps=150)
DETERMINISM_STEPS = 30


@pytest.fixture(scope="module")
def t8(tmp_path_factory):
    out = run_rig(tmp_path_factory.mktemp("t8"), label="a", **T8)
    rows = out["rows"]
    print("T8_SUMMARY steps %d dt %.4g k_c %.4g apply_point_forces %.3e s/step | refill open peak %.3e end %.3e "
          "| rim gap min %.3e max %.3e | contact points %d force %.3e area %.3e | dp_rim %.1f Pa | max penetration "
          "%.3e | KE %.3e <= W %.3e | volume error %.3e J [%.3f, %.3f]"
          % (len(rows), out["dt"], out["k_c"], out["apply_s_per_step"],
             max(abs(r["refill_flow_m3s"]) for r in rows), rows[-1]["refill_flow_m3s"], rows[-1]["rim_gap_min_m"],
             rows[-1]["rim_gap_max_m"], rows[-1]["contact_points"], rows[-1]["contact_force_n"],
             rows[-1]["contact_area_m2"], rows[-1]["dp_rim_pa"], max(r["max_penetration_m"] for r in rows),
             max(r["kinetic_j"] for r in rows), rows[-1]["pump_work_j"], rows[-1]["volume_error"],
             min(r["J_min"] for r in rows), max(r["J_max"] for r in rows)), flush=True)
    return out


def test_t8_pressure_step_shuts_the_refill_path_with_computed_leakage(t8):
    rows = t8["rows"]
    peak = max(abs(r["refill_flow_m3s"]) for r in rows)
    late = rows[-20:]
    # computed, not forced: the flow through the ports is the solver's face flux, equal at two face levels inside
    # the plate (incompressible ports), and stays nonzero (leakage) after the path shuts
    for r in rows:
        assert r["refill_flow_m3s"] == pytest.approx(r["port_mid"], rel=1e-4, abs=1e-14)
    assert all(abs(r["refill_flow_m3s"]) <= 0.1 * peak for r in late)      # shut: < 10 % of the open-state peak
    assert any(r["refill_flow_m3s"] != 0.0 for r in late)                  # ... but not zero: leakage is reported
    # it shuts with the rim inside the kernel half width of the land (2 cells): an under-resolved opening, and the
    # rim-gap classification attributes the late flow to it
    assert all(r["rim_gap_max_m"] < t8["setup"]["kernel_half_width_m"] for r in late)
    for r in late:
        assert r["refill_flow_rim_underresolved_m3s"] == pytest.approx(r["refill_flow_interval_mean_m3s"],
                                                                       rel=1e-12, abs=1e-18)
    # the pressure difference across the rim builds up once the path shuts (it is the pump that loads the flap)
    assert rows[-1]["dp_rim_pa"] > 4.0 * rows[0]["dp_rim_pa"] > 0.0


def test_t8_no_penetration_and_contact_never_exceeds_the_pressure_load(t8):
    rows = t8["rows"]
    # no point centre ever inside a rigid solid (phi >= 0 for every non-tethered point at every step)
    assert max(r["max_penetration_m"] for r in rows) == 0.0
    assert min(r["min_phi_m"] for r in rows) >= 0.0
    # repulsive only, and never more than the pressure load on the contacting area: F_c <= dp_rim * A_c (in diffuse IB
    # the fluid film carries most of that load; the share is reported, the equality is not expected)
    for r in rows:
        assert r["contact_force_n"] >= 0.0
        assert r["contact_force_n"] <= max(r["dp_rim_pa"], 0.0) * r["contact_area_m2"] + 1e-15
    assert min(r["min_D"] for r in rows) > 0.0                               # no empty one-sided support


def test_t8_energy_bounded_and_fluid_volume_conserved(t8):
    rows = t8["rows"]
    for r in rows:
        # fluid kinetic + contact energy never exceed the work done by the pump (elastic energy >= 0, dissipation >= 0)
        assert r["kinetic_j"] + r["contact_energy_j"] <= r["pump_work_j"] * (1.0 + 1e-6) + 1e-15
        # incompressible: no net outflow through the z-min outlet, to the volume balance the projection's CG
        # tolerance (1e-6 relative residual) allows: 1e-4 of the loop throughput (measured ~2e-5)
        assert abs(r["outlet"]) <= 1e-4 * max(abs(q["nozzle"]) for q in rows)
    assert all(math.isfinite(r["volume_error"]) for r in rows)


def test_t8_rows_are_deterministic(t8, tmp_path_factory):
    again = run_rig(tmp_path_factory.mktemp("t8b"), label="b", **dict(T8, steps=DETERMINISM_STEPS))
    assert len(again["rows"]) == DETERMINISM_STEPS
    for first, second in zip(t8["rows"][:DETERMINISM_STEPS], again["rows"]):
        assert first == second
