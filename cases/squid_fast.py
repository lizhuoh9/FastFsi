"""Squid FAST MODE driver (Claude 2026-10-12; Root REQ_401 step 6) — UNVALIDATED ENGINEERING RUN.

Diffuse IB-MPM single-velocity coupling (simulation_core/fast/ib_mpm_fast3d.py) of the two deformable Ecoflex parts
(region 1 valve flap, region 2 main membrane) in the production CartesianFluidSolver on the production squid graded grid
with Root's D2 membrane band (cases/squid_fast_setup.py); rigid regions 3 and 4 are the production static obstacle with
the user's chamber-wall groove; air chamber (a) = shifted-penalty pressure boundary (air_chamber_fast.py, Helper 1);
PHYSICAL flap / membrane contact with the rigid regions (rigid_contact.py, Helper 2: repulsive normal penalty on a
signed-distance field, no seal assumption, no open/close switch — USER 2026-10-12 「别保留理想的了，因为我想看一下
真实情况是什么样的」); graded IB_4 kernel and face validity (graded_kernel.py, Helper 1).
Contract: fast_mode/p05/INTERFACE_CONTRACT_01.md §6 (§4 superseded by the user's decision).

The solver works with the SHIFTED pressure p~ = p - p_air(t); every reported pressure is PHYSICAL (p = p~ + p_air).
Hard stops: non-finite state, |u| > --u-max, |p| > --p-max, total volume error > --volume-error-max, one-sided empty
support (D = 0).  Speed (Root REQ_404): the per-step calls are device kernels only (no host read or write); the checks,
the advective time-step update (1.25 x the last max|u|) and the history rows run every N = --check-interval steps
(default 20), so a hard stop surfaces at most N steps late (the non-finite check is the backstop).  No resume yet
(Root REQ_401: later).  The label UNVALIDATED is written into every output.

usage: python -B -m cases.squid_fast --source-config CONFIG.json --output DIR --end-time S [--max-steps N]
       [--arch cuda] [--setup-only]   (CONFIG.json: your squid simulation_config.json, not distributed;
       or set SQUID_SOURCE_CONFIG)
"""
import argparse
import csv
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

SQUID_FAST_REVISION = "REQ411-squid-fast-v7-vtk-groove-switch"
LABEL = "UNVALIDATED engineering run (fast mode, diffuse IB-MPM); not validated against the sharp reference"
COLUMNS = ["step", "t_s", "dt_s", "dt_limiter", "p_air_pa", "nozzle_flux_m3s", "air_volume_m3",
           "air_volume_rate_m3s", "membrane_apex_dz_m", "flap_rim_min_centre_gap_m", "rim_contact_fraction",
           "contact_force_n", "refill_flow_m3s", "closed_leakage_m3s", "J_min", "J_max", "volume_error",
           "max_u_mps", "max_p_pa", "min_D", "min_D_free", "cg_iterations",
           # Root 2026-10-12 items 1-4: valve log, rim support, penetration, per-body volume error, flap band lift
           "leak_ratio", "valve_label", "rim_gap_min_m", "rim_gap_max_m", "dp_rim_pa", "max_penetration_m",
           "min_D_rim_free", "volume_error_flap", "volume_error_membrane", "tether_max_disp_flap_m",
           "tether_max_uz_flap_m",
           # Root 2026-10-12 hypre decision: equivalence columns (pressure iterations per step, post-projection divergence)
           "pressure_solver", "pressure_iterations_per_step", "hypre_fallbacks", "div_water_max_1ps", "div_outlet_max_1ps",
           "wall_s_per_step", "unix_s"]


class _Interrupted(BaseException):
    """A termination signal turned into an exception so that final.json records the interruption."""


def _raise_interrupted(signum, frame):
    raise _Interrupted(signal.Signals(signum).name)


def parse(argv):
    from cases import squid_fast_setup as S
    p = argparse.ArgumentParser()
    p.add_argument("--source-config", type=Path, default=S.DEFAULT_SOURCE_CONFIG,
                   required=S.DEFAULT_SOURCE_CONFIG is None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--end-time", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--arch", default="cuda")
    p.add_argument("--cfl-elastic", type=float, default=0.3)
    p.add_argument("--cfl-advective", type=float, default=0.4)
    p.add_argument("--kappa-over-mu", type=float, default=10.0)
    p.add_argument("--band-top", type=float, default=S.MEMBRANE_BAND_TOP_M)
    p.add_argument("--band-dz", type=float, default=S.MEMBRANE_BAND_DZ_M)
    p.add_argument("--grid-variant", choices=sorted(S.GRID_VARIANTS), default="base")
    p.add_argument("--cg-tolerance", type=float, default=1.0e-5)
    p.add_argument("--cg-iterations", type=int, default=400)
    p.add_argument("--cg-preconditioner", choices=("fv_multigrid", "fv_multigrid_light"), default="fv_multigrid")
    p.add_argument("--mg-semi-coarsening", action="store_true")
    p.add_argument("--unroll-static", action="store_true")
    p.add_argument("--no-fast-path", action="store_true")
    p.add_argument("--row-interval", type=int, default=20)
    p.add_argument("--check-interval", type=int, default=20)
    p.add_argument("--u-max", type=float, default=50.0)
    p.add_argument("--p-max", type=float, default=5.0e5)
    p.add_argument("--volume-error-max", type=float, default=0.01)
    p.add_argument("--setup-only", action="store_true")
    p.add_argument("--phase-timing-steps", type=int, default=0)
    p.add_argument("--phase-timing-start", type=int, default=5)
    p.add_argument("--groove", choices=("on", "off"), default="on",
                   help="the user's chamber-wall groove (on: production run 1; off: original CAD wall, T9 A2)")
    p.add_argument("--vtk-points-interval-s", type=float, default=0.0,
                   help="VTK of the two axis planes + the material points every this many seconds (0 = off)")
    p.add_argument("--vtk-3d-interval-s", type=float, default=0.0,
                   help="VTK of the full 3-D fluid every this many seconds (0 = off)")
    p.add_argument("--vtk-compress-level", type=int, default=1)
    p.add_argument("--sdf-cache-dir", type=Path, default=None,
                   help="contact SDF cache directory (create-only files keyed by geometry + grid); default: no cache")
    p.add_argument("--pressure-solver", choices=("cg", "hypre_pfmg"), default="cg")
    p.add_argument("--hypre-prefix", default=None, help="hypre install prefix (default: env HYPRE_PREFIX)")
    p.add_argument("--hypre-data-path", choices=("host", "pointer"), default="host")
    p.add_argument("--snapshot-interval-s", type=float, default=0.0,
                   help="body displacement / J and the two axis planes of the fluid every this many seconds (0 = off)")
    return p.parse_args(argv)


def drop_points_in_obstacle(particles, obstacle, grid):
    """Remove the points whose host fluid cell is a rigid obstacle (membrane rim points inside the steel wall voxels:
    the rigid wall itself clamps the rim; in the single-velocity IB the solid mass does not enter the dynamics)."""
    faces = [np.asarray(f) for f in (grid.cell_faces_x_m, grid.cell_faces_y_m, grid.cell_faces_z_m)]
    p = particles['positions']
    idx = [np.clip(np.searchsorted(faces[a], p[:, a], side='right') - 1, 0, len(faces[a]) - 2) for a in range(3)]
    inside = obstacle[idx[0], idx[1], idx[2]] != 0
    kept = {k: v[~inside] for k, v in particles.items()}
    return kept, dict(dropped_points=int(inside.sum()), dropped_tethered=int(np.count_nonzero(inside & (particles['tether'] != 0))),
                      dropped_by_body={str(b): int(np.count_nonzero(inside & (particles['body_id'] == b))) for b in (1, 2)})


def p_air_schedule(spec):
    """Piecewise-linear chamber pressure of the source config (spec.pressure_t*_s / pressure_p*_pa)."""
    times = (float(spec.pressure_t0_s), float(spec.pressure_t1_s), float(spec.pressure_t2_s))
    values = (float(spec.pressure_p0_pa), float(spec.pressure_p1_pa), float(spec.pressure_p2_pa))
    return lambda t: float(np.interp(float(t), times, values))


def proc_memory():
    """(VmRSS, VmHWM) of this process in bytes from /proc/self/status (Linux); NaN where unavailable."""
    values = {}
    try:
        with open('/proc/self/status') as stream:
            for line in stream:
                if line.startswith(('VmRSS:', 'VmHWM:')):
                    key, value = line.split(':')
                    values[key] = int(value.split()[0]) * 1024
    except OSError:
        pass
    return float(values.get('VmRSS', float('nan'))), float(values.get('VmHWM', float('nan')))


class SetupPhases:
    """Wall time, RSS and peak RSS (VmHWM) after each setup phase (Root 2026-10-12, after the 16 GB watchdog stop):
    printed as it goes (the monitor sees where a setup stands) and written to setup_report.json."""

    def __init__(self):
        self.t0 = self.last = time.perf_counter()
        self.rows = []

    def mark(self, name):
        now = time.perf_counter()
        rss, peak = proc_memory()
        self.rows.append(dict(phase=name, wall_s=round(now - self.last, 3), t_s=round(now - self.t0, 3),
                              rss_bytes=rss, peak_rss_bytes=peak))
        print("SQUID_FAST setup %-16s %7.1f s (+%.1f) rss %.2f GB peak %.2f GB"
              % (name, now - self.t0, now - self.last, rss / 1e9, peak / 1e9), flush=True)
        self.last = now


def build(args):
    """Host setup + fluid + body (+ helper modules unless --setup-only).  Returns a namespace dict."""
    from cases import squid_fast_setup as S
    from simulation_core.diagnostics.runtime import TaichiRuntimeConfig, init_taichi
    t0 = time.perf_counter()
    phases = SetupPhases()
    inputs = S.load_squid_inputs(args.source_config, band_top_m=args.band_top, band_dz_m=args.band_dz,
                                 grid_variant=args.grid_variant)
    phases.mark('inputs')
    obstacle, obstacle_report = S.static_obstacle(inputs, groove=(args.groove == "on"))
    phases.mark('obstacle')
    particles, particle_report = S.build_fast_particles(inputs)
    particles, drop_report = drop_points_in_obstacle(particles, obstacle, inputs['grid'])
    phases.mark('particles')
    runtime = TaichiRuntimeConfig(arch=args.arch, strict_arch=(args.arch == "cuda"), default_fp="f32", random_seed=0,
                                  offline_cache=False)
    init_taichi(runtime)
    phases.mark('taichi_init')
    material = inputs['config']['materials'][0]['properties']
    mu, rho_s = float(material['mu_Pa']), float(material['rho_kgm3'])
    grid_rep = S.grid_report(inputs['grid'])
    h_min = min(grid_rep['min_spacing_m'])
    c_s = math.sqrt(mu / rho_s)
    dt_elastic = args.cfl_elastic * h_min / c_s
    fluid, fluid_report = S.build_fluid(inputs, obstacle, runtime, dt_elastic,
                                        multigrid_semi_coarsening=bool(args.mg_semi_coarsening))
    phases.mark('fluid')
    report = dict(revision=SQUID_FAST_REVISION, label=LABEL, grid=grid_rep, obstacle_cells=int(obstacle.sum()),
                  obstacle={k: v for k, v in obstacle_report.items() if isinstance(v, (int, float, str, bool))},
                  particles=particle_report, dropped=drop_report, fluid=fluid_report, mu_pa=mu, rho_s_kgm3=rho_s,
                  kappa_pa=args.kappa_over_mu * mu, c_s_mps=c_s, h_min_m=h_min, dt_elastic_s=dt_elastic,
                  setup_wall_s=time.perf_counter() - t0, args={k: str(v) for k, v in vars(args).items()},
                  geometry_note=geometry_note(obstacle_report),
                  groove=(dict(r_m=list(obstacle_report['groove_r_m']), z_m=list(obstacle_report['groove_z_m']),
                               carved_cells=int(obstacle_report['groove_carved_cells']))
                          if 'groove_r_m' in obstacle_report else None))
    ns = dict(inputs=inputs, obstacle=obstacle, particles=particles, fluid=fluid, report=report, mu=mu, rho_s=rho_s,
              dt_elastic=dt_elastic, h_min=h_min, phases=phases)
    report['setup_phases'] = phases.rows
    if args.setup_only:
        return ns
    from cases.squid_setup.setup import _region_triangles_m
    from simulation_core.fast.air_chamber_fast import AirChamberFast
    from simulation_core.fast.graded_kernel import FaceMask, GradedAxes
    from simulation_core.fast.ib_mpm_fast3d import FastIbMpm3D
    from simulation_core.fast.rigid_contact import RigidContact
    grid = inputs['grid']
    axes = GradedAxes(np.asarray(grid.cell_faces_x_m), np.asarray(grid.cell_faces_y_m), np.asarray(grid.cell_faces_z_m))
    mask = FaceMask(fluid, axes)
    phases.mark('axes_mask')
    tether_k = rho_s * (0.25 / dt_elastic) ** 2
    body = FastIbMpm3D(fluid=fluid, axes=axes, mask=mask, positions_m=particles['positions'],
                       volumes_m3=particles['volumes'], tether_mask=particles['tether'], density_kgm3=rho_s,
                       shear_modulus_pa=mu, kappa_pa=args.kappa_over_mu * mu, tether_stiffness_pa_per_m=tether_k,
                       body_id=particles['body_id'], tag=particles['tag'], fast_path=not args.no_fast_path,
                       unroll_static=bool(args.unroll_static))
    phases.mark('body')
    air = AirChamberFast(fluid, axes, inputs['air_chamber'], body)
    phases.mark('air_chamber')
    analysis = inputs['config'].get('analysis_settings', {})
    rigid_ids = tuple(int(v) for v in analysis.get('solid_obstacle_surface_only_region_ids', (3, 4)))
    rigid_triangles = _region_triangles_m(inputs['config'], rigid_ids)
    contact = RigidContact(fluid, axes, body, rigid_triangles, k_c=tether_k,
                           axis_xy_m=tuple(inputs['air_chamber'].axis_xy_m), land_z_m=1.007000,
                           sdf_cache_dir=args.sdf_cache_dir)
    phases.mark('contact')
    report.update(tether_k_pa_per_m=tether_k, points=int(body.n), rigid_contact_regions=list(rigid_ids))
    # contact distance of the rim on the land (normal along z: the vertical half extent of the rim material boxes)
    d_c_rim = float(np.max(contact.rim_half.to_numpy()[:contact.rim_count])) if contact.rim_count else float('nan')
    report.update(d_c_rim_m=d_c_rim, rim_points=int(contact.rim_count))
    # consistent initial masks (the loop refreshes them every step): air cells at t = 0, face validity, support flags
    air.update(0.0, p_air_schedule(inputs['spec'])(0.0))
    mask.refresh(air.air_cell)
    body.refresh_support_flags()
    phases.mark('initial_masks')
    ns.update(axes=axes, mask=mask, body=body, air=air, contact=contact, d_c_rim_m=d_c_rim)
    # per-body share of the FREE (non-tethered) volume: the total volume-error stop (REQ_382) is over the free points of
    # both bodies, so a body's share says how far its own error moves the total (Root 2026-10-12, item 4 correction)
    d0 = body.diagnostics(rim_land_z_m=1.007000)
    free_total = d0['free_volume_flap_m3'] + d0['free_volume_membrane_m3']
    report.update(free_volume_shares=dict(
        free_volume_flap_m3=d0['free_volume_flap_m3'], free_volume_membrane_m3=d0['free_volume_membrane_m3'],
        flap_share=d0['free_volume_flap_m3'] / free_total if free_total > 0 else float('nan'),
        membrane_share=d0['free_volume_membrane_m3'] / free_total if free_total > 0 else float('nan'),
        note="total volume error = share-weighted sum of the per-body errors"))
    if args.pressure_solver == "hypre_pfmg":
        # Root REQ_407: hypre Struct PCG + PFMG replaces ONLY the linear solve of project() on this fluid instance
        from simulation_core.fast.hypre_pressure import install_hypre_pressure_hook
        ns['hypre'] = install_hypre_pressure_hook(fluid, prefix=args.hypre_prefix, data_path=args.hypre_data_path)
        report.update(pressure_solver=dict(name="hypre_pfmg", authority="Root REQ_411", data_path=ns['hypre'].data_path,
                                           data_path_note=ns['hypre'].data_path_note,
                                           options=dict(ns['hypre'].options)))
        phases.mark('hypre_hook')
    else:
        report.update(pressure_solver=dict(name="cg"))
    return ns


def make_diagnostics(fluid, grid, spec, air_cell):
    """Taichi reductions over the fluid: nozzle exit flux, max |u| and max |p| (physical) over water cells,
    non-finite counts.  air_cell may be None (then every non-obstacle cell is water)."""
    import taichi as ti
    nx, ny, nz = (int(v) for v in grid.grid_nodes)
    faces = [np.asarray(f) for f in (grid.cell_faces_x_m, grid.cell_faces_y_m, grid.cell_faces_z_m)]
    centres = [0.5 * (f[1:] + f[:-1]) for f in faces]
    k_exit = int(np.argmin(np.abs(faces[2] - float(spec.lip_z_m))))
    cx, cy = float(spec.monitor_center_x_m), float(spec.monitor_center_y_m)
    lumen = float(spec.nozzle_radius_m) + 1.0e-4
    exit_area = np.zeros((nx, ny), np.float32)
    rr = np.hypot(centres[0][:, None] - cx, centres[1][None, :] - cy)
    exit_area[rr <= lumen] = (np.diff(faces[0])[:, None] * np.diff(faces[1])[None, :])[rr <= lumen]
    area = ti.field(ti.f32, shape=(nx, ny))
    area.from_numpy(exit_area)
    out = ti.field(ti.f64, shape=4)
    air = air_cell if air_cell is not None else ti.field(ti.i32, shape=(nx, ny, nz))

    @ti.kernel
    def reduce_kernel(p_shift: ti.f32):
        for s in range(4):
            out[s] = 0.0
        for i, j in area:
            if area[i, j] > 0.0:
                ti.atomic_add(out[0], ti.cast(fluid.velocity[i, j, k_exit][2] * area[i, j], ti.f64))
        for i, j, k in fluid.velocity:
            if fluid.obstacle[i, j, k] == 0 and air[i, j, k] == 0:
                u = fluid.velocity[i, j, k]
                speed = ti.cast(u.norm(), ti.f64)
                pressure = ti.cast(fluid.pressure[i, j, k] + p_shift, ti.f64)
                if ti.math.isnan(speed) or ti.math.isinf(speed) or ti.math.isnan(pressure) or ti.math.isinf(pressure):
                    ti.atomic_add(out[3], 1.0)
                else:
                    ti.atomic_max(out[1], speed)
                    ti.atomic_max(out[2], ti.abs(pressure))

    def measure(p_shift_pa):
        reduce_kernel(float(p_shift_pa))
        r = out.to_numpy()
        return dict(nozzle_flux_m3s=float(r[0]), max_u_mps=float(r[1]), max_p_pa=float(r[2]), nonfinite_cells=int(r[3]))
    return measure, dict(k_exit=k_exit, exit_plane_z_m=float(faces[2][k_exit]), exit_capture_area_m2=float(exit_area.sum()))


def hard_stop(code, witness):
    from simulation_core.fast.ib_mpm_fast3d import FastModeHardStop
    raise FastModeHardStop(code, witness)


def write_new(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=1, default=str)
        stream.write(chr(10))


class PhaseTimer:
    """Diagnostic instrument (default OFF, --phase-timing-steps K): for steps start+1 .. start+K every phase is wrapped
    by ti.sync() + perf_counter, and Taichi kernel calls are counted per phase (a wrapper on Kernel.__call__, removed
    afterwards) — the launches-per-step proxy of the smoke report (Root REQ_395/404).  It perturbs the timing of the
    instrumented steps only (the syncs), never the physics."""

    def __init__(self, start, steps):
        self.start, self.steps = int(start), int(steps)
        self.active = False
        self.times, self.calls, self.counted_steps = {}, {}, 0
        self._count, self._restore = None, None

    def begin(self, step):
        want = self.steps > 0 and self.start < step <= self.start + self.steps
        if want and not self.active:
            from taichi.lang import kernel_impl
            original = kernel_impl.Kernel.__call__
            count = {'n': 0}

            def counted(kernel, *args, **kwargs):
                count['n'] += 1
                return original(kernel, *args, **kwargs)
            kernel_impl.Kernel.__call__ = counted
            self._count = count
            self._restore = lambda: setattr(kernel_impl.Kernel, '__call__', original)
            self.active = True
        elif not want and self.active:
            self._restore()
            self.active = False
        if self.active:
            self.counted_steps += 1

    def phase(self, name, fn, *args, **kwargs):
        if not self.active:
            return fn(*args, **kwargs)
        import taichi as ti
        ti.sync()
        n0, t0 = self._count['n'], time.perf_counter()
        result = fn(*args, **kwargs)
        ti.sync()
        self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - t0
        self.calls[name] = self.calls.get(name, 0) + self._count['n'] - n0
        return result

    def report(self):
        k = max(self.counted_steps, 1)
        rows = {name: dict(ms_per_step=1e3 * self.times[name] / k, kernel_calls_per_step=self.calls[name] / k)
                for name in self.times}
        return dict(steps=self.counted_steps, phases=rows,
                    total_ms_per_step=sum(r['ms_per_step'] for r in rows.values()),
                    total_kernel_calls_per_step=sum(r['kernel_calls_per_step'] for r in rows.values()))


VALVE_EDGE_R_M, CAD_WALL_R_M = 0.030, 0.032   # valve flap outer edge and CAD chamber wall radii (VALVE_FACTS_01)


def geometry_note(obstacle_report):
    """Root REQ_410: the user's chamber-wall groove (kept in run 1) widens the refill annulus; one sentence for every
    output, numbers from the groove report of the static obstacle."""
    if 'groove_r_m' not in obstacle_report:
        return ("no groove: the original CAD chamber wall (refill annulus %.1f mm between the valve edge r %.0f mm and "
                "the wall r %.0f mm)" % ((CAD_WALL_R_M - VALVE_EDGE_R_M) * 1e3, VALVE_EDGE_R_M * 1e3, CAD_WALL_R_M * 1e3))
    r0, r1 = (float(v) for v in obstacle_report['groove_r_m'])
    z0, z1 = (float(v) for v in obstacle_report['groove_z_m'])
    return ("chamber-wall groove r %.1f -> %.1f mm, z %.1f - %.1f mm (%d cells; user's 2026-10-03 sharp-path "
            "modification, kept in this run): refill annulus %.1f mm vs CAD %.1f mm, so the refill path is wider than "
            "the real device" % (r0 * 1e3, r1 * 1e3, z0 * 1e3, z1 * 1e3, int(obstacle_report['groove_carved_cells']),
                                 (r1 - VALVE_EDGE_R_M) * 1e3, (CAD_WALL_R_M - VALVE_EDGE_R_M) * 1e3))


ADVECTIVE_SAFETY = 1.25   # max|u| may grow between checks (N steps); the advective dt uses 1.25 x the last measurement
# Root 2026-10-12 (decisions on Helper 2's points; reporting only, no physics change):
VALVE_CLOSURE_LEAK_RATIO = 0.02   # item 1: "port flow ~ 0" = refill-path flow <= 2 % of refill-path + nozzle flow
VALVE_FLOW_FLOOR_M3S = 1.0e-9     # below this total flow there is nothing to classify ("no_flow")
RIM_SUPPORT_STOP = 0.1            # item 2: free rim-point min D below this while the rim is in contact -> stop, report
PENETRATION_REPORT_DC = 2.0       # item 3: max penetration above 2 d_c(rim) -> report (event line), no stop


DIV_OUTLET_REPORT_1PS = 1.0e-3    # Root REQ_411: report div_outlet_max_1ps above 1e-3 1/s ...
DIV_OUTLET_OVER_WATER = 10.0      # ... or above 10 x the water-divergence max


def run_label(args, ns):
    """The tag of every output file: UNVALIDATED engineering run, tree, grid, solver (Root REQ_408/409)."""
    grid = ns['inputs']['grid']
    nodes = "x".join(str(int(v)) for v in grid.grid_nodes)
    solver = "hypre_pfmg (Root REQ_411)" if args.pressure_solver == "hypre_pfmg" else "cg"
    return ("UNVALIDATED engineering run, %s, grid %s %s (graded + D2), groove %s, solver %s, %s"
            % (Path(__file__).resolve().parents[1].name, args.grid_variant, nodes, args.groove, solver,
               SQUID_FAST_REVISION))


def make_vtk_writer(args, ns, directory):
    """Helper 2's VTK series writer (REQ_409/410) on this run's grid, axis planes and material points; returns the
    writer and the static obstacle (rigid regions + groove as fluid.obstacle; the air chamber is a separate mask)."""
    from simulation_core.fast.vtk_export_fast import VtkSeriesWriter
    grid, body = ns['inputs']['grid'], ns['body']
    i, j = _snapshot_planes(ns)
    writer = VtkSeriesWriter(directory, faces_x=np.asarray(grid.cell_faces_x_m), faces_y=np.asarray(grid.cell_faces_y_m),
                             faces_z=np.asarray(grid.cell_faces_z_m), plane_i=i, plane_j=j, X0=body.X0.to_numpy(),
                             body_id=body.body_id.to_numpy(), tag=body.tag.to_numpy(), tether=body.tether.to_numpy(),
                             compress=True, compress_level=int(args.vtk_compress_level), label=run_label(args, ns),
                             geometry_note=ns['report'].get('geometry_note', ''))
    return writer, ns['fluid'].obstacle.to_numpy()


def _num(value):
    return float('nan') if value is None else float(value)


def valve_label(q_refill, q_nozzle, rim_contact_fraction):
    """Item 1: classify the valve state of a row.  closure = the refill path carries <= 2 % of the flow; then
    'contact_closure' when every rim point is within d_c of the land (rim gap <= d_c all around), otherwise the
    closure rests on a water film over the open part of the rim: 'film_closure_numerical_suspected' (no rim point in
    contact) or 'partial_contact_closure_film_numerical_suspected'.  T9 (two grids) separates film from physics."""
    total = abs(q_refill) + abs(q_nozzle)
    if not total >= VALVE_FLOW_FLOOR_M3S:
        return float('nan'), 'no_flow'
    ratio = abs(q_refill) / total
    if ratio > VALVE_CLOSURE_LEAK_RATIO:
        return ratio, 'open'
    if rim_contact_fraction >= 1.0:
        return ratio, 'contact_closure'
    if rim_contact_fraction > 0.0:
        return ratio, 'partial_contact_closure_film_numerical_suspected'
    return ratio, 'film_closure_numerical_suspected'


def _snapshot_planes(ns):
    """Cell indices (i, j) of the vertical planes x = axis_x (yz plane) and y = axis_y (xz plane)."""
    grid, axis = ns['inputs']['grid'], ns['inputs']['air_chamber'].axis_xy_m
    fx, fy = np.asarray(grid.cell_faces_x_m), np.asarray(grid.cell_faces_y_m)
    i = int(np.clip(np.searchsorted(fx, float(axis[0]), side='right') - 1, 0, len(fx) - 2))
    j = int(np.clip(np.searchsorted(fy, float(axis[1]), side='right') - 1, 0, len(fy) - 2))
    return i, j


def write_snapshot_static(ns, directory):
    """Once: reference positions and point attributes, the grid faces, the plane indices and the static obstacle on the
    two planes (create-only)."""
    body, fluid, grid = ns['body'], ns['fluid'], ns['inputs']['grid']
    i, j = _snapshot_planes(ns)
    obstacle = fluid.obstacle.to_numpy()
    with (directory / 'static.npz').open('xb') as stream:
        np.savez(stream, X0=body.X0.to_numpy(), body_id=body.body_id.to_numpy(), tag=body.tag.to_numpy(),
                 tether=body.tether.to_numpy(), vol=body.vol.to_numpy(), faces_x=np.asarray(grid.cell_faces_x_m),
                 faces_y=np.asarray(grid.cell_faces_y_m), faces_z=np.asarray(grid.cell_faces_z_m), plane_i=i, plane_j=j,
                 obstacle_xz=obstacle[:, j, :], obstacle_yz=obstacle[i, :, :], label=LABEL,
                 geometry_note=ns['report'].get('geometry_note', ''))
    return i, j


def write_snapshot(ns, directory, index, step, t, p_shift_pa, planes):
    """Body displacement u (f32, x = X0 + u) and J per point; fluid velocity and PHYSICAL pressure (p~ + p_shift) and
    the air cells on the two axis planes (create-only; diagnostics output, never read back)."""
    body, fluid, air = ns['body'], ns['fluid'], ns['air']
    i, j = planes
    J = np.linalg.det(body.F.to_numpy().astype(np.float64)).astype(np.float32)
    velocity = fluid.velocity.to_numpy()
    pressure = fluid.pressure.to_numpy() + np.float32(p_shift_pa)
    air_cell = air.air_cell.to_numpy()
    with (directory / ('snap_%05d.npz' % index)).open('xb') as stream:
        np.savez(stream, step=step, t_s=t, p_shift_pa=p_shift_pa, u=body.u.to_numpy(), J=J,
                 vel_xz=velocity[:, j, :, :], vel_yz=velocity[i, :, :, :], p_xz=pressure[:, j, :],
                 p_yz=pressure[i, :, :], air_xz=air_cell[:, j, :].astype(np.int8),
                 air_yz=air_cell[i, :, :].astype(np.int8))


def _check(args, ns, measure, p_shift_pa, t, step):
    """Device diagnostics read every N steps (REQ_404): one fluid reduction, the body's one-sided report (D = 0 hard
    stop, surfaced up to N steps late; the non-finite check is the backstop) and the body reductions.  p_shift_pa is the
    air pressure the last projection was solved with (p_air at the step midpoint), so reported pressures are physical."""
    fl = measure(p_shift_pa)
    one_sided = ns['body'].one_sided_report()
    if one_sided['min_D_free'] < 0.25:                      # Root: say where (near the air face or a wall)
        witness = dict(step=step, t_s=t, **ns['body'].min_D_witness(free_only=True))
        with (ns['out'] / "min_D_free_events.jsonl").open("a") as stream:
            stream.write(json.dumps(witness) + chr(10))
        print("SQUID_FAST min_D_free %.3f below 0.25 at %s" % (one_sided['min_D_free'], witness), flush=True)
    d = ns['body'].diagnostics(rim_land_z_m=1.007000)
    if d['nonfinite_points'] or fl['nonfinite_cells']:
        hard_stop('nonfinite_state', dict(step=step, t_s=t, points=d['nonfinite_points'], cells=fl['nonfinite_cells']))
    if fl['max_u_mps'] > args.u_max:
        hard_stop('speed_bound', dict(step=step, t_s=t, max_u_mps=fl['max_u_mps'], bound=args.u_max))
    if fl['max_p_pa'] > args.p_max:
        hard_stop('pressure_bound', dict(step=step, t_s=t, max_p_pa=fl['max_p_pa'], bound=args.p_max))
    if abs(d['volume_error']) > args.volume_error_max:
        hard_stop('total_volume_error', dict(step=step, t_s=t, volume_error=d['volume_error'],
                                             bound=args.volume_error_max))
    return fl, one_sided, d


def run(args, ns, out):
    """The time loop (contract §6; Root REQ_404: no host access on the hot path — the per-step calls are device
    kernels only; checks, the advective dt update and history rows every N = --check-interval steps).
    Returns (status, step, t)."""
    import taichi as ti
    fluid, body, air, contact, mask = ns['fluid'], ns['body'], ns['air'], ns['contact'], ns['mask']
    inputs, h_min = ns['inputs'], ns['h_min']
    ns['out'] = out
    if args.row_interval % args.check_interval:
        raise ValueError("--row-interval must be a multiple of --check-interval")
    p_air = p_air_schedule(inputs['spec'])
    measure, diag_report = make_diagnostics(fluid, inputs['grid'], inputs['spec'], getattr(air, 'air_cell', None))
    ns['report']['diagnostics'] = dict(diag_report, check_interval_steps=args.check_interval,
                                       advective_safety=ADVECTIVE_SAFETY)
    velocity_old = ti.Vector.field(3, ti.f32, shape=tuple(int(v) for v in inputs['grid'].grid_nodes))

    @ti.kernel
    def begin_step():
        # fused: save u^n for the midpoint correction and clear the IB force field in one grid pass
        for i, j, k in velocity_old:
            velocity_old[i, j, k] = fluid.velocity[i, j, k]
            fluid.force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

    stream = (out / "history.csv").open("x", newline="")
    writer = csv.writer(stream)
    writer.writerow(COLUMNS)
    timer = PhaseTimer(args.phase_timing_start, args.phase_timing_steps)
    # projection arguments: the driver's plus the air chamber's requirements (shifted-penalty boundary: z-min outlet
    # backflow 'allow', no pressure accumulation); overlapping keys must agree
    project_kwargs = dict(iterations=args.cg_iterations, pressure_outlet_zmin=True,
                          hibm_pressure_reachability_prepared=True, reset_pressure=False, pressure_solver="fv_cg",
                          cg_tolerance=args.cg_tolerance, cg_preconditioner=args.cg_preconditioner,
                          pressure_solve_failure_policy="raise", read_report=False)
    mg_family = {"fv_multigrid", "fv_multigrid_light"}      # the air chamber needs the MG preconditioner ON
    for key, value in getattr(sys.modules[type(air).__module__], 'PROJECT_REQUIREMENTS', {}).items():
        if key == 'cg_preconditioner' and value in mg_family and project_kwargs[key] in mg_family:
            continue
        if key in project_kwargs and project_kwargs[key] != value:
            raise ValueError("air chamber requires %s=%r, driver has %r" % (key, value, project_kwargs[key]))
        project_kwargs[key] = value
    ns['report']['project_kwargs'] = {k: str(v) for k, v in project_kwargs.items()}
    write_new(out / "run_meta.json", dict(
        revision=SQUID_FAST_REVISION, label=LABEL, grid_variant=args.grid_variant,
        project_kwargs={k: str(v) for k, v in project_kwargs.items()},
        mg=dict(cg_preconditioner=args.cg_preconditioner, semi_coarsening=bool(args.mg_semi_coarsening)),
        air_chamber=dict(module=type(air).__module__, penalty_factor=getattr(air, 'penalty_factor', None),
                         shifted_pressure=("solver solves p~ = p - p_air(t_mid) of each step; every reported pressure "
                                           "is physical (p_air_pa column = schedule value at the row time t)")),
        contact=dict(module=type(contact).__module__, seal_assumption=False, d_c_rim_m=ns['d_c_rim_m']),
        valve_log=dict(rule="Root 2026-10-12 items 1-4 (reporting only, no physics change)",
                       closure_leak_ratio_max=VALVE_CLOSURE_LEAK_RATIO, flow_floor_m3s=VALVE_FLOW_FLOOR_M3S,
                       leak_ratio="|q_refill| / (|q_refill| + |q_nozzle|)",
                       labels=["no_flow", "open", "contact_closure", "partial_contact_closure_film_numerical_suspected",
                               "film_closure_numerical_suspected"],
                       rim_support_stop_min_D=RIM_SUPPORT_STOP,
                       penetration_report_threshold_m=PENETRATION_REPORT_DC * ns['d_c_rim_m'],
                       volume_error_stop="total over free points (REQ_382, 1 %); per-body columns attribute it"),
        snapshots=dict(interval_s=args.snapshot_interval_s,
                       content="u, J per point; fluid velocity, physical pressure, air cells on the two axis planes"),
        body=dict(revision=getattr(sys.modules[type(body).__module__], 'FAST3D_REVISION', None),
                  fast_path=bool(getattr(body, 'fast_path', False)), unroll_static=bool(getattr(body, 'unroll_static', False)),
                  points=int(body.n)),
        checks=dict(interval_steps=args.check_interval, advective_safety=ADVECTIVE_SAFETY, u_max_mps=args.u_max,
                    p_max_pa=args.p_max, volume_error_max=args.volume_error_max),
        geometry_note=ns['report'].get('geometry_note'),
        run_label=run_label(args, ns),
        vtk=dict(points_interval_s=args.vtk_points_interval_s, fluid_3d_interval_s=args.vtk_3d_interval_s,
                 compress_level=args.vtk_compress_level, directory="vtk",
                 note="air is its own mask; obstacle = fluid.obstacle (rigid regions + groove when on)"),
        free_volume_shares=ns['report'].get('free_volume_shares'),
        pressure_solver=ns['report'].get('pressure_solver'),
        reachability_flood=dict(
            batch_levels=ns['report'].get('fluid', {}).get('reachability_flood_batch_levels'),
            bfs_levels=ns['report'].get('fluid', {}).get('reachability_levels'),
            rule=("setup outlet-reachability flood on the production single-level path (fluid_hooks."
                  "single_level_reachability_flood): same BFS; the F2(a) 16-level batch kernel's CUDA compile exceeded "
                  "the 16 GB host cap on the RTX 3080 (Root 2026-10-12: permanent fast-mode default)")),
        setup_phases=ns['report'].get('setup_phases'),
        dt_elastic_s=ns['dt_elastic'], h_min_m=h_min, args={k: str(v) for k, v in vars(args).items()}))
    t, step, dt, limiter = 0.0, 0, ns['dt_elastic'], "elastic"
    last_air, last_wall, last_row_step = None, time.perf_counter(), 0
    valve_stream = (out / "valve_log.jsonl").open("x")       # one line per valve-label change (Root item 1)
    events = dict(stream=None)                                # penetration / rim-support lines, created on first use

    def event(row):
        if events['stream'] is None:
            events['stream'] = (out / "valve_events.jsonl").open("x")
        events['stream'].write(json.dumps(row) + chr(10))
        events['stream'].flush()
    last_label, penetration_high = None, False
    hook = ns.get('hypre')
    last_hypre_iterations, last_iteration_step = 0, 0
    snap_dir, snap_index, next_snap_t, planes = None, 0, 0.0, None
    if args.snapshot_interval_s > 0.0:
        snap_dir = out / "snapshots"
        snap_dir.mkdir(exist_ok=False)
        planes = write_snapshot_static(ns, snap_dir)
    vtk, next_vtk_points_t, next_vtk_3d_t = None, 0.0, 0.0
    if args.vtk_points_interval_s > 0.0 or args.vtk_3d_interval_s > 0.0:
        vtk, static_obstacle = make_vtk_writer(args, ns, out / "vtk")
    solver_events = dict(stream=None, high=False)
    try:
        while t < args.end_time - 1.0e-12 and (args.max_steps <= 0 or step < args.max_steps):
            dt_step = min(dt, args.end_time - t)
            p_mid = p_air(t + 0.5 * dt_step)
            timer.begin(step + 1)
            timer.phase('air_update', air.update, t + 0.5 * dt_step, p_mid)
            timer.phase('mask_refresh', mask.refresh, air.air_cell)
            timer.phase('support_flags', body.refresh_support_flags)
            timer.phase('begin_step', begin_step)
            timer.phase('midpoint_predict', body.midpoint_predict, dt_step)
            timer.phase('contact', contact.apply_point_forces, dt_step)
            timer.phase('spread', body.spread_force, clear=False)
            timer.phase('apply_body_force', fluid.apply_body_force, dt_step, read_report=False)
            timer.phase('fluid_predict', fluid.predict, dt_s=dt_step, advection_scheme="muscl_tvd",
                        pressure_outlet_zmin=True)
            timer.phase('fluid_project', fluid.project, dt_s=dt_step, **project_kwargs)
            timer.phase('air_after_project', air.after_project, dt_step, p_mid)
            timer.phase('midpoint_correct', body.midpoint_correct, velocity_old, dt_step)
            t += dt_step
            step += 1
            if step == 1:
                ti.sync()
                first = dict(first_step_wall_s=time.perf_counter() - last_wall, unix_s=time.time(),
                             note='includes the JIT compile of every kernel of the step (one sync, step 1 only)')
                write_new(out / "first_step.json", first)
                print("SQUID_FAST first_step_wall_s %.2f" % first['first_step_wall_s'], flush=True)
                last_wall, last_row_step = time.perf_counter(), 1
            if timer.steps and step == timer.start + timer.steps:
                timer.begin(step + 1)
                write_new(out / "phase_timing.json", timer.report())
            last = not (t < args.end_time - 1.0e-12 and (args.max_steps <= 0 or step < args.max_steps))
            if step % args.check_interval and not last:
                continue
            fl, one_sided, d = _check(args, ns, measure, p_mid, t, step)
            if snap_dir is not None and t >= next_snap_t - 1.0e-12:
                write_snapshot(ns, snap_dir, snap_index, step, t, p_mid, planes)
                snap_index += 1
                next_snap_t = t + args.snapshot_interval_s
            air_rep = air.report(divergence=True)                  # + max |div + s| (water, outlet), one more read
            due_points = vtk is not None and args.vtk_points_interval_s > 0.0 and t >= next_vtk_points_t - 1.0e-12
            due_3d = vtk is not None and args.vtk_3d_interval_s > 0.0 and t >= next_vtk_3d_t - 1.0e-12
            if due_points or due_3d:                               # host copies only at these diagnostics points
                velocity_h = fluid.velocity.to_numpy()
                pressure_h = fluid.pressure.to_numpy() + p_mid     # physical pressure (p~ + p_air at the midpoint)
                air_h = air.air_cell.to_numpy()
                if due_3d:
                    vtk.write_fluid_3d(step, t, velocity_h, pressure_h, static_obstacle, air_h)
                    next_vtk_3d_t = t + args.vtk_3d_interval_s
                if due_points:
                    vtk.write_planes(step, t, velocity_h, pressure_h, static_obstacle, air_h)
                    J_h = np.linalg.det(body.F.to_numpy().astype(np.float64)).astype(np.float32)
                    vtk.write_points(step, t, body.u.to_numpy(), J_h)
                    next_vtk_points_t = t + args.vtk_points_interval_s
                del velocity_h, pressure_h, air_h
            dt_adv = args.cfl_advective * h_min / max(ADVECTIVE_SAFETY * fl['max_u_mps'], 1.0e-9)
            dt = min(ns['dt_elastic'], dt_adv)
            limiter = "elastic" if dt == ns['dt_elastic'] else "advective"
            if step % args.row_interval and not last:
                continue
            air_volume = float(air_rep.get('air_volume_m3', float('nan')))
            rate = (air_volume - last_air[0]) / (t - last_air[1]) if last_air else float('nan')
            last_air = (air_volume, t)
            c = contact.report(t, dt_step, p_shift_pa=p_mid)
            fraction = float(c.get('rim_contact_fraction') or 0.0)
            q_refill = float(c.get('refill_flow_m3s', float('nan')))
            ratio, label = valve_label(q_refill, fl['nozzle_flux_m3s'], fraction)
            rim_d = float(one_sided.get('min_D_rim_free', float('nan')))
            penetration = float(c.get('max_penetration_m', float('nan')))
            facts = dict(step=step, t_s=t, label=label, leak_ratio=ratio, refill_flow_m3s=q_refill,
                         nozzle_flux_m3s=fl['nozzle_flux_m3s'], rim_contact_fraction=fraction,
                         rim_gap_min_m=c.get('rim_gap_min_m'), rim_gap_max_m=c.get('rim_gap_max_m'),
                         rim_gap_mean_m=c.get('rim_gap_mean_m'), dp_rim_pa=c.get('dp_rim_pa'),
                         min_D_rim_free=rim_d, max_penetration_m=penetration, p_air_mid_pa=p_mid)
            if label != last_label:
                valve_stream.write(json.dumps(facts) + chr(10))
                valve_stream.flush()
                print("SQUID_FAST valve %s at t %.6f (leak ratio %.3g, rim contact %.3f, rim gap max %s)"
                      % (label, t, ratio, fraction, c.get('rim_gap_max_m')), flush=True)
                last_label = label
            threshold = PENETRATION_REPORT_DC * ns['d_c_rim_m']
            if penetration > threshold:                       # item 3: report, no stop
                event(dict(facts, kind="penetration_above_2dc", threshold_m=threshold))
                if not penetration_high:
                    print("SQUID_FAST penetration %.3e m above 2 d_c = %.3e m at t %.6f (reported, no stop)"
                          % (penetration, threshold, t), flush=True)
                penetration_high = True
            else:
                penetration_high = False
            div_out = _num(air_rep.get('divergence_residual_outlet_max_1ps'))
            div_water = _num(air_rep.get('divergence_residual_water_max_1ps'))
            if div_out > DIV_OUTLET_REPORT_1PS or div_out > DIV_OUTLET_OVER_WATER * div_water:   # REQ_411 condition
                row = dict(step=step, t_s=t, kind="div_outlet_above_bound", div_outlet_max_1ps=div_out,
                           div_water_max_1ps=div_water, bound_1ps=DIV_OUTLET_REPORT_1PS,
                           over_water_factor=DIV_OUTLET_OVER_WATER, pressure_solver=args.pressure_solver)
                if solver_events['stream'] is None:
                    solver_events['stream'] = (out / "solver_events.jsonl").open("x")
                solver_events['stream'].write(json.dumps(row) + chr(10))
                solver_events['stream'].flush()
                if not solver_events['high']:
                    print("SQUID_FAST div_outlet %.3e 1/s above the REQ_411 bound (water %.3e) at t %.6f (reported)"
                          % (div_out, div_water, t), flush=True)
                solver_events['high'] = True
            else:
                solver_events['high'] = False
            if fraction > 0.0 and rim_d < RIM_SUPPORT_STOP:   # item 2: stop and report
                event(dict(facts, kind="rim_support_below_0p1_in_contact"))
                hard_stop('rim_support_below_0p1_in_contact', facts)
            if hook is not None:
                total_iterations = int(hook.totals['hypre_iterations'])
                iterations = (total_iterations - last_hypre_iterations) / max(step - last_iteration_step, 1)
                last_hypre_iterations = total_iterations
                fallbacks = int(sum(hook.totals['fallbacks'].values()))
            else:
                iterations = float(getattr(fluid, 'last_project_cg_iterations_total', -1))
                fallbacks = 0
            last_iteration_step = step
            now = time.perf_counter()
            wall = (now - last_wall) / max(step - last_row_step, 1)
            last_wall, last_row_step = now, step
            writer.writerow([step, "%.9g" % t, "%.6g" % dt_step, limiter, "%.6g" % p_air(t),
                             "%.6g" % fl['nozzle_flux_m3s'], "%.9g" % air_volume, "%.6g" % rate,
                             "%.6g" % d['membrane_min_uz_m'], "%.6g" % d['rim_min_gap_m'],
                             "%.6g" % float(c.get('rim_contact_fraction', float('nan'))),
                             "%.6g" % float(c.get('contact_force_n', float('nan'))),
                             "%.6g" % float(c.get('refill_flow_m3s', float('nan'))),
                             "%.6g" % float(c.get('closed_leakage_m3s', float('nan'))), "%.6g" % d['J_min'],
                             "%.6g" % d['J_max'], "%.6g" % d['volume_error'], "%.6g" % fl['max_u_mps'],
                             "%.6g" % fl['max_p_pa'], "%.6g" % one_sided['min_D'],
                             "%.6g" % one_sided['min_D_free'],
                             int(getattr(fluid, 'last_cg_iterations', -1)), "%.6g" % ratio, label,
                             "%.6g" % _num(c.get('rim_gap_min_m')), "%.6g" % _num(c.get('rim_gap_max_m')),
                             "%.6g" % _num(c.get('dp_rim_pa')), "%.6g" % penetration, "%.6g" % rim_d,
                             "%.6g" % d['volume_error_flap'], "%.6g" % d['volume_error_membrane'],
                             "%.6g" % d['tether_max_displacement_flap_m'], "%.6g" % d['tether_max_uz_flap_m'],
                             args.pressure_solver, "%.6g" % iterations, fallbacks,
                             "%.6g" % _num(air_rep.get('divergence_residual_water_max_1ps')),
                             "%.6g" % _num(air_rep.get('divergence_residual_outlet_max_1ps')),
                             "%.6g" % wall, "%.3f" % time.time()])
            stream.flush()
            print("SQUID_FAST step %d t %.6f dt %.3g (%s) p_air %.1f flux %.3e apex %.3e rim_gap %.3e rim_contact %.3f "
                  "J [%.4f %.4f] u %.3g p %.4g minD %.3f rimD %.3f valve %s leak %.3g volerr f/m %.2e/%.2e cg %s "
                  "wall %.3f s/step"
                  % (step, t, dt_step, limiter, p_air(t), fl['nozzle_flux_m3s'], d['membrane_min_uz_m'],
                     d['rim_min_gap_m'], fraction, d['J_min'], d['J_max'], fl['max_u_mps'], fl['max_p_pa'],
                     one_sided['min_D'], rim_d, label, ratio, d['volume_error_flap'], d['volume_error_membrane'],
                     getattr(fluid, 'last_cg_iterations', -1), wall), flush=True)
        reached = t >= args.end_time - 1.0e-12 or (args.max_steps > 0 and step >= args.max_steps)
        return ("COMPLETED" if reached else "STOPPED"), step, t
    finally:
        stream.close()
        valve_stream.close()
        if events['stream'] is not None:
            events['stream'].close()
        if solver_events['stream'] is not None:
            solver_events['stream'].close()
        if vtk is not None:
            vtk.finalize()                                         # the .pvd collections, written once


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status, interrupted_by, step, t = "RUNNING", None, 0, 0.0
    signal.signal(signal.SIGTERM, _raise_interrupted)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    ns = None
    try:
        ns = build(args)
        if not args.setup_only:
            ns['report']['initial_min_D'] = ns['body'].min_D_witness()
            ns['phases'].mark('initial_min_D')
        write_new(args.output / "setup_report.json", ns['report'])
        print("SQUID_FAST_SETUP", json.dumps(dict(points=ns['report'].get('points'), cells=ns['report']['grid']['cells'],
                                                  dt_elastic_s=ns['dt_elastic'], label=LABEL)), flush=True)
        if 'free_volume_shares' in ns['report']:
            print("SQUID_FAST free_volume_shares", json.dumps(ns['report']['free_volume_shares']), flush=True)
        if args.setup_only:
            status = "SETUP_ONLY"
        else:
            ns['contact'].event_log_path = args.output / "contact_events.jsonl"
            status, step, t = run(args, ns, args.output)
    except KeyboardInterrupt:
        status, interrupted_by = "INTERRUPTED", "SIGINT"
        raise
    except _Interrupted as name:
        status, interrupted_by = "INTERRUPTED", str(name)
        raise
    except Exception as error:
        code = getattr(error, 'code', None)
        status = ("HARD_STOP: %s" % code) if code else "FAILED: %s: %s" % (type(error).__name__, str(error)[:400])
        if code:
            write_new(args.output / "hard_stop.json", dict(code=code, witness=getattr(error, 'witness', None)))
        raise
    finally:
        final = dict(status=status, signal=interrupted_by, steps=step, t_s=t, end_time_s=args.end_time,
                     wall_s=time.perf_counter() - started, label=LABEL, revision=SQUID_FAST_REVISION,
                     pressure_solver=args.pressure_solver,
                     geometry_note=None if ns is None else ns['report'].get('geometry_note'))
        if ns is not None and ns.get('hypre') is not None:
            final['hypre'] = ns['hypre'].report()
        (args.output / "final.json").write_text(json.dumps(final, indent=1, default=str))
        if status in ("COMPLETED", "SETUP_ONLY"):
            write_new(args.output / "long_result.json", dict(source="squid_fast", final=final))
        print("SQUID_FAST_END", status, "steps", step, "t", t, flush=True)


if __name__ == "__main__":
    main()
