"""REQ_379 FAST MODE prototype driver (Claude 2026-10-11; v2 2026-10-12: Root REQ_382 volume-drift statistics; v3:
final.json status is COMPLETED only when the configured end time / step count is reached, INTERRUPTED with the signal,
step and t on SIGINT / SIGTERM (v4: SIGINT handler restored even when inherited as ignored), STOPPED_NONFINITE or
FAILED otherwise; v5 (Root REQ_385 dt stability screen): --ramp-time overrides the preset's 2 s inlet ramp, e.g. 0.2 s,
recorded in run_meta.json; v6 (Root REQ_387/388 F3 item #1): --inlet-update values (default) writes only the inlet face
values after the first full inlet write: the inlet magnitude does not change topology, so the pressure nullspace graph,
the velocity-Dirichlet ledger and reachability stay valid and are not re-sealed or re-marked; --inlet-update full keeps
the old per-step rewrite): Turek-Hron FSI1/2/3 with the diffuse IB-MPM single-velocity
coupling of simulation_core/fast/ib_mpm_fast.py (design note FAST_MODE_DESIGN_NOTE_01, approved by Root).

The fluid is the existing CartesianFluidSolver of cases/turek_hron_fsi.py (_build_fluid: channel, cylinder obstacle,
inlet faces with the 2 s cosine ramp, z-min pressure outlet) driven by predict/project with fv_cg + fv_multigrid; no
sharp-mode machinery runs.  The beam is a lattice of material points (spacing <= spacing_over_h * h), extended into
the cylinder, where the points are tethered to their reference positions.

Per step: dt from the elastic shear-wave CFL over the solid cells (uniform grid) and an advective bound; inlet faces at
t + dt; spread the elastic + tether force; body force; predict; project; quasi-2-D filter; move the points.
History (every step): t, dt and its limiter, Turek tip displacement (point A), drag/lift on cylinder + beam by a
control-volume momentum balance (per unit span), J min/max, tether root displacement, CG iterations, wall time, and
(Root REQ_382) the volume drift of the free body: total volume error sum(A J)/sum(A) - 1, the fraction of points with
|J - 1| > 0.5 % and the share of those within 2 cells (the IB_4 half-width) of the reference body surface.

usage: python -B -m cases.turek_fast --preset fsi3 --grid 4,96,576 --end-time 4.0 --output DIR
       [--arch cuda] [--cfl-elastic 0.3] [--cfl-advective 0.4] [--kappa-over-mu 10] [--spacing-over-h 0.45]
       [--cg-tolerance 1e-5] [--max-steps N] [--row-interval 1] [--ramp-time S] [--inlet-update values|full]
"""
import argparse
import csv
import json
import math
import signal
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np


class _Interrupted(BaseException):
    """A termination signal turned into an exception so that final.json records the interruption."""


def _raise_interrupted(signum, frame):
    raise _Interrupted(signal.Signals(signum).name)


def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=("fsi1", "fsi2", "fsi3"), default="fsi3")
    p.add_argument("--grid", default="4,96,576")
    p.add_argument("--end-time", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arch", default="cuda")
    p.add_argument("--cfl-elastic", type=float, default=0.3)
    p.add_argument("--cfl-advective", type=float, default=0.4)
    p.add_argument("--kappa-over-mu", type=float, default=10.0)
    p.add_argument("--spacing-over-h", type=float, default=0.45)
    p.add_argument("--cg-tolerance", type=float, default=1.0e-5)
    p.add_argument("--cg-iterations", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--row-interval", type=int, default=1)
    p.add_argument("--dt-max", type=float, default=1.0e-3)
    p.add_argument("--time-integration", choices=("euler", "midpoint"), default="euler")
    p.add_argument("--no-semi-coarsening", action="store_true")
    p.add_argument("--ramp-time", type=float, default=None)
    p.add_argument("--inlet-update", choices=("values", "full"), default="values")
    return p.parse_args(argv)


def beam_points(case, cfg, spacing, h=None):
    """(y, z) lattice of the beam from the tip to the cylinder centre; tether mask = inside the cylinder; near-surface
    mask = free points within 2 h of the reference beam surface (top, bottom, tip, cylinder junction)."""
    box_min, box_max = case.beam_box_solver_m(cfg)
    zc = case.solver_z_from_turek_hron_x_m(float(cfg.cylinder_center_x_m), cfg)
    yc, r = float(cfg.cylinder_center_y_m), float(cfg.cylinder_radius_m)
    y0, y1 = float(box_min[1]), float(box_max[1])
    z_tip = float(box_min[2])
    ny = max(2, int(math.ceil((y1 - y0) / spacing)))
    nz = max(2, int(math.ceil((zc - z_tip) / spacing)))
    sy, sz = (y1 - y0) / ny, (zc - z_tip) / nz
    ys = y0 + (np.arange(ny) + 0.5) * sy
    zs = z_tip + (np.arange(nz) + 0.5) * sz
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    points = np.column_stack([Y.ravel(), Z.ravel()])
    tether = ((points[:, 0] - yc) ** 2 + (points[:, 1] - zc) ** 2 < r * r).astype(np.int32)
    areas = np.full(len(points), sy * sz)
    tip = int(np.argmin((points[:, 0] - yc) ** 2 + (points[:, 1] - z_tip) ** 2))
    band = 2.0 * float(h if h is not None else min(sy, sz))
    radial = np.sqrt((points[:, 0] - yc) ** 2 + (points[:, 1] - zc) ** 2)
    distance = np.minimum.reduce([points[:, 0] - y0, y1 - points[:, 0], points[:, 1] - z_tip, np.abs(radial - r)])
    near = ((distance < band) & (tether == 0)).astype(np.int32)
    return points, areas, tether, tip, near, dict(spacing_y_m=sy, spacing_z_m=sz, points=len(points),
                                                   tethered=int(tether.sum()), z_tip_m=z_tip, z_cylinder_m=zc,
                                                   near_surface_band_m=band, near_surface_points=int(near.sum()))


def main(argv=None):
    a = parse(sys.argv[1:] if argv is None else argv)
    a.output.mkdir(parents=True, exist_ok=False)
    from simulation_core.diagnostics.runtime import init_taichi, TaichiRuntimeConfig
    runtime = TaichiRuntimeConfig(arch=a.arch, strict_arch=(a.arch == "cuda"), default_fp="f32", random_seed=0,
                                  offline_cache=False)
    init_taichi(runtime)
    from cases import turek_fast_case as case
    from cases.turek_fast_case import _enforce_fluid_quasi_2d
    from simulation_core.fast.ib_mpm_fast import ControlVolumeForce, FAST_MODE_REVISION, FastIbMpmQuasi2D

    grid = tuple(int(v) for v in a.grid.split(","))
    cfg = case.PRESET_BUILDERS[a.preset](grid_nodes=grid, dt_s=1.0e-4, step_count=1, markers_per_side="auto",
                                         markers_per_tip="auto")
    # F2: semi-coarsening gives the quasi-2-D grid a full multigrid hierarchy (8 levels on (4,48,288) instead of 2).
    cfg = replace(cfg, fluid_quasi_2d_x=True, flow_multigrid_semi_coarsening=not a.no_semi_coarsening)
    if a.ramp_time is not None:                    # v5: stability screens reach full inflow sooner (not a benchmark run)
        cfg = replace(cfg, inlet_ramp_time_s=float(a.ramp_time))
    fluid = case._build_fluid(cfg, runtime)
    bounds_min, _ = case._full_bounds(cfg)
    spacing = case.fluid_cell_spacing_m(cfg)
    h = min(spacing[1], spacing[2])
    points, areas, tether, tip, near, layout = beam_points(case, cfg, a.spacing_over_h * h, h)
    mu = float(cfg.young_modulus_pa) / (2.0 * (1.0 + float(cfg.poisson_ratio)))
    rho_s = float(cfg.solid_density_kgm3)
    c_s = math.sqrt(mu / rho_s)
    dt_elastic = a.cfl_elastic * h / c_s
    tether_k = rho_s * (0.25 / dt_elastic) ** 2
    ib = FastIbMpmQuasi2D(fluid=fluid, positions_yz_m=points, areas_m2=areas, tether_mask=tether,
                          density_kgm3=rho_s, shear_modulus_pa=mu, kappa_pa=a.kappa_over_mu * mu,
                          tether_stiffness_pa_per_m=tether_k, grid_origin_m=bounds_min, grid_spacing_m=spacing,
                          grid_nodes=grid, span_m=float(cfg.span_m), near_surface_mask=near)
    # control volume around the cylinder and the beam (Turek x in [0.12, 0.68], y in [0.07, 0.33])
    def k_of(x_turek):
        return int(round((case.solver_z_from_turek_hron_x_m(x_turek, cfg) - bounds_min[2]) / spacing[2]))
    k_range = sorted((k_of(0.68), k_of(0.12)))
    j_range = (int(round((0.07 - bounds_min[1]) / spacing[1])), int(round((0.33 - bounds_min[1]) / spacing[1])))
    cv = ControlVolumeForce(fluid, j_range=j_range, k_range=k_range, origin_m=bounds_min, spacing_m=spacing,
                            grid_nodes=grid, span_m=float(cfg.span_m))
    y_tip0, z_tip0 = float(points[tip, 0]), float(points[tip, 1])
    meta = dict(revision=FAST_MODE_REVISION, args={k: str(v) for k, v in vars(a).items()}, grid=grid,
                spacing_m=spacing, h_min_m=h, preset=a.preset, mu_pa=mu, kappa_pa=a.kappa_over_mu * mu,
                rho_s=rho_s, rho_f=float(cfg.fluid_density_kgm3), mu_f=float(cfg.fluid_viscosity_pa_s),
                mean_inlet_velocity_mps=float(cfg.mean_inlet_velocity_mps), c_s_mps=c_s,
                dt_elastic_s=dt_elastic, tether_k_pa_per_m=tether_k, layout=layout, tip_point_index=tip,
                tip_reference_yz_m=[y_tip0, z_tip0], control_volume=dict(j=j_range, k=k_range),
                inlet_ramp_time_s=float(cfg.inlet_ramp_time_s), inlet_update=a.inlet_update,
                note="quasi-2D x-uniform IB_4 single-velocity coupling; incompressible solid (nu 0.4 SVK of the "
                     "benchmark is modelled as incompressible neo-Hookean: expected stiffer bending, higher "
                     "frequency, smaller amplitude)")
    (a.output / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print("TUREK_FAST_START", json.dumps(meta, default=str)[:600], flush=True)

    u_bound = 3.0 * float(cfg.mean_inlet_velocity_mps)
    dt_advective = a.cfl_advective * h / max(u_bound, 1.0e-6)
    columns = ["step", "t_s", "dt_s", "dt_limiter", "tip_ux_m", "tip_uy_m", "drag_n_per_m", "lift_n_per_m",
               "J_min", "J_max", "tether_max_displacement_m", "cg_iterations", "wall_s_per_step", "nonfinite_points",
               "volume_error", "J_deviated_fraction", "J_deviated_near_surface_share"]
    stream = (a.output / "history.csv").open("x", newline="")
    writer = csv.writer(stream)
    writer.writerow(columns)
    import taichi as ti
    velocity_old = ti.Vector.field(3, ti.f32, shape=grid) if a.time_integration == "midpoint" else None
    ramp_end = float(cfg.inlet_ramp_time_s)
    t, step = 0.0, 0
    inlet_sealed = False                          # v6: the first inlet write is always the full one
    prev_box = prev_beam = None
    last_wall = time.perf_counter()
    started = last_wall
    status = "RUNNING"
    interrupted_by = None
    signal.signal(signal.SIGTERM, _raise_interrupted)
    # a process started as a background job of a non-interactive shell inherits SIGINT as ignored: restore it
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        while t < a.end_time - 1.0e-12 and (a.max_steps <= 0 or step < a.max_steps):
            dt = min(dt_elastic, dt_advective, a.dt_max, a.end_time - t)
            limiter = "elastic" if dt == dt_elastic else ("advective" if dt == dt_advective else "other")
            inlet_rewritten = t <= ramp_end + dt         # the inlet profile is constant after the ramp
            full_inlet_write = inlet_rewritten and (not inlet_sealed or a.inlet_update == "full")
            if full_inlet_write:
                case._write_channel_external_velocity_faces(fluid, cfg, t + dt)
                inlet_sealed = True
            elif inlet_rewritten:                          # v6: magnitude only; topology and ledger unchanged
                case._update_channel_inlet_face_values(fluid, cfg, t + dt)
            if velocity_old is not None:
                velocity_old.copy_from(fluid.velocity)
                ib.midpoint_predict(dt)                    # IBAMR MIDPOINT_RULE: force at X^{n+1/2}
            ib.spread_elastic_force()
            fluid.apply_body_force(dt, read_report=False)
            if full_inlet_write:                           # a full inlet rewrite invalidates the reachability marking
                fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(pressure_outlet_zmin=True)
            fluid.predict(dt_s=dt, advection_scheme="muscl_tvd", pressure_outlet_zmin=True,
                          velocity_inlet_zmax=True)
            _enforce_fluid_quasi_2d(fluid)
            fluid.project(iterations=a.cg_iterations, pressure_outlet_zmin=True,
                          hibm_pressure_reachability_prepared=True, velocity_inlet_zmax=True, dt_s=dt,
                          reset_pressure=False, pressure_solver="fv_cg", cg_tolerance=a.cg_tolerance,
                          cg_preconditioner="fv_multigrid", pressure_solve_failure_policy="raise",
                          read_report=False)
            _enforce_fluid_quasi_2d(fluid)
            if velocity_old is not None:
                ib.midpoint_correct(velocity_old, dt)
            else:
                ib.move_with_fluid(dt)
            t += dt
            step += 1
            if step % a.row_interval == 0:
                d = ib.diagnostics()
                f = cv.measure()
                x = ib.x.to_numpy()[tip]
                box = np.array([f["momentum_y"], f["momentum_z"]])
                beam = np.array([d["momentum_y"], d["momentum_z"]])
                force = np.array([f["surface_y"], f["surface_z"]])
                if prev_box is not None:
                    force = force - (box - prev_box) / (dt * a.row_interval) + (beam - prev_beam) / (dt * a.row_interval)
                prev_box, prev_beam = box, beam
                now = time.perf_counter()
                wall = (now - last_wall) / a.row_interval
                last_wall = now
                cg = int(getattr(fluid, "last_cg_iterations", -1))
                writer.writerow([step, f"{t:.9g}", f"{dt:.6g}", limiter, f"{-(x[1] - z_tip0):.9g}",
                                 f"{x[0] - y_tip0:.9g}", f"{-force[1]:.9g}", f"{force[0]:.9g}",
                                 f"{d['J_min']:.6g}", f"{d['J_max']:.6g}", f"{d['tether_max_displacement_m']:.6g}",
                                 cg, f"{wall:.6g}", d["nonfinite_points"], f"{d['volume_error']:.6g}",
                                 f"{d['J_deviated_fraction']:.6g}", f"{d['J_deviated_near_surface_share']:.6g}"])
                stream.flush()
                if d["nonfinite_points"] or not np.isfinite(x).all():
                    status = "STOPPED_NONFINITE"
                    break
                if step % (50 * a.row_interval) == 0:
                    print("TUREK_FAST step %d t %.5f dt %.3g tip_uy %.5f drag %.2f lift %.2f J [%.4f, %.4f] cg %d "
                          "wall %.4f s/step" % (step, t, dt, x[0] - y_tip0, -force[1], force[0], d["J_min"],
                                                d["J_max"], cg, wall), flush=True)
        if status == "RUNNING":
            reached = t >= a.end_time - 1.0e-12 or (a.max_steps > 0 and step >= a.max_steps)
            status = "COMPLETED" if reached else "STOPPED"
    except KeyboardInterrupt:
        status, interrupted_by = "INTERRUPTED", "SIGINT"
        raise
    except _Interrupted as signal_name:
        status, interrupted_by = "INTERRUPTED", str(signal_name)
        raise
    except Exception as error:
        status = "FAILED: %s: %s" % (type(error).__name__, str(error)[:400])
        raise
    finally:
        stream.close()
        (a.output / "final.json").write_text(json.dumps(dict(
            status=status, signal=interrupted_by, steps=step, t_s=t, end_time_s=a.end_time,
            wall_s=time.perf_counter() - started), indent=1))
        print("TUREK_FAST_END", status, "steps", step, "t", t, flush=True)


if __name__ == "__main__":
    main()
