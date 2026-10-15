"""Turek-Hron FSI1 / FSI2 / FSI3 with the fastfsi diffuse immersed-boundary material-point coupling (example).

The channel, the cylinder, the inlet and the benchmark parameters come from turek_hron_channel.py.  The fluid is the
incompressible MAC-grid solver of fastfsi.fluid (multigrid-preconditioned conjugate gradient, z-min pressure outlet).
The beam is a lattice of material points (spacing <= spacing_over_h * h) of an incompressible neo-Hookean solid, coupled
to the fluid by PlanarImmersedBoundaryCoupler (one velocity field, IB_4 kernel; the points are PlanarMaterialPoints).
The lattice extends into the cylinder, where its points are clamped by direct forcing (--clamp penalty: a penalty
tether instead).

Per step: dt = min(elastic shear-wave CFL over the smallest cell, advective bound, excess-mass spring bound, --dt-max);
inlet faces at t + dt; spread the elastic (and clamp) force; body force; predict; project; quasi-2-D filter; move the
points (explicit Euler, or the midpoint rule with --time-integration midpoint).

Outputs (create-only, in --output):
  run_meta.json  parameters, grid, layout, the control volume and the options in use;
  history.csv    every --row-interval steps: t, dt and its limiter, the displacement of the benchmark point A at the
                 beam tip, drag and lift on the cylinder and the beam per unit span (control-volume momentum balance),
                 J range, clamp drift, conjugate-gradient iterations, wall time, and the volume drift of the free body
                 (sum(A J) / sum(A) - 1, the fraction of points with |J - 1| > 0.5 % and the share of those within
                 2 cells of the reference surface);
  final.json     COMPLETED when the end time or step count is reached, INTERRUPTED on SIGINT / SIGTERM,
                 STOPPED_NONFINITE (nonzero exit), HARD_STOP for invalid J, FAILED otherwise;
  vtk/           optional VTK series (--vtk-interval-s): one fluid x layer and the beam points, with .pvd collections.

The benchmark solid (nu = 0.4, Saint Venant-Kirchhoff) is compressible, the single-velocity IB solid incompressible: its
bending is stiffer unless --equivalent-bending-nu 0.4 matches the plane-strain bending modulus.

Density ratio: FSI2 carries the reference excess mass (rho_s - rho_f) A0 on partner points tied to the beam points by
fixed springs (fastfsi.coupling.excess_mass): symplectic Euler with the default explicit Euler update, Stormer-Verlet
with --time-integration midpoint (dt then comes from the last evaluated spring bound with the library's 0.9 margin;
each step's own bound is checked at its end).  Finite spring stiffness (slip), the incompressible material, the volume
error and the resolution still need physical accuracy gates.  Equal-density FSI1/FSI3 use the original path;
--excess-mass-mode off is a diagnostic only.  With excess mass the library checks the points at the end of every step,
so the per-step solid check of this driver runs only for equal-density bodies.

Control-volume forces (history drag / lift): the momentum change of the fluid box minus the surface stresses, with the
beam's momentum rho_f A0 U from the interpolated point velocities (the excess mass is counted once: the box total
P_grid + P_ex and the body P_fict + P_ex share the same whole-beam P_ex, which cancels); the momentum rate uses the time
difference of successive history rows and the first row has surface terms only; moving_points_outside_cv counts the
points (or partners) outside the box in a row, when the whole-beam sum becomes a diagnostic only.  This is a
same-particle-sum inertia consistency: the beam's reference-area sum is not clipped to the box and differs from the
current-J / grid volume discretisation; it is not a continuum hydrodynamic force accuracy claim.

Solid volumetric modulus: --kappa-over-mu defaults to 2 (1 + nu) / (3 (1 - 2 nu)) with the preset's Poisson ratio
(14/3 for the benchmark's nu = 0.4).

--arch cpu runs the same code on the CPU (slow; for testing and small cases).

Configurations: configs/fsi1.args, configs/fsi2.args and configs/fsi3.args hold the benchmark's reference parameters
as argument files (python -B run_turek_hron_fsi.py @configs/fsi3.args --output DIR); in an argument file a line may
hold several arguments and # starts a comment.

Checkpoint / restart (fastfsi.io.checkpoint, fastfsi.io.restart): --checkpoint-every-s E (0 = off, the default) and
--checkpoint-at T (repeatable) write a create-only checkpoint at the first history row at or after every multiple of E
and at or after every T, into OUTPUT/checkpoints/ (listed in checkpoints.jsonl, never deleted).  A checkpoint holds the
fluid velocity and pressure with the solver's host counters, the beam points (x, F), the excess-mass partner state when
present, and the loop variables.  --restart-from FILE continues the run into a NEW --output directory; the checkpoint's
manifest must match: every argument except the output, the run length, the checkpoint and VTK cadences and the device;
the grid; the beam set-up fingerprint; the source code and the library versions.  A code or library difference is
accepted only with --diagnostic-restart and an output directory whose name contains DIAGNOSTIC (never a result).  At a
restart the inlet faces and the outlet-reachability marking, which depend only on the set-up and the time, are rebuilt
before the checkpoint is loaded.  python -m fastfsi.io.run_segments joins the segments' histories and VTK series.

usage: python -B run_turek_hron_fsi.py --preset fsi3 --grid 4,96,576 --end-time 4.0 --output DIR
       [--arch cuda] [--cfl-elastic 0.3] [--cfl-advective 0.4] [--kappa-over-mu K] [--spacing-over-h 0.45]
       [--cg-tolerance 1e-5] [--max-steps N] [--row-interval 1] [--ramp-time S] [--inlet-update values|full]
       [--vtk-interval-s 0.01 --vtk-start-s 2.5 --vtk-tree NAME [--vtk-dir DIR]]
       [--checkpoint-every-s E] [--checkpoint-at T ...] [--restart-from FILE [--diagnostic-restart]]
       python -B run_turek_hron_fsi.py @configs/fsi3.args --output DIR
"""
import argparse
import csv
import json
import math
import shutil
import signal
import sys
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

EXAMPLE_REVISION = "fastfsi-example-turek-hron-fsi-v7"   # v7: checkpoint / restart, argument files, kappa/mu from nu
# the arguments a restart may change: the output, the run length, the checkpoint and VTK cadences and the device
RESTART_FREE_ARGS = ("output", "end_time", "max_steps", "checkpoint_every_s", "checkpoint_at", "restart_from",
                     "diagnostic_restart", "arch", "vtk_interval_s", "vtk_start_s", "vtk_tree", "vtk_dir")
CHECKPOINT_DISK_HEADROOM_BYTES = 1 << 30


class _Interrupted(BaseException):
    """A termination signal turned into an exception so that final.json records the interruption."""


def _raise_interrupted(signum, frame):
    raise _Interrupted(signal.Signals(signum).name)


class _ArgumentParser(argparse.ArgumentParser):
    """Arguments from the command line and from @files (several arguments per line; # starts a comment)."""

    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split("#", 1)[0].split()


def parse(argv):
    p = _ArgumentParser(fromfile_prefix_chars="@")
    p.add_argument("--preset", choices=("fsi1", "fsi2", "fsi3"), default="fsi3")
    p.add_argument("--grid", default="4,96,576")
    p.add_argument("--end-time", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arch", default="cuda")
    p.add_argument("--cfl-elastic", type=float, default=0.3)
    p.add_argument("--cfl-advective", type=float, default=0.4)
    p.add_argument("--kappa-over-mu", type=float, default=None,
                   help="volumetric / shear modulus of the solid; default 2 (1 + nu) / (3 (1 - 2 nu)) with the "
                        "preset's Poisson ratio")
    p.add_argument("--spacing-over-h", type=float, default=0.45)
    p.add_argument("--cg-tolerance", type=float, default=1.0e-5)
    p.add_argument("--cg-iterations", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--row-interval", type=int, default=1)
    p.add_argument("--dt-max", type=float, default=1.0e-3)
    p.add_argument("--time-integration", choices=("auto", "euler", "midpoint"), default="auto")
    p.add_argument("--excess-mass-mode", choices=("auto", "off"), default="auto",
                   help="auto adds fixed reference excess mass; off is a neutral-inertia diagnostic")
    p.add_argument("--excess-mass-omega-rad-s", type=float, default=None,
                   help="fixed partner spring frequency; default .1 / neutral elastic dt, independent of step dt")
    p.add_argument("--no-semi-coarsening", action="store_true")
    p.add_argument("--ramp-time", type=float, default=None)
    p.add_argument("--inlet-update", choices=("values", "full"), default="values")
    # VTK export, off unless --vtk-interval-s > 0
    p.add_argument("--vtk-interval-s", type=float, default=0.0,
                   help="VTK frame every this many simulated seconds (0 = off); written at history-row points")
    p.add_argument("--vtk-start-s", type=float, default=0.0, help="first VTK frame at the first row with t >= this")
    p.add_argument("--vtk-tree", default="unnamed tree", help="tree name for the VTK label")
    p.add_argument("--vtk-dir", type=Path, default=None, help="VTK series directory (default OUTPUT/vtk; created)")
    p.add_argument("--ib-kernel", choices=("ib4", "ib3"), default="ib4",
                   help="regularized delta of the immersed-boundary coupling (default ib4)")
    p.add_argument("--clamp", choices=("direct", "penalty"), default="direct",
                   help="tethered points as a rigid clamp by direct forcing (direct, default) "
                        "or a penalty tether")
    p.add_argument("--equivalent-bending-nu", type=float, default=None,
                   help="solid shear modulus mu_eq = mu / (2 (1 - NU)) so that the incompressible "
                        "IB solid's plane-strain bending stiffness 4 mu_eq equals the benchmark's 2 mu / (1 - NU); "
                        "default off")
    # checkpoint / restart (module docstring)
    p.add_argument("--checkpoint-every-s", type=float, default=0.0,
                   help="a checkpoint at the first history row at or after every multiple of this time (0 = off)")
    p.add_argument("--checkpoint-at", type=float, action="append", default=[],
                   help="also a checkpoint at the first history row at or after this time (repeatable)")
    p.add_argument("--restart-from", type=Path, default=None,
                   help="continue from this checkpoint into the new (create-only) --output directory")
    p.add_argument("--diagnostic-restart", action="store_true",
                   help="accept a code / library difference from the checkpoint; the output name must contain "
                        "DIAGNOSTIC and the continuation is never a result")
    a = p.parse_args(argv)
    if a.diagnostic_restart and a.restart_from is None:
        p.error("--diagnostic-restart needs --restart-from")
    if not (math.isfinite(a.checkpoint_every_s) and a.checkpoint_every_s >= 0.0):
        p.error("--checkpoint-every-s must be finite and >= 0")
    return a


def reference_kappa_over_mu(nu):
    """kappa / mu = 2 (1 + nu) / (3 (1 - 2 nu)) of a linear-elastic solid, from the decimal value of nu (exactly 14/3
    rounded for nu = 0.4)."""
    nu = Fraction(repr(float(nu)))
    if not (-1 < nu < Fraction(1, 2)):
        raise ValueError("the Poisson ratio must be in (-1, 0.5)")
    return float(2 * (1 + nu) / (3 * (1 - 2 * nu)))


def equivalent_bending_mu(mu_pa, nu):
    """mu_eq = mu / (2 (1 - nu)); None (the default) returns mu unchanged."""
    if nu is None:
        return mu_pa
    nu = float(nu)
    if not (math.isfinite(nu) and -1.0 < nu < 0.5):
        raise ValueError("--equivalent-bending-nu must be finite and in (-1, 0.5)")
    return mu_pa / (2.0 * (1.0 - nu))


def beam_points(channel, cfg, spacing, h=None):
    """(y, z) lattice of the beam from the tip to the cylinder centre; tether mask = inside the cylinder; near-surface
    mask = free points within 2 h of the reference beam surface (top, bottom, tip, cylinder junction)."""
    box_min, box_max = channel.beam_box_solver_m(cfg)
    zc = channel.solver_z(float(cfg.cylinder_center_x_m), cfg)
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


def vtk_label_and_note(a, cfg, grid):
    """FieldData strings of the VTK series."""
    label = "UNVALIDATED engineering run, %s, grid %s, solver fv_cg + fv_multigrid (cg tolerance %g), preset %s" % (
        a.vtk_tree, "x".join(str(v) for v in grid), a.cg_tolerance, a.preset)
    ramp = float(cfg.inlet_ramp_time_s)
    ramp_text = ("inlet cosine ramp over %g s, ACCELERATED (--ramp-time; the benchmark ramps over 2 s)" % ramp
                 if a.ramp_time is not None else "inlet cosine ramp over %g s (preset)" % ramp)
    note = ("%s; time integration %s; inlet update %s; quasi-2-D channel (fields uniform along x, one x layer "
            "written); diffuse IB-MPM single-velocity coupling, incompressible neo-Hookean beam (kappa/mu %g)"
            % (ramp_text, a.time_integration, a.inlet_update, a.kappa_over_mu))
    if getattr(a, "ib_kernel", "ib4") != "ib4":
        note += "; IB kernel %s" % a.ib_kernel
    if getattr(a, "clamp", "penalty") == "direct":
        note += "; rigid clamp by direct forcing of the beam points inside the cylinder"
    if getattr(a, "equivalent_bending_nu", None) is not None:
        note += "; equivalent-bending shear modulus mu_eq = mu / (2 (1 - %g))" % a.equivalent_bending_nu
    return label, note


def write_vtk_frame(writer, fluid, points, step, t):
    """One VTK frame at a history-row point: the host reads happen only here."""
    zmax = fluid.external_velocity_boundary_z_face_value_mps.to_numpy()[1, :, :, 2]
    return writer.write_frame(step, t, fluid.velocity.to_numpy(), fluid.pressure.to_numpy(),
                              fluid.obstacle.to_numpy(), zmax, points.x.to_numpy(), points.F.to_numpy())


def next_vtk_time(t, start, interval):
    """The first frame time start + k * interval strictly after t (frames stay on the interval grid)."""
    if t < start:
        return start
    return start + (math.floor((t - start) / interval + 1.0e-9) + 1) * interval


def validate_solid_state(points, step, t_s, phase):
    """Guard before stress evaluation and after movement, independent of history/VTK cadence."""
    from fastfsi.errors import SimulationHardStop
    try:
        points.validate_state()
    except SimulationHardStop as error:
        error.witness.update(step=int(step), t_s=float(t_s), phase=phase)
        raise


def main(argv=None):
    a = parse(sys.argv[1:] if argv is None else argv)
    if a.preset == "fsi2" and a.excess_mass_mode == "off":
        print("TUREK_HRON_WARNING diagnostic excess mass off: nominal FSI2 density ratio 10 has neutral inertia", flush=True)
    if a.vtk_interval_s < 0.0 or not math.isfinite(a.vtk_interval_s):
        raise ValueError("--vtk-interval-s must be >= 0")
    a.output.mkdir(parents=True, exist_ok=False)
    from fastfsi.io.checkpoint import load_components, read_checkpoint
    saved = read_checkpoint(a.restart_from) if a.restart_from is not None else None   # verified before the build
    from fastfsi.fluid.outlet_reachability import single_level_reachability_flood
    from fastfsi.fluid.solver import enforce_planar_velocity
    from fastfsi.simulation.taichi_runtime import TaichiRuntimeConfig, init_taichi
    runtime = TaichiRuntimeConfig(arch=a.arch, strict_arch=(a.arch == "cuda"), default_fp="f32", random_seed=0,
                                  offline_cache=False)
    init_taichi(runtime)
    import turek_hron_channel as channel
    from fastfsi.coupling.excess_mass import CANDIDATE_DT_MARGIN
    from fastfsi.coupling.immersed_boundary import PlanarImmersedBoundaryCoupler
    from fastfsi.errors import SimulationHardStop
    from fastfsi.simulation.monitors import ControlVolumeForce
    from fastfsi.solid.material_points import PlanarMaterialPoints
    from fastfsi.solid.materials import NeoHookeanMaterial

    grid = tuple(int(v) for v in a.grid.split(","))
    # semi-coarsening gives the quasi-2-D grid a full multigrid hierarchy (8 levels on (4,48,288) instead of 2)
    cfg = channel.preset(a.preset, grid_nodes=grid, dt_s=1.0e-4, multigrid_semi_coarsening=not a.no_semi_coarsening)
    if a.ramp_time is not None:                    # stability screens reach full inflow sooner (not a benchmark run)
        cfg = replace(cfg, inlet_ramp_time_s=float(a.ramp_time))
    if a.kappa_over_mu is None:                    # the benchmark's Poisson ratio
        a.kappa_over_mu = reference_kappa_over_mu(cfg.poisson_ratio)
    fluid = channel.build_fluid(cfg, runtime)
    bounds_min, _ = channel.full_bounds(cfg)
    spacing = channel.cell_spacing_m(cfg)
    h = min(spacing[1], spacing[2])
    points, areas, tether, tip, near, layout = beam_points(channel, cfg, a.spacing_over_h * h, h)
    mu_benchmark = float(cfg.young_modulus_pa) / (2.0 * (1.0 + float(cfg.poisson_ratio)))
    mu = equivalent_bending_mu(mu_benchmark, a.equivalent_bending_nu)     # unchanged when the option is off
    rho_s = float(cfg.solid_density_kgm3)
    rho_f = float(cfg.fluid_density_kgm3)
    # CV total momentum Pgrid + Pex and real body momentum Pfict + Pex share the same Pex; it cancels.
    # Keep rho_f on the interpolated material velocity, not rho_s (which would count the extra inertia twice).
    beam_momentum_scale = rho_f / rho_s
    c_s = math.sqrt(mu / rho_f)
    dt_elastic = a.cfl_elastic * h / c_s
    # Preserve the original anchor force; the new conservative rho_f limit changes only physical advancement.
    tether_dt_nominal = a.cfl_elastic * h / math.sqrt(mu / rho_s)
    tether_k = rho_s * (0.25 / tether_dt_nominal) ** 2
    integrator = "euler" if a.time_integration == "auto" else a.time_integration
    material = NeoHookeanMaterial(density_kgm3=rho_s, shear_modulus_pa=mu, volumetric_modulus_pa=a.kappa_over_mu * mu)
    solid_points = PlanarMaterialPoints(positions_yz_m=points, areas_m2=areas, material=material,
                                        tether_mask=tether, near_surface_mask=near)
    ib = PlanarImmersedBoundaryCoupler(
        fluid=fluid, points=solid_points, tether_stiffness_pa_per_m=tether_k, grid_origin_m=bounds_min,
        grid_spacing_m=spacing, grid_nodes=grid, span_m=float(cfg.span_m), ib_kernel=a.ib_kernel, clamp=a.clamp,
        clamp_density_kgm3=float(cfg.fluid_density_kgm3), excess_mass_mode=a.excess_mass_mode,
        excess_mass_omega_rad_s=a.excess_mass_omega_rad_s, cfl_elastic=a.cfl_elastic)
    density = ib.density_report()
    excess = ib.excess_mass
    # control volume around the cylinder and the beam (Turek x in [0.12, 0.68], y in [0.07, 0.33])
    def k_of(x_turek):
        return int(round((channel.solver_z(x_turek, cfg) - bounds_min[2]) / spacing[2]))
    k_range = sorted((k_of(0.68), k_of(0.12)))
    j_range = (int(round((0.07 - bounds_min[1]) / spacing[1])), int(round((0.33 - bounds_min[1]) / spacing[1])))
    cv = ControlVolumeForce(fluid, j_range=j_range, k_range=k_range, origin_m=bounds_min, spacing_m=spacing,
                            grid_nodes=grid, span_m=float(cfg.span_m))
    y_tip0, z_tip0 = float(points[tip, 0]), float(points[tip, 1])
    from fastfsi.io.restart import (CheckpointPlan, check_restart, checkpoint_bytes, code_manifest, device_identity,
                                    grid_manifest, library_versions, restart_record, write_new_json)
    checkpointing = saved is not None or a.checkpoint_every_s > 0.0 or bool(a.checkpoint_at)
    components = manifest = resume = None
    if checkpointing:                              # what a checkpoint holds and what a restart must match
        components = dict(fluid=fluid, points=solid_points)
        if excess is not None:
            components["excess_mass"] = excess
        manifest = dict(config={k: str(v) for k, v in sorted(vars(a).items()) if k not in RESTART_FREE_ARGS},
                        grid=grid_manifest(grid, (fluid.grid.cell_faces_x_m, fluid.grid.cell_faces_y_m,
                                                  fluid.grid.cell_faces_z_m)),
                        coupler=ib.fingerprint(),
                        code=code_manifest(("fastfsi", "turek_hron_channel"), {"run_turek_hron_fsi.py": __file__}),
                        libraries=library_versions(), device=device_identity(a.arch))
    if saved is not None:
        arrays, host, loop_state, saved_manifest, header = saved
        code_diffs = check_restart(saved_manifest, manifest, diagnostic=a.diagnostic_restart,
                                   output_name=a.output.name)
        # the inlet faces and the outlet-reachability marking depend only on the set-up and the time: rebuilt first,
        # then the checkpoint (with the solver's host counters) is loaded over them
        channel.write_channel_velocity_faces(fluid, cfg, float(loop_state["t"]))
        with single_level_reachability_flood():
            fluid.mark_pressure_outlet_disconnected_nonprojectable_cells(pressure_outlet_zmin=True)
        load_components(components, arrays, host)
        resume = loop_state
        write_new_json(a.output / "restart.json", restart_record(a.restart_from, loop_state, saved_manifest, manifest,
                                                                 header, code_diffs))
        print("TUREK_FAST restart from %s at step %d t %.9g%s" % (a.restart_from, loop_state["step"], loop_state["t"],
                                                                  " (DIAGNOSTIC)" if code_diffs else ""), flush=True)
    vtk_writer, vtk_next = None, None
    if a.vtk_interval_s > 0.0:                     # VTK export, off by default
        from fastfsi.io.vtk import PlanarVtkWriter
        label, note = vtk_label_and_note(a, cfg, grid)
        faces = [np.asarray(f, dtype=np.float64) for f in (fluid.grid.cell_faces_x_m, fluid.grid.cell_faces_y_m,
                                                            fluid.grid.cell_faces_z_m)]
        vtk_writer = PlanarVtkWriter(a.vtk_dir or (a.output / "vtk"), faces_x=faces[0], faces_y=faces[1],
                                     faces_z=faces[2], X0_yz=solid_points.X0.to_numpy(), tether=tether,
                                     near_surface=near, area_m2=areas, tip_index=tip, label=label, note=note)
        vtk_next = float(a.vtk_start_s)
        if resume is not None:                     # the saved schedule, else the first frame after the restart
            vtk_next = resume["vtk_next"] if resume["vtk_next"] is not None else next_vtk_time(
                float(resume["t"]), float(a.vtk_start_s), float(a.vtk_interval_s))
    meta = dict(revision=EXAMPLE_REVISION, args={k: str(v) for k, v in vars(a).items()}, grid=grid,
                spacing_m=spacing, h_min_m=h, preset=a.preset, mu_pa=mu, kappa_pa=a.kappa_over_mu * mu,
                rho_s=rho_s, rho_f=rho_f, mu_f=float(cfg.fluid_viscosity_pa_s),
                effective_solid_density_kgm3=density["effective_reference_density_kgm3"],
                effective_density_ratio=density["effective_reference_density_ratio"],
                density_inertia=density, time_integration=integrator, c_s_density_kgm3=rho_f,
                mean_inlet_velocity_mps=float(cfg.mean_inlet_velocity_mps), c_s_mps=c_s,
                dt_elastic_s=dt_elastic, tether_k_pa_per_m=tether_k, layout=layout, tip_point_index=tip,
                tip_reference_yz_m=[y_tip0, z_tip0], control_volume=dict(
                    j=j_range, k=k_range, beam_momentum_density_kgm3=rho_f,
                    beam_momentum_scale=beam_momentum_scale, momentum_dt="successive_history_row_time_difference",
                    first_row="surface_only_no_previous_momentum_sample"),
                inlet_ramp_time_s=float(cfg.inlet_ramp_time_s), inlet_update=a.inlet_update,
                note="quasi-2D x-uniform IB_4 single-velocity coupling; incompressible solid (nu 0.4 SVK of the "
                     "benchmark is modelled as incompressible neo-Hookean: expected stiffer bending, higher "
                     "frequency, smaller amplitude)",
                mu_benchmark_pa=mu_benchmark, ib_kernel=a.ib_kernel,
                clamp=dict(mode=a.clamp, rule=("approximate anchoring by direct forcing: residual tether drift is reported; force "
                                                "-rho_f A U / dt from the interpolated velocity"
                                                if a.clamp == "direct" else "penalty tether rho_s (0.25/dt)^2")),
                equivalent_bending=None if a.equivalent_bending_nu is None else dict(
                    nu=float(a.equivalent_bending_nu), factor=mu / mu_benchmark,
                    rule="mu_eq = mu / (2 (1 - nu)): plane-strain bending modulus of the incompressible IB solid "
                         "4 mu_eq = the benchmark's 2 mu / (1 - nu); kappa = kappa_over_mu x mu_eq; dt_elastic and "
                         "the tether stiffness use mu_eq"))
    meta["checkpoint"] = dict(every_s=a.checkpoint_every_s, at_s=sorted(a.checkpoint_at),
                              restart_from=None if resume is None else str(Path(a.restart_from).resolve()),
                              restart_step=None if resume is None else int(resume["step"]),
                              diagnostic_restart=bool(a.diagnostic_restart))
    if vtk_writer is not None:
        meta["vtk"] = dict(dir=str(vtk_writer.out_dir), interval_s=a.vtk_interval_s, start_s=a.vtk_start_s,
                           layer_i=vtk_writer.layer_i, layer_x_m=vtk_writer.x_layer, label=vtk_writer.label,
                           note=vtk_writer.note, written_at="history-row points (host reads only there)")
    (a.output / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print("TUREK_FAST_START", json.dumps(meta, default=str)[:600], flush=True)

    u_bound = 3.0 * float(cfg.mean_inlet_velocity_mps)
    dt_advective = a.cfl_advective * h / max(u_bound, 1.0e-6)
    columns = ["step", "t_s", "dt_s", "dt_limiter", "tip_ux_m", "tip_uy_m", "drag_n_per_m", "lift_n_per_m",
               "J_min", "J_max", "tether_max_displacement_m", "cg_iterations", "wall_s_per_step", "nonfinite_points",
               "volume_error", "J_deviated_fraction", "J_deviated_near_surface_share"]
    if excess is not None:
        columns += ["excess_position_slip_m", "excess_velocity_slip_mps", "excess_kinetic_energy",
                    "excess_spring_energy", "excess_beta", "moving_points_outside_cv"]
    stream = (a.output / "history.csv").open("x", newline="")
    writer = csv.writer(stream)
    writer.writerow(columns)
    import taichi as ti
    velocity_old = ti.Vector.field(3, ti.f32, shape=grid) if integrator == "midpoint" else None
    ramp_end = float(cfg.inlet_ramp_time_s)
    t, step = 0.0, 0
    inlet_sealed = False                          # the first inlet write is always the full one
    prev_box = prev_beam = prev_row_t = None
    if resume is not None:                        # the loop variables of the checkpoint
        t, step, inlet_sealed = float(resume["t"]), int(resume["step"]), bool(resume["inlet_sealed"])
        prev_box, prev_beam, prev_row_t = resume["prev_box"], resume["prev_beam"], resume["prev_row_t"]
    plan = None
    if checkpointing:
        plan = CheckpointPlan(a.output, a.checkpoint_every_s, a.checkpoint_at, components, manifest, t,
                              label="TUREK_FAST")
        planned = plan.planned_count(a.end_time, t)
        if planned:                                # the disk check counts every planned checkpoint
            each = checkpoint_bytes(components)
            free = shutil.disk_usage(a.output).free
            print("TUREK_FAST checkpoints planned %d x %.1f MB; disk free %.1f GB"
                  % (planned, each / 1.0e6, free / 1.0e9), flush=True)
            if free < planned * each + CHECKPOINT_DISK_HEADROOM_BYTES:
                raise ValueError("disk check: %d checkpoints of %d bytes + %d headroom exceed %d free bytes"
                                 % (planned, each, CHECKPOINT_DISK_HEADROOM_BYTES, free))
        else:
            plan = None
    last_wall = time.perf_counter()
    started = last_wall
    status = "RUNNING"
    interrupted_by = None
    hard_stop = None
    signal.signal(signal.SIGTERM, _raise_interrupted)
    # a process started as a background job of a non-interactive shell inherits SIGINT as ignored: restore it
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        validate_solid_state(solid_points, step, t, "initial")
        while t < a.end_time - 1.0e-12 and (a.max_steps <= 0 or step < a.max_steps):
            if excess is None:                         # equal density: no spring bound
                dt_spring = float("inf")
            elif velocity_old is None:                 # explicit Euler: the bound where this step spreads
                dt_spring = ib.excess_mass_dt_limit()
            else:                                      # midpoint: spread at X^{n+1/2}, checked at the step's end
                dt_spring = CANDIDATE_DT_MARGIN * ib.excess_mass_candidate_dt_limit()
            dt = min(dt_elastic, dt_advective, dt_spring, a.dt_max, a.end_time - t)
            limiter = ("spring" if dt == dt_spring else
                       "elastic" if dt == dt_elastic else ("advective" if dt == dt_advective else "other"))
            inlet_rewritten = t <= ramp_end + dt         # the inlet profile is constant after the ramp
            full_inlet_write = inlet_rewritten and (not inlet_sealed or a.inlet_update == "full")
            if full_inlet_write:
                channel.write_channel_velocity_faces(fluid, cfg, t + dt)
                inlet_sealed = True
            elif inlet_rewritten:                          # magnitude only; topology and registry unchanged
                channel.update_inlet_face_values(fluid, cfg, t + dt)
            if velocity_old is not None:
                velocity_old.copy_from(fluid.velocity)
                ib.midpoint_predict(dt)                    # IBAMR MIDPOINT_RULE: force at X^{n+1/2}
                if excess is None:                         # with excess mass the library checks the step's end
                    validate_solid_state(solid_points, step + 1, t + 0.5 * dt, "midpoint_predict")
            ib.spread_elastic_force(dt_s=dt)              # the direct clamp needs this step's dt
            fluid.apply_body_force(dt, read_report=False)
            if full_inlet_write:                           # a full inlet rewrite invalidates the reachability marking
                # one breadth-first level per launch (the same reachable set and level count) instead of the 16-level
                # batch kernel, whose CUDA compile can need more than 8 GB of host memory
                with single_level_reachability_flood():
                    fluid.mark_pressure_outlet_disconnected_nonprojectable_cells(pressure_outlet_zmin=True)
            fluid.predict(dt_s=dt, advection_scheme="muscl_tvd", pressure_outlet_zmin=True,
                          velocity_inlet_zmax=True)
            enforce_planar_velocity(fluid)
            fluid.project(iterations=a.cg_iterations, pressure_outlet_zmin=True,
                          pressure_reachability_prepared=True, velocity_inlet_zmax=True, dt_s=dt,
                          reset_pressure=False, pressure_solver="fv_cg", cg_tolerance=a.cg_tolerance,
                          cg_preconditioner="fv_multigrid", pressure_solve_failure_policy="raise",
                          read_report=False)
            enforce_planar_velocity(fluid)
            if velocity_old is not None:
                ib.midpoint_correct(velocity_old, dt)
            else:
                ib.move_with_fluid(dt)
            if excess is None:                             # with excess mass accept() has checked the points
                validate_solid_state(solid_points, step + 1, t + dt, "after_move")
            t += dt
            step += 1
            if step % a.row_interval == 0:
                d = ib.diagnostics()
                f = cv.measure()
                position_h = solid_points.x.to_numpy()
                x = position_h[tip]
                box = np.array([f["momentum_y"], f["momentum_z"]])
                beam = np.array([d["momentum_y"], d["momentum_z"]]) * beam_momentum_scale
                force = np.array([f["surface_y"], f["surface_z"]])
                if prev_box is not None:
                    row_dt = t - prev_row_t
                    force = force - (box - prev_box) / row_dt + (beam - prev_beam) / row_dt
                prev_box, prev_beam, prev_row_t = box, beam, t
                now = time.perf_counter()
                wall = (now - last_wall) / a.row_interval
                last_wall = now
                cg = int(getattr(fluid, "last_cg_iterations", -1))
                row_values = [step, f"{t:.9g}", f"{dt:.6g}", limiter, f"{-(x[1] - z_tip0):.9g}",
                                 f"{x[0] - y_tip0:.9g}", f"{-force[1]:.9g}", f"{force[0]:.9g}",
                                 f"{d['J_min']:.6g}", f"{d['J_max']:.6g}", f"{d['tether_max_displacement_m']:.6g}",
                                 cg, f"{wall:.6g}", d["nonfinite_points"], f"{d['volume_error']:.6g}",
                                 f"{d['J_deviated_fraction']:.6g}", f"{d['J_deviated_near_surface_share']:.6g}"]
                if excess is not None:
                    lower = np.array([bounds_min[1] + j_range[0] * spacing[1], bounds_min[2] + k_range[0] * spacing[2]])
                    upper = np.array([bounds_min[1] + j_range[1] * spacing[1], bounds_min[2] + k_range[1] * spacing[2]])
                    partner = solid_points.X0.to_numpy().astype(np.float64) + excess.displacement.to_numpy()
                    outside = np.any((position_h < lower) | (position_h > upper), axis=1)
                    outside |= np.any((partner < lower) | (partner > upper), axis=1)
                    e = d["excess_mass"]
                    row_values.extend([e["max_position_slip_m"], e["max_velocity_slip_mps"], e["kinetic_energy"],
                                       e["spring_energy"], e["last_beta"], int(np.count_nonzero(outside))])
                writer.writerow(row_values)
                stream.flush()
                if vtk_writer is not None and t >= vtk_next - 1.0e-12:     # frame at this row point
                    write_vtk_frame(vtk_writer, fluid, solid_points, step, t)
                    vtk_next = next_vtk_time(t, float(a.vtk_start_s), float(a.vtk_interval_s))
                    last_wall = time.perf_counter()        # the frame's write time is not charged to the step wall
                if plan is not None and plan.due(t):       # after the row: its momentum is the next row's previous
                    validate_solid_state(solid_points, step, t, "checkpoint")
                    plan.write(step, t, dict(t=t, step=step, inlet_sealed=inlet_sealed, prev_box=prev_box,
                                             prev_beam=prev_beam, prev_row_t=prev_row_t, vtk_next=vtk_next))
                    last_wall = time.perf_counter()        # nor the checkpoint's
                if step % (50 * a.row_interval) == 0:
                    print("TUREK_FAST step %d t %.5f dt %.3g tip_uy %.5f drag %.2f lift %.2f J [%.4f, %.4f] cg %d "
                          "wall %.4f s/step" % (step, t, dt, x[0] - y_tip0, -force[1], force[0], d["J_min"],
                                                d["J_max"], cg, wall), flush=True)
        if status == "RUNNING":
            validate_solid_state(solid_points, step, t, "final")
            reached = t >= a.end_time - 1.0e-12 or (a.max_steps > 0 and step >= a.max_steps)
            status = "COMPLETED" if reached else "STOPPED"
    except KeyboardInterrupt:
        status, interrupted_by = "INTERRUPTED", "SIGINT"
        raise
    except _Interrupted as signal_name:
        status, interrupted_by = "INTERRUPTED", str(signal_name)
        raise
    except SimulationHardStop as error:
        status = "STOPPED_NONFINITE" if error.code == "nonfinite_point_state" else "HARD_STOP: " + error.code
        hard_stop = dict(code=error.code, witness=error.witness)
        raise
    except Exception as error:
        status = "FAILED: %s: %s" % (type(error).__name__, str(error)[:400])
        raise
    finally:
        stream.close()
        final = dict(status=status, signal=interrupted_by, steps=step, t_s=t, end_time_s=a.end_time,
                     wall_s=time.perf_counter() - started,
                     checkpoints=[] if plan is None else [Path(r["file"]).name for r in plan.written],
                     restart_from=None if a.restart_from is None else str(Path(a.restart_from).resolve()))
        if hard_stop is not None:
            final["hard_stop"] = hard_stop
        if vtk_writer is not None:                         # the .pvd collections, once, whatever the status
            final["vtk_pvd"] = [str(p) for p in vtk_writer.finalize()]
        (a.output / "final.json").write_text(json.dumps(final, indent=1))
        print("TUREK_FAST_END", status, "steps", step, "t", t, flush=True)


if __name__ == "__main__":
    main()
