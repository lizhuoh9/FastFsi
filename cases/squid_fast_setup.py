"""Squid FAST MODE setup (Claude 2026-10-12; Root REQ_401 step 6, REQ_399 D1/D2): host-side geometry for the squid fast
run — the graded grid, the fluid, the static obstacle and the material points.  No Taichi kernel of the fast-mode
helpers is used here, so the setup can be built and checked on its own (`squid_fast.py --setup-only`).

Everything is REUSED from the production squid path (cases/squid_soft_robot), only the fast-mode choices differ:
- grid: setup.spec_with_squid_bore_grid(preset='volumetric') region for region, with Root's D2: the axial 0.6 mm band
  runs from the plate (z = 1.006 m) through the membrane travel band to MEMBRANE_BAND_TOP_M (1.045 m), then the
  growth-limited bridge to the 2.0 mm band up to the lid; D1: the 0.375 mm nozzle bands are kept (the same nozzle
  resolution as the sharp reference);
- obstacle: setup.build_source_config_fluid_obstacle_mask (rigid regions 3 and 4 of the user's config,
  solid_obstacle_surface_only_region_ids = [3, 4]) plus the user's chamber-wall groove
  (UDF/m2_inputs/squid_runner_groove.carve_groove, approved 2026-10-03), exactly as UDF/m6_squid_production applies it;
- material points: the two deformable parts (region 1 valve flap, region 2 main membrane) are sampled by the production
  closed-solid classifier (closed_solid_cells.classify_closed_solid_cells, the engine of closed_solid_particle_lattice)
  on a SUB-LATTICE that halves every fluid cell along each axis, so the point spacing is exactly 0.5 Delta per axis
  everywhere on the graded grid (the IB rule; a uniform lattice would need 0.1875 mm everywhere); point volumes are the
  sub-cell volumes rescaled to the exact CAD part volume (as build_squid_volumetric_solid does);
- tethers: the production bond rule of volumetric_solid.build_squid_volumetric_solid — the point layer at each CAD
  anchor face of its own part (source_config_shell_anchor_region_ids: fixed rim region 5, bonded support region 9) is
  held — with the layer thickness = the point's own local spacing (the production rule uses one uniform spacing);
- tags: air-face layer of the membrane (points within one local spacing of the air-pressure face, the air chamber's
  lower bound) and the valve rim (flap points within one local z spacing of the flap's lower surface with r >= 28.0 mm).
UNVALIDATED engineering run: the fast mode is not yet validated against the sharp reference.
"""
import math
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

# FastFsi export: the squid geometry and its simulation_config.json are not distributed; pass --source-config
# or set SQUID_SOURCE_CONFIG (the local default path was removed).
DEFAULT_SOURCE_CONFIG = os.environ.get('SQUID_SOURCE_CONFIG')
MEMBRANE_BAND_BOTTOM_M = 1.006
MEMBRANE_BAND_TOP_M = 1.045
MEMBRANE_BAND_DZ_M = 6.0e-4
UPPER_BAND_DZ_M = 2.0e-3
VALVE_RIM_RADIUS_M = 0.0280
FIXED_RIM_REGION_ID = 5
TAG_AIR_FACE = 1
TAG_VALVE_RIM = 2


def fast_squid_grid_spec(spec, air_chamber, *, band_top_m=MEMBRANE_BAND_TOP_M, band_dz_m=MEMBRANE_BAND_DZ_M,
                         spacing_m=3.75e-4, core_half_width_m=6.0e-3, upstream_core_m=0.012, farfield_spacing_m=3.0e-3,
                         max_growth_ratio=1.2, chamber_spacing_m=1.5e-3, upper_band_dz_m=UPPER_BAND_DZ_M,
                         lateral_margin_m=0.012, valve_band=None):
    """setup.spec_with_squid_bore_grid(preset='volumetric') with the D2 membrane band (see the module docstring).
    T9 grid A (Root REQ_398): chamber_spacing_m = 7.5e-4 refines x and y across the chamber cross-section (the
    production lateral bands), and valve_band = (z_lo, z_hi, dz), e.g. (1.006, 1.013, 3.0e-4), refines z over the valve."""
    from cases.squid_setup.setup import _growth_bridge_m
    from simulation_core.fluids.grid import GradedGridSpec, RefinementRegion, build_graded_grid
    cx, cy = float(spec.monitor_center_x_m), float(spec.monitor_center_y_m)
    radius, exit_z = float(spec.nozzle_radius_m), float(spec.lip_z_m)
    coarse, growth, axial = float(farfield_spacing_m), float(max_growth_ratio), 5.0e-4
    reach = float(air_chamber.outer_radius_m) + float(lateral_margin_m)
    low, high = spec.fluid_bounds_min_m, spec.fluid_bounds_max_m
    spec = replace(spec, fluid_bounds_min_m=(max(low[0], cx - reach), max(low[1], cy - reach), low[2]),
                   fluid_bounds_max_m=(min(high[0], cx + reach), min(high[1], cy + reach), float(air_chamber.lid_z_m)))
    target = float(spacing_m) * (1.0 + 1.0e-10)
    half_width = int(math.ceil(core_half_width_m / spacing_m)) * float(spacing_m)

    def z_band(z_lo, z_hi, dz):
        return RefinementRegion((cx - radius, cy - radius, z_lo), (cx + radius, cy + radius, z_hi), (coarse, coarse, dz))

    regions = [
        RefinementRegion((cx - half_width, cy - half_width, exit_z - math.ceil(4.0 * 2.0 * radius / axial) * axial),
                         (cx + half_width, cy + half_width, exit_z + math.ceil(upstream_core_m / axial) * axial),
                         (target, target, axial * (1.0 + 1.0e-10))),
        z_band(0.984, 1.002, 1.0e-3),
    ]
    chamber_reach = float(air_chamber.inner_radius_m) + 2.0e-3
    inner = half_width + _growth_bridge_m(float(spacing_m), chamber_spacing_m, growth) + 5.0e-4
    for lo, hi in ((-chamber_reach, -inner), (inner, chamber_reach)):
        regions.append(RefinementRegion((cx + lo, cy - radius, 1.020), (cx + hi, cy + radius, 1.021),
                                        (chamber_spacing_m, coarse, coarse)))
        regions.append(RefinementRegion((cx - radius, cy + lo, 1.020), (cx + radius, cy + hi, 1.021),
                                        (coarse, chamber_spacing_m, coarse)))
    regions.append(z_band(MEMBRANE_BAND_BOTTOM_M, float(band_top_m), float(band_dz_m)))
    if valve_band is not None:
        z_lo, z_hi, dz = (float(v) for v in valve_band)
        regions.append(z_band(z_lo, z_hi, dz))
    upper_start = float(band_top_m) + _growth_bridge_m(float(band_dz_m), float(upper_band_dz_m), growth) + 5.0e-4
    if upper_start < float(air_chamber.lid_z_m):
        regions.append(z_band(upper_start, float(air_chamber.lid_z_m), float(upper_band_dz_m)))
    graded = GradedGridSpec(bounds_min_m=spec.fluid_bounds_min_m, bounds_max_m=spec.fluid_bounds_max_m,
                            farfield_spacing_m=coarse, max_growth_ratio=growth, max_cells=None,
                            refinement_regions=tuple(regions))
    grid = build_graded_grid(graded)
    return replace(spec, grid_nodes=grid.grid_nodes, cartesian_grid=None, graded_grid=graded)


GRID_VARIANTS = {
    'base': dict(),                                                        # production + D2 (Root REQ_399)
    'A': dict(chamber_spacing_m=7.5e-4, valve_band=(1.006, 1.013, 3.0e-4)),  # T9 grid A (Root REQ_398)
}


def load_squid_inputs(source_config_path=DEFAULT_SOURCE_CONFIG, *, band_top_m=MEMBRANE_BAND_TOP_M,
                      band_dz_m=MEMBRANE_BAND_DZ_M, grid_variant='base'):
    """Source config, region mapping, measured nozzle exit, air chamber, spec with the fast grid, CartesianGrid."""
    from cases.squid_setup.setup import (cartesian_grid_for_spec, source_config_nozzle_exit_geometry,
                                         spec_with_nozzle_exit)
    from cases.squid_setup.source_config import (load_source_config,
                                                 source_config_pressure_boundary_shell_mapping)
    from cases.squid_setup.spec import infer_spec
    from cases.squid_setup.volumetric_solid import source_config_squid_air_chamber
    path = Path(source_config_path)
    config = load_source_config(path)
    mapping = source_config_pressure_boundary_shell_mapping(config)
    nozzle_exit = source_config_nozzle_exit_geometry(config)
    air_chamber = source_config_squid_air_chamber(config, primary_surface_region_id=mapping.primary_shell_region_id,
                                                  secondary_surface_region_id=mapping.secondary_shell_region_id,
                                                  air_face_region_id=mapping.source_region_id)
    spec = spec_with_nozzle_exit(infer_spec(path, grid_scale=1.0), nozzle_exit)
    if grid_variant not in GRID_VARIANTS:
        raise ValueError("grid variant must be one of %s" % sorted(GRID_VARIANTS))
    spec = fast_squid_grid_spec(spec, air_chamber, band_top_m=band_top_m, band_dz_m=band_dz_m,
                                **GRID_VARIANTS[grid_variant])
    grid = cartesian_grid_for_spec(spec)
    return dict(path=path, config=config, mapping=mapping, nozzle_exit=nozzle_exit, air_chamber=air_chamber,
                spec=spec, grid=grid)


def static_obstacle(inputs, groove=True):
    """Production static obstacle (rigid regions 3, 4 + unreachable water) plus the user's chamber-wall groove
    (groove=False: the original CAD chamber wall, Root T9 A2)."""
    from cases.squid_setup.setup import build_source_config_fluid_obstacle_mask
    from cases.squid_setup.runner_groove import carve_groove
    mask, report = build_source_config_fluid_obstacle_mask(
        config=inputs['config'], source_config_path=inputs['path'], grid=inputs['grid'],
        nozzle_exit=inputs['nozzle_exit'], nozzle_exit_area_tolerance=0.05, connect_surface_seeds_to_zmin=False,
        surface_seed_zmin_connection_max_carve_cells=256)
    if not groove:
        return np.ascontiguousarray(np.asarray(mask).astype(np.int32)), dict(report, groove="off")
    carved, groove_report = carve_groove(mask, inputs['grid'])
    return np.ascontiguousarray(carved.astype(np.int32)), dict(report, **groove_report)


def graded_sublattice(grid, lo_m, hi_m, split=2):
    """CartesianGrid covering the fluid cells that intersect [lo, hi], every cell split into `split` per axis."""
    from simulation_core.fluids.grid import CartesianGrid
    faces = [np.asarray(f, dtype=np.float64) for f in (grid.cell_faces_x_m, grid.cell_faces_y_m, grid.cell_faces_z_m)]
    starts, widths = [], []
    for a in range(3):
        n = len(faces[a]) - 1
        i0 = int(np.clip(np.searchsorted(faces[a], lo_m[a], side='right') - 1, 0, n - 1))
        i1 = int(np.clip(np.searchsorted(faces[a], hi_m[a], side='left'), i0 + 1, n))
        starts.append(float(faces[a][i0]))
        widths.append(tuple(np.repeat(np.diff(faces[a][i0:i1 + 1]) / split, split).tolist()))
    return CartesianGrid(bounds_min_m=tuple(starts), cell_widths_x_m=widths[0], cell_widths_y_m=widths[1],
                         cell_widths_z_m=widths[2])


def part_points(triangles_m, grid, split=2):
    """Points (sub-cell centres inside the closed part), volumes rescaled to the CAD volume, local spacing per point."""
    from cases.squid_setup.closed_solid_cells import classify_closed_solid_cells
    corners = np.asarray(triangles_m, dtype=np.float64).reshape(-1, 3)
    pad = 3.0e-3
    sub = graded_sublattice(grid, corners.min(axis=0) - pad, corners.max(axis=0) + pad, split)
    inside, report = classify_closed_solid_cells(triangles_m, sub, close_thin_walls=False)
    i, j, k = np.nonzero(inside)
    centres = [np.asarray(c) for c in (sub.cell_centers_x_m, sub.cell_centers_y_m, sub.cell_centers_z_m)]
    widths = [np.asarray(w) for w in (sub.cell_widths_x_m, sub.cell_widths_y_m, sub.cell_widths_z_m)]
    positions = np.stack((centres[0][i], centres[1][j], centres[2][k]), axis=1)
    spacing = np.stack((widths[0][i], widths[1][j], widths[2][k]), axis=1)
    volumes = np.prod(spacing, axis=1)
    cad_volume = float(report['enclosed_volume_m3'])
    lattice_volume = float(volumes.sum())
    volumes = volumes * (cad_volume / lattice_volume)
    return positions, volumes, spacing, dict(points=int(len(positions)), cad_volume_m3=cad_volume,
                                             lattice_volume_m3=lattice_volume,
                                             lattice_to_cad_ratio=lattice_volume / cad_volume)


def _near_surface(points, spacing, triangles_m):
    """Points within their own local spacing (max over axes) of the surface (production bond-rule distance query)."""
    from scipy.spatial import cKDTree
    from cases.squid_setup.closed_solid_cells import triangle_surface_samples
    reach = spacing.max(axis=1)
    samples = triangle_surface_samples(np.asarray(triangles_m, dtype=np.float64), float(spacing.min()) / 4.0)[0]
    distance, _ = cKDTree(samples).query(points, distance_upper_bound=2.0 * float(reach.max()))
    return distance <= reach


def build_fast_particles(inputs, *, fixed_rim_region_id=FIXED_RIM_REGION_ID):
    """Material points of the valve flap (body 1) and main membrane (body 2), tethers and tags."""
    from cases.squid_setup.setup import _region_triangles_m, source_config_shell_anchor_region_ids
    from cases.squid_setup.source_config import _face_ids_for_region, source_config_part_material
    from cases.squid_setup.volumetric_solid import _require_outward_closed_body
    config, mapping, grid = inputs['config'], inputs['mapping'], inputs['grid']
    parts = {}
    for body, surface in ((2, mapping.primary_shell_region_id), (1, mapping.secondary_shell_region_id)):
        part = source_config_part_material(config, int(surface))
        if part is None:
            raise ValueError("no CAD part contains FSI surface region %s" % surface)
        parts[body] = int(part['part_region_id'])
    triangles = _region_triangles_m(config, tuple(parts.values()))
    chunks, report = [], {}
    for body in (1, 2):
        _require_outward_closed_body(triangles[parts[body]], "part %d" % parts[body])
        positions, volumes, spacing, rep = part_points(triangles[parts[body]], grid)
        chunks.append((body, positions, volumes, spacing))
        report['body_%d_part_%d' % (body, parts[body])] = rep
    body_id = np.concatenate([np.full(len(c[1]), c[0], np.int32) for c in chunks])
    positions = np.concatenate([c[1] for c in chunks])
    volumes = np.concatenate([c[2] for c in chunks])
    spacing = np.concatenate([c[3] for c in chunks])
    tether = np.zeros(len(positions), np.int32)
    anchors = source_config_shell_anchor_region_ids(config, int(fixed_rim_region_id))
    for region, tris in _region_triangles_m(config, anchors).items():
        owner = next((body for body, part in parts.items()
                      if set(_face_ids_for_region(dict(config), region)) <= set(_face_ids_for_region(dict(config), part))),
                     None)
        if owner is None:
            continue
        sel = np.flatnonzero(body_id == owner)
        held = sel[_near_surface(positions[sel], spacing[sel], tris)]
        tether[held] = 1
        report['anchor_region_%d' % region] = dict(body=owner, tethered_points=int(held.size))
    tag = np.zeros(len(positions), np.int32)
    air_face = _region_triangles_m(config, (int(mapping.source_region_id),))[int(mapping.source_region_id)]
    sel = np.flatnonzero(body_id == 2)
    tag[sel[_near_surface(positions[sel], spacing[sel], air_face)]] |= TAG_AIR_FACE
    axis = np.asarray(inputs['air_chamber'].axis_xy_m, dtype=np.float64)
    flap = np.flatnonzero(body_id == 1)
    r = np.hypot(positions[flap, 0] - axis[0], positions[flap, 1] - axis[1])
    flap_vertices = np.asarray(triangles[parts[1]], dtype=np.float64).reshape(-1, 3)
    r_vertices = np.hypot(flap_vertices[:, 0] - axis[0], flap_vertices[:, 1] - axis[1])
    z_bottom = float(flap_vertices[r_vertices >= VALVE_RIM_RADIUS_M, 2].min())
    rim = (r >= VALVE_RIM_RADIUS_M) & (positions[flap, 2] - z_bottom <= spacing[flap, 2])
    tag[flap[rim]] |= TAG_VALVE_RIM
    report.update(points=int(len(positions)), tethered=int(tether.sum()),
                  air_face_points=int(np.count_nonzero(tag & TAG_AIR_FACE)),
                  valve_rim_points=int(np.count_nonzero(tag & TAG_VALVE_RIM)), flap_bottom_z_m=z_bottom,
                  spacing_min_m=float(spacing.min()), spacing_max_m=float(spacing.max()), parts=parts)
    return dict(positions=positions, volumes=volumes, spacing=spacing, body_id=body_id, tether=tether, tag=tag), report


def build_fluid(inputs, obstacle, runtime, dt_s, *, multigrid_semi_coarsening=False):
    """Production CartesianFluidSolver on the graded grid; closed walls, z-min pressure outlet, static obstacle.
    multigrid_semi_coarsening: the solver's per-axis MG coarsening option (anisotropic graded cells)."""
    from simulation_core.fluids.solver import CartesianFluidSolver
    from simulation_core.fluids.spec import FluidDomainSpec
    grid, spec = inputs['grid'], inputs['spec']
    fluid = CartesianFluidSolver(
        FluidDomainSpec(bounds_min_m=tuple(grid.bounds_min_m), bounds_max_m=tuple(grid.bounds_max_m),
                        grid_nodes=tuple(grid.grid_nodes), density_kgm3=float(spec.water_density_kgm3),
                        viscosity_pa_s=float(spec.water_viscosity_pa_s), dt_s=float(dt_s), cartesian_grid=grid),
        runtime=runtime, multigrid_semi_coarsening=bool(multigrid_semi_coarsening))
    nodes = tuple(int(v) for v in grid.grid_nodes)
    fluid.velocity.from_numpy(np.zeros(nodes + (3,), np.float32))
    fluid.velocity_prev.from_numpy(np.zeros(nodes + (3,), np.float32))
    fluid.pressure.from_numpy(np.zeros(nodes, np.float32))
    fluid.obstacle.from_numpy(obstacle)
    fluid.clear_volume_source()
    fluid.set_velocity_dirichlet_boundary_authority("canonical")
    fluid.prepare_and_seal_velocity_dirichlet_component_ledger()
    from simulation_core.fast.fluid_hooks import FAST_FLOOD_BATCH_LEVELS, single_level_reachability_flood
    with single_level_reachability_flood():          # not the F2(a) batch kernel: its CUDA compile exceeds 16 GB host RSS
        unreached = fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(pressure_outlet_zmin=True)
    return fluid, dict(unreached_cells=int(unreached), reachability_flood_batch_levels=FAST_FLOOD_BATCH_LEVELS,
                       reachability_levels=int(getattr(fluid, 'last_hibm_pressure_reachability_sweeps', -1)))


def grid_report(grid):
    faces = [np.asarray(f) for f in (grid.cell_faces_x_m, grid.cell_faces_y_m, grid.cell_faces_z_m)]
    widths = [np.diff(f) for f in faces]
    return dict(nodes=[int(v) for v in grid.grid_nodes], cells=int(np.prod(grid.grid_nodes)),
                min_spacing_m=[float(w.min()) for w in widths], max_spacing_m=[float(w.max()) for w in widths],
                max_stretch_ratio=[float(np.max(np.maximum(w[1:] / w[:-1], w[:-1] / w[1:]))) for w in widths],
                bounds_min_m=[float(f[0]) for f in faces], bounds_max_m=[float(f[-1]) for f in faces])
