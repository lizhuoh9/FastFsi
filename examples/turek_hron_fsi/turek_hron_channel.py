"""The Turek-Hron FSI benchmark channel (FSI1, FSI2, FSI3) for the fastfsi examples.

S. Turek, J. Hron, "Proposal for numerical benchmarking of fluid-structure interaction between an elastic object and
laminar incompressible flow", in: Fluid-Structure Interaction, Lecture Notes in Computational Science and Engineering
53, Springer (2006) 371-385.

Geometry (benchmark x along the channel, y across it): channel 2.5 m x 0.41 m; rigid cylinder of radius 0.05 m centred
at (0.2, 0.2); elastic beam 0.35 m x 0.02 m attached to the cylinder, its tip at x = 0.6 m.  The flow is quasi-2-D on a
uniform Cartesian grid: x of the solver is the span (fields uniform along it), y is the benchmark y and the solver z
runs against the benchmark x, z = 2.5 m - x, so the inlet is the z-max face and the outlet (p = 0) the z-min face.

Boundary conditions: no-slip walls at y = 0 and y = 0.41 m (directed y faces); parabolic inlet on the z-max face,
u = 1.5 U_mean 4 y (H - y) / H^2 times the cosine ramp 0.5 (1 - cos(pi t / T_ramp)) for t < T_ramp (T_ramp = 2 s in the
benchmark), the two wall corners held at zero; the cylinder is an obstacle mask (cell centres inside the circle).

Fluid: density 1000 kg/m^3, dynamic viscosity 1 Pa s.  Solid presets (benchmark table):
    FSI1: U_mean 0.2 m/s, rho_s 1000 kg/m^3, E 1.4 MPa, nu 0.4
    FSI2: U_mean 1.0 m/s, rho_s 10000 kg/m^3, E 1.4 MPa, nu 0.4
    FSI3: U_mean 2.0 m/s, rho_s 1000 kg/m^3, E 5.6 MPa, nu 0.4

No `from __future__ import annotations` (Taichi kernel module).
"""
import math
from dataclasses import dataclass

import numpy as np
import taichi as ti

from fastfsi.fluid.solver import IncompressibleFluidSolver
from fastfsi.grid.specification import FluidDomainSpecification

WALL_COMPONENTS = 7          # all three velocity components prescribed on a wall / inlet face


@dataclass(frozen=True)
class ChannelBenchmark:
    """Turek-Hron channel, cylinder, beam, flow and solid parameters (SI units)."""
    channel_length_m: float = 2.5
    channel_height_m: float = 0.41
    span_m: float = 0.05
    cylinder_center_x_m: float = 0.2
    cylinder_center_y_m: float = 0.2
    cylinder_radius_m: float = 0.05
    beam_length_m: float = 0.35
    beam_thickness_m: float = 0.02
    beam_tip_x_m: float = 0.6
    mean_inlet_velocity_mps: float = 0.2
    inlet_ramp_time_s: float = 2.0
    fluid_density_kgm3: float = 1000.0
    fluid_viscosity_pa_s: float = 1.0
    solid_density_kgm3: float = 1000.0
    young_modulus_pa: float = 1.4e6
    poisson_ratio: float = 0.4
    grid_nodes: tuple = (4, 48, 288)
    dt_s: float = 5.0e-3
    multigrid_semi_coarsening: bool = False


PRESETS = {
    "fsi1": dict(mean_inlet_velocity_mps=0.2, solid_density_kgm3=1000.0, young_modulus_pa=1.4e6, poisson_ratio=0.4,
                 dt_s=5.0e-3),
    "fsi2": dict(mean_inlet_velocity_mps=1.0, solid_density_kgm3=10000.0, young_modulus_pa=1.4e6, poisson_ratio=0.4,
                 dt_s=1.0e-3),
    "fsi3": dict(mean_inlet_velocity_mps=2.0, solid_density_kgm3=1000.0, young_modulus_pa=5.6e6, poisson_ratio=0.4,
                 dt_s=1.0e-3),
}


def preset(name, **overrides):
    """The benchmark configuration `name` (fsi1, fsi2, fsi3) with keyword overrides."""
    if name not in PRESETS:
        raise ValueError("unknown preset %r (one of %s)" % (name, sorted(PRESETS)))
    return ChannelBenchmark(**{**PRESETS[name], **overrides})


def solver_z(x_m, config):
    """Solver z of the benchmark coordinate x (the solver z runs against x)."""
    return float(config.channel_length_m) - float(x_m)


def cylinder_center_solver_m(config):
    """(y, z) of the cylinder centre in solver coordinates."""
    return float(config.cylinder_center_y_m), solver_z(config.cylinder_center_x_m, config)


def beam_root_x_m(config):
    return float(config.beam_tip_x_m) - float(config.beam_length_m)


def beam_box_solver_m(config):
    """The beam's bounding box in solver coordinates, ((0, y_min, z_tip), (span, y_max, z_root))."""
    half_thickness = 0.5 * float(config.beam_thickness_m)
    y_min = float(config.cylinder_center_y_m) - half_thickness
    y_max = float(config.cylinder_center_y_m) + half_thickness
    z_tip = solver_z(config.beam_tip_x_m, config)
    z_root = solver_z(beam_root_x_m(config), config)
    return (0.0, y_min, z_tip), (float(config.span_m), y_max, z_root)


def cell_spacing_m(config):
    """Uniform cell size (dx, dy, dz)."""
    nx, ny, nz = (int(value) for value in config.grid_nodes)
    return (float(config.span_m) / float(nx), float(config.channel_height_m) / float(ny),
            float(config.channel_length_m) / float(nz))


def full_bounds(config):
    return (0.0, 0.0, 0.0), (float(config.span_m), float(config.channel_height_m), float(config.channel_length_m))


def inlet_ramp_factor(t_s, config):
    """0.5 (1 - cos(pi t / T_ramp)) for 0 < t < T_ramp, 0 before, 1 after (and for T_ramp <= 0)."""
    ramp_time_s = float(config.inlet_ramp_time_s)
    if ramp_time_s <= 0.0 or float(t_s) >= ramp_time_s:
        return 1.0
    if float(t_s) <= 0.0:
        return 0.0
    return 0.5 * (1.0 - math.cos(math.pi * float(t_s) / ramp_time_s))


def cylinder_obstacle_mask(config):
    """i32 mask of the cells whose centre lies inside the cylinder (uniform along the span)."""
    nx, ny, nz = (int(value) for value in config.grid_nodes)
    _, dy, dz = cell_spacing_m(config)
    center_y, center_z = cylinder_center_solver_m(config)
    y_centers = (np.arange(ny, dtype=np.float64) + 0.5) * dy
    z_centers = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    radius_sq = float(config.cylinder_radius_m) ** 2
    distance_sq = (y_centers[:, None] - center_y) ** 2 + (z_centers[None, :] - center_z) ** 2
    mask_yz = (distance_sq <= radius_sq).astype(np.int32)
    return np.broadcast_to(mask_yz[None, :, :], (nx, ny, nz)).copy()


def build_fluid(config, runtime):
    """The incompressible fluid on the channel grid at rest, with the cylinder obstacle and canonical boundary
    authority (the boundary faces are written by write_channel_velocity_faces)."""
    bounds_min, bounds_max = full_bounds(config)
    fluid = IncompressibleFluidSolver(
        FluidDomainSpecification(bounds_min_m=bounds_min, bounds_max_m=bounds_max, grid_nodes=config.grid_nodes,
                                 density_kgm3=config.fluid_density_kgm3, viscosity_pa_s=config.fluid_viscosity_pa_s,
                                 dt_s=config.dt_s),
        runtime=runtime, multigrid_semi_coarsening=bool(config.multigrid_semi_coarsening))
    grid_nodes = tuple(int(value) for value in config.grid_nodes)
    fluid.velocity.from_numpy(np.zeros((*grid_nodes, 3), dtype=np.float32))
    fluid.velocity_prev.from_numpy(np.zeros((*grid_nodes, 3), dtype=np.float32))
    fluid.pressure.from_numpy(np.zeros(grid_nodes, dtype=np.float32))
    fluid.obstacle.from_numpy(cylinder_obstacle_mask(config))
    fluid.clear_volume_source()
    fluid.set_velocity_dirichlet_boundary_authority("canonical")
    return fluid


@ti.kernel
def _channel_velocity_faces_kernel(y_face_active_component_mask: ti.template(), y_face_value_mps: ti.template(),
                                   z_face_active_component_mask: ti.template(), z_face_value_mps: ti.template(),
                                   nx: ti.i32, ny: ti.i32, dy: ti.f32, height_m: ti.f32, peak_scale: ti.f32):
    # walls: the two directed y faces
    for side, i, k in y_face_active_component_mask:
        y_face_active_component_mask[side, i, k] = WALL_COMPONENTS
        y_face_value_mps[side, i, k] = ti.Vector([0.0, 0.0, 0.0])
    # parabolic inlet: the directed z-max face; the wall corners stay at zero
    for i, j in ti.ndrange(nx, ny):
        y = (ti.cast(j, ti.f32) + 0.5) * dy
        parabola = 4.0 * y * (height_m - y) / (height_m * height_m)
        target_z = -peak_scale * parabola
        if j == 0 or j == ny - 1:
            target_z = 0.0
        z_face_active_component_mask[1, i, j] = WALL_COMPONENTS
        z_face_value_mps[1, i, j] = ti.Vector([0.0, 0.0, target_z])


@ti.kernel
def _inlet_face_values_kernel(z_face_value_mps: ti.template(), nx: ti.i32, ny: ti.i32, dy: ti.f32, height_m: ti.f32,
                              peak_scale: ti.f32):
    # the inlet magnitude only: same parabola and wall corners as _channel_velocity_faces_kernel
    for i, j in ti.ndrange(nx, ny):
        y = (ti.cast(j, ti.f32) + 0.5) * dy
        parabola = 4.0 * y * (height_m - y) / (height_m * height_m)
        target_z = -peak_scale * parabola
        if j == 0 or j == ny - 1:
            target_z = 0.0
        z_face_value_mps[1, i, j] = ti.Vector([0.0, 0.0, target_z])


def _inlet_peak_scale(config, t_s):
    return 1.5 * float(config.mean_inlet_velocity_mps) * inlet_ramp_factor(t_s, config)


def write_channel_velocity_faces(fluid, config, t_s):
    """Write the wall and inlet faces for time t_s and rebuild the boundary structures that depend on them (pressure
    nullspace graph, velocity-Dirichlet registry, outlet reachability: invalidated, then the registry is sealed; the caller
    re-marks the outlet reachability)."""
    nx, ny, _ = (int(value) for value in config.grid_nodes)
    _, dy, _ = cell_spacing_m(config)
    fluid._invalidate_pressure_nullspace_component_graph()
    fluid._invalidate_velocity_dirichlet_component_registry()
    fluid._invalidate_pressure_reachability()
    _channel_velocity_faces_kernel(
        fluid.external_velocity_boundary_y_face_active_component_mask,
        fluid.external_velocity_boundary_y_face_value_mps,
        fluid.external_velocity_boundary_z_face_active_component_mask,
        fluid.external_velocity_boundary_z_face_value_mps,
        int(nx), int(ny), float(dy), float(config.channel_height_m), float(_inlet_peak_scale(config, t_s)))
    fluid.prepare_and_seal_velocity_dirichlet_component_registry()


def update_inlet_face_values(fluid, config, t_s):
    """Rewrite only the inlet face values for time t_s.  Precondition: write_channel_velocity_faces ran once on this
    fluid; the magnitude changes no mask, so the nullspace graph, the registry and the reachability stay valid."""
    nx, ny, _ = (int(value) for value in config.grid_nodes)
    _, dy, _ = cell_spacing_m(config)
    _inlet_face_values_kernel(fluid.external_velocity_boundary_z_face_value_mps, int(nx), int(ny), float(dy),
                              float(config.channel_height_m), float(_inlet_peak_scale(config, t_s)))
