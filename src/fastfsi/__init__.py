"""fastfsi: fluid-structure interaction with a diffuse immersed boundary and material-point solids on graded Cartesian
grids, written in Taichi.

Packages (each imports only the ones below it):
  grid        MAC grid geometry and the fluid domain specification
  fluid       incompressible Navier-Stokes on the MAC grid: advection, viscosity, pressure projection and solvers
  solid       material points, hyperelastic materials, penalty contact with rigid surfaces
  coupling    immersed-boundary kernels, interpolation and spreading, face validity, direct-forcing clamps
  boundary    pressure chambers and channel inlets
  geometry    surface meshes, signed distance fields, point seeding
  io          VTK series and history files
  simulation  configuration, time stepping, monitors

Results are engineering results of a research code; validate them against the reference you trust before use.

Known limitations:
  - A solid denser than the fluid carries its excess mass (rho_s - rho_f) V0 on spring-tied partner points
    (coupling.excess_mass).  Gravity on that excess mass is not modelled: the net weight of a body denser than the fluid
    is missing (about 1 Pa on a 1 mm sheet with rho_s - rho_f = 100 kg/m^3).  The springs are finite, so partners lag
    their points (the slip is reported), and the mass uses the reference volume, not the current volume.
  - Solids lighter than the fluid (rho_s < rho_f) are refused.
  - The spring's explicit time-step bound covers the spring kick only, not the pressure projection with inlet, outlet
    or air constraints.
"""

__version__ = "0.1.0.dev0"
