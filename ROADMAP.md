# Roadmap

Planned or considered work, roughly in order. Nothing here is promised.

## Validation

- Turek–Hron FSI1, FSI2 and FSI3 with the reference parameters (`examples/turek_hron_fsi/configs`) against the FeatFlow
  results: harmonic fits of point A, drag and lift over the periodic window, with grid and time-step studies.
- Sample point A exactly: x_A = x_p + F_p (X_A − X_p) from the nearest particle's deformation gradient, or average the
  two particles symmetric about the midline and extrapolate to the tip face; test with a rigid rotation. The bias of
  the nearest-particle probe is measured at about 3 % on u_x in FSI3, so the fix matters for the u_x comparison.
- Quantify the effect of the incompressible immersed solid on the compressible benchmark beam (bending stiffness,
  frequency, amplitude).

## Performance (GPU)

- The hypre pressure solve on device data (no host staging of the right-hand side and the solution).
- Fewer launches and host synchronisations per step: fused predictor kernels, on-device convergence checks in every
  pressure path, setup reuse between steps when the topology is unchanged.
- Multigrid and PFMG settings at unchanged tolerances.
- Taichi's offline kernel cache for production runs (a cached run must reproduce a fresh compile bit for bit).

## Physics and numerics

- Implicit (or semi-implicit) elasticity, to remove the elastic CFL limit on the time step.
- Gravity on the excess mass of dense solids.
- An inlet resistance (orifice loss) model for the pneumatic chamber.
- Better volume conservation of thin membranes.
- A sharp-interface option for thin walls.

## Software

- One set of checkpoint / restart helpers (`fastfsi.io.restart`) for every driver.
- Split the largest modules into smaller units (`fluid/solver.py` about 4,000 lines, `fluid/pressure_projection.py`
  about 2,300, `fluid/advection.py` about 1,600), and fold the three tiny fluid modules (`fluid/outlet_budget.py`,
  `fluid/coupling_constants.py`, `fluid/interface_constants.py`) into the modules that use them.
- More examples and API documentation.
- Multiple GPUs and adaptive mesh refinement are not planned yet.
