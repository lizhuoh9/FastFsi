# Changelog

## 0.1.0 (unreleased)

First version of the package layout `src/fastfsi` (replaces the earlier flat layout of this repository).

- Packages `fluid`, `solid`, `coupling`, `boundary`, `geometry`, `grid`, `io` and `simulation` with generic names; the
  fluid and the solid layers never import each other.
- CUDA device controllers of the pressure solve (`fastfsi.fluid.device_*`): the conjugate-gradient convergence checks,
  the exact-residual confirmation, the component compatibility measures and the outlet-reachability flood read one
  small device packet per poll instead of host scalar arithmetic.
- Excess mass for solids denser than the fluid (`fastfsi.coupling.excess_mass`, penalty-spring partner points).
- Boundary models: pressure chambers and pressure regions, gas regions and sealed gas pockets (isothermal gas law),
  solid occupancy, and a pneumatic chamber (a pressure inlet through a hole that a solid may cover).
- Shifted-pressure chambers carry their projection contract (`PROJECT_REQUIREMENTS`: z-min outlet, outlet backflow
  allowed, no pressure accumulation) as a class attribute; `fastfsi.boundary.outlet_guard.merge_project_requirements`
  merges it into the caller's `project()` arguments and raises when a chamber declares none or a caller value
  conflicts.  `OutletShiftGuard` checks that the solver's and the chamber's halves of the shifted outlet correction meet
  (one `after_project()` per `update()`; `FASTFSI_STRICT_INVARIANTS=1` turns a violation into a hard stop).
  `PneumaticChamber` exports the contract and reports the outlet-layer divergence; `PressureRegions` reports the
  outlet-layer and interior divergence separately.
- Checkpoint / restart: `fastfsi.io.checkpoint` (create-only NPZ checkpoints), `fastfsi.io.restart` (checkpoint
  schedule, manifest checks), `fastfsi.io.run_segments` (joins restarted segments).
- Turek–Hron example: `--checkpoint-every-s`, `--checkpoint-at`, `--restart-from`, `--diagnostic-restart`; argument
  files with the benchmark's reference parameters (`configs/fsi1.args`, `fsi2.args`, `fsi3.args`).
- Turek–Hron example: `--kappa-over-mu` defaults to 2(1 + ν) / (3(1 − 2ν)) with the preset's Poisson ratio (14/3 for
  the benchmark's ν = 0.4; previously a fixed 10), so the solid's parameters all follow the reference.
- `init_taichi` accepts `arch="cpu"` as an explicit opt-in (slow; for testing and small cases);
  `TaichiRuntimeConfig.cpu_max_num_threads` sets the CPU thread count.
- Tests: checkpoint / restart split runs, a golden regression of short Turek–Hron runs, the import graph, field and
  signature contracts, and unit tests of the excess mass, gas pockets and boundary models.
