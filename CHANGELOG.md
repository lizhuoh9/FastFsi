# Changelog

## 0.1.0 (unreleased)

First version of the package layout `src/fastfsi` (replaces the earlier flat layout of this repository).

- Packages `fluid`, `solid`, `coupling`, `boundary`, `geometry`, `grid`, `io` and `simulation` with generic names; the
  fluid and the solid layers never import each other.
- CUDA device controllers of the pressure solve (`fastfsi.fluid.device_*`): the conjugate-gradient convergence checks,
  the exact-residual confirmation, the component compatibility measures and the outlet-reachability flood read one
  small device packet per poll instead of host scalar arithmetic.
- Excess mass for solids denser than the fluid (`fastfsi.coupling.excess_mass`, penalty-spring partner points).
- Whole-body force (`fastfsi.simulation.body_force.WholeBodyForce`, read-only): the force a fixed body and the solids
  it holds exert on its mount (a load cell's reading), by control volumes (boxes of cells; a face on the domain's top
  wall over the body's footprint is the body itself) and by the direct sum (obstacle faces and that lid, minus the
  direct clamp's and the caller's point forces), as impulse-exact interval means with one host read per report.
  A domain wall's shear follows the solver's own wall flags (`no_slip_walls`, resolved by
  `IncompressibleFluidSolver.resolved_no_slip_domain_walls()`, which `predict()` now uses too): a free-slip wall, the
  laminar default on all six, carries none, in a box face and on the lid alike; a no-slip wall carries
  mu u_t / (w/2).  Obstacle faces keep that no-slip estimate: the laminar diffusion treats every fluid-obstacle face
  as no-slip with the wall on the face.
- Held deformation (`ImmersedBoundaryCoupler(held_f="frozen")`, opt-in; the default `"evolve"` is unchanged): the
  direct clamp holds a tethered point's interpolated velocity near 0 while F integrated the interpolated gradient, so a
  held region that does not move carried strain and spread its stress.  Frozen skips the F update of tethered points
  (both F-update kernels; the midpoint predict runs the Euler one), so their F stays I and their stress 0; F is set to I
  at construction and after a checkpoint load (`reset_held_deformation()`, the largest |F − I| removed kept in
  `held_f_reset`).
- Isochoric F update (`ImmersedBoundaryCoupler(f_update="isochoric")`, opt-in; the default `"full"` is unchanged): F
  integrates the interpolated velocity gradient, whose trace is not zero although the fluid is incompressible, so
  det F drifted from the point cloud's own volume and the bulk modulus turned that drift into a spurious mean stress.
  Isochoric updates F with the deviatoric gradient G − tr(G)/3 I in both F updates and renormalises it by
  det(F)^(−1/3), so det F = 1 holds to round-off (held points follow `held_f`); `f_update_report()` returns the
  largest |det F − 1| removed since the last read.
- Boundary models: pressure chambers and pressure regions, gas regions and sealed gas pockets (isothermal gas law),
  solid occupancy, and a pneumatic chamber (a pressure inlet through a hole that a solid may cover).
- Shifted-pressure chambers carry their projection contract (`PROJECT_REQUIREMENTS`: z-min outlet, outlet backflow
  allowed, no pressure accumulation) as a class attribute; `fastfsi.boundary.outlet_guard.merge_project_requirements`
  merges it into the caller's `project()` arguments and raises when a chamber declares none or a caller value
  conflicts.  `OutletShiftGuard` checks that the solver's and the chamber's halves of the shifted outlet correction meet
  (one `after_project()` per `update()`; `FASTFSI_STRICT_INVARIANTS=1` turns a violation into a hard stop).
  `PneumaticChamber` exports the contract and reports the outlet-layer divergence; `PressureRegions` reports the
  outlet-layer and interior divergence separately.
- `OutletShiftGuard` divides by a floored shift: ratio = pre / max(shift, shift_floor), where shift_floor is the shift
  of |p_ref| = 200 Pa on a free outlet face (`SHIFT_FLOOR_PRESSURE_PA`; reported as `outlet_guard_shift_floor_1ps`).
  Near a zero crossing of p_ref the plain ratio exceeded the limit at the solver tolerance, so strict mode stopped
  falsely; a real outlet defect (pre about the shift) still gives a ratio about 1.
- The projection contract is checked where it acts: `IncompressibleFluidSolver.project()` records the contract
  arguments of every call (`project_calls`, `last_project_contract`), and the outlet guard of a shifted-outlet chamber
  raises `ValueError` in `after_project()` when the projection since `update()` ran without the chamber's
  `PROJECT_REQUIREMENTS` (one host comparison per step, no device access; `verify_contract = False` only for tests
  that reproduce the defect on purpose).
- Sealing rule of the gas regions (`DeviceGasRegions`, `PneumaticChamber`; `seal_steps`, `min_cells`): a region cut
  off from the inlet gas becomes a sealed pocket only after it persisted `seal_steps` occupancy updates
  (`N_SEAL` = 5) with at least `min_cells` cells (`M_MIN` = 8); until then it stays inlet gas at the inlet pressure.
  A candidate region that is covered, reconnected or merged before it seals ends without a hard stop (a one-cell
  region from occupancy jitter used to seal and vanish two updates later: `gas_pocket_zero_volume`).  The regions and
  their counters and events are checkpointed (loading checks that the pending regions fit the region tables and that
  every cell names one of them); `seal_steps = min_cells = 1` is the earlier rule.
- Type memory of the gas regions (`DeviceGasRegions`, `type_memory`, default on): every commit records, per box cell,
  the class of its last free code (`last_free`: inlet gas, liquid or a water pocket; checkpointed).  A candidate cell
  that was material reports that class, negated, as its previous code (`HISTORY_*` in `gas_pocket_transitions`), so the
  history is a tie-breaker only: real parents decide whenever a region has one (freed cells ignored, as before), and
  only a sealed region of freed cells alone is typed by its cells' histories -- gas only: a pending region of the inlet
  gas (the sealing rule); liquid only: a water pocket; mixed or none: the old rule, a water pocket.  Every freed-only
  region is counted by its class (candidate counters `history_gas`, `history_liquid`, `history_mixed`,
  `history_none`; a `history` event for the first three; `typed_by_history` on the region's created / sealed / ended
  events).  A one-cell gas-side void that a flickering rasterised surface enclosed for one update used to become a
  water pocket, and the gas contact one update later stopped the run (`gas_region_liquid_to_gas`) although gas and
  liquid never met.
- `raster_interval` (`DeviceGasRegions`, `PneumaticChamber`; default 1): only every Nth refresh re-rasterises the
  solid; the others reuse the last occupancy while volumes, pressures and the stiffness check run at every refresh.
  A restart keeps the cadence bit for bit: `SolidOccupancy`'s checkpoint carries what a skipped refresh reuses (the
  coverage, the column sums and the last statistics).
- Operator component labels by union-find (`PressureNullspace`): the outlet nullspace graph labels each exact
  pressure-operator component by its minimum linear cell index with a lock-free union-find (one union and one
  compression kernel over the grid faces, the interface rows or, without rows, the legacy coupling slots) and certifies
  the labels with one min-label sweep triple, instead of iterating sweep batches until nothing changes; a label the
  certificate changes fails the graph closed (`component_labels_not_converged`).  The labels are the same fixed point;
  the slot passes run only when the init kernel counted slot cells.
- Capped inlet (`PneumaticChamber(inlet_model="capped")`, `fastfsi.boundary.inlet_air`; the default `"ideal"` is
  unchanged): the inlet-connected air keeps its own gas q = p_abs V (isothermal; the volume is the inlet gas's free
  volume, pending candidate regions included, plus an optional line volume) and is fed from the measured line pressure
  through round passages in series (`SeriesRestriction`; `inlet_line_elements` before the lid hole, for the robot the
  gauge's Y fitting's bore, the hose and a straight connector's bore; the lid hole alone by default):
  p_line − p_chamber = a Q + b Q|Q|, with each passage's Poiseuille friction a_i = 128 μ L_i / (π D_i⁴) (a hose
  switches to Blasius above Re 2300) and the joints' dynamic heads from the area ratios (a contraction
  0.5 (1 − A_small / A_large), an expansion (1 − A_small / A_large)², the measuring point and the chamber as large
  volumes; the 2 mm × 7 mm lid hole alone has 1.5 heads, 123 Pa at 35 mL/s at p_atm), with the free flow capped at
  |Q| ≤ Q_cap.  The line's gas is at p_abs, so every loss term scales by p_atm / p_mean with
  p_mean = p_atm + (p_line + p_chamber) / 2 (its volume flow is the free-air Q × p_atm / p_abs); the narrowest
  passage's Mach number is logged (`mach_max`; no compressibility model).  After every accepted geometry a
  backward-Euler step in q solves the law (in closed form while the hose is laminar, by bisection otherwise);
  pockets seal and reopen at p_chamber and their gas leaves or joins q (a ledger q − q₀ = inflow − exchange); a
  covered inlet has no flow, and an inlet region without volume falls back to
  the line pressure (counted and logged); a region back from no volume restarts at the line pressure,
  q = (p_line + p_atm) V (counted, in the ledger).  `inlet_gauge_pa(t, p_line)` gives the step's inlet pressure (the
  line's, or p_chamber), which the driver passes to `update` and `after_project`; a covered hole carries the line
  pressure with either model (no flow, no drop: `inlet_traction_gauge_pa`, the covered-inlet traction).  The region is
  a gas spring like a pocket: ω = sqrt(p_abs A² / (V m)) from the opt-in inlet-spring sums of
  `DeviceGasRegions(inlet_spring=True)` joins `dt_limit()`.  Its state is checkpointed with the line's parameters
  (`line_config`: a, b, Q_cap, V_line, the elements), and a chamber with the other inlet model or another line refuses
  the checkpoint, with one opt-in exception: `InletAirModel.allow_flow_cap_change_on_load` lets the flow cap alone
  differ (a sensitivity restarted from a run's checkpoint with another cap; the model keeps its own cap and records
  `flow_cap_change`); a non-positive absolute pressure is the hard stop `inlet_air_invalid_pressure`.
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
