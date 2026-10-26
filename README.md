# FastFsi

Fluid–structure interaction on the GPU with [Taichi](https://www.taichi-lang.org/): an incompressible fluid on a
staggered (MAC) Cartesian grid is coupled to deformable solids, represented by material points, through a diffuse
immersed boundary. This is research code: it is tested for consistency, not validated for engineering use.

## What is in the package

| Package | Contents |
|---|---|
| `fastfsi.fluid` | Incompressible Navier–Stokes on a graded Cartesian MAC grid: MUSCL-TVD momentum advection, viscous terms, projection with a multigrid-preconditioned conjugate-gradient pressure solve (optionally hypre PFMG), a pressure outlet with outlet-reachability marking, velocity Dirichlet faces and obstacle masks. On CUDA the solver's scalar convergence checks run on the device (`device_*` modules). |
| `fastfsi.solid` | Material points with a neo-Hookean material in 3-D and in a plane (quasi-2-D), state guards (a non-finite state or J ≤ 0 stops the run), penalty contact against rigid level sets (opt-in: each point's contact distance capped at its rest distance to the walls, so nothing is in contact at rest). |
| `fastfsi.coupling` | The immersed-boundary coupling (IB_4 or IB_3 regularized delta), explicit Euler or the midpoint rule, clamps by direct forcing or penalty tethers, the excess mass of solids denser than the fluid (penalty-spring partner points), and opt-in variants of the deformation-gradient update (isochoric; held points kept at F = I). |
| `fastfsi.boundary` | Pressure chambers and per-cell pressure regions, gas regions and sealed gas pockets (isothermal gas law), the solid occupancy of grid cells, and a pneumatic chamber: a pressure inlet through a hole that a solid may cover, either ideal (the line pressure at the hole) or capped (opt-in: the inlet air's own isothermal mass balance behind a series line of round passages, with a free-flow cap). |
| `fastfsi.geometry` | Signed distance and masks of closed triangulated solids. |
| `fastfsi.grid` | Graded Cartesian grids and the fluid-domain specification. |
| `fastfsi.io` | Create-only checkpoints (NPZ, a sha256 per array, type-exact host state, no pickle), restart checks, the joining of restarted segments, and VTK output for ParaView. |
| `fastfsi.simulation` | Taichi runtime set-up and monitors: control-volume forces, and the read-only whole-body force that a fixed body and the solids it holds exert on its mount (a load cell's reading). |

## Installation

Python 3.10 or newer. fastfsi runs on NVIDIA GPUs through Taichi's CUDA backend (it was run on GPUs of compute
capability 7.5 and 8.6). The same code also runs on the CPU as an explicit opt-in (`--arch cpu` in the example,
`TaichiRuntimeConfig(arch="cpu")` in code): slow, meant for testing and small cases. The tests run on the CPU with one
thread (`tests/cpu_runtime.py`), so two runs give the same bits.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Optional: hypre PFMG

The default pressure solve, a multigrid-preconditioned conjugate gradient on the GPU, needs no external library. As an
alternative, `fastfsi.fluid.pressure_hypre.install_hypre_pressure_solver(fluid, prefix=...)` replaces the linear solve
of one fluid instance by hypre's Struct PCG with a PFMG (or SMG) preconditioner, through a thin ctypes layer (no
compiled extension).

- Version: hypre 3.2.0 (https://github.com/hypre-space/hypre), Apache-2.0 or MIT.
- Build: sequential and shared (MPI off, 32-bit `HYPRE_Int`, double precision); the layer refuses other builds. The CPU
  build used here: `cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DHYPRE_ENABLE_MPI=OFF
  -DHYPRE_ENABLE_OPENMP=OFF -DHYPRE_ENABLE_CUDA=OFF`. The CUDA build (CUDA 12.8): `cmake -DCMAKE_BUILD_TYPE=Release
  -DHYPRE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<sm> -DHYPRE_ENABLE_MPI=OFF -DBUILD_SHARED_LIBS=ON
  -DHYPRE_ENABLE_SHARED=ON -DHYPRE_ENABLE_UMPIRE=OFF -DHYPRE_ENABLE_DEVICE_MALLOC_ASYNC=ON`, with `<sm>` the GPU's
  compute capability (75 for Turing, e.g. a GTX 1660; 86 for an RTX 3080 / 3090).
- Location: the `prefix` argument, else the environment variable `HYPRE_PREFIX`; the layer loads
  `<prefix>/lib/libHYPRE.so` and reads `<prefix>/include/HYPRE_config.h` to tell a CUDA build from a CPU build. The CUDA
  runtime is the copy libHYPRE itself loaded (or the library named by `HYPRE_CUDART`).
- Data path: `data_path="host"` (default) stages the matrix and vectors through host memory; `data_path="pointer"`
  hands hypre the Taichi arrays' device pointers (CUDA build with Taichi on CUDA) or host pointers (CPU build).
- The Turek–Hron example uses the default conjugate-gradient solve.

## Turek–Hron FSI benchmarks

`examples/turek_hron_fsi` runs FSI1, FSI2 and FSI3 (Turek & Hron 2006) as a quasi-2-D channel: four cells along the
span with fields uniform along it. The argument files in `configs/` select the benchmark presets, which hold the
reference parameters of the benchmark table (densities, Young's modulus, Poisson ratio, viscosity, mean inflow, the 2 s
inlet ramp), and set only the grid, the run length and the row cadence; nothing is tuned. The solid's volumetric
modulus follows from the reference Poisson ratio: `--kappa-over-mu` defaults to 2(1 + ν) / (3(1 − 2ν)), 14/3 for the
benchmark's ν = 0.4.

```bash
cd examples/turek_hron_fsi
python -B run_turek_hron_fsi.py @configs/fsi3.args --output runs/fsi3
```

A run writes `run_meta.json` (parameters, grid, layout), `history.csv` (time step and its limiter, the displacement of
the material point nearest to point A (its index and reference position are in `run_meta.json`), drag and lift per unit
span from a control-volume momentum balance, the range of J, the conjugate-gradient iterations, the volume drift of the
beam, excess-mass slip), `final.json` (status), and optionally a VTK series (`--vtk-interval-s`).

Long runs can be split by checkpoints and continued in a new output directory; the continuation reproduces the straight
run bit for bit on the same device:

```bash
python -B run_turek_hron_fsi.py @configs/fsi2.args --end-time 5 --checkpoint-every-s 1 --output runs/fsi2_a
python -B run_turek_hron_fsi.py @configs/fsi2.args --end-time 10 --output runs/fsi2_b \
    --restart-from runs/fsi2_a/checkpoints/checkpoint_s<step>.npz
python -m fastfsi.io.run_segments runs/fsi2 runs/fsi2_a runs/fsi2_b --row-interval 10
```

A restart refuses a checkpoint whose set-up differs (any argument other than the output, the run length, the
checkpoint and VTK cadences and the device; the grid; the beam set-up) or whose code or library versions differ;
`--diagnostic-restart` accepts a code difference only into a directory whose name contains `DIAGNOSTIC`.

### Reference values

FeatFlow results on the finest level of the benchmark pages (point A displacement in m, drag and lift in N per unit
depth; periodic cases as mean ± amplitude [frequency in Hz]); the FeatFlow reference time series of FSI2 and FSI3 use
the same time steps. The original proposal (Turek & Hron 2006) lists slightly different values from coarser
computations (second table).

| FeatFlow, finest level | u_x(A) | u_y(A) | Drag | Lift |
|---|---|---|---|---|
| FSI1 (level 7+0, steady) | 2.2705e-5 | 8.2088e-4 | 14.294 | 0.7637 |
| FSI2 (level 4+0, Δt = 0.0005) | −1.485e-2 ± 1.270e-2 [3.86] | 1.30e-3 ± 8.16e-2 [1.93] | 215.06 ± 77.65 [3.86] | 0.61 ± 237.8 [1.93] |
| FSI3 (level 4+0, Δt = 0.00025) | −2.88e-3 ± 2.72e-3 [10.93] | 1.47e-3 ± 3.499e-2 [5.46] | 460.5 ± 27.74 [10.93] | 2.50 ± 153.91 [5.46] |

| Turek & Hron 2006 | u_x(A) | u_y(A) | Drag | Lift |
|---|---|---|---|---|
| FSI1 | 2.27e-5 | 8.209e-4 | 14.295 | 0.7638 |
| FSI2 | −1.458e-2 ± 1.244e-2 [3.8] | 1.23e-3 ± 8.06e-2 [2.0] | 208.83 ± 73.75 [3.8] | 0.88 ± 234.2 [2.0] |
| FSI3 | −2.69e-3 ± 2.53e-3 [10.9] | 1.48e-3 ± 3.438e-2 [5.3] | 457.3 ± 22.66 [10.9] | 2.22 ± 149.78 [5.3] |

## Known limitations

- **Diffuse interface.** The IB_4 kernel spreads forces and interpolates velocities over about four cells: pressure
  jumps across thin walls are smoothed and thin gaps leak unless they span several cells.
- **Incompressible immersed solid.** The single-velocity coupling moves the solid with the divergence-free fluid
  velocity, so J stays close to 1 and the volumetric modulus mainly limits drift. The Turek–Hron beam (Saint
  Venant–Kirchhoff, ν = 0.4) is compressible; the immersed beam bends stiffer (higher frequency, smaller amplitude).
- **Deformation-gradient drift.** The default F update integrates the interpolated velocity gradient, which is not
  trace-free, so the stored det F drifts from the point cloud's own volume over long runs (the history reports the
  range of J). The opt-in isochoric update holds det F = 1 to round-off.
- **Excess mass by penalty springs.** A solid denser than the fluid carries its excess mass on partner points tied to
  the material points by stiff springs; the finite stiffness allows a small slip (reported in the history), and the
  spring bound can limit the time step.
- **Gravity on the excess mass is not modelled.**
- **Inlet line.** The default ideal inlet imposes the line pressure at the hole, with no loss. The opt-in capped inlet
  adds the line's friction and junction losses and a free-flow cap, but no pump or valve dynamics and no
  compressibility in the line (the narrowest passage's Mach number is only logged).
- **Membrane volume error.** Thin membranes of material points gain or lose enclosed volume over a run; drivers should
  monitor the reported volume error and stop at a threshold.
- **Explicit elasticity.** The time step is limited by the elastic shear-wave CFL condition (and by the excess-mass
  spring bound), so stiff solids need many small steps.
- **Quasi-2-D benchmarks.** The Turek–Hron cases are 2-D problems run on the 3-D solver with four span cells.
- **Reproducibility.** GPU runs are not bit-reproducible from run to run (floating-point atomics); CPU runs with one
  thread are.
- One GPU per run; no distributed memory and no adaptive mesh refinement.

## Tests

```bash
pytest                 # everything; the checkpoint split test runs the example in separate processes (minutes)
pytest -m "not slow"   # without it
```

## References

- S. Turek, J. Hron, Proposal for numerical benchmarking of fluid-structure interaction between an elastic object and
  laminar incompressible flow, in: Fluid-Structure Interaction, Lecture Notes in Computational Science and Engineering
  53, Springer (2006) 371–385.
- FeatFlow FSI benchmark results:
  https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_tests/fsi_fsi_tests.html
- C. S. Peskin, The immersed boundary method, Acta Numerica 11 (2002) 479–517.
- A. M. Roma, C. S. Peskin, M. J. Berger, An adaptive version of the immersed boundary method, Journal of
  Computational Physics 153 (1999) 509–534.
- Y. Kim, C. S. Peskin, Penalty immersed boundary method for an elastic boundary with mass, Physics of Fluids 19
  (2007) 053103.
- Y. Hu, Y. Fang, Z. Ge, Z. Qu, Y. Zhu, A. Pradhana, C. Jiang, A moving least squares material point method with
  displacement discontinuity and two-way rigid body coupling, ACM Transactions on Graphics 37(4) (2018) 150.
- Methods of [IBAMR](https://github.com/IBAMR/IBAMR) and [taichi_mpm](https://github.com/yuanming-hu/taichi_mpm) are
  referenced in the code; see [NOTICE](NOTICE).

See [CHANGELOG.md](CHANGELOG.md) for the changes and [ROADMAP.md](ROADMAP.md) for planned work.
