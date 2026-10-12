# FastFsi

Fast fluid-structure interaction (FSI) for soft bodies in water, on the GPU with [Taichi](https://github.com/taichi-dev/taichi).

* **Coupling:** diffuse immersed boundary (IB, in the style of IB/IBFE). A single velocity field covers fluid and
  solid. Forces are spread and velocities interpolated with the 4-point Peskin kernel (IB_4). On graded grids the
  kernel works in index space and gets a first-moment (RKPM) correction.
* **Solid:** material points (MPM-style) of a neo-Hookean body: modified invariants plus a volumetric penalty
  kappa/2 (J - 1)^2. The solid has no momentum equation of its own. Its elastic stress is spread to the fluid as a
  body force; the points move with the interpolated velocity, and F follows the interpolated velocity gradient.
* **Fluid:** incompressible Navier-Stokes on a staggered (MAC), optionally graded Cartesian grid. A finite-volume
  pressure projection is solved with CG preconditioned by geometric multigrid or, optionally, with hypre (PCG
  preconditioned by PFMG).
* **Extras for the squid soft robot:** an air chamber as a shifted-penalty pressure boundary, physical penalty
  contact between the flaps and the rigid parts on a signed-distance field, and VTK output.

> **Status: UNVALIDATED research prototype.** Results have not been validated against experiments or against a
> sharp-interface reference. Every run writes this label into its outputs. Use the code to explore and prototype,
> not to make engineering decisions.

## Contents

| path | what it is |
|---|---|
| `simulation_core/fast/` | the fast mode: `ib_mpm_fast3d.py` (3-D diffuse IB + material points), `ib_mpm_fast.py` (quasi-2-D variant, IB_4 kernel), `graded_kernel.py` (graded-grid kernel, face mask), `air_chamber_fast.py`, `rigid_contact.py`, `fluid_hooks.py`, `hypre_pressure.py` (optional hypre pressure solve), `vtk_export_fast.py` |
| `simulation_core/fluids/` | the Cartesian fluid solver (`solver.py`), grid, spec, CUDA device controllers (`gpu_*.py`) |
| `simulation_core/fluids/shared_definitions.py` | verbatim copies of the few definitions `fluids/*` takes from the sharp-interface coupling package of the original project (that package is not included) |
| `simulation_core/diagnostics/`, `simulation_core/profiles/`, `simulation_core/k9_fast_hook.py` | Taichi runtime set-up, time-step helpers, the physics/execution profile the solver reads, a fast path for Taichi's `data_oriented` attribute hook |
| `cases/turek_fast.py` (+ `turek_fast_case.py`, `turek_fast_kernels.py`) | Turek-Hron FSI1/FSI2/FSI3 benchmark driver (quasi-2-D; the geometry is generated) |
| `cases/squid_fast.py` (+ `squid_fast_setup.py`, `cases/squid_setup/`) | squid soft-robot driver (you supply the geometry and its configuration) |
| `tests/fast/` | CPU tests |
| `tools/cpu_runtime.py` | CPU harness for the tests and for CPU runs of the drivers |

`cases/turek_fast_case.py`, `cases/turek_fast_kernels.py`, `cases/squid_setup/*.py` and
`simulation_core/fluids/shared_definitions.py` hold verbatim copies of definitions from modules of the original
project that are not included here. Every copied block names its source file, its line range and the source file's
sha256.

## Install

Python 3.10, Linux x86_64. A CUDA-capable NVIDIA GPU is needed for real runs. The tests run on the CPU.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt     # numpy, scipy, taichi 1.7.4, trimesh, pytest (pinned)
```

### Optional: hypre for `--pressure-solver hypre_pfmg`

`simulation_core/fast/hypre_pressure.py` loads `libHYPRE.so` through ctypes from the prefix given by
`--hypre-prefix` or by the environment variable `HYPRE_PREFIX`. There is no default location. It needs a sequential
build (MPI off), 32-bit `HYPRE_Int` and double precision, which is the default configuration. Build hypre v3.2.0
from the official source (tag `v3.2.0` of https://github.com/hypre-space/hypre) with CMake.

GPU build. These are the flags we used on RTX 3080 / 3090 cards (compute capability 8.6); set
`CMAKE_CUDA_ARCHITECTURES` to match your card:

```bash
cmake -S hypre/src -B hypre-build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DHYPRE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DHYPRE_ENABLE_MPI=OFF \
  -DBUILD_SHARED_LIBS=ON -DHYPRE_ENABLE_SHARED=ON -DHYPRE_ENABLE_UMPIRE=OFF \
  -DHYPRE_ENABLE_DEVICE_MALLOC_ASYNC=ON -DCMAKE_INSTALL_PREFIX=$HOME/opt/hypre-3.2.0-cuda
cmake --build hypre-build-cuda -j && cmake --install hypre-build-cuda
export HYPRE_PREFIX=$HOME/opt/hypre-3.2.0-cuda
```

CPU build (for the hypre tests):

```bash
cmake -S hypre/src -B hypre-build-cpu -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DHYPRE_ENABLE_MPI=OFF -DHYPRE_ENABLE_OPENMP=OFF -DHYPRE_ENABLE_CUDA=OFF \
  -DHYPRE_ENABLE_HYPRE_BLAS=ON -DHYPRE_ENABLE_HYPRE_LAPACK=ON \
  -DHYPRE_BUILD_EXAMPLES=OFF -DHYPRE_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$HOME/opt/hypre-3.2.0-cpu
cmake --build hypre-build-cpu -j && cmake --install hypre-build-cpu
```

hypre replaces only the linear solve inside the fluid's pressure projection. Its result is accepted only when the
solver's own residual check passes, so the CG remains the reference.

## Running

All drivers write into a new output directory (it must not exist yet) and label every output UNVALIDATED.

### Turek-Hron FSI3 (no input files)

```bash
# GPU
python -B -m cases.turek_fast --preset fsi3 --grid 4,96,576 --end-time 4.0 --output runs/fsi3 --arch cuda
# CPU (slow; for checks): a few steps through the CPU harness
python -B -m tools.cpu_runtime --module cases.turek_fast --preset fsi3 --grid 4,96,576 --end-time 1.0 \
    --max-steps 5 --output runs/fsi3_cpu --arch cpu
```

Outputs: `history.csv` (time, time step and its limiter, tip displacement of point A, drag and lift per unit span
from a control-volume momentum balance, J range, volume error, CG iterations, wall time per step), `run_meta.json`
and `final.json`. Options: `--preset fsi1|fsi2|fsi3`, `--grid nx,ny,nz` (nx = 4 cells across the span, quasi-2-D),
`--ramp-time`, `--cfl-elastic`, `--cfl-advective`, `--kappa-over-mu`, `--spacing-over-h`, `--max-steps`,
`--row-interval` (see `python -m cases.turek_fast --help`).

### Squid soft robot (bring your own geometry)

The squid geometry and its configuration are **not** distributed. `--source-config` (or the environment variable
`SQUID_SOURCE_CONFIG`) points to the `simulation_config.json` exported by the original project's pre-processor. It
holds the surface mesh (STL, or STEP with its surface-mesh cache), the named regions, the materials and the
chamber-pressure load. `cases/squid_fast_setup.py` fixes the roles of the regions and some geometry constants for
the authors' robot: region 1 is the valve flap, region 2 the main membrane, regions 3 and 4 are rigid, region 5 is
the fixed rim. It also fixes the membrane band heights and the valve-rim radius. Adapt these for another geometry.

```bash
python -B -m cases.squid_fast --source-config path/to/simulation_config.json --output runs/squid \
    --end-time 0.05 --arch cuda [--pressure-solver hypre_pfmg --hypre-prefix $HYPRE_PREFIX] \
    [--vtk-points-interval-s 0.005 --vtk-3d-interval-s 0.05] [--snapshot-interval-s 0.01] [--sdf-cache-dir DIR]
python -B -m cases.squid_fast --source-config path/to/simulation_config.json --output runs/squid_setup \
    --arch cuda --setup-only          # build and check the grid, obstacle and material points only
```

Outputs: `setup_report.json`, `history.csv`, `valve_log.jsonl`, `run_meta.json`, `final.json`, and on request
`snapshots/` (npz) and `vtk/` (ParaView `.pvd` collections). Hard stops (non-finite state, velocity or pressure
bound, volume error, empty one-sided kernel support) write `hard_stop.json`.

## Tests (CPU)

```bash
python -B -m tools.cpu_runtime tests/fast -q
```

The production runtime is GPU-only, so the tests go through the CPU harness `tools/cpu_runtime.py`. It sets up
Taichi on one CPU thread and changes only the backend selection. Some tests need inputs that are not part of this
repository and are skipped without them:

* `SQUID_SOURCE_CONFIG=path/to/simulation_config.json` enables the four squid driver tests and one cross-check of
  the air-chamber test;
* `HYPRE_PREFIX=path/to/hypre-cpu-prefix` enables the hypre tests.

## Dormant code paths and unsupported options

* **The fluid solver is shared.** `simulation_core/fluids/solver.py` is the fluid solver of the original project,
  shared with its sharp-interface (HIBM) FSI mode. It still contains that mode's code paths: sharp-interface pressure
  rows, marker constraints, component ledgers and the related reports. The fast mode never activates them, and the
  sharp-interface modules that drive them are not included.
* **Execution switches stay off.** Leave every switch of `UNIFIED_PHYSICS.execution` in
  `simulation_core/profiles/soft_body.py` (`native_dispatch` and the `device_*` switches) at its default `False`.
  `simulation_core/native_dispatch/` is not included, so turning `native_dispatch` on fails with an `ImportError`.
* **Lazy imports with no target.** These imports sit inside functions that the fast mode does not call, and their
  targets are not part of this repository:
  * `simulation_core/fluids/solver.py`, `CartesianFluidSolver._solve_pressure_poisson_fv_cg`:
    `simulation_core.native_dispatch.wave1` (only with `native_dispatch` on)
  * `simulation_core/fluids/gpu_fluid_host.py`, `solve_with_native_regions`: `simulation_core.native_dispatch.regions`
    (only with `native_dispatch` on)
  * `simulation_core/profiles/soft_body.py`: `runtime_identity` (`hibm_native_reference`), `identity`,
    `advance`, `boundary_policy`, `configure_coupling`, `build_boundary`, `marker_projector`
    (`simulation_core.coupling.hibm_mpm.*`), `coupling_config` (`simulation_core.drivers.generic_fsi_solver`),
    `prepare_execution` (`simulation_core.native_dispatch.wave2`, only with `native_dispatch` on). These are the
    sharp-interface profile methods.
* **CUDA path checked statically only.** The CPU tests do not exercise the CUDA-only code: the device controllers
  `simulation_core/fluids/gpu_*.py`, `scalar_maximum` in `shared_definitions.py` (its class `TailKernels` is reduced
  to the one kernel it uses), and the CUDA data path of `hypre_pressure.py`. This code passes the static checks
  (undefined names, imports). The corresponding code of the source tree ran on GPUs, but this export has not yet run
  on a GPU.

## Credits

* IBAMR v0.19.0 (BSD-3-Clause, copyright the IBAMR developers): conventions of the side-centred IB_4 interpolation
  and spreading, the F update (MIDPOINT_RULE), and the target-point / penalty-tether form that the contact penalty is
  based on. These are reused as algorithms; no source is copied.
* taichi_mpm (MIT, copyright Yuanming Hu and the Taichi MPM authors): the MLS-MPM stress term and the
  deformation-gradient update, and the normal-only penetration penalty against level sets. These are reused as
  algorithms.
* A. Pinelli, I. Z. Naqavi, U. Piomelli, J. Favier, "Immersed-boundary methods for general finite-difference and
  finite-volume Navier-Stokes solvers", J. Comput. Phys. 229 (2010) 9073-9091: the reproducing-kernel (moment)
  correction of the delta weights on non-uniform grids used in `graded_kernel.py`.
* hypre (Apache-2.0 or MIT, copyright Lawrence Livermore National Security, LLC and the HYPRE developers): the
  optional Struct PCG + PFMG/SMG pressure solve. It is used as an external library, built separately, and no hypre
  source is included.
* PETSc v3.26.0 (BSD-2-Clause): looked at for the treatment of Dirichlet rows (`MatZeroRowsColumns`); no source copied.
* Taichi (Apache-2.0).

The docstrings' "reuse check" notes refer to `reference_libs/`: read-only local copies of the upstream sources
listed above (IBAMR v0.19.0, PETSc v3.26.0, hypre v3.2.0, taichi_mpm). They are not part of this repository.

No licence has been chosen for this repository yet.
