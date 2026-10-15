"""hypre (v3.2.0) Struct-API pressure solve for the finite-volume pressure operator of the fluid solver.

A thin ctypes layer over libHYPRE.so (no compiled extension, no extra package): the 7-point variable-coefficient Struct
matrix is built from the solver's own operator, solved by PCG with a PFMG (or SMG) V-cycle preconditioner.

FLUID INTEGRATION: install_hypre_pressure_solver(fluid, prefix=...) replaces ONLY the linear solve inside fluid.project()
for that fluid instance (the instance attribute _solve_pressure_poisson_fv_cg).  See HyprePressureSolver for the per-call
algorithm, the stopping criterion (the solver's own ||r||_V check on the device, then the original solve confirms with 0
CG iterations), the upload logic (matrix once per mask change, diagonal box per pressure-chamber change, right-hand side
and warm start per call) and the data paths.

LIBRARY.  Loaded from a configurable prefix: argument `prefix`, else the environment variable HYPRE_PREFIX.  The build type
is read from <prefix>/include/HYPRE_config.h: with HYPRE_USING_CUDA, HYPRE_Initialize(),
HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE) and HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE) are called and every array
handed to the Struct API is a DEVICE pointer.  Requires the default sequential build (MPI off: MPI_Comm = HYPRE_Int,
communicator 0), 32-bit HYPRE_Int and double HYPRE_Real (checked).

SYSTEM (identical to the solver's finite-volume conjugate-gradient system):
    A p = b,  (A p)_i = D_i p_i - sum_f T_if p_nb(f)  per unit volume, i a fluid cell (obstacle == 0),
with D_i the finite-volume denominator (incl. the z-min outlet ghost term 2 w / dz0^2) plus
pressure_interface_matrix_diagonal (e.g. a pressure chamber's penalty), T_if the face products coeff * face_weight (zero
towards obstacles and closed walls), and b = -(rhs_scale (div* + s - volume_source) - R).  The solver's CG uses the
volume-weighted inner product <x, y>_V = sum_i V_i x_i y_i (A is self-adjoint there; V A is symmetric) and stops when
||r||_V <= tol ||b||_V (checked every 16 iterations).  hypre's Struct PCG is Euclidean, so a symmetric form is passed
(scaled_system):
  form "V" (default): V A p = V b, the standard finite-volume M-matrix (couplings -face area / centre distance).  hypre
    then stops on ||V r||_2 / ||V b||_2, a different norm, so solve_to_solver_tolerance() evaluates the solver's
    ||r||_V / ||b||_V on the result and continues with a tighter hypre tolerance until it holds (measured:
    ||r||_V / ||b||_V ~ 1.9 x hypre's ratio; first_tol_factor = 0.4 meets it in one run).
  form "sqrtV": S A S^-1 (S = sqrt V), symmetric with ||r''||_2 = ||r||_V exactly, but not an M-matrix on a graded grid
    (row sums != 0): PFMG red/black Gauss-Seidel / non-Galerkin break down (PCG gamma < 0) and weighted Jacobi needs ~5x
    more iterations.  Kept for reference.
Obstacle cells are identity rows (scaled to a virtual fluid diagonal) with zero right-hand side.  Symmetric storage
(HYPRE_StructMatrixSetSymmetric(A, 1)) keeps the centre and the three minus-side couplings (stencil entries 0, 1, 3, 5).

PRECONDITIONER.  PFMG (semi-coarsening multigrid: coarsens one direction per level, chosen from the operator's
anisotropy, so graded / stretched cells are handled) inside PCG, one V-cycle per application (max_iter 1, tol 0, zero
guess).  Parameters: relax_type 1 = weighted Jacobi (hypre default), 2 = symmetric red/black Gauss-Seidel (3 =
non-symmetric red/black Gauss-Seidel is not valid inside CG); rap_type 0 = Galerkin (default), 1 = non-Galerkin 7-point;
hypre 3.2.0 forces rap_type = 1 whenever relax_type is 2 or 3 (struct_ls/pfmg_setup.c); skip_relax 1 (default) / 0;
num_pre_relax = num_post_relax = 1 (default) or 2.  SMG (full semi-coarsening with line / plane smoothing) is the
alternative.  When the matrix changes every step the solver objects are set up per solve (setup time counts).
Measured on a 1.57 M-cell graded system (single CPU thread; solver CG: 176 iterations): relax_type 2, skip_relax 0,
2 pre / 2 post sweeps: 26-28 PCG iterations, setup 0.06 s, solve 1.7-1.8 s; hypre default: 52-55 iterations.  On a GPU
the best configuration measured was relax_type 1, rap_type 0, skip_relax 0, one sweep (39 iterations, 0.146 s).

HOST <-> DEVICE COPIES (CUDA build), per solve, N cells: coefficients 4 doubles / cell host -> device (then device ->
device into hypre's storage), b and x0 8 N bytes each, the solution 8 N bytes back, plus one download per run for the
host-side ||r||_V check of the "V" form.  data_path="device" packs the coefficients, b and x on the device into Taichi
ndarrays and hands their raw pointers to hypre (both must use the same CUDA context).

HARD STOPS: a hypre error code (other than non-convergence) raises SimulationHardStop('hypre_error'); non-convergence
within max_iter raises SimulationHardStop('hypre_pressure_not_converged') when raise_on_nonconvergence is set.

hypre (Apache-2.0 / MIT, (c) Lawrence Livermore National Security and the HYPRE developers) is used as a library built
from the official source; the API calls follow hypre's documentation and src/examples/ex3.c / ex4.c (Struct PCG + PFMG
/ SMG).  No hypre source is copied into this module.

No `from __future__ import annotations` (module next to Taichi kernels).
"""
import ctypes
import json
import math
import os
import re
import time

import numpy as np

from fastfsi.errors import SimulationHardStop

PREFIX_ENV = "HYPRE_PREFIX"
CUDART_ENV = "HYPRE_CUDART"
HARD_STOP_CODES = {"hypre_error", "hypre_pressure_not_converged"}
HYPRE_ERROR_CONV = 256
HYPRE_MEMORY_DEVICE = 1
HYPRE_EXEC_DEVICE = 1
COMM = 0                                   # MPI_Comm of the sequential build (hypre_MPI_COMM_WORLD)
STENCIL_OFFSETS = ((0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
STORED_ENTRIES = (0, 1, 3, 5)               # symmetric storage: centre, -x, -y, -z
PFMG_SWEEP = tuple(dict(relax_type=relax, rap_type=rap, skip_relax=skip, num_relax=sweeps)
                   for relax in (1, 2) for rap in (0, 1) for skip in (1, 0) for sweeps in (1, 2))


def _config_flags(prefix):
    path = os.path.join(prefix, "include", "HYPRE_config.h")
    with open(path, encoding="utf-8") as stream:
        text = stream.read()
    defined = set(re.findall(r"^#define\s+(HYPRE_\w+)", text, flags=re.MULTILINE))
    version = re.search(r'#define\s+HYPRE_RELEASE_VERSION\s+"([^"]+)"', text)
    return defined, (version.group(1) if version else "unknown")


class CudaRuntime:
    """Minimal ctypes binding of the CUDA runtime that libHYPRE itself links (device buffers for the Struct API)."""

    H2D, D2H, D2D = 1, 2, 3

    def __init__(self):
        candidates = []
        if os.environ.get(CUDART_ENV):
            candidates.append(os.environ[CUDART_ENV])
        try:
            with open("/proc/self/maps", encoding="utf-8") as stream:        # the copy libHYPRE already loaded
                candidates += sorted({line.split()[-1] for line in stream if "libcudart" in line})
        except OSError:
            pass
        candidates += ["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"]
        self.lib = None
        for name in candidates:
            try:
                self.lib = ctypes.CDLL(name)
                self.path = name
                break
            except OSError:
                continue
        if self.lib is None:
            raise OSError("libcudart not found (set %s to its path)" % CUDART_ENV)
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaDeviceSynchronize.argtypes = []
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def check(self, code, what):
        if code != 0:
            raise SimulationHardStop("hypre_error", dict(call=what, cuda_error=int(code),
                                                       message=self.lib.cudaGetErrorString(code).decode()))

    def malloc(self, nbytes):
        pointer = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(max(int(nbytes), 8))), "cudaMalloc")
        return pointer.value

    def free(self, pointer):
        if pointer:
            self.check(self.lib.cudaFree(ctypes.c_void_p(pointer)), "cudaFree")

    def copy(self, dst, src, nbytes, kind):
        self.check(self.lib.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), ctypes.c_size_t(int(nbytes)), kind),
                   "cudaMemcpy")

    def synchronize(self):
        self.check(self.lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


class HypreLibrary:
    """libHYPRE.so from `prefix` with the ctypes signatures used here; one instance per process and prefix."""

    _loaded = {}

    def __new__(cls, prefix=None):
        chosen = prefix or os.environ.get(PREFIX_ENV)
        if not chosen:
            raise ValueError("the hypre installation prefix is not given: pass prefix= or set %s" % PREFIX_ENV)
        prefix = os.path.abspath(chosen)
        if prefix not in cls._loaded:
            if cls._loaded:
                raise RuntimeError("a different libHYPRE is already loaded in this process: %s" % list(cls._loaded))
            instance = super().__new__(cls)
            instance._load(prefix)
            cls._loaded[prefix] = instance
        return cls._loaded[prefix]

    def _load(self, prefix):
        self.prefix = prefix
        flags, self.version = _config_flags(prefix)
        for bad in ("HYPRE_BIGINT", "HYPRE_MIXEDINT", "HYPRE_SINGLE", "HYPRE_LONG_DOUBLE", "HYPRE_COMPLEX"):
            if bad in flags:
                raise RuntimeError("hypre build %s defines %s; this layer needs int32 HYPRE_Int and double" % (prefix, bad))
        if "HYPRE_SEQUENTIAL" not in flags:
            raise RuntimeError("hypre build %s is an MPI build; this layer needs the sequential build" % prefix)
        self.cuda = "HYPRE_USING_CUDA" in flags
        self.flags = sorted(flags)
        self.lib = ctypes.CDLL(os.path.join(prefix, "lib", "libHYPRE.so"))
        I, D, P = ctypes.c_int, ctypes.c_double, ctypes.c_void_p
        PP, PI, PD = ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
        signatures = {
            "HYPRE_Initialize": [], "HYPRE_Initialized": [], "HYPRE_ClearAllErrors": [],
            "HYPRE_SetMemoryLocation": [I], "HYPRE_SetExecutionPolicy": [I],
            "HYPRE_StructGridCreate": [I, I, PP], "HYPRE_StructGridSetExtents": [P, PI, PI],
            "HYPRE_StructGridAssemble": [P], "HYPRE_StructGridDestroy": [P],
            "HYPRE_StructStencilCreate": [I, I, PP], "HYPRE_StructStencilSetEntry": [P, I, PI],
            "HYPRE_StructStencilDestroy": [P],
            "HYPRE_StructMatrixCreate": [I, P, P, PP], "HYPRE_StructMatrixSetSymmetric": [P, I],
            "HYPRE_StructMatrixInitialize": [P], "HYPRE_StructMatrixSetBoxValues": [P, PI, PI, I, PI, P],
            "HYPRE_StructMatrixAssemble": [P], "HYPRE_StructMatrixDestroy": [P],
            "HYPRE_StructMatrixMatvec": [D, P, P, D, P],
            "HYPRE_StructVectorCreate": [I, P, PP], "HYPRE_StructVectorInitialize": [P],
            "HYPRE_StructVectorSetBoxValues": [P, PI, PI, P], "HYPRE_StructVectorGetBoxValues": [P, PI, PI, P],
            "HYPRE_StructVectorAssemble": [P], "HYPRE_StructVectorDestroy": [P],
            "HYPRE_StructPCGCreate": [I, PP], "HYPRE_StructPCGDestroy": [P], "HYPRE_StructPCGSetup": [P, P, P, P],
            "HYPRE_StructPCGSolve": [P, P, P, P], "HYPRE_StructPCGSetTol": [P, D],
            "HYPRE_StructPCGSetAbsoluteTol": [P, D], "HYPRE_StructPCGSetMaxIter": [P, I],
            "HYPRE_StructPCGSetTwoNorm": [P, I], "HYPRE_StructPCGSetRelChange": [P, I],
            "HYPRE_StructPCGSetPrecond": [P, P, P, P], "HYPRE_StructPCGSetLogging": [P, I],
            "HYPRE_StructPCGSetPrintLevel": [P, I], "HYPRE_StructPCGGetNumIterations": [P, PI],
            "HYPRE_StructPCGGetFinalRelativeResidualNorm": [P, PD],
            "HYPRE_StructPFMGCreate": [I, PP], "HYPRE_StructPFMGDestroy": [P], "HYPRE_StructPFMGSetup": [P, P, P, P],
            "HYPRE_StructPFMGSolve": [P, P, P, P], "HYPRE_StructPFMGSetTol": [P, D],
            "HYPRE_StructPFMGSetMaxIter": [P, I], "HYPRE_StructPFMGSetZeroGuess": [P],
            "HYPRE_StructPFMGSetRelaxType": [P, I], "HYPRE_StructPFMGSetRAPType": [P, I],
            "HYPRE_StructPFMGSetNumPreRelax": [P, I], "HYPRE_StructPFMGSetNumPostRelax": [P, I],
            "HYPRE_StructPFMGSetSkipRelax": [P, I], "HYPRE_StructPFMGSetMaxLevels": [P, I],
            "HYPRE_StructPFMGSetLogging": [P, I], "HYPRE_StructPFMGSetPrintLevel": [P, I],
            "HYPRE_StructSMGCreate": [I, PP], "HYPRE_StructSMGDestroy": [P], "HYPRE_StructSMGSetup": [P, P, P, P],
            "HYPRE_StructSMGSolve": [P, P, P, P], "HYPRE_StructSMGSetTol": [P, D],
            "HYPRE_StructSMGSetMaxIter": [P, I], "HYPRE_StructSMGSetZeroGuess": [P],
            "HYPRE_StructSMGSetNumPreRelax": [P, I], "HYPRE_StructSMGSetNumPostRelax": [P, I],
            "HYPRE_StructSMGSetMemoryUse": [P, I], "HYPRE_StructSMGSetLogging": [P, I],
            "HYPRE_StructSMGSetPrintLevel": [P, I],
        }
        for name, argtypes in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = argtypes
            function.restype = ctypes.c_int
        self.lib.HYPRE_DescribeError.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self.lib.HYPRE_DescribeError.restype = None
        self.address = {name: ctypes.cast(getattr(self.lib, name), ctypes.c_void_p).value
                        for name in ("HYPRE_StructPFMGSolve", "HYPRE_StructPFMGSetup",
                                     "HYPRE_StructSMGSolve", "HYPRE_StructSMGSetup")}
        if not self.lib.HYPRE_Initialized():
            self.call("HYPRE_Initialize")
        self.cudart = None
        if self.cuda:
            self.call("HYPRE_SetMemoryLocation", HYPRE_MEMORY_DEVICE)
            self.call("HYPRE_SetExecutionPolicy", HYPRE_EXEC_DEVICE)
            self.cudart = CudaRuntime()

    def call(self, name, *args, allow_conv=False):
        code = getattr(self.lib, name)(*args)
        if code == 0:
            return 0
        if allow_conv and code == HYPRE_ERROR_CONV:
            self.lib.HYPRE_ClearAllErrors()
            return code
        message = ctypes.create_string_buffer(512)
        self.lib.HYPRE_DescribeError(code, message)
        self.lib.HYPRE_ClearAllErrors()
        raise SimulationHardStop("hypre_error", dict(call=name, code=int(code), message=message.value.decode()))

    def synchronize(self):
        if self.cudart is not None:
            self.cudart.synchronize()

    def identity(self):
        return dict(prefix=self.prefix, version=self.version, cuda=self.cuda,
                    cudart=getattr(self.cudart, "path", None), flags=self.flags)


def scaled_system(export, form="V"):
    """The symmetric hypre system from an exported FV system (dict / npz with diag, products (6: x- x+ y- y+ z- z+),
    volume_m3, obstacle, b, p0).  Returns hypre-ordered (x fastest) host arrays: values (N, 4) for stencil entries
    0, 1, 3, 5, rhs (N,), x0 (N,); the unknown's scale (p = x / scale, solver layout); the max relative asymmetry of V A.

    form "V" (default): V A x = V b, x = p -- the standard finite-volume symmetric M-matrix (face area / centre distance
      couplings, zero row sums in the interior); hypre's 2-norm residual is ||V r||_2, NOT the solver's ||r||_V, so the
      caller verifies ||r||_V on the result (see solve_to_solver_tolerance).
    form "sqrtV": S A S^-1 x = S b, x = S p, S = sqrt(V) -- symmetric and ||r''||_2 = ||r||_V exactly, but not an
      M-matrix on a graded grid (row sums != 0), which PFMG handles poorly (measured: breakdown with RB-GS / non-Galerkin,
      ~9x more iterations with weighted Jacobi on a 1.2-stretch grid).
    Obstacle cells: identity rows scaled to the magnitude of a virtual fluid diagonal at that cell (b = x0 = 0)."""
    if form not in ("V", "sqrtV"):
        raise ValueError("form must be 'V' or 'sqrtV'")
    obstacle = np.asarray(export["obstacle"]) != 0
    fluid = ~obstacle
    exported_volume = np.asarray(export["volume_m3"], np.float64)        # the solver's f64 volumes (fluid cells)
    widths = None
    if "cell_widths_x_m" in export:
        widths = [np.asarray(export[key], np.float64)
                  for key in ("cell_widths_x_m", "cell_widths_y_m", "cell_widths_z_m")]
    if widths is not None:
        virtual = (2.0 / widths[0][:, None, None] ** 2 + 2.0 / widths[1][None, :, None] ** 2
                   + 2.0 / widths[2][None, None, :] ** 2)
        cell_volume = widths[0][:, None, None] * widths[1][None, :, None] * widths[2][None, None, :]
    else:
        virtual = np.full(obstacle.shape, float(np.median(np.asarray(export["diag"])[fluid])))
        cell_volume = np.full(obstacle.shape, float(np.median(exported_volume[fluid])))
    # obstacle cells: the export carries no volume there (0); their identity rows use the geometric cell volume
    volume = np.where(fluid, exported_volume, cell_volume)
    sqrt_v = np.sqrt(volume)
    row = volume if form == "V" else np.ones_like(volume)                 # left scaling of A (per row)
    diag = np.where(fluid, np.asarray(export["diag"], np.float64) * row, virtual * row)
    products = np.asarray(export["products"], np.float64)
    lower = []
    asymmetry = 0.0
    for axis in range(3):
        minus = np.where(fluid, products[..., 2 * axis], 0.0)              # coupling of i to i - e_axis
        plus = np.where(fluid, products[..., 2 * axis + 1], 0.0)           # coupling of i to i + e_axis
        index = [slice(None)] * 3
        index[axis] = slice(1, None)
        source = [slice(None)] * 3
        source[axis] = slice(0, -1)
        if form == "V":
            coupling = -minus * volume                                     # -V_i T_{i,i-e} = -(face area / distance)
        else:
            neighbour = np.ones_like(sqrt_v)
            neighbour[tuple(index)] = sqrt_v[tuple(source)]
            coupling = -minus * sqrt_v / neighbour                         # -sqrt(V_i / V_{i-e}) T_{i,i-e}
        vi_t = volume * minus                                              # symmetry: V_i T_{i,i-e} = V_{i-e} T_{i-e,i}
        vj_t = np.zeros_like(vi_t)
        vj_t[tuple(index)] = (volume * plus)[tuple(source)]
        both = (vi_t != 0.0) | (vj_t != 0.0)
        if both.any():
            asymmetry = max(asymmetry, float(np.max(np.abs(vi_t[both] - vj_t[both])
                                                     / np.maximum(np.abs(vi_t[both]), np.abs(vj_t[both])))))
        lower.append(coupling)
    values = np.stack([diag, lower[0], lower[1], lower[2]], axis=-1)        # (nx, ny, nz, 4)
    values = np.ascontiguousarray(values.transpose(2, 1, 0, 3)).reshape(-1, 4)
    scale = np.ones_like(volume) if form == "V" else sqrt_v
    rhs_scale = volume if form == "V" else sqrt_v
    rhs = np.where(fluid, np.asarray(export["b"], np.float64) * rhs_scale, 0.0)
    p0 = np.asarray(export["p0"], np.float64) if "p0" in export else np.zeros_like(rhs)
    x0 = np.where(fluid, p0 * scale, 0.0)
    to_hypre = lambda a: np.ascontiguousarray(a.transpose(2, 1, 0)).reshape(-1)
    return dict(values=values, rhs=to_hypre(rhs), x0=to_hypre(x0), scale=np.where(fluid, scale, 1.0), fluid=fluid,
                asymmetry=asymmetry, shape=tuple(int(v) for v in obstacle.shape), form=form)


def full_stencil_values(values4, shape):
    """Expand symmetric-storage values (N, 4: centre, -x, -y, -z) to all 7 entries (centre, -x, +x, -y, +y, -z, +z):
    the +e coupling of cell i is the -e coupling of cell i + e (zero at the far boundary)."""
    nx, ny, nz = shape
    v = np.asarray(values4).reshape(nz, ny, nx, 4)
    full = np.zeros((nz, ny, nx, 7))
    full[..., 0], full[..., 1], full[..., 3], full[..., 5] = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
    full[:, :, :-1, 2] = v[:, :, 1:, 1]                  # +x of i = -x of i + 1 (x is the last axis here)
    full[:, :-1, :, 4] = v[:, 1:, :, 2]
    full[:-1, :, :, 6] = v[1:, :, :, 3]
    return full.reshape(-1, 7)


def from_hypre_order(vector, shape):
    nx, ny, nz = shape
    return np.ascontiguousarray(np.asarray(vector).reshape(nz, ny, nx).transpose(2, 1, 0))


def fv_apply(export, p):
    """Our operator (per unit volume) applied to p in solver layout: (A p)_i = D_i p_i - sum_f T_if p_nb (fluid cells)."""
    fluid = np.asarray(export["obstacle"]) == 0
    diag = np.asarray(export["diag"], np.float64)
    products = np.asarray(export["products"], np.float64)
    p = np.where(fluid, np.asarray(p, np.float64), 0.0)
    out = diag * p
    for axis in range(3):
        minus = np.zeros_like(p)
        plus = np.zeros_like(p)
        index = [slice(None)] * 3
        index[axis] = slice(1, None)
        source = [slice(None)] * 3
        source[axis] = slice(0, -1)
        minus[tuple(index)] = p[tuple(source)]                   # p_{i - e}
        plus[tuple(source)] = p[tuple(index)]                    # p_{i + e}
        out -= products[..., 2 * axis] * minus + products[..., 2 * axis + 1] * plus
    return np.where(fluid, out, 0.0)


def relative_residual_v(export, p):
    """||b - A p||_V / ||b||_V over fluid cells (the solver's CG criterion)."""
    fluid = np.asarray(export["obstacle"]) == 0
    volume = np.asarray(export["volume_m3"], np.float64)
    b = np.asarray(export["b"], np.float64)
    r = np.where(fluid, b - fv_apply(export, p), 0.0)
    return float(np.sqrt(np.sum(volume * r * r) / max(np.sum(np.where(fluid, volume * b * b, 0.0)), 1e-300)))


class StructPressureSolver:
    """hypre Struct PCG + PFMG / SMG on an (nx, ny, nz) box with the 7-point stencil (symmetric storage)."""

    def __init__(self, shape, *, prefix=None, solver="pfmg", tol=1.0e-5, max_iter=400, relax_type=1, rap_type=0,
                 skip_relax=1, num_relax=1, max_levels=0, raise_on_nonconvergence=False, print_level=0,
                 symmetric_storage=True):
        self.hypre = HypreLibrary(prefix)
        self.shape = tuple(int(v) for v in shape)
        self.cells = int(np.prod(self.shape))
        if solver not in ("pfmg", "smg"):
            raise ValueError("solver must be 'pfmg' or 'smg'")
        if relax_type not in (0, 1, 2):
            raise ValueError("PFMG relax_type inside CG must be 0, 1 (Jacobi) or 2 (symmetric RB-GS)")
        self.options = dict(solver=solver, tol=float(tol), max_iter=int(max_iter), relax_type=int(relax_type),
                            rap_type=int(rap_type), skip_relax=int(skip_relax), num_relax=int(num_relax),
                            max_levels=int(max_levels), symmetric_storage=bool(symmetric_storage))
        self.print_level = int(print_level)
        self.raise_on_nonconvergence = bool(raise_on_nonconvergence)
        self.ilower = (ctypes.c_int * 3)(0, 0, 0)
        self.iupper = (ctypes.c_int * 3)(*(n - 1 for n in self.shape))
        h = self.hypre
        self.grid, self.stencil = ctypes.c_void_p(), ctypes.c_void_p()
        h.call("HYPRE_StructGridCreate", COMM, 3, ctypes.byref(self.grid))
        h.call("HYPRE_StructGridSetExtents", self.grid, self.ilower, self.iupper)
        h.call("HYPRE_StructGridAssemble", self.grid)
        h.call("HYPRE_StructStencilCreate", 3, len(STENCIL_OFFSETS), ctypes.byref(self.stencil))
        for entry, offset in enumerate(STENCIL_OFFSETS):
            h.call("HYPRE_StructStencilSetEntry", self.stencil, entry, (ctypes.c_int * 3)(*offset))
        self.A, self.b, self.x = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        h.call("HYPRE_StructMatrixCreate", COMM, self.grid, self.stencil, ctypes.byref(self.A))
        h.call("HYPRE_StructMatrixSetSymmetric", self.A, 1 if symmetric_storage else 0)
        h.call("HYPRE_StructMatrixInitialize", self.A)
        for vector in (self.b, self.x):
            h.call("HYPRE_StructVectorCreate", COMM, self.grid, ctypes.byref(vector))
            h.call("HYPRE_StructVectorInitialize", vector)
        self.stored = STORED_ENTRIES if symmetric_storage else tuple(range(len(STENCIL_OFFSETS)))
        self.entries = (ctypes.c_int * len(self.stored))(*self.stored)
        self._device = {}
        if h.cuda:                                   # device staging buffers for the Struct API (device pointers)
            for name, count in (("values", len(self.stored) * self.cells), ("rhs", self.cells), ("x", self.cells)):
                self._device[name] = h.cudart.malloc(8 * count)
        self.transfers = dict(h2d_bytes=0, d2h_bytes=0, h2d_s=0.0, d2h_s=0.0)

    # ------------------------------------------------------------------------------------------------ data in / out
    def _pointer_for(self, name, array):
        """A pointer the Struct API may read: the host array (CPU build) or its device copy (CUDA build)."""
        array = np.ascontiguousarray(array, dtype=np.float64)
        if not self.hypre.cuda:
            return array, array.ctypes.data
        start = time.perf_counter()
        self.hypre.cudart.copy(self._device[name], array.ctypes.data, array.nbytes, CudaRuntime.H2D)
        self.transfers["h2d_s"] += time.perf_counter() - start
        self.transfers["h2d_bytes"] += array.nbytes
        return array, self._device[name]

    def set_system_pointers(self, values_ptr, rhs_ptr, x0_ptr):
        """Hand over raw pointers (device pointers for a CUDA build, host pointers otherwise) in hypre order:
        values (N, 4) doubles for entries (0, 1, 3, 5) (symmetric storage; (N, 7) for all entries otherwise), rhs (N,),
        x0 (N,).  Used for zero-host-copy Taichi interop."""
        h = self.hypre
        h.call("HYPRE_StructMatrixSetBoxValues", self.A, self.ilower, self.iupper, len(self.stored), self.entries,
               ctypes.c_void_p(values_ptr))
        h.call("HYPRE_StructMatrixAssemble", self.A)
        h.call("HYPRE_StructVectorSetBoxValues", self.b, self.ilower, self.iupper, ctypes.c_void_p(rhs_ptr))
        h.call("HYPRE_StructVectorAssemble", self.b)
        h.call("HYPRE_StructVectorSetBoxValues", self.x, self.ilower, self.iupper, ctypes.c_void_p(x0_ptr))
        h.call("HYPRE_StructVectorAssemble", self.x)

    def set_matrix_entries(self, source, entries, *, box=None, pointer=False):
        """Set the given stencil entries (subset of the stored ones) over the whole grid or over `box` = (lower, upper)
        (inclusive cell indices) from a host array (cells of the box x fastest, len(entries) values per cell; copied
        to the device first for a CUDA build) or from a raw pointer (pointer=True), then re-assemble.  Used per step for
        the diagonal of the changed (air-chamber) box only: entries=(0,)."""
        entries = tuple(int(e) for e in entries)
        if any(e not in self.stored for e in entries):
            raise ValueError("entries %s are not stored (%s)" % (entries, self.stored))
        if box is None:
            lower, upper, count = self.ilower, self.iupper, self.cells
        else:
            lo, hi = (tuple(int(v) for v in corner) for corner in box)
            if any(a < 0 or b < a or b >= n for a, b, n in zip(lo, hi, self.shape)):
                raise ValueError("box %s outside the grid %s" % (box, self.shape))
            lower, upper = (ctypes.c_int * 3)(*lo), (ctypes.c_int * 3)(*hi)
            count = int(np.prod([b - a + 1 for a, b in zip(lo, hi)]))
        if pointer:
            address = int(source)
        else:
            array = np.ascontiguousarray(source, dtype=np.float64).reshape(-1)[:count * len(entries)]
            if array.size != count * len(entries):
                raise ValueError("matrix values do not match the box")
            _, address = self._pointer_for("values", array)
        h = self.hypre
        h.call("HYPRE_StructMatrixSetBoxValues", self.A, lower, upper, len(entries),
               (ctypes.c_int * len(entries))(*entries), ctypes.c_void_p(address))
        h.call("HYPRE_StructMatrixAssemble", self.A)

    def set_vector(self, which, source, *, pointer=False):
        """Set b ('rhs') or the initial guess ('x') over the whole box from a host array or a raw pointer."""
        vector = {"rhs": self.b, "x": self.x}[which]
        if pointer:
            address = int(source)
        else:
            array = np.ascontiguousarray(source, dtype=np.float64).reshape(self.cells)
            _, address = self._pointer_for(which, array)
        h = self.hypre
        h.call("HYPRE_StructVectorSetBoxValues", vector, self.ilower, self.iupper, ctypes.c_void_p(address))
        h.call("HYPRE_StructVectorAssemble", vector)

    def set_system(self, values, rhs, x0):
        """Host arrays in hypre order (see scaled_system); copies to the device first for a CUDA build."""
        if values.shape[0] == self.cells and values.shape[1] == 4 and len(self.stored) == 7:
            values = full_stencil_values(values, self.shape)
        if values.shape != (self.cells, len(self.stored)) or rhs.shape != (self.cells,) or x0.shape != (self.cells,):
            raise ValueError("system arrays do not match the grid")
        keep = [self._pointer_for(name, array) for name, array in (("values", values), ("rhs", rhs), ("x", x0))]
        self.set_system_pointers(*(pointer for _, pointer in keep))
        self.hypre.synchronize()

    def get_solution_pointer(self, x_ptr):
        self.hypre.call("HYPRE_StructVectorGetBoxValues", self.x, self.ilower, self.iupper, ctypes.c_void_p(x_ptr))

    def get_solution(self):
        out = np.empty(self.cells, np.float64)
        if not self.hypre.cuda:
            self.get_solution_pointer(out.ctypes.data)
            return out
        self.get_solution_pointer(self._device["x"])
        self.hypre.synchronize()
        start = time.perf_counter()
        self.hypre.cudart.copy(out.ctypes.data, self._device["x"], out.nbytes, CudaRuntime.D2H)
        self.transfers["d2h_s"] += time.perf_counter() - start
        self.transfers["d2h_bytes"] += out.nbytes
        return out

    # ------------------------------------------------------------------------------------------------ solve
    def _make_preconditioner(self):
        h, o = self.hypre, self.options
        precond = ctypes.c_void_p()
        if o["solver"] == "pfmg":
            h.call("HYPRE_StructPFMGCreate", COMM, ctypes.byref(precond))
            h.call("HYPRE_StructPFMGSetMaxIter", precond, 1)
            h.call("HYPRE_StructPFMGSetTol", precond, 0.0)
            h.call("HYPRE_StructPFMGSetZeroGuess", precond)
            h.call("HYPRE_StructPFMGSetRelaxType", precond, o["relax_type"])
            h.call("HYPRE_StructPFMGSetRAPType", precond, o["rap_type"])
            h.call("HYPRE_StructPFMGSetNumPreRelax", precond, o["num_relax"])
            h.call("HYPRE_StructPFMGSetNumPostRelax", precond, o["num_relax"])
            h.call("HYPRE_StructPFMGSetSkipRelax", precond, o["skip_relax"])
            if o["max_levels"] > 0:
                h.call("HYPRE_StructPFMGSetMaxLevels", precond, o["max_levels"])
            return precond, h.address["HYPRE_StructPFMGSolve"], h.address["HYPRE_StructPFMGSetup"], \
                "HYPRE_StructPFMGDestroy"
        h.call("HYPRE_StructSMGCreate", COMM, ctypes.byref(precond))
        h.call("HYPRE_StructSMGSetMemoryUse", precond, 0)
        h.call("HYPRE_StructSMGSetMaxIter", precond, 1)
        h.call("HYPRE_StructSMGSetTol", precond, 0.0)
        h.call("HYPRE_StructSMGSetZeroGuess", precond)
        h.call("HYPRE_StructSMGSetNumPreRelax", precond, o["num_relax"])
        h.call("HYPRE_StructSMGSetNumPostRelax", precond, o["num_relax"])
        return precond, h.address["HYPRE_StructSMGSolve"], h.address["HYPRE_StructSMGSetup"], "HYPRE_StructSMGDestroy"

    def setup(self):
        """Create PCG + preconditioner for the current matrix and run the setup (preconditioner hierarchy incl. RAP).
        Must be redone after every matrix change.  Returns the setup wall time (device synchronised)."""
        h, o = self.hypre, self.options
        self.release()
        h.synchronize()
        start = time.perf_counter()
        pcg = ctypes.c_void_p()
        h.call("HYPRE_StructPCGCreate", COMM, ctypes.byref(pcg))
        h.call("HYPRE_StructPCGSetTol", pcg, o["tol"])
        h.call("HYPRE_StructPCGSetAbsoluteTol", pcg, 0.0)
        h.call("HYPRE_StructPCGSetMaxIter", pcg, o["max_iter"])
        h.call("HYPRE_StructPCGSetTwoNorm", pcg, 1)
        h.call("HYPRE_StructPCGSetRelChange", pcg, 0)
        h.call("HYPRE_StructPCGSetLogging", pcg, 1)
        h.call("HYPRE_StructPCGSetPrintLevel", pcg, self.print_level)
        precond, solve_fn, setup_fn, destroy = self._make_preconditioner()
        h.call("HYPRE_StructPCGSetPrecond", pcg, ctypes.c_void_p(solve_fn), ctypes.c_void_p(setup_fn), precond)
        h.call("HYPRE_StructPCGSetup", pcg, self.A, self.b, self.x)
        h.synchronize()
        self._pcg, self._precond, self._precond_destroy = pcg, precond, destroy
        return time.perf_counter() - start

    def run(self, tol=None):
        """PCG iterations from the current x (warm start) to hypre's relative 2-norm tolerance `tol`."""
        h = self.hypre
        if getattr(self, "_pcg", None) is None:
            raise RuntimeError("setup() first")
        tol = self.options["tol"] if tol is None else float(tol)
        h.call("HYPRE_StructPCGSetTol", self._pcg, tol)
        h.synchronize()
        start = time.perf_counter()
        code = h.call("HYPRE_StructPCGSolve", self._pcg, self.A, self.b, self.x, allow_conv=True)
        h.synchronize()
        elapsed = time.perf_counter() - start
        iterations = ctypes.c_int()
        relres = ctypes.c_double()
        h.call("HYPRE_StructPCGGetNumIterations", self._pcg, ctypes.byref(iterations))
        h.call("HYPRE_StructPCGGetFinalRelativeResidualNorm", self._pcg, ctypes.byref(relres))
        return dict(iterations=int(iterations.value), final_relative_residual=float(relres.value),
                    converged=bool(code == 0 and relres.value <= tol), solve_s=elapsed, tol_used=tol)

    def release(self):
        if getattr(self, "_pcg", None) is not None:
            self.hypre.call("HYPRE_StructPCGDestroy", self._pcg)
            self.hypre.call(self._precond_destroy, self._precond)
        self._pcg = self._precond = None

    def solve(self):
        """Setup + one run at the configured tolerance (hypre's own 2-norm criterion)."""
        setup_s = self.setup()
        result = self.run()
        result.update(setup_s=setup_s, **self.options)
        if not result["converged"] and self.raise_on_nonconvergence:
            raise SimulationHardStop("hypre_pressure_not_converged", result)
        return result

    def solve_to_solver_tolerance(self, residual_v, target, max_restarts=4, first_tol_factor=1.0):
        """Setup, then PCG runs until the SOLVER's criterion ||r||_V / ||b||_V <= target holds for the unscaled pressure.
        `residual_v(x_hypre_order) -> float` evaluates that criterion (the caller unscales; benchmark: numpy on the host).
        Starts at hypre tol = first_tol_factor * target; each restart tightens hypre's tolerance by
        max(0.1, 0.5 * target / achieved) and continues from the current x (same PCG / preconditioner setup).
        Returns the totals and the per-run records."""
        setup_s = self.setup()
        tol = float(target) * float(first_tol_factor)
        runs = []
        verify_s = 0.0
        for attempt in range(max_restarts + 1):
            record = self.run(tol)
            start = time.perf_counter()
            achieved = float(residual_v(self.get_solution()))
            verify_s += time.perf_counter() - start
            record["relres_V"] = achieved
            runs.append(record)
            if achieved <= target or not record["converged"]:
                break
            tol *= max(0.1, 0.5 * target / achieved)
        result = dict(setup_s=setup_s, solve_s=sum(r["solve_s"] for r in runs),
                      iterations=sum(r["iterations"] for r in runs), relres_V=runs[-1]["relres_V"],
                      final_relative_residual=runs[-1]["final_relative_residual"], restarts=len(runs) - 1,
                      verify_s=verify_s, converged=bool(runs[-1]["relres_V"] <= target), runs=runs, **self.options)
        if not result["converged"] and self.raise_on_nonconvergence:
            raise SimulationHardStop("hypre_pressure_not_converged", result)
        return result

    def close(self):
        h = self.hypre
        self.release()
        for name in ("A", "b", "x"):
            handle = getattr(self, name, None)
            if handle:
                h.call("HYPRE_StructMatrixDestroy" if name == "A" else "HYPRE_StructVectorDestroy", handle)
                setattr(self, name, None)
        if getattr(self, "stencil", None):
            h.call("HYPRE_StructStencilDestroy", self.stencil)
            self.stencil = None
        if getattr(self, "grid", None):
            h.call("HYPRE_StructGridDestroy", self.grid)
            self.grid = None
        for pointer in self._device.values():
            h.cudart.free(pointer)
        self._device = {}


# ---------------------------------------------------------------------------------------------------- export (Taichi)
def export_pressure_system(fluid):
    """The exact FV operator of `fluid` (diag incl. outlet ghost term + interface diagonal, six face products), cell
    volumes and obstacle mask as host arrays (solver layout).  b / p are not included (take cg_rhs / pressure).
    Raises when interface rows or legacy coupling slots are active (not representable as a 7-point Struct matrix)."""
    import taichi as ti

    if int(fluid.pressure_interface_row_count[None]) != 0:
        raise NotImplementedError("pressure-interface rows are active: not a 7-point operator")
    if int(np.count_nonzero(fluid.pressure_interface_coupling_active.to_numpy())) != 0:
        raise NotImplementedError("pressure-interface coupling slots are active: not a 7-point operator")
    shape = tuple(int(v) for v in fluid.grid.grid_nodes)
    diag = np.zeros(shape, np.float64)
    products = np.zeros(shape + (6,), np.float64)
    volume = np.zeros(shape, np.float64)
    kernel = _export_kernel()
    kernel(fluid, diag, products, volume, int(fluid._velocity_dirichlet_boundary_authority_code()))
    ti.sync()
    return dict(diag=diag, products=products, volume_m3=volume, obstacle=fluid.obstacle.to_numpy().astype(np.int8))


_EXPORT_KERNEL = []


def _export_kernel():
    if _EXPORT_KERNEL:
        return _EXPORT_KERNEL[0]
    import taichi as ti

    @ti.kernel
    def export(fluid: ti.template(), diag: ti.types.ndarray(), products: ti.types.ndarray(),
               volume: ti.types.ndarray(), authority: ti.template()):
        for i, j, k in fluid.obstacle:
            if fluid.obstacle[i, j, k] == 0:
                face, denominator = fluid._fv_pressure_face_products_and_denominator(
                    fluid.obstacle, fluid.velocity_dirichlet_boundary_active,
                    fluid.velocity_dirichlet_boundary_projection_weight,
                    fluid.velocity_dirichlet_pressure_hard_fixed_component_mask,
                    fluid.velocity_dirichlet_boundary_active_component_mask,
                    fluid.velocity_dirichlet_boundary_pressure_mobility,
                    fluid.velocity_dirichlet_boundary_hard_fixed_component_mask,
                    fluid.cell_width_x_m, fluid.cell_width_y_m, fluid.cell_width_z_m,
                    fluid.center_distance_x_m, fluid.center_distance_y_m, fluid.center_distance_z_m,
                    i, j, k, fluid.nx, fluid.ny, fluid.nz, 1, authority)
                diag[i, j, k] = denominator + fluid.pressure_interface_matrix_diagonal[i, j, k]
                for f in ti.static(range(6)):
                    products[i, j, k, f] = face[f]
                volume[i, j, k] = fluid._cell_volume_f64_m3(i, j, k)

    _EXPORT_KERNEL.append(export)
    return export


def describe(result):
    """One JSON line (benchmark output helper)."""
    return json.dumps({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in result.items()},
                      sort_keys=True, default=str)


# ---------------------------------------------------------------------------------------------------- fluid hook
HOOK_DEFAULTS = dict(solver="pfmg", relax_type=1, rap_type=0, skip_relax=0, num_relax=1, tol_factor=0.4,
                     max_restarts=4, max_iter=400, safety=0.999)       # the best measured GPU configuration
_HOOK_KERNELS = []


def _hook_kernels():
    """Taichi kernels of the hook (created on first use so that the module itself does not need Taichi)."""
    if _HOOK_KERNELS:
        return _HOOK_KERNELS[0]
    import types as _types

    import taichi as ti

    @ti.kernel
    def pack_matrix(fluid: ti.template(), authority: ti.template(), rows: ti.types.ndarray(),
                    diag: ti.types.ndarray(), stats: ti.types.ndarray()):
        """V-form rows in hypre order (x fastest): diag = V (den + D) for fluid cells, V (2/wx^2 + 2/wy^2 + 2/wz^2)
        for obstacle identity rows; couplings -V T for -x, -y, -z.  `rows` keeps the last packed values: stats[0] =
        rows whose couplings changed, stats[1] = rows whose diagonal changed, stats[2] = fluid cells with legacy
        pressure-interface coupling slots (not representable: the hook falls back), stats[4:7] / stats[7:10] = lower /
        upper cell index of the box holding every changed diagonal."""
        for c in ti.static(range(3)):
            stats[c] = 0
        stats[4] = fluid.nx
        stats[5] = fluid.ny
        stats[6] = fluid.nz
        stats[7] = -1
        stats[8] = -1
        stats[9] = -1
        for i, j, k in fluid.obstacle:
            n = (k * fluid.ny + j) * fluid.nx + i
            volume = fluid._cell_volume_f64_m3(i, j, k)
            d = ti.cast(0.0, ti.f64)
            cx = ti.cast(0.0, ti.f64)
            cy = ti.cast(0.0, ti.f64)
            cz = ti.cast(0.0, ti.f64)
            if fluid.obstacle[i, j, k] == 0:
                face, den = fluid._fv_pressure_face_products_and_denominator(
                    fluid.obstacle, fluid.velocity_dirichlet_boundary_active,
                    fluid.velocity_dirichlet_boundary_projection_weight,
                    fluid.velocity_dirichlet_pressure_hard_fixed_component_mask,
                    fluid.velocity_dirichlet_boundary_active_component_mask,
                    fluid.velocity_dirichlet_boundary_pressure_mobility,
                    fluid.velocity_dirichlet_boundary_hard_fixed_component_mask,
                    fluid.cell_width_x_m, fluid.cell_width_y_m, fluid.cell_width_z_m,
                    fluid.center_distance_x_m, fluid.center_distance_y_m, fluid.center_distance_z_m,
                    i, j, k, fluid.nx, fluid.ny, fluid.nz, 1, authority)
                d = volume * (den + fluid.pressure_interface_matrix_diagonal[i, j, k])
                cx = -volume * face[0]
                cy = -volume * face[2]
                cz = -volume * face[4]
                if fluid.pressure_interface_coupling_active[i, j, k] != 0:
                    ti.atomic_add(stats[2], 1)
            else:
                wx = ti.cast(fluid.cell_width_x_m[i], ti.f64)
                wy = ti.cast(fluid.cell_width_y_m[j], ti.f64)
                wz = ti.cast(fluid.cell_width_z_m[k], ti.f64)
                d = volume * (2.0 / (wx * wx) + 2.0 / (wy * wy) + 2.0 / (wz * wz))
            if not (rows[n, 1] == cx and rows[n, 2] == cy and rows[n, 3] == cz):
                ti.atomic_add(stats[0], 1)
                rows[n, 1] = cx
                rows[n, 2] = cy
                rows[n, 3] = cz
            if not (rows[n, 0] == d):
                ti.atomic_add(stats[1], 1)
                rows[n, 0] = d
                ti.atomic_min(stats[4], i)
                ti.atomic_min(stats[5], j)
                ti.atomic_min(stats[6], k)
                ti.atomic_max(stats[7], i)
                ti.atomic_max(stats[8], j)
                ti.atomic_max(stats[9], k)
            diag[n] = d

    @ti.kernel
    def pack_vectors(fluid: ti.template(), rhs: ti.types.ndarray(), x0: ti.types.ndarray()):
        """b'' = V cg_rhs and x0 = pressure (fluid cells; 0 in obstacle cells), hypre order."""
        for i, j, k in fluid.obstacle:
            n = (k * fluid.ny + j) * fluid.nx + i
            value = ti.cast(0.0, ti.f64)
            guess = ti.cast(0.0, ti.f64)
            if fluid.obstacle[i, j, k] == 0:
                value = fluid._cell_volume_f64_m3(i, j, k) * ti.cast(fluid.cg_rhs[i, j, k], ti.f64)
                guess = ti.cast(fluid.pressure[i, j, k], ti.f64)
            rhs[n] = value
            x0[n] = guess

    @ti.kernel
    def scatter_solution(fluid: ti.template(), x: ti.types.ndarray(), stats: ti.types.ndarray()):
        """pressure = x (fluid cells), 0 in obstacle cells (as the solver's CG); stats[3] = non-finite values."""
        stats[3] = 0
        for i, j, k in fluid.obstacle:
            n = (k * fluid.ny + j) * fluid.nx + i
            value = ti.cast(0.0, ti.f64)
            if fluid.obstacle[i, j, k] == 0:
                value = x[n]
                if not (ti.abs(value) < 1.0e300):
                    ti.atomic_add(stats[3], 1)
            fluid.pressure[i, j, k] = value

    @ti.kernel
    def pack_box(fluid: ti.template(), rows: ti.types.ndarray(), out: ti.types.ndarray(), x0: ti.i32, y0: ti.i32,
                 z0: ti.i32, bx: ti.i32, by: ti.i32, bz: ti.i32):
        """Diagonal entries of the box [x0, x0 + bx) x [y0, y0 + by) x [z0, z0 + bz), x fastest (hypre box order)."""
        for m in range(bx * by * bz):
            i = x0 + m % bx
            j = y0 + (m // bx) % by
            k = z0 + m // (bx * by)
            out[m] = rows[(k * fluid.ny + j) * fluid.nx + i, 0]

    @ti.kernel
    def fill_probe(probe: ti.types.ndarray()):
        for q in range(probe.shape[0]):
            probe[q] = 1.5 + 0.25 * q

    namespace = _types.SimpleNamespace(pack_matrix=pack_matrix, pack_vectors=pack_vectors, pack_box=pack_box,
                                       scatter_solution=scatter_solution, fill_probe=fill_probe)
    _HOOK_KERNELS.append(namespace)
    return namespace


class HyprePressureSolver:
    """Replaces ONLY the linear solve of fluid.project() on ONE fluid solver instance by hypre Struct PCG + PFMG.
    Installed as the instance attribute fluid._solve_pressure_poisson_fv_cg (the class method and every other instance
    are unchanged); project()'s right-hand side assembly, reachability, pressure-outlet cleanup passes, velocity
    correction and a pressure chamber's after_project are untouched.

    Per call (each FV-CG solve of project(): the main projection and any outlet cleanup pass):
      1. eligibility: z-min outlet, no marker nullspace projector, no nullspace / component projection, no unreached
         cells, no pressure-interface rows or legacy coupling slots -> otherwise the original CG runs (counted);
      2. RHS with the solver's own kernels (_mg_prepare_level0_kernel + _cg_build_positive_rhs_kernel into cg_rhs) and
         ||b||_V with its own weighted dot; the current pressure is the warm start;
      3. matrix: one Taichi kernel packs the V-form rows (solver's _fv_pressure_face_products_and_denominator +
         pressure_interface_matrix_diagonal) and compares them with the last packed rows: couplings changed (mask /
         obstacle change) -> upload all stored entries; only diagonals changed (pressure chamber) -> upload the diagonal;
         nothing changed -> no upload and the PFMG setup is REUSED;
      4. hypre PCG to tol_factor * tol (hypre norm), solution scattered into fluid.pressure, then the SOLVER's
         criterion ||b - A p||_V <= safety * tol ||b||_V checked on the device with the solver's own operator and
         weighted dot (_apply_fv_pressure_operator, _axpby_scalar_field_kernel, _weighted_dot_to_field_kernel; one
         scalar read); not met -> continue PCG with a tighter hypre tolerance (max_restarts);
      5. the ORIGINAL _solve_pressure_poisson_fv_cg is then called with the same arguments: it rebuilds the same RHS,
         evaluates the same criterion on the hypre solution and returns with 0 CG iterations (all of project()'s CG
         bookkeeping is therefore the solver's own); should hypre have failed, it continues with its own CG from the
         hypre solution (or from the restored initial guess if the hypre result was non-finite / worse).
    Hypre statistics: hook.last (this call) and hook.totals; fluid.last_cg_iterations stays the CG's (normally 0).
    A hypre error (SimulationHardStop 'hypre_error', e.g. a CUDA failure) restores the initial guess and lets the original
    CG do that solve (fallback='cg', default; counted in totals['fallbacks']['hypre_error'], next call re-uploads the
    full matrix with a new setup) or is raised (fallback='raise').

    DATA PATH ("host", default): Taichi ndarrays (hypre order) -> numpy (D2H on a CUDA Taichi) -> hypre (H2D on a CUDA
    hypre); solution back the same way.  Per call: rhs + x0 (16 N bytes); the diagonal only over the box of changed
    diagonals (pressure chamber: 8 M bytes, M = box cells; the compact box buffer grows on demand); all 4 entries only on a
    mask change (32 N); solution 8 N per PCG run (+ 8 N back into Taichi).  ("pointer"): the ndarray
    data pointers are passed to hypre directly (no host copies); valid on CPU, and on CUDA only when Taichi and hypre
    share the CUDA context (Taichi makes its context current with cuCtxSetCurrent; the CUDA runtime used by hypre adopts
    a current driver context) -- probed at construction (a Taichi-written ndarray read back through cudaMemcpy), with an
    automatic fall back to "host" if the probe fails.
    """

    def __init__(self, fluid, *, prefix=None, data_path="host", fallback="cg", **options):
        import taichi as ti
        self.ti = ti
        unknown = set(options) - set(HOOK_DEFAULTS)
        if unknown:
            raise ValueError("unknown hook options %s" % sorted(unknown))
        self.options = dict(HOOK_DEFAULTS, **options)
        if fallback not in ("cg", "raise"):
            raise ValueError("fallback must be 'cg' or 'raise'")
        self.fallback = fallback
        self.fluid = fluid
        self.shape = (int(fluid.nx), int(fluid.ny), int(fluid.nz))
        self.cells = int(np.prod(self.shape))
        o = self.options
        self.solver = StructPressureSolver(self.shape, prefix=prefix, solver=o["solver"], tol=o["tol_factor"] * 1e-5,
                                           max_iter=o["max_iter"], relax_type=o["relax_type"],
                                           rap_type=o["rap_type"], skip_relax=o["skip_relax"],
                                           num_relax=o["num_relax"])
        self.kernels = _hook_kernels()
        n = self.cells
        self.rows = ti.ndarray(ti.f64, shape=(n, 4))
        self.rows.fill(float("nan"))                       # first call: every row "changed" -> full upload
        self.diag = ti.ndarray(ti.f64, shape=n)
        self.rhs = ti.ndarray(ti.f64, shape=n)
        self.x0 = ti.ndarray(ti.f64, shape=n)
        self.x = ti.ndarray(ti.f64, shape=n)
        self.stats = ti.ndarray(ti.i32, shape=10)
        self.box_capacity = 0
        self.box = None                                    # compact changed-diagonal box buffer (grown on demand)
        self.setup_valid = False
        # the implementation project() really calls: on CUDA, fastfsi.fluid.device_controls binds the device solve_cg as the
        # instance attribute (a method bound to this fluid; its __func__ takes the fluid like the class method); on
        # CPU there is none and the class method is used.  A second hypre hook, or an unknown wrapper, is refused.
        bound = fluid.__dict__.get("_solve_pressure_poisson_fv_cg")
        if bound is None:
            self.original = type(fluid)._solve_pressure_poisson_fv_cg
        elif getattr(getattr(bound, "__func__", None), "_hypre_pressure_hook", False):
            raise RuntimeError("this fluid instance already has the hypre pressure hook")
        elif getattr(bound, "__self__", None) is fluid and callable(getattr(bound, "__func__", None)):
            self.original = bound.__func__
        else:
            raise RuntimeError("unknown instance override of _solve_pressure_poisson_fv_cg: %r" % (bound,))
        self.previous_instance_attribute = bound
        self.taichi_cuda = ti.lang.impl.current_cfg().arch == ti.cuda
        self.data_path, self.data_path_note = self._choose_data_path(data_path)
        self.totals = dict(calls=0, hypre_calls=0, fallbacks={}, setups=0, setups_reused=0, hypre_iterations=0,
                           hypre_restarts=0, cg_polish_iterations=0, matrix_full_uploads=0, diagonal_uploads=0,
                           h2d_bytes=0, d2h_bytes=0, pack_s=0.0, upload_s=0.0, setup_s=0.0, solve_s=0.0,
                           download_s=0.0, check_s=0.0, confirm_s=0.0)
        self.last = {}
        self.installed = False

    # ------------------------------------------------------------------------------------------------ checkpoint
    # state census: the running totals (history columns: hypre iterations, fallbacks) and the transfer counters
    # are saved.  The packed matrix (rows), the matrix in hypre's memory and the preconditioner setup are NOT: a fresh
    # process starts with rows = NaN, so its first solve uploads the full matrix and sets the preconditioner up again --
    # the same matrix values and the same (deterministic) setup that the continued run holds.
    def state_host(self):
        return dict(totals=self.totals, transfers=self.solver.transfers)

    def load_state_host(self, values):
        if set(values) != {"totals", "transfers"}:
            raise ValueError("hypre hook host state %s, expected totals and transfers" % sorted(values))
        for current, saved in ((self.totals, values["totals"]), (self.solver.transfers, values["transfers"])):
            current.clear()
            current.update(saved)

    # ------------------------------------------------------------------------------------------------ data path
    def _pointer(self, array):
        return int(self.ti.lang.impl.get_runtime().prog.get_ndarray_data_ptr_as_int(array.arr))

    def _choose_data_path(self, requested):
        if requested not in ("host", "pointer"):
            raise ValueError("data_path must be 'host' or 'pointer'")
        if requested == "host":
            return "host", "host staging (numpy)"
        cuda_hypre = self.solver.hypre.cuda
        if cuda_hypre != self.taichi_cuda:
            return "host", "pointer path impossible: hypre %s build with Taichi on %s" % (
                "CUDA" if cuda_hypre else "CPU", "CUDA" if self.taichi_cuda else "CPU")
        if not cuda_hypre:
            return "pointer", "CPU: host pointers shared"
        probe = self.ti.ndarray(self.ti.f64, shape=8)
        self.kernels.fill_probe(probe)
        self.ti.sync()
        out = np.zeros(8, np.float64)
        try:
            self.solver.hypre.cudart.copy(out.ctypes.data, self._pointer(probe), out.nbytes, CudaRuntime.D2H)
        except SimulationHardStop as error:
            return "host", "pointer probe failed (%s): host staging" % (error.witness,)
        if not np.array_equal(out, 1.5 + 0.25 * np.arange(8)):
            return "host", "pointer probe read wrong values: host staging"
        return "pointer", "CUDA: Taichi ndarray readable through the CUDA runtime (shared context)"

    def _upload_matrix(self, entries, box=None):
        """All stored entries (mask change) or the diagonal of the changed box (pressure chamber)."""
        if box is not None:
            (x0, y0, z0), (x1, y1, z1) = box
            count = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1)
            if count > self.box_capacity:
                self.box_capacity = 1 << max(10, int(count - 1).bit_length())
                self.box = self.ti.ndarray(self.ti.f64, shape=self.box_capacity)
            self.kernels.pack_box(self.fluid, self.rows, self.box, x0, y0, z0, x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)
            if self.data_path == "pointer":
                self.solver.set_matrix_entries(self._pointer(self.box), (0,), box=box, pointer=True)
                return 0, 0
            values = self.box.to_numpy()
            self.solver.set_matrix_entries(values[:count], (0,), box=box)
            return 8 * count, values.nbytes
        if self.data_path == "pointer":
            self.solver.set_matrix_entries(self._pointer(self.rows), entries, pointer=True)
            return 0, 0
        array = self.rows.to_numpy()
        self.solver.set_matrix_entries(array, entries)
        return array.nbytes, array.nbytes

    def _upload_vectors(self):
        if self.data_path == "pointer":
            self.solver.set_vector("rhs", self._pointer(self.rhs), pointer=True)
            self.solver.set_vector("x", self._pointer(self.x0), pointer=True)
            return 0, 0
        rhs, x0 = self.rhs.to_numpy(), self.x0.to_numpy()
        self.solver.set_vector("rhs", rhs)
        self.solver.set_vector("x", x0)
        return rhs.nbytes + x0.nbytes, rhs.nbytes + x0.nbytes

    def _download_solution(self):
        if self.data_path == "pointer":
            self.solver.get_solution_pointer(self._pointer(self.x))
            self.solver.hypre.synchronize()
            return 0, 0
        x = self.solver.get_solution()
        self.x.from_numpy(x)
        return x.nbytes, x.nbytes

    # ------------------------------------------------------------------------------------------------ hook
    def install(self):
        fluid = self.fluid
        if self.installed or fluid.__dict__.get("_solve_pressure_poisson_fv_cg") is not self.previous_instance_attribute:
            raise RuntimeError("the pressure solve of this fluid changed since the hook was built (or is hooked)")
        hook = self

        def solve(this, **kwargs):
            return hook._solve(this, kwargs)
        solve._hypre_pressure_hook = True

        # the solver's no-free-face bound-site audit (solver.py assert_no_free_face_rule_bound) follows __wrapped__ to
        # the original's marker; with that rule ON the hook always delegates to the original (see _ineligible)
        solve.__wrapped__ = self.original
        fluid._solve_pressure_poisson_fv_cg = solve.__get__(fluid)
        self.installed = True
        return self

    def uninstall(self):
        if self.installed:
            if self.previous_instance_attribute is None:
                del self.fluid.__dict__["_solve_pressure_poisson_fv_cg"]
            else:
                self.fluid._solve_pressure_poisson_fv_cg = self.previous_instance_attribute
            self.installed = False

    def close(self):
        self.uninstall()
        self.solver.close()

    def _ineligible(self, fluid, kwargs):
        if bool(getattr(fluid, "no_free_face_isolated_cells_solid", False)):
            return "no_free_face_rule"
        if not kwargs.get("pressure_outlet_zmin", False):
            return "no_zmin_pressure_outlet"
        if kwargs.get("pressure_velocity_nullspace_projector") is not None:
            return "marker_nullspace_projector"
        if kwargs.get("remove_nullspace_mean", True):
            return "nullspace_mean_projection"
        if int(kwargs.get("pressure_nullspace_component_count", 0) or 0) > 0:
            return "operator_nullspace_components"
        if int(getattr(fluid, "_pressure_unreached_count", 0) or 0) > 0:
            return "unreached_cells"
        if int(fluid.pressure_interface_row_count[None]) != 0:
            return "pressure_interface_rows"
        return None

    def _relative_residual_v(self, fluid, authority, b_norm):
        fluid._apply_fv_pressure_operator(fluid.pressure, fluid.cg_Ad, 1, authority)
        fluid._axpby_scalar_field_kernel(fluid.cg_r, 1.0, fluid.cg_rhs, -1.0, fluid.cg_Ad)
        fluid._weighted_dot_to_field_kernel(fluid.cg_r, fluid.cg_r, fluid.cg_rr)
        return math.sqrt(max(float(fluid.cg_rr[None]), 0.0)) / b_norm

    def _fallback(self, reason):
        self.totals["fallbacks"][reason] = self.totals["fallbacks"].get(reason, 0) + 1
        self.last["fallback"] = reason
        if self.fallback == "raise" and reason not in ("trivial_rhs", "already_converged"):
            raise SimulationHardStop("hypre_pressure_not_converged", dict(reason=reason))

    def _solve(self, fluid, kwargs):
        ti, k, o, totals = self.ti, self.kernels, self.options, self.totals
        totals["calls"] += 1
        self.last = dict(fallback=None, hypre_iterations=0, restarts=0, setup="none", upload="none")
        reason = self._ineligible(fluid, kwargs)
        if reason is not None:
            self._fallback(reason)
            return self.original(fluid, **kwargs)
        authority = int(fluid._velocity_dirichlet_boundary_authority_code())
        target = float(kwargs["tolerance"])
        clock = time.perf_counter
        ti.sync()
        start = clock()
        fluid._mg_prepare_level0_kernel(fluid._mg_rhs[0], fluid._mg_residual[0], float(kwargs["rhs_scale"]))
        fluid._cg_build_positive_rhs_kernel(fluid._mg_rhs[0], fluid.cg_rhs, 0.0)
        b_norm = math.sqrt(max(float(fluid._weighted_dot_kernel(fluid.cg_rhs, fluid.cg_rhs)), 0.0))
        if b_norm <= 1.0e-30:
            self._fallback("trivial_rhs")
            return self._confirm(fluid, kwargs)
        initial = self._relative_residual_v(fluid, authority, b_norm)
        self.last["initial_relres_V"] = initial
        if initial <= o["safety"] * target:
            self._fallback("already_converged")
            return self._confirm(fluid, kwargs)
        k.pack_matrix(fluid, authority, self.rows, self.diag, self.stats)
        k.pack_vectors(fluid, self.rhs, self.x0)
        stats = self.stats.to_numpy()
        ti.sync()
        packed = clock()
        totals["pack_s"] += packed - start
        if int(stats[2]) > 0:
            self.rows.fill(float("nan"))                  # force a full upload once the slots are gone
            self.setup_valid = False
            self._fallback("legacy_pressure_coupling_slots")
            return self.original(fluid, **kwargs)
        h2d = d2h = 0
        achieved = math.inf
        nonfinite = 0
        tol = o["tol_factor"] * target
        try:
            if int(stats[0]) > 0:
                up, down = self._upload_matrix((0, 1, 3, 5))
                totals["matrix_full_uploads"] += 1
                self.last["upload"] = "matrix"
                self.setup_valid = False
            elif int(stats[1]) > 0:
                box = (tuple(int(v) for v in stats[4:7]), tuple(int(v) for v in stats[7:10]))
                up, down = self._upload_matrix((0,), box=box)
                totals["diagonal_uploads"] += 1
                self.last["upload"] = "diagonal"
                self.last["diagonal_box"] = [list(box[0]), list(box[1])]
                self.setup_valid = False
            else:
                up = down = 0
            h2d, d2h = h2d + up, d2h + down
            up, down = self._upload_vectors()
            h2d, d2h = h2d + up, d2h + down
            self.solver.hypre.synchronize()
            uploaded = clock()
            totals["upload_s"] += uploaded - packed
            if not self.setup_valid:
                totals["setup_s"] += self.solver.setup()
                totals["setups"] += 1
                self.setup_valid = True
                self.last["setup"] = "new"
            else:
                totals["setups_reused"] += 1
                self.last["setup"] = "reused"
            totals["hypre_calls"] += 1
            for attempt in range(o["max_restarts"] + 1):
                run = self.solver.run(tol)
                totals["solve_s"] += run["solve_s"]
                totals["hypre_iterations"] += run["iterations"]
                self.last["hypre_iterations"] += run["iterations"]
                begin = clock()
                up, down = self._download_solution()
                k.scatter_solution(fluid, self.x, self.stats)
                nonfinite = int(self.stats.to_numpy()[3])
                ti.sync()
                middle = clock()
                totals["download_s"] += middle - begin
                h2d, d2h = h2d + up, d2h + down
                achieved = self._relative_residual_v(fluid, authority, b_norm) if nonfinite == 0 else math.inf
                totals["check_s"] += clock() - middle
                if achieved <= o["safety"] * target or nonfinite or not run["converged"]:
                    break
                totals["hypre_restarts"] += 1
                self.last["restarts"] += 1
                tol *= max(0.1, 0.5 * target / achieved)
        except SimulationHardStop as error:
            if error.code != "hypre_error" or self.fallback == "raise":
                raise
            # hypre failed (not a convergence issue): restore the initial guess, force a full re-upload and a new
            # setup next time, and let the solver's own CG do this solve
            k.scatter_solution(fluid, self.x0, self.stats)
            self.rows.fill(float("nan"))
            self.setup_valid = False
            self.last["hypre_error"] = error.witness
            self._fallback("hypre_error")
            return self._confirm(fluid, kwargs)
        totals["h2d_bytes"] += h2d
        totals["d2h_bytes"] += d2h
        self.last.update(relres_V=achieved, hypre_final_tol=tol, h2d_bytes=h2d, d2h_bytes=d2h, nonfinite=nonfinite)
        if not achieved <= o["safety"] * target:
            if nonfinite or not achieved < initial:
                k.scatter_solution(fluid, self.x0, self.stats)     # restore the solver's initial guess
                self.last["restored_initial_guess"] = True
            self._fallback("hypre_not_converged")
        return self._confirm(fluid, kwargs)

    def _confirm(self, fluid, kwargs):
        """The solver's own FV-CG on the current pressure: confirms the criterion with 0 iterations (all CG bookkeeping
        by the solver) or completes the solve with its own CG."""
        start = time.perf_counter()
        result = self.original(fluid, **kwargs)
        self.ti.sync()
        self.totals["confirm_s"] += time.perf_counter() - start
        polish = int(getattr(fluid, "last_cg_iterations", 0))
        self.totals["cg_polish_iterations"] += polish
        self.last.update(cg_polish_iterations=polish,
                         solver_relres=float(getattr(fluid, "last_cg_relative_residual", math.nan)),
                         solver_converged=bool(getattr(fluid, "last_cg_converged", False)))
        return result

    def report(self):
        """JSON-able totals (per-call means included) and the last call."""
        totals = dict(self.totals)
        calls = max(int(totals["hypre_calls"]), 1)
        for key in ("pack_s", "upload_s", "setup_s", "solve_s", "download_s", "check_s", "confirm_s"):
            totals[key.replace("_s", "_s_per_hypre_call")] = totals[key] / calls
        return dict(options=self.options, data_path=self.data_path,
                    data_path_note=self.data_path_note, hypre=self.solver.hypre.identity(), totals=totals,
                    last=dict(self.last))


def install_hypre_pressure_solver(fluid, **kwargs):
    """Convenience: HyprePressureSolver(fluid, **kwargs).install() (driver switch --pressure-solver hypre_pfmg)."""
    return HyprePressureSolver(fluid, **kwargs).install()
