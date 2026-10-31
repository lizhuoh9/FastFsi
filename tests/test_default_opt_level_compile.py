"""The s26 outer chamber's opt-in kernels (Root 2026-10-31) compiled as the runs compile them: the library's own
init_taichi with Taichi's default opt_level (1), in a fresh interpreter.  The other tests run at opt_level 0 (the
deterministic CPU runtime of cpu_runtime.py and the modules' own ti.init), which hid a Taichi 1.7.4 IR-verifier failure
of CavityLedgerGas._finish_kernel at opt_level 1 until the first CPU mechanism run (2026-10-31).  The child runs
  1. the cavity rig of test_cavity_gas: the upper half shifted down 0.5 mm, six accepts (past the hysteresis), the
     subregion count, a state round trip (the air-cell kernels): 480 mm^3, p_atm (512 / 480 - 1), 448 cells, one part;
  2. PressureRegions with an attached external region on the duct of test_pressure_regions (update, projection,
     after_project, report): the 27 block cells at the caller's gauge, the water as the explicit-Dirichlet reference;
  3. per-body materials of test_body_materials through midpoint steps and the spread kernel: body 2's own stress;
  4. the same kernels' default variants (no external region, no table; their text changed with the options): the duct's
     inlet air and its divergence bookkeeping, the single material's deformation."""
import json
import os
import subprocess
import sys

import pytest

from fastfsi.boundary.gas_pockets import P_ATMOSPHERE_PA

CHILD = r"""
import json

import numpy as np
import taichi as ti

from fastfsi.simulation.taichi_runtime import TaichiRuntimeConfig, init_taichi

RUNTIME = TaichiRuntimeConfig(arch="cpu", default_fp="f32", random_seed=0, offline_cache=False, cpu_max_num_threads=1)
init_taichi(RUNTIME)
out = dict(opt_level=int(ti.cfg.opt_level))

import test_cavity_gas as C
cavity, points, air, partner, positions = C.build()
u = np.zeros_like(positions)
u[positions[:, 2] > 12 * C.MM, 2] = -0.5 * C.MM
C.displace(points, positions, u)
for step in range(6):
    r = cavity.accept(1.0e-4 * (step + 1), 0.0)
arrays = {name: cavity.state_array(name) for name in cavity.state_array_names()}
host = cavity.state_host()
cavity.load_state_arrays(arrays)
cavity.load_state_host(host)
out["cavity"] = dict(volume_mm3=r["volume_m3"] / C.MM ** 3, p_gauge_pa=r["p_gauge_pa"], q0_pa_m3=cavity.q0_pa_m3,
                     volume_m3=r["volume_m3"], cells=r["cavity_cells"], subregions=cavity.subregions(),
                     air_cells=int(air.to_numpy().sum()))

import test_pressure_regions as R
import test_pressure_regions_external as E
from fastfsi.boundary.pressure_regions import PressureRegions
from fastfsi.fluid.solver import IncompressibleFluidSolver
from fastfsi.grid.specification import FluidDomainSpecification
grid = R.duct_grid()
fluid = IncompressibleFluidSolver(
    FluidDomainSpecification(bounds_min_m=grid.bounds_min_m, bounds_max_m=grid.bounds_max_m, grid_nodes=grid.grid_nodes,
                             density_kgm3=R.RHO, viscosity_pa_s=R.MU, dt_s=R.DT, cartesian_grid=grid),
    runtime=RUNTIME, multigrid_semi_coarsening=True)
shape = grid.grid_nodes
fluid.velocity.from_numpy(np.zeros(shape + (3,), np.float32))
fluid.velocity_prev.from_numpy(np.zeros(shape + (3,), np.float32))
fluid.pressure.from_numpy(np.zeros(shape, np.float64))
fluid.obstacle.from_numpy(np.zeros(shape, np.int32))
fluid.clear_volume_source()
fluid.set_velocity_dirichlet_boundary_authority("canonical")
fluid.prepare_and_seal_velocity_dirichlet_component_registry()
fluid.mark_pressure_outlet_disconnected_nonprojectable_cells(pressure_outlet_zmin=True)
obstacle = R.obstacle_block()
R.set_obstacle(fluid, obstacle)
u_star = R.random_u_star(7, obstacle)
regions = PressureRegions(fluid, E.REGION_BOX)
code, gauge = E.external_fields()
regions.attach_external_region(E.EXTERNAL_BOX, code, gauge)
regions.set_regions(E.region_codes(), [])
dp = 6000.0
gauge[None] = R.P_IN + dp
fluid.velocity.from_numpy(u_star)
regions.update(0.5, R.P_IN)
R.project(fluid)
regions.after_project(R.DT, R.P_IN)
physical = regions.physical_pressure_field().to_numpy()
report = regions.report()
values = np.full(shape, np.nan)
values[:, :, R.K_AIR:] = R.P_IN
values[E.BLOCK] = R.P_IN + dp
p_ref, _, water = R.dirichlet_reference(grid, obstacle, values, u_star)
out["external"] = dict(cells=report["external_cells"], conflicts=report["external_conflicts"],
                       block_error_pa=float(np.abs(physical[E.BLOCK] - (R.P_IN + dp)).max()),
                       water_error_pa=float(np.abs(physical[water] - p_ref[water]).max()), scale_pa=abs(R.P_IN) + dp)

plain = PressureRegions(fluid, E.REGION_BOX)                # 4. the default variants: no external region, no table
plain.set_regions(E.region_codes(), [])
fluid.velocity.from_numpy(u_star)
fluid.pressure.from_numpy(np.zeros(shape, np.float64))
plain.update(0.5, R.P_IN)
R.project(fluid)
plain.after_project(R.DT, R.P_IN)
default_report = plain.report()

import test_body_materials as B
velocity = ti.Vector.field(3, ti.f32, shape=(8, 8, 8))
shear = np.zeros((8, 8, 8, 3), np.float32)
shear[..., 0] = B.SHEAR * (np.arange(8, dtype=np.float32) + 0.5)[None, None, :]
velocity.from_numpy(shear)
F, tau, force = B.deform_and_spread(*B.coupler(velocity, table={2: B.STIFF}), velocity)
reference = B.neo_hookean_tau(F[2].astype(np.float64), 5., 50.)
out["body"] = dict(tau_relative_error=float(np.abs(tau[2] - reference).max() / np.abs(reference).max()),
                   force_finite=bool(np.isfinite(force).all()), F_xz=float(F[0, 0, 2]),
                   F_xz_expected=B.SHEAR * B.STEPS * B.DT)
F0, tau0, _ = B.deform_and_spread(*B.coupler(velocity), velocity)
out["default"] = dict(inlet_cells=default_report["inlet_cells"], external_cells=default_report.get("external_cells"),
                      water_divergence_1ps=default_report["water_divergence_max_1ps"],
                      inlet_cells_expected=R.NX * R.NY * (R.NZ - R.K_AIR), tau_finite=bool(np.isfinite(tau0).all()),
                      F_xz=float(F0[0, 0, 2]), F_xz_expected=B.SHEAR * B.STEPS * B.DT)
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def child(tmp_path_factory):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p), PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="", TI_SKIP_VERSION_CHECK="ON")
    result = subprocess.run([sys.executable, "-B", "-c", CHILD], env=env, capture_output=True, text=True,
                            timeout=1800, cwd=tmp_path_factory.mktemp("default_opt_level"))
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_child_compiled_at_the_default_opt_level(child):
    assert child["opt_level"] == 1


def test_the_cavity_closes_and_keeps_its_ledger(child):
    c = child["cavity"]
    assert c["volume_mm3"] == pytest.approx(480.0, rel=1e-6)                    # the f32 displacement's rounding
    assert c["p_gauge_pa"] == pytest.approx(c["q0_pa_m3"] / c["volume_m3"] - P_ATMOSPHERE_PA, rel=1e-12)
    assert c["p_gauge_pa"] == pytest.approx(P_ATMOSPHERE_PA * (512.0 / 480.0 - 1.0), rel=1e-5)
    assert c["cells"] == 448 and c["subregions"] == 1 and c["air_cells"] == 448


def test_the_external_region_is_dirichlet_at_the_gauge(child):
    e = child["external"]
    assert e["cells"] == 27 and e["conflicts"] == 0
    assert e["block_error_pa"] <= 1.0e-6 * e["scale_pa"] and e["water_error_pa"] <= 1.0e-6 * e["scale_pa"]


def test_each_body_takes_its_own_material_in_the_spread_kernel(child):
    b = child["body"]
    assert b["tau_relative_error"] <= 1.0e-5 and b["force_finite"]
    assert b["F_xz"] == pytest.approx(b["F_xz_expected"], rel=1e-3)


def test_the_default_variants_compile_and_run(child):
    d = child["default"]
    assert d["inlet_cells"] == d["inlet_cells_expected"] and d["external_cells"] is None
    assert d["water_divergence_1ps"] <= 1.0e-3 and d["tau_finite"]
    assert d["F_xz"] == pytest.approx(d["F_xz_expected"], rel=1e-3)
