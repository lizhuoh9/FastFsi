"""Install private device controllers only for an actual initialized CUDA owner.

CPU methods stay on their original class. Unknown method overrides use their
native implementation; no subclass or third-party callback is rewritten.
"""
import inspect
from types import MethodType
import taichi as ti


def _replace_native(owner, native_type, name, replacement):
    actual = inspect.getattr_static(type(owner), name)
    expected = inspect.getattr_static(native_type, name)
    if actual is expected and name not in owner.__dict__:
        setattr(owner, name, MethodType(replacement, owner))


def install_fluid_device_control(owner):
    if ti.lang.impl.current_cfg().arch != ti.cuda:
        return
    from .solver import CartesianFluidSolver
    from .gpu_fluid_scalars import FluidScalars
    from .gpu_fluid_host import solve_cg, continue_bicgstab, confirm_exact
    from .gpu_pressure_components import (PressureComponents, measure_unreached,
                                         measure_outlet, measure_physical)
    from .gpu_fluid_reachability import flood_frontier
    from .gpu_divergence_report import (DivergenceReport, decode, read_final,
                                         raw_stats, residual_stats, partition_stats)
    from .gpu_divergence_fold import DivergenceFold, fold_report
    owner._h2_fluid_scalars = FluidScalars()
    owner._h2_pressure_components = PressureComponents()
    owner._h2_divergence_report = DivergenceReport()
    if 0 < owner.nx * owner.ny * owner.nz <= 2**31 - 1:
        owner._h2_divergence_fold = DivergenceFold((owner.nx, owner.ny, owner.nz))
        _replace_native(owner, CartesianFluidSolver,
                        '_fold_divergence_final_report_cuda_ordered_kernel', fold_report)
    for name, replacement in {
        '_solve_pressure_poisson_fv_cg': solve_cg,
        '_continue_pressure_poisson_fv_bicgstab': continue_bicgstab,
        '_confirm_pressure_poisson_fv_cg_exact_residual': confirm_exact,
        '_measure_unreached_component_rhs_incompatibility': measure_unreached,
        '_measure_pressure_outlet_nullspace_rhs_incompatibility': measure_outlet,
        '_measure_closed_component_physical_rhs_incompatibility': measure_physical,
        '_flood_hibm_pressure_outlet_reachable_frontier': flood_frontier,
        '_decode_divergence_report_snapshot': decode,
        '_read_divergence_final_report_snapshot': read_final,
        'divergence_stats': raw_stats,
        'divergence_residual_stats': residual_stats,
        'divergence_dirichlet_partition_report_stats': partition_stats,
    }.items():
        _replace_native(owner, CartesianFluidSolver, name, replacement)
    # REQ_373 runtime coverage guard: every bound pressure-gate site must be no-free-face rule aware.
    from .solver import assert_no_free_face_rule_bound
    assert_no_free_face_rule_bound(owner)
