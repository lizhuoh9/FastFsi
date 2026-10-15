"""Per-instance CUDA controllers of the fluid solver: on an actual CUDA runtime, the conjugate-gradient solve and its
exact-residual confirmation, the unreached / outlet component compatibility measures and the outlet-reachability flood
of THIS solver instance are replaced by their device-scalar versions (fastfsi.fluid.device_conjugate_gradient,
device_pressure_components, device_reachability); the CPU methods stay on the class.  A method already overridden by a
subclass or on the instance keeps its own implementation.
"""
import inspect
from types import MethodType

import taichi as ti

# the solver methods that have a device controller, in installation order
REPLACED_METHODS = (
    "_solve_pressure_poisson_fv_cg",
    "_confirm_pressure_poisson_fv_cg_exact_residual",
    "_measure_unreached_component_rhs_incompatibility",
    "_measure_pressure_outlet_nullspace_rhs_incompatibility",
    "_flood_pressure_outlet_reachable_frontier",
)


def device_replacements():
    """{solver method name: device controller function}; imports only, so it can be inspected on any arch."""
    from fastfsi.fluid.device_conjugate_gradient import confirm_exact, solve_cg
    from fastfsi.fluid.device_pressure_components import measure_outlet, measure_unreached
    from fastfsi.fluid.device_reachability import flood_frontier
    functions = (solve_cg, confirm_exact, measure_unreached, measure_outlet, flood_frontier)
    return dict(zip(REPLACED_METHODS, functions))


def _replace_native(owner, native_type, name, replacement):
    actual = inspect.getattr_static(type(owner), name)
    expected = inspect.getattr_static(native_type, name)
    if actual is expected and name not in owner.__dict__:
        setattr(owner, name, MethodType(replacement, owner))


def install_fluid_device_control(owner):
    if ti.lang.impl.current_cfg().arch != ti.cuda:
        return
    from fastfsi.fluid.device_arithmetic import DeviceArithmetic
    from fastfsi.fluid.device_pressure_components import DevicePressureComponents
    from fastfsi.fluid.solver import IncompressibleFluidSolver
    owner._device_arithmetic = DeviceArithmetic()
    owner._device_pressure_components = DevicePressureComponents()
    for name, replacement in device_replacements().items():
        _replace_native(owner, IncompressibleFluidSolver, name, replacement)
