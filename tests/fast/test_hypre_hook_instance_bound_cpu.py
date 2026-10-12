"""Integrator's hypre hook v2 (3080 smoke 2026-10-12: on CUDA, gpu_fluid_install binds the device solve_cg as the
instance attribute _solve_pressure_poisson_fv_cg, which Helper 1's CPU-tested hook refused).  On CPU the CUDA binding
is reproduced with a counting bound method: the hook must take it as its original (the 0-iteration confirmation goes
through it), refuse a second hypre hook, agree with the plain projection, and restore the instance attribute."""
from types import MethodType

import numpy as np
import pytest

from tests.fast import test_hypre_pressure_hook_cpu as H      # skips this module too when the hypre build is absent

hp = H.hp


def test_hook_wraps_the_instance_bound_solve_like_cuda():
    fluid, g, body, air = H.rig()
    obstacle = H.obstacle_mask()
    velocity = H.u_star(5, obstacle)
    zeros = np.zeros((H.NX, H.NY, H.NZ))
    plain = H.projection(fluid, air, velocity, zeros)
    native = type(fluid)._solve_pressure_poisson_fv_cg
    calls = {"n": 0}

    def device_like_solve(self, **kwargs):                # stands for gpu_fluid_host.solve_cg bound per instance
        calls["n"] += 1
        return native(self, **kwargs)
    fluid._solve_pressure_poisson_fv_cg = MethodType(device_like_solve, fluid)
    bound = fluid.__dict__["_solve_pressure_poisson_fv_cg"]
    try:
        hook = hp.HyprePressureHook(fluid)
        assert hook.original is device_like_solve and hook.previous_instance_attribute is bound
        hook.install()
        with pytest.raises(RuntimeError):
            hp.HyprePressureHook(fluid)                   # a second hypre hook is refused
        hooked = H.projection(fluid, air, velocity, zeros)
        totals = dict(hook.totals)
        hook.close()
        assert fluid.__dict__["_solve_pressure_poisson_fv_cg"] is bound          # restored, not deleted
    finally:
        fluid.__dict__.pop("_solve_pressure_poisson_fv_cg", None)
    assert totals["hypre_calls"] >= 1 and not totals["fallbacks"]
    assert calls["n"] >= totals["hypre_calls"]           # every hypre solve was confirmed by the instance's own solve
    assert hooked["cg_iterations"] == 0 and hooked["cg_converged"]
    scale = float(np.abs(plain["p"]).max())
    assert float(np.abs(hooked["p"] - plain["p"]).max()) <= 1.0e-3 * scale
