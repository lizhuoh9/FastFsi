"""The fluid solver's call guards (Codex review 2026-10-26, checked by Root): the defaults of predict(), project() and
apply_body_force() are the supported paths, and an unsupported option is refused before any state changes (before,
apply_body_force(read_report=True) updated the velocity and then raised, so a caller that retried without the report
applied the force twice; project() applied the velocity rows before its first refusal).  The duct rig of
test_pressure_regions.py, on the CPU."""
import numpy as np
import pytest

import test_pressure_regions as duct

REPORT = "per-call solver reports"


@pytest.fixture(scope="module", autouse=True)
def own_duct_rig():
    """As test_outlet_contract.py: this module builds its own rig (another module may re-initialise Taichi)."""
    duct._STATE.clear()
    yield
    duct._STATE.clear()


def snapshot(fluid):
    return fluid.velocity.to_numpy(), fluid.pressure.to_numpy()


def assert_unchanged(fluid, before):
    for old, new in zip(before, snapshot(fluid)):
        assert np.array_equal(old, new)


def project_kwargs(**override):
    return dict(dict(iterations=400, pressure_reachability_prepared=True, dt_s=duct.DT, reset_pressure=False,
                     cg_tolerance=duct.CG_TOLERANCE, **duct.PROJECT_REQUIREMENTS), **override)


def test_unsupported_options_are_refused_before_any_state_change():
    fluid, grid, _ = duct.rig()
    fluid.velocity.from_numpy(duct.random_u_star(3, np.zeros(grid.grid_nodes, np.int32)))
    fluid.force.from_numpy(np.full(grid.grid_nodes + (3,), 50.0, np.float32))
    before = snapshot(fluid)
    with pytest.raises(NotImplementedError, match=REPORT):
        fluid.apply_body_force(duct.DT, read_report=True)
    assert_unchanged(fluid, before)
    with pytest.raises(NotImplementedError, match="muscl_tvd"):
        fluid.predict(dt_s=duct.DT, advection_scheme="euler", pressure_outlet_zmin=True)
    assert_unchanged(fluid, before)
    for bad, match in ((dict(read_report=True), REPORT), (dict(pressure_solver="fv_multigrid"), "fv_cg")):
        with pytest.raises(NotImplementedError, match=match):
            fluid.project(**project_kwargs(**bad))
        assert_unchanged(fluid, before)


def test_the_defaults_are_the_supported_paths():
    fluid, grid, _ = duct.rig()
    fluid.velocity.from_numpy(np.zeros(grid.grid_nodes + (3,), np.float32))
    fluid.force.from_numpy(np.full(grid.grid_nodes + (3,), 50.0, np.float32))
    fluid.apply_body_force(duct.DT)                                    # read_report=False
    assert np.allclose(fluid.velocity.to_numpy(), np.float32(50.0 * duct.DT / duct.RHO))
    fluid.force.from_numpy(np.zeros(grid.grid_nodes + (3,), np.float32))
    fluid.predict(dt_s=duct.DT, pressure_outlet_zmin=True)             # muscl_tvd
    kwargs = project_kwargs()
    del kwargs["pressure_solver"]                                     # fv_cg, failures raise, no report
    assert fluid.project(**kwargs) == {}
    assert np.isfinite(fluid.velocity.to_numpy()).all() and np.isfinite(fluid.pressure.to_numpy()).all()
