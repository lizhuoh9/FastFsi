"""Contracts between classes and the code that relies on their members by name:
  - every CUDA device controller (fastfsi.fluid.device_controls) replaces an existing solver method with the same
    parameters, so a rename can never turn the installation into a silent no-op;
  - the fluid solver's checkpoint fields and host attributes exist on a freshly built solver and its host state
    survives the type-exact encoding;
  - the checkpoint fields of the 3-D and planar material points exist, and a planar point set is restored bit for bit."""
import inspect

import numpy as np
import pytest

from fastfsi.fluid.device_controls import REPLACED_METHODS, device_replacements
from fastfsi.fluid.solver import IncompressibleFluidSolver
from fastfsi.io.checkpoint import (decode_host, encode_host, load_components, read_checkpoint, same_host_value,
                                   write_checkpoint)


def parameters(function):
    return [(p.name, p.kind, p.default) for p in inspect.signature(function).parameters.values()]


def test_every_device_controller_replaces_a_solver_method_with_its_parameters():
    replacements = device_replacements()
    assert tuple(replacements) == REPLACED_METHODS
    for name, function in replacements.items():
        method = inspect.getattr_static(IncompressibleFluidSolver, name)      # on the class or one of its mixins
        assert callable(method), name
        assert parameters(function) == parameters(method), name


@pytest.fixture(scope="module")
def cpu():
    from cpu_runtime import initialize
    initialize()


@pytest.fixture(scope="module")
def fluid(cpu):
    from fastfsi.grid.cartesian_grid import CartesianGrid
    from fastfsi.grid.specification import FluidDomainSpecification
    from fastfsi.simulation.taichi_runtime import TaichiRuntimeConfig
    grid = CartesianGrid(bounds_min_m=(0.0, 0.0, 0.0), cell_widths_x_m=(1.0e-3,) * 4, cell_widths_y_m=(1.0e-3,) * 4,
                         cell_widths_z_m=(1.0e-3,) * 8)
    spec = FluidDomainSpecification(bounds_min_m=grid.bounds_min_m, bounds_max_m=grid.bounds_max_m,
                                    grid_nodes=grid.grid_nodes, density_kgm3=1000.0, viscosity_pa_s=1.0e-3,
                                    dt_s=1.0e-4, cartesian_grid=grid)
    return IncompressibleFluidSolver(spec, runtime=TaichiRuntimeConfig(arch="cpu", default_fp="f32"))


def test_fluid_checkpoint_names_exist_on_a_fresh_solver(fluid):
    for name in IncompressibleFluidSolver.CHECKPOINT_FIELDS:
        assert hasattr(getattr(fluid, name), "to_numpy"), name
    assert [name for name in IncompressibleFluidSolver.CHECKPOINT_HOST if not hasattr(fluid, name)] == []
    assert len(set(IncompressibleFluidSolver.CHECKPOINT_HOST)) == len(IncompressibleFluidSolver.CHECKPOINT_HOST)
    host = fluid.state_host()
    restored = decode_host(encode_host(host))
    assert list(restored) == list(host) and all(same_host_value(host[k], restored[k]) for k in host)


def material():
    from fastfsi.solid.materials import NeoHookeanMaterial
    return NeoHookeanMaterial(density_kgm3=1000.0, shear_modulus_pa=1.0e5, volumetric_modulus_pa=4.7e5)


def test_material_point_checkpoint_fields_exist(cpu):
    from fastfsi.solid.material_points import MaterialPoints
    points = MaterialPoints(positions_m=np.zeros((2, 3)), volumes_m3=np.ones(2), material=material(),
                            tether_mask=np.zeros(2, np.int32))
    for name in MaterialPoints.CHECKPOINT_FIELDS:
        assert hasattr(getattr(points, name), "to_numpy"), name


def test_planar_points_round_trip_bit_for_bit(cpu, tmp_path):
    from fastfsi.solid.material_points import PlanarMaterialPoints
    positions = np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]])

    def build():
        return PlanarMaterialPoints(positions_yz_m=positions, areas_m2=np.full(3, 1.0e-4), material=material(),
                                    tether_mask=np.zeros(3, np.int32))
    saved = build()
    saved.x.from_numpy(positions.astype(np.float32) + np.float32(1.0e-3))
    saved.F.from_numpy(np.tile(np.array([[1.1, 0.05], [0.0, 0.95]], np.float32), (3, 1, 1)))
    write_checkpoint(tmp_path / "points.npz", dict(points=saved), dict(step=1), {})
    arrays, host, loop_state, _, _ = read_checkpoint(tmp_path / "points.npz")
    fresh = build()
    load_components(dict(points=fresh), arrays, host)
    for name in PlanarMaterialPoints.CHECKPOINT_FIELDS:
        assert getattr(fresh, name).to_numpy().tobytes() == getattr(saved, name).to_numpy().tobytes(), name
    assert loop_state == dict(step=1)
