"""WholeBodyForce (simulation.body_force; Root 2026-10-24, the USER's whole-robot thrust record) on small CPU grids
with prescribed fields:
a uniform stream through an empty box carries no net force, and the box's momentum change gives -rho dU V / T exactly;
a closed obstacle in a uniform pressure feels no net force (B) and a closed box none either (A);
a linear pressure gives the obstacle the discrete Archimedes force -G (L + h) L^2 (B);
a cup touching the top wall with pressurised gas inside is force-free only with the lid (the top wall over its
footprint) counted as the body (B), and the box with its top face on that wall reads 0 with no cut face (A);
a box through the body counts cut faces;
the clamp's point forces and the caller's point forces enter B with their interval means."""
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from fastfsi.grid.axes import GradedAxes
from fastfsi.simulation.body_force import WholeBodyForce

N = 10
H = 0.01
RHO, MU = 1000.0, 1.0e-3


@pytest.fixture(scope="module", autouse=True)
def cpu():
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1, opt_level=0, offline_cache=False)


def rig(obstacle=None, air=None):
    faces = np.arange(N + 1) * H
    axes = GradedAxes(faces, faces, faces)
    fluid = SimpleNamespace(velocity=ti.Vector.field(3, ti.f32, shape=(N, N, N)),
                            pressure=ti.field(ti.f64, shape=(N, N, N)), obstacle=ti.field(ti.i32, shape=(N, N, N)),
                            rho=RHO, mu=MU)
    if obstacle is not None:
        fluid.obstacle.from_numpy(obstacle.astype(np.int32))
    gas = None
    if air is not None:
        gas = ti.field(ti.i32, shape=(N, N, N))
        gas.from_numpy(air.astype(np.int32))
    return axes, fluid, gas


def box(lo, hi, name="b"):
    return dict(i=(lo, hi), j=(lo, hi), k=(lo, hi), name=name)


def uniform_velocity(fluid, u):
    fluid.velocity.from_numpy(np.broadcast_to(np.asarray(u, np.float32), (N, N, N, 3)).copy())


def test_a_uniform_stream_carries_no_force_and_the_momentum_change_telescopes():
    axes, fluid, _ = rig()
    uniform_velocity(fluid, (0.0, 0.0, 0.5))
    m = WholeBodyForce(fluid, axes, [box(2, 7)])
    m.begin(0.0)
    p = m.pressure_field(0.0)
    for dt in (1.0e-3, 2.0e-3, 0.5e-3):
        m.accumulate(dt, p)
    r = m.report(3.5e-3)
    assert np.allclose(r["boxes"]["b"]["force_n"], 0.0, atol=1e-12)
    uniform_velocity(fluid, (0.0, 0.0, 0.5))
    m.begin(0.0)
    for dt in (1.0e-3, 2.0e-3):
        m.accumulate(dt, p)
    uniform_velocity(fluid, (0.0, 0.0, 0.8))                    # the box's momentum changes over the interval
    r = m.report(3.0e-3)
    volume = (5 * H) ** 3
    assert r["interval_s"] == pytest.approx(3.0e-3)
    assert r["boxes"]["b"]["momentum_rate_n"][2] == pytest.approx(RHO * 0.3 * volume / 3.0e-3, rel=1e-6)
    # the outflow at the new speed is part of the same interval's flux only through accumulate(); none was added
    assert r["boxes"]["b"]["force_n"][2] == pytest.approx(-RHO * 0.3 * volume / 3.0e-3, rel=1e-6)


def test_a_closed_obstacle_in_a_uniform_pressure_feels_no_net_force():
    obstacle = np.zeros((N, N, N), bool)
    obstacle[4:6, 4:6, 4:6] = True
    axes, fluid, _ = rig(obstacle)
    m = WholeBodyForce(fluid, axes, [box(2, 8)])
    m.begin(0.0)
    m.accumulate(1.0e-3, m.pressure_field(250.0))
    r = m.report(1.0e-3)
    assert np.allclose(r["b"]["obstacle_pressure"], 0.0, atol=1e-9)
    assert np.allclose(r["boxes"]["b"]["force_n"], 0.0, atol=1e-9) and r["boxes"]["b"]["cut_faces"] == 0


def test_a_linear_pressure_gives_the_discrete_archimedes_force():
    obstacle = np.zeros((N, N, N), bool)
    obstacle[3:6, 3:6, 3:6] = True                                # L = 3 h
    axes, fluid, _ = rig(obstacle)
    G = 1.0e4                                                     # Pa/m along z
    z = (np.arange(N) + 0.5) * H
    fluid.pressure.from_numpy(np.broadcast_to(G * z, (N, N, N)).copy())
    m = WholeBodyForce(fluid, axes, [box(1, 9)])
    m.begin(0.0)
    m.accumulate(1.0e-3, fluid.pressure)
    r = m.report(1.0e-3)
    L = 3 * H
    assert r["b"]["obstacle_pressure"][2] == pytest.approx(-G * (L + H) * L * L, rel=1e-9)
    assert np.allclose(r["b"]["obstacle_pressure"][:2], 0.0, atol=1e-9)


def test_a_cup_on_the_top_wall_needs_its_lid_and_the_box_reads_zero():
    obstacle = np.zeros((N, N, N), bool)
    obstacle[3:7, 3:7, 6:] = True                                 # a cup touching the top wall
    obstacle[4:6, 4:6, 7:] = False                                # its cavity, open to the top wall
    air = np.zeros((N, N, N), bool)
    air[4:6, 4:6, 7:] = True
    axes, fluid, gas = rig(obstacle, air)
    pressure = np.zeros((N, N, N))
    pressure[air] = 3000.0                                        # the gas at 3 kPa, the water at 0
    fluid.pressure.from_numpy(pressure)
    footprint = np.zeros((N, N), bool)
    footprint[3:7, 3:7] = True
    lid = WholeBodyForce(fluid, axes, [dict(i=(1, 9), j=(1, 9), k=(2, N), name="top")], top_footprint=footprint,
                         air_cell=gas)
    lid.begin(0.0)
    lid.accumulate(1.0e-3, fluid.pressure)
    r = lid.report(1.0e-3)
    b = np.array(r["b"]["obstacle_pressure"]) + np.array(r["b"]["lid_pressure"])
    assert np.allclose(b, 0.0, atol=1e-9)                         # a closed container: no net force
    assert r["b"]["lid_pressure"][2] == pytest.approx(3000.0 * (2 * H) ** 2)
    assert np.allclose(r["boxes"]["top"]["force_n"], 0.0, atol=1e-9) and r["boxes"]["top"]["cut_faces"] == 0
    open_top = WholeBodyForce(fluid, axes, [dict(i=(1, 9), j=(1, 9), k=(2, N), name="top")], air_cell=gas)
    open_top.begin(0.0)
    open_top.accumulate(1.0e-3, fluid.pressure)
    r = open_top.report(1.0e-3)
    assert r["b"]["obstacle_pressure"][2] == pytest.approx(-3000.0 * (2 * H) ** 2)   # the floor alone, no lid
    assert r["boxes"]["top"]["cut_faces"] > 0                     # the top face meets the gas outside a footprint


def test_a_box_through_the_body_counts_cut_faces():
    obstacle = np.zeros((N, N, N), bool)
    obstacle[3:7, 3:7, 3:7] = True
    axes, fluid, _ = rig(obstacle)
    m = WholeBodyForce(fluid, axes, [box(4, 8)])
    m.begin(0.0)
    m.accumulate(1.0e-3, m.pressure_field(0.0))
    assert m.report(1.0e-3)["boxes"]["b"]["cut_faces"] > 0


def test_the_clamp_and_the_point_forces_enter_with_interval_means():
    axes, fluid, _ = rig()
    force = ti.Vector.field(3, ti.f32, shape=4)
    force.from_numpy(np.array([[1.0, 0.0, 2.0], [0.5, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]], np.float32))
    extra = ti.Vector.field(3, ti.f32, shape=2)
    extra.from_numpy(np.array([[0.0, 0.0, -4.0], [1.0, 0.0, 0.0]], np.float32))
    m = WholeBodyForce(fluid, axes, [box(2, 7)], clamp=SimpleNamespace(point_force=force), point_forces=extra,
                       point_slots=("contact", "contact_inlet"))
    m.begin(0.0)
    p = m.pressure_field(0.0)
    for dt in (1.0e-3, 3.0e-3):
        m.accumulate(dt, p)
        m.add_point_forces("contact", dt)
    m.add_point_forces("contact_inlet", 4.0e-3)
    r = m.report(4.0e-3)
    assert r["b"]["clamp"] == pytest.approx([1.5, 3.0, 1.0])
    assert r["points"]["contact"] == pytest.approx([1.0, 0.0, -4.0])
    assert r["points"]["contact_inlet"] == pytest.approx([1.0, 0.0, -4.0])
