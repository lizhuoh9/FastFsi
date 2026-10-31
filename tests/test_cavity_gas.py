"""The cavity gas body (boundary.cavity_gas, the s26 outer chamber, Root 2026-10-31).  A hollow cube shell (outer
4-20 mm, cavity 8-16 mm, walls 4 mm) of material points on a 0.5 mm lattice, on a 24^3 grid of 1 mm cells, the cavity's
faces as an 8 x 8 subdivided cube surface.  At the start the cavity is its 512 free cells at gauge 0 and its surface
volume 512 mm^3, its stiffness omega^2 = p_abs A^2 / (V m_free); removing 10 % of the gas drops the pressure by 10 % of
p_atm; a rigid shift of the upper half closes the cavity to 384 mm^3 and 384 cells, clears the air cells that left it
and leaves every other air cell alone; a slab pushed across it splits the free cells into 2 parts that stay one body;
an opened wall is the leak hard stop, an exhausted ledger the invalid one, a cavity without free cells the empty one;
bad surfaces are refused; a checkpoint round trip continues bit for bit."""
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from fastfsi.boundary.cavity_gas import CavityLedgerGas, closed_surface_report
from fastfsi.boundary.gas_pockets import P_ATMOSPHERE_PA
from fastfsi.boundary.gas_regions_device import STIFFNESS_LIMIT, STIFFNESS_MARGIN
from fastfsi.boundary.inlet_air import AIR_DENSITY_KG_M3
from fastfsi.errors import SimulationHardStop
from fastfsi.grid.axes import GradedAxes
from fastfsi.io.checkpoint import load_components, read_checkpoint, write_checkpoint
from fastfsi.solid.material_points import MaterialPoints
from fastfsi.solid.materials import NeoHookeanMaterial

MM = 1.0e-3
N = 24
SPACING = 0.5 * MM
BOX = ((1, 23), (1, 23), (1, 23))
FACES = np.arange(N + 1) * MM


@pytest.fixture(scope="module", autouse=True)
def cpu():
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=1, opt_level=0, offline_cache=False)


def shell_points(septum=False):
    """The shell's lattice points; with septum also a 2 mm slab across the cavity (x 11-13 mm), which splits its free
    cells in two while the cavity surface stays the cube."""
    centres = (np.arange(4 * MM + 0.5 * SPACING, 20 * MM, SPACING))
    x, y, z = np.meshgrid(centres, centres, centres, indexing="ij")
    p = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    inner = np.all((p > 8 * MM) & (p < 16 * MM), axis=1)
    if septum:
        inner &= ~((p[:, 0] > 11 * MM) & (p[:, 0] < 13 * MM))
    return p[~inner]


def cube_surface(lo, hi, n):
    """Outward triangulation of the box [lo, hi]^3 with n x n quads per face, shared vertices."""
    ticks = np.linspace(lo, hi, n + 1)
    keys, vertices, triangles = {}, [], []

    def vid(point):
        key = tuple(np.round(np.asarray(point) / (1e-9), 0).astype(np.int64))
        if key not in keys:
            keys[key] = len(vertices)
            vertices.append(point)
        return keys[key]

    for axis in range(3):
        for side, value in ((0, lo), (1, hi)):
            u_axis, v_axis = (axis + 1) % 3, (axis + 2) % 3
            for a in range(n):
                for b in range(n):
                    quad = []
                    for da, db in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        point = [0.0, 0.0, 0.0]
                        point[axis] = value
                        point[u_axis] = ticks[a + da]
                        point[v_axis] = ticks[b + db]
                        quad.append(vid(point))
                    q = quad if side == 1 else quad[::-1]
                    triangles += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64)


class Partner:
    """Stands in for PressureRegions: records the attached external region."""

    def attach_external_region(self, box_cells, code_field, gauge_field):
        self.box_cells, self.code, self.gauge = box_cells, code_field, gauge_field


def build(displacement=None, septum=False, filled=False):
    positions = shell_points(septum)
    points = MaterialPoints(positions_m=positions, volumes_m3=np.full(len(positions), SPACING ** 3),
                            material=NeoHookeanMaterial(1070., 25705., 50. * 25705.),
                            tether_mask=np.zeros(len(positions), np.int32), body_id=np.full(len(positions), 3, np.int32))
    if displacement is not None:
        points.u.from_numpy(displacement.astype(np.float32))
        points.refresh_positions()
    vertices, triangles = cube_surface(8 * MM, 16 * MM, 8)
    fluid = SimpleNamespace(obstacle=ti.field(ti.i32, shape=(N, N, N)))
    if filled:                                  # every cavity cell (8..15 in each axis) a fluid obstacle
        obstacle = np.zeros((N, N, N), np.int32)
        obstacle[8:16, 8:16, 8:16] = 1
        fluid.obstacle.from_numpy(obstacle)
    air = ti.field(ti.i32, shape=(N, N, N))
    partner = Partner()
    cavity = CavityLedgerGas(fluid, GradedAxes(FACES, FACES, FACES), points,
                             body_indices=np.arange(len(positions)), reference_positions_m=positions,
                             spacing_m=np.full((len(positions), 3), SPACING), box_cells=BOX,
                             surface_vertices_m=vertices, surface_triangles=triangles,
                             fixed_vertices=np.zeros(len(vertices), bool), rigid_volume_m3=0.0, air_cell=air,
                             pressure_regions=partner, body_density_kgm3=1070.)
    return cavity, points, air, partner, positions


def displace(points, positions, u):
    points.u.from_numpy(u.astype(np.float32))
    points.refresh_positions()


def test_the_start_is_the_free_cavity_at_gauge_zero():
    cavity, _, air, partner, _ = build()
    r = cavity.report()
    assert r["cavity_cells"] == 512 and r["subregions"] == 1
    assert r["volume_m3"] == pytest.approx(512 * MM ** 3, rel=1e-12)
    assert r["raster_volume_m3"] == pytest.approx(512 * MM ** 3, rel=1e-6)
    assert r["p_gauge_pa"] == 0.0 and cavity.q0_pa_m3 == pytest.approx(P_ATMOSPHERE_PA * 512 * MM ** 3)
    code = partner.code.to_numpy()
    assert code.sum() == 512 and np.all(code[7:15, 7:15, 7:15] == 1)          # box offset 1: cells 8..15
    assert air.to_numpy().sum() == 512 and air.to_numpy()[8:16, 8:16, 8:16].all()
    assert partner.box_cells == BOX and partner.gauge[None] == 0.0
    assert cavity.surface_report["open_edges"] == 0 and cavity.surface_report["orientation_mismatches"] == 0
    omega = np.sqrt((r["p_gauge_pa"] + P_ATMOSPHERE_PA) * r["area_m2"] ** 2 / (r["volume_m3"] * cavity.free_mass_kg))
    assert r["area_m2"] == pytest.approx(6 * (8 * MM) ** 2, rel=1e-12)
    assert r["omega_rad_s"] == pytest.approx(omega, rel=1e-12)
    assert r["dt_limit_s"] == pytest.approx(STIFFNESS_MARGIN * STIFFNESS_LIMIT / omega, rel=1e-12)


def test_the_ledger_follows_the_partner_inflow():
    cavity, _, _, _, _ = build()
    q0 = cavity.q0_pa_m3
    r = cavity.accept(1.0e-3, 0.1 * q0)
    assert r["q_pa_m3"] == pytest.approx(0.9 * q0, rel=1e-12)
    assert r["p_gauge_pa"] == pytest.approx(-0.1 * P_ATMOSPHERE_PA, rel=1e-9)
    assert cavity.gauge[None] == r["p_gauge_pa"]
    assert r["mass_kg"] == pytest.approx(AIR_DENSITY_KG_M3 * 0.9 * q0 / P_ATMOSPHERE_PA, rel=1e-12)
    assert r["moved_mass_kg"] == pytest.approx(AIR_DENSITY_KG_M3 * 0.1 * q0 / P_ATMOSPHERE_PA, rel=1e-12)
    assert r["mass_kg"] + r["moved_mass_kg"] == pytest.approx(AIR_DENSITY_KG_M3 * q0 / P_ATMOSPHERE_PA, rel=1e-14)
    assert 0.0 < r["dt_limit_s"] < np.inf
    assert cavity.ledger_residual(1.0) is None                                     # no reference yet
    partner0 = 3.0e-3                                                               # the partner's gas at the start
    cavity.set_ledger_reference(partner0)
    assert abs(cavity.ledger_residual(partner0 + 0.1 * q0)) <= 1e-16               # the moved gas reached the partner
    assert cavity.ledger_residual(partner0 + 0.1 * q0 + 1e-6, 1e-6) == pytest.approx(0.0, abs=1e-16)   # reappear out
    with pytest.raises(ValueError):
        cavity.set_ledger_reference(float("nan"))


def test_a_rigid_shift_of_the_upper_half_closes_the_cavity():
    cavity, points, _, _, positions = build()
    u = np.zeros_like(positions)
    u[positions[:, 2] > 12 * MM, 2] = -2 * MM
    displace(points, positions, u)
    r = cavity.accept(1.0e-3, 0.0)
    assert r["volume_m3"] == pytest.approx(384 * MM ** 3, rel=1e-7)                 # the f32 displacement's rounding
    assert r["p_gauge_pa"] == cavity.q0_pa_m3 / r["volume_m3"] - P_ATMOSPHERE_PA       # the ledger, exactly
    assert r["p_gauge_pa"] == pytest.approx(P_ATMOSPHERE_PA * (512.0 / 384.0 - 1.0), rel=1e-6)
    assert r["cavity_cells"] == 384


def test_the_air_cells_that_left_the_cavity_clear_and_others_stay():
    cavity, points, air, _, positions = build()
    values = air.to_numpy()
    values[0, 0, 0] = 1                       # another region's air cell, outside the box
    values[2, 2, 2] = 1                       # and one inside the box, in the water outside the shell
    air.from_numpy(values)
    u = np.zeros_like(positions)
    u[positions[:, 2] > 12 * MM, 2] = -2 * MM
    displace(points, positions, u)
    assert cavity.accept(1.0e-3, 0.0)["cavity_cells"] == 384
    after = air.to_numpy()
    assert after[0, 0, 0] == 1 and after[2, 2, 2] == 1                         # not the cavity's: left alone
    assert after[8:16, 8:16, 14:16].sum() == 0 and after[8:16, 8:16, 8:14].all()   # the closed top cleared
    assert after.sum() == 384 + 2


def test_a_slab_across_the_cavity_splits_it_into_parts_of_one_body():
    cavity, _, _, partner, _ = build(septum=True)
    r = cavity.accept(1.0e-3, 0.02 * cavity.q0_pa_m3)
    assert cavity.subregions() == 2 and r["cavity_cells"] == 384               # 2 x 3 x 8 x 8 free cells
    code = partner.code.to_numpy()
    assert code[7:10].sum() == 192 and code[12:15].sum() == 192                 # cells 8-10 and 13-15 in x
    assert code[10:12].sum() == 0 and r["p_gauge_pa"] < 0.0                    # the slab; one gauge for both parts


def test_an_opened_wall_is_the_leak_hard_stop():
    cavity, points, _, _, positions = build()
    u = np.zeros_like(positions)
    strip = (positions[:, 2] > 16 * MM) & (positions[:, 0] > 10 * MM) & (positions[:, 0] < 14 * MM)
    u[strip, 2] = 4 * MM                      # a strip of the top plate lifts: the cavity meets the water at y < 4 mm
    displace(points, positions, u)
    for step in range(4):                     # the hysteresis frees a vacated cell at its 5th update without material
        assert cavity.accept(1.0e-3 * (step + 1), 0.0)["cavity_cells"] >= 512
    with pytest.raises(SimulationHardStop) as stop:
        cavity.accept(5.0e-3, 0.0)
    assert stop.value.code == "cavity_gas_leak" and stop.value.witness["boundary_cells"] > 0


def test_an_exhausted_ledger_is_the_invalid_hard_stop():
    cavity, _, _, _, _ = build()
    with pytest.raises(SimulationHardStop) as stop:
        cavity.accept(1.0e-3, cavity.q0_pa_m3)
    assert stop.value.code == "cavity_gas_invalid"


def test_a_cavity_without_free_cells_is_the_empty_hard_stop():
    with pytest.raises(SimulationHardStop) as stop:
        build(filled=True)                    # no seed finds a free cell: the gas body would act nowhere
    assert stop.value.code == "cavity_gas_empty" and stop.value.witness["seeds"] == 0


def test_bad_surfaces_are_refused():
    vertices, triangles = cube_surface(8 * MM, 16 * MM, 2)
    assert closed_surface_report(triangles) == dict(open_edges=0, multi_edges=0, orientation_mismatches=0)
    assert closed_surface_report(triangles[1:])["open_edges"] > 0
    assert closed_surface_report(np.vstack([triangles[:1, ::-1], triangles[1:]]))["orientation_mismatches"] > 0
    positions = shell_points()
    points = MaterialPoints(positions_m=positions, volumes_m3=np.full(len(positions), SPACING ** 3),
                            material=NeoHookeanMaterial(1070., 1., 50.), tether_mask=np.zeros(len(positions), np.int32))
    common = dict(body_indices=np.arange(len(positions)), reference_positions_m=positions,
                  spacing_m=np.full((len(positions), 3), SPACING), box_cells=BOX, rigid_volume_m3=0.0,
                  air_cell=ti.field(ti.i32, shape=(N, N, N)), pressure_regions=Partner(), body_density_kgm3=1070.)
    fluid = SimpleNamespace(obstacle=ti.field(ti.i32, shape=(N, N, N)))
    axes = GradedAxes(FACES, FACES, FACES)
    for tri, match in ((triangles[1:], "closed"), (triangles[:, ::-1], "outward")):
        with pytest.raises(ValueError, match=match):
            CavityLedgerGas(fluid, axes, points, surface_vertices_m=vertices, surface_triangles=tri,
                            fixed_vertices=np.zeros(len(vertices), bool), **common)
    with pytest.raises(ValueError, match="inside the box"):
        CavityLedgerGas(fluid, axes, points, surface_vertices_m=vertices + 20 * MM, surface_triangles=triangles,
                        fixed_vertices=np.zeros(len(vertices), bool), **common)


def test_a_checkpoint_round_trip_continues_bit_for_bit(tmp_path):
    cavity, points, air, partner, positions = build()
    u = np.zeros_like(positions)
    u[positions[:, 2] > 12 * MM, 2] = -1 * MM
    displace(points, positions, u)
    cavity.accept(1.0e-3, 0.05 * cavity.q0_pa_m3)
    path = tmp_path / "cavity.npz"
    write_checkpoint(path, dict(points=points, cavity=cavity), dict(step=1), dict(test="cavity"))
    after = cavity.accept(2.0e-3, 0.07 * cavity.q0_pa_m3)
    arrays, host, _, _, _ = read_checkpoint(path)
    cavity2, points2, air2, partner2, _ = build()
    load_components(dict(points=points2, cavity=cavity2), arrays, host)
    assert np.array_equal(partner2.code.to_numpy(), arrays["cavity"]["code"])
    assert np.array_equal(air2.to_numpy()[1:23, 1:23, 1:23], arrays["cavity"]["code"])
    again = cavity2.accept(2.0e-3, 0.07 * cavity.q0_pa_m3)
    for key in ("q_pa_m3", "p_gauge_pa", "volume_m3", "raster_volume_m3", "cavity_cells", "seeds"):
        assert again[key] == after[key], key
