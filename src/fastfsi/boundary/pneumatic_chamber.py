"""An ideal pressure inlet through a hole in a chamber lid, with sealed air pockets, on the fluid grid:
SolidOccupancy (where the body is) + DeviceGasRegions (inlet air, water, pockets and their gas, on the device) +
PressureRegions (per-cell Dirichlet pressures) + the covered-inlet traction, behind the ColumnPressureChamber driving
protocol (update / after_project / air_cell / report / physical pressure / checkpoint) plus accept().

INLET.  A disc of diameter d in the lid (the top face of the fluid domain) centred at (x, y).  Its inlet cells are the
box's top-layer cells whose centres lie in the disc, plus the cell of the disc centre.  Inlet-connected air takes the
line pressure p_in(t) with no resistance (an ideal inlet).
COVERED INLET.  When no inlet cell is free, no Dirichlet cell carries p_in any more.  The line pressure
then acts on the covering membrane as the force p_in A_hole along the lid normal into the chamber (downward for
p_in > 0, upward under suction).  It is distributed over the chamber-surface points (tag `surface_tag`) whose deformed
top-face footprint overlaps the disc, by the overlapping area (8 x 8 samples of the footprint parallelogram spanned by
F e_x s_x and F e_y s_y), and added to the points' f_extra (the coupler spreads it and zeroes f_extra).  A covered inlet
with no overlapping surface point is the hard stop pneumatic_chamber_inlet_without_surface (the line pressure would be lost).
STIFFNESS.  dt_limit() = 0.9 x 0.5 / max pocket omega of the last accept; accept() checks omega dt <= 0.5 for the
step just taken (hard stop gas_pocket_stiffness).
BOX.  The chamber footprint (r <= radius from the axis) in x-y, from box_z_lo_m to the lid in z; outside the footprint
and the fluid's obstacle cells are wall.  The box floor must stay free of material (assertion: hard stop
pneumatic_chamber_floor_material, checked in accept()).

PER STEP (driver):
    update(t_mid, p_in(t_mid))       the rows of this step's solve (inlet air at p_in, pockets at their pressures)
    mask.refresh(air_cell)           (F1) faces next to Dirichlet cells
    ... midpoint predict, contact ...
    apply_inlet_traction(p_in(t_mid))  only acts while the inlet is covered
    ... spread, predict, project ...
    after_project(dt, p_in(t_mid))
    ... midpoint correct -> the accepted geometry ...
    accept(t_new, dt_taken)          occupancy, topology (on flips), volumes, pressures, the rows of the next step
POCKET RULE (gas_regions_device): seal_steps (default N_SEAL = 5) and min_cells (M_MIN = 8) decide when a region cut
off from the inlet seals; until then it is inlet air.  raster_interval N re-rasterises the body every Nth accept only
(default 1).  The candidate events wait in candidate_events_pending until the driver drains them
(drain_candidate_events()).
TRANSFERS: see gas_regions_device (two reads under 1 KB without flips); while the inlet is covered one more 8-byte
read (the covering area) per accept.  The run totals are in run_summary()["transfers"].

No `from __future__ import annotations` (Taichi kernel module).
"""
import math

import numpy as np
import taichi as ti

from fastfsi.boundary.gas_pocket_transitions import M_MIN, N_SEAL
from fastfsi.boundary.gas_pockets import P_ATMOSPHERE_PA
from fastfsi.boundary.gas_regions_device import DeviceGasRegions
from fastfsi.boundary.solid_occupancy import (DEFAULT_BOX_SAMPLES, DEFAULT_HYSTERESIS_STEPS, DEFAULT_STRETCH_FACTOR,
                                        DEFAULT_SUB, SolidOccupancy)
from fastfsi.boundary.pressure_regions import DEFAULT_PENALTY_FACTOR, PressureRegions
from fastfsi.boundary.pressure_regions import PROJECT_REQUIREMENTS  # noqa: F401 (the contract here too)
from fastfsi.errors import SimulationHardStop
from fastfsi.point_tags import TAG_CHAMBER_SURFACE

HARD_STOP_CODES = {"pneumatic_chamber_inlet_without_surface", "pneumatic_chamber_floor_material"}
NOISE_EVENT_LOG_MAX = 200                # the first events kept in full (the counts are complete)
FOOTPRINT_SAMPLES = 8


def chamber_box(faces, axis_xy_m, radius_m, z_lo_m, lid_z_m):
    """Half-open cell ranges of the box: x-y cells meeting the footprint square, z from the cell holding z_lo_m to the
    top layer (the lid must be the domain top)."""
    fx, fy, fz = (np.asarray(f, dtype=np.float64) for f in faces)
    if abs(fz[-1] - float(lid_z_m)) > 1.0e-6:
        raise ValueError("the lid %.6f m is not the top face of the grid %.6f m" % (lid_z_m, fz[-1]))
    ranges = []
    for f, centre in ((fx, axis_xy_m[0]), (fy, axis_xy_m[1])):
        lo = max(int(np.searchsorted(f, centre - radius_m, side="right")) - 2, 0)
        hi = min(int(np.searchsorted(f, centre + radius_m, side="left")) + 1, len(f) - 1)
        ranges.append((lo, hi))
    k0 = int(np.searchsorted(fz, z_lo_m, side="right")) - 1
    if not 0 <= k0 < len(fz) - 2:
        raise ValueError("box_z_lo_m %.6f is outside the grid" % z_lo_m)
    ranges.append((k0, len(fz) - 1))
    return tuple(ranges)


def _no_run_transfers():
    return dict(reads=0, read_bytes=0, writes=0, write_bytes=0, max_reads_per_accept=0, max_read_bytes_per_accept=0,
                max_writes_per_accept=0, max_write_bytes_per_accept=0, topology_accepts=0, host_stage_accepts=0)


@ti.data_oriented
class PneumaticChamber:
    """See the module docstring."""

    PROJECT_REQUIREMENTS = PROJECT_REQUIREMENTS   # its PressureRegions' contract

    def __init__(self, fluid, axes, points, *, body_indices, reference_positions_m, spacing_m, point_mass_kg,
                 axis_xy_m, radius_m, lid_z_m, box_z_lo_m, inlet_centre_xy_m, inlet_diameter_m,
                 ambient_pressure_pa=P_ATMOSPHERE_PA, pressure_bounds_abs_pa=None, sub=DEFAULT_SUB,
                 stretch_factor=DEFAULT_STRETCH_FACTOR, box_samples=DEFAULT_BOX_SAMPLES,
                 hysteresis_steps=DEFAULT_HYSTERESIS_STEPS, penalty_factor=DEFAULT_PENALTY_FACTOR,
                 inlet_traction=True, surface_tag=TAG_CHAMBER_SURFACE, max_pockets=4096, flood_batch=16,
                 staging=65536, seal_steps=N_SEAL, min_cells=M_MIN, raster_interval=1):
        faces = tuple(np.asarray(f, dtype=np.float64) for f in axes.faces_m)
        box = chamber_box(faces, axis_xy_m, float(radius_m), float(box_z_lo_m), float(lid_z_m))
        self.fluid = fluid
        self.points = points
        self.box_cells = box
        self.shape = tuple(int(v) for v in fluid.grid.grid_nodes)
        self.axis_xy_m = (float(axis_xy_m[0]), float(axis_xy_m[1]))
        self.radius_m = float(radius_m)
        self.inlet_centre_xy_m = (float(inlet_centre_xy_m[0]), float(inlet_centre_xy_m[1]))
        self.inlet_diameter_m = float(inlet_diameter_m)
        if not (math.isfinite(self.inlet_diameter_m) and self.inlet_diameter_m > 0.0):
            raise ValueError("inlet_diameter_m must be finite and positive")
        self.inlet_area_m2 = math.pi * 0.25 * self.inlet_diameter_m ** 2
        self.inlet_traction = bool(inlet_traction)
        self.surface_tag = int(surface_tag)
        self.occupancy = SolidOccupancy(axes, box, points, body_indices, reference_positions_m, spacing_m, sub=sub,
                                       stretch_factor=stretch_factor, box_samples=box_samples,
                                       hysteresis_steps=hysteresis_steps, point_mass_kg=point_mass_kg)
        self.chamber = PressureRegions(fluid, box, penalty_factor=penalty_factor, max_pockets=max_pockets)
        self.air_cell = ti.field(ti.i32, shape=self.shape)
        cx, cy, cz = (0.5 * (f[lo:hi] + f[lo + 1:hi + 1]) for f, (lo, hi) in zip(faces, box))
        r = np.hypot(cx[:, None] - self.axis_xy_m[0], cy[None, :] - self.axis_xy_m[1])
        obstacle = fluid.obstacle.to_numpy()[box[0][0]:box[0][1], box[1][0]:box[1][1], box[2][0]:box[2][1]] != 0
        wall = obstacle | (r > self.radius_m)[:, :, None]
        disc = np.hypot(cx[:, None] - self.inlet_centre_xy_m[0], cy[None, :] - self.inlet_centre_xy_m[1])
        inlet = np.zeros(wall.shape, dtype=bool)
        inlet[:, :, -1] = disc <= 0.5 * self.inlet_diameter_m
        ic = int(np.searchsorted(faces[0], self.inlet_centre_xy_m[0], side="right")) - 1 - box[0][0]
        jc = int(np.searchsorted(faces[1], self.inlet_centre_xy_m[1], side="right")) - 1 - box[1][0]
        if not (0 <= ic < wall.shape[0] and 0 <= jc < wall.shape[1]):
            raise ValueError("the inlet centre is outside the chamber box")
        inlet[ic, jc, -1] = True
        inlet &= ~wall
        if not inlet.any():
            raise ValueError("the inlet disc has no chamber cell")
        self.inlet_cells = int(inlet.sum())
        self.regions = DeviceGasRegions(self.occupancy, self.chamber, self.air_cell, wall, inlet,
                                        ambient_pressure_pa=ambient_pressure_pa,
                                        pressure_bounds_abs_pa=pressure_bounds_abs_pa, max_pockets=max_pockets,
                                        flood_batch=flood_batch, staging=staging, seal_steps=seal_steps,
                                        min_cells=min_cells, raster_interval=raster_interval)
        self.candidate_events_pending = []      # candidate-region events until the driver drains them (not carried)
        self.area = ti.field(ti.f64, shape=self.occupancy.n_body)
        self.area_total = ti.field(ti.f64, shape=())
        self.covered = False
        self.traction_steps = 0
        self.accepts = 0
        self.noise_run = [dict(volume_from_m3=v, max_abs_dp_pa=0.0, sum_dp2_pa2=0.0, count=0)
                          for v in [0.0] + list(self.regions.volume_bins_m3)]
        self.ke_max_j = 0.0
        self.transfer_run = _no_run_transfers()
        self.noise_events = []
        self.noise_event_refreshes = 0
        self.noise_event_pockets = 0
        self.last = None

    # ------------------------------------------------------------------------------------- covered-inlet traction
    @ti.kernel
    def _traction_area_kernel(self, cx: ti.f64, cy: ti.f64, radius: ti.f64, tag: ti.i32):
        self.area_total[None] = 0.0
        for b in range(self.occupancy.n_body):
            p = self.occupancy.body_index[b]
            area = ti.cast(0.0, ti.f64)
            if (self.points.tag[p] & tag) != 0:
                x = self.occupancy._position(p)
                F = self.points.F[p]
                s = self.occupancy.spacing[b]
                ax = ti.cast(F[0, 0] * s[0], ti.f64)
                ay = ti.cast(F[1, 0] * s[0], ti.f64)
                bx = ti.cast(F[0, 1] * s[1], ti.f64)
                by = ti.cast(F[1, 1] * s[1], ti.f64)
                reach = 0.5 * (ti.sqrt(ax * ax + ay * ay) + ti.sqrt(bx * bx + by * by))
                if (x[0] - cx) ** 2 + (x[1] - cy) ** 2 <= (radius + reach) ** 2:
                    inside = 0
                    for u in range(FOOTPRINT_SAMPLES):
                        for v in range(FOOTPRINT_SAMPLES):
                            fu = (u + 0.5) / FOOTPRINT_SAMPLES - 0.5
                            fv = (v + 0.5) / FOOTPRINT_SAMPLES - 0.5
                            px = x[0] + fu * ax + fv * bx - cx
                            py = x[1] + fu * ay + fv * by - cy
                            if px * px + py * py <= radius * radius:
                                inside += 1
                    area = ti.abs(ax * by - ay * bx) * inside / (FOOTPRINT_SAMPLES * FOOTPRINT_SAMPLES)
            self.area[b] = area
            ti.atomic_add(self.area_total[None], area)

    @ti.kernel
    def _traction_apply_kernel(self, force_n: ti.f64):
        total = self.area_total[None]
        if total > 0.0:
            for b in range(self.occupancy.n_body):
                if self.area[b] > 0.0:
                    p = self.occupancy.body_index[b]
                    value = self.points.f_extra[p]
                    value.z += ti.cast(-force_n * self.area[b] / total, ti.f32)
                    self.points.f_extra[p] = value

    def apply_inlet_traction(self, p_in_pa):
        """While the inlet is covered: the line pressure on the covering surface, p_in A_hole downward; no-op
        otherwise (and with inlet_traction=False)."""
        if not (self.inlet_traction and self.covered):
            return False
        self._traction_area_kernel(self.inlet_centre_xy_m[0], self.inlet_centre_xy_m[1], 0.5 * self.inlet_diameter_m,
                                   self.surface_tag)
        self._traction_apply_kernel(float(p_in_pa) * self.inlet_area_m2)
        self.traction_steps += 1
        return True

    # ------------------------------------------------------------------------------------------------ protocol
    def accept(self, t_s, inlet_gauge_pa, dt_taken_s=None):
        """The accepted geometry at t_s: refresh everything for the next step; checks the step just taken against the
        pocket stiffness bound and the covering area of the last traction.  Returns the diagnostics."""
        area_reads = 0
        if self.covered and self.inlet_traction and self.traction_steps:
            total = float(self.area_total.to_numpy())
            area_reads = 1
            if total <= 0.0:
                raise SimulationHardStop("pneumatic_chamber_inlet_without_surface", dict(
                    time_s=float(t_s), inlet_area_m2=self.inlet_area_m2, inlet_centre_xy_m=list(self.inlet_centre_xy_m)))
        diagnostics = self.regions.refresh(t_s, inlet_gauge_pa)
        floor = self._floor_material()
        if floor:
            raise SimulationHardStop("pneumatic_chamber_floor_material", dict(time_s=float(t_s), cells=floor))
        if dt_taken_s is not None:
            self.regions.check_step(dt_taken_s)
        self.covered = bool(diagnostics["inlet_covered"])
        self.accepts += 1
        for run, step in zip(self.noise_run, diagnostics["noise"]):
            run["max_abs_dp_pa"] = max(run["max_abs_dp_pa"], step["max_abs_dp_pa"])
            run["sum_dp2_pa2"] += step["sum_dp2_pa2"]
            run["count"] += step["count"]
        self.ke_max_j = max(self.ke_max_j, diagnostics["pocket_footprint_kinetic_energy_j"])
        event = diagnostics.get("noise_event")
        if event:
            self.noise_event_refreshes += 1
            self.noise_event_pockets += int(event["count"])
            if len(self.noise_events) < NOISE_EVENT_LOG_MAX:
                self.noise_events.append(dict(event))
        step = dict(diagnostics["transfers"])
        step["reads"] += area_reads
        step["read_bytes"] += 8 * area_reads
        run = self.transfer_run
        for key in ("reads", "read_bytes", "writes", "write_bytes"):
            run[key] += step[key]
            run["max_" + key + "_per_accept"] = max(run["max_" + key + "_per_accept"], step[key])
        topology = diagnostics.get("topology")
        run["topology_accepts"] += int(topology is not None)
        run["host_stage_accepts"] += int(bool(topology and topology.get("host_stage")))
        self.last = dict(diagnostics, inlet_covered=self.covered, traction_steps=self.traction_steps, transfers=step)
        self.candidate_events_pending.extend(dict(e) for e in diagnostics.get("candidate_events_new") or [])
        return dict(self.last)

    def run_summary(self):
        """The run's pocket statistics (the noise check) and event counters."""
        noise = [dict(run, rms_dp_pa=(math.sqrt(run["sum_dp2_pa2"] / run["count"]) if run["count"] else None))
                 for run in self.noise_run]
        regions = self.regions
        return dict(accepts=self.accepts, noise=noise, pocket_footprint_kinetic_energy_max_j=self.ke_max_j,
                    seal_events=regions.seal_events, exchange_pa_m3=regions.exchange_pa_m3,
                    topology_changes=regions.topology_changes, host_stages=regions.host_stages,
                    pocket_set_unchanged=regions.pocket_set_unchanged, host_stage_s=regions.host_stage_s,
                    flood_levels=regions.flood_levels, traction_steps=self.traction_steps,
                    pockets_at_end=regions.pockets, inlet_covered_at_end=self.covered,
                    transfers=dict(self.transfer_run), noise_event_refreshes=self.noise_event_refreshes,
                    noise_event_pockets=self.noise_event_pockets, noise_events=[dict(e) for e in self.noise_events],
                    seal_steps=regions.seal_steps, min_cells=regions.min_cells,
                    raster_interval=regions.raster_interval, raster_skips=regions.raster_skips,
                    type_memory=regions.type_memory, candidate_counters=dict(regions.candidate_counters),
                    candidate_events=[dict(e) for e in regions.candidate_events],
                    candidates_pending_at_end=len(regions.pending_regions))

    def drain_candidate_events(self):
        """The candidate-region events since the last drain (created / history / sealed / ended), then cleared."""
        out, self.candidate_events_pending = self.candidate_events_pending, []
        return out

    def _floor_material(self):
        """Material cells on the box floor: counted by the classification of a topology change (unchanged otherwise)."""
        topology = (self.regions.last or {}).get("topology")
        return int(topology["floor_material_cells"]) if topology else 0

    def dt_limit(self):
        return self.regions.dt_limit()

    def update(self, t_s, p_in_pa):
        self.chamber.update(t_s, p_in_pa)

    def after_project(self, dt_s, p_in_pa):
        self.chamber.after_project(dt_s, p_in_pa)

    def physical_pressure_into(self, out_field):
        return self.chamber.physical_pressure_into(out_field)

    def physical_pressure_field(self):
        return self.chamber.physical_pressure_field()

    def report(self, *, divergence=False):
        """ColumnPressureChamber-compatible keys (air_volume_m3, divergence_residual_water_max_1ps, ..._outlet_...) plus the
        pocket diagnostics of the last accept."""
        rep = self.chamber.report()
        last = self.last or {}
        return dict(rep, air_volume_m3=last.get("air_volume_m3"), air_cells=last.get("air_cells"),
                    divergence_residual_water_max_1ps=rep["water_divergence_max_1ps"] if divergence else None,
                    divergence_residual_outlet_max_1ps=rep["outlet_divergence_max_1ps"] if divergence else None,
                    pockets=last.get("pockets"),
                    inlet_covered=self.covered, traction_steps=self.traction_steps,
                    pocket_volume_m3=last.get("pocket_volume_m3"), pocket_p_min_abs_pa=last.get("pocket_p_min_abs_pa"),
                    pocket_p_max_abs_pa=last.get("pocket_p_max_abs_pa"), omega_max_rad_s=last.get("omega_max_rad_s"),
                    dt_limit_s=last.get("dt_limit_s"), noise=last.get("noise"),
                    pocket_footprint_kinetic_energy_j=last.get("pocket_footprint_kinetic_energy_j"),
                    occupancy=last.get("occupancy"), seal_events=last.get("seal_events"),
                    exchange_pa_m3=last.get("exchange_pa_m3"), transfers=last.get("transfers"),
                    transfer_run=dict(self.transfer_run), topology_changes=self.regions.topology_changes,
                    host_stages=self.regions.host_stages, pocket_set_unchanged=self.regions.pocket_set_unchanged,
                    noise_event_refreshes=self.noise_event_refreshes, noise_event_pockets=self.noise_event_pockets,
                    candidates_pending=last.get("candidates_pending"),
                    candidate_cells_pending=last.get("candidate_cells_pending"),
                    candidate_counters=last.get("candidate_counters"), raster_skips=last.get("raster_skips"))

    # checkpoint: the sub-components' state under prefixes
    def _parts(self):
        return (("occupancy.", self.occupancy), ("regions.", self.regions), ("chamber.", self.chamber))

    def state_array_names(self):
        return tuple(prefix + name for prefix, part in self._parts() for name in part.state_array_names())

    def state_array(self, name):
        for prefix, part in self._parts():
            if name.startswith(prefix):
                return part.state_array(name[len(prefix):])
        raise KeyError(name)

    def load_state_arrays(self, arrays):
        for prefix, part in self._parts():
            part.load_state_arrays({name[len(prefix):]: value for name, value in arrays.items()
                                    if name.startswith(prefix)})

    def state_host(self):
        host = {prefix + name: value for prefix, part in self._parts() for name, value in part.state_host().items()}
        host.update(covered=self.covered, traction_steps=self.traction_steps, accepts=self.accepts,
                    noise_run=[dict(run) for run in self.noise_run], ke_max_j=self.ke_max_j,
                    transfer_run=dict(self.transfer_run), noise_events=[dict(e) for e in self.noise_events],
                    noise_event_refreshes=self.noise_event_refreshes, noise_event_pockets=self.noise_event_pockets)
        return host

    def load_state_host(self, values):
        for prefix, part in self._parts():
            part.load_state_host({name[len(prefix):]: value for name, value in values.items()
                                  if name.startswith(prefix)})
        self.covered = values["covered"]
        self.traction_steps = values["traction_steps"]
        self.accepts = values["accepts"]
        self.noise_run = [dict(run) for run in values["noise_run"]]
        self.ke_max_j = values["ke_max_j"]
        self.transfer_run = dict(values["transfer_run"])
        self.noise_events = [dict(e) for e in values["noise_events"]]
        self.noise_event_refreshes = values["noise_event_refreshes"]
        self.noise_event_pockets = values["noise_event_pockets"]
