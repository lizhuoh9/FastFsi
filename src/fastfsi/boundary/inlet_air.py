"""The inlet air of a lid-inlet chamber as a lumped gas volume behind a flow-limited line (the capped inlet, stage 1;
Root 2026-10-23 on the USER's decision "同意把入口改成气室空气质量平衡加限流入口，先在现有流程上做出来并验证。").

IDEAL (the default): the inlet-connected air takes the line pressure p_line(t) with no resistance; nothing is integrated
and PneumaticChamber's path is unchanged.
CAPPED: the inlet-connected air region (DeviceGasRegions' INLET_GAS cells, the pending candidate regions included, plus
an optional line volume V_line) holds its own gas amount q = p_abs V (Pa m^3, isothermal like the sealed pockets; its
mass is m = rho_atm q / p_atm).  The line feeds it from the measured line pressure through round passages in series,
in flow order (SeriesRestriction; the robot, USER 2026-10-23: the gauge's Y fitting -> its through bore -> the hose ->
the straight connector's bore -> the lid hole -> the chamber; by default the lid hole alone):
    p_line - p_chamber = dp(Q) = a Q + b Q |Q|  (+ a turbulent hose's friction)
                                                Q > 0 into the chamber, free air at p_atm (m^3/s)
  FRICTION: each passage's laminar (Poiseuille) a_i = 128 mu L_i / (pi D_i^4); a hose ("tube") switches to Blasius above
    Re = rho U D / mu = 2300: dp_i = f (L_i / D_i) rho U^2 / 2, f = 0.3164 Re^(-1/4) (no transition model: where the law
    jumps up, the flow stays at Re = 2300).  The short passages (the hole, the fittings' bores) keep the laminar law.
  JOINTS (dynamic heads rho U^2 / 2 of the smaller section): the measuring point and the chamber are large volumes; a
    contraction loses 0.5 (1 - A_small / A_large) heads and an expansion (1 - A_small / A_large)^2 (Borda-Carnot), so
    the lid hole alone has its entrance 0.5 and exit 1.0 heads: b = 1.5 rho / (2 A^2).
(Derived from the geometry, never fitted: the 2 mm hole through the 7 mm lid gives 123 Pa at 35 mL/s, 112 Pa of it the
dynamic heads, at p_atm.)  DENSITY (Root 2026-10-23): the gas in the line is at p_abs, so its volume flow is the
free-air Q x p_atm / p_abs and every loss term (friction, heads, Blasius; Re depends on the mass flow only) scales by
s = p_atm / p_mean, p_mean = p_atm + (p_line + p_chamber) / 2 (the mean is exact for isothermal Poiseuille flow;
p_chamber of the last step).  The flow is capped at the pump's free flow |Q| <= Q_cap, and dq/dt = p_atm Q.
MACH (a logged limit, no model): the narrowest passage's |Q| s / (A_min c), c = 343.2 m/s; mach_max is kept.
STEP (after the accepted geometry gave V_{n+1}; backward Euler in q): with q' the gas after the step's pocket exchange,
p_0 = q' / V - p_atm (no flow) and c = p_atm dt / V, the line's law at the new state is s dp(Q) + c Q = D with
D = p_line - p_0.  While every hose is laminar it is s b Q|Q| + (s a + c) Q = D, solved in closed form,
Q = sign(D) 2|D| / ((s a + c) + sqrt((s a + c)^2 + 4 s b |D|)) (b = 0: the linear law Q = D / (s a + c)); a hose above
Re = 2300 at that root only adds resistance, and the root is bisected below it.  Then |Q| <= Q_cap (Q_capped
otherwise), q = q' + p_atm dt Q and p_chamber = q / V - p_atm.  The step is stable for any time constant
V / (p_atm dQ/dp); the fluid solve uses the state's p_chamber (lagged by at most half a step).
COVERED (no free inlet cell): Q = 0 and q is kept (the pending air under the membrane is closed); the covered hole
itself carries the LINE pressure (no flow, no drop along the line: PneumaticChamber.inlet_traction_gauge_pa).  With no
gas volume at all (V = 0 with V_line = 0) p_chamber falls back to p_line: the line pressure acts through the hole; every
fallback is counted and the first FALLBACK_LOG_MAX are logged.  REAPPEARANCE (Root's review 2026-10-23): a region that
has volume again after a step (or a start) without it restarts in equilibrium with the line, q = (p_line + p_atm) V --
the hole's gas sat at p_line while nothing flowed -- instead of the near-vacuum its empty gas amount would give; the gas
set so is counted (reappear_pa_m3, in the ledger) and the first FALLBACK_LOG_MAX events are logged.
POCKETS: a region sealing off the inlet air takes its gas from q and a share reopening to it returns its gas
(gas_pocket_transitions.allocate_pocket_gas's exchange, evaluated at p_atm + p_chamber).
LEDGER: inflow = sum p_atm Q dt, exchange = sum of the pocket exchanges and reappear = the gas set at reappearances;
q - q_0 = inflow - exchange + reappear to round-off.
GUARD: a non-finite or non-positive absolute pressure of the capped region is the hard stop inlet_air_invalid_pressure.
CHECKPOINT: the state carries the line's parameters (line_config: a, b, Q_cap, V_line, the elements); a model with
another line refuses it.
"""
import math
from dataclasses import dataclass

from fastfsi.boundary.gas_pockets import P_ATMOSPHERE_PA
from fastfsi.errors import SimulationHardStop

AIR_DENSITY_KG_M3 = 1.204          # dry air at 20 C and 101325 Pa: the free-air density of the line's law
AIR_VISCOSITY_PA_S = 1.81e-5       # dry air at 20 C
HOLE_LOSS_HEADS = 1.5              # a sharp-edged hole between large volumes: entrance 0.5 + exit 1.0 dynamic heads
CONTRACTION_HEADS = 0.5            # a sudden contraction: 0.5 (1 - A_small / A_large) heads of the small section
LAMINAR_REYNOLDS_MAX = 2300.0      # a hose's friction: Poiseuille up to here, Blasius above
BLASIUS_FACTOR = 0.3164            # Darcy f = 0.3164 Re^(-1/4), a smooth tube
BISECTION_STEPS_MAX = 200
SOUND_SPEED_M_S = 343.2            # dry air at 20 C: the logged Mach number's reference (no compressibility model)
ELEMENT_KINDS = ("fitting", "tube", "hole")
INLET_MODELS = ("ideal", "capped")
LPM_TO_M3_S = 1.0e-3 / 60.0        # L/min of free air to m^3/s
FALLBACK_LOG_MAX = 50
HARD_STOP_CODES = {"inlet_air_invalid_pressure"}


def hole_law(diameter_m, length_m, *, viscosity_pa_s=AIR_VISCOSITY_PA_S, density_kg_m3=AIR_DENSITY_KG_M3,
             loss_heads=HOLE_LOSS_HEADS):
    """(a, b) of the round hole's pressure drop dp = a Q + b Q |Q| (Q free air in m^3/s): the laminar friction
    a = 128 mu L / (pi D^4) (Poiseuille) and the entrance and exit losses b = loss_heads rho / (2 A^2),
    A = pi D^2 / 4."""
    if not (math.isfinite(diameter_m) and diameter_m > 0.0 and math.isfinite(length_m) and length_m >= 0.0):
        raise ValueError("the hole needs a finite positive diameter and a finite length >= 0")
    area = 0.25 * math.pi * diameter_m ** 2
    return (128.0 * viscosity_pa_s * length_m / (math.pi * diameter_m ** 4),
            loss_heads * density_kg_m3 / (2.0 * area * area))


def hole_pressure_drop_pa(q_m3_s, a, b):
    """The hole law: the pressure drop of the flow q (sign of q)."""
    return a * q_m3_s + b * q_m3_s * abs(q_m3_s)


def solve_hole_flow(d_pa, a, b):
    """The flow Q with b Q |Q| + a Q = d (a, b >= 0): the root of the sign of d, in the cancellation-free form
    Q = sign(d) 2 |d| / (a + sqrt(a^2 + 4 b |d|)); infinite (sign of d) for a = b = 0 and d != 0."""
    if d_pa == 0.0:
        return 0.0
    size = abs(d_pa)
    denominator = a + math.sqrt(a * a + 4.0 * b * size)
    if denominator <= 0.0:
        return math.copysign(math.inf, d_pa)
    return math.copysign(2.0 * size / denominator, d_pa)


@dataclass(frozen=True)
class FlowElement:
    """One round passage of the inlet line, in flow order: "tube" (a hose: Blasius above Re 2300), "fitting" (a barb's
    bore) or "hole" (the lid hole); the short passages keep the laminar friction.  length_m = 0: the joints only."""
    name: str
    kind: str
    diameter_m: float
    length_m: float = 0.0

    def __post_init__(self):
        if self.kind not in ELEMENT_KINDS:
            raise ValueError("a line element's kind must be one of %s" % (ELEMENT_KINDS,))
        if not (math.isfinite(self.diameter_m) and self.diameter_m > 0.0 and math.isfinite(self.length_m)
                and self.length_m >= 0.0):
            raise ValueError("line element %r needs a finite positive diameter and a finite length >= 0" % (self.name,))

    @property
    def area_m2(self):
        return 0.25 * math.pi * self.diameter_m ** 2


class SeriesRestriction:
    """The line between the measured pressure and the chamber (module docstring): FlowElements in series, or given
    coefficients (a, b) with no elements (tests; (0, 0): no restriction).  The lid hole alone is hole_law's a and b
    bit for bit."""

    def __init__(self, elements=(), *, coefficients=None, viscosity_pa_s=AIR_VISCOSITY_PA_S,
                 density_kg_m3=AIR_DENSITY_KG_M3, laminar_reynolds_max=LAMINAR_REYNOLDS_MAX):
        self.elements = tuple(elements)
        if not all(isinstance(e, FlowElement) for e in self.elements):
            raise TypeError("the line's elements must be FlowElements")
        self.viscosity_pa_s = float(viscosity_pa_s)
        self.density_kg_m3 = float(density_kg_m3)
        self.laminar_reynolds_max = float(laminar_reynolds_max)
        self.joints = []
        self.tubes = []                 # each hose with a length: (laminar a, Blasius k, the laminar range's Q limit)
        self.min_area_m2 = min((e.area_m2 for e in self.elements), default=None)    # the Mach number's passage
        if coefficients is not None:
            if self.elements:
                raise ValueError("give the line's elements or its coefficients, not both")
            self.a, self.b = (float(v) for v in coefficients)
            if not all(math.isfinite(v) and v >= 0.0 for v in (self.a, self.b)):
                raise ValueError("the line's coefficients must be finite and >= 0")
            self.a_short = self.a
            self.source = "coefficients"
            return
        self.source = "geometry"
        self.a = self.b = self.a_short = 0.0
        for element, heads in zip(self.elements, self._joint_heads()):
            area = element.area_m2
            a_i = 128.0 * self.viscosity_pa_s * element.length_m / (math.pi * element.diameter_m ** 4)
            self.a += a_i
            self.b += heads * self.density_kg_m3 / (2.0 * area * area)
            if element.kind == "tube" and element.length_m > 0.0:
                self.tubes.append((a_i, self._blasius_k(element), self.laminar_reynolds_max * self.viscosity_pa_s
                                   * area / (self.density_kg_m3 * element.diameter_m)))
            else:
                self.a_short += a_i

    def _joint_heads(self):
        """The joints' dynamic heads per passage (each of the smaller section at a change of area); records the
        joints."""
        heads = [0.0] * len(self.elements)
        areas = [e.area_m2 for e in self.elements]
        for i in range(len(areas) + 1):
            up = areas[i - 1] if i > 0 else math.inf               # the measuring point: a large volume
            down = areas[i] if i < len(areas) else math.inf        # the chamber: a large volume
            if down < up:                                          # a contraction into passage i
                k, at, kind = CONTRACTION_HEADS * (1.0 - down / up), i, "entrance" if i == 0 else "contraction"
            elif up < down:                                        # an expansion out of passage i - 1
                k, at, kind = (1.0 - up / down) ** 2, i - 1, "exit" if i == len(areas) else "expansion"
            else:
                continue
            heads[at] += k
            self.joints.append(dict(kind=kind, heads=k, section=self.elements[at].name,
                                    upstream=self.elements[i - 1].name if i > 0 else "measuring point",
                                    downstream=self.elements[i].name if i < len(areas) else "chamber"))
        return heads

    def _blasius_k(self, element):
        """dp = k |Q|^1.75 of a turbulent hose: f (L / D) rho U^2 / 2 with f = 0.3164 Re^(-1/4) and U = Q / A."""
        d, rho = element.diameter_m, self.density_kg_m3
        return (BLASIUS_FACTOR * (self.viscosity_pa_s / (rho * d)) ** 0.25 * (element.length_m / d) * 0.5 * rho
                / element.area_m2 ** 1.75)

    def _drop(self, size):
        """dp at the flow size >= 0 (free air at p_atm)."""
        drop = self.a_short * size + self.b * size * size
        for a_t, k_t, q_laminar in self.tubes:
            drop += a_t * size if size <= q_laminar else k_t * size ** 1.75
        return drop

    def pressure_drop_pa(self, q_m3_s):
        """The line's law dp(Q) at p_atm (odd in Q)."""
        return math.copysign(self._drop(abs(q_m3_s)), q_m3_s) if q_m3_s else 0.0

    def solve(self, d_pa, extra_linear=0.0, loss_scale=1.0):
        """The flow Q with loss_scale dp(Q) + extra_linear Q = d_pa and its regime: "laminar" (closed form, every hose
        laminar), "turbulent" (a hose above Re 2300: bisected below the laminar root) or "transition" (d falls where a
        hose's law jumps: Q stays at its Re = 2300).  loss_scale: the density factor p_atm / p_mean."""
        flow = solve_hole_flow(d_pa, loss_scale * self.a + extra_linear, loss_scale * self.b)
        size = abs(flow)
        if not any(size > q_laminar for _, _, q_laminar in self.tubes):
            return flow, "laminar"
        target, low, high = abs(d_pa), 0.0, size
        for _ in range(BISECTION_STEPS_MAX):
            middle = 0.5 * (low + high)
            if not low < middle < high:
                break
            if loss_scale * self._drop(middle) + extra_linear * middle <= target:
                low = middle
            else:
                high = middle
        regime = "transition" if any(low <= q_laminar < high for _, _, q_laminar in self.tubes) else "turbulent"
        return math.copysign(low, d_pa), regime

    def describe(self):
        """The line for run_meta: its passages, joints and composed a and b, and the hoses' laws."""
        return dict(source=self.source, a_pa_s_m3=self.a, b_pa_s2_m6=self.b,
                    elements=[dict(name=e.name, kind=e.kind, diameter_m=e.diameter_m, length_m=e.length_m)
                              for e in self.elements],
                    joints=[dict(j) for j in self.joints],
                    tubes=[dict(laminar_a_pa_s_m3=a_t, blasius_k=k_t, laminar_q_max_m3_s=q_laminar)
                           for a_t, k_t, q_laminar in self.tubes],
                    air=dict(density_kg_m3=self.density_kg_m3, viscosity_pa_s=self.viscosity_pa_s),
                    friction="Poiseuille a_i = 128 mu L_i / (pi D_i^4); a tube: Blasius f = 0.3164 Re^(-1/4) above "
                             "Re %g (the flow stays at Re %g where the law jumps)"
                             % (self.laminar_reynolds_max, self.laminar_reynolds_max),
                    joint_rule="heads rho U^2 / 2 of the smaller section: contraction 0.5 (1 - A_small / A_large), "
                               "expansion (1 - A_small / A_large)^2; the measuring point and the chamber are large")


class InletAirModel:
    """The inlet air behind the line (module docstring).  ideal: a pass-through (pressure_gauge_pa() returns the line
    pressure; start() and advance() keep no state).  capped: a few host floats per accept.  hole_a / hole_b are the
    line's composed a and b (the lid hole's alone by default)."""

    HOST_FIELDS = ("q_pa_m3", "q0_pa_m3", "p_gauge_pa", "volume_m3", "q_m3_s", "capped", "covered", "fallback",
                   "inflow_pa_m3", "exchange_pa_m3", "steps", "capped_steps", "covered_steps", "fallbacks",
                   "fallback_events", "q_capped_max_m3_s", "p_min_gauge_pa", "p_max_gauge_pa", "regime",
                   "turbulent_steps", "transition_steps", "loss_scale", "mach", "mach_max", "reappearances",
                   "reappear_pa_m3", "reappear_events")

    def __init__(self, kind="ideal", *, hole_a_pa_s_m3=0.0, hole_b_pa_s2_m6=0.0, restriction=None,
                 flow_cap_m3_s=math.inf, line_volume_m3=0.0, ambient_pa=P_ATMOSPHERE_PA):
        if kind not in INLET_MODELS:
            raise ValueError("the inlet model must be one of %s" % (INLET_MODELS,))
        if restriction is None:
            restriction = SeriesRestriction(coefficients=(hole_a_pa_s_m3, hole_b_pa_s2_m6))
        elif hole_a_pa_s_m3 or hole_b_pa_s2_m6:
            raise ValueError("give the line's restriction or the hole coefficients, not both")
        if not (math.isfinite(line_volume_m3) and line_volume_m3 >= 0.0):
            raise ValueError("the line volume must be finite and >= 0")
        if not (flow_cap_m3_s > 0.0) or math.isnan(flow_cap_m3_s):
            raise ValueError("the flow cap must be positive (inf: no cap)")
        if not (math.isfinite(ambient_pa) and ambient_pa > 0.0):
            raise ValueError("the ambient pressure must be finite and positive")
        self.kind = kind
        self.restriction = restriction
        self.hole_a = restriction.a
        self.hole_b = restriction.b
        self.flow_cap_m3_s = float(flow_cap_m3_s)
        self.line_volume_m3 = float(line_volume_m3)
        self.ambient_pa = float(ambient_pa)
        self.q_pa_m3 = None
        self.q0_pa_m3 = None
        self.p_gauge_pa = None
        self.volume_m3 = None
        self.q_m3_s = 0.0
        self.capped = False
        self.covered = False
        self.fallback = False
        self.regime = "laminar"
        self.loss_scale = 1.0
        self.mach = 0.0
        self.mach_max = 0.0
        self.inflow_pa_m3 = 0.0
        self.exchange_pa_m3 = 0.0
        self.steps = 0
        self.capped_steps = 0
        self.covered_steps = 0
        self.fallbacks = 0
        self.turbulent_steps = 0
        self.transition_steps = 0
        self.fallback_events = []
        self.reappearances = 0
        self.reappear_pa_m3 = 0.0
        self.reappear_events = []
        self.q_capped_max_m3_s = 0.0
        self.p_min_gauge_pa = None
        self.p_max_gauge_pa = None

    @property
    def is_capped(self):
        return self.kind == "capped"

    def line_config(self):
        """The line's parameters a checkpoint must match (JSON-ready: no inf, lists)."""
        return dict(kind=self.kind, a_pa_s_m3=self.hole_a, b_pa_s2_m6=self.hole_b,
                    flow_cap_m3_s=self.flow_cap_m3_s if math.isfinite(self.flow_cap_m3_s) else None,
                    line_volume_m3=self.line_volume_m3, ambient_pa=self.ambient_pa,
                    elements=[[e.name, e.kind, e.diameter_m, e.length_m] for e in self.restriction.elements])

    def describe(self):
        """The model's parameters (run_meta / setup report)."""
        return dict(kind=self.kind, hole_a_pa_s_m3=self.hole_a, hole_b_pa_s2_m6=self.hole_b,
                    flow_cap_m3_s=(self.flow_cap_m3_s if math.isfinite(self.flow_cap_m3_s) else None),
                    line_volume_m3=self.line_volume_m3, ambient_pa=self.ambient_pa,
                    restriction=self.restriction.describe(),
                    law="p_line - p_chamber = s dp(Q), dp(Q) = a Q + b Q|Q| (+ a turbulent hose's friction; free air "
                        "at p_atm), |Q| <= Q_cap, dq/dt = p_atm Q, q = p_abs V",
                    density="s = p_atm / p_mean, p_mean = p_atm + (p_line + p_chamber) / 2 (p_chamber of the last "
                            "step): the line's volume flow is the free-air Q x p_atm / p_abs",
                    mach="logged, no model: the narrowest passage's |Q| s / (A_min c), c = %g m/s" % SOUND_SPEED_M_S,
                    integration="backward Euler in q after the accepted geometry, the law solved in closed form "
                                "(bisected while a hose is turbulent); the fluid solve uses the state's p_chamber "
                                "(lagged by at most half a step)")

    def pressure_gauge_pa(self, p_line_gauge_pa):
        """The inlet air's gauge pressure for the next solve: ideal -> the line pressure; capped -> the state's."""
        if not self.is_capped or self.p_gauge_pa is None:
            return float(p_line_gauge_pa)
        return self.p_gauge_pa

    def start(self, p_line_gauge_pa, air_volume_m3):
        """The first accepted geometry: the inlet air in equilibrium with the line (capped only)."""
        if not self.is_capped:
            return
        volume = float(air_volume_m3) + self.line_volume_m3
        self.p_gauge_pa = float(p_line_gauge_pa)
        self.volume_m3 = volume
        self.q_pa_m3 = (self.p_gauge_pa + self.ambient_pa) * volume
        self.q0_pa_m3 = self.q_pa_m3
        self.p_min_gauge_pa = self.p_max_gauge_pa = self.p_gauge_pa

    def advance(self, dt_s, p_line_gauge_pa, air_volume_m3, covered, exchange_pa_m3, t_s=None):
        """One accepted step of length dt_s (capped only): the pocket exchange, then the flow through the line at the
        new volume; returns the new gauge pressure (the line pressure for ideal)."""
        if not self.is_capped:
            return float(p_line_gauge_pa)
        dt, p_line = float(dt_s), float(p_line_gauge_pa)
        exchange = float(exchange_pa_m3)
        q = self.q_pa_m3 - exchange
        self.exchange_pa_m3 += exchange
        volume = float(air_volume_m3) + self.line_volume_m3
        if volume > 0.0 and not self.volume_m3 > 0.0:         # back after a step (or a start) without volume:
            q_set = (p_line + self.ambient_pa) * volume        # the hole's gas sat at p_line while nothing flowed
            self.reappear_pa_m3 += q_set - q
            self.reappearances += 1
            if len(self.reappear_events) < FALLBACK_LOG_MAX:
                self.reappear_events.append(dict(time_s=t_s, p_line_gauge_pa=p_line, volume_m3=volume,
                                                 q_before_pa_m3=q, covered=bool(covered)))
            q = q_set
        flow, capped, fallback, regime, scale = 0.0, False, False, "laminar", 1.0
        if not volume > 0.0:
            fallback = True                         # no gas volume: the line pressure acts through the hole
        elif not covered:
            p_free = q / volume - self.ambient_pa
            c = self.ambient_pa * dt / volume
            p_mean = self.ambient_pa + 0.5 * (p_line + self.p_gauge_pa)    # the line's mean absolute pressure
            if not (math.isfinite(p_mean) and p_mean > 0.0):
                raise SimulationHardStop("inlet_air_invalid_pressure", dict(
                    time_s=t_s, p_mean_abs_pa=p_mean, p_line_gauge_pa=p_line, p_gauge_pa=self.p_gauge_pa))
            scale = self.ambient_pa / p_mean                               # the density: volume flow Q p_atm / p_abs
            flow, regime = self.restriction.solve(p_line - p_free, c, scale)
            if abs(flow) > self.flow_cap_m3_s:
                flow, capped = math.copysign(self.flow_cap_m3_s, flow), True
            q += self.ambient_pa * dt * flow
        pressure = p_line if fallback else q / volume - self.ambient_pa
        if not (math.isfinite(pressure) and pressure + self.ambient_pa > 0.0):
            raise SimulationHardStop("inlet_air_invalid_pressure", dict(
                time_s=t_s, p_gauge_pa=pressure, q_pa_m3=q, volume_m3=volume, flow_m3_s=flow, p_line_gauge_pa=p_line,
                exchange_pa_m3=exchange, covered=bool(covered)))
        self.q_pa_m3 = q
        self.p_gauge_pa = pressure
        self.volume_m3 = volume
        self.q_m3_s = flow
        self.capped = capped
        self.covered = bool(covered)
        self.fallback = fallback
        self.regime = regime
        self.loss_scale = scale
        area = self.restriction.min_area_m2
        self.mach = abs(flow) * scale / (area * SOUND_SPEED_M_S) if area else 0.0
        self.mach_max = max(self.mach_max, self.mach)
        self.inflow_pa_m3 += self.ambient_pa * dt * flow
        self.steps += 1
        self.capped_steps += int(capped)
        self.covered_steps += int(bool(covered))
        self.turbulent_steps += int(regime == "turbulent")
        self.transition_steps += int(regime == "transition")
        if capped:
            self.q_capped_max_m3_s = max(self.q_capped_max_m3_s, abs(flow))
        self.p_min_gauge_pa = min(self.p_min_gauge_pa, pressure)
        self.p_max_gauge_pa = max(self.p_max_gauge_pa, pressure)
        if fallback:
            self.fallbacks += 1
            if len(self.fallback_events) < FALLBACK_LOG_MAX:
                self.fallback_events.append(dict(time_s=t_s, p_line_gauge_pa=p_line, covered=bool(covered),
                                                 q_pa_m3=q))
        return pressure

    def mass_kg(self, density_kg_m3=AIR_DENSITY_KG_M3):
        """The inlet air's mass m = rho_atm q / p_atm (None before start)."""
        return None if self.q_pa_m3 is None else density_kg_m3 * self.q_pa_m3 / self.ambient_pa

    def ledger(self):
        """q - q_0 against inflow - exchange + reappear (Pa m^3) and the residual."""
        if self.q_pa_m3 is None:
            return None
        change = self.q_pa_m3 - self.q0_pa_m3
        return dict(q_change_pa_m3=change, inflow_pa_m3=self.inflow_pa_m3, exchange_pa_m3=self.exchange_pa_m3,
                    reappear_pa_m3=self.reappear_pa_m3,
                    residual_pa_m3=change - (self.inflow_pa_m3 - self.exchange_pa_m3 + self.reappear_pa_m3))

    def summary(self):
        return dict(self.describe(), steps=self.steps, capped_steps=self.capped_steps,
                    covered_steps=self.covered_steps, fallbacks=self.fallbacks,
                    turbulent_steps=self.turbulent_steps, transition_steps=self.transition_steps,
                    mach_max=self.mach_max, reappearances=self.reappearances,
                    fallback_events=[dict(e) for e in self.fallback_events],
                    reappear_events=[dict(e) for e in self.reappear_events], q_capped_max_m3_s=self.q_capped_max_m3_s,
                    p_min_gauge_pa=self.p_min_gauge_pa, p_max_gauge_pa=self.p_max_gauge_pa, ledger=self.ledger(),
                    mass_kg=self.mass_kg())

    def state_host(self):
        values = {name: getattr(self, name) for name in self.HOST_FIELDS}
        values["fallback_events"] = [dict(e) for e in self.fallback_events]
        values["reappear_events"] = [dict(e) for e in self.reappear_events]
        values["line_config"] = self.line_config()
        return values

    def load_state_host(self, values):
        """The state of a checkpoint written with the same line (else ValueError: a run cannot change its line)."""
        if values.get("line_config") != self.line_config():
            raise ValueError("the checkpoint's inlet line %s is not this model's %s"
                             % (values.get("line_config"), self.line_config()))
        for name in self.HOST_FIELDS:
            setattr(self, name, values[name])
        self.fallback_events = [dict(e) for e in values["fallback_events"]]
        self.reappear_events = [dict(e) for e in values["reappear_events"]]
