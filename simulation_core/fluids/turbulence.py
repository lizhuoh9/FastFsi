"""Pure NumPy building blocks for the Menter SST-2003 k-omega model.

This module deliberately does not compute wall distance, discretize transport
or diffusion, or couple turbulence to :class:`CartesianFluidSolver`.  It only
implements local algebra from the SST-2003 model.  Callers supply wall distance
and the required local gradients/strain invariant as arrays.

The equations and constants follow NASA's Turbulence Modeling Resource entry
for SST-2003/SST-2003m:
https://tmbwg.github.io/turbmodels/sst.html
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
_TANH_ARGUMENT_CAP = 1.0e3


def _readonly_array(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _broadcast_named(**values: ArrayLike) -> dict[str, FloatArray]:
    names = tuple(values)
    try:
        arrays = tuple(np.asarray(values[name], dtype=np.float64) for name in names)
        broadcast = np.broadcast_arrays(*arrays)
    except (TypeError, ValueError) as exc:
        joined = ", ".join(names)
        raise ValueError(f"SST inputs are not numeric and broadcast-compatible: {joined}") from exc
    return {name: np.asarray(value, dtype=np.float64) for name, value in zip(names, broadcast, strict=True)}


def _require_finite(name: str, value: FloatArray) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")


def _require_positive(name: str, value: FloatArray) -> None:
    _require_finite(name, value)
    if np.any(value <= 0.0):
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: FloatArray) -> None:
    _require_finite(name, value)
    if np.any(value < 0.0):
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class SSTConstants:
    """Coefficients for the standard SST-2003 model."""

    sigma_k_1: float = 0.85
    sigma_k_2: float = 1.0
    sigma_omega_1: float = 0.5
    sigma_omega_2: float = 0.856
    beta_1: float = 0.075
    beta_2: float = 0.0828
    gamma_1: float = 5.0 / 9.0
    gamma_2: float = 0.44
    beta_star: float = 0.09
    kappa: float = 0.41
    a1: float = 0.31
    production_limit_factor: float = 10.0
    cd_kw_floor: float = 1.0e-10

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            value = float(getattr(self, descriptor.name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{descriptor.name} must be positive and finite")


SST2003_CONSTANTS = SSTConstants()


@dataclass(frozen=True)
class SSTValidatedState:
    turbulent_kinetic_energy: FloatArray
    specific_dissipation_rate: FloatArray
    density: FloatArray
    kinematic_viscosity: FloatArray
    wall_distance: FloatArray

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            object.__setattr__(self, descriptor.name, _readonly_array(getattr(self, descriptor.name)))


@dataclass(frozen=True)
class SSTBlendingFunctions:
    f1: FloatArray
    f2: FloatArray
    cd_kw: FloatArray

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            object.__setattr__(self, descriptor.name, _readonly_array(getattr(self, descriptor.name)))


@dataclass(frozen=True)
class SSTBlendedCoefficients:
    sigma_k: FloatArray
    sigma_omega: FloatArray
    beta: FloatArray
    gamma: FloatArray

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            object.__setattr__(self, descriptor.name, _readonly_array(getattr(self, descriptor.name)))


@dataclass(frozen=True)
class SSTLocalSourceStep:
    """One forward-Euler update using local SST source terms only.

    ``k_source`` has units of k/time and ``omega_source`` has units of
    omega/time.  Diffusive and advective transport are intentionally absent.
    """

    f1: FloatArray
    f2: FloatArray
    eddy_viscosity: FloatArray
    raw_production: FloatArray
    limited_production: FloatArray
    cross_diffusion: FloatArray
    k_source: FloatArray
    omega_source: FloatArray
    k_next: FloatArray
    omega_next: FloatArray

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            object.__setattr__(self, descriptor.name, _readonly_array(getattr(self, descriptor.name)))


@dataclass(frozen=True)
class SSTWallCorrelationResult:
    """Local Fluent-style, y+-insensitive SST wall-correlation quantities.

    The result is local algebra only: it neither infers wall distance nor
    applies a boundary condition to a transport field.  Every array is a
    read-only copy so callers cannot mutate the calculation after the fact.
    """

    u_star: FloatArray
    y_plus: FloatArray
    u_laminar_plus: FloatArray
    u_turbulent_plus: FloatArray
    u_plus: FloatArray
    u_tau: FloatArray
    wall_shear_stress: FloatArray
    kinematic_wall_traction_coefficient: FloatArray
    d_u_turbulent_plus_d_y_plus: FloatArray
    omega_laminar_plus: FloatArray
    omega_turbulent_plus: FloatArray
    omega_plus: FloatArray
    wall_specific_dissipation_rate: FloatArray
    production_laminar: FloatArray
    production_turbulent: FloatArray
    wall_production: FloatArray

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            object.__setattr__(self, descriptor.name, _readonly_array(getattr(self, descriptor.name)))


def sst_wall_correlation(
    *,
    relative_tangential_velocity: ArrayLike,
    wall_distance: ArrayLike,
    turbulent_kinetic_energy: ArrayLike,
    specific_dissipation_rate: ArrayLike,
    density: ArrayLike,
    kinematic_viscosity: ArrayLike,
) -> SSTWallCorrelationResult:
    """Evaluate Fluent's y+-insensitive ``correlation`` wall algebra.

    ``relative_tangential_velocity`` may be signed; its magnitude defines the
    wall law.  Turbulent kinetic energy is non-negative, while the other fluid
    state inputs are positive.  ``specific_dissipation_rate`` is the cell
    value used by the correlation-production branch (not the wall omega
    target). This function intentionally leaves transport and boundary
    enforcement to the caller.
    """

    values = _broadcast_named(
        relative_tangential_velocity=relative_tangential_velocity,
        wall_distance=wall_distance,
        turbulent_kinetic_energy=turbulent_kinetic_energy,
        specific_dissipation_rate=specific_dissipation_rate,
        density=density,
        kinematic_viscosity=kinematic_viscosity,
    )
    relative_velocity = values["relative_tangential_velocity"]
    _require_finite("relative_tangential_velocity", relative_velocity)
    _require_positive("wall_distance", values["wall_distance"])
    _require_non_negative(
        "turbulent_kinetic_energy",
        values["turbulent_kinetic_energy"],
    )
    _require_positive("specific_dissipation_rate", values["specific_dissipation_rate"])
    _require_positive("density", values["density"])
    _require_positive("kinematic_viscosity", values["kinematic_viscosity"])

    # Fluent's y+-insensitive SST correlation constants, kept here rather
    # than borrowing the transport-model constants because the two models use
    # different calibrated wall-law values (notably kappa).
    kappa = 0.4187
    e_constant = 9.793
    c_mu = beta_star = 0.09
    beta_i = 0.075
    c_calib = 1.0 / 3.0
    c_exp = 1.3

    dy = values["wall_distance"]
    k = values["turbulent_kinetic_energy"]
    omega = values["specific_dissipation_rate"]
    rho = values["density"]
    nu = values["kinematic_viscosity"]
    speed = np.abs(relative_velocity)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        u_star = np.sqrt(nu * speed / dy + np.sqrt(c_mu) * k)
        y_plus = dy * u_star / nu
        u_laminar_plus = y_plus
        u_turbulent_plus = np.log(e_constant * np.maximum(y_plus, 0.2)) / kappa
        # Algebraically equivalent to ``(a**-4 + b**-4)**-0.25``, but
        # remains finite when the laminar branch is extremely small (the
        # production solver evaluates this law in f32).
        u_plus_min = np.minimum(u_laminar_plus, u_turbulent_plus)
        u_plus_max = np.maximum(u_laminar_plus, u_turbulent_plus)
        u_plus_ratio = np.divide(
            u_plus_min,
            u_plus_max,
            out=np.zeros_like(u_plus_min),
            where=u_plus_max > 0.0,
        )
        u_plus = u_plus_min / (1.0 + u_plus_ratio**4.0) ** 0.25
        u_tau = np.divide(
            speed,
            u_plus,
            out=np.zeros_like(speed),
            where=u_plus > 0.0,
        )
        wall_shear_stress = rho * u_tau * u_star
        kinematic_wall_traction_coefficient = np.divide(
            u_star,
            u_plus,
            out=np.array(nu / dy, copy=True),
            where=u_plus > 0.0,
        )
        d_u_turbulent_plus_d_y_plus = np.where(
            y_plus > 0.2,
            1.0 / (kappa * y_plus),
            0.0,
        )
        omega_laminar_plus = np.divide(
            c_calib * 6.0,
            beta_i * y_plus**2.0,
            out=np.full_like(y_plus, np.inf),
            where=y_plus > 0.0,
        )
        omega_turbulent_plus = d_u_turbulent_plus_d_y_plus / np.sqrt(beta_star)
        omega_plus = omega_laminar_plus * (
            1.0 + (omega_turbulent_plus / omega_laminar_plus) ** c_exp
        ) ** (1.0 / c_exp)
        regular_wall_omega = u_star**2.0 / nu * omega_plus
        laminar_wall_omega = c_calib * 6.0 * nu / (beta_i * dy**2.0)
        wall_specific_dissipation_rate = np.where(
            y_plus <= 0.2,
            laminar_wall_omega,
            regular_wall_omega,
        )
        dynamic_viscosity = rho * nu
        production_laminar = (
            rho
            * k
            / omega
            * (wall_shear_stress / dynamic_viscosity) ** 2.0
        )
        production_turbulent = (
            wall_shear_stress**2.0
            / dynamic_viscosity
            * d_u_turbulent_plus_d_y_plus
        )
        wall_production = np.divide(
            production_laminar * production_turbulent,
            production_laminar + production_turbulent,
            out=np.zeros_like(production_laminar),
            where=(production_laminar + production_turbulent) > 0.0,
        )

    for name, value in (
        ("u_star", u_star),
        ("y_plus", y_plus),
        ("u_laminar_plus", u_laminar_plus),
        ("u_turbulent_plus", u_turbulent_plus),
        ("u_plus", u_plus),
        ("u_tau", u_tau),
        ("wall_shear_stress", wall_shear_stress),
        ("kinematic_wall_traction_coefficient", kinematic_wall_traction_coefficient),
        ("d_u_turbulent_plus_d_y_plus", d_u_turbulent_plus_d_y_plus),
        ("omega_turbulent_plus", omega_turbulent_plus),
        ("wall_specific_dissipation_rate", wall_specific_dissipation_rate),
        ("production_laminar", production_laminar),
        ("production_turbulent", production_turbulent),
        ("wall_production", wall_production),
    ):
        _require_finite(name, value)

    # At the exact laminar limit y+ == 0 the two nondimensional omega
    # quantities diverge, while their dimensional product has the finite
    # analytic limit evaluated above.  Preserve that mathematical distinction
    # instead of hiding it behind an arbitrary positive y+ floor.
    for name, value in (
        ("omega_laminar_plus", omega_laminar_plus),
        ("omega_plus", omega_plus),
    ):
        if np.any(np.isnan(value)) or np.any(value <= 0.0):
            raise ValueError(f"{name} must be positive and not NaN")

    return SSTWallCorrelationResult(
        u_star=u_star,
        y_plus=y_plus,
        u_laminar_plus=u_laminar_plus,
        u_turbulent_plus=u_turbulent_plus,
        u_plus=u_plus,
        u_tau=u_tau,
        wall_shear_stress=wall_shear_stress,
        kinematic_wall_traction_coefficient=kinematic_wall_traction_coefficient,
        d_u_turbulent_plus_d_y_plus=d_u_turbulent_plus_d_y_plus,
        omega_laminar_plus=omega_laminar_plus,
        omega_turbulent_plus=omega_turbulent_plus,
        omega_plus=omega_plus,
        wall_specific_dissipation_rate=wall_specific_dissipation_rate,
        production_laminar=production_laminar,
        production_turbulent=production_turbulent,
        wall_production=wall_production,
    )


def validate_sst_state(
    *,
    turbulent_kinetic_energy: ArrayLike,
    specific_dissipation_rate: ArrayLike,
    density: ArrayLike,
    kinematic_viscosity: ArrayLike,
    wall_distance: ArrayLike,
) -> SSTValidatedState:
    """Validate and broadcast the local state without modifying its inputs.

    Zero turbulent kinetic energy is admissible (for example at a resolved
    wall).  Omega, density, molecular kinematic viscosity, and the supplied
    wall distance must be strictly positive.
    """

    values = _broadcast_named(
        turbulent_kinetic_energy=turbulent_kinetic_energy,
        specific_dissipation_rate=specific_dissipation_rate,
        density=density,
        kinematic_viscosity=kinematic_viscosity,
        wall_distance=wall_distance,
    )
    _require_non_negative("turbulent_kinetic_energy", values["turbulent_kinetic_energy"])
    _require_positive("specific_dissipation_rate", values["specific_dissipation_rate"])
    _require_positive("density", values["density"])
    _require_positive("kinematic_viscosity", values["kinematic_viscosity"])
    _require_positive("wall_distance", values["wall_distance"])
    return SSTValidatedState(**values)


def blend_sst_coefficients(
    f1: ArrayLike,
    *,
    constants: SSTConstants = SST2003_CONSTANTS,
) -> SSTBlendedCoefficients:
    """Blend inner (1) and outer (2) coefficients with ``F1``."""

    values = _broadcast_named(f1=f1)
    f1_array = values["f1"]
    _require_finite("f1", f1_array)
    if np.any((f1_array < 0.0) | (f1_array > 1.0)):
        raise ValueError("f1 must lie in the closed interval [0, 1]")

    def blend(inner: float, outer: float) -> FloatArray:
        return f1_array * inner + (1.0 - f1_array) * outer

    return SSTBlendedCoefficients(
        sigma_k=blend(constants.sigma_k_1, constants.sigma_k_2),
        sigma_omega=blend(constants.sigma_omega_1, constants.sigma_omega_2),
        beta=blend(constants.beta_1, constants.beta_2),
        gamma=blend(constants.gamma_1, constants.gamma_2),
    )


def sst_blending_functions(
    *,
    turbulent_kinetic_energy: ArrayLike,
    specific_dissipation_rate: ArrayLike,
    wall_distance: ArrayLike,
    kinematic_viscosity: ArrayLike,
    density: ArrayLike,
    grad_k_dot_grad_omega: ArrayLike = 0.0,
    constants: SSTConstants = SST2003_CONSTANTS,
) -> SSTBlendingFunctions:
    """Evaluate SST-2003 ``F1`` and ``F2`` from caller-supplied local data."""

    values = _broadcast_named(
        turbulent_kinetic_energy=turbulent_kinetic_energy,
        specific_dissipation_rate=specific_dissipation_rate,
        wall_distance=wall_distance,
        kinematic_viscosity=kinematic_viscosity,
        density=density,
        grad_k_dot_grad_omega=grad_k_dot_grad_omega,
    )
    state = validate_sst_state(
        turbulent_kinetic_energy=values["turbulent_kinetic_energy"],
        specific_dissipation_rate=values["specific_dissipation_rate"],
        density=values["density"],
        kinematic_viscosity=values["kinematic_viscosity"],
        wall_distance=values["wall_distance"],
    )
    grad_dot = values["grad_k_dot_grad_omega"]
    _require_finite("grad_k_dot_grad_omega", grad_dot)

    f1, f2, cd_kw = _sst_blending_arrays(state, grad_dot, constants)
    return SSTBlendingFunctions(f1=f1, f2=f2, cd_kw=cd_kw)


def _sst_blending_arrays(
    state: SSTValidatedState, grad_dot: FloatArray, constants: SSTConstants,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate blending from validated, broadcast-compatible local arrays."""
    k = state.turbulent_kinetic_energy
    omega = state.specific_dissipation_rate
    rho = state.density
    nu = state.kinematic_viscosity
    distance = state.wall_distance

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        cd_kw = np.maximum(
            2.0 * rho * constants.sigma_omega_2 * grad_dot / omega,
            constants.cd_kw_floor,
        )
        viscous_argument = 500.0 * nu / (distance * distance * omega)
        wall_argument = np.sqrt(k) / (constants.beta_star * omega * distance)
        cross_argument = (
            4.0 * rho * constants.sigma_omega_2 * k / (cd_kw * distance * distance)
        )
        argument_1 = np.minimum(np.maximum(wall_argument, viscous_argument), cross_argument)
        argument_2 = np.maximum(2.0 * wall_argument, viscous_argument)

    _require_finite("cd_kw", cd_kw)
    _require_finite("SST F1 argument", argument_1)
    _require_finite("SST F2 argument", argument_2)
    f1 = np.tanh(np.minimum(argument_1, _TANH_ARGUMENT_CAP) ** 4)
    f2 = np.tanh(np.minimum(argument_2, _TANH_ARGUMENT_CAP) ** 2)
    return f1, f2, cd_kw


def sst_eddy_viscosity(
    *,
    density: ArrayLike,
    turbulent_kinetic_energy: ArrayLike,
    specific_dissipation_rate: ArrayLike,
    strain_rate_magnitude: ArrayLike,
    f2: ArrayLike,
    constants: SSTConstants = SST2003_CONSTANTS,
) -> FloatArray:
    """Return dynamic eddy viscosity with the SST-2003 stress limiter."""

    values = _broadcast_named(
        density=density,
        turbulent_kinetic_energy=turbulent_kinetic_energy,
        specific_dissipation_rate=specific_dissipation_rate,
        strain_rate_magnitude=strain_rate_magnitude,
        f2=f2,
    )
    _require_positive("density", values["density"])
    _require_non_negative("turbulent_kinetic_energy", values["turbulent_kinetic_energy"])
    _require_positive("specific_dissipation_rate", values["specific_dissipation_rate"])
    _require_non_negative("strain_rate_magnitude", values["strain_rate_magnitude"])
    _require_finite("f2", values["f2"])
    if np.any((values["f2"] < 0.0) | (values["f2"] > 1.0)):
        raise ValueError("f2 must lie in the closed interval [0, 1]")

    return _readonly_array(_sst_eddy_viscosity_array(
        values["density"], values["turbulent_kinetic_energy"],
        values["specific_dissipation_rate"], values["strain_rate_magnitude"],
        values["f2"], constants,
    ))


def _sst_eddy_viscosity_array(
    density: FloatArray, turbulent_kinetic_energy: FloatArray,
    specific_dissipation_rate: FloatArray, strain_rate_magnitude: FloatArray,
    f2: FloatArray, constants: SSTConstants,
) -> FloatArray:
    denominator = np.maximum(
        constants.a1 * specific_dissipation_rate,
        strain_rate_magnitude * f2,
    )
    eddy_viscosity = (
        density
        * constants.a1
        * turbulent_kinetic_energy
        / denominator
    )
    _require_finite("eddy_viscosity", eddy_viscosity)
    return eddy_viscosity


def sst_production_limiter(
    *,
    production: ArrayLike,
    density: ArrayLike,
    turbulent_kinetic_energy: ArrayLike,
    specific_dissipation_rate: ArrayLike,
    constants: SSTConstants = SST2003_CONSTANTS,
) -> FloatArray:
    """Apply the SST-2003 ``10 beta* rho k omega`` production cap."""

    values = _broadcast_named(
        production=production,
        density=density,
        turbulent_kinetic_energy=turbulent_kinetic_energy,
        specific_dissipation_rate=specific_dissipation_rate,
    )
    _require_non_negative("production", values["production"])
    _require_positive("density", values["density"])
    _require_non_negative("turbulent_kinetic_energy", values["turbulent_kinetic_energy"])
    _require_positive("specific_dissipation_rate", values["specific_dissipation_rate"])
    return _readonly_array(_sst_limited_production_array(
        values["production"], values["density"], values["turbulent_kinetic_energy"],
        values["specific_dissipation_rate"], constants,
    ))


def _sst_limited_production_array(
    production: FloatArray, density: FloatArray, turbulent_kinetic_energy: FloatArray,
    specific_dissipation_rate: FloatArray, constants: SSTConstants,
) -> FloatArray:
    cap = (
        constants.production_limit_factor
        * constants.beta_star
        * density
        * turbulent_kinetic_energy
        * specific_dissipation_rate
    )
    limited = np.minimum(production, cap)
    _require_finite("limited_production", limited)
    return limited


def sst_local_source_step(
    *,
    turbulent_kinetic_energy: ArrayLike,
    specific_dissipation_rate: ArrayLike,
    density: ArrayLike,
    kinematic_viscosity: ArrayLike,
    wall_distance: ArrayLike,
    strain_rate_magnitude: ArrayLike,
    grad_k_dot_grad_omega: ArrayLike = 0.0,
    dt_s: float,
    constants: SSTConstants = SST2003_CONSTANTS,
) -> SSTLocalSourceStep:
    """Advance one explicit step using only the local SST-2003m sources.

    A step that would create negative ``k`` or non-positive ``omega`` is
    rejected instead of silently clipping the state.  The caller can then
    reduce ``dt_s`` or substep explicitly.
    """

    dt = float(dt_s)
    if not isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be positive and finite")
    values = _broadcast_named(
        turbulent_kinetic_energy=turbulent_kinetic_energy,
        specific_dissipation_rate=specific_dissipation_rate,
        density=density,
        kinematic_viscosity=kinematic_viscosity,
        wall_distance=wall_distance,
        strain_rate_magnitude=strain_rate_magnitude,
        grad_k_dot_grad_omega=grad_k_dot_grad_omega,
    )
    state = validate_sst_state(
        turbulent_kinetic_energy=values["turbulent_kinetic_energy"],
        specific_dissipation_rate=values["specific_dissipation_rate"],
        density=values["density"],
        kinematic_viscosity=values["kinematic_viscosity"],
        wall_distance=values["wall_distance"],
    )
    strain = values["strain_rate_magnitude"]
    grad_dot = values["grad_k_dot_grad_omega"]
    _require_non_negative("strain_rate_magnitude", strain)
    _require_finite("grad_k_dot_grad_omega", grad_dot)

    f1, f2, _ = _sst_blending_arrays(state, grad_dot, constants)
    coefficients = blend_sst_coefficients(f1, constants=constants)
    eddy_viscosity = _sst_eddy_viscosity_array(
        state.density, state.turbulent_kinetic_energy, state.specific_dissipation_rate,
        strain, f2, constants,
    )
    raw_production = eddy_viscosity * strain * strain
    _require_non_negative("production", raw_production)
    limited_production = _sst_limited_production_array(
        raw_production, state.density, state.turbulent_kinetic_energy,
        state.specific_dissipation_rate, constants,
    )

    k = state.turbulent_kinetic_energy
    omega = state.specific_dissipation_rate
    rho = state.density
    k_source = limited_production / rho - constants.beta_star * omega * k
    # Evaluate limited_production / mu_t algebraically.  Both numerator and
    # denominator vanish linearly as k -> 0, so a guarded division would
    # incorrectly erase the finite omega-production limit at a resolved wall.
    eddy_viscosity_denominator = np.maximum(
        constants.a1 * omega,
        strain * f2,
    )
    production_over_eddy_viscosity = np.minimum(
        strain * strain,
        (
            constants.production_limit_factor
            * constants.beta_star
            * omega
            * eddy_viscosity_denominator
            / constants.a1
        ),
    )
    omega_production = coefficients.gamma * production_over_eddy_viscosity
    cross_diffusion = (
        2.0
        * (1.0 - f1)
        * constants.sigma_omega_2
        * grad_dot
        / omega
    )
    omega_source = omega_production - coefficients.beta * omega * omega + cross_diffusion
    k_next = k + dt * k_source
    omega_next = omega + dt * omega_source

    for name, value in (
        ("raw_production", raw_production),
        ("cross_diffusion", cross_diffusion),
        ("k_source", k_source),
        ("omega_source", omega_source),
        ("k_next", k_next),
        ("omega_next", omega_next),
    ):
        _require_finite(name, value)
    if np.any(k_next < 0.0) or np.any(omega_next <= 0.0):
        raise ValueError(
            "local SST source step must preserve non-negative turbulent_kinetic_energy "
            "and positive specific_dissipation_rate; reduce dt_s"
        )

    return SSTLocalSourceStep(
        f1=f1,
        f2=f2,
        eddy_viscosity=eddy_viscosity,
        raw_production=raw_production,
        limited_production=limited_production,
        cross_diffusion=cross_diffusion,
        k_source=k_source,
        omega_source=omega_source,
        k_next=k_next,
        omega_next=omega_next,
    )


__all__ = [
    "SST2003_CONSTANTS",
    "SSTBlendedCoefficients",
    "SSTBlendingFunctions",
    "SSTConstants",
    "SSTLocalSourceStep",
    "SSTValidatedState",
    "SSTWallCorrelationResult",
    "blend_sst_coefficients",
    "sst_blending_functions",
    "sst_eddy_viscosity",
    "sst_local_source_step",
    "sst_production_limiter",
    "sst_wall_correlation",
    "validate_sst_state",
]
