"""Hyperelastic material laws of the material points.

Neo-Hookean solid with modified invariants and a volumetric term:

    W = mu/2 (I1b - 3) + kappa/2 (J - 1)^2,   I1b = J^(-2/3) tr(b),   b = F F^T,   J = det F,

with the Kirchhoff stress tau = mu J^(-2/3) dev(b) + kappa (J - 1) J I.  The volumetric term is a numerical
stabiliser of a nearly incompressible solid: in a single-velocity immersed-boundary model the solid moves with the
divergence-free fluid velocity, so its volume is kept by the projection and kappa only damps the drift of J.

evaluate_neo_hookean_kirchhoff: the 3-D stress (3 x 3).
evaluate_neo_hookean_kirchhoff_planar: the plane-strain stress of a quasi-2-D body (the (y, z) block, F = diag(1, F2)).
NeoHookeanMaterial: the three parameters, for configurations.

No `from __future__ import annotations` (Taichi kernel module).
"""
import math
from dataclasses import dataclass

import taichi as ti


@dataclass(frozen=True)
class NeoHookeanMaterial:
    """Density (kg/m^3), shear modulus mu (Pa) and volumetric modulus kappa (Pa) of a neo-Hookean solid."""

    density_kgm3: float
    shear_modulus_pa: float
    volumetric_modulus_pa: float

    def __post_init__(self):
        for name in ("density_kgm3", "shear_modulus_pa", "volumetric_modulus_pa"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError("%s must be finite and positive, got %r" % (name, value))

    @property
    def shear_wave_speed_mps(self):
        return math.sqrt(self.shear_modulus_pa / self.density_kgm3)


@ti.func
def evaluate_neo_hookean_kirchhoff(F, mu, kappa):
    """Kirchhoff stress of W = mu/2 (I1b - 3) + kappa/2 (J - 1)^2 (modified invariants; the 3-D form of
    evaluate_neo_hookean_kirchhoff_planar)."""
    J = F.determinant()
    b = F @ F.transpose()
    jm = ti.pow(J, -2.0 / 3.0)
    identity = ti.Matrix.identity(ti.f32, 3)
    return mu * jm * (b - (b.trace() / 3.0) * identity) + kappa * (J - 1.0) * J * identity


@ti.func
def evaluate_neo_hookean_kirchhoff_planar(F2, mu, kappa):
    """Plane-strain Kirchhoff stress (yz block) of the neo-Hookean solid with modified invariants plus the volumetric
    stabiliser W = mu/2 (I1b - 3) + kappa/2 (J - 1)^2.  F = diag(1, F2)."""
    J = F2.determinant()
    b = F2 @ F2.transpose()
    i1 = b.trace() + 1.0
    jm = ti.pow(J, -2.0 / 3.0)
    identity = ti.Matrix.identity(ti.f32, 2)
    return mu * jm * (b - (i1 / 3.0) * identity) + kappa * (J - 1.0) * J * identity
