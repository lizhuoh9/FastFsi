"""Errors shared by every package.

SimulationHardStop is raised when a run must end because its state can no longer be trusted (a non-finite value, an
empty interpolation support, a guard limit).  It carries a machine-readable code and a JSON-serialisable witness that
says where and when the condition was found, so that a driver can write it to its output before stopping.
"""

HARD_STOP_CODES = (
    "nonfinite_point_state",               # a material point's position, velocity or deformation is not finite
    "invalid_deformation_gradient",        # a finite material-point Jacobian is zero or negative
    "one_sided_empty_support",             # a point's interpolation support has no valid face left
    "excess_mass_dt_limit",                # selected dt exceeds the current mass-overlap spring kick bound
    "pressure_chamber_column_without_surface",   # a chamber column has no surface point and no surface height
    "pressure_chamber_no_chamber_cells",   # the pressure chamber has no cell left
    "contact_nonfinite_state",             # the penalty contact found a non-finite distance or force
    "contact_setup_invalid",               # the penalty contact was given invalid point data
)


class SimulationHardStop(RuntimeError):
    """A hard stop with its machine-readable code and witness (a JSON-serialisable dict)."""

    def __init__(self, code, witness):
        if not isinstance(code, str) or not code:
            raise ValueError("a hard stop needs a non-empty string code, got %r" % (code,))
        super().__init__("%s: %s" % (code, witness))
        self.code = code
        self.witness = witness
