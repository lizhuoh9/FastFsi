"""Bits of the per-point `tag` field shared by the coupler, the pressure chamber and the contact diagnostics.

TAG_CHAMBER_SURFACE: the point lies on a surface that faces a pressure chamber; the chamber's cells start above the
                     highest such point of each grid column (boundary.pressure_chamber).
TAG_CONTACT_RIM:     the point's height above a reference plane is reported and its interpolation support is
                     monitored separately (contact diagnostics).
"""

TAG_CHAMBER_SURFACE = 1
TAG_CONTACT_RIM = 2
