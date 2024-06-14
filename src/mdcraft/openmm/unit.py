"""
OpenMM physical constants and unit conversions
==============================================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains physical constants and functions for unit
reduction.
"""

from openmm import unit

from ..algorithm import unit as u

VACUUM_PERMITTIVITY = 8.854187812813e-12 * unit.farad / unit.meter

def get_scale_factors(
        bases: dict[str, unit.Quantity], other: dict[str, list] = {}
    ) -> dict[str, unit.Quantity]:

    """
    Evaluates scaling factors for reduced units.

    .. seealso::

       This function is an alias for
       :func:`mdcraft.algorithm.unit.get_scale_factors`.
    """

    return u.get_scale_factors(bases, other)

def get_lj_scale_factors(
        bases: dict[str, unit.Quantity], other: dict[str, list] = {}
    ) -> dict[str, unit.Quantity]:

    """
    Evaluates scaling factors for reduced Lennard-Jones units.

    .. seealso::

       This function is an alias for
       :func:`mdcraft.algorithm.unit.get_lj_scale_factors`.
    """

    return u.get_lj_scale_factors(bases, other)