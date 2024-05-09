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

def get_scaling_factors(
        bases: dict[str, unit.Quantity], other: dict[str, list] = {}
    ) -> dict[str, unit.Quantity]:

    """
    Evaluates scaling factors for reduced units.

    .. seealso::

       This is an alias function. For more information, see
       :func:`mdcraft.algorithm.unit.get_scaling_factors`.
    """

    return u.get_scaling_factors(bases, other)

def get_lj_scaling_factors(
        bases: dict[str, unit.Quantity], other: dict[str, list] = {}
    ) -> dict[str, unit.Quantity]:

    """
    Evaluates scaling factors for reduced Lennard-Jones units.

    .. seealso::

       This is an alias function. For more information, see
       :func:`mdcraft.algorithm.unit.get_lj_scaling_factors`.
    """

    return u.get_lj_scaling_factors(bases, other)