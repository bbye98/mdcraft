"""
Unit manipulation
=================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains helper functions for unit manipulation and reduction.
"""

from numbers import Number
from typing import Any, Union

import numpy as np

from .. import FOUND_OPENMM, Q_, ureg

if FOUND_OPENMM:
    from openmm import unit
    from ..openmm.unit import VACUUM_PERMITTIVITY


def get_scale_factors(
    bases: dict[str, Union["unit.Quantity", Q_]], other: dict[str, list] = None
) -> dict[str, Union["unit.Quantity", Q_]]:
    r"""
    Evaluates scale factors for reduced units.

    Parameters
    ----------
    bases : `dict`
        Fundamental quantities: molar mass (:math:`m`), length
        (:math:`\sigma`), and energy (:math:`\epsilon`).

        .. container::

           **Format**:

           .. code::

              {
                "mass": <openmm.unit.Quantity> | <pint.Quantity>,
                "length": <openmm.unit.Quantity> | <pint.Quantity>,
                "energy": <openmm.unit.Quantity> | <pint.Quantity>
              }

        **Reference units**: :math:`\mathrm{g/mol}`, :math:`\mathrm{nm}`,
        and :math:`\mathrm{kJ}`.

    other : `dict`, optional
        Other scale factors to compute. The key should be the name of
        the scale factor, and the value should contain `tuple`
        objects with the names of bases or default scale factors and
        their powers.

        **Example**:
        :code:`{"diffusivity": (("length", 2), ("time", -1))}`.

    Returns
    -------
    scales : `dict`
        Scale factors.
    """

    if other is not None:
        for name, params in other.items():
            factor = 1
            for base, power in params:
                factor *= bases[base] ** power
            bases[name] = factor

    return bases


def get_lj_scale_factors(
    bases: dict[str, Union["unit.Quantity", Q_]], other: dict[str, list] = None
) -> dict[str, Union["unit.Quantity", Q_]]:
    r"""
    Evaluates scale factors for Lennard-Jones reduced units.

    By default, the following scale factors are calculated:

    ===========================  ===================================================
    Quantity (`dict` key)        Expression
    ===========================  ===================================================
    :code:`"molar_energy"`       :math:`N_\mathrm{A}\epsilon`
    :code:`"time"`               :math:`\sqrt{m\sigma^2/\epsilon}`
    :code:`"velocity"`           :math:`\sigma/\tau`
    :code:`"force"`              :math:`\epsilon/\sigma`
    :code:`"temperature"`        :math:`\epsilon/k_\mathrm{B}T`
    :code:`"pressure"`           :math:`\epsilon/\sigma^3`
    :code:`"dynamic_viscosity"`  :math:`\epsilon\tau/\sigma^3`
    :code:`"charge"`             :math:`\sqrt{4\pi\varepsilon_0\sigma\epsilon}`
    :code:`"dipole"`             :math:`\sqrt{4\pi\varepsilon_0\sigma^3\epsilon}`
    :code:`"electric_field"`     :math:`\sqrt{\epsilon/(4\pi\varepsilon_0\sigma^3)}`
    :code:`"mass_density"`       :math:`m/\sigma^3`
    ===========================  ===================================================

    Parameters
    ----------
    bases : `dict`
        Fundamental quantities: molar mass (:math:`m`), length
        (:math:`\sigma`), and energy (:math:`\epsilon`).

        .. container::

           **Format**:

           .. code::

              {
                "mass": <openmm.unit.Quantity> | <pint.Quantity>,
                "length": <openmm.unit.Quantity> | <pint.Quantity>,
                "energy": <openmm.unit.Quantity> | <pint.Quantity>
              }

        **Reference units**: :math:`\mathrm{g/mol}`, :math:`\mathrm{nm}`,
        and :math:`\mathrm{kJ}`.

    other : `dict`, optional
        Other scale factors to compute. The key should be the name of
        the scale factor, and the value should contain `tuple`
        objects with the names of bases or default scale factors and
        their powers.

        **Example**:
        :code:`{"diffusivity": (("length", 2), ("time", -1))}`.

    Returns
    -------
    scales : `dict`
        Scale factors.
    """

    if bases["mass"].__module__ == "pint":
        avogadro_constant = ureg.avogadro_constant
        boltzmann_constant = ureg.boltzmann_constant
        bases["molar_energy"] = bases["energy"] * avogadro_constant
        bases["time"] = np.sqrt(
            bases["mass"] * bases["length"] ** 2 / bases["molar_energy"]
        ).to(ureg.picosecond)
        bases["charge"] = np.sqrt(
            4 * np.pi * ureg.vacuum_permittivity * bases["length"] * bases["energy"]
        ).to(ureg.elementary_charge)
    else:
        avogadro_constant = unit.AVOGADRO_CONSTANT_NA
        boltzmann_constant = unit.BOLTZMANN_CONSTANT_kB
        bases["molar_energy"] = bases["energy"] * avogadro_constant
        bases["time"] = (
            (bases["mass"] * bases["length"] ** 2 / bases["molar_energy"])
            .sqrt()
            .in_units_of(unit.picosecond)
        )
        bases["charge"] = (
            (4 * np.pi * VACUUM_PERMITTIVITY * bases["length"] * bases["energy"])
            .sqrt()
            .in_units_of(unit.elementary_charge)
        )

    # Define the default scale factors
    bases["velocity"] = bases["length"] / bases["time"]
    bases["force"] = bases["molar_energy"] / bases["length"]
    bases["temperature"] = bases["energy"] / boltzmann_constant
    bases["pressure"] = bases["energy"] / bases["length"] ** 3
    bases["dynamic_viscosity"] = bases["pressure"] * bases["time"]
    bases["dipole"] = bases["length"] * bases["charge"]
    bases["electric_field"] = bases["force"] / bases["charge"]
    bases["mass_density"] = bases["mass"] / (bases["length"] ** 3 * avogadro_constant)

    return get_scale_factors(bases, other)


def strip_unit(
    value: Union[Number, str, "unit.Quantity", Q_],
    unit_: Union[str, "unit.Unit", ureg.Unit] = None,
) -> tuple[Number, Union[None, "unit.Unit", ureg.Unit]]:
    """
    Strips the unit from an :obj:`openmm.unit.quantity.Quantity` or
    :obj:`pint.Quantity` object.

    Parameters
    ----------
    value : `numbers.Number`, `str`, `openmm.unit.Quantity`, or \
    `pint.Quantity`
        Physical quantity for which to get the magnitude of in the
        unit specified in `unit_`.

    unit_ : `str`, `openmm.unit.Unit`, or `pint.Unit`, optional
        Unit to convert to. If not specified, the original unit is used.

    Returns
    -------
    value : `numbers.Number`
        Magnitude of the physical quantity in the specified unit.

    unit_ : `openmm.unit.Unit` or `pint.Unit`
        Unit of the physical quantity.

    Examples
    --------
    For any quantity other than a :obj:`openmm.unit.quantity.Quantity`
    or :obj:`pint.Quantity` object, the raw quantity and user-specified
    unit are returned.

    >>> strip_unit(90.0, "deg")
    (90.0, 'deg')
    >>> strip_unit(90.0, ureg.degree)
    (90.0, <Unit('degree')>)

    If no target unit is specified, the magnitude and original unit of
    the quantity are returned.

    >>> strip_unit(1.380649e-23)
    (1.380649e-23, None)
    >>> strip_unit(1.380649e-23 * ureg.joule * ureg.kelvin ** -1)
    (1.380649e-23, <Unit('joule / kelvin')>)

    If a target unit using the same module as the quantity is specified,
    the quantity is first converted to the target unit, if necessary,
    before its magnitude and unit are returned.

    >>> g = 9.80665 * ureg.meter / ureg.second ** 2
    >>> strip_unit(g, "meter/second**2")
    (9.80665, <Unit('meter / second ** 2')>)
    >>> strip_unit(g, ureg.foot / ureg.second ** 2)
    (32.17404855643044, <Unit('foot / second ** 2')>)

    If a target unit using a different module than the quantity is
    specified, the quantity is converted to the specified target unit in
    the new module, if necessary, before its magnitude and unit are
    returned.

    >>> strip_unit(8.205736608095969e-05 * unit.meter ** 3 * unit.atmosphere
    ...            / (unit.kelvin * unit.mole),
    ...            ureg.joule / (ureg.kelvin * ureg.mole))
    (8.31446261815324, <Unit('joule / kelvin / mole')>)
    >>> strip_unit(8.205736608095969e-05 * ureg.meter ** 3 * ureg.atmosphere
    ...            / (ureg.kelvin * ureg.mole),
    ...            unit.joule / (unit.kelvin * unit.mole))
    (8.31446261815324, Unit({BaseUnit(..., name="kelvin", ...): -1.0, ...}))

    This function also supports strings directly:

    >>> strip_unit("299792458 m/s")
    (299792458.0, 'meter / second')
    >>> strip_unit("299792458 m/s", "ft/s")
    (983571056.4304463, 'foot / second')
    >>> strip_unit("299792458 m/s", unit.foot / unit.second)
    (983571056.4304463, Unit({BaseUnit(..., name="foot", ...): 1.0, ...}))
    >>> strip_unit("299792458 m/s", ureg.foot / ureg.second)
    (983571056.4304463, <Unit('foot / second')>)
    """

    def convert_openmm_to_pint(ou: "unit.Unit") -> ureg.Unit:
        """
        Converts an OpenMM unit to a Pint unit.

        Parameters
        ----------
        ou : `openmm.unit.Unit`
            OpenMM unit to convert.

        Returns
        -------
        pu : `pint.Unit`
            Pint unit equivalent to the OpenMM unit.
        """

        pu = ureg.Unit("")
        for u, p in ou.iter_base_or_scaled_units():
            pu *= ureg.Unit(u.name.replace(" ", "_")) ** p
        return pu

    if isinstance(value, Q_):

        # No target unit (unit_) specified; return Pint magnitude and
        # unit (unit__)
        if unit_ is None:
            unit__ = value.units
            value = value.magnitude
        else:

            # Convert OpenMM target unit (unit__ = unit_) to Pint unit
            # (unit_) and return unit__
            if getattr(unit_, "__module__", None) == "openmm.unit.unit":
                unit_, unit__ = convert_openmm_to_pint(unit_), unit_

            # Convert str or Pint target unit (unit_) to Pint unit
            # (unit__)
            else:
                unit__ = ureg.Unit(unit_)

            # Get magnitude of Pint quantity in Pint unit (unit_)
            value = value.m_as(unit_)

    elif getattr(value, "__module__", None) == "openmm.unit.quantity":
        swap = False

        # No target unit (unit_) specified; return OpenMM magnitude and
        # unit (unit__)
        if unit_ is None:
            unit_ = unit__ = value.unit

        # Store OpenMM target unit (unit_) in return value (unit__)
        elif getattr(unit_, "__module__", None) == "openmm.unit.unit":
            unit__ = unit_
        else:

            # Determine whether unit_ and unit__ need to be swapped at
            # the end; str target unit should give OpenMM unit, while
            # Pint target unit should give Pint unit
            swap = not isinstance(unit_, str)

            # Convert str or Pint target unit (unit_) to OpenMM unit
            # (unit__)
            unit_ = ureg.Unit(unit_)
            try:
                unit__ = unit.dimensionless
                for u, p in unit_._units.items():
                    unit__ *= getattr(unit, u) ** p
            except AttributeError:
                emsg = (
                    "strip_unit() relies on the pint module for "
                    "parsing units. At least one unit in 'unit' is "
                    "not defined the same way in openmm.unit and pint, "
                    "so the unit conversion cannot be performed. Try "
                    "specifying a openmm.unit.Quantity object for "
                    "'unit' instead."
                )
                raise ValueError(emsg)
        value = value.value_in_unit(unit__)
        if swap:
            unit_, unit__ = unit__, unit_

    elif isinstance(value, str):
        value = ureg(value)
        if isinstance(value, Number):
            return value, unit_
        else:
            if unit_ is not None:
                unit__ = unit_
                if getattr(unit_, "__module__", None) == "openmm.unit.unit":
                    unit_ = convert_openmm_to_pint(unit_)
                value = value.to(unit_)
                if isinstance(unit__, str):
                    unit__ = str(value.units)
                value = value.magnitude
            else:
                value, unit__ = value.magnitude, str(value.units)

    else:
        unit__ = unit_

    return value, unit__


def is_unitless(value: Any) -> bool:
    """
    Determines whether a value is unitless.

    Parameters
    ----------
    value : `Any`
        Value to check for unitlessness.

    Returns
    -------
    is_unitless : `bool`
        Whether the value is unitless.

    Examples
    --------
    >>> is_unitless(90.0)
    True
    >>> is_unitless("90 degrees")
    False
    >>> is_unitless(90.0 * ureg.degree)
    False
    >>> is_unitless(90.0 * unit.degree)
    False
    >>> is_unitless({"quantity": 90 * ureg.degree})
    True
    """

    return strip_unit(value)[1] is None
