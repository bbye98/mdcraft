from typing import Any, Union

from .. import FOUND_OPENMM, Q_, U_

if FOUND_OPENMM:
    from openmm import unit


def _convert_openmm_unit_to_pint_unit(openmm_unit: "unit.Unit") -> U_:
    """
    Converts an OpenMM unit to a Pint unit.

    Parameters
    ----------
    openmm_unit : `openmm.unit.Unit`
        OpenMM unit to convert.

    Returns
    -------
    pint_unit : `pint.Unit`
        Pint unit equivalent to the OpenMM unit.
    """

    pint_unit = U_("")
    for base_unit, power in openmm_unit.iter_base_or_scaled_units():
        pint_unit *= U_(base_unit.name.replace(" ", "_")) ** power
    return pint_unit


def _convert_pint_unit_to_openmm_unit(pint_unit: U_) -> "unit.Unit":
    """
    Converts a Pint unit to an OpenMM unit.

    Parameters
    ----------
    pint_unit : `pint.Unit`
        Pint unit to convert.

    Returns
    -------
    openmm_unit : `openmm.unit.Unit`
        OpenMM unit equivalent to the Pint unit.
    """

    openmm_unit = unit.Unit({})
    for base_unit, power in pint_unit._units.items():
        openmm_unit *= getattr(unit, base_unit) ** power
    return openmm_unit


def strip_unit(
    quantity: Any,
    output_unit: Union[str, "unit.Unit", U_, None] = None,
    /,
) -> tuple[Any, Union[str, "unit.Unit", U_, None]]:
    """
    Separates the unit from a physical quantity after, optionally,
    converting it to a different unit.

    Parameters
    ----------
    quantity : any, positional-only
        Physical quantity from which to strip the unit.

    output_unit : `str`, `openmm.unit.Unit`, or `pint.Unit`, \
    positional-only, optional
        Unit to convert `quantity` to. If not specified, the original
        unit is returned.

    Returns
    -------
    value : any
        Value of the physical quantity.

    unit : `str`, `openmm.unit.Unit`, or `pint.Unit`
        Unit of the physical quantity.

    Examples
    --------
    This function supports a number of different input types. The
    following examples provide an exhaustive overview of the possible
    inputs and outputs. :code:`unit` refers to the `openmm.unit` module,
    :code:`ureg` refers to a `pint.UnitRegistry` instance, and :code:`Q_`
    refers to the `pint.Quantity` class attached to the :code:`ureg`
    instance.

    Strings containing quantities without units are typecasted to
    floating-point numbers and converted, if necessary, before being
    returned with the unaltered user-specified unit:

    >>> strip_unit("9.80665")
    (9.80665, None)
    >>> strip_unit("9.80665", "m/s^2")
    (9.80665, 'm/s^2')
    >>> strip_unit("9.80665", unit.meter / unit.second**2)
    (9.80665, Unit({BaseUnit(name="meter", ...): 1.0, BaseUnit(name="second", ...): -2.0}))
    >>> strip_unit("9.80665", ureg.meter / ureg.second**2)
    (9.80665, <Unit('meter / second ** 2')>)

    Strings containing quantities with units have their numeric parts
    typecasted to floating-point numbers and returned with the
    user-specified unit or a formatted string containing the original
    unit:

    >>> strip_unit("9.80665 meter/second^2")
    (9.80665, 'meter / second ** 2')
    >>> strip_unit("9.80665 meters/second^2", "ft/s^2")
    (32.17404855643044, 'ft/s^2')
    >>> strip_unit("9.80665 m/s^2", unit.foot / unit.second**2)
    (32.17404855643044, Unit({BaseUnit(name="foot", ...): 1.0, BaseUnit(name="second", ...): -2.0}))
    >>> strip_unit("9.80665 m/s^2", ureg.foot / ureg.second**2)
    (32.17404855643044, <Unit('foot / second ** 2')>)

    Dimensionless OpenMM quantities have their underlying values
    returned with `None` or an OpenMM or Pint unit if a unit was
    specified by the user:

    >>> strip_unit(unit.Quantity(9.80665))
    (9.80665, None)
    >>> strip_unit(unit.Quantity(9.80665), "m/s^2")
    (9.80665, Unit({BaseUnit(name="meter", ...): 1.0, BaseUnit(name="second", ...): -2.0}))
    >>> strip_unit(unit.Quantity(9.80665), unit.meter / unit.second**2)
    (9.80665, Unit({BaseUnit(name="meter", ...): 1.0, BaseUnit(name="second", ...): -2.0}))
    >>> strip_unit(unit.Quantity(9.80665), ureg.meter / ureg.second**2)
    (9.80665, <Unit('meter / second ** 2')>)

    OpenMM quantities have their underlying values returned with the
    original OpenMM unit, a user-specified unit if it is an OpenMM or
    Pint unit, or an equivalent OpenMM unit if the user-specified unit
    is a string:

    >>> strip_unit(9.80665 * unit.meter / unit.second**2)
    (9.80665, Unit({BaseUnit(name="meter", ...): 1.0, BaseUnit(name="second", ...): -2.0}))
    >>> strip_unit(9.80665 * unit.meter / unit.second**2, "ft/s^2")
    (32.17404855643044, Unit({BaseUnit(name="foot", ...): 1.0, BaseUnit(name="second", ...): -2.0}))
    >>> strip_unit(9.80665 * unit.meter / unit.second**2, unit.foot / unit.second**2)
    (32.17404855643044, Unit({BaseUnit(name="foot", ...): 1.0, BaseUnit(name="second", ...): -2.0}))
    >>> strip_unit(9.80665 * unit.meter / unit.second**2, ureg.foot / ureg.second**2)
    (32.17404855643044, <Unit('foot / second ** 2')>)

    Similarly, dimensionless Pint quantities have their underlying
    values returned with `None` or a Pint unit if a unit was specified
    by the user:

    >>> strip_unit(Q_(9.80665))
    (9.80665, None)
    >>> strip_unit(Q_("9.80665"), "m/s^2")
    (9.80665, 'meter / second ** 2')
    >>> strip_unit(Q_("9.80665"), unit.meter / unit.second**2)
    (9.80665, Unit({BaseUnit(name="meter", ...): 1.0, BaseUnit(name="second", ...): -2.0}))
    >>> strip_unit(Q_("9.80665"), ureg.meter / ureg.second**2)
    (9.80665, <Unit('meter / second ** 2')>)

    Finally, Pint quantities have their underlying values returned with
    the original Pint unit, a user-specified unit if it is an OpenMM or
    Pint unit, or an equivalent Pint unit if the user-specified unit is
    a string:

    >>> strip_unit(Q_("9.80665 m/s^2"))
    (9.80665, <Unit('meter / second ** 2')>)
    >>> strip_unit(Q_("9.80665 m/s^2"), "ft/s^2")
    (32.17404855643044, <Unit('foot / second ** 2')>)
    >>> strip_unit(Q_("9.80665 m/s^2"), unit.foot / unit.second**2)
    (32.17404855643044, Unit({BaseUnit(name="foot", ...): 1.0, BaseUnit(name="second", ...): -2.0}))
    >>> strip_unit(Q_("9.80665 m/s^2"), ureg.foot / ureg.second**2)
    (32.17404855643044, <Unit('foot / second ** 2')>)
    """

    if not (
        output_unit is None
        or isinstance(output_unit, (str, U_))
        or getattr(output_unit, "__module__", None) == "openmm.unit.unit"
    ):
        raise TypeError(
            "`output_unit` must be `None`, a `str`, "
            "an `openmm.unit.Unit`, or a `pint.Unit`."
        )

    if isinstance(quantity, str):
        quantity = Q_(quantity)
        if quantity.unitless:
            actual_unit = output_unit
        elif output_unit is None:
            actual_unit = str(quantity.units)
        else:
            actual_unit = conversion_unit = output_unit
            if getattr(output_unit, "__module__", None) == "openmm.unit.unit":
                conversion_unit = _convert_openmm_unit_to_pint_unit(output_unit)
            quantity = quantity.to(conversion_unit)
            if isinstance(output_unit, str):
                actual_unit = str(quantity.units)
        value = quantity.magnitude
    elif getattr(quantity, "__module__", None) == "openmm.unit.quantity":
        if quantity.unit == unit.dimensionless:
            actual_unit = (
                _convert_pint_unit_to_openmm_unit(U_(output_unit))
                if isinstance(output_unit, str)
                else output_unit
            )
            conversion_unit = quantity.unit
        elif isinstance(output_unit, U_):
            actual_unit = output_unit
            conversion_unit = _convert_pint_unit_to_openmm_unit(output_unit)
        else:
            actual_unit = conversion_unit = (
                _convert_pint_unit_to_openmm_unit(U_(output_unit))
                if isinstance(output_unit, str)
                else (
                    output_unit if isinstance(output_unit, unit.Unit) else quantity.unit
                )
            )
        value = quantity.value_in_unit(conversion_unit)
    elif isinstance(quantity, Q_):
        if quantity.unitless:
            actual_unit = (
                U_(output_unit) if isinstance(output_unit, str) else output_unit
            )
            conversion_unit = quantity.units
        elif getattr(output_unit, "__module__", None) == "openmm.unit.unit":
            actual_unit = output_unit
            conversion_unit = _convert_openmm_unit_to_pint_unit(output_unit)
        else:
            actual_unit = conversion_unit = (
                U_(output_unit)
                if isinstance(output_unit, str)
                else output_unit if isinstance(output_unit, U_) else quantity.units
            )
        value = quantity.m_as(conversion_unit)
    else:
        value = quantity
        actual_unit = output_unit
    return value, actual_unit
