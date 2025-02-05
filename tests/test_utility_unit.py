import pathlib
import sys

import numpy as np
from openmm import unit
import pytest

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft import Q_, ureg  # noqa: E402
from mdcraft.utility.unit import strip_unit  # noqa: E402


def test_func_strip_unit():
    # Invalid output units
    with pytest.raises(TypeError):
        strip_unit(180.0, np.pi)

    # Strings without units
    assert strip_unit("9.80665") == (9.80665, None)
    assert strip_unit("9.80665", "m/s^2") == (9.80665, "m/s^2")
    assert strip_unit("9.80665", unit.meter / unit.second**2) == (
        9.80665,
        unit.meter / unit.second**2,
    )
    assert strip_unit("9.80665", ureg.meter / ureg.second**2) == (
        9.80665,
        ureg.meter / ureg.second**2,
    )

    # Strings with units
    assert strip_unit("9.80665 meter/second^2") == (9.80665, "meter / second ** 2")
    assert strip_unit("9.80665 meters/second^2", "ft/s^2") == (
        32.17404855643044,
        "foot / second ** 2",
    )
    assert strip_unit("9.80665 m/s^2", unit.foot / unit.second**2) == (
        32.17404855643044,
        unit.foot / unit.second**2,
    )
    assert strip_unit("9.80665 m/s^2", ureg.foot / ureg.second**2) == (
        32.17404855643044,
        ureg.foot / ureg.second**2,
    )

    # Dimensionless OpenMM quantities
    assert strip_unit(unit.Quantity(9.80665)) == (9.80665, None)
    assert strip_unit(unit.Quantity(9.80665), "m/s^2") == (
        9.80665,
        unit.meter / unit.second**2,
    )
    assert strip_unit(unit.Quantity(9.80665), unit.meter / unit.second**2) == (
        9.80665,
        unit.meter / unit.second**2,
    )
    assert strip_unit(unit.Quantity(9.80665), ureg.meter / ureg.second**2) == (
        9.80665,
        ureg.meter / ureg.second**2,
    )

    # OpenMM quantities
    assert strip_unit(9.80665 * unit.meter / unit.second**2) == (
        9.80665,
        unit.meter / unit.second**2,
    )
    assert strip_unit(9.80665 * unit.meter / unit.second**2, "ft/s^2") == (
        32.17404855643044,
        unit.foot / unit.second**2,
    )
    assert strip_unit(
        9.80665 * unit.meter / unit.second**2, unit.foot / unit.second**2
    ) == (32.17404855643044, unit.foot / unit.second**2)
    assert strip_unit(
        9.80665 * unit.meter / unit.second**2, ureg.foot / ureg.second**2
    ) == (32.17404855643044, ureg.foot / ureg.second**2)

    # Dimensionless Pint quantities
    assert strip_unit(Q_("9.80665")) == (9.80665, None)
    assert strip_unit(Q_("9.80665"), "m/s^2") == (
        9.80665,
        ureg.meter / ureg.second**2,
    )
    assert strip_unit(Q_("9.80665"), unit.meter / unit.second**2) == (
        9.80665,
        unit.meter / unit.second**2,
    )
    assert strip_unit(Q_("9.80665"), ureg.meter / ureg.second**2) == (
        9.80665,
        ureg.meter / ureg.second**2,
    )

    # Pint quantities
    assert strip_unit(Q_("9.80665 m/s^2")) == (9.80665, ureg.meter / ureg.second**2)
    assert strip_unit(Q_("9.80665 m/s^2"), "ft/s^2") == (
        32.17404855643044,
        ureg.foot / ureg.second**2,
    )
    assert strip_unit(Q_("9.80665 m/s^2"), unit.foot / unit.second**2) == (
        32.17404855643044,
        unit.foot / unit.second**2,
    )
    assert strip_unit(Q_("9.80665 m/s^2"), ureg.foot / ureg.second**2) == (
        32.17404855643044,
        ureg.foot / ureg.second**2,
    )

    # Non-quantity objects
    for quantity in [
        42,
        (unit.AVOGADRO_CONSTANT_NA, unit.BOLTZMANN_CONSTANT_kB),
        (Q_("3 nm"), Q_("4 nm"), Q_("5 nm")),
    ]:
        assert strip_unit(quantity) == (quantity, None)
