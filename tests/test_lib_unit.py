import pathlib
import sys

import numpy as np
from openmm import unit
import pytest

sys.path.insert(
    0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src"
)
from mdcraft import Q_, ureg
from mdcraft.lib.unit import strip_unit


class TestFunctionStripUnit:
    def test_invalid_output_units(self):
        with pytest.raises(TypeError):
            strip_unit(180.0, np.pi)

    def test_non_quantity_objects(self):
        for quantity in [
            42,
            (unit.AVOGADRO_CONSTANT_NA, unit.BOLTZMANN_CONSTANT_kB),
            (Q_("3 nm"), Q_("4 nm"), Q_("5 nm")),
        ]:
            assert strip_unit(quantity) == (quantity, None)

    def test_strings_without_units(self):
        assert strip_unit("9.80665") == (9.80665, None)
        assert strip_unit("9.80665", u := "m/s^2") == (9.80665, u)
        assert strip_unit("9.80665", u := unit.meter / unit.second**2) == (
            9.80665,
            u,
        )
        assert strip_unit("9.80665", u := ureg.meter / ureg.second**2) == (
            9.80665,
            u,
        )

    def test_strings_with_units(self):
        assert strip_unit("9.80665 m/s^2") == (
            9.80665,
            ureg.meter / ureg.second**2,
        )
        assert strip_unit("9.80665 m/s^2", "ft/s^2") == (
            32.17404855643044,
            ureg.foot / ureg.second**2,
        )
        assert strip_unit("9.80665 m/s^2", unit.foot / unit.second**2) == (
            32.17404855643044,
            unit.foot / unit.second**2,
        )
        assert strip_unit("9.80665 m/s^2", ureg.foot / ureg.second**2) == (
            32.17404855643044,
            ureg.foot / ureg.second**2,
        )

    def test_dimensionless_openmm_quantities(self):
        assert strip_unit(unit.Quantity(9.80665)) == (9.80665, None)
        assert strip_unit(unit.Quantity(9.80665), "m/s^2") == (
            9.80665,
            unit.meter / unit.second**2,
        )
        assert strip_unit(
            unit.Quantity(9.80665), unit.meter / unit.second**2
        ) == (
            9.80665,
            unit.meter / unit.second**2,
        )
        assert strip_unit(
            unit.Quantity(9.80665), ureg.meter / ureg.second**2
        ) == (
            9.80665,
            ureg.meter / ureg.second**2,
        )

    def test_openmm_quantities(self):
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

    def test_dimensionless_pint_quantities(self):
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

    def test_pint_quantities(self):
        assert strip_unit(Q_("9.80665 m/s^2")) == (
            9.80665,
            ureg.meter / ureg.second**2,
        )
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
