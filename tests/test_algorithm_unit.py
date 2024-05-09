import pathlib
import sys

import numpy as np
from openmm import unit
import pytest

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft import ureg
from mdcraft.algorithm.unit import get_lj_scaling_factors, strip_unit # noqa: E402

def test_func_get_lj_scaling_factors():

    # TEST CASE 1: Lennard-Jones scaling factors
    pint_factors = get_lj_scaling_factors({
        "mass": 39.948 * ureg.gram / ureg.mole,
        "energy": 3.9520829798737548e-25 * ureg.kilocalorie,
        "length": 3.4 * ureg.angstrom
    })
    openmm_factors = get_lj_scaling_factors({
        "mass": 39.948 * unit.gram / unit.mole,
        "energy": 0.238 * unit.kilocalorie_per_mole / unit.AVOGADRO_CONSTANT_NA,
        "length": 3.4 * unit.angstrom
    })
    for key in openmm_factors.keys():
        value, unit_ = strip_unit(openmm_factors[key])
        assert np.isclose(strip_unit(pint_factors[key], unit_)[0],
                          value)

def test_func_strip_unit():

    # TEST CASE 1: Strip unit from non-Quantity
    assert strip_unit(90.0, "deg") == (90.0, "deg")
    assert strip_unit(90.0, ureg.degree) == (90.0, ureg.degree)
    assert strip_unit(90.0, unit.degree) == (90.0, unit.degree)

    # TEST CASE 2: Strip unit from Quantity
    k_ = 1.380649e-23
    assert strip_unit(k_) == (k_, None)
    assert strip_unit(k_ * ureg.joule * ureg.kelvin ** -1) \
           == (k_, ureg.joule * ureg.kelvin ** -1)
    assert strip_unit(k_ * unit.joule * unit.kelvin ** -1) \
           == (k_, unit.joule * unit.kelvin ** -1)

    # TEST CASE 3: Strip unit from Quantity with compatible unit specified
    g_ = 32.17404855643044
    g = 9.80665 * ureg.meter / ureg.second ** 2
    assert strip_unit(g_, "foot/second**2") \
           == (g_, ureg.foot / ureg.second ** 2)
    assert strip_unit(g, ureg.foot / ureg.second ** 2) \
           == (g_, ureg.foot / ureg.second ** 2)
    g = 9.80665 * unit.meter / unit.second ** 2
    assert strip_unit(g, "foot/second**2") \
           == (g_, unit.foot / unit.second ** 2)
    assert strip_unit(g, unit.foot / unit.second ** 2) \
           == (g_, unit.foot / unit.second ** 2)

    # TEST CASE 4: Strip unit from Quantity with incompatible unit specified
    R_ = 8.31446261815324
    R__ = 8.205736608095969e-05
    assert strip_unit(
        R__ * ureg.meter ** 3 * ureg.atmosphere / (ureg.kelvin * ureg.mole),
        unit.joule / (unit.kelvin * unit.mole)
    ) == (R_, unit.joule / (unit.kelvin * unit.mole))
    assert strip_unit(
        R__ * unit.meter ** 3 * unit.atmosphere / (unit.kelvin * unit.mole),
        ureg.joule / (ureg.kelvin * ureg.mole)
    ) == (R_, ureg.joule / (ureg.kelvin * ureg.mole))

    # TEST CASE 5: Strip unit from Quantity with non-standard
    # incompatible unit specified
    with pytest.raises(ValueError):
        strip_unit(
            R_ * unit.joule / (unit.kelvin * unit.mole),
            ureg.meter ** 3 * ureg.atmosphere / (ureg.kelvin * ureg.mole)
        )