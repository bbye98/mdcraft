import pathlib
import sys

from openmm import unit

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft.openmm import unit as u # noqa: E402

def test_func_lj_scaling():

    temp = 300 * unit.kelvin

    # TEST CASE 1: Correct units for complex scaling factors
    scales = u.get_lj_scaling_factors(
        {"mass": 18.0153 * unit.amu,
         "length": 0.275 * unit.nanometer,
         "energy": (unit.BOLTZMANN_CONSTANT_kB * temp).in_units_of(unit.kilojoule)}
    )
    assert scales["molar_energy"].unit == unit.kilojoule_per_mole
    assert scales["velocity"].unit == unit.nanometer / unit.picosecond
    assert scales["electric_field"].unit \
           == unit.kilojoule_per_mole / (unit.nanometer * unit.elementary_charge)

    # TEST CASE 2: No default scaling factors
    scales = u.get_scaling_factors(
        {"mass": 18.0153 * unit.amu,
         "length": 0.275 * unit.nanometer,
         "energy": (unit.BOLTZMANN_CONSTANT_kB * temp).in_units_of(unit.kilojoule),
         "charge": 1 * unit.elementary_charge},
        {"surface_charge_density": (("charge", 1), ("length", -2))}
    )
    assert "time" not in scales

    # TEST CASE 3: Custom scaling factors
    assert scales["surface_charge_density"].unit == unit.elementary_charge / unit.nanometer ** 2