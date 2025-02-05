from .. import ureg

INTERNAL_UNITS = {
    "charge": ureg.elementary_charge,
    "energy": ureg.kilojoule / ureg.mole,
    "length": ureg.nanometer,
    "mass": ureg.unified_atomic_mass_unit,
    "temperature": ureg.kelvin,
    "time": ureg.picosecond,
}
