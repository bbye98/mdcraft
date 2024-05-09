"""
MDCraft
=======
"""

from importlib.util import find_spec

from pint import Quantity, UnitRegistry

VERSION = "1.0.0"
FOUND_OPENMM = find_spec("openmm") is not None

Q_ = Quantity
ureg = UnitRegistry(auto_reduce_dimensions=True)

from . import algorithm, analysis, fit, lammps, plot # noqa: E402

__all__ = ["FOUND_OPENMM", "VERSION", "algorithm", "analysis", "fit", "lammps",
           "plot"]
if FOUND_OPENMM:
    __all__.append("openmm")