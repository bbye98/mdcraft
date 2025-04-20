from importlib.util import find_spec

from pint import UnitRegistry

__version__ = "2.0.0"

FOUND = {dep: find_spec(dep) is not None for dep in {"MDAnalysis", "openmm"}}

ureg = UnitRegistry()
Q_ = ureg.Quantity
U_ = ureg.Unit
