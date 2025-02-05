from importlib.util import find_spec

from pint import UnitRegistry

__version__ = "2.0.0"

FOUND_OPENMM = find_spec("openmm") is not None

ureg = UnitRegistry()
Q_ = ureg.Quantity
U_ = ureg.Unit
