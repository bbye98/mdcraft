from importlib.util import find_spec

from pint import UnitRegistry

__all__ = ["algorithm", "analysis", "core", "io", "utility"]
__version__ = "2.0.0"

FOUND = {
    dep: find_spec(dep) is not None
    for dep in {"MDAnalysis", "netCDF4", "openmm"}
}

ureg = UnitRegistry()
Q_ = ureg.Quantity
U_ = ureg.Unit
