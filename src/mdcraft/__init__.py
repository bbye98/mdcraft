from importlib.util import find_spec
from typing import TypeVar

import numpy as np
from pint import UnitRegistry

__all__ = ["analysis", "io", "lib"]
__version__ = "2.0.0"

FOUND = {
    dep: find_spec(dep) is not None
    for dep in {"MDAnalysis", "netCDF4", "openmm"}
}

ureg = UnitRegistry()
Q_ = ureg.Quantity
U_ = ureg.Unit

int_t = TypeVar("int_t", bound=np.integer)
float_t = TypeVar("float_t", bound=np.floating)
complex_t = TypeVar("complex_t", bound=np.complexfloating)
