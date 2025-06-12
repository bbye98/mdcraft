from .. import Q_
from ..utility.unit import strip_unit

class Integrator:
    def __init__(self, dt: float | Q_) -> None:
        self._dt_ps = strip_unit(dt, "ps")[0]