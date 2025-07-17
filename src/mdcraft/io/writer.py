import warnings

import numpy as np

from .. import __version__, FOUND, Q_, ureg
from ..lib.unit import strip_unit
from .base import BaseTrajectoryWriter

if FOUND["netCDF4"]:
    import netCDF4 as nc


class NetCDFWriter(BaseTrajectoryWriter):
    """
    AMBER NetCDF trajectory/restart file writer.

    .. seealso::

       For more information on the AMBER NetCDF file format, see the
       `AMBER NetCDF Trajectory/Restart Convention
       <https://ambermd.org/netcdf/nctraj.xhtml>`_.

    Parameters
    ----------
    filename : `str` or `pathlib.Path`, positional-only
        Filename or path to the NetCDF file.

    mode : `str`, optional, default: ``"w"``
        NetCDF file access mode.

    restart : `bool`, default: :code:`False`
        Specifies whether the NetCDF file is a trajectory
        (:code:`False`) or restart file (:code:`True`).

    Examples
    --------
    """

    _EXTENSIONS = {".nc", ".ncdf"}
    _FORMAT = "NETCDF"
    _REMD_DIMENSIONS = {
        "temperature": 1,
        "partial": 2,
        "hamiltonian": 3,
        "ph": 4,
        "redox": 5,
    }
    _units = {
        "charge": ureg.elementary_charge,
        "energy": ureg.kilocalorie / ureg.mole,
        "length": ureg.angstrom,
        "mass": ureg.gram / ureg.mole,
        "temperature": ureg.kelvin,
        "time": ureg.picosecond,
    }

    def __init__(
        self, filename: str, mode: str = "w", *, restart: bool = False, **kwargs
    ) -> None:
        if not FOUND["netCDF4"]:
            raise RuntimeError(
                "mdcraft.io.writer.NetCDFWriter requires the 'netcdf4' "
                "package, but it could not be found or imported."
            )

        super().__init__(filename)
        self._file = nc.Dataset(filename, mode, format="NETCDF3_64BIT_OFFSET")
        self._file.set_always_mask(False)
        self._initialized = hasattr(self._file, "Conventions")
        self._restart = (
            self._file.Conventions == "AMBERRESTART"
            if self._initialized
            else restart
        )

    def _write_header(
        self,
        N: int,
        cell: bool,
        time: bool,
        positions: bool,
        velocities: bool,
        forces: bool,
        *,
        title: str | None = None,
        remd_dimensions: list[int | str] | None = None,
        reduced: bool = False,
    ) -> None:
        # Set global attributes
        self._file.Conventions = "AMBER"
        if self._restart:
            self._file.Conventions += "RESTART"
        self._file.ConventionVersion = "1.0"
        self._file.program = "MDCraft"
        self._file.programVersion = __version__
        if title is not None:
            self._file.title = title

        # Set dimensions
        self._file.createDimension("frame", self._restart or None)
        self._file.createDimension("spatial", 3)
        self._file.createDimension("atom", N)

        # Create variables
        if self._restart:
            fp_t = "d"
            shape = ("atom", "spatial")
        else:
            fp_t = "f"
            shape = ("frame", "atom", "spatial")

        if reduced:
            if time:
                self._file.createVariable(
                    "time", fp_t, shape[-3::-1]
                ).units = ""
            if positions:
                self._file.createVariable("coordinates", fp_t, shape).units = ""
            if velocities:
                self._file.createVariable("velocities", fp_t, shape).units = ""
            if forces:
                self._file.createVariable("forces", fp_t, shape).units = ""
        else:
            if time:
                self._file.createVariable(
                    "time", fp_t, shape[-3::-1]
                ).units = "picosecond"
            if positions:
                self._file.createVariable(
                    "coordinates", fp_t, shape
                ).units = "angstrom"
            if velocities:
                self._file.createVariable(
                    "velocities", fp_t, shape
                ).units = "angstrom/picosecond"
            if forces:
                self._file.createVariable(
                    "forces", fp_t, shape
                ).units = "kilocalorie/mole/angstrom"

        # Set dimensions and create variables for unit cell information,
        # if available
        if cell:
            self._file.createDimension("cell_spatial", 3)
            self._file.createDimension("cell_angular", 3)
            self._file.createDimension("label", 5)

        # Set dimensinos and create variables for replica exchange
        # molecular dynamics (REMD) information, if available
        if remd_dimensions is not None and len(remd_dimensions) > 0:
            for i, remd_dim in enumerate(remd_dimensions):
                if (
                    isinstance(remd_dim, str)
                    and remd_dim in self._REMD_DIMENSIONS
                ):
                    remd_dimensions[i] = self._REMD_DIMENSIONS[remd_dim]
                elif not isinstance(remd_dim, int) or not 0 < remd_dim <= len(
                    self._REMD_DIMENSIONS
                ):
                    raise ValueError(
                        f"Invalid REMD dimension '{remd_dim}'. Valid values: "
                        "'" + "', '".join(self._REMD_DIMENSIONS.keys()) + "'."
                    )

        self._initialized = True
