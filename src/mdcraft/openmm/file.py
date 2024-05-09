"""
Custom OpenMM topology and trajectory file readers and writers
==============================================================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module provides custom topology and trajectory file readers and
writers for OpenMM.
"""

import platform
from typing import Any, Union
import warnings

import netCDF4 as nc
import numpy as np
import openmm
from openmm import app, unit

from .. import VERSION

class NetCDFFile():

    """
    Interface for reading and writing AMBER NetCDF trajectory and
    restart files.

    Parameters
    ----------
    file : `str` or `netcdf4.Dataset`
        NetCDF file. If `file` is a filename and does not have the
        :code:`.nc` or :code:`.ncdf` extension, :code:`.nc` will
        automatically be appended.

    mode : `str`
        NetCDF file access mode.

    restart : `bool`, default: :code:`False`
        Specifies whether the NetCDF file is a trajectory or restart file.

    **kwargs
        Keyword arguments to be passed to :code:`netCDF4.Dataset`.
    """

    def __init__(
            self, file: Union[str, nc.Dataset], mode: str,
            restart: bool = False, **kwargs):

        if isinstance(file, str):
            if not file.endswith((".nc", ".ncdf")):
                file += ".nc"
            self._nc = nc.Dataset(file, mode=mode,
                                  format="NETCDF3_64BIT_OFFSET", **kwargs)
        else:
            self._nc = file
        self._nc.set_always_mask(False)

        if mode == "r":
            self._frame = self._nc.variables["time"].shape[0]
            self._restart = self._nc.Conventions == "AMBERRESTART"
        else:
            self._frame = 0
            self._restart = restart

    def get_dimensions(
            self, frames: Union[int, list[int], slice] = None,
            units: bool = True
        ) -> Union[tuple[np.ndarray[float], np.ndarray[float]],
                   tuple[unit.Quantity, unit.Quantity]]:

        """
        Get the simulation box dimensions.

        Parameters
        ----------
        frames : `int`, `list`, or `slice`, optional
            Frame indices. If :code:`None`, the dimensions across all
            frames are returned.

        units : `bool`, default: :code:`True`
            Determines whether the dimensions are returned with units.

        Returns
        -------
        cell_lengths : `numpy.ndarray`, optional
            Simulation box dimensions.

            **Reference unit**: :math:`\\mathrm{Å}`.

        cell_angles : `numpy.ndarray`, optional
            Angles that define the shape of the simulation box.

            **Reference unit**: :math:`^\\circ`.
        """

        cell_lengths = (self._nc.variables["cell_lengths"][:] if frames is None
                        else self._nc.variables["cell_lengths"][frames])
        cell_angles = (self._nc.variables["cell_angles"][:] if frames is None
                       else self._nc.variables["cell_angles"][frames])
        if units:
            cell_lengths *= unit.angstrom
            cell_angles *= unit.degree
        return cell_lengths, cell_angles

    def get_num_frames(self) -> int:

        """
        Get the number of frames.

        Returns
        -------
        num_frames : `int`
            Number of frames.
        """

        return self._nc.dimensions["frame"].size

    def get_num_atoms(self) -> int:

        """
        Get the number of atoms.

        Returns
        -------
        num_atoms : `int`
            Number of atoms.
        """

        return self._nc.dimensions["atom"].size

    def get_times(
            self, frames: Union[int, list[int], slice] = None,
            units: bool = True) -> Union[np.ndarray[float], unit.Quantity]:

        """
        Get simulation times.

        Parameters
        ----------
        frames : `int`, `list`, or `slice`, optional
            Frame indices. If :code:`None`, the times across all
            frames are returned.

        units : `bool`, default: :code:`True`
            Determines whether the times are returned with units.

        Returns
        -------
        times : `numpy.ndarray` or `openmm.unit.Quantity`
            Simulation times.

            **Reference unit**: :math:`\\mathrm{ps}`.
        """

        times = (self._nc.variables["time"][:] if frames is None
                 else self._nc.variables["time"][frames])
        if units:
            times *= unit.picosecond
        return times

    def get_positions(
            self, frames: Union[int, list[int], slice] = None,
            units: bool = True) -> Union[np.ndarray[float], unit.Quantity]:

        """
        Get the atom positions.

        Parameters
        ----------
        frames : `int`, `list`, or `slice`, optional
            Frame indices. If :code:`None`, the positions across all
            frames are returned.

        units : `bool`, default: :code:`True`
            Determines whether the positions are returned with units.

        Returns
        -------
        positions : `numpy.ndarray` or `openmm.unit.Quantity`
            Atom positions.

            **Reference unit**: :math:`\\mathrm{Å}`.
        """

        positions = (self._nc.variables["coordinates"][:] if frames is None
                     else self._nc.variables["coordinates"][frames])
        if units:
            positions *= unit.angstrom
        return positions

    def get_velocities(
            self, frames: Union[int, list[int], slice] = None,
            units: bool = True) -> Union[np.ndarray[float], unit.Quantity]:

        """
        Get atom velocities.

        Parameters
        ----------
        frames : `int`, `list`, or `slice`, optional
            Frame indices. If :code:`None`, the velocities across all
            frames are returned.

        units : `bool`, default: :code:`True`
            Determines whether the velocities are returned with units.

        Returns
        -------
        velocities : `numpy.ndarray` or `openmm.unit.Quantity`
            Atom velocities. If the NetCDF file does not contain
            this information, :code:`None` is returned.

            **Reference unit**: :math:`\\mathrm{Å/ps}`.
        """

        if "velocities" not in self._nc.variables:
            wmsg = ("The NetCDF file does not contain information about "
                    "the atom velocities.")
            warnings.warn(wmsg)
            return None

        velocities = (self._nc.variables["velocities"][:] if frames is None
                      else self._nc.variables["velocities"][frames])
        if units:
            velocities *= unit.angstrom / unit.picosecond
        return velocities

    def get_forces(
            self, frames: Union[int, list[int], slice] = None,
            units: bool = True) -> Union[np.ndarray[float], unit.Quantity]:

        """
        Get the forces acting on the atoms.

        Parameters
        ----------
        frames : `int`, `list`, or `slice`, optional
            Frame indices. If :code:`None`, the forces across all frames
            are returned.

        units : `bool`, default: :code:`True`
            Determines whether the forces are returned with units.

        Returns
        -------
        forces : `numpy.ndarray` or `openmm.unit.Quantity`
            Forces acting on the atoms. If the NetCDF file does not
            contain this information, :code:`None` is returned.

            **Reference unit**: :math:`\\mathrm{Å/ps}`.
        """

        if "forces" not in self._nc.variables:
            wmsg = ("The NetCDF file does not contain information about "
                    "the forces acting on the atoms.")
            warnings.warn(wmsg)
            return None

        forces = (self._nc.variables["forces"][:] if frames is None
                  else self._nc.variables["forces"][frames])
        if units:
            forces *= unit.kilocalorie_per_mole / unit.angstrom
        return forces

    def write_header(
            self: Any, N: int, cell: bool, velocities: bool, forces: bool,
            restart: bool = False, *, remd: str = None, temp0: float = None,
            remd_dimtype: np.ndarray[int] = None,
            remd_indices: np.ndarray[int] = None, remd_repidx: int = -1,
            remd_crdidx: int = -1, remd_values: np.ndarray[float] = None
        ) -> "NetCDFFile":

        """
        Initialize a NetCDF file according to `AMBER NetCDF
        Trajectory/Restart Convention Version 1.0, Revision C
        <https://ambermd.org/netcdf/nctraj.xhtml>`_.

        .. note::

           This function can be used as either a static or instance
           method.

        Parameters
        ----------
        self : `str`, `netcdf4.Dataset`, or `mdcraft.openmm.file.NetCDFFile`
            If :meth:`write_header` is called as a static method, you
            must provide a filename or a NetCDF file object. Otherwise,
            the NetCDF file embedded in the current instance is used.

        N : `int`
            Number of atoms.

        cell : `bool`
            Specifies whether simulation box length and angle
            information is available.

        velocities : `bool`
            Specifies whether atom velocities should be written.

        forces : `bool`
            Specifies whether forces exerted on atoms should be
            written.

        restart : `bool`, default: :code:`False`
            Specifies whether the NetCDF file is a trajectory or restart
            file.

        remd : `str`, keyword-only, optional
            Specifies whether information about a replica exchange
            molecular dynamics (REMD) simulation is written.

            .. container::

               **Valid values**:

               * :code:`"temp"` for regular REMD.
               * :code:`"multi"` for multi-dimensional REMD.

        temp0 : `float`, keyword-only, optional
            Temperature that the thermostat is set to maintain for a
            REMD restart file only.

            **Reference unit**: :math:`\\mathrm{K}`.

        remd_dimtype : array-like, keyword-only, optional
            Array specifying the exchange type(s) for the REMD
            dimension(s). Required for a multi-dimensional REMD restart
            file.

        remd_indices : array-like, keyword-only, optional
            Array specifying the position in all dimensions that each
            frame is in. Required for a multi-dimensional REMD restart
            file.

        remd_repidx : `int`, keyword-only, optional
            Overall index of the frame in replica space.

        remd_crdidx : `int`, keyword-only, optional
            Overall index of the frame in coordinate space.

        remd_values : array-like, keyword-only, optional
            Replica value the specified replica dimension has for that
            given frame. Required for a multi-dimensional REMD restart
            file.

        Returns
        -------
        netcdf_file : `mdcraft.openmm.file.NetCDFFile`
            NetCDF file object. Only returned when this function is used
            as a static method.
        """

        # Create NetCDF object if it doesn't already exist
        if not isinstance(self, NetCDFFile):
            self = NetCDFFile(self, "w", restart=restart)

        self._nc.Conventions = "AMBER"
        if self._restart:
            self._nc.Conventions += "RESTART"
        self._nc.ConventionVersion = "1.0"
        self._nc.program = "MDCraft"
        self._nc.programVersion = VERSION
        self._nc.title = (f"OpenMM {openmm.Platform.getOpenMMVersion()} / "
                          f"{platform.node()}")

        if self._restart:
            self._nc.createDimension("frame", 1)
        else:
            self._nc.createDimension("frame", None)

        if remd == "multi": # pragma: no cover
            self._nc.createDimension("remd_dimension", len(remd_dimtype))
        self._nc.createDimension("spatial", 3)
        self._nc.createDimension("atom", N)

        if self._restart:
            self._nc.createVariable("coordinates", "d", ("atom", "spatial"))
        else:
            self._nc.createVariable("coordinates", "f",
                                    ("frame", "atom", "spatial"))
        self._nc.variables["coordinates"].units = "angstrom"

        self._nc.createVariable("time", "d", ("frame",))
        self._nc.variables["time"].units = "picosecond"

        if cell:
            self._nc.createDimension("cell_spatial", 3)
            self._nc.createDimension("cell_angular", 3)
            self._nc.createDimension("label", 5)
            self._nc.createVariable("spatial", "c", ("spatial",))
            self._nc.variables["spatial"][:] = list("xyz")
            self._nc.createVariable("cell_spatial", "c", ("cell_spatial",))
            self._nc.variables["cell_spatial"][:] = list("abc")
            self._nc.createVariable("cell_angular", "c",
                                    ("cell_angular", "label"))
            self._nc.variables["cell_angular"][:] = [list("alpha"),
                                                     list("beta "),
                                                     list("gamma")]

            if self._restart:
                self._nc.createVariable("cell_lengths", "d", ("cell_spatial",))
                self._nc.createVariable("cell_angles", "d", ("cell_angular",))
            else:
                self._nc.createVariable("cell_lengths", "f",
                                        ("frame", "cell_spatial"))
                self._nc.createVariable("cell_angles", "f",
                                        ("frame", "cell_angular"))
            self._nc.variables["cell_lengths"].units = "angstrom"
            self._nc.variables["cell_angles"].units = "degree"

        if velocities:
            if self._restart:
                self._nc.createVariable("velocities", "d", ("atom", "spatial"))
            else:
                self._nc.createVariable("velocities", "f",
                                        ("frame", "atom", "spatial"))
            self._nc.variables["velocities"].units = "angstrom/picosecond"
            self._nc.variables["velocities"].scale_factor = 20.455

        if forces:
            if self._restart:
                self._nc.createVariable("forces", "d", ("atom", "spatial"))
            else:
                self._nc.createVariable("forces", "f",
                                        ("frame", "atom", "spatial"))
            self._nc.variables["forces"].units = "kilocalorie/mole/angstrom"

        if remd is not None: # pragma: no cover
            if remd == "temp":
                self._nc.createVariable("temp0", "d", ("frame",))
                if self._restart:
                    if temp0 is None:
                        emsg = ("Temperature must be provided for a REMD "
                                "restart file.")
                        raise ValueError(emsg)
                    self._nc.variables["temp0"][0] = temp0
                self._nc.variables["temp0"].units = "kelvin"

            elif remd == "multi":
                self._nc.createVariable("remd_dimtype", "i",
                                        ("remd_dimension",))
                self._nc.createVariable("remd_repidx", "i", ("frame",))
                self._nc.createVariable("remd_crdidx", "i", ("frame",))
                if self._restart:
                    if remd_dimtype is None:
                        emsg = ("Dimension types must be provided for a "
                                "multi-dimensional REMD restart file.")
                        raise ValueError(emsg)
                    self._nc.variables["remd_dimtype"] = remd_dimtype

                    self._nc.createVariable("remd_indices", "i",
                                            ("remd_dimension",))
                    if remd_indices is None:
                        emsg = ("Dimension indices must be provided for a "
                                "multi-dimensional REMD restart file.")
                        raise ValueError(emsg)
                    self._nc.variables["remd_indices"] = remd_indices

                    self._nc.variables["remd_repidx"][0] = remd_repidx
                    self._nc.variables["remd_crdidx"][0] = remd_crdidx

                    self._nc.createVariable("remd_values", "d",
                                            ("remd_dimension",))
                    if remd_values is None:
                        emsg = ("Replica values must be provided for a "
                                "multi-dimensional REMD restart file.")
                        raise ValueError(emsg)
                    self._nc.variables["remd_values"][:] = remd_values

                else:
                    self._nc.createVariable("remd_indices", "i",
                                            ("frame", "remd_dimension"))
                    self._nc.createVariable("remd_values", "d",
                                            ("frame", "remd_dimension"))

        return self

    def write_file(self: Any, state: openmm.State) -> "NetCDFFile":

        """
        Write a single simulation state to a restart NetCDF file.

        .. note::

           This function can be used as either a static or instance
           method.

        Parameters
        ----------
        self : `str`, `netcdf4.Dataset`, or `mdcraft.openmm.file.NetCDFFile`
            If :meth:`write_file` is called as a static method, you must
            provide a filename or a NetCDF file object. Otherwise, the
            NetCDF file embedded in the current instance is used.

        state : `openmm.State`
            OpenMM simulation state from which to retrieve cell
            dimensions and atom positions, velocities, and forces.

        Returns
        -------
        netcdf_file : `mdcraft.openmm.file.NetCDFFile`
            NetCDF file object. Only returned when this function is used
            as a static method.
        """

        # Collect all available data in the state
        data = {}
        pbv = state.getPeriodicBoxVectors()
        if pbv is not None:
            (a, b, c, alpha, beta, gamma) = \
                app.internal.unitcell.computeLengthsAndAngles(pbv)
            data["cell_lengths"] = 10 * np.array((a, b, c))
            data["cell_angles"] = 180 * np.array((alpha, beta, gamma)) / np.pi
        data["coordinates"] = (state.getPositions(asNumpy=True)
                               .value_in_unit(unit.angstrom))
        try:
            data["velocities"] \
                = state.getVelocities(asNumpy=True).value_in_unit(
                    unit.angstrom / unit.picosecond
                )
        except openmm.OpenMMException: # pragma: no cover
            pass
        try:
            data["forces"] \
                = state.getForces(asNumpy=True).value_in_unit(
                    unit.kilocalorie_per_mole / unit.angstrom
                )
        except openmm.OpenMMException: # pragma: no cover
            pass

        # Create NetCDF file or object if it doesn't already exist
        if not isinstance(self, NetCDFFile):
            self = NetCDFFile(self, "w", restart=True)
        if not hasattr(self._nc, "Conventions"):
            self.write_header(data["coordinates"].shape[0],
                              "cell_lengths" in data or "cell_angles" in data,
                              "velocities" in data, "forces" in data)
            self._nc.set_always_mask(False)
        elif self._nc.Conventions != "AMBERRESTART":
            raise ValueError("The NetCDF file must be a restart file.")

        # Write data to NetCDF file
        for k, v in data.items():
            self._nc.variables[k][:] = v
        self._nc.sync()

        return self

    def write_model(
            self: Any, time: Union[float, np.ndarray[float]],
            coordinates: np.ndarray[float],
            velocities: np.ndarray[float] = None,
            forces: np.ndarray[float] = None,
            cell_lengths: np.ndarray[float] = None,
            cell_angles: np.ndarray[float] = None, *, restart: bool = False
        ) -> "NetCDFFile":

        """
        Write simulation state(s) to a NetCDF file.

        .. note::

           This function can be used as either a static or instance
           method.

        Parameters
        ----------
        self : `str`, `netcdf4.Dataset`, or `mdcraft.openmm.file.NetCDFFile`
            If :meth:`write_model` is called as a static method, you must
            provide a filename or a NetCDF file object. Otherwise, the
            NetCDF file embedded in the current instance is used.

        time : `float` or `numpy.ndarray`
            Time(s). The dimensionality determines whether a single or
            multiple frames are written.

            **Reference unit**: :math:`\\mathrm{ps}`.

        coordinates : `numpy.ndarray`
            Coordinates of :math:`N` atoms over :math:`N_t` frames. The
            dimensionality depends on whether a single or multiple
            frames are to be written and must be compatible with that
            for `time`.

            **Shape**: :math:`(N,\\,3)` or :math:`(N_t,\\,N,\\,3)`.

            **Reference unit**: :math:`\\mathrm{Å}`.

        velocities : `numpy.ndarray`, optional
            Velocities of :math:`N` atoms over :math:`N_t` frames. The
            dimensionality depends on whether a single or multiple
            frames are to be written and must be compatible with that
            for `time`.

            **Shape**: :math:`(N,\\,3)` or :math:`(N_t,\\,N,\\,3)`.

            **Reference unit**: :math:`\\mathrm{Å/ps}`.

        forces : `numpy.ndarray`, optional
            Forces exerted on :math:`N` atoms over :math:`N_t` frames.
            The dimensionality depends on whether a single or multiple
            frames are to be written and must be compatible with that
            for `time`.

            **Shape**: :math:`(N,\\,3)` or :math:`(N_t,\\,N,\\,3)`.

            **Reference unit**: :math:`\\mathrm{Å/ps}`.

        cell_lengths : `numpy.ndarray`, optional
            Simulation box dimensions.

            **Shape**: :math:`(3,)`.

            **Reference unit**: :math:`\\mathrm{Å}`.

        cell_angles : `numpy.ndarray`, optional
            Angles that define the shape of the simulation box.

            **Shape**: :math:`(3,)`.

            **Reference unit**: :math:`^\\circ`.

        restart : `bool`, keyword-only, default: :code:`False`
            Prevents the frame index from being incremented if writing a
            NetCDF restart file.

        Returns
        -------
        netcdf_file : `mdcraft.openmm.file.NetCDFFile`
            NetCDF file object. Only returned when this function is used
            as a static method.
        """

        # Create NetCDF file or object if it doesn't already exist
        if not isinstance(self, NetCDFFile):
            self = NetCDFFile(self, "w", restart=restart)
        if not hasattr(self._nc, "Conventions"):
            self.write_header(coordinates.shape[0],
                              cell_lengths is not None or cell_angles is not None,
                              velocities is not None, forces is not None)
            self._nc.set_always_mask(False)

        # Write model to NetCDF file
        n_frames = len(time) if isinstance(time, (tuple, list, np.ndarray)) else 1
        frames = slice(self._frame, self._frame + n_frames)
        self._nc.variables["time"][frames] = time
        self._nc.variables["coordinates"][frames] = coordinates
        if velocities is not None:
            self._nc.variables["velocities"][frames] = velocities
        if forces is not None:
            self._nc.variables["forces"][frames] = forces
        if cell_lengths is not None:
            self._nc.variables["cell_lengths"][frames] = cell_lengths
        if cell_angles is not None:
            self._nc.variables["cell_angles"][frames] = cell_angles
        self._nc.sync()
        if not restart:
            self._frame += n_frames

        return self