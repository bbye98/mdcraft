import concurrent.futures
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, "/mnt/c/Users/Benjamin/Documents/GitHub/mdcraft-dev/src")
from mdcraft.io.reader import BaseTrajectoryReader  # , NetCDFReader  # dev MDCraft

# os.chdir("/mnt/c/Users/Benjamin/Downloads")
# filename = "example.nc"
# reader = NetCDFReader(filename, parallel=True)

### TESTING

os.chdir(
    "/mnt/e/caltech/research/projects/gcme/methodology/data/polyanion_counterion_solvent/edl/ic"
)
filename = "nvt_N_96000_Np_60_xp_0.005_rp_78.0_A_25.224_dV_0.000__0.nc"

import netCDF4 as nc
from scipy.io import netcdf_file


class NetCDFReader(BaseTrajectoryReader):
    """ """

    _EXTENSIONS = {"nc", "ncdf"}
    _IS_PARALLELIZABLE = False

    def __init__(
        self,
        filename: str | Path,
        /,
        *,
        module: str = "scipy",
        parallel: bool = False,
        n_threads: int | None = None,
        starting_frame: int = 0,
    ) -> None:

        super().__init__(
            filename,
            parallel=parallel,
            n_threads=n_threads,
            starting_frame=starting_frame,
        )

        # Store which module to use for reading
        self._module = module.lower()

        # Create and store handle to file
        self.open()

        # Store trajectory properties
        if self._module == "scipy":
            self._n_atoms = self._file.dimensions["atom"]
            self._n_frames = self._file._recs
            self._is_restart = self._file.Conventions == b"AMBERRESTART"
        else:
            self._n_atoms = self._file.dimensions["atom"].size
            self._n_frames = self._file.dimensions["frame"].size
            self._is_restart = self._file.Conventions == "AMBERRESTART"

        # Create finalizer
        self._finalizer = weakref.finalize(self, self.close)

    @property
    def n_atoms(self) -> int:
        """
        Number of atoms in each frame.
        """

        return self._n_atoms

    @property
    def n_frames(self) -> int:
        """
        Number of frames in the trajectory.
        """

        return self._n_frames

    def open(self) -> None:
        """
        Opens the NetCDF file and stores a handle to it.
        """

        self._file = self._open()

    def close(self) -> None:
        """
        Closes the NetCDF file and deletes the handle.
        """

        if hasattr(self, "_file"):
            self._file.close()
            del self._file

    def get_dimensions(
        self,
        frames: int | list[int] | slice | None = None,
        **kwargs,
    ) -> tuple[np.ndarray[float] | np.ndarray[float]]:
        """
        Gets the dimensions (lattice parameters) of the simulation box.

        Parameters
        ----------
        frames : `int`, `list`, or `slice`, optional
            Frame indices. If :code:`None`, the dimensions across all
            frames are returned.

        Returns
        -------
        dimensions : `numpy.ndarray`
            Simulation box dimensions.

            **Shape**: :math:`(6,)` or :math:`(N_\\mathrm{frames},6)`.

            **Reference unit**: :math:`\\mathrm{Å}` for the lengths and
            :math:`^\\circ` for the angles.
        """

        if frames is None:
            frames = slice(None)

        file = kwargs.get("file") or getattr(self, "_file", None)
        if file is None:
            file = self._open()
            manual = True
        else:
            manual = False

        dimensions = np.hstack(
            (
                file.variables["cell_lengths"][frames],
                file.variables["cell_angles"][frames],
            )
        )

        if manual:
            file.close()

        return dimensions

    def get_forces(
        self,
        frames: int | list[int] | slice | None = None,
        *,
        verbose: bool = True,
        **kwargs,
    ) -> np.ndarray[float]:
        """
        Gets the forces acting on the atoms.

        Parameters
        ----------
        frames : `int`, `list`, or `slice`, optional
            Indices of frames. If :code:`None`, the forces across all frames
            are returned.

        verbose : `bool`, keyword-only, default: :code:`True`
            Determines whether a warning is issued if the NetCDF file does
            not contain force information.

        Returns
        -------
        forces : `numpy.ndarray` or `openmm.unit.Quantity`
            Forces acting on the atoms. If the NetCDF file does not
            contain this information, :code:`None` is returned.

            **Shape**: :math:`(N_\\mathrm{atoms},3)` or
            :math:`(N_\\mathrm{frames},N_\\mathrm{atoms},3)`.

            **Reference unit**: :math:`\\mathrm{Å/ps}`.
        """

        if frames is None:
            frames = slice(None)

        file = kwargs.get("file") or getattr(self, "_file", None)
        if file is None:
            file = self._open()
            manual = True
        else:
            manual = False

        if "forces" not in file.variables:
            if verbose:
                warnings.warn(
                    "The NetCDF file "
                    f"'{Path(file.filepath()).resolve().name}' does "
                    "not contain the forces acting on the atoms."
                )
            return None

        forces = file.variables["forces"][frames]

        if manual:
            file.close()

        return forces

    def get_positions(
        self, frames: int | list[int] | slice | None = None, **kwargs
    ) -> np.ndarray[float]:
        """
        Gets the atom positions.

        Parameters
        ----------
        frames : `int`, `list`, or `slice`, optional
            Indices of frames. If :code:`None`, the positions across all
            frames are returned.

        Returns
        -------
        positions : `numpy.ndarray`
            Atom positions.

            **Shape**: :math:`(N_\\mathrm{atoms},3)` or
            :math:`(N_\\mathrm{frames},N_\\mathrm{atoms},3)`.

            **Reference unit**: :math:`\\mathrm{Å}`.
        """

        if frames is None:
            frames = slice(None)

        file = kwargs.get("file") or getattr(self, "_file", None)
        if file is None:
            file = self._open()
            manual = True
        else:
            manual = False

        positions = file.variables["coordinates"][frames]

        if manual:
            file.close()

        return positions

    def get_times(
        self, frames: int | list[int] | slice | None = None, **kwargs
    ) -> int | np.ndarray[float]:
        """
        Gets the simulation times.

        Parameters
        ----------
        frames : `int`, `list`, or `slice`, optional
            Indices of frames. If :code:`None`, the times across all
            frames are returned.

        Returns
        -------
        times : `int` or `numpy.ndarray`
            Simulation times.

            **Shape**: Scalar or :math:`(N_\\mathrm{frames},)`.

            **Reference unit**: :math:`\\mathrm{ps}`.
        """

        if frames is None:
            frames = slice(None)

        file = kwargs.get("file") or getattr(self, "_file", None)
        if file is None:
            file = self._open()
            manual = True
        else:
            manual = False

        times = file.variables["time"][frames]

        if manual:
            file.close()

        return times

    def get_velocities(
        self,
        frames: int | list[int] | slice | None = None,
        /,
        *,
        verbose: bool = True,
        **kwargs,
    ) -> np.ndarray[float]:
        """
        Gets the atom velocities.

        Parameters
        ----------
        frames : `int`, `list`, or `slice`, optional
            Indices of frames. If :code:`None`, the velocities across all
            frames are returned.

        verbose : `bool`, default: :code:`True`
            Determines whether a warning is issued if the NetCDF file does
            not contain velocity information.

        Returns
        -------
        velocities : `numpy.ndarray` or `openmm.unit.Quantity`
            Atom velocities. If the NetCDF file does not contain
            this information, :code:`None` is returned.

            **Shape**: :math:`(N_\\mathrm{atoms},3)` or
            :math:`(N_\\mathrm{frames},N_\\mathrm{atoms},3)`.

            **Reference unit**: :math:`\\mathrm{Å/ps}`.
        """

        if frames is None:
            frames = slice(None)

        file = kwargs.get("file") or getattr(self, "_file", None)
        if file is None:
            file = self._open()
            manual = True
        else:
            manual = False

        if "velocities" not in file.variables:
            if verbose:
                warnings.warn(
                    "The NetCDF file "
                    f"'{Path(file.filepath()).resolve().name}' does "
                    "not contain atom velocities."
                )
            return None

        velocities = file.variables["velocities"][frames]

        if manual:
            file.close()

        return velocities

    def _open(self) -> netcdf_file | nc.Dataset:
        if self._module == "scipy":
            file = netcdf_file(self._filename, "r")
        else:
            file = nc.Dataset(self._filename, mode="r")
            file.set_auto_mask(False)
        return file

    def read_frames(
        self, frames: int | Iterable[int], *, parallel: bool
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Reads data from one or more frames from the NetCDF file.

        Parameters
        ----------
        frames : `int` or `list`
            Indices of frames to read.

        parallel : `bool`, keyword-only
            Determines whether the file is read in parallel.

        Returns
        -------
        data : `dict` or `list`
            Data from the frames.
        """

        def read_frame(file: nc.Dataset, frame: int) -> dict[str, Any]:
            """
            Reads data from a single frame in the specified NetCDF file.

            Parameters
            ----------
            file : `netCDF4.Dataset`
                Handle to the NetCDF file.

            frame : `int`
                Frame index to read.

            Returns
            -------
            frame_data : `dict`
                Data from the frame.
            """

            return {
                "time": self.get_times(frame, file=file),
                "n_atoms": self.n_atoms,
                "positions": self.get_positions(frame, file=file),
                "dimensions": self.get_dimensions(frame, file=file),
                "forces": self.get_forces(frame, verbose=False, file=file),
                "velocities": self.get_velocities(frame, verbose=False, file=file),
            }

        # Open file for parallel reading, if necessary
        file = self._open() if parallel else self._file

        # Read data from frame(s)
        data = (
            read_frame(file, frames)
            if isinstance(frames, int)
            else [read_frame(file, frame) for frame in frames]
        )

        # Close file, if necessary
        if parallel:
            file.close()

        return data


n_frames = 2500

start = datetime.now()
data_serial = []
file = netcdf_file(filename)
for i in range(n_frames):
    data_serial.append(file.variables["coordinates"][i])
print(f"Serial: {datetime.now() - start}")


def read_parallel(file, i):
    return file.variables["coordinates"][i]


start = datetime.now()
data_parallel = []
file = nc.Dataset(filename)
with concurrent.futures.ProcessPoolExecutor() as executor:
    for future in concurrent.futures.as_completed(
        executor.submit(read_parallel, file, i) for i in range(n_frames)
    ):
        data_serial.append(future.result())
print(f"Parallel: {datetime.now() - start}")

### BENCHMARK

# os.chdir("/mnt/e/research/gcme/methodology/data/polyanion_counterion_solvent/edl/ic")
# filename = "nvt_N_96000_Np_60_xp_0.005_rp_78.0_A_25.224_dV_0.000__0.nc"

# print("Parallel (scipy.io.netcdf_file):")
# start = datetime.now()
# reader = NetCDFReader(filename, parallel=True)
# n_frames = reader.n_frames
# n_workers = 20
# frames_per_worker = np.ceil(n_frames / n_workers).astype(int)
# frames = []
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     for future in concurrent.futures.as_completed(
#         executor.submit(
#             reader.get_frames,
#             slice(n * frames_per_worker, (n + 1) * frames_per_worker),
#             parallel=True
#         )
#         for n in range(n_workers)
#     ):
#         frames.extend(future.result())
# print(f"  Read {n_frames} frames: {datetime.now() - start}")

# print("Parallel (netCDF4.Dataset):")
# start = datetime.now()
# reader = NetCDFReader(filename, module="netcdf4", parallel=True)
# n_frames = reader.n_frames
# n_workers = 20
# frames_per_worker = np.ceil(n_frames / n_workers).astype(int)
# frames = []
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     for future in concurrent.futures.as_completed(
#         executor.submit(
#             reader.get_frames,
#             slice(n * frames_per_worker, (n + 1) * frames_per_worker),
#             parallel=True
#         )
#         for n in range(n_workers)
#     ):
#         frames.extend(future.result())
# print(f"  Read {n_frames} frames: {datetime.now() - start}")

# print("Serial (scipy.io.netcdf_file):")
# start = datetime.now()
# reader = NetCDFReader(filename)
# n_frames = reader.n_frames
# frames_serial_scipy = reader.get_frames(range(n_frames))
# print(f"  Read {n_frames} frames: {datetime.now() - start}")

# print("Serial (netCDF4.Dataset):")
# start = datetime.now()
# reader = NetCDFReader(filename, module="netcdf4")
# n_frames = reader.n_frames
# frames_serial_netcdf4 = reader.get_frames(range(n_frames))
# print(f"  Read {n_frames} frames: {datetime.now() - start}")

debug = True
