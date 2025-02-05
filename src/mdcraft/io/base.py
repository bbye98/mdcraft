from __future__ import annotations
from abc import abstractmethod
from collections.abc import Iterable
import importlib.util
from pathlib import Path
from types import TracebackType
from typing import Any, Self, TextIO
import weakref

import numpy as np
import psutil

from .. import U_

FOUND_MDANALYSIS = importlib.util.find_spec("MDAnalysis") is not None


class BaseReader:
    """
    Base class for topology and trajectory readers.

    Subclasses must implement the :meth:`open` and :meth:`close` methods
    to handle the opening and closing of the file.

    Parameters
    ----------
    filename : `str` or `pathlib.Path`, positional-only
        Filename or path to the topology or trajectory file.

    parallel : `bool`, keyword-only
        Determines whether the file is read in parallel.

    n_workers : `int`, keyword-only
        Number of threads to use when reading the file in parallel.
        If not specified, the number of logical threads available is
        used.
    """

    _PARALLELIZABLE: bool

    def __init__(
        self, filename: str | Path, /, *, parallel: bool, n_workers: int = None
    ) -> None:
        # Resolve full path to file
        self._filename = Path(filename).resolve(True)

        # Store settings
        self._parallel = parallel
        self._n_workers = n_workers or psutil.cpu_count()

        # Create finalizer
        self._finalizer = weakref.finalize(self, self.close)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._finalizer()

    def __getstate__(self) -> dict[str, Any]:
        # Close file before pickling for parallel reading
        self.close()
        return self.__dict__

    @abstractmethod
    def open(self) -> None:
        """
        Opens the topology or trajectory file and stores a handle to it.
        """

        pass

    @abstractmethod
    def close(self) -> None:
        """
        Closes the topology and trajectory file and deletes the handle.
        """

        pass


class BaseTopologyReader(BaseReader):
    """
    Base class for topology readers.

    Subclasses must set values for

    * the :attr:`_FORMAT` and :attr:`_EXTENSIONS` attributes, which
      specify the format and standard extension(s) of the topology
      file, respectively,
    * the :attr:`_UNITS` attribute, which specifies the units used by
      the simulation software that generated the topology file,
    * the :attr:`_PARALLELIZABLE` attribute, which specifies whether the
      reader can process a file in parallel,
    * the :attr:`_reduced` attribute, which specifies whether the data
      is in reduced units,
    * the :meth:`dimensions` property, which specifies the simulation
      box dimensions (or lattice parameters), and
    * the :meth:`n_atoms`, :meth:`n_bonds`, :meth:`n_angles`,
      :meth:`n_dihedrals`, :meth:`n_improper_dihedrals`,
      :meth:`n_residues`, :meth:`n_segments`, :meth:`n_chains`, and
      :meth:`n_molecules` properties, which specify the number of atoms,
      bonds, angles, dihedrals, improper dihedrals, residues, segments,
      chains, and molecules, respectively,

    and implement

    * the :meth:`__repr__` method to provide a string representation of
      the reader that can be used to recreate it, and
    * the :meth:`open` and :meth:`close` methods to handle the opening
      and closing of the file.

    Parameters
    ----------
    filename : `str` or `pathlib.Path`, positional-only
        Filename or path to the topology file.

    parallel : `bool`, keyword-only
        Determines whether the file is read in parallel.

    n_workers : `int`, keyword-only
        Number of threads to use when reading the file in parallel.
        If not specified, the number of logical threads available is
        used.
    """

    _EXTENSIONS: set[str]
    _FORMAT: str
    _UNITS: dict[str, U_]
    _reduced: bool

    def __init__(
        self,
        filename: str | Path,
        /,
        *,
        parallel: bool,
        n_workers: int,
    ) -> None:
        super().__init__(filename, parallel=parallel, n_workers=n_workers)

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}: '{self._filename.name}', "
            f"{self.n_atoms:,} atoms"
        )

    @staticmethod
    def _get_supported_formats() -> dict[str, object]:
        """
        Supported topology formats.
        """

        return {r._FORMAT: r for r in BaseTopologyReader.__subclasses__()}

    @abstractmethod
    def _parse_topology(self, file: TextIO) -> dict[str, Any]:
        pass

    def read_topology(
        self,
        /,
        *,
        parallel: bool | None = None,
        _convert_units: bool = True,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        pass

    @property
    @abstractmethod
    def dimensions(self) -> np.ndarray[float] | None:
        """
        Simulation box dimensions (or lattice parameters). If `None`,
        the system size could not be determined from the topology.

        **Reference units**: :math:`\\mathrm{nm}` for lengths and
        degrees (:math:`^\\circ`) for angles.
        """

        pass

    @property
    @abstractmethod
    def n_atoms(self) -> int:
        """
        Number of atoms.
        """

        pass

    @property
    @abstractmethod
    def n_bonds(self) -> int:
        """
        Number of bonds.
        """

        pass

    @property
    @abstractmethod
    def n_angles(self) -> int:
        """
        Number of angles.
        """

        pass

    @property
    @abstractmethod
    def n_dihedrals(self) -> int:
        """
        Number of dihedrals.
        """

        pass

    @property
    @abstractmethod
    def n_residues(self) -> int:
        """
        Number of residues.
        """

        pass

    @property
    @abstractmethod
    def n_segments(self) -> int:
        """
        Number of segments.
        """

        pass

    @property
    @abstractmethod
    def n_chains(self) -> int:
        """
        Number of chains.
        """

        pass

    @property
    @abstractmethod
    def n_molecules(self) -> int:
        """
        Number of molecules.
        """

        pass

    # TODO: Finish specification before implementing readers.
    # TODO: Support smaller subdivisions like molecules (residues) and segments (chains)?


class BaseTrajectoryReader(BaseReader):
    """
    Base class for trajectory readers.

    Subclasses must set values for

    * the :attr:`_FORMAT` and :attr:`_EXTENSIONS` attributes, which
      specify the format and standard extension(s) of the trajectory
      file, respectively,
    * the :attr:`_UNITS` attribute, which specifies the units used by
      the simulation software that generated the trajectory file,
    * the :attr:`_PARALLELIZABLE` attribute, which specifies whether the
      reader can process a file in parallel,
    * the :attr:`_reduced` attribute, which specifies whether the data
      is in reduced units,
    * the :meth:`dt` and :meth:`time_step` properties, which specify the
      time step size between timesteps and the time step between frames,
      respectively,
    * the :meth:`times` and :meth:`timesteps` properties, which specify
      the simulation times and timesteps found in the trajectory, and
    * the :meth:`n_atoms` and :meth:`n_frames` properties, which specify
      the number of frames in the trajectory and the number of atoms in
      each frame, respectively,

    and implement

    * the :meth:`__repr__` method to provide a string representation of
      the reader that can be used to recreate it,
    * the :meth:`_parse_frame` method to read and parse data from a
      single frame in the trajectory file, and
    * the :meth:`open` and :meth:`close` methods to handle the opening
      and closing of the file.

    Parameters
    ----------
    filename : `str` or `pathlib.Path`, positional-only
        Filename or path to the trajectory file.

    parallel : `bool`, keyword-only
        Determines whether the file is read in parallel.

    n_workers : `int`, keyword-only
        Number of threads to use when reading the file in parallel.
        If not specified, the number of logical threads available is
        used.
    """

    _EXTENSIONS: set[str]
    _FORMAT: str
    _UNITS: dict[str, U_]
    _reduced: bool

    def __init__(
        self,
        filename: str | Path,
        /,
        *,
        parallel: bool,
        n_workers: int,
    ) -> None:
        super().__init__(filename, parallel=parallel, n_workers=n_workers)

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        string = (
            f"{self.__class__.__name__}: '{self._filename.name}', "
            f"{self.n_frames:,} frame(s)"
        )
        if self.n_atoms is not None:
            string += f", {self.n_atoms:,} atom(s)"
        return string

    @staticmethod
    def _get_supported_formats() -> dict[str, object]:
        """
        Supported trajectory formats.
        """

        return {r._FORMAT: r for r in BaseTrajectoryReader.__subclasses__()}

    def _check_frame(self, frame_index: int) -> None:
        """
        Checks if a frame index is valid.

        Parameters
        ----------
        frame_index : `int`
            Frame index to check.
        """

        if frame_index >= self.n_frames:
            raise EOFError(
                f"Frame with index {frame_index} was requested from "
                f"'{self._filename.name}', but it only has "
                f"{self.n_frames} frames."
            )

    @abstractmethod
    def _parse_frame(
        self, file: TextIO, frame_index: int, convert_units: bool
    ) -> dict[str, Any]:
        """
        Reads data from a single frame in the specified trajectory
        file.

        Parameters
        ----------
        file : `io.TextIO`
            Handle to the trajectory file.

        frame_index : `int`
            Frame index to read.

        convert_units : `bool`
            Specifies whether to convert the data from LAMMPS units to
            consistent MDCraft units.

        Returns
        -------
        frame_data : `dict`
            Data from the frame.
        """

        pass

    @property
    @abstractmethod
    def dt(self) -> float | None:
        """
        Time step size between timesteps in the trajectory. If `None`,
        the time step size is not constant across frames or could not be
        determined from the trajectory.

        **Reference units**: :math:`\\mathrm{ps}`.
        """

        pass

    @property
    @abstractmethod
    def time_step(self) -> float | None:
        """
        Time step between frames in the trajectory. If `None`, the time
        step is not constant across frames.

        **Reference units**: :math:`\\mathrm{ps}`.
        """

        pass

    @property
    @abstractmethod
    def times(self) -> np.ndarray[float]:
        """
        Simulation times found in the trajectory.

        **Reference units**: :math:`\\mathrm{ps}`.
        """

        pass

    @property
    @abstractmethod
    def timesteps(self) -> float | None:
        """
        Simulation timesteps found in the trajectory. If `None`, the
        timesteps could not be determined from the trajectory.
        """

        pass

    @property
    @abstractmethod
    def n_atoms(self) -> int | None:
        """
        Number of atoms in each frame. If `None`, the number of atoms
        is not constant across frames.
        """

        pass

    @property
    @abstractmethod
    def n_frames(self) -> int:
        """
        Number of frames in the trajectory.
        """

        pass

    def read_frames(
        self,
        frame_indices: int | slice | Iterable[int],
        /,
        *,
        parallel: bool | None = None,
        _convert_units: bool = True,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Reads data from one or more frames from the trajectory file.

        Parameters
        ----------
        frame_indices : `int`, `slice`, or array-like, positional-only
            Indices of frames to read.

        parallel : `bool`, keyword-only, optional
            Determines whether the file is read in parallel.

        Returns
        -------
        data : `dict` or `list`
            Data from the frame(s).
        """

        # Validate indices of frames
        if isinstance(frame_indices, (int, np.integer)):
            self._check_frame(frame_indices)
        else:
            if isinstance(frame_indices, slice):
                frame_indices = range(*frame_indices.indices(self.n_frames))
            for fi in frame_indices:
                self._check_frame(fi)

        # Open file for parallel reading, if necessary
        if parallel is None:
            parallel = self._parallel and self._PARALLELIZABLE
        if parallel:
            file = open(self._filename, "r")
        else:
            self.open()
            file = self._file

        # Read data from frame(s)
        data = (
            self._parse_frame(file, frame_indices, _convert_units and not self._reduced)
            if isinstance(frame_indices, (int, np.integer))
            else [
                self._parse_frame(file, fi, _convert_units and not self._reduced)
                for fi in frame_indices
            ]
        )

        # Close file, if necessary
        if parallel:
            file.close()

        return data
