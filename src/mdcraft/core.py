"""
Core analysis objects
=====================
.. moduleauthor:: Benjamin B. Ye <GitHub: @bbye98>

This submodule contains the core analysis objects for interacting with
simulation topologies, trajectories, and other outputs.
"""

from __future__ import annotations
from bisect import bisect
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Iterable, Self

import numpy as np

from .io.reader import BaseTopologyReader, BaseTrajectoryReader


class System:  # TODO
    """ """

    class SystemSubset:  # TODO
        """ """

        def __init__(self) -> None:
            pass

    def __init__(self) -> None:
        # TODO: Support (multiple) Topology objects and OpenMM topology objects
        # TODO: Support Trajectory objects and OpenMM context/state objects
        pass


class Topology:  # TODO
    """
    Simulation topology.

    A topology defines the structure, interactions, and parameters
    required to describe the simulation system. Information includes
    topological objects, such as atoms, residues, segments, chains, and
    molecules, and their force field properties, such as masses,
    charges, bonds, angles, dihedrals, and improper dihedrals.

    The topology hierarchy is organized as follows:

    * Atoms are the basic building blocks.
    * Residues are groups of atoms that form a chemically distinct unit
      within a larger structure, like an amino acid in a protein or a
      monomer in a polymer.
    * Segments are groups of residues that represent a structural or
      functional subregion within a molecule, like a loop region in a
      protein or a block of monomers in a copolymer. Oftentimes,
      there is no distinction between segments and chains.
    * Chains are sequences of connected residues that form a single,
      continuous strand in a molecule, like a polypeptide chain in a
      protein or a full polymer chain.
    * Molecules are groups of atoms that are chemically bonded to each
      other, and can range from small (e.g., water) to very large
      macromolecules (e.g., proteins, DNA, and polymers).

    Parameters
    ----------
    filename : `str` or `pathlib.Path`
        Path to the topology file.

    format : `str`, keyword-only, optional
        Format of the topology file. If not specified, the format is
        determined from the file extension.

    **kwargs : `dict`
        Additional keyword arguments to pass to the topology reader.
    """

    def __init__(self, filename: str | Path, *, format: str = None, **kwargs) -> None:
        self._filename = Path(filename).resolve(True)
        if format is None:
            try:
                self._reader = next(
                    r
                    for r in BaseTopologyReader.__subclasses__()
                    if self._filename.suffix in r._EXTENSIONS
                )(filename, **kwargs)
            except StopIteration:
                raise RuntimeError(
                    "Could not determine the format of the topology file."
                )
        else:
            supported_formats = BaseTopologyReader.supported_formats()
            format = format.upper()
            if format not in supported_formats:
                raise ValueError(
                    f"Invalid or unsupported format '{format}'. Valid "
                    f"values: '" + "', '".join(supported_formats) + "'."
                )
            self._reader = supported_formats[format](filename, **kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._filename.name}, "
            f"format='{self._reader._FORMAT}', ...)"
        )

    def __str__(self) -> str:
        string = f"{self.__class__.__name__}: '{self._filename.name}'"
        if self.n_atoms is not None:
            string += f", {self.n_atoms:,} atoms"
        return string

    @property
    def dimensions(self) -> np.ndarray[float] | None:
        """
        Simulation box dimensions (or lattice parameters). If `None`,
        the dimensions are not available in the topology.

        **Reference units**: :math:`\\mathrm{nm}` for lengths and
        degrees (:math:`^\\circ`) for angles.
        """

        return self._reader.dimensions

    @property
    def n_atoms(self) -> int:
        """
        Number of atoms.
        """

        return self._reader.n_atoms

    # TODO: Add other properties from BaseTopologyReader


class ForceField:
    pass


class Atom:
    pass


class Bond:
    pass


class Angle:
    pass


class Dihedral:
    pass


class Residue:
    pass


class Segment:
    pass


class Chain:
    pass


class Molecule:
    pass


class Trajectory:
    """
    Simulation trajectory.

    A trajectory contains all data found in the frames from one or more
    trajectory files. Information includes simulation times, timesteps,
    atom positions, forces, velocities, and other global, group, or
    per-atom attributes depending on the trajectory format.

    When multiple trajectory files are provided, they are first ordered
    by their starting times and then by their ending times to form a
    continuous trajectory proceeding forward in time.

    .. note::

       All trajectory files must use the same unit system, especially
       for LAMMPS dump files.

    When the trajectory files have overlapping frames, the frames from
    the later files are used. With :code:`o` denoting kept frames,
    :code:`x` denoting discarded frames, and :code:`f` denoting frames
    available in the stitched trajectory, a simple example is:

    .. code::

       file 0:     ooxxxx
       file 1:       oooooo
       file 2:             oooooo
       trajectory: ffffffffffffff

    More complex interleaving of frames is also possible, but is not
    guaranteed to be handled correctly. Examples include:

    .. code::

       file 0:     xooxoo
       file 1:     o  o  o  x  o  x
       file 2:              o o o o
       trajectory: fffffff  f fff f

    and:

    .. code::

       file 0:     x o o x o o
       file 1:     o  x  o  x  o  x  o
       file 2:      o o o o
       file 3:              o o o o o o
       trajectory: ffffffffffffff f fff

    Individual frames can be accessed through indexing or iteration.

    Parameters
    ----------
    filenames : `str`, `pathlib.Path`, or array-like
        Path to the trajectory files.

    formats : `str`, keyword-only, optional
        Formats of the trajectory files. If not specified, the formats
        are determined from the file extensions.

    **kwargs : `dict`
        Additional keyword arguments to pass to all trajectory readers.
    """

    def __init__(
        self,
        filenames: str | Path | Iterable[str | Path],
        *,
        formats: str | Iterable[str] = None,
        **kwargs,
    ) -> None:
        # Resolve full paths to files
        if isinstance(filenames, (str, Path)):
            self._filenames = np.array((Path(filenames).resolve(True),))
        else:
            self._filenames = np.fromiter(
                (Path(f).resolve(True) for f in filenames),
                dtype=object,
                count=len(filenames),
            )

        # Determine the format of and instantiate readers for the
        # trajectory files
        if formats is None:
            try:
                self._readers = np.fromiter(
                    (
                        next(
                            r
                            for r in BaseTrajectoryReader.__subclasses__()
                            if (filename := f).suffix in r._EXTENSIONS
                        )(f, **kwargs)
                        for f in self._filenames
                    ),
                    dtype=object,
                    count=len(self._filenames),
                )
            except StopIteration:
                raise RuntimeError(
                    "Could not determine the format of trajectory file "
                    f"'{filename.name}'."
                )
        else:
            supported_formats = BaseTrajectoryReader._get_supported_formats()
            if isinstance(formats, str):
                formats = formats.upper()
                if formats not in supported_formats:
                    raise ValueError(
                        f"Invalid or unsupported format '{formats}'. Valid "
                        f"values: '" + "', '".join(supported_formats) + "'."
                    )
                Reader = supported_formats[format]
                self._readers = np.fromiter(
                    (Reader(f, **kwargs) for f in self._filenames),
                    dtype=object,
                    count=len(self._filenames),
                )
            elif len(formats) == len(self._filenames):
                try:
                    self._readers = np.fromiter(
                        (
                            supported_formats[fmt.upper()](f, **kwargs)
                            for f, fmt in zip(self._filenames, formats)
                        ),
                        dtype=object,
                        count=len(self._filenames),
                    )
                except KeyError as e:
                    raise ValueError(
                        f"Invalid or unsupported format '{e.args[0]}'. Valid "
                        f"values: '" + "', '".join(supported_formats) + "'."
                    )
            else:
                raise ValueError(
                    "Number of formats in `formats` does not match the "
                    "number of files in `filenames`."
                )

        # Order readers by starting time then by ending time
        time_ranges = [(r.times[0], r.times[-1]) for r in self._readers]
        order = [x[0] for x in sorted(enumerate(time_ranges), key=lambda x: x[1])]
        time_ranges = np.array(time_ranges)
        if not all(o > order[i] for i, o in enumerate(order[1:])):
            time_ranges = time_ranges[order]
            self._filenames = self._filenames[order]
            self._readers = self._readers[order]

        self._overlap_frames = {}  # <time>: (<reader_index>, <reader_frame_index>)
        if len(self._readers) > 1:
            # Combine trajectories from individual readers into a
            # continuous trajectory
            self._n_frames = 0
            self._start_frames = []
            self._offset_frames = []
            trajectory_end_time = time_ranges[0, 1]
            seen_times = set()
            unseen_start_time = -np.inf

            for reader_index, reader in enumerate(self._readers[:-1]):
                # Get next reader
                next_reader = self._readers[reader_index + 1]

                # Get the ending time (inclusive) of the current reader
                reader_end_time = reader.times[-1]

                # Update the trajectory ending time
                trajectory_end_time = max(trajectory_end_time, reader_end_time)

                # Find overlapping times (inclusive) between the readers
                overlap_start_time = next_reader.times[0]
                overlap_end_time = max(
                    min(reader_end_time, next_reader.times[-1]), trajectory_end_time
                )

                # Account for normal frames in the current reader before
                # the overlapping region
                is_normal_frame = (reader.times > unseen_start_time) & (
                    reader.times < overlap_start_time
                )
                n_normal_frames = np.count_nonzero(is_normal_frame)
                self._start_frames.append(self._n_frames)
                self._offset_frames.append(is_normal_frame.argmax())
                self._n_frames += n_normal_frames
                seen_times.update(reader.times[is_normal_frame])

                # Account for overlapping frames in both readers
                reader_indices = (reader_index, reader_index + 1)
                overlap_reader_frame_indices = (
                    (
                        (reader.times >= overlap_start_time)
                        & (reader.times <= overlap_end_time)
                    ).nonzero()[0],
                    (
                        (next_reader.times >= overlap_start_time)
                        & (next_reader.times <= overlap_end_time)
                    ).nonzero()[0],
                )
                overlap_reader_times = (
                    reader.times[overlap_reader_frame_indices[0]],
                    next_reader.times[overlap_reader_frame_indices[1]],
                )

                for ri, rfis, rts in zip(
                    reader_indices, overlap_reader_frame_indices, overlap_reader_times
                ):
                    for rfi, rt in zip(rfis, rts):
                        if rt not in seen_times:
                            seen_times.add(rt)
                            self._n_frames += 1
                        self._overlap_frames[rt] = ri, rfi

                # Update the starting time for the next iteration
                unseen_start_time = overlap_end_time

            # Account for normal frames, if any, in the last reader
            reader = self._readers[-1]
            if trajectory_end_time < reader.times[-1]:
                is_normal_frame = reader.times > trajectory_end_time
                self._start_frames.append(self._n_frames)
                self._offset_frames.append(is_normal_frame.argmax())
                self._n_frames += np.count_nonzero(is_normal_frame)
                seen_times.update(reader.times[is_normal_frame])

            # Replace times with trajectory indices in the dictionary of
            # overlap frames
            self._overlap_frames = {
                index: info
                for index, info in zip(
                    np.searchsorted(sorted(seen_times), list(self._overlap_frames)),
                    self._overlap_frames.values(),
                )
            }
        else:
            self._n_frames = self._readers[0].n_frames
            self._start_frames = [0]
            self._offset_frames = [0]

        # Store current frame index
        self._frame_index = 0

    def __getitem__(
        self, frame_indices: int | slice | Iterable[int]
    ) -> TrajectoryFrame | TrajectorySubset:
        if isinstance(frame_indices, int):
            return self.get_frames(frame_indices)
        if isinstance(frame_indices, slice):
            frame_indices = range(*frame_indices.indices(self.n_frames))
        for fi in frame_indices:
            self._check_frame(fi)
        return TrajectorySubset(self, frame_indices)

    def __iter__(self) -> Self:
        return self

    def __len__(self) -> int:
        return self.n_frames

    def __next__(self) -> TrajectoryFrame:
        if self._frame_index >= self.n_frames:
            self._frame_index = 0
            raise StopIteration

        frame_index = self._frame_index
        self._frame_index += 1
        return self.get_frames(frame_index)

    def __repr__(self) -> str:
        if len(self._filenames) == 1:
            filenames = self._filenames[0]
            formats = self._readers[0]._FORMAT
        else:
            filenames = [f.name for f in self._filenames]
            formats = [r._FORMAT for r in self._readers]
        return f"{self.__class__.__name__}({filenames}, formats={formats}, ...)"

    def __str__(self) -> str:
        filenames = (
            self._filenames[0]
            if len(self._filenames) == 1
            else [f.name for f in self._filenames]
        )
        string = f"{self.__class__.__name__}: {filenames}, {self.n_frames:,} frame(s)"
        if self.n_atoms is not None:
            string += f", {self.n_atoms:,} atom(s)"
        return string

    def _check_frame(self, frame_index: int) -> None:
        """
        Checks if a frame index is valid.

        Parameters
        ----------
        frame_index : `int`
            Frame index to check.
        """

        if not -self.n_frames <= frame_index < self.n_frames:
            raise EOFError(
                f"Frame with index {frame_index} was requested from "
                f"a trajectory with only {self.n_frames} frames."
            )

    def _get_reader_indices(
        self, frame_indices: int | Iterable[int], /
    ) -> tuple[int, int] | list[tuple[int, int]]:
        """
        Get the reader and reader frame indices for the desired
        trajectory frames.

        Parameters
        ----------
        frame_indices : `int` or array-like
            Trajectory frame indices.

        Returns
        -------
        reader_indices : `tuple` or array-like
            A tuple or list of tuples of reader and reader frame
            indices.
        """

        if isinstance(frame_indices, int):
            if frame_indices < 0:
                frame_indices %= self.n_frames
            if frame_indices in self._overlap_frames:
                reader_index, reader_frame_index = self._overlap_frames[frame_indices]
            else:
                reader_index = bisect(self._start_frames, frame_indices) - 1
                reader_frame_index = (
                    frame_indices
                    - self._start_frames[reader_index]
                    + self._offset_frames[reader_index]
                )
            return reader_index, reader_frame_index
        return [self._get_reader_indices(i) for i in frame_indices]

    @cached_property
    def dt(self) -> float | None:
        """
        Time step size between timesteps in the trajectory. If `None`,
        the time step size is not constant across frames or could not be
        determined from the trajectory.

        **Reference unit**: :math:`\\mathrm{ps}`.
        """

        dts = set(
            (
                (r.dt for r in self._readers)
                if self.timesteps is None
                else np.round(
                    np.diff(self.times) / np.diff(self.timesteps),
                    np.finfo(float).precision,
                )
            )
        )
        return None if len(dts) != 1 else dts.pop()

    @cached_property
    def time_step(self) -> float | None:
        """
        Time step between frames in the trajectory. If `None`, the time
        step is not constant across frames.

        **Reference units**: :math:`\\mathrm{ps}`.
        """

        time_steps = set(np.round(np.diff(self.times), np.finfo(float).precision))
        return None if len(time_steps) != 1 else time_steps.pop()

    @cached_property
    def times(self) -> np.ndarray[float]:
        """
        Simulation times found in the trajectory.

        **Reference units**: :math:`\\mathrm{ps}`.
        """

        return np.fromiter(
            (
                self._readers[ri].times[rfi]
                for ri, rfi in self._get_reader_indices(range(self.n_frames))
            ),
            dtype=float,
            count=self.n_frames,
        )

    @cached_property
    def timesteps(self) -> np.ndarray[int] | None:
        """
        Simulation timesteps found in the trajectory. If `None`, the
        timesteps could not be determined from the trajectory.
        """

        if any(r.timesteps is None for r in self._readers):
            return None
        return np.fromiter(
            (
                self._readers[ri].timesteps[rfi]
                for ri, rfi in self._get_reader_indices(range(self.n_frames))
            ),
            dtype=float,
            count=self.n_frames,
        )

    @cached_property
    def n_atoms(self) -> int | None:
        """
        Number of atoms in each frame. If `None`, the number of atoms
        is not constant across frames or the trajectory does not contain
        atom data.
        """

        n_atoms = {r.n_atoms for r in self._readers}
        return None if len(n_atoms) != 1 else n_atoms.pop()

    @property
    def n_frames(self) -> int:
        """
        Number of frames in the trajectory.
        """

        return self._n_frames

    def get_frames(
        self,
        frame_indices: int | slice | Iterable[int],
        /,
        *,
        parallel: bool | None = None,
    ) -> TrajectoryFrame | list[TrajectoryFrame]:
        """
        Gets one or more frames from the trajectory.

        Parameters
        ----------
        frame_indices : `int`, `slice`, or array-like, positional-only
            Indices of frames to get.

        parallel : `bool`, keyword-only, optional
            Specifies whether this method will be run in parallel. If
            not specified, the setting used for the trajectory reader is
            used.

        Returns
        -------
        frames : `TrajectoryFrame` or `list`
            Trajectory frame(s).
        """

        # Get single frame
        if isinstance(frame_indices, int):
            self._check_frame(frame_indices)
            ri, rfi = self._get_reader_indices(frame_indices)
            return TrajectoryFrame(
                frame_indices % self.n_frames,
                **self._readers[ri].read_frames(rfi, parallel=parallel),
            )

        # Convert slices to iterable ranges
        if isinstance(frame_indices, slice):
            frame_indices = range(*frame_indices.indices(self.n_frames))
        for fi in frame_indices:
            self._check_frame(fi)

        # Get reader and reader frame indices
        reader_indices = defaultdict(list)
        for ri, rfi in self._get_reader_indices(frame_indices):
            reader_indices[ri].append(rfi)

        # Read frames from each reader
        frames = []
        for ri, rfis in reader_indices.items():
            frames.extend(self._readers[ri].read_frames(rfis, parallel=parallel))
        return [
            TrajectoryFrame(f % self.n_frames, **d)
            for f, d in zip(frame_indices, sorted(frames, key=lambda f: f["time"]))
        ]


class TrajectorySubset:
    """
    Trajectory subset.

    Individual frames can be accessed through relative indexing or
    iteration.

    Parameters
    ----------
    trajectory : `Trajectory`
        Full simulation trajectory.

    frame_indices : `slice` or array-like
        Indices of frames to keep in the trajectory subset.
    """

    def __init__(
        self, trajectory: Trajectory, frame_indices: slice | range | Iterable[int]
    ) -> None:
        self._trajectory = trajectory
        self._frame_indices = (
            range(*frame_indices.indices(self._trajectory.n_frames))
            if isinstance(frame_indices, slice)
            else frame_indices
        )
        self._index = 0

    def __getitem__(self, indices: int | slice | Iterable[int]) -> TrajectoryFrame:
        if isinstance(indices, int):
            self._check_frame(indices)
            return self.get_frames(indices)
        if isinstance(indices, slice):
            indices = range(*indices.indices(self.n_frames))
        for i in indices:
            self._check_frame(i)
        return TrajectorySubset(self._trajectory, self._frame_indices[indices])

    def __iter__(self) -> Self:
        return self

    def __len__(self) -> int:
        return self.n_frames

    def __next__(self) -> TrajectoryFrame:
        if self._index >= self.n_frames:
            self._index = 0
            raise StopIteration

        frame = self._frame_indices[self._index]
        self._index += 1
        return self._trajectory.get_frames(frame)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({repr(self._trajectory)}, frames={self._frame_indices})"
        )

    def __str__(self) -> str:
        string = (
            f"{self.__class__.__name__}: "
            f"{[f.name for f in self._trajectory._filenames]}, "
            f"{len(self._frame_indices):,} frame(s)"
        )
        if self._trajectory.n_atoms is not None:
            string += f", {self._trajectory.n_atoms:,} atom(s)"
        return string

    def _check_frame(self, index: int) -> None:
        """
        Checks if a frame index is valid.

        Parameters
        ----------
        index : `int`
            Frame index to check.
        """

        if not -self.n_frames <= index < self.n_frames:
            raise EOFError(
                f"Frame with index {index} was requested from a "
                f"trajectory subset with only {self.n_frames} frames."
            )

    @cached_property
    def dt(self) -> float | None:
        """
        Time step size between timesteps in the trajectory subset. If
        `None`, the time step size is not constant across frames or
        could not be determined from the subset.

        **Reference unit**: :math:`\\mathrm{ps}`.
        """

        dts = set(
            (
                self._trajectory._readers[ri].dt
                for ri in {
                    ri[0]
                    for ri in self._trajectory._get_reader_indices(self._frame_indices)
                }
            )
            if self.timesteps is None
            else np.round(
                np.diff(self.times) / np.diff(self.timesteps), np.finfo(float).precision
            )
        )
        return None if len(dts) != 1 else dts.pop()

    @cached_property
    def time_step(self) -> float | None:
        """
        Time step between frames in the trajectory subset. If `None`,
        the time step is not constant across frames.

        **Reference units**: :math:`\\mathrm{ps}`.
        """

        time_steps = set(np.round(np.diff(self.times), np.finfo(float).precision))
        return None if len(time_steps) != 1 else time_steps.pop()

    @cached_property
    def times(self) -> np.ndarray[float]:
        """
        Simulation times found in the trajectory subset.

        **Reference units**: :math:`\\mathrm{ps}`.
        """

        return np.fromiter(
            (
                self._trajectory.times[self._frame_indices[i]]
                for i in range(self.n_frames)
            ),
            dtype=float,
            count=self.n_frames,
        )

    @cached_property
    def timesteps(self) -> np.ndarray[int] | None:
        """
        Simulation timesteps found in the trajectory subset. If `None`,
        the timesteps could not be determined from the subset.
        """

        return np.fromiter(
            (
                self._trajectory.timesteps[self._frame_indices[i]]
                for i in range(self.n_frames)
            ),
            dtype=float,
            count=self.n_frames,
        )

    @property
    def n_atoms(self) -> int:
        """
        Number of atoms in each frame. If `None`, the number of atoms
        is not constant across frames or the trajectory subset does not
        contain atom data.
        """

        n_atoms = {
            self._trajectory._readers[ri].n_atoms
            for ri in {
                ri[0]
                for ri in self._trajectory._get_reader_indices(self._frame_indices)
            }
        }
        return None if len(n_atoms) != 1 else n_atoms.pop()

    @property
    def n_frames(self) -> int:
        """
        Number of frames in the trajectory subset.
        """

        return len(self._frame_indices)

    def get_frames(
        self,
        indices: int | slice | Iterable[int],
        /,
        *,
        parallel: bool | None = None,
    ) -> TrajectoryFrame | list[TrajectoryFrame]:
        """
        Gets one or more frames from the trajectory subset.

        Parameters
        ----------
        indices : `int`, `slice`, or array-like, positional-only
            Indices of frames in the subset to get.

        parallel : `bool`, keyword-only, optional
            Specifies whether this method will be run in parallel. If
            not specified, the setting used for the trajectory reader is
            used.

        Returns
        -------
        frames : `TrajectoryFrame` or `list`
            Trajectory frame(s).
        """

        return self._trajectory.get_frames(
            self._frame_indices[indices], parallel=parallel
        )


class TrajectoryFrame:
    """
    Trajectory frame.

    This class holds the data of a single frame in a trajectory.

    Parameters
    ----------
    frame : `int`
        Frame number.

    time : `float`
        Simulation time.

        **Reference unit**: :math:`\\mathrm{ps}`.

    timestep : `int`, keyword-only, optional
        Timestep number.

    dimensions : `numpy.ndarray`, keyword-only, optional
        Simulation box dimensions (or lattice parameters).

        **Reference units**: :math:`\\mathrm{nm}` for lengths and
        degrees (:math:`^\\circ`) for angles.

    n_atoms : `int`, keyword-only, optional
        Number of atoms.

    ids : `numpy.ndarray`, keyword-only, optional
        Atom indices or identifiers.

    positions : `numpy.ndarray`, keyword-only, optional
        Atom positions.

        **Reference unit**: :math:`\\mathrm{nm}`.

    forces : `numpy.ndarray`, keyword-only, optional
        Forces exerted on the atoms.

        **Reference unit**: :math:`\\mathrm{kJ/(mol\\cdot nm)}`.

    velocities : `numpy.ndarray`, keyword-only, optional
        Atom velocities.

        **Reference unit**: :math:`\\mathrm{nm/ps}`.

    **kwargs : `dict`
        Additional or non-standard per-atom attributes to be stored in
        the `extra` attribute.

        .. container::

           **Examples**:

           * Topology information, like masses and charges.
           * LAMMPS-specific attributes, like image flags.
           * Custom attributes, like LAMMPS computes, fixes, and
             variables.

        .. note::

           All additional attributes are in the original simulation
           units specified and will likely not be consistent with the
           internal set of units used by MDCraft.

    Attributes
    ----------
    extra : `dict`
        Additional or non-standard per-atom attributes.

        .. container::

           **Examples**:

           +-------------------------------------+--------------------------------------------+------------------------+
           | Attribute                           | Source(s)                                  | Per-atom data type     |
           +=====================================+============================================+========================+
           | :code:`"molecule_ids"`              | * LAMMPS dump: :code:`mol` attribute       | `int`                  |
           +-------------------------------------+--------------------------------------------+------------------------+
           | :code:`"types"`                     | * LAMMPS dump: :code:`type` attribute      | `int`                  |
           +-------------------------------------+--------------------------------------------+------------------------+
           | :code:`"labels"`                    | * LAMMPS dump: :code:`typelabel` attribute | `str`                  |
           +-------------------------------------+--------------------------------------------+------------------------+
           | :code:`"elements"`                  | * LAMMPS dump: :code:`element` attribute   | `str`                  |
           +-------------------------------------+--------------------------------------------+------------------------+
           | :code:`"masses"`                    | * LAMMPS dump: :code:`mass` attribute      | `float`                |
           +-------------------------------------+--------------------------------------------+------------------------+
           | :code:`"charges"`                   | * LAMMPS dump: :code:`charge` attribute    | `float`                |
           +-------------------------------------+--------------------------------------------+------------------------+
           | :code:`"image_flags"`               | * LAMMPS dump: :code:`ix`, :code:`iy`, and | `numpy.ndarray[int]`   |
           |                                     |   :code:`iz` attributes                    |                        |
           +-------------------------------------+--------------------------------------------+------------------------+
           | :code:`"dipole_moments"`            | * LAMMPS dump: :code:`mux`, :code:`muy`,   | `numpy.ndarray[float]` |
           |                                     |   and :code:`muz` attributes               |                        |
           +-------------------------------------+--------------------------------------------+------------------------+
           | :code:`"dipole_moments_magnitudes"` | * LAMMPS dump: :code:`mu` attribute        | `float`                |
           +-------------------------------------+--------------------------------------------+------------------------+
           | :code:`"angular_velocities"`        | * LAMMPS dump: :code:`omegax`,             | `numpy.ndarray[float]` |
           |                                     |   :code:`omegay`, and :code:`omegaz`       |                        |
           |                                     |   attributes                               |                        |
           +-------------------------------------+--------------------------------------------+------------------------+
           | :code:`"angular_momenta"`           | * LAMMPS dump: :code:`angmomx`,            | `numpy.ndarray[float]` |
           |                                     |   :code:`angmomy`, and :code:`angmomz`     |                        |
           |                                     |   attributes                               |                        |
           +-------------------------------------+--------------------------------------------+------------------------+
           | :code:`"torques"`                   | * LAMMPS dump: :code:`tqx`, :code:`tqy`,   | `numpy.ndarray[float]` |
           |                                     |   and :code:`tqz` attributes               |                        |
           +-------------------------------------+--------------------------------------------+------------------------+
           | custom                              | * LAMMPS dump: attributes beginning with   | any                    |
           |                                     |   :code:`c_`, :code:`d_`, :code:`d2_`,     |                        |
           |                                     |   :code:`f_`, :code:`i_`, :code:`i2_`, and |                        |
           |                                     |   :code:`v_`                               |                        |
           +-------------------------------------+--------------------------------------------+------------------------+
    """

    def __init__(
        self,
        frame: int,
        time: float,
        *,
        timestep: int = None,
        dimensions: np.ndarray[float] = None,
        n_atoms: int = None,
        ids: np.ndarray[int] = None,
        positions: np.ndarray[float] = None,
        forces: np.ndarray[float] = None,
        velocities: np.ndarray[float] = None,
        **kwargs,
    ) -> None:
        self._frame = frame
        self._n_atoms = n_atoms
        self._time = time
        self._timestep = timestep

        self._dimensions = dimensions
        if self._dimensions is not None:
            self._dimensions = np.asarray(self._dimensions)
            if self._dimensions.ndim != 1 or self._dimensions.shape[0] != 6:
                raise ValueError("Invalid shape for `dimensions`.")

        self._ids = ids
        if self._ids is not None:
            self._ids = np.asarray(self._ids)
            if self._ids.ndim != 1 or self._ids.shape[0] != self._n_atoms:
                raise ValueError("Invalid shape for `ids`.")

        self._positions = positions
        if self._positions is None:
            self._positions = np.asarray(positions)
            if (
                self._positions.ndim != 2
                or self._positions.shape[0] != self._n_atoms
                or self._positions.shape[1] != 3
            ):
                raise ValueError("Invalid shape for `positions`.")

        self._forces = forces
        if self._forces is not None:
            self._forces = np.asarray(self._forces)
            if (
                self._forces.ndim != 2
                or self._forces.shape[0] != self._n_atoms
                or self._forces.shape[1] != 3
            ):
                raise ValueError("Invalid shape for `forces`.")

        self._velocities = velocities
        if self._velocities is not None:
            self._velocities = np.asarray(self._velocities)
            if (
                self._velocities.ndim != 2
                or self._velocities.shape[0] != self._n_atoms
                or self._velocities.shape[1] != 3
            ):
                raise ValueError("Invalid shape for `velocities`.")

        self.extra = kwargs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(frame={self.frame}, "
            f"n_atoms={self._n_atoms}, "
            "positions=<class 'numpy.ndarray>, ...)"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}: "
            f"frame {self._frame}, {self._n_atoms} atoms"
        )

    @property
    def frame(self) -> int:
        """
        Frame number.

        .. container::

           **Sources**:

           * LAMMPS dump: Enumeration.
        """

        return self._frame

    @property
    def timestep(self) -> int:
        """
        Timestep number.

        .. container::

           **Sources**:

           * LAMMPS dump: :code:`ITEM: TIMESTEP` header.
        """

        return self._timestep

    @property
    def time(self) -> float:
        """
        Simulation time.

        .. container::

           **Sources**:

           * LAMMPS dump: :code:`ITEM: TIME` header.

        **Reference unit**: :math:`\\mathrm{ps}`.
        """

        return self._time

    @property
    def dimensions(self) -> np.ndarray[float]:
        """
        Simulation box dimensions (or lattice parameters).

        .. container::

           **Sources**:

           * LAMMPS dump: :code:`ITEM: BOX BOUNDS [...]` header.

        **Reference units**: :math:`\\mathrm{nm}` for lengths and
        degrees (:math:`^\\circ`) for angles.
        """

        return self._dimensions

    @property
    def n_atoms(self) -> int:
        """
        Number of atoms.

        .. container::

           **Sources**:

           * LAMMPS dump: :code:`ITEM: NUMBER OF ATOMS` header.
        """

        return self._n_atoms

    @property
    def ids(self) -> np.ndarray[int]:
        """
        Atom indices or identifiers.

        .. container::

           **Sources**:

           * LAMMPS dump: :code:`id` attribute or enumeration.
        """

        return self._ids

    @property
    def positions(self) -> np.ndarray[float]:
        """
        Atom positions.

        .. container::

           **Sources**:

           * LAMMPS dump: :code:`x[su]`, :code:`y[su]`, and :code:`z[su]`
             attributes.

        **Reference unit**: :math:`\\mathrm{nm}`.
        """

        return self._positions

    @property
    def forces(self) -> np.ndarray[float]:
        """
        Forces exerted on the atoms.

        .. container::

           **Sources**:

           * LAMMPS dump: :code:`fx`, :code:`fy`, and :code:`fz`
             attributes.

        **Reference unit**: :math:`\\mathrm{kJ/(mol\\cdot nm)}`.
        """

        return self._forces

    @property
    def velocities(self) -> np.ndarray[float]:
        """
        Atom velocities.

        .. container::

           **Sources**:

           * LAMMPS dump: :code:`vx`, :code:`vy`, and :code:`vz`
             attributes.

        **Reference unit**: :math:`\\mathrm{nm/ps}`.
        """

        return self._velocities
