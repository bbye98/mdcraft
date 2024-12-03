"""
Topology and trajectory readers
===============================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains topology and trajectory readers for use with MDAnalysis.
"""

import concurrent.futures
import os
from pathlib import Path
from typing import TextIO, Union
import warnings

import psutil
from MDAnalysis.coordinates.base import ReaderBase
from MDAnalysis.lib.distances import transform_StoR
from MDAnalysis.lib.util import anyopen, store_init_arguments
import numpy as np

from ..algorithm.topology import convert_cell_representation


class LAMMPSDumpTrajectoryReader(ReaderBase):
    r"""
    LAMMPS dump trajectory reader.

    Parameters
    ----------
    filename : `str`
        Filename of the LAMMPS dump file.

    conventions : `str` or `tuple`, optional
        Coordinate conventions for the :math:`x`, :math:`y`, and
        :math:`z`-positions. If a `str` is provided, the same value is
        used for all axes. Determined automatically if not provided.

        .. container::

           **Valid values:**

           * :code:`None` to ignore an axis (when coordinate information
             is unavailable in that axis).
           * :code:`""` for unscaled and wrapped coordinates.
           * :code:`"u"` for unscaled and unwrapped coordinates.
           * :code:`"s"` for scaled and wrapped coordinates.
           * :code:`"su"` for scaled and unwrapped coordinates.

    unwrap : `bool`, default: :code:`False`
        Determines whether atom positions are unwrapped.

    extras : `list`, optional
        Extra per-atom information to be read from the dump file.
        Topology attributes, like type, mass, and charge, are not
        expected to change and are not supported.

        .. note::

           For named LAMMPS vector attributes, the stored array will
           always be arrays of shape :math:`(N,\,3)`, where :math:`N` is
           the number of atoms. If data for an axis is not found in the
           dump file, the corresponding column in the array will be
           filled with zeros.

           For custom LAMMPS vector attributes, the stored array will
           have as many columns as there are instances of the attribute
           in the dump file. For example, if :code:`"c_custom"` was
           provided to `extras` and the dump file contains the
           instances :code:`c_custom[2]`, :code:`c_custom[4]`, and
           :code:`c_custom[1]`, in that order, the array will have shape
           :math:`(N,\,3)`. The instances are always sorted, such that
           the columns in the example array correspond to
           :code:`c_custom[1]`, :code:`c_custom[2]`, and
           :code:`c_custom[4]`, respectively.

        **Valid values:**

        ==================================  ===================================================  ======================
        Keyword (`Reader.data` key)         LAMMPS dump attribute(s)                             Type
        ==================================  ===================================================  ======================
        :code:`"dipole_moments"`            (:code:`mux`, :code:`muy`, :code:`muz`)              `numpy.ndarray[float]`
        :code:`"dipole_moment_magnitudes"`  :code:`mu`                                           `float`
        :code:`"angular_velocities"`        (:code:`omegax`, :code:`omegay`, :code:`omegaz`)     `numpy.ndarray[float]`
        :code:`"angular_momentums"`         (:code:`angmomx`, :code:`angmomy`, :code:`angmomz`)  `numpy.ndarray[float]`
        :code:`"torques"`                   (:code:`tqx`, :code:`tqy`, :code:`tqz`)              `numpy.ndarray[float]`
        :code:`"c_{compute_id}"`            (:code:`c_{compute_id}[i]`, ...)                     `numpy.ndarray[float]`
        :code:`"d_{name}"`                  (:code:`d_{name}[i]`, ...)                           `numpy.ndarray[float]`
        :code:`"d2_{name}[i]"`              (:code:`d2_{name}[i][j]`, ...)                       `numpy.ndarray[float]`
        :code:`"f_{fix_id}"`                (:code:`f_{fix_id}[i]`, ...)                         `numpy.ndarray[float]`
        :code:`"i_{name}"`                  (:code:`i_{name}[i]`, ...)                           `numpy.ndarray[int]`
        :code:`"i2_{name}[i]"`              (:code:`i2_{name}[i][j]`, ...)                       `numpy.ndarray[int]`
        :code:`"v_{name}"`                  (:code:`v_{name}[i]`, ...)                           `numpy.ndarray[float]`
        ==================================  ===================================================  ======================

    Examples
    --------
    First, this trajectory reader must be registered to MDAnalysis by
    importing this submodule or this class:

    >>> from mdcraft.analysis import reader

    This will overwrite the built-in MDAnalysis LAMMPS dump reader.

    Then, to read a LAMMPS dump file :code:`simulation.lammpsdump` and
    extract both the topology and trajectory:

    >>> universe = mda.Universe("simulation.lammpsdump")

    If the dump file contains extra information, like the per-atom
    dipole moments, it can be specified in the :code:`extras` argument:

    >>> universe = mda.Universe("simulation.lammpsdump", extras=["dipole_moments"])

    The extra information will be stored in the :code:`data` attribute of
    the trajectory object:

    >>> dipole_moments = universe.trajectory.data["dipole_moments"]

    If the dump file does not have the :code:`.lammpsdump` extension,
    the format can be specified in the :code:`format` argument:

    >>> universe = mda.Universe("simulation.dump", format="LAMMPSDUMP")
    """

    _CONVENTIONS = ["u", "su", "", "s"]
    _CUSTOM_ATTRIBUTE_PREFIXES = ("c_", "d_", "d2_", "f_", "i_", "i2_", "v_")
    _EXTRA_ATTRIBUTES = {
        "dipole_moments": ("mux", "muy", "muz"),
        "dipole_moment_magnitudes": ("mu",),
        "angular_velocities": ("omegax", "omegay", "omegaz"),
        "angular_momentums": ("angmomx", "angmomy", "angmomz"),
        "torques": ("tqx", "tqy", "tqz"),
    }

    format = "LAMMPSDUMP"

    @staticmethod
    def _get_offsets(
        file: Union[str, Path, TextIO],
        start: int,
        end: int,
        grid: bool,
        parallel: bool,
    ) -> list[int]:
        """
        Finds the byte offsets for frames in a LAMMPS dump file.

        Parameters
        ----------
        file : `str`, `pathlib.Path`, or `io.TextIO`
            Filename, path, or handle to the LAMMPS dump file.

        start : `int`
            Byte offset to start reading from.

        end : `int`
            Byte offset to stop reading at.

        grid : `bool`
            Specifies whether the file uses the :code:`style grid`
            format.

        parallel : `bool`
            Determines whether the file is read in parallel.
        """

        manual = isinstance(file, (str, Path))
        if manual:
            file = open(file, "r")
        byte_counter = file.seek(start)

        offsets = []
        if parallel:
            while byte_counter < end:
                line = file.readline()
                byte_counter += len(line)
                if (line_index := line.find("ITEM: TIMESTEP")) != -1:
                    offsets.append(byte_counter - len(line) + line_index)
                    break
            while byte_counter < end:
                byte_counter += len(file.readline())  # <timestep>
                byte_counter += len(file.readline())  # ITEM: NUMBER OF ATOMS
                n_atoms = file.readline()
                byte_counter += len(n_atoms)
                for _ in range(5 + 4 * grid + int(n_atoms)):
                    byte_counter += len(file.readline())
                    if byte_counter >= end:
                        break
                else:
                    offsets.append(byte_counter)
                byte_counter += len(file.readline())  # ITEM: TIMESTEP
        else:
            line_counter = 0
            lines_per_frame = 9 + 4 * grid
            line = True
            while line:
                if (relative_line := line_counter % lines_per_frame) == 0 and (
                    offset := file.tell()
                ) < end:
                    offsets.append(offset)
                elif relative_line == 4:
                    lines_per_frame = 9 + 4 * grid + int(line)
                line = file.readline()
                line_counter += 1

        if manual:
            file.close()

        return offsets

    @store_init_arguments
    def __init__(
        self,
        filename: str,
        conventions: Union[str, tuple[str]] = None,
        unwrap: bool = False,
        *,
        extras: list[str] = None,
        parallel: bool = False,
        n_threads: int = None,
        **kwargs,
    ) -> None:

        super().__init__(filename, **kwargs)

        if conventions is not None:
            if isinstance(conventions, str):
                if conventions not in self._CONVENTIONS:
                    emsg = (
                        f"Invalid convention '{conventions}'. Valid "
                        "values: '" + "', '".join(self._CONVENTIONS) + "'."
                    )
                    raise ValueError(emsg)
                self._conventions = 3 * [conventions]
            elif len(conventions) != 3:
                emsg = "'conventions' must have length 3 when it is an array."
                raise ValueError(emsg)
            else:
                for c in conventions:
                    if c not in self._CONVENTIONS:
                        emsg = (
                            f"Invalid convention '{c}'. Valid values: "
                            "'" + "', '".join(self._CONVENTIONS) + "'."
                        )
                        raise ValueError(emsg)
                self._conventions = conventions
        else:
            self._conventions = []

        with anyopen(self.filename) as f:
            for _ in range(3):
                f.readline()
            self.n_atoms = int(f.readline())

            for _ in range(4):
                f.readline()
            self.is_style_grid = f.readline().rstrip() == "ITEM: DIMENSION"

        file_size = os.path.getsize(filename)
        if parallel:
            n_threads = n_threads or psutil.cpu_count()
            chunk_size = np.ceil(file_size / n_threads).astype(int)
            self._offsets = []
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_threads
            ) as executor:
                for future in concurrent.futures.as_completed(
                    executor.submit(
                        self._get_offsets,
                        filename,
                        i * chunk_size,
                        min((i + 1) * chunk_size, file_size),
                        self.is_style_grid,
                        True,
                    )
                    for i in range(n_threads)
                ):
                    self._offsets.extend(future.result())
            self._offsets.sort()
        else:
            self._file = open(filename, "r")
            self._offsets = self._get_offsets(
                self._file, 0, file_size, self.is_style_grid, False
            )

        if isinstance(extras, str):
            if extras not in self._EXTRA_ATTRIBUTES and not extras.startswith(
                self._CUSTOM_ATTRIBUTE_PREFIXES
            ):
                raise ValueError(f"Invalid attribute '{extras}' in 'extras'.")
            extras = [extras]
        elif extras is not None:
            for attr in extras:
                if attr not in self._EXTRA_ATTRIBUTES and not attr.startswith(
                    self._CUSTOM_ATTRIBUTE_PREFIXES
                ):
                    raise ValueError(f"Invalid attribute '{attr}' in 'extras'.")
        self._extras = extras

        self._unwrap = unwrap

        self._reopen()
        self._read_next_timestep()

    def _reopen(self) -> None:
        """
        Reopens the LAMMPS dump file.
        """

        self.close()
        self._file = anyopen(self.filename)
        self.ts = self._Timestep(self.n_atoms, **self._ts_kwargs)
        self.ts.frame = -1

    def _read_frame(self, frame: int) -> ReaderBase._Timestep:
        """
        Reads a specific frame from the LAMMPS dump file.

        Parameters
        ----------
        frame : `int`, optional
            Frame to read.

        Returns
        -------
        timestep : `ReaderBase._Timestep`
            Timestep object with all information from a frame in the
            trajectory.
        """

        self._file.seek(self._offsets[frame])
        self.ts.frame = frame - 1
        return self._read_next_timestep()

    def _read_next_timestep(self, ts: int = None) -> ReaderBase._Timestep:
        """
        Reads the next timestep from the LAMMPS dump file.

        Returns
        -------
        timestep : `ReaderBase._Timestep`
            Timestep object with all information from a frame in the
            trajectory.
        """

        # Set up timestep
        ts = self.ts
        ts.frame += 1
        if ts.frame >= self.n_frames:
            emsg = (
                f"'{self.filename}' only has {self.n_frames} frames "
                f"and does not contain a frame {ts.frame}."
            )
            raise EOFError(emsg)

        # Get file and read timestep information
        f = self._file
        f.readline()  # ITEM: TIMESTEP
        ts.data["step"] = int(f.readline())
        ts.data["time"] = ts.data["step"] * ts.dt

        # Read number of atoms
        f.readline()  # ITEM: NUMBER OF ATOMS
        if (n_atoms := int(f.readline())) != self.n_atoms:
            emsg = (
                f"Timestep {ts.data['step']} has {n_atoms} atoms, "
                f"but the topology has {self.n_atoms} atoms."
            )
            raise RuntimeError(emsg)

        # Read box information
        if "xy xz yz" in (_ := f.readline()):  # ITEM: BOX BOUNDS
            xlo, xhi, xy = (float(v) for v in f.readline().split())
            ylo, yhi, xz = (float(v) for v in f.readline().split())
            zlo, zhi, yz = (float(v) for v in f.readline().split())

            # https://docs.lammps.org/Howto_triclinic.html\
            # #output-of-restricted-and-general-triclinic-boxes-in-a-dump-file
            xlo -= min(0, xy, xz, xy + xz)
            xhi -= max(0, xy, xz, xy + xz)
            ylo -= min(0, yz)
            yhi -= max(0, yz)

            box_vectors = np.zeros((3, 3))
            box_vectors[0, 0] = xhi - xlo
            box_vectors[1, :2] = xy, yhi - ylo
            box_vectors[2] = xz, yz, zhi - zlo
            ts.dimensions = convert_cell_representation(vectors=box_vectors)
        elif "abc origin" in _:
            box_vectors = np.empty((3, 3))
            for i in range(3):
                box_vectors[i] = (float(v) for v in f.readline().split()[:3])
            ts.dimensions = convert_cell_representation(vectors=box_vectors)
        else:
            xlo, xhi = (float(v) for v in f.readline().split())
            ylo, yhi = (float(v) for v in f.readline().split())
            zlo, zhi = (float(v) for v in f.readline().split())
            ts.dimensions = np.array((xhi - xlo, yhi - ylo, zhi - zlo, 90, 90, 90))

        if self.is_style_grid:
            for _ in range(4):
                f.readline()

        # Get all attributes in dump file
        attributes = {a: i for i, a in enumerate(f.readline().rstrip().split()[2:])}

        # Get coordinate axis indices and verify conventions
        self._axis_indices = np.empty(3, dtype=int)
        if self._conventions:
            for i, (ax, c) in enumerate(zip("xyz", self._conventions)):
                if c is None:
                    self._axis_indices[i] = -1
                else:
                    if (col := attributes.get(f"{ax}{c}")) is None:
                        emsg = (
                            f"No {ax}-coordinate information with the "
                            f"specified convention '{c}' found in "
                            f"'{self.filename}'."
                        )
                        raise RuntimeError(emsg)
                    self._axis_indices[i] = col
        else:
            for i, ax in enumerate("xyz"):
                for c in self._CONVENTIONS:
                    if (n := f"{ax}{c}") in attributes:
                        self._axis_indices[i] = attributes[n]
                        self._conventions.append(c)
                        break
                else:
                    self._axis_indices[i] = -1
                    self._conventions.append(None)
        self._scaled_where = np.asarray(
            [i for i, c in enumerate(self._conventions) if "s" in c], dtype=int
        )
        self._has_scaled = np.any(self._scaled_where)

        # Get image flag indices if unwrapping is desired
        if self._unwrap:
            self._image_indices = np.fromiter(
                (
                    -1 if ai == -1 or "u" in c else (attributes.get(f"i{ax}") or -1)
                    for ax, ai, c in zip("xyz", self._axis_indices, self._conventions)
                ),
                dtype=int,
                count=3,
            )
            if np.any(self._image_indices >= 0):
                self._image_where = np.where(self._image_indices >= 0)[0]
            else:
                wmsg = "No image flags found in dump file. " "Setting 'unwrap=False'."
                warnings.warn(wmsg)
                self._unwrap = False

        # Check if velocity and force information is available
        self._velocity_indices = np.fromiter(
            (attributes.get(f"v{ax}") or -1 for ax in "xyz"), dtype=int, count=3
        )
        ts.has_velocities = np.any(self._velocity_indices)
        self._force_indices = np.fromiter(
            (attributes.get(f"f{ax}") or -1 for ax in "xyz"), dtype=int, count=3
        )
        ts.has_forces = np.any(self._force_indices)

        # Preallocate arrays for extra information, if requested
        if self._extras:
            self._extras_indices = {}
            for attr in self._extras:
                if attr.startswith(self._CUSTOM_ATTRIBUTE_PREFIXES):
                    la = len(attr)
                    self._extras_indices[attr] = [
                        j
                        for _, j in sorted(
                            (int(a[a.find("[", la) + 1 : a.find("]", la)]), i)
                            for a, i in attributes.items()
                            if a.startswith(attr)
                        )
                    ]
                    ts.data[attr] = np.empty(
                        (n_atoms, len(self._extras_indices[attr])),
                        dtype=np.int64 if attr.startswith("i_") else np.float32,
                    )
                else:
                    self._extras_indices[attr] = [
                        attributes.get(a) or -1 for a in self._EXTRA_ATTRIBUTES[attr]
                    ]
                    ts.data[attr] = np.empty(
                        (n_atoms, len(self._extras_indices[attr])), dtype=np.float32
                    )

        # Iterate over atoms and read their information
        if has_id := "id" in attributes:
            atom_indices = np.empty(self.n_atoms, dtype=int)
        for i in range(self.n_atoms):
            values = f.readline().split()

            # Store atom ID
            if has_id:
                atom_indices[i] = values[attributes["id"]]

            # Store atom positions (missing dimensions are left as 0)
            ts.positions[i] = tuple(
                0 if ai == -1 else values[ai] for ai in self._axis_indices
            )

            # Unscale coordinates before unwrapping, if necessary
            if self._has_scaled:
                ts.positions[i, self._scaled_where] = transform_StoR(
                    ts.positions[i], ts.dimensions
                )[self._scaled_where]

            # Unwrap coordinates, if necessary
            if self._unwrap:
                ts.positions[i, self._image_where] += ts.dimensions[
                    self._image_where
                ] * tuple(
                    int(values[ii]) for ii in self._image_indices[self._image_where]
                )

            # Shift unwrapped and/or unscaled coordinates by the origin
            if self._has_scaled or self._unwrap:
                ts.positions += (xlo, ylo, zlo)

            # Store atom velocities and forces, if available
            if ts.has_velocities:
                ts.velocities[i] = tuple(
                    0 if vi == -1 else values[vi] for vi in self._velocity_indices
                )
            if ts.has_forces:
                ts.forces[i] = tuple(
                    0 if fi == -1 else values[fi] for fi in self._force_indices
                )

            # Store extra information, if requested
            if self._extras:
                for attr, attribute_indices in self._extras_indices.items():
                    if attr.startswith(self._CUSTOM_ATTRIBUTE_PREFIXES):
                        ts.data[attr][i] = np.fromiter(
                            (values[ai] for ai in attribute_indices),
                            dtype=np.int64 if attr.startswith("i_") else np.float32,
                            count=len(attribute_indices),
                        )
                    else:
                        ts.data[attr][i] = tuple(
                            0 if ai == -1 else float(values[ai])
                            for ai in attribute_indices
                        )

        # Reorder atoms by ID, if available
        if has_id:
            order = np.argsort(atom_indices)
            ts.positions = ts.positions[order]
            if ts.has_velocities:
                ts.velocities = ts.velocities[order]
            if ts.has_forces:
                ts.forces = ts.forces[order]
            if self._extras:
                for attr in self._extras:
                    ts.data[attr] = ts.data[attr][order]

        return ts

    @property
    def n_frames(self) -> int:
        return len(self._offsets)

    def close(self) -> None:
        """
        Closes the LAMMPS dump file.
        """

        if hasattr(self, "_file"):
            self._file.close()
