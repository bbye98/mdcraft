from __future__ import annotations
from collections import defaultdict
import concurrent.futures
import importlib.util
from pathlib import Path
from typing import Any, TextIO
import warnings

import numpy as np
import pandas as pd

from . import INTERNAL_UNITS
from .base import BaseTopologyReader, BaseTrajectoryReader
from .. import ureg, U_
from ..utility.topology import convert_cell_representation, scale_triclinic_coordinates

FOUND_MDANALYSIS = importlib.util.find_spec("MDAnalysis") is not None


class LAMMPSDataReader(BaseTopologyReader):  # TODO
    """ """

    _ATOM_ATTRIBUTES = {
        "atom-ID": "ids",
        "molecule-ID": "molecule_ids",
        "atom-type": "types",
        ("q", "charge"): "charges",
        "mass": "masses",
        ("x", "y", "z"): "positions",
        ("bodyflag", "ellipsoidflag", "lineflag", "triangleflag"): "flags",
        "diameter": "sizes",
        ("rho", "density"): "densities",
        ("mux", "muy", "muz"): "dipole_moments",
    }

    _ATOM_STYLES = {
        "amoeba": ["atom-ID", "molecule-ID", "atom-type", "q", "x", "y", "z"],
        "angle": ["atom-ID", "molecule-ID", "atom-type", "x", "y", "z"],
        "atomic": ["atom-ID", "atom-type", "x", "y", "z"],
        "body": ["atom-ID", "atom-type", "bodyflag", "mass", "x", "y", "z"],
        "bond": ["atom-ID", "molecule-ID", "atom-type", "x", "y", "z"],
        "bpm/sphere": [
            "atom-ID",
            "molecule-ID",
            "atom-type",
            "diameter",
            "density",
            "x",
            "y",
            "z",
        ],
        "charge": ["atom-ID", "atom-type", "q", "x", "y", "z"],
        "dielectric": [
            "atom-ID",
            "atom-type",
            "q",
            "x",
            "y",
            "z",
            "mux",
            "muy",
            "muz",
            "area",
            "ed",
            "em",
            "epsilon",
            "curvature",
        ],
        "dipole": ["atom-ID", "atom-type", "q", "x", "y", "z", "mux", "muy", "muz"],
        "dpd": ["atom-ID", "atom-type", "theta", "x", "y", "z"],
        "edpd": ["atom-ID", "atom-type", "edpd_temp", "edpd_cv", "x", "y", "z"],
        "electron": ["atom-ID", "atom-type", "q", "espin", "eradius", "x", "y", "z"],
        "ellipsoid": [
            "atom-ID",
            "atom-type",
            "ellipsoidflag",
            "density",
            "x",
            "y",
            "z",
        ],
        "full": ["atom-ID", "molecule-ID", "atom-type", "q", "x", "y", "z"],
        "line": [
            "atom-ID",
            "molecule-ID",
            "atom-type",
            "lineflag",
            "density",
            "x",
            "y",
            "z",
        ],
        "mdpd": ["atom-ID", "atom-type", "rho", "x", "y", "z"],
        "molecular": ["atom-ID", "molecule-ID", "atom-type", "x", "y", "z"],
        "oxdna": ["atom-ID", "atom-type", "x", "y", "z"],
        "peri": ["atom-ID", "atom-type", "volume", "density", "x", "y", "z"],
        "rheo": ["atom-ID", "atom-type", "status", "rho", "x", "y", "z"],
        "rheo/thermal": [
            "atom-ID",
            "atom-type",
            "status",
            "rho",
            "energy",
            "x",
            "y",
            "z",
        ],
        "smd": [
            "atom-ID",
            "atom-type",
            "molecule-ID",
            "volume",
            "mass",
            "kradius",
            "cradius",
            "x0",
            "y0",
            "z0",
            "x",
            "y",
            "z",
        ],
        "sph": ["atom-ID", "atom-type", "rho", "esph", "cv", "x", "y", "z"],
        "sphere": ["atom-ID", "atom-type", "diameter", "density", "x", "y", "z"],
        "spin": ["atom-ID", "atom-type", "x", "y", "z", "spx", "spy", "spz", "sp"],
        "tdpd": ["atom-ID", "atom-type", "x", "y", "z", "cc*"],
        "template": [
            "atom-ID",
            "atom-type",
            "molecule-ID",
            "template-index",
            "template-atom",
            "x",
            "y",
            "z",
        ],
        "tri": [
            "atom-ID",
            "molecule-ID",
            "atom-type",
            "triangleflag",
            "density",
            "x",
            "y",
            "z",
        ],
        "wavepacket": [
            "atom-ID",
            "atom-type",
            "charge",
            "espin",
            "eradius",
            "etag",
            "cs_re",
            "cs_im",
            "x",
            "y",
            "z",
        ],
    }
    _EXTENSIONS = {".data"}
    _FORMAT = "LAMMPSDATA"
    _HEADER_KEYWORDS = {
        "atoms",
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
        "atom types",
        "bond types",
        "angle types",
        "dihedral types",
        "improper types",
        "ellipsoids",
        "lines",
        "triangles",
        "bodies",
        "xlo xhi",
        "ylo yhi",
        "zlo zhi",
        "xy xz yz",
        "avec",
        "bvec",
        "cvec",
        "abc origin",
    }
    _SECTION_KEYWORDS = {
        "Atoms": "atoms",
        "Velocities": "atoms",
        "Masses": "atom_types",
        "Ellipsoids": "ellipsoids",
        "Lines": "lines",
        "Triangles": "triangles",
        "Bodies": "bodies",
        "Bonds": "bonds",
        "Angles": "angles",
        "Dihedrals": "dihedrals",
        "Impropers": "impropers",
        "Atom Type Labels": "atom_types",
        "Bond Type Labels": "bond_types",
        "Angle Type Labels": "angle_types",
        "Dihedral Type Labels": "dihedral_types",
        "Improper Type Labels": "improper_types",
        "Pair Coeffs": "atom_types",
        "PairIJ Coeffs": None,
        "Bond Coeffs": "bond_types",
        "Angle Coeffs": "angle_types",
        "Dihedral Coeffs": "dihedral_types",
        "Improper Coeffs": "improper_types",
        "BondBond Coeffs": "angle_types",
        "BondAngle Coeffs": "angle_types",
        "MiddleBondTorsion Coeffs": "dihedral_types",
        "EndBondTorsion Coeffs": "dihedral_types",
        "AngleTorsion Coeffs": "dihedral_types",
        "AngleAngleTorsion Coeffs": "dihedral_types",
        "BondBond13 Coeffs": "dihedral_types",
        "AngleAngle Coeffs": "improper_types",
    }

    def __init__(
        self,
        filename: str | Path,
        /,
        atom_style: str | list[str] | None = None,
        *,
        reduced: bool = False,
        parallel: bool = False,
        n_workers: int | None = None,
    ) -> None:
        super().__init__(
            filename,
            parallel=parallel,
            n_workers=n_workers,
        )

        # Create and store handle to file
        self.open()

        # Skip first line
        self._file.readline()

        # Read header and get topology summary and simulation box size information
        # NOTE: https://docs.lammps.org/read_data.html
        self._n_ = defaultdict(int)
        self._dimensions = {}
        while line := self._file.readline():
            if line := line.rstrip():
                if line in self._SECTION_KEYWORDS:
                    break
                *value, header = line.split(
                    maxsplit=1
                    + ("lo" in line)
                    + 2 * any(s in line for s in ("xy", "vec", "abc"))
                )
                if header not in self._HEADER_KEYWORDS:
                    raise RuntimeError(
                        f"Invalid LAMMPS data file '{self._filename.name}'. "
                        f"Unknown header keyword '{header}'."
                    )
                if len(value) == 1:
                    self._n_[header.replace(" ", "_")] = int(value[0])
                else:
                    self._dimensions[header] = np.array(value, dtype=float)

        if self._n_["atoms"] == 0:
            raise RuntimeError(
                f"Invalid LAMMPS data file '{self._filename.name}'. No atoms found."
            )

        if "xlo xhi" in self._dimensions:
            xlo, xhi = self._dimensions["xlo xhi"]
            try:
                ylo, yhi = self._dimensions["ylo yhi"]
                zlo, zhi = self._dimensions["zlo zhi"]
            except KeyError:
                raise RuntimeError(
                    f"Invalid LAMMPS data file '{self._filename.name}'. "
                    "Incomplete simulation box size information."
                )
            box_vectors = np.diag((xhi - xlo, yhi - ylo, zhi - zlo))
            if "xy xz yz" in self._dimensions:
                *box_vectors[1:, 0], box_vectors[2, 1] = self._dimensions["xy xz yz"]
        elif "avec" in self._dimensions:
            box_vectors = np.stack(
                (
                    self._dimensions["avec"],
                    self._dimensions["bvec"],
                    self._dimensions["cvec"],
                )
            )
        self._dimensions = convert_cell_representation(box_vectors, "parameters")

        # Find all sections
        file_size = self._filename.stat().st_size
        start = self._file.tell() - len(line) - 1
        if self._parallel:
            chunk_size = np.ceil((file_size - start) / self._n_workers).astype(int)
            self._offsets = {}
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self._n_workers
            ) as executor:
                for future in concurrent.futures.as_completed(
                    executor.submit(
                        self._find_sections,
                        self._filename,
                        i * chunk_size,
                        min((i + 1) * chunk_size, file_size),
                    )
                    for i in range(self._n_workers)
                ):
                    offsets = future.result()
                    if offsets is not None:
                        self._offsets |= offsets
        else:
            self._offsets = self._find_sections(self._file, start, file_size)

        # Figure out and store atom style
        self._file.seek(self._offsets["Atoms"])
        self._file.readline()
        self._file.readline()
        n_attributes = len(self._file.readline().split())
        if atom_style is None:
            if n_attributes == 5:
                self._atom_style = "atomic"
            elif n_attributes == 7:
                self._atom_style = "full"
            else:
                raise RuntimeError(
                    "Could not determine atom style for "
                    f"'{self._filename.name}'. Use the `atom_style` "
                    "parameter to specify it."
                )
            warnings.warn(
                f"Atom style for '{self._filename.name}' assumed to "
                f"be '{self._atom_style}'."
            )
            self._atom_style = self._ATOM_STYLES[self._atom_style]
        else:
            atom_attributes = set(
                attribute
                for attributes in self._ATOM_STYLES.values()
                for attribute in attributes
            )
            self._atom_style = []
            if isinstance(atom_style, str):
                atom_style = atom_style.split()
            for style in atom_style:
                if style in self._ATOM_STYLES:
                    for attribute in self._ATOM_STYLES[style]:
                        if attribute not in self._atom_style:
                            self._atom_style.append(attribute)
                elif (
                    style in atom_attributes or style.startswith("cc")
                ) and style not in self._atom_style:
                    self._atom_style.append(style)
            if len(self._atom_style) != n_attributes:
                raise RuntimeError(
                    f"Invalid atom style for '{self._filename.name}'. "
                    f"Expected {n_attributes} attributes, but got "
                    f"{len(self._atom_style)}: {self._atom_style}."
                )

        # Store settings
        self._reduced = reduced

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"('{self._filename.name}', reduced={self._reduced}, "
            f"parallel={self._parallel}, n_workers={self._n_workers})"
        )

    def _find_sections(
        self, file: str | Path | TextIO, start: int, end: int
    ) -> dict[str, int]:
        """
        Finds all sections in the LAMMPS data file.

        Parameters
        ----------
        file : `str`, `pathlib.Path`, or `io.TextIO`
            Filename, path, or handle to the data file.

        start : `int`
            Byte offset to start reading from.

        end : `int`
            Byte offset to stop reading at.

        Returns
        -------
        sections : `dict`
            Sections found in the data file and their starting
            byte offsets.
        """

        manual = isinstance(file, (str, Path))
        if manual:
            file = open(file, "r")
        byte_counter = file.seek(start)

        sections = {}
        while byte_counter < end:
            line = file.readline()
            byte_counter += len(line)
            if (l := line.rstrip()) in self._SECTION_KEYWORDS:
                sections[l] = byte_counter - len(line)
        return sections

    def _parse_section(
        self, file: TextIO, section: str
    ) -> dict[str, np.ndarray[int | float]]:

        file.seek(self._offsets[section])
        n_lines = (
            (self.n_atom_types + 1) * self.n_atom_types // 2
            if section == "PairIJ Coeffs"
            else self._n_[self._SECTION_KEYWORDS[section]]
        )
        file.readline()  # <header keyword>
        file.readline()  # blank line

        section_data = {}
        if "Labels" in section:
            section_data["types"] = np.empty(n_lines, dtype=int)
            section_data["labels"] = np.empty(n_lines, dtype=str)
            for i in range(n_lines):
                section_data["types"][i], section_data["labels"][i] = (
                    file.readline().split()
                )
        else:
            # data = np.loadtxt(file, max_rows=n_lines)
            data = pd.read_csv(file, sep="\\s+", header=None, nrows=n_lines).to_numpy()
            if section == "Atoms":
                for i, attribute in enumerate(self._atom_style):
                    if attribute.endswith("*"):
                        section_data[attribute[:-1]] = data[
                            :,
                            i : (
                                -n_columns_left
                                if (n_columns_left := len(self._atom_style) - i - 1)
                                else None
                            ),
                        ]
                    else:
                        section_data[attribute] = data[:, i]
            elif section == "Bodies":
                debug = True
            elif section == "Ellipsoids":
                section_data["ID"] = data[:, 0].astype(int)
                section_data["shape"] = data[:, 1:4]
                section_data["quaternion"] = data[:, 4:]
            elif section == "Lines":
                section_data["ID"] = data[:, 0].astype(int)
                section_data["point1"] = data[:, 1:3]
                section_data["point2"] = data[:, 3:]
            elif section == "Masses":
                section_data["atom-type"] = data[:, 0].astype(int)
                section_data["mass"] = data[:, 1]
            elif section == "Triangles":
                section_data["ID"] = data[:, 0].astype(int)
                section_data["corner1"] = data[:, 1:4]
                section_data["corner2"] = data[:, 4:7]
                section_data["corner3"] = data[:, 7:]
            elif section == "Velocities":
                debug = True
            elif "Coeffs" in section:
                if "IJ" in section:
                    section_data["atom-type1"] = data[:, 0].astype(int)
                    section_data["atom-type2"] = data[:, 1].astype(int)
                    section_data["coeffs"] = data[:, 2:]
                else:
                    section_data["atom-type"] = data[:, 0].astype(int)
                    section_data["coeffs"] = data[:, 1:]
            else:  # Bonds, Angles, Dihedrals, Impropers
                section_data["ID"] = data[:, 0].astype(int)
                section_data["type"] = data[:, 1]
                section_data["atom-types"] = data[:, 2:].astype(int)

        return section_data

    def _parse_topology(
        self, file: TextIO, convert_units: bool = True, parallel: bool | None = None
    ) -> dict[str, dict[str, np.ndarray[int | float]]]:

        sections = {}

        for section in self._offsets:
            sections[section] = self._parse_section(file, section)

        topology = {}

        return topology

    @property
    def dimensions(self) -> np.ndarray[float] | None:
        """
        Simulation box dimensions (or lattice parameters). If `None`,
        the system size could not be determined from the topology.

        **Reference units**: :math:`\\mathrm{nm}` for lengths and
        degrees (:math:`^\\circ`) for angles.
        """

        return self._dimensions

    @property
    def n_atoms(self) -> int:
        """
        Number of atoms.
        """

        return self._n_["atoms"]

    @property
    def n_bonds(self) -> int:
        """
        Number of bonds.
        """

        return self._n_["bonds"]

    @property
    def n_angles(self) -> int:
        """
        Number of angles.
        """

        return self._n_["angles"]

    @property
    def n_dihedrals(self) -> int:
        """
        Number of proper dihedrals.
        """

        return self._n_["dihedrals"]

    @property
    def n_improper_dihedrals(self) -> int:
        """
        Number of improper dihedrals.
        """

        return self._n_["impropers"]

    @property
    def n_atom_types(self) -> int:
        """
        Number of atom types.
        """

        return self._n_["atom_types"]

    @property
    def n_bond_types(self) -> int:
        """
        Number of bond types.
        """

        return self._n_["bond_types"]

    @property
    def n_angle_types(self) -> int:
        """
        Number of angle types.
        """

        return self._n_["angle_types"]

    @property
    def n_dihedral_types(self) -> int:
        """
        Number of proper dihedral types.
        """

        return self._n_["dihedral_types"]

    @property
    def n_improper_dihedral_types(self) -> int:
        """
        Number of improper dihedral types.
        """

        return self._n_["improper_types"]

    @property
    def n_ellipsoids(self) -> int:
        """
        Number of ellipsoids.
        """

        return self._n_["ellipsoids"]

    @property
    def n_lines(self) -> int:
        """
        Number of lines.
        """

        return self._n_["lines"]

    @property
    def n_triangles(self) -> int:
        """
        Number of triangles.
        """

        return self._n_["triangles"]

    @property
    def n_bodies(self) -> int:
        """
        Number of bodies.
        """

        return self._n_["bodies"]

    @property
    def n_residues(self) -> int:
        """
        Number of residues.
        """

        raise NotImplementedError

    @property
    def n_chains(self) -> int:
        """
        Number of chains.
        """

        raise NotImplementedError

    def open(self) -> None:
        """
        Opens the LAMMPS data file and stores a handle to it.
        """

        self._file = open(self._filename, "r")

    def close(self) -> None:
        """
        Closes the LAMMPS data file and deletes the handle.
        """

        if hasattr(self, "_file"):
            self._file.close()
            del self._file


class LAMMPSDumpReader(BaseTrajectoryReader):
    """
    LAMMPS dump file reader.

    This class is not a full-featured LAMMPS dump file reader, but can
    process most, if not all, dump files written using a single
    :code:`dump` command in the text (not binary or gzip) format and
    with :code:`atom`, :code:`custom`, :code:`grid`, or :code:`local`
    styles. Notably, it supports

    * frame headers with units and/or time information (using
      :code:`dump_modify units yes` and/or
      :code:`dump_modify time yes`, respectively),
    * frames with different numbers of atoms/entities,
    * orthogonal, restricted triclinic, and general triclinic simulation
      boxes,
    * any combination of (un)scaled and (un)wrapped coordinates, and
    * all standard and custom attributes.

    Furthermore, it can read in parallel, which can be much faster for
    parsing large dump files and performing analyses on multiple frames
    in those dump files.

    .. important::

       If the :code:`dump_modify` command is used to alter the output in
       the dump file, the :code:`colname` and :code:`header` keywords
       must *not* be used. If the dump file is written in the
       :code:`local` style, the :code:`label ATOMS` option must also
       *not* be used.

    .. note::

       Frames in a dump file are ordered by their timestep numbers and
       are expected to proceed forward in time. If the
       :code:`reset_timestep` command is used in the LAMMPS input script
       used to generate the dump file, this trajectory reader may behave
       unexpectedly.

    .. seealso::

       For more information on the LAMMPS dump file format, see the
       `dump command <https://docs.lammps.org/dump.html>`_ page of the
       LAMMPS documentation.

    Parameters
    ----------
    filename : `str` or `pathlib.Path`, positional-only
        Filename or path to the dump file.

    coordinate_formats : `str` or `list` of `str`, optional
        Coordinate formats. If a `str` is provided, it is used for
        all axes. If the trajectory does not contain particle
        positions for a specific axis, use `None` for that axis.

        .. container::

            **Valid values**:

            * :code:`""` for unscaled and wrapped coordinates,
            * :code:`"s"` for scaled and wrapped coordinates,
            * :code:`"u"` for unscaled and unwrapped coordinates, or
            * :code:`"su"` for scaled and unwrapped coordinates.

    dt : `float`, optional
        Simulation time step size. Only used when the dump file was not
        written with :code:`dump_modify time yes`. If not specified and
        the dump file does not contain time information, the step size
        is assumed to be the LAMMPS default for the style of units used.

    extras : `bool`, `str` or `list` of `str`, keyword-only, optional
        Extra per-atom information to read if the dump file was written
        in the :code:`atom` or :code:`custom` styles. If `True`, all
        extra attributes found in the trajectory are read.

        **Valid values:**

        +-------------------------------------+-----------------------------------------------+------------------------+
        | Attribute                           | LAMMPS dump attribute(s)                      | Per-atom data type     |
        +=====================================+===============================================+========================+
        | :code:`"dipole_moments"`            | (:code:`mux`, :code:`muy`, :code:`muz`)       | `numpy.ndarray[float]` |
        +-------------------------------------+-----------------------------------------------+------------------------+
        | :code:`"dipole_moments_magnitudes"` | :code:`mu`                                    | `float`                |
        +-------------------------------------+-----------------------------------------------+------------------------+
        | :code:`"angular_velocities"`        | (:code:`omegax`, :code:`omegay`,              | `numpy.ndarray[float]` |
        |                                     | :code:`omegaz`)                               |                        |
        +-------------------------------------+-----------------------------------------------+------------------------+
        | :code:`"angular_momenta"`           | (:code:`angmomx`, :code:`angmomy`,            | `numpy.ndarray[float]` |
        |                                     | :code:`angmomz`)                              |                        |
        +-------------------------------------+-----------------------------------------------+------------------------+
        | :code:`"torques"`                   | (:code:`tqx`, :code:`tqy`, :code:`tqz`)       | `numpy.ndarray[float]` |
        +-------------------------------------+-----------------------------------------------+------------------------+
        | :code:`"c_{compute_id}"`            | (:code:`c_{compute_id}[i]`, ...)              | `numpy.ndarray[float]` |
        +-------------------------------------+-----------------------------------------------+------------------------+
        | :code:`"d_{name}"`                  | (:code:`d_{name}[i]`, ...)                    | `numpy.ndarray[float]` |
        +-------------------------------------+-----------------------------------------------+------------------------+
        | :code:`"d2_{name}[i]"`              | (:code:`d2_{name}[i][j]`, ...)                | `numpy.ndarray[float]` |
        +-------------------------------------+-----------------------------------------------+------------------------+
        | :code:`"f_{fix_id}"`                | (:code:`f_{fix_id}[i]`, ...)                  | `numpy.ndarray[float]` |
        +-------------------------------------+-----------------------------------------------+------------------------+
        | :code:`"i_{name}"`                  | (:code:`i_{name}[i]`, ...)                    | `numpy.ndarray[int]`   |
        +-------------------------------------+-----------------------------------------------+------------------------+
        | :code:`"i2_{name}[i]"`              | (:code:`i2_{name}[i][j]`, ...)                | `numpy.ndarray[int]`   |
        +-------------------------------------+-----------------------------------------------+------------------------+
        | :code:`"v_{name}"`                  | (:code:`v_{name}[i]`, ...)                    | `numpy.ndarray[float]` |
        +-------------------------------------+-----------------------------------------------+------------------------+

    units_style : `str`, optional
        Style of units used in the dump file. If not specified, the
        style of units is determined from the file (when
        :code:`dump_modify units yes` was used) or assumed to be
        :code:`units lj` (the LAMMPS default) otherwise.

        **Valid values:** :code:`"lj"`, :code:`"real"`, :code:`"metal"`,
        :code:`"si"`, :code:`"cgs"`, :code:`"electron"`,
        :code:`"micro"`, and :code:`"nano"`.

    parallel : `bool`, keyword-only, optional
        Determines whether the file is read in parallel. This instance
        setting can be overridden in methods that have a `parallel`
        keyword parameter.

    n_workers : `int`, keyword-only, optional
        Number of threads to use when reading the file in parallel.
        If not specified, the number of logical threads available is
        used.

        .. important::

           `n_workers` should be set to a value that is less than half
           the number of frames in the dump file if `dt` is not
           specified and is to be determined automatically.

    Examples
    --------
    The simplest way to load a LAMMPS dump file :code:`dump.lammpstrj`
    containing atom trajectories is:

    >>> reader = LAMMPSDumpReader("dump.lammpstrj")

    Trajectory properties, like the coordinate formats, time step size,
    and style of units, are automatically determined from the file (if
    possible). Alternatively, the :class:`LAMMPSDumpReader` constructor
    accepts keyword arguments—`coordinate_formats`, `dt`, and
    `units_style`, respectively—to directly specify these properties.
    Additionally, the constructor also accepts the `extras` keyword
    argument to specify extra attributes to read from the dump file, and
    the `parallel` and `n_workers` keyword arguments to control parallel
    parsing of the dump file.

    For example, to parse the same dump file in parallel with 4 CPU
    threads and specify that it is in reduced Lennard-Jones units, the
    coordinates are scaled and unwrapped in the :math:`x`- and
    :math:`y`-directions but unscaled and wrapped in the
    :math:`z`-direction, the time step size is
    :math:`\\Delta t^*=0.005`, and all extra attributes should be read:

    >>> reader = LAMMPSDumpReader(
    ...    "dump.lammpstrj",
    ...    ["su", "su", ""],
    ...    dt=0.005,
    ...    extras=True,
    ...    units_style="lj",
    ...    parallel=True,
    ...    n_workers=4
    ... )

    Then, the `reader` instance can be used to get basic trajectory
    properties:

    >>> reader.dt
    0.005
    >>> reader.timestep
    5
    >>> reader.times
    array([ 0.,  5., 10., 15., 20.])
    >>> reader.n_frames
    5
    >>> reader.n_atoms
    1000

    or get raw data, like dimensions and positions, from frames:

    >>> reader.read_frames([0, 1])
    [{'time': 0, 'timestep': 0, 'dimensions': array([10., 10., 10., 90., 90., 90.]), ...},
     {'time': 5, 'timestep': 1, 'dimensions': array([10., 10., 10., 90., 90., 90.]), ...}]
    """

    _ATTRIBUTES = {
        "id",
        "mol",
        "proc",
        "procp1",
        "type",
        "typelabel",
        "element",
        "mass",
        "x",
        "y",
        "z",
        "xs",
        "ys",
        "zs",
        "xu",
        "yu",
        "zu",
        "xsu",
        "ysu",
        "xsu",
        "ix",
        "iy",
        "iz",
        "vx",
        "vy",
        "vz",
        "fx",
        "fy",
        "fz",
        "q",
        "index",
    }
    _COORDINATE_FORMATS = ("u", "su", "", "s")
    _CUSTOM_ATTRIBUTE_PREFIXES = ("c_", "d_", "d2_", "f_", "i_", "i2_", "v_")
    _DEFAULT_TIME_STEP_SIZES = {
        "lj": 0.005,
        "real": 1.0,
        "metal": 0.001,
        "si": 1e-8,
        "cgs": 1e-8,
        "electron": 0.001,
        "micro": 2.0,
        "nano": 0.00045,
    }
    _EXTENSIONS = {".dump", ".lammpsdump", ".lammpstrj"}
    _EXTRA_ATTRIBUTES = {
        "dipole_moments": ("mux", "muy", "muz"),
        "dipole_moment_magnitudes": ("mu",),
        "angular_velocities": ("omegax", "omegay", "omegaz"),
        "angular_momenta": ("angmomx", "angmomy", "angmomz"),
        "torques": ("tqx", "tqy", "tqz"),
    }
    _FORMAT = "LAMMPSDUMP"
    _PARALLELIZABLE = True
    _UNITS: dict[str, U_]
    _UNIT_STYLES = {
        "lj": {
            "charge": ureg.dimensionless,
            "energy": ureg.dimensionless,
            "length": ureg.dimensionless,
            "mass": ureg.dimensionless,
            "temperature": ureg.dimensionless,
            "time": ureg.dimensionless,
        },
        "real": {
            "charge": ureg.elementary_charge,
            "energy": ureg.kilocalorie / ureg.mole,
            "length": ureg.angstrom,
            "mass": ureg.gram / ureg.mole,
            "temperature": ureg.kelvin,
            "time": ureg.femtosecond,
        },
        "metal": {
            "charge": ureg.elementary_charge,
            "energy": ureg.electron_volt,
            "length": ureg.angstrom,
            "mass": ureg.gram / ureg.mole,
            "temperature": ureg.kelvin,
            "time": ureg.picosecond,
        },
        "si": {
            "charge": ureg.coulomb,
            "energy": ureg.joule,
            "length": ureg.meter,
            "mass": ureg.kilogram,
            "temperature": ureg.kelvin,
            "time": ureg.second,
        },
        "cgs": {
            "charge": ureg.franklin,
            "energy": ureg.erg,
            "length": ureg.centimeter,
            "mass": ureg.gram,
            "temperature": ureg.kelvin,
            "time": ureg.second,
        },
        "electron": {
            "charge": ureg.elementary_charge,
            "energy": ureg.hartree,
            "length": ureg.bohr,
            "mass": ureg.unified_atomic_mass_unit,
            "temperature": ureg.kelvin,
            "time": ureg.femtosecond,
        },
        "micro": {
            "charge": ureg.picocoulomb,
            "energy": ureg.picogram * ureg.micrometer**2 / ureg.microsecond**2,
            "length": ureg.micrometer,
            "mass": ureg.picogram,
            "temperature": ureg.kelvin,
            "time": ureg.microsecond,
        },
        "nano": {
            "charge": ureg.elementary_charge,
            "energy": ureg.attogram * ureg.nanometer**2 / ureg.nanosecond**2,
            "length": ureg.nanometer,
            "mass": ureg.attogram,
            "temperature": ureg.kelvin,
            "time": ureg.nanosecond,
        },
    }
    _VECTOR_ATTRIBUTES = {
        "positions",
        "velocities",
        "forces",
        "dipole_moments",
        "angular_velocities",
        "angular_momenta",
        "torques",
    }

    def __init__(
        self,
        filename: str | Path,
        /,
        coordinate_formats: str | list[str] | None = None,
        *,
        dt: float | None = None,
        extras: bool | str | list[str] | None = None,
        units_style: str | None = None,
        parallel: bool = False,
        n_workers: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            filename,
            parallel=parallel,
            n_workers=n_workers,
        )

        # Create and store handle to file
        self.open()

        # Loosely check if file is a LAMMPS dump file
        line = self._file.readline().rstrip()
        if not line.startswith("ITEM:"):
            raise ValueError(
                f"'{self._filename.name}' is not a valid LAMMPS dump file."
            )

        # Figure out per-frame format and style of units
        self._has_units_header = line == "ITEM: UNITS"
        if self._has_units_header:
            _units_style = self._file.readline().rstrip()
            if units_style is not None and units_style != _units_style:
                warnings.warn(
                    f"LAMMPS dump file '{self._filename.name}' uses "
                    f"`units {_units_style}`, but `{units_style=}` "
                    "was specified."
                )
            self._units_style = _units_style
            line = self._file.readline().rstrip()
        elif units_style is None:
            self._units_style = "lj"
        else:
            if units_style not in self._UNIT_STYLES:
                raise ValueError(
                    f"Invalid units style '{units_style}'. Valid values: "
                    "'" + "', '".join(self._UNIT_STYLES) + "'."
                )
            self._units_style = units_style
        self._UNITS = self._UNIT_STYLES[self._units_style]
        self._has_time_header = line == "ITEM: TIME"
        if self._has_time_header:
            self._file.readline()
            self._file.readline()
        self._file.readline()  # <timestep number>

        # Figure out dump style and number of header lines
        self._dump_style = None
        line = self._file.readline().rstrip()
        if line.startswith("ITEM: BOX BOUNDS"):
            self._dump_style = "grid"
            attributes_header_prefix = "ITEM: GRID CELLS "
            extras = True
            n_skip_lines = 6
        else:
            if line == "ITEM: NUMBER OF ATOMS":
                self._dump_style = "custom"
                self._entity_name = "atoms"
            else:
                self._dump_style = "local"
                self._entity_name = line.split()[-1].lower()
                extras = True
            attributes_header_prefix = f"ITEM: {self._entity_name.upper()} "
            n_skip_lines = 5
        for _ in range(n_skip_lines):
            self._file.readline()
        if self._dump_style == "grid":
            self._n_entities = np.prod(np.array(self._file.readline().split(), int))

        # Get and store attributes available in file
        self._attributes = {
            attr: col
            for col, attr in enumerate(
                self._file.readline().removeprefix(attributes_header_prefix).split()
            )
        }

        # Find and note columns for standard attributes
        self._attribute_columns = defaultdict(list)
        if (
            col := self._attributes.get("id", self._attributes.get("index"))
        ) is not None:
            self._attribute_columns["ids"] = col
        if (col := self._attributes.get("mol")) is not None:
            self._attribute_columns["molecule_ids"] = col
        if (col := self._attributes.get("type")) is not None:
            self._attribute_columns["types"] = col
        if (col := self._attributes.get("typelabel")) is not None:
            self._attribute_columns["labels"] = col
        if (col := self._attributes.get("element")) is not None:
            self._attribute_columns["elements"] = col
        if (col := self._attributes.get("mass")) is not None:
            self._attribute_columns["masses"] = col
        if (col := self._attributes.get("q")) is not None:
            self._attribute_columns["charges"] = col

        # Figure out available information in atom trajectories
        if self._dump_style == "custom":
            # Validate and store coordinate columns and formats
            if coordinate_formats is None:
                self._coordinate_formats = []
                for axis in "xyz":
                    for fmt in self._COORDINATE_FORMATS:
                        if col := self._attributes.get(f"{axis}{fmt}"):
                            self._attribute_columns["positions"].append(col)
                            self._coordinate_formats.append(fmt)
                            break
                    else:
                        self._attribute_columns["positions"].append(None)
                        self._coordinate_formats.append(None)
            elif isinstance(coordinate_formats, str):
                if coordinate_formats not in self._COORDINATE_FORMATS:
                    raise ValueError(
                        f"Invalid format '{coordinate_formats}' in "
                        "`coordinate_formats`. Valid values: '"
                        "', '".join(self._COORDINATE_FORMATS) + "'."
                    )
                self._coordinate_formats = []
                for axis in "xyz":
                    if col := self._attributes.get(f"{axis}{coordinate_formats}"):
                        self._attribute_columns["positions"].append(col)
                        self._coordinate_formats.append(coordinate_formats)
                    else:
                        warnings.warn(
                            f"Coordinate format '{coordinate_formats}' "
                            f"was specified, but '{self._filename.name}' "
                            "does not contain the "
                            f"`{axis}{coordinate_formats}` attribute."
                        )
                        self._attribute_columns["positions"].append(None)
                        self._coordinate_formats.append(None)
                if not any(self._coordinate_formats):
                    raise ValueError(
                        f"Coordinate format '{coordinate_formats}' was "
                        f"specified, but '{self._filename.name}' does not "
                        f"contain any of the `x{coordinate_formats}`, "
                        f"`y{coordinate_formats}`, or "
                        f"`z{coordinate_formats}` attributes."
                    )
            else:
                if len(coordinate_formats) == 3:
                    for axis, fmt in zip("xyz", coordinate_formats):
                        if fmt is None:
                            self._attribute_columns["positions"].append(None)
                            continue
                        if fmt not in self._COORDINATE_FORMATS:
                            raise ValueError(
                                f"Invalid format '{fmt}' in "
                                "`coordinate_formats`. Valid values: '"
                                "', '".join(self._COORDINATE_FORMATS) + "'."
                            )
                        col = self._attributes.get(f"{axis}{fmt}")
                        if col is None:
                            raise ValueError(
                                f"{axis}-coordinate format '{fmt}' was specified, "
                                f"but '{self._filename.name}' does not contain "
                                f"the `{axis}{fmt}` attribute."
                            )
                        self._attribute_columns["positions"].append(col)
                    self._coordinate_formats = coordinate_formats
                else:
                    raise ValueError(
                        "`coordinate_formats` must have length 3 when "
                        f"it is an array, not {len(coordinate_formats)}."
                    )

            # Check if image flags and velocity and force information are
            # available
            image_flag_columns = []
            force_columns = []
            velocity_columns = []
            for axis in "xyz":
                image_flag_columns.append(self._attributes.get(f"i{axis}"))
                force_columns.append(self._attributes.get(f"f{axis}"))
                velocity_columns.append(self._attributes.get(f"v{axis}"))
            if any(image_flag_columns):
                self._attribute_columns["image_flags"] = image_flag_columns
            if any(force_columns):
                self._attribute_columns["forces"] = force_columns
            if any(velocity_columns):
                self._attribute_columns["velocities"] = velocity_columns

        # Validate and note extra attributes to read
        self._extra_attribute_indices = defaultdict(list)
        if extras is None:
            extra_attribute_columns = {}
        elif isinstance(extras, bool):
            if extras:
                extra_attribute_columns = defaultdict(dict)
                for attr, col in self._attributes.items():
                    if attr not in self._ATTRIBUTES:
                        if attr.startswith(self._CUSTOM_ATTRIBUTE_PREFIXES):
                            if (split_index := attr.rfind("[")) != -1:
                                name = attr[:split_index]
                                index = int(attr[split_index + 1 : -1])
                                self._extra_attribute_indices[name].append(index)
                                extra_attribute_columns[name][index] = col
                            else:
                                self._extra_attribute_indices[attr] = None
                                extra_attribute_columns[attr] = col
                        else:
                            for name, attrs in self._EXTRA_ATTRIBUTES.items():
                                if attr in attrs:
                                    extra_attribute_columns[name][
                                        attrs.index(attr)
                                    ] = col
                for name, cols in extra_attribute_columns.items():
                    if name in self._EXTRA_ATTRIBUTES:
                        self._extra_attribute_indices[name] = extra_attribute_columns[
                            name
                        ] = [i for _, i in sorted(cols.items())]
                    elif self._extra_attribute_indices[name] is not None:
                        self._extra_attribute_indices[name].sort()
                        extra_attribute_columns[name] = [
                            cols[i] for i in self._extra_attribute_indices[name]
                        ]
            else:
                extra_attribute_columns = {}
        else:
            if isinstance(extras, str):
                extras = [extras]
            extra_attribute_columns = defaultdict(dict)
            for attr in extras:
                if attr in self._EXTRA_ATTRIBUTES:
                    columns = tuple(
                        self._attributes.get(attr)
                        for attr in self._EXTRA_ATTRIBUTES[attr]
                    )
                    if not any(columns):
                        raise ValueError(
                            f"Extra attribute '{attr}' was requested, but "
                            f"'{self._filename.name}' does not contain any of the "
                            f"attributes in {self._EXTRA_ATTRIBUTES[attr]}."
                        )
                    extra_attribute_columns[attr] = columns
                elif attr.startswith(self._CUSTOM_ATTRIBUTE_PREFIXES):
                    mapping = {
                        int(name[len(attr) + 1 : -1]): col
                        for name, col in self._attributes.items()
                        if name.startswith(attr)
                    }
                    self._extra_attribute_indices[attr] = sorted(mapping.keys())
                    extra_attribute_columns[attr] = [
                        mapping[i] for i in self._extra_attribute_indices[attr]
                    ]
                else:
                    raise ValueError(f"Invalid attribute '{attr}' in `extras`.")
        self._attribute_columns |= extra_attribute_columns

        # Get time step, number of atoms, and byte offsets for frames in file
        # NOTE: Possible values for `*dt`, `*time_step`, and
        #       `*n_entities` are
        #       - `True` if the value has not been set yet,
        #       - `False` if the value has not yet been found in the
        #         trajectory (subset),
        #       - an `int` if the value is constant, or
        #       - `None` if the value is not constant.
        file_size = self._filename.stat().st_size
        self._parallel = parallel
        if self._parallel:
            chunk_size = np.ceil(file_size / self._n_workers).astype(int)
            self._dt = self._time_step = True
            self._offsets = []
            self._times = []
            self._timesteps = []
            if self._dump_style != "grid":
                self._n_entities = True
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self._n_workers
            ) as executor:
                for future in concurrent.futures.as_completed(
                    executor.submit(
                        self._find_frames,
                        self._filename,
                        i * chunk_size,
                        min((i + 1) * chunk_size, file_size),
                    )
                    for i in range(self._n_workers)
                ):
                    _dt, time_step, n_entities, offsets, times, timesteps = (
                        future.result()
                    )
                    if _dt is not False and self._dt != _dt:
                        if self._dt is True:
                            self._dt = _dt
                        elif _dt is not None:
                            self._dt = None
                    if time_step is not False and self._time_step != time_step:
                        if self._time_step is True:
                            self._time_step = time_step
                        elif time_step is not None:
                            self._time_step = None
                    if self._dump_style != "grid":
                        if n_entities is not False and self._n_entities != n_entities:
                            if self._n_entities is True:
                                self._n_entities = n_entities
                            elif n_entities is not None:
                                self._n_entities = None
                    self._offsets.extend(offsets)
                    if times is not None:
                        self._times.extend(times)
                    self._timesteps.extend(timesteps)
            order = np.argsort(self._offsets)
            self._offsets = np.array(self._offsets)[order]
            if self._times:
                self._times = np.array(self._times)[order]
            self._timesteps = np.array(self._timesteps)[order]
        else:
            (
                self._dt,
                self._time_step,
                n_entities,
                self._offsets,
                self._times,
                self._timesteps,
            ) = self._find_frames(self._file, 0, file_size)
            self._offsets = np.array(self._offsets)
            if self._times is not None:
                self._times = np.array(self._times)
            self._timesteps = np.array(self._timesteps)
            if self._dump_style != "grid":
                self._n_entities = n_entities

        # Give properties default values if not set
        if dt:
            if self._dt != dt and not isinstance(self._dt, bool):
                warnings.warn(
                    f"`{dt=}` was specified, but the time step size "
                    "was determined to be "
                    f"{'variable' if self._dt is None else self._dt}."
                )
                self._dt = dt
        elif isinstance(self._dt, bool):
            self._dt = self._DEFAULT_TIME_STEP_SIZES[self._units_style]
        if self._times is None or isinstance(self._times, bool):
            self._times = self._dt * self._timesteps
        if isinstance(self._time_step, bool):
            self._time_step = (
                None
                if len(self._times) == 1
                or len(time_steps := set(np.round(np.diff(self._times), 15))) > 1
                else time_steps.pop()
            )
        if isinstance(self._n_entities, bool):
            self._n_entities = None

        # Store other settings
        self._reduced = self._units_style == "lj"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}('{self._filename.name}', "
            f"coordinate_formats={getattr(self, '_coordinate_formats', None)}, "
            f"dt={self.dt}, extras={list(self._extra_attribute_indices.keys())}, "
            f"units_style='{self._units_style}', parallel={self._parallel}, "
            f"n_workers={self._n_workers})"
        )

    def _find_frames(self, file: str | Path | TextIO, start: int, end: int) -> tuple[
        bool | float | None,
        bool | float | None,
        bool | int | None,
        list[int],
        list[float] | None,
        list[int],
    ]:
        """
        Finds the time step size, time step between frames, number of
        entities, byte offsets, times, and timesteps for frames in (a
        subset of) the LAMMPS dump file.

        Parameters
        ----------
        file : `str`, `pathlib.Path`, or `io.TextIO`
            Filename, path, or handle to the dump file.

        start : `int`
            Byte offset to start reading from.

        end : `int`
            Byte offset to stop reading at.

        Returns
        -------
        dt : `bool` or `float`
            Time step size between timesteps. Is `False` if
            :code:`dump_modify time yes` was not used or `None` if the
            step size is not constant.

            **Reference units**: :math:`\\mathrm{ps}`.

        time_step : `bool` or `float`
            Time step between frames. Is `False` if
            :code:`dump_modify time yes` was not used or `None` if the
            time step is not constant.

            **Reference units**: :math:`\\mathrm{ps}`.

        n_entities : `bool` or `int`
            Number of entities in each frame. Is `False` if the
            number of entities is not found in the trajectory (subset)
            or `None` if the number of entities is not constant.

        offsets : `list`
            Byte offsets for frames in the file.

        times : `list`
            Simulation times of the frames in the file. Is `None` if
            :code:`dump_modify time yes` was not used.

            **Reference units**: :math:`\\mathrm{ps}`.

        timesteps : `list`
            Simuulation timesteps for frames in the file.
        """

        manual = isinstance(file, (str, Path))
        if manual:
            filename = file
            file = open(file, "r")
        else:
            filename = file.name
        byte_counter = file.seek(start)

        is_style_grid = self._dump_style == "grid"
        frame_marker = "ITEM: TIME" if self._has_time_header else "ITEM: TIMESTEP"
        n_preheader_lines = 5 + 4 * is_style_grid

        dt = time_step = n_entities = False
        offsets = []
        times = []
        timesteps = []

        # Find first frame
        while byte_counter < end:
            line = file.readline()
            byte_counter += len(line)
            if line.rstrip() == frame_marker:
                offsets.append(byte_counter - len(line))
                if self._has_time_header:
                    _time = file.readline()
                    byte_counter += len(_time)  # <time>
                    _time = float(_time)
                    times.append(_time)
                    byte_counter += len(file.readline())  # ITEM: TIMESTEP
                _timestep = file.readline()
                byte_counter += len(_timestep)  # <timestep>
                _timestep = int(_timestep)
                timesteps.append(_timestep)
                if is_style_grid:
                    n_entities = _n_entities = self._n_entities
                else:
                    byte_counter += len(file.readline())  # ITEM: NUMBER OF [...]
                    _n_entities = file.readline()
                    byte_counter += len(_n_entities)  # <n_entities>
                    n_entities = _n_entities = int(_n_entities)
                n_lines_per_frame = n_preheader_lines + _n_entities
                break

        # Find remaining frames
        while byte_counter < end:
            for _ in range(n_lines_per_frame):
                byte_counter += len(file.readline())
                if byte_counter >= end:
                    break
            else:  # End of file not reached
                offsets.append(byte_counter)
                byte_counter += len(file.readline())
                if self._has_time_header:
                    time = file.readline()
                    byte_counter += len(time)  # <time>
                    time = float(time)
                    times.append(time)
                    byte_counter += len(file.readline())  # ITEM: TIMESTEP
                timestep = file.readline()
                byte_counter += len(timestep)  # <timestep>
                timestep = int(timestep)
                timesteps.append(timestep)
                if self._has_time_header:
                    _time_step = time - _time
                    if not np.isclose(time_step, _time_step):
                        if time_step is False:
                            time_step = _time_step
                        elif time_step is not None:
                            time_step = None
                    if not np.isclose(dt, _dt := _time_step / (timestep - _timestep)):
                        if dt is False:
                            dt = _dt
                        elif dt is not None:
                            dt = None
                    _time, _timestep = time, timestep
                if not is_style_grid:
                    byte_counter += len(file.readline())  # ITEM: NUMBER OF [...]
                    _n_entities = file.readline()
                    byte_counter += len(_n_entities)  # <n_entities>
                    _n_entities = int(_n_entities)
                    if n_entities != _n_entities and n_entities is not None:
                        n_entities = None
                    n_lines_per_frame = n_preheader_lines + _n_entities

        if manual:
            file.close()

        if len(offsets) != len(timesteps):
            raise RuntimeError(
                "Number of frames and timesteps found in the LAMMPS "
                f"dump file '{filename}' do not match."
            )
        if self._has_time_header and len(offsets) != len(times):
            raise RuntimeError(
                "Number of frames and times found in the LAMMPS "
                f"dump file '{filename}' do not match."
            )

        return (
            dt,
            time_step,
            n_entities,
            offsets,
            times if self._has_time_header else None,
            timesteps,
        )

    def _parse_frame(
        self, file: TextIO, frame: int, convert_units: bool
    ) -> dict[str, Any]:
        """
        Reads data from a single frame in the specified LAMMPS dump
        file.

        Parameters
        ----------
        file : `io.TextIO`
            Handle to the dump file.

        frame : `int`
            Frame index to read.

        convert_units : `bool`
            Specifies whether to convert the data from LAMMPS units to
            consistent MDCraft units.

        Returns
        -------
        frame_data : `dict`
            Data from the frame.
        """

        # Seek to frame
        file.seek(self._offsets[frame])

        # Initialize data dictionary
        frame_data = {}

        # Read time, if available
        if self._has_time_header:
            file.readline()  # ITEM: TIME
            frame_data["time"] = float(file.readline())

        # Read timestep
        file.readline()  # ITEM: TIMESTEP
        frame_data["timestep"] = int(file.readline())
        if not self._has_time_header and self._dt is not None:
            frame_data["time"] = frame_data["timestep"] * self._dt

        # Read number of atoms or entities
        if self._dump_style == "grid":
            n_entities = self.n_grids
        else:
            file.readline()  # ITEM: NUMBER OF [...]
            frame_data[f"n_{self._entity_name}"] = n_entities = int(file.readline())

        # Read system dimensions
        if "xy xz yz" in (box_header := file.readline()):  # ITEM: BOX BOUNDS
            xlo, xhi, xy = (float(val) for val in file.readline().split())
            ylo, yhi, xz = (float(val) for val in file.readline().split())
            zlo, zhi, yz = (float(val) for val in file.readline().split())

            # https://docs.lammps.org/Howto_triclinic.html
            # #output-of-restricted-and-general-triclinic-boxes-in-a-dump-file
            xlo -= min(0, xy, xz, xy + xz)
            xhi -= max(0, xy, xz, xy + xz)
            ylo -= min(0, yz)
            yhi -= max(0, yz)

            box_vectors = np.zeros((3, 3))
            box_vectors[0, 0] = xhi - xlo
            box_vectors[1, :2] = xy, yhi - ylo
            box_vectors[2] = xz, yz, zhi - zlo
            frame_data["dimensions"] = convert_cell_representation(
                box_vectors, "parameters"
            )
        elif is_general_triclinic := "abc origin" in box_header:
            box_vectors = np.vstack(
                [[float(val) for val in file.readline().split()[:3]] for _ in range(3)]
            )
            frame_data["dimensions"] = convert_cell_representation(
                box_vectors, "parameters"
            )
        else:
            xlo, xhi = (float(val) for val in file.readline().split())
            ylo, yhi = (float(val) for val in file.readline().split())
            zlo, zhi = (float(val) for val in file.readline().split())
            frame_data["dimensions"] = np.array(
                (xhi - xlo, yhi - ylo, zhi - zlo, 90, 90, 90)
            )
            box_vectors = convert_cell_representation(
                frame_data["dimensions"], "vectors"
            )
        for _ in range(1 + 4 * (self._dump_style == "grid")):
            file.readline()

        # Read frame data
        # data = np.loadtxt(file, max_rows=n_entities)
        data = pd.read_csv(file, sep="\\s+", header=None, nrows=n_entities).to_numpy()
        for name, columns in self._attribute_columns.items():
            frame_data[name] = data[:, columns]
            if name in {"ids", "molecule_ids", "types"} or name.startswith(
                ("i_", "i2_")
            ):
                frame_data[name] = frame_data[name].astype(int)

        # Recover Cartesian coordinates from scaled coordinates and system dimensions
        if self._dump_style == "custom":
            scaled_flags = ["s" in fmt for fmt in self._coordinate_formats]
            if any(scaled_flags) and np.allclose(frame_data["dimensions"][3:], 90):
                frame_data["positions"][:, scaled_flags] *= frame_data["dimensions"][
                    :3
                ][scaled_flags]

        # Rotate coordinates and per-atom vector quantities for general triclinic boxes
        if is_general_triclinic:
            restricted_box_vectors = convert_cell_representation(
                frame_data["dimensions"], "vectors"
            )
            if self._dump_style == "custom":
                for attr in frame_data.keys() & self._VECTOR_ATTRIBUTES:
                    scale_triclinic_coordinates(
                        frame_data[attr], box_vectors, scaled_flags
                    )
                    frame_data[attr] @= restricted_box_vectors.T

        # Convert from LAMMPS units to consistent MDCraft units
        if convert_units:
            frame_data["dimensions"][:3] = (
                frame_data["dimensions"][:3] * self._UNITS["length"]
            ).m_as(INTERNAL_UNITS["length"])
            if self._dump_style == "custom":
                frame_data["positions"] = (
                    frame_data["positions"] * self._UNITS["length"]
                ).m_as(INTERNAL_UNITS["length"])
                if "forces" in frame_data:
                    frame_data["forces"] = (
                        frame_data["forces"]
                        * self._UNITS["energy"]
                        / self._UNITS["length"]
                    )
                    if "[substance]" not in frame_data["forces"].dimensionality:
                        frame_data["forces"] *= ureg.avogadro_constant
                    frame_data["forces"] = frame_data["forces"].m_as(
                        INTERNAL_UNITS["energy"] / INTERNAL_UNITS["length"]
                    )
                if "velocities" in frame_data:
                    frame_data["velocities"] = (
                        frame_data["velocities"]
                        * self._UNITS["length"]
                        / self._UNITS["time"]
                    ).m_as(INTERNAL_UNITS["length"] / INTERNAL_UNITS["time"])

        return frame_data

    @property
    def dt(self) -> float | None:
        """
        Time step size between timesteps in the trajectory. If `None`,
        the step size is not constant across frames.

        **Reference units**: :math:`\\mathrm{ps}`.
        """

        return self._dt

    @property
    def time_step(self) -> float | None:
        """
        Time step between frames in the trajectory. If `None`, the time
        step is not constant across frames.

        **Reference units**: :math:`\\mathrm{ps}`.
        """

        return self._time_step

    @property
    def times(self) -> np.ndarray[float]:
        """
        Simuulation times found in the trajectory. May not be accurate
        if the time step size (`dt`) was not specified and could not be
        determined from the dump file.

        **Reference units**: :math:`\\mathrm{ps}`.
        """

        return self._times

    @property
    def timesteps(self) -> float:
        """
        Simulation timesteps found in the trajectory.
        """

        return self._timesteps

    @property
    def n_atoms(self) -> int | None:
        """
        Number of atoms in each frame. Only available if the dump file
        was written in the :code:`atom` or :code:`custom` styles. If
        `None`, the number of atoms is not constant across frames.
        """

        return self._n_entities if self._dump_style == "custom" else None

    @property
    def n_entities(self) -> int | None:
        """
        Number of atoms (:code:`atom` or :code:`custom` dump styles),
        grids (:code:`grid` dump style), or local entities (:code:`local`
        dump style) in each frame. If `None`, the number of entities is
        not constant across frames.
        """

        return self._n_entities

    @property
    def n_frames(self) -> int:
        """
        Number of frames in the trajectory.
        """

        return len(self._offsets)

    @property
    def n_grids(self) -> tuple[int, int, int] | None:
        """
        Number of grid cells in each dimension. Only available if the
        dump file was written in the :code:`grid` style.
        """

        return self._n_entities if self._dump_style == "grid" else None

    def open(self) -> None:
        """
        Opens the LAMMPS dump file and stores a handle to it.
        """

        self._file = open(self._filename, "r")

    def close(self) -> None:
        """
        Closes the LAMMPS dump file and deletes the handle.
        """

        if hasattr(self, "_file"):
            self._file.close()
            del self._file


if FOUND_MDANALYSIS:
    from MDAnalysis.coordinates.base import ReaderBase

    class _LAMMPSDumpReader(LAMMPSDumpReader, ReaderBase):
        """
        LAMMPS dump file reader compatible with MDAnalysis.

        .. seealso::

        For more information on the features of this reader, see the
        documentation for :class:`mdcraft.io.LAMMPSDumpReader`.

        Parameters
        ----------
        filename : `str` or `pathlib.Path`, positional-only
            Filename or path to the dump file.

        coordinate_formats : `str` or `list` of `str`, optional
            Coordinate formats. If a `str` is provided, it is used for
            all axes. If the trajectory does not contain particle
            positions for a specific axis, use `None` for that axis.

            .. container::

                **Valid values**:

                * :code:`""` for unscaled and wrapped coordinates,
                * :code:`"s"` for scaled and wrapped coordinates,
                * :code:`"u"` for unscaled and unwrapped coordinates, or
                * :code:`"su"` for scaled and unwrapped coordinates.

        extras : `bool`, `str` or `list` of `str`, keyword-only, optional
            Extra per-atom information to read. If `True`, all extra
            attributes found in the trajectory are read.

            **Valid values:**

            +-------------------------------------+-------------------------------------------+------------------------+
            | Attribute                           | LAMMPS dump attribute(s)                  | Per-atom data type     |
            +=====================================+===========================================+========================+
            | :code:`"dipole_moments"`            | (:code:`mux`, :code:`muy`, :code:`muz`)   | `numpy.ndarray[float]` |
            +-------------------------------------+-------------------------------------------+------------------------+
            | :code:`"dipole_moments_magnitudes"` | :code:`mu`                                | `float`                |
            +-------------------------------------+-------------------------------------------+------------------------+
            | :code:`"angular_velocities"`        | (:code:`omegax`, :code:`omegay`,          | `numpy.ndarray[float]` |
            |                                     | :code:`omegaz`)                           |                        |
            +-------------------------------------+-------------------------------------------+------------------------+
            | :code:`"angular_momenta"`           | (:code:`angmomx`, :code:`angmomy`,        | `numpy.ndarray[float]` |
            |                                     | :code:`angmomz`)                          |                        |
            +-------------------------------------+-------------------------------------------+------------------------+
            | :code:`"torques"`                   | (:code:`tqx`, :code:`tqy`, :code:`tqz`)   | `numpy.ndarray[float]` |
            +-------------------------------------+-------------------------------------------+------------------------+
            | :code:`"c_{compute_id}"`            | (:code:`c_{compute_id}[i]`, ...)          | `numpy.ndarray[float]` |
            +-------------------------------------+-------------------------------------------+------------------------+
            | :code:`"d_{name}"`                  | (:code:`d_{name}[i]`, ...)                | `numpy.ndarray[float]` |
            +-------------------------------------+-------------------------------------------+------------------------+
            | :code:`"d2_{name}[i]"`              | (:code:`d2_{name}[i][j]`, ...)            | `numpy.ndarray[float]` |
            +-------------------------------------+-------------------------------------------+------------------------+
            | :code:`"f_{fix_id}"`                | (:code:`f_{fix_id}[i]`, ...)              | `numpy.ndarray[float]` |
            +-------------------------------------+-------------------------------------------+------------------------+
            | :code:`"i_{name}"`                  | (:code:`i_{name}[i]`, ...)                | `numpy.ndarray[int]`   |
            +-------------------------------------+-------------------------------------------+------------------------+
            | :code:`"i2_{name}[i]"`              | (:code:`i2_{name}[i][j]`, ...)            | `numpy.ndarray[int]`   |
            +-------------------------------------+-------------------------------------------+------------------------+
            | :code:`"v_{name}"`                  | (:code:`v_{name}[i]`, ...)                | `numpy.ndarray[float]` |
            +-------------------------------------+-------------------------------------------+------------------------+

        reduced : `bool`, keyword-only, optional
            Specifies whether the data is in reduced units.

        parallel : `bool`, keyword-only, optional
            Determines whether the file is read in parallel to determine
            the frame offsets.

        n_workers : `int`, keyword-only, optional
            Number of threads to use when reading the file in parallel.
            If not specified, the number of logical threads available is
            used.

        **kwargs : `dict`, optional
            Additional keyword arguments to pass to the
            :class:`MDAnalysis.coordinates.base.ReaderBase` constructor.
        """

        format = "LAMMPSDUMP"

        def __init__(
            self,
            filename: str | Path,
            /,
            coordinate_formats: str | list[str] | None = None,
            *,
            extras: bool | str | list[str] | None = None,
            parallel: bool = False,
            n_workers: int | None = None,
            **kwargs,
        ) -> None:
            LAMMPSDumpReader.__init__(
                self,
                filename,
                coordinate_formats=coordinate_formats,
                extras=extras,
                parallel=parallel,
                n_workers=n_workers,
            )
            if self._dump_style != "custom":
                raise ValueError(
                    "MDAnalysis only supports LAMMPS dump files "
                    "written in the `atom` or `custom` styles."
                )
            ReaderBase.__init__(self, filename, **kwargs)
            self._reopen()
            self._read_next_timestep()

        def __repr__(self) -> str:
            return (
                f"<LAMMPSDumpReader {self._filename.name} with "
                f"{self.n_frames} frames of {self.n_atoms} atoms>"
            )

        def _reopen(self) -> None:
            """
            Reopens the LAMMPS dump file and stores a handle to it.
            """

            self.close()
            self.open()
            self.ts = self._Timestep(self.n_atoms, **self._ts_kwargs)
            self.ts.frame = -1

        def _read_frame(self, frame: int) -> "ReaderBase._Timestep":
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

        def _read_next_timestep(self) -> "ReaderBase._Timestep":
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
            self._check_frame(ts.frame)

            data = self.read_frames(ts.frame, parallel=False, _convert_units=False)
            ts.data["step"] = data["timestep"]
            ts.data["time"] = data["timestep"] * ts.dt
            ts.dimensions = data["dimensions"]
            ts.positions = data["positions"]
            ts.has_forces = "forces" in data
            if ts.has_forces:
                ts.forces = data["forces"]
            ts.has_velocities = "velocities" in data
            if ts.has_velocities:
                ts.velocities = data["velocities"]
            for attr in self._extra_attribute_indices:
                ts.data[attr] = data[attr]

            if "ids" in data:
                order = np.argsort(data["ids"])
                ts.positions = ts.positions[order]
                if ts.has_forces:
                    ts.forces = ts.forces[order]
                if ts.has_velocities:
                    ts.velocities = ts.velocities[order]
                for attr in self._extra_attribute_indices:
                    ts.data[attr] = ts.data[attr][order]

            return ts
