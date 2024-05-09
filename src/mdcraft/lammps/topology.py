"""
Topology transformations
========================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains implementations of common LAMMPS topology
transformations, like the generation of initial particle positions.
"""

from io import TextIOWrapper
from numbers import Real
from typing import Any, Union

import numpy as np

from ..algorithm import topology as topo

def create_atoms(*args, **kwargs) -> Any:

    """
    Generates initial particle positions for coarse-grained simulations.

    .. seealso::

       This is an alias function. For more information, see
       :func:`mdcraft.algorithm.topology.create_atoms`.
    """

    return topo.create_atoms(*args, **kwargs)

def write_data(
        file: Union[str, TextIOWrapper], positions: tuple[np.ndarray[float]],
        *, bonds: tuple[np.ndarray[int]] = None,
        angles: tuple[np.ndarray[int]] = None,
        dihedrals: tuple[np.ndarray[int]] = None,
        impropers: tuple[np.ndarray[int]] = None,
        dimensions: np.ndarray[float] = None, tilt: np.ndarray[float] = None,
        charges: np.ndarray[float] = None, masses: np.ndarray[float] = None,
    ) -> None:

    r"""
    Writes topological data to a LAMMPS data file in :code:`atom_style full`.

    Parameters
    ----------
    file : `str` or `_io.TextIOWrapper`
        LAMMPS data file.

    positions : `tuple`
        Atomic positions. Each element of the tuple should contain
        atoms of the same atom type.

        **Shape**: Tuple of arrays with shape :math:`(*,\,3)`.

        **Reference units**: :math:`\mathrm{Å}`.

    bonds : `tuple`, keyword-only, optional
        Pairs of indices of bonded atoms. Each element of the tuple
        should contain bonds of the same bond type.

        **Shape**: Tuple of arrays with shape :math:`(*,\,2)`.

    angles : `tuple`, keyword-only, optional
        Triples of indices of atoms that form an angle. Each element of
        the tuple should contain angles of the same angle type.

        **Shape**: Tuple of arrays with shape :math:`(*,\,3)`.

    dihedrals : `tuple`, keyword-only, optional
        Quadruples of indices of atoms that form a dihedral. Each
        element of the tuple should contain dihedrals of the same
        dihedral type.

        **Shape**: Tuple of arrays with shape :math:`(*,\,4)`.

    impropers : `tuple`, keyword-only, optional
        Quadruples of indices of atoms that form an improper. Each
        element of the tuple should contain impropers of the same
        improper type.

        **Shape**: Tuple of arrays with shape :math:`(*,\,4)`.

    dimensions : array-like, keyword-only, optional
        Box dimensions. If three values are provided, the box
        dimensions are assumed to be from :math:`0` to the specified
        values. If six values are provided, the box dimensions go from
        the three values in the first column to the three values in the
        second column.

        **Shape**: :math:`(3,)` or :math:`(3,\,2)`.

        **Reference units**: :math:`\mathrm{Å}`.

    tilt : array-like, keyword-only, optional
        Box :math:`xy`, :math:`xz`, and :math:`yz` tilt factors.

        **Shape**: :math:`(3,)`.

    charges : array-like, keyword-only, optional
        Atomic charges.

        **Shape**: :math:`(N,)`.

    masses : array-like, keyword-only, optional
        Atomic masses.

        **Shape**: :math:`(N,)`.
    """

    if isinstance(file, str):
        file = open(file, "w")

    # Write header
    file.write("LAMMPS Description\n\n")
    n_atoms_type = [len(p) for p in positions]
    n_atoms = sum(n_atoms_type)
    file.write(f"{n_atoms} atoms\n")
    file.write(f"{len(positions)} atom types\n")
    if bonds is not None:
        n_bonds_type = [len(b) for b in bonds]
        file.write(f"{sum(n_bonds_type)} bonds\n")
        file.write(f"{len(bonds)} bond types\n")
    if angles is not None:
        n_angles_type = [len(a) for a in angles]
        file.write(f"{sum(n_angles_type)} angles\n")
        file.write(f"{len(angles)} angle types\n")
    if dihedrals is not None:
        n_dihedrals_type = [len(d) for d in dihedrals]
        file.write(f"{sum(n_dihedrals_type)} dihedrals\n")
        file.write(f"{len(dihedrals)} dihedral types\n")
    if impropers is not None:
        n_impropers_type = [len(i) for i in impropers]
        file.write(f"{sum(n_impropers_type)} impropers\n")
        file.write(f"{len(impropers)} improper types\n")
    if dimensions is not None:
        if dimensions.ndim == 1:
            dimensions = np.vstack((np.zeros(3), dimensions)).T
        for i, d in enumerate(dimensions):
            a = chr(120 + i)
            file.write(f"{d[0]:.6g} {d[1]:.6g} {a}lo {a}hi\n")
    if tilt is not None:
        file.write(f"{tilt[0]:.6g} {tilt[1]:.6g} {tilt[2]:.6g} xy xz yz\n")

    # Write masses
    if masses is not None:
        if len(masses) != len(positions):
            emsg = "Number of masses must match number of atom types."
            raise ValueError(emsg)
        file.write("\nMasses\n\n")
        for i, m in enumerate(masses):
            file.write(f"{i + 1} {m:.6g}\n")

    # Write atom positions
    if charges is None:
        charges = np.zeros(n_atoms)
    if len(charges) == len(positions):
        charges = list(charges)
        for i, (qs, n) in enumerate(zip(charges, n_atoms_type)):
            if isinstance(qs, Real):
                charges[i] *= np.ones(n)
    elif len(charges) == n_atoms:
        charges = np.array_split(charges, np.cumsum(n_atoms)[:-1])
    else:
        raise ValueError("'charges' has an invalid shape.")
    file.write("\nAtoms # full\n\n")
    for t, (pos, qs) in enumerate(zip(positions, charges)):
        start = sum(n_atoms_type[:t])
        for i, (p, q) in enumerate(zip(pos, qs)):
            file.write(f"{start + i + 1} {start + i + 1} {t + 1} {q:.6g} "
                       f"{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}\n")

    # Write bonds
    if bonds is not None:
        file.write("\nBonds\n\n")
        for t, b in enumerate(bonds):
            start = sum(n_bonds_type[:t])
            for i, (a, b) in enumerate(b):
                file.write(f"{start + i + 1} {t + 1} {a} {b}\n")

    # Write angles
    if angles is not None:
        file.write("\nAngles\n\n")
        for t, a in enumerate(angles):
            start = sum(n_angles_type[:t])
            for i, (a, b, c) in enumerate(a):
                file.write(f"{start + i + 1} {t + 1} {a} {b} {c}\n")

    # Write dihedrals
    if dihedrals is not None:
        file.write("\nDihedrals\n\n")
        for t, d in enumerate(dihedrals):
            start = sum(n_dihedrals_type[:t])
            for i, (a, b, c, d) in enumerate(d):
                file.write(f"{start + i + 1} {t + 1} {a} {b} {c} {d}\n")

    # Write impropers
    if impropers is not None:
        file.write("\nImpropers\n\n")
        for t, i in enumerate(impropers):
            start = sum(n_impropers_type[:t])
            for j, (a, b, c, d) in enumerate(i):
                file.write(f"{start + j + 1} {t + 1} {a} {b} {c} {d}\n")

    file.close()