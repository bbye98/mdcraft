"""
Topology transformations
========================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains algorithms for initializing or transforming
topologies.
"""

from typing import Any, Union
import warnings

import MDAnalysis as mda
from MDAnalysis.lib.distances import minimize_vectors
import numpy as np
from .. import FOUND_OPENMM, Q_, ureg
from .molecule import center_of_mass
from .utility import (is_lower_triangular, get_closest_factors, replicate, 
                      find_connected_nodes)
from .unit import strip_unit

if FOUND_OPENMM:
    from openmm import app, unit

def create_atoms(
        dimensions: Union[np.ndarray[float], "unit.Quantity", Q_,
                          "app.Topology"],
        N: int = None, N_p: int = 1, *, lattice: str = None,
        length: Union[float, "unit.Quantity"] = 0.34,
        flexible: bool = False, bonds: bool = False, angles: bool = False,
        dihedrals: bool = False, randomize: bool = False,
        length_unit: Union["unit.Unit", ureg.Unit] = None, wrap: bool = False
    ) -> Any:

    """
    Generates initial particle positions for coarse-grained simulations.

    Parameters
    ----------
    dimensions : `numpy.ndarray`, `openmm.unit.Quantity`, \
    `pint.Quantity`, or `openmm.app.Topology`
        System dimensions, box vectors, or lattice parameters provided
        as an array, or an OpenMM topology with system dimension 
        information.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    N : `int`, optional
        Total number of particles :math:`N`. Must be provided for random
        melts or polymers.

    N_p : `int`, default: :code:`1`
        Number of atoms (monomers) :math:`N_\\mathrm{p}` in each
        segment (polymer chain).

        **Valid values**: :math:`1\\leq N_\\mathrm{p}\\leq N`, with
        :math:`N` divisible by :math:`N_\\mathrm{p}`.

    lattice : `str`, keyword-only, optional
        Lattice type, with the relevant length scale specified in
        `length`. If `lattice` is not specified, particle positions will
        be assigned randomly.

        .. tip::

           To build walls with the correct periodicity, set the
           :math:`z`-dimension to :code:`0` in `dimensions` and
           :code:`flexible=True`. This function will then return
           the wall particle positions and the :math:`x`- and
           :math:`y`-dimensions closest to those specified in
           `dimensions` that satisfy the lattice periodicity.

           Walls should only be built in the :math:`z`-direction.

        .. container::

           **Valid values**:

           * :code:`"fcc"`: Face-centered cubic (FCC) lattice,
             determined by the particle size :math:`a`.
           * :code:`"hcp"`: Hexagonal close-packed (HCP) lattice,
             determined by the particle size :math:`a`.
           * :code:`"cubic"`: Cubic crystal system, determined by the
             particle size :math:`a`.
           * :code:`"honeycomb"`: Honeycomb lattice (e.g., graphene),
             determined by the bond length :math:`b`.

    length : `float` or `openmm.unit.Quantity`, default: :code:`0.34`
        For random polymer positions, `length` is the bond length used in
        the random walk. For lattice systems, `length` is either the
        particle size or the bond length, depending on the lattice type.
        Has no effect if :code:`N_p=1` or :code:`lattice=None`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    flexible : `bool`, default: :code:`False`
        Specifies whether `dimensions` can be modified to satisfy the
        lattice periodicity, if applicable. For example, if the provided
        :math:`z`-dimension can hold a non-integer 19.7 replicas of a
        lattice, then it is updated to reflect the width of 20 replicas.
        To ignore a direction (and make that dimension constant), such
        as when creating walls, set that dimension to :code:`0` in
        `dimensions`.

    bonds : `bool`, default: :code:`False`
        Determines whether bond information is returned for polymeric
        systems. Has no effect if :code:`N_p=1`.

    angles : `bool`, default: :code:`False`
        Determines whether angle information is returned for polymeric
        systems. Has no effect if :code:`N_p=1`.

    dihedrals : `bool`, default: :code:`False`
        Determines whether dihedral information is returned for polymeric
        systems. Has no effect if :code:`N_p=1`.

    randomize : `bool`, default: :code:`False`
        Determines whether the order of the replicated polymer positions
        are randomized. Has no effect if :code:`N_p=1`.

    length_unit : `openmm.unit.Unit` or `pint.Unit`, optional
        Length unit. If not specified, it is determined automatically
        from `dimensions` or `length`.

    wrap : `bool`, default: :code:`False`
        Determines whether particles outside the simulation box are
        wrapped back into the main unit cell.

    Returns
    -------
    positions : `numpy.ndarray` or `openmm.unit.Quantity`
        Particle positions.

        **Shape**: :math:`(N,\\,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    bonds : `numpy.ndarray`
        Pairs of all bonded particle indices. Only returned if
        :code:`bonds=True`.

        **Shape**: :math:`(N_\\mathrm{bonds},\\,2)`.

    angles : `numpy.ndarray`
        Triples of all particle indices that form an angle. Only
        returned if :code:`angles=True`.

        **Shape**: :math:`(N_\\mathrm{angles},\\,3)`.

    dihedrals : `numpy.ndarray`
        Quadruples of all particle indices that form a dihedral. Only
        returned if :code:`dihedrals=True`.

        **Shape**: :math:`(N_\\mathrm{dihedrals},\\,4)`.

    dimensions : `numpy.ndarray` or `openmm.unit.Quantity`
        Dimensions for lattice systems. Only returned if `lattice` is
        not :code:`None`.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{nm}`.
    """

    # Get raw numerical dimensions and length
    if isinstance(dimensions, app.Topology):
        pbv, length_unit = strip_unit(dimensions.getPeriodicBoxVectors(), 
                                      length_unit)
        pbv = np.asarray(pbv)
    else:
        dimensions, length_unit = strip_unit(dimensions, length_unit)
        pbv = get_cell_representation(np.asarray(dimensions), "vectors")
    dimensions = np.diag(pbv)
    length, length_unit = strip_unit(length, length_unit)
    length_unit = length_unit or 1

    if lattice is None:

        # Ensure the user-specified values are valid
        if N is None:
            raise ValueError("The number of particles N must be specified.")
        if not isinstance(N, (int, np.integer)):
            raise ValueError("The number of particles N must be an integer.")
        if not (1 <= N_p <= N and isinstance(N_p, (int, np.integer))):
            emsg = ("The number of particles N_p in each segment must "
                    "be an integer between 1 and N.")
            raise ValueError(emsg)
        if N_p > 1 and N % N_p != 0:
            emsg = (f"{N=} particles cannot be evenly divided into segments "
                    f"with {N_p=} particles.")
            raise ValueError(emsg)

        # Generate particle positions for a random melt
        if N_p == 1:
            return (np.random.rand(N, 3) @ pbv.T) * length_unit
        else:
            topo = []

            # Determine unit cell information for each segment
            segments = N // N_p
            n_cells = get_closest_factors(segments, 3)
            cell_dims = dimensions / n_cells

            # Randomly generate a segment within the unit cell
            cell_pos = np.zeros((N_p, 3))
            cell_pos[0] = cell_dims / 4
            rng = np.random.default_rng()
            for i in range(1, N_p):
                vec = rng.random(3) * 2 - 1
                cell_pos[i] = cell_pos[i - 1] + length * vec / np.linalg.norm(vec)

            # Replicate unit cell in x-, y-, and z-directions (and
            # randomize, if desired)
            pos = replicate(cell_dims, cell_pos, n_cells)
            if randomize:
                pos = np.vstack(rng.permutation(pos.reshape((segments, -1, 3))))

            # Wrap particles past the system boundaries
            if wrap:
                for i in range(3):
                    pos[pos[:, i] < 0, i] += dimensions[i]
                    pos[pos[:, i] > dimensions[i], i] -= dimensions[i]

            topo.append((pos / dimensions @ pbv.T) * length_unit)

            # Determine all bonds
            if bonds:
                topo.append(np.array([(i * N_p + j, i * N_p + j + 1)
                                      for i in range(segments)
                                      for j in range(N_p - 1)]))

            # Determine all angles
            if angles:
                topo.append(np.array([np.arange(i * N_p + j, i * N_p + j + 3)
                                      for i in range(segments)
                                      for j in range(N_p - 2)]))

            # Determine all dihedrals
            if dihedrals:
                topo.append(np.array([np.arange(i * N_p + j, i * N_p + j + 4)
                                      for i in range(segments)
                                      for j in range(N_p - 3)]))

            return topo[0] if len(topo) == 1 else tuple(topo)
    else:
        around = np.around if flexible else np.floor

        # Set unit cell information
        if lattice == "cubic":
            dims = dimensions.copy()
            dims[np.isclose(dims, 0)] = 1
            n_cells = around(dims / length).astype(int)
            cell_dims = length * np.ones(3, dtype=float)
            x, y, z = (length * np.arange(n, dtype=float) for n in n_cells)
            pos = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
        else:
            if lattice == "fcc":
                cell_dims = length * np.array((1, np.sqrt(3), 3 * np.sqrt(6) / 3))
                cell_pos = length * np.array((
                    (0, 0, 0),
                    (0.5, np.sqrt(3) / 2, 0),
                    (0.5, np.sqrt(3) / 6, np.sqrt(6) / 3),
                    (0, 2 * np.sqrt(3) / 3, np.sqrt(6) / 3),
                    (0, np.sqrt(3) / 3, 2 * np.sqrt(6) / 3),
                    (0.5, 5 * np.sqrt(3) / 6, 2 * np.sqrt(6) / 3),
                ))
            elif lattice == "hcp":
                cell_dims = length * np.array((1, np.sqrt(3), 2 * np.sqrt(6) / 3))
                cell_pos = length * np.array((
                    (0, 0, 0),
                    (0.5, np.sqrt(3) / 2, 0),
                    (0.5, np.sqrt(3) / 6, np.sqrt(6) / 3),
                    (0, 2 * np.sqrt(3) / 3, np.sqrt(6) / 3)
                ))
            elif lattice == "honeycomb":
                cell_dims = length * np.array((np.sqrt(3), 3, np.inf))
                cell_pos = length * np.array((
                    (0, 0, 0),
                    (0, 1, 0),
                    (np.sqrt(3) / 2, 1.5, 0),
                    (np.sqrt(3) / 2, 2.5, 0)
                ))

            # Determine unit cell multiples
            n_cells = around(dimensions / cell_dims).astype(int)
            n_cells[n_cells == 0] = 1
            cell_dims[np.isinf(cell_dims)] = 0

            # Replicate unit cell in x-, y-, and z-directions
            pos = replicate(cell_dims, cell_pos, n_cells)

        # Remove particles outside of system boundaries
        if flexible:
            n_cells[np.isclose(dimensions, 0)] = 0
            pos = pos[~np.any(pos[:, np.isclose(dimensions, 0)] > 0, axis=1)]
        else:
            pos = pos[~np.any(pos > dimensions, axis=1)]

        return (
            (np.divide(pos, dimensions, 
                       out=np.zeros_like(pos),
                       where=dimensions != 0) @ pbv.T) * length_unit, 
            n_cells * cell_dims * length_unit
        )

def unwrap(
        positions: np.ndarray[float], positions_old: np.ndarray[float],
        dimensions: np.ndarray[float], *, thresholds: float = None,
        images: np.ndarray[int] = None, in_place: bool = True
    ) -> Union[None, tuple[np.ndarray[float], np.ndarray[float],
                           np.ndarray[int]]]:

    r"""
    Globally unwraps particle positions.

    Parameters
    ----------
    positions : `numpy.ndarray`
        Particle positions.

        **Shape**: :math:`(N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    positions_old : `numpy.ndarray`
        Previous particle positions.

        **Shape**: :math:`(N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    dimensions : `numpy.ndarray`
        System dimensions.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    thresholds : `float`, keyword-only, optional
        Maximum distances in each direction a particle can move before
        it is considered to have crossed a boundary.

        **Reference unit**: :math:`\mathrm{nm}`.

    images : `numpy.ndarray`, keyword-only, optional
        Current image flags (or number of boundary crossings).

        **Shape**: :math:`(N,\,3)`.

    in_place : `bool`, keyword-only, default: :code:`False`
        Determines whether the input array is modified in-place.

    Returns
    -------
    positions : `numpy.ndarray`
        Unwrapped particle positions. Only returned if
        :code:`in_place=False`.

        **Shape**: :math:`(N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    positions_old : `numpy.ndarray`
        Updated previous particle positions. Only returned if
        :code:`in_place=False`.

        **Shape**: :math:`(N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    images : `numpy.ndarray`
        Updated image flags (or number of boundary crossings). Only
        returned if :code:`in_place=False`.

        **Shape**: :math:`(N,\,3)`.
    """

    if thresholds is None:
        thresholds = np.min(dimensions) / 2
    if images is None:
        images = np.zeros_like(positions, dtype=int)

    dpos = positions - positions_old
    mask = np.abs(dpos) >= thresholds
    if in_place:
        images[mask] -= np.sign(dpos[mask]).astype(int)
        positions_old[:] = positions[:]
        positions += images * dimensions
    else:
        positions = positions.copy()
        positions_old = positions.copy()
        images = images.copy()
        images[mask] -= np.sign(dpos[mask]).astype(int)
        positions += images * dimensions
        return positions, positions_old, images

def unwrap_edge(
        group: mda.AtomGroup = None, *, positions: np.ndarray[float] = None,
        bonds: np.ndarray[int] = None, dimensions: np.ndarray[float] = None,
        thresholds: np.ndarray[float] = None, masses: np.ndarray[float] = None,
    ) -> np.ndarray[float]:

    r"""
    Locally unwraps the positions of molecules at the edge of the
    simulation box.

    Parameters
    ----------
    group : `MDAnalysis.AtomGroup`, optional
        Atom group. If not provided, the atom positions, bonds, and
        system dimensions must be provided in `positions`, `bonds`, and
        `dimensions`, respectively.

    positions : `numpy.ndarray`, keyword-only, optional
        Atom positions.

        **Shape**: :math:`(N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    bonds : `numpy.ndarray`, keyword-only, optional
        Pairs of all bonded atom indices in reference to `positions`.

        **Shape**: :math:`(N_\mathrm{bonds},\,2)`.

    dimensions : `numpy.ndarray`, keyword-only, optional
        System dimensions and, optionally, angles.

        **Shape**: :math:`(3,)` or :math:`(6,)`.

        **Reference unit**: :math:`\mathrm{nm}` (lengths) and
        :math:`^\circ` (angles).

    thresholds : `numpy.ndarray`, keyword-only, optional
        Maximum distances in each direction an atom can move before it
        is considered to have crossed a boundary.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    masses : `numpy.ndarray`, keyword-only, optional
        Atom masses. If not specified, all atoms are assumed to have the
        same mass.

        **Shape**: :math:`(N,)`.

        **Reference unit**: :math:`\mathrm{g/mol}`.

    Returns
    -------
    positions : `numpy.ndarray`
        Unwrapped atom positions.

        **Shape**: :math:`(N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.
    """

    if group is not None:
        return group.unwrap()

    elif positions is not None:
        if bonds is None:
            emsg = "Bond information must be specified in 'bonds'."
            raise ValueError(emsg)
        if dimensions is None:
            emsg = "System dimensions must be specified in 'dimensions'."
            raise ValueError(emsg)
        elif len(dimensions) == 3:
            dimensions = np.concatenate((dimensions, (90, 90, 90)))
        if thresholds is None:
            thresholds = dimensions[:3] / 2

        # Get graph of connected atoms
        bond_pairs = {}
        for a, b in bonds:
            bond_pairs.setdefault(a, []).append(b)
            bond_pairs.setdefault(b, []).append(a)

        # Determine indices of connected atoms in each molecule
        molecules = find_connected_nodes(bond_pairs)

        # Specify which atoms have had their positions updated
        done = {m[0] for m in molecules}
        todo = set(range(len(positions))).difference(done)

        # Upwrap atom positions using that of the nearest bonded atom
        # that has already been unwrapped
        while len(todo) > 0:
            for particle_index in todo:
                for bonded_index in bond_pairs[particle_index]:
                    if bonded_index in done:
                        positions[particle_index] = (
                            positions[bonded_index]
                            + minimize_vectors(positions[particle_index]
                                               - positions[bonded_index],
                                               dimensions)
                        )
                        done.add(particle_index)
                        break
            todo -= done

        if masses is None:
            wmsg = ("No masses specified. All atoms are assumed to "
                    "have a mass of 1.")
            warnings.warn(wmsg)
            masses = np.ones(len(positions))
        elif len(masses) == len(molecules):
            masses = np.concatenate(masses)
        elif len(masses) != len(positions):
            emsg = ("The number of masses must be equal to the number "
                    "of atoms or the number of molecules.")
            raise ValueError(emsg)

        # Recenter molecules using their centers of mass
        for molecule_indices in molecules:
            com = center_of_mass(positions=positions[molecule_indices],
                                 masses=masses[molecule_indices])
            positions[molecule_indices] \
                += wrap(com, dimensions[:3], in_place=False) - com

        return positions
    else:
        raise ValueError("Either 'group' or 'positions' must be specified.")

def wrap(
        positions: np.ndarray[float], dimensions: np.ndarray[float], *,
        in_place: bool = True) -> np.ndarray[float]:

    r"""
    Wraps particle positions back into the primary simulation cell.

    Parameters
    ----------
    positions : `numpy.ndarray`
        Particle positions.

        **Shape**: :math:`(N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    dimensions : `numpy.ndarray`
        System dimensions.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    in_place : `bool`, keyword-only, default: :code:`False`
        Determines whether the input array is modified in-place.

    Returns
    -------
    positions : `numpy.ndarray`
        Wrapped particle positions. Only returned if
        :code:`in_place=False`.

        **Shape**: :math:`(N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.
    """

    wrap_indices = (positions < 0) | (positions > dimensions)
    if in_place:
        positions[wrap_indices] -= (
            np.floor(positions / dimensions) * dimensions
        )[wrap_indices]
    else:
        positions = positions.copy()
        positions[wrap_indices] -= (
            np.floor(positions / dimensions) * dimensions
        )[wrap_indices]
        return positions

def reduce_box_vectors(vectors: np.ndarray[float]) -> np.ndarray[float]:

    r"""
    Performs lattice reduction on box vectors.

    Parameters
    ----------
    vectors : `numpy.ndarray`
        Box vectors :math:`\mathbf{a}`, :math:`\mathbf{b}`, and
        :math:`\mathbf{c}`, provided as rows in a matrix.

        **Shape**: :math:`(3,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    Returns
    -------
    reduced_vectors : `numpy.ndarray`
        Reduced box vectors.

        **Shape**: :math:`(3,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.
    """

    vectors = np.asarray(vectors)
    if is_lower_triangular(vectors):
        return vectors
    a, b, c = vectors
    c -= b * np.round(c[1] / b[1])
    c -= a * np.round(c[0] / a[0])
    b -= a * np.round(b[0] / a[0])
    return np.vstack((a, b, c))

def convert_cell_representation(
        *, parameters: np.ndarray[float] = None,
        vectors: np.ndarray[float] = None) -> np.ndarray[float]:
    
    r"""
    Converts between crystallographic lattice parameters and triclinic
    box vectors.

    Parameters
    ----------
    parameters : array-like, optional
        Lattice parameters :math:`(a,\,b,\,c,\,\alpha,\,\beta,\,\gamma)`.

        **Shape**: :math:`(6,)`.

        **Reference unit**: :math:`\mathrm{nm}` (lengths) and
        :math:`^\circ` (angles).

    vectors : array-like, optional
        Box vectors :math:`\mathbf{a}`, :math:`\mathbf{b}`, and
        :math:`\mathbf{c}`, provided as rows in a matrix, in their
        reduced forms:

        .. math::

           \begin{align*}
             \mathbf{a} &= (a_x,\,0,\,0) \\
             \mathbf{b} &= (b_x,\,b_y,\,0) \\
             \mathbf{c} &= (c_x,\,c_y,\,c_z)
           \end{align*}
           \quad\mathrm{where}\quad 
           a_x>0,\,b_y>0,\,c_z>0,\,
           a_x\geq2|b_x|,\,a_x\geq2|c_x|,\,b_y\geq2|c_y|

        **Shape**: :math:`(3,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    Returns
    -------
    representation : `numpy.ndarray`
        Lattice parameters or box vectors.
    """
    
    if parameters is None and vectors is None:
        emsg = "Either 'parameters' or 'vectors' must be specified."
        raise ValueError(emsg)
    elif parameters is not None and vectors is not None:
        emsg = "Only one of 'parameters' or 'vectors' can be specified."
        raise ValueError(emsg)

    if parameters is not None:
        if len(parameters) == 3:
            warnings.warn("Only cell lengths are specified. Assuming cubic cell.")
            parameters = np.concatenate((parameters, (90, 90, 90)))
        elif len(parameters) != 6:
            emsg = "Invalid number of lattice parameters in 'parameters'."
            raise ValueError(emsg)
        else:
            parameters = parameters.copy()

        alpha, beta, gamma = np.radians(parameters[3:])
        vectors = np.zeros((3, 3))
        vectors[0, 0] = parameters[0]
        vectors[1, 0] = parameters[1] * np.cos(gamma)
        vectors[1, 1] = parameters[1] * np.sin(gamma)
        vectors[2, 0] = parameters[2] * np.cos(beta)
        vectors[2, 1] = (parameters[2] * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) 
                         / np.sin(gamma))
        vectors[2, 2] = np.sqrt(parameters[2] ** 2 - vectors[2, 0] ** 2 
                                - vectors[2, 1] ** 2)
        vectors[np.isclose(vectors, 0, atol=5e-6)] = 0
        return vectors
    
    else:
        if not is_lower_triangular(vectors):
            vectors = reduce_box_vectors(vectors)
        parameters = np.empty(6)
        parameters[:3] = np.linalg.norm(vectors, axis=1)
        parameters[3] = np.degrees(np.arccos(np.dot(vectors[1], vectors[2]) 
                                             / (parameters[1] * parameters[2])))
        parameters[4] = np.degrees(np.arccos(np.dot(vectors[0], vectors[2]) 
                                             / (parameters[0] * parameters[2])))
        parameters[5] = np.degrees(np.arccos(np.dot(vectors[0], vectors[1]) 
                                             / (parameters[0] * parameters[1])))
        return parameters
    
def get_cell_representation(
        rep: np.ndarray[float], output: str, /
    ) -> np.ndarray[float]:

    r"""
    Gets the desired cell representation given a starting cell 
    representation.

    Parameters
    ----------
    rep : `numpy.ndarray`, positional-only
        Starting cell representation. Can be system dimensions,
        lattice parameters, or box vectors.

        **Shape**: :math:`(3,)`, :math:`(6,)`, or :math:`(3,\,3)`.

    output : `str`, positional-only
        Desired cell representation format.

        **Valid values**: :code:`"dimensions"`, :code:`"parameters"`,
        and :code:`"vectors"`.

    Returns
    -------
    representation : `numpy.ndarray`
        Desired cell representation.

        **Shape**: :math:`(3,)`, :math:`(6,)`, or :math:`(3,\,3)`.
    """

    rep = np.asarray(rep)
    if rep.ndim == 1:
        if rep.shape[0] == 3:
            input_ = "dimensions"
        elif rep.shape[0] == 6:
            input_ = "parameters"
        else:
            emsg = ("Invalid shape for 'rep'. Dimensions or lattice "
                    "parameters must be provided as a 3- or 6-element "
                    "array, respectively.")
            raise ValueError(emsg)
    elif rep.ndim == 2:
        if rep.shape != (3, 3):
            emsg = ("Invalid shape for 'rep'. Box vectors must be "
                    "provided as rows in a 3-by-3 matrix.")
            raise ValueError(emsg)
        input_ = "vectors"
    else:
        emsg = ("Invalid shape for 'rep'. Must be a 3- or 6-element "
                "array, or a 3-by-3 matrix.")
        raise ValueError(emsg)

    if input_ == output:
        return rep

    if output == "dimensions":
        if input_ == "parameters":
            return np.diag(convert_cell_representation(parameters=rep))
        elif input_ == "vectors":
            return np.diag(rep)
    elif output == "parameters":
        if input_ == "dimensions":
            return np.concatenate((rep, (90, 90, 90)))
        elif input_ == "vectors":
            return convert_cell_representation(vectors=rep)
    elif output == "vectors":
        if input_ == "dimensions":
            return np.diag(rep)
        elif input_ == "parameters":
            return convert_cell_representation(parameters=rep)