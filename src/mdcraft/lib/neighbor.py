from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING

from numba import njit
from numba.typed import List
import numpy as np

from .. import Q_
from .bit import count_leading_zeros
from .cell import (
    _invert_box_vectors,
    _scale_coordinates,
    check_orthogonality,
    convert_cell_representation,
    reduce_box_vectors,
)
from ..lib.unit import strip_unit

if TYPE_CHECKING:  # pragma: no cover
    from .. import int_t, float_t


@njit(fastmath=True, inline="always")  # pragma: no cover
def _check_positions(
    positions: np.ndarray[float_t], dimensions: np.ndarray[float_t]
) -> np.bool_:
    """
    Checks whether all particle positions are within the bounds of the
    simulation box.

    Parameters
    ----------
    positions : `numpy.ndarray`
        Particle positions :math:`\\mathrm{r}` in Cartesian or
        fractional coordinates.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

        **Reference unit**: :math:`\\mathrm{nm}` (Cartesian) or unitless
        (fractional).

    dimensions : `numpy.ndarray`
        Box dimensions :math:`(L_x,L_y[,L_z])` (Cartesian) or
        :math:`(1,1[,1])` (fractional).

        **Shape**: :math:`(2,)` or :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{nm}` (Cartesian) or unitless
        (fractional).

    Returns
    -------
    all_inside : `bool`
        Whether all particles are inside the simulation box.
    """
    for pid in range(positions.shape[0]):
        for dim in range(positions.shape[1]):
            if (
                positions[pid, dim] < 0.0
                or positions[pid, dim] > dimensions[dim]
            ):
                return False
    return True


@njit(fastmath=True, inline="always")  # pragma: no cover
def _compute_squared_distance_orthogonal(
    position_i: np.ndarray[float_t],
    position_j: np.ndarray[float_t],
    dimensions: np.ndarray[float_t],
    pbc: np.bool_,
) -> float_t:
    """
    Computes the squared separation distance :math:`r_{ij}^2` between
    two particles in an orthogonal simulation box.

    Parameters
    ----------
    position_i : `numpy.ndarray`
        Position of the first particle :math:`\\mathrm{r}_i`.

        **Shape**: :math:`(2,)` or :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    position_j : `numpy.ndarray`
        Position of the second particle :math:`\\mathrm{r}_j`.

        **Shape**: :math:`(2,)` or :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    dimensions : `numpy.ndarray`
        Box dimensions :math:`(L_x,L_y[,L_z])`.

        **Shape**: :math:`(2,)` or :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    pbc : `bool`
        Specifies whether to apply periodic boundary conditions (PBC)
        and use the minimum image convention when calculating the
        squared separation distance between the two particles.

    Returns
    -------
    dr_squared : `float`
        Squared separation distance :math:`r_{ij}^2` between the two
        particles.
    """
    dr_squared = 0.0
    for dim in range(position_i.shape[0]):
        dr = position_j[dim] - position_i[dim]
        if pbc:
            dr -= dimensions[dim] * round(dr / dimensions[dim])
        dr_squared += dr * dr
    return dr_squared


@njit(fastmath=True, inline="always")  # pragma: no cover
def _compute_squared_distance(
    position_i: np.ndarray[float_t],
    position_j: np.ndarray[float_t],
    box_vectors: np.ndarray[float_t],
    pbc: np.bool_,
) -> float_t:
    """
    Computes the squared separation distance :math:`r_{ij}^2` between
    two particles in a triclinic simulation box.

    Parameters
    ----------
    position_i : `numpy.ndarray`
        Position of the first particle :math:`\\mathrm{r}_i`.

        **Shape**: :math:`(2,)` or :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    position_j : `numpy.ndarray`
        Position of the second particle :math:`\\mathrm{r}_j`.

        **Shape**: :math:`(2,)` or :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    box_vectors : `numpy.ndarray`
        Box vectors :math:`(\\mathbf{A};\\mathbf{B}[;\\mathbf{C}])`.

        **Shape**: :math:`(2,2)` or :math:`(3,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    pbc : `bool`
        Specifies whether to apply periodic boundary conditions (PBC)
        and use the minimum image convention when calculating the
        squared separation distance between the two particles.

    Returns
    -------
    dr_squared : `float`
        Squared separation distance :math:`r_{ij}^2` between the two
        particles.
    """
    # Compute the separation distance vector
    n_dimensions = position_i.shape[0]
    dr = np.empty(n_dimensions, position_i.dtype)
    for dim in range(n_dimensions):
        dr[dim] = position_j[dim] - position_i[dim]

    # Apply periodic boundary conditions (PBC) to get the minimum image
    # convention
    if pbc:
        for axis in range(n_dimensions - 1, -1, -1):
            # Compute how many box vectors to subtract by projecting the
            # displacement onto the current box vector
            factor = np.floor(dr[axis] / box_vectors[axis, axis] + 0.5)

            # Wrap the displacement back into the central image along
            # this axis
            for dim in range(n_dimensions):
                dr[dim] -= factor * box_vectors[axis, dim]

    # Compute the squared separation distance
    dr_squared = 0.0
    for dim in range(n_dimensions):
        dr_squared += dr[dim] * dr[dim]
    return dr_squared


@njit(fastmath=True)  # pragma: no cover
def _build_neighbor_list_orthogonal_brute_force(
    positions: np.ndarray[float_t],
    cutoff: float_t,
    dimensions: np.ndarray[float_t],
    pbc: np.bool_,
) -> List[set[np.uint32]]:
    """
    Builds a half neighbor list for particles in an orthogonal
    simulation box using a brute-force approach.

    Parameters
    ----------
    positions : `numpy.ndarray`
        Particle positions :math:`\\mathrm{r}`.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    cutoff : `float`
        Cutoff distance :math:`r_\\mathrm{cutoff}` for neighbor search.

        **Reference unit**: :math:`\\mathrm{nm}`.

    dimensions : `numpy.ndarray`
        Box dimensions :math:`(L_x,L_y[,L_z])`.

        **Shape**: :math:`(2,)` or :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    pbc : `bool`
        Specifies whether to apply periodic boundary conditions (PBC)
        and use the minimum image convention when calculating
        separation distances between particles.

    Returns
    -------
    neighbor_lists : `list`
        A list of neighbor lists (sets) for all particles :math:`i`,
        with each inner variable-length set containing the indices of
        nearby particles :math:`j`, where :math:`i<j`, that are within
        the cutoff distance of particle :math:`i`.

        **Shape**: :math:`(N,)`.
    """
    n_particles = positions.shape[0]
    cutoff_squared = cutoff * cutoff
    neighbor_lists = List()
    for pid in range(n_particles):
        neighbor_list = set()
        for nid in range(pid + 1, n_particles):
            if (
                _compute_squared_distance_orthogonal(
                    positions[pid], positions[nid], dimensions, pbc
                )
                < cutoff_squared
            ):
                neighbor_list.add(np.uint32(nid))
        neighbor_lists.append(neighbor_list)
    return neighbor_lists


@njit(fastmath=True)  # pragma: no cover
def _build_neighbor_list_brute_force(
    positions: np.ndarray[float_t],
    cutoff: float_t,
    box_vectors: np.ndarray[float_t],
    pbc: np.bool_,
) -> List[set[np.uint32]]:
    """
    Builds a half neighbor list for particles in a triclinic simulation
    box using a brute-force approach.

    Parameters
    ----------
    positions : `numpy.ndarray`
        Particle positions :math:`\\mathrm{r}` in Cartesian coordinates.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    scaled_positions : `numpy.ndarray`
        Particle positions :math:`\\mathrm{r}` in fractional
        coordinates.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

    cutoff : `float`
        Cutoff distance :math:`r_\\mathrm{cutoff}` for neighbor search.

        **Reference unit**: :math:`\\mathrm{nm}`.

    box_vectors : `numpy.ndarray`
        Box vectors :math:`(\\mathbf{A};\\mathbf{B}[;\\mathbf{C}])`.

        **Shape**: :math:`(2,2)` or :math:`(3,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    pbc : `bool`
        Specifies whether to apply periodic boundary conditions (PBC)
        and use the minimum image convention when calculating
        separation distances between particles.

    Returns
    -------
    neighbor_lists : `list`
        A list of neighbor lists (sets) for all particles :math:`i`,
        with each inner variable-length set containing the indices of
        nearby particles :math:`j`, where :math:`i<j`, that are within
        the cutoff distance of particle :math:`i`.

        **Shape**: :math:`(N,)`.
    """
    n_particles = positions.shape[0]
    cutoff_squared = cutoff * cutoff
    neighbor_lists = List()
    for pid in range(n_particles):
        neighbor_list = set()
        for nid in range(pid + 1, n_particles):
            if (
                _compute_squared_distance(
                    positions[pid], positions[nid], box_vectors, pbc
                )
                < cutoff_squared
            ):
                neighbor_list.add(np.uint32(nid))
        neighbor_lists.append(neighbor_list)
    return neighbor_lists


@njit(fastmath=True, inline="always")  # pragma: no cover
def _get_cell_offsets_orthogonal(
    n_dimensions: np.uint8,
) -> tuple[np.uint8, np.ndarray[np.int8]]:
    """
    Returns the number and index offsets of neighboring cells in the
    forward half-shell that may contain particles within a cutoff
    distance in an orthogonal simulation box.

    Parameters
    ----------
    n_dimensions : `numpy.uint8`
        Dimensionality of the simulation box :math:`d`.

        **Valid values**: :code:`2` or :code:`3`.

    Returns
    -------
    n_offsets : `numpy.uint8`
        Number of neighboring cells :math:`N_\\mathrm{offsets}`.

    cell_offsets : `numpy.ndarray`
        Index offsets of neighboring cells.

        **Shape**: :math:`(N_\\mathrm{offsets},d)`.
    """
    if n_dimensions == 2:
        return np.uint8(5), np.array(
            ((0, 0), (0, 1), (1, -1), (1, 0), (1, 1)), np.int8
        )

    # n_dimensions == 3
    return np.uint8(14), np.array(
        (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, -1),
            (0, 1, 0),
            (0, 1, 1),
            (1, -1, -1),
            (1, -1, 0),
            (1, -1, 1),
            (1, 0, -1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, -1),
            (1, 1, 0),
            (1, 1, 1),
        ),
        np.int8,
    )


@njit(fastmath=True, inline="always")  # pragma: no cover
def _build_cell_lists_orthogonal(
    positions: np.ndarray[float_t],
    cutoff: float_t,
    dimensions: np.ndarray[float_t],
    pbc: np.bool_,
) -> tuple[
    np.ndarray[np.uint32],
    np.ndarray[np.uint32],
    np.ndarray[int_t],
    np.ndarray[int_t],
]:
    """
    Builds cell lists for particles in an orthogonal simulation box.

    Parameters
    ----------
    positions : `numpy.ndarray`
        Particle positions :math:`\\mathrm{r}`.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    cutoff : `float`
        Cutoff distance :math:`r_\\mathrm{cutoff}` for neighbor search.

        **Reference unit**: :math:`\\mathrm{nm}`.

    dimensions : `numpy.ndarray`
        Box dimensions :math:`(L_x,L_y[,L_z])`.

        **Shape**: :math:`(2,)` or :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    pbc : `bool`
        Specifies whether to apply periodic boundary conditions (PBC)
        and use the minimum image convention when calculating
        separation distances between particles.

    Returns
    -------
    n_cells : `numpy.ndarray`
        Number of cells in each dimension :math:`(N_x,N_y[,N_z])`
        that the simulation box is split into.

        **Shape**: :math:`(2,)` or :math:`(3,)`.

    particle_cell_indices : `numpy.ndarray`
        Cell indices for each particle :math:`(i_x,i_y[,i_z])`.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

    cell_heads : `numpy.ndarray`
        Linked list heads for each cell, where each entry contains the
        index of the first particle in the cell.

        **Shape**: :math:`(N_x,N_y[,N_z])`.

    cell_lists : `numpy.ndarray`
        Linked list of particles in each cell, where each entry contains
        either the index of the next particle in the cell or :code:`-1`
        to indicate the end of the list.

        **Shape**: :math:`(N,)`.
    """
    # Split simulation domain into cells
    n_particles, n_dimensions = positions.shape
    n_cells = np.empty(n_dimensions, np.uint32)
    inv_cell_sizes = np.empty(n_dimensions, dimensions.dtype)
    for dim in range(n_dimensions):
        n_cells[dim] = max(np.uint32(dimensions[dim] / cutoff), 1)
        inv_cell_sizes[dim] = n_cells[dim] / dimensions[dim]

    # Get cell indices for each particle and create linked list for
    # each cell
    cell_heads = np.full(
        (n_cells[0], n_cells[1], 1 if n_dimensions == 2 else n_cells[2]),
        -1,
        np.int64,
    )
    cell_lists = np.empty(n_particles, np.int64)
    particle_cell_indices = np.empty((n_particles, n_dimensions), np.uint32)
    for pid in range(n_particles):
        for dim in range(n_dimensions):
            particle_cell_indices[pid, dim] = np.uint32(
                positions[pid, dim] * inv_cell_sizes[dim]
            )
            if pbc:
                particle_cell_indices[pid, dim] %= n_cells[dim]
        if n_dimensions == 2:
            cix, ciy = particle_cell_indices[pid]
            ciz = 0
        else:
            cix, ciy, ciz = particle_cell_indices[pid]
        cell_lists[pid] = cell_heads[cix, ciy, ciz]
        cell_heads[cix, ciy, ciz] = pid

    return n_cells, particle_cell_indices, cell_heads, cell_lists


@njit(fastmath=True)  # pragma: no cover
def _build_neighbor_list_orthogonal_cell_list(
    positions: np.ndarray[float_t],
    cutoff: float_t,
    dimensions: np.ndarray[float_t],
    pbc: np.bool_,
) -> List[set[np.uint32]]:
    """
    Builds a neighbor list for particles in an orthogonal simulation
    box using the cell list algorithm.

    Parameters
    ----------
    positions : `numpy.ndarray`
        Particle positions :math:`\\mathrm{r}`.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    cutoff : `float`
        Cutoff distance :math:`r_\\mathrm{cutoff}` for neighbor search.

        **Reference unit**: :math:`\\mathrm{nm}`.

    dimensions : `numpy.ndarray`
        Box dimensions :math:`(L_x,L_y[,L_z])`.

        **Shape**: :math:`(2,)` or :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    pbc : `bool`
        Specifies whether to apply periodic boundary conditions (PBC)
        and use the minimum image convention when calculating
        separation distances between particles.

    Returns
    -------
    neighbor_lists : `list`
        A list of neighbor lists (sets) for all particles :math:`i`,
        with each inner variable-length set containing the indices of
        nearby particles :math:`j`, where :math:`i<j`, that are within
        the cutoff distance of particle :math:`i`.

        **Shape**: :math:`(N,)`.
    """
    # Build cell lists
    n_dimensions = positions.shape[1]
    n_cells, particle_cell_indices, cell_heads, cell_lists = (
        _build_cell_lists_orthogonal(positions, cutoff, dimensions, pbc)
    )

    # Define offsets for neighboring cells
    n_offsets, cell_offsets = _get_cell_offsets_orthogonal(n_dimensions)

    # Build neighbor list for each particle
    neighbor_lists = List()
    cutoff_squared = cutoff * cutoff
    for pid in range(positions.shape[0]):
        neighbor_list = set()
        ix, iy = particle_cell_indices[pid, :2]

        # Check current and forward neighboring cells
        for idx in range(n_offsets):
            jx = (ix + cell_offsets[idx, 0]) % n_cells[0]
            jy = (iy + cell_offsets[idx, 1]) % n_cells[1]
            if n_dimensions == 2:
                if not pbc and max(abs(ix - jx), abs(iy - jy)) > 1:
                    continue
                nid = cell_heads[jx, jy, 0]
            else:
                iz = particle_cell_indices[pid, 2]
                jz = (iz + cell_offsets[idx, 2]) % n_cells[2]
                if (
                    not pbc
                    and max(abs(ix - jx), abs(iy - jy), abs(iz - jz)) > 1
                ):
                    continue
                nid = cell_heads[jx, jy, jz]

            # Traverse linked list of particles in current and
            # neighboring cells
            while nid != -1:
                if pid != nid:
                    if (
                        _compute_squared_distance_orthogonal(
                            positions[pid], positions[nid], dimensions, pbc
                        )
                        < cutoff_squared
                    ):
                        if pid < nid:
                            neighbor_list.add(np.uint32(nid))
                        else:
                            neighbor_lists[nid].add(np.uint32(pid))
                nid = cell_lists[nid]

        # Add the neighbor list for the current particle
        neighbor_lists.append(neighbor_list)

    return neighbor_lists


@njit(fastmath=True, inline="always")  # pragma: no cover
def _get_cell_offsets(
    cutoff: float_t,
    box_vectors: np.ndarray[float_t],
    n_cells: np.ndarray[np.uint32],
) -> tuple[np.uint8, np.ndarray[np.int8]]:
    """
    Returns the number and index offsets of neighboring cells in the
    forward half-shell that may contain particles within a cutoff
    distance in a triclinic simulation box.

    Parameters
    ----------
    cutoff : `float`
        Cutoff distance :math:`r_\\mathrm{cutoff}` for neighbor search.

        **Reference unit**: :math:`\\mathrm{nm}`.

    box_vectors : `numpy.ndarray`
        Box vectors :math:`(\\mathbf{A};\\mathbf{B}[;\\mathbf{C}])`.

        **Shape**: :math:`(2,2)` or :math:`(3,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    n_cells : `numpy.ndarray`
        Number of cells in each dimension :math:`(N_x,N_y[,N_z])`
        that the simulation box is split into.

        **Shape**: :math:`(2,)` or :math:`(3,)`.

    Returns
    -------
    n_offsets : `numpy.uint8`
        Number of neighboring cells :math:`N_\\mathrm{offsets}`.

    cell_offsets : `numpy.ndarray`
        Index offsets of neighboring cells.

        **Shape**: :math:`(N_\\mathrm{offsets},d)`.

    cutoff_extents : `numpy.ndarray`
        Number of cell offsets to consider in each dimension.

        **Shape**: :math:`(d,)`.
    """
    # Compute the cell vectors
    n_dimensions = box_vectors.shape[0]
    cell_vectors = box_vectors.copy()
    for dim in range(n_dimensions):
        for axis in range(n_dimensions):
            cell_vectors[dim, axis] /= n_cells[dim]

    # Compute the cross products of the cell vectors
    cell_cross_products = np.empty(
        (n_dimensions, n_dimensions), box_vectors.dtype
    )
    for ai in range(n_dimensions):
        aj = (ai + 1) % n_dimensions
        ak = (ai + 2) % n_dimensions
        for di in range(n_dimensions):
            dj = (di + 1) % n_dimensions
            dk = (di + 2) % n_dimensions
            cell_cross_products[ai, di] = (
                cell_vectors[aj, dj] * cell_vectors[ak, dk]
                - cell_vectors[aj, dk] * cell_vectors[ak, dj]
            )

    # Compute the minimum distances between cells
    cell_minimum_distances = np.empty(n_dimensions, box_vectors.dtype)
    cutoff_extents = np.empty(n_dimensions, np.int8)
    for axis in range(n_dimensions):
        cell_minimum_distances[axis] = 0.0
        cross_product_magnitude_squared = 0.0
        for dim in range(n_dimensions):
            cell_minimum_distances[axis] += (
                cell_vectors[axis, dim] * cell_cross_products[axis, dim]
            )
            cross_product_magnitude_squared += (
                cell_cross_products[axis, dim] * cell_cross_products[axis, dim]
            )
        cell_minimum_distances[axis] = abs(cell_minimum_distances[axis]) / sqrt(
            cross_product_magnitude_squared
        )
        cutoff_extents[axis] = np.ceil(cutoff / cell_minimum_distances[axis])

    # Determine the number and index offsets of neighboring cells
    n_offsets = np.uint8(1)
    for extent in cutoff_extents:
        n_offsets *= 2 * extent + 1
    n_offsets //= 2
    n_offsets += 1
    cell_offsets = np.empty((n_offsets, n_dimensions), np.int8)
    cell_offsets[0] = 0
    coi = 1
    if n_dimensions == 2:
        for ix in range(-cutoff_extents[0], cutoff_extents[0] + 1):
            for iy in range(-cutoff_extents[1], cutoff_extents[1] + 1):
                if ix > 0 or (ix == 0 and iy > 0):
                    cell_offsets[coi, 0] = ix
                    cell_offsets[coi, 1] = iy
                    coi += 1
    else:  # n_dimensions == 3
        for ix in range(-cutoff_extents[0], cutoff_extents[0] + 1):
            for iy in range(-cutoff_extents[1], cutoff_extents[1] + 1):
                for iz in range(-cutoff_extents[2], cutoff_extents[2] + 1):
                    if (
                        ix > 0
                        or (ix == 0 and iy > 0)
                        or (ix == 0 and iy == 0 and iz > 0)
                    ):
                        cell_offsets[coi, 0] = ix
                        cell_offsets[coi, 1] = iy
                        cell_offsets[coi, 2] = iz
                        coi += 1
    return n_offsets, cell_offsets, cutoff_extents


@njit(fastmath=True, inline="always")  # pragma: no cover
def _build_cell_lists_triclinic(
    scaled_positions: np.ndarray[float_t],
    cutoff: float_t,
    box_vectors: np.ndarray[float_t],
    pbc: np.bool_,
) -> tuple[
    np.ndarray[np.uint32],
    np.ndarray[np.uint32],
    np.ndarray[int_t],
    np.ndarray[int_t],
]:
    """
    Builds cell lists for particles in a triclinic simulation box.

    Parameters
    ----------
    scaled_positions : `numpy.ndarray`
        Particle positions :math:`\\mathrm{r}` in fractional
        coordinates.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

    cutoff : `float`
        Cutoff distance :math:`r_\\mathrm{cutoff}` for neighbor search.

        **Reference unit**: :math:`\\mathrm{nm}`.

    box_vectors : `numpy.ndarray`
        Box vectors :math:`(\\mathbf{A};\\mathbf{B}[;\\mathbf{C}])`.

        **Shape**: :math:`(2,2)` or :math:`(3,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    pbc : `bool`
        Specifies whether to apply periodic boundary conditions (PBC)
        and use the minimum image convention when calculating
        separation distances between particles.

    Returns
    -------
    n_cells : `numpy.ndarray`
        Number of cells in each dimension :math:`(N_x,N_y[,N_z])`
        that the simulation box is split into.

        **Shape**: :math:`(2,)` or :math:`(3,)`.

    particle_cell_indices : `numpy.ndarray`
        Cell indices for each particle :math:`(i_x,i_y[,i_z])`.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

    cell_heads : `numpy.ndarray`
        Linked list heads for each cell, where each entry contains the
        index of the first particle in the cell.

        **Shape**: :math:`(N_x,N_y[,N_z])`.

    cell_lists : `numpy.ndarray`
        Linked list of particles in each cell, where each entry contains
        either the index of the next particle in the cell or :code:`-1`
        to indicate the end of the list.

        **Shape**: :math:`(N,)`.
    """
    # Split simulation domain into cells
    n_particles, n_dimensions = scaled_positions.shape
    n_cells = np.empty(n_dimensions, np.uint32)
    for dim in range(n_dimensions):
        box_length_squared = 0.0
        for axis in range(n_dimensions):
            box_length_squared += (
                box_vectors[dim, axis] * box_vectors[dim, axis]
            )
        n_cells[dim] = max(np.uint32(sqrt(box_length_squared) / cutoff), 1)

    # Get cell indices for each particle and create linked list for
    # each cell
    cell_heads = np.full(
        (n_cells[0], n_cells[1], 1 if n_dimensions == 2 else n_cells[2]),
        -1,
        np.int64,
    )
    cell_lists = np.empty(n_particles, np.int64)
    particle_cell_indices = np.empty((n_particles, n_dimensions), np.uint32)
    for pid in range(n_particles):
        for dim in range(n_dimensions):
            scaled_position = scaled_positions[pid, dim]
            if pbc:
                scaled_position %= 1.0
            particle_cell_indices[pid, dim] = np.uint32(
                scaled_position * n_cells[dim]
            )
        if n_dimensions == 2:
            cix, ciy = particle_cell_indices[pid]
            ciz = 0
        else:
            cix, ciy, ciz = particle_cell_indices[pid]
        cell_lists[pid] = cell_heads[cix, ciy, ciz]
        cell_heads[cix, ciy, ciz] = pid

    return n_cells, particle_cell_indices, cell_heads, cell_lists


@njit(fastmath=True)  # pragma: no cover
def _build_neighbor_list_cell_list(
    positions: np.ndarray[float_t],
    scaled_positions: np.ndarray[float_t],
    cutoff: float_t,
    box_vectors: np.ndarray[float_t],
    pbc: np.bool_,
) -> List[set[np.uint32]]:
    """
    Builds a half neighbor list for particles in a triclinic simulation
    box using the cell list algorithm.

    Parameters
    ----------
    positions : `numpy.ndarray`
        Particle positions :math:`\\mathrm{r}` in Cartesian coordinates.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    scaled_positions : `numpy.ndarray`
        Particle positions :math:`\\mathrm{r}` in fractional
        coordinates.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

    cutoff : `float`
        Cutoff distance :math:`r_\\mathrm{cutoff}` for neighbor search.

        **Reference unit**: :math:`\\mathrm{nm}`.

    box_vectors : `numpy.ndarray`
        Box vectors :math:`(\\mathbf{A};\\mathbf{B}[;\\mathbf{C}])`.

        **Shape**: :math:`(2,2)` or :math:`(3,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    pbc : `bool`
        Specifies whether to apply periodic boundary conditions (PBC)
        and use the minimum image convention when calculating
        separation distances between particles.

    Returns
    -------
    neighbor_lists : `list`
        A list of neighbor lists (sets) for all particles :math:`i`,
        with each inner variable-length set containing the indices of
        nearby particles :math:`j`, where :math:`i<j`, that are within
        the cutoff distance of particle :math:`i`.

        **Shape**: :math:`(N,)`.
    """
    # Build cell lists
    n_dimensions = scaled_positions.shape[1]
    n_cells, particle_cell_indices, cell_heads, cell_lists = (
        _build_cell_lists_triclinic(scaled_positions, cutoff, box_vectors, pbc)
    )

    # Define offsets for neighboring cells
    n_offsets, cell_offsets, cutoff_extents = _get_cell_offsets(
        cutoff, box_vectors, n_cells
    )

    # Build neighbor list for each particle
    neighbor_lists = List()
    cutoff_squared = cutoff * cutoff
    for pid in range(scaled_positions.shape[0]):
        neighbor_list = set()
        ix, iy = particle_cell_indices[pid, :2]

        # Check current and forward neighboring cells
        for idx in range(n_offsets):
            jx = (ix + cell_offsets[idx, 0]) % n_cells[0]
            jy = (iy + cell_offsets[idx, 1]) % n_cells[1]
            if n_dimensions == 2:
                if not pbc and (
                    abs(ix - jx) > cutoff_extents[0]
                    or abs(iy - jy) > cutoff_extents[1]
                ):
                    continue
                nid = cell_heads[jx, jy, 0]
            else:
                iz = particle_cell_indices[pid, 2]
                jz = (iz + cell_offsets[idx, 2]) % n_cells[2]
                if not pbc and (
                    abs(ix - jx) > cutoff_extents[0]
                    or abs(iy - jy) > cutoff_extents[1]
                    or abs(iz - jz) > cutoff_extents[2]
                ):
                    continue
                nid = cell_heads[jx, jy, jz]

            # Traverse linked list of particles in current and
            # neighboring cells
            while nid != -1:
                if pid != nid:
                    if (
                        _compute_squared_distance(
                            positions[pid],
                            positions[nid],
                            box_vectors,
                            pbc,
                        )
                        < cutoff_squared
                    ):
                        if pid < nid:
                            neighbor_list.add(np.uint32(nid))
                        else:
                            neighbor_lists[nid].add(np.uint32(pid))
                nid = cell_lists[nid]

        # Add the neighbor list for the current particle
        neighbor_lists.append(neighbor_list)

    return neighbor_lists


@njit(fastmath=True, inline="always")  # pragma: no cover
def _expand_10_bit_int(v: np.uint32) -> np.uint32:
    """
    Expands a 10-bit integer into a 32-bit integer for bit interleaving.

    Parameters
    ----------
    v : `numpy.uint32`
        A 10-bit integer to be expanded.

    Returns
    -------
    v : `numpy.uint32`
        The expanded 32-bit integer.
    """
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


@njit(fastmath=True, inline="always")  # pragma: no cover
def _compute_morton_codes(
    positions: np.ndarray[float_t], dimensions: np.ndarray[float_t]
) -> np.ndarray[np.uint32]:
    """
    Maps multidimensional particle positions to one-dimensional Morton
    codes for efficient spatial indexing.

    Parameters
    ----------
    positions : `numpy.ndarray`
        Particle positions :math:`\\mathrm{r}`.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    dimensions : `numpy.ndarray`
        Box dimensions :math:`(L_x,L_y[,L_z])`.

        **Shape**: :math:`(2,)` or :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    Returns
    -------
    morton_codes : `numpy.ndarray`
        Morton codes for each particle position.
    """
    n_particles, n_dimensions = positions.shape
    inv_cell_size = np.empty(n_dimensions, dimensions.dtype)
    for dim in range(n_dimensions):
        inv_cell_size[dim] = 1_024.0 / dimensions[dim]
    morton_codes = np.zeros(n_particles, np.uint32)
    for pid in range(n_particles):
        for dim in range(n_dimensions):
            morton_codes[pid] |= (
                _expand_10_bit_int(
                    min(
                        np.uint32(positions[pid, dim] * inv_cell_size[dim]),
                        1_023,
                    )
                )
                << dim
            )
    return morton_codes


@njit(fastmath=True, inline="always")  # pragma: no cover
def _build_leaf_nodes(positions, indices, nodes, traversal_indices):
    """"""
    n_particles, n_dimensions = positions.shape
    for nid in range(n_particles):
        pid = indices[nid]
        r_fp32 = positions[pid].astype(np.float32)
        nodes[nid, :n_dimensions] = np.nextafter(r_fp32, -np.inf)
        nodes[nid, n_dimensions:] = np.nextafter(r_fp32, np.inf)
        traversal_indices[nid, 0] = -indices[nid] - 1
        traversal_indices[nid, 1] = -1


@njit(fastmath=True, inline="always")  # pragma: no cover
def _find_split_index(morton_codes, first, last):
    """"""
    # Split range in half if all codes are identical
    first_code = morton_codes[first]
    last_code = morton_codes[last]
    if first_code == last_code:
        return (first + last) >> 1

    # Get common prefix for all codes
    common_prefix = count_leading_zeros(first_code ^ last_code, 32)

    # Search for furthest index where the prefix is still longer than
    # the common prefix using binary search
    split = first
    step = last - first
    while step > 1:
        step >>= 1
        mid = split + step
        if mid < last:
            mid_code = morton_codes[mid]
            prefix = count_leading_zeros(first_code ^ mid_code, 32)
            if prefix > common_prefix:
                split = mid
    return split


@njit(fastmath=True)  # pragma: no cover
def _build_internal_nodes(
    morton_codes, first, last, next_free, nodes, traversal_indices
):
    """"""
    if first == last:
        return first

    split = _find_split_index(morton_codes, first, last)
    left = _build_internal_nodes(
        morton_codes, first, split, next_free, nodes, traversal_indices
    )
    right = _build_internal_nodes(
        morton_codes, split + 1, last, next_free, nodes, traversal_indices
    )

    idx = next_free[0]
    next_free[0] += 1

    n_dimensions = nodes.shape[1] // 2
    for dim in range(n_dimensions):
        nodes[idx, dim] = min(nodes[left, dim], nodes[right, dim])
        ub_idx = dim + n_dimensions
        nodes[idx, ub_idx] = max(nodes[left, ub_idx], nodes[right, ub_idx])
    traversal_indices[idx, 0] = left
    traversal_indices[idx, 1] = right

    return idx


@njit(fastmath=True)  # pragma: no cover
def _assign_ropes(idx, rope, traversal_indices):
    if traversal_indices[idx, 0] < 0:
        traversal_indices[idx, 1] = rope
        return

    left = traversal_indices[idx, 0]
    right = traversal_indices[idx, 1]

    _assign_ropes(left, right, traversal_indices)
    _assign_ropes(right, rope, traversal_indices)

    traversal_indices[idx, 1] = rope


# @njit(fastmath=True)  # pragma: no cover
def _build_neighbor_list_orthogonal_bvh(
    positions: np.ndarray[float_t],
    cutoff: float_t,
    dimensions: np.ndarray[float_t],
    pbc: np.bool_,
) -> List[set[np.uint32]]:
    """"""
    # Compute Morton codes for particle positions and sort them
    morton_codes = _compute_morton_codes(positions, dimensions)
    sorted_indices = np.argsort(morton_codes)

    # Preallocate array to store information about leaf and internal nodes
    n_particles, n_dimensions = positions.shape
    n_nodes = 2 * n_particles - 1
    nodes = np.empty((n_nodes, 2 * n_dimensions), np.float32)
    traversal_indices = np.empty((n_nodes, 2), np.int64)

    # Build leaf nodes
    _build_leaf_nodes(positions, sorted_indices, nodes, traversal_indices)

    # Build internal nodes
    next_free = np.array((n_particles,), np.uint32)
    root = _build_internal_nodes(
        morton_codes[sorted_indices],
        0,
        n_particles - 1,
        next_free,
        nodes,
        traversal_indices,
    )

    # Assign ropes to internal nodes
    _assign_ropes(root, -1, traversal_indices)

    # TODO

    return


def build_neighbor_list(
    positions: np.ndarray[float_t] | Q_,
    cutoff: float_t | Q_,
    box_size: np.ndarray[float_t] | Q_ | None = None,
    *,
    pbc: bool = True,
    algorithm: str = "cell_list",
) -> List[set[np.uint32]]:
    """
    Builds a half neighbor list containing particle pairs within a
    cutoff distance.

    Parameters
    ----------
    positions : `numpy.ndarray` or `pint.Quantity`
        Particle positions :math:`\\mathbf{r}`.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    cutoff : `float` or `pint.Quantity`
        Cutoff distance :math:`r_\\mathrm{cutoff}` for neighbor search.

        **Reference unit**: :math:`\\mathrm{nm}`.

    box_size : `numpy.ndarray` or `pint.Quantity`, optional
        Size of the simulation box as dimensions :math:`(L_x,L_y[,L_z])`,
        lattice parameters :math:`(a,b[,c,\\alpha,\\beta],\\gamma)`, or
        box vectors :math:`(\\mathbf{a};\\mathbf{b}[;\\mathbf{c}])`. If
        not provided, the simulation box is assumed to be orthogonal and
        non-periodic.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    pbc : `bool`, keyword-only, default: :code:`True`
        Specifies whether to apply periodic boundary conditions (PBC)
        and use the minimum image convention when calculating
        separation distances between particles.

    algorithm : `str`, keyword-only, default: :code:`"cell_list"`
        Algorithm to use for building the neighbor list.

        **Valid values**: :code:`"brute_force"` and :code:`"cell_list"`.

    Returns
    -------
    neighbor_lists : `list`
        A list of neighbor lists (sets) for all particles :math:`i`,
        with each inner variable-length set containing the indices of
        nearby particles :math:`j`, where :math:`i<j`, that are within
        the cutoff distance of particle :math:`i`.

        **Shape**: :math:`(N,)`.
    """
    # Validate input arguments
    if algorithm not in (algorithms := {"brute_force", "bvh", "cell_list"}):
        raise ValueError(
            f"Invalid `algorithm` value: {algorithm}. Valid values: '"
            + "', '".join(algorithms)
            + "'."
        )
    positions = np.asarray(strip_unit(positions, "nm")[0])
    if positions.ndim != 2 or positions.shape[1] not in {2, 3}:
        raise ValueError(
            "`positions` must be a two-dimensional array with shape "
            "(N, 2) or (N, 3)."
        )
    n_dimensions = positions.shape[1]
    cutoff = strip_unit(cutoff, "nm")[0]

    # Keep track of additional keyword arguments for specific algorithms
    kwargs = {}

    # Assume non-periodic orthogonal box if no box size is provided
    if box_size is None:
        return globals()[f"_build_neighbor_list_orthogonal_{algorithm}"](
            positions - positions.min(axis=0),  # Shift positions to origin
            cutoff,
            np.fromiter(  # Use maximum distance between particles as box size
                (
                    np.nextafter(
                        positions[:, dim].max() - positions[:, dim].min(),
                        np.inf,
                    )
                    for dim in range(n_dimensions)
                ),
                positions.dtype,
                n_dimensions,
            ),
            False,
            **kwargs,
        )
    else:
        box_size = convert_cell_representation(
            strip_unit(box_size, "nm")[0], "vectors", n_dimensions=n_dimensions
        )

        # Check if the box is orthogonal by testing whether the box
        # vectors form a diagonal matrix
        if check_orthogonality(box_size):
            if pbc and cutoff > np.diag(box_size).min() / 2:
                raise ValueError(
                    "`cutoff` must be less than or equal to half the "
                    "minimum box length when `pbc` is True."
                )

            box_size = convert_cell_representation(box_size, "dimensions")
            if not pbc and not _check_positions(positions, box_size):
                raise ValueError(
                    "`positions` must be within the bounds of the "
                    "simulation box defined by `box_size`."
                )

            return globals()[f"_build_neighbor_list_orthogonal_{algorithm}"](
                positions, cutoff, box_size, pbc, **kwargs
            )

        box_size = reduce_box_vectors(box_size)
        if pbc and cutoff > np.diag(box_size).min() / 2:
            raise ValueError(
                "`cutoff` must be less than or equal to half the "
                "minimum box length when `pbc` is True."
            )

        if algorithm == "cell_list":
            # Compute scaled positions for triclinic box
            kwargs["scaled_positions"] = scaled_positions = positions.copy()
            _scale_coordinates(
                scaled_positions,
                box_size,
                _invert_box_vectors(box_size),
                np.full(n_dimensions, False, np.bool_),
            )

            if not pbc and not _check_positions(
                scaled_positions, np.ones(n_dimensions)
            ):
                raise ValueError(
                    "`positions` must be within the bounds of the "
                    "simulation box defined by `box_size`."
                )

        return globals()[f"_build_neighbor_list_{algorithm}"](
            positions, cutoff=cutoff, box_vectors=box_size, pbc=pbc, **kwargs
        )
