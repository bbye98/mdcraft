from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING

from numba import njit
from numba.typed import List
import numpy as np

from .. import Q_
from ..utility.topology import (
    _invert_box_vectors,
    _scale_coordinates,
    convert_cell_representation,
)
from ..utility.unit import strip_unit

if TYPE_CHECKING:  # pragma: no cover
    from .. import float_t, int_t


@njit(fastmath=True, inline="always")  # pragma: no cover
def _check_positions(
    positions: np.ndarray[float_t],
    box_lengths: np.ndarray[float_t],
) -> np.bool_:
    for pid in range(positions.shape[0]):
        for dim in range(positions.shape[1]):
            if (
                positions[pid, dim] < 0.0
                or positions[pid, dim] > box_lengths[dim]
            ):
                return False
    return True


@njit(fastmath=True, inline="always")  # pragma: no cover
def _compute_squared_separation_distance_orthogonal(
    position_i: np.ndarray[float_t],
    position_j: np.ndarray[float_t],
    dimensions: np.ndarray[float_t],
    pbc: np.bool_,
) -> float_t:
    dr_squared = 0.0
    for dim in range(position_i.shape[0]):
        dr = position_j[dim] - position_i[dim]
        if pbc:
            dr -= dimensions[dim] * round(dr / dimensions[dim])
        dr_squared += dr * dr
    return dr_squared


@njit(fastmath=True, inline="always")  # pragma: no cover
def _compute_squared_separation_distance_triclinic(
    scaled_position_i: np.ndarray[float_t],
    scaled_position_j: np.ndarray[float_t],
    box_vectors: np.ndarray[float_t],
    pbc: np.bool_,
) -> float_t:
    n_dimensions = scaled_position_i.shape[0]
    dr_vector = np.zeros(n_dimensions, scaled_position_i.dtype)
    for dim in range(n_dimensions):
        scaled_dr = scaled_position_j[dim] - scaled_position_i[dim]
        if pbc:
            scaled_dr -= round(scaled_dr)
        for axis in range(n_dimensions):
            dr_vector[axis] += scaled_dr * box_vectors[dim, axis]
    dr_squared = 0.0
    for dim in range(n_dimensions):
        dr_squared += dr_vector[dim] * dr_vector[dim]
    return dr_squared


@njit(fastmath=True, inline="always")  # pragma: no cover
def _get_forward_cell_offsets(
    n_dimensions: np.uint8,
) -> tuple[np.uint8, np.ndarray[np.int8]]:
    if n_dimensions == 2:
        return np.uint8(5), np.array(
            ((0, 0), (0, 1), (1, -1), (1, 0), (1, 1)), np.int8
        )
    else:
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
    # Split simulation domain into cells
    n_particles, n_dimensions = scaled_positions.shape
    n_cells = np.empty(n_dimensions, np.uint32)
    for dim in range(n_dimensions):
        box_length = 0.0
        for axis in range(n_dimensions):
            box_length += box_vectors[dim, axis] * box_vectors[dim, axis]
        box_length = sqrt(box_length)
        n_cells[dim] = max(np.uint32(box_length / cutoff), 1)

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
            particle_cell_index = np.trunc(
                scaled_positions[pid, dim] * n_cells[dim]
            )
            if pbc:
                particle_cell_index %= n_cells[dim]
            particle_cell_indices[pid, dim] = np.uint32(particle_cell_index)
        if n_dimensions == 2:
            cix, ciy = particle_cell_indices[pid]
            ciz = 0
        else:
            cix, ciy, ciz = particle_cell_indices[pid]
        cell_lists[pid] = cell_heads[cix, ciy, ciz]
        cell_heads[cix, ciy, ciz] = pid

    return n_cells, particle_cell_indices, cell_heads, cell_lists


@njit(fastmath=True)  # pragma: no cover
def _build_neighbor_list_orthogonal(
    positions: np.ndarray[float_t],
    cutoff: float_t,
    dimensions: np.ndarray[float_t],
    pbc: np.bool_,
) -> List[set[np.uint32]]:
    # Build cell lists
    n_dimensions = positions.shape[1]
    n_cells, particle_cell_indices, cell_heads, cell_lists = (
        _build_cell_lists_orthogonal(positions, cutoff, dimensions, pbc)
    )

    # Define offsets for neighboring cells
    n_offsets, cell_offsets = _get_forward_cell_offsets(n_dimensions)

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
                        _compute_squared_separation_distance_orthogonal(
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


@njit(fastmath=True)  # pragma: no cover
def _build_neighbor_list_triclinic(
    positions: np.ndarray[float_t],
    scaled_positions: np.ndarray[float_t],
    cutoff: float_t,
    box_vectors: np.ndarray[float_t],
    pbc: np.bool_,
) -> List[set[np.uint32]]:
    # Build cell lists
    n_dimensions = positions.shape[1]
    n_cells, particle_cell_indices, cell_heads, cell_lists = (
        _build_cell_lists_triclinic(scaled_positions, cutoff, box_vectors, pbc)
    )

    # Define offsets for neighboring cells
    n_offsets, cell_offsets = _get_forward_cell_offsets(n_dimensions)

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
                        _compute_squared_separation_distance_triclinic(
                            scaled_positions[pid],
                            scaled_positions[nid],
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


def build_neighbor_list(
    positions: np.ndarray[float_t] | Q_,
    cutoff: float_t | Q_,
    box_size: np.ndarray[float_t] | Q_ | None = None,
    *,
    pbc: bool = True,
) -> List[set[np.uint32]]:
    """
    Builds a neighbor list for interacting particles using a cell list
    algorithm.

    Parameters
    ----------
    positions : `numpy.ndarray` or `pint.Quantity`
        Particle positions.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    cutoff : `float` or `pint.Quantity`
        Cutoff distance for neighbor search.

        **Reference unit**: :math:`\\mathrm{nm}`.

    box_size : `numpy.ndarray` or `pint.Quantity`, optional
        Size of the simulation box in dimensions :math:`(L_x,L_y[,L_z])`,
        lattice parameters :math:`(a,b[,c,\\alpha,\\beta],\\gamma)`, or
        box vectors :math:`(\\mathbf{a};\\mathbf{b}[;\\mathbf{c}])`. If
        not provided, the simulation box is assumed to be orthogonal and
        non-periodic.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    pbc : `bool`, keyword-only, default: `True`
        Specifies whether to apply periodic boundary conditions (PBC)
        and use the minimum image convention when calculating
        separation distances between particles.

    Returns
    -------
    neighbor_lists : `list`
        Neighbor lists for each particle, with each list being a
        set of particle indices that are within the cutoff distance
        from the corresponding particle.

        **Shape**: :math:`(N,)`.
    """

    positions = np.asarray(strip_unit(positions, "nm")[0])
    if positions.ndim != 2 or positions.shape[1] not in {2, 3}:
        raise ValueError(
            "`positions` must be a two-dimensional array with shape "
            "(N, 2) or (N, 3)."
        )
    positions -= positions.min(axis=0)

    n_dimensions = positions.shape[1]
    cutoff = strip_unit(cutoff, "nm")[0]
    if box_size is None:
        dtype = positions.dtype
        return _build_neighbor_list_orthogonal(
            positions,
            cutoff,
            np.fromiter(
                (
                    positions[:, dim].max()
                    - positions[:, dim].min()
                    + 2 * np.finfo(dtype).eps
                    for dim in range(n_dimensions)
                ),
                dtype,
                n_dimensions,
            ),
            False,
        )
    else:
        box_size = convert_cell_representation(
            strip_unit(box_size, "nm")[0], "vectors", n_dimensions
        )
        if np.array_equal(box_size, np.diag(np.diag(box_size))):
            box_size = convert_cell_representation(box_size, "dimensions")
            if not pbc and not _check_positions(positions, box_size):
                raise ValueError(
                    "`positions` must be within the bounds of the "
                    "simulation box defined by `box_size`."
                )
            return _build_neighbor_list_orthogonal(
                positions, cutoff, box_size, pbc
            )

        scaled_positions = positions.copy()
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
        return _build_neighbor_list_triclinic(
            positions, scaled_positions, cutoff, box_size, pbc
        )
