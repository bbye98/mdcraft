from __future__ import annotations

from math import floor

from numba import njit
from numba.typed import List
import numpy as np


@njit(fastmath=True)
def build_neighbor_list(
    positions: np.ndarray[np.floating],
    cutoff: np.floating,
    dimensions: np.ndarray[np.floating],
) -> List[set[np.uint32]]:
    """
    Builds a neighbor list for particles in an orthogonal
    simulation box using a cell list algorithm.

    Parameters
    ----------
    positions : `numpy.ndarray`
        Particle positions.

        **Shape**: :math:`(N,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    cutoff : `float`
        Cutoff distance for neighbor search.

        **Reference unit**: :math:`\\mathrm{nm}`.

    dimensions : `numpy.ndarray`
        Dimensions of the orthogonal simulation box.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    Returns
    -------
    neighbor_lists : `list`
        Neighbor lists for each particle, with each list being a
        set of particle indices that are within the cutoff distance
        from the corresponding particle.

        **Shape**: :math:`(N,)`.
    """

    # Split simulation domain into cells
    n_particles = len(positions)
    n_dimensions = len(dimensions)
    n_cells = np.empty(n_dimensions, np.uint32)
    inv_cell_sizes = np.empty(n_dimensions, np.float64)
    for dim in range(n_dimensions):
        n_cells[dim] = floor(dimensions[dim] / cutoff)
        inv_cell_sizes[dim] = n_cells[dim] / dimensions[dim]

    # Get cell indices for each particle and create linked list for
    # each cell
    cell_heads = np.full(
        (n_cells[0], n_cells[1], 1 if n_dimensions == 2 else n_cells[2]),
        -1,
        np.int64,
    )
    cell_linked_lists = np.empty(n_particles, np.int32)
    particle_cell_indices = np.empty((n_particles, n_dimensions), np.uint32)
    for pid in range(n_particles):
        for dim in range(n_dimensions):
            particle_cell_indices[pid, dim] = (
                np.uint32(positions[pid, dim] * inv_cell_sizes[dim]) % n_cells[dim]
            )
        if n_dimensions == 2:
            cix, ciy = particle_cell_indices[pid]
            ciz = 0
        else:
            cix, ciy, ciz = particle_cell_indices[pid]
        cell_linked_lists[pid] = cell_heads[cix, ciy, ciz]
        cell_heads[cix, ciy, ciz] = pid

    # Define offsets for neighboring cells
    if n_dimensions == 2:
        n_offsets = 5
        cell_offsets = np.array(((0, 0), (0, 1), (1, -1), (1, 0), (1, 1)), np.int8)
    else:
        n_offsets = 14
        cell_offsets = np.array(
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

    # Build neighbor list for each particle
    neighbor_lists = List()
    cutoff_squared = cutoff * cutoff
    for pid in range(n_particles):
        neighbor_list = set()
        ix, iy = particle_cell_indices[pid, :2]

        # Check current and forward neighboring cells
        for idx in range(n_offsets):
            jx = (ix + cell_offsets[idx, 0]) % n_cells[0]
            jy = (iy + cell_offsets[idx, 1]) % n_cells[1]
            if n_dimensions == 2:
                nid = cell_heads[jx, jy, 0]
            else:
                nid = cell_heads[
                    jx,
                    jy,
                    (particle_cell_indices[pid, 2] + cell_offsets[idx, 2]) % n_cells[2],
                ]

            # Traverse linked list of particles in current and
            # neighboring cells
            while nid != -1:
                if pid != nid:
                    dr_squared = 0.0
                    for dim in range(n_dimensions):
                        dr = positions[nid, dim] - positions[pid, dim]
                        dr -= dimensions[dim] * round(dr / dimensions[dim])
                        dr_squared += dr * dr
                    if dr_squared < cutoff_squared:
                        if pid < nid:
                            neighbor_list.add(nid)
                        else:
                            neighbor_lists[nid].add(np.uint32(pid))
                nid = cell_linked_lists[nid]

        # Add the neighbor list for the current particle
        neighbor_lists.append(neighbor_list)

    return neighbor_lists
