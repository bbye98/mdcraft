from math import floor

from numba import njit
from numba.typed import List
import numpy as np

from .. import Q_
from ..utility.unit import strip_unit


class State:
    def __init__(
        self,
        *,
        dimensions: np.ndarray[np.float64] | Q_,
        n_particles: int | None = None,
        positions: np.ndarray[np.float64] | Q_ | None = None,
        velocities: np.ndarray[np.float64] | Q_ | None = None,
        forces: np.ndarray[np.float64] | Q_ | None = None,
        cutoff: float | Q_ | None = None,
    ) -> None:
        self._dimensions_nm = np.asarray(strip_unit(dimensions, "nm")[0], np.float64)
        if not 2 <= len(self._dimensions_nm) <= 3:
            raise ValueError(
                "`dimensions` must be a one-dimensional array with length 2 or 3."
            )

        rng = np.random.default_rng()
        if n_particles is None:
            if positions is None:
                raise ValueError(
                    "Either `n_particles` or `positions` must be provided."
                )
            self._n_particles = positions.shape[0]
            self._positions_nm = np.asarray(strip_unit(positions, "nm")[0], np.float64)
        else:
            self._n_particles = n_particles
            self._positions_nm = (
                rng.random((self._n_particles, 3), np.float64) * self._dimensions_nm
            )
        self._velocities_nm_per_ps = (
            np.zeros_like(self._positions_nm)
            if velocities is None
            else np.asarray(strip_unit(velocities, "nm/ps")[0], np.float64)
        )
        self._forces_kJ_per_mol_nm = (
            np.zeros_like(self._positions_nm)
            if forces is None
            else np.asarray(strip_unit(forces, "kJ/(mol*nm)")[0], np.float64)
        )

        self._cutoff_nm = strip_unit(cutoff, "nm")[0]
        if self._cutoff_nm is None:
            self._neighbor_list = np.empty((0, 0), np.uint32)
            self._neighbor_counts = np.empty(0, np.uint32)
        else:
            self._build_neighbor_list(
                self._n_particles,
                self._dimensions_nm,
                self._positions_nm,
                self._cutoff_nm,
            )

    @staticmethod
    @njit(fastmath=True)
    def build_neighbor_list(
        dimensions: np.ndarray[np.float64],
        positions: np.ndarray[np.float64],
        cutoff: np.float64,
    ) -> None:
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
                        (particle_cell_indices[pid, 2] + cell_offsets[idx, 2])
                        % n_cells[2],
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


class System:
    def __init__(self) -> None:
        pass


class Simulation:
    def __init__(self) -> None:
        pass
