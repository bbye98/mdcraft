import pathlib
import sys

from MDAnalysis.lib.distances import capped_distance
import numpy as np
import pytest

sys.path.insert(
    0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src"
)
from mdcraft import ureg
from mdcraft.algorithm import neighbor
from mdcraft.utility.topology import (
    convert_cell_representation,
    scale_coordinates,
)

RNG = np.random.default_rng(42)


class TestFunctionBuildNeighborList:
    @classmethod
    def setup_class(cls):
        cls.positions = (
            np.array(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (3.0, 0.0, 0.0),
                )
            )
            * ureg.nm
        )
        cls.cutoff = 15 * ureg.angstrom
        cls.dimensions = np.array([0.004, 0.004, 0.004]) * ureg.um
        cls.positions_nm = cls.positions.m_as(ureg.nm)
        cls.cutoff_nm = cls.cutoff.m_as(ureg.nm)
        cls.dimensions_nm = cls.dimensions.m_as(ureg.nm)

        cls.random_cutoff = 10.0
        cls.random_lattice_parameters = np.array(
            (30.0, 40.0, 50.0, 45.0, 45.0, 45.0)
        )
        cls.random_positions = cls.random_lattice_parameters[:3] * RNG.random(
            (50, 3)
        )

    @staticmethod
    def get_row_differences(nl_i, nl_j):
        return (
            np.setdiff1d(
                nl_i.view([("", nl_i.dtype)] * nl_i.shape[1]),
                nl_j.view([("", nl_j.dtype)] * nl_j.shape[1]),
            )
            .view(nl_i.dtype)
            .reshape(-1, nl_i.shape[1])
        )

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            neighbor.build_neighbor_list(
                positions=np.empty((4,)),
                cutoff=self.cutoff,
            )

    def test_units_orthogonal_nbc_2d(self):
        neighbor_list = neighbor.build_neighbor_list(
            positions=self.positions[:, :2], cutoff=self.cutoff
        )
        assert (
            neighbor_list[0] == {1}
            and len(neighbor_list[1]) == len(neighbor_list[2]) == 0
        )

    def test_dimensionless_orthogonal_nbc_2d(self):
        neighbor_list = neighbor.build_neighbor_list(
            positions=self.positions_nm[:, :2],
            cutoff=self.cutoff_nm,
            box_size=self.dimensions_nm[:2],
            pbc=False,
        )
        assert (
            neighbor_list[0] == {1}
            and len(neighbor_list[1]) == len(neighbor_list[2]) == 0
        )

    def test_units_orthogonal_pbc_2d(self):
        neighbor_list = neighbor.build_neighbor_list(
            positions=self.positions[:, :2],
            cutoff=self.cutoff,
            box_size=self.dimensions[:2],
        )
        assert (
            neighbor_list[0] == {1, 2}
            and len(neighbor_list[1]) == len(neighbor_list[2]) == 0
        )

    def test_dimensionless_orthogonal_pbc_2d(self):
        neighbor_list = neighbor.build_neighbor_list(
            positions=self.positions_nm[:, :2],
            cutoff=self.cutoff_nm,
            box_size=self.dimensions_nm[:2],
        )
        assert (
            neighbor_list[0] == {1, 2}
            and len(neighbor_list[1]) == len(neighbor_list[2]) == 0
        )

    def test_units_orthogonal_nbc_3d(self):
        neighbor_list = neighbor.build_neighbor_list(
            positions=self.positions, cutoff=self.cutoff
        )
        assert (
            neighbor_list[0] == {1}
            and len(neighbor_list[1]) == len(neighbor_list[2]) == 0
        )

    def test_dimensionless_orthogonal_nbc_3d(self):
        neighbor_list = neighbor.build_neighbor_list(
            positions=self.positions_nm,
            cutoff=self.cutoff_nm,
            box_size=self.dimensions_nm,
            pbc=False,
        )
        assert (
            neighbor_list[0] == {1}
            and len(neighbor_list[1]) == len(neighbor_list[2]) == 0
        )

    def test_units_orthogonal_pbc_3d(self):
        neighbor_list = neighbor.build_neighbor_list(
            positions=self.positions,
            cutoff=self.cutoff,
            box_size=self.dimensions,
        )
        assert (
            neighbor_list[0] == {1, 2}
            and len(neighbor_list[1]) == len(neighbor_list[2]) == 0
        )

    def test_dimensionless_orthogonal_pbc_3d(self):
        neighbor_list = neighbor.build_neighbor_list(
            positions=self.positions_nm,
            cutoff=self.cutoff_nm,
            box_size=self.dimensions_nm,
        )
        assert (
            neighbor_list[0] == {1, 2}
            and len(neighbor_list[1]) == len(neighbor_list[2]) == 0
        )

    def test_random_dimensionless_triclinic_pbc_3d(self):
        neighbor_list_mdanalysis = np.unique(
            np.sort(
                capped_distance(
                    self.random_positions,
                    self.random_positions,
                    self.random_cutoff,
                    0,
                    self.random_lattice_parameters,
                )[0],
                axis=1,
            ),
            axis=0,
        )
        neighbor_list_mdcraft = neighbor.build_neighbor_list(
            self.random_positions,
            self.random_cutoff,
            self.random_lattice_parameters,
        )
        neighbor_list_mdcraft = np.array(
            [
                (pid, nid)
                for pid in range(len(neighbor_list_mdcraft))
                for nid in neighbor_list_mdcraft[pid]
            ]
        )
        neighbor_list_mdcraft = neighbor_list_mdcraft[
            np.lexsort(neighbor_list_mdcraft.T[::-1])
        ]

        scaled_positions = self.random_positions.copy()
        scale_coordinates(scaled_positions, self.random_lattice_parameters)
        box_vectors = convert_cell_representation(
            self.random_lattice_parameters, "vectors"
        )
        assert (
            len(
                self.get_row_differences(
                    neighbor_list_mdcraft, neighbor_list_mdanalysis
                )
            )
            == 0
            and (
                np.fromiter(
                    (
                        neighbor._compute_squared_separation_distance_triclinic(
                            scaled_positions[i],
                            scaled_positions[j],
                            box_vectors,
                            True,
                        )
                        for i, j in self.get_row_differences(
                            neighbor_list_mdanalysis, neighbor_list_mdcraft
                        )
                    ),
                    np.float64,
                )
                > self.random_cutoff**2
            ).all()
        )

        """
array([[ 1, 23],
       [ 1, 48],
       [ 3, 32],
       [ 3, 34],
       [ 5, 25],
       [ 5, 31],
       [ 5, 44],
       [ 6, 17],
       [ 6, 49],
       [ 8, 18],
       [11, 14],
       [11, 26],
       [17, 24],
       [18, 29],
       [21, 46],
       [22, 28],
       [25, 33],
       [25, 34],
       [25, 38],
       [26, 29],
       [27, 39],
       [28, 35],
       [31, 34],
       [31, 38],
       [33, 37],
       [46, 49]])
        """
