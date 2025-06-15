import pathlib
import sys

from MDAnalysis.lib.distances import capped_distance
import numpy as np

sys.path.insert(
    0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src"
)
from mdcraft import ureg
from mdcraft.algorithm import neighbor

RNG = np.random.default_rng()


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
