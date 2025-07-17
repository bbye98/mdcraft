import pathlib
import sys

from MDAnalysis.lib.distances import capped_distance
import numpy as np
import pytest

sys.path.insert(
    0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src"
)
from mdcraft import ureg
from mdcraft.lib.neighbor import build_neighbor_list

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
        cls.cutoff = 15.0 * ureg.angstrom
        cls.dimensions = np.array([0.004, 0.004, 0.004]) * ureg.um
        cls.positions_nm = cls.positions.m_as(ureg.nm)
        cls.cutoff_nm = cls.cutoff.m_as(ureg.nm)
        cls.dimensions_nm = cls.dimensions.m_as(ureg.nm)

        cls.random_cutoff = 10.0
        cls.random_lattice_parameters = np.array(
            (30.0, 40.0, 50.0, 45.0, 45.0, 45.0)
        )
        cls.random_positions = cls.random_lattice_parameters[:3] * RNG.random(
            (900, 3)
        )

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            build_neighbor_list(
                positions=np.empty((4,)),
                cutoff=self.cutoff,
            )

    def test_units_orthogonal_nbc_2d(self):
        neighbor_list = build_neighbor_list(
            positions=self.positions[:, :2], cutoff=self.cutoff
        )
        assert (
            neighbor_list[0] == {1}
            and len(neighbor_list[1]) == len(neighbor_list[2]) == 0
        )

    def test_dimensionless_orthogonal_nbc_2d(self):
        neighbor_list = build_neighbor_list(
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
        neighbor_list = build_neighbor_list(
            positions=self.positions[:, :2],
            cutoff=self.cutoff,
            box_size=self.dimensions[:2],
        )
        assert (
            neighbor_list[0] == {1, 2}
            and len(neighbor_list[1]) == len(neighbor_list[2]) == 0
        )

    def test_dimensionless_orthogonal_pbc_2d(self):
        neighbor_list = build_neighbor_list(
            positions=self.positions_nm[:, :2],
            cutoff=self.cutoff_nm,
            box_size=self.dimensions_nm[:2],
        )
        assert (
            neighbor_list[0] == {1, 2}
            and len(neighbor_list[1]) == len(neighbor_list[2]) == 0
        )

    def test_units_orthogonal_nbc_3d(self):
        neighbor_list = build_neighbor_list(
            positions=self.positions, cutoff=self.cutoff
        )
        assert (
            neighbor_list[0] == {1}
            and len(neighbor_list[1]) == len(neighbor_list[2]) == 0
        )

    def test_dimensionless_orthogonal_nbc_3d(self):
        neighbor_list = build_neighbor_list(
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
        neighbor_list = build_neighbor_list(
            positions=self.positions,
            cutoff=self.cutoff,
            box_size=self.dimensions,
        )
        assert (
            neighbor_list[0] == {1, 2}
            and len(neighbor_list[1]) == len(neighbor_list[2]) == 0
        )

    def test_dimensionless_orthogonal_pbc_3d(self):
        neighbor_list = build_neighbor_list(
            positions=self.positions_nm,
            cutoff=self.cutoff_nm,
            box_size=self.dimensions_nm,
        )
        assert (
            neighbor_list[0] == {1, 2}
            and len(neighbor_list[1]) == len(neighbor_list[2]) == 0
        )

    def test_random_dimensionless_orthogonal_nbc_3d(self):
        neighbor_list_mdanalysis = np.unique(
            np.sort(
                capped_distance(
                    self.random_positions,
                    self.random_positions,
                    self.random_cutoff,
                    0.0,
                )[0],
                axis=1,
            ),
            axis=0,
        )
        neighbor_list_mdcraft = build_neighbor_list(
            self.random_positions, self.random_cutoff
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
        assert np.array_equal(neighbor_list_mdcraft, neighbor_list_mdanalysis)

    def test_random_dimensionless_orthogonal_pbc_3d(self):
        neighbor_list_mdanalysis = np.unique(
            np.sort(
                capped_distance(
                    self.random_positions,
                    self.random_positions,
                    self.random_cutoff,
                    0.0,
                    np.concatenate(
                        (self.random_lattice_parameters[:3], (90.0, 90.0, 90.0))
                    ),
                )[0],
                axis=1,
            ),
            axis=0,
        )
        neighbor_list_mdcraft = build_neighbor_list(
            self.random_positions,
            self.random_cutoff,
            self.random_lattice_parameters[:3],
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
        assert np.array_equal(neighbor_list_mdcraft, neighbor_list_mdanalysis)

    def test_random_dimensionless_triclinic_nbc_3d(self): ...

    def test_random_dimensionless_triclinic_pbc_3d(self):
        neighbor_list_mdanalysis = np.unique(
            np.sort(
                capped_distance(
                    self.random_positions,
                    self.random_positions,
                    self.random_cutoff,
                    0.0,
                    self.random_lattice_parameters,
                )[0],
                axis=1,
            ),
            axis=0,
        )
        neighbor_list_mdcraft = build_neighbor_list(
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
        assert np.array_equal(neighbor_list_mdcraft, neighbor_list_mdanalysis)
