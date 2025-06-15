import pathlib
import sys

import numpy as np
from openmm import unit
import pytest

sys.path.insert(
    0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src"
)
from mdcraft import ureg
from mdcraft.utility.topology import (
    convert_cell_representation,
    reduce_box_vectors,
    scale_coordinates,
)


class TestFunctionReduceBoxVectors:
    @classmethod
    def setup_class(cls):
        cls.box_vectors_3d = np.array(
            (
                (9 / np.sqrt(11), 3 / np.sqrt(11), 3 / np.sqrt(11)),
                (-4 / np.sqrt(6), 8 / np.sqrt(6), 4 / np.sqrt(6)),
                (5 / np.sqrt(66), 20 / np.sqrt(66), -35 / np.sqrt(66)),
            )
        )
        cls.reduced_box_vectors_3d = np.array(
            ((3.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 5.0))
        )
        cls.box_vectors_2d = np.array(
            (
                (9 / np.sqrt(10), 3 / np.sqrt(10)),
                (-4 / np.sqrt(10), 12 / np.sqrt(10)),
            )
        )
        cls.reduced_box_vectors_2d = cls.reduced_box_vectors_3d[:2, :2]

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            reduce_box_vectors(np.empty((2, 3)))

    def test_reduced_2d(self):
        assert np.allclose(
            reduce_box_vectors(self.reduced_box_vectors_2d),
            self.reduced_box_vectors_2d,
        )

    def test_reduced_3d(self):
        assert np.allclose(
            reduce_box_vectors(self.reduced_box_vectors_3d),
            self.reduced_box_vectors_3d,
        )

    def test_general_triclinic_2d(self):
        assert np.allclose(
            reduce_box_vectors(self.box_vectors_2d), self.reduced_box_vectors_2d
        )

    def test_general_triclinic_3d(self):
        assert np.allclose(
            reduce_box_vectors(self.box_vectors_3d), self.reduced_box_vectors_3d
        )

    def test_openmm_quantity_2d(self):
        assert np.allclose(
            reduce_box_vectors(self.box_vectors_2d * unit.nanometer),
            self.reduced_box_vectors_2d * unit.nanometer,
        )

    def test_openmm_quantity_3d(self):
        assert np.allclose(
            reduce_box_vectors(self.box_vectors_3d * unit.nanometer),
            self.reduced_box_vectors_3d * unit.nanometer,
        )

    def test_pint_quantity_2d(self):
        assert np.allclose(
            reduce_box_vectors(self.box_vectors_2d * ureg.nanometer),
            self.reduced_box_vectors_2d * ureg.nanometer,
        )

    def test_pint_quantity_3d(self):
        assert np.allclose(
            reduce_box_vectors(self.box_vectors_3d * ureg.nanometer),
            self.reduced_box_vectors_3d * ureg.nanometer,
        )


class TestFunctionConvertCellRepresentation:
    @classmethod
    def setup_class(cls):
        cls.dimensions_3d = np.array((3.0, 4.0, 5.0))
        cls.dimensions_with_units_3d = cls.dimensions_3d * ureg.nanometer
        cls.parameters_3d = np.array((*cls.dimensions_3d, 90.0, 90.0, 90.0))
        cls.vectors_3d = np.diag(cls.dimensions_3d)
        cls.vectors_with_units_3d = cls.vectors_3d * ureg.nanometer

    def test_invalid_shape(self):
        with pytest.raises(ValueError):
            convert_cell_representation(np.empty(4), "vectors")

    def test_dimensions_to_parameters_3d(self):
        assert np.allclose(
            convert_cell_representation(self.dimensions_3d, "parameters"),
            self.parameters_3d,
        )

    def test_dimensions_to_vectors_3d(self):
        assert np.allclose(
            convert_cell_representation(self.dimensions_3d, "vectors"),
            self.vectors_3d,
        )

    def test_dimensions_with_units_to_vectors_3d(self):
        assert np.allclose(
            convert_cell_representation(
                self.dimensions_with_units_3d, "vectors"
            ),
            self.vectors_with_units_3d,
        )

    def test_parameters_to_dimensions_3d(self):
        assert np.allclose(
            convert_cell_representation(self.parameters_3d, "dimensions"),
            self.dimensions_3d,
        )

    def test_parameters_to_parameters_3d(self):
        assert np.allclose(
            convert_cell_representation(self.parameters_3d, "parameters"),
            self.parameters_3d,
        )

    def test_parameters_to_vectors_3d(self):
        assert np.allclose(
            convert_cell_representation(self.parameters_3d, "vectors"),
            self.vectors_3d,
        )

    def test_vectors_to_dimensions_3d(self):
        assert np.allclose(
            convert_cell_representation(self.vectors_3d, "dimensions"),
            self.dimensions_3d,
        )

    def test_vectors_to_parameters_3d(self):
        assert np.allclose(
            convert_cell_representation(self.vectors_3d, "parameters"),
            self.parameters_3d,
        )

    def test_vectors_with_units_to_dimensions_3d(self):
        assert np.allclose(
            convert_cell_representation(
                self.vectors_with_units_3d, "dimensions"
            ),
            self.dimensions_with_units_3d,
        )


class TestFunctionScaleTriclinicCoordinates:
    @classmethod
    def setup_class(cls):
        cls.fractional_coordinates = np.array(
            ((1 / 2, 1 / 2, 1 / 2), (1 / 3, 1 / 2, 3 / 5))
        )
        cls.box_vectors = np.array(
            (
                (9 / np.sqrt(11), 3 / np.sqrt(11), 3 / np.sqrt(11)),
                (-4 / np.sqrt(6), 8 / np.sqrt(6), 4 / np.sqrt(6)),
                (5 / np.sqrt(66), 20 / np.sqrt(66), -35 / np.sqrt(66)),
            )
        )
        cls.coordinates = cls.fractional_coordinates @ cls.box_vectors.T

    def test_invalid_coordinates_type(self):
        with pytest.raises(TypeError):
            scale_coordinates([0, 0, 0], self.box_vectors)

    def test_invalid_coordinates_shape(self):
        with pytest.raises(ValueError):
            scale_coordinates(np.empty(1), self.box_vectors)

    def test_invalid_box_vectors_shape(self):
        with pytest.raises(ValueError):
            scale_coordinates(self.coordinates, self.box_vectors[:2])

    def test_invalid_scaled_flags_length(self):
        with pytest.raises(ValueError):
            scale_coordinates(
                self.coordinates, self.box_vectors, [False, False]
            )

    def test_unscaled_xyz(self):
        test = self.coordinates.copy()
        scale_coordinates(test, self.box_vectors)
        assert np.allclose(test, self.fractional_coordinates)

    def test_unscaled_xy(self):
        test = self.coordinates.copy()
        test[:, 2] = self.fractional_coordinates[:, 2]
        scale_coordinates(test, self.box_vectors, [False, False, True])
        assert np.allclose(test, self.fractional_coordinates)

    def test_unscaled_y(self):
        test = self.coordinates.copy()
        test[:, [0, 2]] = self.fractional_coordinates[:, [0, 2]]
        scale_coordinates(test, self.box_vectors, [True, False, True])
        assert np.allclose(test, self.fractional_coordinates)
