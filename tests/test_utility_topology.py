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
    reduce_box_vectors,
    convert_cell_representation,
    scale_triclinic_coordinates,
)


class TestFunctionReduceBoxVectors:
    @classmethod
    def setup_class(cls):
        cls.box_vectors = np.array(
            (
                (9 / np.sqrt(11), 3 / np.sqrt(11), 3 / np.sqrt(11)),
                (-4 / np.sqrt(6), 8 / np.sqrt(6), 4 / np.sqrt(6)),
                (5 / np.sqrt(66), 20 / np.sqrt(66), -35 / np.sqrt(66)),
            )
        )
        cls.reduced_box_vectors = np.array(
            ((3.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 5.0))
        )

    def test_invalid_box_vectors_shape(self):
        with pytest.raises(ValueError):
            reduce_box_vectors(np.empty((2, 3)))

    def test_reduced_box_vectors(self):
        assert np.allclose(
            reduce_box_vectors(self.reduced_box_vectors),
            self.reduced_box_vectors,
        )

    def test_general_triclinic_simulation_box(self):
        assert np.allclose(
            reduce_box_vectors(self.box_vectors), self.reduced_box_vectors
        )

    def test_openmm_quantities(self):
        assert np.allclose(
            reduce_box_vectors(self.box_vectors * unit.nanometer),
            self.reduced_box_vectors * unit.nanometer,
        )

    def test_pint_quantities(self):
        assert np.allclose(
            reduce_box_vectors(self.box_vectors * ureg.nanometer),
            self.reduced_box_vectors * ureg.nanometer,
        )


class TestFunctionConvertCellRepresentation:
    @classmethod
    def setup_class(cls):
        cls.dimensions = np.array((3.0, 4.0, 5.0))
        cls.dimensions_with_units = cls.dimensions * ureg.nanometer
        cls.parameters = np.array((*cls.dimensions, 90.0, 90.0, 90.0))
        cls.vectors = np.diag(cls.dimensions)
        cls.vectors_with_units = cls.vectors * ureg.nanometer

    def test_invalid_input_shape(self):
        with pytest.raises(ValueError):
            convert_cell_representation(np.empty(4), "vectors")

    def test_dimensions_to_parameters(self):
        assert np.allclose(
            convert_cell_representation(self.dimensions, "parameters"),
            self.parameters,
        )

    def test_dimensions_to_vectors(self):
        assert np.allclose(
            convert_cell_representation(self.dimensions, "vectors"),
            self.vectors,
        )

    def test_dimensions_with_units_to_vectors(self):
        assert np.allclose(
            convert_cell_representation(self.dimensions_with_units, "vectors"),
            self.vectors_with_units,
        )

    def test_parameters_to_dimensions(self):
        assert np.allclose(
            convert_cell_representation(self.parameters, "dimensions"),
            self.dimensions,
        )

    def test_parameters_to_parameters(self):
        assert np.allclose(
            convert_cell_representation(self.parameters, "parameters"),
            self.parameters,
        )

    def test_parameters_to_vectors(self):
        assert np.allclose(
            convert_cell_representation(self.parameters, "vectors"),
            self.vectors,
        )

    def test_vectors_to_dimensions(self):
        assert np.allclose(
            convert_cell_representation(self.vectors, "dimensions"),
            self.dimensions,
        )

    def test_vectors_to_parameters(self):
        assert np.allclose(
            convert_cell_representation(self.vectors, "parameters"),
            self.parameters,
        )

    def test_vectors_with_units_to_dimensions(self):
        assert np.allclose(
            convert_cell_representation(self.vectors_with_units, "dimensions"),
            self.dimensions_with_units,
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
            scale_triclinic_coordinates([0, 0, 0], self.box_vectors)

    def test_invalid_coordinates_shape(self):
        with pytest.raises(ValueError):
            scale_triclinic_coordinates(np.empty(1), self.box_vectors)

    def test_invalid_box_vectors_shape(self):
        with pytest.raises(ValueError):
            scale_triclinic_coordinates(self.coordinates, self.box_vectors[:2])

    def test_invalid_scaled_flags_length(self):
        with pytest.raises(ValueError):
            scale_triclinic_coordinates(
                self.coordinates, self.box_vectors, [False, False]
            )

    def test_unscaled_xyz(self):
        test = self.coordinates.copy()
        scale_triclinic_coordinates(test, self.box_vectors)
        assert np.allclose(test, self.fractional_coordinates)

    def test_unscaled_xy(self):
        test = self.coordinates.copy()
        test[:, 2] = self.fractional_coordinates[:, 2]
        scale_triclinic_coordinates(
            test, self.box_vectors, [False, False, True]
        )
        assert np.allclose(test, self.fractional_coordinates)

    def test_unscaled_y(self):
        test = self.coordinates.copy()
        test[:, [0, 2]] = self.fractional_coordinates[:, [0, 2]]
        scale_triclinic_coordinates(test, self.box_vectors, [True, False, True])
        assert np.allclose(test, self.fractional_coordinates)
