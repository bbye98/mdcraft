import pathlib
import sys

import numpy as np
from openmm import unit
import pytest

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft import ureg  # noqa: E402
from mdcraft.utility.topology import (
    reduce_box_vectors,
    convert_cell_representation,
    scale_triclinic_coordinates,
)  # noqa: E402


def test_func_reduce_box_vectors():
    # Box vectors array with invalid shape
    with pytest.raises(ValueError):
        reduce_box_vectors(np.zeros((2, 3)))

    # Reduced box vectors
    reduced_box_vectors = np.array(((3.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 5.0)))
    assert np.allclose(reduce_box_vectors(reduced_box_vectors), reduced_box_vectors)

    # Box vectors for general triclinic simulation box
    box_vectors = np.array(
        (
            (9 / np.sqrt(11), 3 / np.sqrt(11), 3 / np.sqrt(11)),
            (-4 / np.sqrt(6), 8 / np.sqrt(6), 4 / np.sqrt(6)),
            (5 / np.sqrt(66), 20 / np.sqrt(66), -35 / np.sqrt(66)),
        )
    )
    assert np.allclose(reduce_box_vectors(box_vectors), reduced_box_vectors)

    # OpenMM quantities
    assert np.allclose(
        reduce_box_vectors(box_vectors * unit.nanometer),
        reduced_box_vectors * unit.nanometer,
    )

    # Pint quantities
    assert np.allclose(
        reduce_box_vectors(box_vectors * ureg.nanometer),
        reduced_box_vectors * ureg.nanometer,
    )


def test_func_convert_cell_representation():
    # References
    dimensions = np.array((3.0, 4.0, 5.0))
    parameters = np.array((3.0, 4.0, 5.0, 90.0, 90.0, 90.0))
    vectors = np.array(((3.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 5.0)))

    # Input array with invalid shape
    with pytest.raises(ValueError):
        convert_cell_representation(np.zeros(4), "vectors")

    # Dimensions
    assert np.allclose(
        convert_cell_representation(dimensions, "parameters"), parameters
    )
    assert np.allclose(convert_cell_representation(dimensions, "vectors"), vectors)

    # Dimensions with units
    assert np.allclose(
        convert_cell_representation(dimensions * ureg.nanometer, "vectors"),
        vectors * ureg.nanometer,
    )

    # Lattice parameters
    assert np.allclose(
        convert_cell_representation(parameters, "dimensions"), dimensions
    )
    assert np.allclose(
        convert_cell_representation(parameters, "parameters"), parameters
    )
    assert np.allclose(convert_cell_representation(parameters, "vectors"), vectors)

    # Box vectors
    assert np.allclose(convert_cell_representation(vectors, "dimensions"), dimensions)
    assert np.allclose(convert_cell_representation(vectors, "parameters"), parameters)

    # Box vectors with units
    assert np.allclose(
        convert_cell_representation(vectors * ureg.nanometer, "dimensions"),
        dimensions * ureg.nanometer,
    )


def test_func_scale_triclinic_coordinates():
    # References
    fractional_coordinates = np.array(((1 / 2, 1 / 2, 1 / 2), (1 / 3, 1 / 2, 3 / 5)))
    box_vectors = np.array(
        (
            (9 / np.sqrt(11), 3 / np.sqrt(11), 3 / np.sqrt(11)),
            (-4 / np.sqrt(6), 8 / np.sqrt(6), 4 / np.sqrt(6)),
            (5 / np.sqrt(66), 20 / np.sqrt(66), -35 / np.sqrt(66)),
        )
    )
    coordinates = fractional_coordinates @ box_vectors.T

    # Coordinates not stored in NumPy array
    with pytest.raises(TypeError):
        scale_triclinic_coordinates([0, 0, 0], box_vectors)

    # Coordinates array with invalid shape
    with pytest.raises(ValueError):
        scale_triclinic_coordinates(np.zeros(1), box_vectors)

    # Box vectors array with invalid shape
    with pytest.raises(ValueError):
        scale_triclinic_coordinates(coordinates, box_vectors[:2])

    # Wrong number of scaled flags
    with pytest.raises(ValueError):
        scale_triclinic_coordinates(coordinates, box_vectors, [False, False])

    # All axes are unscaled
    test = coordinates.copy()
    scale_triclinic_coordinates(test, box_vectors)
    assert np.allclose(test, fractional_coordinates)

    # x- and y-axes are unscaled
    test = coordinates.copy()
    test[:, 2] = fractional_coordinates[:, 2]
    scale_triclinic_coordinates(test, box_vectors, [False, False, True])
    assert np.allclose(test, fractional_coordinates)

    # y-axis is unscaled
    test = coordinates.copy()
    test[:, [0, 2]] = fractional_coordinates[:, [0, 2]]
    scale_triclinic_coordinates(test, box_vectors, [True, False, True])
    assert np.allclose(test, fractional_coordinates)
