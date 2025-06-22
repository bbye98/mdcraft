from __future__ import annotations
from typing import TYPE_CHECKING
import warnings

from numba import njit
import numpy as np

from .. import FOUND, Q_
from ..utility.unit import strip_unit

if FOUND["openmm"]:
    from openmm import unit

if TYPE_CHECKING:  # pragma: no cover
    from .. import float_t


def convert_cell_representation(
    representation: np.ndarray[float_t] | "unit.Quantity" | Q_,
    output_format: str,
    n_dimensions: int | None = None,
    /,
) -> np.ndarray[float_t] | "unit.Quantity" | Q_:
    """
    Converts between cell representations for a simulation box.

    .. dropdown:: Two-dimensional (2D) systems

       For a square simulation box, the supported input and output
       formats are

       * its dimensions :math:`(L_x,L_y)`, where :math:`L_x` and
         :math:`L_y` are the lengths along the :math:`x`- and
         :math:`y`-axes, respectively,
       * its lattice parameters :math:`(a,b,\\gamma)`, where :math:`a`
         and :math:`b` are the cell lengths and :math:`\\gamma` is the
         cell angle, and
       * its box vectors :math:`(\\mathbf{a};\\mathbf{b})`.

    .. dropdown:: Three-dimensional (3D) systems
       :open:

       For a cubic simulation box, the supported input and output
       formats are

       * its dimensions :math:`(L_x,L_y,L_z)`, where :math:`L_x`,
         :math:`L_y`, and :math:`L_z` are the lengths along the
         :math:`x`-, :math:`y`-, and :math:`z`-axes, respectively,
       * its lattice parameters :math:`(a,b,c,\\alpha,\\beta,\\gamma)`,
         where :math:`a`, :math:`b`, and :math:`c` are the cell lengths
         and :math:`\\alpha`, :math:`\\beta`, and :math:`\\gamma` are
         the cell angles, and
       * its box vectors :math:`(\\mathbf{a};\\mathbf{b};\\mathbf{c})`.

    Parameters
    ----------
    representation : `numpy.ndarray`, `openmm.unit.Quantity`, or \
    `pint.Quantity`, positional-only
        Dimensions :math:`(L_x,L_y[,L_z])`, lattice parameters
        :math:`(a,b[,c,\\alpha,\\beta],\\gamma)`, or box
        vectors :math:`(\\mathbf{a};\\mathbf{b}[;\\mathbf{c}])`.

        .. note::

           Lattice parameters should always be provided in an array
           without explicit units.

        .. container::

           **Shapes**:

           * 2D: :math:`(2,)` for dimensions, :math:`(3,)` for lattice
             parameters, or :math:`(2,2)` for box vectors.
           * 3D: :math:`(3,)` for dimensions, :math:`(6,)` for lattice
             parameters, or :math:`(3,3)` for box vectors.

        **Reference units**: :math:`\\mathrm{nm}` for lengths and
        degrees (:math:`^\\circ`) for angles.

    output_format : `str`, positional-only
        Desired cell representation.

        .. container::

           **Valid values**:

           * :code:`"dimensions"` for dimensions,
           * :code:`"parameters"` for lattice parameters, or
           * :code:`"vectors"` for box vectors.

    n_dimensions : `int`, positional-only, default: :code:`3`
        Dimensionality of the simulation box.

        **Valid values**: :code:`2` or :code:`3`.

    Returns
    -------
    new_representation : `numpy.ndarray`, `openmm.unit.Quantity`, or \
    `pint.Quantity`
        Cell representation in the desired format.

        .. note::

           Lattice parameters will always be returned in an array
           without explicit units, even if the starting cell
           representation is an OpenMM or Pint quantity.

        .. container::

           **Shapes**:

           * 2D: :math:`(2,)` for dimensions, :math:`(3,)` for lattice
             parameters, or :math:`(2,2)` for box vectors.
           * 3D: :math:`(3,)` for dimensions, :math:`(6,)` for lattice
             parameters, or :math:`(3,3)` for box vectors.

        **Reference units**: :math:`\\mathrm{nm}` for lengths and
        degrees (:math:`^\\circ`) for angles.

    Examples
    --------
    Let us start with a cubic simulation box with dimensions
    :math:`(3,4,5)~\\mathrm{nm}`:

    >>> dimensions = np.array((3.0, 4.0, 5.0))

    We can convert the dimensions to lattice parameters using

    >>> convert_cell_representation(dimensions, "parameters")
    array([3., 4., 5., 90., 90., 90.])

    Alternatively, we can convert the dimensions to box vectors using

    >>> convert_cell_representation(dimensions, "vectors")
    array([[3., 0., 0.],
           [0., 4., 0.],
           [0., 0., 5.]])

    If the dimensions are provided as an OpenMM or Pint quantity, the
    output will have units attached:

    >>> convert_cell_representation(dimensions * unit.nanometer, "vectors")
    Quantity(value=array([[3., 0., 0.],
           [0., 4., 0.],
           [0., 0., 5.]]), unit=nanometer)
    """

    representation, length_unit = strip_unit(representation)
    representation = np.asarray(representation)

    if (shape := representation.shape) == (2,):
        n_dimensions = 2
        input_format = "dimensions"
    elif shape == (2, 2):
        n_dimensions = 2
        input_format = "vectors"
    elif shape == (3,):
        if n_dimensions == 2:
            input_format = "parameters"
        else:
            if n_dimensions is None:
                n_dimensions = 3
                warnings.warn(
                    "When `representation` has shape (3,), it can be either "
                    "2D lattice parameters or 3D dimensions. As "
                    "`n_dimensions` is not specified, it is assumed to be "
                    "the latter."
                )
            input_format = "dimensions"
    elif shape == (3, 3):
        n_dimensions = 3
        input_format = "vectors"
    elif shape == (6,):
        n_dimensions = 3
        input_format = "parameters"
    else:
        raise ValueError(
            f"Invalid shape {shape} for `representation`. "
            "Valid shapes: (2,), (2, 2), (3,), (3, 3), (6,)."
        )

    if n_dimensions == 2:
        if input_format == "dimensions":
            if output_format == "parameters":
                return np.concatenate((representation, (90.0,)))
            elif output_format == "vectors":
                representation = np.diag(representation)
        elif input_format == "parameters":
            gamma = np.radians(representation[2])
            if output_format == "dimensions":
                gamma = np.radians(representation[2])
                return np.array(
                    (representation[0], representation[1] * np.sin(gamma))
                )
            elif output_format == "vectors":
                vectors = np.zeros((2, 2))
                vectors[0, 0] = representation[0]
                vectors[1, 0] = representation[1] * np.cos(gamma)
                vectors[1, 1] = representation[1] * np.sin(gamma)
                vectors[np.isclose(vectors, 0, atol=5e-6)] = 0
                return vectors
            return representation  # output_format == "parameters"
        else:  # input_format == "vectors"
            representation = reduce_box_vectors(representation)
            if output_format == "parameters":
                parameters = np.empty(3, dtype=representation.dtype)
                parameters[:2] = np.linalg.norm(representation, axis=1)
                parameters[2] = np.degrees(
                    np.arccos(
                        np.dot(representation[0], representation[1])
                        / (parameters[0] * parameters[1])
                    )
                )
                return parameters
            elif output_format == "dimensions":
                representation = np.diag(representation)
    else:
        if input_format == "dimensions":
            if output_format == "parameters":
                return np.concatenate((representation, (90.0, 90.0, 90.0)))
            elif output_format == "vectors":
                representation = np.diag(representation)
        elif input_format == "parameters":
            alpha, beta, gamma = np.radians(representation[3:])
            if output_format == "dimensions":
                return np.array(
                    (
                        representation[0],
                        representation[1] * np.sin(gamma),
                        np.sqrt(
                            representation[2] ** 2
                            - (representation[2] * np.cos(beta)) ** 2
                            - (
                                representation[2]
                                * (np.cos(alpha) - np.cos(beta) * np.cos(gamma))
                                / np.sin(gamma)
                            )
                            ** 2
                        ),
                    )
                )
            elif output_format == "vectors":
                vectors = np.zeros((3, 3))
                vectors[0, 0] = representation[0]
                vectors[1, 0] = representation[1] * np.cos(gamma)
                vectors[1, 1] = representation[1] * np.sin(gamma)
                vectors[2, 0] = representation[2] * np.cos(beta)
                vectors[2, 1] = (
                    representation[2]
                    * (np.cos(alpha) - np.cos(beta) * np.cos(gamma))
                    / np.sin(gamma)
                )
                vectors[2, 2] = np.sqrt(
                    representation[2] ** 2
                    - vectors[2, 0] ** 2
                    - vectors[2, 1] ** 2
                )
                vectors[np.isclose(vectors, 0, atol=5e-6)] = 0
                return vectors
            return representation  # output_format == "parameters"
        else:  # input_format == "vectors"
            representation = reduce_box_vectors(representation)
            if output_format == "parameters":
                return np.concatenate(
                    (
                        parameters := np.linalg.norm(representation, axis=1),
                        np.degrees(
                            np.arccos(
                                (
                                    np.dot(representation[1], representation[2])
                                    / (parameters[1] * parameters[2]),
                                    np.dot(representation[0], representation[2])
                                    / (parameters[0] * parameters[2]),
                                    np.dot(representation[0], representation[1])
                                    / (parameters[0] * parameters[1]),
                                )
                            )
                        ),
                    )
                )
            elif output_format == "dimensions":
                representation = np.diag(representation)

    return (
        representation if length_unit is None else representation * length_unit
    )


def reduce_box_vectors(
    box_vectors: np.ndarray[float_t] | "unit.Quantity" | Q_, /
) -> np.ndarray[float_t] | "unit.Quantity" | Q_:
    """
    Reduces the box vectors of a general triclinic simulation box to
    those of a restricted one.

    .. dropdown:: Two-dimensional box vectors

       General parallelogram box vectors have the form

       .. math::

          \\begin{align*}
            \\mathbf{A}&=(A_x,A_y),\\\\
            \\mathbf{B}&=(B_x,B_y),
          \\end{align*}

       whereas restricted box vectors have the form

       .. math::

          \\begin{align*}
            \\mathbf{a}&=(a_x,0),\\\\
            \\mathbf{b}&=(b_x,b_y),
          \\end{align*}

       where :math:`a_x>0`, :math:`b_y>0`, and :math:`a_x\\geq2|b_x|`.

       The conversion is done using

       .. math::

          \\begin{align*}
            a_x&=\\|\\mathbf{A}\\|,\\\\
            b_x&=\\mathbf{B}\\cdot\\hat{\\mathbf{A}},\\\\
            b_y&=\\|\\hat{\\mathbf{A}}\\times\\mathbf{B}\\|,
          \\end{align*}

       where :math:`v=\\|\\mathbf{v}\\|` is the norm of a vector
       :math:`\\mathbf{v}` and
       :math:`\\hat{\\mathbf{v}}\\equiv\\mathbf{v}/\\|\\mathbf{v}\\|` is
       the unit vector in the direction of :math:`\\mathbf{v}`.

    .. dropdown:: Three-dimensional box vectors
       :open:

       General triclinic box vectors have the form

       .. math::

          \\begin{align*}
            \\mathbf{A}&=(A_x,A_y,A_z),\\\\
            \\mathbf{B}&=(B_x,B_y,B_z),\\\\
            \\mathbf{C}&=(C_x,C_y,C_z),
          \\end{align*}

       whereas restricted box vectors have the form

       .. math::

          \\begin{align*}
            \\mathbf{a}&=(a_x,0,0),\\\\
            \\mathbf{b}&=(b_x,b_y,0),\\\\
            \\mathbf{c}&=(c_x,c_y,c_z),
          \\end{align*}

       where :math:`a_x>0`, :math:`b_y>0`, :math:`c_z>0`,
       :math:`a_x\\geq2|b_x|`, :math:`a_x\\geq2|c_x|`, and
       :math:`b_y\\geq2|c_y|`.

       The conversion is done using

       .. math::

          \\begin{align*}
            a_x&=\\|\\mathbf{A}\\|,\\\\
            b_x&=\\mathbf{B}\\cdot\\hat{\\mathbf{A}},\\\\
            b_y&=\\|\\hat{\\mathbf{A}}\\times\\mathbf{B}\\|,\\\\
            c_x&=\\mathbf{C}\\cdot\\hat{\\mathbf{A}},\\\\
            c_y&=\\mathbf{C}\\cdot[
            (\\widehat{\\mathbf{A}\\times\\mathbf{B}})
            \\times\\hat{\\mathbf{A}}],\\\\
            c_z&=\\|\\mathbf{C}\\cdot
            (\\widehat{\\mathbf{A}\\times\\mathbf{B}})\\|,
          \\end{align*}

       where :math:`v=\\|\\mathbf{v}\\|` is the norm of a vector
       :math:`\\mathbf{v}` and
       :math:`\\hat{\\mathbf{v}}\\equiv\\mathbf{v}/\\|\\mathbf{v}\\|` is
       the unit vector in the direction of :math:`\\mathbf{v}`.

    Parameters
    ----------
    box_vectors : `numpy.ndarray`, `openmm.unit.Quantity`, or \
    `pint.Quantity`, positional-only
        Box vectors :math:`(\\mathbf{A};\\mathbf{B}[;\\mathbf{C}])`.

        **Shape**: :math:`(2,2)` or :math:`(3,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    Returns
    -------
    reduced_box_vectors : `numpy.ndarray`, `openmm.unit.Quantity`, or \
    `pint.Quantity`
        Reduced box vectors :math:`(\\mathbf{a};\\mathbf{b}[;\\mathbf{c}])`.

        **Shape**: :math:`(2,2)` or :math:`(3,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    Examples
    --------
    If the provided box vectors are already in their reduced forms,
    they are returned as is:

    >>> box_vectors = np.array(((1, 0, 0), (2, 3, 0), (4, 5, 6)))
    >>> reduce_box_vectors(box_vectors)
    array([[1, 0, 0],
           [2, 3, 0],
           [4, 5, 6]])

    If the box vectors for a general triclinic simulation box are
    provided, they are reduced to those of a restricted one:

    >>> box_vectors = np.array(
    ...     (
    ...         (9 / np.sqrt(11), 3 / np.sqrt(11), 3 / np.sqrt(11)),
    ...         (-4 / np.sqrt(6), 8 / np.sqrt(6), 4 / np.sqrt(6)),
    ...         (5 / np.sqrt(66), 20 / np.sqrt(66), -35 / np.sqrt(66))
    ...     )
    ... )
    >>> reduce_box_vectors(box_vectors)
    array([[3., 0., 0.],
           [0., 4., 0.],
           [0., 0., 5.]])

    With :code:`unit` referring to the `openmm.unit` module and
    :code:`ureg` referring to a `pint.UnitRegistry` instance, this
    function also supports OpenMM and Pint quantities:

    >>> reduce_box_vectors(box_vectors * unit.nanometer)
    Quantity(value=array([[3., 0., 0.],
           [0., 4., 0.],
           [0., 0., 5.]]), unit=nanometer)
    >>> reduce_box_vectors(box_vectors * ureg.nanometer)
    <Quantity([[3., 0., 0.],
     [0., 4., 0.],
     [0., 0., 5.]], 'nanometer')>
    """

    reduced_box_vectors, length_unit = strip_unit(box_vectors)
    reduced_box_vectors = np.asarray(reduced_box_vectors)
    if reduced_box_vectors.shape not in {(2, 2), (3, 3)}:
        raise ValueError(
            f"Invalid shape {reduced_box_vectors.shape} for "
            "`box_vectors`. Valid shapes: (2, 2) and (3, 3)."
        )

    if len(reduced_box_vectors) == 2:
        if (
            np.allclose(reduced_box_vectors, np.tril(reduced_box_vectors))
            and np.all(np.diag(reduced_box_vectors) > 0)
            and reduced_box_vectors[0, 0]
            >= 2 * np.abs(reduced_box_vectors[1, 0])
        ):
            return box_vectors

        reduced_box_vectors = convert_cell_representation(
            (
                (a := np.linalg.norm((A := reduced_box_vectors[0]))),
                (b := np.linalg.norm((B := reduced_box_vectors[1]))),
                np.degrees(np.arccos(np.dot(A, B) / (a * b))),
            ),
            "vectors",
            2,
        )
    elif (
        np.allclose(reduced_box_vectors, np.tril(reduced_box_vectors))
        and np.all(np.diag(reduced_box_vectors) > 0)
        and reduced_box_vectors[0, 0] >= 2 * np.abs(reduced_box_vectors[1, 0])
        and reduced_box_vectors[0, 0] >= 2 * np.abs(reduced_box_vectors[2, 0])
        and reduced_box_vectors[1, 1] >= 2 * np.abs(reduced_box_vectors[2, 1])
    ):
        return box_vectors
    else:
        reduced_box_vectors = convert_cell_representation(
            (
                (a := np.linalg.norm((A := reduced_box_vectors[0]))),
                (b := np.linalg.norm((B := reduced_box_vectors[1]))),
                (c := np.linalg.norm((C := reduced_box_vectors[2]))),
                np.degrees(np.arccos(np.dot(B, C) / (b * c))),
                np.degrees(np.arccos(np.dot(A, C) / (a * c))),
                np.degrees(np.arccos(np.dot(A, B) / (a * b))),
            ),
            "vectors",
        )

    return (
        reduced_box_vectors
        if length_unit is None
        else reduced_box_vectors * length_unit
    )


@njit(fastmath=True, inline="always")  # pragma: no cover
def _invert_box_vectors(
    box_vectors: np.ndarray[float_t],
) -> np.ndarray[float_t]:
    """
    Numba-accelerated function for inverting the box vectors of a
    general parallelogram or triclinic simulation box.

    Parameters
    ----------
    box_vectors : `numpy.ndarray`
        Box vectors :math:`(\\mathbf{A};\\mathbf{B}[;\\mathbf{C}])`.

        **Shape**: :math:`(2,2)` or :math:`(3,3)`.

    Returns
    -------
    inv_box_vectors : `numpy.ndarray`
        Inverted box vectors
        :math:`(\\mathbf{A}^*;\\mathbf{B}^*[;\\mathbf{C}^*])`.

        **Shape**: :math:`(2,2)` or :math:`(3,3)`.
    """

    n_dimensions = len(box_vectors)
    inv_box_vectors = np.empty(
        (n_dimensions, n_dimensions), dtype=box_vectors.dtype
    )
    if n_dimensions == 2:
        determinant = (
            box_vectors[0, 0] * box_vectors[1, 1]
            - box_vectors[0, 1] * box_vectors[1, 0]
        )
        inv_box_vectors[0, 0] = box_vectors[1, 1] / determinant
        inv_box_vectors[0, 1] = -box_vectors[0, 1] / determinant
        inv_box_vectors[1, 0] = -box_vectors[1, 0] / determinant
        inv_box_vectors[1, 1] = box_vectors[0, 0] / determinant
    else:
        bv00, bv01, bv02 = box_vectors[0]
        bv10, bv11, bv12 = box_vectors[1]
        bv20, bv21, bv22 = box_vectors[2]
        inv_box_vectors[0, 0] = bv11 * bv22 - bv12 * bv21
        inv_box_vectors[0, 1] = bv02 * bv21 - bv01 * bv22
        inv_box_vectors[0, 2] = bv01 * bv12 - bv02 * bv11
        inv_box_vectors[1, 0] = bv12 * bv20 - bv10 * bv22
        inv_box_vectors[1, 1] = bv00 * bv22 - bv02 * bv20
        inv_box_vectors[1, 2] = bv02 * bv10 - bv00 * bv12
        inv_box_vectors[2, 0] = bv10 * bv21 - bv11 * bv20
        inv_box_vectors[2, 1] = bv01 * bv20 - bv00 * bv21
        inv_box_vectors[2, 2] = bv00 * bv11 - bv01 * bv10
        determinant = (
            bv00 * inv_box_vectors[0, 0]
            + bv01 * inv_box_vectors[1, 0]
            + bv02 * inv_box_vectors[2, 0]
        )
        for i in range(3):
            for j in range(3):
                inv_box_vectors[i, j] /= determinant
    return inv_box_vectors


@njit(fastmath=True, inline="always")  # pragma: no cover
def _scale_coordinates(
    coordinates: np.ndarray[float_t],
    box_vectors: np.ndarray[float_t],
    inv_box_vectors: np.ndarray[float_t],
    scaled_flags: np.ndarray[np.bool_],
) -> None:
    """
    Numba-accelerated function for scaling the coordinates of entities
    in a general parallelogram or triclinic simulation box to get the
    fractional coordinates.

    Parameters
    ----------
    coordinates : `numpy.ndarray`
        Coordinates of :math:`N` entities.

        .. note::

           This function modifies this NumPy array in-place.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    box_vectors : `numpy.ndarray`
        Box vectors :math:`(\\mathbf{A};\\mathbf{B}[;\\mathbf{C}])`.

        **Shape**: :math:`(2,2)` or :math:`(3,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    inv_box_vectors : `numpy.ndarray`
        Inverted box vectors
        :math:`(\\mathbf{A}^*;\\mathbf{B}^*[;\\mathbf{C}^*])`.

        **Shape**: :math:`(2,2)` or :math:`(3,3)`.

    scaled_flags : `numpy.ndarray`
        Flags indicating whether the coordinates are already scaled
        along the respective axes.

        **Shape**: :math:`(3,)`.
    """

    # All coordinates are already scaled; nothing to do
    if scaled_flags.all():
        return

    # Get number of entities and dimensions
    n_entities, n_dimensions = coordinates.shape

    # All coordinates are unscaled
    if not scaled_flags.any():
        entity_scaled_coordinates = np.empty(
            n_dimensions, dtype=coordinates.dtype
        )
        for eid in range(n_entities):
            for dim in range(n_dimensions):
                entity_scaled_coordinates[dim] = 0.0
                for axis in range(n_dimensions):
                    entity_scaled_coordinates[dim] += (
                        coordinates[eid, axis]
                        * inv_box_vectors[axis, dim]  # TODO
                    )
            for dim in range(n_dimensions):
                coordinates[eid, dim] = entity_scaled_coordinates[dim]
        return

    # Define the indices of the other axes for each axis
    if n_dimensions == 2:
        other_axis_indices = np.array(((1,), (0,)), np.uint8)
    else:
        other_axis_indices = np.array(((1, 2), (0, 2), (0, 1)), np.uint8)

    # Scale coordinates
    for axis_index in range(n_dimensions):
        if not scaled_flags[axis_index]:
            other_indices = other_axis_indices[axis_index]
            other_scaled_flags = scaled_flags[other_indices]
            if other_scaled_flags.all():
                for eid in range(n_entities):
                    for other_index in other_indices:
                        coordinates[eid, axis_index] -= (
                            coordinates[eid, other_index]
                            * box_vectors[other_index, axis_index]
                        )
                    coordinates[eid, axis_index] /= box_vectors[
                        axis_index, axis_index
                    ]
            else:
                if other_scaled_flags[0]:
                    scaled_index = other_indices[0]
                    unscaled_index = other_indices[1]
                else:
                    scaled_index = other_indices[1]
                    unscaled_index = other_indices[0]
                for eid in range(n_entities):
                    coordinates[eid, axis_index] = (
                        box_vectors[unscaled_index, unscaled_index]
                        * coordinates[eid, axis_index]
                        - box_vectors[unscaled_index, axis_index]
                        * coordinates[eid, unscaled_index]
                        + coordinates[eid, scaled_index]
                        * (
                            box_vectors[unscaled_index, axis_index]
                            * box_vectors[scaled_index, unscaled_index]
                            - box_vectors[unscaled_index, unscaled_index]
                            * box_vectors[scaled_index, axis_index]
                        )
                    ) / (
                        box_vectors[axis_index, axis_index]
                        * box_vectors[unscaled_index, unscaled_index]
                        - box_vectors[unscaled_index, axis_index]
                        * box_vectors[axis_index, unscaled_index]
                    )
            scaled_flags[axis_index] = True


def scale_coordinates(
    coordinates: np.ndarray[float_t],
    box_size: np.ndarray[float_t] | "unit.Quantity" | Q_,
    scaled_flags: np.ndarray[bool] | None = None,
) -> None:
    """
    Scales the coordinates of entities in a general parallelogram or
    triclinic simulation box to get the fractional coordinates.

    The relationship between Cartesian coordinates
    :math:`\\mathbf{r}` and the fractional coordinates
    :math:`\\mathbf{f}` can be described by the matrix transformation
    :math:`\\mathbf{r}=\\mathbf{h}\\mathbf{f}`, where

    .. dropdown:: Two-dimensional systems

       :math:`\\mathbf{h}` is the cell tensor (matrix of box vectors
       :math:`\\mathrm{A}` and :math:`\\mathrm{B}`):

       .. math::

          \\begin{pmatrix}r_x\\\\r_y\\end{pmatrix}
          =\\begin{pmatrix}A_x&A_y\\\\B_x&B_y\\end{pmatrix}
          \\begin{pmatrix}f_x\\\\f_y\\end{pmatrix}.

    .. dropdown:: Three-dimensional systems
       :open:

       :math:`\\mathbf{h}` is the cell tensor (matrix of box vectors
       :math:`\\mathrm{A}`, :math:`\\mathrm{B}`, and :math:`\\mathrm{C}`):

       .. math::

          \\begin{pmatrix}r_x\\\\r_y\\\\r_z\\end{pmatrix}
          =\\begin{pmatrix}A_x&A_y&A_z\\\\B_x&B_y&B_z\\\\C_x&C_y&C_z
          \\end{pmatrix}\\begin{pmatrix}f_x\\\\f_y\\\\f_z\\end{pmatrix}.

    As such, the Cartesian coordinates can be scaled to fractional
    coordinates using :math:`\\mathbf{f}=\\mathbf{h}^{-1}\\mathbf{r}`.

    Parameters
    ----------
    coordinates : `numpy.ndarray`
        Coordinates of :math:`N` entities.

        .. note::

           This function modifies this NumPy array in-place.

        **Shape**: :math:`(N,2)` or :math:`(N,3)`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    box_size : `numpy.ndarray`, `openmm.unit.Quantity`, or \
    `pint.Quantity`
        Dimensions :math:`(L_x,L_y[,L_z])`, lattice parameters
        :math:`(a,b[,c,\\alpha,\\beta],\\gamma)`, or box
        vectors :math:`(\\mathbf{a};\\mathbf{b}[;\\mathbf{c}])`.

        .. note::

           Lattice parameters should always be provided in an array
           without explicit units.

        .. container::

           **Shapes**:

           * 2D: :math:`(2,)` for dimensions, :math:`(3,)` for lattice
             parameters, or :math:`(2,2)` for box vectors.
           * 3D: :math:`(3,)` for dimensions, :math:`(6,)` for lattice
             parameters, or :math:`(3,3)` for box vectors.

        **Reference units**: :math:`\\mathrm{nm}` for lengths and
        degrees (:math:`^\\circ`) for angles.

    scaled_flags : array-like, optional
        Flags indicating whether the coordinates are already scaled
        along the respective axes. If not provided, all flags are
        assumed to be :code:`False`.

        **Shape**: :math:`(3,)`.

    Examples
    --------
    Let us start with the coordinates of two entities in a general
    triclinic simulation box:

    >>> coordinates = np.array(
    ...     (
    ...         (2.2613350843332274, 1.6329931618554523, -0.6154574548966636),
    ...         (1.899521470839911, 2.0684580050169066, -1.1488539158071056)
    ...     )
    ... )
    >>> box_vectors = np.array(
    ...     (
    ...         (9 / np.sqrt(11), 3 / np.sqrt(11), 3 / np.sqrt(11)),
    ...         (-4 / np.sqrt(6), 8 / np.sqrt(6), 4 / np.sqrt(6)),
    ...         (5 / np.sqrt(66), 20 / np.sqrt(66), -35 / np.sqrt(66))
    ...     )
    ... )

    We can scale the coordinates to get the fractional coordinates:

    >>> scale_coordinates(coordinates, box_vectors)
    >>> coordinates
    array([[0.5       , 0.5       , 0.5       ],
           [0.33333333, 0.5       , 0.6       ]])

    Now, let us assume that the coordinates are already scaled along
    the :math:`z`-axis. We can scale the coordinates in just the
    :math:`x`- and :math:`y`-axes:

    >>> coordinates = np.array(
    ...     (
    ...         (2.2613350843332274, 1.6329931618554523, 0.5),
    ...         (1.899521470839911, 2.0684580050169066, 0.6)
    ...     )
    ... )
    >>> scale_coordinates(coordinates, box_vectors, [False, False, True])
    >>> coordinates
    array([[0.5       , 0.5       , 0.5       ],
           [0.33333333, 0.5       , 0.6       ])

    Finally, let us assume that the coordinates are already scaled
    along the :math:`x`- and :math:`z`-axes. We can scale the
    coordinates in just the :math:`y`-axis:

    >>> coordinates = np.array(
    ...     (
    ...         (0.5, 1.6329931618554523, 0.5),
    ...         (1 / 3, 2.0684580050169066, 0.6)
    ...     )
    ... )
    >>> scale_coordinates(coordinates, box_vectors, [True, False, True])
    >>> coordinates
    array([[0.5       , 0.5       , 0.5       ],
           [0.33333333, 0.5       , 0.6       ])
    """

    # Check user input
    if not isinstance(coordinates, np.ndarray):
        raise TypeError("`coordinates` must be a NumPy array.")
    if coordinates.ndim != 2 or coordinates.shape[1] not in {2, 3}:
        raise ValueError(
            f"Invalid shape {coordinates.shape} for `coordinates`. "
            "Valid shapes: (N, 2) or (N, 3)."
        )
    n_dimensions = coordinates.shape[1]
    box_size = np.asarray(strip_unit(box_size, "nm")[0])
    if box_size.shape != (n_dimensions, n_dimensions):
        try:
            box_size = convert_cell_representation(
                box_size, "vectors", n_dimensions
            )
        except ValueError:
            raise ValueError(
                "`box_size` must be compatible with `coordinates`."
            )
    if scaled_flags is None:
        scaled_flags = np.array((False, False, False), dtype=np.bool_)
    elif len(scaled_flags) != n_dimensions:
        raise ValueError(
            f"`scaled_flags` must have length {n_dimensions} to be "
            "compatible with `coordinates`."
        )
    else:
        scaled_flags = np.asarray(scaled_flags, dtype=np.bool_)

    # Call Numba function to scale coordinates
    _scale_coordinates(
        coordinates, box_size, _invert_box_vectors(box_size), scaled_flags
    )
