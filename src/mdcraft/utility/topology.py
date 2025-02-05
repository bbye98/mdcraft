from typing import Union

import numpy as np

from .. import FOUND_OPENMM, Q_
from ..utility.unit import strip_unit

if FOUND_OPENMM:
    from openmm import unit


def reduce_box_vectors(
    vectors: Union[np.ndarray[float], "unit.Quantity", Q_], /
) -> Union[np.ndarray[float], "unit.Quantity", Q_]:
    """
    Reduces the box vectors of a general triclinic simulation box to
    those of a restricted one.

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
         c_y&=\\mathbf{C}\\cdot[(\\widehat{\\mathbf{A}\\times\\mathbf{B}})\\times\\hat{\\mathbf{A}}],\\\\
         c_z&=\\|\\mathbf{C}\\cdot(\\widehat{\\mathbf{A}\\times\\mathbf{B}})\\|,
       \\end{align*}

    where :math:`v=\\|\\mathbf{v}\\|` is the norm of a vector
    :math:`\\mathbf{v}` and :math:`\\hat{\\mathbf{v}}\\equiv\\mathbf{v}/\\|\\mathbf{v}\\|`
    is the unit vector in the direction of :math:`\\mathbf{v}`.

    Parameters
    ----------
    vectors : `numpy.ndarray`, `openmm.unit.Quantity`, or \
    `pint.Quantity`, positional-only
        Box vectors :math:`(\\mathbf{a};\\mathbf{b};\\mathbf{c})`.

        **Shape**: :math:`(3,3)`.

        **Reference units**: :math:`\\mathrm{nm}`.

    Returns
    -------
    reduced_vectors : `numpy.ndarray`, `openmm.unit.Quantity`, or \
    `pint.Quantity`
        Reduced box vectors
        :math:`(\\mathbf{a};\\mathbf{b};\\mathbf{c})`.

        **Shape**: :math:`(3,3)`.

        **Reference units**: :math:`\\mathrm{nm}`.

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

    vectors, length_unit = strip_unit(vectors)
    vectors = np.asarray(vectors)
    if vectors.shape != (3, 3):
        raise ValueError(
            f"Invalid shape {vectors.shape} for `vectors`. Valid shape: (3, 3)."
        )

    if (
        not np.allclose(vectors, np.tril(vectors))
        or not np.all(np.diag(vectors) > 0)
        or vectors[0, 0] < 2 * np.abs(vectors[1, 0])
        or vectors[0, 0] < 2 * np.abs(vectors[2, 0])
        or vectors[1, 1] < 2 * np.abs(vectors[2, 1])
    ):
        vectors = convert_cell_representation(
            (
                (a := np.linalg.norm((A := vectors[0]))),
                (b := np.linalg.norm((B := vectors[1]))),
                (c := np.linalg.norm((C := vectors[2]))),
                np.degrees(np.arccos(np.dot(B, C) / (b * c))),
                np.degrees(np.arccos(np.dot(A, C) / (a * c))),
                np.degrees(np.arccos(np.dot(A, B) / (a * b))),
            ),
            "vectors",
        )

    return vectors if length_unit is None else vectors * length_unit


def convert_cell_representation(
    representation: Union[np.ndarray[float], "unit.Quantity", Q_], output_format: str, /
) -> Union[np.ndarray[float], "unit.Quantity", Q_]:
    """
    Converts between cell representations for a simulation box.

    The supported input and output formats are

    * the dimensions :math:`(L_x,L_y,L_z)` of a cubic simulation box,
      where :math:`L_x`, :math:`L_y`, and :math:`L_z` are the lengths
      along the :math:`x`, :math:`y`, and :math:`z` axes, respectively,
    * the lattice parameters :math:`(a,b,c,\\alpha,\\beta,\\gamma)` for
      a triclinic simulation box, where :math:`a`, :math:`b`, and
      :math:`c` are the cell lengths and :math:`\\alpha`,
      :math:`\\beta`, and :math:`\\gamma` are the cell angles, and
    * the box vectors :math:`(\\mathbf{a};\\mathbf{b};\\mathbf{c})` for
      a triclinic simulation box.

    Parameters
    ----------
    representation : `numpy.ndarray`, `openmm.unit.Quantity`, or \
    `pint.Quantity`, positional-only
        Dimensions :math:`(L_x,L_y,L_z)`, lattice parameters
        :math:`(a,b,c,\\alpha,\\beta,\\gamma)`, or box
        vectors :math:`(\\mathbf{a};\\mathbf{b};\\mathbf{c})`.

        .. note::

           Lattice parameters should always be provided in an array
           without explicit units.

        .. container::

           **Shapes**:

           * :math:`(3,)` for dimensions,
           * :math:`(6,)` for lattice parameters, or
           * :math:`(3,3)` for box vectors.

        **Reference units**: :math:`\\mathrm{nm}` for lengths and
        degrees (:math:`^\\circ`) for angles.

    output_format : `str`, positional-only
        Desired cell representation.

        .. container::

           **Valid values**:

           * :code:`"dimensions"` for dimensions,
           * :code:`"parameters"` for lattice parameters, or
           * :code:`"vectors"` for box vectors.

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

           * :math:`(3,)` for dimensions,
           * :math:`(6,)` for lattice parameters, or
           * :math:`(3,3)` for box vectors.

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
    match representation.shape:
        case (3,):
            input_format = "dimensions"
        case (6,):
            input_format = "parameters"
        case (3, 3):
            input_format = "vectors"
        case _:
            raise ValueError(
                f"Invalid shape {representation.shape} for `representation`. "
                "Valid shapes: (3,), (6,), (3, 3)."
            )

    if input_format == "dimensions":
        if output_format == "parameters":
            return np.concatenate((representation, (90.0, 90.0, 90.0)))
        elif output_format == "vectors":
            representation = np.diag(representation)
        return representation if length_unit is None else representation * length_unit
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
                representation[2] ** 2 - vectors[2, 0] ** 2 - vectors[2, 1] ** 2
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
        return representation if length_unit is None else representation * length_unit


def scale_triclinic_coordinates(
    coordinates: np.ndarray[float],
    box_vectors: np.ndarray[float],
    scaled_flags: list[bool] | None = None,
) -> None:
    """
    Scales the coordinates of entities in a general triclinic
    simulation box with origin :math:`(0,0,0)` to get the fractional
    coordinates.

    The relationship between Cartesian coordinates :math:`\\mathbf{r}`
    and the fractional coordinates :math:`\\mathbf{f}` can be described
    by the matrix transformation
    :math:`\\mathbf{r}=\\mathbf{h}\\mathbf{f}`, where
    :math:`\\mathbf{h}` is the cell tensor (matrix of box vectors
    :math:`\\mathrm{A}`, :math:`\\mathrm{B}`, and :math:`\\mathrm{C}`):

    .. math::

       \\begin{pmatrix}r_x\\\\r_y\\\\r_z\\end{pmatrix}
       =\\begin{pmatrix}A_x&A_y&A_z\\\\B_x&B_y&B_z\\\\C_x&C_y&C_z\\end{pmatrix}
       \\begin{pmatrix}f_x\\\\f_y\\\\f_z\\end{pmatrix}.

    As such, the Cartesian coordinates can be scaled to fractional
    coordinates using :math:`\\mathbf{f}=\\mathbf{h}^{-1}\\mathbf{r}`.

    .. note::

       This function modifies the input NumPy array in-place.

    Parameters
    ----------
    coordinates : `numpy.ndarray`
        Coordinates of :math:`N` entities.

        **Shape**: :math:`(N,3)`.

        **Reference units**: :math:`\\mathrm{nm}`.

    box_vectors : `numpy.ndarray`
        Box vectors of the general triclinic simulation box.

        **Shape**: :math:`(3,3)`.

        **Reference units**: :math:`\\mathrm{nm}`.

    scaled_flags : `list`, optional
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

    >>> scale_triclinic_coordinates(coordinates, box_vectors)
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
    >>> scale_triclinic_coordinates(coordinates, box_vectors, [False, False, True])
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
    >>> scale_triclinic_coordinates(coordinates, box_vectors, [True, False, True])
    >>> coordinates
    array([[0.5       , 0.5       , 0.5       ],
           [0.33333333, 0.5       , 0.6       ])
    """

    if not isinstance(coordinates, np.ndarray):
        raise TypeError("`coordinates` must be a NumPy array.")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            f"Invalid shape {coordinates.shape} for `coordinates`. "
            "Valid shape: (N, 3)."
        )
    if box_vectors.shape != (3, 3):
        raise ValueError(
            f"Invalid shape {box_vectors.shape} for `box_vectors`. "
            "Valid shape: (3, 3)."
        )
    if scaled_flags is None:
        scaled_flags = [False, False, False]
    elif len(scaled_flags) != 3:
        raise ValueError("`scaled_flags` must be an array with length 3.")

    if not any(scaled_flags):
        coordinates @= np.linalg.inv(box_vectors).T
        return
    elif not all(scaled_flags):
        for axis_index, scaled_flag in enumerate(scaled_flags):
            if not scaled_flag:
                other_indices = [0, 1, 2]
                other_scaled_flags = scaled_flags.copy()
                del other_indices[axis_index], other_scaled_flags[axis_index]
                if all(other_scaled_flags):
                    for other_index in other_indices:
                        coordinates[:, axis_index] -= (
                            coordinates[:, other_index]
                            * box_vectors[axis_index, other_index]
                        )
                    coordinates[:, axis_index] /= box_vectors[axis_index, axis_index]
                else:
                    scaled_index, unscaled_index = other_indices[
                        :: (2 * other_scaled_flags[0] - 1)
                    ]
                    coordinates[:, axis_index] = (
                        box_vectors[unscaled_index, unscaled_index]
                        * coordinates[:, axis_index]
                        - box_vectors[axis_index, unscaled_index]
                        * coordinates[:, unscaled_index]
                        + coordinates[:, scaled_index]
                        * (
                            box_vectors[axis_index, unscaled_index]
                            * box_vectors[unscaled_index, scaled_index]
                            - box_vectors[unscaled_index, unscaled_index]
                            * box_vectors[axis_index, scaled_index]
                        )
                    ) / (
                        box_vectors[axis_index, axis_index]
                        * box_vectors[unscaled_index, unscaled_index]
                        - box_vectors[axis_index, unscaled_index]
                        * box_vectors[unscaled_index, axis_index]
                    )
                scaled_flags[axis_index] = True
