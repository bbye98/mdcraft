"""
Molecular structure
===================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains algorithms for computing structural information
for a collection of particles.
"""

from typing import Union

import MDAnalysis as mda
import numpy as np

def center_of_mass(
        group: mda.AtomGroup = None, grouping: str = None, *,
        masses: Union[np.ndarray[float], list[np.ndarray[float]]] = None,
        positions: Union[np.ndarray[float], list[np.ndarray[float]]] = None,
        images: Union[np.ndarray[int], list[np.ndarray[int]]] = None,
        dimensions: np.ndarray[float] = None, n_groups: int = None,
        raw: bool = False
    ) -> Union[np.ndarray[float],
               tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]]:

    r"""
    Computes the centers of mass :math:`\mathbf{R}_\mathrm{com}` for a
    collection of atoms.

    For a group of :math:`N` atoms with masses :math:`m_i` and positions
    :math:`\mathbf{r}_i`, the center of mass is defined as

    .. math::

       \mathbf{R}_\mathrm{com}=\frac{\sum_{i=1}^Nm_i
       \mathbf{r}_i}{\sum_{i=1}^Nm_i}

    .. note::

       This function supports a wide variety of inputs, depending on
       how the atom information is provided and what should be
       calculated.

       When an :class:`MDAnalysis.core.groups.AtomGroup` object is
       provided in `group`, the atom masses and positions are retrieved
       from it and do not need to be provided in `masses` and
       `positions`, respectively. If the :code:`AtomGroup` abides by the
       standard topological heirarchy, you can specify the desired
       grouping in `grouping` and the appropriate centers of mass will
       be calculated. Otherwise, if and only if the :code:`AtomGroup`
       contains equisized or identical groups corresponding to the
       desired grouping (i.e., the :code:`AtomGroup` has atoms that are
       or can be treated as nonbonded entities or topological groups
       with the same number of but not necessarily identical
       constituents), you can provide the total number of groups in
       `n_groups` and the atom masses and positions will be distributed
       accordingly.

       If the trajectory is not unwrapped, the number of periodic
       boundary crossings (and optionally, the system dimensions if they
       are not embedded in the :code:`AtomGroup`) can be provided in
       `images` (and `dimensions`).

       If the :code:`AtomGroup` does not have the correct structural
       information and the residues or segments do not contain the same
       number of atoms, the atom masses and positions can each be
       provided directly as a :class:`numpy.ndarray` or list in `masses`
       and `positions`, respectively. To calculate the overall center of
       mass, the array-like object holding the masses should be
       one-dimensional, while that containing the positions should be
       two-dimensional. To calculate centers of mass for multiple
       groups, the array-like object holding the masses should be
       two-dimensional (indices: group, atom), while that containing the
       positions should be three-dimensional (indices: group, atom,
       axis). When a list is used, the inner arrays do not have to be
       homogeneously shaped, thus allowing you to calculate the centers
       of mass for residues or segments with different numbers of atoms.

       You may also provide only one of the atom masses or positions, in
       which case the missing information will be retrieved from the
       :code:`AtomGroup` provided in `group`. This is generally not
       recommended since the shapes of the provided and retrieved arrays
       may be incompatible.

    Parameters
    ----------
    group : `MDAnalysis.AtomGroup`, optional
        Collection of atoms to compute the centers of mass for. If not
        provided, the atom masses and posititions must be provided
        directly in `masses` and `positions`, respectively.

    grouping : `str`, optional
        Determines which center of mass is calculated if atom masses and
        positions are retrieved from `group`.

        .. container::

           **Valid values**:

           * :code:`None`: Center of mass of all atoms in `group`.
           * :code:`"residues"`: Centers of mass for each residue or
             molecule in `group`.
           * :code:`"segments"`: Centers of mass for each segment or
             chain in `group`.

    masses : array-like, keyword-only, optional
        Atom masses.

        .. container::

           **Shape**:

           The general ungrouped shape is :math:`(N,)`.

           For equisized or identical groups, the :class:`numpy.ndarray`
           object should have shape

           * :math:`(N,)` for the overall center of mass,
           * :math:`(N_\mathrm{res},\,N/N_\mathrm{res})` for the residue
             centers of mass, where :math:`N_\mathrm{res}` is
             the number of residues, or
           * :math:`(N_\mathrm{seg},\,N/N_\mathrm{seg}` for the segment
             centers of mass, where :math:`N_\mathrm{seg}` is
             the number of segments.

           For groups with different numbers of atoms, the list should
           contain inner array-like objects holding the masses of the
           atoms in each group.

        **Reference unit**: :math:`\mathrm{g/mol}`.

    positions : array-like, keyword-only, optional
        Atom positions.

        .. container::

           **Shape**:

           The general ungrouped shape is :math:`(N,\,3)`.

           For equisized or identical groups, the :class:`numpy.ndarray`
           object should have shape

           * :math:`(N,\,3)` for the overall center of mass,
           * :math:`(N_\mathrm{res},\,N/N_\mathrm{res},\,3)` for the
             residue centers of mass, or
           * :math:`(N_\mathrm{seg},\,N/N_\mathrm{seg},\,3)` for the
             segment centers of mass.

           For groups with different numbers of atoms, the list should
           contain inner array-like objects holding the positions of the
           atoms in each group.

        **Reference unit**: :math:`\mathrm{Å}`.

    images : array-like, keyword-only, optional
        Image flags for the atoms. Must be provided to get correct
        results if the trajectory is wrapped.

        **Shape**: Same as `positions`.

    dimensions : `numpy.ndarray`, keyword-only, optional
        System dimensions. Must be provided if `images` is provided and
        `group` is not provided or does not contain the system
        dimensions.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    n_groups : `int`, keyword-only, optional
        Number of residues or segments. Must be provided if `group` has
        an irregular topological heirarchy or the `masses` and
        `positions` arrays have the general ungrouped shapes.

    raw : `bool`, keyword-only, default: :code:`False`
        Determines whether atom masses and positions are returned.

    Returns
    -------
    com : `numpy.ndarray`
        Centers of mass.

        .. container::

           **Shape**:

           * :math:`(3,)` for :code:`grouping=None`.
           * :math:`(N_\mathrm{res},\,3)` for
             :code:`grouping="residues"`.
           * :math:`(N_\mathrm{seg},\,3)` for
             :code:`grouping="segments"`.

    masses : `numpy.ndarray`
        Atom masses. Only returned if `group` was provided and contains
        equisized or identical groups, and :code:`raw=True`.

        .. container::

           **Shape**:

           * :math:`(N,)` for :code:`grouping=None`.
           * :math:`(N_\mathrm{res},\,N/N_\mathrm{res})` for
             :code:`grouping="residues"`.
           * :math:`(N_\mathrm{seg},\,N/N_\mathrm{seg})` for
             :code:`grouping="segments"`.

        **Reference unit**: :math:`\mathrm{g/mol}`.

    positions : `numpy.ndarray`
        Unwrapped atom positions. Only returned if `group` was
        provided and contains equisized or identical groups, and
        :code:`raw=True`.

        .. container::

           **Shape**:

           * :math:`(N,\,3)` for :code:`grouping=None`.
           * :math:`(N_\mathrm{res},\,N/N_\mathrm{res},\,3)` for
             :code:`grouping="residues"`.
           * :math:`(N_\mathrm{seg},\,N/N_\mathrm{seg},\,3)` for
             :code:`grouping="segments"`.

        **Reference unit**: :math:`\mathrm{Å}`.

    Examples
    --------
    For an :class:`MDAnalysis.core.groups.AtomGroup` object with all
    necessary topological information and an unwrapped trajectory, the
    overall, per-residue, and per-segment centers of mass can be
    calculated as follows:

    >>> universe = mda.Universe("topology.pdb", "trajectory.nc")
    >>> group = universe.select_atoms("all")
    >>> com_ovr = center_of_mass(group)
    >>> com_res = center_of_mass(group, "residues")
    >>> com_seg = center_of_mass(group, "segments")

    If the :code:`AtomGroup` does not contain residue or segment
    information and the per-residue or per-segment centers of mass,
    respectively, are desired, the number of residues or segments can be
    provided in `n_groups`:

    >>> com_res = center_of_mass(group, "residues", n_groups=2) # 2 residues

    If the trajectory is wrapped, the number of periodic boundary
    crossings must be provided in `images`. Additionally, if the
    system dimensions are not embedded in the :code:`AtomGroup`, they
    must also be provided in `dimensions`:

    >>> images = np.array(((1, 0, 0), (0, 0, 0), (0, -1, 0),
    ...                    (0, 1, 0), (0, 0, 1), (-1, 1, 0)))
    >>> dimensions = np.array((5.0, 8.0, 10.0))
    >>> com_ovr = center_of_mass(group, images=images, dimensions=dimensions)

    Alternatively, if the atom masses and positions are directly
    available as :class:`numpy.ndarray` objects, the overall center of
    mass can be calculated as follows:

    >>> masses = np.array((12.01, 1.01, 1.01, 12.01, 1.01, 1.01))
    >>> positions = np.array(((0.0, -0.07579, 0.0),
    ...                       (0.86681, 0.60144, 0.0),
    ...                       (-0.86681, 0.60144, 0.0),
    ...                       (0.0, -0.07579, 1.0),
    ...                       (0.86681, 0.60144, 1.0),
    ...                       (-0.86681, 0.60144, 1.0)))
    >>> com_ovr = center_of_mass(masses=masses, positions=positions)

    If the per-residue center of mass is desired, the number of residues
    can be provided in `n_groups`:

    >>> com_res = center_of_mass(masses=masses, positions=positions, n_groups=2)

    or the arrays containing the atom masses and positions can be
    reshaped to the appropriate shapes:

    >>> n_groups = 2
    >>> com_res = center_of_mass(masses=masses.reshape((n_groups, -1)),
    ...                          positions=positions.reshape((n_groups, -1, 3)))

    Like before, if the trajectory is wrapped, the number of periodic
    boundary crossings and system dimensions must be provided in
    `images` and `dimensions`, respectively:

    >>> images = np.array(((0, 0, 0), (0, 0, 0), (0, 0, 0),
    ...                    (0, 0, 1), (0, 0, 1), (0, 0, 1)))
    >>> dimensions = np.array((12.0, 12.0, 12.0))
    >>> com_ovr = center_of_mass(masses=masses, positions=positions,
    ...                          images=images, dimensions=dimensions)

    Finally, if the per-residue or per-segment center of mass is
    desired but the groups contain different numbers of atoms, the
    the atom masses and positions can be provided as lists of arrays
    holding the masses and positions of the atoms in each group:

    >>> masses = [(12.01, 1.01, 1.01), (22.99,), (12.01, 1.01)]
    >>> positions = [((0.0, -0.07579, 0.0),
    ...               (0.86681, 0.60144, 0.0),
    ...               (-0.86681, 0.60144, 0.0)),
    ...              ((0.0, 0.0, 0.0),),
    ...              ((0.0, -0.07579, 1.0),
    ...               (0.86681, 0.60144, 1.0))]
    >>> com_res = center_of_mass(masses=masses, positions=positions)

    It is still possible to pass in the number of periodic boundary
    crossings if the trajectory is wrapped, but attention must be paid
    to the shape of the array:

    >>> images = [((0, 0, 0), (0, 0, 0), (0, 0, 0)),
    ...           ((1, 1, 1),),
    ...           ((0, 1, 0), (0, 1, 0))]
    >>> dimensions = np.array((10.0, 10.0, 10.0))
    >>> com_res = center_of_mass(masses=masses, positions=positions,
    ...                          images=images, dimensions=dimensions)
    """

    # Check whether grouping is valid
    if grouping not in {None, "residues", "segments"}:
        emsg = (f"Invalid grouping: '{grouping}'. Valid options are "
                "None, 'residues', and 'segments'.")
        raise ValueError(emsg)

    # Get system dimensions if image flags are provided
    if images is not None:
        if dimensions is None:
            try:
                dimensions = group.dimensions[:3]
            except (NameError, TypeError):
                emsg = ("Image flags were provided, but no system "
                        "dimensions were provided or found in the "
                        "trajectory.")
                raise ValueError(emsg)
        else:
            dimensions = np.asarray(dimensions)

    # Get particle masses and positions from the trajectory, if
    # necessary
    missing = (masses is None, positions is None)
    if any(missing):
        if group is None:
            emsg = ("Either a group of atoms or atom positions and "
                    "masses must be provided.")
            raise ValueError(emsg)

        # Check whether the groups have equal numbers of atoms
        if grouping is None:
            same = True
        else:
            groups = getattr(group, grouping)

            # Calculate and return the centers of mass for different
            # groups here if unwrapping and the mass and position arrays
            # are not needed
            if not (same := all(g.atoms.n_atoms == groups[0].atoms.n_atoms
                            for g in groups)) and images is None and not raw:
                return np.array([g.atoms.center_of_mass() for g in groups])

        # Get and unwrap particle positions, if necessary
        if missing[1]:
            positions = group.positions
            if images is not None:
                positions += images * dimensions[:3]

        # Get particle masses and ensure correct dimensionality, if
        # necessary
        if same:
            if missing[0]:
                masses = group.masses
            if grouping is not None or n_groups is not None:
                shape = (n_groups or getattr(group, f"n_{grouping}"), -1, 3)
                masses = masses.reshape(shape[:-1])
                positions = positions.reshape(shape)
        else:
            if missing[0]:
                masses = [g.atoms.masses for g in groups]
            if missing[1]:
                positions = [positions[g.atoms.ix] for g in groups]
    else:

        # Try to convert arrays to NumPy arrays if they are not already
        # to take advantage of vectorized operations later
        try:
            positions = np.asarray(positions)
            masses = np.asarray(masses)
            if images is not None:
                positions += images
        except ValueError:
            pass
        if type(masses) != type(positions):
            emsg = ("The shapes of the arrays containing the masses "
                    "and positions are incompatible.")
            raise ValueError(emsg)
        if images is not None and type(images) != type(positions):
            emsg = ("The shapes of the arrays containing the positions "
                    "and image flags are incompatible.")
            raise ValueError(emsg)

    # Calculate the centers of mass for the specified grouping
    if isinstance(positions, np.ndarray):

        # Reshape the mass and position arrays based on the specified
        # number of groups
        if n_groups is not None:
            masses = masses.reshape((n_groups, -1))
            positions = positions.reshape((n_groups, -1, 3))

        com = (np.einsum("...a,...ad->...d", masses, positions)
               / masses.sum(axis=-1, keepdims=True))
    else:
        if images is not None:
            for j, (p, i) in enumerate(zip(positions, images)):
                positions[j] = p + i * dimensions
        com = np.array([np.dot(m, p) / m.sum()
                        for m, p in zip(masses, positions)])

    # Return raw masses and positions, if desired
    if raw:
        return com, masses, positions

    return com

def radius_of_gyration(
        group: mda.AtomGroup = None, grouping: str = None, *,
        masses: Union[np.ndarray[float], list[np.ndarray[float]]] = None,
        positions: Union[np.ndarray[float], list[np.ndarray[float]]] = None,
        com: np.ndarray[float] = None,
        images: Union[np.ndarray[int], list[np.ndarray[int]]] = None,
        dimensions: np.ndarray[float] = None, n_groups: int = None,
        components: bool = False) -> Union[float, np.ndarray[float]]:

    r"""
    Computes the radii of gyration :math:`R_\mathrm{g}` for a collection
    of atoms.

    For a group of :math:`N` atoms with masses :math:`m_i` and positions
    :math:`\mathbf{r}_i`, the radius of gyration is defined as

    .. math::

       R_\mathrm{g}=\sqrt{
       \frac{\sum_i^Nm_i\|\mathbf{r}_i-\mathbf{R}_\mathrm{com}\|^2}
       {\sum_i^Nm_i}}

    where :math:`\mathbf{R}_\mathrm{com}` is the center of mass.

    Alternatively, the radii of gyration around the coordinate axes can
    be calculated by only summing the radii components orthogonal to
    each axis. For example, the radius of gyration around the
    :math:`x`-axis is

    .. math::

       R_{\mathrm{g},\,x}=\sqrt{\frac{\sum_i^Nm_i
       \left[(\mathbf{r}_{i,\,y}-\mathbf{R}_{\mathrm{com},\,y})^2
       +(\mathbf{r}_{i,\,z}-\mathbf{R}_{\mathrm{com},\,z})^2\right]}
       {\sum_i^Nm_i}}

    .. note::

       This function supports a wide variety of inputs, depending on
       how the atom information is provided and what should be
       calculated.

       When an :class:`MDAnalysis.core.groups.AtomGroup` object is
       provided in `group`, the atom masses and positions are retrieved
       from it and do not need to be provided in `masses` and
       `positions`, respectively. If the :code:`AtomGroup` abides by the
       standard topological heirarchy, you can specify the desired
       grouping in `grouping` and the appropriate radii of gyration will
       be calculated. Otherwise, if and only if the :code:`AtomGroup`
       contains equisized or identical groups corresponding to the
       desired grouping (i.e., the :code:`AtomGroup` has atoms that are
       or can be treated as nonbonded entities or topological groups
       with the same number of but not necessarily identical
       constituents), you can provide the total number of groups in
       `n_groups` and the atom masses and positions will be distributed
       accordingly.

       If the trajectory is not unwrapped, the number of periodic
       boundary crossings (and optionally, the system dimensions if they
       are not embedded in the :code:`AtomGroup`) can be provided in
       `images` (and `dimensions`).

       If the :code:`AtomGroup` does not have the correct structural
       information and the residues or segments do not contain the same
       number of atoms, the atom masses and positions can each be
       provided directly as a :class:`numpy.ndarray` or list in `masses`
       and `positions`, respectively. To calculate the overall radius of
       gyration, the array-like object holding the masses should be
       one-dimensional, while that containing the positions should be
       two-dimensional. To calculate radii of gyration for multiple
       groups, the array-like object holding the masses should be
       two-dimensional (indices: group, atom), while that containing the
       positions should be three-dimensional (indices: group, atom,
       axis). When a list is used, the inner arrays do not have to be
       homogeneously shaped, thus allowing you to calculate the radii of
       gyration for residues or segments with different numbers of atoms.

       You may also provide only one of the atom masses or positions, in
       which case the missing information will be retrieved from the
       :code:`AtomGroup` provided in `group`. This is generally not
       recommended since the shapes of the provided and retrieved arrays
       may be incompatible.

    Parameters
    ----------
    group : `MDAnalysis.AtomGroup`, optional
        Collection of atoms to compute the radii of gyration for. If not
        provided, the atom masses and posititions must be provided
        directly in `masses` and `positions`, respectively.

    grouping : `str`, optional
        Determines which radius of gyration is calculated if atom masses
        and positions are retrieved from `group`.

        .. container::

           **Valid values**:

           * :code:`None`: Radius of gyration of all atoms in `group`.
           * :code:`"residues"`: Radius of gyration for each residue or
             molecule in `group`.
           * :code:`"segments"`: Radius of gyration for each segment or
             chain in `group`.

    masses : array-like, keyword-only, optional
        Atom masses.

        .. container::

           **Shape**:

           The general ungrouped shape is :math:`(N,)`.

           For equisized or identical groups, the :class:`numpy.ndarray`
           object should have shape

           * :math:`(N,)` for the overall radius of gyration,
           * :math:`(N_\mathrm{res},\,N/N_\mathrm{res})` for the residue
             radii of gyration, where :math:`N_\mathrm{res}` is the
             number of residues, or
           * :math:`(N_\mathrm{seg},\,N/N_\mathrm{seg}` for the segment
             radii of gyration, where :math:`N_\mathrm{seg}` is the
             number of segments.

           For groups with different numbers of atoms, the list should
           contain inner array-like objects holding the masses of the
           atoms in each group.

        **Reference unit**: :math:`\mathrm{g/mol}`.

    positions : array-like, keyword-only, optional
        Atom positions.

        .. container::

           **Shape**:

           The general ungrouped shape is :math:`(N,\,3)`.

           For equisized or identical groups, the :class:`numpy.ndarray`
           object should have shape

           * :math:`(N,\,3)` for the overall radius of gyration,
           * :math:`(N_\mathrm{res},\,N/N_\mathrm{res},\,3)` for the
             residue radii of gyration, or
           * :math:`(N_\mathrm{seg},\,N/N_\mathrm{seg},\,3)` for the
             segment radii of gyration.

           For groups with different numbers of atoms, the list should
           contain inner array-like objects holding the positions of the
           atoms in each group.

        **Reference unit**: :math:`\mathrm{Å}`.

    com : `numpy.ndarray`, keyword-only, optional
        Centers of mass.

        .. container::

           **Shape**:

           * :math:`(3,)` for the overall radius of gyration.
           * :math:`(N_\mathrm{res},\,3)` for the residue radii of
             gyration.
           * :math:`(N_\mathrm{seg},\,3)` for the segment radii of
             gyration.

    images : array-like, keyword-only, optional
        Image flags for the atoms. Must be provided to get correct
        results if the trajectory is wrapped.

        **Shape**: Same as `positions`.

    dimensions : `numpy.ndarray`, keyword-only, optional
        System dimensions. Must be provided if `images` is provided and
        `group` is not provided or does not contain the system
        dimensions.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    n_groups : `int`, keyword-only, optional
        Number of residues or segments. Must be provided if `group` has
        an irregular topological heirarchy or the `masses` and
        `positions` arrays have the general ungrouped shapes.

    components : `bool`, keyword-only, default: :code:`False`
        Specifies whether the components of the radii of gyration are
        calculated and returned instead.

    Returns
    -------
    gyradii : `float` or `numpy.ndarray`
        Radii of gyration.

        .. container::

           **Shape**:

           * Scalar for :code:`grouping=None`.
           * :math:`(N_\mathrm{res},)` for :code:`grouping="residues"`.
           * :math:`(N_\mathrm{seg},)` for :code:`grouping="segments"`.

           If :code:`components=True`, an additional axis with length
           :math:`3` is added to the end.

        **Reference unit**: :math:`\mathrm{Å}`.

    Examples
    --------
    For an :class:`MDAnalysis.core.groups.AtomGroup` object with all
    necessary topological information and an unwrapped trajectory, the
    overall, per-residue, and per-segment radii of gyration can be
    calculated as follows:

    >>> universe = mda.Universe("topology.pdb", "trajectory.nc")
    >>> group = universe.select_atoms("all")
    >>> Rg_ovr = radius_of_gyration(group)
    >>> Rg_res = radius_of_gyration(group, "residues")
    >>> Rg_seg = radius_of_gyration(group, "segments")

    If the :code:`AtomGroup` does not contain residue or segment
    information and the per-residue or per-segment centers of mass,
    respectively, are desired, the number of residues or segments can be
    provided in `n_groups`:

    >>> Rg_res = radius_of_gyration(group, "residues", n_groups=2) # 2 residues

    If the trajectory is wrapped, the number of periodic boundary
    crossings must be provided in `images`. Additionally, if the
    system dimensions are not embedded in the :code:`AtomGroup`, they
    must also be provided in `dimensions`:

    >>> images = np.array(((1, 0, 0), (0, 0, 0), (0, -1, 0),
    ...                    (0, 1, 0), (0, 0, 1), (-1, 1, 0)))
    >>> dimensions = np.array((5.0, 8.0, 10.0))
    >>> Rg_ovr = radius_of_gyration(group, images=images, dimensions=dimensions)

    Alternatively, if the atom masses and positions are directly
    available as :class:`numpy.ndarray` objects, the overall radius of
    gyration can be calculated as follows:

    >>> masses = np.array((12.01, 1.01, 1.01, 12.01, 1.01, 1.01))
    >>> positions = np.array(((0.0, -0.07579, 0.0),
    ...                       (0.86681, 0.60144, 0.0),
    ...                       (-0.86681, 0.60144, 0.0),
    ...                       (0.0, -0.07579, 1.0),
    ...                       (0.86681, 0.60144, 1.0),
    ...                       (-0.86681, 0.60144, 1.0)))
    >>> Rg_ovr = radius_of_gyration(masses=masses, positions=positions)

    If the per-residue radius of gyration is desired, the number of
    residues can be provided in `n_groups`:

    >>> Rg_res = radius_of_gyration(masses=masses, positions=positions,
    ...                             n_groups=2)

    or the arrays containing the atom masses and positions can be
    reshaped to the appropriate shapes:

    >>> n_groups = 2
    >>> Rg_res = radius_of_gyration(
    ...     masses=masses.reshape((n_groups, -1)),
    ...     positions=positions.reshape((n_groups, -1, 3))
    ... )

    Like before, if the trajectory is wrapped, the number of periodic
    boundary crossings and system dimensions must be provided in
    `images` and `dimensions`, respectively:

    >>> images = np.array(((0, 0, 0), (0, 0, 0), (0, 0, 0),
    ...                    (0, 0, 1), (0, 0, 1), (0, 0, 1)))
    >>> dimensions = np.array((12.0, 12.0, 12.0))
    >>> Rg_ovr = radius_of_gyration(masses=masses, positions=positions,
    ...                             images=images, dimensions=dimensions)

    Finally, if the per-residue or per-segment radius of gyration is
    desired but the groups contain different numbers of atoms, the
    the atom masses and positions can be provided as lists of arrays
    holding the masses and positions of the atoms in each group:

    >>> masses = [(12.01, 1.01, 1.01), (22.99,), (12.01, 1.01)]
    >>> positions = [((0.0, -0.07579, 0.0),
    ...               (0.86681, 0.60144, 0.0),
    ...               (-0.86681, 0.60144, 0.0)),
    ...              ((0.0, 0.0, 0.0),),
    ...              ((0.0, -0.07579, 1.0),
    ...               (0.86681, 0.60144, 1.0))]
    >>> Rg_res = radius_of_gyration(masses=masses, positions=positions)

    It is still possible to pass in the number of periodic boundary
    crossings if the trajectory is wrapped, but attention must be paid
    to the shape of the array:

    >>> images = [((0, 0, 0), (0, 0, 0), (0, 0, 0)),
    ...           ((1, 1, 1),),
    ...           ((0, 1, 0), (0, 1, 0))]
    >>> dimensions = np.array((10.0, 10.0, 10.0))
    >>> Rg_res = radius_of_gyration(masses=masses, positions=positions,
    ...                             images=images, dimensions=dimensions)

    For any of the examples above, the components of the radii of
    gyration can be calculated and returned by setting
    :code:`components=True`:

    >>> Rg_ovr = radius_of_gyration(group, components=True)
    """

    # Check whether grouping is valid
    if grouping not in {None, "residues", "segments"}:
        emsg = (f"Invalid grouping: '{grouping}'. Valid options are "
                "None, 'residues', and 'segments'.")
        raise ValueError(emsg)

    # Get particle masses and positions from the trajectory and the
    # center(s) of mass, if necessary
    missing = (masses is None, positions is None, com is None)
    if any(missing[:2]):
        com, masses, positions = center_of_mass(
            group,
            grouping,
            masses=masses,
            positions=positions,
            images=images,
            dimensions=dimensions,
            n_groups=n_groups,
            raw=True
        )
    elif missing[2]:
        com, masses, positions = center_of_mass(
            masses=masses,
            positions=positions,
            images=images,
            dimensions=dimensions,
            n_groups=n_groups,
            raw=True
        )

    if isinstance(positions, np.ndarray):
        if components:
            cpos = (positions
                    - np.expand_dims(com, axis=positions.ndim - 2)) ** 2

            # Compute the radii of gyration in each direction for
            # equisized or identical groups
            if grouping or n_groups:
                return np.sqrt(
                    np.einsum("ga,gad->gd", masses,
                              np.stack((cpos[:, :, (1, 2)].sum(axis=2),
                                        cpos[:, :, (0, 2)].sum(axis=2),
                                        cpos[:, :, (0, 1)].sum(axis=2)),
                                       axis=2))
                    / masses.sum(axis=1, keepdims=True)
                )

            # Compute the radius of gyration in each direction for all
            # atoms
            return np.sqrt(
                np.dot(
                    masses,
                    np.hstack((cpos[:, (1, 2)].sum(axis=1, keepdims=True),
                               cpos[:, (0, 2)].sum(axis=1, keepdims=True),
                               cpos[:, (0, 1)].sum(axis=1, keepdims=True)))
                ) / masses.sum()
            )

        # Compute the overall radii of gyration for equisized or
        # identical groups
        elif grouping or n_groups:
            return np.sqrt(
                np.einsum("ga,gad->gd", masses,
                          (positions - com[:, None]) ** 2).sum(axis=1)
                / masses.sum(axis=1)
            )

        # Compute the overall radius of gyration for all atoms
        return np.sqrt(np.dot(masses, (positions - com) ** 2).sum()
                       / masses.sum())

    # Compute the radii of gyration in each direction for asymmetric
    # groups
    if components:
        gyradii = np.empty(com.shape)
        for i, (m, p, c) in enumerate(zip(masses, positions, com)):
            cpos = (p - c) ** 2
            gyradii[i] = np.array(
                (np.dot(m, cpos[:, (1, 2)].sum(axis=1)),
                 np.dot(m, cpos[:, (0, 2)].sum(axis=1)),
                 np.dot(m, cpos[:, (0, 1)].sum(axis=1)))
            ) / m.sum()
        return np.sqrt(gyradii)

    # Compute the overall radii of gyration for asymmetric groups
    return np.sqrt(
        [np.einsum("a,ad->d", m, (p - c) ** 2).sum() / m.sum()
         for m, p, c in zip(masses, positions, com)]
    )