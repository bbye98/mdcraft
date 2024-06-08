"""
Bulk structural analysis
========================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains classes to analyze the structure of bulk fluid and
electrolyte systems.
"""

from itertools import combinations_with_replacement
from typing import Union
import warnings

import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
import numba
import numpy as np
from scipy.integrate import simpson
from scipy.signal import argrelextrema
from scipy.special import jv

from .base import Hash, DynamicAnalysisBase, NumbaAnalysisBase
from .. import FOUND_OPENMM, Q_, ureg
from ..algorithm import accelerated
from ..algorithm.molecule import center_of_mass
from ..algorithm.unit import is_unitless, strip_unit
from ..algorithm.utility import get_closest_factors

if FOUND_OPENMM:
    from openmm import unit

def radial_histogram(
        positions_1: np.ndarray[float], positions_2: np.ndarray[float], /,
        n_bins: int, range_: np.ndarray[float], dimensions: np.ndarray[float],
        bin_edges: np.ndarray[float] = None, *, exclusion: tuple[int] = None
    ) -> np.ndarray[float]:

    r"""
    Computes the radial histogram of distances between particles of
    the same species :math:`\alpha` or two different species
    :math:`\alpha` and :math:`\beta`.

    Parameters
    ----------
    positions_1 : `numpy.ndarray`, positional-only
        Positions or centers of mass of entities belonging to species
        :math:`\alpha`.

        **Shape**: :math:`(N_\alpha,\,3)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    positions_2 : `numpy.ndarray`, positional-only
        Positions or centers of mass of entities belonging to species
        :math:`\beta`.

        **Shape**: :math:`(N_\beta,\,3)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    n_bins : `int`
        Number of histogram bins :math:`N_\mathrm{bins}`.

    range_ : array-like
        Range of radii values.

        **Shape**: :math:`(2,)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    dimensions : array-like
        System dimensions and orthogonality.

        **Shape**: :math:`(6,)`.

        **Reference unit**: :math:`\mathrm{Å}` (dimensions),
        :math:`^\circ` (orthogonality).

    bin_edges : `numpy.ndarray`, optional
        Bin edges.

        **Shape**: :math:`(N_\mathrm{bins}+1,)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    exclusion : array-like, keyword-only, optional
        Tiles to exclude from the interparticle distances.

        **Shape**: :math:`(2,)`.

        **Example**: :code:`(1, 1)` to exclude self-interactions.

    Returns
    -------
    histogram : `numpy.ndarray`
        Radial histogram.

        **Shape**: :math:`(N_\mathrm{bins},)`.
    """

    # Get pair separation distances of atom pairs within range
    pairs, distances = capped_distance(
        positions_1,
        positions_2,
        range_[1],
        range_[0] - np.finfo(np.float64).eps,
        box=dimensions
    )

    # Exclude atom pairs with the same atoms or atoms from the
    # same residue
    if exclusion is not None:
        distances = distances[np.where(pairs[:, 0] // exclusion[0]
                                       != pairs[:, 1] // exclusion[1])[0]]

    return accelerated.numba_histogram(
        distances,
        n_bins,
        bin_edges or accelerated.numba_histogram_bin_edges(
            np.asarray(range_, dtype=float),
            n_bins
        )
    )

def zeroth_order_hankel_transform(
        r: np.ndarray[float], f: np.ndarray[float], q: np.ndarray[float]
    ) -> np.ndarray[float]:

    r"""
    Computes the Hankel transform :math:`F_0(q)` of discrete data
    :math:`f(r)` using the zeroth-order Bessel function :math:`J_0`.

    .. math::

       F_0(q)=\int_0^\infty f(r)J_0(qr)r\,dr

    Parameters
    ----------
    r : `numpy.ndarray`
        Radii :math:`r`.

        **Reference unit**: :math:`\mathrm{Å}`.

    f : `numpy.ndarray`
        Discrete data :math:`f(r)` to Hankel transform.

        **Shape**: Same as `r`.

    q : `numpy.ndarray`
        Wavenumbers :math:`q` to evaluate the Hankel transforms at.

        **Reference unit**: :math:`\mathrm{Å}^{-1}`.

    Returns
    -------
    ht : `numpy.ndarray`
        Hankel transform of the discrete data.

        **Shape**: Same as `q`.
    """

    ht = 2 * np.pi * simpson(f * r * jv(0, q * r), r)
    if 0 in q:
        ht[q == 0] = 2 * np.pi * simpson(f * r, r)
    return ht

def radial_fourier_transform(
        r: np.ndarray[float], f: np.ndarray[float], q: np.ndarray[float]
    ) -> np.ndarray[float]:

    r"""
    Computes the radial Fourier transform :math:`\hat{f}(q)` of
    discrete data :math:`f(r)`.

    .. math::

       \hat{f}(q)=\frac{4\pi}{q}\int_0^\infty f(r)r\sin(qr)\,dr

    Parameters
    ----------
    r : `numpy.ndarray`
        Radii :math:`r`.

        **Reference unit**: :math:`\mathrm{Å}`.

    f : `numpy.ndarray`
        Discrete data :math:`f(r)` to Fourier transform.

        **Shape**: Same as `r`.

    q : `numpy.ndarray`
        Wavenumbers :math:`q` to evaluate the Fourier transforms at.

        **Reference unit**: :math:`\mathrm{Å}^{-1}`.

    Returns
    -------
    rft : `numpy.ndarray`
        Radial Fourier transform of the discrete data.

        **Shape**: Same as `q`.
    """

    rft = 4 * np.pi * np.divide(simpson(f * r * np.sin(np.outer(q, r)), x=r), q)
    if 0 in q:
        rft[q == 0] = 4 * np.pi * simpson(f * r ** 2, x=r)
    return rft

def calculate_coordination_numbers(
        bins: np.ndarray[float], rdf: np.ndarray[float], rho: float, *,
        n_coord_nums: int = 2, n_dims: int = 3, threshold: float = 0.1
    ) -> np.ndarray[float]:

    r"""
    Calculates coordination numbers :math:`n_k` from a radial
    distribution function :math:`g_{\alpha\beta}(r)`.

    It is defined as

    .. math::

       n_k=4\pi\rho_j\int_{r_{k-1}}^{r_k}r^2g_{\alpha\beta}(r)\,dr

    for three-dimensional systems and

    .. math::

       n_k=2\pi\rho_j\int_{r_{k-1}}^{r_k}rg_{\alpha\beta}(r)\,dr

    for two-dimensional systems, where :math:`k` is the index,
    :math:`\rho_\beta` is the bulk number density of species
    :math:`\beta` and :math:`r_k` is the :math:`(k + 1)`-th local
    minimum of :math:`g_{\alpha\beta}(r)`.

    If the radial distribution function :math:`g_{\alpha\beta}(r)` does
    not contain as many local minima as `n_coord_nums`, this method will
    return `numpy.nan` for the coordination numbers that could not be
    calculated.

    Parameters
    ----------
    bins : `numpy.ndarray`
        Centers of the histogram bins.

        **Shape**: :math:`(N_\mathrm{bins},)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    rdf : `numpy.ndarray`
        Radial distribution function :math:`g_{\alpha\beta}(r)`.

        **Shape**: :math:`(N_\mathrm{bins},)`.

    rho : `float`
        Number density :math:`\rho_\beta` of species :math:`\beta`.

        **Reference unit**: :math:`\mathrm{Å}^{n_\mathrm{dims}}`.

    n_coord_nums : `int`, keyword-only, default: :code:`2`
        Number of coordination numbers :math:`n_\mathrm{coord}` to
        calculate.

    n_dims : `int`, keyword-only, default: :code:`3`
        Number of dimensions :math:`n_\mathrm{dims}`.

    threshold : `float`, keyword-only, default: :code:`0.1`
        Minimum :math:`g_{\alpha\beta}(r)` value that must be reached
        before local minima are found.

    Returns
    -------
    coord_nums : `numpy.ndarray`
        Coordination numbers :math:`n_k`.
    """

    if n_dims not in {2, 3}:
        raise ValueError("Invalid number of dimensions.")

    def f(r: np.ndarray[float], rdf: np.ndarray[float], rho: float, start: int,
          stop: int) -> np.ndarray[float]:
        if n_dims == 3:
            return 4 * np.pi * rho * simpson(r ** 2 * rdf[start:stop], r)
        else:
            return 2 * np.pi * rho * simpson(r * rdf[start:stop], r)

    coord_nums = np.empty(n_coord_nums)
    coord_nums[:] = np.nan

    # Find indices of minima in the radial distribution function
    i_min, = argrelextrema(rdf, np.less)
    i_min = i_min[rdf[i_min] >= threshold]
    n_min = len(i_min)

    # Integrate the radial distribution function to get the coordination
    # number(s)
    if n_min:
        r = bins[:i_min[0] + 1]
        coord_nums[0] = f(r, rdf, rho, None, i_min[0] + 1)
        for i in range(min(n_coord_nums, n_min) - 1):
            r = bins[i_min[i]:i_min[i + 1] + 1]
            coord_nums[i + 1] = f(r, rdf, rho, i_min[i], i_min[i + 1] + 1)
    else:
        warnings.warn("No local minima found.")

    return coord_nums

def calculate_structure_factor(
        r: np.ndarray[float], g: np.ndarray[float], equal: bool, rho: float,
        x_1: float = 1, x_2: float = None, q: np.ndarray[float] = None, *,
        q_lower: float = None, q_upper: float = None, n_q: int = 1_000,
        n_dims: int = 3, formalism: str = "FZ"
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:

    r"""
    Calculates the static or partial structure factor
    :math:`S_{\alpha\beta}(q)` using the radial histogram bins :math:`r`
    and the radial distribution function :math:`g_{\alpha\beta}(r)` for
    an isotropic fluid.

    Parameters
    ----------
    r : `numpy.ndarray`
        Radii :math:`r`.

        **Reference unit**: :math:`\mathrm{Å}`.

    g : `numpy.ndarray`
        Radial distribution function :math:`g_{\alpha\beta}(r)`.

        **Shape**: Same as `r`.

    equal : `bool`
        Specifies whether `g` is between the same species
        (:math:`\alpha=\beta`). If :code:`False`, the number
        concentrations of species :math:`\alpha` and :math:`\beta` must
        be specified in `x_1` and `x_2`.

    rho : `float`
        Bulk number density :math:`\rho` or surface density
        :math:`\sigma`.

        **Reference unit**: :math:`\mathrm{Å}^{-3}` or
        :math:`\mathrm{Å}^{-2}`.

    x_1 : `float`, default: :code:`1`
        Number concentration of species :math:`\alpha`. Required if
        :code:`equal=False`.

    x_2 : `float`, optional
        Number concentration of species :math:`\beta`. Required if
        :code:`equal=False`.

    q : `numpy.ndarray`, optional
        Wavenumbers :math:`q`. If not specified, wavenumbers are
        generated using `q_lower`, `q_upper`, and `n_q`.

        **Reference unit**: :math:`\mathrm{Å}^{-1}`.

    q_lower : `float`, keyword-only, optional
        Lower bound for the wavenumbers :math:`q`. Has no effect if `q`
        is specified.

        **Reference unit**: :math:`\mathrm{Å}^{-1}`.

    q_upper : `float`, keyword-only, optional
        Upper bound for the wavenumbers :math:`q`. Has no effect if `q`
        is specified.

        **Reference unit**: :math:`\mathrm{Å}^{-1}`.

    n_q : `int`, keyword-only, default: :code:`1_000`
        Number of wavenumbers :math:`q` to generate. Has no effect if `q`
        is specified.

    n_dims : `int`, keyword-only, default: :code:`3`
        Number of dimensions :math:`n_\mathrm{dims}`.

    formalism : `str`, keyword-only, default: :code:`"FZ"`
        Formalism to use for the partial structure factor. Has no effect
        if :code:`equal=True`.

        .. container::

           **Valid values**:

           * :code:`"general"`: A general formalism given by

             .. math::

                S_{\alpha\beta}(q)=1+x_\alpha x_\beta\frac{4\pi\rho}{q}
                \int_0^\infty(g_{\alpha\beta}(r)-1)\sin{(qr)}r\,dr

           * :code:`"FZ"`: Faber–Ziman formalism [1]_

             .. math::

                S_{\alpha\beta}(q)=1+\frac{4\pi\rho}{q}\int_0^\infty
                (g_{\alpha\beta}(r)-1)\sin{(qr)}r\,dr

           * :code:`"AL"`: Ashcroft–Langreth formalism [2]_

             .. math::

                S_{\alpha\beta}(q)=\delta_{ij}+(x_\alpha x_\beta)^{1/2}
                \frac{4\pi\rho}{q}
                \int_0^\infty(g_{\alpha\beta}(r)-1)\sin{(qr)}r\,dr

           In two-dimensional systems, the second term is

           .. math::

              2\pi\rho\int_0^\infty (g_{\alpha\beta}(r)-1)J_0(qr)r\,dr

           instead, where :math:`J_0` is the zeroth-order Bessel
           function.

    Returns
    -------
    q : `numpy.ndarray`
        Wavenumbers :math:`q`.

        **Shape**: :math:`(N_q,)`.

    S : `numpy.ndarray`
        Static or partial structure factor :math:`S_{\alpha\beta}(q)`.

        **Shape**: :math:`(N_q,)`.

    References
    ----------
    .. [1] T. E. Faber and J. M. Ziman, A Theory of the Electrical
       Properties of Liquid Metals: III. the Resistivity of Binary
       Alloys, *Philosophical Magazine* **11**, **153** (1965).
       https://doi.org/10.1080/14786436508211931

    .. [2] N. W. Ashcroft and D. C. Langreth, Structure of Binary Liquid
       Mixtures. I, *Phys. Rev.* **156**, **685** (1967).
       https://doi.org/10.1103/PhysRev.156.685
    """

    if q is None:
        if q_lower is None:
            q_lower = 2 * np.pi / r[-1]
        if q_upper is None:
            q_upper = 2 * np.pi / r[0]
        q = np.linspace(q_lower, q_upper,
                        int((q_upper - q_lower) / q_lower)
                        if n_q is None else n_q)

    if n_dims == 3:
        _transform = radial_fourier_transform
    elif n_dims == 2:
        _transform = zeroth_order_hankel_transform
    else:
        raise ValueError("Invalid number of dimensions.")

    rho_sft = rho * _transform(r, g - 1, q)
    if equal or formalism == "FZ":
        return q, 1 + rho_sft
    elif not equal:
        if formalism == "AL":
            return q, (x_1 == x_2) + np.sqrt(x_1 * x_2) * rho_sft
        elif formalism == "general":
            return q, 1 + x_1 * x_2 * rho_sft
    raise ValueError("Invalid formalism.")

class RadialDistributionFunction(DynamicAnalysisBase):

    """
    Serial and parallel implementations to calculate the radial
    distribution function (RDF) :math:`g_{ij}(r)` between types
    :math:`i` and :math:`j` and its related properties for two-
    and three-dimensional systems.

    The RDF is given by

    .. math::

       g_{\\alpha\\beta}^\\mathrm{3D}(r)
       =\\frac{V}{4\\pi r^2N_\\alpha N_\\beta}\\sum_{i=1}^{N_\\alpha}
       \\sum_{j=1}^{N_\\beta}\\left\\langle\\delta
       \\left(|\\mathbf{r}_i-\\mathbf{r}_j|-r\\right)\\right\\rangle\\\\
       g_{\\alpha\\beta}^\\mathrm{2D}(r)
       =\\frac{A}{2\\pi rN_\\alpha N_\\beta}\\sum_{i=1}^{N_\\alpha}
       \\sum_{j=1}^{N_\\beta}\\left\\langle\\delta
       \\left(|\\mathbf{r}_i-\\mathbf{r}_j|-r\\right)\\right\\rangle

    where :math:`V` and :math:`A` are the system volume and area,
    :math:`N_\\alpha` and :math:`N_\\beta` are the number of particles
    of species :math:`\\alpha` and :math:`\\beta`, and
    :math:`\\mathbf{r}_i` and :math:`\\mathbf{r}_j` are the positions of
    particles :math:`i` and :math:`j` belonging to species
    :math:`\\alpha` and :math:`\\beta`, respectively. The RDF is
    normalized such that
    :math:`\\lim_{r\\rightarrow\\infty}g_{\\alpha\\beta}(r)=1` in a
    homogeneous system.

    A closely related quantity is the single particle density
    :math:`n_{\\alpha\\beta}(r)=\\rho_\\beta g_{\\alpha\\beta}(r)`,
    where :math:`\\rho_\\beta` is the number density of species
    :math:`\\beta`.

    The cumulative RDF is

    .. math::

       G_{\\alpha\\beta}^\\mathrm{3D}(r)
       =4\\pi\\int_0^rR^2g_{\\alpha\\beta}(R)\\,dR\\\\
       G_{\\alpha\\beta}^\\mathrm{2D}(r)
       =2\\pi\\int_0^rRg_{\\alpha\\beta}(R)\\,dR

    and the average number of :math:`\\beta` particles found within
    radius :math:`r` is

    .. math::

       N_{\\alpha\\beta}(r)=\\rho_\\beta G_{\\alpha\\beta}(r)

    The expression above can be used to compute the coordination numbers
    (number of neighbors in each solvation shell) by setting :math:`r`
    to values where :math:`g_{\\alpha\\beta}(r)` is locally minimized,
    which signify the solvation shell boundaries.

    .. container::

       The RDF can also be used to obtain other relevant structural
       properties, such as

       * the potential of mean force

         .. math::

            w_{\\alpha\\beta}(r)
            =-k_\\mathrm{B}T\\ln{g_{\\alpha\\beta}(r)}

         where :math:`k_\mathrm{B}` is the Boltzmann constant and
         :math:`T` is the system temperature, and

       * the static or partial structure factor (see
         :func:`calculate_structure_factor` for the possible
         definitions).

    Parameters
    ----------
    group_1 : `MDAnalysis.AtomGroup`
        Atom group containing species :math:`\\alpha`.

    group_2 : `MDAnalysis.AtomGroup`
        Atom group containing species :math:`\\beta`. If not specified,
        `group_1` is used for both species.

    n_bins : `int`, default: :code:`201`
        Number of histogram bins :math:`N_\\mathrm{bins}`.

    range_ : array-like, default: :code:`(0.0, 15.0)`
        Range of radii values. The upper bound should be less than half
        the smallest system dimension.

        **Shape**: :math:`(2,)`.

        **Reference unit**: :math:`\\mathrm{Å}`.

    exclusion : array-like, keyword-only, optional
        Tiles to exclude from the interparticle distances. The
        `groupings` parameter dictates what a tile represents.

        **Shape**: :math:`(2,)`.

        **Example**: :code:`(1, 1)` to exclude self-interactions.

    groupings : `str` or array-like, keyword-only, \
    default: :code:`"atoms"`
        Determines whether the centers of mass are used in lieu of
        individual atom positions. If `groupings` is a `str`, the same
        value is used for all `groups`.

        .. note::

           If the desired grouping is not :code:`"atoms"`,

           * the trajectory file should have segments (or chains)
             containing residues (or molecules) and residues containing
             atoms, and

           * residues and segments should be locally unwrapped at the
             simulation box edges, if not already, using
             :class:`MDAnalysis.transformations.wrap.unwrap`,
             :meth:`MDAnalysis.core.groups.AtomGroup.unwrap`, or
             :func:`MDAnalysis.lib.mdamath.make_whole`.

        .. container::

           **Valid values**:

           * :code:`"atoms"`: Atom positions (generally or for
             coarse-grained simulations).
           * :code:`"residues"`: Residues' centers of mass (for
             atomistic simulations).
           * :code:`"segments"`: Segments' centers of mass (for
             atomistic polymer simulations).

    drop_axis : `int` or `str`, keyword-only, optional
        Axis in three-dimensional space to ignore in the two-dimensional
        analysis.

        **Valid values**: :code:`0` or :code:`x` for the :math:`x`-axis,
        :code:`1` or :code:`y` for the :math:`y`-axis, and :code:`2` or
        :code:`z` for the :math:`z`-axis.

    norm : `str`, keyword-only, default: :code:`"rdf"`
        Determines how the radial histograms are normalized.

        .. container::

           **Valid values**:

           * :code:`norm="rdf"`: The radial distribution function
             :math:`g_{\\alpha\\beta}(r)` is computed.
           * :code:`norm="density"`: The single particle density
             :math:`n_{\\alpha\\beta}(r)` is computed.
           * :code:`norm=None`: The raw particle pair count in the
             radial histogram bins is returned.

    n_batches : `int`, keyword-only, optional
        Number of batches to divide the histogram calculation into.
        This is useful for large systems that cannot be processed in a
        single pass.

        .. note::

           If you use too few bins and too many batches, the histogram
           counts may be off by a few due to the floating-point nature
           of the cutoffs. However, when the RDF is averaged over a
           long trajectory with many particles, the difference should
           be negligible.

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units.

    parallel : `bool`, keyword-only, default: :code:`False`
        Determines whether the analysis is performed in parallel.

        .. note::

           The Dask backend generally provides the best performance.

    verbose : `bool`, keyword-only, default: :code:`True`
        Determines whether detailed progress is shown.

    **kwargs
        Additional keyword arguments to pass to
        :class:`MDAnalysis.analysis.base.AnalysisBase`.

    Attributes
    ----------
    universe : `MDAnalysis.Universe`
        :class:`MDAnalysis.core.universe.Universe` object containing all
        information describing the system.

    results.units : `dict`
        Reference units for the results. For example, to get the
        reference units for :code:`results.bins`, call
        :code:`results.units["bins"]`.

    results.bin_edges : `numpy.ndarray`
        Bin edges.

        **Shape**: :math:`(N_\\mathrm{bins}+1,)`.

        **Reference unit**: :math:`\\textrm{Å}`.

    results.bins : `numpy.ndarray`
        Bin centers :math:`r`.

        **Shape**: :math:`(N_\\mathrm{bins},)`.

        **Reference unit**: :math:`\\textrm{Å}`.

    results.counts : `numpy.ndarray`
        Raw particle pair counts in the radial histogram bins.

        **Shape**: :math:`(N_\\mathrm{bins},)`.

    results.rdf : `numpy.ndarray`
        .. container::

           One of

           * :code:`norm="rdf"`: the radial distribution function
             :math:`g_{\\alpha\\beta}(r)`,
           * :code:`norm="density"`: the single particle density
             :math:`n_{\\alpha\\beta}(r)`, or
           * :code:`norm=None`: the raw particle pair count in the
             radial histogram bins.

        **Shape**: :math:`(N_\mathrm{bins},)`.

    results.coordination_numbers : `numpy.ndarray`
        Coordination numbers :math:`n_k`. Only available after running
        :meth:`calculate_coordination_numbers`.

    results.pmf : `numpy.ndarray`
        Potential of mean force :math:`w_{\\alpha\\beta}(r)`. Only
        available after running :meth:`calculate_pmf`.

        **Shape**: :math:`(N_\\mathrm{bins},)`.

        **Reference unit**: :math:`\\mathrm{kJ/mol}`.

    results.wavenumbers : `numpy.ndarray`
        Wavenumbers :math:`q`. Only available after running
        :meth:`calculate_structure_factor`.

        **Reference unit**: :math:`\\textrm{Å}^{-1}`.

    results.ssf : `numpy.ndarray`
        Static or partial structure factor. Only available after running
        :meth:`calculate_structure_factor`.

        **Shape**: Same as `results.wavenumbers`.
    """

    def __init__(
            self, group_1: mda.AtomGroup, group_2: mda.AtomGroup = None,
            n_bins: int = 201, range_: tuple[float] = (0.0, 15.0), *,
            exclusion: tuple[int] = None,
            groupings: Union[str, tuple[str]] = "atoms",
            drop_axis: Union[int, str] = None, norm: str = "rdf",
            n_batches: int = None, reduced: bool = False,
            parallel: bool = False, verbose: bool = True, **kwargs) -> None:

        self._groups = (group_1, group_1 if group_2 is None else group_2)
        self.universe = self.group_1.universe
        if self.universe.dimensions is None:
            raise ValueError("Trajectory does not contain system "
                             "dimension information.")
        super().__init__(self.universe.trajectory, parallel, verbose, **kwargs)

        GROUPINGS = {"atoms", "residues", "segments"}
        if isinstance(groupings, str):
            if groupings not in GROUPINGS:
                emsg = (f"Invalid grouping '{groupings}'. Valid values: "
                        "'" + "', '".join(GROUPINGS) + "'.")
                raise ValueError(emsg)
            self._groupings = 2 * [groupings]
        else:
            if len(groupings) != 2:
                emsg = "'groupings' must be a string or an array with length 2."
                raise ValueError(emsg)
            for gr in groupings:
                if gr not in GROUPINGS:
                    emsg = (f"Invalid grouping '{gr}'. Valid values: "
                            "'" + "', '".join(GROUPINGS) + "'.")
                    raise ValueError(emsg)
            self._groupings = groupings

        self._drop_axis = (ord(drop_axis) - 120 if isinstance(drop_axis, str)
                           else drop_axis)
        if self._drop_axis not in {0, 1, 2, None}:
            raise ValueError("Invalid axis in 'drop_axis'.")

        self._n_bins = n_bins
        self._range = range_
        self._norm = norm
        self._exclusion = exclusion
        self._reduced = reduced
        self._n_batches = n_batches
        self._verbose = verbose

    def _prepare(self) -> None:

        # Preallocate floating-point number for system volume or area
        if not self._parallel and self._norm == "rdf":
            self._area_or_volume = 0.0

        # Preallocate arrays to store results
        self.results.bin_edges = accelerated.numba_histogram_bin_edges(
            np.asarray(self._range, dtype=float),
            self._n_bins
        )
        self.results.bins = (self.results.bin_edges[:-1]
                             + self.results.bin_edges[1:]) / 2
        if not self._parallel:
            self.results.counts = np.zeros(self._n_bins, dtype=int)

        # Store reference units
        self.results.units = Hash({"bins": ureg.angstrom,
                                   "edges": ureg.angstrom})

    def _single_frame(self) -> None:

        # Get system dimensions and orthogonality
        dimensions = self._ts.dimensions

        # Get positions or centers of mass of entities in the groups
        positions = [ag.positions if gr == "atoms" else center_of_mass(ag, gr)
                     for ag, gr in zip(self._groups, self._groupings)]

        # Add system volume or area analyzed in current frame
        if self._drop_axis is None:
            if self._norm == "rdf":
                self._area_or_volume += self._ts.volume
        else:

            # Apply corrections to avoid including periodic images in
            # the dimension to exclude
            positions[0][:, self._drop_axis] \
                = positions[1][:, self._drop_axis] = 0
            dimensions[self._drop_axis] = dimensions[:3].max()

            if self._norm == "rdf":
                self._area_or_volume += np.delete(dimensions[:3],
                                                  self._drop_axis).prod()

        # Tally counts in each pair separation distance bin
        if self._n_batches:
            edges = np.array_split(self.results.bin_edges, self._n_batches)
            ranges_indices = {
                e: np.where((self.results.bins > e[0])
                            & (self.results.bins < e[1]))[0]
                for e in [(self._range[0], edges[0][-1]),
                          *((a[-1], b[-1])
                            for a, b in zip(edges[:-1], edges[1:]))]
            }
            for r, i in ranges_indices.items():
                self.results.counts[i] += radial_histogram(
                    *positions,
                    n_bins=i.shape[0],
                    range_=r,
                    dimensions=dimensions,
                    exclusion=self._exclusion
                )
        else:
            self.results.counts += radial_histogram(
                *positions,
                n_bins=self._n_bins,
                range_=self._range,
                dimensions=dimensions,
                exclusion=self._exclusion
            )

    def _single_frame_parallel(self, index: int) -> np.ndarray[float]:

        # Set current trajectory frame
        _ts = self._sliced_trajectory[index]

        # Preallocate array to store entity counts followed by system
        # volume
        results = np.empty(1 + self._n_bins)

        # Get system dimensions and orthogonality
        dimensions = _ts.dimensions

        # Get positions or centers of mass of entities in the groups
        positions = [ag.positions if gr == "atoms" else center_of_mass(ag, gr)
                     for ag, gr in zip(self._groups, self._groupings)]

        # Store system volume or area analyzed in current frame
        if self._drop_axis is None:
            results[self._n_bins] = _ts.volume
        else:

            # Apply corrections to avoid including periodic images in
            # the dimension to exclude
            positions[0][:, self._drop_axis] \
                = positions[1][:, self._drop_axis] = 0
            dimensions[self._drop_axis] = dimensions[:3].max()

            results[self._n_bins] = np.delete(dimensions[:3], self._drop_axis).prod()

        # Compute radial histogram for current frame
        if self._n_batches:
            edges = np.array_split(self.results.bin_edges, self._n_batches)
            ranges_indices = {
                e: np.where((self.results.bins > e[0])
                            & (self.results.bins < e[1]))[0]
                for e in [(self._range[0], edges[0][-1]),
                          *((a[-1], b[-1])
                            for a, b in zip(edges[:-1], edges[1:]))]
            }
            for r, i in ranges_indices.items():
                results[i] = radial_histogram(
                    *positions,
                    n_bins=i.shape[0],
                    range_=r,
                    dimensions=dimensions,
                    exclusion=self._exclusion
                )
        else:
            results[:self._n_bins] = radial_histogram(
                *positions,
                n_bins=self._n_bins,
                range_=self._range,
                dimensions=dimensions,
                exclusion=self._exclusion
            )

        return results

    def _conclude(self) -> None:

        # Tally counts in each pair separation distance bin over all
        # frames
        if self._parallel:
            self._results = np.vstack(self._results).sum(axis=0)
            self.results.counts = self._results[:self._n_bins]
            self._area_or_volume = self._results[self._n_bins]

        # Compute the normalization factor
        norm = self.n_frames
        if self._norm is not None:
            if self._drop_axis is None:
                norm *= 4 * np.pi * np.diff(self.results.edges ** 3) / 3
            else:
                norm *= np.pi * np.diff(self.results.edges ** 2)
            if self._norm == "rdf":
                _N2 = getattr(self._groups[1], f"n_{self._groupings[1]}")
                if self._exclusion:
                    _N2 -= self._exclusion[1]
                norm *= (getattr(self._groups[0], f"n_{self._groupings[0]}")
                         * _N2 * self.n_frames / self._area_or_volume)

        # Compute and store the radial distribution function, the single
        # particle density, or the raw radial pair counts
        self.results.rdf = self.results.counts / norm

    def _get_rdf(self) -> np.ndarray[float]:

        r"""
        Returns the existing radial distribution function (RDF) if
        :code:`norm="rdf"` was passed to the :class:`RDF` constructor.
        Otherwise, the RDF is calculated and returned.

        Returns
        -------
        rdf : `numpy.ndarray`
            Radial distribution function :math:`g_{\alpha\beta}(r)`.
        """

        if self._norm == "rdf":
            return self.results.rdf

        _N2 = getattr(self._groups[1], f"n_{self._groupings[1]}")
        if self._exclusion:
            _N2 -= self._exclusion[1]
        if self._drop_axis is None:
            norm = 4 * np.diff(self.results.edges ** 3) / 3
        else:
            norm = np.diff(self.results.edges ** 2)
        return self._area_or_volume * self.results.counts / (
            np.pi * self.n_frames ** 2 * _N2 * norm
            * getattr(self._groups[0], f"n_{self._groupings[0]}")
        )

    def calculate_coordination_numbers(
            self, rho: float, *, n_coord_nums: int = 2, threshold: float = 0.1
        ) -> None:

        r"""
        Calculates the coordination numbers :math:`n_k`.

        If the radial distribution function :math:`g_{\alpha\beta}(r)`
        does not contain :math:`k` local minima, this method will return
        `numpy.nan` for the coordination numbers that could not be
        calculated.

        Parameters
        ----------
        rho : `float`
            Number density :math:`\rho_\beta` of species :math:`\beta`.

            **Reference unit**: :math:`\mathrm{nm}^{-3}`.

        n_coord_nums : `int`, keyword-only, default: :code:`2`
            Number of coordination numbers :math:`n_\mathrm{coord}` to
            calculate.

        threshold : `float`, keyword-only, default: :code:`0.1`
            Minimum :math:`g_{\alpha\beta}(r)` value that must be
            reached before local minima are found.
        """

        self.results.coordination_numbers = calculate_coordination_numbers(
            self.results.bins,
            self._get_rdf(),
            rho,
            n_coord_nums=n_coord_nums,
            n_dims=2 + (self._drop_axis is None),
            threshold=threshold
        )

    def calculate_pmf(
            self, temperature: Union[float, "unit.Quantity", Q_]) -> None:

        r"""
        Calculates the potential of mean force
        :math:`w_{\alpha\beta}(r)`.

        Parameters
        ----------
        temperature : `float` or `openmm.unit.Quantity`
            System temperature :math:`T`.

            .. note::

               If :code:`reduced=True` was set in the :class:`RDF`
               constructor, `temperature` should be equal to the energy
               scale. When the Lennard-Jones potential is used, it
               generally means that :math:`T^*=1`, or `temperature=1`.

            **Reference unit**: :math:`\mathrm{K}`.
        """

        if self._reduced and not is_unitless(temperature):
            emsg = "'temperature' cannot have units when reduced=True."
            raise ValueError(emsg)

        self.results.units["pmf"] = ureg.kilojoule / ureg.mole
        kBT = strip_unit(temperature, "K")[0]
        if not self._reduced:
            kBT *= (ureg.avogadro_constant * ureg.boltzmann_constant
                    * ureg.kelvin).m_as(self.results.units["pmf"])
        self.results.pmf = -kBT * np.log(self._get_rdf())

    def calculate_structure_factor(
            self, rho: float, x_1: float = None, x_2: float = None,
            q: np.ndarray[float] = None, *, q_lower: float = None,
            q_upper: float = None, n_q: int = 1_000, formalism: str = "FZ"
        ) -> None:

        r"""
        Computes the static or partial structure factor
        :math:`S_{\alpha\beta}(q)` using the radial histogram bins
        :math:`r` and the radial distribution function
        :math:`g_{\alpha\beta}(r)` for an isotropic fluid.

        Parameters
        ----------
        rho : `float`
            Bulk number density :math:`\rho`.

            **Reference unit**: :math:`\mathrm{Å}^{-3}`.

        x_1 : `float`, default: :code:`1`
            Number concentration of species :math:`\alpha`. Required if
            the two atom groups are not identical.

        x_2 : `float`, optional
            Number concentration of species :math:`\beta`. Required if
            the two atom groups are not identical.

        q : `numpy.ndarray`, optional
            Wavenumbers :math:`q`.

            **Reference unit**: :math:`\mathrm{Å}^{-1}`.

        q_lower : `float`, keyword-only, optional
            Lower bound for the wavenumbers :math:`q`. Has no effect if `q`
            is specified.

            **Reference unit**: :math:`\mathrm{Å}^{-1}`.

        q_upper : `float`, keyword-only, optional
            Upper bound for the wavenumbers :math:`q`. Has no effect if `q`
            is specified.

            **Reference unit**: :math:`\mathrm{Å}^{-1}`.

        n_q : `int`, keyword-only, default: :code:`1_000`
            Number of wavenumbers :math:`q` to generate. Has no effect if `q`
            is specified.

        formalism :`str`, keyword-only, default: :code:`"FZ"`
            Formalism to use for the partial structure factor. Has no effect
            if the two atom groups are the same.

            .. container::

               **Valid values**:

               * :code:`"general"`: A general formalism similar to that
                 of the static structure factor, except the second term
                 is multiplied by :math:`x_ix_j`.
               * :code:`"FZ"`: Faber–Ziman formalism.
               * :code:`"AL"`: Ashcroft–Langreth formalism.

               .. seealso::

                  For more information, see
                  :func:`calculate_structure_factor`.
        """

        self.results.wavenumbers, self.results.ssf = calculate_structure_factor(
            self.results.bins,
            self._get_rdf(),
            self._groups[0] == self._groups[1],
            rho,
            x_1,
            x_2,
            q=q,
            q_lower=q_lower,
            q_upper=q_upper,
            n_q=n_q,
            n_dims=2 + (self._drop_axis is None),
            formalism=formalism
        )

@numba.njit("c16(f8[:],f8[:])", fastmath=True)
def numba_delta_fourier_transform(
        q: np.ndarray[float], r: np.ndarray[float]) -> complex:

    r"""
    Serial Numba-accelerated Fourier transform of a Dirac delta function
    involving two one-dimensional NumPy arrays :math:`\mathbf{q}` and
    :math:`\mathbf{r}`, each with shape :math:`(3,)`.

    .. math::

        \mathcal{F}[\delta(\mathbf{q}-\mathbf{r})]
        =\exp(i\mathbf{q}\cdot\mathbf{r})

    Parameters
    ----------
    q : `np.ndarray`
        First vector :math:`\mathbf{q}`.

        **Shape**: :math:`(3,)`.

    r : `np.ndarray`
        Second vector :math:`\mathbf{r}`.

        **Shape**: :math:`(3,)`.

    Returns
    -------
    F : `complex`
        Fourier transform of the Dirac delta function,
        :math:`\mathcal{F}[\delta(\mathbf{q}-\mathbf{r})]`.
    """

    return np.exp(1j * accelerated.numba_dot(q, r))

def delta_fourier_transform_sum(
        qs: np.ndarray[float], rs: np.ndarray[float]) -> np.ndarray[complex]:

    r"""
    Evaluates the Fourier transforms of Dirac delta functions involving
    all possible combinations of multiple one-dimensional NumPy arrays
    :math:`\mathbf{q}` and :math:`\mathbf{r}`, each with shape
    :math:`(3,)`, summed over all :math:`\mathbf{r}`.

    .. math::

        \sum_\mathbf{r}\mathcal{F}[\delta(\mathbf{q}-\mathbf{r})]
        =\sum_\mathbf{r}\exp(i\mathbf{q}\cdot\mathbf{r})

    Parameters
    ----------
    qs : `np.ndarray`
        Multiple vectors :math:`\mathbf{q}`.

        **Shape**: :math:`(N_q,\,3)`.

    rs : `np.ndarray`
        Multiple vectors :math:`\mathbf{r}`.

        **Shape**: :math:`(N_r,\,3)`.

    Returns
    -------
    F : `np.ndarray`
        Fourier transforms of the Dirac delta functions, summed over all
        :math:`\mathbf{r}`.

        **Shape**: :math:`(N_q,)`.
    """

    F = np.empty(qs.shape[0], dtype=np.complex128)
    for i in numba.prange(qs.shape[0]):
        F[i] = 0.0j
        for j in range(rs.shape[0]):
            F[i] += numba_delta_fourier_transform(qs[i], rs[j])
    return F

@numba.njit("f8(f8[:])", fastmath=True)
def numba_pythagorean_trigonometric_identity(r: np.ndarray[float]) -> float:

    r"""
    Serial Numba-accelerated evaluation of the Pythagorean trigonometric
    identity for a one-dimensional NumPy array :math:`\mathbf{r}`.

    .. math::

       \left(\sum_{i=1}^3\cos(r_i)\right)^2
       +\left(\sum_{i=1}^3\sin(r_i)\right)^2

    Parameters
    ----------
    r : `np.ndarray`
        Vector :math:`\mathbf{r}`.

        **Shape**: :math:`(N_r,)`.

    Returns
    -------
    c2_s2 : `float`
        Pythagorean trigonometric identity for the vector
        :math:`\mathbf{r}`.
    """

    c = s = 0
    for i in range(r.shape[0]):
        c += np.cos(r[i])
        s += np.sin(r[i])
    return c ** 2 + s ** 2

@numba.njit("f8(f8[:],f8[:])", fastmath=True)
def numba_cross_pythagorean_trigonometric_identity(
        r: np.ndarray[float], s: np.ndarray[float]) -> float:

    r"""
    Serial Numba-accelerated evaluation of the cross Pythagorean
    trigonometric identity for two one-dimensional NumPy arrays
    :math:`\mathbf{r}` and :math:`\mathbf{s}`.

    .. math::

       2\left(\sum_{i=1}^3\cos(r_i)\sum_{j=1}^3\cos(s_j)
       +\sum_{i=1}^3\sin(r_i)\sum_{j=1}^3\sin(s_j)\right)

    Parameters
    ----------
    r : `np.ndarray`
        First vector :math:`\mathbf{r}`.

        **Shape**: :math:`(N_r,)`.

    s : `np.ndarray`
        Second vector :math:`\mathbf{s}`.

        **Shape**: :math:`(N_s,)`.

    Returns
    -------
    c2_s2 : `float`
        Cross Pythagorean trigonometric identity for the vectors
        :math:`\mathbf{r}` and :math:`\mathbf{s}`.
    """

    c1 = c2 = s1 = s2 = 0
    for i in range(r.shape[0]):
        c1 += np.cos(r[i])
        s1 += np.sin(r[i])
    for j in range(s.shape[0]):
        c2 += np.cos(s[j])
        s2 += np.sin(s[j])
    return 2 * (c1 * c2 + s1 * s2)

def ssf_trigonometric(qrs: np.ndarray[float]) -> np.ndarray[float]:

    r"""
    Computes the static structure factors using a two-dimensional NumPy
    array containing :math:`\mathbf{q}\cdot\mathbf{r}` using the
    trigonometric form.

    .. math::

        NS(q)=\left\langle\left(\sum_{j=1}^N
        \cos{(\mathbf{q}\cdot\mathbf{r}_j)}\right)^2+\left(
        \sum_{j=1}^N\sin{(\mathbf{q}\cdot\mathbf{r}_j)}
        \right)^2\right\rangle

    Parameters
    ----------
    qrs : `np.ndarray`
        Inner products :math:`\mathbf{q}\cdot\mathbf{r}_j`.

        **Shape**: :math:`(N_q,\,N_r)`.

    Returns
    -------
    ssf : `np.ndarray`
        Static structure factors.

        **Shape**: :math:`(N_q,)`.
    """

    ssf = np.empty(qrs.shape[0])
    for i in numba.prange(qrs.shape[0]):
        ssf[i] = numba_pythagorean_trigonometric_identity(qrs[i])
    return ssf

def psf_trigonometric(
        qrs1: np.ndarray[float], qrs2: np.ndarray[float]) -> np.ndarray[float]:

    r"""
    Computes the partial structure factors given two two-dimensional
    NumPy arrays, each containing :math:`\mathbf{q}\cdot\mathbf{r}`,
    using the trigonometric form.

    .. math::

        \frac{NS_{\alpha\beta}(q)}{2-\delta_{\alpha\beta}}
        =\left\langle
        \sum_{j=1}^{N_\alpha}\cos{(\mathbf{q}\cdot\mathbf{r}_j)}
        \sum_{k=1}^{N_\beta}\cos{(\mathbf{q}\cdot\mathbf{r}_k)}
        +\sum_{j=1}^{N_\alpha}\sin{(\mathbf{q}\cdot\mathbf{r}_j)}
        \sum_{k=1}^{N_\beta}\sin{(\mathbf{q}\cdot\mathbf{r}_k)}
        \right\rangle

    Parameters
    ----------
    qrs1 : `np.ndarray`
        First set of inner products :math:`\mathbf{q}\cdot\mathbf{r}_j`.

        **Shape**: :math:`(N_q,\,N_r)`.

    qrs2 : `np.ndarray`
        Second set of inner products :math:`\mathbf{q}\cdot\mathbf{r}_k`.

        **Shape**: :math:`(N_q,\,N_r)`.

    Returns
    -------
    ssf : `np.ndarray`
        Partial structure factors.

        **Shape**: :math:`(N_q,)`.
    """

    ssf = np.empty(qrs1.shape[0])
    for i in numba.prange(qrs1.shape[0]):
        ssf[i] = numba_cross_pythagorean_trigonometric_identity(qrs1[i],
                                                                qrs2[i])
    return ssf

class StructureFactor(NumbaAnalysisBase):

    """
    Serial and parallel implementations to calculate the static
    structure factor :math:`S(q)` or partial structure factor
    :math:`S_{\\alpha\\beta}(q)` for species :math:`\\alpha` and
    :math:`\\beta`.

    The static structure factor is a measure of how a material scatters
    incident radiation, and can be computed directly from a molecular
    dynamics simulation trajectory using

    .. math::

        S(q)&=\\frac{1}{N}\\left\\langle\\sum_{j=1}^N\\sum_{k=1}^N
        \\exp{[-i\\mathbf{q}\\cdot(\\mathbf{r}_j-\\mathbf{r}_k)]}
        \\right\\rangle\\\\&=\\frac{1}{N}\\left\\langle\\left(
        \\sum_{j=1}^N\\sin{(\\mathbf{q}\\cdot\\mathbf{r}_j)}\\right)^2
        +\\left(\\sum_{j=1}^N\\cos{(\\mathbf{q}\\cdot\\mathbf{r}_j)}
        \\right)^2\\right\\rangle

    where :math:`N` is the total number of particles or centers of mass,
    :math:`\\mathbf{q}` is the scattering wavevector, and
    :math:`\\mathbf{r}_i` is the position of the :math:`i`-th particle.

    For multicomponent systems, the equation above can be generalized to
    get the partial structure factors

    .. math::

       S_{\\alpha\\beta}(q)&=\\frac{2-\\delta_{\\alpha\\beta}}{N}
       \\left\\langle\\sum_{j=1}^{N_\\alpha}\\sum_{k=1}^{N_\\beta}
       \\exp{[-i\\mathbf{q}\\cdot(\\mathbf{r}_j-\\mathbf{r}_k)]}
       \\right\\rangle\\\\
       &=\\frac{2-\\delta_{\\alpha\\beta}}{N}\\left\\langle
       \\sum_{j=1}^{N_\\alpha}\\cos{(\\mathbf{q}\\cdot\\mathbf{r}_j)}
       \\sum_{k=1}^{N_\\beta}\\cos{(\\mathbf{q}\\cdot\\mathbf{r}_k)}
       +\\sum_{j=1}^{N_\\alpha}\\sin{(\\mathbf{q}\\cdot\\mathbf{r}_j)}
       \\sum_{k=1}^{N_\\beta}\\sin{(\\mathbf{q}\\cdot\\mathbf{r}_k)}
       \\right\\rangle

    where :math:`\\delta_{ij}` is the Kronecker delta, :math:`N_\\alpha`
    and :math:`N_\\beta` are the numbers of particles or centers of mass
    for species :math:`\\alpha` and :math:`\\beta`.

    The partial structure factors :math:`S_{\\alpha\\beta}(q)` and the
    static structure factor :math:`S(q)` are related via

    .. math::

       S(q)=\\sum_{\\alpha=1}^{N_\\mathrm{g}}
       \\sum_{\\beta=\\alpha}^{N_\\mathrm{g}}S_{\\alpha\\beta}(q)

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Groups of atoms that share the same grouping type. If
        :code:`mode=None`, all atoms in the universe must be assigned to
        a group. If :code:`mode="pair"`, there must be exactly one or
        two groups.

    groupings : `str` or array-like, default: :code:`"atoms"`
        Determines whether the centers of mass are used in lieu of
        individual atom positions. If `groupings` is a `str`, the same
        value is used for all `groups`.

        .. note::

           If the desired grouping is not :code:`"atoms"`,

           * the trajectory file should have segments (or chains)
             containing residues (or molecules) and residues containing
             atoms, and

           * residues and segments should be locally unwrapped at the
             simulation box edges, if not already, using
             :class:`MDAnalysis.transformations.wrap.unwrap`,
             :meth:`MDAnalysis.core.groups.AtomGroup.unwrap`, or
             :func:`MDAnalysis.lib.mdamath.make_whole`.

        .. container::

           **Valid values**:

           * :code:`"atoms"`: Atom positions (generally or for
             coarse-grained simulations).
           * :code:`"residues"`: Residues' centers of mass (for
             atomistic simulations).
           * :code:`"segments"`: Segments' centers of mass (for
             atomistic polymer simulations).

    mode : `str`, keyword-only, optional
        Evaluation mode.

        .. container::

           **Valid values**:

           * :code:`None`: The static structure factor is computed.
           * :code:`"pair"`: The partial structure factor is computed
             between the groups in `groups`.
           * :code:`"partial"`: The partial structure factors for all
             unique pairs in `groups` is computed.

    form : `str`, keyword-only, default: :code:`"exp"`
        Expression used to evaluate the structure factors.

        .. container::

           **Valid values**:

           * :code:`"exp"`: Exponential form. Slightly faster due to
             fewer mathematical operations.
           * :code:`"trig"`: Trigonometric form. Slightly slower but
             doesn't have overflow issues.

    dimensions : array-like, `openmm.unit.Quantity`, or \
    `pint.Quantity`, keyword-only, optional
        System dimensions :math:`(L_x,\\,L_y,\\,L_z)`. Only used to
        determine the scattering wavevectors when `wavevectors` is not
        specified.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{Å}`.

    n_points : `int`, keyword-only, default: :code:`32`
        Number of points :math:`n_\mathrm{points}` in the scattering
        wavevector grid. Additional wavevectors can be introduced via
        `n_surfaces` and `n_surface_points` for more accurate structure
        factors at small wavenumbers. Alternatively, the desired
        wavevectors can be specified directly in `wavevectors`.

    n_surfaces : `int`, keyword-only, optional
        Number of spherical surfaces in the first octant that intersect
        with the grid wavevectors along the three coordinate axes for
        which to introduce extra wavevectors for more accurate structure
        factor values. Only available if the system is perfectly cubic.

    n_surface_points : `int`, keyword-only, default: :code:`8`
        Number of extra wavevectors to introduce per spherical surface.
        Has no effect if `n_surfaces` is not specified.

    q_max : `float`, `openmm.unit.Quantity`, or `pint.Quantity`, \
    keyword-only, optional
        Maximum wavenumber :math:`q_\\mathrm{max}`.

        **Reference unit**: :math:`\\mathrm{Å}^{-1}`.

    wavevectors : array-like, `openmm.unit.Quantity`, or `pint.Quantity`, \
    keyword-only, optional
        Scattering wavevectors

        .. math::

           \\mathbf{q}=2\\pi\\left(\\frac{a}{L_x},\\,\\frac{b}{L_y},\\,
           \\frac{c}{L_z}\\right)

        for which to compute structure factors. :math:`a`, :math:`b`, and
        :math:`c` are integers from :math:`0` up to
        :math:`n_\\mathrm{points}-1`. Has precedence over `n_points`,
        `n_surfaces`, and `n_surface_points` if specified.

        **Shape**: :math:`(N_q,\\,3)`.

        **Reference unit**: :math:`\\mathrm{Å}^{-1}`.

    sort : `bool`, keyword-only, default: :code:`True`
        Determines whether the results are sorted by the wavenumbers.

    unique : `bool`, keyword-only, default: :code:`True`
        Determines whether structure factors for the same wavenumber
        are grouped and averaged.

    parallel : `bool`, keyword-only, default: :code:`False`
        Determines whether the analysis is performed in parallel.

    verbose : `bool`, keyword-only, default: :code:`True`
        Determines whether detailed progress is shown.

    **kwargs
        Additional keyword arguments to pass to
        :class:`MDAnalysis.analysis.base.AnalysisBase`.

    Attributes
    ----------
    universe : `MDAnalysis.Universe`
        :class:`MDAnalysis.core.universe.Universe` object containing all
        information describing the system.

    results.units : `dict`
        Reference units for the results. For example, to get the
        reference units for :code:`results.wavenumbers`, call
        :code:`results.units["wavenumbers"]`.

    results.pairs : `tuple`
        All unique pairs of indices of the groups of atoms in `groups`.
        The ordering coincides with the column indices in
        `results.ssf`.

    results.wavenumbers : `numpy.ndarray`
        Wavenumbers :math:`q`.

        **Shape**: :math:`(N_q,)`.

        **Reference unit**: :math:`\\mathrm{Å}^{-1}`.

    results.ssf : `numpy.ndarray`
        Static structure factor :math:`S(q)` or partial structure
        factors :math:`S_{\\alpha\\beta}(q)`.

        **Shape**: :math:`(1,\\,N_q)` or
        :math:`(C(N_\\mathrm{g}+1,\\,2),\\,N_q)`.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            groupings: Union[str, tuple[str]] = "atoms", *,
            mode: str = None, form: str = "exp",
            dimensions: Union[np.ndarray[float], "unit.Quantity", Q_] = None,
            n_points: int = 32, n_surfaces: int = None,
            n_surface_points: int = 8,
            q_max: Union[float, "unit.Quantity", Q_] = None,
            wavevectors: np.ndarray[float] = None, sort: bool = True,
            unique: bool = True, parallel: bool = False, verbose: bool = True,
            **kwargs) -> None:

        self._groups = [groups] if isinstance(groups, mda.AtomGroup) else groups
        self.universe = self._groups[0].universe
        super().__init__(self.universe.trajectory, verbose, **kwargs)

        GROUPINGS = {"atoms", "residues"}
        self._n_groups = len(self._groups)
        if isinstance(groupings, str):
            if groupings not in GROUPINGS:
                emsg = (f"Invalid grouping '{groupings}'. Valid values: "
                        "'" + "', '".join(GROUPINGS) + "'.")
                raise ValueError(emsg)
            self._groupings = self._n_groups * [groupings]
        else:
            if self._n_groups != len(groupings):
                emsg = ("The shape of 'groupings' is incompatible with "
                        "that of 'groups'.")
                raise ValueError(emsg)
            for gr in groupings:
                if gr not in GROUPINGS:
                    emsg = (f"Invalid grouping '{gr}'. Valid values: "
                            "'" + "', '".join(GROUPINGS) + "'.")
                    raise ValueError(emsg)
            self._groupings = groupings

        self._mode = mode
        if self._mode == "pair" and not 1 <= len(self._groups) <= 2:
            emsg = ("There must be exactly one or two atom groups in "
                    "'groups' when mode='pair'.")
            raise ValueError(emsg)
        elif self._mode is None:
            if sum(g.n_atoms for g in self._groups) \
                    != self.universe.atoms.n_atoms:
                emsg = ("The atom groups in 'groups' must contain all "
                        "atoms in the simulation when 'mode=None'.")
                raise ValueError(emsg)

        if wavevectors is not None:
            self._wavevectors = wavevectors
        else:
            dimensions = strip_unit(
                dimensions or self.universe.dimensions[:3], "Å"
            )[0]
            if dimensions is None:
                emsg = ("System dimensions were not found, but are "
                        "required when 'wavevectors' is not specified.")
                raise ValueError(emsg)
            elif len(dimensions) != 3:
                raise ValueError("'dimensions' must have length 3.")
            dimensions = np.asarray(dimensions)

            if np.allclose(dimensions, dimensions[0]):
                grid = 2 * np.pi * np.arange(n_points) / dimensions[0]
                self._wavevectors = (np.stack(np.meshgrid(grid, grid, grid), -1)
                                    .reshape(-1, 3))
                if n_surfaces:
                    n_theta, n_phi = get_closest_factors(n_surface_points, 2,
                                                        reverse=True)
                    theta = np.linspace(np.pi / (2 * n_theta + 4),
                                        np.pi / 2 - np.pi / (2 * n_theta + 4),
                                        n_theta)
                    phi = np.linspace(np.pi / (2 * n_phi + 4),
                                    np.pi / 2 - np.pi / (2 * n_phi + 4),
                                    n_phi)
                    self._wavevectors = np.vstack((
                        self._wavevectors,
                        np.einsum(
                            "o,tpd->otpd",
                            grid[1:n_surfaces + 1],
                            np.stack(
                                (np.sin(theta) * np.cos(phi)[:, None],
                                np.sin(theta) * np.sin(phi)[:, None],
                                np.tile(np.cos(theta)[None, :], (n_phi, 1))),
                                axis=-1
                            )
                        ).reshape((n_surfaces * n_surface_points, 3))
                    ))
            else:
                self._wavevectors = np.stack(
                    np.meshgrid(*[2 * np.pi * np.arange(n_points) / L
                                for L in dimensions]),
                    axis=-1
                ).reshape(-1, 3)
        self._wavenumbers = np.linalg.norm(self._wavevectors, axis=1)

        if q_max is not None:
            q_max = strip_unit(q_max, "Å^-1")[0]
            keep = self._wavenumbers <= q_max
            self._wavevectors = self._wavevectors[keep]
            self._wavenumbers = self._wavenumbers[keep]

        self._Ns = np.fromiter(
            (getattr(a, f"n_{g}")
             for (a, g) in zip(self._groups, self._groupings)),
            dtype=int,
            count=self._n_groups
        )
        self._N = self._Ns.sum()
        self._slices = []
        _ = 0
        for N in self._Ns:
            self._slices.append(slice(_, _+ N))
            _ += N

        self._njit = lambda s: numba.njit(s, fastmath=True, parallel=parallel)
        self._delta_fourier_transform_sum = self._njit(
            "c16[:](f8[:,:],f8[:,:])"
        )(delta_fourier_transform_sum)
        self._ssf_trigonometric = self._njit(
            "f8[:](f8[:,:])"
        )(ssf_trigonometric)
        self._psf_trigonometric = self._njit(
            "f8[:](f8[:,:],f8[:,:])"
        )(psf_trigonometric)
        self._inner = (accelerated.numba_inner_parallel if parallel
                       else accelerated.numba_inner)

        self._form = form
        self._sort = sort
        self._unique = unique
        self._verbose = verbose

    def _prepare(self) -> None:

        # Determine all unique pairs
        self.results.pairs = (
            tuple(combinations_with_replacement(range(self._n_groups), 2))
            if self._mode == "partial"
            else ((0, self._n_groups - 1),) if self._mode == "pair"
            else ((None, None),)
        )

        # Create a persisting array to hold atom positions for a single
        # frame so that it doesn't have to be recreated every frame
        self._positions = np.empty((self._N, 3))

        # Preallocate arrays to store structure factors
        self.results.ssf = np.zeros((len(self.results.pairs),
                                     len(self._wavenumbers)))

        # Determine the unique wavenumbers, if desired
        self.results.wavenumbers = (np.unique(self._wavenumbers.round(11))
                                    if self._unique else self._wavenumbers)

        # Store reference units
        self.results.units = Hash({"wavenumbers": ureg.angstrom ** -1})

    def _single_frame(self) -> None:

        # Store positions or centers of mass in the current frame
        for g, gr, s in zip(self._groups, self._groupings, self._slices):
            self._positions[s] = (g.positions if gr == "atoms"
                                  else center_of_mass(g, gr))

        # Compute the structure factor by multiplying exp(iqr) by its
        # conjugates
        if self._form == "exp":
            if self._mode is None:
                rhos = self._delta_fourier_transform_sum(self._wavevectors,
                                                         self._positions)
                self.results.ssf += (rhos * rhos.conj()).real
            else:
                for i, (j, k) in enumerate(self.results.pairs):
                    rhos_j = self._delta_fourier_transform_sum(
                        self._wavevectors, self._positions[self._slices[j]]
                    )
                    if j == k:
                        self.results.ssf[i] += (rhos_j * rhos_j.conj()).real
                    else:
                        self.results.ssf[i] += 2 * (
                            rhos_j * self._delta_fourier_transform_sum(
                                self._wavevectors,
                                self._positions[self._slices[k]]
                            ).conj()
                        ).real

        # Compute the structure factor by summing cos(qr)^2 and sin(qr)^2
        elif self._form == "trig":
            if self._mode is None:
                self.results.ssf += self._ssf_trigonometric(
                    self._inner(self._wavevectors, self._positions)
                )
            else:
                for i, (j, k) in enumerate(self.results.pairs):
                    qrs_j = self._inner(self._wavevectors,
                                        self._positions[self._slices[j]])
                    if j == k:
                        self.results.ssf[i] += self._ssf_trigonometric(qrs_j)
                    else:
                        self.results.ssf[i] += self._psf_trigonometric(
                            qrs_j,
                            self._inner(self._wavevectors,
                                        self._positions[self._slices[k]])
                        )

    def _conclude(self) -> None:

        # Normalize the structure factors by the number of entities and
        # timesteps
        self.results.ssf /= self.n_frames * self._N

        # Combine values sharing the same wavenumber, if desired
        if self._unique:
            self.results.ssf = np.hstack(
                [self.results.ssf[:, np.isclose(q, self._wavenumbers)]
                 .mean(axis=1, keepdims=True)
                for q in self.results.wavenumbers]
            )

        # Sort the results by wavenumber, if desired
        if self._sort:
            order = np.argsort(self.results.wavenumbers)
            self.results.wavenumbers = self.results.wavenumbers[order]
            self.results.ssf = self.results.ssf[:, order]

        # Clean up memory by deleting arrays that will not be reused
        del self._positions

@numba.njit("f8(f8[:])", fastmath=True)
def numba_cosine_sum(x: np.ndarray[float]) -> float:

    r"""
    Serial Numba-accelerated sum of the cosines of the elements in a
    one-dimensional NumPy array :math:`\mathbf{x}`.

    .. math::

       \sum_{i=1}^N\cos(x_i)

    Parameters
    ----------
    x : `np.ndarray`
        Vector :math:`\mathbf{x}`.

        **Shape**: :math:`(N,)`.

    Returns
    -------
    s : `float`
        Sum of the cosines of the elements in the vector
        :math:`\mathbf{x}`.
    """

    s = 0
    for i in range(x.shape[0]):
        s += np.cos(x[i])
    return s

def cosine_column_sum(xs: np.ndarray[float]) -> np.ndarray[float]:

    r"""
    Evaluates the column-wise sum of the cosines of the elements in a
    two-dimensional NumPy array :math:`\mathbf{x}`.

    .. math::

       \sum_{i=1}^N\cos(x_{ij})

    Parameters
    ----------
    xs : `np.ndarray`
        Matrix :math:`\mathbf{x}`.

        **Shape**: :math:`(M,\,N)`.

    Returns
    -------
    s : `np.ndarray`
        Column-wise sums of the cosines of the elements in the matrix
        :math:`\mathbf{x}`.

        **Shape**: :math:`(M,)`.
    """

    s = np.empty(xs.shape[0])
    for i in numba.prange(xs.shape[0]):
        s[i] = numba_cosine_sum(xs[i])
    return s

def cosine_column_sum_inplace(
        xs: np.ndarray[float], s: np.ndarray[float]) -> None:

    r"""
    Evaluates the column-wise sum of the cosines of the elements in a
    two-dimensional NumPy array :math:`\mathbf{x}`.

    .. math::

       \sum_{i=1}^N\cos(x_{ij})

    Parameters
    ----------
    xs : `np.ndarray`
        Matrix :math:`\mathbf{x}`.

        **Shape**: :math:`(M,\,N)`.

    s : `np.ndarray`
        Array to hold the column-wise sums of the cosines of the
        elements in the matrix :math:`\mathbf{x}`.

        **Shape**: :math:`(N,)`.
    """

    assert s.shape[0] == xs.shape[0]
    for i in numba.prange(xs.shape[0]):
        s[i] = numba_cosine_sum(xs[i])

@numba.njit("f8(f8[:])", fastmath=True)
def numba_sine_sum(x: np.ndarray[float]) -> float:

    r"""
    Serial Numba-accelerated sum of the sines of the elements in a
    one-dimensional NumPy array :math:`\mathbf{x}`.

    .. math::

       \sum_{i=1}^N\sin(x_i)

    Parameters
    ----------
    x : `np.ndarray`
        Vector :math:`\mathbf{x}`.

        **Shape**: :math:`(N,)`.

    Returns
    -------
    s : `float`
        Sum of the sines of the elements in the vector
        :math:`\mathbf{x}`.
    """

    s = 0
    for i in range(x.shape[0]):
        s += np.sin(x[i])
    return s

def sine_column_sum(xs: np.ndarray[float]) -> None:

    r"""
    Evaluates the column-wise sum of the sines of the elements in a
    two-dimensional NumPy array :math:`\mathbf{x}`.

    .. math::

       \sum_{i=1}^N\sin(x_{ij})

    Parameters
    ----------
    xs : `np.ndarray`
        Matrix :math:`\mathbf{x}`.

        **Shape**: :math:`(M,\,N)`.

    Returns
    -------
    s : `np.ndarray`
        Column-wise sums of the sines of the elements in the matrix
        :math:`\mathbf{x}`.

        **Shape**: :math:`(M,)`.
    """

    s = np.empty(xs.shape[0])
    for i in numba.prange(xs.shape[0]):
        s[i] = numba_sine_sum(xs[i])
    return s

def sine_column_sum_inplace(
        xs: np.ndarray[float], s: np.ndarray[float]) -> None:

    r"""
    Evaluates the column-wise sum of the sines of the elements in a
    two-dimensional NumPy array :math:`\mathbf{x}`.

    .. math::

       \sum_{i=1}^N\sin(x_{ij})

    Parameters
    ----------
    xs : `np.ndarray`
        Matrix :math:`\mathbf{x}`.

        **Shape**: :math:`(M,\,N)`.

    s : `np.ndarray`
        Array to hold the column-wise sums of the sines of the elements
        in the matrix :math:`\mathbf{x}`.

        **Shape**: :math:`(M,)`.
    """

    assert s.shape[0] == xs.shape[0]
    for i in numba.prange(xs.shape[0]):
        s[i] = numba_sine_sum(xs[i])

class IntermediateScatteringFunction(StructureFactor):

    """
    Serial and parallel implementations to calculate the coherent and
    incoherent (or self) parts of the intermediate scattering function,
    :math:`F(q,\\,t)` and :math:`F_\\mathrm{s}(q,\\,t)`, respectively, and
    the partial intermediate scattering functions,
    :math:`F_{\\alpha\\beta}(q,\\,t)`, for species :math:`\\alpha` and
    :math:`\\beta`.

    The coherent intermediate scattering function is a measure of the time
    evolution of the structure factor and can be computed directly from
    a molecular dynamics simulation trajectory using

    .. math::

       F(q,\\,t)&=\\frac{1}{N}\\left\\langle\\sum_{j=1}^N\\sum_{k=1}^N
       \\exp{[-i\\mathbf{q}\\cdot(\\mathbf{r}_j(t_0+t)
       -\\mathbf{r}_k(t_0))]}\\right\\rangle\\\\
       &=\\frac{1}{N}\\left\\langle
       \\sum_{j=1}^N\\cos(\\mathbf{q}\\cdot\\mathbf{r}_j(t_0+t))
       \\sum_{j=1}^N\\cos(\\mathbf{q}\\cdot\\mathbf{r}_j(t_0))
       +\\sum_{j=1}^N\\sin(\\mathbf{q}\\cdot\\mathbf{r}_j(t_0+t))
       \\sum_{j=1}^N\\sin(\\mathbf{q}\\cdot\\mathbf{r}_j(t_0))
       \\right\\rangle

    where :math:`N` is the total number of particles or centers of mass,
    :math:`\\mathbf{q}` is the scattering wavevector, :math:`t_0` and
    :math:`t` are the initial and lag times, and
    :math:`\\mathbf{r}_i` is the position of the :math:`i`-th particle.

    For multicomponent systems, the equation above can be generalized to
    get the partial coherent intermediate scattering functions

    .. math::

       F_{\\alpha\\beta}(q,\\,t)&=\\frac{2-\\delta_{\\alpha\\beta}}{N}
       \\left\\langle\\sum_{j=1}^{N_\\alpha}\\sum_{k=1}^{N_\\beta}
       \\exp{[-i\\mathbf{q}\\cdot(\\mathbf{r}_j(t_0+t)
       -\\mathbf{r}_k(t_0))]}\\right\\rangle\\\\
       &=\\frac{1}{(1+\\delta_{\\alpha\\beta})N}\\left\\langle
       \\sum_{j=1}^{N_\\alpha}
       \\cos(\\mathbf{q}\\cdot\\mathbf{r}_j(t_0+t))
       \\sum_{k=1}^{N_\\beta}\\cos(\\mathbf{q}\\cdot\\mathbf{r}_k(t_0))
       +\\sum_{j=1}^{N_\\alpha}
       \\sin(\\mathbf{q}\\cdot\\mathbf{r}_j(t_0+t))
       \\sum_{k=1}^{N_\\beta}\\sin(\\mathbf{q}\\cdot\\mathbf{r}_k(t_0))
       +\\sum_{k=1}^{N_\\beta}
       \\cos(\\mathbf{q}\\cdot\\mathbf{r}_k(t_0+t))
       \\sum_{j=1}^{N_\\alpha}\\cos(\\mathbf{q}\\cdot\\mathbf{r}_j(t_0))
       +\\sum_{k=1}^{N_\\beta}
       \\sin(\\mathbf{q}\\cdot\\mathbf{r}_k(t_0+t))
       \\sum_{j=1}^{N_\\alpha}\\sin(\\mathbf{q}\\cdot\\mathbf{r}_j(t_0))
       \\right\\rangle

    where :math:`\\delta_{ij}` is the Kronecker delta, :math:`N_\\alpha`
    and :math:`N_\\beta` are the numbers of particles or centers of mass
    for species :math:`\\alpha` and :math:`\\beta`.

    The partial coherent intermediate scattering functions
    :math:`F_{\\alpha\\beta}(q,\\,t)` and the coherent intermediate
    scattering function :math:`F(q,\\,t)` are related via

    .. math::

       F(q,\\,t)=\\sum_{\\alpha=1}^{N_\\mathrm{g}}
       \\sum_{\\beta=\\alpha}^{N_\\mathrm{g}}F_{\\alpha\\beta}(q,\\,t)

    and are related to the static and partial structure factors via

    .. math::

         F(q,\\,0)=S(q),\\,F_{\\alpha\\beta}(q,\\,0)=S_{\\alpha\\beta}(q)

    The incoherent intermediate scattering function characterizes the
    mean relaxation time of a system, and its spatial fluctuations
    provide information about dynamic heterogeneities. It is defined as

    .. math::

       F_\\mathrm{s}(q,\\,t)&=\\frac{1}{N}\\left\\langle
       \\sum_{j=1}^N\\exp{[-i\\mathbf{q}\\cdot(\\mathbf{r}_j(t_0+t)
       -\\mathbf{r}_j(t_0))]}\\right\\rangle\\\\
       &=\\frac{1}{N}\\left\\langle\\sum_{j=1}^N\\cos[\\mathbf{q}
       \\cdot(\\mathbf{r}_j(t_0+t)-\\mathbf{r}_j(t_0))]
       -\\Re\\left(i\\sum_{j=1}^N\\sin[\\mathbf{q}\\cdot(
       \\mathbf{r}_j(t_0+t)-\\mathbf{r}_j(t_0))]\\right)\\right\\rangle

    Similarly, partial incoherent intermediate scattering functions
    :math:`F_{\\mathrm{s},\\,\\alpha}(q,\\,t)` can be defined for a
    single species :math:`\\alpha`:

    .. math::

       F_{\\mathrm{s},\\,\\alpha}(q,\\,t)&=\\frac{1}{N}
       \\left\\langle\\sum_{j=1}^{N_\\alpha}
       \\exp{[-i\\mathbf{q}\\cdot(\\mathbf{r}_j(t_0+t)
       -\\mathbf{r}_j(t_0))]}\\right\\rangle\\\\
       &=\\frac{1}{N}\\left\\langle\\sum_{j=1}^{N_\\alpha}
       \\cos[\\mathbf{q}\\cdot(\\mathbf{r}_j(t_0+t)-\\mathbf{r}_j(t_0))]
       -\\Re\\left(i\\sum_{j=1}^{N_\\alpha}\\sin[\\mathbf{q}\\cdot(
       \\mathbf{r}_j(t_0+t)-\\mathbf{r}_j(t_0))]\\right)\\right\\rangle

    and related to the incoherent intermediate scattering function via

    .. math::

       F_\\mathrm{s}(q,\\,t)=\\sum_{\\alpha=1}^{N_\\mathrm{g}}
       F_{\\mathrm{s},\\,\\alpha}(q,\\,t)

    .. note::

       The simulation must have been run with a constant timestep
       :math:`\\Delta t` and the frames to be analyzed must be evenly
       spaced and proceed forward in time.

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Groups of atoms that share the same grouping type. If
        :code:`mode=None`, all atoms in the universe must be assigned to
        a group. If :code:`mode="pair"`, there must be exactly one or
        two groups.

    groupings : `str` or array-like, default: :code:`"atoms"`
        Determines whether the centers of mass are used in lieu of
        individual atom positions. If `groupings` is a `str`, the same
        value is used for all `groups`.

        .. note::

           If the desired grouping is not :code:`"atoms"`,

           * the trajectory file should have segments (or chains)
             containing residues (or molecules) and residues containing
             atoms, and

           * residues and segments should be locally unwrapped at the
             simulation box edges, if not already, using
             :class:`MDAnalysis.transformations.wrap.unwrap`,
             :meth:`MDAnalysis.core.groups.AtomGroup.unwrap`, or
             :func:`MDAnalysis.lib.mdamath.make_whole`.

        .. container::

           **Valid values**:

           * :code:`"atoms"`: Atom positions (generally or for
             coarse-grained simulations).
           * :code:`"residues"`: Residues' centers of mass (for
             atomistic simulations).
           * :code:`"segments"`: Segments' centers of mass (for
             atomistic polymer simulations).

    mode : `str`, keyword-only, optional
        Evaluation mode.

        .. container::

           **Valid values**:

           * :code:`None`: The coherent intermediate scattering function
             is computed.
           * :code:`"pair"`: The partial coherent intermediate
             scattering function is computed between the groups in
             `groups`.
           * :code:`"partial"`: The partial coherent intermediate
             scattering functions for all unique pairs in `groups` is
             computed.

    form : `str`, keyword-only, default: :code:`"exp"`
        Expression used to evaluate the intermediate scattering
        functions.

        .. container::

           **Valid values**:

           * :code:`"exp"`: Exponential form. Slightly faster due to
             fewer mathematical operations.
           * :code:`"trig"`: Trigonometric form. Slightly slower but
             doesn't have overflow issues.

    dimensions : array-like, `openmm.unit.Quantity`, or \
    `pint.Quantity`, keyword-only, optional
        System dimensions :math:`(L_x,\\,L_y,\\,L_z)`. Only used to
        determine the scattering wavevectors when `wavevectors` is not
        specified.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{Å}`.

    dt : `float`, `openmm.unit.Quantity`, or `pint.Quantity`, \
    keyword-only, optional
        Time between frames :math:`\\Delta t`. While this is normally
        determined from the trajectory, the trajectory may not have the
        correct timestep information.

        **Reference unit**: :math:`\\mathrm{ps}`.

    n_points : `int`, keyword-only, default: :code:`32`
        Number of points :math:`n_\mathrm{points}` in the scattering
        wavevector grid. Additional wavevectors can be introduced via
        `n_surfaces` and `n_surface_points` for more accurate structure
        factors at small wavenumbers. Alternatively, the desired
        wavevectors can be specified directly in `wavevectors`.

    n_surfaces : `int`, keyword-only, optional
        Number of spherical surfaces in the first octant that intersect
        with the grid wavevectors along the three coordinate axes for
        which to introduce extra wavevectors for more accurate
        intermediate scattering function values. Only available if the
        system is perfectly cubic.

    n_surface_points : `int`, keyword-only, default: :code:`8`
        Number of extra wavevectors to introduce per spherical surface.
        Has no effect if `n_surfaces` is not specified.

    q_max : `float`, `openmm.unit.Quantity`, or `pint.Quantity`, \
    keyword-only, optional
        Maximum wavenumber :math:`q_\\mathrm{max}`.

        **Reference unit**: :math:`\\mathrm{Å}^{-1}`.

    wavevectors : array-like, `openmm.unit.Quantity`, or `pint.Quantity`, \
    keyword-only, optional
        Scattering wavevectors

        .. math::

           \\mathbf{q}=2\\pi\\left(\\frac{a}{L_x},\\,\\frac{b}{L_y},\\,
           \\frac{c}{L_z}\\right)

        for which to compute structure factors. :math:`a`, :math:`b`, and
        :math:`c` are integers from :math:`0` up to
        :math:`n_\\mathrm{points}-1`. Has precedence over `n_points`,
        `n_surfaces`, and `n_surface_points` if specified.

        **Shape**: :math:`(N_q,\\,3)`.

        **Reference unit**: :math:`\\mathrm{Å}^{-1}`.

    sort : `bool`, keyword-only, default: :code:`True`
        Determines whether the results are sorted by the wavenumbers.

    unique : `bool`, keyword-only, default: :code:`True`
        Determines whether intermediate scattering functions for the
        same wavenumber are grouped and averaged.

    n_lags : `int`, keyword-only, optional
        Number of time lags :math:`t` or "windows" for which to evaluate
        the intermediate scattering functions, including zero. If not
        specified, the number of frames in the trajectory is used.

    incoherent : `bool`, keyword-only, default: :code:`False`
        Determines whether the incoherent intermediate scattering
        function is computed.

    parallel : `bool`, keyword-only, default: :code:`False`
        Determines whether the analysis is performed in parallel.

    verbose : `bool`, keyword-only, default: :code:`True`
        Determines whether detailed progress is shown.

    **kwargs
        Additional keyword arguments to pass to
        :class:`MDAnalysis.analysis.base.AnalysisBase`.

    Attributes
    ----------
    universe : `MDAnalysis.Universe`
        :class:`MDAnalysis.core.universe.Universe` object containing all
        information describing the system.

    results.units : `dict`
        Reference units for the results. For example, to get the
        reference units for :code:`results.wavenumbers`, call
        :code:`results.units["wavenumbers"]`.

    results.pairs : `tuple`
        All unique pairs of indices of the groups of atoms in `groups`.
        The ordering coincides with the column indices in
        `results.cisf` and `results.iisf`.

    results.times : `numpy.ndarray`
        Time lags :math:`t`.

        **Shape**: :math:`(N_t,)`.

        **Reference unit**: :math:`\\mathrm{ps}`.

    results.wavenumbers : `numpy.ndarray`
        Wavenumbers :math:`q`.

        **Shape**: :math:`(N_q,)`.

        **Reference unit**: :math:`\\mathrm{Å}^{-1}`.

    results.cisf : `numpy.ndarray`
        Coherent intermediate scattering function :math:`F(q,\\,t)` or
        partial coherent intermediate scattering functions
        :math:`F_{\\alpha\\beta}(q,\\,t)`.

        **Shape**: :math:`(N_t,\\,1,\\,N_q)` or
        :math:`(N_t,\\,C(N_\\mathrm{g}+1,\\,2),\\,N_q)`.

    results.iisf : `numpy.ndarray`
        Incoherent intermediate scattering function
        :math:`F_\\mathrm{s}(q,\\,t)` or partial incoherent intermediate
        scattering functions :math:`F_{\\mathrm{s},\\,\\alpha}(q,\\,t)`.
        Only available if :code:`incoherent=True`.

        **Shape**: :math:`(N_t,\\,1,\\,N_q)` or
        :math:`(N_t,\\,N_\\mathrm{g},\\,N_q)`.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            groupings: Union[str, tuple[str]] = "atoms", *,
            mode: str = None, form: str = "exp",
            dimensions: Union[np.ndarray[float], "unit.Quantity", Q_] = None,
            dt: Union[float, "unit.Quantity", Q_] = None,
            n_points: int = 32, n_surfaces: int = None,
            n_surface_points: int = 8,
            q_max: Union[float, "unit.Quantity", Q_] = None,
            wavevectors: np.ndarray[float] = None, sort: bool = True,
            unique: bool = True, n_lags: int = None, incoherent: bool = False,
            parallel: bool = False, verbose: bool = True, **kwargs) -> None:

        super().__init__(
            groups, groupings, mode=mode, form=form, dimensions=dimensions,
            n_points=n_points, n_surfaces=n_surfaces,
            n_surface_points=n_surface_points, q_max=q_max,
            wavevectors=wavevectors, sort=sort, unique=unique,
            parallel=parallel, verbose=verbose, **kwargs
        )

        self._dt = strip_unit(dt or self._trajectory.dt, "ps")[0]

        self._cosine_column_sum = self._njit(
            "f8[:](f8[:,:])"
        )(cosine_column_sum)
        self._cosine_column_sum_inplace = self._njit(
            "void(f8[:,:],f8[:])"
        )(cosine_column_sum_inplace)
        self._sine_column_sum = self._njit("f8[:](f8[:,:])")(sine_column_sum)
        self._sine_column_sum_inplace = self._njit(
            "void(f8[:,:],f8[:])"
        )(sine_column_sum_inplace)

        self._n_lags = n_lags
        self._incoherent = incoherent

    def _prepare(self) -> None:

        # Update number of time lags now that the user has specified
        # the frames to analyze, if necessary
        self._n_lags = min(self._n_lags or np.inf, self.n_frames)

        # Ensure frames are evenly spaced and proceed forward in time
        if hasattr(self._sliced_trajectory, "frames"):
            df = np.diff(self._sliced_trajectory.frames)
            if df[0] <= 0 or not np.allclose(df, df[0]):
                emsg = ("The selected frames must be evenly spaced and "
                        "proceed forward in time.")
                raise ValueError(emsg)
            df = df[0]
        else:
            if (df := self.step) <= 0:
                raise ValueError("The analysis must proceed forward in time.")

        # Determine all unique pairs
        self.results.pairs = (
            tuple(combinations_with_replacement(range(self._n_groups), 2))
            if self._mode == "partial"
            else ((0, self._n_groups - 1),) if self._mode == "pair"
            else ((None, None),)
        )

        # Preallocate arrays to store results
        self._positions = np.zeros((self._n_lags, self._N, 3))
        shape = (self._n_lags,
                 1 if self._mode is None else self._n_groups,
                 len(self._wavenumbers))
        if self._form == "exp":
            self._exp_sum = np.empty(shape, dtype=complex)
        elif self._form == "trig":
            self._cos_sum = np.empty(shape)
            self._sin_sum = np.empty(shape)
        self.results.cisf = np.zeros((
            self._n_lags,
            1 if self._mode is None else len(self.results.pairs),
            len(self._wavenumbers)
        ))
        if self._incoherent:
            self.results.iisf = np.zeros((
                self._n_lags,
                1 if self._mode is None else self._n_groups,
                len(self._wavenumbers)
            ))
        self.results.times = df * self._dt * np.arange(self._n_lags)

        # Determine the unique wavenumbers, if desired
        if self._unique:
            self.results.wavenumbers = np.unique(self._wavenumbers.round(11))
        else:
            self.results.wavenumbers = self._wavenumbers

        # Store reference units
        self.results.units = Hash({"times": ureg.picosecond,
                                   "wavenumbers": ureg.angstrom ** -1})

    def _single_frame(self) -> None:

        # Relative current frame index
        rcfi = self._frame_index % self._n_lags

        # Store atom or center-of-mass positions in the current frame
        for g, gr, s in zip(self._groups, self._groupings, self._slices):
            self._positions[rcfi, s] = (
                g.positions if gr == "atoms" else center_of_mass(g, gr)
            )

        # Note: Although the intermediate scattering functions can be
        # calculated using correlation functions more efficiently than
        # using sliding windows, the memory requirement to store the
        # exp(iqr) or cos(qr) and sin(qr) terms for all frames,
        # wavevectors, and positions is prohibitive for most consumer
        # machines. (For example, a 10-frame, 32,768-wavevector,
        # 10,000-particle array requires over 52 GB of RAM.)

        # Calculate intermediate scattering functions using exponential
        # form
        if self._form == "exp":
            if self._mode is None:
                self._exp_sum[rcfi] = self._delta_fourier_transform_sum(
                    self._wavevectors, self._positions[rcfi]
                )
                for time_lag in range(min(self._n_lags,
                                          self._frame_index + 1)):
                    rifi = (self._frame_index - time_lag) % self._n_lags
                    self.results.cisf[time_lag] += (
                        self._exp_sum[rifi] * self._exp_sum[rcfi].conj()
                    ).real
                    if self._incoherent:
                        self.results.iisf[time_lag] \
                            += self._delta_fourier_transform_sum(
                                self._wavevectors,
                                self._positions[rcfi] - self._positions[rifi]
                            ).real
            else:
                for i in range(self._n_groups):
                    self._exp_sum[rcfi, i] = self._delta_fourier_transform_sum(
                        self._wavevectors,
                        self._positions[rcfi, self._slices[i]]
                    )
                for time_lag in range(min(self._n_lags,
                                          self._frame_index + 1)):
                    rifi = (self._frame_index - time_lag) % self._n_lags
                    for i, (j, k) in enumerate(self.results.pairs):
                        if j == k:
                            self.results.cisf[time_lag, i] += (
                                self._exp_sum[rifi, j]
                                * self._exp_sum[rcfi, j].conj()
                            ).real
                            if self._incoherent:
                                self.results.iisf[time_lag, j] \
                                    += self._delta_fourier_transform_sum(
                                        self._wavevectors,
                                        self._positions[rcfi, self._slices[j]]
                                        - self._positions[rifi, self._slices[j]]
                                    ).real
                        else:
                            self.results.cisf[time_lag, i] += (
                                (self._exp_sum[rifi, j]
                                 * self._exp_sum[rcfi, k].conj()).real
                                + (self._exp_sum[rifi, k]
                                   * self._exp_sum[rcfi, j].conj()).real
                            )

        # Calculate intermediate scattering functions using
        # trigonometric form
        elif self._form == "trig":
            if self._mode is None:
                qrs = self._inner(self._wavevectors,
                                  self._positions[rcfi])
                self._cosine_column_sum_inplace(qrs, self._cos_sum[rcfi, 0])
                self._sine_column_sum_inplace(qrs, self._sin_sum[rcfi, 0])
                for time_lag in range(min(self._n_lags,
                                          self._frame_index + 1)):
                    rifi = (self._frame_index - time_lag) % self._n_lags
                    self.results.cisf[time_lag] += (
                        self._cos_sum[rifi] * self._cos_sum[rcfi]
                        + self._sin_sum[rifi] * self._sin_sum[rcfi]
                    )
                    if self._incoherent:
                        qrs = self._inner(self._wavevectors,
                                          self._positions[rcfi]
                                          - self._positions[rifi])
                        self.results.iisf[time_lag] += (
                            self._cosine_column_sum(qrs)
                            - 1j * self._sine_column_sum(qrs)
                        ).real
            else:
                for i in range(self._n_groups):
                    qrs = self._inner(self._wavevectors,
                                      self._positions[rcfi, self._slices[i]])
                    self._cosine_column_sum_inplace(qrs, self._cos_sum[rcfi, i])
                    self._sine_column_sum_inplace(qrs, self._sin_sum[rcfi, i])
                for time_lag in range(min(self._n_lags,
                                          self._frame_index + 1)):
                    rifi = (self._frame_index - time_lag) % self._n_lags
                    for i, (j, k) in enumerate(self.results.pairs):
                        if j == k:
                            self.results.cisf[time_lag, i] += (
                                self._cos_sum[rifi, j] * self._cos_sum[rcfi, j]
                                + self._sin_sum[rifi, j]
                                  * self._sin_sum[rcfi, j]
                            )
                            if self._incoherent:
                                qrs = self._inner(
                                    self._wavevectors,
                                    self._positions[rcfi, self._slices[j]]
                                    - self._positions[rifi, self._slices[j]]
                                )
                                self.results.iisf[time_lag, j] += (
                                    self._cosine_column_sum(qrs)
                                    - 1j * self._sine_column_sum(qrs)
                                ).real
                        else:
                            self.results.cisf[time_lag, i] += (
                                self._cos_sum[rifi, j] * self._cos_sum[rcfi, k]
                                + self._sin_sum[rifi, j]
                                  * self._sin_sum[rcfi, k]
                                + self._cos_sum[rifi, k]
                                  * self._cos_sum[rcfi, j]
                                + self._sin_sum[rifi, k]
                                  * self._sin_sum[rcfi, j]
                            )

    def _conclude(self) -> None:

        # Normalize the intermediate scattering functions by the number
        # of particles and timesteps
        norm = (
            self._N
            * np.arange(self.n_frames,
                        self.n_frames - self._n_lags, -1)[:, None, None]
        )
        self.results.cisf /= norm
        if self._incoherent:
            self.results.iisf /= norm

        # Combine values sharing the same wavenumber, if desired
        if self._unique:
            self.results.cisf = np.stack(
                [self.results.cisf[:, :, np.isclose(q, self._wavenumbers)]
                 .mean(axis=2) for q in self.results.wavenumbers],
                axis=-1
            )
            if self._incoherent:
                self.results.iisf = np.stack(
                    [self.results.iisf[:, :, np.isclose(q, self._wavenumbers)]
                     .mean(axis=2) for q in self.results.wavenumbers],
                    axis=-1
                )

        # Sort the results by wavenumber, if desired
        if self._sort:
            order = np.argsort(self.results.wavenumbers)
            self.results.wavenumbers = self.results.wavenumbers[order]
            self.results.cisf = self.results.cisf[:, :, order]
            if self._incoherent:
                self.results.iisf = self.results.iisf[:, :, order]

        # Clean up memory by deleting arrays that will not be reused
        del self._positions
        if self._form == "exp":
            del self._exp_sum
        elif self._form == "trig":
            del self._cos_sum, self._sin_sum