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
from MDAnalysis.lib import distances
import numba
import numpy as np
from scipy.integrate import simpson
from scipy.signal import argrelextrema
from scipy.special import jv

from .base import DynamicAnalysisBase, NumbaAnalysisBase
from .. import FOUND_OPENMM, ureg, Q_
from ..algorithm import accelerated
from ..algorithm.molecule import center_of_mass
from ..algorithm.unit import strip_unit
from ..algorithm.utility import get_closest_factors

if FOUND_OPENMM:
    from openmm import unit

def radial_histogram(
        pos1: np.ndarray[float], pos2: np.ndarray[float], n_bins: int,
        range: tuple[float], dims: tuple[float], *,
        exclusion: tuple[int] = None) -> np.ndarray[float]:

    r"""
    Computes the radial histogram of distances between particles of
    the same type or two different types.

    Parameters
    ----------
    pos1 : `numpy.ndarray`
        :math:`N_1` positions or center of masses of particles in the
        first group.

        **Shape**: :math:`(N_1,\,3)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    pos2 : `numpy.ndarray`
        :math:`N_2` positions or center of masses of particles in the
        second group.

        **Shape**: :math:`(N_2,\,3)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    n_bins : `int`
        Number of histogram bins :math:`N_\mathrm{bins}`.

    range : array-like
        Range of radii values.

        **Shape**: :math:`(2,)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    dims : array-like
        System dimensions and orthogonality.

        **Shape**: :math:`(6,)`.

        **Reference unit**: :math:`\mathrm{Å}` (dimensions),
        :math:`^\circ` (orthogonality).

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
    pairs, dist = distances.capped_distance(
        pos1, pos2, range[1], range[0] - np.finfo(np.float64).eps,
        box=dims
    )

    # Exclude atom pairs with the same atoms or atoms from the
    # same residue
    if exclusion is not None:
        dist = dist[np.where(pairs[:, 0] // exclusion[0]
                             != pairs[:, 1] // exclusion[1])[0]]

    return np.histogram(dist, bins=n_bins, range=range)[0]

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
    distribution function :math:`g_{ij}(r)`.

    It is defined as

    .. math::

       n_k=4\pi\rho_j\int_{r_{k-1}}^{r_k}r^2g_{ij}(r)\,dr

    for three-dimensional systems and

    .. math::

       n_k=2\pi\rho_j\int_{r_{k-1}}^{r_k}rg_{ij}(r)\,dr

    for two-dimensional systems, where :math:`k` is the index,
    :math:`\rho_j` is the bulk number density of species :math:`j` and
    :math:`r_k` is the :math:`(k + 1)`-th local minimum of
    :math:`g_{ij}(r)`.

    If the radial distribution function :math:`g_{ij}(r)` does not
    contain as many local minima as `n_coord_nums`, this method will
    return `numpy.nan` for the coordination numbers that could not be
    calculated.

    Parameters
    ----------
    bins : `numpy.ndarray`
        Centers of the histogram bins.

        **Shape**: :math:`(N_\mathrm{bins},)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    rdf : `numpy.ndarray`
        Radial distribution function :math:`g_{ij}(r)`.

        **Shape**: :math:`(N_\mathrm{bins},)`.

    rho : `float`
        Number density :math:`\rho_j` of species :math:`j`.

        **Reference unit**: :math:`\mathrm{Å}^D`, where :math:`D` is the
        number of dimensions.

    n_coord_nums : `int`, keyword-only, default: :code:`2`
        Number of coordination numbers to calculate.

    n_dims : `int`, keyword-only, default: :code:`3`
        Number of dimensions :math:`D`.

    threshold : `float`, keyword-only, default: :code:`0.1`
        Minimum :math:`g_{ij}(r)` value that must be reached before
        local minima are found.

    Returns
    -------
    coord_nums : `numpy.ndarray`
        Coordination numbers :math:`n_k`.
    """

    if n_dims not in {2, 3}:
        raise ValueError("Invalid number of dimensions.")

    def f(r, rdf, rho, start, stop):
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
        x_i: float = 1, x_j: float = None, q: np.ndarray[float] = None, *,
        q_lower: float = None, q_upper: float = None, n_q: int = 1_000,
        n_dims: int = 3, formalism: str = "FZ"
    ) -> tuple[np.ndarray[float], np.ndarray[float]]:

    r"""
    Calculates the (partial) static structure factor :math:`S_{ij}(q)`
    using the radial histogram bins :math:`r` and the radial
    distribution function :math:`g_{ij}(r)` for an isotropic fluid.

    Parameters
    ----------
    r : `numpy.ndarray`
        Radii :math:`r`.

        **Reference unit**: :math:`\mathrm{Å}`.

    g : `numpy.ndarray`
        Radial distribution function :math:`g_{ij}(r)`.

        **Shape**: Same as `r`.

    equal : `bool`
        Specifies whether `g` is between the same species, or
        :math:`i = j`. If :code:`False`, the number concentrations of
        species :math:`i` and :math:`j` must be specified in `x_i` and
        `x_j`.

    rho : `float`
        Bulk number density :math:`\rho` or surface density
        :math:`\sigma`.

        **Reference unit**: :math:`\mathrm{Å}^{-3}` or
        :math:`\mathrm{Å}^{-2}`.

    x_i : `float`, default: :code:`1`
        Number concentration of species :math:`i`. Required if
        :code:`equal=False`.

    x_j : `float`, optional
        Number concentration of species :math:`j`. Required if
        :code:`equal=False`.

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

    n_dims : `int`, keyword-only, default: :code:`3`
        Number of dimensions :math:`D`.

    formalism : `str`, keyword-only, default: :code:`"FZ"`
        Formalism to use for the partial structure factor. Has no effect
        if :code:`equal=True`.

        .. container::

           **Valid values**:

           * :code:`"general"`: A general formalism given by

             .. math::

                S_{ij}(q)=1+x_ix_j\frac{4\pi\rho}{q}\int_0^\infty
                (g_{ij}(r)-1)\sin{(qr)}r\,dr

           * :code:`"FZ"`: Faber–Ziman formalism [1]_

             .. math::

                S_{ij}(q)=1+\frac{4\pi\rho}{q}\int_0^\infty
                (g_{ij}(r)-1)\sin{(qr)}r\,dr

           * :code:`"AL"`: Ashcroft–Langreth formalism [2]_

             .. math::

                S_{ij}(q)=\delta_{ij}+(x_ix_j)^{1/2}\frac{4\pi\rho}{q}
                \int_0^\infty (g_{ij}(r)-1)\sin{(qr)}r\,dr

           In two-dimensional systems, the second term is

           .. math::

              2\pi\rho\int_0^\infty (g_{ij}(r)-1)J_0(qr)r\,dr

           instead, where :math:`J_0` is the zeroth-order Bessel
           function.

    Returns
    -------
    q : `numpy.ndarray`
        Wavenumbers :math:`q`.

        **Shape**: :math:`(N_q,)`.

    S : `numpy.ndarray`
        (Partial) static structure factor :math:`S(q)`.

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
            return q, (x_i == x_j) + np.sqrt(x_i * x_j) * rho_sft
        elif formalism == "general":
            return q, 1 + x_i * x_j * rho_sft
    raise ValueError("Invalid formalism.")

class RadialDistributionFunction(DynamicAnalysisBase):

    r"""
    Serial and parallel implementations to calculate the radial
    distribution function (RDF) :math:`g_{ij}(r)` between types
    :math:`i` and :math:`j` and its related properties for two-
    and three-dimensional systems.

    The RDF is given by

    .. math::

       g_{ij}^\mathrm{3D}(r)=\frac{V}{4\pi r^2N_iN_j}
       \sum_{\alpha=1}^{N_i}\sum_{\beta=1}^{N_j}\left\langle
       \delta\left(|\mathbf{r}_\alpha-\mathbf{r}_\beta|-r\right)
       \right\rangle\\
       g_{ij}^\mathrm{2D}(r)=\frac{A}{2\pi rN_iN_j}
       \sum_{\alpha=1}^{N_i}\sum_{\beta=1}^{N_j}\left\langle
       \delta\left(|\mathbf{r}_\alpha-\mathbf{r}_\beta|-r\right)
       \right\rangle

    where :math:`V` and :math:`A` are the system volume and area,
    :math:`N_i` and :math:`N_j` are the number of particles, and
    :math:`\mathbf{r}_\alpha` and :math:`\mathbf{r}_\beta` are the
    positions of particles :math:`\alpha` and :math:`\beta` belonging
    to species :math:`i` and :math:`j`, respectively. The RDF is
    normalized such that :math:`\lim_{r\rightarrow\infty}g_{ij}(r)=1` in
    a homogeneous system.

    (A closely related quantity is the single particle density
    :math:`n_{ij}(r)=\rho_jg_{ij}(r)`, where :math:`\rho_j` is the
    number density of species :math:`j`.)

    The cumulative RDF is

    .. math::

       G_{ij}^\mathrm{3D}(r)=4\pi\int_0^rR^2g_{ij}(R)\,dR\\
       G_{ij}^\mathrm{2D}(r)=2\pi\int_0^rRg_{ij}(R)\,dR

    and the average number of :math:`j` particles found within radius
    :math:`r` is

    .. math::

       N_{ij}(r)=\rho_jG_{ij}(r)

    The expression above can be used to compute the coordination numbers
    (number of neighbors in each solvation shell) by setting :math:`r`
    to the :math:`r`-values where :math:`g_{ij}(r)` is locally
    minimized, which signify the solvation shell boundaries.

    .. container::

       The RDF can also be used to obtain other relevant structural
       properties, such as

       * the potential of mean force

         .. math::

            w_{ij}(r)=-k_\mathrm{B}T\ln{g_{ij}(r)}

         where :math:`k_\mathrm{B}` is the Boltzmann constant and
         :math:`T` is the system temperature, and

       * the (partial) static structure factor (see
         :func:`calculate_structure_factor` for the possible
         definitions).

    Parameters
    ----------
    ag1 : `MDAnalysis.AtomGroup`
        First atom group :math:`i`.

    ag2 : `MDAnalysis.AtomGroup`
        Second atom group :math:`j`.

    n_bins : `int`, default: :code:`201`
        Number of histogram bins :math:`N_\mathrm{bins}`.

    range : array-like, default: :code:`(0.0, 15.0)`
        Range of radii values. The upper bound should be less than half
        the largest system dimension.

        **Shape**: :math:`(2,)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    drop_axis : `int` or `str`, keyword-only, default: :code:`2`
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
             :math:`g_{ij}(r)` is computed.
           * :code:`norm="density"`: The single particle density
             :math:`n_{ij}(r)` is computed.
           * :code:`norm=None`: The raw particle pair count in the
             radial histogram bins is returned.

    exclusion : array-like, keyword-only, optional
        Tiles to exclude from the interparticle distances. The
        `groupings` parameter dictates what a tile represents.

        **Shape**: :math:`(2,)`.

        **Example**: :code:`(1, 1)` to exclude self-interactions.

    groupings : `str` or array-like, keyword-only, default: :code:`"atoms"`
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

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units.

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
        reference units for :code:`results.bins`, call
        :code:`results.units["results.bins"]`.

    results.edges : `numpy.ndarray`
        Edges of the histogram bins.

        **Shape**: :math:`(N_\mathrm{bins}+1,)`.

        **Reference unit**: :math:`\textrm{Å}`.

    results.bins : `numpy.ndarray`
        Centers of the histogram bins.

        **Shape**: :math:`(N_\mathrm{bins},)`.

        **Reference unit**: :math:`\textrm{Å}`.

    results.counts : `numpy.ndarray`
        Raw particle pair counts in the radial histogram bins.

        **Shape**: :math:`(N_\mathrm{bins},)`.

    results.rdf : `numpy.ndarray`
        .. container::

           One of

           * :code:`norm="rdf"`: the radial distribution function
             :math:`g_{ij}(r)`,
           * :code:`norm="density"`: the single particle density
             :math:`n_{ij}(r)`, or
           * :code:`norm=None`: the raw particle pair count in the
             radial histogram bins.

        **Shape**: :math:`(N_\mathrm{bins},)`.

    results.coordination_numbers : `numpy.ndarray`
        Coordination numbers :math:`n_k`. Only available after running
        :meth:`calculate_coordination_numbers`.

    results.pmf : `numpy.ndarray`
        Potential of mean force :math:`w(r)`. Only available after
        running :meth:`calculate_pmf`.

        **Shape**: :math:`(N_\mathrm{bins},)`.

        **Reference unit**: :math:`\mathrm{kJ/mol}`.

    results.wavenumbers : `numpy.ndarray`
        Wavenumbers :math:`q`. Only available after running
        :meth:`calculate_structure_factor`.

        **Reference unit**: :math:`\textrm{Å}^{-1}`.

    results.ssf : `numpy.ndarray`
        (Partial) static structure factor. Only available after running
        :meth:`calculate_structure_factor`.

        **Shape**: Same as `results.wavenumbers`.
    """

    def __init__(
            self, ag1: mda.AtomGroup, ag2: mda.AtomGroup = None,
            n_bins: int = 201, range: tuple[float] = (0.0, 15.0), *,
            drop_axis: Union[int, str] = None, norm: str = "rdf",
            exclusion: tuple[int] = None,
            groupings: Union[str, tuple[str]] = "atoms",
            reduced: bool = False, n_batches: int = None,
            parallel: bool = False, verbose: bool = True, **kwargs) -> None:

        self.ag1 = ag1
        self.ag2 = ag1 if ag2 is None else ag2
        self.universe = self.ag1.universe
        if self.universe.dimensions is None and self._ts.dimensions is None:
            raise ValueError("Trajectory does not contain system "
                             "dimension information.")

        super().__init__(self.universe.trajectory, parallel, verbose, **kwargs)

        if isinstance(groupings, str):
            if groupings not in {"atoms", "residues", "segments"}:
                emsg = (f"Invalid grouping '{groupings}'. The options are "
                        "'atoms', 'residues', and 'segments'.")
                raise ValueError(emsg)
            self._groupings = 2 * [groupings]
        else:
            for g in groupings:
                if g not in {"atoms", "residues", "segments"}:
                    emsg = (f"Invalid grouping '{g}'. The options are "
                            "'atoms', 'residues', and 'segments'.")
                    raise ValueError(emsg)
            self._groupings = (2 * groupings if len(groupings) == 1
                               else groupings)

        self._drop_axis = (ord(drop_axis) - 120 if isinstance(drop_axis, str)
                           else drop_axis)
        if self._drop_axis not in {0, 1, 2, None}:
            raise ValueError("Invalid axis to drop.")

        self._n_bins = n_bins
        self._range = range
        self._norm = norm
        self._exclusion = exclusion
        self._reduced = reduced
        self._n_batches = n_batches
        self._verbose = verbose

    def _prepare(self) -> None:

        # Preallocate arrays to store results
        self.results.edges = np.linspace(*self._range, self._n_bins + 1)
        self.results.bins = (self.results.edges[:-1]
                             + self.results.edges[1:]) / 2
        self.results.counts = np.zeros(self._n_bins, dtype=int)
        self.results.units = {"results.bins": ureg.angstrom,
                              "results.edges": ureg.angstrom}

        # Preallocate floating-point number for total volume (or area)
        # analyzed (for when system dimensions can change, such as
        # during NpT equilibration)
        if not self._parallel and self._norm == "rdf":
            self._area_or_volume = 0.0

    def _single_frame(self) -> None:

        dims = self._ts.dimensions
        pos1 = (self.ag1.positions if self._groupings[0] == "atoms"
                else center_of_mass(self.ag1, self._groupings[0]))
        pos2 = (self.ag2.positions if self._groupings[1] == "atoms"
                else center_of_mass(self.ag2, self._groupings[1]))

        if self._drop_axis is None:
            if self._norm == "rdf":
                self._area_or_volume += self._ts.volume
        else:

            # Apply corrections to avoid including periodic images in
            # the dimension to exclude
            pos1[:, self._drop_axis] = pos2[:, self._drop_axis] = 0
            dims[self._drop_axis] = dims[:3].max()

            if self._norm == "rdf":
                self._area_or_volume += np.delete(dims[:3],
                                                  self._drop_axis).prod()

        # Tally counts in each pair separation distance bin
        if self._n_batches:
            edges = np.array_split(self.results.edges, self._n_batches)
            ranges_indices = {
                e: np.where((self.results.bins > e[0])
                            & (self.results.bins < e[1]))[0]
                for e in [(self._range[0], edges[0][-1]),
                          *((a[-1], b[-1])
                            for a, b in zip(edges[:-1], edges[1:]))]
            }
            for r, i in ranges_indices.items():
                self.results.counts[i] += radial_histogram(
                    pos1=pos1, pos2=pos2, n_bins=i.shape[0], range=r,
                    dims=dims, exclusion=self._exclusion
                )
        else:
            self.results.counts += radial_histogram(
                pos1=pos1, pos2=pos2, n_bins=self._n_bins, range=self._range,
                dims=dims, exclusion=self._exclusion
            )

    def _single_frame_parallel(
            self, frame: int, index: int) -> np.ndarray[float]:

        _ts = self._trajectory[frame]
        result = np.empty(1 + self._n_bins)

        dims = _ts.dimensions
        pos1 = (self.ag1.positions if self._groupings[0] == "atoms"
                else center_of_mass(self.ag1, self._groupings[0]))
        pos2 = (self.ag2.positions if self._groupings[1] == "atoms"
                else center_of_mass(self.ag2, self._groupings[1]))

        # Apply corrections to avoid including periodic images in the
        # dimension to exclude
        if self._drop_axis is None:
            result[self._n_bins] = _ts.volume
        else:
            pos1[:, self._drop_axis] = pos2[:, self._drop_axis] = 0
            dims[self._drop_axis] = dims[:3].max()
            result[self._n_bins] = np.delete(dims[:3], self._drop_axis).prod()

        # Compute radial histogram for a single frame
        if self._n_batches:
            edges = np.array_split(self.results.edges, self._n_batches)
            ranges_indices = {
                e: np.where((self.results.bins > e[0])
                            & (self.results.bins < e[1]))[0]
                for e in [(self._range[0], edges[0][-1]),
                          *((a[-1], b[-1])
                            for a, b in zip(edges[:-1], edges[1:]))]
            }
            for r, i in ranges_indices.items():
                result[i] = radial_histogram(
                    pos1=pos1, pos2=pos2, n_bins=i.shape[0], range=r,
                    dims=_ts.dimensions, exclusion=self._exclusion
                )
        else:
            result[:self._n_bins] = radial_histogram(
                pos1=pos1, pos2=pos2, n_bins=self._n_bins, range=self._range,
                dims=dims, exclusion=self._exclusion
            )

        return result

    def _conclude(self):

        # Tally counts in each pair separation distance bin over all
        # frames
        if self._parallel:
            self._results = np.vstack(self._results).sum(axis=0)
            self.results.counts[:] = self._results[:self._n_bins]
            self._area_or_volume = self._results[self._n_bins]

        # Compute the normalization factor
        norm = self.n_frames
        if self._norm is not None:
            if self._drop_axis is None:
                norm *= 4 * np.pi * np.diff(self.results.edges ** 3) / 3
            else:
                norm *= np.pi * np.diff(self.results.edges ** 2)
            if self._norm == "rdf":
                _N2 = getattr(self.ag2, f"n_{self._groupings[1]}")
                if self._exclusion:
                    _N2 -= self._exclusion[1]
                norm *= (getattr(self.ag1, f"n_{self._groupings[0]}") * _N2
                         * self.n_frames / self._area_or_volume)

        # Compute and store the radial distribution function, the single
        # particle density, or the raw radial pair counts
        self.results.rdf = self.results.counts / norm

    def _get_rdf(self) -> np.ndarray[float]:

        """
        Returns the existing radial distribution function (RDF) if
        :code:`norm="rdf"` was passed to the :class:`RDF` constructor.
        Otherwise, the RDF is calculated and returned.

        Returns
        -------
        rdf : `numpy.ndarray`
            Radial distribution function :math:`g_{ij}(r)`.
        """

        if self._norm == "rdf":
            return self.results.rdf
        else:
            _N2 = getattr(self.ag2, f"n_{self._groupings[1]}")
            if self._exclusion:
                _N2 -= self._exclusion[1]

            if self._drop_axis is None:
                norm = 4 * np.diff(self.results.edges ** 3) / 3
            else:
                norm = np.diff(self.results.edges ** 2)
            return self._area_or_volume * self.results.counts / (
                np.pi * self.n_frames ** 2 * _N2 * norm
                * getattr(self.ag1, f"n_{self._groupings[0]}")
            )

    def calculate_coordination_numbers(
            self, rho: float, *, n_coord_nums: int = 2, threshold: float = 0.1
        ) -> None:

        r"""
        Calculates the coordination numbers :math:`n_k`.

        If the radial distribution function :math:`g_{ij}(r)` does not
        contain :math:`k` local minima, this method will return
        `numpy.nan` for the coordination numbers that could not be
        calculated.

        Parameters
        ----------
        rho : `float`
            Number density :math:`\rho_j` of species :math:`j`.

            **Reference unit**: :math:`\mathrm{nm}^{-3}`.

        n_coord_nums : `int`, keyword-only, default: :code:`2`
            Number of coordination numbers to calculate.

        threshold : `float`, keyword-only, default: :code:`0.1`
            Minimum :math:`g_{ij}(r)` value for a local minimum to be
            considered the boundary of a radial shell.
        """

        self.results.coordination_numbers = calculate_coordination_numbers(
            self.results.bins, self._get_rdf(), rho, n_coord_nums=n_coord_nums,
            n_dims=2 + (self._drop_axis is None), threshold=threshold
        )

    def calculate_pmf(
            self, temperature: Union[float, "unit.Quantity", Q_]) -> None:

        r"""
        Calculates the potential of mean force :math:`w_{ij}(r)`.

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

        self.results.units["results.pmf"] = ureg.kilojoule / ureg.mole

        temperature, unit_ = strip_unit(temperature, "kelvin")
        if self._reduced:
            if isinstance(unit_, str):
                emsg = "'temperature' cannot have units when reduced=True."
                raise ValueError(emsg)
            kBT = temperature
        else:
            kBT = (
                ureg.avogadro_constant * ureg.boltzmann_constant
                * temperature * ureg.kelvin
            ).m_as(self.results.units["results.pmf"])
        self.results.pmf = -kBT * np.log(self._get_rdf())

    def calculate_structure_factor(
            self, rho: float, x_i: float = None, x_j: float = None,
            q: np.ndarray[float] = None, *, q_lower: float = None,
            q_upper: float = None, n_q: int = 1_000, formalism: str = "FZ"
        ) -> None:

        r"""
        Computes the (partial) static structure factor :math:`S_{ij}(q)`
        using the radial histogram bins :math:`r` and the radial
        distribution function :math:`g_{ij}(r)` for an isotropic fluid.

        Parameters
        ----------
        rho : `float`
            Bulk number density :math:`\rho`.

            **Reference unit**: :math:`\mathrm{Å}^{-3}`.

        x_i : `float`, default: :code:`1`
            Number concentration of species :math:`i`. Required if
            the two atom groups are not identical.

        x_j : `float`, optional
            Number concentration of species :math:`j`. Required if
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
            self.results.bins, self._get_rdf(), self.ag1 == self.ag2,
            rho, x_i, x_j, q=q, q_lower=q_lower, q_upper=q_upper,
            n_q=n_q, n_dims=2 + (self._drop_axis is None), formalism=formalism
        )

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
        Group(s) of atoms that share the same grouping type. If
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
             between the group(s) in `groups`.
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
        System dimensions. If not provided, they are retrieved from the
        topology or trajectory. Only necessary if `wavevectors` is not
        specified.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{Å}`.

    n_points : `int`, keyword-only, default: :code:`32`
        Number of points in the scattering wavevector grid. Additional
        wavevectors can be introduced via `n_surfaces` and
        `n_surface_points` for more accurate structure factors at small
        wavenumbers. Alternatively, the desired wavevectors can be
        specified directly in `wavevectors`.

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
        Maximum scattering wavevector magnitude.

        **Reference unit**: :math:`\\mathrm{Å}^{-1}`.

    wavevectors : `numpy.ndarray`, keyword-only, optional
        Scattering wavevectors for which to compute structure
        factors. Has precedence over `n_points`, `n_surfaces`, and
        `n_surface_points` if specified.

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
        :code:`results.units["results.wavenumbers"]`.

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

    @staticmethod
    def ssf_trigonometric_2d(qrs: np.ndarray[float]) -> np.ndarray[float]:

        r"""
        Computes the static structure factors using a two-dimensional
        NumPy array containing :math:`\mathbf{q}\cdot\mathbf{r}` using
        the trigonometric form.

        .. math::

           S(q)=\frac{1}{N}\left\langle\left(\sum_{j=1}^N
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
            ssf[i] = accelerated.pythagorean_trigonometric_identity_1d(qrs[i])
        return ssf

    @staticmethod
    def psf_trigonometric_2d_2d(
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
            ssf[i] = accelerated.pythagorean_trigonometric_identity_1d_1d(
                qrs1[i], qrs2[i]
            )
        return ssf

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

        self._n_groups = len(self._groups)
        if isinstance(groupings, str):
            if groupings not in (GROUPINGS := {"atoms", "residues"}):
                emsg = (f"Invalid grouping '{groupings}'. Valid "
                        f"values: {', '.join(GROUPINGS)}.")
                raise ValueError(emsg)
            self._groupings = self._n_groups * [groupings]
        else:
            if self._n_groups != len(groupings):
                emsg = ("The number of grouping values is not equal to "
                        "the number of groups.")
                raise ValueError(emsg)
            for g in groupings:
                if g not in (GROUPINGS := {"atoms", "residues"}):
                    emsg = (f"Invalid grouping '{groupings}'. Valid "
                            f"values: {', '.join(GROUPINGS)}.")
                    raise ValueError(emsg)
            self._groupings = groupings

        self._mode = mode
        if self._mode == "pair" and not 1 <= len(self._groups) <= 2:
            emsg = "There must be exactly one or two groups when mode='pair'."
            raise ValueError(emsg)
        elif self._mode is None:
            if sum(g.n_atoms for g in self._groups) \
                    != self.universe.atoms.n_atoms:
                emsg = ("The provided atom groups do not contain all atoms "
                        "in the universe.")
                raise ValueError(emsg)

        if dimensions is not None:
            if len(dimensions) != 3:
                raise ValueError("'dimensions' must have length 3.")
            self._dimensions = np.asarray(strip_unit(dimensions, "angstrom")[0])
        elif self.universe.dimensions is not None:
            self._dimensions = self.universe.dimensions[:3].copy()
        elif wavevectors is None:
            raise ValueError("No system dimensions found or provided.")

        # Determine the wavevectors and their corresponding magnitudes
        if wavevectors is not None:
            self._wavevectors = wavevectors
        elif np.allclose(self._dimensions, self._dimensions[0]):
            grid = 2 * np.pi * np.arange(n_points) / self._dimensions[0]
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
                              for L in self._dimensions]),
                axis=-1
            ).reshape(-1, 3)
        self._wavenumbers = np.linalg.norm(self._wavevectors, axis=1)

        if q_max is not None:
            q_max, _ = strip_unit(q_max, "angstrom^-1")
            keep = self._wavenumbers <= q_max
            self._wavevectors = self._wavevectors[keep]
            self._wavenumbers = self._wavenumbers[keep]

        # Determine the number of particles in each group and their
        # corresponding indices
        self._Ns = np.fromiter(
            (getattr(a, f"n_{g}")
             for (a, g) in zip(self._groups, self._groupings)),
            dtype=int,
            count=self._n_groups
        )
        self._N = self._Ns.sum()
        self._slices = []
        index = 0
        for N in self._Ns:
            self._slices.append(slice(index, index + N))
            index += N

        # Define the functions to use depending on whether the user
        # wants parallelization
        self._njit = lambda s: numba.njit(s, fastmath=True, parallel=parallel)
        self._ssf_trigonometric = self._njit("f8[:](f8[:,:])")(
            self.ssf_trigonometric_2d
        )
        self._psf_trigonometric = self._njit("f8[:](f8[:,:],f8[:,:])")(
            self.psf_trigonometric_2d_2d
        )
        if parallel:
            self._delta_fourier_transform_sum \
                = accelerated.delta_fourier_transform_sum_parallel_2d_2d
            self._inner = accelerated.inner_parallel_2d_2d
        else:
            self._delta_fourier_transform_sum \
                = accelerated.delta_fourier_transform_sum_2d_2d
            self._inner = accelerated.inner_2d_2d

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

        # Preallocate arrays to store results
        self.results.ssf = np.zeros((len(self.results.pairs),
                                     len(self._wavenumbers)))

        # Determine the unique wavenumbers, if desired
        self.results.wavenumbers = (np.unique(self._wavenumbers.round(11))
                                    if self._unique else self._wavenumbers)

        # Store reference units
        self.results.units = {"results.wavenumbers": ureg.angstrom ** -1}

    def _single_frame(self) -> None:

        # Store atom or center-of-mass positions in the current frame
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

        # Normalize the structure factors by the number of particles and
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
        Group(s) of atoms that share the same grouping type. If
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
             scattering function is computed between the group(s) in
             `groups`.
           * :code:`"partial"`: The partial coherent intermediate
             scattering functions for all unique pairs in `groups` is
             computed.

    form : `str`, keyword-only, default: :code:`"exp"`
        Expression used to evaluate the intermediate scattering
        function(s).

        .. container::

           **Valid values**:

           * :code:`"exp"`: Exponential form. Slightly faster due to
             fewer mathematical operations.
           * :code:`"trig"`: Trigonometric form. Slightly slower but
             doesn't have overflow issues.

    dimensions : array-like, `openmm.unit.Quantity`, or \
    `pint.Quantity`, keyword-only, optional
        System dimensions. If not provided, they are retrieved from the
        topology or trajectory. Only necessary if `wavevectors` is not
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
        Number of points in the scattering wavevector grid. Additional
        wavevectors can be introduced via `n_surfaces` and
        `n_surface_points` for more accurate structure factors at small
        wavenumbers. Alternatively, the desired wavevectors can be
        specified directly in `wavevectors`.

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
        Maximum scattering wavevector magnitude.

        **Reference unit**: :math:`\\mathrm{Å}^{-1}`.

    wavevectors : `numpy.ndarray`, keyword-only, optional
        Scattering wavevectors for which to compute the intermediate
        scattering functions. Has precedence over `n_points`,
        `n_surfaces`, and `n_surface_points` if specified.

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
        :code:`results.units["results.wavenumbers"]`.

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

        self._dt = strip_unit(dt or self._trajectory.dt, "picosecond")[0]

        # Define the functions to use depending on whether the user
        # wants parallelization
        if parallel:
            self._cosine_sum_2d = accelerated.cosine_sum_parallel_2d
            self._cosine_sum_inplace_2d \
                = accelerated.cosine_sum_inplace_parallel_2d
            self._sine_sum_2d = accelerated.sine_sum_parallel_2d
            self._sine_sum_inplace_2d \
                = accelerated.sine_sum_inplace_parallel_2d
        else:
            self._cosine_sum_2d = accelerated.cosine_sum_2d
            self._cosine_sum_inplace_2d = accelerated.cosine_sum_inplace_2d
            self._sine_sum_2d = accelerated.sine_sum_2d
            self._sine_sum_inplace_2d = accelerated.sine_sum_inplace_2d

        self._n_lags = n_lags
        self._incoherent = incoherent

    def _prepare(self) -> None:

        # Update number of time lags now that the user has specified
        # the frames to analyze, if necessary
        self._n_lags = self._n_lags or self.n_frames

        # Ensure frames are evenly spaced and proceed forward in time
        if hasattr(self._sliced_trajectory, "frames"):
            df = np.diff(self._sliced_trajectory.frames)
            if df[0] <= 0 or not np.allclose(df, df[0]):
                emsg = ("The selected frames must be evenly spaced and "
                        "proceed forward in time.")
                raise ValueError(emsg)
            df = df[0]
        elif hasattr(self._sliced_trajectory, "step"):
            if self._sliced_trajectory.step <= 0:
                raise ValueError("The analysis must proceed forward in time.")
            df = self._sliced_trajectory.step

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
        self.results.units = {"results.times": ureg.picosecond,
                              "results.wavenumbers": ureg.angstrom ** -1}

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
                self._cosine_sum_inplace_2d(qrs, self._cos_sum[rcfi, 0])
                self._sine_sum_inplace_2d(qrs, self._sin_sum[rcfi, 0])
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
                            self._cosine_sum_2d(qrs)
                            - 1j * self._sine_sum_2d(qrs)
                        ).real
            else:
                for i in range(self._n_groups):
                    qrs = self._inner(self._wavevectors,
                                      self._positions[rcfi, self._slices[i]])
                    self._cosine_sum_inplace_2d(qrs, self._cos_sum[rcfi, i])
                    self._sine_sum_inplace_2d(qrs, self._sin_sum[rcfi, i])
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
                                    self._cosine_sum_2d(qrs)
                                    - 1j * self._sine_sum_2d(qrs)
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
        normalization = (
            self._N
            * np.arange(self.n_frames,
                        self.n_frames - self._n_lags, -1)[:, None, None]
        )
        self.results.cisf /= normalization
        if self._incoherent:
            self.results.iisf /= normalization

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