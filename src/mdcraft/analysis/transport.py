"""
Transport properties
====================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains classes to evaluate the transport properties of
fluid systems.
"""

import itertools
from typing import Union
import warnings

import MDAnalysis as mda
from MDAnalysis.lib.log import ProgressBar
from MDAnalysis.lib.mdamath import make_whole
import numpy as np
from scipy import optimize

from .base import SerialAnalysisBase
from .. import FOUND_OPENMM, Q_, ureg
from ..algorithm import correlation
from ..algorithm.molecule import center_of_mass
from ..algorithm.topology import unwrap, wrap
from ..algorithm.unit import strip_unit
from ..fit.polynomial import poly1

if FOUND_OPENMM:
    from openmm import unit

def msd_fft(
        r_i: np.ndarray[float], r_j: np.ndarray[float] = None, /,
        axis: int = None, *, average: bool = True) -> np.ndarray[float]:

    r"""
    Evaluates the mean squared displacements (MSD) or the analogous
    cross displacements (CD) of positions :math:`\mathbf{r}_i(t)` and
    :math:`\mathbf{r}_j(t)` using fast Fourier transforms (FFT).

    .. seealso..

       This is an alias function. For more information, see
       :func:`mdcraft.algorithm.correlation.msd_fft`.
    """

    return correlation.msd_fft(r_i, r_j, axis, average=average)

def msd_shift(
        r_i: np.ndarray[float], r_j: np.ndarray[float] = None, /,
        axis: int = None, *, average: bool = True) -> np.ndarray[float]:

    r"""
    Evaluates the mean squared displacements (MSD) or the analogous
    cross displacements (CD) of positions :math:`\mathbf{r}_i(t)` and
    :math:`\mathbf{r}_j(t)` using the Einstein relation.

    .. seealso..

       This is an alias function. For more information, see
       :func:`mdcraft.algorithm.correlation.msd_shift`.
    """

    return correlation.msd_shift(r_i, r_j, axis, average=average)

def calculate_transport_coefficients(
        time: np.ndarray[float], msd_cross: np.ndarray[float],
        msd_self: np.ndarray[float], Ns: np.ndarray[int],
        dimensions: np.ndarray[float], kBT: float, start: int = 1,
        stop: int = None, scale: str = "log", *, start_self: int = None,
        stop_self: int = None, scale_self: str = None,
        enforce_linear: bool = True, verbose: bool = False
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:

    r"""
    Fits the mean squared displacements (MSDs) or the analogous cross
    displacements (CDs) to evaluate the self-diffusion coefficients
    :math:`D_i` and the Onsager transport coefficients :math:`L_{ij}`
    and :math:`L_{ii}^\mathrm{self}`.

    Parameters
    ----------
    time : `numpy.ndarray`
        Changes in time :math:`t-t_0`.

        **Shape**: :math:`(N_t,)`.

        **Reference unit**: :math:`\mathrm{ps}`.

    msd_cross : `numpy.ndarray`
        MSDs/CDs for the :math:`N_\mathrm{g}` groups over
        :math:`N_\mathrm{b}` blocks of :math:`N_t` trajectory frames
        each. Includes the dimensionality scaling factor.

        **Shape**: :math:`(C(N_\mathrm{g}+1,\,2),\,N_t)` or
        :math:`(C(N_\mathrm{g}+1,\,2),\,N_\mathrm{b},\,N_t)`.

        **Reference unit**: :math:`\mathrm{Å}^2`.

    msd_self : `numpy.ndarray`
        Self MSDs for the :math:`N_\mathrm{g}` groups over
        :math:`N_\mathrm{b}` blocks of :math:`N_t` trajectory frames
        each. Includes the dimensionality scaling factor.

        **Shape**: :math:`(N_\mathrm{g},\,N_t)` or
        :math:`(N_\mathrm{g},\,N_\mathrm{b},\,N_t)`.

        **Reference unit**: :math:`\mathrm{Å}^2`.

    Ns : `numpy.ndarray`
        Number of particles or centers of masses :math:`N` in each
        group.

    dimensions : `numpy.ndarray`
        System dimensions.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    kBT : `float`
        Thermal energy scale.

        **Reference unit**: :math:`\mathrm{kJ/mol}`.

    start : `int`, default: :code:`1`
        Starting frame with respect to the interval used in
        :meth:`Onsager.run` for fitting the MSDs (or their analogs).

    stop : `int`, optional
        Ending frame with respect to the interval used in
        :meth:`Onsager.run` for fitting the MSDs (or their analogs).

    scale : `str`, :code:`{"log", "linear"}`
        Data scaling for fitting the MSDs (or their analogs) against
        time.

        .. container::

           **Valid values**:

           * :code:`"linear"`: Linear :math:`x`- and :math:`y`-axes.
           * :code:`"log"`: Logarithmic :math:`x`- and :math:`y`-axes.

    start_self : `int`, keyword-only, optional
        Starting frame for fitting the self MSDs. If not provided,
        `start_self` shares a value with `start`.

    stop_self : `int`, keyword-only, optional
        Ending frame for fitting the self MSDs. If not provided,
        `stop_self` shares a value with `stop`.

    scale_self : `str`, keyword-only, optional
        Data scaling for fitting the self MSDs against time. If not
        provided, `scale_self` shares a value with `scale`.

        .. container::

           **Valid values**:

           * :code:`"linear"`: Linear :math:`x`- and :math:`y`-axes.
           * :code:`"log"`: Logarithmic :math:`x`- and :math:`y`-axes.

    enforce_linear : `bool`, keyword-only, default: :code:`True`
        Enforce linear fits for data on a logarithmic scale by
        setting the slope to :math:`1`, or

        .. math::

           \log(\mathrm{MSD})=\log(t)+b

    verbose : `bool`, keyword-only, default: :code:`False`
        Determines if detailed progress is shown.

    Returns
    -------
    L_ij : `numpy.ndarray`
        Onsager transport coefficients :math:`L_{ij}`.

        **Shape**: :math:`(N_\mathrm{b},\,N_\mathrm{g},\,N_\mathrm{g})`.

        **Reference unit**:
        :math:`\mathrm{mol/(kJ\cdotÅ\cdot\mathrm ps)}`.

    L_ii_self : `numpy.ndarray`
        Self Onsager transport coefficient terms
        :math:`L_{ii}^\mathrm{self}`.

        **Shape**: :math:`(N_\mathrm{b},\,N_\mathrm{g})`.

        **Reference unit**:
        :math:`\mathrm{mol/(kJ\cdotÅ\cdot\mathrm ps)}`.

    D_i : `numpy.ndarray`
        Self-diffusion coefficients :math:`D_i`.

        **Shape**: :math:`(N_\mathrm{b},\,N_\mathrm{g})`.

        **Reference unit**: :math:`\mathrm{Å}/\mathrm{ps)}`.
    """

    # Set default settings for fitting self MSDs
    if start_self is None:
        start_self = start
    if stop_self is None:
        stop_self = stop
    if scale_self is None:
        scale_self = scale

    # Preallocate arrays to hold Onsager transport coefficients and
    # self-diffusion coefficients
    ndim = msd_self.ndim
    if ndim not in (2, 3):
        emsg = ("The arrays containing the cross- and self-MSDs have "
                "invalid shapes.")
        raise ValueError(emsg)
    elif ndim == 2:
        n_groups = msd_self.shape[0]
        n_blocks = 1
    elif ndim == 3:
        n_groups, n_blocks = msd_self.shape[:2]
    L_ij = np.zeros((n_blocks, n_groups, n_groups))
    D_i = np.zeros((n_blocks, n_groups))

    # Get indices of upper-triangular entries in L_ij and D_i
    # arrays
    r_ud, c_ud = np.triu_indices(n_groups)

    # Calculate MSD "normalization" factor (kBT * V)
    denom = kBT * dimensions[~np.isclose(dimensions, 0)].prod()

    # Iterate through all blocks
    for b in ProgressBar(range(n_blocks), verbose=verbose):
        # Iterate through all unique group pairings
        for i, msd in enumerate(msd_cross[:, b] / denom):
            y = msd[start:stop]
            valid = np.isfinite(y) & (y > 0)
            y = y[valid]
            x = time[start:stop][valid]

            if len(x) > 1:

                # Calculate L_ij
                if scale == "linear":
                    L_ij[b, r_ud[i], c_ud[i]] = np.polyfit(x, y, 1)[0]
                elif scale == "log":
                    if enforce_linear:
                        L_ij[b, r_ud[i], c_ud[i]] = np.exp(
                            optimize.curve_fit(lambda x, b: poly1(x, 1, b),
                                               np.log(x), np.log(y))[0]
                        )
                    else:
                        fit = np.polyfit(np.log(x), np.log(y), 1)
                        if abs(1 - fit[0]) >= 0.01:
                            wmsg = ("The slope for log(MSDc) vs. log(t) fit "
                                    f"is {fit[0]:.6f}.")
                            warnings.warn(wmsg)
                        L_ij[b, r_ud[i], c_ud[i]] = np.exp(fit[1])
            else:
                L_ij[b, r_ud[i], c_ud[i]] = np.nan

        # Mirror the L_ij matrix over the diagonal
        L_ij[b] = L_ij[b] + L_ij[b].T - np.diag(np.diag(L_ij[b]))

        # Iterate through all self groups
        for i, msd in enumerate(msd_self[:, b]):
            y = msd[start_self:stop_self]
            valid = np.isfinite(y) & (y > 0)
            y = y[valid]
            x = time[start_self:stop_self][valid]

            if len(x) > 1:

                # Calculate D_i
                if scale_self == "linear":
                    D_i[b, i] = np.polyfit(x, y, 1)[0]
                elif scale_self == "log":
                    if enforce_linear:
                        D_i[b, i] = np.exp(
                            optimize.curve_fit(lambda x, b: poly1(x, 1, b),
                                               np.log(x), np.log(y))[0]
                        )
                    else:
                        fit = np.polyfit(np.log(x), np.log(y), 1)
                        if abs(1 - fit[0]) >= 0.01:
                            wmsg = ("The slope for log(MSD) vs. log(t) fit is "
                                    f"{fit[0]:.6f}!")
                            warnings.warn(wmsg)
                        D_i[b, i] = np.exp(fit[1])
            else:
                D_i[b, i] = np.nan

    return L_ij, Ns * D_i / denom, D_i

def calculate_conductivity(
        L_ij: np.ndarray[float], z: np.ndarray[float], *, reduced: bool = False
    ) -> np.ndarray[float]:

    r"""
    Calculates the ionic conductivity :math:`\kappa` using the
    Onsager transport coefficients :math:`L_{ij}`.

    Parameters
    ----------
    L_ij : `numpy.ndarray`
        Onsager transport coefficients :math:`L_{ij}` for the atoms or
        residues in the :math:`N_\mathrm{g}` groups and
        :math:`N_\mathrm{b}` trajectory blocks.

        **Shape**: :math:`(N_\mathrm{b},\,N_\mathrm{g},\,N_\mathrm{g})`.

        **Reference unit**:
        :math:`\mathrm{mol/(kJ}\cdot\mathrm{Å}\cdot\mathrm{ps)}`.

    z : array-like
        Charge numbers :math:`z_i` of the atoms or residues in the
        :math:`N_\mathrm{g}` groups.

        **Shape**: :math:`(N_\mathrm{g},)`.

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units.

    Returns
    -------
    kappas : `numpy.ndarray`
        Conductivities :math:`\kappa` for the :math:`N_\mathrm{b}`
        trajectory blocks.

        **Shape**: :math:`(N_\mathrm{b},\,)`.

        **Reference unit**: :math:`\mathrm{C}^2/(\mathrm{kJ}\cdot
        \mathrm{Å}\cdot\mathrm{ps})`.

        **To SI unit**: :math:`1\times10^{19}\,\mathrm{S}/\mathrm{m}`.
    """

    kappas = np.einsum("bij,ij->b", L_ij, z * z[:, None])
    if not reduced:
        kappas = (kappas * ureg.avogadro_constant
                  * ureg.elementary_charge ** 2 * ureg.mole
                  / ureg.coulomb ** 2).to_reduced_units().magnitude
    return kappas

def calculate_electrophoretic_mobility(
        L_ij: np.ndarray[float], z: np.ndarray[float], rho: np.ndarray[float],
        *, reduced: bool = False) -> np.ndarray[float]:

    r"""
    Calculates the electrophoretic mobility :math:`\mu_i` of each
    species using the Onsager transport coefficients :math:`L_{ij}`.

    Parameters
    ----------
    L_ij : `numpy.ndarray`
        Onsager transport coefficients :math:`L_{ij}` for the atoms or
        residues in the :math:`N_\mathrm{g}` groups and
        :math:`N_\mathrm{b}` trajectory blocks.

        **Shape**: :math:`(N_\mathrm{b},\,N_\mathrm{g},\,N_\mathrm{g})`.

        **Reference unit**:
        :math:`\mathrm{mol/(kJ}\cdot\mathrm{Å}\cdot\mathrm{ps)}`.

    z : array-like
        Charge numbers :math:`z_i` of the atoms or residues in the
        :math:`N_\mathrm{g}` groups.

        **Shape**: :math:`(N_\mathrm{g},)`.

    rho : array-like
        Number densities :math:`n_i` of the atoms or residues in the
        :math:`N_\mathrm{g}` groups.

        **Shape**: :math:`(N_\mathrm{g},)`.

        **Reference unit**: :math:`\mathrm{Å}^{-3}`.

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units.

    Returns
    -------
    mus : `numpy.ndarray`
        Electrophoretic mobilities :math:`\mu_i` for the atoms or
        residues in the :math:`N_\mathrm{g}` groups and
        :math:`N_\mathrm{b}` trajectory blocks.

        **Shape**: :math:`(N_\mathrm{b},\,N_\mathrm{g})`.

        **Reference unit**:
        :math:`\mathrm{Å}^2\cdot\mathrm{C/(kJ}\cdot\mathrm{ps)}`.

        **To SI unit**: :math:`1\times 10^{-11}\,\mathrm{m}^2 /
        (\mathrm{V}\cdot\mathrm{s})`.
    """

    mus = (L_ij * z / rho[:, None]).sum(axis=-1)
    if not reduced:
        mus = (mus * ureg.avogadro_constant * ureg.elementary_charge
               * ureg.mole / ureg.coulomb).to_reduced_units().magnitude
    return mus

def calculate_transference_number(
        L_ij: np.ndarray[float], z: np.ndarray[float]) -> np.ndarray[float]:

    r"""
    Calculates the transference number :math:`t_i` of each species using
    the Onsager transport coefficients :math:`L_{ij}`.

    Parameters
    ----------
    L_ij : `numpy.ndarray`
        Onsager transport coefficients :math:`L_{ij}` for the atoms or
        residues in the :math:`N_\mathrm{g}` groups and
        :math:`N_\mathrm{b}` trajectory blocks.

        **Shape**: :math:`(N_\mathrm{b},\,N_\mathrm{g},\,N_\mathrm{g})`.

        **Reference unit**:
        :math:`\mathrm{mol/(kJ}\cdot\mathrm{Å}\cdot\mathrm{ps)}`.

    z : array-like
        Charge numbers :math:`z_i` of the atoms or residues in the
        :math:`N_\mathrm{g}` groups.

        **Shape**: :math:`(N_\mathrm{g},)`.

    Returns
    -------
    ts : `numpy.ndarray`
        Transference numbers :math:`t_i` for the atoms or residues in
        the :math:`N_\mathrm{g}` groups and :math:`N_\mathrm{b}`
        trajectory blocks.

        **Shape**: :math:`(N_\mathrm{b},\,N_\mathrm{g})`.
    """

    s = z * (L_ij * z).sum(axis=-1)
    return s / s.sum(axis=-1)

class Onsager(SerialAnalysisBase):

    """
    A serial implementation to calculate the Onsager transport
    coefficients and its related properties.

    .. note::

       The simulation must have been run with a constant timestep
       :math:`\\Delta t` and the frames to be analyzed must be evenly
       spaced and proceed forward in time for this analysis module to
       function correctly.

    The Onsager transport framework [1]_ can be used to analyze
    transport properties in bulk constant-volume fluids and
    electrolytic systems, such as those in the canonical
    (:math:`NVT`), microcanonical (:math:`NVE`), and grand
    canonical (:math:`\\mu VT`) ensembles.

    The Onsager transport equation

    .. math::

       \\pmb{J}_i=-\\sum_j L_{ij}\\nabla\\bar{\\mu}_j

    relates the flux :math:`\\pmb{J}_i` of species :math:`i` to the
    Onsager transport coefficients :math:`L_{ij}` and the
    electrochemical potential :math:`\\bar{\\mu}_j` of species
    :math:`j`. There is an Onsager transport coefficient for each pair
    of species that, unlike the Nernst–Einstein equation, captures the
    strong cross-correlations in electrolytes.

    The Onsager transport coefficients can be calculated from the
    particle positions over time using the Einstein relation

    .. math::

       L_{ij}=\\frac{1}{6k_\\mathrm{B}TV}
       \\lim_{t\\rightarrow\\infty}\\frac{d}{dt}\\left\\langle\\sum_\\alpha
       \\left[\\pmb{r}_{i,\\alpha}(t)-\\pmb{r}_{i,\\alpha}(0)\\right]
       \\cdot\\sum_\\beta\\left[\\pmb{r}_{j,\\beta}(t)
       -\\pmb{r}_{j,\\beta}(0)\\right]\\right\\rangle

    where :math:`k_\\mathrm{B}` is the Boltzmann constant, :math:`T` is
    the system temperature, :math:`V` is the system volume, :math:`t` is
    time, and :math:`\\pmb{r}_\\alpha` and :math:`\\pmb{r}_\\beta` are
    the positions of particles :math:`\\alpha` and :math:`\\beta`
    belonging to species :math:`i` and :math:`j`, respectively. The
    angular brackets denote the ensemble average. It is evident that
    :math:`L_{ij}=L_{ji}`; hence, the equation above is an Onsager
    reciprocal relation.

    The diagonal terms :math:`L_{ii}` in the matrix of Onsager transport
    coefficients captures the self-diffusion and self-correlations for
    a single species :math:`i` and has two contributions:

    .. math::

       L_{ii}=L_{ii}^\\mathrm{self}+L_{ii}^\\mathrm{distinct}

    The self term

    .. math::

       L_{ii}^\\mathrm{self}=\\frac{1}{6k_\\mathrm{B}TV}
       \\lim_{t\\rightarrow\\infty}\\frac{d}{dt}
       \\sum_\\alpha\\left\\langle\\left[
       \\pmb{r}_{i,\\alpha}(t)-\\pmb{r}_{i,\\alpha}(0)
       \\right]^2\\right\\rangle

    is given by the autocorrelation function of the flux of a single
    particle :math:`\\alpha` when :math:`\\alpha=\\beta`, while the
    distinct term

    .. math::

       L_{ii}^\\mathrm{self}=\\frac{1}{6k_\\mathrm{B}TV}
       \\lim_{t\\rightarrow\\infty}\\frac{d}{dt}
       \\sum_\\alpha\\sum_{\\beta\\neq\\alpha}\\left\\langle\\left[
       \\pmb{r}_{i,\\alpha}(t)-\\pmb{r}_{i,\\alpha}(0)\\right]\\cdot
       \\left[\\pmb{r}_{i,\\beta}(t)-\\pmb{r}_{i,\\beta}(0)\\right]
       \\right\\rangle

    is given by the cross-correlation between two distinct particles
    when :math:`\\alpha\\neq\\beta`. Naturally, the self term is
    related to the self-diffusion coefficient :math:`D_i` via

    .. math::

       L_{ii}^\\mathrm{self}=\\frac{D_i\\rho_i}{k_\\mathrm{B}T}

    where :math:`\\rho_i` is the number density of species :math:`i`.

    Finally, the Onsager transport coefficients can be used to
    obtain experimentally relevant transport properties:

    * Ionic conductivity:

      .. math::

         \\kappa=F^2\\sum_i\\sum_j z_iz_jL_{ij}

      :math:`F` is the Faraday constant.

    * Electrophoretic mobility:

      .. math::

         \\mu_i=\\frac{F}{\\rho_i}\\sum_j z_jL_{ij}

    * Transference number:

      .. math::

         t_i=\\frac{\\sum_j z_iz_jL_{ij}}{\\sum_k\\sum_l z_kz_lL_{kl}}

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Group(s) of atoms for which the mean squared displacements (or
        analogs) are calculated.

    groupings : `str` or array-like, default: :code:`"atoms"`
        Determines whether the centers of mass are used in lieu of
        individual atom positions. If `groupings` is a `str`, the same
        value is used for all `groups`.

        .. note::

           In a standard trajectory file, segments (or chains) contain
           residues (or molecules), and residues contain atoms. This
           heirarchy must be adhered to for this analysis module to
           function correctly, unless your selected grouping is always
           :code:`"atoms"`.

        .. container::

           **Valid values**:

           * :code:`"atoms"`: Atom positions (generally for
             coarse-grained simulations).
           * :code:`"residues"`: Residues' centers of mass (for
             atomistic simulations).
           * :code:`"segments"`: Segments' centers of mass (for
             atomistic polymer simulations).

    temperature : `float`, `openmm.unit.Quantity`, or `pint.Quantity`, \
    default: :code:`300`
        System temperature :math:`T`.

        .. note::

           If :code:`reduced=True`, `temperature` should be equal to the
           energy scale. When the Lennard-Jones potential is used, it
           generally means that :math:`T^* = 1`, or `temperature=1`.

        **Reference unit**: :math:`\\mathrm{K}`.

    charges : array-like, `openmm.unit.Quantity`, or `pint.Quantity`, \
    keyword-only, optional
        Charge numbers :math:`z_i` for the specified `groupings` in the
        :math:`N_\\mathrm{g}` `groups`. If not provided, it will be
        retrieved from the topology or trajectory.

        **Shape**: :math:`(N_\\mathrm{g},)`.

        **Reference unit**: :math:`\\mathrm{e}`.

    dimensions : array-like, `openmm.unit.Quantity`, or \
    `pint.Quantity`, keyword-only, optional
        System dimensions. If not provided, they are retrieved from the
        topology or trajectory.

        .. tip::

           You can zero out dimensions by setting them to :code:`0` if
           your simulation is pseudo-1D or pseudo-2D. For example, if
           you have a one-layer thick slab in the :math:`xy`-plane, you
           can set :code:`dimensions=np.array((L_x, L_y, 0))` to
           evaluate the transport properties in 2D. Note that this
           affects the reference units of the Onsager transport
           coefficients and its related properties, but not the
           self-diffusivity.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\\mathrm{Å}`.

    dt : `float`, `openmm.unit.Quantity`, or `pint.Quantity`, \
    keyword-only, optional
        Time between frames :math:`\\Delta t`. While this is normally
        determined from the trajectory, the trajectory may not have the
        correct information if the data is in reduced units. For
        example, if your reduced timestep is :math:`0.01` and you output
        trajectory data every :math:`10,000` timesteps, then
        :math:`\\Delta t=100`.

        **Reference unit**: :math:`\\mathrm{ps}`.

    n_blocks : `int`, keyword-only, default: :code:`1`
        Number of blocks :math:`N_\\mathrm{b}` to split the trajectory
        into.

    center : `bool`, keyword-only, default: :code:`False`
        Determines whether the system center of mass is subtracted from
        the positions used to compute the transport properties.

    center_atom : `bool`, keyword-only, default: :code:`False`
        Determines whether the system center of mass is computed using
        the atom positions, regardless of `groupings`. Has no effect if
        :code:`center=False`.

    center_wrap : `bool`, keyword-only, default: :code:`False`
        Determines whether the system center of mass is computed using
        the wrapped particle positions. Has no effect if
        :code:`center=False`.

    fft : `bool`, keyword-only, default: :code:`True`
        Determines whether fast Fourier transforms (FFT) are used to
        evaluate the mean squared displacements (or analogs).

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units. Affects
        `temperature`, `results.times`, etc.

    unwrap : `bool`, keyword-only, default: :code:`False`
        Determines if atom positions are unwrapped. Ensure that
        :code:`unwrap=False` when the trajectory already contains
        unwrapped particle positions, as this parameter is used in
        conjunction with `center_wrap` to determine the appropriate
        system center of mass.

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
        reference units for :code:`results.times`, call
        :code:`results.units["results.times"]`.

    results.pairs : `tuple`
        All unique pairs of indices of the groups of atoms in `groups`.
        The ordering coincides with the column indices in
        `results.msd`.

    results.times : `numpy.ndarray`
        Changes in time :math:`t-t_0`.

        **Shape**: :math:`(N_t,)`.

        **Reference unit**: :math:`\\mathrm{ps}`.

    results.msd_cross : `numpy.ndarray`
        MSDs (or analogs) for the :math:`N_\\mathrm{g}` groups over
        :math:`N_\\mathrm{b}` blocks of :math:`N_t` trajectory frames
        each. Includes the dimensionality scaling factor. See Notes
        about what this value actually means.

        **Shape**:
        :math:`(_{N_\\mathrm{g}+1}\\mathrm{C}_{2},\\,N_\\mathrm{b},\\,
        N_t)`.

        **Reference unit**: :math:`\\mathrm{Å}^2`.

    results.msd_self : `numpy.ndarray`
        Self MSDs (or analogs) for the :math:`N_\\mathrm{g}` groups over
        :math:`N_\\mathrm{b}` blocks of :math:`N_t` trajectory frames
        each. Includes the dimensionality scaling factor. See Notes
        about what this value actually means.

        **Shape**: :math:`(N_\\mathrm{g},\\,N_\\mathrm{b},\\,N_t)`.

        **Reference unit**: :math:`\\mathrm{Å}^2`.

    results.L_ij : `numpy.ndarray`
        Onsager transport coefficients :math:`L_{ij}`. Only available
        after running :meth:`calculate_transport_coefficients`.

        **Shape**:
        :math:`(N_\\mathrm{b},\\,N_\\mathrm{g},\\,N_\\mathrm{g})`.

        **Reference unit**:
        :math:`\\mathrm{mol/(kJ}\\cdot\\mathrm{Å}\\cdot\\mathrm{ps)}`.

    results.L_ii_self : `numpy.ndarray`
        Self Onsager transport coefficient terms
        :math:`L_{ii}^\\mathrm{self}`. Note that
        :math:`L_{ii}^\\mathrm{self}` is related to :math:`D_i` via

        .. math::

            L_{ii}^\\mathrm{self}=\\dfrac{N}{k_\\mathrm{B}TV}D_i

        Only available after running
        :meth:`calculate_transport_coefficients`.

        **Shape**: :math:`(N_\\mathrm{b},\\,N_\\mathrm{g})`.

        **Reference unit**:
        :math:`\\mathrm{mol/(kJ}\\cdot\\mathrm{Å}\\cdot\\mathrm{ps)}`.

    results.D_i : `numpy.ndarray`
        Self-diffusion coefficients :math:`D_i`. Only available after
        running :meth:`calculate_transport_coefficients`.

        **Shape**: :math:`(N_\\mathrm{b},\\,N_\\mathrm{g})`.

        **Reference unit**: :math:`\\mathrm{Å}^2/\\mathrm{ps)}`.

    results.conductivities : `numpy.ndarray`
        Conductivities :math:`\\kappa`. Only available after running
        :meth:`calculate_conductivity`.

        **Shape**: :math:`(N_\\mathrm{b},\\,N_\\mathrm{g})`.

        **Reference unit**: :math:`\\mathrm{C}^2/(\\mathrm{kJ}\\cdot
        \\mathrm{Å}\\cdot\\mathrm{ps})`.

        **To SI unit**: :math:`1\times10^{19}\\,\\mathrm{S}/\\mathrm{m}`.

    results.electrophoretic_mobilities : `numpy.ndarray`
        Electrophoretic mobilities :math:`\\mu_i`. Only available after
        running :meth:`calculate_electrophoretic_mobility`.

        **Shape**: :math:`(N_\\mathrm{b},\\,N_\\mathrm{g})`.

        **Reference unit**:
        :math:`\\mathrm{Å}^2\\cdot\\mathrm{C/(kJ}\\cdot\\mathrm{ps)}`.

        **To SI unit**: :math:`1\times10^{-11}\\,\\mathrm{m}^2/
        (\\mathrm{V}\\cdot\\mathrm{s})`.

    results.transference_numbers : `numpy.ndarray`
        Transference numbers :math:`t_i`. Only available after running
        :meth:`calculate_transference_number`.

        **Shape**: :math:`(N_\\mathrm{b},\\,N_\\mathrm{g})`.

    Notes
    -----
    * The values in `results.msd_cross` are actually "summed squared
      displacements"—that is, `results.msd_cross` contains the sum of
      all particles' squared displacements instead of their average.
      However, it is extremely unlikely the raw values in
      `results.msd_cross` will be used other than to calculate the
      Onsager transport coefficients. On the other hand,
      `results.msd_self` is averaged over all particles. Therefore, it
      can be plotted against `results.times` to get the self-diffusion
      coefficients.

    References
    ----------
    .. [1] Fong, K. D.; Self, J.; McCloskey, B. D.; Persson, K. A.
       Onsager Transport Coefficients and Transference Numbers in
       Polyelectrolyte Solutions and Polymerized Ionic Liquids.
       *Macromolecules* **2020**, **53** (21), 9503–9512.
       https://doi.org/10.1021/acs.macromol.0c02001.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            groupings: Union[str, tuple[str]] = "atoms",
            temperature: Union[float, "unit.Quantity", Q_] = 300, *,
            charges: Union[np.ndarray[float], "unit.Quantity", Q_] = None,
            dimensions: Union[np.ndarray[float], "unit.Quantity", Q_] = None,
            dt: Union[float, "unit.Quantity", Q_] = None,
            n_blocks: int = 1, center: bool = False, center_atom: bool = False,
            center_wrap: bool = False, fft: bool = True, reduced: bool = False,
            unwrap: bool = False, verbose: bool = True, **kwargs) -> None:

        self._groups = [groups] if isinstance(groups, mda.AtomGroup) else groups
        self.universe = self._groups[0].universe
        super().__init__(self.universe.trajectory, verbose=verbose, **kwargs)

        self._n_groups = len(self._groups)
        if isinstance(groupings, str):
            if groupings not in (GROUPINGS := {"atoms", "residues",
                                               "segments"}):
                emsg = (f"Invalid grouping '{groupings}'. Valid values: "
                        f"{', '.join(GROUPINGS)}.")
                raise ValueError(emsg)
            self._groupings = self._n_groups * [groupings]
        else:
            if self._n_groups != len(groupings):
                emsg = ("The number of grouping values is not equal to the "
                        "number of groups.")
                raise ValueError(emsg)
            for g in groupings:
                if g not in GROUPINGS:
                    emsg = (f"Invalid grouping '{g}'. Valid values: "
                            f"{', '.join(GROUPINGS)}.")
                    raise ValueError(emsg)
            self._groupings = groupings

        temperature, unit_ = strip_unit(temperature, "temperature")
        if reduced:
            if not isinstance(unit_, str):
                emsg = "'temperature' cannot have units when reduced=True."
                raise TypeError(emsg)
            self._kBT = temperature
        else:
            self._kBT = (
                ureg.avogadro_constant * ureg.boltzmann_constant
                * temperature * ureg.kelvin
            ).m_as(self.results.units["_kBT"])

        if dimensions is not None:
            if len(dimensions) != 3:
                raise ValueError("'dimensions' must have length 3.")
            self._dimensions = np.asarray(strip_unit(dimensions, "angstrom")[0])
        elif self.universe.dimensions is not None:
            self._dimensions = self.universe.dimensions[:3].copy()
        else:
            raise ValueError("No system dimensions found or provided.")

        self._dt, unit_ = strip_unit(dt or self._trajectory.dt, "picosecond")
        if reduced and not isinstance(unit_,  str):
            raise TypeError("'dt' cannot have units when reduced=True.")

        if charges is not None:
            if len(charges) != self._n_groups:
                emsg = ("The number of group charges is not equal to "
                        "the number of groups.")
                raise ValueError(emsg)
            charges, unit_ = strip_unit(charges, "elementary_charge")
            if reduced and not isinstance(unit_, str):
                emsg = "'charges' cannot have units when reduced=True."
                raise TypeError(emsg)
            self._charges = np.asarray(charges)
        elif hasattr(self.universe.atoms, "charges"):
            self._charges = np.fromiter(
                (getattr(g, gr).charges[0]
                 for g, gr in zip(self._groups, self._groupings)),
                count=self._n_groups,
                dtype=int
            )
        else:
            self._charges = None

        # Determine the number of particles in each group and their
        # corresponding indices
        self._Ns = tuple(getattr(a, f"n_{g}")
                         for (a, g) in zip(self._groups, self._groupings))
        self._N = sum(self._Ns)
        self._slices = []
        index = 0
        for N in self._Ns:
            self._slices.append(slice(index, index + N))
            index += N

        if np.all(~np.isclose(self._dimensions, 0)):
            self._rhos = np.fromiter(
                (getattr(g, f"n_{gr}")
                for g, gr in zip(self._groups, self._groupings)),
                count=self._n_groups,
                dtype=float
            ) / self._dimensions.prod()

        self._n_blocks = n_blocks
        self._center = center
        self._center_atom = center_atom
        self._center_wrap = center_wrap
        self._fft = fft
        self._reduced = reduced
        self._unwrap = unwrap
        self._verbose = verbose

    def _prepare(self) -> None:

        # Ensure frames are evenly spaced and proceed forward in time
        if hasattr(self._sliced_trajectory, "frames"):
            df = np.diff(self._sliced_trajectory.frames)
            if df[0] <= 0 or not np.allclose(df, df[0]):
                emsg = ("The selected frames must be evenly spaced and "
                        "proceed forward in time.")
                raise ValueError(emsg)
        elif hasattr(self._sliced_trajectory, "step") \
                and self._sliced_trajectory.step <= 0:
            raise ValueError("The analysis must proceed forward in time.")

        # Find all unique AtomGroup combinations
        self.results.pairs = tuple(
            itertools.combinations_with_replacement(range(self._n_groups), 2)
        )

        # Preallocate arrays to store positions and number of boundary
        # crossings
        self._positions = np.empty((self.n_frames, self._N, 3))
        if self._unwrap:
            self.universe.trajectory[
                self._sliced_trajectory.frames[0]
                if hasattr(self._sliced_trajectory, "frames")
                else (self.start or 0)
            ]
            for f in self.universe.atoms.fragments:
                make_whole(f)
            self._positions_old = self.universe.atoms.positions
            self._images = np.zeros((self.universe.atoms.n_atoms, 3), dtype=int)
            self._thresholds = self._dimensions / 2

        # Determine number of frames used when the trajectory is split
        # into blocks
        self._n_frames_block = self.n_frames // self._n_blocks
        self._n_frames = self._n_blocks * self._n_frames_block
        extra_frames = self.n_frames - self._n_frames
        if extra_frames > 0:
            wmsg = (f"The trajectory is not divisible into {self._n_blocks:,} "
                    f"blocks, so the last {extra_frames:,} frame(s) will be "
                    "discarded. To maximize performance, set appropriate "
                    "starting and ending frames in run() so that the number "
                    "of frames to be analyzed is divisible by the number of "
                    "blocks.")
            warnings.warn(wmsg)

        # Preallocate arrays to store results
        self.results.times = self.step * self._dt * np.arange(self._n_frames //
                                                              self._n_blocks)
        self.results.msd_cross = np.empty(
            (len(self.results.pairs), self._n_blocks, self._n_frames_block),
            dtype=float
        )
        self.results.msd_self = np.empty(
            (self._n_groups, self._n_blocks, self._n_frames_block),
            dtype=float
        )

        # Store reference units
        self.results.units = {"results.times": ureg.picosecond}
        self.results.units["results.msd_cross"] = \
            self.results.units["results.msd_self"] = ureg.angstrom ** 2

    def _single_frame(self) -> None:

        positions = self.universe.atoms.positions.copy()
        if self._unwrap:
            unwrap(positions, self._positions_old, self._dimensions,
                   thresholds=self._thresholds, images=self._images)

        for g, gr, s in zip(self._groups, self._groupings, self._slices):
            self._positions[self._frame_index, s] = (
                positions[g.indices] if gr == "atoms"
                else center_of_mass(
                    g, gr,
                    images=self._images[g.indices] if hasattr(self, "_images")
                           else None
                )
            )

        # Subtract system center-of-mass from particle or molecule
        # center-of-mass positions
        if self._center:
            if self._center_atom:
                if self._center_wrap:
                    wrap(positions, self._dimensions)
                scom = center_of_mass(positions=positions,
                                      masses=self.universe.atoms.masses)
            else:
                if self._center_wrap:
                    positions = wrap(self._positions[self._frame_index],
                                     self._dimensions, in_place=False)
                else:
                    positions = self._positions[self._frame_index]
                scom = center_of_mass(
                    positions=positions,
                    masses=np.concatenate(
                        [getattr(g, gr).masses
                         for g, gr in zip(self._groups, self._groupings)]
                    )
                )
            self._positions[self._frame_index] -= scom

    def _conclude(self) -> None:

        # Truncate the positions array if there are extra frames
        if self.n_frames != self._n_frames:
            self._positions = self._positions[:self._n_frames]

        # Compute the MSDs (or their analogs) for each unique AtomGroup
        # pair
        msd = msd_fft if self._fft else msd_shift
        delete_dimensions = np.isclose(self._dimensions, 0)
        for i, (i1, i2) in enumerate(ProgressBar(self.results.pairs,
                                                 verbose=self._verbose)):
            if i1 == i2:
                if self._Ns[i1]:
                    positions = self._positions[:, self._slices[i1]].reshape(
                        (self._n_blocks, -1, self._Ns[i1], 3)
                    )
                    positions[:, :, :, delete_dimensions] = 0
                    self.results.msd_cross[i] = msd(positions.sum(axis=2),
                                                    axis=1)
                    self.results.msd_self[i1] = (
                        msd(positions, axis=1, average=False).sum(axis=-1) /
                        self._Ns[i1]
                    )
                else:
                    self.results.msd_cross[i] \
                        = self.results.msd_self[i1] = np.nan
            elif self._Ns[i1] and self._Ns[i2]:
                positions1 = self._positions[:, self._slices[i1]].reshape(
                    (self._n_blocks, -1, self._Ns[i1], 3)
                ).sum(axis=2)
                positions2 = self._positions[:, self._slices[i2]].reshape(
                    (self._n_blocks, -1, self._Ns[i2], 3)
                ).sum(axis=2)
                positions1[:, :, delete_dimensions] \
                    = positions2[:, :, delete_dimensions] = 0
                self.results.msd_cross[i] = msd(positions1, positions2, axis=1)
            else:
                self.results.msd_cross[i] = np.nan

        # Account for dimensionality by dividing by 2 * D
        D = 2 * (~delete_dimensions).sum()
        self.results.msd_cross /= D
        self.results.msd_self /= D

    def calculate_transport_coefficients(
            self, start: int = 1, stop: int = None, scale: str = "log", *,
            start_self: int = None, stop_self: int = None,
            scale_self: str = None, enforce_linear: bool = True) -> None:

        r"""
        Fits the mean squared displacements (MSDs) (or analogs) to
        evaluate the Onsager transport coefficients :math:`L_{ij}` and
        :math:`L_{ii}^\mathrm{self}`.

        Parameters
        ----------
        start : `int`, default: :code:`1`
            Starting frame with respect to the interval used in
            :meth:`Onsager.run` for fitting the MSDs (or their analogs).

        stop : `int`, optional
            Ending frame with respect to the interval used in
            :meth:`Onsager.run` for fitting the MSDs (or their analogs).

        scale : `str`, :code:`{"log", "linear"}`
            Data scaling for fitting the MSDs (or their analogs) against
            time.

            .. container::

               **Valid values**:

               * :code:`"linear"`: Linear :math:`x`- and :math:`y`-axes.
               * :code:`"log"`: Logarithmic :math:`x`- and :math:`y`-axes.

        start_self : `int`, keyword-only, optional
            Starting frame with respect to the interval used in
            :meth:`Onsager.run` for fitting the self MSDs. If not
            provided, `start_self` shares a value with `start`.

        stop_self : `int`, keyword-only, optional
            Ending frame with respect to the interval used in
            :meth:`Onsager.run` for fitting the self MSDs. If not
            provided, `stop_self` shares a value with `stop`.

        scale_self : `str`, keyword-only, optional
            Data scaling for fitting the self MSDs against time. If not
            provided, `scale_self` shares a value with `scale`.

            .. container::

               **Valid values**:

               * :code:`"linear"`: Linear :math:`x`- and :math:`y`-axes.
               * :code:`"log"`: Logarithmic :math:`x`- and :math:`y`-axes.

        enforce_linear : `bool`, keyword-only, default: :code:`True`
            Enforce linear fits for data on a logarithmic scale by
            setting the slope to :math:`1`, or

            .. math::

                \log(\mathrm{MSD})=\log(t)+b
        """

        if not hasattr(self.results, "msd_cross"):
            emsg = ("Call Onsager.run() before "
                    "Onsager.calculate_transport_coefficients().")
            raise RuntimeError(emsg)

        self.results.L_ij, self.results.L_ii_self, self.results.D_i \
            = calculate_transport_coefficients(
                self.results.times,
                self.results.msd_cross,
                self.results.msd_self,
                self._Ns,
                self._dimensions,
                self._kBT,
                start,
                stop,
                scale,
                start_self=start_self,
                stop_self=stop_self,
                scale_self=scale_self,
                enforce_linear=enforce_linear,
                verbose=self._verbose
            )

        # Store reference units
        if not self._reduced:
            self.results.units["results.D_i"] = (ureg.angstrom ** 2
                                                 / ureg.picosecond)
            self.results.units["results.L_ii_self"] \
                = self.results.units["results.L_ij"] \
                = 1 / (ureg.kilojoule * ureg.angstrom * ureg.picosecond
                       / ureg.mole)

    def calculate_conductivity(
            self, *,
            charges: Union[np.ndarray[float], "unit.Quantity", Q_] = None
        ) -> None:

        """
        Calculates the ionic conductivity :math:`\\kappa` using the
        Onsager transport coefficients :math:`L_{ij}`.

        Parameters
        ----------
        charges : array-like, `openmm.unit.Quantity`, or \
        `pint.Quantity`, keyword-only, optional
            Charge numbers :math:`z_i` of the groupings in the
            :math:`N_\\mathrm{g}` groups. This argument is optional only
            if `charges` has previously been passed to a calculation
            method belonging to this class.

            **Shape**: :math:`(N_\\mathrm{g},)`.

            **Reference unit**: :math:`\\mathrm{e}`.
        """

        if not hasattr(self.results, "L_ij"):
            emsg = ("Call Onsager.calculate_transport_coefficients() before "
                    "Onsager.calculate_conductivity().")
            raise RuntimeError(emsg)

        if charges is not None:
            if len(charges) != self._n_groups:
                emsg = ("The number of group charges is not equal to "
                        "the number of groups.")
                raise ValueError(emsg)
            charges, unit_ = strip_unit(charges, "elementary_charge")
            if self._reduced and not isinstance(unit_, str):
                emsg = "'charges' cannot have units when reduced=True."
                raise TypeError(emsg)
            self._charges = np.asarray(charges)
        if self._charges is None:
            raise ValueError("No charge number information available.")

        self.results.conductivities = calculate_conductivity(
            self.results.L_ij, self._charges,
            reduced=self._reduced
        )

        self.results.units["results.conductivities"] = \
            ureg.coulomb ** 2 / (ureg.kilojoule * ureg.angstrom
                                 * ureg.picosecond)

    def calculate_electrophoretic_mobility(
            self, *,
            charges: Union[np.ndarray[float], "unit.Quantity", Q_] = None,
            rhos: Union[np.ndarray[float], "unit.Quantity", Q_] = None
        ) -> None:

        """
        Calculates the electrophoretic mobility :math:`\\mu_i` of each
        species using the Onsager transport coefficients :math:`L_{ij}`.

        Parameters
        ----------
        charges : array-like, `openmm.unit.Quantity`, or \
        `pint.Quantity`, keyword-only, optional
            Charge numbers :math:`z_i` of the groupings in the
            :math:`N_\\mathrm{g}` groups. This argument is optional only
            if charge information is present in the topology or
            `charges` has previously been passed to a calculation
            method belonging to this class.

            **Shape**: :math:`(N_\\mathrm{g},)`.

            **Reference unit**: :math:`\\mathrm{e}`.

        rhos : array-like, `openmm.unit.Quantity`, or \
        `pint.Quantity`, keyword-only, optional
            Number densities :math:`n_i` of the groupings in the
            :math:`N_\\mathrm{g}` groups.

            **Shape**: :math:`(N_\\mathrm{g},)`.

            **Reference unit**: :math:`\\mathrm{Å}^{-3}`.
        """

        if not hasattr(self.results, "L_ij"):
            emsg = ("Call Onsager.calculate_transport_coefficients() before "
                    "Onsager.calculate_electrophoretic_mobility().")
            raise RuntimeError(emsg)

        if charges is not None:
            if len(charges) != self._n_groups:
                emsg = ("The number of group charges is not equal to "
                        "the number of groups.")
                raise ValueError(emsg)
            charges, unit_ = strip_unit(charges, "elementary_charge")
            if self._reduced and not isinstance(unit_, str):
                emsg = "'charges' cannot have units when reduced=True."
                raise TypeError(emsg)
            self._charges = np.asarray(charges)
        if self._charges is None:
            raise ValueError("No charge number information available.")

        if rhos is not None:
            if len(rhos) != self._n_groups:
                emsg = ("The number of group number densities is not "
                        "equal to the number of groups.")
                raise ValueError(emsg)
            rhos, unit_ = strip_unit(rhos, "angstrom**-3")
            if self._reduced and not isinstance(unit_, str):
                emsg = "'rhos' cannot have units when reduced=True."
                raise TypeError(emsg)
            self._rhos = np.asarray(rhos)
        if self._rhos is None:
            raise ValueError("No number density information available.")

        self.results.electrophoretic_mobilities = calculate_electrophoretic_mobility(
            self.results.L_ij, self._charges, self._rhos,
            reduced=self._reduced
        )

        self.results.units["results.electrophoretic_mobilities"] = \
            ureg.angstrom ** 2 * ureg.coulomb / (ureg.kilojoule
                                                 * ureg.picosecond)

    def calculate_transference_number(
            self, *,
            charges: Union[np.ndarray[float], "unit.Quantity", Q_] = None
        ) -> None:

        """
        Calculates the transference number of each species using the
        Onsager transport coefficients :math:`L_{ij}`.

        Parameters
        ----------
        charges : array-like, `openmm.unit.Quantity`, or \
        `pint.Quantity`, keyword-only, optional
            Charge numbers :math:`z_i` of the groupings in the
            :math:`N_\\mathrm{g}` groups. This argument is optional only
            if charge information is present in the topology or
            `charges` has previously been passed to a calculation
            method belonging to this class.

            **Shape**: :math:`(N_\\mathrm{g},)`.

            **Reference unit**: :math:`\\mathrm{e}`.
        """

        if not hasattr(self.results, "L_ij"):
            emsg = ("Call Onsager.calculate_transport_coefficients() before "
                    "Onsager.calculate_transference_number().")
            raise RuntimeError(emsg)

        if charges is not None:
            if len(charges) != self._n_groups:
                emsg = ("The number of group charges is not equal to "
                        "the number of groups.")
                raise ValueError(emsg)
            charges, unit_ = strip_unit(charges, "elementary_charge")
            if self._reduced and not isinstance(unit_, str):
                emsg = "'charges' cannot have units when reduced=True."
                raise TypeError(emsg)
            self._charges = np.asarray(charges)
        if self._charges is None:
            raise ValueError("No charge number information available.")

        self.results.transference_numbers = calculate_transference_number(
            self.results.L_ij, self._charges
        )