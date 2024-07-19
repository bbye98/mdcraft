"""
Polymeric properties
====================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains classes to determine the structural
and dynamical properties of polymeric systems.
"""

from typing import Union
import warnings

import MDAnalysis as mda
from MDAnalysis.lib.log import ProgressBar
import numba
import numpy as np
from scipy import optimize, special

from . import structure
from .base import Hash, DynamicAnalysisBase, NumbaAnalysisBase
from .. import FOUND_OPENMM, Q_, ureg
from ..algorithm import accelerated, correlation
from ..algorithm.molecule import center_of_mass, radius_of_gyration
from ..algorithm.topology import unwrap, unwrap_edge
from ..algorithm.unit import strip_unit
from ..algorithm.utility import get_closest_factors
from ..fit.exponential import stretched_exp

if FOUND_OPENMM:
    from openmm import unit

class _PolymerAnalysisBase(DynamicAnalysisBase):

    r"""
    An analysis base object for polymer systems.

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Groups of polymers to be analyzed.

        .. note::

           All polymers in each group must have the same chain length.

    groupings : `str` or array-like, default: :code:`"atoms"`
        Determines whether the centers of mass are used in lieu of
        individual atom positions. If `groupings` is a `str`, the same
        value is used for all `groups`.

        .. note::

           In a standard trajectory file, segments (or chains) contain
           residues (or molecules), and residues contain atoms. This
           heirarchy must be adhered to for this analysis module to
           function correctly. If your trajectory file does not contain
           the correct residue or segment information, provide the
           number of chains and chain lengths in `n_chains` and
           `n_monomers`, respectively.

        .. container::

           **Valid values**:

           * :code:`"atoms"`: Atom positions (generally or for
             coarse-grained simulations).
           * :code:`"residues"`: Residues' centers of mass (for
             atomistic simulations).

    n_chains : `int` or array-like, optional
        Number of chains :math:`M` in each polymer group. Must be
        provided if the trajectory does not adhere to the standard
        container heirarchy. If an `int` is provided, the same value is
        used for all groups.

        **Shape**: :math:`(N_\mathrm{groups},)`.

    n_monomers : `int` or array-like, optional
        Number of monomers :math:`N` in each chain in each polymer
        group. Must be provided if the trajectory does not adhere to the
        standard container heirarchy. If an `int` is provided, the same
        value is used for all groups.

        **Shape**: :math:`(N_\mathrm{groups},)`.

    unwrap : `bool`, keyword-only, default: :code:`False`
        Determines whether atom positions are unwrapped.

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
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            groupings: Union[str, tuple[str]] = "atoms",
            n_chains: Union[int, tuple[int]] = None,
            n_monomers: Union[int, tuple[int]] = None, *, unwrap: bool = False,
            parallel: bool = False, verbose: bool = True, **kwargs) -> None:

        self._groups = [groups] if isinstance(groups, mda.AtomGroup) else groups
        self.universe = self._groups[0].universe
        super().__init__(self.universe.trajectory, parallel, verbose, **kwargs)

        self._dimensions = self.universe.dimensions
        if self._dimensions is not None:
            self._dimensions = self._dimensions[:3].copy()

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
                    emsg = (f"Invalid grouping '{gr}'. Valid "
                            "values: '" + "', '".join(GROUPINGS) + "'.")
                    raise ValueError(emsg)
            self._groupings = groupings

        if n_chains is None or n_monomers is None:
            self._internal = True
            self._n_chains = np.empty(self._n_groups, dtype=int)
            self._n_monomers = np.empty_like(self._n_chains)
            for i, ag in enumerate(self._groups):
                self._n_chains[i] = ag.segments.n_segments
                self._n_monomers[i] = ag.n_atoms // self._n_chains[i]
        else:
            self._internal = False
            if isinstance(n_chains, (int, np.integer)):
                self._n_chains = n_chains * np.ones(self._n_groups, dtype=int)
            elif self._n_groups == len(n_chains):
                self._n_chains = n_chains
            else:
                emsg = ("The shape of 'n_chains' is incompatible with "
                        "that of 'groups'.")
                raise ValueError(emsg)
            if isinstance(n_monomers, (int, np.integer)):
                self._n_monomers = n_monomers * np.ones(n_monomers, dtype=int)
            elif self._n_groups == len(n_monomers):
                self._n_monomers = n_monomers
            else:
                emsg = ("The shape of 'n_monomers' is incompatible "
                        "with that of 'groups'.")
                raise ValueError(emsg)

        self._unwrap = unwrap
        self._verbose = verbose

class Gyradius(_PolymerAnalysisBase):

    """
    Serial and parallel implementations to calculate the radius of
    gyration :math:`R_\\mathrm{g}` of a polymer.

    The radius of gyration is used to describe the dimensions of a
    polymer chain, and is defined as

    .. math::

        R_\\mathrm{g}=\\sqrt{
        \\frac{\\sum_i^N m_i\\|\\mathbf{r}_i
        -\\mathbf{R}_\\mathrm{com}\\|^2}{\\sum_i^N m_i}}

    where :math:`m_i` and :math:`\\mathbf{r}_i` are the mass and
    position, respectively, of particle :math:`i`, and
    :math:`\\mathbf{R}_\\mathrm{com}` is the center of mass.

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Groups of polymers to be analyzed.

        .. note::

           All polymers in each group must have the same chain length.

    groupings : `str` or array-like, default: :code:`"atoms"`
        Determines whether the centers of mass are used in lieu of
        individual atom positions. If `groupings` is a `str`, the same
        value is used for all `groups`.

        .. note::

           In a standard trajectory file, segments (or chains) contain
           residues (or molecules), and residues contain atoms. This
           heirarchy must be adhered to for this analysis module to
           function correctly. If your trajectory file does not contain
           the correct residue or segment information, provide the
           number of chains and chain lengths in `n_chains` and
           `n_monomers`, respectively.

        .. container::

           **Valid values**:

           * :code:`"atoms"`: Atom positions (generally or for
             coarse-grained simulations).
           * :code:`"residues"`: Residues' centers of mass (for
             atomistic simulations).

    n_chains : `int` or array-like, optional
        Number of chains :math:`M` in each polymer group. Must be
        provided if the trajectory does not adhere to the standard
        container heirarchy. If an `int` is provided, the same value is
        used for all groups.

        **Shape**: :math:`(N_\\mathrm{groups},)`.

    n_monomers : `int` or array-like, optional
        Number of monomers :math:`N` in each chain in each polymer
        group. Must be provided if the trajectory does not adhere to the
        standard container heirarchy. If an `int` is provided, the same
        value is used for all groups.

        **Shape**: :math:`(N_\\mathrm{groups},)`.

    components : `bool`, keyword-only, default: :code:`False`
        Specifies whether the components of the radii of gyration are
        calculated and returned instead.

    unwrap : `bool`, keyword-only, default: :code:`False`
        Determines whether atom positions are unwrapped.

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
        reference units for `results.gyradii`, call
        :code:`results.units["gyradii"]`.

    results.gyradii : `numpy.ndarray`
        Radii of gyration.

        **Shape**: :math:`(N_\\mathrm{groups},\\,N_\\mathrm{frames})` or
        :math:`(N_\\mathrm{groups},\\,N_\\mathrm{frames},\\,3)`.

        **Reference unit**: :math:`\\mathrm{Å}`.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            groupings: Union[str, tuple[str]] = "atoms",
            n_chains: Union[int, tuple[int]] = None,
            n_monomers: Union[int, tuple[int]] = None, *,
            components: bool = False, unwrap: bool = False,
            parallel: bool = False, verbose: bool = True, **kwargs):

        super().__init__(groups, groupings, n_chains, n_monomers,
                         unwrap=unwrap, parallel=parallel, verbose=verbose,
                         **kwargs)

        if self.universe.dimensions is None and unwrap:
            emsg = ("System dimensions were not found, but are "
                    "required when 'unwrap=True'.")
            raise ValueError(emsg)

        self._Ns_p = self._n_chains * self._n_monomers
        self._N_p = self._Ns_p.sum()
        self._slices = []
        _ = 0
        for N_p in self._Ns_p:
            self._slices.append(slice(_, _ + N_p))
            _ += N_p

        self._components = components

    def _prepare(self) -> None:

        if self._unwrap:

            # Navigate to first frame in analysis
            ts = self._sliced_trajectory[0]

            # Get system dimensions
            dimensions = ts.dimensions[:3]

            # Preallocate arrays to determine the number of periodic
            # boundary crossings for each entity
            self._positions_old = np.empty((self._N_p, 3))
            for ag, gr, s, M, N in zip(self._groups, self._groupings,
                                       self._slices, self._n_chains,
                                       self._n_monomers):
                if self._internal and gr == "residues":
                    ag.unwrap()
                    self._positions_old[s] = center_of_mass(ag, gr)
                else:
                    positions = unwrap_edge(
                        positions=ag.positions,
                        bonds=np.array([(i * N + j, i * N + j + 1)
                                        for i in range(M) for j in range(N - 1)]),
                        dimensions=dimensions,
                        masses=ag.masses
                    )
                    self._positions_old[s] = (
                        positions if gr == "atoms"
                        else center_of_mass(
                            positions=positions.reshape(M, N, -1, 3),
                            masses=ag.masses.reshape(M, N, -1)
                        )
                    )
            self._images = np.zeros((self._N_p, 3), dtype=int)

            # Store unwrapped entity positions in a shared memory array
            # for parallel analysis
            if self._parallel:
                self._positions = np.empty((self.n_frames, self._N_p, 3))
                for i, _ in enumerate(self._sliced_trajectory):
                    for ag, gr, s, M, N in zip(self._groups, self._groupings,
                                               self._slices, self._n_chains,
                                               self._n_monomers):
                        if self._internal and gr == "residues":
                            self._positions[i, s] = center_of_mass(ag, gr)
                        else:
                            self._positions[i, s] = (
                                ag.positions if gr == "atoms"
                                else center_of_mass(
                                    positions=ag.positions.reshape(M, N, -1, 3),
                                    masses=ag.masses.reshape(M, N, -1)
                                )
                            )

                    # Globally unwrap entity positions for correct
                    # centers of mass and radii of gyration
                    unwrap(
                        self._positions[i],
                        self._positions_old,
                        dimensions,
                        thresholds=dimensions / 2,
                        images=self._images
                    )

        # Preallocate arrays to hold radii of gyration
        if not self._parallel:
            shape = [self._n_groups, self.n_frames]
            if self._components:
                shape.append(3)
            self.results.gyradii = np.empty(shape)

        # Store reference units
        self.results.units = Hash({"gyradii": ureg.angstrom})

    def _single_frame(self) -> None:

        for i, (ag, gr, M, N, s) in enumerate(
                zip(self._groups, self._groupings, self._n_chains,
                    self._n_monomers, self._slices)
            ):

            # Store atom or center-of-mass positions in the current frame
            if self._internal and gr == "residues":
                positions = center_of_mass(ag, gr)
            else:
                positions = (
                    ag.positions if gr == "atoms"
                    else center_of_mass(
                        positions=ag.positions.reshape(M, N, -1, 3),
                        masses=ag.masses.reshape(M, N, -1)
                    )
                )

            # Globally unwrap entity positions for correct centers of
            # mass and radii of gyration
            if self._unwrap:
                dimensions = self._ts.dimensions[:3]
                unwrap(
                    positions,
                    self._positions_old[s],
                    dimensions,
                    thresholds=dimensions / 2,
                    images=self._images[s]
                )

            # Compute the radii of gyration
            self.results.gyradii[i, self._frame_index] \
                = radius_of_gyration(
                    grouping="segments",
                    positions=positions.reshape((M, N, 3)),
                    masses=ag.masses.reshape((M, N)),
                    components=self._components
                ).mean(axis=0)

    def _single_frame_parallel(
            self, index: int) -> tuple[int, np.ndarray[float]]:

        # Set current trajectory frame
        self._sliced_trajectory[index]

        # Preallocate array to store radii of gyration
        shape = [self._n_groups]
        if self._components:
            shape.append(3)
        gyradii = np.empty(shape)

        for i, (ag, gr, M, N, s) in enumerate(
                zip(self._groups, self._groupings, self._n_chains,
                    self._n_monomers, self._slices)
            ):

            # Calculate or get entity positions
            if self._unwrap:
                positions = self._positions[index, s]
            elif self._internal and gr == "residues":
                positions = center_of_mass(ag, gr)
            else:
                positions = (
                    ag.positions if gr == "atoms"
                    else center_of_mass(
                        positions=ag.positions.reshape(M, N, -1, 3),
                        masses=ag.masses.reshape(M, N, -1)
                    )
                )

            # Compute the radii of gyration
            gyradii[i] = radius_of_gyration(
                grouping="segments",
                positions=positions.reshape((M, N, 3)),
                masses=ag.masses.reshape((M, N)),
                components=self._components
            ).mean(axis=0)

        return index, gyradii

    def _conclude(self) -> None:

        # Consolidate parallel results and clean up memory by deleting
        # arrays that will not be reused
        if self._parallel:
            self._results = sorted(self._results)
            self.results.gyradii \
                = np.stack([r[1] for r in self._results], axis=1)

            del self._results
            if self._unwrap:
                del self._positions
        if self._unwrap:
            del self._positions_old, self._images

def correlation_fft(
        x: np.ndarray[Union[float, complex]],
        y: np.ndarray[Union[float, complex]] = None, /, axis: int = None, *,
        average: bool = False, double: bool = False, vector: bool = False
    ) -> np.ndarray[Union[float, complex]]:


    r"""
    Evaluates the autocorrelation functions (ACF)
    :math:`\mathrm{R_\mathbf{XX}}(\tau)` or cross-correlation functions
    (CCF) :math:`\mathrm{R_\mathbf{XY}}(\tau)` of time series
    :math:`\mathbf{X}(t)` and :math:`\mathbf{Y}(t)` using fast Fourier
    transforms (FFT).

    .. seealso::

       This function is an alias for
       :func:`mdcraft.algorithm.correlation.correlation_fft`.
    """

    return correlation.correlation_fft(x, y, axis, average=average,
                                       double=double, vector=vector)

def correlation_shift(
        x: np.ndarray[Union[float, complex]],
        y: np.ndarray[Union[float, complex]] = None, /, axis: int = None, *,
        average: bool = False, double: bool = False, vector: bool = False
    ) -> np.ndarray[Union[float, complex]]:

    r"""
    Evaluates the autocorrelation functions (ACF)
    :math:`\mathrm{R_\mathbf{XX}}(\tau)` or cross-correlation functions
    (CCF) :math:`\mathrm{R_\mathbf{XY}}(\tau)` of time series
    :math:`\mathbf{X}(t)` and :math:`\mathbf{Y}(t)` directly by using
    sliding windows.

    .. seealso::

       This function is an alias for
       :func:`mdcraft.algorithm.correlation.correlation_shift`.
    """

    return correlation.correlation_shift(x, y, axis, average=average,
                                         double=double, vector=vector)

def calculate_relaxation_time(
        times: np.ndarray[float], acf: np.ndarray[float]) -> float:

    r"""
    Calculates the orientational relaxation time :math:`\tau_\mathrm{r}`
    of polymers using end-to-end vector autocorrelation function (ACF)
    time series :math:`C_\mathrm{ee}(t)`.

    A stretched exponential function with :math:`\tau` and :math:`\beta`
    as coefficients,

    .. math::

       C_\mathrm{ee}(t)=\exp{\left[-(t/\tau)^\beta\right]}

    is fitted to the ACF time series, and the relaxation time is
    estimated using the first moment of :math:`C_\mathrm{ee}`,

    .. math::

       \tau_\mathrm{r}=\int_0^\infty C_\mathrm{ee}(t)\,dt
       =\frac{\tau}{\beta}\Gamma\left(\frac{1}{\beta}\right)

    where :math:`\Gamma` is the gamma function.

    Parameters
    ----------
    times : `numpy.ndarray`
        Changes in time :math:`t-t_0`.

        **Shape**: :math:`(N_\mathrm{frames},)`.

        **Reference unit**: :math:`\mathrm{ps}`.

    acf : `numpy.ndarray`
        End-to-end vector ACF :math:`C_\mathrm{ee}(t)`.

        **Shape**:
        :math:`(N_\mathrm{groups},\,N_\mathrm{bins},\,N_\mathrm{frames})`.

    Returns
    -------
    relaxation_time : `float`
        Average orientational relaxation time :math:`\tau_\mathrm{r}`.

        **Reference unit**: :math:`\mathrm{ps}`.
    """

    tau_r, beta = optimize.curve_fit(stretched_exp, times / times[1], acf,
                                     bounds=(0, np.inf))[0]
    return tau_r * times[1] * special.gamma((beta + 1) / beta)

class EndToEndVector(_PolymerAnalysisBase):

    """
    A serial implementation to calculate the end-to-end vector
    autocorrelation function (ACF) :math:`C_\\mathrm{ee}(t)` and
    estimate the orientational relaxation time :math:`\\tau_\\mathrm{r}`
    of a polymer.

    The end-to-end vector ACF is defined as

    .. math::

       C_\\mathrm{ee}(t)=\\frac{\\langle\\mathbf{R}_\\mathrm{ee}(t)
       \\cdot\\mathbf{R}_\\mathrm{ee}(0)\\rangle}
       {\\langle\\mathbf{R}_\\mathrm{ee}^2\\rangle}

    where :math:`\\mathbf{R}_\\mathrm{ee}=\\mathbf{r}_N-\\mathbf{r}_1`
    is the end-to-end vector.

    The orientational relaxation time can then be estimated by fitting
    a stretched exponential function

    .. math::

       C_\\mathrm{ee}=\\exp{\\left[-(t/\\tau)^\\beta\\right]}

    to the end-to-end vector ACF and evaluating

    .. math::

       \\tau_\\mathrm{r}=\\int_0^\\infty C_\\mathrm{ee}\\,dt
       =\\frac{\\tau}{\\beta}\\Gamma\\left(\\frac{1}{\\beta}\\right)

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Groups of polymers to be analyzed.

        .. note::

           All polymers in each group must have the same chain length.

    groupings : `str` or array-like, default: :code:`"atoms"`
        Determines whether the centers of mass are used in lieu of
        individual atom positions. If `groupings` is a `str`, the same
        value is used for all `groups`.

        .. note::

           In a standard trajectory file, segments (or chains) contain
           residues (or molecules), and residues contain atoms. This
           heirarchy must be adhered to for this analysis module to
           function correctly. If your trajectory file does not contain
           the correct residue or segment information, provide the
           number of chains and chain lengths in `n_chains` and
           `n_monomers`, respectively.

        .. container::

           **Valid values**:

           * :code:`"atoms"`: Atom positions (generally or for
             coarse-grained simulations).
           * :code:`"residues"`: Residues' centers of mass (for
             atomistic simulations).

    n_chains : `int` or array-like, optional
        Number of chains :math:`M` in each polymer group. Must be
        provided if the trajectory does not adhere to the standard
        container heirarchy. If an `int` is provided, the same value is
        used for all groups.

        **Shape**: :math:`(N_\\mathrm{groups},)`.

    n_monomers : `int` or array-like, optional
        Number of monomers :math:`N` in each chain in each polymer
        group. Must be provided if the trajectory does not adhere to the
        standard container heirarchy. If an `int` is provided, the same
        value is used for all groups.

        **Shape**: :math:`(N_\\mathrm{groups},)`.

    n_blocks : `int`, keyword-only, default: :code:`1`
        Number of blocks to split the trajectory into.

    dt : `float`, `openmm.unit.Quantity`, or `pint.Quantity`, \
    keyword-only, optional
        Time between frames :math:`\\Delta t`. While this is normally
        determined from the trajectory, the trajectory may not have the
        correct information if the data is in reduced units. For
        example, if the reduced timestep is :math:`0.01` and trajectory
        data was outputted every :math:`10,000` timesteps, then
        :math:`\\Delta t=100`.

        **Reference unit**: :math:`\\mathrm{ps}`.

    fft : `bool`, keyword-only, default: :code:`True`
        Determines whether fast Fourier transforms (FFT) are used to
        evaluate the ACFs.

    unwrap : `bool`, keyword-only, default: :code:`False`
        Determines whether atom positions are unwrapped.

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
        reference units for `results.times`, call
        :code:`results.units["times"]`.

    results.times : `numpy.ndarray`
        Changes in time :math:`t-t_0`.

        **Shape**: :math:`(N_\\mathrm{frames},)`.

        **Reference unit**: :math:`\\mathrm{ps}`.

    results.acf : `numpy.ndarray`
        End-to-end vector ACFs :math:`C_\\mathrm{ee}(t)`.

        **Shape**:
        :math:`(N_\\mathrm{groups},\\,N_\\mathrm{blocks},\\,N_\\mathrm{frames})`.

    results.relaxation_times : `numpy.ndarray`
        Average orientational relaxation times :math:`\\tau_\\mathrm{r}`.

        **Shape**: :math:`(N_\\mathrm{groups},\\,N_\\mathrm{blocks})`.

        **Reference units**: :math:`\\mathrm{ps}`.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            groupings: Union[str, tuple[str]] = "atoms",
            n_chains: Union[int, tuple[int]] = None,
            n_monomers: Union[int, tuple[int]] = None, *, n_blocks: int = 1,
            dt: Union[float, "unit.Quantity", Q_] = None, fft: bool = True,
            unwrap: bool = False, verbose: bool = True, **kwargs) -> None:

        # Disable parallel support for this analysis class
        if "parallel" in kwargs:
            wmsg = ("The 'EndToEndVector' analysis class does not "
                    "support multithreading.")
            warnings.warn(wmsg)
            del kwargs["parallel"]

        super().__init__(groups, groupings, n_chains, n_monomers,
                         unwrap=unwrap, verbose=verbose, **kwargs)

        self._M = self._n_chains.sum()
        self._slices = []
        _ = 0
        for M in self._n_chains:
            self._slices.append(slice(_, _ + M))
            _ += M

        self._n_blocks = n_blocks
        self._dt = strip_unit(dt, "ps")[0] or self._trajectory.dt
        self._fft = fft

    def _prepare(self) -> None:

        # Ensure frames are evenly spaced and proceed forward in time
        if hasattr(self._sliced_trajectory, "frames"):
            dfs = np.diff(self._sliced_trajectory.frames)
            if (df := dfs[0]) <= 0 or not np.allclose(dfs, df):
                emsg = ("The selected frames must be evenly spaced and "
                        "proceed forward in time.")
                raise ValueError(emsg)
        elif (df := self.step) <= 0:
            raise ValueError("The analysis must proceed forward in time.")

        # Determine number of frames used when the trajectory is split
        # into blocks
        self._n_frames_block = self.n_frames // self._n_blocks
        self._n_frames = self._n_blocks * self._n_frames_block
        if (extra_frames := self.n_frames - self._n_frames) > 0:
            wmsg = (f"The trajectory is not divisible into {self._n_blocks:,} "
                    f"blocks, so the last {extra_frames:,} frame(s) will be "
                    "discarded. To maximize performance, set appropriate "
                    "starting and ending frames in run() so that the number "
                    "of frames to be analyzed is divisible by the number of "
                    "blocks.")
            warnings.warn(wmsg)

        # Preallocate arrays to store end-to-end vectors
        self._e2e = np.empty((self.n_frames, self._M, 3))

        if self._unwrap:

            # Navigate to first frame in analysis
            self._sliced_trajectory[0]

            # Preallocate arrays to determine the number of periodic
            # boundary crossings for the first and last monomer in each
            # chain
            self._positions_end_old = np.empty((self._M, 2, 3))
            for ag, gr, s, M, N in zip(self._groups, self._groupings,
                                       self._slices, self._n_chains,
                                       self._n_monomers):
                if self._internal and gr == "residues":
                    ag.unwrap()
                    self._positions_end_old[s] = np.stack(
                        [center_of_mass(s.residues[[0, -1]].atoms, "residues")
                         for s in ag.segments]
                    )
                else:
                    positions = unwrap_edge(
                        positions=ag.positions,
                        bonds=np.array([(i * N + j, i * N + j + 1)
                                        for i in range(M) for j in range(N - 1)]),
                        dimensions=self._dimensions,
                        masses=ag.masses
                    ).reshape(M, N, -1, 3)[:, (0, -1)]
                    self._positions_end_old[s] = (
                        positions[:, :, 0] if gr == "atoms"
                        else center_of_mass(
                            positions=positions,
                            masses=ag.masses.reshape(M, N, -1)[:, (0, -1)]
                        )
                    )
            self._images = np.zeros((self._M, 2, 3), dtype=int)
            self._thresholds = self._dimensions / 2

        # Preallocate array to store end-to-end vector ACFs
        self.results.acf = np.empty(
            (self._n_groups, self._n_blocks, self._n_frames_block)
        )

        # Store time changes
        self.results.times = df * self._dt * np.arange(self._n_frames_block)

        # Store reference units
        self.results.units = {"times": ureg.picosecond}

    def _single_frame(self) -> None:

        for ag, gr, s, M, N in zip(self._groups, self._groupings,
                                  self._slices, self._n_chains,
                                  self._n_monomers):

            # Store ending monomer or center-of-mass positions in the
            # current frame
            if self._internal and gr == "residues":
                positions_end = np.stack(
                    center_of_mass(s.residues[[0, -1]].atoms, "residues")
                    for s in ag.segments
                )
            else:
                positions_end = ag.positions.reshape(M, N, -1, 3)[:, (0, -1)]
                positions_end = (
                    positions_end[:, :, 0] if gr == "atoms"
                    else center_of_mass(
                        positions=positions_end,
                        masses=ag.masses.reshape(M, N, -1)[:, (0, -1)]
                    )
                )

            # Globally unwrap entity positions for correct end-to-end
            # vectors
            if self._unwrap:
                unwrap(
                    positions_end,
                    self._positions_end_old[s],
                    self._dimensions,
                    thresholds=self._thresholds,
                    images=self._images[s]
                )

            # Compute the end-to-end vectors
            self._e2e[self._frame_index, s] \
                = np.diff(positions_end, axis=1)[:, 0]

    def _conclude(self) -> None:

        # Clean up memory by deleting arrays that will not be reused
        if self._unwrap:
            del self._positions_end_old, self._images, self._thresholds

        # Compute end-to-end vector ACFs
        _acf = correlation_fft if self._fft else correlation_shift
        for i, (s, M) in ProgressBar(enumerate(zip(self._slices,
                                                   self._n_chains))):
            self.results.acf[i] = _acf(
                (self._e2e[:, s]
                 / np.linalg.norm(self._e2e[:, s], axis=-1, keepdims=True))
                .reshape(self._n_blocks, -1, M, 3),
                average=True,
                vector=True
            )

    def calculate_relaxation_times(self) -> None:

        """
        Calculates the orientational relaxation times.
        """

        if not hasattr(self.results, "acf"):
            emsg = ("Call EndToEndVector.run() before "
                    "EndToEndVector.calculate_relaxation_times().")
            raise RuntimeError(emsg)

        self.results.relaxation_times = np.empty((self._n_groups,
                                                  self._n_blocks))
        self.results.units["relaxation_times"] = ureg.picosecond

        for i, block in enumerate(self.results.acf):
            for j, acf in enumerate(block):
                valid = np.where(acf >= 0)[0]
                self.results.relaxation_times[i, j] = calculate_relaxation_time(
                    self.results.times[valid], acf[valid]
                )

class SingleChainStructureFactor(NumbaAnalysisBase, _PolymerAnalysisBase):

    """
    Serial and parallel implementations to calculate the single-chain
    structure factor :math:`S_\\mathrm{sc}(q)` of a homopolymer.

    It is defined as

    .. math::

       S_{\\mathrm{sc}}(q)=\\frac{1}{MN}\\left\\langle
       \\sum_{m=1}^M\\sum_{j=1}^{N}\\sum_{k=1}^{N}\\exp{
       [i\\mathbf{q}\\cdot(\\mathbf{r}_j-\\mathbf{r}_k)]}\\right\\rangle

    where :math:`M` is the number of chains, :math:`N` is the chain
    length, :math:`\\mathbf{q}` and :math:`q` are the scattering
    wavevector and its magnitude, respectively, and
    :math:`\\mathbf{r}_i` is the position of the :math:`i`-th monomer.

    .. container::

       The single-chain structure factor reveals information about the
       characteristic length scales of the polymer:

       * In the Guinier regime (:math:`qR_g\\ll1`),
         :math:`S_{\\mathrm{sc}}(q)^{-1}\\approx N(1-(qR_g)^2/3)`
         can be used to determine the radius of gyration :math:`R_g`.
       * In the Porod regime (:math:`qR_g\\gg1`),
         :math:`S_{\\mathrm{sc}}(q)=1` since the only contribution is
         the self-scattering of the monomers.
       * In the intermediate regime, the slope :math:`s` of the log-log
         plot of :math:`S_{\\mathrm{sc}}(q)` is related to the scaling
         exponent :math:`\\nu` via :math:`\\nu=-1/s`.

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Groups of polymers to be analyzed.

        .. note::

           All polymers in each group must have the same chain length.

    groupings : `str` or array-like, default: :code:`"atoms"`
        Determines whether the centers of mass are used in lieu of
        individual atom positions. If `groupings` is a `str`, the same
        value is used for all `groups`.

        .. note::

           In a standard trajectory file, segments (or chains) contain
           residues (or molecules), and residues contain atoms. This
           heirarchy must be adhered to for this analysis module to
           function correctly. If your trajectory file does not contain
           the correct residue or segment information, provide the
           number of chains and chain lengths in `n_chains` and
           `n_monomers`, respectively.

        .. container::

           **Valid values**:

           * :code:`"atoms"`: Atom positions (generally or for
             coarse-grained simulations).
           * :code:`"residues"`: Residues' centers of mass (for
             atomistic simulations).

    n_chains : `int` or array-like, optional
        Number of chains :math:`M` in each polymer group. Must be
        provided if the trajectory does not adhere to the standard
        container heirarchy. If an `int` is provided, the same value is
        used for all groups.

        **Shape**: :math:`(N_\\mathrm{groups},)`.

    n_monomers : `int` or array-like, optional
        Number of monomers :math:`N` in each chain in each polymer
        group. Must be provided if the trajectory does not adhere to the
        standard container heirarchy. If an `int` is provided, the same
        value is used for all groups.

        **Shape**: :math:`(N_\\mathrm{groups},)`.

    form : `str`, keyword-only, default: :code:`"exp"`
        Expression used to evaluate the single-chain structure factors.

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
        Number of points :math:`n_\\mathrm{points}` in the scattering
        wavevector grid to generate

        .. math::

           \\mathbf{q}=2\\pi\\left(\\frac{a}{L_x},\\,\\frac{b}{L_y},\\,
           \\frac{c}{L_z}\\right)

        where :math:`a`, :math:`b`, and :math:`c` are integers from
        :math:`0` up to :math:`n_\\mathrm{points}-1`.

        Additional wavevectors can be introduced via `n_surfaces` and
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
        Maximum wavenumber :math:`q_\\mathrm{max}`.

        **Reference unit**: :math:`\\mathrm{Å}^{-1}`.

    wavevectors : array-like, `openmm.unit.Quantity`, or `pint.Quantity`, \
    keyword-only, optional
        Scattering wavevectors for which to compute structure factors.
        Has precedence over `n_points`, `n_surfaces`, and
        `n_surface_points` if specified.

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

    results.wavenumbers : `numpy.ndarray`
        Wavenumbers :math:`q`.

        **Shape**: :math:`(N_q,)`.

        **Reference unit**: :math:`\\mathrm{Å}^{-1}`.

    results.scsf : `numpy.ndarray`
        Single-chain structure factors :math:`S_\\mathrm{sc}(q)`.

        **Shape**: :math:`(N_q,)`.
    """

    def __init__(
            self, groups: mda.AtomGroup, groupings: str = "atoms",
            n_chains: int = None, n_monomers: int = None, *, form: str = "exp",
            dimensions: Union[np.ndarray[float], "unit.Quantity", Q_] = None,
            n_points: int = 32, n_surfaces: int = None,
            n_surface_points: int = 8,
            q_max: Union[float, "unit.Quantity", Q_] = None,
            wavevectors: np.ndarray[float] = None, sort: bool = True,
            unique: bool = True, unwrap: bool = False, parallel: bool = False,
            verbose: bool = True, **kwargs) -> None:

        # Specify 'parallel=False' to use the NumbaAnalysisBase.run method
        # instead of the ParallelAnalysisBase.run method
        _PolymerAnalysisBase.__init__(groups, groupings, n_chains, n_monomers,
                                      unwrap=unwrap, parallel=False,
                                      verbose=verbose, **kwargs)

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

        self._njit = lambda s: numba.njit(s, fastmath=True, parallel=parallel)
        self._delta_fourier_transform_sum = self._njit(
            "c16[:](f8[:,:],f8[:,:])"
        )(structure.delta_fourier_transform_sum)
        self._ssf_trigonometric = self._njit(
            "f8[:](f8[:,:])"
        )(structure.ssf_trigonometric)
        self._psf_trigonometric = self._njit(
            "f8[:](f8[:,:],f8[:,:])"
        )(structure.psf_trigonometric)
        self._inner = (accelerated.numba_inner_parallel if parallel
                       else accelerated.numba_inner)

        self._form = form
        self._sort = sort
        self._unique = unique
        self._verbose = verbose

    def _prepare(self) -> None:

        # Preallocate array to store single-chain structure factors
        if not self._parallel:
            self.results.scsf = np.zeros((self._n_groups,
                                          len(self._wavenumbers)))

        # Determine the unique wavenumbers
        self.results.wavenumbers = np.unique(self._wavenumbers.round(11))

        # Store reference units
        self.results.units = Hash({"results.wavenumbers": ureg.angstrom ** -1})

    def _single_frame(self) -> None:

        for ig, (ag, gr, M, N) in enumerate(
                zip(self._groups, self._groupings, self._n_chains,
                    self._n_monomers)
            ):

            # Get entity positions in the current frame
            if self._internal and gr == "residues":
                positions = center_of_mass(ag, gr)
            else:
                positions = (
                    ag.positions if gr == "atoms"
                    else center_of_mass(
                        positions=ag.positions.reshape(M, N, -1, 3),
                        masses=ag.masses.reshape(M, N, -1)
                    )
                )

            # Calculate single-chain structure factor contributions
            if self._form == "exp":
                for chain in positions.reshape((M, N, 3)):
                    rhos = self._delta_fourier_transform_sum(self._wavevectors,
                                                             chain)
                    self.results.scsf[ig] += (rhos * rhos.conj()).real
            elif self._form == "trig":
                for chain in positions.reshape((M, N, 3)):
                    self.results.scsf[ig] += self._ssf_trigonometric(
                        self._inner(self._wavevectors, chain)
                    )

    def _conclude(self) -> None:

        # Normalize the single-chain structure factor by the number of
        # monomers and timesteps
        self.results.scsf /= self._n_chains * self._n_monomers * self.n_frames

        # Combine values sharing the same wavenumber, if desired
        if self._unique:
            self.results.scsf = np.hstack(
                [self.results.scsf[:, np.isclose(q, self._wavenumbers)]
                 .mean(axis=1, keepdims=True)
                for q in self.results.wavenumbers]
            )

        # Sort the results by wavenumber, if desired
        if self._sort:
            order = np.argsort(self.results.wavenumbers)
            self.results.wavenumbers = self.results.wavenumbers[order]
            self.results.scsf = self.results.scsf[:, order]