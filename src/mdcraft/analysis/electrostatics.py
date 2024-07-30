"""
Electrostatic properties
========================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains classes to quantify electrostatic properties, such
as the instantaneous dipole moment.
"""

from numbers import Real
from typing import Union

import MDAnalysis as mda
import numpy as np

from .base import DynamicAnalysisBase
from .. import FOUND_OPENMM, Q_, ureg
from ..algorithm.topology import unwrap
from ..algorithm.unit import is_unitless, strip_unit

if FOUND_OPENMM:
    from openmm import unit

def calculate_relative_permittivity(
        M: np.ndarray[float], temperature: float, volume: float, *,
        reduced: bool = False) -> float:

    r"""
    Calculates the relative permittivity (or static dielectric constant)
    :math:`\varepsilon_\mathrm{r}` of a medium using dipole moments
    :math:`\mathbf{M}`.

    The dipole moment fluctuation formula [1]_ relates the relative
    permittivity to the dipole moment via

    .. math::

       \varepsilon_\mathrm{r}=1+\frac{\overline{\langle\mathbf{M}^2\rangle
       -\langle\mathbf{M}\rangle^2}}{3\varepsilon_0 Vk_\mathrm{B}T}

    where the angular brackets :math:`\langle\,\cdot\,\rangle` denote
    the ensemble average, the overline signifies the spatial average,
    :math:`\varepsilon_0` is the vacuum permittivity,
    :math:`k_\mathrm{B}` is the Boltzmann constant, and :math:`T` is
    the system temperature.

    .. note::

       If residues (molecules) in your system have net charges, the
       dipole moments must be made position-independent by subtracting
       the product of the net charges and the centers of mass or
       geometry.

    Parameters
    ----------
    M : array-like
        Dipole moment vectors :math:`\mathbf{M}`.

        **Shape**: :math:`(N_\mathrm{frames},\,3)`.

        **Reference unit**: :math:`\mathrm{e\cdotÅ}`.

    temperature : `float`
        System temperature :math:`T`.

        .. note::

           If :code:`reduced=True`, `temperature` should be equal to the
           energy scale. When the Lennard-Jones potential is used, it
           generally means that :math:`T^*=1`, or `temperature=1`.

        **Reference unit**: :math:`\mathrm{K}`.

    volume : `float`
        System volume :math:`V`.

        **Reference unit**: :math:`\mathrm{Å^3}`.

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units.

    Returns
    -------
    dielectric : `float`
        Relative permittivity (or static dielectric constant).

    References
    ----------
    .. [1] Neumann, M. Dipole Moment Fluctuation Formulas in Computer
       Simulations of Polar Systems. *Molecular Physics* **1983**,
       *50* (4), 841–858. https://doi.org/10.1080/00268978300102721.
    """

    conversion_factor = 4 * np.pi if reduced else (
        1 * ureg.elementary_charge ** 2
        / (ureg.vacuum_permittivity * ureg.angstrom
           * ureg.boltzmann_constant * ureg.kelvin)
    ).m_as(ureg.dimensionless)

    return 1 + (conversion_factor * (M ** 2 - M.mean(axis=0) ** 2).mean()
                / (volume * temperature))

class DipoleMoment(DynamicAnalysisBase):

    """
    Serial and parallel implementations to calculate the dipole moment
    :math:`\\mathbf{M}`.

    For a system with :math:`N` atoms or molecules, the dipole moment is
    given by

    .. math::

       \\mathbf{M}=\\sum_i^{N}q_i\\mathbf{z}_i

    where :math:`q_i` and :math:`\\mathbf{z}_i` are the charge and
    position of entity :math:`i`.

    The dipole moment can be used to estimate the relative permittivity
    (or static dielectric constant) via the dipole moment fluctuation
    formula [1]_:

    .. math::

       \\varepsilon_\\mathrm{r}=1+\\frac{\\overline{
       \\langle\\mathbf{M}^2\\rangle-\\langle\\mathbf{M}\\rangle^2
       }}{3\\varepsilon_0 Vk_\\mathrm{B}T}

    where the angular brackets :math:`\\langle\,\\cdot\,\\rangle` denote
    the ensemble average, the overline signifies the time average,
    :math:`\\varepsilon_0` is the vacuum permittivity,
    :math:`k_\\mathrm{B}` is the Boltzmann constant, and :math:`T` is
    the temperature.

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Groups of atoms for which the dipole moments are calculated.

        .. note::

           If :code:`neutralize=True` or :code:`unwrap=True`, all atoms
           of any particular molecule must belong to the same atom
           group.

    charges : array-like, `openmm.unit.Quantity`, or \
    `pint.Quantity`, keyword-only,optional
        Charges :math:`q_i` for the atoms in the
        :math:`N_\\mathrm{groups}` groups in `groups`. If not provided,
        it should be available in and will be retrieved from the main
        :class:`MDAnalysis.core.universe.Universe` object.

        **Shape**: :math:`(N_\\mathrm{groups},)` array of real numbers
        or :math:`(N_i,)` arrays, where :math:`N_i` is the number of
        atoms in group :math:`i`.

        **Reference unit**: :math:`\\mathrm{e}`.

    dim_scales : `float` or array-like, keyword-only, optional
        Scale factors for the system dimensions. If an `int` is
        provided, the same value is used for all axes.

        **Shape**: :math:`(3,)`.

    average : `bool`, keyword-only, default: :code:`False`
        Determines whether the dipole moment vectors and volumes are
        averaged over the :math:`N_\\mathrm{frames}` analysis frames.

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units. Only affects
        :meth:`calculate_relative_permittivity` calls.

    neutralize : `bool`, keyword-only, default: :code:`False`
        Specifies whether net charges of molecules are subtracted at the
        center of mass. Must be enabled if your system contains
        molecules with net charges and you want to calculate the
        relative permittivity, but should be disabled if you are only
        interested in the dipole moment.

        .. note::

           The topology must contain residue (molecule) information.

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
        information describing the simulation system.

    results.units : `dict`
        Reference units for the results. For example, to get the
        reference units for `results.times`, call
        :code:`results.units["times"]`.

    results.times : `numpy.ndarray`
        Times :math:`t`. Only available if :code:`average=False`.

        **Shape**: :math:`(N_\\mathrm{frames},)`.

        **Reference unit**: :math:`\\mathrm{ps}`.

    results.dipoles : `numpy.ndarray`
        Dipole moment vectors :math:`\\mathbf{M}`.

        **Shape**: :math:`(N_\\mathrm{groups},\\,3)` or
        :math:`(N_\\mathrm{frames},\\,N_\\mathrm{groups},\\,3)`.

        **Reference unit**: :math:`\\mathrm{e\\cdotÅ}`.

    results.volumes : `numpy.ndarray`
        System volumes :math:`V`.

        **Shape**: :math:`(N_\\mathrm{frames},)`.

        **Reference unit**: :math:`\\mathrm{Å^3}`.

    results.dielectrics : `numpy.ndarray`
        Relative permittivity (or static dielectric constant)
        :math:`\\varepsilon_\\mathrm{r}` in each dimension.

        **Shape**: :math:`(3,)`.
        
    Example
    --------
    First, this analysis class must be imported:

    >>> from mdcraft.analysis.electrostatics import DipoleMoment

    Then, after loading a simulation trajectory:

    >>> universe = mda.Universe("simulation.nc", "topology.cif")

    We must then select the atom-groups to be analyzed (typically the solvent):

    >>> ag = universe.select_atoms("resname SOL")
    
    The `DipoleMoment` class can be instantiated with the selected atom-groups, remembering to specify the charges:
    
    >>> dpm = DipoleMoment(ag, charges=ag.charges)
    >>> dpm.run()

    The results can be obtained under the `results` attribute:
    
    >>> dpm.results.dipoles
    
    To calculate the relative permittivity, the temperature must be provided:
    
    >>> dpm.calculate_relative_permittivity(300)
    
    These results can be saved to a file using the `save` method:
        
    >>> dpm.save("dpm")

    References
    ----------
    .. [1] Neumann, M. Dipole Moment Fluctuation Formulas in Computer
       Simulations of Polar Systems. *Molecular Physics* **1983**,
       *50* (4), 841–858. https://doi.org/10.1080/00268978300102721.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            charges: Union[np.ndarray[float], "unit.Quantity", Q_] = None,
            dim_scales: Union[float, tuple[float]] = 1, average: bool = False,
            reduced: bool = False, neutralize: bool = False,
            unwrap: bool = False, parallel: bool = False, verbose: bool = True,
            **kwargs) -> None:

        self._groups = [groups] if isinstance(groups, mda.AtomGroup) else groups
        self._n_groups = len(self._groups)
        self.universe = self._groups[0].universe
        if self.universe.dimensions is None:
            raise ValueError("Trajectory does not contain system "
                             "dimension information.")
        super().__init__(self.universe.trajectory, parallel, verbose, **kwargs)

        if (isinstance(dim_scales, Real)
            or (len(dim_scales) == 3
                and all(isinstance(f, Real) for f in dim_scales))):
            self._dim_scales = dim_scales
        else:
            emsg = ("'dim_scales' must be a floating-point number "
                    "or an array with shape (3,).")
            raise ValueError(emsg)

        self._Ns = np.fromiter((ag.n_atoms for ag in self._groups), dtype=int,
                               count=self._n_groups)
        self._N = self._Ns.sum()
        self._slices = []
        _ = 0
        for N in self._Ns:
            self._slices.append(slice(_, _ + N))
            _ += N

        if charges is not None:
            charges = list(charges)
            if len(charges) == self._n_groups:
                for i, (ag, q) in enumerate(zip(self._groups, charges)):
                    charges[i] = strip_unit(q, "e")[0]
                    if isinstance(charges[i], Real):
                        charges[i] *= np.ones(ag.n_atoms)
                    elif ag.n_atoms != len(charges[i]):
                        emsg = ("The number of charges in "
                                f"'charges[{i}]' is not equal to the "
                                "number of atoms in the corresponding "
                                "group.")
                        raise ValueError(emsg)
                self._charges = charges
            else:
                emsg = ("The number of group charge arrays is not "
                        "equal to the number of groups.")
                raise ValueError(emsg)
        elif hasattr(self.universe.atoms, "charges"):
            self._charges = [ag.charges for ag in self._groups]
        else:
            raise ValueError("The topology has no charge information.")

        self._all_neutral = np.allclose(
            self.universe.atoms.total_charge(compound="residues"), 0,
            atol=1e-6
        )

        self._average = average
        self._neutralize = neutralize
        self._reduced = reduced
        self._unwrap = unwrap
        self._verbose = verbose

    def _prepare(self) -> None:

        if self._unwrap:

            # Navigate to first frame in analysis
            ts = self._sliced_trajectory[0]

            # Get (scaled) system dimensions
            dimensions = self._dim_scales * ts.dimensions[:3]

            # Preallocate arrays to determine the number of periodic
            # boundary crossings for each atom
            self._positions_old = np.empty((self._N, 3))
            for ag, s in zip(self._groups, self._slices):

                # Locally unwrap molecules at the edge of the simulation
                # system
                self._positions_old[s] = ag.unwrap()

            self._images = np.zeros((self._N, 3), dtype=int)

            # Store unwrapped particle positions in a shared memory array
            # for parallel analysis
            if self._parallel:
                self._positions = np.empty((self.n_frames, self._N, 3))
                for i, _ in enumerate(self._sliced_trajectory):
                    for ag, s in zip(self._groups, self._slices):
                        self._positions[i, s] = ag.positions

                    # Globally unwrap atom positions
                    unwrap(
                        self._positions[i],
                        self._positions_old,
                        dimensions,
                        thresholds=dimensions / 2,
                        images=self._images
                    )

        # Store reference units
        self.results.units = {"dipoles": ureg.elementary_charge * ureg.angstrom,
                              "volumes": ureg.angstrom ** 3}

        # Preallocate arrays to store dipole moments and system volumes
        self.results.dipoles = np.empty((self.n_frames, self._n_groups, 3))
        self.results.volumes = np.empty(self.n_frames)

        # Store time information, if necessary
        if not self._average:
            self.results.times = np.fromiter(
                (ts.time for ts in self._sliced_trajectory),
                dtype=float,
                count=self.n_frames
            )
            self.results.units["times"] = ureg.picosecond

        # Preallocate array to hold atom positions for a given frame
        # (so that one doesn't have to be recreated each frame)
        if not self._parallel:
            self._positions = np.empty((self._N, 3))

    def _single_frame(self) -> None:

        # Get (scaled) system dimensions
        if self._unwrap:
            dimensions = self._dim_scales * self._ts.dimensions[:3]

        for i, (ag, s, q) in enumerate(zip(self._groups, self._slices,
                                           self._charges)):

            # Store atom positions in the current frame
            self._positions[s] = ag.positions

            # Globally unwrap atom positions
            if self._unwrap:
                unwrap(
                    self._positions[s],
                    self._positions_old[s],
                    dimensions,
                    thresholds=dimensions / 2,
                    images=self._images[s]
                )

            # Subtract the net charge at the center of mass of each
            # molecule, if necessary
            if self._neutralize:
                q -= q * np.concatenate([r.atoms.masses / r.mass
                                         for r in ag.residues])

            # Calculate the dipole moment for the current frame
            self.results.dipoles[self._frame_index, i] = q @ self._positions[s]

        # Store the system volume for the current frame
        self.results.volumes[self._frame_index] \
            = np.prod(self._dim_scales) * self._ts.volume

    def _single_frame_parallel(
            self, index: int) -> tuple[int, float, np.ndarray[float]]:

        # Set current trajectory frame
        ts = self._sliced_trajectory[index]

        # Preallocate arrays to hold
        dipoles = np.empty((self._n_groups, 3))

        for i, (ag, s, q) in enumerate(zip(self._groups, self._slices,
                                           self._charges)):

            # Get atom positions in the current frame
            positions = (self._positions[index, s] if self._unwrap
                         else ag.positions)

            # Subtract the net charge at the center of mass of each
            # molecule, if necessary
            if self._neutralize:
                q -= q * np.concatenate([r.atoms.masses / r.mass
                                         for r in ag.residues])

            # Calculate the dipole moment for the current frame
            dipoles[i] = q @ positions

        return index, np.prod(self._dim_scales) * ts.volume, dipoles

    def _conclude(self) -> None:

        # Consolidate parallel results and clean up memory by deleting
        # arrays that will not be reused
        if self._parallel:
            self._results = sorted(self._results)
            self.results.volumes = np.fromiter((r[1] for r in self._results),
                                               dtype=float, count=self.n_frames)
            self.results.dipoles = np.stack([r[2] for r in self._results])

            del self._results
            if self._unwrap:
                del self._positions
        else:
            del self._positions
        if self._unwrap:
            del self._positions_old, self._images

        # Average results, if requested
        if self._average:
            self.results.dipoles = self.results.dipoles.mean(axis=0)
            self.results.volumes = self.results.volumes.mean()

    def calculate_relative_permittivity(
            self, temperature: Union[float, "unit.Quantity", Q_]) -> None:

        r"""
        Calculates the relative permittivity (or static dielectric
        constant) :math:`\varepsilon_\mathrm{r}` of a medium using
        dipole moments :math:`\mathbf{M}`.

        Parameters
        ----------
        temperature : `float`, `openmm.unit.Quantity`, or `pint.Quantity`
            System temperature :math:`T`.

            .. note::

               If :code:`reduced=True` was set in the
               :class:`DipoleMoment` constructor, `temperature` should
               be equal to the energy scale. When the Lennard-Jones
               potential is used, it generally means that
               :math:`T^*=1`, or `temperature=1`.

            **Reference unit**: :math:`\mathrm{K}`.
        """

        if self._average:
            emsg = ("Cannot compute relative permittivity using the"
                    "averaged dipole moment.")
            raise RuntimeError(emsg)
        elif not self._all_neutral and not self._neutralize:
            emsg = ("Cannot compute relative permittivity for a "
                    "non-neutral system or a system with ions unless "
                    "the net charge is subtracted at the center of "
                    "mass of each molecule carrying a net charge.")
            raise RuntimeError(emsg)

        if self._reduced and not is_unitless(temperature):
            emsg = ("'temperature' cannot have units when reduced=True.")
            raise ValueError(emsg)
        temperature = strip_unit(temperature, "K")[0]

        dipoles = self.results.dipoles
        if self._n_groups > 1:
            dipoles = dipoles.sum(axis=dipoles.ndim - 2)
        self.results.dielectrics = calculate_relative_permittivity(
            dipoles,
            temperature,
            self.results.volumes.mean(),
            reduced=self._reduced
        )