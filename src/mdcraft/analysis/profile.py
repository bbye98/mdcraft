"""
Linear profiles
===============
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains classes to compute number density and charge
density profiles and related quantities, like surface charge densities
and potential profiles.
"""

import logging
from numbers import Real
from typing import Any, Union
import warnings

import MDAnalysis as mda
import numpy as np
from scipy import integrate, sparse

from .base import Hash, DynamicAnalysisBase
from .. import FOUND_OPENMM, Q_, ureg
from ..algorithm.accelerated import (histogram_bin_edges_1d_int,
                                     histogram_1d_int_1d)
from ..algorithm.molecule import center_of_mass
from ..algorithm.topology import unwrap, wrap
from ..algorithm.unit import is_unitless, strip_unit

if FOUND_OPENMM:
    from openmm import unit

_ELECTROSTATIC_CONVERSION_FACTORS = (
    (1 * ureg.elementary_charge
     / (ureg.vacuum_permittivity * ureg.angstrom)).m_as(ureg.volt),
    4 * np.pi
)

def calculate_surface_charge_density(
        bins: np.ndarray[float], charge_density_profile: np.ndarray[float],
        dielectric: float = None, *, L: float = None,
        dV: Union[float, np.ndarray[float]] = None, reduced: bool = False
    ) -> Union[float, np.ndarray[float]]:

    r"""
    Calculates the surface charge density :math:`\sigma_q` using the
    charge density profile :math:`\rho_q(z)`.

    The surface charge density is given by

    .. math::

       \sigma_q=\frac{1}{L}\left(
       \varepsilon_0\varepsilon_\mathrm{r}\Delta V
       -\int_0^L z\rho_q(z)\,\mathrm{d}z\right)

    where :math:`\varepsilon_0` and :math:`\varepsilon_\mathrm{r}` are
    the vacuum and relative permittivities, respectively,
    :math:`\Delta V` is the potential difference, and :math:`L` is the
    system dimension.

    The first (static) term accounts for the applied potential across
    the dielectric medium, while the second (polarization) term accounts
    for the distribution of charged species.

    Parameters
    ----------
    bins : array-like
        Histogram bin centers :math:`z` for the charge density profile
        in `charge_density_profile`.

        **Shape**: :math:`(N_\mathrm{bins},)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    charge_density_profile : array-like
        Charge density profile :math:`\rho_q(z)`.

        **Shape**: :math:`(N_\mathrm{bins},)` or
        :math:`(N_\mathrm{frames},\,N_\mathrm{bins})`.

        **Reference unit**: :math:`\mathrm{e/Å}^{-3}`.

    dielectric : `float`, optional
        Relative permittivity or static dielectric constant
        :math:`\varepsilon_\mathrm{r}`. Required if `dV` is provided.

    L : `float`, keyword-only, optional
        System size :math:`L` in the dimension that `bins` and
        `charge_density_profile` were calculated in. If not specified,
        it is determined using the first and last values in `bins`,
        assuming that the bin centers are uniformly spaced.

        **Reference unit**: :math:`\mathrm{Å}`.

    dV : `float` or array-like, keyword-only, optional
        Potential difference :math:`\Delta V` across the system
        dimension specified in `axis`. If not provided, only the
        polarizable part (second term) of the surface charge density is
        calculated.

        **Shape**: Scalar or :math:`(N_\mathrm{frames},)`.

        **Reference unit**: :math:`\mathrm{V}`.

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units.

    Returns
    -------
    surface_charge_density : `numpy.ndarray`
        Surface charge density :math:`\sigma_q`.

        **Shape**: Scalar or :math:`(N_\mathrm{frames},)`.

        **Reference unit**: :math:`\mathrm{e/Å}^2`.
    """

    # Check input array shapes
    if bins.ndim != 1:
        raise ValueError("'bins' must be a one-dimensional array.")
    if charge_density_profile.ndim not in {1, 2}:
        emsg = ("'charge_density_profile' must be a one- or "
                "two-dimensional array.")
        raise ValueError(emsg)
    if bins.shape[0] != charge_density_profile.shape[-1]:
        emsg = ("'bins' and 'charge_density_profile' must have the same"
                "length in the last dimension.")
        raise ValueError(emsg)
    if dV is not None:
        if dielectric is None:
            raise ValueError("'dielectric' must be specified when 'dV' is.")
        if not isinstance(dV, Real) \
                and dV.shape[0] != charge_density_profile.shape[0]:
            emsg = ("The number of potential differences in 'dVs' must "
                    "be equal to the number of frames in "
                    "'charge_density_profile'.")
            raise ValueError(emsg)

    # Determine system size, if not provided
    dz = bins[1] - bins[0]
    if (uniform := np.allclose(np.diff(bins), dz)) \
            and not np.isclose(bins[0], dz / 2):
        wmsg = ("'bins' currently does not start at zero, so it will "
                f"be shifted left by {(shift := bins[0] - dz / 2):.6g}.")
        warnings.warn(wmsg)
        bins -= shift
    if L is None:
        if not uniform:
            emsg = "'L' must be specified when 'bins' is not uniformly spaced."
            raise ValueError(emsg)
        L = bins[-1] - bins[0] + dz

    y = -integrate.trapezoid(bins * charge_density_profile, bins)
    if dV is not None:
        y += dielectric * dV / _ELECTROSTATIC_CONVERSION_FACTORS[reduced]
    return y / L

def calculate_potential_profile(
        bins: np.ndarray[float], charge_density_profile: np.ndarray[float],
        dielectric: float, *, L: float = None,
        sigma_q: Union[float, np.ndarray[float]] = None,
        dV: Union[float, np.ndarray[float]] = None, threshold: float = 1e-5,
        V0: Union[float, np.ndarray[float]] = 0, method: str = "integral",
        pbc: bool = False, reduced: bool = False) -> np.ndarray[float]:

    r"""
    Calculates the potential profile :math:`\Psi(z)` using the charge
    density profile :math:`\rho_q(z)` by numerically solving Poisson's
    equation for electrostatics.

    Poisson's equation is given by

    .. math::

       \varepsilon_0\varepsilon_\mathrm{r}\nabla^2\Psi(z)=-\rho_q(z)

    where :math:`\varepsilon_0` and :math:`\varepsilon_\mathrm{r}` are
    the vacuum and relative permittivities, respectively.

    The boundary conditions (BCs) are

    .. math::

       \left.\frac{\partial\Psi}{\partial z}\right\rvert_{z=0}&
       =-\frac{\sigma_q}{\varepsilon_0\varepsilon_\mathrm{r}}\\
       \left.\Psi\right\rvert_{z=0}&=\Psi_0

    The first BC is used to ensure that the electric field in the bulk
    of the solution is zero, while the second BC is used to set the
    potential on the left electrode.

    Poisson's equation can be evaluated by using the trapezoidal rule
    to numerically integrate the charge density profile twice:

    1. Integrate the charge density profile.
    2. Apply the first BC by subtracting :math:`\sigma_q` from all points.
    3. Integrate the profile from Step 2.
    4. Divide by :math:`-\varepsilon_0\varepsilon_\mathrm{r}`.
    5. Apply the second BC by adding :math:`\Psi_0` to all points.

    This method is fast but requires many histogram bins to accurately
    resolve the potential profile.

    Alternatively, Poisson's equation can be written as a system of
    linear equations

    .. math::

       A\mathbf{\Psi}=\mathbf{b}

    using second-order finite differences. :math:`A` is a tridiagonal
    matrix and :math:`\mathbf{b}` is a vector containing the charge
    density profile, with boundary conditions applied in the first and
    last rows.

    The inner equations are given by

    .. math::

       \Psi_i^{''}=\frac{\Psi_{i-1}-2\Psi_i+\Psi_{i+1}}{h^2}
       =-\frac{1}{\varepsilon_0\varepsilon_\mathrm{r}}\rho_{q,\,i}

    where :math:`i` is the bin index and :math:`h` is the bin width.

    In the case of periodic boundary conditions, the first and last
    equations are given by

    .. math::

       \Psi_0^{''}&=\frac{\Psi_{N-1}-2\Psi_0+\Psi_1}{h^2}
       =-\frac{1}{\varepsilon_0\varepsilon_\mathrm{r}}\rho_{q,\,0}\\
       \Psi_{N-1}^{''}&=\frac{\Psi_{N-2}-2\Psi_{N-1}+\Psi_0}{h^2}
       =-\frac{1}{\varepsilon_0\varepsilon_\mathrm{r}}\rho_{q,\,N-1}

    When the system has slab geometry, the boundary conditions are
    implemented via

    .. math::

       \Psi_0&=\Psi_0\\
       \Psi_0^\prime&=\frac{-3\Psi_0+4\Psi_1-\Psi_2}{2h}
       =\frac{\sigma_q}{\varepsilon_0\varepsilon_\mathrm{r}}

    This method is slower but can be more accurate even with fewer
    histogram bins for bulk systems with periodic boundary conditions.

    Parameters
    ----------
    bins : array-like
        Histogram bin centers :math:`z` for the charge density profile
        in `charge_density_profile`.

        **Shape**: :math:`(N_\mathrm{bins},)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    charge_density_profile : array-like
        Charge density profile :math:`\rho_q(z)`.

        **Shape**: :math:`(N_\mathrm{bins},)` or
        :math:`(N_\mathrm{frames},\,N_\mathrm{bins})`.

        **Reference unit**: :math:`\mathrm{e/Å}^{-3}`.

    dielectric : `float`
        Relative permittivity or static dielectric constant
        :math:`\varepsilon_\mathrm{r}`.

    L : `float`, keyword-only, optional
        System size in the dimension that `bins` and
        `charge_density_profile` were calculated in. If not specified,
        it is determined using the first and last values in `bins`,
        assuming that the bin centers are uniformly spaced.

        **Reference unit**: :math:`\mathrm{Å}`.

    sigma_q : `float` or array-like, keyword-only, optional
        Surface charge density :math:`\sigma_q`. Used as a boundary
        condition. If not provided, it is determined using `dV` and
        `charge_density_profile`. If `dV` is also not provided, the
        average value in the center of the integrated charge density
        profile is used if :code:`method="integral"`.

        .. note::

           :math:`\sigma_q` and :math:`\Delta\Psi` should have the same
           sign.

        **Shape**: Scalar or :math:`(N_\mathrm{frames},)`.

        **Reference unit**: :math:`\mathrm{e/Å^2}`.

    dV : `float` or array-like, keyword-only, optional
        Potential difference :math:`\Delta\Psi` across the system
        dimension specified in `axis`. Only used if `sigma_q` was not
        provided.

        **Shape**: Scalar or :math:`(N_\mathrm{frames},)`.

        **Reference unit**: :math:`\mathrm{V}`.

    V0 : `float` or array-like, keyword-only, default: :code:`0`
        Potential :math:`\Psi_0` at the left boundary.

        **Shape**: Scalar or :math:`(N_\mathrm{frames},)`.

        **Reference unit**: :math:`\mathrm{V}`.

    threshold : `float`, keyword-only, default: :code:`1e-5`
        Threshold for determining the plateau regions in the centers of
        the integrated charge density profiles to be used as estimates
        of `sigma_q`. Only used if `sigma_q` was not provided and
        cannot be calculated using `dV` and `charge_density_profile`.

    method : `str`, keyword-only, default: :code:`"integral"`
        Method used to calculate the potential profiles.

        **Valid values**: :code:`"integral"`, :code:`"matrix"`.

    pbc : `bool`, keyword-only, default: :code:`False`
        Specifies whether the axis has periodic boundary conditions.
        Only used when :code:`method="matrix"`.

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units.

    Returns
    -------
    potential : `numpy.ndarray`
        Potential profile :math:`\Psi(z)`.

        **Shape**: Same as `charge_density_profile`.

        **Reference unit**: :math:`\mathrm{V}`.
    """

    ecf = _ELECTROSTATIC_CONVERSION_FACTORS[reduced]

    # Check input array shapes
    if bins.ndim != 1:
        raise ValueError("'bins' must be a one-dimensional array.")
    if charge_density_profile.ndim not in {1, 2}:
        emsg = ("'charge_density_profile' must be a one- or "
                "two-dimensional array.")
        raise ValueError(emsg)
    if bins.shape[0] != charge_density_profile.shape[-1]:
        emsg = ("'bins' and 'charge_density_profile' must have the same"
                "length in the last dimension.")
        raise ValueError(emsg)
    if sigma_q is not None and not isinstance(sigma_q, Real) \
            and sigma_q.shape[0] != charge_density_profile.shape[0]:
        emsg = ("The number of surface charge densities in 'sigmas_q' "
                "must be equal to the number of frames in "
                "'charge_density_profile'.")
        raise ValueError(emsg)
    if dV is not None and not isinstance(dV, Real) \
            and dV.shape[0] != charge_density_profile.shape[0]:
        emsg = ("The number of potential differences in 'dVs' must be "
                "equal to the number of frames in "
                "'charge_density_profile'.")
        raise ValueError(emsg)
    if V0 is not None and not isinstance(V0, Real) \
            and V0.shape[0] != charge_density_profile.shape[0]:
        emsg = ("The number of potentials in 'V0s' must be equal to "
                "the number of frames in 'charge_density_profile'.")
        raise ValueError(emsg)

    # Determine system size, if not provided
    dz = bins[1] - bins[0]
    if (uniform := np.allclose(np.diff(bins), dz)) \
            and not np.isclose(bins[0], dz / 2):
        wmsg = ("'bins' currently does not start at zero, so it will "
                f"be shifted left by {(shift := bins[0] - dz / 2):.6g}.")
        warnings.warn(wmsg)
        bins -= shift
    if L is None:
        if not uniform:
            emsg = "'L' must be specified when 'bins' is not uniformly spaced."
            raise ValueError(emsg)
        L = bins[-1] - bins[0] + dz

    # Calculate surface charge density, if not provided
    if sigma_q is None and dV is not None:
        sigma_q = (
            dielectric * dV / ecf
            - integrate.trapezoid(bins * charge_density_profile, bins)
        ) / L

    if method == "integral":
        potential = integrate.cumulative_trapezoid(charge_density_profile,
                                                   bins, initial=0)

        if sigma_q is None:
            wmsg = ("No surface charge density information. The value "
                    "will be extracted from the integrated charge "
                    "density profile, which may be inaccurate due to "
                    "numerical errors.")
            warnings.warn(wmsg)

            # Get surface charge density from the integrated charge
            # density profile
            cut_indices = np.where(
                np.diff(
                    np.abs(
                        np.gradient(
                            potential if potential.ndim == 1
                            else potential.mean(axis=0)
                        )
                    ) < threshold
                )
            )[0] + 1
            if len(cut_indices) == 0:
                logging.warning(
                    "No bulk plateau region found in the integrated "
                    "charge density profile. The average value over "
                    "the entire profile will be used."
                )
                sigma_q = potential.mean(axis=-1, keepdims=True)
            else:
                target_index = potential.shape[-1] // 2
                if potential.ndim == 1:
                    sigma_q = potential[
                        cut_indices[cut_indices <= target_index][-1]:
                        cut_indices[cut_indices >= target_index][0]
                    ].mean()
                else:
                    sigma_q = potential[
                        :,
                        cut_indices[cut_indices <= target_index][-1]:
                        cut_indices[cut_indices >= target_index][0]
                    ].mean(axis=-1, keepdims=True)

        return V0 - ecf * (
            integrate.cumulative_trapezoid(potential - sigma_q, bins,
                                           initial=0)
        ) / dielectric

    elif method == "matrix":
        if sigma_q is None:
            emsg = ("No surface charge density information. Either "
                    "'sigma_q' or 'dV' must be provided when "
                    "method='matrix'.")
            raise ValueError(emsg)

        if not np.allclose(np.diff(bins), dz):
            raise ValueError("'bins' must be uniformly spaced.")

        # Set up matrix and load vector for second-order finite
        # difference method
        N = len(bins)
        A = sparse.diags((1, -2, 1), (-1, 0, 1), shape=(N, N), format="csc")
        b = charge_density_profile.copy().T
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",
                                  category=sparse.SparseEfficiencyWarning)

            # Apply boundary conditions and solve
            if pbc:
                A[0, -1] = A[-1, 0] = 1
                b *= -ecf * dz ** 2 / dielectric
                psi = np.empty_like(b)
                psi[1:] = sparse.linalg.spsolve(A[1:, 1:], b[1:])
                psi[0] = psi[-1]
                return psi
            else:
                A[0, :3] = -1.5, 2, -0.5
                A[-1, 0] = 1
                A[-1, -2:] = 0
                b[0] = ecf * dz * sigma_q / dielectric
                b[1:-1] *= -ecf * dz ** 2 / dielectric
                b[-1] = V0
                return sparse.linalg.spsolve(A, b)

class DensityProfile(DynamicAnalysisBase):

    """
    Serial and parallel implementations to calculate the number and
    charge density profiles :math:`\\rho_i(z)` and :math:`\\rho_q(z)` of
    a constant-volume system along the specified axes.

    The microscopic number density profile of species :math:`i` is
    calculated by binning particle positions along an axis :math:`z`
    using

    .. math::

       \\rho_i(z)=\\frac{V}{N_\\mathrm{bins}}\\left\\langle
       \\sum_\\alpha\\delta(z-z_\\alpha)\\right\\rangle

    where :math:`V` is the system volume and :math:`N_\\mathrm{bins}` is
    the number of bins. The angular brackets denote an ensemble average.

    If the species carry charges, the charge density profile can be
    obtained using

    .. math::

       \\rho_q(z)=e\\sum_i z_i\\rho_i(z)

    where :math:`z_i` is the charge number of species :math:`i` and
    :math:`e` is the elementary charge.

    With the charge density profile, the surface charge density is
    given by

    .. math::

       \\sigma_q=\\frac{1}{L}\\left(
       \\varepsilon_0\\varepsilon_\\mathrm{r}\\Delta\\Psi
       -\\int_0^L z\\rho_q(z)\\,\\mathrm{d}z\\right)

    where :math:`\\varepsilon_0` and :math:`\\varepsilon_\\mathrm{r}`
    are the vacuum and relative permittivities, respectively,
    :math:`\\Delta\\Psi` is the potential difference, and :math:`L` is
    the system dimension, and the potential profile can be computed by
    numerically solving Poisson's equation for electrostatics:

    .. math::

       \\varepsilon_0\\varepsilon_\\mathrm{r}\\nabla^2\\Psi(z)=-\\rho_q(z)

    Parameters
    ----------
    groups : `MDAnalysis.AtomGroup` or array-like
        Groups of atoms for which density profiles are calculated.

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

    axes : `int`, `str`, or array-like, default: :code:`"xyz"`
        Axes along which to compute the density profiles.

        .. container::

           **Examples**:

           * :code:`2` for the :math:`z`-direction.
           * :code:`"xy"` for the :math:`x`- and :math:`y`-directions.
           * :code:`(0, 1)` for the :math:`x`- and :math:`y`-directions.

    n_bins : `int` or array-like
        Number of histogram bins :math:`N_\\mathrm{bins}` for each axis
        in `axes`. If an `int` is provided, the same value is used for
        all axes.

    charges : array-like, keyword-only, optional
        Charge numbers :math:`z_i` for the entities in the
        :math:`N_\\mathrm{groups}` atom groups in `groups`. If not
        provided, they will be retrieved from the main
        :class:`MDAnalysis.core.universe.Universe` object only if it
        contains charge information.

        .. note::

           Depending on the grouping for a specific atom group, all
           entities (atoms, residues, or segments) must carry the same
           charge. Otherwise, the charge density contribution for that
           atom group would not make sense. If this condition does not
           hold, change how the atoms are grouped in the atom groups so
           that all entities share the same charge.

        **Shape**: :math:`(N_\\mathrm{groups},)`.

        **Reference unit**: :math:`\\mathrm{e}`.

    dimensions : array-like, keyword-only, optional
        System dimensions :math:`(L_x,\\,L_y,\\,L_z)`. If the
        :class:`MDAnalysis.core.universe.Universe` object that the
        atom groups in `groups` belong to does not contain
        dimensionality information, provide it here. Affected by
        `dim_scales`.

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

    dim_scales : array-like, keyword-only, optional
        Scale factors for the system dimensions. If an `int` is
        provided, the same value is used for all axes.

        **Shape**: :math:`(3,)`.

    average : `bool`, keyword-only, default: :code:`True`
        Determines whether the density profiles are averaged over the
        :math:`N_\\mathrm{frames}` analysis frames.

    recenter : `int`, `list`, `MDAnalysis.AtomGroup`, or `tuple`, \
    keyword-only, optional
        Constrains the center of mass of an atom group by adjusting the
        particle coordinates every analysis frame. Either specify an
        :class:`MDAnalysis.core.groups.AtomGroup`, its index within
        `groups`, a list of atom groups or their indices, or a tuple
        containing the aforementioned information and the fixed center
        of mass coordinates, in that order. To avoid recentering in a
        specific dimension, set the coordinate to :code:`numpy.nan`. If
        the center of mass is not specified, the center of the
        simulation box is used.

        **Shape**: :math:`(3,)` for the fixed center of mass.

    reduced : `bool`, keyword-only, default: :code:`False`
        Specifies whether the data is in reduced units. Affects
        `results.number_densities`, `results.charge_densities`, etc.

    parallel : `bool`, keyword-only, default: :code:`False`
        Determines whether the analysis is performed in parallel.

        .. note::

           The Joblib threading backend generally provides the best
           performance.

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
        reference units for `results.bins`, call
        :code:`results.units["bins"]`.

    results.times : `numpy.ndarray`
        Times :math:`t`. Only available if :code:`average=False`.

        **Shape**: :math:`(N_\\mathrm{frames},)`.

        **Reference unit**: :math:`\\mathrm{ps}`.

    results.bins : `dict`
        Bin centers :math:`z` corresponding to the density profiles in
        each dimension. The key is the axis, e.g.,
        :code:`results.bins["z"]` for the :math:`z`-axis.

        **Shape**: Each array has shape :math:`(N_\\mathrm{bins},)`.

        **Reference unit**: :math:`\\mathrm{Å}`.

    results.bin_edges : `dict`
        Bin edges corresponding to the density profiles in each
        dimension. The key is the axis, e.g.,
        :code:`results.bin_edges["z"]` for the :math:`z`-axis.

        **Shape**: Each array has shape :math:`(N_\\mathrm{bins}+1,)`.

        **Reference unit**: :math:`\\mathrm{Å}`.

    results.number_densities : `dict`
        Number density profiles :math:`\\rho(z)`. The key is the axis,
        e.g., :code:`results.number_densities["z"]` for the
        :math:`z`-axis.

        **Shape**: Each array has shape
        :math:`(N_\\mathrm{groups},\\,N_\\mathrm{bins},)`. If
        :code:`average=False`, an additional second dimension of
        length :math:`N_\\mathrm{frames}` is present.

        **Reference unit**: :math:`\\mathrm{Å}^{-3}`.

    results.charge_densities : `dict`
        Charge density profiles :math:`\\rho_q(z)`. Only available if
        charge information was found or provided. The key is the axis,
        e.g., :code:`results.charge_densities["z"]` for the
        :math:`z`-axis.

        **Shape**: Each array has shape :math:`(N_\\mathrm{bins},)`. If
        :code:`average=False`, an additional first dimension of length
        :math:`N_\\mathrm{frames}` is present.

        **Reference unit**: :math:`\\mathrm{e/Å}^{-3}`.

    results.surface_charge_densities : `numpy.ndarray`
        Surface charge densities :math:`\\sigma_q`. Only available after
        running :meth:`calculate_surface_charge_densities`.

        **Shape**: :math:`(N_\\mathrm{axes},)` or
        :math:`(N_\\mathrm{axes},\,N_\\mathrm{frames})`.

    results.potentials : `dict`
        Potential profiles :math:`\\Psi(z)`. Only available after
        running :meth:`calculate_potential_profiles`. The key is the
        axis, e.g., :code:`results.potentials["z"]` for the
        :math:`z`-axis.

        **Shape**: Each array has shape :math:`(N_\\mathrm{bins},)`. If
        :code:`average=False`, an additional second dimension of
        length :math:`N_\\mathrm{frames}` is present.

        **Reference unit**: :math:`\\mathrm{V}`.
    """

    def __init__(
            self, groups: Union[mda.AtomGroup, tuple[mda.AtomGroup]],
            groupings: Union[str, tuple[str]] = "atoms",
            axes: Union[int, str, tuple[Union[int, str]]] = "xyz",
            n_bins: Union[int, tuple[int]] = 201, *,
            charges: Union[np.ndarray[float], "unit.Quantity", Q_] = None,
            dimensions: Union[np.ndarray[float], "unit.Quantity", Q_] = None,
            dt: Union[float, "unit.Quantity", Q_] = None,
            recenter: Union[
                mda.AtomGroup, int, list[mda.AtomGroup, int],
                tuple[Union[mda.AtomGroup, int, list[mda.AtomGroup, int]],
                      np.ndarray[float]]
            ] = None,
            dim_scales: Union[float, tuple[float]] = 1,
            average: bool = True, reduced: bool = False,
            parallel: bool = False, verbose: bool = True, **kwargs) -> None:

        self._groups = [groups] if isinstance(groups, mda.AtomGroup) else groups
        self.universe = self._groups[0].universe
        super().__init__(self.universe.trajectory, parallel, verbose, **kwargs)

        GROUPINGS = {"atoms", "residues", "segments"}
        self._n_groups = len(self._groups)
        if isinstance(groupings, str):
            if groupings not in GROUPINGS:
                emsg = (f"Invalid grouping '{groupings}'. Valid values: "
                        "'" + "', '".join(GROUPINGS) + "'.")
                raise ValueError(emsg)
            self._groupings = self._n_groups * [groupings]
        else:
            if self._n_groups != len(groupings):
                emsg = ("The number of grouping values is not equal to "
                        "the number of atom groups.")
                raise ValueError(emsg)
            for gr in groupings:
                if gr not in GROUPINGS:
                    emsg = (f"Invalid grouping '{groupings}'. Valid values: "
                            "'" + "', '".join(GROUPINGS) + "'.")
                    raise ValueError(emsg)
            self._groupings = groupings

        if isinstance(axes, int):
            self._axis_indices = np.array((axes,), dtype=int)
        else:
            self._axis_indices = np.fromiter(
                (ord(ax.lower()) - 120 if isinstance(ax, str) else ax
                 for ax in axes),
                count=len(axes),
                dtype=int
            )
        self._axes = [chr(120 + i) for i in self._axis_indices]
        if not all(ax in "xyz" for ax in self._axes):
            raise ValueError("Invalid axis passed in 'axes'.")
        self._n_axes = len(self._axes)

        if isinstance(n_bins, int):
            self._n_bins = n_bins * np.ones(len(self._axes), dtype=int)
        elif isinstance(n_bins, str):
            emsg = "'n_bins' must be an integer or an iterable object."
            raise ValueError(emsg)
        else:
            if len(n_bins) != len(self._axes):
                emsg = ("The shape of 'n_bins' is incompatible with "
                        "the number of axes to calculate density "
                        "profiles along.")
                raise ValueError(emsg)
            if not all(isinstance(n, int) for n in n_bins):
                emsg = "All bin counts in 'n_bins' must be integers."
                raise ValueError(emsg)
            self._n_bins = n_bins

        if charges is not None:
            if len(charges) != self._n_groups:
                emsg = ("The number of group charges is not equal to "
                        "the number of groups.")
                raise ValueError(emsg)
            if reduced and not is_unitless(charges):
                emsg = "'charges' cannot have units when 'reduced=True'."
                raise TypeError(emsg)
            self._charges = np.asarray(strip_unit(charges, "e")[0])
        elif hasattr(self.universe.atoms, "charges"):
            self._charges = np.empty(self._n_groups)
            for i, (ag, gr) in enumerate(zip(self._groups, self._groupings)):
                qs = getattr(ag, gr).charges
                if not np.allclose(qs, q := qs[0]):
                    self._charges = None
                    wmsg = (f"Not all {gr} in `groups[{i}]` share the "
                            "same charge. The charge density profile will "
                            "not be calculated.")
                    warnings.warn(wmsg)
                    break
                self._charges[i] = q
        else:
            self._charges = None

        if dimensions is not None:
            if len(dimensions) != 3:
                raise ValueError("'dimensions' must have length 3.")
            if reduced and not is_unitless(dimensions):
                emsg = "'dimensions' cannot have units when 'reduced=True'."
                raise TypeError(emsg)
            self._dimensions = np.asarray(strip_unit(dimensions, "Å")[0])
        elif self.universe.dimensions is not None:
            self._dimensions = self.universe.dimensions[:3].copy()
        else:
            raise ValueError("No system dimensions found or provided.")

        if dt is None:
            self._dt = self._trajectory.dt
        else:
            if reduced and not is_unitless(dt):
                raise TypeError("'dt' cannot have units when 'reduced=True'.")
            self._dt = strip_unit(dt, "ps")[0]

        if (isinstance(dim_scales, Real)
            or (len(dim_scales) == 3
                and all(isinstance(f, Real) for f in dim_scales))):
            self._dimensions *= dim_scales
        else:
            emsg = ("'dim_scales' must be a floating-point number "
                    "or an array with shape (3,).")
            raise ValueError(emsg)

        self._Ns = np.fromiter(
            (getattr(ag, f"n_{gr}")
             for (ag, gr) in zip(self._groups, self._groupings)),
            dtype=int,
            count=self._n_groups
        )
        self._N = self._Ns.sum()
        self._slices = []
        _ = 0
        for N in self._Ns:
            self._slices.append(slice(_, _ + N))
            _ += N

        if recenter is None:
            self._recenter = recenter
        else:
            if isinstance(recenter, (int, mda.AtomGroup)):
                grp = recenter
                com = self._dimensions / 2
            elif isinstance(recenter, tuple) and len(recenter) == 2:
                grp, com = recenter
                if len(com) != 3:
                    emsg = ("The target center of mass in 'recenter' "
                            "must have length 3.")
                    raise ValueError(emsg)
                com = np.asarray(com)
            else:
                emsg = ("Invalid value passed to 'recenter'. The "
                        "argument must either be an atom group, its "
                        "index in 'groups', multiple groups/indices, or "
                        "a tuple containing the aforementioned "
                        "information and a target center of mass, in "
                        "that order.")
                raise ValueError(emsg)

            if isinstance(grp, int):
                if not 0 <= grp < self._n_groups:
                    emsg = "Invalid group index passed to 'recenter'."
                    raise ValueError(emsg)
                grp = self._slices[grp]
            elif isinstance(grp, mda.AtomGroup):
                try:
                    grp = self._slices[self._groups.index(grp)]
                except ValueError:
                    emsg = ("The atom group passed to 'recenter' is not "
                            "in 'groups'.")
                    raise ValueError(emsg)
            else:
                try:
                    grp = np.r_[
                        *(
                            self._slices[
                                g if isinstance(g, int)
                                else self._groups.index(g)
                            ] for g in grp
                        )
                    ]
                except ValueError:
                    emsg = "Invalid atom group or index passed to 'recenter'."
                    raise ValueError(emsg)
            self._recenter = (grp, com)

        self._dielectrics = np.empty(self._n_axes)
        self._dVs = np.empty(self._n_axes)
        self._dielectrics[:] = self._dVs[:] = np.nan

        self._average = average
        self._reduced = reduced
        self._verbose = verbose

    def _prepare(self) -> None:

        # Specify bin centers and edges for each axis
        self.results.bins = Hash()
        self.results.bin_edges = Hash()
        for ia, ax, n in zip(self._axis_indices, self._axes, self._n_bins):
            d = self._dimensions[ia]
            self.results.bins[ax] = np.linspace(s := d / (2 * n), d - s, n)
            self.results.bin_edges[ax] \
                = histogram_bin_edges_1d_int(np.asarray((0, d)), n)

        # Store entity masses for center of mass calculations
        self._masses = np.empty(self._N)
        for ag, gr, s in zip(self._groups, self._groupings, self._slices):
            self._masses[s] = getattr(ag, gr).masses

        if self._recenter is not None:

            # Navigate to first frame in analysis
            self._sliced_trajectory[0]

            # Preallocate arrays to determine the number of periodic
            # boundary crossings for each entity
            self._positions_old = np.empty((self._N, 3))
            for ag, gr, s in zip(self._groups, self._groupings, self._slices):
                self._positions_old[s] = (ag.positions if gr == "atoms"
                                          else center_of_mass(ag, gr))
            self._images = np.zeros((self._N, 3), dtype=int)
            self._thresholds = self._dimensions / 2

            if self._parallel:

                # Preallocate array to hold processed entity positions
                # for all frames to be analyzed in parallel
                self._positions = np.empty((self.n_frames, self._N, 3))
                for i, _ in enumerate(self._sliced_trajectory):

                    # Get raw entity positions for current frame
                    for ag, gr, s in zip(self._groups, self._groupings,
                                         self._slices):
                        self._positions[i, s] = (ag.positions if gr == "atoms"
                                                 else center_of_mass(ag, gr))

                    # Globally unwrap entity positions for correct
                    # center of mass
                    unwrap(
                        self._positions[i],
                        self._positions_old,
                        self._dimensions,
                        thresholds=self._thresholds,
                        images=self._images
                    )

                    # Shift entity positions by the difference between
                    # the group and target centers of mass
                    self._positions[i] -= np.fromiter(
                        (0 if np.isnan(tx) else gx - tx for gx, tx in zip(
                            center_of_mass(
                                positions=self._positions[i, self._recenter[0]],
                                masses=self._masses[self._recenter[0]]
                            ),
                            self._recenter[1]
                        )),
                        dtype=float,
                        count=3
                    )

                # Wrap entity positions back into the simulation box so that
                # they belong to a histogram bin
                wrap(self._positions, self._dimensions)

        # Preallocate arrays to hold entity positions for a given frame
        # (so that one doesn't have to be recreated each frame) and
        # number density profiles for serial analysis
        if not self._parallel:
            self._positions = np.empty((self._N, 3))
            shape = [self._n_groups]
            if not self._average:
                shape.append(self.n_frames)
            self.results.number_densities = Hash({
                ax: np.zeros((*shape, n))
                for ax, n in zip(self._axes, self._n_bins)
            })

        # Store reference units
        self.results.units = Hash({
            "bins": ureg.angstrom,
            "number_densities": ureg.angstrom ** -3
        })

        # Preallocate dictionary to hold charge density profiles, if
        # necessary
        if self._charges is not None:
            self.results.charge_densities = Hash()
            self.results.units["charge_densities"] = (
                ureg.elementary_charge / ureg.angstrom ** 3
            )

        # Store time information, if necessary
        if not self._average:
            try:
                self.results.times \
                    = self._dt * np.asarray(self._sliced_trajectory.frames)
            except AttributeError:
                self.results.times \
                    = self._dt * np.arange(self.start, self.stop, self.step)
            self.results.units["times"] = ureg.picosecond

    def _single_frame(self):

        # Store atom or center-of-mass positions in the current frame
        for ag, gr, s in zip(self._groups, self._groupings, self._slices):
            self._positions[s] = (ag.positions if gr == "atoms"
                                  else center_of_mass(ag, gr))

        if self._recenter is not None:

            # Globally unwrap entity positions for correct center of mass
            unwrap(
                self._positions,
                self._positions_old,
                self._dimensions,
                thresholds=self._thresholds,
                images=self._images
            )

            # Shift entity positions by the difference between the group
            # and target centers of mass
            self._positions -= np.fromiter(
                (0 if np.isnan(tx) else gx - tx for gx, tx in zip(
                    center_of_mass(
                        positions=self._positions[self._recenter[0]],
                        masses=self._masses[self._recenter[0]]
                    ),
                    self._recenter[1]
                )),
                dtype=float,
                count=3
            )

        # Wrap entity positions back into the simulation box so that
        # they belong to a histogram bin
        wrap(self._positions, self._dimensions)

        # Compute and tally the bin counts for the entity positions
        for ax, ia, n in zip(self._axes, self._axis_indices, self._n_bins):
            for ig, (gr, s) in enumerate(zip(self._groupings, self._slices)):
                if self._average:
                    self.results.number_densities[ax][ig] \
                        += histogram_1d_int_1d(self._positions[s, ia], n,
                                               self.results.bin_edges[ax])
                else:
                    self.results.number_densities[ax][ig, self._frame_index] \
                        = histogram_1d_int_1d(self._positions[s, ia], n,
                                              self.results.bin_edges[ax])

    def _single_frame_parallel(
            self, index: int) -> tuple[int, np.ndarray[float]]:

        # Set current trajectory frame
        self._sliced_trajectory[index]

        # Preallocate arrays to hold bin counts for the current frame
        results = {ax: np.empty((self._n_groups, n))
                   for ax, n in zip(self._axes, self._n_bins)}

        # Calculate or get entity positions
        if self._recenter is None:
            positions = np.empty((self._N, 3))
            for ag, gr, s in zip(self._groups, self._groupings, self._slices):
                positions[s] = (ag.positions if gr == "atoms"
                                else center_of_mass(ag, gr))
            wrap(positions, self._dimensions)
        else:
            positions = self._positions[index]

        # Compute and tally the bin counts for the entity positions
        for ax, ia, n in zip(self._axes, self._axis_indices, self._n_bins):
            for ig, (gr, s) in enumerate(zip(self._groupings, self._slices)):
                results[ax][ig] = histogram_1d_int_1d(
                    positions[s, ia], n, self.results.bin_edges[ax]
                )

        return index, results

    def _conclude(self):

        # Consolidate parallel results and clean up memory by deleting
        # arrays that will not be reused
        if self._parallel:
            self._results = sorted(self._results)
            self.results.number_densities = Hash()
            for ax in self._axes:
                self.results.number_densities[ax] \
                    = np.stack([r[1][ax] for r in self._results], axis=1)
                if self._average:
                    self.results.number_densities[ax] \
                        = self.results.number_densities[ax].sum(axis=1)
            del self._results
            if self._recenter is not None:
                del self._positions
        else:
            del self._positions
        if self._recenter is not None:
            del self._positions_old, self._images, self._thresholds

        # Normalize histograms by bin volume
        volume = np.prod(self._dimensions)
        for ax, n in zip(self._axes, self._n_bins):
            denom = volume / n
            if self._average:
                denom *= self.n_frames
            self.results.number_densities[ax] /= denom

            if self._charges is not None:
                self.results.charge_densities[ax] = np.einsum(
                    "g,g...b->...b",
                    self._charges,
                    self.results.number_densities[ax]
                )

    def _validate_input(
            self, value: Any, unit_: str, name: str, n_axes: int
        ) -> Union[Real, np.ndarray[Real]]:

        """
        Validates and processes input values to calculation methods.

        Parameters
        ----------
        value : `Any`
            Value to validate and process.

        unit_ : `str`
            Reference unit for the value.

        name : `str`
            Name of the value being validated.

        n_axes : `int`
            Number of axes along which to calculate the value.

        Returns
        -------
        `Real` or `numpy.ndarray`
            Processed value.
        """

        if value is None:
            value = n_axes * [value]
        else:
            if self._reduced and not is_unitless(value):
                emsg = f"'{name}' cannot have units when 'reduced=True'."
                raise TypeError(emsg)
            value = strip_unit(value, unit_)[0]
            if isinstance(value, Real):
                value = n_axes * [value]
            elif len(value) != n_axes:
                emsg = (f"The length of '{name}' must match the "
                        "number of axes.")
                raise ValueError(emsg)
        return value

    def calculate_surface_charge_densities(
            self, axes: Union[str, tuple[str]] = None,
            dielectrics: Union[float, tuple[float]] = None, *,
            dVs: Union[float, np.ndarray[float], "unit.Quantity", Q_] = None
        ) -> None:

        """
        Calculates the surface charge densities :math:`\\sigma_q` for
        the specified system dimensions using the charge density
        profiles :math:`\\rho_q(z)`.

        Parameters
        ----------
        axes : `str` or array-like, optional
            Axes along which to compute the potential profiles. If not
            specified, all axes for which charge density profiles were
            calculated will be used.

            **Examples**::code:`"xy"` or :code:`("x", "y")`
            for the :math:`x`- and :math:`y`-directions.

        dielectrics : `float`, optional
            Relative permittivities or dielectric constants
            :math:`\\varepsilon_\\mathrm{r}`. Only optional if
            previously provided to another calculation method in this
            class.

        dVs : `float`, array-like, `openmm.unit.Quantity`, or \
        `pint.Quantity`, keyword-only, optional
            Potential differences :math:`\\Delta\\Psi` across the system
            dimensions specified in `axes`. Can be retrieved if
            previously provided to another calculation method in this
            class. Has no effect if `sigmas_q` is provided since this
            value is used solely to calculate `sigmas_q`.

            **Shapes**: :math:`(N_\\mathrm{axes},)` or
            :math:`(N_\\mathrm{axes},\\,N_\\mathrm{frames})`.

            **Reference unit**: :math:`\\mathrm{V}`.
        """

        # Ensure charge density profiles have already been calculated
        if not hasattr(self.results, "charge_densities"):
            emsg = ("Either call run() before calculate_potential_profiles() "
                    "or provide charge information when initializing "
                    "the DensityProfile object.")
            raise RuntimeError(emsg)

        # Validate inputs
        if axes is None:
            axes = self._axes
            axis_indices = self._axis_indices
        else:
            if isinstance(axes, Real) \
                    or any(not isinstance(ax, str) for ax in axes):
                raise ValueError("'axes' must only contain strings.")
            axes = tuple(axes)
            axis_indices = [ord(ax.lower()) - 120 for ax in axes]
        try:
            relative_axes = [self._axes.index(ax) for ax in axes]
        except ValueError:
            raise ValueError("Invalid axis passed in 'axes'.")
        n_axes = len(axes)
        dimensions = self._dimensions[axis_indices]

        if dielectrics is not None:
            self._dielectrics[relative_axes] = self._validate_input(
                dielectrics, "", "dielectrics", n_axes
            )
        elif np.any(np.isnan(self._dielectrics)):
            raise ValueError("No dielectric constants found or provided.")

        if dVs is not None:
            self._dVs[relative_axes] = self._validate_input(
                dVs, "V", "dVs", n_axes
            )

        # Preallocate dictionary to hold potential profiles
        if not hasattr(self.results, "surface_charge_densities"):
            shape = [self._n_axes]
            if not self._average:
                shape.append(self.n_frames)
            self.results.surface_charge_densities = np.empty(shape)
            self.results.surface_charge_densities[:] = np.nan
            self.results.units["surface_charge_densities"] \
                = ureg.elementary_charge / ureg.angstrom ** 2

        # Calculate surface charge densities
        for i, (ax, dielectric, L, dV) in enumerate(
                zip(axes, self._dielectrics, dimensions, self._dVs)
            ):
            self.results.surface_charge_densities[i] \
                = calculate_surface_charge_density(
                    self.results.bins[ax],
                    self.results.charge_densities[ax],
                    dielectric,
                    L=L,
                    dV=None if np.isnan(dV) else dV,
                    reduced=self._reduced
                )

    def calculate_potential_profiles(
            self, axes: Union[str, tuple[str]] = None,
            dielectrics: Union[float, tuple[float]] = None, *,
            sigmas_q: Union[float, np.ndarray[float], "unit.Quantity", Q_] = None,
            dVs: Union[float, np.ndarray[float], "unit.Quantity", Q_] = None,
            thresholds: Union[float, np.ndarray[float]] = 1e-5,
            V0s: Union[float, np.ndarray[float], "unit.Quantity", Q_] = 0,
            methods: Union[str, tuple[str]] = "integral",
            pbcs: Union[bool, tuple[bool]] = False) -> None:

        """
        Calculates the potential profiles in the specified dimensions
        using the charge density profiles by numerically solving
        Poisson's equation for electrostatics.

        Parameters
        ----------
        axes : `str` or array-like, optional
            Axes along which to compute the potential profiles. If not
            specified, all axes for which charge density profiles were
            calculated will be used.

            **Examples**::code:`"xy"` or :code:`("x", "y")`
            for the :math:`x`- and :math:`y`-directions.

        dielectrics : `float`, optional
            Relative permittivities or dielectric constants
            :math:`\\varepsilon_\\mathrm{r}`. Only optional if
            previously provided to another calculation method in this
            class.

        sigmas_q : `float`, array-like, `openmm.unit.Quantity`, or \
        `pint.Quantity`, keyword-only, optional
            Surface charge densities :math:`\\sigma_q`. Used to ensure
            that the electric field in the bulk of the solution is zero.
            If not provided, it is determined using `dVs` and the charge
            density profiles, or the average values in the centers of
            the integrated charge density profiles.

            .. note::

               :math:`\sigma_q` and :math:`\Delta\Psi` should have the
               same sign.

            **Shapes**: :math:`(N_\\mathrm{axes},)` or
            :math:`(N_\\mathrm{axes},\\,N_\\mathrm{frames})`.

            **Reference unit**: :math:`\\mathrm{e/Å^2}`.

        dVs : `float`, array-like, `openmm.unit.Quantity`, or \
        `pint.Quantity`, keyword-only, optional
            Potential differences :math:`\\Delta\\Psi` across the system
            dimensions specified in `axes`. Can be retrieved if
            previously provided to another calculation method in this
            class. Has no effect if `sigmas_q` is provided since this
            value is used solely to calculate `sigmas_q`.

            **Shapes**: :math:`(N_\\mathrm{axes},)` or
            :math:`(N_\\mathrm{axes},\\,N_\\mathrm{frames})`.

            **Reference unit**: :math:`\\mathrm{V}`.

        thresholds : `float` or array-like, keyword-only, \
        default: :code:`1e-5`
            Thresholds for determining the plateau regions of the
            integrals of the charge density profiles to calculate
            `sigmas_q`. Has no effect if `sigmas_q` is provided, or if
            `sigmas_q` can be calculated using `dVs` and
            `charge_density_profiles`.

        V0s : `float`, array-like, `openmm.unit.Quantity`, or \
        `pint.Quantity`, keyword-only, default: :code:`0`
            Potentials :math:`\\Psi_0` at the left boundary.

            **Shapes**: :math:`(N_\\mathrm{axes},)` or
            :math:`(N_\\mathrm{axes},\\,N_\\mathrm{frames})`.

            **Reference unit**: :math:`\\mathrm{V}`.

        methods : `str` or array-like, keyword-only, \
        default: :code:`"integral"`
            Methods to use to calculate the potential profiles.

            **Valid values**: :code:`"integral"`, :code:`"matrix"`.

        pbcs : `bool`, keyword-only, default: :code:`False`
            Specifies whether the system has periodic boundary
            conditions in each of the axes. Only used when
            :code:`method="matrix"`.
        """

        # Ensure charge density profiles have already been calculated
        if not hasattr(self.results, "charge_densities"):
            emsg = ("Either call run() before calculate_potential_profiles() "
                    "or provide charge information when initializing "
                    "the DensityProfile object.")
            raise RuntimeError(emsg)

        # Preallocate dictionary to hold potential profiles
        if not hasattr(self.results, "potentials"):
            self.results.potentials = Hash()
            self.results.units["potentials"] = ureg.volt

        # Validate inputs
        if axes is None:
            axes = self._axes
            axis_indices = self._axis_indices
        else:
            if isinstance(axes, Real) \
                    or any(not isinstance(ax, str) for ax in axes):
                raise ValueError("'axes' must only contain strings.")
            axes = tuple(axes)
            axis_indices = [ord(ax.lower()) - 120 for ax in axes]
        try:
            relative_axes = [self._axes.index(ax) for ax in axes]
        except ValueError:
            raise ValueError("Invalid axis passed in 'axes'.")
        n_axes = len(axes)
        dimensions = self._dimensions[axis_indices]

        if dielectrics is not None:
            self._dielectrics[relative_axes] = self._validate_input(
                dielectrics, "", "dielectrics", n_axes
            )
        elif np.any(np.isnan(self._dielectrics)):
            raise ValueError("No dielectric constants found or provided.")

        if sigmas_q is None:
            if hasattr(self.results, "surface_charge_densities") \
                    and np.all(np.isfinite(self._dVs[relative_axes])):
                sigmas_q = -self.results.surface_charge_densities
            else:
                sigmas_q = self._validate_input(sigmas_q, "e/Å^2", "sigmas_q",
                                                n_axes)
                if dVs is not None:
                    self._dVs[relative_axes] = self._validate_input(
                        dVs, "V", "dVs", n_axes
                    )
        else:
            sigmas_q = self._validate_input(sigmas_q, "e/Å^2", "sigmas_q",
                                            n_axes)

        V0s = self._validate_input(V0s, "V", "V0s", n_axes)
        thresholds = self._validate_input(thresholds, "", "thresholds", n_axes)

        METHODS = {"integral", "matrix"}
        if isinstance(methods, str) and methods in METHODS:
            methods = n_axes * [methods]
        elif len(methods) != n_axes:
            emsg = "The length of 'methods' must match the number of axes."
            raise ValueError(emsg)
        else:
            for method in methods:
                if method not in METHODS:
                    emsg = (f"Invalid method '{method}'. Valid values: "
                            "'" + "', '".join(METHODS) + "'.")
                    raise ValueError(emsg)

        if isinstance(pbcs, bool):
            pbcs = n_axes * [pbcs]
        elif len(pbcs) != n_axes:
            emsg = "The length of 'pbcs' must match the number of axes."
            raise ValueError(emsg)
        elif not all(isinstance(pbc, bool) for pbc in pbcs):
            raise ValueError("All values in 'pbcs' must be booleans.")

        # Calculate potential profiles
        for ax, dielectric, L, sigma_q, dV, threshold, V0, method, pbc in zip(
                axes, self._dielectrics, dimensions, sigmas_q, self._dVs,
                thresholds, V0s, methods, pbcs
            ):
            self.results.potentials[ax] = calculate_potential_profile(
                self.results.bins[ax],
                self.results.charge_densities[ax],
                dielectric,
                L=L,
                sigma_q=sigma_q,
                dV=None if np.isnan(dV) else dV,
                threshold=threshold,
                V0=V0,
                method=method,
                pbc=pbc,
                reduced=self._reduced
            )