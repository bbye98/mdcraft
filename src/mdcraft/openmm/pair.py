"""
Custom OpenMM pair potentials
=============================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains implementations of commonly used pair potentials
that are not available in OpenMM, such as the Gaussian and Yukawa
potentials. Generally, the pair potentials are named after their LAMMPS
:code:`pair_style` counterparts, if available.
"""

from typing import Union

import numpy as np
import openmm
from openmm import unit

from .unit import VACUUM_PERMITTIVITY

def _setup_pair(
        cnbforce: openmm.CustomNonbondedForce,
        cutoff: Union[float, unit.Quantity],
        global_params: dict[str, Union[float, unit.Quantity]],
        per_params: list[str],
        tab_funcs: dict[str, Union[np.ndarray[Union[int, float]],
                                   openmm.unit.Quantity,
                                   openmm.Discrete2DFunction]],
        method: int = openmm.CustomNonbondedForce.CutoffPeriodic
    ) -> None:

    r"""
    Sets up a :class:`openmm.CustomNonbondedForce` object.

    Parameters
    ----------
    cnbforce : `openmm.CustomNonbondedForce`
        Custom nonbonded force object.

    cutoff : `float` or `openmm.unit.Quantity`
        Shared cutoff distance :math:`r_\mathrm{cut}` for all nonbonded
        interactions in the simulation sytem. Must be less than half the
        periodic simulation box dimensions.

        **Reference unit**: :math:`\mathrm{nm}`.

    global_params : `dict`, optional
        Global parameters.

    per_params : `list`, optional
        Per-particle parameters.

    tab_funcs : `dict`, optional
        Tabulated functions.

    method : `int`, default: :code:`2`
        Cutoff method.
    """

    if global_params:
        for param in global_params.items():
            cnbforce.addGlobalParameter(*param)
    if per_params:
        for param in per_params:
            cnbforce.addPerParticleParameter(param)
    if tab_funcs:
        for name, func in tab_funcs.items():
            if not isinstance(func, openmm.Discrete2DFunction):
                func = openmm.Discrete2DFunction(*func.shape, func.ravel().tolist())
            cnbforce.addTabulatedFunction(name, func)
    cnbforce.setCutoffDistance(cutoff)
    cnbforce.setNonbondedMethod(method)

def coul_gauss(
        cutoff: Union[float, unit.Quantity], tol: float = 1e-4, *,
        g_ewald: Union[float, unit.Quantity] = None,
        dims: Union[np.ndarray[float], unit.Quantity] = None,
        mix: str = "default", per_params: list = None,
        global_params: dict[str, Union[float, unit.Quantity]] = None,
        tab_funcs: dict[str, Union[np.ndarray[Union[int, float]],
                                   openmm.unit.Quantity,
                                   openmm.Discrete2DFunction]] = None
    ) -> tuple[openmm.CustomNonbondedForce, openmm.NonbondedForce]:

    r"""
    Implements the smeared Coulomb pair potential.

    The charges have spherical Gaussian-distributed charge
    distributions [1]_ [2]_

    .. math::

       u_\mathrm{Coul}(r_{12})=\frac{q_1q_2}{4\pi\varepsilon_0r_{12}}
       \mathrm{erf}(\alpha_{12}r_{12})

    where :math:`q_1` and :math:`q_2` are the particle charges in
    :math:`\mathrm{e}`, :math:`\varepsilon_0` is the vacuum
    permittivity, and :math:`\alpha_{12}=\sqrt{\alpha_1^2+\alpha_2^2}`
    is an inverse screening length or damping parameter in
    :math:`\mathrm{nm}^{-1}`. Effectively, electrostatic interactions
    are dampened for charges at small separation distances but remain
    unchanged at large separation distances.

    Additionally, an implementation of the electrostatic pair potential
    in the Gaussian core model

    .. math::

       u_\mathrm{Coul}(r_{12})=\frac{q_1q_2}{4\pi\varepsilon_0r_{12}}
       \mathrm{erf}\left(\frac{\pi^{1/2}}{2^{1/2}a_{12}}r_{12}\right)

    is available, where :math:`a_{12}=\sqrt{a_1^2+a_2^2}` with
    :math:`a_1` and :math:`a_2` being the electrostatic smearing radii
    in :math:`\mathrm{nm}`.

    To account for the solvent polarization implicitly using its
    relative permittivity :math:`\varepsilon_\mathrm{r}`, scale the
    particle charges :math:`q_1` and :math:`q_2` by
    :math:`1/\sqrt{\varepsilon_\mathrm{r}}`.

    After creating the pair potentials, particles should be registered
    using :meth:`openmm.openmm.NonbondedForce.addParticle` and
    :meth:`openmm.openmm.CustomNonbondedForce.addParticle`.

    Parameters
    ----------
    cutoff : `float` or `openmm.unit.Quantity`
        Shared cutoff distance for all nonbonded interactions in the
        simulation sytem. Must be less than half the minimum periodic
        simulation box dimension.

        **Reference unit**: :math:`\mathrm{nm}`.

    tol : `float`, default: :code:`1e-4`
        Error tolerance :math:`\delta` for Ewald summation.

    g_ewald : `float` or `openmm.unit.Quantity`, keyword-only, optional
        Ewald splitting parameter :math:`g_\mathrm{Ewald}`. If not
        provided, it is computed using

        .. math::

           g_\mathrm{Ewald}=\frac{\sqrt{-\log(2\delta)}}{r_\mathrm{cut}}

        **Reference unit**: :math:`\mathrm{nm}^{-1}`.

    dims : `array_like` or `openmm.unit.Quantity`, keyword-only, optional
        Simulation system dimensions. Must be provided with `g_ewald` to
        calculate the number of mesh nodes :math:`n_\mathrm{mesh}`.
        Both `dims` and `g_ewald` must either have units or be unitless.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    mix : `str`, keyword-only, default: :code:`"default"`
        Mixing rule for :math:`\alpha_{12}`.

        .. container::

           **Valid values**:

           * :code:`"default"`: Default mixing rule.

             .. math::

                \alpha_{12}=\frac{\alpha_1\alpha_2}
                {\sqrt{\alpha_1^2+\alpha_2^2}}

             **Per-particle parameters**:

             * :class:`openmm.openmm.CustomNonbondedForce`:
               :math:`(q_i,\,\alpha_i)`.
             * :class:`openmm.openmm.NonbondedForce`:
               :math:`(q_i,\,\sigma_i,\,\epsilon_i)`.

           * :code:`"core"`: Gaussian core model.

             .. math::

                \begin{gather*}
                  a_{12}^2=a_1^2+a_2^2\\
                  \alpha_{12}=\sqrt{\frac{\pi}{2a_{12}^2}}
                \end{gather*}

             This is equivalent to setting
             :math:`\alpha_i=\sqrt{\pi/(2a_i^2)}` in the default mixing
             rule.

             * :class:`openmm.openmm.CustomNonbondedForce`:
               :math:`(q_i,\,a_i)`.
             * :class:`openmm.openmm.NonbondedForce`:
               :math:`(q_i,\,\sigma_i,\,\epsilon_i)`.

           * :code:`"alpha12 = ...;"`: Custom mixing rule. The string
             containing the expression for :math:`\alpha_{12}` must be
             written in valid C++ syntax, with any custom global and
             per-particle parameters and tabulated functions defined in
             `global_params`, `per_params`, and `tab_funcs`,
             respectively.

             **Per-particle parameters**:

             * :class:`openmm.openmm.CustomNonbondedForce`:
               :math:`(q_i,\,\ldots)`.
             * :class:`openmm.openmm.NonbondedForce`:
               :math:`(q_i,\,\sigma_i,\,\epsilon_i)`.

        .. tip::

           To disable the Lennard-Jones potential, set
           :math:`\sigma_i=0\,\mathrm{nm}` and
           :math:`\epsilon_i=0\,\mathrm{kJ/mol}` for all particles.

    global_params : `dict`, keyword-only, optional
        Additional global parameters for use in the definition of
        :math:`\alpha_{12}`.

    per_params : `list`, keyword-only, optional
        Additional per-particle parameters for use in the definition of
        :math:`\alpha_{12}`.

    tab_funcs : `dict`, keyword-only, optional
        Optional tabulated functions for use in the definition of
        :math:`\alpha_{12}`.

    Returns
    -------
    pair_coul_gauss_dir : `openmm.CustomNonbondedForce`
        Short-range electrostatic contribution evaluated in real space.

    pair_coul_gauss_rec : `openmm.NonbondedForce`
        Long-range electrostatic contribution evaluated in Fourier
        (reciprocal) space.

    References
    ----------
    .. [1] Kiss, P. T.; Sega, M.; Baranyai, A. Efficient Handling of
       Gaussian Charge Distributions: An Application to Polarizable
       Molecular Models. *J. Chem. Theory Comput.* **2014**, *10* (12),
       5513–5519. https://doi.org/10.1021/ct5009069.

    .. [2] Eslami, H.; Khani, M.; Müller-Plathe, F. Gaussian Charge
       Distributions for Incorporation of Electrostatic Interactions in
       Dissipative Particle Dynamics: Application to Self-Assembly of
       Surfactants. *J. Chem. Theory Comput.* **2019**, *15* (7),
       4197–4207. https://doi.org/10.1021/acs.jctc.9b00174.
    """

    if g_ewald is None:
        g_ewald = np.sqrt(-np.log(2 * tol)) / cutoff

    if global_params is None:
        global_params = {}
    global_params |= {
        "G_EWALD": g_ewald,
        "ONE_4PI_EPS0": unit.AVOGADRO_CONSTANT_NA / (4 * np.pi
                                                     * VACUUM_PERMITTIVITY)
    }
    if mix == "default":
        mix = "alpha12=alpha1*alpha2/sqrt(alpha1^2+alpha2^2);"
        per_params = ["alpha"]
    elif mix == "core":
        mix = f"alpha12=sqrt({np.pi}/(2*(a1^2+a2^2)));"
        per_params = ["a"]

    pair_coul_gauss_dir = openmm.CustomNonbondedForce(
        f"ONE_4PI_EPS0*q1*q2*(erf(alpha12*r)-erf(G_EWALD*r))/r;{mix}"
    )
    pair_coul_gauss_dir.addPerParticleParameter("q")
    _setup_pair(pair_coul_gauss_dir, cutoff, global_params, per_params,
                tab_funcs)
    pair_coul_gauss_rec = lj_coul(cutoff, tol, g_ewald=g_ewald, dims=dims)
    pair_coul_gauss_rec.setIncludeDirectSpace(False)
    return pair_coul_gauss_dir, pair_coul_gauss_rec

def dpd(
        cutoff: Union[float, unit.Quantity],
        cutoff_dpd: Union[float, unit.Quantity] = None, *,
        mix: str = None, per_params: list = None,
        global_params: dict[str, Union[float, unit.Quantity]] = None,
        tab_funcs: dict[str, Union[np.ndarray[Union[int, float]],
                                   unit.Quantity,
                                   openmm.Discrete2DFunction]] = None
    ) -> openmm.CustomNonbondedForce:

    r"""
    Implements the conservative part of the dissipative particle
    dynamics (DPD) force.

    .. note::

       This does not include an implementation of the DPD thermostat.

    The potential energy between two DPD beads is given by:

    .. math::

       u_\mathrm{DPD}(r_{12})=\frac{1}{2}A_{12}r_\mathrm{cut}
       \left(1-\frac{r}{r_\mathrm{cut}}\right)^2

    where :math:`A_{12}` is the conservative force parameter in
    :math:`\mathrm{kJ/(mol\cdot nm)}` and :math:`r_\mathrm{cut}` is the
    interaction cutoff distance in :math:`\mathrm{nm}`.

    As :math:`A_{12}` has no well-defined mixing rule, it must be

    * evaluated using a custom mixing rule in `mix` with necessary
      per-particle parameters in `per_params`,
    * specified as a global parameter in `global_params`, or
    * provided for each pair of atom types in `tab_funcs` while
      specifying :code:`mix="A12=A(type1,type2);"` and
      :code:`per_params=("type",)`.

    After creating the pair potential, particles should be registered
    using :meth:`openmm.openmm.CustomNonbondedForce.addParticle`.

    Parameters
    ----------
    cutoff : `float` or `openmm.unit.Quantity`
        Shared cutoff distance for all nonbonded interactions in the
        simulation sytem. Must be less than half the minimum periodic
        simulation box dimension.

        **Reference unit**: :math:`\mathrm{nm}`.

    cutoff_dpd : `float` or `openmm.unit.Quantity`, optional
        Cutoff distance :math:`r_\mathrm{cut}` for the DPD potential.
        Must be less than the shared cutoff distance. If not provided,
        it is set to the shared cutoff distance.

        **Reference unit**: :math:`\mathrm{nm}`.

    mix : `str`, keyword-only, optional
        Mixing rule for :math:`A_{12}`. Must be defined if
        :math:`A_{12}` is not a global parameter. The string containing
        the expression for :math:`A_{12}` must be written in valid C++
        syntax, with any custom global and per-particle parameters and
        tabulated functions defined in `global_params`, `per_params`,
        and `tab_funcs`, respectively.

    global_params : `dict`, keyword-only, optional
        Additional global parameters for use in the definition of
        :math:`A_{12}`.

    per_params : `list`, keyword-only, optional
        Additional per-particle parameters for use in the definition of
        :math:`A_{12}`.

    tab_funcs : `dict`, keyword-only, optional
        Optional tabulated functions for use in the definition of
        :math:`A_{12}`.

    Returns
    -------
    pair_dpd : `openmm.CustomNonbondedForce`
        DPD pair potential.
    """

    if isinstance(cutoff, unit.Quantity):
        cutoff = cutoff.value_in_unit(unit.nanometer)
    if cutoff_dpd is None:
        cutoff_dpd = cutoff
    else:
        if isinstance(cutoff_dpd, unit.Quantity):
            cutoff_dpd = cutoff_dpd.value_in_unit(unit.nanometer)
        if cutoff_dpd > cutoff:
            emsg = ("The cutoff distance for the dissipative particle "
                    "dynamics (DPD) potential must be less than the "
                    "shared cutoff distance.")
            raise ValueError(emsg)

    energy = f"0.5*A12*{cutoff_dpd}*(1-r/{cutoff_dpd})^2;"
    if mix:
        energy = f"{energy}{mix}"

    pair_dpd = openmm.CustomNonbondedForce(energy)
    _setup_pair(pair_dpd, cutoff, global_params, per_params, tab_funcs)
    return pair_dpd

def gauss(
        cutoff: Union[float, unit.Quantity],
        cutoff_gauss: Union[float, unit.Quantity] = None, *,
        shift: bool = True, mix: str = "geometric",
        global_params: dict[str, Union[float, unit.Quantity]] = None,
        per_params: list = None,
        tab_funcs: dict[str, Union[np.ndarray[Union[int, float]],
                                   unit.Quantity,
                                   openmm.Discrete2DFunction]] = None
    ) -> openmm.CustomNonbondedForce:

    r"""
    Implements the generalized intermolecular Gaussian pair potential.

    The potential energy between two Gaussian particles is given by

    .. math::

       u_\mathrm{Gauss}(r_{12})=\alpha_{12}\exp{(-\beta_{12}r_{12}^2)}

    where :math:`\alpha_{12}` is a magnitude scaling constant in
    :math:`\mathrm{kJ/mol}` and :math:`\beta_{12}` is a screening length
    in :math:`\mathrm{nm}`.

    In addition to arithmetic and geometric mixing rules for
    :math:`\alpha_{12}` and :math:`\beta_{12}`, an implementation of the
    excluded volume pair potential in the Gaussian core model

    .. math::

        u_\mathrm{Gauss}(r_{12})=A\left(\frac{3}{2\pi\sigma_{12}^2}
        \right)^{3/2}\exp{\left(-\frac{3}{2\sigma_{12}^2}r_{12}^2\right)}

    is available, where :math:`A` is the repulsion parameter in
    :math:`\mathrm{nm}^3\cdot\mathrm{kJ/mol}` and :math:`\sigma_{12}` is
    the average particle size in :math:`\mathrm{nm}`.

    After creating the pair potential, particles should be registered
    using :meth:`openmm.openmm.CustomNonbondedForce.addParticle`.

    Parameters
    ----------
    cutoff : `float` or `openmm.unit.Quantity`
        Shared cutoff distance for all nonbonded interactions in the
        simulation sytem. Must be less than half the minimum periodic
        simulation box dimension.

        **Reference unit**: :math:`\mathrm{nm}`.

    cutoff_gauss : `float` or `openmm.unit.Quantity`, optional
        Cutoff distance :math:`r_\mathrm{cut}` for the Gaussian
        potential. Must be less than the shared cutoff distance. If not
        provided, it is set to the shared cutoff distance.

        **Reference unit**: :math:`\mathrm{nm}`.

    shift : `bool`, keyword-only, default: :code:`True`
        Determines whether the Gaussian potential is shifted at its
        cutoff to :math:`0\,\mathrm{kJ/mol}`.

    mix : `str`, keyword-only, default: :code:`"geometric"`
        Mixing rule for :math:`\alpha_{12}` and :math:`\beta_{12}`.

        .. container::

           **Valid values**:

           * :code:`"arithmetic"`: Arithmetic combining rule.

             .. math::

                \begin{gather*}
                  \alpha_{12}=\sqrt{\alpha_1\alpha_2}\\
                  \beta_{12}=\frac{2}{\beta_1^{-1}+\beta_2^{-1}}
                \end{gather*}

             **Per-particle parameters**: :math:`(\alpha_i,\,\beta_i)`.

           * :code:`"geometric"`: Geometric mixing rule.

             .. math::

                \begin{gather*}
                  \alpha_{12}=\sqrt{\alpha_1\alpha_2}\\
                  \beta_{12}=\sqrt{\beta_1\beta_2}
                \end{gather*}

             **Per-particle parameters**: :math:`(\alpha_i,\,\beta_i)`.

           * :code:`"core"`: Gaussian core model.

             .. math::

                \begin{gather*}
                  \sigma_{12}^2=\sigma_1^2+\sigma_2^2\\
                  \beta_{12}=\frac{3}{2\sigma_{12}^2}\\
                  \alpha_{12}=A\left(
                  \frac{\beta_{12}}{\pi}\right)^{3/2}
                \end{gather*}

             **Global parameter**: :math:`A`.

             **Per-particle parameter**: :math:`\sigma_i`.

           * :code:`"alpha12 = ...; beta12 = ...;"`: Custom mixing rule.
             The string containing the expressions for
             :math:`\alpha_{12}` and :math:`\beta_{12}` must be written
             in valid C++ syntax, with any custom global and
             per-particle parameters and tabulated functions defined in
             `global_params`, `per_params`, and `tab_funcs`,
             respectively.

    global_params : `dict`, keyword-only, optional
        Optional global parameters for use in the definition of
        :math:`\alpha_{12}` and :math:`\beta_{12}`.

    per_params : `list`, keyword-only, optional
        Optional per-particle parameters for use in the definition of
        :math:`\alpha_{12}` and :math:`\beta_{12}`.

    tab_funcs : `dict`, keyword-only, optional
        Optional tabulated functions for use in the definition of
        :math:`\alpha_{12}` and :math:`\beta_{12}`.

    Returns
    -------
    pair_gauss : `openmm.CustomNonbondedForce`
        Gaussian pair potential.
    """

    if isinstance(cutoff, unit.Quantity):
        cutoff = cutoff.value_in_unit(unit.nanometer)
    if cutoff_gauss is None:
        cutoff_gauss = cutoff
    else:
        if isinstance(cutoff_gauss, unit.Quantity):
            cutoff_gauss = cutoff_gauss.value_in_unit(unit.nanometer)
        if cutoff_gauss > cutoff:
            emsg = ("The cutoff distance for the Gaussian potential "
                    "must be less than the shared cutoff distance.")
            raise ValueError(emsg)

    prefix = f"step({cutoff_gauss}-r)*(" if cutoff != cutoff_gauss else "("
    root = "alpha12*exp(-beta12*r^2)"
    suffix = (f"-ucut);ucut=alpha12*exp(-beta12*{cutoff_gauss}^2);"
              if shift else ");")
    if mix == "arithmetic":
        mix = "alpha12=sqrt(alpha1*alpha2);beta12=2/(1/beta1+1/beta2);"
        per_params = ["alpha", "beta"]
    elif mix == "geometric":
        mix = "alpha12=sqrt(alpha1*alpha2);beta12=sqrt(beta1*beta2);"
        per_params = ["alpha", "beta"]
    elif "core" in mix:
        mix = mix.replace(
            "core",
            f"alpha12=A*(beta12/{np.pi})^(3/2);beta12=3/(2*sigma12sq);"
            "sigma12sq=sigma1^2+sigma2^2"
        )
        if not mix.endswith(";"):
            mix += ";"
        if "A" not in mix and "A" not in global_params:
            raise ValueError("Global parameter 'A' not specified.")
        if per_params is None:
            per_params = []
        per_params += ["sigma"]

    pair_gauss = openmm.CustomNonbondedForce(f"{prefix}{root}{suffix}{mix}")
    _setup_pair(pair_gauss, cutoff, global_params, per_params, tab_funcs)
    return pair_gauss

def lj_coul(
        cutoff: Union[float, unit.Quantity], tol: float = 1e-4, *,
        g_ewald: Union[float, unit.Quantity] = None,
        dims: Union[float, unit.Quantity] = None) -> openmm.NonbondedForce:

    r"""
    Implements the standard Lennard-Jones (LJ) and Coulomb pair
    potentials.

    The potential energy between two charged LJ particles is given by

    .. math::

       \begin{gather*}
         u_\mathrm{Coul}(r_{12})=\frac{q_1q_2}{4\pi\varepsilon_0r_{12}}\\
         u_\mathrm{LJ}(r_{12})=4\epsilon_{12}\left[\left(
         \frac{\sigma_{12}}{r_{12}}\right)^{12}
         -\left(\frac{\sigma_{12}}{r_{12}} \right)^6\right]
       \end{gather*}

    where :math:`q_1` and :math:`q_2` are the particle charges in
    :math:`\mathrm{e}`, :math:`\varepsilon_0` is the vacuum
    permittivity, :math:`\sigma_{12}` is the averged particle size in
    :math:`\mathrm{nm}`, and :math:`\epsilon_{12}` is the dispersion
    energy in :math:`\mathrm{kJ/mol}`. :math:`q_1`, :math:`q_2`,
    :math:`\sigma_{12}`, and :math:`\epsilon_{12}` are determined from
    per-particle parameters `charge`, `sigma`, and `epsilon`,
    respectively, which are set in the main script using
    :meth:`openmm.openmm.NonbondedForce.addParticle`.

    The mixing rule for :math:`\sigma_{12}` and :math:`\epsilon_{12}` is
    the Lorentz–Berthelot combining rule:

    .. math::

       \begin{gather*}
         \sigma_{12}=\frac{\sigma_1+\sigma_2}{2}\\
         \epsilon_{12}=\sqrt{\epsilon_1\epsilon_2}
       \end{gather*}

    To disable the LJ potential and use a different excluded volume
    interaction potential but keep the Coulomb potential, set
    :math:`\sigma_{12}=0\,\mathrm{nm}` and
    :math:`\epsilon_{12}=0\,\mathrm{kJ/mol}` for all particles.

    To account for a solvent implicitly using its relative permittivity
    :math:`\varepsilon_\mathrm{r}`, scale the particle charges
    :math:`q_i` by :math:`1/\sqrt{\varepsilon_\mathrm{r}}`.

    Parameters
    ----------
    cutoff : `float` or `openmm.unit.Quantity`
        Shared cutoff distance :math:`r_\mathrm{cut}` for all nonbonded
        interactions in the simulation sytem. Must be less than half the
        minimum periodic simulation box dimension.

        **Reference unit**: :math:`\mathrm{nm}`.

    tol : `float`, default: :code:`1e-4`
        Error tolerance :math:`\delta` for Ewald summation.

    g_ewald : `float` or `openmm.unit.Quantity`, keyword-only, optional
        Ewald splitting parameter :math:`g_\mathrm{Ewald}`. If not
        provided, it is computed using

        .. math::

           g_\mathrm{Ewald}=\frac{\sqrt{-\log(2\delta)}}{r_\mathrm{cut}}

        **Reference unit**: :math:`\mathrm{nm}^{-1}`.

    dims : `float` or `openmm.unit.Quantity`, keyword-only, optional
        Simulation system dimensions. Must be provided with `g_ewald` to
        calculate the number of mesh nodes :math:`n_\mathrm{mesh}`.
        Both `dims` and `g_ewald` must either have units or be unitless.

        **Shape**: :math:`(3,)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    Returns
    -------
    pair_lj_coul : `openmm.NonbondedForce`
        Hybrid LJ and Coulomb pair potentials.
    """

    pair_lj_coul = openmm.NonbondedForce()
    pair_lj_coul.setCutoffDistance(cutoff)
    pair_lj_coul.setNonbondedMethod(openmm.NonbondedForce.PME)
    if g_ewald is None or dims is None:
        pair_lj_coul.setEwaldErrorTolerance(tol)
    else:
        n_mesh = np.ceil(2 * g_ewald * dims / (3 * tol ** (1 / 5)))
        pair_lj_coul.setPMEParameters(g_ewald, *n_mesh)
    return pair_lj_coul

def ljts(
        cutoff: Union[float, unit.Quantity],
        cutoff_ljts: Union[float, unit.Quantity] = None, *,
        coefs: Union[dict[str, float], tuple[float, float, float]] = (1, 1, 4),
        powers: Union[dict[str, float], tuple[float, float]] = (12, 6),
        shift: bool = True, mix: str = "arithmetic", mie: bool = False,
        wca: bool = False,
        global_params: dict[str, Union[float, unit.Quantity]] = None,
        per_params: list = None,
        tab_funcs: dict[str, Union[np.ndarray[Union[int, float]],
                                   unit.Quantity,
                                   openmm.Discrete2DFunction]] = None
    ) -> openmm.CustomNonbondedForce:

    r"""
    Implements the generalized Lennard-Jones truncated and shifted
    (LJTS) pair potential and its derivatives, such as the Mie
    and Weeks–Chandler–Andersen (WCA) potentials.

    The generalized LJ potential is given by

    .. math::

       u_\mathrm{LJ}(r_{12})=C\epsilon_{12}
       \left[A\left(\frac{\sigma_{12}}{r_{12}}\right)^{\gamma_\mathrm{r}}
       -B\left(\frac{\sigma_{12}}{r_{12}}\right)^{\gamma_\mathrm{a}}\right]

    where :math:`\sigma_{12}` is the average particle size in
    :math:`\mathrm{nm}`, :math:`\epsilon_{12}` is the dispersion
    energy in :math:`\mathrm{kJ/mol}`, and :math:`A`, :math:`B`,
    :math:`C`, :math:`\gamma_\mathrm{r}`, and :math:`\gamma_\mathrm{a}`
    are constants.

    The standard 12-6 LJ potential, which is the most frequently used
    form, uses :math:`A=1`, :math:`B=1`, :math:`C=4`,
    :math:`\gamma_\mathrm{r}=12`, and :math:`\gamma_\mathrm{a}=6`.

    While :math:`\gamma_\mathrm{a}=6` is justified by the London
    dispersion force, no such rationale exists for
    :math:`\gamma_\mathrm{r}=12`. By relaxing the restrictions on what
    values :math:`\gamma_\mathrm{r}` and :math:`\gamma_\mathrm{a}` can
    take on and keeping :math:`A=1` and :math:`B=1`, the Mie potential
    is obtained:

    .. math::

       \begin{gather*}
         u_\mathrm{Mie}(r_{12})=C_\mathrm{Mie}\epsilon_{12}
         \left[\left(\frac{\sigma_{12}}{r_{12}}\right)^{\gamma_\mathrm{r}}
         -\left(\frac{\sigma_{12}}{r_{12}}\right)^{\gamma_\mathrm{a}}\right]\\
         C_\mathrm{Mie}=\frac{\gamma_\mathrm{r}}
         {\gamma_\mathrm{r}-\gamma_\mathrm{a}}
         \left(\frac{\gamma_\mathrm{r}}{\gamma_\mathrm{a}}\right)
         ^{\gamma_\mathrm{a}/(\gamma_\mathrm{r}-\gamma_\mathrm{a})}
       \end{gather*}

    In dense fluids, it is common to use a purely repulsive potential.
    The standard 12-6 LJ potential can be truncated and shifted at a
    cutoff of :math:`r_\mathrm{cut}=2^{1/6}\sigma_{12}` to get the WCA
    potential, which is widely employed as a reference potential in
    simulations of liquid systems. A generalized WCA potential can be
    acquired by truncating and shifting the Mie potential instead at a
    cutoff of :math:`r_\mathrm{WCA}=(\gamma_\mathrm{r}/\gamma_\mathrm{a})
    ^{1/(\gamma_\mathrm{r}-\gamma_\mathrm{a})}\sigma_{12}`:

    .. math::

       u_\mathrm{WCA}(r_{12})=\begin{cases}
         C_\mathrm{Mie}\epsilon_{12}\left[
         \left(\dfrac{\sigma_{12}}{r_{12}}\right)^{\gamma_\mathrm{r}}
         -\left(\dfrac{\sigma_{12}}{r_{12}}\right)^{\gamma_\mathrm{a}}
         \right]+\epsilon_{12},&\mathrm{if}\,r_{12}<r_\mathrm{WCA}\\
         0,&\mathrm{if}\,r_{12}\geq r_\mathrm{WCA}
       \end{cases}

    In the case where flexibility is allowed for all constants and the
    cutoff, the generalized LJTS potential is obtained:

    .. math::

       u_\mathrm{LJTS}(r_{12})=\begin{cases}
         u_\mathrm{LJ}(r_{12})-u_\mathrm{LJ}(r_\mathrm{cut}),
         &\mathrm{if}\,r_{12}<r_\mathrm{cut}\\
         0,&\mathrm{if}\,r_{12}\geq r_\mathrm{cut}
       \end{cases}

    After creating the pair potentials, particles should be registered
    using :meth:`openmm.openmm.CustomNonbondedForce.addParticle`.

    Parameters
    ----------
    cutoff : `float` or `openmm.unit.Quantity`
        Shared cutoff distance for all nonbonded interactions in the
        simulation sytem. Must be less than half the minimum periodic
        simulation box dimension.

        **Reference unit**: :math:`\mathrm{nm}`.

    cutoff_ljts : `float` or `openmm.unit.Quantity`, optional
        Cutoff distance :math:`r_\mathrm{cut}` for the LJTS potential.
        Must be less than the shared cutoff distance. If not provided,
        it is set to the shared cutoff distance. If the WCA potential is
        used, the dynamic cutoff :math:`r_\mathrm{WCA}` is used instead.

        **Reference unit**: :math:`\mathrm{nm}`.

    coefs : `tuple` or `dict`, keyword-only, default: :code:`(1, 1, 4)`
        Coefficients :math:`A`, :math:`B`, and :math:`C`, in that order.
        If a `dict` is provided, use the keys :code:`"A"`, :code:`"B"`,
        and :code:`"C"`, respectively. If the Mie or WCA potential is
        used, the appropriate coefficients are used instead.

    powers : `tuple` or `dict`, keyword-only, default: :code:`(12, 6)`
        Powers :math:`\gamma_\mathrm{r}` and :math:`\gamma_\mathrm{a}`,
        in that order. If a `dict` is provided, use the keys :code:`"r"`
        and :code:`"a"`, respectively.

    shift : `bool`, keyword-only, default: :code:`True`
        Determines whether the LJTS potential is shifted at its cutoff
        to :math:`0\,\mathrm{kJ/mol}`. If the WCA potential is used, a
        dynamic shift of :math:`\epsilon_{12}` is used instead.

    mie : `bool`, keyword-only, default: :code:`False`
        Determines whether to use the Mie potential. If :code:`True`,
        `coefs` is superseded by the correct values for the Mie
        potential. Only one of `mie` and `wca` can be :code:`True`.

    wca : `bool`, keyword-only, default: :code:`False`
        Determines whether to use the WCA potential. If :code:`True`,
        `cutoff_ljts`, `coefs`, and `shift` are superseded by the
        correct values for the WCA potential. Only one of `mie` and
        `wca` can be :code:`True`.

    mix : `str`, keyword-only, default: :code:`"arithmetic"`
        Mixing rule for :math:`\sigma_{12}` and :math:`\epsilon_{12}`.

        .. container::

           **Valid values**:

           * :code:`"arithmetic"`: Lorentz-Berthelot combining rule.

             .. math::

                \begin{gather*}
                  \sigma_{12}=\frac{\sigma_1+\sigma_2}{2}\\
                  \epsilon_{12}=\sqrt{\epsilon_1\epsilon_2}
                \end{gather*}

             **Per-particle parameters**: :math:`(\sigma_i,\,\epsilon_i)`.

           * :code:`"geometric"`: Geometric mixing rule.

             .. math::

                \begin{gather*}
                  \sigma_{12}=\sqrt{\sigma_1\sigma_2}\\
                  \epsilon_{12}=\sqrt{\epsilon_1\epsilon_2}
                \end{gather*}

             **Per-particle parameters**: :math:`(\sigma_i,\,\epsilon_i)`.

           * :code:`"sixthpower"`: Sixth-power mixing rule.

             .. math::

                \begin{gather*}
                  \sigma_{12}=\left(\frac{\sigma_1^6
                  +\sigma_2^6}{2}\right)^{1/6}\\
                  \epsilon_{12}=\frac{2\sqrt{\epsilon_1\epsilon_2}
                  \sigma_1^3\sigma_2^3}{\sigma_1^6+\sigma_2^6}
                \end{gather*}

             **Per-particle parameters**: :math:`(\sigma_i,\,\epsilon_i)`.

           * :code:`"sigma12 = ...; epsilon12 = ...;"`: Custom mixing
             rule. The string containing the expression for
             :math:`\sigma_{12}` and :math:`\epsilon_{12}` must be
             written in valid C++ syntax, with any custom global and
             per-particle parameters and tabulated functions defined in
             `global_params`, `per_params`, and `tab_funcs`,
             respectively.

    global_params : `dict`, keyword-only, optional
        Additional global parameters for use in the definition of
        :math:`\sigma_{12}` and :math:`\epsilon_{12}`.

    per_params : `list`, keyword-only, optional
        Additional per-particle parameters for use in the definition of
        :math:`\sigma_{12}` and :math:`\epsilon_{12}`.

    tab_funcs : `dict`, keyword-only, optional
        Optional tabulated functions for use in the definition of
        :math:`\sigma_{12}` and :math:`\epsilon_{12}`.

    Returns
    -------
    pair_ljts : `openmm.CustomNonbondedForce`
        LJTS pair potential.
    """

    if isinstance(cutoff, unit.Quantity):
        cutoff = cutoff.value_in_unit(unit.nanometer)
    if cutoff_ljts is None:
        cutoff_ljts = cutoff
    else:
        if isinstance(cutoff_ljts, unit.Quantity):
            cutoff_ljts = cutoff_ljts.value_in_unit(unit.nanometer)
        if cutoff_ljts > cutoff:
            emsg = ("The cutoff distance for the LJTS potential must be "
                    "less than the shared cutoff distance.")
            raise ValueError(emsg)

    if mie and wca:
        raise ValueError("Both 'mie' and 'wca' are set to True.")
    elif mie or wca:
        coef_mie = (powers[0] / (powers[0] - powers[1])
                    * (powers[0] / powers[1]) ** (powers[1] / (powers[0] - powers[1])))
    if isinstance(powers, dict):
        powers = (powers["r"], powers["a"])
    if wca:
        cutoff_wca = (powers[0] / powers[1]) ** (1 / (powers[0] - powers[1]))
        root = (f"{coef_mie}*epsilon12*((sigma12/r)^{powers[0]}"
                f"-(sigma12/r)^{powers[1]})")
        prefix = f"step({cutoff_wca}*sigma12-r)*("
        suffix = "+epsilon12);"
    else:
        if mie:
            coefs = (1, 1, coef_mie)
        elif isinstance(coefs, dict):
            coefs = (coefs["A"], coefs["B"], coefs["C"])
        root = (f"{coefs[2]}*epsilon12*({coefs[0]}*(sigma12/r)^{powers[0]}"
                f"-{coefs[1]}*(sigma12/r)^{powers[1]})")
        prefix = f"step({cutoff_ljts}-r)*(" if cutoff != cutoff_ljts else "("
        suffix = ((f"-ucut);ucut={coefs[2]}*epsilon12"
                   f"*({coefs[0]}*(sigma12/{cutoff_ljts})^{powers[0]}"
                   f"-{coefs[1]}*(sigma12/{cutoff_ljts})^{powers[1]});")
                  if shift else ");")
    if mix == "arithmetic":
        mix = "sigma12=(sigma1+sigma2)/2;epsilon12=sqrt(epsilon1*epsilon2);"
        per_params = ["sigma", "epsilon"]
    elif mix == "geometric":
        mix = "sigma12=sqrt(sigma1*sigma2);epsilon12=sqrt(epsilon1*epsilon2);"
        per_params = ["sigma", "epsilon"]
    elif mix == "sixthpower":
        mix = ("sigma12=((sigma1^6+sigma2^6)/2)^(1/6);"
               "epsilon12=2*sqrt(epsilon1*epsilon2)*sigma1^3*sigma2^3"
               "/(sigma1^6+sigma2^6);")
        per_params = ["sigma", "epsilon"]

    pair_ljts = openmm.CustomNonbondedForce(f"{prefix}{root}{suffix}{mix}")
    _setup_pair(pair_ljts, cutoff, global_params, per_params, tab_funcs)
    return pair_ljts

def solvation(
        cutoff: Union[float, unit.Quantity],
        cutoff_solvation: Union[float, unit.Quantity] = None, *,
        mix: str = "arithmetic", per_params: list = None,
        global_params: dict[str, Union[float, unit.Quantity]] = None,
        tab_funcs: dict[str, Union[np.ndarray[Union[int, float]],
                                   openmm.unit.Quantity,
                                   openmm.Discrete2DFunction]] = None
    ) -> openmm.CustomNonbondedForce:

    r"""
    Implements the solvation pair potential.

    The solvation energy between particles is given by

    .. math::

       u_\mathrm{solv}(r_{12})=-S_{12}\left[\left(
       \frac{\sigma_{12}}{r_{12}}\right)^4
       -\left(\frac{\sigma_{12}}{r_\mathrm{cut}}\right)^4\right]

    where :math:`\sigma_{12}` is the size of the particle in
    :math:`\mathrm{nm}` and :math:`S_{12}` is the solvation strength
    in :math:`\mathrm{kJ/mol}`.

    After creating the pair potential, particles should be registered
    using :meth:`openmm.openmm.CustomNonbondedForce.addParticle`.

    Parameters
    ----------
    cutoff : `float` or `openmm.unit.Quantity`
        Shared cutoff distance for all nonbonded interactions in the
        simulation sytem. Must be less than half the minimum periodic
        simulation box dimension.

        **Reference unit**: :math:`\mathrm{nm}`.

    cutoff_solvation : `float` or `openmm.unit.Quantity`, optional
        Cutoff distance :math:`r_\mathrm{cut}` for the solvation
        potential. Must be less than the shared cutoff distance. If not
        provided, it is set to the shared cutoff distance.

        **Reference unit**: :math:`\mathrm{nm}`.

    mix : `str`, keyword-only, default: :code:`"arithmetic"`
        Mixing rule for :math:`\sigma_{12}` and :math:`S_{12}`.

        .. container::

           **Valid values**:

           * :code:`"arithmetic"`: Arithmetic mixing rule.

             .. math::

                \begin{gather*}
                  \sigma_{12}=\frac{\sigma_1+\sigma_2}{2}\\
                  S_{12}=\sqrt{S_1S_2}
                \end{gather*}

             **Per-particle parameters**: :math:`(\sigma_i, S_i)`.

           * :code:`"geometric"`: Geometric mixing rule.

             .. math::

                \begin{gather*}
                  \sigma_{12}=\sqrt{\sigma_1\sigma_2}\\
                  S_{12}=\sqrt{S_1S_2}
                \end{gather*}

             **Per-particle parameters**: :math:`(\sigma_i, S_i)`.

           * :code:`"sigma12 = ...; S12 = ...;"`: Custom mixing rule.
             The string containing the expressions for
             :math:`\sigma_{12}` and :math:`S_{12}` must be written
             in valid C++ syntax, with any custom global and
             per-particle parameters and tabulated functions defined in
             `global_params`, `per_params`, and `tab_funcs`,
             respectively.

    global_params : `dict`, keyword-only, optional
        Additional global parameters for use in the definition of
        :math:`\sigma_{12}` and :math:`S_{12}`.

    per_params : `list`, keyword-only, optional
        Additional per-particle parameters for use in the definition of
        :math:`\sigma_{12}` and :math:`S_{12}`.

    tab_funcs : `dict`, keyword-only, optional
        Optional tabulated functions for use in the definition of
        :math:`\sigma_{12}` and :math:`S_{12}`.

    Returns
    -------
    pair_solv : `openmm.CustomNonbondedForce`
        Solvation pair potential.
    """

    if isinstance(cutoff, unit.Quantity):
        cutoff = cutoff.value_in_unit(unit.nanometer)
    if cutoff_solvation is None:
        cutoff_solvation = cutoff
    else:
        if isinstance(cutoff_solvation, unit.Quantity):
            cutoff_solvation = cutoff_solvation.value_in_unit(unit.nanometer)
        if cutoff_solvation > cutoff:
            emsg = ("The cutoff distance for the solvation potential "
                    "must be less than the shared cutoff distance.")
            raise ValueError(emsg)

    root = "-S12*((sigma12/r)^4-(sigma12/cut)^4)"
    if mix == "arithmetic":
        mix = "sigma12=(sigma1+sigma2)/2;S12=sqrt(S1*S2);"
        per_params = ["sigma", "S"]
    elif mix == "geometric":
        mix = "sigma12=sqrt(sigma1*sigma2);S12=sqrt(S1*S2);"
        per_params = ["sigma", "S"]

    pair_solv = openmm.CustomNonbondedForce(f"{root}{mix}")
    _setup_pair(pair_solv, cutoff, global_params, per_params, tab_funcs)
    return pair_solv

def wca(cutoff: Union[float, unit.Quantity], *, mix: str = "arithmetic",
        powers: Union[dict[str, float], tuple[float, float]] = (12, 6),
        global_params: dict[str, Union[float, unit.Quantity]] = None,
        per_params: list = None,
        tab_funcs: dict[str, Union[np.ndarray[Union[int, float]],
                                   unit.Quantity,
                                   openmm.Discrete2DFunction]] = None
    ) -> openmm.CustomNonbondedForce:

    r"""
    Implements the generalized Weeks–Chander–Andersen (WCA) pair
    potential.

    The potential energy between two WCA particles is given by

    .. math::

       \begin{gather*}
         u_\mathrm{WCA}(r_{12})=\begin{cases}
           C_\mathrm{Mie}\epsilon_{12}\left[
           \left(\dfrac{\sigma_{12}}{r_{12}}\right)^{\gamma_\mathrm{r}}
           -\left(\dfrac{\sigma_{12}}{r_{12}}\right)^{\gamma_\mathrm{a}}
           \right]+\epsilon_{12},&\mathrm{if}\,r_{12}<r_\mathrm{WCA}\\
           0,&\mathrm{if}\,r_{12}\geq r_\mathrm{WCA}
         \end{cases}\\
         C_\mathrm{Mie}=\frac{\gamma_\mathrm{r}}
         {\gamma_\mathrm{r}-\gamma_\mathrm{a}}
         \left(\frac{\gamma_\mathrm{r}}{\gamma_\mathrm{a}}\right)
         ^{\gamma_\mathrm{a}/(\gamma_\mathrm{r}-\gamma_\mathrm{a})}
       \end{gather*}

    where :math:`\sigma_{12}` is the average particle size in
    :math:`\mathrm{nm}`, :math:`\epsilon_{12}` is the dispersion energy
    in :math:`\mathrm{kJ/mol}`, :math:`\gamma_\mathrm{r}` and
    :math:`\gamma_\mathrm{a}` are the powers for the repulsive and
    attractive terms, respectively, and
    :math:`r_\mathrm{WCA}=(\gamma_\mathrm{r}/\gamma_\mathrm{a})
    ^{1/(\gamma_\mathrm{r}-\gamma_\mathrm{a})}\sigma_{12}` is the
    cutoff distance.

    After creating the pair potentials, particles should be registered
    using :meth:`openmm.openmm.CustomNonbondedForce.addParticle`.

    Parameters
    ----------
    cutoff : `float` or `openmm.unit.Quantity`
        Shared cutoff distance :math:`r_\mathrm{cut}` for all nonbonded
        interactions in the simulation sytem. Must be less than half the
        minimum periodic simulation box dimension.

        **Reference unit**: :math:`\mathrm{nm}`.

    powers : `tuple` or `dict`, keyword-only, default: :code:`(12, 6)`
        Powers :math:`\gamma_\mathrm{r}` and :math:`\gamma_\mathrm{a}`,
        in that order. If a `dict` is provided, use the keys :code:`"r"`
        and :code:`"a"`, respectively.

    mix : `str`, keyword-only, default: :code:`"arithmetic"`
        Mixing rule for :math:`\sigma_{12}` and :math:`\epsilon_{12}`.

        .. container::

           **Valid values**:

           * :code:`"arithmetic"`: Lorentz-Berthelot combining rule.

             .. math::

                \begin{gather*}
                  \sigma_{12}=\frac{\sigma_1+\sigma_2}{2}\\
                  \epsilon_{12}=\sqrt{\epsilon_1\epsilon_2}
                \end{gather*}

             **Per-particle parameters**: :math:`(\sigma_i,\,\epsilon_i)`.

           * :code:`"geometric"`: Geometric mixing rule.

             .. math::

                \begin{gather*}
                  \sigma_{12}=\sqrt{\sigma_1\sigma_2}\\
                  \epsilon_{12}=\sqrt{\epsilon_1\epsilon_2}
                \end{gather*}

             **Per-particle parameters**: :math:`(\sigma_i,\,\epsilon_i)`.

           * :code:`"sixthpower"`: Sixth-power mixing rule.

             .. math::

                \begin{gather*}
                  \sigma_{12}=\left(\frac{\sigma_1^6
                  +\sigma_2^6}{2}\right)^{1/6}\\
                  \epsilon_{12}=\frac{2\sqrt{\epsilon_1\epsilon_2}
                  \sigma_1^3\sigma_2^3}{\sigma_1^6+\sigma_2^6}
                \end{gather*}

             **Per-particle parameters**: :math:`(\sigma_i,\,\epsilon_i)`.

           * :code:`"sigma12 = ...; epsilon12 = ...;"`: Custom mixing
             rule. The string containing the expression for
             :math:`\sigma_{12}` and :math:`\epsilon_{12}` must be
             written in valid C++ syntax, with any custom global and
             per-particle parameters and tabulated functions defined in
             `global_params`, `per_params`, and `tab_funcs`,
             respectively.

    global_params : `dict`, keyword-only, optional
        Additional global parameters for use in the definition of
        :math:`\sigma_{12}` and :math:`\epsilon_{12}`.

    per_params : `list`, keyword-only, optional
        Additional per-particle parameters for use in the definition of
        :math:`\sigma_{12}` and :math:`\epsilon_{12}`.

    tab_funcs : `dict`, keyword-only, optional
        Optional tabulated functions for use in the definition of
        :math:`\sigma_{12}` and :math:`\epsilon_{12}`.

    Returns
    -------
    pair_wca : `openmm.CustomNonbondedForce`
        WCA pair potential.
    """

    return ljts(cutoff, powers=powers, mix=mix, wca=True,
                global_params=global_params, per_params=per_params,
                tab_funcs=tab_funcs)

def yukawa(
        cutoff: Union[float, unit.Quantity],
        cutoff_yukawa: Union[float, unit.Quantity] = None, *,
        shift: bool = True, mix: str = "geometric", per_params: list = None,
        global_params: dict[str, Union[float, unit.Quantity]] = None,
        tab_funcs: dict[str, Union[np.ndarray[Union[int, float]],
                                   openmm.unit.Quantity,
                                   openmm.Discrete2DFunction]] = None
    ) -> openmm.CustomNonbondedForce:

    r"""
    Implements the generalized intermolecular Yukawa pair potential.

    The potential energy between two Yukawa particles is given by

    .. math::

       u_\mathrm{Yukawa}(r_{12})=\frac{\alpha_{12}
       \exp(-\kappa r_{12})}{r_{12}}

    where :math:`\alpha_{12}` is a magnitude scaling constant in
    :math:`\mathrm{nm}\cdot\mathrm{kJ/mol}` and :math:`\kappa` is a
    screening length in :math:`\mathrm{nm}^{-1}`.

    After creating the pair potential, particles should be registered
    using :meth:`openmm.openmm.CustomNonbondedForce.addParticle`.

    Parameters
    ----------
    cutoff : `float` or `openmm.unit.Quantity`
        Shared cutoff distance for all nonbonded interactions in the
        simulation sytem. Must be less than half the minimum periodic
        simulation box dimension.

        **Reference unit**: :math:`\mathrm{nm}`.

    cutoff_yukawa : `float` or `openmm.unit.Quantity`, optional
        Cutoff distance :math:`r_\mathrm{cut}` for the Yukawa potential.
        Must be less than the shared cutoff distance. If not provided,
        it is set to the shared cutoff distance.

        **Reference unit**: :math:`\mathrm{nm}`.

    shift : `bool`, keyword-only, default: :code:`True`
        Determines whether the Yukawa potential is shifted at its
        cutoff to :math:`0\,\mathrm{kJ/mol}`.

    mix : `str`, keyword-only, default: :code:`"geometric"`
        Mixing rule for :math:`\alpha_{12}`.

        .. container::

           **Valid values**:

           * :code:`"geometric"`: Geometric mixing rule.

             .. math::

                \alpha_{12}=\sqrt{\alpha_1\alpha_2}

             **Global parameter**: :math:`\kappa`.

             **Per-particle parameter**: :math:`\alpha_i`.

           * :code:`"alpha12 = ...;"`: Custom mixing rule. The string
             containing the expressions for :math:`\alpha_{12}` (and
             optionally, :math:`\kappa`) must be written in valid C++
             syntax, with any custom global and per-particle parameters
             and tabulated functions defined in `global_params`,
             `per_params`, and `tab_funcs`, respectively.

             Should you want to use the geometric mixing rule for
             :math:`\alpha_{12}` and specify an alternate definition for
             :math:`\kappa`, you can include :code:`"geometric;"` in the
             string but

             Be aware that the
             parameters and functions used in the geometric mixing rule
             will take precedence over those defined here.

    global_params : `dict`, keyword-only, optional
        Additional global parameters for use in the definition of
        :math:`\alpha_{12}` (and optionally, :math:`\kappa`).

    per_params : `list`, keyword-only, optional
        Additional per-particle parameters for use in the definition of
        :math:`\alpha_{12}` (and optionally, :math:`\kappa`).

    tab_funcs : `dict`, keyword-only, optional
        Optional tabulated functions for use in the definition of
        :math:`\alpha_{12}` (and optionally, :math:`\kappa`).

    Returns
    -------
    pair_yukawa : `openmm.CustomNonbondedForce`
        Yukawa pair potential.
    """

    if isinstance(cutoff, unit.Quantity):
        cutoff = cutoff.value_in_unit(unit.nanometer)
    if cutoff_yukawa is None:
        cutoff_yukawa = cutoff
    else:
        if isinstance(cutoff_yukawa, unit.Quantity):
            cutoff_yukawa = cutoff_yukawa.value_in_unit(unit.nanometer)
        if cutoff_yukawa > cutoff:
            emsg = ("The cutoff distance for the LJTS potential must be "
                    "less than the shared cutoff distance.")
            raise ValueError(emsg)

    prefix = f"step({cutoff_yukawa}-r)*(" if cutoff != cutoff_yukawa else "("
    root = "alpha12*exp(-kappa*r)/r"
    suffix = (f"-ucut);ucut=alpha12*exp(-kappa*{cutoff_yukawa})/{cutoff_yukawa};"
              if shift else ");")
    if "geometric" in mix:
        mix = mix.replace("geometric", "alpha12=sqrt(alpha1*alpha2)")
        if not mix.endswith(";"):
            mix += ";"
        if "kappa" not in mix and "kappa" not in global_params:
            raise ValueError("Global parameter 'kappa' not defined.")
        if per_params is None:
            per_params = []
        per_params += ["alpha"]

    pair_yukawa = openmm.CustomNonbondedForce(f"{prefix}{root}{suffix}{mix}")
    _setup_pair(pair_yukawa, cutoff, global_params, per_params, tab_funcs)
    return pair_yukawa