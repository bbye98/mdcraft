"""
OpenMM system extensions and tools
==================================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains implementations of common OpenMM system
transformations, like the slab correction method for pseudo-2D slab
systems, the method of image charges, and an applied potential
difference.
"""

import logging
from typing import Any, Union
import warnings

import mpmath
import numpy as np
import openmm
from openmm import app, unit
from scipy import special

from .unit import VACUUM_PERMITTIVITY

try:
    from openmm_ic import ICLangevinIntegrator
    FOUND_ICPLUGIN = True
except ImportError:
    try:
        from constvplugin import ConstVLangevinIntegrator as ICLangevinIntegrator
        FOUND_ICPLUGIN = True
    except ImportError:
        FOUND_ICPLUGIN = False

def register_particles(
        system: openmm.System, topology: app.Topology,
        N: int = 0, mass: Union[float, unit.Quantity] = 0.0, *,
        chain: app.Chain = None, element: app.Element = None, name: str = "",
        resname: str = "", nbforce: openmm.NonbondedForce = None,
        charge: Union[float, unit.Quantity] = 0.0,
        sigma: Union[float, unit.Quantity] = 0.0,
        epsilon: Union[float, unit.Quantity] = 0.0,
        cnbforces: dict[openmm.CustomNonbondedForce, tuple[Any]] = None
    ) -> None:

    """
    Sequentially registers particles of the same type to the simulation
    system, topology, and nonbonded pair potentials.

    Parameters
    ----------
    system : `openmm.System`
        OpenMM molecular system. Can be :code:`None` if particles are to
        be registered to the topology only.

    topology : `openmm.app.Topology`
        Topological information about an OpenMM system.

    N : `int`, default: :code:`0`
        Number of atom(s). If not specified, no particles are added.

    mass : `float` or `openmm.unit.Quantity`, default: :code:`0`
        Molar mass. If not specified, particles are massless and will
        not move.

        **Reference unit**: :math:`\\mathrm{g/mol}`.

    chain : `openmm.app.Chain`, keyword-only, optional
        Chain that the atom(s) should be added to. If not provided,
        a new chain is created for each atom.

    element : `openmm.app.Element`, keyword-only, optional
        Chemical element of the atom(s) to add.

    name : `str`, keyword-only, optional
        Name of the atom(s) to add.

    resname : `str`, keyword-only, optional
        Name of the residue(s) to add. If not specified, `name` is used
        if available.

    nbforce : `openmm.NonbondedForce`, keyword-only, optional
        Standard OpenMM pair potential object implementing the nonbonded
        Lennard-Jones and Coulomb potentials.

    charge : `float` or `openmm.unit.Quantity`, keyword-only, \
    default: :code`0`
        Charge :math:`q` of the atom(s) for use in the Coulomb potential
        in `nbforce`.

        **Reference unit**: :math:`\\mathrm{e}`.

    sigma : `float` or `openmm.unit.Quantity`, keyword-only, \
    default: :code`0`
        :math:`\\sigma` parameter of the Lennard-Jones potential in
        `nbforce`.

        **Reference unit**: :math:`\\mathrm{nm}`.

    epsilon : `float` or `openmm.unit.Quantity`, keyword-only, \
    default: :code`0`
        :math:`\\epsilon` parameter of the Lennard-Jones potential in
        `nbforce`.

        **Reference unit**: :math:`\\mathrm{kJ/mol}`.

    cnbforces : `dict`, keyword-only, optional
        Custom pair potential objects implementing other non-standard
        pair potentials and their corresponding per-particle parameters.

        **Example**: :code:`{gauss: (0.3 * unit.nanometer,)}`, where
        `gauss` is a custom Gaussian potential obtained using
        :func:`mdcraft.openmm.pair.gauss()`.
    """

    has_nbforce = nbforce is not None
    has_system = system is not None
    per_chain = chain is None
    cnbforces = cnbforces or {}
    for _ in range(N):
        if has_system:
            system.addParticle(mass)
        if per_chain:
            chain = topology.addChain()
        residue = topology.addResidue(resname or name, chain)
        topology.addAtom(name, element, residue)
        if has_nbforce:
            nbforce.addParticle(charge, sigma, epsilon)
        for cnbforce, param in cnbforces.items():
            cnbforce.addParticle(param)

def add_slab_correction(
        system: openmm.System, topology: app.Topology,
        nbforce: Union[openmm.NonbondedForce, openmm.CustomNonbondedForce],
        temp: Union[float, unit.Quantity], fric: Union[float, unit.Quantity],
        dt: Union[float, unit.Quantity], axis: int = 2, *,
        charge_index: int = 0, z_scale: float = 3, method: str = "force"
    ) -> openmm.Integrator:

    r"""
    Implements a slab correction so that efficient three-dimensional
    Ewald methods can continue to be used to evaluate the electrostatics
    for systems that are periodic in the :math:`x`- and
    :math:`y`-directions but not the :math:`z`-direction. Effectively,
    the system is treated as if it were periodic in the
    :math:`z`-direction, but with empty volume added between the slabs
    and the slab–slab interactions removed.

    For electroneutral systems, the Yeh–Berkowitz correction [1]_ is
    applied:

    .. math::

       \begin{gather*}
         U^\mathrm{corr}=\frac{N_A}{2\varepsilon_0V}M_z^2\\
         u_i^\mathrm{corr}=\frac{N_A}{2\varepsilon_0V}q_i\left(z_iM_z
         -\frac{\sum_i q_iz_i^2}{2}\right)\\
         f_{i,z}^\mathrm{corr}=-\frac{N_A}{\varepsilon_0V}q_iM_z
       \end{gather*}

    For systems with a net electric charge, the Ballenegger–Arnold–Cerdà
    correction [2]_ is applied instead:

    .. math::

       \begin{gather*}
         U^\mathrm{corr}=\frac{N_A}{2\varepsilon_0V}
         \left(M_z^2-q_\mathrm{tot}\sum_i q_iz_i^2
         -\frac{q_\mathrm{tot}^2L_z^2}{12}\right)\\
         u_i^\mathrm{corr}=\frac{N_A}{2\varepsilon_0V}q_i
         \left(z_iM_z-\frac{\sum_i q_iz_i^2+q_\mathrm{tot}z_i^2}{2}
         -\frac{q_\mathrm{tot}L_z^2}{12}\right)\\
         f_{i,z}^\mathrm{corr}=-\frac{N_A}{\varepsilon_0V}q_i
         \left(M_z-q_\mathrm{tot}z_i\right)
       \end{gather*}

    Note that the the relative permittivity
    :math:`\varepsilon_\mathrm{r}` does not appear in the equations
    above because it is accounted for by scaling the particle charges in
    the Coulomb potential.

    Parameters
    ----------
    system : `openmm.System`
        OpenMM molecular system.

    topology : `openmm.app.Topology`
        Topological information about an OpenMM system.

    nbforce : `openmm.NonbondedForce` or `openmm.CustomNonbondedForce`
        Pair potential object containing particle charge information.

        .. note::

           It is assumed that the charge :math:`q` information is the
           first per-particle parameter stored in `nbforce`. If not, the
           index can be specified in `charge_index`.

    temp : `float` or `openmm.unit.Quantity`
        System temperature :math:`T`.

        **Reference unit**: :math:`\mathrm{K}`.

    fric : `float` or `openmm.unit.Quantity`
        Friction coefficient :math:`\gamma` that couples the system to
        the heat bath.

        **Reference unit**: :math:`\mathrm{ps}^{-1}`.

    dt : `float` or `openmm.unit.Quantity`
        Integration step size :math:`\Delta t`.

        **Reference unit**: :math:`\mathrm{ps}`.

    axis : `int`, default: :code:`2`
        Axis along which to apply the slab correction, with :math:`x`
        being :code:`0`, :math:`y` being :code:`1`, and :math:`z` being
        :code:`2`. The source code and outputs, if any, will refer to
        this dimension as :code:`z` regardless of this value.

    charge_index : `int`, keyword-only, default: :code:`0`
        Index of charge :math:`q` information in the per-particle
        parameters stored in `nbforce`.

    z_scale : `float`, keyword-only, default: :code:`3`
        Scaling factor for the dimension specified in `axis`.

    method : `str`, keyword-only, default: :code:`"force"`
        Slab correction methodology.

        .. container::

           **Valid values**:

           * :code:`"force"`: Collective implementation via
             :class:`openmm.CustomExternalForce` and
             :class:`openmm.CustomCVForce` [3]_. This is
             generally the most efficient.
           * :code:`"integrator"`: Per-particle implementation via an
             :class:`openmm.CustomIntegrator`.

    Returns
    -------
    integrator : `openmm.Integrator` or `openmm.CustomIntegrator`
        Integrator that simulates a system using Langevin dynamics, with
        the LFMiddle discretization.

    References
    ----------
    .. [1] Yeh, I.-C.; Berkowitz, M. L. Ewald Summation for Systems with
       Slab Geometry. *J. Chem. Phys.* **1999**, *111* (7), 3155–3162.
       https://doi.org/10.1063/1.479595.

    .. [2] Ballenegger, V.; Arnold, A.; Cerdà, J. J. Simulations of
       Non-Neutral Slab Systems with Long-Range Electrostatic
       Interactions in Two-Dimensional Periodic Boundary Conditions.
       *J. Chem. Phys.* **2009**, *131* (9), 094107.
       https://doi.org/10.1063/1.3216473.

    .. [3] Son, C. Y.; Wang, Z.-G. Image-Charge Effects on Ion
       Adsorption near Aqueous Interfaces. *Proc. Natl. Acad. Sci.
       U.S.A.* **2021**, *118* (19), e2020615118.
       https://doi.org/10.1073/pnas.2020615118.
    """

    # Get system dimensions
    dims = np.array(
        topology.getUnitCellDimensions().value_in_unit(unit.nanometer)
    ) * unit.nanometer
    pbv = system.getDefaultPeriodicBoxVectors()

    # Scale system z-dimension by specified z-scale
    if z_scale < 2:
        wmsg = ("A z-scaling factor that is less than 2 may introduce "
                "unwanted slab–slab interactions. The recommended "
                "value is 3.")
        warnings.warn(wmsg)
    elif z_scale > 5:
        wmsg = ("A z-scaling factor that is greater than 5 may "
                "penalize performance. The recommended value is 3.")
        warnings.warn(wmsg)
    dims[axis] *= z_scale
    pbv[axis] *= z_scale

    # Set new system dimensions
    topology.setUnitCellDimensions(dims)
    system.setDefaultPeriodicBoxVectors(*pbv)

    # Obtain particle charge information
    if isinstance(nbforce.getParticleParameters(0)[charge_index], unit.Quantity):
        qs = np.fromiter(
            (nbforce.getParticleParameters(i)[charge_index]
             .value_in_unit(unit.elementary_charge)
             for i in range(nbforce.getNumParticles())),
            dtype=float
        )
    else:
        qs = np.fromiter(
            (nbforce.getParticleParameters(i)[charge_index]
             for i in range(nbforce.getNumParticles())),
            dtype=float
        )
    neutral = qs.min() == qs.max()
    if not neutral:
        q_tot = qs.sum()
        electroneutral = np.isclose(q_tot, 0)

    # Calculate coefficient for slab correction
    coef = unit.AVOGADRO_CONSTANT_NA / \
           (2 * VACUUM_PERMITTIVITY * dims[0] * dims[1] * dims[2])

    # Get letter representation of axis for formula
    z = chr(120 + axis)

    if neutral:

        # Instantiate an integrator that simulates a system using
        # Langevin dynamics, with the LFMiddle discretization
        integrator = openmm.LangevinMiddleIntegrator(temp, fric, dt)

    else:
        if method == "integrator":

            # Implement an integrator that simulates a system using Langevin
            # dynamics, with the LFMiddle discretization
            integrator = openmm.CustomIntegrator(dt)
            integrator.addGlobalVariable("a", np.exp(-fric * dt))
            integrator.addGlobalVariable("b", np.sqrt(1 - np.exp(-2 * fric * dt)))
            integrator.addGlobalVariable(
                "kT", unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB * temp
            )
            integrator.addPerDofVariable("x1", 0)
            integrator.addUpdateContextState()
            integrator.addComputePerDof("v", "v+dt*f/m")
            integrator.addConstrainVelocities()
            integrator.addComputePerDof("x", "x+dt*v/2")
            integrator.addComputePerDof("v", "a*v+b*sqrt(kT/m)*gaussian")
            integrator.addComputePerDof("x", "x+dt*v/2")
            integrator.addComputePerDof("x1", "x")
            integrator.addConstrainPositions()
            integrator.addComputePerDof("v", "v+(x-x1)/dt")

            # Initialize per-degree-of-freedom variable q for charge
            integrator.addPerDofVariable("q", 0)

            # Add global dipole moment computation to integrator
            integrator.addComputeSum("M_z", "q*x")
            integrator.addComputeSum("M_zz", "q*x^2")

            # Give particle charge information to integrator
            q_vectors = np.zeros((len(qs), 3))
            q_vectors[:, axis] = qs
            integrator.setPerDofVariableByName("q", q_vectors)

            # Implement per-particle slab correction
            if electroneutral:
                slab_corr = openmm.CustomExternalForce(
                    f"coef*q*({z}*M_z-M_zz/2)"
                )
            else:
                slab_corr = openmm.CustomExternalForce(
                    f"coef*q*({z}*M_z-(M_zz+q_tot*{z}^2)/2-q_tot*dim_z^2/12)"
                )
                slab_corr.addGlobalParameter("dim_z", dims[axis])
                slab_corr.addGlobalParameter("q_tot", q_tot)
            slab_corr.addGlobalParameter("M_z", 0)
            slab_corr.addGlobalParameter("M_zz", 0)
            slab_corr.addGlobalParameter("coef", coef)
            slab_corr.addPerParticleParameter("q")

            # Register real particles to the slab correction
            for i, q in enumerate(qs):
                slab_corr.addParticle(i, (q,))

        elif method == "force":

            # Instantiate an integrator that simulates a system using
            # Langevin dynamics, with the LFMiddle discretization
            integrator = openmm.LangevinMiddleIntegrator(temp, fric, dt)

            # Calculate instantaneous system dipole
            M_z = openmm.CustomExternalForce(f"q*{z}")
            M_z.addPerParticleParameter("q")

            # Implement collective slab correction
            if electroneutral:
                slab_corr = openmm.CustomCVForce("coef*M_z^2")
            else:
                M_zz = openmm.CustomExternalForce(f"q*{z}^2")
                M_zz.addPerParticleParameter("q")
                slab_corr = openmm.CustomCVForce(
                    "coef*(M_z^2-q_tot*M_zz-q_tot^2*dim_z^2/12)"
                )
                slab_corr.addCollectiveVariable("M_zz", M_zz)
                slab_corr.addGlobalParameter("dim_z", dims[axis])
                slab_corr.addGlobalParameter("q_tot", q_tot)
            slab_corr.addCollectiveVariable("M_z", M_z)
            slab_corr.addGlobalParameter("coef", coef)

            # Register real particles to the slab correction
            for i, q in enumerate(qs):
                M_z.addParticle(i, (q,))
                if not electroneutral:
                    M_zz.addParticle(i, (q,))

        # Register slab correction to the system
        system.addForce(slab_corr)

    return integrator

def add_image_charges(
        system: openmm.System, topology: app.Topology,
        positions: Union[np.ndarray[float], unit.Quantity],
        temp: Union[float, unit.Quantity], fric: Union[float, unit.Quantity],
        dt: Union[float, unit.Quantity], *, gamma: float = -1,
        n_cells: int = 2, nbforce: openmm.NonbondedForce = None,
        cnbforces: dict[openmm.CustomNonbondedForce, dict[str, Any]] = None,
        wall_indices: np.ndarray[int] = None, exclude: bool = False
    ) -> tuple[unit.Quantity, openmm.Integrator]:

    r"""
    Implements the method of image charges for dielectric boundaries.
    For more information about the method, see Refs. [1]_, [2]_, and
    [3]_ for perfectly conducting boundaries, and Refs. [4]_, [5]_, and
    [6]_ otherwise.

    .. note::

       The boundaries must be in the :math:`xy`-plane and along the
       :math:`z`-axis.

    Parameters
    ----------
    system : `openmm.System`
        OpenMM molecular system.

    topology : `openmm.app.Topology`
        Topological information about an OpenMM system.

    positions : `numpy.ndarray`
        Positions of the :math:`N` particles in the real system.

        **Shape**: :math:`(N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    temp : `float` or `openmm.unit.Quantity`
        System temperature :math:`T`.

        **Reference unit**: :math:`\mathrm{K}`.

    fric : `float` or `openmm.unit.Quantity`
        Friction coefficient :math:`\gamma` that couples the system to
        the heat bath.

        **Reference unit**: :math:`\mathrm{ps}^{-1}`.

    dt : `float` or `openmm.unit.Quantity`
        Integration step size :math:`\Delta t`.

        **Reference unit**: :math:`\mathrm{ps}`.

    gamma : `float`, keyword-only, default: :code:`-1`
        Scaled image charge magnitude :math:`\gamma`.

    n_cells : `int`, keyword-only, default: :code:`2`
        Total number :math:`n_\mathrm{cell}` of cells, with :math:`1`
        real and :math:`n_\mathrm{cell}-1` image cells.

        .. tip::

           Use :math:`n_\mathrm{cell}>2` to instantiate a
           highly-confined simulation system with periodic dimensions
           large enough to satisfy the constraint that the force field
           cutoff be less than half the box size.

        **Valid values**: Even integers greater than :math:`1` when
        :math:`\gamma=-1`.

    nbforce : `openmm.NonbondedForce`, keyword-only, optional
        Standard OpenMM pair potential object implementing the nonbonded
        Lennard-Jones and Coulomb potentials. For the image charges, the
        charges :math:`q` have their signs flipped, and the LJ
        parameters :math:`\sigma` and :math:`\epsilon` are both set to
        :math:`0`.

    cnbforces : `dict`, keyword-only, optional
        Custom pair potential objects implementing other non-standard
        pair potentials. The keys are the potentials and the values are
        the corresponding per-particle parameters to use, which can be
        provided as follows:

        .. container::

           * `None`: All per-particle parameters set to :math:`0`.
           * `dict`: Information on how to transform the real particle
             parameters into those for the image charges. The valid
             (optional) key-value pairs are:

             * :code:`"charge"`: Index (`int`) where the charge
               :math:`q` information is stored.
             * :code:`"zero"`: Index (`int`) or indices (`list`,
               `slice`, or `numpy.ndarray`) of parameters to zero out.
             * :code:`"replace"`: `dict` containing key-value pairs of
               indices (`int`) of parameters to change and their
               replacement value(s). If the value is an `int`, all
               particles receive that value for the current pair
               potential. If the value is another `dict`, the new
               parameter value (value of inner `dict`) is determined by
               the current parameter value (key of inner `dict`).

        To prevent unintended behavior, ensure that each parameter index
        for a custom pair potential is used only once across the keys
        listed above. If one does appear more than once, the order
        in which parameter values are updated follows that of the list
        above.

        See the Examples section below for a simple illustration of the
        possibilities outlined above.

    wall_indices : array-like, keyword-only, optional
        Atom indices corresponding to wall particles. If not provided,
        the wall particles are guessed using the system dimensions.

    exclude : `bool`, keyword-only, default: :code:`False`
        Specifies whether interactions between a wall particle and all
        image wall particles should be excluded. If :code:`False`, only
        the interaction between a wall particle and its own image is
        removed.

        .. tip::

           If you want accurate forces acting on wall particles, set
           :code:`exclude=True`.

    Returns
    -------
    positions : `numpy.ndarray`
        Positions of the :math:`N` particles in the real system and
        their images.

        **Shape**: :math:`(2N,\,3)`.

        **Reference unit**: :math:`\mathrm{nm}`.

    integrator : `openmm.Integrator`
        Integrator that simulates a system using Langevin dynamics, with
        the LFMiddle discretization.

    Examples
    --------
    Prepare the `cnbforces` `dict` argument for a system with the
    following custom pair potentials:

    * :code:`pair_gauss`: Gaussian potential with per-particle
      parameters `type` and `beta`.

      * `type` is the particle type, and is used to look up a
        tabulated function for parameter values. As such, the new
        values will vary and depend on the old ones. For example, for
        a system with :math:`2` types of particles, :math:`0` becomes
        :math:`2` and :math:`1` becomes :math:`3`.
      * `beta` is assumed to be constant, with the value for the real
        particles differing from that for image charges. For this
        example, the new value is :math:`\beta = 42` (arbitrary
        units).

    * :code:`pair_wca`: Weeks–Chander–Andersen potential with per-
      particle parameters `sigma` and `epsilon`, both of which
      should be set to :math:`0` to turn off this potential for
      image charges.
    * :code:`pair_coul_recip`: Smeared Coulomb potential with per-
      particle parameter `charge`, which should be flipped for image
      charges, and `dummy`, a value that should be set to :math:`0`.

    .. code::

       NEW_BETA_VALUE = 42
       cnbforces = {
           pair_gauss: {"replace": {0: {0: 2, 1: 3}, 1: NEW_BETA_VALUE}},
           pair_wca: None,
           pair_coul_recip: {"charge": 0, "zero": 1}
       }

    References
    ----------
    .. [1] Hautman, J.; Halley, J. W.; Rhee, Y. ‐J. Molecular Dynamics
       Simulation of Water Beween Two Ideal Classical Metal Walls.
       *J. Chem. Phys.* **1989**, *91* (1), 467–472.
       https://doi.org/10.1063/1.457481.

    .. [2] Dwelle, K. A.; Willard, A. P. Constant Potential,
       Electrochemically Active Boundary Conditions for Electrochemical
       Simulation. *J. Phys. Chem. C* **2019**, *123* (39), 24095–24103.
       https://doi.org/10.1021/acs.jpcc.9b06635.

    .. [3] Son, C. Y.; Wang, Z.-G. Image-Charge Effects on Ion
       Adsorption near Aqueous Interfaces. *Proc. Natl. Acad. Sci.
       U.S.A.* **2021**, *118* (19), e2020615118.
       https://doi.org/10.1073/pnas.2020615118.

    .. [4] Dos Santos, A. P.; Girotto, M.; Levin, Y. Simulations of
       Coulomb Systems with Slab Geometry Using an Efficient 3D Ewald
       Summation Method. *The Journal of Chemical Physics* **2016**,
       **144** (14), 144103. https://doi.org/10.1063/1.4945560.

    .. [5] Dos Santos, A. P.; Girotto, M.; Levin, Y. Simulations of
       Coulomb Systems Confined by Polarizable Surfaces Using Periodic
       Green Functions. *The Journal of Chemical Physics* **2017**,
       **147** (18), 184105. https://doi.org/10.1063/1.4997420.

    .. [6] Son, C. Y.; Wang, Z.-G. Manuscript in preparation.
    """

    if not FOUND_ICPLUGIN:
        emsg = ("An integrator capable of simulating a system with "
                "image charges was not found. As such, the method of "
                "image charges is unavailable unless the openmm-ic "
                "(https://github.com/bbye98/mdcraft/tree/main/lib/openmm-ic-plugin) "
                "or the constvplugin (https://github.com/scychon/openmm_constV) "
                "package is installed.")
        raise ImportError(emsg)

    if np.isclose(gamma, 0):
        emsg = ("Use the slab correction, available via "
                "mdcraft.openmm.system.slab_correction(), for gamma=0.")
        raise ValueError(emsg)
    if not np.isclose(gamma, -1) and n_cells != 2:
        emsg = ("The method of image charges with gamma != -1 is only "
                "implemented for n_cells=2.")
        raise ValueError(emsg)

    def _ic_beta(gamma: float, x: float) -> float:

        r"""
        Computes the :math:`\beta` value used in the higher-order term
        correction for the method of image charges with
        :math:`\gamma\neq\pm1`.

        Parameters
        ----------
        gamma : `float`
            Scaled image charge magnitude :math:`\gamma`.

        x : `float`
            Scaled :math:`x`-coordinate of the ion.

        Returns
        -------
        beta : `float`
            Approximated :math:`\beta` value.
        """

        if not 0 <= x <= 1:
            raise ValueError("'x' must be between 0 and 1.")
        if np.isclose(x, 0.5):
            return float(2 * special.zeta(3, 1.5)
                         - 2 * gamma ** 4 * mpmath.lerchphi(gamma ** 2, 3, 1.5))
        else:
            return (
                special.zeta(2, 2 - x) - special.zeta(2, 1 + x) - gamma ** 4 *
                float(mpmath.lerchphi(gamma ** 2, 2, 2 - x)
                      - mpmath.lerchphi(gamma ** 2, 2, 1 + x))
            ) / (2 * x - 1)

    # Get system information
    dims = np.asarray(
        topology.getUnitCellDimensions().value_in_unit(unit.nanometer)
    ) * unit.nanometer
    pbv = system.getDefaultPeriodicBoxVectors()
    N_real_atoms = positions.shape[0]
    if isinstance(positions, unit.Quantity):
        positions = positions.value_in_unit(unit.nanometer)

    # Guess indices of left and right walls if not provided
    if wall_indices is None:
        wall_indices = np.concatenate(
            (np.isclose(positions[:, 2], 0).nonzero()[0],
             np.isclose(positions[:, 2],
                        dims[2].value_in_unit(unit.nanometer)).nonzero()[0])
        )

    # Find averaged beta value for image charges with gamma =/= +/-1
    beta = (_ic_beta(gamma, 0) + _ic_beta(gamma, 0.5)) / 2

    # Set up higher-order image charge and slab corrections
    cv_E_corr = openmm.CustomExternalForce("q*(1-2*z/L)")
    cv_E_corr.addGlobalParameter("L", dims[2]) # real system z-dimension such
                                               # that 0 <= (x = z / L_z) <= 1
    cv_E_corr.addPerParticleParameter("q")
    cv_M_z = openmm.CustomExternalForce("q*z")
    cv_M_z.addPerParticleParameter("q")
    cv_M_zz = openmm.CustomExternalForce("q*z^2")
    cv_M_zz.addPerParticleParameter("q")

    # Obtain particle charge information and register charges to
    # corrections
    if nbforce is None:
        charge_index = None
        for force, params in cnbforces:
            if "charge" in params:
                charge_index = params["charge"]
                break
        if charge_index is None:
            raise ValueError("No charge information provided.")
    else:
        force = nbforce
        charge_index = 0

    q_tot = 0
    if isinstance(force.getParticleParameters(0)[charge_index], unit.Quantity):
        for i in range(force.getNumParticles()):
            q = (force.getParticleParameters(i)[charge_index]
                 .value_in_unit(unit.elementary_charge))
            q_tot += q
            if not np.isclose(q, 0):
                cv_E_corr.addParticle(i, (q,))
                cv_M_z.addParticle(i, (q,))
                cv_M_zz.addParticle(i, (q,))
    else:
        for i in range(force.getNumParticles()):
            q = force.getParticleParameters(i)[charge_index]
            q_tot += q
            if not np.isclose(q, 0):
                cv_E_corr.addParticle(i, (q,))
                cv_M_z.addParticle(i, (q,))
                cv_M_zz.addParticle(i, (q,))
    electroneutral = np.isclose(q_tot, 0)

    # Update and set new system dimensions
    dims[2] *= n_cells
    topology.setUnitCellDimensions(dims)
    pbv[2] *= n_cells
    system.setDefaultPeriodicBoxVectors(*pbv)
    logging.info(f"Increased z-dimension to {dims[2]}.")

    # Determine correction energy expression
    corr_energy = ""
    corr = openmm.CustomCVForce(corr_energy)
    if not np.isclose(beta, 0):
        corr_energy += "coef1*E_corr*M_z"
        corr.addCollectiveVariable("E_corr", cv_E_corr)
        corr.addGlobalParameter(
            "coef1",
            (unit.AVOGADRO_CONSTANT_NA * gamma * beta
             / (4 * np.pi * VACUUM_PERMITTIVITY * dims[2] ** 2))
            .in_units_of(unit.kilojoule_per_mole /
                         (unit.elementary_charge ** 2 * unit.nanometer))
        )
    if not np.isclose(gamma, -1):
        corr_energy += "+coef2*M_z^2"
    if not electroneutral:
        if np.isclose(gamma, 1):
            corr_energy += "-coef2*q_tot*M_z*L_z"
        elif np.isclose(gamma, -1):
            corr_energy += "+coef2*q_tot*(M_z*L_z-M_zz)"
        else:
            corr_energy += "-coef2*q_tot*M_zz"
        corr.addGlobalParameter("q_tot", q_tot)
    if "coef2" in corr_energy:
        corr.addGlobalParameter(
            "coef2",
            (unit.AVOGADRO_CONSTANT_NA
             / (2 * VACUUM_PERMITTIVITY * dims[0] * dims[1] * dims[2]))
            .in_units_of(unit.kilojoule_per_mole
                         / (unit.elementary_charge * unit.nanometer) ** 2)
        )
    if "L_z" in corr_energy:
        corr.addGlobalParameter("L_z", dims[2]) # periodic system z-dimension
    if "M_z" in corr_energy:
        corr.addCollectiveVariable("M_z", cv_M_z)
    if "M_zz" in corr_energy:
        corr.addCollectiveVariable("M_zz", cv_M_zz)
    if corr_energy:
        if corr_energy.startswith("+"):
            corr_energy = corr_energy.lstrip("+")
        corr.setEnergyFunction(corr_energy)
        system.addForce(corr)
        logging.info("Added higher-order image charge and/or slab "
                     "correction(s).")

    # Mirror particle positions
    if n_cells == 2:
        positions = np.concatenate(
            (positions, positions * np.array((1, 1, -1), dtype=int))
        ) * unit.nanometer
    else:
        positions = np.tile(positions, (n_cells, 1))
        for cell_index in range(1, n_cells):
            start = cell_index * N_real_atoms
            stop = (cell_index + 1) * N_real_atoms
            positions[start:stop, 2] = (
                (1 - 2 * (cell_index % 2)) * positions[start:stop, 2]
                - 2 * np.floor(cell_index / 2)
                * dims[2].value_in_unit(unit.nanometer)
            )
        positions *= unit.nanometer
    logging.info(f"Replicated {N_real_atoms:,} particles {n_cells - 1} "
                 "time(s) over the z-axis.")

    # Instantiate an integrator that simulates a system using
    # Langevin dynamics and updates the image charge positions
    integrator = ICLangevinIntegrator(temp, fric, dt, n_cells)

    # Register image charges to the system, topology, and force field
    cnbforces = cnbforces or {}
    N_real_chains = topology.getNumChains()
    atoms = list(topology.atoms())
    residues = list(topology.residues())
    coefs = (1, gamma)
    for c in range(1, n_cells):
        coef = coefs[c % 2]
        chains_ic = [topology.addChain() for _ in range(N_real_chains)]
        residues_ic = [topology.addResidue(f"IC_{r.name}",
                                           chains_ic[r.chain.index])
                       for r in residues]
        for i, atom in enumerate(atoms):
            system.addParticle(0)
            topology.addAtom(f"IC_{atom.name}", atom.element,
                             residues_ic[atom.residue.index])
            if nbforce is not None:
                nbforce.addParticle(
                    0 if i in wall_indices
                    else coef * nbforce.getParticleParameters(i)[0],
                    0, 0
                )
            for force, kwargs in cnbforces.items():
                params = np.array(force.getParticleParameters(i))
                if kwargs is None:
                    params[:] = 0
                else:
                    if "charge" in kwargs:
                        params[kwargs["charge"]] *= (0 if i in wall_indices
                                                     else coef)
                    if "zero" in kwargs:
                        params[kwargs["zero"]] = 0
                    if "replace" in kwargs:
                        for index, value in kwargs["replace"].items():
                            params[index] = (value[params[index]]
                                             if isinstance(value, dict)
                                             else value)
                force.addParticle(params)
    logging.info(f"Registered {system.getNumParticles() - N_real_atoms:,} "
                 "image particles to the force field.")

    # Add existing particle exclusions to mirrored image charges
    for i in range(nbforce.getNumExceptions()):
        i1, i2, qq = nbforce.getExceptionParameters(i)[:3]
        if i1 not in wall_indices and i2 not in wall_indices:
            for c in range(1, n_cells):
                nbforce.addException(c * N_real_atoms + i1,
                                     c * N_real_atoms + i2, qq, 0, 0)
                for force in cnbforces:
                    i1, i2 = force.getExclusionParticles(i)
                    force.addExclusion(c * N_real_atoms + i1,
                                       c * N_real_atoms + i2)
    logging.info("Mirrored excluded non-wall image particle–image "
                 "particle interactions.")

    # Prevent wall particles from interacting with all image wall
    # particles
    if exclude:
        for i in wall_indices:
            for j in wall_indices:
                for c in range(1, n_cells):
                    nbforce.addException(i, c * N_real_atoms + j, 0, 0, 0)
                    for force in cnbforces:
                        force.addExclusion(i, c * N_real_atoms + j)

    # Prevent wall particles from interacting their images
    else:
        for i in wall_indices:
            for c in range(1, n_cells):
                nbforce.addException(i, c * N_real_atoms + i, 0, 0, 0)
                for force in cnbforces:
                    force.addExclusion(i, c * N_real_atoms + i)
    logging.info("Removed wall–image wall interactions.")

    return positions, integrator

def add_electric_field(
        system: openmm.System, nbforce: openmm.NonbondedForce,
        E: Union[float, unit.Quantity], *, axis: int = 2,
        dielectric: float = 1, charge_index: int = 0,
        atom_indices: Union[int, np.ndarray[int]] = None
    ) -> None:

    r"""
    Adds an electric field to all charged particles by adding a force
    :math:`f_i=q_iE` in the axis specified in `axis`, where :math:`q_i`
    is the per-particle charge and :math:`E` is the electric field.

    .. hint::

       The following schematic shows how directionality is handled:

       .. code::

          |-| (-) ---> |+|
          |-| <-- E -- |+|
          |-| <--- (+) |+|

       With a positive potential difference
       (:math:`\Delta V>0\;\mathrm{V}`), the electric field is negative
       (:math:`E<0\;\mathrm{V/m}`) such that it is pointing from the
       right (positive) plate to the left (negative) plate. If an ion
       has a positive charge (:math:`q_i>0\;\mathrm{e}`), the force will
       be negative, indicating that it will be drawn to the left plate,
       and vice versa.

    Parameters
    ----------
    system : `openmm.System`
        OpenMM molecular system.

    nbforce : `openmm.NonbondedForce` or `openmm.CustomNonbondedForce`
        Pair potential object containing particle charge information.

        .. note::

           It is assumed that the charge :math:`q` information is the
           first per-particle parameter stored in `nbforce`. If not, the
           index can be specified in `charge_index`.

    E : `float` or `openmm.unit.Quantity`
        Electric field :math:`E`.

        **Reference unit**: :math:`\mathrm{kJ/(mol\cdot nm\cdot e)}`.

    axis : `int`, keyword-only, default: :code:`2`
        Axis along which the walls are placed. :code:`0`, :code:`1`, and
        :code:`2` correspond to :math:`x`, :math:`y`, and :math:`z`,
        respectively.

    dielectric : `float`, keyword-only, default: :code:`1`
        Relative permittivity :math:`\varepsilon_\mathrm{r}` of the
        medium. Used to scale the particle charges by
        :math:`\sqrt{\varepsilon_\mathrm{r}}` and recover the original
        values.

    charge_index : `int`, keyword-only, default: :code:`0`
        Index of charge :math:`q` information in the per-particle
        parameters stored in `nbforce`.

    atom_indices : `int` or array-like, keyword-only, optional
        Indices of atoms to apply the electric field to. By default,
        the electric field is applied to all atoms, but this can be
        computationally expensive when there are charged particles that
        do not need to be included, such as image charges. If an `int`
        is provided, all atoms up to that index are included.
    """

    # Get letter representation of axis for formula
    z = chr(120 + axis)

    # Get indices of atoms that are affected by the electric field
    if atom_indices is None:
        atom_indices = range(nbforce.getNumParticles())
    elif isinstance(atom_indices, int):
        atom_indices = range(atom_indices)

    # Create and register particles to the electric field
    efield = openmm.CustomExternalForce(f"-q*E*{z}")
    efield.addGlobalParameter("E", E)
    efield.addPerParticleParameter("q")

    for i in atom_indices:
        q = nbforce.getParticleParameters(i)[charge_index]
        if isinstance(q, unit.Quantity):
            q = q.value_in_unit(unit.elementary_charge)
        if not np.isclose(q, 0):
            efield.addParticle(i, (q * np.sqrt(dielectric),))

    system.addForce(efield)

def estimate_pressure_tensor(
        context: openmm.Context, dh: float = 1e-5, *, diag: bool = False
    ) -> np.ndarray[float]:

    r"""
    Computes the estimated pressure tensor using a central finite
    difference for the virial contribution.

    The pressure tensor is given by

    .. math::

       \mathbf{p}=\frac{1}{V}\left(
       \sum_{i=1}^N m_i\mathbf{v}_i^\mathsf{T}\mathbf{v}_i
       +\mathbf{h}^\mathsf{T}\frac{dU}{d\mathbf{h}}\right)

    where :math:`V` is the volume, :math:`m_i` and :math:`\mathbf{v}_i`
    are the mass and velocity vector of particle :math:`i`,
    :math:`\mathbf{h}` is a :math:`3\times 3` matrix where the rows
    contain the box vectors, and :math:`U` is the total pairwise
    potential energy.

    To evaluate :math:`dU/d\mathbf{h}`, the box vectors are perturbed by
    `dh` in each of the six "directions" and the positions are updated
    accordingly. Then, OpenMM reevaluates the potential
    energy based on the new periodic box vectors and particle positions,
    and the difference in potential energy is used in the central finite
    difference formula to estimate the derivative.

    Parameters
    ----------
    context : `openmm.Context`
        OpenMM simulation context.

    dh : `float`, default: :code:`1e-5`
        Finite difference step size.

    diag : `bool`, keyword-only, default: :code:`False`
        Determines whether only the values in the main diagonal of the
        pressure tensor (i.e., the values that, when summed, gives the
        system pressure) are calculated.

    Returns
    -------
    pres : `numpy.ndarray`
        Estimated pressure tensor (or diagonal components).

        **Shape**: :math:`(3,)` or :math:`(3,\,3)`.

        **Reference unit**: :math:`\mathrm{atm}`.
    """

    try:
        state = context.getState(getPositions=True, getVelocities=True,
                                 getEnergy=True)
        box = state.getPeriodicBoxVectors(asNumpy=True)
        positions = state.getPositions(asNumpy=True)
        velocities = state.getVelocities(asNumpy=True)
        volume = box[0, 0] * box[1, 1] * box[2, 2]
    except openmm.OpenMMException:
        emsg = ("The simulation context must have information about "
                "the particle positions and velocities.")
        raise ValueError(emsg)

    system = context.getSystem()
    masses = np.fromiter(
        (system.getParticleMass(i).value_in_unit(unit.dalton)
         for i in range(system.getNumParticles())),
        dtype=float
    ) * unit.dalton

    if diag:

        # Compute the ideal contribution
        p_kinetic = (masses * velocities ** 2).sum(axis=0)

        # Estimate the virial contribution with a central finite difference
        p_virial = np.zeros(3) * unit.kilojoule_per_mole
        for i in range(3):
            box_ = box.copy()
            box_[i, i] += dh
            context.setPeriodicBoxVectors(*box_)
            context.setPositions(
                np.dot(positions,
                       np.divide(box_, box, out=np.zeros_like(box),
                                 where=box.value_in_unit(unit.nanometer) != 0))
            )
            U_plus = context.getState(getEnergy=True).getPotentialEnergy()
            box_ = box.copy()
            box_[i, i] -= dh
            context.setPeriodicBoxVectors(*box_)
            context.setPositions(
                np.dot(positions,
                       np.divide(box_, box, out=np.zeros_like(box),
                                 where=box.value_in_unit(unit.nanometer) != 0))
            )
            U_minus = context.getState(getEnergy=True).getPotentialEnergy()
            p_virial[i] = U_plus - U_minus
        p_virial = (p_virial / (2 * dh)).in_units_of(p_kinetic.unit)

    else:

        # Compute the ideal contribution
        p_kinetic = (masses * velocities * velocities[:, :, None]).sum(axis=0)

        # Estimate the virial contribution with a central finite difference
        p_virial = np.zeros((3, 3)) * unit.kilojoule_per_mole
        for i in range(3):
            for j in range(i + 1):
                box_ = box.copy()
                box_[i, j] += dh
                context.setPeriodicBoxVectors(*box_)
                context.setPositions(
                    np.dot(positions,
                           np.divide(box_, box, out=np.zeros_like(box),
                                     where=box.value_in_unit(unit.nanometer) != 0))
                )
                U_plus = context.getState(getEnergy=True).getPotentialEnergy()
                box_ = box.copy()
                box_[i, j] -= dh
                context.setPeriodicBoxVectors(*box_)
                context.setPositions(
                    np.dot(positions,
                           np.divide(box_, box, out=np.zeros_like(box),
                                     where=box.value_in_unit(unit.nanometer) != 0))
                )
                U_minus = context.getState(getEnergy=True).getPotentialEnergy()
                p_virial[i, j] = U_plus - U_minus
        p_virial = (p_virial / (2 * dh)).in_units_of(p_kinetic.unit)
        p_virial = (
            p_virial._value + np.tril(p_virial).T
            - np.diag(np.diag(p_virial))
        ) * p_virial.unit

    return (
        (p_kinetic + p_virial) / (unit.AVOGADRO_CONSTANT_NA * volume)
    ).in_units_of(unit.atmosphere)