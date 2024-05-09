"""
Custom OpenMM bond potentials
=============================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains implementations of commonly used bond potentials
that are not available in OpenMM, such as the finite extension nonlinear
elastic (FENE) potential. Generally, the bond potentials are named after
their LAMMPS :code:`bond_style` counterparts, if available.
"""

from typing import Union

import openmm
from openmm import unit

from .pair import wca as pwca

def _setup_bond(
        cbforce: openmm.CustomBondForce,
        global_params: dict[str, Union[float, unit.Quantity]],
        per_params: list[str]) -> None:

    """
    Sets up a :class:`openmm.CustomBondForce` object.

    Parameters
    ----------
    cbforce : `openmm.CustomBondForce`
        Custom bond force object.

    global_params : `dict`
        Global parameters.

    per_params : `list`
        Per-particle parameters.
    """

    for param in global_params.items():
        cbforce.addGlobalParameter(*param)
    for param in per_params:
        cbforce.addPerBondParameter(param)

def fene(
        global_args: dict[str, Union[float, unit.Quantity]] = None,
        wca: bool = True, **kwargs
    ) -> tuple[openmm.CustomBondForce, openmm.CustomNonbondedForce]:

    r"""
    Implements the finite extensible nonlinear elastic (FENE) potential
    used for bead-spring polymer models.

    The potential energy between two bonded particles is given by

    .. math::

       u_\textrm{FENE}=-\frac{1}{2}k_{12}r_{0,12}^2
       \ln{\left[1-\left(\frac{r_{12}}{r_{0,12}}\right)^2\right]}
       +4\epsilon_{12}\left[\left(\frac{\sigma_{12}}{r_{12}}\right)^{12}
       -\left(\frac{\sigma_{12}}{r_{12}}\right)^6\right]+\epsilon_{12}

    where :math:`k_{12}` is the bond coefficient in
    :math:`\textrm{kJ}/(\textrm{nm}^2\cdot\textrm{mol})`,
    :math:`r_{0,12}` is the equilibrium bond length in
    :math:`\textrm{nm}`, :math:`\sigma_{12}` is the average particle
    size in :math:`\textrm{nm}`, and :math:`\epsilon_{12}` is the
    dispersion energy in :math:`\textrm{kJ/mol}`. :math:`k_{12}`,
    :math:`r_{0,12}`, :math:`\sigma_{12}` and :math:`\epsilon_{12}` are
    determined from per-bond and per-particle parameters `k`, `r0`,
    `sigma` and `epsilon`, respectively, which are set using
    :meth:`openmm.openmm.CustomBondForce.addBond` and
    :meth:`openmm.openmm.NonbondedForce.addParticle`.

    Parameters
    ----------
    global_args : `dict`, optional
        Constant values :math:`k_{12}` and :math:`r_{0,12}` to use
        instead of per-bond parameters. The corresponding per-bond
        parameters will not be registered, but the remaining
        per-bond parameters will still have to be provided in their
        default order.

    wca : `bool`, default: :code:`True`
        Determines whether the Weeks–Chandler–Andersen (WCA) potential
        is included.

    **kwargs
        Keyword arguments to be passed to
        :meth:`mdcraft.openmm.pair.wca` if :code:`wca=True`.

    Returns
    -------
    bond_fene : `openmm.CustomBondForce`
        FENE bond potential.

    pair_wca : `openmm.CustomNonbondedForce`
        WCA pair potential, if :code:`wca=True`.
    """

    bond_fene = openmm.CustomBondForce("-0.5*k*r0^2*log(1-(r/r0)^2)")
    per_args = ["k", "r0"]
    for param in global_args.keys():
        if param in per_args:
            per_args.remove(param)
    _setup_bond(bond_fene, global_args, per_args)

    if wca:
        pair_wca = pwca(**kwargs)
        return bond_fene, pair_wca

    return bond_fene