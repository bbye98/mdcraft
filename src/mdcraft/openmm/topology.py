"""
OpenMM topology transformations
===============================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains implementations of common OpenMM topology
transformations, like the generation of initial particle positions and
subsetting existing topology objects.
"""

from itertools import repeat
from typing import Any, Iterable, Union

import numpy as np
from openmm import app

from ..algorithm import topology as t

def create_atoms(*args, **kwargs) -> Any:

    """
    Generates initial particle positions.

    This is an alias function. For more information, see
    :func:`mdcraft.algorithm.topology.create_atoms`.
    """

    return t.create_atoms(*args, **kwargs)

def _get_hierarchy_indices(
        item: Union[app.Atom, app.topology.Bond, app.Residue, app.Chain],
        bonds: dict[str, list]
    ) -> tuple[set[int], set[int], set[int], set[int]]:

    """
    Get unique indices of all topology items related to the one passed
    to this function.

    For an atom, the indices of itself, the residue it belongs to, and
    the chain that its residue belongs to are returned.

    For a bond, the indices of the two atoms it connects, itself, the
    residue(s) that the two atoms belong to, and the chain(s) that the
    residue(s) belong to are returned.

    For a residue, the indices of all underlying atoms and bonds,
    itself, and the chain it belongs to are returned.

    For a chain, the indices of all underlying atoms, bonds, and
    residues, and itself are returned.

    Parameters
    ----------
    item : `openmm.app.Atom`, `openmm.app.topology.Bond`,
    `openmm.app.Residue`, or `openmm.app.Chain`
        Topology item of interest.

    Returns
    -------
    atoms : `set`
        Indices of all atoms associated with `item`.

    bonds : `set`
        Indices of all bonds associated with `item`.

    residues : `set`
        Indices of all residues associated with `item`.

    chains : `set`
        Indices of all chains associated with `item`.
    """

    if isinstance(item, app.Atom):
        return {item.index}, set(), {item.residue.index}, \
               {item.residue.chain.index}

    elif isinstance(item, app.topology.Bond):
        return {item.atom1.index, item.atom2.index}, {bonds.index(item)}, \
               {item.atom1.residue.index, item.atom2.residue.index}, \
               {item.atom1.residue.chain.index, item.atom2.residue.chain.index}

    elif isinstance(item, app.Residue):
        return {a.index for a in item.atoms()}, \
               {bonds.index(b) for b in item.bonds()}, {item.index}, \
               {item.chain.index}

    elif isinstance(item, app.Chain):
        atom_indices = set()
        bond_indices = set()
        residue_indices = set()
        for residue in item.residues():
            a, b, r, _ = _get_hierarchy_indices(residue, bonds)
            atom_indices |= a
            bond_indices |= b
            residue_indices |= r
        return atom_indices, bond_indices, residue_indices, {item.index}

def _is_topology_object(obj: Any):

    """
    Check if the argument is a topology item.

    Parameters
    ----------
    obj : `Any`
        Any object.

    Returns
    -------
    is_topology_object : `bool`
        Boolean value indicating whether `obj` is a topology item.
    """
    return isinstance(obj, (app.Atom, app.topology.Bond, app.Residue, app.Chain))

def get_subset(
        topology: app.Topology, positions: np.ndarray[float], *,
        delete: list[Any] = None, keep: list[Any] = None,
        types: Union[str, Iterable[str]] = None
    ) -> Union[app.Topology, np.ndarray[float]]:

    r"""
    Creates a topology subset and gets its corresponding particle positions.

    Parameters
    ----------
    topology : `openmm.app.Topology`
        OpenMM topology.

    positions : `numpy.ndarray`
        Positions of the :math:`N` particles in the topology.

        **Shape**: :math:`(N,\,3)`.

    delete : array-like, keyword-only, optional
        `openmm.app.Atom`, `openmm.app.Bond`, `openmm.app.Residue`,
        and/or `openmm.app.Chain` objects, or the indices of those
        objects. If indices are provided, their corresponding object
        types (:code:`"atom"`, :code:`"residue"`, :code:`"chain"`) must
        be provided in `types`. The specified items will be deleted from
        the model.

        .. note::

           Only one of `delete` and `keep` can be specified.

    keep : array-like, keyword-only, optional
        `openmm.app.Atom`, `openmm.app.Bond`, `openmm.app.Residue`,
        and/or `openmm.app.Chain` objects, or the indices of those
        objects. If indices are provided, their corresponding object
        types (:code:`"atom"`, :code:`"residue"`, :code:`"chain"`) must
        be provided in `types`. The specified items will be kept in the
        model.

        .. note::

           Only one of `delete` and `keep` can be specified.

    types : `str` or array-like, keyword-only, optional
        Object types corresponding to the indices provided in `delete`
        or `keep`. If a `str` is provided, all items in the array are
        assumed to have the same object type. Must be provided if not
        all items in `delete` or `keep` are OpenMM topology objects.

    Returns
    -------
    topology : `openmm.app.Topology`
        OpenMM topology subset.

    positions : `numpy.ndarray`
        Positions of the remaining :math:`N_\mathrm{rem}` particles in
        the topology.

        **Shape**: :math:`(N_\mathrm{rem},\,3)`.
    """

    # Set boolean flags for subroutines below
    found = (delete is not None, keep is not None)

    # Check if both delete and keep arguments were provided
    if all(found):
        emsg = ("Only specify topology items to either delete or keep. "
                "When both types are specified, the atoms, bonds, "
                "residues, and/or chains to be removed from the topology "
                "become ambiguous.")
        raise ValueError(emsg)

    # Return original topology and positions if no items are specified
    elif not any(found):
        return topology, positions

    # Ensure object type(s) are provided if not all items are topology
    # objects
    elif types is None \
            and not all(_is_topology_object(i) for i in
                        next(a for a in [delete, keep] if a is not None)):
        emsg = ("Object types must be specified for the topology items "
                f"to be {'kept' if found[0] else 'deleted'}.")
        raise ValueError(emsg)

    # Create a generator of types if a string is provided
    elif isinstance(types, str):
        same = True
        types = repeat(types)
    elif types is not None:
        same = all(t == "atoms" for t in types)

    # Create OpenMM Modeller
    modeller = app.Modeller(topology, positions)

    if types is not None:

        # Create dictionary with topology subitems
        model = {
            "atom": list(topology.atoms()),
            "bond": list(topology.bonds()),
            "chain": list(topology.chains()),
            "residue": list(topology.residues())
        }

        # If indices and types of objects to be deleted are specified,
        # create an iterable object of corresponding items from the
        # dictionary
        if found[0]:
            delete = (i if _is_topology_object(i) else model[t][i]
                    for i, t in zip(delete, types))
        else:

            # Preallocate sets to store indices of atoms, residues, and
            # chains to keep
            atoms = set()
            bonds = set()
            residues = set()
            chains = set()

            # Remove items to be kept from the master list of topology
            # subitems to delete
            for item, item_type in zip(keep, types):
                if not _is_topology_object(item):
                    item = model[item_type][item]
                a, b, r, c = _get_hierarchy_indices(item, model["bond"])
                atoms |= a
                bonds |= b
                residues |= r
                chains |= c

            model["atom"] = np.delete(model["atom"], list(atoms))
            model["residue"] = np.delete(model["residue"], list(residues))
            model["chain"] = np.delete(model["chain"], list(chains))
            if not bonds and same:
                model["bond"] = []
            else:
                for i in sorted(bonds, reverse=True):
                    del model["bond"][i]
            delete = [i for t in model.values() for i in t]

    # Create subset by deleting objects from original topology
    modeller.delete(delete)

    return modeller.topology, modeller.positions