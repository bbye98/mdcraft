import pathlib
import sys

import numpy as np
import openmm
from openmm import app
import pytest

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft.openmm import system as s, topology as t # noqa: E402

def test_func_subset_errors():

    topology = app.Topology()
    topology.addAtom("", "", topology.addResidue("", topology.addChain()))
    positions = np.array((0, 0, 0))

    # TEST CASE 1: Both items to delete and keep are specified
    with pytest.raises(ValueError):
        t.get_subset(topology, positions, delete=[0], keep=[0])

    # TEST CASE 2: No item type specified
    with pytest.raises(ValueError):
        t.get_subset(topology, positions, delete=[0])

def test_func_subset_polymer():

    M = 100
    N_p = 25
    N = M * N_p
    dims = np.array((10, 10, 10))
    positions = t.create_atoms(dims, M * N_p, N_p)
    system = openmm.System()
    topology = app.Topology()
    for _ in range(M):
        chain = topology.addChain()
        s.register_particles(system, topology, N=N_p, chain=chain)
    atoms = list(topology.atoms())
    for m in range(M):
        for n in range(N_p - 1):
            i = m * N_p + n
            topology.addBond(atoms[i], atoms[i + 1])

    # TEST CASE 1: Nothing to do
    topo_sub, pos_sub = t.get_subset(topology, positions)
    assert topo_sub is topology
    assert pos_sub is positions

    n_atoms = 100
    n_chains = n_atoms // N_p
    n_bonds = n_atoms - n_chains

    # TEST CASE 2: Delete everything but the first 100 atoms, with no type specified
    topo_sub, pos_sub = t.get_subset(topology, positions, delete=atoms[n_atoms:])
    assert topo_sub.getNumAtoms() == n_atoms
    assert topo_sub.getNumBonds() == n_bonds
    assert topo_sub.getNumResidues() == n_atoms
    assert topo_sub.getNumChains() == n_chains

    # TEST CASE 3: Delete everything but the first 100 atoms, with types specified
    topo_sub, pos_sub = t.get_subset(topology, positions, delete=np.arange(100, N),
                                 types="atom")
    assert topo_sub.getNumAtoms() == n_atoms
    assert topo_sub.getNumBonds() == n_bonds
    assert topo_sub.getNumResidues() == n_atoms
    assert topo_sub.getNumChains() == n_chains

    # TEST CASE 4: Keep first 100 atoms
    topo_sub, pos_sub = t.get_subset(topology, positions, keep=np.arange(n_atoms),
                                 types="atom")
    assert topo_sub.getNumAtoms() == n_atoms
    assert topo_sub.getNumBonds() == n_bonds
    assert topo_sub.getNumResidues() == n_atoms
    assert topo_sub.getNumChains() == n_chains

    # TEST CASE 5: Keep first 96 bonds
    topo_sub, pos_sub = t.get_subset(topology, positions, keep=np.arange(n_bonds),
                                 types=n_bonds * ["bond"])
    assert topo_sub.getNumAtoms() == n_atoms
    assert topo_sub.getNumBonds() == n_bonds
    assert topo_sub.getNumResidues() == n_atoms
    assert topo_sub.getNumChains() == n_chains

    # TEST CASE 6: Keep first 100 residues
    topo_sub, pos_sub = t.get_subset(topology, positions, keep=np.arange(n_atoms),
                                 types="residue")
    assert topo_sub.getNumAtoms() == n_atoms
    assert topo_sub.getNumBonds() == n_bonds
    assert topo_sub.getNumResidues() == n_atoms
    assert topo_sub.getNumChains() == n_chains

    # TEST CASE 7: Keep first 4 chains
    topo_sub, pos_sub = t.get_subset(topology, positions, keep=np.arange(n_chains),
                                 types="chain")
    assert topo_sub.getNumAtoms() == n_atoms
    assert topo_sub.getNumBonds() == n_bonds
    assert topo_sub.getNumResidues() == n_atoms
    assert topo_sub.getNumChains() == n_chains