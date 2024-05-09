import pathlib
import sys

import numpy as np
from openmm import app, unit
import pytest

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft.algorithm import topology # noqa: E402

rng = np.random.default_rng()
dims = np.array((10.0, 10.0, 10.0))

def test_func_create_atoms():

    # TEST CASE 1: N not specified
    with pytest.raises(ValueError):
        topology.create_atoms(dims)

    # TEST CASE 2: N not an integer
    with pytest.raises(ValueError):
        topology.create_atoms(dims, np.pi)

    # TEST CASE 3: Invalid N_p
    with pytest.raises(ValueError):
        topology.create_atoms(dims, N=9000, N_p=9001)

    # TEST CASE 4: N not divisible by N_p
    with pytest.raises(ValueError):
        topology.create_atoms(dims, N=10, N_p=3)

    # TEST CASE 5: Random melt in reduced units
    N = rng.integers(1, 1000)
    pos = topology.create_atoms(dims, N)
    assert pos.shape == (N, 3)

    # TEST CASE 6: Random melt with default length unit
    pos = topology.create_atoms(dims * unit.nanometer, N)
    assert pos.shape == (N, 3) and pos.unit == unit.nanometer

    # TEST CASE 7: Random melt with specific length unit
    pos = topology.create_atoms(dims * unit.nanometer, N, length_unit=unit.angstrom)
    assert pos.shape == (N, 3) and pos.unit == unit.angstrom

    # TEST CASE 8: Topology provided instead of dimensions
    topo = app.Topology()
    topo.setUnitCellDimensions(dims)
    pos = topology.create_atoms(topo, N)
    assert pos.shape == (N, 3)

    # TEST CASE 9: Random polymer melt
    M = rng.integers(1, 100)
    N_p = rng.integers(4, 100)
    N = M * N_p
    pos = topology.create_atoms(dims, N, N_p)
    assert pos.shape == (N, 3)

    # TEST CASE 10: Random polymer melt with bond, angle, and dihedral
    # information and wrapped positions
    pos, bonds, angles, dihedrals = topology.create_atoms(
        dims, N, N_p, bonds=True, angles=True, dihedrals=True, randomize=True,
        wrap=True
    )
    assert pos.shape == (N, 3)
    assert bonds.shape[0] == N - M
    assert angles.shape[0] == N - 2 * M
    assert dihedrals.shape[0] == N - 3 * M
    assert np.all((pos[:, 0] > 0) & (pos[:, 0] < dims[0]))
    assert np.all((pos[:, 1] > 0) & (pos[:, 2] < dims[1]))
    assert np.all((pos[:, 1] > 0) & (pos[:, 2] < dims[2]))

    # TEST CASE 11: FCC lattice with flexible dimensions
    pos, new_dims = topology.create_atoms(dims, lattice="fcc", length=0.8,
                                          flexible=True)
    assert np.allclose(pos[4], 0.8 * np.array((0, np.sqrt(3) / 3, 2 * np.sqrt(6) / 3)))
    assert np.allclose(dims, new_dims, atol=1)

    # TEST CASE 12: HCP lattice with flexible dimensions
    pos, new_dims = topology.create_atoms(dims, lattice="hcp", length=0.8,
                                          flexible=True)
    assert np.allclose(pos[1], 0.8 * np.array((0.5, np.sqrt(3) / 2, 0)))
    assert np.allclose(dims, new_dims, atol=1)

    # TEST CASE 13: HCP lattice to fill specified dimensions
    pos, new_dims = topology.create_atoms(dims, lattice="hcp", length=0.8)
    assert np.allclose(pos[1], 0.8 * np.array((0.5, np.sqrt(3) / 2, 0)))
    assert np.allclose(dims, new_dims, atol=1)

    # TEST CASE 14: Graphene wall
    pos, new_dims = topology.create_atoms(dims, lattice="honeycomb",
                                          length=0.142 * unit.nanometer,
                                          flexible=True)
    assert pos[1, 1] == 0.142 * unit.nanometer
    assert np.allclose(dims[:2], new_dims[:2], atol=1)
    assert new_dims[2] == 0 * unit.nanometer

    # TEST CASE 15: Cubic crystal lattice
    pos, new_dims = topology.create_atoms(dims, lattice="cubic", length=1)
    assert np.allclose(pos[-1], dims - 1)

def test_func_unwrap():

    pos_old = np.array(((2.0, 2.0, 2.0),))
    images = np.zeros_like(pos_old, dtype=int)
    thresholds = dims / 2
    pos = np.array(((8.0, 8.0, 8.0),))

    # TEST CASE 1: Unwrap not in-place
    pos_unwrapped, pos_old_updated, images = topology.unwrap(
        pos, pos_old, dims, thresholds=thresholds, images=images,
        in_place=False
    )
    assert (np.allclose(pos_unwrapped[0], -2)
            and np.allclose(pos, pos_old_updated))

    # TEST CASE 2: Unwrap in-place
    topology.unwrap(pos, pos_old, dims)
    assert np.allclose(pos[0], -2)

def test_func_wrap():

    pos = np.array(((9.0, 10.0, 11.0),))

    # TEST CASE 1: Wrap not in-place
    pos_wrapped = topology.wrap(pos, dims, in_place=False)
    assert np.allclose(pos_wrapped[0], (9, 10, 1))

    # TEST CASE 2: Wrap in-place
    topology.wrap(pos, dims)
    assert np.allclose(pos[0], (9, 10, 1))