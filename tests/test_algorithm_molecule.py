import pathlib
import sys

import MDAnalysis as mda
from MDAnalysis.tests.datafiles import DCD, PSF
import numpy as np
import pytest

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft.algorithm import molecule # noqa: E402

# Load sample topology and trajectory
universe = mda.Universe(PSF, DCD)
protein = universe.select_atoms("protein")
core = universe.select_atoms("protein and (resid 1-29 60-121 160-214)")
nmp = universe.select_atoms("protein and resid 30-59")
lid = universe.select_atoms("protein and resid 122-159")
arg = universe.select_atoms("resname ARG")

def test_func_center_of_mass():

    """
    Test cases 5–15 are inspired by the example in the "Working with
    AtomGroups" section of the MDAnalysis Tutorial
    (https://www.mdanalysis.org/MDAnalysisTutorial/atomgroups.html).
    """

    # TEST CASE 1: No topology or trajectory
    with pytest.raises(ValueError):
        molecule.center_of_mass()

    # TEST CASE 2: Invalid grouping
    with pytest.raises(ValueError):
        molecule.center_of_mass(universe.atoms, "atoms")

    # TEST CASE 3: No system dimension information when number of periodic
    # boundary crossings is provided
    with pytest.raises(ValueError):
        molecule.center_of_mass(universe.atoms,
                                images=np.zeros((universe.atoms.n_atoms, 3)))

    # TEST CASE 4: Incompatible mass and position arrays
    with pytest.raises(ValueError):
        molecule.center_of_mass(
            masses=universe.atoms.masses,
            positions=[r.atoms.positions for r in universe.residues]
        )

    # TEST CASE 5: Center of mass of domains in AdK
    assert np.allclose(molecule.center_of_mass(core), core.center_of_mass())
    assert np.allclose(molecule.center_of_mass(nmp), nmp.center_of_mass())
    assert np.allclose(molecule.center_of_mass(lid), lid.center_of_mass())

    # TEST CASE 6: Center of mass of all particles using AtomGroup
    com = universe.atoms.center_of_mass()
    assert np.allclose(molecule.center_of_mass(universe.atoms), com)

    # TEST CASE 7: Center of mass of all particles using AtomGroup, but
    # with the masses and unwrapped positions returned
    c, m, p = molecule.center_of_mass(
        universe.atoms,
        images=np.zeros((universe.atoms.n_atoms, 3), dtype=int),
        dimensions=np.array((0, 0, 0)),
        raw=True
    )
    assert np.allclose(c, com)
    assert np.allclose(m, universe.atoms.masses)
    assert np.allclose(p, universe.atoms.positions)

    # TEST CASE 8: Centers of mass of different residues using AtomGroup
    res_coms = np.array([r.atoms.center_of_mass()
                         for r in universe.residues])
    assert np.allclose(molecule.center_of_mass(universe.atoms, "residues"),
                       res_coms)

    # TEST CASE 9: Centers of mass of different residues using raw masses
    # and positions from AtomGroup
    assert np.allclose(
        molecule.center_of_mass(
            universe.atoms, "residues",
            masses=[r.atoms.masses for r in universe.residues]
        ),
        res_coms
    )

    # TEST CASE 10: Centers of mass of different residues using raw masses
    # and positions
    assert np.allclose(
        molecule.center_of_mass(
            masses=[r.atoms.masses for r in universe.residues],
            positions=[r.atoms.positions for r in universe.residues]
        ),
        res_coms
    )

    # TEST CASE 11: Centers of mass of arginine residues using AtomGroup
    arg_coms = np.array([r.atoms.center_of_mass() for r in arg.residues])
    assert np.allclose(molecule.center_of_mass(arg, "residues"), arg_coms)

    # TEST CASE 12: Centers of mass of arginine residues using AtomGroup
    # and specified number of residues
    assert np.allclose(molecule.center_of_mass(arg, n_groups=13), arg_coms)

    # TEST CASE 13: Centers of mass of arginine residues using raw masses
    # and positions
    assert np.allclose(molecule.center_of_mass(masses=arg.masses,
                                               positions=arg.positions,
                                               n_groups=13),
                       arg_coms)

    # TEST CASE 14: Centers of mass of only segment in AtomGroup
    assert np.allclose(molecule.center_of_mass(universe.atoms, "segments"),
                       com)

    # TEST CASE 15: Centers of mass of only segment in AtomGroup
    # containing the arginine residues
    assert np.allclose(molecule.center_of_mass(arg, "segments"),
                       arg.center_of_mass())

def test_radius_of_gyration():

    """
    The reference implementation is adapted from the "Writing your own
    trajectory analysis" section of the MDAnalysis User Guide
    (https://userguide.mdanalysis.org/stable/examples/analysis/custom_trajectory_analysis.html).
    """

    def radius_of_gyration(group):
        positions = group.positions
        masses = group.masses
        center_of_mass = group.center_of_mass()
        r_sq = (positions - center_of_mass) ** 2
        r_ssq = np.array((r_sq.sum(axis=1),
                          (r_sq[:, [1, 2]]).sum(axis=1),
                          (r_sq[:, [0, 2]]).sum(axis=1),
                          (r_sq[:, [0, 1]]).sum(axis=1)))
        return np.sqrt((masses * r_ssq).sum(axis=1) / masses.sum())

    # TEST CASE 1: Invalid grouping
    with pytest.raises(ValueError):
        molecule.radius_of_gyration(universe.atoms, "atoms")

    # TEST CASE 2: Overall radius of gyration
    ref = radius_of_gyration(universe.atoms)
    assert np.isclose(molecule.radius_of_gyration(universe.atoms), ref[0])
    assert np.allclose(molecule.radius_of_gyration(universe.atoms,
                                                   components=True),
                       ref[1:])

    # TEST CASE 3: Radii of gyration of arginine residues
    ref = np.array([radius_of_gyration(g.atoms) for g in arg.residues])
    assert np.allclose(molecule.radius_of_gyration(arg, "residues"),
                       ref[:, 0])
    assert np.allclose(molecule.radius_of_gyration(arg, "residues",
                                                   components=True),
                       ref[:, 1:])

    # TEST CASE 4: Radii of gyration of different residues
    ref = np.array([radius_of_gyration(g.atoms) for g in universe.residues])
    assert np.allclose(molecule.radius_of_gyration(universe.atoms, "residues"),
                       ref[:, 0])
    assert np.allclose(molecule.radius_of_gyration(universe.atoms, "residues",
                                                   components=True),
                       ref[:, 1:])