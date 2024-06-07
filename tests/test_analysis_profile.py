import pathlib
import sys

import MDAnalysis as mda
from MDAnalysis.tests.datafiles import waterDCD, waterPSF
from MDAnalysis.analysis.lineardensity import LinearDensity
import numpy as np
import pytest

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft.analysis.profile import DensityProfile # noqa: E402

universe = mda.Universe(waterPSF, waterDCD)

def test_class_density_profile():

    """
    The first two test cases are adapted from the "Computing mass and
    charge density on each axis" page from the MDAnalysis User Guide
    (https://userguide.mdanalysis.org/stable/examples/analysis/volumetric/linear_density.html).
    """

    ld = LinearDensity(universe.atoms, "residues").run()
    dp = DensityProfile(universe.atoms, "residues", axes="xy", n_bins=200,
                        average=False).run()
    pdp = DensityProfile(universe.atoms, "residues", axes="xy", n_bins=200,
                         parallel=True).run(module="joblib", n_jobs=1)

    for ax in "xy":
        number_density = (0.602214076 * ld.results[ax].mass_density
                          / universe.residues.masses[0])
        charge_density = 0.602214076 * ld.results[ax].charge_density

        # TEST CASE 1: Number density profiles
        assert np.allclose(number_density,
                           dp.results.number_densities[ax].mean(axis=1))
        assert np.allclose(number_density, pdp.results.number_densities[ax])

        # TEST CASE 2: Charge density profiles
        assert np.allclose(charge_density,
                           dp.results.charge_densities[ax].mean(axis=0))
        assert np.allclose(charge_density, pdp.results.charge_densities[ax])

    # TEST CASE 3: Wrong number of dielectric constants
    with pytest.raises(ValueError):
        dp.calculate_potential_profiles(dielectrics=(78, 78, 78))

    # TEST CASE 4: Invalid axes
    for axes in ["a", 0, [0, 1]]:
        with pytest.raises(ValueError):
            dp.calculate_potential_profiles(axes, 78)

    # TEST CASE 5: Invalid or wrong number of surface charge densities
    tests = [(0, 0, 0), np.zeros((3, 9))]
    for sigmas_q in tests:
        with pytest.raises(ValueError):
            dp.calculate_potential_profiles(dielectrics=78, sigmas_q=sigmas_q)

    # TEST CASE 6: Invalid or wrong number of potential differences
    for dVs in tests:
        with pytest.raises(ValueError):
            dp.calculate_potential_profiles(dielectrics=78, dVs=dVs)

    # TEST CASE 7: Invalid or wrong number of thresholds
    for thresholds in tests:
        with pytest.raises(ValueError):
            dp.calculate_potential_profiles(dielectrics=78,
                                            thresholds=thresholds)

    # TEST CASE 8: Invalid or wrong number of left boundary potentials
    for V0s in tests:
        with pytest.raises(ValueError):
            dp.calculate_potential_profiles(dielectrics=78, V0s=V0s)

    # TEST CASE 9: Wrong number of methods
    with pytest.raises(ValueError):
        dp.calculate_potential_profiles(
            dielectrics=78,
            methods=("integral", "integral", "matrix")
        )

    # TEST CASE 10: Wrong number of booleans for 'pbcs'
    with pytest.raises(ValueError):
        dp.calculate_potential_profiles(dielectrics=78,
                                        pbcs=(True, True, False))

    # TEST CASE 11: Invalid axis
    with pytest.raises(ValueError):
        dp.calculate_potential_profiles("z", 78)

    # TEST CASE 12: Potential profiles from integration
    dp.calculate_potential_profiles(dielectrics=78, sigmas_q=0,
                                    methods="integral")
    for ax in "xy":
        assert np.allclose(dp.results.potentials[ax], 0)

    # TEST CASE 13: Potential profiles from system of equations
    dp.calculate_potential_profiles(dielectrics=78, sigmas_q=0,
                                    methods="matrix")
    for ax in "xy":
        assert np.allclose(dp.results.potentials[ax], 0)