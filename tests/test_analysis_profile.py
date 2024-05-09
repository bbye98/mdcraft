import pathlib
import sys

import MDAnalysis as mda
from MDAnalysis.tests.datafiles import waterDCD, waterPSF
from MDAnalysis.analysis.lineardensity import LinearDensity
import numpy as np

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft.analysis import profile # noqa: E402

universe = mda.Universe(waterPSF, waterDCD)

def test_class_density_profile():

    """
    The test cases are adapted from the "Computing mass and charge
    density on each axis" page from the MDAnalysis User Guide
    (https://userguide.mdanalysis.org/stable/examples/analysis/volumetric/linear_density.html).
    """

    density = LinearDensity(universe.atoms, grouping="residues").run()
    density_profile = profile.DensityProfile(
        universe.atoms, "residues", n_bins=200
    ).run()
    parallel_density_profile = profile.DensityProfile(
        universe.atoms, "residues", n_bins=200, parallel=True
    ).run()

    for i, axis in enumerate("xyz"):

        number_density = (
            0.602214076 * getattr(density.results, axis).mass_density
            / universe.residues.masses[0]
        )
        charge_density = (0.602214076
                          * getattr(density.results, axis).charge_density)

        # TEST CASE 1: Number density profiles
        assert(np.allclose(number_density,
                           density_profile.results.number_densities[i]))
        assert(np.allclose(number_density,
                           parallel_density_profile.results.number_densities[i]))

        # TEST CASE 2: Charge density profiles
        assert(np.allclose(charge_density,
                           density_profile.results.charge_densities[i]))
        assert(np.allclose(charge_density,
                           parallel_density_profile.results.charge_densities[i]))