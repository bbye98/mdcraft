import pathlib
import sys

import MDAnalysis as mda
from MDAnalysis.tests.datafiles import DCD, PSF
from MDAnalysis.analysis.base import AnalysisFromFunction
import numpy as np

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft.analysis import polymer # noqa: E402

rng = np.random.default_rng()
universe = mda.Universe(PSF, DCD)
protein = universe.select_atoms("protein")

def test_class_gyradius():

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

    rog = AnalysisFromFunction(radius_of_gyration, universe.trajectory,
                               protein).run()

    # TEST CASE 1: Time series of overall radii of gyration
    gyradius = polymer.Gyradius(protein, grouping="residues").run()
    gyradius_parallel = polymer.Gyradius(
        protein, grouping="residues", parallel=True
    ).run()
    assert np.allclose(rog.results["timeseries"][:, 0],
                       gyradius.results.gyradii[0])
    assert np.allclose(rog.results["timeseries"][:, 0],
                       gyradius_parallel.results.gyradii[0])

    # TEST CASE 2: Time series of radius of gyration components
    gyradius = polymer.Gyradius(protein, grouping="residues",
                                components=True).run()
    gyradius_parallel = polymer.Gyradius(
        protein, grouping="residues", components=True, parallel=True
    ).run()
    assert np.allclose(rog.results["timeseries"][:, 1:],
                       gyradius.results.gyradii[0])
    assert np.allclose(rog.results["timeseries"][:, 1:],
                       gyradius_parallel.results.gyradii[0])