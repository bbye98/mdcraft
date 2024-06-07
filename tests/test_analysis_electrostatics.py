import pathlib
import sys

import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF_TRICLINIC, DCD_TRICLINIC
from MDAnalysis.analysis.dielectric import DielectricConstant
import numpy as np

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft.analysis.electrostatics import DipoleMoment # noqa: E402

def test_class_dipole_moment():

    """
    The test cases are adapted from the "Dielectric —
    :code:`MDAnalysis.analysis.dielectric`" page from the MDAnalysis
    User Guide (https://docs.mdanalysis.org/stable/documentation_pages/analysis/dielectric.html).
    """

    universe = mda.Universe(PSF_TRICLINIC, DCD_TRICLINIC)

    diel = DielectricConstant(universe.atoms)
    diel.run()

    dm = DipoleMoment(universe.atoms).run()
    dm.calculate_relative_permittivity(300)

    pdm = DipoleMoment(universe.atoms, parallel=True).run(module="joblib",
                                                          n_jobs=1)
    pdm.calculate_relative_permittivity(300)

    # TEST CASE 1: Relative permittivity of water system
    assert np.isclose(diel.results.eps_mean, dm.results.dielectrics)
    assert np.isclose(diel.results.eps_mean, pdm.results.dielectrics)