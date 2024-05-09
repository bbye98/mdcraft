import os
import pathlib
import sys
import tarfile
import urllib
import warnings

import ase.io
import dynasor
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.tests.datafiles import TPR, XTC
import numpy as np

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft.analysis import structure # noqa: E402

warnings.filterwarnings("ignore")
rng = np.random.default_rng()

def test_func_radial_histogram():

    L = 20
    half_L = L // 2
    dims = np.array((L, L, L, 90, 90, 90), dtype=int)
    origin = half_L * np.ones(3)

    N = 1_000
    norm = L // 2 * rng.random(N)
    counts = np.histogram(norm, bins=half_L, range=(0, half_L + 1))[0]
    neighbors = rng.random((N, 3))
    neighbors *= norm[:, None] / np.linalg.norm(neighbors, axis=1, keepdims=True)
    neighbors += dims[:3] / 2

    # TEST CASE 1: Correct radial histogram for randomly placed particles
    assert np.array_equal(counts,
                          structure.radial_histogram(origin, neighbors,
                                                     n_bins=half_L,
                                                     range=(0, half_L + 1),
                                                     dims=dims))

def test_func_radial_fourier_transform():

    alpha = 1 + 9 * rng.random()
    r = np.linspace(1e-8, 20, 2_000)
    q = 1 / r
    f = np.exp(-alpha * r) / r
    F = 4 * np.pi / (alpha ** 2 + q ** 2)

    # TEST CASE 1: Radial Fourier transform of function exp(-ar)/r,
    # which has analytical form 4*pi/(a^2+q^2)
    assert np.allclose(F, structure.radial_fourier_transform(r, f, q),
                       atol=4e-5)

"""
The following test cases (test_class_rdf_*) are adapted from the
"Average radial distribution functions" page from the MDAnalysis User
Guide (https://userguide.mdanalysis.org/stable/examples/analysis/structure/average_rdf.html).
"""

def test_class_rdf():

    universe = mda.Universe(TPR, XTC)
    res60 = universe.select_atoms("resid 60")
    water = universe.select_atoms("resname SOL")
    thr = universe.select_atoms("resname THR")
    n_bins = 75

    ### residue60/water
    rdf = InterRDF(res60, water, nbins=n_bins).run()

    # TEST CASE 1: Batched serial RDF calculation
    serial_rdf = structure.RadialDistributionFunction(
        res60, water, n_bins=n_bins, n_batches=2
    ).run()
    assert np.allclose(rdf.results.bins, serial_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, serial_rdf.results.rdf)

    # TEST CASE 2: Batched parallel RDF calculation
    parallel_rdf = structure.RadialDistributionFunction(
        res60, water, n_bins=n_bins, n_batches=2, parallel=True
    ).run()
    assert np.allclose(rdf.results.bins, parallel_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, parallel_rdf.results.rdf)

    ### residue60/residue60 w/ self exclusion
    exclusion = (1, 1)
    rdf = InterRDF(res60, res60, nbins=n_bins, exclusion_block=exclusion).run()

    # TEST CASE 1: Serial RDF calculation
    serial_rdf = structure.RadialDistributionFunction(
        res60, n_bins=n_bins, exclusion=exclusion
    ).run()
    assert np.allclose(rdf.results.bins, serial_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, serial_rdf.results.rdf)

    # TEST CASE 2: Parallel RDF calculation
    parallel_rdf = structure.RadialDistributionFunction(
        res60, n_bins=n_bins, exclusion=exclusion, parallel=True
    ).run()
    assert np.allclose(rdf.results.bins, parallel_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, parallel_rdf.results.rdf)

    ### threonine/threonine w/ self exclusion
    exclusion = (14, 14)
    rdf = InterRDF(thr, thr, nbins=n_bins, exclusion_block=exclusion).run()

    # TEST CASE 1: Serial RDF calculation
    serial_rdf = structure.RadialDistributionFunction(
        thr, n_bins=n_bins, exclusion=exclusion
    ).run()
    assert np.allclose(rdf.results.bins, serial_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, serial_rdf.results.rdf)

    # TEST CASE 2: Parallel RDF calculation
    parallel_rdf = structure.RadialDistributionFunction(
        thr, n_bins=n_bins, exclusion=exclusion, parallel=True
    ).run()
    assert np.allclose(rdf.results.bins, parallel_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, parallel_rdf.results.rdf)

    ### threonine/threonine w/ carbon exclusion
    exclusion = (4, 10)
    rdf = InterRDF(thr, thr, nbins=n_bins, exclusion_block=exclusion).run()

    # TEST CASE 1: Serial RDF calculation
    serial_rdf = structure.RadialDistributionFunction(
        thr, n_bins=n_bins, exclusion=exclusion
    ).run()
    assert np.allclose(rdf.results.bins, serial_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, serial_rdf.results.rdf)

    # TEST CASE 2: Parallel RDF calculation
    parallel_rdf = structure.RadialDistributionFunction(
        thr, n_bins=n_bins, exclusion=exclusion, parallel=True
    ).run()
    assert np.allclose(rdf.results.bins, parallel_rdf.results.bins)
    assert np.allclose(rdf.results.rdf, parallel_rdf.results.rdf)

def test_class_structurefactor():

    """
    The following test cases are adapted from the "Static structure
    factor in halide perovskite (CsPbI3)" page from the dynasor documentation
    (https://dynasor.materialsmodeling.org/dev/tutorials/static_structure_factor.html).
    """

    path = os.getcwd()
    if "tests" in path:
        path_split = path.split("/")
        path = "/".join(path_split[:path_split.index("tests") + 1])
    else:
        path += "/tests"
    if not os.path.isdir(f"{path}/data/ssf"):
        os.makedirs(f"{path}/data/ssf")
    os.chdir(f"{path}/data/ssf")

    if not os.path.isdir("md_runs"):
        with urllib.request.urlopen(
                "https://zenodo.org/records/10149723/files/md_runs.tar.gz"
            ) as r:
            with open("md_runs.tar.gz", "wb") as f:
                f.write(r.read())
        with tarfile.open("md_runs.tar.gz", "r:gz") as tar:
            tar.extractall()
        os.remove("md_runs.tar.gz")
    os.chdir("md_runs/NVT_tetra_size8_T450_nframes1000")

    q_max = 2.2
    stop = 10
    atoms = ase.io.read("model.xyz")
    atomic_indices = atoms.symbols.indices()
    atom_types = sorted(atomic_indices.keys())
    traj = dynasor.Trajectory("movie.nc", trajectory_format="nc",
                              atomic_indices=atomic_indices,
                              frame_stop=stop)
    q_points = dynasor.get_spherical_qpoints(traj.cell, q_max=q_max)
    sample = dynasor.compute_static_structure_factors(traj, q_points)
    q_norms = np.linalg.norm(sample.q_points, axis=1)
    q_norms_unique = np.unique(np.round(q_norms, 11))

    universe = mda.Universe("model.xyz", "movie.nc")
    groups = [universe.select_atoms(f"element {e}") for e in atom_types]

    # TEST CASE 1: Partial structure factors
    psf_exp = structure.StructureFactor(
        groups, mode="partial", q_max=q_max, parallel=True
    ).run(stop=stop)
    psf_trig = structure.StructureFactor(
        groups, mode="partial", q_max=q_max, form="trig", parallel=True
    ).run(stop=stop)
    psf_dynasor = np.empty((len(psf_exp.results.pairs), len(q_norms_unique)))
    for i, (j, k) in enumerate(psf_exp.results.pairs):
        psf_dynasor[i] = np.fromiter(
            (sample[f"Sq_{atom_types[j]}_{atom_types[k]}"]
             [np.isclose(q, q_norms)].mean() for q in q_norms_unique),
            dtype=float,
            count=len(q_norms_unique)
        )
    assert np.allclose(psf_exp.results.wavenumbers, q_norms_unique)
    assert np.allclose(psf_exp.results.ssf, psf_dynasor, atol=1e-6)
    assert np.allclose(psf_trig.results.ssf, psf_dynasor, atol=1e-6)

    # TEST CASE 2: Static structure factors
    ssf_exp = structure.StructureFactor(
        groups, wavevectors=q_points, sort=False, unique=False, parallel=True
    ).run(stop=stop)
    ssf_trig = structure.StructureFactor(
        groups, form="trig", wavevectors=q_points, sort=False, unique=False,
        parallel=True
    ).run(stop=stop)
    assert np.allclose(ssf_exp.results.ssf[0], sample.Sq[:, 0])
    assert np.allclose(ssf_trig.results.ssf[0], sample.Sq[:, 0])

def test_class_intermediatescatteringfunction():

    path = os.getcwd()
    if "tests" in path:
        path_split = path.split("/")
        path = "/".join(path_split[:path_split.index("tests") + 1])
    else:
        path += "/tests"
    if not os.path.isdir(f"{path}/data/ssf"):
        os.makedirs(f"{path}/data/ssf")
    os.chdir(f"{path}/data/ssf")

    if not os.path.isdir("md_runs"):
        with urllib.request.urlopen(
                "https://zenodo.org/records/10149723/files/md_runs.tar.gz"
            ) as r:
            with open("md_runs.tar.gz", "wb") as f:
                f.write(r.read())
        with tarfile.open("md_runs.tar.gz", "r:gz") as tar:
            tar.extractall()
        os.remove("md_runs.tar.gz")
    os.chdir("md_runs/NVT_tetra_size8_T450_nframes1000")

    stop = 20
    n_lags = stop // 2
    atoms = ase.io.read("model.xyz")
    atomic_indices = atoms.symbols.indices()
    atom_types = sorted(atomic_indices.keys())
    traj = dynasor.Trajectory("movie.nc", trajectory_format="nc",
                              atomic_indices=atomic_indices,
                              frame_stop=stop)
    q_points = dynasor.get_spherical_qpoints(traj.cell, q_max=2,
                                             max_points=2_000)
    sample = dynasor.compute_dynamic_structure_factors(
        traj, q_points, dt=1, window_size=n_lags, calculate_incoherent=True
    )
    q_norms = np.linalg.norm(sample.q_points, axis=1)
    q_norms_unique = np.unique(np.round(q_norms, 11))

    universe = mda.Universe("model.xyz", "movie.nc")
    groups = [universe.select_atoms(f"element {e}") for e in atom_types]
    isf_exp = structure.IntermediateScatteringFunction(
        groups, mode="partial", wavevectors=q_points, n_lags=n_lags,
        incoherent=True, parallel=True
    ).run(stop=stop)
    isf_trig = structure.IntermediateScatteringFunction(
        groups, mode="partial", form="trig", wavevectors=q_points,
        n_lags=n_lags, incoherent=True, parallel=True
    ).run(stop=stop)

    # TEST CASE 1: Partial coherent intermediate scattering functions
    cisf_dynasor = np.empty((n_lags, len(isf_exp.results.pairs),
                             len(q_norms_unique)))
    for i, (j, k) in enumerate(isf_exp.results.pairs):
        for iq, q in enumerate(q_norms_unique):
            cisf_dynasor[:, i, iq] = (
                sample[f"Fqt_coh_{atom_types[j]}_{atom_types[k]}"]
                [np.isclose(q, q_norms), :n_lags].mean(axis=0)
            )
    assert np.allclose(isf_exp.results.cisf, cisf_dynasor)
    assert np.allclose(isf_trig.results.cisf, cisf_dynasor)

    # TEST CASE 2: Partial incoherent intermediate scattering functions
    iisf_dynasor = np.empty((n_lags, len(atom_types), len(q_norms_unique)))
    for i in range(len(atom_types)):
        for iq, q in enumerate(q_norms_unique):
            iisf_dynasor[:, i, iq] = (
                sample[f"Fqt_incoh_{atom_types[i]}"]
                [np.isclose(q, q_norms), :n_lags].mean(axis=0)
            )
    assert np.allclose(isf_exp.results.iisf, iisf_dynasor)
    assert np.allclose(isf_trig.results.iisf, iisf_dynasor)

    isf_exp = structure.IntermediateScatteringFunction(
        groups, wavevectors=q_points, n_lags=n_lags, incoherent=True,
        sort=False, unique=False, parallel=True
    ).run(stop=stop)
    isf_trig = structure.IntermediateScatteringFunction(
        groups, form="trig", wavevectors=q_points, n_lags=n_lags,
        incoherent=True, sort=False, unique=False, parallel=True
    ).run(stop=stop)

    # TEST CASE 3: Coherent intermediate scattering function
    assert np.allclose(isf_exp.results.cisf[:, 0], sample.Fqt_coh[:, :n_lags].T)
    assert np.allclose(isf_trig.results.cisf[:, 0], sample.Fqt_coh[:, :n_lags].T)

    # TEST CASE 4: Incoherent intermediate scattering function
    assert np.allclose(isf_exp.results.iisf[:, 0], sample.Fqt_incoh[:, :n_lags].T)
    assert np.allclose(isf_trig.results.iisf[:, 0], sample.Fqt_incoh[:, :n_lags].T)