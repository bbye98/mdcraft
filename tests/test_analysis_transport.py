import glob
import os
import pathlib
import sys
import urllib

import MDAnalysis as mda
from MDAnalysis.analysis.msd import EinsteinMSD
from MDAnalysis.tests.datafiles import RANDOM_WALK, RANDOM_WALK_TOPO
import numpy as np
from scipy.stats import linregress

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft.analysis import transport # noqa: E402

def test_class_onsager_msd():

    universe = mda.Universe(RANDOM_WALK_TOPO, RANDOM_WALK)

    start = 20
    stop = 60
    time = np.arange(universe.trajectory.n_frames)
    msd = EinsteinMSD(universe).run().results.timeseries / 6
    diff = linregress(time[start:stop], msd[start:stop]).slope

    # TEST CASE 1: MSD and diffusion coefficients of a random walk
    # calculated using the Einstein relation
    universe.dimensions = np.array((np.inf, np.inf, np.inf, 90, 90, 90))
    onsager_shift = transport.Onsager(universe.atoms, fft=False, reduced=True).run()
    onsager_shift.calculate_transport_coefficients(start, stop, scale="linear")
    assert np.allclose(onsager_shift.results.msd_self[0, 0], msd)
    assert np.isclose(diff, onsager_shift.results.D_i[0, 0])

    # TEST CASE 2: MSD and diffusion coefficients of a random walk
    # calculated using the FFT-based algorithm
    universe.dimensions = np.array((np.inf, np.inf, np.inf, 90, 90, 90))
    onsager_fft = transport.Onsager(universe.atoms, reduced=True).run()
    onsager_fft.calculate_transport_coefficients(start, stop, scale="linear")
    assert np.allclose(onsager_fft.results.msd_self[0, 0], msd)
    assert np.isclose(diff, onsager_fft.results.D_i[0, 0])

def test_class_onsager_transport_coefficients():

    """
    The test cases are adapted from the "Mean Squared Displacement —
    :code:`MDAnalysis.analysis.msd`" page from the MDAnalysis User Guide
    (https://docs.mdanalysis.org/stable/documentation_pages/analysis/msd.html)
    and uses data from the paper "Onsager Transport Coefficients and
    Transference Numbers in Polyelectrolyte Solutions and Polymerized
    Ionic Liquids" by Fong et al.
    (https://doi.org/10.1021/acs.macromol.0c02001).
    """

    def acf_fft(x):
        N = len(x)
        f = np.fft.fft(x, n=2 * N)
        return np.fft.ifft(f * f.conj())[:N].real / (N * np.ones(N)
                                                          - np.arange(0, N))

    def msd_fft(r):
        N = len(r)
        D = np.append(np.square(r).sum(axis=1), 0)
        Q = 2 * D.sum()
        S1 = np.zeros(N)
        for m in range(N):
            Q = Q - D[m - 1] - D[N - m]
            S1[m] = Q / (N - m)
        return S1 - 2 * sum(acf_fft(r[:, i]) for i in range(r.shape[1]))

    def ccf_fft(x, y):
        N = len(x)
        return np.fft.ifft(
            np.fft.fft(x, n=2 ** (2 * N - 1).bit_length())
            * np.fft.fft(y, n=2 ** (2 * N - 1).bit_length()).conj()
        )[:N].real / (N * np.ones(N) - np.arange(0, N))

    def msd_cross_fft(r, k):
        N = len(r)
        D = np.append(np.multiply(r, k).sum(axis=1), 0)
        Q = 2 * D.sum()
        S1 = np.zeros(N)
        for m in range(N):
            Q = Q - D[m - 1] - D[N - m]
            S1[m] = Q / (N - m)
        return S1 - sum(ccf_fft(r[:, i], k[:, i]) for i in range(r.shape[1])) \
               - sum(ccf_fft(k[:, i], r[:, i]) for i in range(k.shape[1]))

    def calc_L_ii_self(positions):
        L_ii_self = np.zeros(positions.shape[0])
        for i in range(positions.shape[1]):
            L_ii_self += msd_fft(positions[:, i, :])
        return L_ii_self

    def calc_L_ii(positions):
        return msd_fft(positions.sum(axis=1))

    def calc_L_ij(cation_positions, anion_positions):
        return msd_cross_fft(cation_positions.sum(axis=1),
                             anion_positions.sum(axis=1))

    def compute_L_ij(anion_positions, cation_positions, volume):
        return np.vstack(
            (
                calc_L_ii(anion_positions),
                calc_L_ij(cation_positions, anion_positions),
                calc_L_ii(cation_positions),
                calc_L_ii_self(anion_positions),
                calc_L_ii_self(cation_positions)
            )
        ) / (6 * volume)

    def fit_data(times, f, start, stop):
        return linregress(times[start:stop], f[start:stop])[0]

    path = os.getcwd()
    if "tests" in path:
        path_split = path.split("/")
        path = "/".join(path_split[:path_split.index("tests") + 1])
    else:
        path += "/tests"
    if not os.path.isdir(f"{path}/data/onsager"):
        os.makedirs(f"{path}/data/onsager")
    os.chdir(f"{path}/data/onsager")

    url = "https://raw.githubusercontent.com/kdfong/transport-coefficients-MSD/master/example-data"
    if not os.path.isfile("system.data"):
        with urllib.request.urlopen(f"{url}/system.data") as r:
            with open("system.data", "w") as f:
                f.write(r.read().decode())
    for i in range(1, 6):
        if not os.path.isfile(f"traj_{i}.dcd"):
            with urllib.request.urlopen(f"{url}/traj_{i}.dcd") as r:
                with open(f"traj_{i}.dcd", "wb") as f:
                    f.write(r.read())

    dt = 50
    start = 40
    fit_start = 2
    fit_stop = 20
    fit_start_self = 20
    fit_stop_self = 50

    universe = mda.Universe("system.data", glob.glob("*.dcd"), format="LAMMPS")
    groups = [universe.select_atoms(f"type {i}") for i in range(1, 3)]
    positions = [np.zeros((universe.trajectory.n_frames - start, g.n_atoms, 3))
                 for g in groups]
    for i, _ in enumerate(universe.trajectory[40:]):
        com = universe.atoms.center_of_mass(wrap=True)
        for g, p in zip(groups, positions):
            p[i] = g.positions - com
    volume = universe.dimensions[:3].prod()
    times = np.arange(0, (universe.trajectory.n_frames - start) * dt, dt,
                      dtype=int)
    msds = compute_L_ij(*positions, volume)

    onsager = transport.Onsager(groups, temperature=1, center=True,
                                center_atom=True, center_wrap=True,
                                reduced=True, dt=dt).run(start=40)
    onsager.calculate_transport_coefficients(fit_start, fit_stop,
                                             start_self=fit_start_self,
                                             stop_self=fit_stop_self,
                                             scale="linear", enforce_linear=False)

    L_ij_array = np.triu(onsager.results.L_ij)
    for i, (msd, L_ij) in enumerate(zip(msds, L_ij_array[L_ij_array != 0])):

        # TEST CASE 1: Cross displacements of polyelectrolyte system
        assert np.allclose(msd, onsager.results.msd_cross[i, 0] / volume,
                           atol=1e-3)

        # TEST CASE 2: Onsager transport coefficients of polyelectrolyte
        # system
        assert np.isclose(fit_data(times, msd, fit_start, fit_stop), L_ij)

    for i, (msd, L_ii_self) in enumerate(zip(msds[3:], onsager.results.L_ii_self[0])):

        # TEST CASE 1: MSDs of polyelectrolyte system
        assert np.allclose(
            msd, groups[i].n_atoms * onsager.results.msd_self[i, 0] / volume,
            atol=1e-6
        )

        # TEST CASE 2: Self Onsager transport coefficients of
        # polyelectrolyte system
        assert np.isclose(
            fit_data(times, msd, fit_start_self, fit_stop_self),
            L_ii_self
        )