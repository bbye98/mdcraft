import concurrent.futures
from datetime import datetime
import os
import sys

import MDAnalysis as mda
import numpy as np

sys.path.insert(0, "/mnt/c/Users/Benjamin/Documents/GitHub/mdcraft-dev/src")
# from mdcraft.analysis.core import Trajectory
from mdcraft.io.reader import LAMMPSDataReader, LAMMPSDumpReader


os.chdir("/mnt/c/Users/Benjamin/Documents/GitHub/mdcraft-dev/tests/data/topologies")

data_reader = LAMMPSDataReader("benzene.data")
test = data_reader._parse_topology(data_reader._file)
debug = True

# os.chdir("/mnt/e/caltech/research/testing/ananya/data/N_3000_Lx_100_mu_1.600_off_0.001")
os.chdir("/mnt/c/Users/Benjamin/Downloads")

### TOPOLOGY / TESTING

# topology = LAMMPSDataReader("topology.data")

### TOPOLOGY / BENCHMARK

# topology_file = "topology.data"

### TRAJECTORY / TESTING
# # dump style grid
# start = datetime.now()
# sreader = LAMMPSDumpReader("/mnt/e/caltech/research/testing/lammps/tmp.dump.3d")
# print("Serial: ", datetime.now() - start)
# sframe = sreader.read_frames(0)
# start = datetime.now()
# preader = LAMMPSDumpReader("/mnt/e/caltech/research/testing/lammps/tmp.dump.3d", parallel=True, n_threads=1)
# print("Parallel: ", datetime.now() - start)
# pframe = preader.read_frames(0)
# assert sreader.n_frames == preader.n_frames == 201
# assert sreader.n_grids == preader.n_grids == 125
# assert sreader.dt == preader.dt == 0.0025
# assert sreader.time_step == preader.time_step == 0.25
# assert np.allclose(sreader.timesteps, preader.timesteps)

# # trajectory slicing
# trajectory = Trajectory(["subset1.lammpstrj", "subset2.lammpstrj", "subset3.lammpstrj"])
# frames = trajectory.get_frames([9, 10, 11, 12])
# for ts in trajectory:
#     print(ts)
# subset = trajectory[::2]
# for ts in subset[::2]:
#     print(ts)

# # MDAnalysis integration
# universe = mda.Universe(
#     "topology.data", "subset.lammpsdump", atom_style="id type x y z", extras=True
# )

### TRAJECTORY / BENCHMARK
# trajectory_file = "trajectory.lammpsdump"
trajectory_file = "dump.dipoleProd"

# with open(trajectory_file, "r") as i:
#     with open("subset.lammpstrj", "w") as o:
#         for _ in range(1000 * 6009):
#             o.write(i.readline())

# trajectory_file = "subset.lammpstrj"

# print("Parallel:")
# n_workers = 8
# start = datetime.now()
# preader = LAMMPSDumpReader(
#     trajectory_file, parallel=True, n_threads=n_workers
# )  # , extras=True)
# print(f"  Initialize reader and get frame offsets: {datetime.now() - start}")
# n_frames = preader.n_frames // 10
# frames_per_worker = np.ceil(n_frames / n_workers).astype(int)
# start = datetime.now()
# frames_parallel = []
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     for future in concurrent.futures.as_completed(
#         executor.submit(
#             preader.read_frames,
#             slice(n * frames_per_worker, (n + 1) * frames_per_worker),
#             parallel=True,
#         )
#         for n in range(n_workers)
#     ):
#         # future.result()
#         frames_parallel.extend(future.result())
# print(f"  Read {n_frames} frames: {datetime.now() - start}")

# print("Serial:")
# start = datetime.now()
# sreader = LAMMPSDumpReader(trajectory_file)  # , extras=True)
# print(f"  Initialize reader and get frame offsets: {datetime.now() - start}")
# n_frames = sreader.n_frames // 10
# start = datetime.now()
# # for i in range(n_frames):
# #     sreader.read_frames(i)
# frames_serial = sreader.read_frames(range(n_frames))
# print(f"  Read {n_frames} frames: {datetime.now() - start}")

# assert np.allclose(sreader._offsets, preader._offsets)
# assert sreader.n_atoms == preader.n_atoms == 6_000
# assert sreader.dt == preader.dt == 0.005

debug = True

"""
1000 frames / LAPTOP-BYE-DELL:

np.loadtxt:

Parallel:
  Initialize reader and get frame offsets: 0:00:03.519072
  Read 1000 frames: 0:00:06.925570
Serial:
  Initialize reader and get frame offsets: 0:00:15.314994
  Read 1000 frames: 0:00:21.754503

pd.read_csv:

Parallel:
  Initialize reader and get frame offsets: 0:00:03.490919
  Read 1000 frames: 0:00:05.743207
Serial:
  Initialize reader and get frame offsets: 0:00:09.324469
  Read 1000 frames: 0:00:14.362038
"""
