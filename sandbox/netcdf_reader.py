from __future__ import annotations
import os

from mdcraft.core import Trajectory
from mdcraft.io.reader import NetCDFReader


os.chdir("/mnt/c/Users/Benjamin/Downloads")
reader = NetCDFReader("example.nc", module="netcdf4", dt=0.8058974)
trajectory = Trajectory("example.nc", module="netcdf4", dt=0.8058974)
debug = True

# n_frames = 2500

# start = datetime.now()
# data_serial = []
# file = netcdf_file(filename)
# for i in range(n_frames):
#     data_serial.append(file.variables["coordinates"][i])
# print(f"Serial: {datetime.now() - start}")


# def read_parallel(file, i):
#     return file.variables["coordinates"][i]


# start = datetime.now()
# data_parallel = []
# file = nc.Dataset(filename)
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     for future in concurrent.futures.as_completed(
#         executor.submit(read_parallel, file, i) for i in range(n_frames)
#     ):
#         data_serial.append(future.result())
# print(f"Parallel: {datetime.now() - start}")

### BENCHMARK

# os.chdir("/mnt/e/research/gcme/methodology/data/polyanion_counterion_solvent/edl/ic")
# filename = "nvt_N_96000_Np_60_xp_0.005_rp_78.0_A_25.224_dV_0.000__0.nc"

# print("Parallel (scipy.io.netcdf_file):")
# start = datetime.now()
# reader = NetCDFReader(filename, parallel=True)
# n_frames = reader.n_frames
# n_workers = 20
# frames_per_worker = np.ceil(n_frames / n_workers).astype(int)
# frames = []
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     for future in concurrent.futures.as_completed(
#         executor.submit(
#             reader.get_frames,
#             slice(n * frames_per_worker, (n + 1) * frames_per_worker),
#             parallel=True
#         )
#         for n in range(n_workers)
#     ):
#         frames.extend(future.result())
# print(f"  Read {n_frames} frames: {datetime.now() - start}")

# print("Parallel (netCDF4.Dataset):")
# start = datetime.now()
# reader = NetCDFReader(filename, module="netcdf4", parallel=True)
# n_frames = reader.n_frames
# n_workers = 20
# frames_per_worker = np.ceil(n_frames / n_workers).astype(int)
# frames = []
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     for future in concurrent.futures.as_completed(
#         executor.submit(
#             reader.get_frames,
#             slice(n * frames_per_worker, (n + 1) * frames_per_worker),
#             parallel=True
#         )
#         for n in range(n_workers)
#     ):
#         frames.extend(future.result())
# print(f"  Read {n_frames} frames: {datetime.now() - start}")

# print("Serial (scipy.io.netcdf_file):")
# start = datetime.now()
# reader = NetCDFReader(filename)
# n_frames = reader.n_frames
# frames_serial_scipy = reader.get_frames(range(n_frames))
# print(f"  Read {n_frames} frames: {datetime.now() - start}")

# print("Serial (netCDF4.Dataset):")
# start = datetime.now()
# reader = NetCDFReader(filename, module="netcdf4")
# n_frames = reader.n_frames
# frames_serial_netcdf4 = reader.get_frames(range(n_frames))
# print(f"  Read {n_frames} frames: {datetime.now() - start}")

debug = True
