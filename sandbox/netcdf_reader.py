import os

from mdcraft.core import Trajectory
from mdcraft.io.writer import NetCDFWriter


os.chdir(
    "/mnt/c/Users/Benjamin/Documents/GitHub/mdcraft-dev/tests/data/trajectories/lammps"
)
trajectory = Trajectory("ljmelt_lj.nc")
reader = trajectory._readers[0]
nc = reader._file
frame = trajectory[0]

# os.chdir(
#     "/mnt/c/Users/Benjamin/Documents/GitHub/mdcraft-dev/tests/data/trajectories"
# )
# writer = NetCDFWriter("test.nc")
# writer._write_header(
#     N=1000,
#     cell=True,
#     time=True,
#     positions=True,
#     velocities=True,
#     forces=True,
#     title="Test",
#     remd_dimensions=["Temperature", "Partial", "Hamiltonian", "pH", "RedOx"],
# )

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
