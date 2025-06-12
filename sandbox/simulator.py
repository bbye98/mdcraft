import sys

import MDAnalysis as mda
import numpy as np
from scipy.spatial import KDTree

sys.path.insert(
    0, "/mnt/c/Users/Benjamin/Documents/Repositories/mdcraft-dev/src"
)
from mdcraft.algorithm.neighbor import build_neighbor_list

rng = np.random.default_rng(seed=42)
n_particles = 10
dimensions_nm = np.array((100.0, 100.0, 100.0))
positions_nm = rng.random((n_particles, 3), np.float64) * dimensions_nm
cutoff_nm = 7.5


from time import time

build_neighbor_list(np.array(((0.0, 0.0, 0.0),)), cutoff_nm, dimensions_nm)
start = time()
for _ in range(10):
    build_neighbor_list(
        positions_nm,
        cutoff_nm,
        dimensions_nm,
    )
print("mdcraft.algorithm.neighbor.build_neighbor_list: ", (time() - start) / 10)

lattice = np.concatenate((dimensions_nm, (90.0, 90.0, 90.0)))
start = time()
for _ in range(2):
    mda.lib.distances.capped_distance(
        positions_nm, positions_nm, cutoff_nm, 0, lattice
    )
print("MDAnalysis.lib.distances.capped_distance: ", (time() - start) / 2)

start = time()
for _ in range(10):
    tree = KDTree(positions_nm, boxsize=dimensions_nm)
    tree.query_pairs(cutoff_nm)
print("scipy.spatial.KDTree: ", (time() - start) / 10)

# # MDCraft
# mdcraft_nl = build_neighbor_list(positions_nm, cutoff_nm, dimensions_nm)
# mdcraft_pairs = []
# for pid in range(n_particles):
#     mdcraft_pairs.extend([(pid, j) for j in mdcraft_nl[pid]])
# mdcraft_pairs = np.array(mdcraft_pairs, dtype=np.uint32)
# mdcraft_pairs = mdcraft_pairs[np.lexsort(mdcraft_pairs.T[::-1])]

# # MDAnalysis
# mda_pairs = mda.lib.distances.capped_distance(
#     positions_nm,
#     positions_nm,
#     cutoff_nm,
#     0,
#     np.concatenate((dimensions_nm, (90.0, 90.0, 90.0))),
# )[0]
# mda_pairs = np.unique(np.sort(mda_pairs, axis=1), axis=0)

# # SciPy
# tree = KDTree(positions_nm, boxsize=dimensions_nm)
# scipy_pairs = np.asarray(tuple(tree.query_pairs(cutoff_nm)))
# scipy_pairs = scipy_pairs[np.lexsort(scipy_pairs.T[::-1])]

# assert np.array_equal(mdcraft_pairs, mda_pairs)
# assert np.array_equal(mdcraft_pairs, scipy_pairs)

debug = True
