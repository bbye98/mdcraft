import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft.algorithm import utility # noqa: E402

rng = np.random.default_rng()

def test_func_closest_factors():

    # TEST CASE 1: Cube root of perfect cube
    factors = utility.get_closest_factors(1000, 3)
    assert np.allclose(factors, 10 * np.ones(3, dtype=int))

    # TEST CASE 2: Three closest factors in ascending order
    factors = utility.get_closest_factors(35904, 3)
    assert factors.tolist() == [32, 33, 34]

    # TEST CASE 3: Four closest factors in descending order
    factors = utility.get_closest_factors(73440, 4, reverse=True)
    assert factors.tolist() == [18, 17, 16, 15]

def test_func_replicate():

    # TEST CASE 1: Replicate two vectors
    dims = rng.integers(1, 5, size=3)
    n_cells = rng.integers(2, 10, size=3)
    pos = utility.replicate(dims, np.array(((0, 0, 0), dims // 2)), n_cells)
    assert pos.shape[0] == 2 * n_cells.prod()
    assert np.allclose(pos[2], (dims[0], 0, 0))

def test_func_rebin():

    # TEST CASE 1: Rebin 1D array
    arr = np.arange(50)
    ref = np.arange(2, 52, 5)
    assert np.allclose(utility.rebin(arr), ref)

    # TEST CASE 2: Rebin 2D array
    assert np.allclose(utility.rebin(np.tile(arr[None, :], (5, 1))),
                       np.tile(ref[None, :], (5, 1)))

    # TEST CASE 3: No factor specified and cannot be determined
    with pytest.raises(ValueError):
        utility.rebin(np.empty((17,)))