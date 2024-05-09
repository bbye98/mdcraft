import pathlib
import sys

import numpy as np
import pytest
import tidynamics

sys.path.insert(0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src")
from mdcraft.algorithm import correlation  # noqa: E402

# Generate random times series
rng = np.random.default_rng()
shape = (*rng.integers(2, 50, size=3), 3)
vec_ones = np.ones(shape)
vec_random_1 = rng.random(shape)
vec_random_2 = rng.random(shape)

# Calculate reference ACFs and CCFs for the random time series using
# tidynamics
acf_scalar = tidynamics.acf(vec_random_1[0, :, 0, 0])
acf_multi_scalar = np.stack(
    [tidynamics.acf(v) for v in vec_random_1[0, :, :, 0].T]
).T
acf_block_scalar = np.stack(
    [tidynamics.acf(v) for v in vec_random_1[:, :, 0, 0]]
)
acf_vector = tidynamics.acf(vec_random_1[0, :, 0])
acf_multi_vector = np.stack(
    [tidynamics.acf(v) for v in np.swapaxes(vec_random_1[0], 0, 1)]
).T
acf_block_vector = np.stack(
    [tidynamics.acf(v) for v in vec_random_1[:, :, 0]]
)
ccf_scalar = tidynamics.correlation(vec_random_1[0, :, 0, 0],
                                    vec_random_2[0, :, 0, 0])
ccf_multi_scalar = np.stack(
    [tidynamics.correlation(v1, v2)
     for v1, v2 in zip(vec_random_1[0, :, :, 0].T,
                       vec_random_2[0, :, :, 0].T)]
).T
ccf_block_scalar = np.stack(
    [tidynamics.correlation(v1, v2)
     for v1, v2 in zip(vec_random_1[:, :, 0, 0],
                       vec_random_2[:, :, 0, 0])]
)
ccf_vector = tidynamics.correlation(vec_random_1[0, :, 0],
                                    vec_random_2[0, :, 0])
ccf_multi_vector = np.stack(
    [tidynamics.correlation(v1, v2)
     for v1, v2 in zip(np.swapaxes(vec_random_1[0], 0, 1),
                       np.swapaxes(vec_random_2[0], 0, 1))]
).T
ccf_block_vector = np.stack(
    [tidynamics.correlation(v1, v2)
     for v1, v2 in zip(vec_random_1[:, :, 0],
                       vec_random_2[:, :, 0])]
)

def test_func_correlation_fft():

    # TEST CASE 1: ACF of empty 1D array; raises ValueError
    with pytest.raises(ValueError):
        correlation.correlation_fft(np.empty(0))

    # TEST CASE 2: ACF of empty 2D array; raises ValueError
    with pytest.raises(ValueError):
        correlation.correlation_fft(np.empty((0, 3)))

    # TEST CASE 3: ACF of 5D array; raises ValueError
    with pytest.raises(ValueError):
        correlation.correlation_fft(np.empty((2, 2, 2, 2, 2)))

    # TEST CASE 4: ACF of 3D array along invalid axis; raises ValueError
    with pytest.raises(ValueError):
        correlation.correlation_fft(np.empty((2, 2, 2)), axis=2)

    # TEST CASE 5: ACF of time series of ones
    assert np.allclose(correlation.correlation_fft(vec_ones[0, :, 0, 0]), 1)

    # TEST CASE 6: ACF of time series of ones for multiple entities
    with pytest.warns(UserWarning):
        assert np.allclose(
            correlation.correlation_fft(vec_ones[0, :, :, 0]),
            1
        )

    # TEST CASE 7: ACF of blocked time series of ones
    with pytest.warns(UserWarning):
        assert np.allclose(
            correlation.correlation_fft(vec_ones[:, :, 0, 0]),
            1
        )

    # TEST CASE 8: ACF of blocked time series of ones for multiple entities
    with pytest.warns(UserWarning):
        assert np.allclose(
            correlation.correlation_fft(vec_ones[:, :, :, 0]),
            1
        )

    # TEST CASE 9: ACF of time series of 1-vectors
    with pytest.warns(UserWarning):
        assert np.allclose(
            correlation.correlation_fft(vec_ones[0, :, 0], vector=True),
            shape[-1]
        )

    # TEST CASE 10: ACF of time series of 1-vectors for multiple entities
    with pytest.warns(UserWarning):
        assert np.allclose(
            correlation.correlation_fft(vec_ones[0], vector=True),
            shape[-1]
        )

    # TEST CASE 11: ACF of blocked time series of 1-vectors
    with pytest.warns(UserWarning):
        assert np.allclose(
            correlation.correlation_fft(vec_ones[:, :, 0], vector=True),
            shape[-1]
        )

    # TEST CASE 12: ACF of blocked time series of 1-vectors for multiple
    # entities
    assert np.allclose(correlation.correlation_fft(vec_ones, vector=True),
                       shape[-1])

    # TEST CASE 13: ACF of time series of random scalars
    assert np.allclose(correlation.correlation_fft(vec_random_1[0, :, 0, 0]),
                       acf_scalar)

    # TEST CASE 14: ACF of time series of random scalars for multiple
    # entities
    acf = correlation.correlation_fft(vec_random_1[0, :, :, 0], axis=0)
    assert np.allclose(acf, acf_multi_scalar)
    assert np.allclose(
        correlation.correlation_fft(vec_random_1[0, :, :, 0], average=True,
                                    axis=0),
        acf.mean(axis=1)
    )

    # TEST CASE 15: ACF of blocked time series of random scalars
    assert np.allclose(
        correlation.correlation_fft(vec_random_1[:, :, 0, 0], axis=1),
        acf_block_scalar
    )

    # TEST CASE 16: ACF of blocked time series of random scalars for
    # multiple entities
    acf = correlation.correlation_fft(vec_random_1[:, :, :, 0], axis=1)
    assert np.allclose(acf[0], acf_multi_scalar)
    assert np.allclose(acf[:, :, 0], acf_block_scalar)
    assert np.allclose(
        correlation.correlation_fft(vec_random_1[:, :, :, 0], average=True,
                                    axis=1),
        acf.mean(axis=2)
    )

    # TEST CASE 17: ACF of time series of random vectors
    assert np.allclose(
        correlation.correlation_fft(vec_random_1[0, :, 0], axis=0, vector=True),
        acf_vector
    )

    # TEST CASE 18: ACF of time series of random vectors for multiple
    # entities
    assert np.allclose(
        correlation.correlation_fft(vec_random_1[0], axis=0, vector=True),
        acf_multi_vector
    )

    # TEST CASE 19: ACF of blocked time series of random vectors
    assert np.allclose(
        correlation.correlation_fft(vec_random_1[:, :, 0], axis=1, vector=True),
        acf_block_vector
    )

    # TEST CASE 20: ACF of blocked time series of random vectors for
    # multiple entities
    acf = correlation.correlation_fft(vec_random_1, vector=True)
    assert np.allclose(acf[0], acf_multi_vector)
    assert np.allclose(acf[:, :, 0], acf_block_vector)

    # TEST CASE 21: CCF of empty 1D arrays
    with pytest.raises(ValueError):
        correlation.correlation_fft(np.empty(0), np.empty(0))

    # TEST CASE 22: CCF of empty 2D array
    with pytest.raises(ValueError):
        correlation.correlation_fft(np.empty((0, 3)), np.empty((0, 3)))

    # TEST CASE 23: CCF of 5D arrays
    with pytest.raises(ValueError):
        correlation.correlation_fft(np.empty((2, 2, 2, 2, 2)),
                                    np.empty((2, 2, 2, 2, 2)))

    # TEST CASE 24: CCF of asymmetric 2D arrays
    with pytest.raises(ValueError):
        correlation.correlation_fft(np.empty((2, 3)), np.empty((3, 2)))

    # TEST CASE 25: CCF of 3D arrays along invalid axis
    with pytest.raises(ValueError):
        correlation.correlation_fft(np.empty((2, 2, 2)), np.empty((2, 2, 2)),
                                    axis=2)

    # TEST CASE 26: CCF of time series of random scalars
    assert np.allclose(
        correlation.correlation_fft(vec_random_1[0, :, 0, 0].tolist(),
                                    vec_random_2[0, :, 0, 0].tolist()),
        ccf_scalar
    )

    # TEST CASE 27: CCF of time series of random scalars for multiple
    # entities
    assert np.allclose(
        correlation.correlation_fft(vec_random_1[0, :, :, 0],
                                    vec_random_2[0, :, :, 0], axis=0),
        ccf_multi_scalar
    )

    # TEST CASE 28: CCF of blocked time series of random scalars
    assert np.allclose(
        correlation.correlation_fft(vec_random_1[:, :, 0, 0],
                                    vec_random_2[:, :, 0, 0], axis=1),
        ccf_block_scalar
    )

    # TEST CASE 29: CCF of blocked time series of random scalars for
    # multiple entities
    ccf = correlation.correlation_fft(vec_random_1[:, :, :, 0],
                                      vec_random_2[:, :, :, 0], axis=1)
    assert np.allclose(ccf[0], ccf_multi_scalar)
    assert np.allclose(ccf[:, :, 0], ccf_block_scalar)

    # TEST CASE 30: CCF of time series of random vectors
    assert np.allclose(
        correlation.correlation_fft(vec_random_1[0, :, 0],
                                    vec_random_2[0, :, 0],
                                    axis=0, vector=True),
        ccf_vector
    )

    # TEST CASE 31: CCF of time series of random vectors for multiple
    # entities
    assert np.allclose(
        correlation.correlation_fft(vec_random_1[0], vec_random_2[0],
                                    axis=0, vector=True),
        ccf_multi_vector
    )

    # TEST CASE 32: CCF of blocked time series of random vectors
    assert np.allclose(
        correlation.correlation_fft(vec_random_1[:, :, 0],
                                    vec_random_2[:, :, 0],
                                    axis=1, vector=True),
        ccf_block_vector
    )

    # TEST CASE 33: CCF of blocked time series of random vectors for
    # multiple entities
    ccf = correlation.correlation_fft(vec_random_1, vec_random_2,
                                      vector=True)
    assert np.allclose(ccf[0], ccf_multi_vector)
    assert np.allclose(ccf[:, :, 0], ccf_block_vector)

def test_func_correlation_shift():

    # TEST CASE 1: ACF of empty 1D array
    with pytest.raises(ValueError):
        correlation.correlation_shift(np.empty(0))

    # TEST CASE 2: ACF of empty 2D array
    with pytest.raises(ValueError):
        correlation.correlation_shift(np.empty((0, 3)))

    # TEST CASE 3: ACF of 5D array
    with pytest.raises(ValueError):
        correlation.correlation_shift(np.empty((2, 2, 2, 2, 2)))

    # TEST CASE 4: ACF of 3D array along invalid axis
    with pytest.raises(ValueError):
        correlation.correlation_shift(np.empty((2, 2, 2)), axis=2)

    # TEST CASE 5: ACF of time series of random scalars
    assert np.allclose(correlation.correlation_shift(vec_random_1[0, :, 0, 0]),
                       acf_scalar)

    # TEST CASE 6: ACF of time series of random scalars for multiple
    # entities
    with pytest.warns(UserWarning):
        acf = correlation.correlation_shift(vec_random_1[0, :, :, 0])
    assert np.allclose(acf, acf_multi_scalar)
    assert np.allclose(
        correlation.correlation_shift(vec_random_1[0, :, :, 0],
                                      average=True, axis=0),
        acf.mean(axis=1)
    )

    # TEST CASE 7: ACF of blocked time series of random scalars
    assert np.allclose(
        correlation.correlation_shift(vec_random_1[:, :, 0, 0], axis=1),
        acf_block_scalar
    )

    # TEST CASE 8: ACF of blocked time series of random scalars for
    # multiple entities
    acf = correlation.correlation_shift(vec_random_1[:, :, :, 0], axis=1)
    assert np.allclose(acf[0], acf_multi_scalar)
    assert np.allclose(acf[:, :, 0], acf_block_scalar)
    assert np.allclose(
        correlation.correlation_shift(vec_random_1[:, :, :, 0],
                                      average=True, axis=1),
        acf.mean(axis=2)
    )

    # TEST CASE 9: ACF of time series of random vectors
    assert np.allclose(
        correlation.correlation_shift(vec_random_1[0, :, 0], axis=0,
                                      vector=True),
        acf_vector
    )

    # TEST CASE 10: ACF of time series of random vectors for multiple
    # entities
    assert np.allclose(
        correlation.correlation_shift(vec_random_1[0], axis=0,
                                      vector=True),
        acf_multi_vector
    )

    # TEST CASE 11: ACF of blocked time series of random vectors
    assert np.allclose(
        correlation.correlation_shift(vec_random_1[:, :, 0], axis=1,
                                      vector=True),
        acf_block_vector
    )

    # TEST CASE 12: ACF of blocked time series of random vectors for
    # multiple entities
    acf = correlation.correlation_shift(vec_random_1, vector=True)
    assert np.allclose(acf[0], acf_multi_vector)
    assert np.allclose(acf[:, :, 0], acf_block_vector)

    # TEST CASE 13: CCF of empty 1D arrays
    with pytest.raises(ValueError):
        correlation.correlation_shift(np.empty(0), np.empty(0))

    # TEST CASE 14: CCF of empty 2D array
    with pytest.raises(ValueError):
        correlation.correlation_shift(np.empty((0, 3)), np.empty((0, 3)))

    # TEST CASE 15: CCF of 5D arrays
    with pytest.raises(ValueError):
        correlation.correlation_shift(np.empty((2, 2, 2, 2, 2)),
                                      np.empty((2, 2, 2, 2, 2)))

    # TEST CASE 16: CCF of asymmetric 2D arrays
    with pytest.raises(ValueError):
        correlation.correlation_shift(np.empty((2, 3)), np.empty((3, 2)))

    # TEST CASE 17: CCF of 3D arrays along invalid axis
    with pytest.raises(ValueError):
        correlation.correlation_shift(np.empty((2, 2, 2)), np.empty((2, 2, 2)),
                                      axis=2)

    # TEST CASE 18: CCF of time series of random scalars
    assert np.allclose(
        correlation.correlation_shift(vec_random_1[0, :, 0, 0].tolist(),
                                      vec_random_2[0, :, 0, 0].tolist()),
        ccf_scalar
    )

    # TEST CASE 19: CCF of time series of random scalars for multiple
    # entities
    assert np.allclose(
        correlation.correlation_shift(vec_random_1[0, :, :, 0],
                                      vec_random_2[0, :, :, 0], axis=0),
        ccf_multi_scalar
    )

    # TEST CASE 20: CCF of blocked time series of random scalars
    assert np.allclose(
        correlation.correlation_shift(vec_random_1[:, :, 0, 0],
                                      vec_random_2[:, :, 0, 0], axis=1),
        ccf_block_scalar
    )

    # TEST CASE 21: CCF of blocked time series of random scalars for
    # multiple entities
    ccf = correlation.correlation_shift(vec_random_1[:, :, :, 0],
                                        vec_random_2[:, :, :, 0], axis=1)
    assert np.allclose(ccf[0], ccf_multi_scalar)
    assert np.allclose(ccf[:, :, 0], ccf_block_scalar)

    # TEST CASE 22: CCF of time series of random vectors
    assert np.allclose(
        correlation.correlation_shift(vec_random_1[0, :, 0],
                                      vec_random_2[0, :, 0],
                                      axis=0, vector=True),
        ccf_vector
    )

    # TEST CASE 23: CCF of time series of random vectors for multiple
    # entities
    assert np.allclose(
        correlation.correlation_shift(vec_random_1[0], vec_random_2[0],
                                      axis=0, vector=True),
        ccf_multi_vector
    )

    # TEST CASE 24: CCF of blocked time series of random vectors
    assert np.allclose(
        correlation.correlation_shift(vec_random_1[:, :, 0],
                                      vec_random_2[:, :, 0],
                                      axis=1, vector=True),
        ccf_block_vector
    )

    # TEST CASE 25: CCF of blocked time series of random vectors for
    # multiple entities
    ccf = correlation.correlation_shift(vec_random_1, vec_random_2,
                                        vector=True)
    assert np.allclose(ccf[0], ccf_multi_vector)
    assert np.allclose(ccf[:, :, 0], ccf_block_vector)

    # TEST CASE 26: Doubled ACF for time series of random scalars
    assert np.allclose(
        correlation.correlation_shift(vec_random_1[0, :, 0, 0], double=True),
        correlation.correlation_shift(vec_random_1[0, :, 0, 0], double=True)
    )

    # TEST CASE 27: Overlapped CCF for time series of random scalars
    assert np.allclose(
        correlation.correlation_shift(vec_random_1[0, :, 0, 0],
                                      vec_random_2[0, :, 0, 0],
                                      axis=0, double=True),
        correlation.correlation_fft(vec_random_1[0, :, 0, 0],
                                    vec_random_2[0, :, 0, 0],
                                    axis=0, double=True)
    )

    # TEST CASE 28: Overlapped CCF for blocked time series of random scalars
    assert np.allclose(
        correlation.correlation_shift(vec_random_1[:, :, 0, 0],
                                      vec_random_2[:, :, 0, 0],
                                      axis=1, double=True),
        correlation.correlation_fft(vec_random_1[:, :, 0, 0],
                                    vec_random_2[:, :, 0, 0],
                                    axis=1, double=True)
    )

# Generate simple trajectories and calculate their MSDs and CD
traj_1 = np.array(((0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)))
msd_1 = np.einsum("td,td->t", traj_1, traj_1)
traj_2 = np.array(((0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 8)))
msd_2 = ((traj_2 - traj_2[0]) ** 2).sum(axis=1)
cd = (traj_1 * (traj_2 - traj_2[0])).sum(axis=1)

def test_func_msd_fft():

    # TEST CASE 1: MSD of empty trajectory
    with pytest.raises(ValueError):
        correlation.msd_fft(np.empty(0))

    # TEST CASE 2: MSD of 5D trajectory
    with pytest.raises(ValueError):
        correlation.msd_fft(np.empty((2, 2, 2, 2, 2)))

    # TEST CASE 3: CD of asymmetric trajectories
    with pytest.raises(ValueError):
        correlation.msd_fft(traj_1, traj_2[:1])

    # TEST CASE 4: MSD of 3D array along invalid axis
    with pytest.raises(ValueError):
        correlation.msd_fft(np.empty((2, 2, 2)), axis=2)

    # TEST CASE 5: MSD of first simple trajectory
    assert np.allclose(correlation.msd_fft(traj_1.tolist()), msd_1)

    # TEST CASE 6: MSD of second simple trajectory
    assert np.allclose(correlation.msd_fft(traj_2), msd_2)

    # TEST CASE 7: CD of simple trajectories
    assert np.allclose(correlation.msd_fft(traj_1, traj_2.tolist()), cd)

    # TEST CASE 8: MSD of first simple trajectory replicated for multiple
    # particles
    with pytest.warns(UserWarning):
        assert np.allclose(
            correlation.msd_fft(
                np.tile(traj_1[:, None], (1, 2, 1)),
                average=False
            )[:, 0],
            msd_1
        )

    # TEST CASE 9: MSD of first simple trajectory replicated for multiple
    # blocks and particles
    assert np.allclose(
        correlation.msd_fft(
            np.tile(traj_1[None, :, None], (2, 1, 2, 1)),
            average=False
        )[0, :, 0],
        msd_1
    )

    # TEST CASE 10: CD of simple trajectories replicated for multiple
    # blocks and particles
    assert np.allclose(
        correlation.msd_fft(
            np.tile(traj_1[None, :, None], (2, 1, 2, 1)),
            np.tile(traj_2[None, :, None], (2, 1, 2, 1))
        )[0],
        cd
    )

def test_func_msd_shift():

    # TEST CASE 1: MSD of empty trajectory
    with pytest.raises(ValueError):
        correlation.msd_shift(np.empty(0))

    # TEST CASE 2: MSD of 5D trajectory
    with pytest.raises(ValueError):
        correlation.msd_shift(np.empty((2, 2, 2, 2, 2)))

    # TEST CASE 3: CD of asymmetric trajectories
    with pytest.raises(ValueError):
        correlation.msd_shift(traj_1, traj_2[:1])

    # TEST CASE 4: MSD of 3D array along invalid axis
    with pytest.raises(ValueError):
        correlation.msd_shift(np.empty((2, 2, 2)), axis=2)

    # TEST CASE 5: MSD of first simple trajectory
    assert np.allclose(correlation.msd_shift(traj_1.tolist()), msd_1)

    # TEST CASE 6: MSD of second simple trajectory
    assert np.allclose(correlation.msd_shift(traj_2), msd_2)

    # TEST CASE 7: CD of simple trajectories
    assert np.allclose(correlation.msd_shift(traj_1, traj_2.tolist()), cd)

    # TEST CASE 8: MSD of first simple trajectory replicated for multiple
    # particles
    with pytest.warns(UserWarning):
        assert np.allclose(
            correlation.msd_shift(
                np.tile(traj_1[:, None], (1, 2, 1)),
                average=False
            )[:, 0],
            msd_1
        )

    # TEST CASE 9: MSD of first simple trajectory replicated for multiple
    # blocks and particles
    assert np.allclose(
        correlation.msd_shift(
            np.tile(traj_1[None, :, None], (2, 1, 2, 1)),
            average=False
        )[0, :, 0],
        msd_1
    )

    # TEST CASE 10: CD of simple trajectories replicated for multiple
    # blocks and particles
    assert np.allclose(
        correlation.msd_shift(
            np.tile(traj_1[None, :, None], (2, 1, 2, 1)),
            np.tile(traj_2[None, :, None], (2, 1, 2, 1))
        )[0],
        cd
    )