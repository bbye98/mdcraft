import pathlib
import sys

import numpy as np
import pytest
import tidynamics

sys.path.insert(
    0, f"{pathlib.Path(__file__).parents[1].resolve().as_posix()}/src"
)
from mdcraft.algorithm import correlation


RNG = np.random.default_rng()


class TestFunctionCorrelation:

    @classmethod
    def setup_class(cls):
        # Randomly choose the shape of the time series data
        # (b)locks, (t)ime, (e)ntities, (d)imensions
        # cls.shape = np.array((*RNG.integers(2, 100, size=3), 3))
        cls.shape = np.array((100, 100, 100, 3))
        cls.shape_ccf = cls.shape.copy()
        cls.shape_ccf[1] = 2 * cls.shape_ccf[1] - 1

        # Generate time series data
        cls.ones = np.ones(cls.shape)
        cls.r1 = RNG.random(cls.shape)
        cls.r2 = RNG.random(cls.shape)

        # Calculate ACF solutions using tidynamics
        cls.acf_te = np.stack(
            [tidynamics.acf(v) for v in cls.r1[0, ..., 0].T]
        ).T
        cls.acf_t = cls.acf_te[:, 0]
        cls.acf_ted = np.stack(
            [tidynamics.acf(v) for v in np.swapaxes(cls.r1[0], 0, 1)]
        ).T
        cls.acf_td = cls.acf_ted[:, 0]
        cls.acf_bt = np.stack([tidynamics.acf(v) for v in cls.r1[..., 0, 0]])
        cls.acf_btd = np.stack([tidynamics.acf(v) for v in cls.r1[:, :, 0]])

        # Calculate CCF solutions using tidynamics
        cls.ccf_te = np.stack(
            [
                tidynamics.correlation(v1, v2)
                for v1, v2 in zip(cls.r1[0, ..., 0].T, cls.r2[0, ..., 0].T)
            ]
        ).T
        cls.ccf_t = cls.ccf_te[:, 0]
        cls.ccf_ted = np.stack(
            [
                tidynamics.correlation(v1, v2)
                for v1, v2 in zip(
                    np.swapaxes(cls.r1[0], 0, 1),
                    np.swapaxes(cls.r2[0], 0, 1),
                )
            ]
        ).T
        cls.ccf_td = cls.ccf_ted[:, 0]
        cls.ccf_bt = np.stack(
            [
                tidynamics.correlation(v1, v2)
                for v1, v2 in zip(cls.r1[..., 0, 0], cls.r2[..., 0, 0])
            ]
        )
        cls.ccf_btd = np.stack(
            [
                tidynamics.correlation(v1, v2)
                for v1, v2 in zip(cls.r1[:, :, 0], cls.r2[:, :, 0])
            ]
        )

        # Calculate symmetrize CCF solutions
        cls.symmetrized_ccf_ted = np.vstack(
            (
                2 * cls.ccf_ted[cls.shape[1] - 1],
                (
                    cls.ccf_ted[cls.shape[1] :]
                    + cls.ccf_ted[cls.shape[1] - 2 :: -1]
                ),
            )
        )
        cls.symmetrized_ccf_btd = np.hstack(
            (
                2 * cls.ccf_btd[:, cls.shape[1] - 1, None],
                (
                    cls.ccf_btd[:, cls.shape[1] :]
                    + cls.ccf_btd[:, cls.shape[1] - 2 :: -1]
                ),
            )
        )

    def test_acf_empty_1d(self):
        """
        Computes the ACF of an empty one-dimensional array.

        A ValueError should be raised.
        """
        with pytest.raises(ValueError):
            correlation.correlation(np.empty(0))

    def test_acf_empty_2d(self):
        """
        Computes the ACF of an empty two-dimensional array.

        A ValueError should be raised.
        """
        with pytest.raises(ValueError):
            correlation.correlation(np.empty((0, 0)))

    def test_acf_invalid_ndim(self):
        """
        Computes the ACF of a five-dimensional array with shape
        (1, 1, 1, 1, 1).

        A ValueError should be raised.
        """
        with pytest.raises(ValueError):
            correlation.correlation(np.empty((1, 1, 1, 1, 1)))

    def test_acf_invalid_axis(self):
        """
        Computes the ACF of a three-dimensional array with shape
        (1, 1, 1) along an invalid axis.

        A ValueError should be raised.
        """
        with pytest.raises(ValueError):
            correlation.correlation(np.empty((1, 1, 1)), axis=2)

    def test_acf_invalid_vector_ndim(self):
        """
        Computes the ACF of a time series of vectors, but a
        one-dimensional array with shape (1,) is provided.

        A ValueError should be raised.
        """
        with pytest.raises(ValueError):
            correlation.correlation(np.empty(1), vector=True)

    def test_acf_fft_ones_t(self):
        """
        Computes the ACF of a time series of ones with shape (N_t,)
        using FFTs.

        The expected result is an array of ones with the same shape.
        """
        assert (
            acf := correlation.correlation(self.ones[0, :, 0, 0])
        ).shape == self.shape[1] and np.allclose(acf, 1)

    def test_acf_fft_ones_te(self):
        """
        Computes the ACF of a time series of ones for multiple entities
        with shape (N_t, N_e) using FFTs.

        The expected result is an array of ones with the same shape.

        A UserWarning about the axis along which to compute the ACF
        should be raised.
        """
        with pytest.warns(UserWarning):
            assert np.allclose(
                (acf := correlation.correlation(self.ones[0, ..., 0])).shape,
                self.shape[1:3],
            ) and np.allclose(acf, 1)

    def test_acf_fft_ones_bt(self):
        """
        Computes the ACF of a segmented time series of ones with shape
        (N_b, N_t) using FFTs.

        The expected result is an array of ones with the same shape.
        """
        assert np.allclose(
            (
                acf := correlation.correlation(self.ones[..., 0, 0], axis=1)
            ).shape,
            self.shape[:2],
        ) and np.allclose(acf, 1)

    def test_acf_fft_ones_bte(self):
        """
        Computes the ACF of a segmented time series of ones for multiple
        entities with shape (N_b, N_t, N_e) using FFTs.

        The expected result is an array of ones with the same shape.
        """
        assert np.allclose(
            (acf := correlation.correlation(self.ones[..., 0], axis=1)).shape,
            self.shape[:3],
        ) and np.allclose(acf, 1)

    def test_acf_fft_ones_td(self):
        """
        Computes the ACF of a time series of one-vectors with shape
        (N_t, 3) using FFTs.

        The expected result is an array of threes with shape (N_t,).

        A UserWarning about the axis along which to compute the ACF should
        be raised.
        """
        with pytest.warns(UserWarning):
            assert np.allclose(
                (
                    acf := correlation.correlation(
                        self.ones[0, :, 0], vector=True
                    )
                ).shape,
                self.shape[1],
            ) and np.allclose(acf, 3)

    def test_acf_fft_ones_ted(self):
        """
        Computes the ACF of a time series of one-vectors for multiple
        entities with shape (N_t, N_e, 3) using FFTs.

        The expected result is an array of threes with shape (N_t, N_e).

        A UserWarning about the axis along which to compute the ACF
        should be raised.
        """

        with pytest.warns(UserWarning):
            assert np.allclose(
                (
                    acf := correlation.correlation(self.ones[0], vector=True)
                ).shape,
                self.shape[1:3],
            ) and np.allclose(acf, 3)

    def test_acf_fft_ones_btd(self):
        """
        Computes the ACF of a segmented time series of one-vectors with
        shape (N_b, N_t, 3).

        The expected result is an array of threes with shape (N_b, N_t).
        """
        assert np.allclose(
            (
                acf := correlation.correlation(
                    self.ones[:, :, 0], axis=1, vector=True
                )
            ).shape,
            self.shape[:2],
        ) and np.allclose(acf, 3)

    def test_acf_fft_ones_bted(self):
        """
        Computes the ACF of a segmented time series of one-vectors for
        multiple entities with shape (N_b, N_t, N_e, 3) using FFTs.

        The expected result is an array of threes with shape
        (N_b, N_t, N_e).
        """
        assert np.allclose(
            (acf := correlation.correlation(self.ones, vector=True)).shape,
            self.shape[:3],
        ) and np.allclose(acf, 3)

    def test_acf_fft_random_t(self):
        """
        Computes the ACF of a time series of random scalars with shape
        (N_t,) using FFTs.

        The expected result is an solution array with the same shape.
        """
        assert (
            acf := correlation.correlation(self.r1[0, :, 0, 0])
        ).shape == self.shape[1] and np.allclose(acf, self.acf_t)

    def test_acf_fft_random_te(self):
        """
        Computes the ACF of a time series of random scalars for multiple
        entities with shape (N_t, N_e) using FFTs.

        The expected result is a solution array of the same shape.
        """
        assert (
            np.allclose(
                (
                    acf := correlation.correlation(self.r1[0, ..., 0], axis=0)
                ).shape,
                self.shape[1:3],
            )
            and np.allclose(acf, self.acf_te)
            and np.allclose(
                correlation.correlation(
                    self.r1[0, ..., 0], average=True, axis=0
                ),
                acf.mean(axis=1),
            )
        )

    def test_acf_fft_random_bt(self):
        """
        Computes the ACF of a segmented time series of random scalars
        with shape (N_b, N_t) using FFTs.

        The expected result is a solution array of the same shape.
        """
        assert np.allclose(
            (acf := correlation.correlation(self.r1[..., 0, 0], axis=1)).shape,
            self.shape[:2],
        ) and np.allclose(acf, self.acf_bt)

    def test_acf_fft_random_bte(self):
        """
        Computes the ACF of a segmented time series of random scalars
        for multiple entities with shape (N_b, N_t, N_e) using FFTs.

        The expected result is a solution array of the same shape.
        """
        assert (
            np.allclose(
                (
                    acf := correlation.correlation(self.r1[..., 0], axis=1)
                ).shape,
                self.shape[:3],
            )
            and np.allclose(acf[0], self.acf_te)
            and np.allclose(acf[..., 0], self.acf_bt)
            and np.allclose(
                correlation.correlation(self.r1[..., 0], axis=1, average=True),
                acf.mean(axis=2),
            )
        )

    def test_acf_fft_random_td(self):
        """
        Computes the ACF of a time series of random vectors with shape
        (N_t, 3) using FFTs.

        The expected result is a solution array with shape (N_t,).
        """
        assert np.allclose(
            (
                acf := correlation.correlation(
                    self.r1[0, :, 0], axis=0, vector=True
                )
            ).shape,
            self.shape[1],
        ) and np.allclose(acf, self.acf_td)

    def test_acf_fft_random_ted(self):
        """
        Computes the ACF of a time series of random vectors for multiple
        entities with shape (N_t, N_e, 3) using FFTs.

        The expected result is a solution array with shape (N_t, N_e).
        """
        assert (
            np.allclose(
                (
                    acf := correlation.correlation(
                        self.r1[0], axis=0, vector=True
                    )
                ).shape,
                self.shape[1:3],
            )
            and np.allclose(acf, self.acf_ted)
            and np.allclose(
                correlation.correlation(
                    self.r1[0], axis=0, average=True, vector=True
                ),
                acf.mean(axis=1),
            )
        )

    def test_acf_fft_random_btd(self):
        """
        Computes the ACF of a segmented time series of random vectors
        with shape (N_b, N_t, 3) using FFTs.

        The expected result is a solution array with shape (N_b, N_t).
        """
        assert np.allclose(
            (
                acf := correlation.correlation(
                    self.r1[:, :, 0], axis=1, vector=True
                )
            ).shape,
            self.shape[:2],
        ) and np.allclose(acf, self.acf_btd)

    def test_acf_fft_random_bted(self):
        """
        Computes the ACF of a segmented time series of random vectors
        for multiple entities with shape (N_b, N_t, N_e, 3) using FFTs.

        The expected result is a solution array with shape
        (N_b, N_t, N_e).
        """
        assert (
            np.allclose(
                (acf := correlation.correlation(self.r1, vector=True)).shape,
                self.shape[:3],
            )
            and np.allclose(acf[0], self.acf_ted)
            and np.allclose(acf[..., 0], self.acf_btd)
        )

    def test_acf_shift_random_t(self):
        """
        Computes the ACF of a time series of random scalars with shape
        (N_t,) using sliding windows.

        The expected result is a solution array with the same shape.
        """
        assert (
            acf := correlation.correlation(self.r1[0, :, 0, 0], fft=False)
        ).shape == self.shape[1] and np.allclose(acf, self.acf_t)

    def test_acf_shift_random_te(self):
        """
        Computes the ACF of a time series of random scalars for multiple
        entities with shape (N_t, N_e) using sliding windows.

        The expected result is a solution array of the same shape.
        """
        assert (
            np.allclose(
                (
                    acf := correlation.correlation(
                        self.r1[0, ..., 0], axis=0, fft=False
                    )
                ).shape,
                self.shape[1:3],
            )
            and np.allclose(acf, self.acf_te)
            and np.allclose(
                correlation.correlation(
                    self.r1[0, ..., 0], average=True, axis=0, fft=False
                ),
                acf.mean(axis=1),
            )
        )

    def test_acf_shift_random_bt(self):
        """
        Computes the ACF of a segmented time series of random scalars
        with shape (N_b, N_t) using sliding windows.

        The expected result is a solution array of the same shape.
        """
        assert np.allclose(
            (
                acf := correlation.correlation(
                    self.r1[..., 0, 0], axis=1, fft=False
                )
            ).shape,
            self.shape[:2],
        ) and np.allclose(acf, self.acf_bt)

    def test_acf_shift_random_bte(self):
        """
        Computes the ACF of a segmented time series of random scalars
        for multiple entities with shape (N_b, N_t, N_e) using sliding
        windows.

        The expected result is a solution array of the same shape.
        """
        assert (
            np.allclose(
                (
                    acf := correlation.correlation(self.r1[..., 0], axis=1)
                ).shape,
                self.shape[:3],
            )
            and np.allclose(acf[0], self.acf_te)
            and np.allclose(acf[..., 0], self.acf_bt)
            and np.allclose(
                correlation.correlation(
                    self.r1[..., 0], axis=1, average=True, fft=False
                ),
                acf.mean(axis=2),
            )
        )

    def test_acf_shift_random_td(self):
        """
        Computes the ACF of a time series of random vectors with shape
        (N_t, 3) using sliding windows.

        The expected result is a solution array with shape (N_t,).
        """
        assert (
            acf := correlation.correlation(
                self.r1[0, :, 0], axis=0, fft=False, vector=True
            )
        ).shape == self.shape[1] and np.allclose(acf, self.acf_td)

    def test_acf_shift_random_ted(self):
        """
        Computes the ACF of a time series of random vectors for multiple
        entities with shape (N_t, N_e, 3) using sliding windows.

        The expected result is a solution array with shape (N_t, N_e).
        """
        assert (
            np.allclose(
                (
                    acf := correlation.correlation(
                        self.r1[0], axis=0, fft=False, vector=True
                    )
                ).shape,
                self.shape[1:3],
            )
            and np.allclose(acf, self.acf_ted)
            and np.allclose(
                correlation.correlation(
                    self.r1[0], axis=0, average=True, fft=False, vector=True
                ),
                acf.mean(axis=1),
            )
        )

    def test_acf_shift_random_btd(self):
        """
        Computes the ACF of a segmented time series of random vectors
        with shape (N_b, N_t, 3) using sliding windows.

        The expected result is a solution array with shape (N_b, N_t).
        """
        assert np.allclose(
            (
                acf := correlation.correlation(
                    self.r1[:, :, 0], axis=1, fft=False, vector=True
                )
            ).shape,
            self.shape[:2],
        ) and np.allclose(acf, self.acf_btd)

    def test_acf_shift_random_bted(self):
        """
        Computes the ACF of a segmented time series of random vectors
        for multiple entities with shape (N_b, N_t, N_e, 3) using
        sliding windows.

        The expected result is a solution array with shape
        (N_b, N_t, N_e).
        """
        assert (
            np.allclose(
                (
                    acf := correlation.correlation(
                        self.r1, fft=False, vector=True
                    )
                ).shape,
                self.shape[:3],
            )
            and np.allclose(acf[0], self.acf_ted)
            and np.allclose(acf[..., 0], self.acf_btd)
        )

    def test_doubled_acf_fft_random_bted(self):
        """
        Computes the doubled ACF of a segmented time series of random
        vectors for multiple entities with shape (N_b, N_t, N_e, 3)
        using FFTs.

        The expected result is a solution array with shape
        (N_b, N_t, N_e).
        """
        assert (
            np.allclose(
                (
                    acf := correlation.correlation(
                        self.r1,
                        symmetrize=True,
                        vector=True,
                    )
                    / 2
                ).shape,
                self.shape[:3],
            )
            and np.allclose(acf[0], self.acf_ted)
            and np.allclose(acf[..., 0], self.acf_btd)
        )

    def test_doubled_acf_shift_random_bted(self):
        """
        Computes the doubled ACF of a segmented time series of random
        vectors for multiple entities with shape (N_b, N_t, N_e, 3)
        using sliding windows.

        The expected result is a solution array with shape
        (N_b, N_t, N_e).
        """
        assert (
            np.allclose(
                (
                    acf := correlation.correlation(
                        self.r1,
                        symmetrize=True,
                        fft=False,
                        vector=True,
                    )
                    / 2
                ).shape,
                self.shape[:3],
            )
            and np.allclose(acf[0], self.acf_ted)
            and np.allclose(acf[..., 0], self.acf_btd)
        )

    def test_ccf_asymmetric_arrays(self):
        """
        Computes the CCF of two asymmetric arrays with different shapes.

        A ValueError should be raised.
        """
        with pytest.raises(ValueError):
            correlation.correlation(np.empty(1), np.empty(2))

    def test_ccf_fft_random_t(self):
        """
        Computes the CCF of two time series of random scalars with shape
        (N_t,) using FFTs.

        The expected result is a solution array with the same shape.
        """
        assert (
            ccf := correlation.correlation(
                self.r1[0, :, 0, 0], self.r2[0, :, 0, 0]
            )
        ).shape == self.shape_ccf[1] and np.allclose(ccf, self.ccf_t)

    def test_ccf_fft_random_te(self):
        """
        Computes the CCF of two time series of random scalars for
        multiple entities with shape (N_t, N_e) using FFTs.

        The expected result is a solution array of the same shape.
        """
        assert (
            np.allclose(
                (
                    ccf := correlation.correlation(
                        self.r1[0, ..., 0], self.r2[0, ..., 0], axis=0
                    )
                ).shape,
                self.shape_ccf[1:3],
            )
            and np.allclose(ccf, self.ccf_te)
            and np.allclose(
                correlation.correlation(
                    self.r1[0, ..., 0],
                    self.r2[0, ..., 0],
                    average=True,
                    axis=0,
                ),
                ccf.mean(axis=1),
            )
        )

    def test_ccf_fft_random_bt(self):
        """
        Computes the CCF of two segmented time series of random scalars
        with shape (N_b, N_t) using FFTs.

        The expected result is a solution array of the same shape.
        """
        assert np.allclose(
            (
                ccf := correlation.correlation(
                    self.r1[..., 0, 0], self.r2[..., 0, 0], axis=1
                )
            ).shape,
            self.shape_ccf[:2],
        ) and np.allclose(ccf, self.ccf_bt)

    def test_ccf_fft_random_bte(self):
        """
        Computes the CCF of two segmented time series of random scalars
        for multiple entities with shape (N_b, N_t, N_e) using FFTs.

        The expected result is a solution array of the same shape.
        """
        assert (
            np.allclose(
                (
                    ccf := correlation.correlation(
                        self.r1[..., 0], self.r2[..., 0], axis=1
                    )
                ).shape,
                self.shape_ccf[:3],
            )
            and np.allclose(ccf[0], self.ccf_te)
            and np.allclose(ccf[..., 0], self.ccf_bt)
            and np.allclose(
                correlation.correlation(
                    self.r1[..., 0], self.r2[..., 0], axis=1, average=True
                ),
                ccf.mean(axis=2),
            )
        )

    def test_ccf_fft_random_td(self):
        """
        Computes the CCF of two time series of random vectors with shape
        (N_t, 3) using FFTs.

        The expected result is a solution array with shape (N_t,).
        """
        assert (
            ccf := correlation.correlation(
                self.r1[0, :, 0], self.r2[0, :, 0], axis=0, vector=True
            )
        ).shape == self.shape_ccf[1] and np.allclose(ccf, self.ccf_td)

    def test_ccf_fft_random_ted(self):
        """
        Computes the CCF of two time series of random vectors for
        multiple entities with shape (N_t, N_e, 3) using FFTs.

        The expected result is a solution array with shape (N_t, N_e).
        """
        assert (
            np.allclose(
                (
                    ccf := correlation.correlation(
                        self.r1[0], self.r2[0], axis=0, vector=True
                    )
                ).shape,
                self.shape_ccf[1:3],
            )
            and np.allclose(ccf, self.ccf_ted)
            and np.allclose(
                correlation.correlation(
                    self.r1[0], self.r2[0], axis=0, average=True, vector=True
                ),
                ccf.mean(axis=1),
            )
        )

    def test_ccf_fft_random_btd(self):
        """
        Computes the CCF of two segmented time series of random vectors
        with shape (N_b, N_t, 3) using FFTs.

        The expected result is a solution array with shape (N_b, N_t).
        """
        assert np.allclose(
            (
                ccf := correlation.correlation(
                    self.r1[:, :, 0], self.r2[:, :, 0], axis=1, vector=True
                )
            ).shape,
            self.shape_ccf[:2],
        ) and np.allclose(ccf, self.ccf_btd)

    def test_ccf_fft_random_bted(self):
        """
        Computes the CCF of two segmented time series of random vectors
        for multiple entities with shape (N_b, N_t, N_e, 3) using FFTs.

        The expected result is a solution array with shape
        (N_b, N_t, N_e).
        """
        assert (
            np.allclose(
                (
                    ccf := correlation.correlation(
                        self.r1, self.r2, vector=True
                    )
                ).shape,
                self.shape_ccf[:3],
            )
            and np.allclose(ccf[0], self.ccf_ted)
            and np.allclose(ccf[..., 0], self.ccf_btd)
        )

    def test_ccf_shift_random_t(self):
        """
        Computes the CCF of two time series of random scalars with shape
        (N_t,) using sliding windows.

        The expected result is a solution array with the same shape.
        """
        assert (
            ccf := correlation.correlation(
                self.r1[0, :, 0, 0], self.r2[0, :, 0, 0], fft=False
            )
        ).shape == self.shape_ccf[1] and np.allclose(ccf, self.ccf_t)

    def test_ccf_shift_random_te(self):
        """
        Computes the CCF of two time series of random scalars for
        multiple entities with shape (N_t, N_e) using sliding windows.

        The expected result is a solution array of the same shape.
        """
        assert (
            np.allclose(
                (
                    ccf := correlation.correlation(
                        self.r1[0, ..., 0], self.r2[0, ..., 0], axis=0
                    )
                ).shape,
                self.shape_ccf[1:3],
            )
            and np.allclose(ccf, self.ccf_te)
            and np.allclose(
                correlation.correlation(
                    self.r1[0, ..., 0],
                    self.r2[0, ..., 0],
                    average=True,
                    axis=0,
                    fft=False,
                ),
                ccf.mean(axis=1),
            )
        )

    def test_ccf_shift_random_bt(self):
        """
        Computes the CCF of two segmented time series of random scalars
        with shape (N_b, N_t) using sliding windows.

        The expected result is a solution array of the same shape.
        """
        assert np.allclose(
            (
                ccf := correlation.correlation(
                    self.r1[..., 0, 0], self.r2[..., 0, 0], axis=1, fft=False
                )
            ).shape,
            self.shape_ccf[:2],
        ) and np.allclose(ccf, self.ccf_bt)

    def test_ccf_shift_random_bte(self):
        """
        Computes the CCF of two segmented time series of random scalars
        for multiple entities with shape (N_b, N_t, N_e) using sliding
        windows.

        The expected result is a solution array of the same shape.
        """
        assert (
            np.allclose(
                (
                    ccf := correlation.correlation(
                        self.r1[..., 0], self.r2[..., 0], axis=1, fft=False
                    )
                ).shape,
                self.shape_ccf[:3],
            )
            and np.allclose(ccf[0], self.ccf_te)
            and np.allclose(ccf[..., 0], self.ccf_bt)
            and np.allclose(
                correlation.correlation(
                    self.r1[..., 0], self.r2[..., 0], axis=1, average=True
                ),
                ccf.mean(axis=2),
            )
        )

    def test_ccf_shift_random_td(self):
        """
        Computes the CCF of two time series of random vectors with shape
        (N_t, 3) using sliding windows.

        The expected result is a solution array with shape (N_t,).
        """
        assert (
            ccf := correlation.correlation(
                self.r1[0, :, 0],
                self.r2[0, :, 0],
                axis=0,
                fft=False,
                vector=True,
            )
        ).shape == self.shape_ccf[1] and np.allclose(ccf, self.ccf_td)

    def test_ccf_shift_random_ted(self):
        """
        Computes the CCF of two time series of random vectors for
        multiple entities with shape (N_t, N_e, 3) using sliding
        windows.

        The expected result is a solution array with shape (N_t, N_e).
        """
        assert (
            np.allclose(
                (
                    ccf := correlation.correlation(
                        self.r1[0], self.r2[0], axis=0, fft=False, vector=True
                    )
                ).shape,
                self.shape_ccf[1:3],
            )
            and np.allclose(ccf, self.ccf_ted)
            and np.allclose(
                correlation.correlation(
                    self.r1[0],
                    self.r2[0],
                    axis=0,
                    average=True,
                    fft=False,
                    vector=True,
                ),
                ccf.mean(axis=1),
            )
        )

    def test_ccf_shift_random_btd(self):
        """
        Computes the CCF of two segmented time series of random vectors
        with shape (N_b, N_t, 3) using sliding windows.

        The expected result is a solution array with shape (N_b, N_t).
        """
        assert np.allclose(
            (
                ccf := correlation.correlation(
                    self.r1[:, :, 0],
                    self.r2[:, :, 0],
                    axis=1,
                    fft=False,
                    vector=True,
                )
            ).shape,
            self.shape_ccf[:2],
        ) and np.allclose(ccf, self.ccf_btd)

    def test_ccf_shift_random_bted(self):
        """
        Computes the CCF of two segmented time series of random vectors
        for multiple entities with shape (N_b, N_t, N_e, 3) using
        sliding windows.

        The expected result is a solution array with shape
        (N_b, N_t, N_e).
        """
        assert (
            np.allclose(
                (
                    ccf := correlation.correlation(
                        self.r1, self.r2, fft=False, vector=True
                    )
                ).shape,
                self.shape_ccf[:3],
            )
            and np.allclose(ccf[0], self.ccf_ted)
            and np.allclose(ccf[..., 0], self.ccf_btd)
        )

    def test_symmetrized_ccf_fft_random_bted(self):
        """
        Computes the symmetrized CCF of two segmented time series of
        random vectors for multiple entities with shape
        (N_b, N_t, N_e, 3) using FFTs.

        The expected result is a solution array with shape
        (N_b, N_t, N_e).
        """
        assert (
            np.allclose(
                (
                    ccf := correlation.correlation(
                        self.r1,
                        self.r2,
                        symmetrize=True,
                        vector=True,
                    )
                ).shape,
                self.shape[:3],
            )
            and np.allclose(ccf[0], self.symmetrized_ccf_ted)
            and np.allclose(ccf[..., 0], self.symmetrized_ccf_btd)
        )

    def test_symmetrized_ccf_shift_random_bted(self):
        """
        Computes the symmetrized CCF of two segmented time series of
        random vectors for multiple entities with shape
        (N_b, N_t, N_e, 3) using sliding windows.

        The expected result is a solution array with shape
        (N_b, N_t, N_e).
        """
        assert (
            np.allclose(
                (
                    ccf := correlation.correlation(
                        self.r1,
                        self.r2,
                        symmetrize=True,
                        fft=False,
                        vector=True,
                    )
                ).shape,
                self.shape[:3],
            )
            and np.allclose(ccf[0], self.symmetrized_ccf_ted)
            and np.allclose(ccf[..., 0], self.symmetrized_ccf_btd)
        )
