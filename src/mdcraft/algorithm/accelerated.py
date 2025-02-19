"""
Accelerated algorithms
======================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains miscellaneous Numba-accelerated common algorithms.
"""

import numba
import numpy as np


@numba.njit(fastmath=True)
def numba_histogram_bin_edges(
    array: np.ndarray[float], n_bins: int
) -> np.ndarray[float]:
    r"""
    Serial Numba-accelerated function to compute the uniform histogram
    bin edges for a one-dimensional NumPy array :math:`\mathbf{a}`
    and a specified number of bins :math:`N_\mathrm{bins}`.

    Parameters
    ----------
    array : `np.ndarray`
        One-dimensional array :math:`\mathbf{a}`.

    n_bins : `int`
        Number of bins :math:`N_\mathrm{bins}`.

    Returns
    -------
    bin_edges : `np.ndarray`
        Uniform histogram bin edges for the array :math:`\mathbf{a}`.
    """

    min_, max_ = array.min(), array.max()
    n_edges = n_bins + 1
    bin_edges = np.empty(n_edges)
    delta = (max_ - min_) / n_bins
    for i in range(n_edges):
        bin_edges[i] = min_ + i * delta
    bin_edges[-1] = max_
    return bin_edges


@numba.njit(fastmath=True)
def numba_histogram(
    array: np.ndarray[float], n_bins: int, bin_edges: np.ndarray[float]
) -> np.ndarray[int]:
    r"""
    Serial Numba-accelerated function to compute the histogram of a
    one-dimensional NumPy array :math:`\mathbf{a}` using predetermined
    bin edges for :math:`N_\mathrm{bins}` bins.

    Parameters
    ----------
    array : `np.ndarray`
        One-dimensional array :math:`\mathbf{a}`.

    n_bins : `int`
        Number of bins :math:`N_\mathrm{bins}`.

    bin_edges : `np.ndarray`
        Bin edges.

    Returns
    -------
    histogram_ : `np.ndarray`
        Histogram of the array :math:`\mathbf{a}`.
    """

    min_, max_ = bin_edges[0], bin_edges[-1]
    histogram_ = np.zeros(n_bins, dtype=np.intp)
    for x in array:
        if x == max_:
            bin_ = n_bins - 1
        else:
            bin_ = int(n_bins * (x - min_) / (max_ - min_))
        if 0 <= bin_ < n_bins:
            histogram_[bin_] += 1
    return histogram_


@numba.njit(fastmath=True)
def numba_dot(a: np.ndarray[float], b: np.ndarray[float]) -> float:
    r"""
    Serial Numba-accelerated dot product between two one-dimensional
    NumPy arrays :math:`\mathbf{a}` and :math:`\mathbf{b}`, each with
    shape :math:`(3,)`.

    .. math::

       \mathbf{a}\cdot\mathbf{b}=a_1b_1+a_2b_2+a_3b_3

    Parameters
    ----------
    a : `np.ndarray`
        First vector :math:`\mathbf{a}`.

        **Shape**: :math:`(3,)`.

    b : `np.ndarray`
        Second vector :math:`\mathbf{b}`.

        **Shape**: :math:`(3,)`.

    Returns
    -------
    ab : `float`
        Dot product :math:`\mathbf{a}\cdot\mathbf{b}`.
    """

    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@numba.njit(fastmath=True)
def numba_inner(qs: np.ndarray[float], rs: np.ndarray[float]) -> np.ndarray[float]:
    r"""
    Serial Numba-accelerated inner product between all possible
    combinations of multiple one-dimensional NumPy arrays
    :math:`\mathbf{q}` and :math:`\mathbf{r}`, each with shape
    :math:`(3,)`.

    .. math::

       \mathbf{q}_i\cdot\mathbf{r}_j
       =q_{i1}r_{j1}+q_{i2}r_{j2}+q_{i3}r_{j3}

    Parameters
    ----------
    qs : `np.ndarray`
        Multiple vectors :math:`\mathbf{q}`.

        **Shape**: :math:`(N_q,\,3)`.

    rs : `np.ndarray`
        Multiple vectors :math:`\mathbf{r}`.

        **Shape**: :math:`(N_r,\,3)`.

    Returns
    -------
    s : `np.ndarray`
        Inner products of the vectors,
        :math:`\mathbf{q}_i\cdot\mathbf{r}_j`.

        **Shape**: :math:`(N_q,\,N_r)`.
    """

    s = np.empty((qs.shape[0], rs.shape[0]))
    for i in range(qs.shape[0]):
        for j in range(rs.shape[0]):
            s[i, j] = numba_dot(qs[i], rs[j])
    return s


@numba.njit(fastmath=True, parallel=True)
def numba_inner_parallel(
    qs: np.ndarray[float], rs: np.ndarray[float]
) -> np.ndarray[float]:
    r"""
    Parallel Numba-accelerated inner product between all possible
    combinations of multiple one-dimensional NumPy arrays
    :math:`\mathbf{q}` and :math:`\mathbf{r}`, each with shape
    :math:`(3,)`.

    .. math::

       \mathbf{q}_i\cdot\mathbf{r}_j
       =q_{i1}r_{j1}+q_{i2}r_{j2}+q_{i3}r_{j3}

    Parameters
    ----------
    qs : `np.ndarray`
        Multiple vectors :math:`\mathbf{q}`.

        **Shape**: :math:`(N_q,\,3)`.

    rs : `np.ndarray`
        Multiple vectors :math:`\mathbf{r}`.

        **Shape**: :math:`(N_r,\,3)`.

    Returns
    -------
    s : `np.ndarray`
        Inner products of the vectors,
        :math:`\mathbf{q}_i\cdot\mathbf{r}_j`.

        **Shape**: :math:`(N_q,\,N_r)`.
    """

    s = np.empty((qs.shape[0], rs.shape[0]))
    for i in numba.prange(qs.shape[0]):
        for j in range(rs.shape[0]):
            s[i, j] = numba_dot(qs[i], rs[j])
    return s
