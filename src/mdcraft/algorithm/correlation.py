"""
Correlation functions
=====================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains algorithms for evaluating the spatial and temporal
correlation between data sets, such as the autocorrelation and
cross-correlation functions and the closely related mean squared
displacement.
"""

from typing import Union
import warnings

import numpy as np
from scipy import fft

def correlation_fft(
        x: np.ndarray[Union[float, complex]],
        y: np.ndarray[Union[float, complex]] = None, /, axis: int = None, *,
        average: bool = False, double: bool = False, vector: bool = False
    ) -> np.ndarray[Union[float, complex]]:

    r"""
    Evaluates the autocorrelation functions (ACF)
    :math:`\mathrm{R_\mathbf{XX}}(\tau)` or cross-correlation functions
    (CCF) :math:`\mathrm{R_\mathbf{XY}}(\tau)` of time series
    :math:`\mathbf{X}(t)` and :math:`\mathbf{Y}(t)` using fast Fourier
    transforms (FFT).

    The fast convolution algorithm (FCA) [1]_ [2]_ is associated wtih
    the Wiener–Khinchin theorem and has a time complexity of
    :math:`\mathcal{O}(N\log{N})`.

    The ACF for a data set :math:`\mathbf{X}(t)` can be computed using

    .. math::

       \begin{gather*}
         \hat{\mathbf{X}}(\omega)=\mathcal{F}\{\mathbf{X}(t)\}\\
         \mathrm{R}_{\mathbf{XX}}(\tau)=\mathcal{F}^{-1}
         \{\hat{\mathbf{X}}(\omega)\hat{\mathbf{X}}^*(\omega)\}
       \end{gather*}

    where :math:`\tau` is the time lag and the asterisk (:math:`^*`)
    denotes the complex conjugate.

    Similarly, the CCF for data sets :math:`\mathbf{X}(t)` and
    :math:`\mathbf{Y}(t)` can be computed using

    .. math::

       \mathrm{R}_{\mathbf{XY}}(\tau)=\mathcal{F}^{-1}(\mathcal{F}
       (\mathbf{X}(t))\cdot\mathcal{F}(\mathbf{Y}(t)))

    Parameters
    ----------
    x : `numpy.ndarray`, positional-only
        Time evolution of :math:`d`-dimensional data for :math:`N`
        entities over :math:`N_\mathrm{b}` blocks of :math:`N_t` times
        each.

        .. container::

           **Shape**:

           * Scalar: :math:`(N_t,)`, :math:`(N_t,\,N)`,
             :math:`(N_\mathrm{b},\,N_t)`, or
             :math:`(N_\mathrm{b},\,N_t,\,N)`.
           * Vector: :math:`(N_t,\,d)`, :math:`(N_t,\,N,\,d)`,
             :math:`(N_\mathrm{b},\,N_t,\,d)`, or
             :math:`(N_\mathrm{b},\,N_t,\,N,\,d)`.

    y : `numpy.ndarray`, positional-only, optional
        Time evolution of :math:`d`-dimensional data for another
        :math:`N` entities over :math:`N_\mathrm{b}` blocks of
        :math:`N_t` times each. If provided, the CCF for `x` and `y` is
        calculated. Otherwise, the ACF for `x` is calculated.

        **Shape**: Same as `x`.

    axis : `int`, optional
        Axis along which time evolves. If not specified, the axis is
        determined automatically using the shape of `x`.

    average : `bool`, keyword-only, default: :code:`True`
        Determines whether the ACF/CCF is averaged over all entities.
        Only available if `x` and `y` contain information for multiple
        entities.

    double : `bool`, keyword-only, default: :code:`False`
        Determines whether the ACF is doubled or the negative and
        positive time lags are combined for the CCF.

    vector : `bool`, keyword-only, default: :code:`False`
        Specifies whether `x` and `y` contain vectors. If :code:`True`,
        the ACF/CCF is summed over the last axis.

    Returns
    -------
    corr : `numpy.ndarray`
        ACF or CCF.

        .. container::

           **Shape**:

           For ACF, the shape is that of `x` but with the following
           modifications:

           * If :code:`average=True`, the axis containing the :math:`N`
             entities is removed.
           * If :code:`vector=True`, the last axis is removed.

           For CCF, the shape is that of `x` but with the following
           modifications:

           * If :code:`average=True`, the axis containing the :math:`N`
             entities is removed.
           * If :code:`double=False`, the axis containing the
             :math:`N_t` times now has a length of :math:`2N_t-1` to
             accomodate negative and positive time lags.
           * If :code:`vector=True`, the last axis is removed.

    References
    ----------
    .. [1] Kneller, G. R.; Keiner, V.; Kneller, M.; Schiller, M.
       NMOLDYN: A Program Package for a Neutron Scattering Oriented
       Analysis of Molecular Dynamics Simulations. *Computer Physics
       Communications* **1995**, *91* (1–3), 191–214.
       https://doi.org/10.1016/0010-4655(95)00048-K.

    .. [2] Calandrini, V.; Pellegrini, E.; Calligari, P.; Hinsen, K.;
       Kneller, G. R. NMoldyn - Interfacing Spectroscopic Experiments,
       Molecular Dynamics Simulations and Models for Time Correlation
       Functions. *JDN* **2011**, *12*, 201–232.
       https://doi.org/10.1051/sfn/201112010.
    """

    # Ensure arrays have valid shapes
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("The arrays cannot be empty.")
    ndim = x.ndim
    if not 1 <= ndim <= 4:
        emsg = ("The arrays must be one-, two-, three-, or four-"
                "dimensional.")
        raise ValueError(emsg)
    if y is not None:
        y = np.asarray(y)
        if x.shape != y.shape:
            raise ValueError("The arrays must have the same shape.")

    # Check or set axis along which to compute the ACF/CCF
    if axis is None:
        if ndim == 4:
            axis = 1
        else:
            axis = 0
            if ndim > 1:
                wmsg = ("The axis along which to compute the ACF/CCF "
                        "was not specified and is ambiguous for a "
                        "multidimensional array. As such, it has been "
                        "set to the first axis by default.")
                warnings.warn(wmsg)
    elif axis not in {0, 1}:
        emsg = ("The ACF/CCF can only be computed along the first or "
                "second axis.")
        raise ValueError(emsg)

    # Determine whether faster real-valued FFTs can be used
    if (real := np.isrealobj(x) and (y is None or np.isrealobj(y))):
        fft_ = fft.rfft
        ifft = fft.irfft
    else:
        fft_ = fft.fft
        ifft = fft.ifft

    # Compute the power spectral density by first zero-padding the
    # arrays for linear convolution and then inverting it to get the
    # ACF/CCF
    N_t = x.shape[axis]
    n = 2 * fft.next_fast_len(N_t, real=real)
    if y is None:
        f = fft_(x, n=n, axis=axis)
        r = ifft(f * f.conj(), axis=axis)
        r = (double + 1) * (r[:, :N_t] if axis else r[:N_t])
    else:
        fx = fft_(x, n=n, axis=axis)
        fy = fft_(y, n=n, axis=axis)
        f = fx.conj() * fy
        if double:
            r = ifft(f + fx * fy.conj(), axis=axis)
            r = r[:, :N_t] if axis else r[:N_t]
        else:
            r = ifft(f, axis=axis)

    # Sum over the last axis if the arrays contain vectors
    if vector:
        r = r.sum(axis=-1)

    # Determine the axes over which to expand the dimensions of the
    # reversed time array for correct matrix division
    axes = list(range(ndim - vector))
    axes.remove(axis)

    # Normalize the ACF/CCF
    if axis:
        r[:, :N_t] /= np.expand_dims(np.arange(N_t, 0, -1), axes)
        if r.shape[axis] != N_t:
            r[:, 1 - N_t:] /= np.expand_dims(np.arange(1, N_t), axes)
            r = np.hstack((r[:, 1 - N_t:], r[:, :N_t]))
    else:
        r[:N_t] /= np.expand_dims(np.arange(N_t, 0, -1), axes)
        if r.shape[axis] != N_t:
            r[1 - N_t:] /= np.expand_dims(np.arange(1, N_t), axes)
            r = np.concatenate((r[1 - N_t:], r[:N_t]))

    # Average over all entities, if desired
    if average:
        axis_avg = ndim - vector - 1
        if axis != axis_avg:
            return r.mean(axis=axis_avg)

    return r

def correlation_shift(
        x: np.ndarray[Union[float, complex]],
        y: np.ndarray[Union[float, complex]] = None, /, axis: int = None, *,
        average: bool = False, double: bool = False, vector: bool = False
    ) -> np.ndarray[Union[float, complex]]:

    r"""
    Evaluates the autocorrelation functions (ACF)
    :math:`\mathrm{R_\mathbf{XX}}(\tau)` or cross-correlation functions
    (CCF) :math:`\mathrm{R_\mathbf{XY}}(\tau)` of time series
    :math:`\mathbf{X}(t)` and :math:`\mathbf{Y}(t)` directly by using
    sliding windows.

    The ACF for a data set :math:`\mathbf{X}(t)` can be computed using

    .. math::

       \mathrm{R}_{\mathbf{XX}}(\tau)
       =\langle\mathbf{X}(t_0+\tau)\cdot\mathbf{X}^*(t_0)\rangle
       =\dfrac{1}{N_\tau}\sum_{j=1}^{N_\tau}
       \textbf{X}(t_j+\tau)\cdot\textbf{X}^*(t_j)

    where :math:`\tau` is the time lag, :math:`t_j` is an arbitrary
    reference time, :math:`N_\tau` is the number of possible reference
    times, and the asterisk (:math:`^*`) denotes the complex conjugate.

    Similarly, the CCF for data sets :math:`\mathbf{X}(t)` and
    :math:`\mathbf{Y}(t)` can be computed using

    .. math::

       \mathrm{R}_{\mathbf{XY}}(\tau)
       =\langle\mathbf{X}(t_0+\tau)\cdot\mathbf{Y}^*(t_0)\rangle
       =\dfrac{1}{N_\tau}\sum_{j=1}^{N_\tau}
       \textbf{X}(t_j+\tau)\cdot\textbf{Y}^*(t_j)

    To minimize statistical noise, the ACF/CCF is calculated for and
    averaged over  all possible reference times :math:`t_0`. As such,
    this algorithm has a time complexity of :math:`\mathcal{O}(N^2)`.
    With large data sets, this approach is too slow to be useful. If
    your machine supports fast Fourier transforms (FFT), use the much
    more performant FFT-based algorithm implemented in
    :func:`mdcraft.algorithm.correlation.correlation_fft` instead.

    Parameters
    ----------
    x : `numpy.ndarray`, positional-only
        Time evolution of :math:`d`-dimensional data for :math:`N`
        entities over :math:`N_\mathrm{b}` blocks of :math:`N_t` times
        each.

        .. container::

           **Shape**:

           * Scalar: :math:`(N_t,)`, :math:`(N_t,\,N)`,
             :math:`(N_\mathrm{b},\,N_t)`, or
             :math:`(N_\mathrm{b},\,N_t,\,N)`.
           * Vector: :math:`(N_t,\,d)`, :math:`(N_t,\,N,\,d)`,
             :math:`(N_\mathrm{b},\,N_t,\,d)`, or
             :math:`(N_\mathrm{b},\,N_t,\,N,\,d)`.

    y : `numpy.ndarray`, positional-only, optional
        Time evolution of :math:`d`-dimensional data for another
        :math:`N` entities over :math:`N_\mathrm{b}` blocks of
        :math:`N_t` times each. If provided, the CCF for `x` and `y` is
        calculated. Otherwise, the ACF for `x` is calculated.

        **Shape**: Same as `x`.

    axis : `int`, optional
        Axis along which time evolves. If not specified, the axis is
        determined automatically using the shape of `x`.

    average : `bool`, keyword-only, default: :code:`True`
        Determines whether the ACF/CCF is averaged over all entities.
        Only available if `x` and `y` contain information for multiple
        entities.

    double : `bool`, keyword-only, default: :code:`False`
        Determines whether the ACF is doubled or the negative and
        positive time lags are combined for the CCF.

    vector : `bool`, keyword-only, default: :code:`False`
        Specifies whether `x` and `y` contain vectors. If :code:`True`,
        the ACF/CCF is summed over the last axis.

    Returns
    -------
    corr : `numpy.ndarray`
        ACF or CCF.

        .. container::

           **Shape**:

           For ACF, the shape is that of `x` but with the following
           modifications:

           * If :code:`average=True`, the axis containing the :math:`N`
             entities is removed.
           * If :code:`vector=True`, the last axis is removed.

           For CCF, the shape is that of `x` but with the following
           modifications:

           * If :code:`average=True`, the axis containing the :math:`N`
             entities is removed.
           * If :code:`double=False`, the axis containing the
             :math:`N_t` times now has a length of :math:`2N_t-1` to
             accomodate negative and positive time lags.
           * If :code:`vector=True`, the last axis is removed.
    """

    # Ensure arrays have valid shapes
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("The arrays cannot be empty.")
    ndim = x.ndim
    if not 1 <= ndim <= 4:
        emsg = ("The arrays must be one-, two-, three-, or four-"
                "dimensional.")
        raise ValueError(emsg)
    if y is not None:
        y = np.asarray(y)
        if x.shape != y.shape:
            raise ValueError("The arrays must have the same shape.")

    # Check or set axis along which to compute the ACF/CCF
    if axis is None:
        if ndim == 4:
            axis = 1
        else:
            axis = 0
            if ndim > 1:
                wmsg = ("The axis along which to compute the ACF/CCF "
                        "was not specified and is ambiguous for a "
                        "multidimensional array. As such, it has been "
                        "set to the first axis by default.")
                warnings.warn(wmsg)
    elif axis not in {0, 1}:
        emsg = ("The ACF/CCF can only be computed along the first or "
                "second axis.")
        raise ValueError(emsg)

    # Calculate the ACF/CCF
    N_t = x.shape[axis]
    if y is None:
        if ndim == 1:
            r = np.fromiter(
                (np.dot(x[i:], x[:-i if i else None]) for i in range(N_t)),
                dtype=float,
                count=N_t,
            )
        elif axis:
            axes = f"bt...{'d' * vector}"
            r = np.stack(
                [np.einsum(f"{axes},{axes}->b...",
                           x[:, i:], x[:, :-i if i else None])
                 for i in range(N_t)],
                axis=1
            )
        else:
            axes = f"t...{'d' * vector}"
            r = np.stack(
                [np.einsum(f"{axes},{axes}->...", x[i:], x[:-i if i else None])
                 for i in range(N_t)]
            )
    else:
        start = np.r_[np.zeros(N_t - 1, dtype=int), 0:N_t]
        stop = np.r_[1:N_t + 1, N_t * np.ones(N_t - 1, dtype=int)]
        if ndim == 1:
            r = np.fromiter(
                (np.dot(x[i:j], y[k:m])
                 for i, j, k, m in zip(start[::-1], stop[::-1], start, stop)),
                dtype=float,
                count=2 * N_t - 1
            )
        elif axis:
            axes = f"bt...{'d' * vector}"
            r = np.stack(
                [np.einsum(f"{axes},{axes}->b...", x[:, i:j], y[:, k:m])
                 for i, j, k, m in zip(start[::-1], stop[::-1], start, stop)],
                axis=1
            )
        else:
            axes = f"t...{'d' * vector}"
            r = np.stack(
                [np.einsum(f"{axes},{axes}->...", x[i:j], y[k:m])
                 for i, j, k, m in zip(start[::-1], stop[::-1], start, stop)]
            )

    # Double the ACF or overlap the negative and positive time lags for
    # the CCF, if desired
    if double:
        if y is None:
            r *= 2
        elif axis:
            r = r[:, N_t - 1:] + r[:, N_t - 1::-1]
        else:
            r = r[N_t - 1:] + r[N_t - 1::-1]

    # Determine the axes over which to expand the dimensions of the
    # reversed time array for correct matrix division
    axes = list(range(ndim - vector))
    axes.remove(axis)

    # Normalize the ACF/CCF
    if axis:
        r[:, -N_t:] /= np.expand_dims(np.arange(N_t, 0, -1), axes)
        if r.shape[axis] != N_t:
            r[:, :N_t - 1] /= np.expand_dims(np.arange(1, N_t), axes)
    else:
        r[-N_t:] /= np.expand_dims(np.arange(N_t, 0, -1), axes)
        if r.shape[axis] != N_t:
            r[:N_t - 1] /= np.expand_dims(np.arange(1, N_t), axes)

    # Average over all entities, if desired
    if average:
        axis_avg = ndim - 1 - vector
        if axis != axis_avg:
            return r.mean(axis=axis_avg)

    return r

def msd_fft(
        r_i: np.ndarray[float], r_j: np.ndarray[float] = None, /,
        axis: int = None, *, average: bool = True) -> np.ndarray[float]:

    r"""
    Evaluates the mean squared displacements (MSD) or the analogous
    cross displacements (CD) of positions :math:`\mathbf{r}_i(t)` and
    :math:`\mathbf{r}_j(t)` using fast Fourier transforms (FFT).

    For a set of positions :math:`\mathbf{r}_i(t)`, the MSD is computed
    using the algorithm [1]_ [2]_

    .. math::

       \mathrm{MSD}_{i,\,m}&=\frac{1}{N_t-m}\sum_{k=0}^{N_t-m-1}
       [\textbf{r}_{i,\,k+m}-\textbf{r}_{i,\,k}]^2\\
       &=\frac{1}{N_t-m}\sum_{k=0}^{N_t-m-1}
       \left[\textbf{r}_{i,\,k+m}^2+\textbf{r}_{i,\,k}^2\right]
       -\frac{2}{N_t-m}\sum_{k=0}^{N_t-m-1}
       \textbf{r}_{i,\,k}\cdot\textbf{r}_{i,\,k+m}\\
       &=S_{ii,\,m}-2\mathrm{R}_{ii,\,m}

    where :math:`i` is the species index, :math:`m` is the index
    corresponding to time lag :math:`\tau`, :math:`\mathrm{R}_{ii,\,m}`
    is the autocorrelation of :math:`\mathbf{r}_i(t)`, and :math:`S_m`
    is evaluated using the recursive relation

    .. math::

       \begin{gather*}
         D_{ii,\,m}=\textbf{r}_{i,\,m}^2\\
         Q_{ii,\,-1}=2\sum_{k=0}^{N_t-1}D_{ii,\,k}\\
         Q_{ii,\,m}=Q_{ii,\,m-1}-D_{ii,\,m-1}-D_{ii,\,N_t-m}\\
         S_{ii,\,m}=\frac{Q_{ii,\,m}}{N_t-m}
       \end{gather*}

    Similarly, the CD for two sets of positions :math:`\mathbf{r}_i(t)`
    and :math:`\mathbf{r}_j(t)` is computed using

    .. math::

       \mathrm{CD}_{ij,m}=S_{ij,\,m}-2\mathrm{R}_{ij,\,m}

    where :math:`\mathrm{R}_{ij,\,m}` is the cross-correlation of
    :math:`\mathbf{r}_i(t)` and :math:`\mathbf{r}_j(t)`, and
    :math:`S_{ij,\,m}` is evaluated using the recursive relation

    .. math::

       \begin{gather*}
         D_{ij,\,m}=\textbf{r}_{i,\,m}\cdot\textbf{r}_{j,\,m}\\
         Q_{ij,\,-1}=2\sum_{k=0}^{N_t-1}D_{ij,\,k}\\
         Q_{ij,\,m}=Q_{ij,\,m-1}-D_{ij,\,m-1}-D_{ij,\,N_t-m}\\
         S_{ij,\,m}=\frac{Q_{ij,\,m}}{N_t-m}
       \end{gather*}

    .. note::

       To evaluate the sum in the expression used to calculate the
       Onsager transport coefficients [3]_

       .. math::

          L_{ij}=\frac{1}{6k_\mathrm{B}T}\lim_{t\rightarrow\infty}
          \frac{d}{d\tau}\left\langle
          \sum_{\alpha=1}^{N_i}[\mathbf{r}_{i,\,\alpha}(t_0+\tau)
          -\mathbf{r}_{i,\,\alpha}(t_0)]\cdot
          \sum_{\beta=1}^{N_j}[\mathbf{r}_{j,\,\beta}(t_0+\tau)
          -\mathbf{r}_{j,\,\beta(t_0)}]\right\rangle

       `r_i` and `r_j` should be summed over all entities before being
       passed to this function.

    Parameters
    ----------
    r_i : `numpy.ndarray`, positional-only
        Time evolution of individual or averaged positions for :math:`N`
        entities over :math:`N_\mathrm{b}` blocks of :math:`N_t` times
        each.

        **Shape**: :math:`(N_t,\,3)`, :math:`(N_t,\,N,\,3)`,
        :math:`(N_\mathrm{b},\,N_t,\,3)`, or
        :math:`(N_\mathrm{b},\,N_t,\,N,\,3)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    r_j : `numpy.ndarray`, positional-only, optional
        Time evolution of individual or averaged positions for another
        :math:`N` entities over :math:`N_\mathrm{b}` blocks of
        :math:`N_t` times each.

        **Shape**: Same as `r_i`.

        **Reference unit**: :math:`\mathrm{Å}`.

    axis : `int`, optional
        Axis along which time evolves. If not specified, the axis is
        determined automatically using the shape of `r_i`.

    average : `bool`, keyword-only, default: :code:`True`
        Determines whether the MSD/CD is averaged over all entities.
        Only available if `r_i` and `r_j` contain information for
        multiple entities.

    Returns
    -------
    disp : `numpy.ndarray`
        MSD or CD.

        **Shape**: Same as the shape of `r_i`, except with the last axis
        removed. If :code:`average=True`, the axis containing the
        :math:`N` entities is also removed.

        **Reference unit**: :math:`\text{Å}^2`.

    References
    ----------
    .. [1] Kneller, G. R.; Keiner, V.; Kneller, M.; Schiller, M.
       NMOLDYN: A Program Package for a Neutron Scattering Oriented
       Analysis of Molecular Dynamics Simulations. *Computer Physics
       Communications* **1995**, *91* (1–3), 191–214.
       https://doi.org/10.1016/0010-4655(95)00048-K.

    .. [2] Calandrini, V.; Pellegrini, E.; Calligari, P.; Hinsen, K.;
       Kneller, G. R. NMoldyn - Interfacing Spectroscopic Experiments,
       Molecular Dynamics Simulations and Models for Time Correlation
       Functions. *JDN* **2011**, *12*, 201–232.
       https://doi.org/10.1051/sfn/201112010.

    .. [3] Fong, K. D.; Self, J.; McCloskey, B. D.; Persson, K. A.
       Onsager Transport Coefficients and Transference Numbers in
       Polyelectrolyte Solutions and Polymerized Ionic Liquids.
       *Macromolecules* **2020**, *53* (21), 9503–9512.
       https://doi.org/10.1021/acs.macromol.0c02001.
    """

    # Ensure arrays have valid shapes
    r_i = np.asarray(r_i)
    if r_i.size == 0:
        raise ValueError("The position arrays cannot be empty.")
    if r_i.shape[-1] != 3:
        emsg = ("The position arrays must have three components in "
                "the last axis.")
        raise ValueError(emsg)
    ndim = r_i.ndim
    if not 2 <= ndim <= 4:
        emsg = ("The position arrays must be two-, three-, or four-"
                "dimensional.")
        raise ValueError(emsg)
    if r_j is not None:
        r_j = np.asarray(r_j)
        if r_i.shape != r_j.shape:
            raise ValueError("The position arrays must have the same shape.")

    # Check or set axis along which to compute the MSD/CD
    if axis is None:
        if ndim == 4:
            axis = 1
        else:
            axis = 0
            if ndim == 3:
                emsg = ("The axis along which to compute the MSD/CD "
                        "was not specified and is ambiguous for a "
                        "three-dimensional array. As such, it has been "
                        "set to the first axis by default.")
                warnings.warn(emsg)
    elif axis not in {0, 1}:
        emsg = ("The MSD/CD can only be computed along the first or "
                "second axis.")
        raise ValueError(emsg)

    # Get intermediate quantities required for the MSD/CD calculation
    R_ij = correlation_fft(r_i, r_j, axis, average=False, double=True,
                           vector=True)
    D_ij = (r_i * (r_i if r_j is None else r_j)).sum(axis=-1)

    N_t = r_i.shape[axis]
    if ndim - axis == 3:

        # Calculate the MSD/CD for each entity
        if not average:
            stack = np.vstack if ndim == 3 else np.hstack
            shape = np.asarray(r_i.shape[:-1])
            mask = np.ones_like(shape, dtype=bool)
            mask[axis] = False
            D_k = stack((D_ij, np.expand_dims(np.zeros(shape[mask]), axis)))
            if axis:
                Q_ij = (
                    2 * D_k.sum(axis=axis, keepdims=True) * np.ones((1, N_t, 1))
                    - np.cumsum(
                        D_k[:, np.arange(-1, N_t - 1)] + D_k[:, N_t:0:-1],
                        axis=axis
                    )
                )
            else:
                Q_ij = (
                    2 * D_k.sum(axis=axis) * np.ones((N_t, 1))
                    - np.cumsum(D_k[np.arange(-1, N_t - 1)] + D_k[N_t:0:-1],
                                axis=axis)
                )
            return Q_ij / np.arange(N_t, 0, -1)[:, None] - R_ij

        # Average the intermediate quantities over all particles
        R_ij = R_ij.mean(axis=ndim - 2)
        D_ij = D_ij.mean(axis=ndim - 2)

    # Calculate the averaged MSD/CD
    if axis:
        Q_ij = (
            2 * D_ij.sum(axis=axis, keepdims=True) * np.ones((1, N_t))
            - np.insert(
                np.cumsum(D_ij[:, :N_t - 1] + D_ij[:, N_t - 1:0:-1], axis=axis),
                0,
                0,
                axis=axis
            )
        )
    else:
        Q_ij = (
            2 * D_ij.sum() * np.ones(N_t)
            - np.insert(np.cumsum(D_ij[:N_t - 1] + D_ij[N_t - 1:0:-1]), 0, 0)
        )
    return Q_ij / np.arange(N_t, 0, -1) - R_ij

def msd_shift(
        r_i: np.ndarray[float], r_j: np.ndarray[float] = None, /,
        axis: int = None, *, average: bool = True) -> np.ndarray[float]:

    r"""
    Evaluates the mean squared displacements (MSD) or the analogous
    cross displacements (CD) of positions :math:`\mathbf{r}_i(t)` and
    :math:`\mathbf{r}_j(t)` using the Einstein relation.

    For a set of positions :math:`\mathbf{r}_i(t)`, the MSD is defined
    as

    .. math::

        \mathrm{MSD}_i(\tau)=\left\langle[\textbf{r}_i(t_0+\tau)
        -\textbf{r}_i(t_0)]^2\right\rangle
        =\dfrac{1}{N_\tau}\sum_{k=1}^{N_\tau}[\textbf{r}_i(t_k+\tau)
        -\textbf{r}_i(t_k)]^2

    where :math:`\tau` is the time lag, :math:`t_j` is an arbitrary
    reference time, and :math:`N_\tau` is the number of possible
    reference times.

    Similarly, the CD for two sets of positions :math:`\mathbf{r}_i(t)`
    and :math:`\mathbf{r}_j(t)` is defined as

    .. math::

       \mathrm{CD}_{ij}(\tau)&=\langle
       [\textbf{r}_i(t_0+\tau)-\textbf{r}_i(t_0)]\cdot
       [\textbf{r}_j(t_0+\tau)-\textbf{r}_j(t_0)]\rangle\\
       &=\dfrac{1}{N_\tau}\sum_{k=1}^{N_\tau}
       [\textbf{r}_i(t_k+\tau)-\textbf{r}_i(t_k))\cdot
       (\textbf{r}_j(t_k+\tau)-\textbf{r}_j(t_k)]

    To minimize statistical noise, the MSD/CD is calculated for and
    averaged over all possible reference times :math:`t_0`.

    .. note::

       To evaluate the sum in the expression used to calculate the
       Onsager transport coefficients [1]_

       .. math::

          L_{ij}=\frac{1}{6k_\mathrm{B}T}\lim_{t\rightarrow\infty}
          \frac{d}{d\tau}\left\langle
          \sum_{\alpha=1}^{N_i}[\mathbf{r}_{i,\,\alpha}(t_0+\tau)
          -\mathbf{r}_{i,\,\alpha}(t_0)]\cdot
          \sum_{\beta=1}^{N_j}[\mathbf{r}_{j,\,\beta}(t_0+\tau)
          -\mathbf{r}_{j,\,\beta(t_0)}]\right\rangle

       `r_i` and `r_j` should be summed over all entities before being
       passed to this function.

    Parameters
    ----------
    r_i : `numpy.ndarray`, positional-only
        Time evolution of individual or averaged positions for :math:`N`
        entities over :math:`N_\mathrm{b}` blocks of :math:`N_t` times
        each.

        **Shape**: :math:`(N_t,\,3)`, :math:`(N_t,\,N,\,3)`,
        :math:`(N_\mathrm{b},\,N_t,\,3)`, or
        :math:`(N_\mathrm{b},\,N_t,\,N,\,3)`.

        **Reference unit**: :math:`\mathrm{Å}`.

    r_j : `numpy.ndarray`, positional-only, optional
        Time evolution of individual or averaged positions for another
        :math:`N` entities over :math:`N_\mathrm{b}` blocks of
        :math:`N_t` times each.

        **Shape**: Same as `r_i`.

        **Reference unit**: :math:`\mathrm{Å}`.

    axis : `int`, optional
        Axis along which time evolves. If not specified, the axis is
        determined automatically using the shape of `r_i`.

    average : `bool`, keyword-only, default: :code:`True`
        Determines whether the MSD/CD is averaged over all entities.
        Only available if `r_i` and `r_j` contain information for
        multiple entities.

    Returns
    -------
    disp : `numpy.ndarray`
        MSD or CD.

        **Shape**: Same as the shape of `r_i`, except with the last axis
        removed. If :code:`average=True`, the axis containing the
        :math:`N` entities is also removed.

        **Reference unit**: :math:`\text{Å}^2`.

    References
    ----------
    .. [1] Fong, K. D.; Self, J.; McCloskey, B. D.; Persson, K. A.
       Onsager Transport Coefficients and Transference Numbers in
       Polyelectrolyte Solutions and Polymerized Ionic Liquids.
       *Macromolecules* **2020**, *53* (21), 9503–9512.
       https://doi.org/10.1021/acs.macromol.0c02001.
    """

    # Ensure arrays have valid shapes
    r_i = np.asarray(r_i)
    if r_i.size == 0:
        raise ValueError("The position arrays cannot be empty.")
    if r_i.shape[-1] != 3:
        emsg = ("The position arrays must have three components in "
                "the last axis.")
        raise ValueError(emsg)
    ndim = r_i.ndim
    if not 2 <= ndim <= 4:
        emsg = ("The position arrays must be two-, three-, or four-"
                "dimensional.")
        raise ValueError(emsg)
    if r_j is not None:
        r_j = np.asarray(r_j)
        if r_i.shape != r_j.shape:
            raise ValueError("The position arrays must have the same shape.")

    # Check or set axis along which to compute the MSD/CD
    if axis is None:
        if ndim == 4:
            axis = 1
        else:
            axis = 0
            if ndim == 3:
                emsg = ("The axis along which to compute the MSD/CD "
                        "was not specified and is ambiguous for a "
                        "three-dimensional array. As such, it has been "
                        "set to the first axis by default.")
                warnings.warn(emsg)
    elif axis not in {0, 1}:
        emsg = ("The MSD/CD can only be computed along the first or "
                "second axis.")
        raise ValueError(emsg)

    # Calculate the MSD/CD for each entity
    N_t = r_i.shape[axis]
    if r_j is None:
        if axis:
            disp = np.stack(
                [((r_i[:, :-i if i else None] - r_i[:, i:]) ** 2)
                 .sum(axis=-1).mean(axis=axis)
                 for i in range(N_t)],
                axis=1
            )
        else:
            disp = np.stack([((r_i[:-i if i else None] - r_i[i:]) ** 2)
                             .sum(axis=-1).mean(axis=axis)
                             for i in range(N_t)])
    else:
        if axis:
            disp = np.stack(
                [
                    np.einsum(
                        "bt...d,bt...d->bt...",
                        r_i[:, :-i if i else None] - r_i[:, i:],
                        r_j[:, :-i if i else None] - r_j[:, i:]
                    ).mean(axis=axis)
                    for i in range(N_t)
                ],
                axis=1
            )
        else:
            disp = np.stack([
                np.einsum(
                    "t...d,t...d->t...",
                    r_i[:-i if i else None] - r_i[i:],
                    r_j[:-i if i else None] - r_j[i:]
                ).mean(axis=axis)
                for i in range(N_t)
            ])

    # Average over all entities, if desired
    if ndim - axis == 3 and average:
        disp = disp.mean(axis=ndim - 2)

    return disp