from __future__ import annotations
import warnings

import numpy as np
from scipy import fft as pfft


def correlation(
    x: np.ndarray[float | complex],
    y: np.ndarray[float | complex] = None,
    /,
    axis: int = None,
    *,
    average: bool = False,
    fft: bool = True,
    symmetric: bool = False,
    vector: bool = False,
) -> np.ndarray[float | complex]:
    """
    Evaluates the autocorrelation function (ACF)
    :math:`\\mathrm{R_\\mathbf{XX}}(\\tau)` or cross-correlation
    function (CCF) :math:`\\mathrm{R_\\mathbf{XY}}(\\tau)` of time
    series :math:`\\mathbf{X}(t)` and :math:`\\mathbf{Y}(t)`.

    Using a naive sliding window technique, the ACF for a data set
    :math:`\\mathbf{X}(t)` can be computed using

    .. math::

       \\mathrm{R}_{\\mathbf{XX}}(\\tau)
       =\\langle\\mathbf{X}(t_0+\\tau)\\cdot\\mathbf{X}^*(t_0)\\rangle
       =\\dfrac{1}{N_\\tau}\\sum_{j=1}^{N_\\tau}
       \\textbf{X}(t_j+\\tau)\\cdot\\textbf{X}^*(t_j)

    where :math:`\\tau` is the time lag, :math:`t_j` is an arbitrary
    reference time, :math:`N_\\tau` is the number of possible reference
    times, and the asterisk (:math:`^*`) denotes the complex conjugate.

    Similarly, the CCF for data sets :math:`\\mathbf{X}(t)` and
    :math:`\\mathbf{Y}(t)` can be computed using

    .. math::

       \\mathrm{R}_{\\mathbf{XY}}(\\tau)
       =\\langle\\mathbf{X}(t_0+\\tau)\\cdot\\mathbf{Y}^*(t_0)\\rangle
       =\\dfrac{1}{N_\\tau}\\sum_{j=1}^{N_\\tau}
       \\textbf{X}(t_j+\\tau)\\cdot\\textbf{Y}^*(t_j)

    To minimize statistical noise, the ACF/CCF is calculated for and
    averaged over all possible reference times :math:`t_j`. As such,
    this approach has a time complexity of :math:`\\mathcal{O}(N^2)`,
    making it infeasible for large data sets.

    Alternatively, the ACF/CCF can be efficiently computed using the
    fast convolution algorithm (FCA) [1]_ [2]_, which leverages the
    Wiener–Khinchin theorem. FCA uses fast Fourier transforms (FFT) and
    has a time complexity of :math:`\\mathcal{O}(N\\log{N})`.

    Using FCA, the ACF for a data set :math:`\\mathbf{X}(t)` is computed
    using

    .. math::

       \\begin{gather*}
         \\hat{\\mathbf{X}}(\\omega)=\\mathcal{F}[\\mathbf{X}(t)]\\\\
         \\mathrm{R}_{\\mathbf{XX}}(\\tau)=\\mathcal{F}^{-1}
         [\\hat{\\mathbf{X}}(\\omega)\\hat{\\mathbf{X}}^*(\\omega)]
       \\end{gather*}

    and the CCF for data sets :math:`\\mathbf{X}(t)` and
    :math:`\\mathbf{Y}(t)` is computed using

    .. math::

       \\mathrm{R}_{\\mathbf{XY}}(\\tau)=\\mathcal{F}^{-1}[\\mathcal{F}
       [\\mathbf{X}(t)]\\cdot\\mathcal{F}[\\mathbf{Y}(t)]]

    Parameters
    ----------
    x : `numpy.ndarray`, positional-only
        Time evolution of :math:`d`-dimensional data for :math:`N`
        entities over :math:`N_\\mathrm{b}` blocks of :math:`n_t` times
        each.

        .. container::

           **Shape**:

           * Scalar data: :math:`(n_t,)`, :math:`(n_t,\\,N)`,
             :math:`(N_\\mathrm{b},\\,n_t)`, or
             :math:`(N_\\mathrm{b},\\,n_t,\\,N)`.
           * Vector data: :math:`(n_t,\\,d)`, :math:`(n_t,\\,N,\\,d)`,
             :math:`(N_\\mathrm{b},\\,n_t,\\,d)`, or
             :math:`(N_\\mathrm{b},\\,n_t,\\,N,\\,d)`.

    y : `numpy.ndarray`, positional-only, optional
        Time evolution of :math:`d`-dimensional data for another
        :math:`N` entities over :math:`N_\\mathrm{b}` blocks of
        :math:`n_t` times each. If provided, the CCF for `x` and `y` is
        evaluated. Otherwise, the ACF for `x` is evaluated.

        **Shape**: Same as `x`.

    axis : `int`, optional
        Axis along which time evolves. If not specified, the axis is
        determined automatically using the shape of `x`.

    average : `bool`, keyword-only, default: :code:`True`
        Specifies whether to average the ACF/CCFd over all entities.
        Only available if `x` and `y` contain information for multiple
        entities.

    fft : `bool`, keyword-only, default: :code:`True`
        Specifies whether to use fast Fourier transforms (FFT) to
        evaluate the ACF/CCF.

    symmetric : `bool`, keyword-only, default: :code:`False`
        Specifies whether to double the ACF or to combine the negative
        and positive time lags for the CCF.

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

           * If :code:`average=True`, the axis corresponding to the
             :math:`N` entities is no longer present.
           * If :code:`vector=True`, the last axis is no longer present.

           For CCF, the shape is that of `x` but with the following
           modifications:

           * If :code:`average=True`, the axis corresponding to the
             :math:`N` entities is no longer present.
           * If :code:`symmetric=False`, the axis corresponding to the
             :math:`n_t` times now has a length of :math:`2n_t-1` to
             accomodate negative and positive time lags.
           * If :code:`vector=True`, the last axis is no longer present.

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

    # Ensure that arguments are valid and consistent
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("The arrays cannot be empty.")
    n_dim = x.ndim
    if not 1 <= n_dim <= 4:
        raise ValueError(
            "The arrays must be one-, two-, three-, or four-dimensional."
        )
    if vector and n_dim == 1:
        raise ValueError(
            "The arrays cannot be one-dimensional if " "`vector=True`."
        )
    if y is not None:
        y = np.asarray(y)
        if x.shape != y.shape:
            raise ValueError("The arrays must have the same shape.")

    # Check or set axis along which to compute the ACF/CCF
    if axis is None:
        if n_dim == 4:
            axis = 1
        else:
            axis = 0
            if n_dim > 1:
                warnings.warn(
                    "The axis along which to compute the ACF/CCF was"
                    "not specified and is ambiguous for a "
                    "multidimensional array. By default, the ACF/CCF "
                    "will be evaluated along the first axis (`axis=0`)."
                )
    elif axis not in {0, 1}:
        raise ValueError(
            "The ACF/CCF can only be computed along the first "
            "(`axis=0`) or second axis (`axis=1`)."
        )

    # Compute the ACF/CCF
    n_t = x.shape[axis]
    slices = (slice(None),) * axis
    if fft:
        # Determine whether faster real-valued FFT algorithms can be used
        real = np.isrealobj(x) and (y is None or np.isrealobj(y))
        if real:
            f_fft = pfft.rfft
            f_ifft = pfft.irfft
        else:
            f_fft = pfft.fft
            f_ifft = pfft.ifft

        # Initialize array with axis slice(s) needed for normalization later
        axis_slices = [slice(n_t)]

        # Compute the power spectral density by first zero-padding the
        # arrays for linear convolution and then inverting it to get the
        # ACF/CCF
        n_fft = 2 * pfft.next_fast_len(n_t, real=real)
        if y is None:
            ft = f_fft(x, n=n_fft, axis=axis)
            corr = f_ifft(ft * ft.conj(), axis=axis)
            corr = (symmetric + 1) * (corr[:, :n_t] if axis else corr[:n_t])
        else:
            ft_x = f_fft(x, n=n_fft, axis=axis)
            ft_y = f_fft(y, n=n_fft, axis=axis)
            ft = ft_x.conj() * ft_y
            if symmetric:
                corr = f_ifft(ft + ft_x * ft_y.conj(), axis=axis)
                corr = corr[:, :n_t] if axis else corr[:n_t]
            else:
                corr = f_ifft(ft, axis=axis)
                axis_slices.append(slice(1 - n_t, None))

        # Sum over the last axis if the arrays contain vectors
        if vector:
            corr = corr.sum(axis=-1)

    else:
        # Initialize array with axis slice(s) needed for normalization later
        axis_slices = [slice(-n_t, None)]

        # Use forward and backward slices to get relevant time windows
        # for the ACF/CCF
        if y is None:
            if n_dim == 1:
                corr = np.fromiter(
                    (
                        np.dot(x[i:], x[: -i if i else None])
                        for i in range(n_t)
                    ),
                    dtype=float,
                    count=n_t,
                )
            else:
                ss_prefix = "b" * axis
                ss_lhs = f"{ss_prefix}t...{'d' * vector}"
                corr = np.stack(
                    [
                        np.einsum(
                            f"{ss_lhs},{ss_lhs}->{ss_prefix}...",
                            x[*slices, i:],
                            x[*slices, : -i if i else None],
                        )
                        for i in range(n_t)
                    ],
                    axis=axis,
                )
        else:
            start = np.r_[np.zeros(n_t - 1, dtype=int), 0:n_t]
            stop = np.r_[1 : n_t + 1, n_t * np.ones(n_t - 1, dtype=int)]
            if n_dim == 1:
                corr = np.fromiter(
                    (
                        np.dot(x[i:j], y[k:m])
                        for i, j, k, m in zip(
                            start[::-1], stop[::-1], start, stop
                        )
                    ),
                    dtype=float,
                    count=2 * n_t - 1,
                )
            else:
                ss_prefix = "b" * axis
                ss_lhs = f"{ss_prefix}t...{'d' * vector}"
                corr = np.stack(
                    [
                        np.einsum(
                            f"{ss_lhs},{ss_lhs}->{ss_prefix}...",
                            x[*slices, i:j],
                            y[*slices, k:m],
                        )
                        for i, j, k, m in zip(
                            start[::-1], stop[::-1], start, stop
                        )
                    ],
                    axis=axis,
                )

        # Double the ACF or combine the negative and positive time lags for
        # the CCF, if desired
        if symmetric:
            if y is None:
                corr *= 2
            else:
                corr = corr[*slices, n_t - 1 :] + corr[*slices, n_t - 1 :: -1]
        else:
            axis_slices.append(slice(n_t - 1))

    # Determine the axes over which to expand the dimensions of the
    # reversed time array for correct matrix division
    axes = list(range(n_dim - vector))
    axes.remove(axis)

    # Normalize the ACF/CCF
    corr[*slices, axis_slices[0]] /= np.expand_dims(
        np.arange(n_t, 0, -1), axes
    )
    if corr.shape[axis] != n_t:
        corr[*slices, axis_slices[1]] /= np.expand_dims(
            np.arange(1, n_t), axes
        )
        if fft:
            corr = pfft.fftshift(corr, axis)[
                *slices, (start := n_fft // 2 - n_t + 1) : start + 2 * n_t - 1
            ]

    # Average over all entities, if desired
    if average and axis != (axis_avg := n_dim - vector - 1):
        return corr.mean(axis=axis_avg)

    return corr
