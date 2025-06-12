from __future__ import annotations
from typing import TYPE_CHECKING
import warnings

import numpy as np
from scipy import fft as pfft

if TYPE_CHECKING:
    from .. import float_t, complex_t


def correlation(
    x: np.ndarray[float_t | complex_t],
    y: np.ndarray[float_t | complex_t] | None = None,
    /,
    axis: int | None = None,
    *,
    average: bool = False,
    fft: bool = True,
    symmetrize: bool = False,
    vector: bool = False,
) -> np.ndarray[float_t | complex_t]:
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
       \\mathbf{X}(t_j+\\tau)\\cdot\\mathbf{X}^*(t_j)

    where :math:`\\tau` is the time lag, :math:`t_j` is an arbitrary
    reference time, :math:`N_\\tau` is the number of possible reference
    times, and the asterisk (:math:`^*`) denotes the complex conjugate.

    Similarly, the CCF for data sets :math:`\\mathbf{X}(t)` and
    :math:`\\mathbf{Y}(t)` can be computed using

    .. math::

       \\mathrm{R}_{\\mathbf{XY}}(\\tau)
       =\\langle\\mathbf{X}(t_0+\\tau)\\cdot\\mathbf{Y}^*(t_0)\\rangle
       =\\dfrac{1}{N_\\tau}\\sum_{j=1}^{N_\\tau}
       \\mathbf{X}(t_j+\\tau)\\cdot\\mathbf{Y}^*(t_j)

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
        entities over :math:`N_\\mathrm{b}` blocks of :math:`N_t` times
        each.

        .. container::

           **Shape**:

           * Scalar data: :math:`(N_t,)`, :math:`(N_t,N)`,
             :math:`(N_\\mathrm{b},N_t)`, or
             :math:`(N_\\mathrm{b},N_t,N)`.
           * Vector data: :math:`(N_t,d)`, :math:`(N_t,N,d)`,
             :math:`(N_\\mathrm{b},N_t,d)`, or
             :math:`(N_\\mathrm{b},N_t,N,d)`.

    y : `numpy.ndarray`, positional-only, optional
        Time evolution of :math:`d`-dimensional data for another
        :math:`N` entities over :math:`N_\\mathrm{b}` blocks of
        :math:`N_t` times each. If provided, the CCF for `x` and `y` is
        evaluated. Otherwise, the ACF for `x` is evaluated.

        **Shape**: Same as `x`.

    axis : `int`, optional
        Axis along which time evolves. If not specified, the axis is
        determined automatically using the shape of `x`.

    average : `bool`, keyword-only, default: :code:`True`
        Specifies whether to average the ACF/CCF over all entities.
        Only available if `x` and `y` contain information for multiple
        entities.

    fft : `bool`, keyword-only, default: :code:`True`
        Specifies whether to use fast Fourier transforms (FFT) to
        evaluate the ACF/CCF.

    symmetrize : `bool`, keyword-only, default: :code:`False`
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
           * If :code:`symmetrize=False`, the axis corresponding to the
             :math:`N_t` times now has a length of :math:`2N_t-1` to
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
            "The arrays cannot be one-dimensional if `vector=True`."
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
                    "not specified and is ambiguous for "
                    "multidimensional arrays. By default, the ACF/CCF "
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
            corr = (symmetrize + 1) * (corr[:, :n_t] if axis else corr[:n_t])
        else:
            ft_x = f_fft(x, n=n_fft, axis=axis)
            ft_y = f_fft(y, n=n_fft, axis=axis)
            ft = ft_x.conj() * ft_y
            if symmetrize:
                corr = f_ifft(ft + ft_x * ft_y.conj(), axis=axis)[*slices, :n_t]
            else:
                corr = f_ifft(ft, axis=axis)
                axis_slices.append(slice(1 - n_t, None))

        # Sum over the last axis if the arrays contain vectors
        if vector:
            corr = corr.sum(axis=-1)

    else:
        # Initialize array with axis slice(s) needed for normalization later
        axis_slices = [slice(-n_t, None)]

        # Use opposite sliding windows to get relevant data for each time lag
        if y is None:
            if n_dim == 1:
                corr = np.fromiter(
                    (np.dot(x[i:], x[: -i if i else None]) for i in range(n_t)),
                    x.dtype,
                    n_t,
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
            start = np.r_[np.zeros(n_t - 1, np.uint32), 0:n_t]
            stop = np.r_[1 : n_t + 1, np.full(n_t - 1, n_t, np.uint32)]
            if n_dim == 1:
                corr = np.fromiter(
                    (
                        np.dot(x[i:j], y[k:m])
                        for i, j, k, m in zip(
                            start[::-1], stop[::-1], start, stop
                        )
                    ),
                    x.dtype,
                    2 * n_t - 1,
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
        if symmetrize:
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
    corr[*slices, axis_slices[0]] /= np.expand_dims(np.arange(n_t, 0, -1), axes)
    if corr.shape[axis] != n_t:
        corr[*slices, axis_slices[1]] /= np.expand_dims(np.arange(1, n_t), axes)
        if fft:
            corr = pfft.fftshift(corr, axis)[
                *slices, (start := n_fft // 2 - n_t + 1) : start + 2 * n_t - 1
            ]

    # Average over all entities, if desired
    if average and axis != (axis_avg := n_dim - vector - 1):
        return corr.mean(axis=axis_avg)

    return corr


def msd(
    r_i: np.ndarray[np.float64],
    r_j: np.ndarray[np.float64] | None = None,
    /,
    axis: int | None = None,
    *,
    average: bool = True,
    fft: bool = True,
) -> np.ndarray[np.float64]:
    """
    Evaluates the mean squared displacement (MSD) or the cross mean
    squared displacement (CMSD) of positions :math:`\\mathbf{r}_i(t)`
    and :math:`\\mathbf{r}_j(t)`.

    Using the Einstein relation, the MSD for a set of positions
    :math:`\\mathbf{r}_i(t)` can be computed using

    .. math::

        \\mathrm{MSD}_i(\\tau)=\\langle[\\mathbf{r}_i(t_0+\\tau)
        -\\mathbf{r}_i(t_0)]^2\\rangle
        =\\dfrac{1}{N_\\tau}\\sum_{k=1}^{N_\\tau}
        [\\mathbf{r}_i(t_k+\\tau)-\\mathbf{r}_i(t_k)]^2

    where :math:`\\tau` is the time lag, :math:`t_j` is an arbitrary
    reference time, and :math:`N_\\tau` is the number of possible
    reference times.

    Similarly, the CMSD for two sets of positions
    :math:`\\mathbf{r}_i(t)` and :math:`\\mathbf{r}_j(t)` can be
    computed using

    .. math::

       \\mathrm{CMSD}_{ij}(\\tau)&=\\langle
       [\\mathbf{r}_i(t_0+\\tau)-\\mathbf{r}_i(t_0)]\\cdot
       [\\mathbf{r}_j(t_0+\\tau)-\\mathbf{r}_j(t_0)]\\rangle\\\\
       &=\\dfrac{1}{N_\\tau}\\sum_{k=1}^{N_\\tau}
       [\\mathbf{r}_i(t_k+\\tau)-\\mathbf{r}_i(t_k)]\\cdot
       [\\mathbf{r}_j(t_k+\\tau)-\\mathbf{r}_j(t_k)]

    To minimize statistical noise, the MSD/CMSD is calculated for and
    averaged over all possible reference times :math:`t_k`.

    Alternatively, the MSD/CMSD can be efficiently computed using the
    fast convolution algorithm (FCA) [1]_ [2]_, which leverages the
    Wiener–Khinchin theorem. FCA uses fast Fourier transforms (FFT) and
    has a time complexity of :math:`\\mathcal{O}(N\\log{N})`.

    Using FCA, the MSD for a set of positions :math:`\\mathbf{r}_i(t)`
    is computed using

    .. math::

       \\mathrm{MSD}_{i,m}&=\\frac{1}{N_t-m}
       \\sum_{k=0}^{N_t-m-1}
       [\\mathbf{r}_{i,k+m}-\\mathbf{r}_{i,k}]^2\\\\
       &=\\frac{1}{N_t-m}\\sum_{k=0}^{N_t-m-1}
       \\left[\\mathbf{r}_{i,k+m}^2+\\mathbf{r}_{i,k}^2\\right]
       -\\frac{2}{N_t-m}\\sum_{k=0}^{N_t-m-1}
       \\mathbf{r}_{i,k}\\cdot\\mathbf{r}_{i,k+m}\\\\
       &=\\mathrm{S}_{ii,m}-2\\mathrm{R}_{ii,m}

    where :math:`i` is the species index, :math:`m` is the index
    corresponding to time lag :math:`\\tau`, :math:`\\mathrm{R}_{ii,m}`
    is the autocorrelation of :math:`\\mathbf{r}_i(t)`, and :math:`S_m`
    is evaluated using the recursive relation

    .. math::

       \\begin{gather*}
         D_{ii,m}=\\mathbf{r}_{i,m}^2\\\\
         Q_{ii,-1}=2\\sum_{k=0}^{N_t-1}D_{ii,k}\\\\
         Q_{ii,m}=Q_{ii,m-1}-D_{ii,m-1}-D_{ii,N_t-m}\\\\
         S_{ii,m}=\\frac{Q_{ii,m}}{N_t-m}
       \\end{gather*}

    Similarly, the CMSD for two sets of positions
    :math:`\\mathbf{r}_i(t)` and :math:`\\mathbf{r}_j(t)` is computed
    using

    .. math::

       \\mathrm{CMSD}_{ij,m}=S_{ij,m}-2\\mathrm{R}_{ij,m}

    where :math:`\\mathrm{R}_{ij,m}` is the cross-correlation of
    :math:`\\mathbf{r}_i(t)` and :math:`\\mathbf{r}_j(t)`, and
    :math:`S_{ij,m}` is evaluated using the recursive relation

    .. math::

       \\begin{gather*}
         D_{ij,m}=\\mathbf{r}_{i,m}\\cdot\\mathbf{r}_{j,m}\\\\
         Q_{ij,-1}=2\\sum_{k=0}^{N_t-1}D_{ij,k}\\\\
         Q_{ij,m}=Q_{ij,m-1}-D_{ij,m-1}-D_{ij,N_t-m}\\\\
         S_{ij,m}=\\frac{Q_{ij,m}}{N_t-m}
       \\end{gather*}

    .. note::

       `r_i` and `r_j` should be summed over all entities before being
       passed to this function if it is being used to evaluate the
       cross terms in the Onsager transport coefficients [3]_

       .. math::

          L_{ij}=\\frac{1}{6k_\\mathrm{B}T}\\lim_{t\\rightarrow\\infty}
          \\frac{d}{d\\tau}\\left\\langle\\sum_{\\alpha=1}^{N_i}
          [\\mathbf{r}_{i,\\alpha}(t_0+\\tau)
          -\\mathbf{r}_{i,\\alpha}(t_0)]\\cdot
          \\sum_{\\beta=1}^{N_j}[\\mathbf{r}_{j,\\beta}(t_0+\\tau)
          -\\mathbf{r}_{j,\\beta}(t_0)]\\right\\rangle

    Parameters
    ----------
    r_i : `numpy.ndarray`, positional-only
        Time evolution of individual or summed :math:`d`-dimensional
        positions for :math:`N` entities over :math:`N_\\mathrm{b}`
        blocks of :math:`N_t` times each.

        **Shape**: :math:`(N_t,d)`, :math:`(N_t,N,d)`,
        :math:`(N_\\mathrm{b},N_t,d)`, or
        :math:`(N_\\mathrm{b},N_t,N,d)`.

        **Reference unit**: :math:`\\mathrm{Å}`.

    r_j : `numpy.ndarray`, positional-only, optional
        Time evolution of individual or summed :math:`d`-dimensional
        positions for another :math:`N` entities over
        :math:`N_\\mathrm{b}` blocks of :math:`N_t` times each.

        **Shape**: Same as `r_i`.

        **Reference unit**: :math:`\\mathrm{Å}`.

    axis : `int`, optional
        Axis along which time evolves. If not specified, the axis is
        determined automatically using the shape of `r_i`.

    average : `bool`, keyword-only, default: :code:`True`
        Specifies whether to average the MSD/CMSD over all entities.
        Only available if `r_i` and `r_j` contain information for
        multiple entities.

    fft : `bool`, keyword-only, default: :code:`True`
        Specifies whether to use fast Fourier transforms (FFT) to
        evaluate the MSD/CMSD.

    Returns
    -------
    disp : `numpy.ndarray`
        MSD or CMSD.

        **Shape**: Same as the shape of `r_i`, except the last axis is
        no longer present. If :code:`average=True`, the axis indexing
        the :math:`N` entities is also no longer present.

        **Reference unit**: :math:`\\mathrm{Å}^2`.

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
    ndim = r_i.ndim
    if not 2 <= ndim <= 4:
        raise ValueError(
            "The position arrays must be two-, three-, or four-dimensional."
        )
    if r_j is not None:
        r_j = np.asarray(r_j)
        if r_i.shape != r_j.shape:
            raise ValueError("The position arrays must have the same shape.")

    # Check or set axis along which to compute the MSD/CMSD
    if axis is None:
        if ndim == 4:
            axis = 1
        else:
            axis = 0
            if ndim == 3:
                warnings.warn(
                    "The axis along which to compute the MSD/CMSD "
                    "was not specified and is ambiguous for "
                    "three-dimensional position arrays. By default, "
                    "the MSD/CMSD will be evaluated along the first "
                    "axis (`axis=0`)."
                )
    elif axis not in {0, 1}:
        raise ValueError(
            "The MSD/CMSD can only be computed along the first "
            "(`axis=0`) or second axis (`axis=1`)."
        )

    # Compute the MSD/CMSD
    n_t = r_i.shape[axis]
    slices = (slice(None),) * axis
    if fft:
        # Evaluate necessary intermediate quantities
        R_ij = correlation(
            r_i, r_j, axis, average=False, symmetrize=True, vector=True
        )
        D_ij = (r_i * (r_i if r_j is None else r_j)).sum(axis=-1)

        if ndim - axis == 3:
            # Evaluate the MSD/CMSD for each entity
            if not average:
                D_k = (np.vstack if ndim == 3 else np.hstack)(
                    (
                        D_ij,
                        np.zeros(
                            r_i.shape[:axis] + (1,) + r_i.shape[axis + 1 : -1]
                        ),
                    )
                )
                return (
                    2
                    * D_k.sum(axis=axis, keepdims=axis)
                    * np.ones((*(1,) * axis, n_t, 1))
                    - np.cumsum(
                        D_k[*slices, np.arange(-1, n_t - 1)]
                        + D_k[*slices, n_t:0:-1],
                        axis=axis,
                    )
                ) / np.arange(n_t, 0, -1)[:, None] - R_ij

            # Average the intermediate quantities over all entities
            R_ij = R_ij.mean(axis=ndim - 2)
            D_ij = D_ij.mean(axis=ndim - 2)

        # Evaluate the ensemble-averaged MSD/CMSD
        return (
            2 * D_ij.sum(axis=axis, keepdims=axis) * np.ones((1, n_t))
            - np.concatenate(
                (
                    np.zeros_like(D_ij[*slices, :1]),
                    np.cumsum(
                        D_ij[*slices, : n_t - 1]
                        + D_ij[*slices, n_t - 1 : 0 : -1],
                        axis=axis,
                    ),
                ),
                axis=axis,
            )
        ) / np.arange(n_t, 0, -1) - R_ij

    else:
        if r_j is None:
            disp = np.stack(
                [
                    (
                        (r_i[*slices, : -i if i else None] - r_i[*slices, i:])
                        ** 2
                    )
                    .sum(axis=-1)
                    .mean(axis=axis)
                    for i in range(n_t)
                ],
                axis=axis,
            )
        else:
            ss_prefix = "b" * axis
            disp = np.stack(
                [
                    np.einsum(
                        f"{ss_prefix}t...d,{ss_prefix}t...d->{ss_prefix}t...",
                        r_i[*slices, : -i if i else None] - r_i[*slices, i:],
                        r_j[*slices, : -i if i else None] - r_j[*slices, i:],
                    ).mean(axis=axis)
                    for i in range(n_t)
                ],
                axis=axis,
            )

        # Average over all entities, if desired
        if ndim - axis == 3 and average:
            return disp.mean(axis=ndim - 2)

        return disp
