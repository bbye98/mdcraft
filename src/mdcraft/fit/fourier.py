r"""
Fourier series models
=====================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

The Fourier series is a sum of sine and cosine functions that describes
a periodic signal, and can be represented in trigonometric form as

.. math::

   y=a_0+\sum_{k=1}^na_k\cos{(k\omega x)}+b_k\sin{(k\omega x)}

where :math:`a_0` models a constant (intercept) term in the data and is
associated with the :math:`k=0` cosine term, :math:`\omega` is the
fundamental frequency of the signal, and :math:`n` is the number of
terms (harmonics) in the series.

This module provides the general Fourier series model above for any
number of terms :math:`k`, as well as convenience functions for Fourier
series models with :math:`1\leq k\leq 8` analogous to MATLAB's
:code:`fourier1`, :code:`fourier2`, etc.
"""

import numpy as np

def fourier(
        x: np.ndarray, omega: float, a0: float, *args: float) -> np.ndarray:

    r"""
    General Fourier series model.

    .. math::

       y=a_0+\sum_{k=1}^na_i\cos{(k\omega x)}+b_i\sin{(k\omega x)}

    Parameters
    ----------
    x : `numpy.ndarray`
        :math:`x`-values.

    omega : `float`
        Fundamental frequency :math:`\omega` of the signal.

    a0 : `float`
        Constant (intercept) term :math:`a_0` for the :math:`k=0`
        cosine term.

    *args : `float`
        Fitting parameters for the Fourier series term(s), ordered as
        :math:`a_1,\,b_1,\,a_2,\,b_2,\ldots,\,a_n,\,b_n`, where :math:`n`
        is the number of terms in the model. As such, the number of
        variable positional arguments must be even.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.

    Examples
    --------
    Generate :math:`x`- and :math:`y`-values (with error), and then use
    :meth:`scipy.optimize.curve_fit` to fit coefficients for a one-term
    Fourier series model.

    >>> from scipy import optimize
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(0, 5, 20)
    >>> err = (2 * rng.random(x.shape) - 1) / 10
    >>> y = 1 + 2 * np.cos(x / 2) + 3 * np.sin(x / 2) + err
    >>> pk, _ = optimize.curve_fit(
            lambda x, omega, a0, a1, b1: fourier(x, omega, a0, a1, b1), x, y
        )
    >>> pk
    array([0.51185734, 1.15090712, 1.87471839, 2.87117784])

    Evaluate the fitted :math:`y`-values using the coefficients.

    >>> fourier(x, *pk)
    array([3.0256255 , 3.39422104, 3.72217562, 4.00354785, 4.23324026,
           4.40709163, 4.52195238, 4.57574164, 4.56748494, 4.49733187,
           4.36655334, 4.1775186 , 3.9336523 , 3.63937245, 3.30001035,
           2.92171405, 2.51133696, 2.07631367, 1.62452525, 1.16415655])
    """

    n = len(args)
    assert n >= 2 and n % 2 == 0, \
        "Number of fitting parameters must be greater than 2 and even."
    kwx = np.arange(1, n // 2+1)[:, None] * omega * x
    return a0+args[::2] @ np.cos(kwx)+args[1::2] @ np.sin(kwx)

def fourier1(
        x: np.ndarray, a0: float, a1: float, b1: float, omega: float
    ) -> np.ndarray:

    r"""
    Convenience function for the :code:`fourier1` model from MATLAB.

    .. math::

       y=a_0+a_1\cos{(\omega x)}+b_1\sin{(\omega x)}

    Parameters
    ----------
    x : `numpy.ndarray`
        :math:`x`-values.

    a0 : `float`
        Constant (intercept) term :math:`a_0` for the :math:`k=0`
        cosine term.

    a1 : `float`
        Coefficient :math:`a_1` for the first :math:`\cos` term.

    b1 : `float`
        Coefficient :math:`b_1` for the first :math:`\sin` term.

    omega : `float`
        Fundamental frequency :math:`\omega` of the signal.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return fourier(x, omega, a0, a1, b1)

def fourier2(
        x: np.ndarray, a0: float, a1: float, b1: float, a2: float,
        b2: float, omega: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`fourier2` model from MATLAB.

    .. math::

       y=a_0+a_1\cos{(\omega x)}+b_1\sin{(\omega x)}
       +a_2\cos{(2\omega x)}+b_2\sin{(2\omega x)}

    Parameters
    ----------
    x : `numpy.ndarray`
        :math:`x`-values.

    a0 : `float`
        Constant (intercept) term :math:`a_0` for the :math:`k=0`
        cosine term.

    a1 : `float`
        Coefficient :math:`a_1` for the first :math:`\cos` term.

    b1 : `float`
        Coefficient :math:`b_1` for the first :math:`\sin` term.

    a2 : `float`
        Coefficient :math:`a_2` for the second :math:`\cos` term.

    b2 : `float`
        Coefficient :math:`b_2` for the second :math:`\sin` term.

    omega : `float`
        Fundamental frequency :math:`\omega` of the signal.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return fourier(x, omega, a0, a1, b1, a2, b2)

def fourier3(
        x: np.ndarray, a0: float, a1: float, b1: float, a2: float,
        b2: float, a3: float, b3: float, omega: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`fourier3` model from MATLAB.

    .. math::

       y=a_0+a_1\cos{(\omega x)}+b_1\sin{(\omega x)}
       +a_2\cos{(2\omega x)}+b_2\sin{(2\omega x)}
       +a_3\cos{(3\omega x)}+b_3\sin{(3\omega x)}

    Parameters
    ----------
    x : `numpy.ndarray`
        :math:`x`-values.

    a0 : `float`
        Constant (intercept) term :math:`a_0` for the :math:`k=0`
        cosine term.

    a1 : `float`
        Coefficient :math:`a_1` for the first :math:`\cos` term.

    b1 : `float`
        Coefficient :math:`b_1` for the first :math:`\sin` term.

    a2 : `float`
        Coefficient :math:`a_2` for the second :math:`\cos` term.

    b2 : `float`
        Coefficient :math:`b_2` for the second :math:`\sin` term.

    a3 : `float`
        Coefficient :math:`a_3` for the third :math:`\cos` term.

    b3 : `float`
        Coefficient :math:`b_3` for the third :math:`\sin` term.

    omega : `float`
        Fundamental frequency :math:`\omega` of the signal.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return fourier(x, omega, a0, a1, b1, a2, b2, a3, b3)

def fourier4(
        x: np.ndarray, a0: float, a1: float, b1: float, a2: float,
        b2: float, a3: float, b3: float, a4: float, b4: float, omega: float
    ) -> np.ndarray:

    r"""
    Convenience function for the :code:`fourier4` model from MATLAB.

    .. math::

       y=a_0+a_1\cos{(\omega x)}+b_1\sin{(\omega x)}
       +a_2\cos{(2\omega x)}+b_2\sin{(2\omega x)}
       +a_3\cos{(3\omega x)}+b_3\sin{(3\omega x)}
       +a_4\cos{(4\omega x)}+b_4\sin{(4\omega x)}

    Parameters
    ----------
    x : `numpy.ndarray`
        :math:`x`-values.

    a0 : `float`
        Constant (intercept) term :math:`a_0` for the :math:`k=0`
        cosine term.

    a1 : `float`
        Coefficient :math:`a_1` for the first :math:`\cos` term.

    b1 : `float`
        Coefficient :math:`b_1` for the first :math:`\sin` term.

    a2 : `float`
        Coefficient :math:`a_2` for the second :math:`\cos` term.

    b2 : `float`
        Coefficient :math:`b_2` for the second :math:`\sin` term.

    a3 : `float`
        Coefficient :math:`a_3` for the third :math:`\cos` term.

    b3 : `float`
        Coefficient :math:`b_3` for the third :math:`\sin` term.

    a4 : `float`
        Coefficient :math:`a_4` for the fourth :math:`\cos` term.

    b4 : `float`
        Coefficient :math:`b_4` for the fourth :math:`\sin` term.

    omega : `float`
        Fundamental frequency :math:`\omega` of the signal.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return fourier(x, omega, a0, a1, b1, a2, b2, a3, b3, a4, b4)

def fourier5(
        x: np.ndarray, a0: float, a1: float, b1: float, a2: float,
        b2: float, a3: float, b3: float, a4: float, b4: float,
        a5: float, b5: float, omega: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`fourier5` model from MATLAB.

    .. math::

       y=a_0+a_1\cos{(\omega x)}+b_1\sin{(\omega x)}
       +a_2\cos{(2\omega x)}+b_2\sin{(2\omega x)}+\cdots
       +a_5\cos{(5\omega x)}+b_5\sin{(5\omega x)}

    Parameters
    ----------
    x : `numpy.ndarray`
        :math:`x`-values.

    a0 : `float`
        Constant (intercept) term :math:`a_0` for the :math:`k=0`
        cosine term.

    a1 : `float`
        Coefficient :math:`a_1` for the first :math:`\cos` term.

    b1 : `float`
        Coefficient :math:`b_1` for the first :math:`\sin` term.

    a2 : `float`
        Coefficient :math:`a_2` for the second :math:`\cos` term.

    b2 : `float`
        Coefficient :math:`b_2` for the second :math:`\sin` term.

    a3 : `float`
        Coefficient :math:`a_3` for the third :math:`\cos` term.

    b3 : `float`
        Coefficient :math:`b_3` for the third :math:`\sin` term.

    a4 : `float`
        Coefficient :math:`a_4` for the fourth :math:`\cos` term.

    b4 : `float`
        Coefficient :math:`b_4` for the fourth :math:`\sin` term.

    a5 : `float`
        Coefficient :math:`a_5` for the fifth :math:`\cos` term.

    b5 : `float`
        Coefficient :math:`b_5` for the fifth :math:`\sin` term.

    omega : `float`
        Fundamental frequency :math:`\omega` of the signal.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return fourier(x, omega, a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5)

def fourier6(
        x: np.ndarray, a0: float, a1: float, b1: float, a2: float,
        b2: float, a3: float, b3: float, a4: float, b4: float,
        a5: float, b5: float, a6: float, b6: float, omega: float
    ) -> np.ndarray:

    r"""
    Convenience function for the :code:`fourier6` model from MATLAB.

    .. math::

       y=a_0+a_1\cos{(\omega x)}+b_1\sin{(\omega x)}
       +a_2\cos{(2\omega x)}+b_2\sin{(2\omega x)}+\cdots
       +a_6\cos{(6\omega x)}+b_6\sin{(6\omega x)}

    Parameters
    ----------
    x : `numpy.ndarray`
        :math:`x`-values.

    a0 : `float`
        Constant (intercept) term :math:`a_0` for the :math:`k=0`
        cosine term.

    a1 : `float`
        Coefficient :math:`a_1` for the first :math:`\cos` term.

    b1 : `float`
        Coefficient :math:`b_1` for the first :math:`\sin` term.

    a2 : `float`
        Coefficient :math:`a_2` for the second :math:`\cos` term.

    b2 : `float`
        Coefficient :math:`b_2` for the second :math:`\sin` term.

    a3 : `float`
        Coefficient :math:`a_3` for the third :math:`\cos` term.

    b3 : `float`
        Coefficient :math:`b_3` for the third :math:`\sin` term.

    a4 : `float`
        Coefficient :math:`a_4` for the fourth :math:`\cos` term.

    b4 : `float`
        Coefficient :math:`b_4` for the fourth :math:`\sin` term.

    a5 : `float`
        Coefficient :math:`a_5` for the fifth :math:`\cos` term.

    b5 : `float`
        Coefficient :math:`b_5` for the fifth :math:`\sin` term.

    a6 : `float`
        Coefficient :math:`a_6` for the sixth :math:`\cos` term.

    b6 : `float`
        Coefficient :math:`b_6` for the sixth :math:`\sin` term.

    omega : `float`
        Fundamental frequency :math:`\omega` of the signal.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return fourier(x, omega, a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5,
                   a6, b6)

def fourier7(
        x: np.ndarray, a0: float, a1: float, b1: float, a2: float,
        b2: float, a3: float, b3: float, a4: float, b4: float,
        a5: float, b5: float, a6: float, b6: float, a7: float,
        b7: float, omega: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`fourier7` model from MATLAB.

    .. math::

       y=a_0+a_1\cos{(\omega x)}+b_1\sin{(\omega x)}
       +a_2\cos{(2\omega x)}+b_2\sin{(2\omega x)}+\cdots
       +a_7\cos{(7\omega x)}+b_7\sin{(7\omega x)}

    Parameters
    ----------
    x : `numpy.ndarray`
        :math:`x`-values.

    a0 : `float`
        Constant (intercept) term :math:`a_0` for the :math:`k=0`
        cosine term.

    a1 : `float`
        Coefficient :math:`a_1` for the first :math:`\cos` term.

    b1 : `float`
        Coefficient :math:`b_1` for the first :math:`\sin` term.

    a2 : `float`
        Coefficient :math:`a_2` for the second :math:`\cos` term.

    b2 : `float`
        Coefficient :math:`b_2` for the second :math:`\sin` term.

    a3 : `float`
        Coefficient :math:`a_3` for the third :math:`\cos` term.

    b3 : `float`
        Coefficient :math:`b_3` for the third :math:`\sin` term.

    a4 : `float`
        Coefficient :math:`a_4` for the fourth :math:`\cos` term.

    b4 : `float`
        Coefficient :math:`b_4` for the fourth :math:`\sin` term.

    a5 : `float`
        Coefficient :math:`a_5` for the fifth :math:`\cos` term.

    b5 : `float`
        Coefficient :math:`b_5` for the fifth :math:`\sin` term.

    a6 : `float`
        Coefficient :math:`a_6` for the sixth :math:`\cos` term.

    b6 : `float`
        Coefficient :math:`b_6` for the sixth :math:`\sin` term.

    a7 : `float`
        Coefficient :math:`a_7` for the seventh :math:`\cos` term.

    b7 : `float`
        Coefficient :math:`b_7` for the seventh :math:`\sin` term.

    omega : `float`
        Fundamental frequency :math:`\omega` of the signal.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return fourier(x, omega, a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5,
                   a6, b6, a7, b7)

def fourier8(
        x: np.ndarray, a0: float, a1: float, b1: float, a2: float,
        b2: float, a3: float, b3: float, a4: float, b4: float,
        a5: float, b5: float, a6: float, b6: float, a7: float,
        b7: float, a8: float, b8: float, omega: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`fourier8` model from MATLAB.

    .. math::

       y=a_0+a_1\cos{(\omega x)}+b_1\sin{(\omega x)}
       +a_2\cos{(2\omega x)}+b_2\sin{(2\omega x)}+\cdots
       +a_8\cos{(8\omega x)}+b_8\sin{(8\omega x)}

    Parameters
    ----------
    x : `numpy.ndarray`
        :math:`x`-values.

    a0 : `float`
        Constant (intercept) term :math:`a_0` for the :math:`k=0`
        cosine term.

    a1 : `float`
        Coefficient :math:`a_1` for the first :math:`\cos` term.

    b1 : `float`
        Coefficient :math:`b_1` for the first :math:`\sin` term.

    a2 : `float`
        Coefficient :math:`a_2` for the second :math:`\cos` term.

    b2 : `float`
        Coefficient :math:`b_2` for the second :math:`\sin` term.

    a3 : `float`
        Coefficient :math:`a_3` for the third :math:`\cos` term.

    b3 : `float`
        Coefficient :math:`b_3` for the third :math:`\sin` term.

    a4 : `float`
        Coefficient :math:`a_4` for the fourth :math:`\cos` term.

    b4 : `float`
        Coefficient :math:`b_4` for the fourth :math:`\sin` term.

    a5 : `float`
        Coefficient :math:`a_5` for the fifth :math:`\cos` term.

    b5 : `float`
        Coefficient :math:`b_5` for the fifth :math:`\sin` term.

    a6 : `float`
        Coefficient :math:`a_6` for the sixth :math:`\cos` term.

    b6 : `float`
        Coefficient :math:`b_6` for the sixth :math:`\sin` term.

    a7 : `float`
        Coefficient :math:`a_7` for the seventh :math:`\cos` term.

    b7 : `float`
        Coefficient :math:`b_7` for the seventh :math:`\sin` term.

    a8 : `float`
        Coefficient :math:`a_8` for the eigth :math:`\cos` term.

    b8 : `float`
        Coefficient :math:`b_8` for the eigth :math:`\sin` term.

    omega : `float`
        Fundamental frequency :math:`\omega` of the signal.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return fourier(x, omega, a0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5,
                   a6, b6, a7, b7, a8, b8)