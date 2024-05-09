r"""
Polynomial models
=================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

Polynomial models for curves are given by

.. math::

   y=\sum_{k=0}^np_kx^k

where :math:`n+1` is the order of the polynomial and :math:`n` is the
degree of the polynomial. The order gives the number of coefficients to
be fit, and the degree gives the highest power of the predictor variable.

Polynomials are often used when a simple empirical model is required.
You can use the polynomial model for interpolation or extrapolation, or
to characterize data using a global fit.

This module provides the general polynomial model above for any integer
:math:`n\geq0`, as well as convenience functions for polynomial models
with :math:`1\leq n \leq9` analogous to MATLAB's :code:`poly1`,
:code:`poly2`, etc.

Note that in MATLAB, the polynomial models have the form

.. math::

   y=\sum_{i=1}^{n+1}p_ix^{n+1-i}
"""

import numpy as np

def poly(x: np.ndarray, *args: float) -> np.ndarray:

    r"""
    General polynomial model.

    .. math::

       y=\sum_{k=0}^np_kx^k

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    *args : `float`
        Fitting parameters for each :math:`x^k` term, starting with the
        :math:`x^0` term. The number of positional arguments,
        :math:`n+1`, determines the order :math:`n` of the polynomial
        model.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.

    Examples
    --------
    Generate :math:`x`- and :math:`y`-values (with error), and then use
    :meth:`scipy.optimize.curve_fit` to fit coefficients for a third-order
    polynomial.

    >>> from scipy import optimize
    >>> rng = np.random.default_rng()
    >>> x = np.arange(-2, 3)
    >>> err = (2 * rng.random(x.shape) - 1) / 10
    >>> y = x ** 2+2 * x+1+err
    >>> pk, _ = optimize.curve_fit(lambda x, p0, p1, p2: poly(x, p0, p1, p2), x, y)
    >>> pk
    array([0.98225396, 2.0243695 , 1.01370609])

    Evaluate the fitted :math:`y`-values using the coefficients.

    >>> poly(x, *pk)
    array([ 0.9883393 , -0.02840946,  0.98225396,  4.02032955,  9.08581731])
    """

    return args @ (x ** np.arange(len(args))[:, None])

def poly1(x: np.ndarray, p1: float, p2: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`poly1` model from MATLAB:

    .. math::

       y=p_1x+p_2

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    p1 : `float`
        Coefficient :math:`p_1` for the :math:`x` term.

    p2 : `float`
        Constant term :math:`p_2`.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return poly(x, p2, p1)

def poly2(x: np.ndarray, p1: float, p2: float, p3: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`poly2` model from MATLAB:

    .. math::

       y=p_1x^2+p_2x+p_3

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    p1 : `float`
        Coefficient :math:`p_1` for the :math:`x^2` term.

    p2 : `float`
        Coefficient :math:`p_2` for the :math:`x` term.

    p3 : `float`
        Constant term :math:`p_3`.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return poly(x, p3, p2, p1)

def poly3(
        x: np.ndarray, p1: float, p2: float, p3: float, p4: float
    ) -> np.ndarray:

    r"""
    Convenience function for the :code:`poly3` model from MATLAB:

    .. math::

       y=p_1x^3+p_2x^2+p_3x+p_4

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    p1 : `float`
        Coefficient :math:`p_1` for the :math:`x^3` term.

    p2 : `float`
        Coefficient :math:`p_2` for the :math:`x^2` term.

    p3 : `float`
        Coefficient :math:`p_3` for the :math:`x` term.

    p4 : `float`
        Constant term :math:`p_4`.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return poly(x, p4, p3, p2, p1)

def poly4(
        x: np.ndarray, p1: float, p2: float, p3: float, p4: float, p5: float
    ) -> np.ndarray:

    r"""
    Convenience function for the :code:`poly4` model from MATLAB:

    .. math::

       y=p_1x^4+p_2x^3+\cdots+p_5

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    p1 : `float`
        Coefficient :math:`p_1` for the :math:`x^4` term.

    p2 : `float`
        Coefficient :math:`p_2` for the :math:`x^3` term.

    p3 : `float`
        Coefficient :math:`p_3` for the :math:`x^2` term.

    p4 : `float`
        Coefficient :math:`p_4` for the :math:`x` term.

    p5 : `float`
        Constant term :math:`p_5`.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return poly(x, p5, p4, p3, p2, p1)

def poly5(
        x: np.ndarray, p1: float, p2: float, p3: float, p4: float, p5: float,
        p6: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`poly5` model from MATLAB:

    .. math::

       y=p_1x^5+p_2x^4+\cdots+p_6

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    p1 : `float`
        Coefficient :math:`p_1` for the :math:`x^5` term.

    p2 : `float`
        Coefficient :math:`p_2` for the :math:`x^4` term.

    p3 : `float`
        Coefficient :math:`p_3` for the :math:`x^3` term.

    p4 : `float`
        Coefficient :math:`p_4` for the :math:`x^2` term.

    p5 : `float`
        Coefficient :math:`p_5` for the :math:`x` term.

    p6 : `float`
        Constant term :math:`p_6`.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return poly(x, p6, p5, p4, p3, p2, p1)

def poly6(
        x: np.ndarray, p1: float, p2: float, p3: float, p4: float, p5: float,
        p6: float, p7: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`poly6` model from MATLAB:

    .. math::

       y=p_1x^6+p_2x^5+\cdots+p_7

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    p1 : `float`
        Coefficient :math:`p_1` for the :math:`x^6` term.

    p2 : `float`
        Coefficient :math:`p_2` for the :math:`x^5` term.

    p3 : `float`
        Coefficient :math:`p_3` for the :math:`x^4` term.

    p4 : `float`
        Coefficient :math:`p_4` for the :math:`x^3` term.

    p5 : `float`
        Coefficient :math:`p_5` for the :math:`x^2` term.

    p6 : `float`
        Coefficient :math:`p_6` for the :math:`x` term.

    p7 : `float`
        Constant term :math:`p_7`.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return poly(x, p7, p6, p5, p4, p3, p2, p1)

def poly7(
        x: np.ndarray, p1: float, p2: float, p3: float, p4: float, p5: float,
        p6: float, p7: float, p8: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`poly7` model from MATLAB:

    .. math::

       y=p_1x^7+p_2x^6+\cdots+p_8

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    p1 : `float`
        Coefficient :math:`p_1` for the :math:`x^7` term.

    p2 : `float`
        Coefficient :math:`p_2` for the :math:`x^6` term.

    p3 : `float`
        Coefficient :math:`p_3` for the :math:`x^5` term.

    p4 : `float`
        Coefficient :math:`p_4` for the :math:`x^4` term.

    p5 : `float`
        Coefficient :math:`p_5` for the :math:`x^3` term.

    p6 : `float`
        Coefficient :math:`p_6` for the :math:`x^2` term.

    p7 : `float`
        Coefficient :math:`p_7` for the :math:`x` term.

    p8 : `float`
        Constant term :math:`p_8`.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return poly(x, p8, p7, p6, p5, p4, p3, p2, p1)

def poly8(
        x: np.ndarray, p1: float, p2: float, p3: float, p4: float, p5: float,
        p6: float, p7: float, p8: float, p9: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`poly8` model from MATLAB:

    .. math::

       y=p_1x^8+p_2x^7+\cdots+p_9

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    p1 : `float`
        Coefficient :math:`p_1` for the :math:`x^8` term.

    p2 : `float`
        Coefficient :math:`p_2` for the :math:`x^7` term.

    p3 : `float`
        Coefficient :math:`p_3` for the :math:`x^6` term.

    p4 : `float`
        Coefficient :math:`p_4` for the :math:`x^5` term.

    p5 : `float`
        Coefficient :math:`p_5` for the :math:`x^4` term.

    p6 : `float`
        Coefficient :math:`p_6` for the :math:`x^3` term.

    p7 : `float`
        Coefficient :math:`p_7` for the :math:`x^2` term.

    p8 : `float`
        Coefficient :math:`p_8` for the :math:`x` term.

    p9 : `float`
        Constant term :math:`p_9`.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return poly(x, p9, p8, p7, p6, p5, p4, p3, p2, p1)

def poly9(
        x: np.ndarray, p1: float, p2: float, p3: float, p4: float, p5: float,
        p6: float, p7: float, p8: float, p9: float, p10: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`poly9` model from MATLAB:

    .. math::

       y=p_1x^9+p_2x^8+\cdots+p_{10}

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    p1 : `float`
        Coefficient :math:`p_1` for the :math:`x^9` term.

    p2 : `float`
        Coefficient :math:`p_2` for the :math:`x^8` term.

    p3 : `float`
        Coefficient :math:`p_3` for the :math:`x^7` term.

    p4 : `float`
        Coefficient :math:`p_4` for the :math:`x^6` term.

    p5 : `float`
        Coefficient :math:`p_5` for the :math:`x^5` term.

    p6 : `float`
        Coefficient :math:`p_6` for the :math:`x^4` term.

    p7 : `float`
        Coefficient :math:`p_7` for the :math:`x^3` term.

    p8 : `float`
        Coefficient :math:`p_9` for the :math:`x^2` term.

    p9 : `float`
        Coefficient :math:`p_9` for the :math:`x` term.

    p10 : `float`
        Constant term :math:`p_{10}`.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return poly(x, p10, p9, p8, p7, p6, p5, p4, p3, p2, p1)