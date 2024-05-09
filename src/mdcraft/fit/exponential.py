r"""
Exponential models
==================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

Exponentials are often used when the rate of change of a quantity is
proportional to the initial amount of the quantity. The general
exponential model is given by

.. math::

   y=\sum_{k=1}^na_k\exp{(b_kx)}

If the coefficient :math:`b_k` for an :math:`\exp` term is negative,
that term represents exponential decay. If the coefficient is positive,
that term represents exponential growth.

This module provides the general exponential model above for any number
of terms :math:`k`, as well as convenience functions for the one-
(:math:`k=1`) and two-term (:math:`k=2`) exponential models
analogous to MATLAB's :code:`exp1` and :code:`exp2`, respectively.

Additionally, this module has the stretched exponential function, also
known as the complementary cumulative Weibull distribution, given by

.. math::

   y=\exp{\left[-\left(\frac{x}{\alpha}\right)^\beta\right]}

where :math:`\beta` is the stretching exponent. This expression is
obtained by inserting a fractional power law into the exponential
function.

This model is generally meaningful only for :math:`x>0`. The graph of
:math:`\log{(y)}` vs. :math:`x` is characteristically stretched when
:math:`0\leq\beta\leq 1` and compressed when :math:`\beta>1` (the latter
case has less practical importance). When :math:`\beta=1`, the one-term
exponential model is recovered. When :math:`\beta=2`, the probability
density function for the normal distribution is obtained.
"""

import numpy as np

def exp(x: np.ndarray, *args: float) -> np.ndarray:

    r"""
    General exponential model.

    .. math::

       y=\sum_{k=1}^na_k\exp{(b_kx)}

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    *args : `float`
        Fitting parameters for the exponential term(s), ordered as
        :math:`a_1,\,b_1,\,a_2,\,b_2,\ldots,\,a_n,\,b_n`, where
        :math:`n` is the number of terms in the model. As such, the
        number of variable positional arguments must be even.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.

    Examples
    --------
    Generate :math:`x`- and :math:`y`-values (with error), and then use
    :func:`scipy.optimize.curve_fit` to fit coefficients for a two-term
    exponential model.

    >>> from scipy import optimize
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-0.1, 0.1, 10)
    >>> err = (2 * rng.random(x.shape) - 1) / 10
    >>> y = np.exp(-8 * x) + np.exp(12 * x) + err
    >>> pk, _ = optimize.curve_fit(
            lambda x, a1, b1, a2, b2: exp(x, a1, b1, a2, b2), x, y
        )
    >>> pk
    array([ 1.13072662, -6.90042351,  0.88706719, 12.87854508])

    Evaluate the fitted :math:`y`-values using the coefficients.

    >>> exp(x, *pk)
    array([2.49915084, 2.25973234, 2.09274413, 2.00061073, 1.98962716,
           2.07080543, 2.26106343, 2.58486089, 3.07642312, 3.78274065])
    """

    n = len(args)
    if n < 2 or n % 2 != 0:
        emsg = "Number of fitting parameters must be greater than 2 and even."
        raise ValueError(emsg)
    return np.exp(args[1::2] * x[:, None]) @ args[::2]

def exp1(x: np.ndarray, a: float, b: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`exp1` model from MATLAB.

    .. math::

       y=a\exp{(bx)}

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    a : `float`
        Coefficient :math:`a` for the :math:`\exp` term.

    b : `float`
        Coefficient :math:`b` for the :math:`x` term in the :math:`\exp`
        term.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return exp(x, a, b)

def exp2(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`exp2` model from MATLAB.

    .. math::

       y=a\exp{(bx)}+c\exp{(dx)}

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    a : `float`
        Coefficient :math:`a` for the first :math:`\exp` term.

    b : `float`
        Coefficient :math:`b` for the :math:`x` term in the first
        :math:`\exp` term.

    c : `float`
        Coefficient :math:`a` for the second :math:`\exp` term.

    d : `float`
        Coefficient :math:`b` for the :math:`x` term in the second
        :math:`\exp` term.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return exp(x, a, b, c, d)

def biexp(
        x: np.ndarray, y0: float, a: float, b: float, c: float, d: float
    ) -> np.ndarray:

    r"""
    Bi-exponential function.

    .. math::

       y=y_0+a\exp{\left(-\frac{x}{b}\right)}+c\exp{\left(-\frac{x}{d}\right)}

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    y0 : `float`
        Offset :math:`y_0`.

    a : `float`
        Coefficient :math:`a` for the first :math:`\exp` term.

    b : `float`
        Coefficient :math:`b` for the :math:`x` term in the first
        :math:`\exp` term.

    c : `float`
        Coefficient :math:`a` for the second :math:`\exp` term.

    d : `float`
        Coefficient :math:`b` for the :math:`x` term in the second
        :math:`\exp` term.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return y0 + a * np.exp(-x / b) + c * np.exp(-x / d)

def stretched_exp(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:

    r"""
    Stretched exponential function.

    .. math::

       y=\exp{\left[-\left(\frac{x}{\alpha}\right)^\beta\right]}

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    alpha : `float`
        Scaling parameter :math:`\alpha` for :math:`x`.

    beta : `float`
        Stretching exponent :math:`\beta`.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return np.exp(-(x / alpha) ** beta)