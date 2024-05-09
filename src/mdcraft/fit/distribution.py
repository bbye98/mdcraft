r"""
Weibull distribution models
===========================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

The Weibull distribution is widely used in reliability and life (failure
rate) data analysis. This module provides the one-, two-, and
three-parameter Weibull distributions.

The three-parameter Weibull distribution is

.. math::

   y=ab(x-c)^{b-1}\exp{[-a(x-c)^b]}

where :math:`a` is the scale parameter, :math:`b` is the shape
parameter, and :math:`c` is the location parameter.

The two-parameter Weibull distribution

.. math::

   y=abx^{b-1}\exp{(-ax^b)}

has :math:`x-c` replaced with :math:`x`.

The one-parameter Weibull distribution has the shape parameter fixed,
so only the scale parameter is fitted.
"""

import numpy as np

def weibull(x: np.ndarray, a: float, b: float, c: float = 0) -> np.ndarray:

    r"""
    General three-parameter Weibull distribution.

    .. math::

       y=ab(x-c)^{b-1}\exp{[-a(x-c)^b]}

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    a : `float`
        Scale parameter :math:`a`.

    b : `float`
        Shape parameter :math:`b`. If specified to be a constant, the
        one-parameter Weibull distribution is used.

    c : `float`, keyword-only, default: :code:`0`
        Location parameter :math:`c`. If not specified as a parameter,
        the two-parameter Weibull distribution is used.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.

    Examples
    --------
    Create a three-parameter Weibull distribution model for fitting.

    >>> model = lambda x, a, b, c: weibull(x, a, b, c)

    Create a two-parameter Weibull distribution model for fitting.

    >>> model = lambda x, a, b: weibull(x, a, b)

    Create a one-parameter Weibull distribution model for fitting, with
    :math:`b = 1`.

    >>> model = lambda x, a: weibull(x, a, 1)
    """

    return a * b * (x - c) ** (b - 1) * np.exp(-a * (x - c) ** b)