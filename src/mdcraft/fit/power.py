r"""
Power models
============
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

The power model is given by

.. math::

   y=ax^b+c

This module provides the power model above and convenience functions for
the one- (:math:`c = 0`) and two-term power models in MATLAB,
:code:`power1` and :code:`power2`, respectively.
"""

import numpy as np

def power(x: np.ndarray, a: float, b: float, c: float = 0) -> np.ndarray:

    r"""
    General power model.

    .. math::

       y=ax^b+c

    Parameters
    ----------
    x : `numpy.ndarray`
        :math:`x`-values.

    a : `float`
        Coefficient for the :math:`x^b` term.

    b : `float`
        Power constant :math:`b` for the :math:`x^b` term.

    c : `float`, keyword-only, default: :code:`0`
        Constant for the :math:`y`-intercept.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return a * x ** b + c

def power1(x: np.ndarray, a: float, b: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`power1` model from MATLAB.

    .. math::

       y=ax^b

    Parameters
    ----------
    x : `numpy.ndarray`
        :math:`x`-values.

    a : `float`
        Coefficient for the :math:`x^b` term.

    b : `float`
        Power constant :math:`b` for the :math:`x^b` term.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return power(x, a, b)

def power2(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`power2` model from MATLAB.

    .. math::

       y=ax^b+c

    Parameters
    ----------
    x : `numpy.ndarray`
        :math:`x`-values.

    a : `float`
        Coefficient for the :math:`x^b` term.

    b : `float`
        Power constant :math:`b` for the :math:`x^b` term.

    c : `float`
        Constant for the :math:`y`-intercept.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return power(x, a, b, c)