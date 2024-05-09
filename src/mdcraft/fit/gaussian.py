r"""
Gaussian models
===============
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

The Gaussian model fits peaks, and is given by

.. math::

   y=\sum_{k=1}^na_k\exp{\left[-\left(\frac{x-b_k}{c_k}\right)^2\right]}

where :math:`a` is the amplitude, :math:`b` is the centroid (location),
:math:`c` is related to the peak width, and :math:`n` is the number of
peaks to fit.

This module provides the general Gaussian model above for any number of
terms :math:`k`, as well as convenience functions for Gaussian models
with :math:`1\leq n\leq8` analogous to MATLAB's :code:`gauss1`,
:code:`gauss2`, etc.
"""

import numpy as np

def gauss(x: np.ndarray, *args: float) -> np.ndarray:

    r"""
    General Gaussian model.

    .. math::

       y=\sum_{k=1}^na_k\exp{\left[
       -\left(\frac{x-b_k}{c_k}\right)^2\right]}

    Parameters
    ----------
    x : `numpy.ndarray`
        :math:`x`-values.

    *args : `float`
        Fitting parameters for the Gaussian term(s), ordered as
        :math:`a_1,\,b_1,\,c_1,\,a_2,\,b_2,\,c_2,\ldots,\,a_n,\,b_n,\,c_n`,
        where :math:`n` is the number of terms in the model. As such,
        the number of variable positional arguments must be divisible by
        :math:`3`.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.

    Examples
    --------
    Generate :math:`x`- and :math:`y`-values (with error), and then use
    :func:`scipy.optimize.curve_fit` to fit coefficients for a one-term
    Gaussian model.

    >>> from scipy import optimize
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-3, 7, 10)
    >>> err = (2 * rng.random(x.shape) - 1) / 10
    >>>y=np.exp(-((x - 2) / 3) ** 2) + err
    >>> pk, _ = optimize.curve_fit(lambda x, a1, b1, c1: gauss(x, a1, b1, c1), x, y)
    >>> pk
    array([1.07262377, 1.90290018, 2.90033242])

    Evaluate the fitted :math:`y`-values using the coefficients.

    >>> gauss(x, *pk)
    array([0.06157175, 0.19415629, 0.45650326, 0.80031095, 1.04615509,
           1.01966105, 0.74103382, 0.40155281, 0.16224441, 0.04887866])
    """

    n = len(args)
    assert n >= 3 and n % 3 == 0, \
        "Number of fitting parameters must be greater than and divisible by 3."
    return np.exp(-((x[:, None] - args[1::3]) / args[2::3]) ** 2) @ args[::3]

def gauss1(x: np.ndarray, a1: float, b1: float, c1: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`gauss1` model from MATLAB.

    .. math::

       y=a_1\exp{\left[-\left(\frac{x-b_1}{c_1}\right)^2\right]}

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    a1 : `float`
        Amplitude :math:`a_1` of the first Gaussian term.

    b1 : `float`
        Centroid :math:`b_1` of the first Gaussian term.

    c1 : `float`
        Peak width :math:`c_1` of the first Gaussian term.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return gauss(x, a1, b1, c1)

def gauss2(
        x: np.ndarray, a1: float, b1: float, c1: float, a2: float, b2: float,
        c2: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`gauss2` model from MATLAB:

    .. math::

       y=a_1\exp{\left[-\left(\frac{x-b_1}{c_1}\right)^2\right]}
       +a_2\exp{\left[-\left(\frac{x-b_2}{c_2}\right)^2\right]}

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    a1 : `float`
        Amplitude :math:`a_1` of the first Gaussian term.

    b1 : `float`
        Centroid :math:`b_1` of the first Gaussian term.

    c1 : `float`
        Peak width :math:`c_1` of the first Gaussian term.

    a2 : `float`
        Amplitude :math:`a_2` of the second Gaussian term.

    b2 : `float`
        Centroid :math:`b_2` of the second Gaussian term.

    c2 : `float`
        Peak width :math:`c_2` of the second Gaussian term.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return gauss(x, a1, b1, c1, a2, b2, c2)

def gauss3(
        x: np.ndarray, a1: float, b1: float, c1: float, a2: float, b2: float,
        c2: float, a3: float, b3: float, c3: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`gauss3` model from MATLAB.

    .. math::

       y=a_1\exp{\left[-\left(\frac{x-b_1}{c_1}\right)^2\right]}
       +a_2\exp{\left[-\left(\frac{x-b_2}{c_2}\right)^2\right]}
       +a_3\exp{\left[-\left(\frac{x-b_3}{c_3}\right)^2\right]}

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    a1 : `float`
        Amplitude :math:`a_1` of the first Gaussian term.

    b1 : `float`
        Centroid :math:`b_1` of the first Gaussian term.

    c1 : `float`
        Peak width :math:`c_1` of the first Gaussian term.

    a2 : `float`
        Amplitude :math:`a_2` of the second Gaussian term.

    b2 : `float`
        Centroid :math:`b_2` of the second Gaussian term.

    c2 : `float`
        Peak width :math:`c_2` of the second Gaussian term.

    a3 : `float`
        Amplitude :math:`a_3` of the third Gaussian term.

    b3 : `float`
        Centroid :math:`b_3` of the third Gaussian term.

    c3 : `float`
        Peak width :math:`c_3` of the third Gaussian term.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return gauss(x, a1, b1, c1, a2, b2, c2, a3, b3, c3)

def gauss4(
        x: np.ndarray, a1: float, b1: float, c1: float, a2: float, b2: float,
        c2: float, a3: float, b3: float, c3: float, a4: float, b4: float,
        c4: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`gauss4` model from MATLAB.

    .. math::

       y=a_1\exp{\left[-\left(\frac{x-b_1}{c_1}\right)^2\right]}
       +a_2\exp{\left[-\left(\frac{x-b_2}{c_2}\right)^2\right]}
       +a_3\exp{\left[-\left(\frac{x-b_3}{c_3}\right)^2\right]}
       +a_4\exp{\left[-\left(\frac{x-b_4}{c_4}\right)^2\right]}

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    a1 : `float`
        Amplitude :math:`a_1` of the first Gaussian term.

    b1 : `float`
        Centroid :math:`b_1` of the first Gaussian term.

    c1 : `float`
        Peak width :math:`c_1` of the first Gaussian term.

    a2 : `float`
        Amplitude :math:`a_2` of the second Gaussian term.

    b2 : `float`
        Centroid :math:`b_2` of the second Gaussian term.

    c2 : `float`
        Peak width :math:`c_2` of the second Gaussian term.

    a3 : `float`
        Amplitude :math:`a_3` of the third Gaussian term.

    b3 : `float`
        Centroid :math:`b_3` of the third Gaussian term.

    c3 : `float`
        Peak width :math:`c_3` of the third Gaussian term.

    a4 : `float`
        Amplitude :math:`a_4` of the fourth Gaussian term.

    b4 : `float`
        Centroid :math:`b_4` of the fourth Gaussian term.

    c4 : `float`
        Peak width :math:`c_4` of the fourth Gaussian term.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return gauss(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4)

def gauss5(
        x: np.ndarray, a1: float, b1: float, c1: float, a2: float, b2: float,
        c2: float, a3: float, b3: float, c3: float, a4: float, b4: float,
        c4: float, a5: float, b5: float, c5: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`gauss5` model from MATLAB.

    .. math::

       y=a_1\exp{\left[-\left(\frac{x-b_1}{c_1}\right)^2\right]}
       +a_2\exp{\left[-\left(\frac{x-b_2}{c_2}\right)^2\right]}
       +\cdots+a_5\exp{\left[-\left(\frac{x-b_5}{c_5}\right)^2\right]}

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    a1 : `float`
        Amplitude :math:`a_1` of the first Gaussian term.

    b1 : `float`
        Centroid :math:`b_1` of the first Gaussian term.

    c1 : `float`
        Peak width :math:`c_1` of the first Gaussian term.

    a2 : `float`
        Amplitude :math:`a_2` of the second Gaussian term.

    b2 : `float`
        Centroid :math:`b_2` of the second Gaussian term.

    c2 : `float`
        Peak width :math:`c_2` of the second Gaussian term.

    a3 : `float`
        Amplitude :math:`a_3` of the third Gaussian term.

    b3 : `float`
        Centroid :math:`b_3` of the third Gaussian term.

    c3 : `float`
        Peak width :math:`c_3` of the third Gaussian term.

    a4 : `float`
        Amplitude :math:`a_4` of the fourth Gaussian term.

    b4 : `float`
        Centroid :math:`b_4` of the fourth Gaussian term.

    c4 : `float`
        Peak width :math:`c_4` of the fourth Gaussian term.

    a5 : `float`
        Amplitude :math:`a_5` of the fifth Gaussian term.

    b5 : `float`
        Centroid :math:`b_5` of the fifth Gaussian term.

    c5 : `float`
        Peak width :math:`c_5` of the fifth Gaussian term.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return gauss(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4,
                 a5, b5, c5)

def gauss6(
        x: np.ndarray, a1: float, b1: float, c1: float, a2: float, b2: float,
        c2: float, a3: float, b3: float, c3: float, a4: float, b4: float,
        c4: float, a5: float, b5: float, c5: float, a6: float, b6: float,
        c6: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`gauss6` model from MATLAB.

    .. math::

       y=a_1\exp{\left[-\left(\frac{x-b_1}{c_1}\right)^2\right]}
       +a_2\exp{\left[-\left(\frac{x-b_2}{c_2}\right)^2\right]}
       +\cdots+a_6\exp{\left[-\left(\frac{x-b_6}{c_6}\right)^2\right]}

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    a1 : `float`
        Amplitude :math:`a_1` of the first Gaussian term.

    b1 : `float`
        Centroid :math:`b_1` of the first Gaussian term.

    c1 : `float`
        Peak width :math:`c_1` of the first Gaussian term.

    a2 : `float`
        Amplitude :math:`a_2` of the second Gaussian term.

    b2 : `float`
        Centroid :math:`b_2` of the second Gaussian term.

    c2 : `float`
        Peak width :math:`c_2` of the second Gaussian term.

    a3 : `float`
        Amplitude :math:`a_3` of the third Gaussian term.

    b3 : `float`
        Centroid :math:`b_3` of the third Gaussian term.

    c3 : `float`
        Peak width :math:`c_3` of the third Gaussian term.

    a4 : `float`
        Amplitude :math:`a_4` of the fourth Gaussian term.

    b4 : `float`
        Centroid :math:`b_4` of the fourth Gaussian term.

    c4 : `float`
        Peak width :math:`c_4` of the fourth Gaussian term.

    a5 : `float`
        Amplitude :math:`a_5` of the fifth Gaussian term.

    b5 : `float`
        Centroid :math:`b_5` of the fifth Gaussian term.

    c5 : `float`
        Peak width :math:`c_5` of the fifth Gaussian term.

    a6 : `float`
        Amplitude :math:`a_6` of the sixth Gaussian term.

    b6 : `float`
        Centroid :math:`b_6` of the sixth Gaussian term.

    c6 : `float`
        Peak width :math:`c_6` of the sixth Gaussian term.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return gauss(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4,
                 a5, b5, c5, a6, b6, c6)

def gauss7(
        x: np.ndarray, a1: float, b1: float, c1: float, a2: float, b2: float,
        c2: float, a3: float, b3: float, c3: float, a4: float, b4: float,
        c4: float, a5: float, b5: float, c5: float, a6: float, b6: float,
        c6: float, a7: float, b7: float, c7: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`gauss7` model from MATLAB.

    .. math::

       y=a_1\exp{\left[-\left(\frac{x-b_1}{c_1}\right)^2\right]}
       +a_2\exp{\left[-\left(\frac{x-b_2}{c_2}\right)^2\right]}
       +\cdots+a_7\exp{\left[-\left(\frac{x-b_7}{c_7}\right)^2\right]}

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    a1 : `float`
        Amplitude :math:`a_1` of the first Gaussian term.

    b1 : `float`
        Centroid :math:`b_1` of the first Gaussian term.

    c1 : `float`
        Peak width :math:`c_1` of the first Gaussian term.

    a2 : `float`
        Amplitude :math:`a_2` of the second Gaussian term.

    b2 : `float`
        Centroid :math:`b_2` of the second Gaussian term.

    c2 : `float`
        Peak width :math:`c_2` of the second Gaussian term.

    a3 : `float`
        Amplitude :math:`a_3` of the third Gaussian term.

    b3 : `float`
        Centroid :math:`b_3` of the third Gaussian term.

    c3 : `float`
        Peak width :math:`c_3` of the third Gaussian term.

    a4 : `float`
        Amplitude :math:`a_4` of the fourth Gaussian term.

    b4 : `float`
        Centroid :math:`b_4` of the fourth Gaussian term.

    c4 : `float`
        Peak width :math:`c_4` of the fourth Gaussian term.

    a5 : `float`
        Amplitude :math:`a_5` of the fifth Gaussian term.

    b5 : `float`
        Centroid :math:`b_5` of the fifth Gaussian term.

    c5 : `float`
        Peak width :math:`c_5` of the fifth Gaussian term.

    a6 : `float`
        Amplitude :math:`a_6` of the sixth Gaussian term.

    b6 : `float`
        Centroid :math:`b_6` of the sixth Gaussian term.

    c6 : `float`
        Peak width :math:`c_6` of the sixth Gaussian term.

    a7 : `float`
        Amplitude :math:`a_7` of the seventh Gaussian term.

    b7 : `float`
        Centroid :math:`b_7` of the seventh Gaussian term.

    c7 : `float`
        Peak width :math:`c_7` of the seventh Gaussian term.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return gauss(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4,
                 a5, b5, c5, a6, b6, c6, a7, b7, c7)

def gauss8(
        x: np.ndarray, a1: float, b1: float, c1: float, a2: float, b2: float,
        c2: float, a3: float, b3: float, c3: float, a4: float, b4: float,
        c4: float, a5: float, b5: float, c5: float, a6: float, b6: float,
        c6: float, a7: float, b7: float, c7: float, a8: float, b8: float,
        c8: float) -> np.ndarray:

    r"""
    Convenience function for the :code:`gauss8` model from MATLAB.

    .. math::

       y=a_1\exp{\left[-\left(\frac{x-b_1}{c_1}\right)^2\right]}
       +a_2\exp{\left[-\left(\frac{x-b_2}{c_2}\right)^2\right]}
       +\cdots+a_8\exp{\left[-\left(\frac{x-b_8}{c_8}\right)^2\right]}

    Parameters
    ----------
    x : `numpy.ndarray`
        One-dimensional array containing :math:`x`-values.

    a1 : `float`
        Amplitude :math:`a_1` of the first Gaussian term.

    b1 : `float`
        Centroid :math:`b_1` of the first Gaussian term.

    c1 : `float`
        Peak width :math:`c_1` of the first Gaussian term.

    a2 : `float`
        Amplitude :math:`a_2` of the second Gaussian term.

    b2 : `float`
        Centroid :math:`b_2` of the second Gaussian term.

    c2 : `float`
        Peak width :math:`c_2` of the second Gaussian term.

    a3 : `float`
        Amplitude :math:`a_3` of the third Gaussian term.

    b3 : `float`
        Centroid :math:`b_3` of the third Gaussian term.

    c3 : `float`
        Peak width :math:`c_3` of the third Gaussian term.

    a4 : `float`
        Amplitude :math:`a_4` of the fourth Gaussian term.

    b4 : `float`
        Centroid :math:`b_4` of the fourth Gaussian term.

    c4 : `float`
        Peak width :math:`c_4` of the fourth Gaussian term.

    a5 : `float`
        Amplitude :math:`a_5` of the fifth Gaussian term.

    b5 : `float`
        Centroid :math:`b_5` of the fifth Gaussian term.

    c5 : `float`
        Peak width :math:`c_5` of the fifth Gaussian term.

    a6 : `float`
        Amplitude :math:`a_6` of the sixth Gaussian term.

    b6 : `float`
        Centroid :math:`b_6` of the sixth Gaussian term.

    c6 : `float`
        Peak width :math:`c_6` of the sixth Gaussian term.

    a7 : `float`
        Amplitude :math:`a_7` of the seventh Gaussian term.

    b7 : `float`
        Centroid :math:`b_7` of the seventh Gaussian term.

    c7 : `float`
        Peak width :math:`c_7` of the seventh Gaussian term.

    a8 : `float`
        Amplitude :math:`a_8` of the eigth Gaussian term.

    b8 : `float`
        Centroid :math:`b_8` of the eigth Gaussian term.

    c8 : `float`
        Peak width :math:`c_8` of the eigth Gaussian term.

    Returns
    -------
    fit : `numpy.ndarray`
        Fitted :math:`y`-values.
    """

    return gauss(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4,
                 a5, b5, c5, a6, b6, c6, a7, b7, c7, a8, b8, c8)