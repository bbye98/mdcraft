"""
Plot colors
===========
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module provides functions to select colors for plots.
"""

import colorsys
from typing import Union

import matplotlib.colors as mc

def adjust_lightness(
        colors: Union[str, tuple[float], list[Union[str, tuple[float]]]],
        amount: float) -> Union[tuple[float], list[tuple[float]]]:

    """
    Adjusts the lightness of colors.

    Parameters
    ----------
    color : `str`, `tuple`, or `list`.
        The colors to adjust. A single color can be specified as a tuple
        of normalized RGB values or a string containing its name or a
        hexadecimal value. Multiple colors can be provided in a `list`.

        **Examples**: :code:`"aquamarine"`, :code:`"#080085"`,
        :code:`(0.269, 0.269, 0.269)`.

    amount : `float`
        The amount to adjust the luminosity by. A value betwen :math:`0`
        and :math:`1` darkens the color, while a value greater than 
        :math:`1` lightens the color.

    Returns
    -------
    colors : `tuple` or `list`
        The adjusted colors.
    """

    if isinstance(colors, list):
        for i, color in enumerate(colors):
            colors[i] = adjust_lightness(color, amount)
        return colors

    colors = colorsys.rgb_to_hls(
        *mc.to_rgb(mc.cnames[colors] if colors in mc.cnames else colors)
    )
    return colorsys.hls_to_rgb(colors[0], max(0, min(1, amount * colors[1])),
                               colors[2])