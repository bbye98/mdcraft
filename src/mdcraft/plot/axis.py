"""
Axis components
===============
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module provides additional functionality for Matplotlib axes.
"""

from typing import Any

import matplotlib as mpl
import numpy as np

def set_up_tabular_legend(
        rows: list[str], cols: list[str], *, hlabel: str = None,
        vlabel: str = None, hla: str = "left", vla: str = "top",
        condense: bool = False, **kwargs) -> tuple[dict[str, Any], int, int]:

    r"""
    Sets up a tabular legend for a :class:`matplotlib.axes.Axes` object.

    Parameters
    ----------
    rows : `tuple` or `list`
        Raw string representations of the row values.

    cols : `tuple` or `list`
        Raw string representations of the column values.

    hlabel : `str`, keyword-only, optional
        Horizontal label for column values.

    vlabel : `str`, keyword-only, optional
        Vertical label for row values.

    hla : `str`, keyword-only, default: :code:`"left"`
        Alignment for `hlabel`.

        .. container::

           **Valid values**:

           * :code:`"left"`: Left-aligned text.
           * :code:`"center"`: Horizontally centered text.

    vla : `str`, keyword-only, default: :code:`"top"`
        Alignment for `vlabel`.

        .. container::

           **Valid values**:

           * :code:`"top"`: Top-aligned text.
           * :code:`"center"`: Vertically centered text.

    condense : `bool`, keyword-only, default: :code:`False`
        Condenses the legend by placing `vlabel` in the empty top-left
        corner. Cannot be used when no `vlabel` is specified or in
        conjuction with :code:`vla="center"` (which will take priority).

    **kwargs :
        Keyword arguments passed to :meth:`matplotlib.axes.Axes.legend`.

    Returns
    -------
    properties : `dict`
        Properties of the tabular legend to be unpacked and used in the
        :meth:`matplotlib.axes.Axes.legend` call.

        .. container::

           * handles (`list`): :obj:`matplotlib.artist` objects to be
             added to the legend.
           * labels (`list`): Labels to be shown next to the
             :obj:`matplotlib.artist` objects in the legend.
           * ncol (`int`): Number of columns in the legend.
           * kwargs (`dict`): Keyword arguments passed to
             :meth:`matplotlib.axes.Axes.legend`.

    nrow : `int`
        Number of rows in the legend.

    idx_start : `int`
        Index at which to start storing handles for
        :obj:`matplotlib.artist` objects.

    Notes
    -----
    Condensing the legend can cause alignment issues in the first column
    containing the row values if the row values are not of comparable
    length to the rows label. An easy but imprecise fix is to center the
    shorter values using the width of the largest
    :class:`matplotlib.transforms.Bbox` in the first column. Sample code
    for use in your main script utilizing the outputs of this function is
    provided below:

    .. code-block::

       (props, nrow, start) = tabular_legend(..., handletextpad=-5/4)
       fig, ax = plt.subplots(...)
       for i, x in enumerate(...):
           for j, y in enumerate(...):
               props["handles"][start + i * nrow + j], = ax.plot(...)
       ax.set_xlabel(...)
       ax.set_ylabel(...)
       lgd = ax.legend(**props)
       fig.canvas.draw()
       texts = lgd.get_texts()[:nrow]
       bounds = [t.get_window_extent().bounds[2] / 2 for t in texts]
       center = max(bounds)
       for k, t in enumerate(texts):
           t.set_position((center - bounds[k], 0))
       plt.show()
    """

    hpad = bool(vlabel) - condense + 1
    vpad = bool(hlabel) + 1
    nrow = len(rows) + vpad
    ncol = len(cols) + hpad

    labels = ["" for _ in range(nrow * ncol)]
    if vlabel:
        labels[vpad + (len(rows) // 2 if vla == "center" else -condense)] = vlabel
    iv = vpad + nrow * (bool(vlabel) - condense)
    labels[iv:iv + len(rows)] = rows
    if hlabel:
        labels[
            (2 + (hla == "center") * (int(np.ceil(len(cols) / 2)) - 1)) * nrow
        ] = hlabel
    labels[hpad * nrow + bool(hlabel)::nrow] = cols

    return {
        "handles": [mpl.patches.Rectangle((0, 0), 0.1, 0.1, ec="none", fill=False)
                    for _ in range(len(labels))],
        "labels": labels,
        "ncol": ncol,
        **kwargs
    }, nrow, iv + nrow