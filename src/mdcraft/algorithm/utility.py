"""
Utility algorithms
==================
.. moduleauthor:: Benjamin Ye <GitHub: @bbye98>

This module contains logical and mathematical utility functions used by
other MDCraft modules.
"""

from typing import Any

import numpy as np
import sympy

def get_closest_factors(
        value: int, n_factors: int, reverse: bool = False
    ) -> np.ndarray[int]:

    """
    Finds the :math:`n` closest factors for a given number :math:`N`,
    sorted in ascending order.

    Parameters
    ----------
    value : `int`
        Number :math:`N` to be factorized.

    n_factors : `int`
        Number of factors :math:`n` to return.

    reverse : `bool`, optional, default: :code:`False`
        Specifies whether to sort in descending order.

    Returns
    -------
    factors : `np.ndarray`
        :math:`n` closest factors for `N`.

        **Shape**: :math:`(n,)`.
    """

    # Take the n-th root of N
    rt = value ** (1 / n_factors)
    rt_int = int(np.round(rt))
    if np.isclose(rt, rt_int):
        return rt_int * np.ones(n_factors, dtype=int)

    # Get all factors of N
    _factors = np.fromiter(
        (factor for factor, power in sympy.ntheory.factorint(value).items()
         for _ in range(power)),
        dtype=int
    )

    # Find n closest factors
    i = 0
    factors = np.ones(n_factors, dtype=int)
    for j, f in enumerate(_factors[::-1]):
        while True:
            if i < n_factors:
                m = factors[i] * f
                if m <= rt_int or j < n_factors and factors[i] == 1:
                    factors[i] = m
                    break
                i += 1
            else:
                factors[np.argmin(factors)] *= f
                break

    if reverse:
        return np.sort(factors)[::-1]
    return np.sort(factors)

def replicate(
        cell_dims: np.ndarray[float], cell_pos: np.ndarray[float],
        n_cells: np.ndarray[int]) -> np.ndarray[float]:

    r"""
    Replicates points in an unit cell along the :math:`x`-, :math:`y`-,
    and :math:`z`-directions.

    Parameters
    ----------
    cell_dims : `numpy.ndarray`
        Dimensions of the unit cell.

        **Shape**: :math:`(3,)`.

    cell_pos : `numpy.ndarray`
        Positions of the :math:`N` points inside the unit cell.

        **Shape**: :math:`(N,\,3)`.

    n_cells : `numpy.ndarray`
        Number of times to replicate the unit cell in each direction.

        **Shape**: :math:`(3,)`.

    Returns
    -------
    pos : `numpy.ndarray`
        Positions of the original and replicated points.
    """

    # Add cell x-dimensions to cell x-positions and replicate them
    # n_y * n_z times
    x = np.tile(
        np.concatenate(cell_pos[:, 0]
                       + (cell_dims[0] * np.arange(n_cells[0]))[:, None]),
        reps=n_cells[1] * n_cells[2]
    )

    # Replicate cell y-positions n_x times, add cell y-dimensions to
    # them, and then replicate them n_z times
    y = np.tile(
        np.concatenate(np.tile(cell_pos[:, 1], reps=n_cells[0])
                       + (np.arange(n_cells[1]) * cell_dims[1])[:, None]),
        reps=n_cells[2]
    )

    # Replicate cell z-positions n_x * n_y times and add cell
    # z-dimensions to them
    z = np.concatenate(np.tile(cell_pos[:, 2], reps=n_cells[0] * n_cells[1])
                       + cell_dims[2] * np.arange(n_cells[2])[:, None])

    return np.vstack((x, y, z)).T

def rebin(x: np.ndarray[float], factor: int = None) -> np.ndarray[float]:

    r"""
    Rebins discrete data.

    Parameters
    ----------
    x : `numpy.ndarray`
        Discrete data to be rebinned in the last dimension.

    factor : `int`, optional
        Size reduction factor. If not specified, the biggest factor
        on the order of :math:`\mathcal{O}(1)`, if available, is used.

    Returns
    -------
    xr : `numpy.ndarray`
        Rebinned discrete data.
    """

    if factor is None:
        factors = np.array(sympy.divisors(x.shape[-1])[1:])
        factor_indices = np.where(factors < 10)[0]
        if len(factor_indices):
            factor = factors[factor_indices[-1]]
        else:
            raise ValueError("No factor provided for rebinning.")

    return x.reshape((*x.shape[:-1], -1, factor)).mean(axis=-1)

def depth_first_search(
        graph: dict[Any, list[Any]], start: Any, visited: bool,
        group: list[Any]) -> None:

    """
    Implements the depth-first search algorithm to find connected
    components in a graph.

    Parameters
    ----------
    graph : `dict`
        Graph to search.

    start : `Any`
        Starting node.

    visited : `dict`
        Visited nodes. This `dict` is updated in-place.

    group : `list`
        Connected nodes. This `list` is updated in-place.
    """

    visited[start] = True
    group.append(start)
    for neighbor in graph[start]:
        if not visited[neighbor]:
            depth_first_search(graph, neighbor, visited, group)

def find_connected_nodes(graph: dict[Any, list[Any]]) -> list[list[Any]]:

    """
    Finds connected components in a graph.

    Parameters
    ----------
    graph : `dict`
        Graph to search.

    Returns
    -------
    results : `list`
        Connected components.
    """

    visited = {node: False for node in graph}
    results = []
    for start in graph:
        if not visited[start]:
            group = []
            depth_first_search(graph, start, visited, group)
            results.append(group)
    return results