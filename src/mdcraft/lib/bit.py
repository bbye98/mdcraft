from numba import njit
import numpy as np

from .. import int_t


@njit(fastmath=True, inline="always")
def count_leading_zeros(x: int | int_t, bits: int | int_t) -> int:
    if x == 0:
        return bits
    n = 0
    mask = 1 << (bits - 1)
    while (x & mask) == 0:
        n += 1
        x <<= 1
    return n
