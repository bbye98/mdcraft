from numba import njit

from .. import int_t


@njit(fastmath=True, inline="always")
def count_leading_zeros(x: int | int_t, bits: int | int_t) -> int:
    """
    Counts the number of leading zeros in the binary representation of
    an integer.

    Parameters
    ----------
    x : `int`
        An integer to count leading zeros in.

    bits : `int`
        The number of bits in the binary representation.

    Returns
    -------
    clz : `int`
        The number of leading zeros in the binary representation of `x`.
        If `x` is zero, `bits` is returned.

    Examples
    --------
    >>> n = 1
    >>> f"{n:0>32b}"
    '00000000000000000000000000000001'
    >>> count_leading_zeros(n, 32)
    31
    >>> n = 2 ** 63 - 1
    >>> f"{n:0>64b}"
    '0111111111111111111111111111111111111111111111111111111111111111'
    >>> count_leading_zeros(n, 64)
    1
    """
    # Prevents operations with mismatched integer types from promoting
    # the return value to a floating-point number
    bits = type(x)(bits)

    if x == 0:
        return bits
    n = 0
    mask = 1 << (bits - 1)
    while (x & mask) == 0:
        n += 1
        x <<= 1
    return n
