from enum import IntEnum


class IndexBase(IntEnum):
    """
    Base index for time series data.

    ZERO: int
        First element is indexed from 0.
    ONE: int
        First element is indexed from 1.
    """

    ZERO = 0
    ONE = 1
