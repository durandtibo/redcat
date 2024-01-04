from __future__ import annotations

__all__ = [
    "argsort_along_batch",
    "array_equal",
    "equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "not_equal",
]

from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from redcat.ba.core import BatchedArray

TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")

#################################
#     Comparison operations     #
#################################

equal = np.equal
greater = np.greater
greater_equal = np.greater_equal
less = np.less
less_equal = np.less_equal
not_equal = np.not_equal


def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan: bool = False) -> bool:
    r"""Indicate if two arrays are equal.

    Args:
        a1: Specifies the first array.
        a2: Specifies the second array.
        equal_nan: Whether to compare NaN’s as equal.
            If the dtype of a1 and a2 is complex, values will be
            considered equal if either the real or the imaginary
            component of a given value is nan.

    Returns:
        ``True`` if the arrays are equal, otherwise ``False``.

    Example usage:

    ```pycon
    >>> from redcat import ba
    >>> x1 = ba.ones((2, 3))
    >>> x2 = ba.ones((2, 3))
    >>> x3 = ba.zeros((2, 3))
    >>> ba.array_equal(x1, x2)
    True
    >>> ba.array_equal(x1, x3)
    False

    ```
    """
    if isinstance(a1, BatchedArray):
        return a1.allequal(a2, equal_nan)
    return np.array_equal(a1, a2, equal_nan)


###########################################
#     Item selection and manipulation     #
###########################################


def argsort_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> TBatchedArray:
    r"""Sort the elements of the batch along the batch axis in monotonic
    order by value.

    Args:
        a: Array to sort.
        args: See the documentation of ``numpy.argsort``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.argsort``.
            ``axis`` should not be passed.

    Returns:
        The indices that sort the batch along the batch dimension.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> x = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> y = ba.argsort_along_batch(x)
    >>> y
    array([[0, 0],
           [1, 1],
           [2, 2],
           [3, 3],
           [4, 4]], batch_axis=0)
    >>> x = ba.BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> y = ba.argsort_along_batch(x)
    >>> y
    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]], batch_axis=1)

    ```
    """
    return a.argsort_along_batch(*args, **kwargs)
