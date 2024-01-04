from __future__ import annotations

__all__ = [
    "argsort_along_batch",
    "equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "not_equal",
]

from typing import Any, TypeVar

import numpy as np

from redcat.ba.core import BatchedArray

TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")

#################################
#     Comparison operations     #
#################################

# TODO: array_equal

equal = np.equal
greater = np.greater
greater_equal = np.greater_equal
less = np.less
less_equal = np.less_equal
not_equal = np.not_equal


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
