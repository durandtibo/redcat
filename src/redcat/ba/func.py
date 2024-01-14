from __future__ import annotations

__all__ = [
    "allclose",
    "argsort",
    "argsort_along_batch",
    "array_equal",
    "cumprod",
    "cumprod_along_batch",
    "cumsum",
    "cumsum_along_batch",
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


def allclose(
    a: ArrayLike, b: ArrayLike, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    r"""Indicate if two arrays are element-wise equal within a tolerance.

    Args:
        a: Specifies the first array.
        b: Specifies the second array.
        rtol: The relative tolerance parameter (see Notes in ``numpy.allclose``).
        atol: The absolute tolerance parameter (see Notes in ``numpy.allclose``).
        equal_nan: Whether to compare NaN’s as equal. If ``True``,
            NaN’s in a will be considered equal to NaN’s in b in the
            output array.

    Returns:
        ``True`` if the arrays are element-wise equal within a
            tolerance, otherwise ``False``.

    Example usage:

    ```pycon
    >>> from redcat import ba
    >>> x1 = ba.ones((2, 3))
    >>> x2 = ba.ones((2, 3)) + 1e-4
    >>> x3 = ba.zeros((2, 3))
    >>> ba.allclose(x1, x2, atol=1e-3)
    True
    >>> ba.allclose(x1, x3)
    False

    ```
    """
    if isinstance(a, BatchedArray):
        return a.allclose(b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return np.allclose(a=a, b=b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan: bool = False) -> bool:
    r"""Indicate if two arrays are element-wise equal.

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
    return np.array_equal(a1=a1, a2=a2, equal_nan=equal_nan)


###########################################
#     Item selection and manipulation     #
###########################################

argsort = np.argsort
cumprod = np.cumprod
cumsum = np.cumsum


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


def cumprod_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> TBatchedArray:
    r"""Return the cumulative product of elements along a batch axis.

    Args:
        a: Array to sort.
        args: See the documentation of ``numpy.cumprod``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.cumprod``.
            ``axis`` should not be passed.

    Returns:
        The cumulative product of elements along a batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> x = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> y = ba.cumprod_along_batch(x)
    >>> y
    array([[  0,   1],
           [  0,   3],
           [  0,  15],
           [  0, 105],
           [  0, 945]], batch_axis=0)
    >>> x = ba.BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> y = ba.cumprod_along_batch(x)
    >>> y
    array([[    0,     0,     0,     0,     0],
           [    5,    30,   210,  1680, 15120]], batch_axis=1)

    ```
    """
    return a.cumprod_along_batch(*args, **kwargs)


def cumsum_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> TBatchedArray:
    r"""Return the cumulative sum of elements along a batch axis.

    Args:
        a: Array to sort.
        args: See the documentation of ``numpy.cumsum``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.cumsum``.
            ``axis`` should not be passed.

    Returns:
        The cumulative sum of elements along a batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> x = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> y = ba.cumsum_along_batch(x)
    >>> y
    array([[ 0,  1],
           [ 2,  4],
           [ 6,  9],
           [12, 16],
           [20, 25]], batch_axis=0)
    >>> x = ba.BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> y = ba.cumsum_along_batch(x)
    >>> y
    array([[ 0,  1,  3,  6, 10],
           [ 5, 11, 18, 26, 35]], batch_axis=1)

    ```
    """
    return a.cumsum_along_batch(*args, **kwargs)
