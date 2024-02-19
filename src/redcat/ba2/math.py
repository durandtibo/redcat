from __future__ import annotations

__all__ = [
    "cumprod",
    "cumprod_along_batch",
    "cumsum",
    "cumsum_along_batch",
    "nancumprod",
    "nancumprod_along_batch",
    "nancumsum",
    "nancumsum_along_batch",
]

from typing import SupportsIndex, TypeVar

import numpy as np
from numpy.typing import DTypeLike

from redcat.ba2.core import BatchedArray, implements

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")


@implements(np.cumprod)
def cumprod(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike = None,
    out: np.ndarray | None = None,
) -> TBatchedArray | np.ndarray:
    r"""See ``numpy.cumprod`` documentation."""
    return a.cumprod(axis=axis, dtype=dtype, out=out)


def cumprod_along_batch(a: TBatchedArray, dtype: DTypeLike = None) -> TBatchedArray:
    r"""Return the cumulative product of elements along the batch axis.

    Args:
        a: The input array.
        dtype: Type of the returned array and of the accumulator
            in which the elements are multiplied. If dtype is not
            specified, it defaults to the dtype of ``self``,
            unless a has an integer dtype with a precision less
            than that of  the default platform integer.
            In that case, the default platform integer is used.

    Returns:
        The cumulative product of elements along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.arange(10).reshape(5, 2))
    >>> ba2.cumprod_along_batch(batch)
    array([[  0,   1],
           [  0,   3],
           [  0,  15],
           [  0, 105],
           [  0, 945]], batch_axis=0)

    ```
    """
    return a.cumprod_along_batch(dtype=dtype)


@implements(np.cumsum)
def cumsum(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike = None,
    out: np.ndarray | None = None,
) -> TBatchedArray | np.ndarray:
    r"""See ``numpy.cumsum`` documentation."""
    return a.cumsum(axis=axis, dtype=dtype, out=out)


def cumsum_along_batch(a: TBatchedArray, dtype: DTypeLike = None) -> TBatchedArray:
    r"""Return the cumulative sum of elements along the batch axis.

    Args:
        a: The input array.
        dtype: Type of the returned array and of the accumulator
            in which the elements are summed. If dtype is not
            specified, it defaults to the dtype of ``self``,
            unless a has an integer dtype with a precision less
            than that of  the default platform integer.
            In that case, the default platform integer is used.

    Returns:
        The cumulative sum of elements along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.arange(10).reshape(5, 2))
    >>> ba2.cumsum_along_batch(batch)
    array([[ 0,  1],
           [ 2,  4],
           [ 6,  9],
           [12, 16],
           [20, 25]], batch_axis=0)

    ```
    """
    return a.cumsum_along_batch(dtype=dtype)


@implements(np.nancumprod)
def nancumprod(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike = None,
    out: np.ndarray | None = None,
) -> TBatchedArray | np.ndarray:
    r"""See ``numpy.nancumprod`` documentation."""
    return a.nancumprod(axis=axis, dtype=dtype, out=out)


def nancumprod_along_batch(a: TBatchedArray, dtype: DTypeLike = None) -> TBatchedArray:
    r"""Return the cumulative product of elements along the batch axis.

    Args:
        a: The input array.
        dtype: Type of the returned array and of the accumulator
            in which the elements are multiplied. If dtype is not
            specified, it defaults to the dtype of ``self``,
            unless a has an integer dtype with a precision less
            than that of  the default platform integer.
            In that case, the default platform integer is used.

    Returns:
        The cumulative product of elements along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba2.nancumprod_along_batch(batch)
    array([[ 1.,  1.,  2.],
           [ 3.,  4., 10.]], batch_axis=0)

    ```
    """
    return a.nancumprod_along_batch(dtype=dtype)


@implements(np.nancumsum)
def nancumsum(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike = None,
    out: np.ndarray | None = None,
) -> TBatchedArray | np.ndarray:
    r"""See ``numpy.nancumsum`` documentation."""
    return a.nancumsum(axis=axis, dtype=dtype, out=out)


def nancumsum_along_batch(a: TBatchedArray, dtype: DTypeLike = None) -> TBatchedArray:
    r"""Return the cumulative sum of elements along the batch axis.

    Args:
        a: The input array.
        dtype: Type of the returned array and of the accumulator
            in which the elements are summed. If dtype is not
            specified, it defaults to the dtype of ``self``,
            unless a has an integer dtype with a precision less
            than that of  the default platform integer.
            In that case, the default platform integer is used.

    Returns:
        The cumulative sum of elements along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba2.nancumsum_along_batch(batch)
    array([[1., 0., 2.],
           [4., 4., 7.]], batch_axis=0)

    ```
    """
    return a.nancumsum_along_batch(dtype=dtype)
