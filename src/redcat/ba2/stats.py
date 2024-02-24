from __future__ import annotations

__all__ = [
    "mean",
    "mean_along_batch",
    "median",
    "median_along_batch",
]

from typing import SupportsIndex, TypeVar

import numpy as np
from numpy.typing import DTypeLike

from redcat.ba2.core import BatchedArray

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")


def mean(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike = None,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return the arithmetic mean along the specified axis.

    Args:
        a: The input array.
        axis: Axis or axes along which to operate. By default,
            flattened input is used.
        dtype: Type of the returned array and of the accumulator
            in which the elements are summed. If dtype is not
            specified, it defaults to the dtype of ``self``,
            unless a has an integer dtype with a precision less
            than that of  the default platform integer.
            In that case, the default platform integer is used.
        out: Alternative output array in which to place the result.
            It must have the same shape and buffer length as the
            expected output but the type will be cast if necessary.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the original array.

    Returns:
        The arithmetic mean along the specified axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba2.mean(batch)
    3.5
    >>> ba2.mean(batch, axis=0)
    array([2. , 5. , 3.5])
    >>> ba2.mean(batch, axis=0, keepdims=True)
    array([[2. , 5. , 3.5]])
    >>> batch = ba2.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba2.mean(batch, axis=1)
    array([3., 4.])

    ```
    """
    return a.mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def mean_along_batch(
    a: TBatchedArray,
    dtype: DTypeLike = None,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return tne arithmetic mean along the batch axis.

    Args:
        a: The input array.
        dtype: Type of the returned array and of the accumulator
            in which the elements are summed. If dtype is not
            specified, it defaults to the dtype of ``self``,
            unless a has an integer dtype with a precision less
            than that of  the default platform integer.
            In that case, the default platform integer is used.
        out: Alternative output array in which to place the result.
            It must have the same shape and buffer length as the
            expected output but the type will be cast if necessary.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the original array.

    Returns:
        The arithmetic mean along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba2.mean_along_batch(batch)
    array([2. , 5. , 3.5])
    >>> ba2.mean_along_batch(batch, keepdims=True)
    array([[2. , 5. , 3.5]])
    >>> batch = ba2.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba2.mean_along_batch(batch)
    array([3., 4.])

    ```
    """
    return a.mean_along_batch(out=out, dtype=dtype, keepdims=keepdims)


def median(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return the median along the specified axis.

    Args:
        a: The input array.
        axis: Axis or axes along which to operate. By default,
            flattened input is used.
        out: Alternative output array in which to place the result.
            It must have the same shape and buffer length as the
            expected output but the type will be cast if necessary.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the original array.

    Returns:
        The median along the specified axis. If the input contains
            integers or floats smaller than float64, then the
            output data-type is np.float64. Otherwise, the
            data-type of the output is the same as that of the
            input. If out is specified, that array is returned
            instead.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba2.mean(batch)
    3.5
    >>> ba2.mean(batch, axis=0)
    array([2. , 5. , 3.5])
    >>> ba2.mean(batch, axis=0, keepdims=True)
    array([[2. , 5. , 3.5]])
    >>> batch = ba2.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba2.mean(batch, axis=1)
    array([3., 4.])

    ```
    """
    return a.median(axis=axis, out=out, keepdims=keepdims)


def median_along_batch(
    a: TBatchedArray,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return tne median along the batch axis.

    Args:
        a: The input array.
        out: Alternative output array in which to place the result.
            It must have the same shape and buffer length as the
            expected output but the type will be cast if necessary.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the original array.

    Returns:
        The median along the specified axis. If the input contains
            integers or floats smaller than float64, then the
            output data-type is np.float64. Otherwise, the
            data-type of the output is the same as that of the
            input. If out is specified, that array is returned
            instead.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba2.mean_along_batch(batch)
    array([2. , 5. , 3.5])
    >>> ba2.mean_along_batch(batch, keepdims=True)
    array([[2. , 5. , 3.5]])
    >>> batch = ba2.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba2.mean_along_batch(batch)
    array([3., 4.])

    ```
    """
    return a.median_along_batch(out=out, keepdims=keepdims)
