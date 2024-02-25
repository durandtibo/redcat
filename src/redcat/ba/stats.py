r"""Contain functions to compute statistics on ``BatchedArray``."""

from __future__ import annotations

__all__ = [
    "mean",
    "mean_along_batch",
    "median",
    "median_along_batch",
    "nanmean",
    "nanmean_along_batch",
    "nanmedian",
    "nanmedian_along_batch",
]

from typing import TYPE_CHECKING, SupportsIndex, TypeVar

import numpy as np

from redcat.ba.core import BatchedArray, implements

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")


@implements(np.mean)
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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba.mean(batch)
    3.5
    >>> ba.mean(batch, axis=0)
    array([2. , 5. , 3.5])
    >>> ba.mean(batch, axis=0, keepdims=True)
    array([[2. , 5. , 3.5]])
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.mean(batch, axis=1)
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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba.mean_along_batch(batch)
    array([2. , 5. , 3.5])
    >>> ba.mean_along_batch(batch, keepdims=True)
    array([[2. , 5. , 3.5]])
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.mean_along_batch(batch)
    array([3., 4.])

    ```
    """
    return a.mean_along_batch(out=out, dtype=dtype, keepdims=keepdims)


@implements(np.median)
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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba.mean(batch)
    3.5
    >>> ba.mean(batch, axis=0)
    array([2. , 5. , 3.5])
    >>> ba.mean(batch, axis=0, keepdims=True)
    array([[2. , 5. , 3.5]])
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.mean(batch, axis=1)
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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba.mean_along_batch(batch)
    array([2. , 5. , 3.5])
    >>> ba.mean_along_batch(batch, keepdims=True)
    array([[2. , 5. , 3.5]])
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.mean_along_batch(batch)
    array([3., 4.])

    ```
    """
    return a.median_along_batch(out=out, keepdims=keepdims)


@implements(np.nanmean)
def nanmean(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike = None,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return the arithmetic mean along the specified axis, ignoring
    NaNs.

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
        The arithmetic mean along the specified axis, ignoring NaNs.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanmean(batch)
    3.0
    >>> ba.nanmean(batch, axis=0)
    array([2. , 4. , 3.5])
    >>> ba.nanmean(batch, axis=0, keepdims=True)
    array([[2. , 4. , 3.5]])
    >>> ba.nanmean(batch, axis=1)
    array([1.5, 4. ])

    ```
    """
    return a.nanmean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def nanmean_along_batch(
    a: TBatchedArray,
    dtype: DTypeLike = None,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return tne arithmetic mean along the batch axis, ignoring NaNs.

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
        The arithmetic mean along the batch axis, ignoring NaNs.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanmean_along_batch(batch)
    array([2. , 4. , 3.5])
    >>> ba.nanmean_along_batch(batch, keepdims=True)
    array([[2. , 4. , 3.5]])
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanmean_along_batch(batch)
    array([1.5, 4. ])

    ```
    """
    return a.nanmean_along_batch(out=out, dtype=dtype, keepdims=keepdims)


@implements(np.nanmedian)
def nanmedian(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return the median along the specified axis, ignoring NaNs.

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
        The median along the specified axis, ignoring NaNs.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanmedian(batch)
    3.0
    >>> ba.nanmedian(batch, axis=0)
    array([2. , 4. , 3.5])
    >>> ba.nanmedian(batch, axis=0, keepdims=True)
    array([[2. , 4. , 3.5]])
    >>> ba.nanmedian(batch, axis=1)
    array([1.5, 4. ])

    ```
    """
    return a.nanmedian(axis=axis, out=out, keepdims=keepdims)


def nanmedian_along_batch(
    a: TBatchedArray,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return tne median along the batch axis, ignoring NaNs.

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
        The median along the batch axis, ignoring NaNs.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanmedian_along_batch(batch)
    array([2. , 4. , 3.5])
    >>> ba.nanmedian_along_batch(batch, keepdims=True)
    array([[2. , 4. , 3.5]])
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanmedian_along_batch(batch)
    array([1.5, 4. ])

    ```
    """
    return a.nanmedian_along_batch(out=out, keepdims=keepdims)
