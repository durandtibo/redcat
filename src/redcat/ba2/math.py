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
    "nanprod",
    "nanprod_along_batch",
    "nansum",
    "nansum_along_batch",
    "prod",
    "prod_along_batch",
    "sum",
    "sum_along_batch",
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


def nanprod(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike = None,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> TBatchedArray | np.ndarray:
    r"""Return the product of elements along a given axis treating Not a
    Numbers (NaNs) as one.

    Args:
        axis: Axis along which the cumulative product is computed.
            By default, the input is flattened.
        dtype: Type of the returned array and of the accumulator
            in which the elements are multiplied. If dtype is not
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
        The product of elements along a given axis treating Not a
            Numbers (NaNs) as one.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba2.nanprod(batch, axis=0)
    array([ 3., 4., 10.])
    >>> ba2.nanprod(batch, axis=0, keepdims=True)
    array([[ 3., 4., 10.]])
    >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba2.nanprod(batch, axis=1)
    array([ 2., 60.])

    ```
    """
    return a.nanprod(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def nanprod_along_batch(
    a: TBatchedArray, dtype: DTypeLike = None, keepdims: bool = False
) -> TBatchedArray:
    r"""Return the product of elements along the batch axis treating Not
    a Numbers (NaNs) as one.

    Args:
        dtype: Type of the returned array and of the accumulator
            in which the elements are multiplied. If dtype is not
            specified, it defaults to the dtype of ``self``,
            unless a has an integer dtype with a precision less
            than that of  the default platform integer.
            In that case, the default platform integer is used.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the original array.

    Returns:
        The product of elements along the batch axis treating Not a
            Numbers (NaNs) as one.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba2.nanprod_along_batch(batch)
    array([ 3., 4., 10.])

    ```
    """
    return a.nanprod_along_batch(dtype=dtype, keepdims=keepdims)


def nansum(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike = None,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> TBatchedArray | np.ndarray:
    r"""Return the sum of elements along a given axis treating Not a
    Numbers (NaNs) as zero.

    Args:
        axis: Axis along which the cumulative product is computed.
            By default, the input is flattened.
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
        The sum of elements along a given axis treating Not a
            Numbers (NaNs) as zero.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba2.nansum(batch, axis=0)
    array([4., 4., 7.])
    >>> ba2.nansum(batch, axis=0, keepdims=True)
    array([[4., 4., 7.]])
    >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba2.nansum(batch, axis=1)
    array([ 3., 12.])

    ```
    """
    return a.nansum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def nansum_along_batch(
    a: TBatchedArray, dtype: DTypeLike = None, keepdims: bool = False
) -> TBatchedArray:
    r"""Return the sum of elements along the batch axis treating Not a
    Numbers (NaNs) as zero.

    Args:
        dtype: Type of the returned array and of the accumulator
            in which the elements are summed. If dtype is not
            specified, it defaults to the dtype of ``self``,
            unless a has an integer dtype with a precision less
            than that of  the default platform integer.
            In that case, the default platform integer is used.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the original array.

    Returns:
        The sum of elements along the batch axis treating Not a
            Numbers (NaNs) as zero.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba2.nansum_along_batch(batch)
    array([4., 4., 7.])

    ```
    """
    return a.nansum_along_batch(dtype=dtype, keepdims=keepdims)


def prod(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike = None,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> TBatchedArray | np.ndarray:
    r"""Return the product of elements along a given axis.

    Args:
        axis: Axis along which the cumulative product is computed.
            By default, the input is flattened.
        dtype: Type of the returned array and of the accumulator
            in which the elements are multiplied. If dtype is not
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
        The product of elements along a given axis treating.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba2.prod(batch, axis=0)
    array([ 3, 24, 10])
    >>> ba2.prod(batch, axis=0, keepdims=True)
    array([[ 3, 24, 10]])
    >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba2.prod(batch, axis=1)
    array([12, 60])

    ```
    """
    return a.prod(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def prod_along_batch(
    a: TBatchedArray, dtype: DTypeLike = None, keepdims: bool = False
) -> TBatchedArray:
    r"""Return the product of elements along the batch axis.

    Args:
        dtype: Type of the returned array and of the accumulator
            in which the elements are multiplied. If dtype is not
            specified, it defaults to the dtype of ``self``,
            unless a has an integer dtype with a precision less
            than that of  the default platform integer.
            In that case, the default platform integer is used.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the original array.

    Returns:
        The product of elements along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba2.prod_along_batch(batch)
    array([ 3, 24, 10])

    ```
    """
    return a.prod_along_batch(dtype=dtype, keepdims=keepdims)


def sum(  # noqa: A001
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike = None,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> TBatchedArray | np.ndarray:
    r"""Return the sum of elements along a given axis treating Not a
    Numbers (NaNs) as zero.

    Args:
        axis: Axis along which the cumulative product is computed.
            By default, the input is flattened.
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
        The sum of elements along a given axis treating Not a
            Numbers (NaNs) as zero.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba2.sum(batch, axis=0)
    array([ 4, 10, 7])
    >>> ba2.sum(batch, axis=0, keepdims=True)
    array([[ 4, 10, 7]])
    >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba2.sum(batch, axis=1)
    array([ 9, 12])

    ```
    """
    return a.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def sum_along_batch(
    a: TBatchedArray, dtype: DTypeLike = None, keepdims: bool = False
) -> TBatchedArray:
    r"""Return the sum of elements along the batch axis treating Not a
    Numbers (NaNs) as zero.

    Args:
        dtype: Type of the returned array and of the accumulator
            in which the elements are summed. If dtype is not
            specified, it defaults to the dtype of ``self``,
            unless a has an integer dtype with a precision less
            than that of  the default platform integer.
            In that case, the default platform integer is used.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the original array.

    Returns:
        The sum of elements along the batch axis treating Not a
            Numbers (NaNs) as zero.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba2.sum_along_batch(batch)
    array([ 4, 10, 7])

    ```
    """
    return a.sum_along_batch(dtype=dtype, keepdims=keepdims)
