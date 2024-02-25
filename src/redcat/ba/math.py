r"""Contain mathematical functions for ``BatchedArray``."""

from __future__ import annotations

__all__ = [
    "add",
    "cumprod",
    "cumprod_along_batch",
    "cumsum",
    "cumsum_along_batch",
    "diff",
    "diff_along_batch",
    "divide",
    "floor_divide",
    "max",
    "max_along_batch",
    "min",
    "min_along_batch",
    "multiply",
    "nancumprod",
    "nancumprod_along_batch",
    "nancumsum",
    "nancumsum_along_batch",
    "nanmax",
    "nanmax_along_batch",
    "nanmin",
    "nanmin_along_batch",
    "nanprod",
    "nanprod_along_batch",
    "nansum",
    "nansum_along_batch",
    "prod",
    "prod_along_batch",
    "subtract",
    "sum",
    "sum_along_batch",
    "true_divide",
]

from typing import TYPE_CHECKING, SupportsIndex, TypeVar

import numpy as np

from redcat.ba.core import BatchedArray, implements

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike

TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")

add = np.add
divide = np.divide
floor_divide = np.floor_divide
multiply = np.multiply
subtract = np.subtract
true_divide = np.true_divide


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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> ba.cumprod_along_batch(batch)
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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> ba.cumsum_along_batch(batch)
    array([[ 0,  1],
           [ 2,  4],
           [ 6,  9],
           [12, 16],
           [20, 25]], batch_axis=0)

    ```
    """
    return a.cumsum_along_batch(dtype=dtype)


@implements(np.diff)
def diff(
    a: TBatchedArray,
    n: int = 1,
    axis: SupportsIndex = -1,
    prepend: ArrayLike = np._NoValue,
    append: ArrayLike = np._NoValue,
) -> TBatchedArray | np.ndarray:
    r"""Calculate the n-th discrete difference along the given axis.

    Args:
        a: The input array.
        n: The number of times values are differenced. If zero,
            the input is returned as-is.
        axis: The axis along which the difference is taken,
            default is the last axis.
        prepend: Values to prepend to the current array along axis
            prior to performing the difference. Scalar values are
            expanded to arrays with length 1 in the direction of
            axis and the shape of the input array in along all
            other axes. Otherwise the dimension and shape must
            match the current array except along axis.
        append: Values to append to the current array along axis
            prior to performing the difference. Scalar values are
            expanded to arrays with length 1 in the direction of
            axis and the shape of the input array in along all
            other axes. Otherwise the dimension and shape must
            match the current array except along axis.

    Returns:
        The n-th differences. The shape of the output is the same
            as the current array except along axis where the
            dimension is smaller by ``n``. The type of the output
            is the same as the type of the difference between any
            two elements of the array. This is the same as the type
            of the current array in most cases.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]]))
    >>> ba.diff(batch, n=1, axis=0)
    array([[ 0, -1],
           [ 1,  7],
           [-7, -9],
           [ 6,  7]])
    >>> batch = BatchedArray(np.array([[9, 3, 7, 4, 0], [6, 6, 2, 3, 3]]), batch_axis=1)
    >>> ba.diff(batch, axis=1)
    array([[-6,  4, -3, -4], [ 0, -4,  1,  0]])

    ```
    """
    return a.diff(n=n, axis=axis, prepend=prepend, append=append)


def diff_along_batch(
    a: TBatchedArray,
    n: int = 1,
    prepend: ArrayLike = np._NoValue,
    append: ArrayLike = np._NoValue,
) -> TBatchedArray | np.ndarray:
    r"""Calculate the n-th discrete difference along the batch axis.

    Args:
        a: The input array.
        n: The number of times values are differenced. If zero,
            the input is returned as-is.
        prepend: Values to prepend to the array along the batch
            axis prior to performing the difference. Scalar values
            are expanded to arrays with length 1 in the direction
            of axis and the shape of the input array in along all
            other axes. Otherwise the dimension and shape must
            match the current array except along axis.
        append: Values to append to the array along the batch
            axis prior to performing the difference. Scalar values
            are expanded to arrays with length 1 in the direction
            of axis and the shape of the input array in along all
            other axes. Otherwise the dimension and shape must
            match the current array except along axis.

    Returns:
        The n-th differences. The shape of the output is the same
            as the current array except along the batch axis where
            the dimension is smaller by ``n``. The type of the
            output is the same as the type of the difference
            between any two elements of the array. This is the same
            as the type of the current array in most cases.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = BatchedArray(np.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]]))
    >>> ba.diff_along_batch(batch, n=1)
    array([[ 0, -1],
           [ 1,  7],
           [-7, -9],
           [ 6,  7]])
    >>> batch = BatchedArray(np.array([[9, 3, 7, 4, 0], [6, 6, 2, 3, 3]]), batch_axis=1)
    >>> ba.diff_along_batch(batch, n=1)
    array([[-6,  4, -3, -4], [ 0, -4,  1,  0]])

    ```
    """
    return a.diff_along_batch(n=n, prepend=prepend, append=append)


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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nancumprod_along_batch(batch)
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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nancumsum_along_batch(batch)
    array([[1., 0., 2.],
           [4., 4., 7.]], batch_axis=0)

    ```
    """
    return a.nancumsum_along_batch(dtype=dtype)


@implements(np.nanprod)
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
        a: The input array.
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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanprod(batch, axis=0)
    array([ 3., 4., 10.])
    >>> ba.nanprod(batch, axis=0, keepdims=True)
    array([[ 3., 4., 10.]])
    >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanprod(batch, axis=1)
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
        a: The input array.
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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanprod_along_batch(batch)
    array([ 3., 4., 10.])

    ```
    """
    return a.nanprod_along_batch(dtype=dtype, keepdims=keepdims)


@implements(np.nansum)
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
        a: The input array.
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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nansum(batch, axis=0)
    array([4., 4., 7.])
    >>> ba.nansum(batch, axis=0, keepdims=True)
    array([[4., 4., 7.]])
    >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nansum(batch, axis=1)
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
        a: The input array.
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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nansum_along_batch(batch)
    array([4., 4., 7.])

    ```
    """
    return a.nansum_along_batch(dtype=dtype, keepdims=keepdims)


@implements(np.prod)
def prod(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike = None,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> TBatchedArray | np.ndarray:
    r"""Return the product of elements along a given axis.

    Args:
        a: The input array.
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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba.prod(batch, axis=0)
    array([ 3, 24, 10])
    >>> ba.prod(batch, axis=0, keepdims=True)
    array([[ 3, 24, 10]])
    >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.prod(batch, axis=1)
    array([12, 60])

    ```
    """
    return a.prod(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


def prod_along_batch(
    a: TBatchedArray, dtype: DTypeLike = None, keepdims: bool = False
) -> TBatchedArray:
    r"""Return the product of elements along the batch axis.

    Args:
        a: The input array.
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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba.prod_along_batch(batch)
    array([ 3, 24, 10])

    ```
    """
    return a.prod_along_batch(dtype=dtype, keepdims=keepdims)


@implements(np.sum)
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
        a: The input array.
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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba.sum(batch, axis=0)
    array([ 4, 10, 7])
    >>> ba.sum(batch, axis=0, keepdims=True)
    array([[ 4, 10, 7]])
    >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.sum(batch, axis=1)
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
        a: The input array.
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
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba.sum_along_batch(batch)
    array([ 4, 10, 7])

    ```
    """
    return a.sum_along_batch(dtype=dtype, keepdims=keepdims)


@implements(np.max)
def max(  # noqa: A001
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return the maximum of an array or maximum along an axis.

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
        The maximum of an array or maximum along an axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba.max(batch)
    6
    >>> ba.max(batch, axis=0)
    array([3, 6, 5])
    >>> ba.max(batch, axis=0, keepdims=True)
    array([[3, 6, 5]])
    >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.max(batch, axis=1)
    array([6, 5])

    ```
    """
    return a.max(axis=axis, out=out, keepdims=keepdims)


def max_along_batch(
    a: TBatchedArray,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return the maximum along the batch axis.

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
        The maximum along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba.max_along_batch(batch)
    array([3, 6, 5])
    >>> ba.max_along_batch(batch, keepdims=True)
    array([[3, 6, 5]])
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.max_along_batch(batch)
    array([6, 5])

    ```
    """
    return a.max_along_batch(out=out, keepdims=keepdims)


@implements(np.min)
def min(  # noqa: A001
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return the minimum of an array or minimum along an axis.

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
        The minimum of an array or minimum along an axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba.min(batch)
    1
    >>> ba.min(batch, axis=0)
    array([1, 4, 2])
    >>> ba.min(batch, axis=0, keepdims=True)
    array([[1, 4, 2]])
    >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.min(batch, axis=1)
    array([1, 3])

    ```
    """
    return a.min(axis=axis, out=out, keepdims=keepdims)


def min_along_batch(
    a: TBatchedArray,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return the minimum along the batch axis.

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
        The minimum along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba.min_along_batch(batch)
    array([1, 4, 2])
    >>> ba.min_along_batch(batch, keepdims=True)
    array([[1, 4, 2]])
    >>> batch = ba.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.min_along_batch(batch)
    array([1, 3])

    ```
    """
    return a.min_along_batch(out=out, keepdims=keepdims)


@implements(np.nanmax)
def nanmax(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return the maximum of an array or maximum along an axis, ignoring
    any NaNs.

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
        The maximum of an array or maximum along an axis, ignoring
            any NaNs.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanmax(batch)
    5.0
    >>> ba.nanmax(batch,axis=0)
    array([3., 4., 5.])
    >>> ba.nanmax(batch, axis=0, keepdims=True)
    array([[3., 4., 5.]])
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanmax(batch, axis=1)
    array([2., 5.])

    ```
    """
    return a.nanmax(axis=axis, out=out, keepdims=keepdims)


def nanmax_along_batch(
    a: TBatchedArray,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return the maximum along the batch axis, ignoring any NaNs.

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
        The maximum along the batch axis, ignoring any NaNs.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanmax_along_batch(batch)
    array([3., 4., 5.])
    >>> ba.nanmax_along_batch(batch, keepdims=True)
    array([[3., 4., 5.]])
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanmax_along_batch(batch)
    array([2., 5.])

    ```
    """
    return a.nanmax_along_batch(out=out, keepdims=keepdims)


@implements(np.nanmin)
def nanmin(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return the minimum of an array or minimum along an axis, ignoring
    any NaNs.

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
        The minimum of an array or minimum along an axis, ignoring
            any NaNs.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[np.nan, 6, 2], [3, 4, 5]]))
    >>> ba.nanmin(batch)
    2.0
    >>> ba.nanmin(batch,axis=0)
    array([3., 4., 2.])
    >>> ba.nanmin(batch, axis=0, keepdims=True)
    array([[3., 4., 2.]])
    >>> batch = ba.BatchedArray(np.array([[np.nan, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanmin(batch, axis=1)
    array([2., 3.])

    ```
    """
    return a.nanmin(axis=axis, out=out, keepdims=keepdims)


def nanmin_along_batch(
    a: TBatchedArray,
    out: np.ndarray | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return the minimum along the batch axis, ignoring any NaNs.

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
        The minimum along the batch axis, ignoring any NaNs.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[np.nan, 6, 2], [3, 4, 5]]))
    >>> ba.nanmin_along_batch(batch)
    array([3., 4., 2.])
    >>> ba.nanmin_along_batch(batch, keepdims=True)
    array([[3., 4., 2.]])
    >>> batch = ba.BatchedArray(np.array([[np.nan, 6, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanmin_along_batch(batch)
    array([2., 3.])

    ```
    """
    return a.nanmin_along_batch(out=out, keepdims=keepdims)
