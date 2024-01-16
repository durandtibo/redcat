from __future__ import annotations

__all__ = [
    "allclose",
    "argmax",
    "argmax_along_batch",
    "argmin",
    "argmin_along_batch",
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
    "max",
    "max_along_batch",
    "mean",
    "mean_along_batch",
    "median",
    "median_along_batch",
    "min",
    "min_along_batch",
    "nanmean",
    "nanmean_along_batch",
    "nanmedian",
    "nanmedian_along_batch",
    "nanprod",
    "nanprod_along_batch",
    "nansum",
    "nansum_along_batch",
    "not_equal",
    "permute_along_axis",
    "permute_along_batch",
    "prod",
    "prod_along_batch",
    "shuffle_along_axis",
    "shuffle_along_batch",
    "sort",
    "sort_along_batch",
    "sum",
    "sum_along_batch",
    "nanmin_along_batch",
    "nanmin",
    "nanmax",
    "nanmax_along_batch",
]

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from redcat.ba.core import BatchedArray
from redcat.types import RNGType

TBatchedArray = TypeVar("TBatchedArray", np.ndarray, "BatchedArray")

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
sort = np.sort


def argsort_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> TBatchedArray:
    r"""Return the indices that would sort the batch along the batch
    dimension.

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
        a: Specifies the input array.
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
        a: Specifies the input array.
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


def permute_along_axis(
    a: TBatchedArray, permutation: np.ndarray | Sequence, axis: int = 0
) -> TBatchedArray:
    r"""Permute the values of an array along a given axis.

    Args:
        a: Specifies the array to permute.
        permutation: Specifies the permutation to use on the array.
            The dimension of this array should be compatible with the
            shape of the array to permute.
        axis: Specifies the axis used to permute the array.

    Returns:
        The permuted array.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.arange(4))
    >>> ba.permute_along_axis(batch, permutation=np.array([0, 2, 1, 3]))
    array([0, 2, 1, 3], batch_axis=0)
    >>> batch = ba.BatchedArray(np.arange(20).reshape(4, 5))
    >>> ba.permute_along_axis(batch, permutation=np.array([0, 2, 1, 3]))
    array([[ 0,  1,  2,  3,  4],
           [10, 11, 12, 13, 14],
           [ 5,  6,  7,  8,  9],
           [15, 16, 17, 18, 19]], batch_axis=0)
    >>> batch = ba.BatchedArray(np.arange(20).reshape(4, 5))
    >>> ba.permute_along_axis(batch, permutation=np.array([0, 4, 2, 1, 3]), axis=1)
    array([[ 0,  4,  2,  1,  3],
           [ 5,  9,  7,  6,  8],
           [10, 14, 12, 11, 13],
           [15, 19, 17, 16, 18]], batch_axis=0)
    >>> batch = ba.BatchedArray(np.arange(20).reshape(2, 2, 5))
    >>> ba.permute_along_axis(batch, permutation=np.array([0, 4, 2, 1, 3]), axis=2)
    array([[[ 0,  4,  2,  1,  3],
            [ 5,  9,  7,  6,  8]],
           [[10, 14, 12, 11, 13],
            [15, 19, 17, 16, 18]]], batch_axis=0)

    ```
    """
    return a.permute_along_axis(permutation, axis)


def permute_along_batch(a: TBatchedArray, permutation: np.ndarray | Sequence) -> TBatchedArray:
    r"""Permute the values of an array along the batch axis.

    Args:
        a: Specifies the array to permute.
        permutation: Specifies the permutation to use on the array.
            The dimension of this array should be compatible with the
            shape of the array to permute.

    Returns:
        The permuted array along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.arange(4))
    >>> ba.permute_along_batch(batch, permutation=np.array([0, 2, 1, 3]))
    array([0, 2, 1, 3], batch_axis=0)
    >>> batch = ba.BatchedArray(np.arange(20).reshape(4, 5))
    >>> ba.permute_along_batch(batch, permutation=np.array([0, 2, 1, 3]))
    array([[ 0,  1,  2,  3,  4],
           [10, 11, 12, 13, 14],
           [ 5,  6,  7,  8,  9],
           [15, 16, 17, 18, 19]], batch_axis=0)
    >>> batch = ba.BatchedArray(np.arange(20).reshape(4, 5), batch_axis=1)
    >>> ba.permute_along_batch(batch, permutation=np.array([0, 4, 2, 1, 3]))
    array([[ 0,  4,  2,  1,  3],
           [ 5,  9,  7,  6,  8],
           [10, 14, 12, 11, 13],
           [15, 19, 17, 16, 18]], batch_axis=1)

    ```
    """
    return a.permute_along_batch(permutation)


def shuffle_along_axis(
    a: TBatchedArray, axis: int, generator: RNGType | None = None
) -> TBatchedArray:
    r"""Shuffle the batch along a given axis.

    Args:
        a: Specifies the array to shuffle.
        axis: Specifies the shuffle axis.
        generator: Specifies the pseudorandom number generator for
            sampling or the random seed for the random number
            generator.

    Returns:
        A new batch with shuffled data along the given axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> ba.shuffle_along_axis(batch, axis=0)
    array([[...]], batch_axis=0)

    ```
    """
    return a.shuffle_along_axis(axis=axis, generator=generator)


def shuffle_along_batch(a: TBatchedArray, generator: RNGType | None = None) -> TBatchedArray:
    r"""Shuffle the batch along the batch axis.

    Args:
        a: Specifies the array to shuffle.
        generator: Specifies the pseudorandom number generator for
            sampling or the random seed for the random number
            generator.

    Returns:
        A new batch with shuffled data along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> ba.shuffle_along_batch(batch)
    array([[...]], batch_axis=0)

    ```
    """
    return a.shuffle_along_batch(generator=generator)


def sort_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> TBatchedArray:
    r"""Return a sorted copy of an array along the batch axis.

    Args:
        a: Array to sort.
        args: See the documentation of ``numpy.argsort``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.argsort``.
            ``axis`` should not be passed.

    Returns:
        A sorted copy of an array along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> x = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> y = ba.sort_along_batch(x)
    >>> y
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]], batch_axis=0)
    >>> x = ba.BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> y = ba.sort_along_batch(x)
    >>> y
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]], batch_axis=1)

    ```
    """
    return np.sort(a, *args, axis=a.batch_axis, **kwargs)


#####################
#     Reduction     #
#####################


def argmax(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Return indices of the maximum values along an axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.argmax``.
        kwargs: See the documentation of ``numpy.argmax``.

    Returns:
        The indices of the maximum values along an axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> ba.argmax(batch, axis=0)
    array([4, 4])
    >>> ba.argmax(batch, axis=0, keepdims=True)
    array([[4, 4]])
    >>> batch = ba.BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> ba.argmax(batch, axis=1)
    array([4, 4])

    ```
    """
    return a.argmax(*args, **kwargs)


def argmax_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Return indices of the maximum values along the batch axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.argmax``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.argmax``.
            ``axis`` should not be passed.

    Returns:
        The indices of the maximum values along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> ba.argmax_along_batch(batch)
    array([4, 4])
    >>> ba.argmax_along_batch(batch, keepdims=True)
    array([[4, 4]])
    >>> batch = ba.BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> ba.argmax_along_batch(batch)
    array([4, 4])

    ```
    """
    return a.argmax_along_batch(*args, **kwargs)


def argmin(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Return indices of the minimum values along an axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.argmin``.
        kwargs: See the documentation of ``numpy.argmin``.

    Returns:
        The indices of the minimum values along an axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> ba.argmin(batch, axis=0)
    array([0, 0])
    >>> ba.argmin(batch, axis=0, keepdims=True)
    array([[0, 0]])
    >>> batch = ba.BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> ba.argmin(batch, axis=1)
    array([0, 0])

    ```
    """
    return a.argmin(*args, **kwargs)


def argmin_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Return indices of the minimum values along the batch axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.argmin``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.argmin``.
            ``axis`` should not be passed.

    Returns:
        The indices of the minimum values along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> ba.argmin_along_batch(batch)
    array([0, 0])
    >>> ba.argmin_along_batch(batch, keepdims=True)
    array([[0, 0]])
    >>> batch = ba.BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> ba.argmin_along_batch(batch)
    array([0, 0])

    ```
    """
    return a.argmin_along_batch(*args, **kwargs)


def max(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:  # noqa: A001
    r"""Return the maximum along a given axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.max``.
        kwargs: See the documentation of ``numpy.max``.

    Returns:
        The maximum values along an axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> ba.max(batch, axis=0)
    array([8, 9])
    >>> ba.max(batch, axis=0, keepdims=True)
    array([[8, 9]])
    >>> batch = ba.BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> ba.max(batch, axis=1)
    array([4, 9])

    ```
    """
    return a.max(*args, **kwargs)


def max_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Return the maximum along the batch axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.max``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.max``.
            ``axis`` should not be passed.

    Returns:
        The maximum values along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> ba.max_along_batch(batch)
    array([8, 9])
    >>> ba.max_along_batch(batch, keepdims=True)
    array([[8, 9]])
    >>> batch = ba.BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> ba.max_along_batch(batch)
    array([4, 9])

    ```
    """
    return a.max_along_batch(*args, **kwargs)


def mean(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Return the mean along a given axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.mean``.
        kwargs: See the documentation of ``numpy.mean``.

    Returns:
        The mean values along an axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat.ba import BatchedArray
    >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
    >>> batch.mean(axis=0)
    array([4., 5.])
    >>> batch.mean(axis=0, keepdims=True)
    array([[4., 5.]])
    >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> batch.mean(axis=1)
    array([2., 7.])

    ```
    """
    return a.mean(*args, **kwargs)


def mean_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Return the mean along the batch axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.mean``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.mean``.
            ``axis`` should not be passed.

    Returns:
        The mean values along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat.ba import BatchedArray
    >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
    >>> batch.mean_along_batch()
    array([4., 5.])
    >>> batch.mean_along_batch(keepdims=True)
    array([[4., 5.]])
    >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> batch.mean_along_batch()
    array([2., 7.])

    ```
    """
    return a.mean_along_batch(*args, **kwargs)


def median(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Return the median along a given axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.median``.
        kwargs: See the documentation of ``numpy.median``.

    Returns:
        The median values along an axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat.ba import BatchedArray
    >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
    >>> batch.median(axis=0)
    array([4., 5.])
    >>> batch.median(axis=0, keepdims=True)
    array([[4., 5.]])
    >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> batch.median(axis=1)
    array([2., 7.])

    ```
    """
    return a.median(*args, **kwargs)


def median_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Return the median along the batch axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.median``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.median``.
            ``axis`` should not be passed.

    Returns:
        The median values along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat.ba import BatchedArray
    >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
    >>> batch.median_along_batch()
    array([4., 5.])
    >>> batch.median_along_batch(keepdims=True)
    array([[4., 5.]])
    >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> batch.median_along_batch()
    array([2., 7.])

    ```
    """
    return a.median_along_batch(*args, **kwargs)


def min(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:  # noqa: A001
    r"""Return the minimum along a given axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.min``.
        kwargs: See the documentation of ``numpy.min``.

    Returns:
        The minimum values along an axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> ba.min(batch, axis=0)
    array([0, 1])
    >>> ba.min(batch, axis=0, keepdims=True)
    array([[0, 1]])
    >>> batch = ba.BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> ba.min(batch, axis=1)
    array([0, 5])

    ```
    """
    return a.min(*args, **kwargs)


def min_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Return the minimum along the batch axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.min``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.min``.
            ``axis`` should not be passed.

    Returns:
        The minimum values along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.arange(10).reshape(5, 2))
    >>> ba.min_along_batch(batch)
    array([0, 1])
    >>> ba.min_along_batch(batch, keepdims=True)
    array([[0, 1]])
    >>> batch = ba.BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
    >>> ba.min_along_batch(batch)
    array([0, 5])

    ```
    """
    return a.min_along_batch(*args, **kwargs)


def nanmax(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Compute the maximum or the maximum along the specified axis,
    ignoring NaNs.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.nanmax``.
        kwargs: See the documentation of ``numpy.nanmax``.

    Returns:
        The maximum or the maximum along the specified axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanmax(batch, axis=0)
    array([3., 4., 5.])
    >>> ba.nanmax(batch, axis=0, keepdims=True)
    array([[3., 4., 5.]])
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanmax(batch, axis=1)
    array([2., 5.])

    ```
    """
    return a.nanmax(*args, **kwargs)


def nanmax_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Compute the maximum or the maximum along the batch axis, ignoring
    NaNs.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.nanmax``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.nanmax``.
            ``axis`` should not be passed.

    Returns:
        The maximum or the maximum along the batch axis.

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
    return a.nanmax_along_batch(*args, **kwargs)


def nanmean(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Compute the arithmetic mean along the specified axis, ignoring
    NaNs.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.nanmean``.
        kwargs: See the documentation of ``numpy.nanmean``.

    Returns:
        The arithmetic mean along the specified axis, ignoring NaNs.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanmean(batch, axis=0)
    array([2. , 4. , 3.5])
    >>> ba.nanmean(batch, axis=0, keepdims=True)
    array([[2. , 4. , 3.5]])
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanmean(batch, axis=1)
    array([1.5, 4. ])

    ```
    """
    return a.nanmean(*args, **kwargs)


def nanmean_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Compute the arithmetic mean along the batch axis, ignoring NaNs.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.nanmean``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.nanmean``.
            ``axis`` should not be passed.

    Returns:
        The arithmetic mean along the batch axis, ignoring NaNs.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanmean_along_batch(batch)
    array([2. , 4. , 3.5])
    >>> ba.nanmean_along_batch(batch,keepdims=True)
    array([[2. , 4. , 3.5]])
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanmean_along_batch(batch)
    array([1.5, 4. ])

    ```
    """
    return a.nanmean_along_batch(*args, **kwargs)


def nanmedian(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Compute the median along the specified axis, ignoring NaNs.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.nanmedian``.
        kwargs: See the documentation of ``numpy.nanmedian``.

    Returns:
        The median along an axis, ignoring NaNs.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanmedian(batch, axis=0)
    array([2. , 4. , 3.5])
    >>> ba.nanmedian(batch, axis=0, keepdims=True)
    array([[2. , 4. , 3.5]])
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanmedian(batch, axis=1)
    array([1.5, 4. ])

    ```
    """
    return a.nanmedian(*args, **kwargs)


def nanmedian_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Compute the median along the batch axis, ignoring NaNs.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.nanmedian``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.nanmedian``.
            ``axis`` should not be passed.

    Returns:
        The median along the batch axis, ignoring NaNs.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanmedian_along_batch(batch)
    array([2. , 4. , 3.5])
    >>> ba.nanmedian_along_batch(batch,keepdims=True)
    array([[2. , 4. , 3.5]])
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanmedian_along_batch(batch)
    array([1.5, 4. ])

    ```
    """
    return a.nanmedian_along_batch(*args, **kwargs)


def nanmin(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Compute the minimum or the minimum along the specified axis,
    ignoring NaNs.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.nanmin``.
        kwargs: See the documentation of ``numpy.nanmin``.

    Returns:
        The minimum or the minimum along the specified axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanmin(batch, axis=0)
    array([1., 4., 2.])
    >>> ba.nanmin(batch, axis=0, keepdims=True)
    array([[1., 4., 2.]])
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanmin(batch, axis=1)
    array([1., 3.])

    ```
    """
    return a.nanmin(*args, **kwargs)


def nanmin_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Compute the minimum or the minimum along the batch axis, ignoring
    NaNs.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.nanmin``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.nanmin``.
            ``axis`` should not be passed.

    Returns:
        The minimum or the minimum along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanmin_along_batch(batch)
    array([1., 4., 2.])
    >>> ba.nanmin_along_batch(batch, keepdims=True)
    array([[1., 4., 2.]])
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanmin_along_batch(batch)
    array([1., 3.])

    ```
    """
    return a.nanmin_along_batch(*args, **kwargs)


def nanprod(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Compute the product along the specified axis, ignoring NaNs.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.nanprod``.
        kwargs: See the documentation of ``numpy.nanprod``.

    Returns:
        The product along the specified axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanprod(batch, axis=0)
    array([ 3., 4., 10.])
    >>> ba.nanprod(batch, axis=0, keepdims=True)
    array([[ 3., 4., 10.]])
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanprod(batch, axis=1)
    array([ 2., 60.])

    ```
    """
    return a.nanprod(*args, **kwargs)


def nanprod_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Compute the product along the batch axis, ignoring NaNs.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.nanprod``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.nanprod``.
            ``axis`` should not be passed.

    Returns:
        The product along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nanprod_along_batch(batch)
    array([ 3., 4., 10.])
    >>> ba.nanprod_along_batch(batch, keepdims=True)
    array([[ 3., 4., 10.]])
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nanprod_along_batch(batch)
    array([ 2., 60.])

    ```
    """
    return a.nanprod_along_batch(*args, **kwargs)


def nansum(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Compute the sum along the specified axis, ignoring NaNs.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.nansum``.
        kwargs: See the documentation of ``numpy.nansum``.

    Returns:
        The sum along the specified axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nansum(batch,axis=0)
    array([4., 4., 7.])
    >>> ba.nansum(batch,axis=0, keepdims=True)
    array([[4., 4., 7.]])
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nansum(batch, axis=1)
    array([ 3., 12.])

    ```
    """
    return a.nansum(*args, **kwargs)


def nansum_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Compute the sum along the batch axis, ignoring NaNs.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.nansum``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.nansum``.
            ``axis`` should not be passed.

    Returns:
        The sum along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba.nansum_along_batch(batch)
    array([4., 4., 7.])
    >>> ba.nansum_along_batch(batch, keepdims=True)
    array([[4., 4., 7.]])
    >>> batch = ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.nansum_along_batch(batch)
    array([ 3., 12.])

    ```
    """
    return a.nansum_along_batch(*args, **kwargs)


def prod(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Compute the product along the specified axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.prod``.
        kwargs: See the documentation of ``numpy.prod``.

    Returns:
        The product along the specified axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]))
    >>> ba.prod(batch, axis=0)
    array([ 3, 12, 10])
    >>> ba.prod(batch, axis=0, keepdims=True)
    array([[ 3, 12, 10]])
    >>> batch = ba.BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.prod(batch, axis=1)
    array([ 6, 60])

    ```
    """
    return a.prod(*args, **kwargs)


def prod_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Compute the product along the batch axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.prod``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.prod``.
            ``axis`` should not be passed.

    Returns:
        The product along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]))
    >>> ba.prod_along_batch(batch)
    array([ 3, 12, 10])
    >>> ba.prod_along_batch(batch, keepdims=True)
    array([[ 3, 12, 10]])
    >>> batch = ba.BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.prod_along_batch(batch)
    array([ 6, 60])

    ```
    """
    return a.prod_along_batch(*args, **kwargs)


def sum(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:  # noqa: A001
    r"""Compute the sum along the specified axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.sum``.
        kwargs: See the documentation of ``numpy.sum``.

    Returns:
        The sum along the specified axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]))
    >>> ba.sum(batch, axis=0)
    array([4, 7, 7])
    >>> ba.sum(batch, axis=0, keepdims=True)
    array([[4, 7, 7]])
    >>> batch = ba.BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.sum(batch, axis=1)
    array([ 6, 12])

    ```
    """
    return a.sum(*args, **kwargs)


def sum_along_batch(a: TBatchedArray, *args: Any, **kwargs: Any) -> np.ndarray:
    r"""Compute the sum along the batch axis.

    Args:
        a: Input array.
        args: See the documentation of ``numpy.sum``.
            ``axis`` should not be passed.
        kwargs: See the documentation of ``numpy.sum``.
            ``axis`` should not be passed.

    Returns:
        The sum along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]))
    >>> ba.sum_along_batch(batch)
    array([4, 7, 7])
    >>> ba.sum_along_batch(batch, keepdims=True)
    array([[4, 7, 7]])
    >>> batch = ba.BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]), batch_axis=1)
    >>> ba.sum_along_batch(batch)
    array([ 6, 12])

    ```
    """
    return a.sum_along_batch(*args, **kwargs)
