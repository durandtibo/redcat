from __future__ import annotations

__all__ = [
    "argmax",
    "argmax_along_batch",
    "argmin",
    "argmin_along_batch",
    "sort",
    "sort_along_batch",
    "nanargmax",
    "nanargmax_along_batch",
]

from typing import SupportsIndex, TypeVar

import numpy as np

from redcat.ba2.core import BatchedArray, SortKind, implements

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")


@implements(np.sort)
def sort(
    a: TBatchedArray, axis: SupportsIndex | None = -1, kind: SortKind | None = None
) -> TBatchedArray:
    r"""See ``numpy.sort`` documentation."""
    x = a.copy()
    x.sort(axis=axis, kind=kind)
    return x


def sort_along_batch(a: TBatchedArray, kind: SortKind | None = None) -> TBatchedArray:
    r"""Sort an array in-place along the batch dimension.

    Args:
        a: The input array.
        kind: Sorting algorithm. The default is ‘quicksort’.
            Note that both ‘stable’ and ‘mergesort’ use timsort
            under the covers and, in general, the actual
            implementation will vary with datatype.
            The ‘mergesort’ option is retained for backwards
            compatibility.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba2.sort_along_batch(batch)
    array([[1, 4, 2],
           [3, 6, 5]], batch_axis=0)

    ```
    """
    return sort(a, axis=a.batch_axis, kind=kind)


@implements(np.argmax)
def argmax(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    out: np.ndarray | None = None,
    *,
    keepdims: bool = False,
) -> np.ndarray:
    r"""See ``numpy.argmax`` documentation."""
    return a.argmax(axis=axis, out=out, keepdims=keepdims)


def argmax_along_batch(
    a: TBatchedArray,
    out: np.ndarray | None = None,
    *,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return the indices of the maximum values along the batch axis.

    Args:
        a: The input array.
        out: If provided, the result will be inserted into this
            array. It should be of the appropriate shape and dtype.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the array.

    Returns:
        The indices of the maximum values along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba2.argmax_along_batch(batch)
    array([1, 0, 1])
    >>> ba2.argmax_along_batch(batch, keepdims=True)
    array([[1, 0, 1]])

    ```
    """
    return a.argmax_along_batch(out=out, keepdims=keepdims)


@implements(np.argmin)
def argmin(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    out: np.ndarray | None = None,
    *,
    keepdims: bool = False,
) -> np.ndarray:
    r"""See ``numpy.argmin`` documentation."""
    return a.argmin(axis=axis, out=out, keepdims=keepdims)


def argmin_along_batch(
    a: TBatchedArray,
    out: np.ndarray | None = None,
    *,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return the indices of the minimum values along the batch axis.

    Args:
        a: The input array.
        out: If provided, the result will be inserted into this
            array. It should be of the appropriate shape and dtype.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the array.

    Returns:
        The indices of the minimum values along the batch axis.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
    >>> ba2.argmin_along_batch(batch)
    array([0, 1, 0])
    >>> ba2.argmin_along_batch(batch, keepdims=True)
    array([[0, 1, 0]])

    ```
    """
    return a.argmin_along_batch(out=out, keepdims=keepdims)


@implements(np.nanargmax)
def nanargmax(
    a: TBatchedArray,
    axis: SupportsIndex | None = None,
    out: np.ndarray | None = None,
    *,
    keepdims: bool = False,
) -> np.ndarray:
    r"""See ``numpy.nanargmax`` documentation."""
    return a.nanargmax(axis=axis, out=out, keepdims=keepdims)


def nanargmax_along_batch(
    a: TBatchedArray,
    out: np.ndarray | None = None,
    *,
    keepdims: bool = False,
) -> np.ndarray:
    r"""Return the indices of the maximum values along the batch axis
    ignoring NaNs.

    Args:
        a: The input array.
        out: If provided, the result will be inserted into this
            array. It should be of the appropriate shape and dtype.
        keepdims: If this is set to True, the axes which are
            reduced are left in the result as dimensions with size
            one. With this option, the result will broadcast
            correctly against the array.

    Returns:
        The indices of the maximum values along the batch axis
            ignoring NaNs.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
    >>> ba2.nanargmax_along_batch(batch)
    array([1, 1, 1])
    >>> ba2.nanargmax_along_batch(batch, keepdims=True)
    array([[1, 1, 1]])

    ```
    """
    return a.nanargmax_along_batch(out=out, keepdims=keepdims)
