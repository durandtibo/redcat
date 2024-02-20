from __future__ import annotations

__all__ = [
    "sort",
    "sort_along_batch",
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
