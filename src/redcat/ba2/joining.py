from __future__ import annotations

__all__ = ["concatenate", "concatenate_along_batch"]

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

from redcat.ba2.core import BatchedArray, implements

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")


@implements(np.concatenate)
def concatenate(
    arrays: Sequence[TBatchedArray], axis: int | None = 0
) -> TBatchedArray | np.ndarray:
    r"""See ``numpy.concatenate`` documentation."""
    return arrays[0].concatenate(arrays[1:], axis=axis)


def concatenate_along_batch(arrays: Sequence[TBatchedArray]) -> TBatchedArray | np.ndarray:
    r"""Join a sequence of arrays along the batch axis.

    Args:
        arrays: The arrays must have the same shape, except in the
            dimension corresponding to axis.

    Returns:
        The concatenated array.

    Raises:
        RuntimeError: if the batch axes are different.

    Example usage:

    ```pycon
    >>> from redcat import ba2
    >>> arrays = [
    ...     ba2.batched_array([[0, 1, 2], [4, 5, 6]]),
    ...     ba2.batched_array([[10, 11, 12], [13, 14, 15]]),
    ... ]
    >>> out = ba2.concatenate_along_batch(arrays)
    >>> out
    array([[ 0,  1,  2],
           [ 4,  5,  6],
           [10, 11, 12],
           [13, 14, 15]], batch_axis=0)

    ```
    """
    return arrays[0].concatenate_along_batch(arrays[1:])
