r"""Contain joining functions for ``BatchedArray``."""

from __future__ import annotations

__all__ = ["concatenate", "concatenate_along_batch"]

from typing import TYPE_CHECKING, TypeVar

import numpy as np

from redcat.ba.core import BatchedArray, implements

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    >>> from redcat import ba
    >>> arrays = [
    ...     ba.array([[0, 1, 2], [4, 5, 6]]),
    ...     ba.array([[10, 11, 12], [13, 14, 15]]),
    ... ]
    >>> out = ba.concatenate_along_batch(arrays)
    >>> out
    array([[ 0,  1,  2],
           [ 4,  5,  6],
           [10, 11, 12],
           [13, 14, 15]], batch_axis=0)

    ```
    """
    return arrays[0].concatenate_along_batch(arrays[1:])
