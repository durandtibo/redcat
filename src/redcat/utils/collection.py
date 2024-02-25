r"""Contain utility functions for collection objects."""

from __future__ import annotations

__all__ = ["to_list"]

from numpy import ndarray
from torch import Tensor

from redcat import BaseBatch


def to_list(data: list | tuple | Tensor | ndarray | BaseBatch) -> list:
    r"""Convert an input data to a list.

    Args:
        data: Specifies the data to convert.

    Returns:
        The data.

    Example usage:

    ```pycon
    >>> from redcat import BatchList
    >>> from redcat.utils.collection import to_list
    >>> to_list(BatchList([1, 2, 3]))
    [1, 2, 3]

    ```
    """
    if isinstance(data, list):
        return data
    if isinstance(data, BaseBatch):
        data = data.data
    if isinstance(data, (Tensor, ndarray)):
        return data.tolist()
    return list(data)
