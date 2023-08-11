from __future__ import annotations

__all__ = ["to_list"]

from numpy import ndarray
from torch import Tensor

from redcat import BaseBatch


def to_list(data: list | tuple | Tensor | ndarray | BaseBatch) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, BaseBatch):
        data = data.data
    if isinstance(data, (Tensor, ndarray)):
        return data.tolist()
    return list(data)
