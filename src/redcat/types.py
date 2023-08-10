from __future__ import annotations

__all__ = ["IndexType", "IndicesType", "RNGType", "RNGOrSeedType"]

import random
from collections import namedtuple
from collections.abc import Sequence
from typing import TypeVar, Union
from unittest.mock import Mock

from black import Any
from coola import objects_are_equal
from coola.types import Tensor, ndarray
from coola.utils import is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()


if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

T = TypeVar("T")

IndexType = Union[int, slice, list[int], Tensor, None]

IndicesType = Union[Sequence[int], Tensor, ndarray]

RNGType = Union[random.Random, np.random.Generator, torch.Generator]
RNGOrSeedType = Union[np.random.Generator, torch.Generator, random.Random, int, None]


class SortReturnType(namedtuple("SortReturnType", ["values", "indices"])):
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SortReturnType):
            return False
        return objects_are_equal(self.values, other.values) and objects_are_equal(
            self.indices, other.indices
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)
