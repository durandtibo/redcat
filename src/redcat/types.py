from __future__ import annotations

__all__ = ["IndexType", "IndicesType", "RNGType"]

import random
from collections.abc import Sequence
from typing import TypeVar, Union
from unittest.mock import Mock

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
