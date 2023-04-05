__all__ = ["DeviceType", "IndexType"]

from collections.abc import Sequence
from typing import Union

import torch
from torch import Tensor

DeviceType = Union[torch.device, str, int]
IndexType = Union[None, int, slice, str, Tensor, Sequence]
