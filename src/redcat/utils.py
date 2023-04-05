__all__ = ["DeviceType", "IndexType", "get_available_devices"]

from collections.abc import Sequence
from typing import Union

import torch
from torch import Tensor

DeviceType = Union[torch.device, str, int]
IndexType = Union[None, int, slice, str, Tensor, Sequence]


def get_available_devices() -> tuple[str, ...]:
    r"""Gets the available PyTorch devices on the machine.

    Returns
    -------
        tuple: The available devices.

    Example usage:

    .. code-block:: python

        >>> from redcat.utils import get_available_devices
        >>> get_available_devices()
        ('cpu', 'cuda:0')
    """
    if torch.cuda.is_available():
        return ("cpu", "cuda:0")
    return ("cpu",)
