__all__ = ["DeviceType", "IndexType", "align_to_batch_first", "get_available_devices"]

from collections.abc import Sequence
from typing import Union

import torch
from torch import Tensor

DeviceType = Union[torch.device, str, int]
IndexType = Union[None, int, slice, str, Tensor, Sequence]


def align_to_batch_first(tensor: Tensor, batch_dim: int) -> Tensor:
    r"""Aligns the input tensor format to ``(batch_size, *)`` where `*` means
    any number of dimensions.

    Args:
        tensor (``torch.Tensor``): Specifies the tensor to change
            format.
        batch_dim (int): Specifies the batch dimension in the input
            tensor.

    Returns:
        ``torch.Tensor``: A tensor of shape ``(batch_size, *)`` where
            `*` means any number of dimensions.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from redcat.utils import align_to_batch_first
        >>> align_to_batch_first(torch.arange(20).view(4, 5), batch_dim=1)
        tensor([[ 0,  5, 10, 15],
                [ 1,  6, 11, 16],
                [ 2,  7, 12, 17],
                [ 3,  8, 13, 18],
                [ 4,  9, 14, 19]])
    """
    if batch_dim == 0:
        return tensor
    return tensor.transpose(0, batch_dim)


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
