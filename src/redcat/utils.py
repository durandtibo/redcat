__all__ = ["DeviceType", "IndexType", "align_to_batch_first", "get_available_devices", "swap2"]

import copy
from collections.abc import MutableSequence, Sequence
from typing import Union, overload

import numpy as np
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


@overload
def swap2(sequence: Tensor, index0: int, index1: int) -> Tensor:
    r"""``swap2`` for a ``torch.Tensor``."""


@overload
def swap2(sequence: np.ndarray, index0: int, index1: int) -> np.ndarray:
    r"""``swap2`` for a ``numpy.ndarray``."""


@overload
def swap2(sequence: MutableSequence, index0: int, index1: int) -> MutableSequence:
    r"""``swap2`` for a mutable sequence."""


def swap2(
    sequence: Union[Tensor, np.ndarray, MutableSequence], index0: int, index1: int
) -> Union[Tensor, np.ndarray, MutableSequence]:
    r"""Swaps two values in a mutable sequence.

    Args:
        sequence (``torch.Tensor`` or ``numpy.ndarray`` or
            ``MutableSequence``): Specifies the sequence to update.
        index0 (int): Specifies the index of the first value to swap.
        index1 (int): Specifies the index of the second value to swap.

    Returns:
        ``torch.Tensor`` or ``numpy.ndarray`` or ``MutableSequence``:
            The updated sequence.

    Example usage:

    .. code-block:: python

        >>> from redcat.utils import swap2
        >>> seq = [1, 2, 3, 4, 5]
        >>> swap2(seq, 2, 0)
        >>> seq
        [3, 2, 1, 4, 5]
    """
    tmp = copy.deepcopy(sequence[index0])
    sequence[index0] = sequence[index1]
    sequence[index1] = tmp
    return sequence
