from __future__ import annotations

__all__ = [
    "DeviceType",
    "align_to_batch_first",
    "align_to_batch_seq",
    "align_to_seq_batch",
    "compute_batch_seq_permutation",
    "get_available_devices",
    "get_torch_generator",
    "permute_along_dim",
    "swap2",
]

import copy
from collections.abc import MutableSequence
from typing import Union, overload

import numpy as np
import torch
from torch import Tensor

DeviceType = Union[torch.device, str, int]


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
        >>> from redcat.utils.tensor import align_to_batch_first
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


def align_to_batch_seq(tensor: Tensor, batch_dim: int, seq_dim: int) -> Tensor:
    r"""Aligns the input tensor format to ``(batch_size, sequence_length,
    *)`` where `*` means any number of dimensions.

    Args:
        tensor (``torch.Tensor``): Specifies the tensor to change
            format.
        batch_dim (int): Specifies the batch dimension in the input
            tensor.
        seq_dim (int): Specifies the sequence dimension in the input
            tensor.

    Returns:
        ``torch.Tensor``: A tensor of shape
            ``(batch_size, sequence_length, *)`` where `*` means any
            number of dimensions.
    """
    return tensor.permute(
        *compute_batch_seq_permutation(
            num_dims=tensor.dim(),
            old_batch_dim=batch_dim,
            old_seq_dim=seq_dim,
            new_batch_dim=0,
            new_seq_dim=1,
        )
    )


def align_to_seq_batch(tensor: Tensor, batch_dim: int, seq_dim: int) -> Tensor:
    r"""Aligns the input tensor format to ``(sequence_length, batch_size,
    *)`` where `*` means any number of dimensions.

    Args:
        tensor (``torch.Tensor``): Specifies the tensor to change
            format.
        batch_dim (int): Specifies the batch dimension in the input
            tensor.
        seq_dim (int): Specifies the sequence dimension in the input
            tensor.

    Returns:
        ``torch.Tensor``: A tensor of shape
            ``(sequence_length, batch_size, *)`` where `*` means any
            number of dimensions.
    """
    return tensor.permute(
        *compute_batch_seq_permutation(
            num_dims=tensor.dim(),
            old_batch_dim=batch_dim,
            old_seq_dim=seq_dim,
            new_batch_dim=1,
            new_seq_dim=0,
        )
    )


def compute_batch_seq_permutation(
    num_dims: int,
    old_batch_dim: int,
    old_seq_dim: int,
    new_batch_dim: int,
    new_seq_dim: int,
) -> list[int]:
    r"""Computes the permutation to update the batch and sequence
    dimensions.

    Args:
        num_dims (int): Specifies the number of dimensions.
        old_batch_dim (int): Specifies the old batch dimension.
        old_seq_dim (int): Specifies the old sequence dimension.
        new_batch_dim (int): Specifies the new batch dimension.
        new_seq_dim (int): Specifies the new sequence dimension.

    Returns:
        list: The permutation to update the batch and sequence
            dimensions.

    Example usage:

    .. code-block:: python

        >>> from redcat.utils.tensor import compute_batch_seq_permutation
        >>> compute_batch_seq_permutation(5, 0, 1, 1, 0)
        [1, 0, 2, 3, 4]
        >>> compute_batch_seq_permutation(2, 0, 1, 1, 0)
        [1, 0]
        >>> compute_batch_seq_permutation(5, 0, 1, 2, 0)
        [1, 2, 0, 3, 4]
        >>> compute_batch_seq_permutation(5, 0, 1, 1, 2)
        [2, 0, 1, 3, 4]
    """
    if old_batch_dim == old_seq_dim:
        raise RuntimeError(
            f"Incorrect old_batch_dim ({old_batch_dim}) and old_seq_dim ({old_seq_dim}). "
            "The dimensions should be different"
        )
    if new_batch_dim == new_seq_dim:
        raise RuntimeError(
            f"Incorrect new_batch_dim ({new_batch_dim}) and new_seq_dim ({new_seq_dim}). "
            "The dimensions should be different"
        )
    dims = list(range(num_dims))
    swap2(dims, old_batch_dim, new_batch_dim)  # Swap batch dim
    if old_batch_dim == new_seq_dim and old_seq_dim == new_batch_dim:
        return dims  # Swapping batch dims also swaps sequence dims
    if new_batch_dim == old_seq_dim:
        # Update the old sequence dim because it changes during the batch dim swap
        old_seq_dim = old_batch_dim
    swap2(dims, old_seq_dim, new_seq_dim)  # Swap sequence dim
    return dims


def get_available_devices() -> tuple[str, ...]:
    r"""Gets the available PyTorch devices on the machine.

    Returns
    -------
        tuple: The available devices.

    Example usage:

    .. code-block:: python

        >>> from redcat.utils.tensor import get_available_devices
        >>> get_available_devices()
        ('cpu', 'cuda:0')
    """
    if torch.cuda.is_available():
        return ("cpu", "cuda:0")
    if torch.backends.mps.is_available():
        return ("cpu", "mps:0")
    return ("cpu",)


def get_torch_generator(
    random_seed: int = 1, device: torch.device | str | None = "cpu"
) -> torch.Generator:
    r"""Creates a ``torch.Generator`` initialized with a given seed.

    Args:
    ----
        random_seed (int, optional): Specifies a random seed.
            Default: ``1``
        device (``torch.device`` or str or ``None``, optional):
            Specifies the desired device for the generator.
            Default: ``'cpu'``

    Returns:
    -------
        ``torch.Generator``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from redcat.utils.tensor import get_torch_generator
        >>> generator = get_torch_generator(42)
        >>> torch.rand(2, 4, generator=generator)
        tensor([[0.8823, 0.9150, 0.3829, 0.9593],
                [0.3904, 0.6009, 0.2566, 0.7936]])
        >>> generator = get_torch_generator(42)
        >>> torch.rand(2, 4, generator=generator)
        tensor([[0.8823, 0.9150, 0.3829, 0.9593],
                [0.3904, 0.6009, 0.2566, 0.7936]])
    """
    generator = torch.Generator(device)
    generator.manual_seed(random_seed)
    return generator


def permute_along_dim(tensor: Tensor, permutation: Tensor, dim: int = 0) -> Tensor:
    r"""Permutes the values of a tensor along a given dimension.

    Args:
    ----
        tensor (``torch.Tensor``): Specifies the tensor to permute.
        permutation (``torch.Tensor`` of type long and shape
            ``(dimension,)``): Specifies the permutation to use on the
            tensor. The dimension of this tensor should be compatible
            with the shape of the tensor to permute.
        dim (int, optional): Specifies the dimension used to permute the
            tensor. Default: ``0``

    Returns:
    -------
        ``torch.Tensor``: The permuted tensor.

    Example usage:

    .. code-block:: python

        >>> from redcat.utils.tensor import permute_along_dim
        >>> permute_along_dim(tensor=torch.arange(4), permutation=torch.tensor([0, 2, 1, 3]))
        tensor([0, 2, 1, 3])
        >>> permute_along_dim(
        ...     tensor=torch.arange(20).view(4, 5),
        ...     permutation=torch.tensor([0, 2, 1, 3]),
        ... )
        tensor([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [15, 16, 17, 18, 19]])
        >>> permute_along_dim(
        ...     tensor=torch.arange(20).view(4, 5),
        ...     permutation=torch.tensor([0, 4, 2, 1, 3]),
        ...     dim=1,
        ... )
        tensor([[ 0,  4,  2,  1,  3],
                [ 5,  9,  7,  6,  8],
                [10, 14, 12, 11, 13],
                [15, 19, 17, 16, 18]])
        >>> permute_along_dim(
        ...     tensor=torch.arange(20).view(2, 2, 5),
        ...     permutation=torch.tensor([0, 4, 2, 1, 3]),
        ...     dim=2,
        ... )
        tensor([[[ 0,  4,  2,  1,  3],
                 [ 5,  9,  7,  6,  8]],
                [[10, 14, 12, 11, 13],
                 [15, 19, 17, 16, 18]]])
    """
    return tensor.transpose(0, dim)[permutation].transpose(0, dim).contiguous()


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
    sequence: Tensor | np.ndarray | MutableSequence, index0: int, index1: int
) -> Tensor | np.ndarray | MutableSequence:
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

        >>> from redcat.utils.tensor import swap2
        >>> seq = [1, 2, 3, 4, 5]
        >>> swap2(seq, 2, 0)
        >>> seq
        [3, 2, 1, 4, 5]
    """
    tmp = copy.deepcopy(sequence[index0])
    sequence[index0] = sequence[index1]
    sequence[index1] = tmp
    return sequence
