r"""Contain utility functions for ``torch.Tensor``s."""

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
    "to_tensor",
]

from typing import TYPE_CHECKING, Union

import torch
from coola.utils.tensor import get_available_devices
from torch import Tensor

from redcat.base import BaseBatch
from redcat.utils.common import swap2

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

DeviceType = Union[torch.device, str, int]


def align_to_batch_first(tensor: Tensor, batch_dim: int) -> Tensor:
    r"""Aligns the input tensor format to ``(batch_size, *)`` where `*`
    means any number of dimensions.

    Args:
        tensor: Specifies the tensor to change format.
        batch_dim: Specifies the batch dimension in the input tensor.

    Returns:
        A tensor of shape ``(batch_size, *)`` where `*` means any
            number of dimensions.

    Example usage:

    ```pycon
    >>> import torch
    >>> from redcat.utils.tensor import align_to_batch_first
    >>> align_to_batch_first(torch.arange(20).view(4, 5), batch_dim=1)
    tensor([[ 0,  5, 10, 15],
            [ 1,  6, 11, 16],
            [ 2,  7, 12, 17],
            [ 3,  8, 13, 18],
            [ 4,  9, 14, 19]])

    ```
    """
    if batch_dim == 0:
        return tensor
    return tensor.transpose(0, batch_dim)


def align_to_batch_seq(tensor: Tensor, batch_dim: int, seq_dim: int) -> Tensor:
    r"""Aligns the input tensor format to ``(batch_size, sequence_length,
    *)`` where `*` means any number of dimensions.

    Args:
        tensor: Specifies the tensor to change format.
        batch_dim: Specifies the batch dimension in the input tensor.
        seq_dim: Specifies the sequence dimension in the input tensor.

    Returns:
        A tensor of shape ``(batch_size, sequence_length, *)`` where
            `*` means any number of dimensions.

    Example usage:

    ```pycon
    >>> import torch
    >>> from redcat.utils.tensor import align_to_batch_seq
    >>> align_to_batch_seq(torch.arange(20).view(4, 5), batch_dim=1, seq_dim=0)
    tensor([[ 0,  5, 10, 15],
            [ 1,  6, 11, 16],
            [ 2,  7, 12, 17],
            [ 3,  8, 13, 18],
            [ 4,  9, 14, 19]])

    ```
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
        tensor: Specifies the tensor to change format.
        batch_dim: Specifies the batch dimension in the input tensor.
        seq_dim: Specifies the sequence dimension in the input tensor.

    Returns:
        A tensor of shape ``(sequence_length, batch_size, *)`` where
            `*` means any number of dimensions.

    Example usage:

    ```pycon
    >>> import torch
    >>> from redcat.utils.tensor import align_to_seq_batch
    >>> align_to_seq_batch(torch.arange(20).view(4, 5), batch_dim=0, seq_dim=1)
    tensor([[ 0,  5, 10, 15],
            [ 1,  6, 11, 16],
            [ 2,  7, 12, 17],
            [ 3,  8, 13, 18],
            [ 4,  9, 14, 19]])

    ```
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
    r"""Compute the permutation to update the batch and sequence
    dimensions.

    Args:
        num_dims: Specifies the number of dimensions.
        old_batch_dim: Specifies the old batch dimension.
        old_seq_dim: Specifies the old sequence dimension.
        new_batch_dim: Specifies the new batch dimension.
        new_seq_dim: Specifies the new sequence dimension.

    Returns:
        The permutation to update the batch and sequence dimensions.

    Example usage:

    ```pycon
    >>> from redcat.utils.tensor import compute_batch_seq_permutation
    >>> compute_batch_seq_permutation(5, 0, 1, 1, 0)
    [1, 0, 2, 3, 4]
    >>> compute_batch_seq_permutation(2, 0, 1, 1, 0)
    [1, 0]
    >>> compute_batch_seq_permutation(5, 0, 1, 2, 0)
    [1, 2, 0, 3, 4]
    >>> compute_batch_seq_permutation(5, 0, 1, 1, 2)
    [2, 0, 1, 3, 4]

    ```
    """
    if old_batch_dim == old_seq_dim:
        msg = (
            f"Incorrect old_batch_dim ({old_batch_dim}) and old_seq_dim ({old_seq_dim}). "
            "The dimensions should be different"
        )
        raise RuntimeError(msg)
    if new_batch_dim == new_seq_dim:
        msg = (
            f"Incorrect new_batch_dim ({new_batch_dim}) and new_seq_dim ({new_seq_dim}). "
            "The dimensions should be different"
        )
        raise RuntimeError(msg)
    dims = list(range(num_dims))
    swap2(dims, old_batch_dim, new_batch_dim)  # Swap batch dim
    if old_batch_dim == new_seq_dim and old_seq_dim == new_batch_dim:
        return dims  # Swapping batch dims also swaps sequence dims
    if new_batch_dim == old_seq_dim:
        # Update the old sequence dim because it changes during the batch dim swap
        old_seq_dim = old_batch_dim
    swap2(dims, old_seq_dim, new_seq_dim)  # Swap sequence dim
    return dims


def get_torch_generator(
    random_seed: int = 1, device: torch.device | str | None = "cpu"
) -> torch.Generator:
    r"""Create a ``torch.Generator`` initialized with a given seed.

    Args:
        random_seed: Specifies a random seed.
        device: Specifies the desired device for the generator.

    Returns:
        A PyTorch pseudo random number generator.

    Example usage:

    ```pycon
    >>> import torch
    >>> from redcat.utils.tensor import get_torch_generator
    >>> generator = get_torch_generator(42)
    >>> torch.rand(2, 4, generator=generator)
    tensor([[...]])
    >>> generator = get_torch_generator(42)
    >>> torch.rand(2, 4, generator=generator)
    tensor([[...]])

    ```
    """
    generator = torch.Generator(device)
    generator.manual_seed(random_seed)
    return generator


def permute_along_dim(tensor: Tensor, permutation: Tensor, dim: int = 0) -> Tensor:
    r"""Permutes the values of a tensor along a given dimension.

    Args:
        tensor: Specifies the tensor to permute.
        permutation: Specifies the permutation to use on the tensor.
            The dimension of this tensor should be compatible with
            the shape of the tensor to permute.
        dim: Specifies the dimension used to permute the tensor.

    Returns:
        The permuted tensor.

    Example usage:

    ```pycon
    >>> import torch
    >>> from redcat.utils.tensor import permute_along_dim
    >>> permute_along_dim(tensor=torch.arange(4), permutation=torch.tensor([0, 2, 1, 3]))
    tensor([0, 2, 1, 3])
    >>> permute_along_dim(
    ...     tensor=torch.arange(20).view(4, 5),
    ...     permutation=torch.tensor([0, 2, 1, 3]),
    ... )
    tensor([[ 0,  1,  2,  3,  4],
            [10, 11, 12, 13, 14],
            [ 5,  6,  7,  8,  9],
            [15, 16, 17, 18, 19]])
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

    ```
    """
    return tensor.transpose(0, dim)[permutation].transpose(0, dim).contiguous()


def to_tensor(data: BaseBatch | Sequence | Tensor | np.ndarray) -> Tensor:
    r"""Convert the input to a ``torch.Tensor``.

    Args:
        data: Specifies the data to convert to a tensor.

    Returns:
        A tensor.

    Example usage:

    ```pycon
    >>> from redcat.utils.tensor import to_tensor
    >>> x = to_tensor([1, 2, 3, 4, 5])
    >>> x
    tensor([1, 2, 3, 4, 5])

    ```
    """
    if isinstance(data, BaseBatch):
        data = data.data
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)
    return data
