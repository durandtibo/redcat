from __future__ import annotations

__all__ = [
    "check_batch_dims",
    "check_data_and_dim",
    "check_seq_dims",
    "get_batch_dims",
    "get_seq_dims",
    "swap2",
]

import copy
from collections.abc import Iterable, Mapping, MutableSequence
from typing import Any, TypeVar, overload

import numpy as np
from numpy import ndarray
from torch import Tensor

from redcat.base import BaseBatch

T = TypeVar("T")


def check_batch_dims(dims: set[int]) -> None:
    r"""Gets the batch dimensions from the inputs.

    Args:
        dims: Specifies the batch dims to check.

    Raises:
        RuntimeError: if there are more than one batch dimension.

    Example usage:

    ```pycon

    >>> from redcat.utils.common import check_batch_dims
    >>> check_batch_dims({0})

    ```
    """
    if len(dims) != 1:
        raise RuntimeError(f"The batch dimensions do not match. Received multiple values: {dims}")


def check_data_and_dim(data: ndarray | Tensor, batch_dim: int) -> None:
    r"""Checks if the array ``data`` and ``batch_dim`` are correct.

    Args:
        data: Specifies the array in the batch.
        batch_dim: Specifies the batch dimension in the array object.

    Raises:
        RuntimeError: if one of the input is incorrect.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> import torch
    >>> from redcat.utils.common import check_data_and_dim
    >>> check_data_and_dim(np.ones((2, 3)), batch_dim=0)
    >>> check_data_and_dim(torch.ones(2, 3), batch_dim=0)

    ```
    """
    ndim = data.ndim
    if ndim < 1:
        raise RuntimeError(f"data needs at least 1 dimensions (received: {ndim})")
    if batch_dim < 0 or batch_dim >= ndim:
        raise RuntimeError(
            f"Incorrect batch_dim ({batch_dim}) but the value should be in [0, {ndim - 1}]"
        )


def check_seq_dims(dims: set[int]) -> None:
    r"""Gets the sequence dimensions from the inputs.

    Args:
        dims: Specifies the sequence dims to check.

    Raises:
        RuntimeError: if there are more than one sequence dimension.

    Example usage:

    ```pycon

    >>> from redcat.utils.common import check_seq_dims
    >>> check_seq_dims({1})

    ```
    """
    if len(dims) != 1:
        raise RuntimeError(
            f"The sequence dimensions do not match. Received multiple values: {dims}"
        )


def get_batch_dims(args: Iterable[Any], kwargs: Mapping[str, Any] | None = None) -> set[int]:
    r"""Gets the batch dimensions from the inputs.

    Args:
        args: Variable length argument list.
        kwargs: Arbitrary keyword arguments.

    Returns:
        The batch dimensions.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> import torch
    >>> from redcat import BatchedArray, BatchedTensor
    >>> from redcat.utils.common import get_batch_dims
    >>> get_batch_dims(
    ...     args=(BatchedArray(torch.ones(2, 3)), BatchedArray(torch.ones(2, 6))),
    ...     kwargs={"batch": BatchedArray(torch.ones(2, 4))},
    ... )
    {0}
    >>> get_batch_dims(
    ...     args=(BatchedTensor(torch.ones(2, 3)), BatchedTensor(torch.ones(2, 6))),
    ...     kwargs={"batch": BatchedTensor(torch.ones(2, 4))},
    ... )
    {0}

    ```
    """
    kwargs = kwargs or {}
    dims = {val._batch_dim for val in args if hasattr(val, "_batch_dim")}
    dims.update({val._batch_dim for val in kwargs.values() if hasattr(val, "_batch_dim")})
    return dims


def get_data(data: BaseBatch[T] | Any) -> T:
    r"""Gets the data from a batch or the input data.

    Args:
        data: Specifies the data.

    Returns:
        The data.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> import torch
    >>> from redcat import BatchedArray, BatchedTensor
    >>> from redcat.utils.common import get_data
    >>> get_data(BatchedArray(np.ones((2, 3))))
    array([[1., 1., 1.],
           [1., 1., 1.]])
    >>> get_data(BatchedTensor(torch.ones(2, 3)))
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    >>> get_data(torch.ones(2, 3))
    tensor([[1., 1., 1.],
            [1., 1., 1.]])

    ```
    """
    from redcat import BatchedArray  # TODO: remove BatchedArray when finish

    if isinstance(data, (BaseBatch, BatchedArray)):
        data = data.data
    return data


def get_seq_dims(args: Iterable[Any, ...], kwargs: Mapping[str, Any] | None = None) -> set[int]:
    r"""Gets the sequence dimensions from the inputs.

    Args:
        args: Variable length argument list.
        kwargs: Arbitrary keyword arguments.

    Returns:
        The sequence dimensions.

    Example usage:

    ```pycon
    >>> import torch
    >>> from redcat import BatchedTensorSeq
    >>> from redcat.utils.common import get_seq_dims
    >>> get_seq_dims(
    ...     args=(BatchedTensorSeq(torch.ones(2, 3)), BatchedTensorSeq(torch.ones(2, 6))),
    ...     kwargs={"batch": BatchedTensorSeq(torch.ones(2, 4))},
    ... )
    {1}

    ```
    """
    kwargs = kwargs or {}
    dims = {val._seq_dim for val in args if hasattr(val, "_seq_dim")}
    dims.update({val._seq_dim for val in kwargs.values() if hasattr(val, "_seq_dim")})
    return dims


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

    The swap is performed in-place.

    Args:
        sequence: Specifies the sequence to update.
        index0: Specifies the index of the first value to swap.
        index1: Specifies the index of the second value to swap.

    Returns:
        The updated sequence.

    Example usage:

    ```pycon
    >>> from redcat.utils.common import swap2
    >>> seq = [1, 2, 3, 4, 5]
    >>> swap2(seq, 2, 0)
    >>> seq
    [3, 2, 1, 4, 5]

    ```
    """
    tmp = copy.deepcopy(sequence[index0])
    sequence[index0] = sequence[index1]
    sequence[index1] = tmp
    return sequence
