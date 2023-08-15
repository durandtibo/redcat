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
from typing import Any, overload

import numpy as np
from numpy import ndarray
from torch import Tensor


def check_batch_dims(dims: set[int]) -> None:
    r"""Gets the batch dimensions from the inputs.

    Args:
    ----
        dims (set): Specifies the batch dims to check.

    Raises:
    ------
        RuntimeError if there are more than one batch dimension.

    Example usage:

    .. code-block:: pycon

        >>> from redcat.utils.common import check_batch_dims
        >>> check_batch_dims({0})
    """
    if len(dims) != 1:
        raise RuntimeError(f"The batch dimensions do not match. Received multiple values: {dims}")


def check_data_and_dim(data: ndarray | Tensor, batch_dim: int) -> None:
    r"""Checks if the array ``data`` and ``batch_dim`` are correct.

    Args:
    ----
        data ( ``numpy.ndarray`` or ``torch.Tensor``): Specifies
            the array in the batch.
        batch_dim (int): Specifies the batch dimension in the
            array object.

    Raises:
    ------
        RuntimeError: if one of the input is incorrect.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> import torch
        >>> from redcat.utils.common import check_data_and_dim
        >>> check_data_and_dim(np.ones((2, 3)), batch_dim=0)
        >>> check_data_and_dim(torch.ones(2, 3), batch_dim=0)
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
    ----
        dims (set): Specifies the sequence dims to check.

    Raises:
    ------
        RuntimeError if there are more than one sequence dimension.

    Example usage:

    .. code-block:: pycon

        >>> from redcat.utils.common import get_seq_dims
        >>> get_seq_dims({1})
    """
    if len(dims) != 1:
        raise RuntimeError(
            f"The sequence dimensions do not match. Received multiple values: {dims}"
        )


def get_batch_dims(args: Iterable[Any], kwargs: Mapping[str, Any] | None = None) -> set[int]:
    r"""Gets the batch dimensions from the inputs.

    Args:
    ----
        args: Variable length argument list.
        kwargs: Arbitrary keyword arguments.

    Returns:
    -------
        set: The batch dimensions.

    Example usage:

    .. code-block:: pycon

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
    """
    kwargs = kwargs or {}
    dims = {val._batch_dim for val in args if hasattr(val, "_batch_dim")}
    dims.update({val._batch_dim for val in kwargs.values() if hasattr(val, "_batch_dim")})
    return dims


def get_seq_dims(args: Iterable[Any, ...], kwargs: Mapping[str, Any] | None = None) -> set[int]:
    r"""Gets the sequence dimensions from the inputs.

    Args:
    ----
        args: Variable length argument list.
        kwargs: Arbitrary keyword arguments.

    Returns:
    -------
        set: The sequence dimensions.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from redcat import BatchedTensorSeq
        >>> from redcat.utils.common import get_seq_dims
        >>> get_seq_dims(
        ...     args=(BatchedTensorSeq(torch.ones(2, 3)), BatchedTensorSeq(torch.ones(2, 6))),
        ...     kwargs={"batch": BatchedTensorSeq(torch.ones(2, 4))},
        ... )
        {1}
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
    ----
        sequence (``torch.Tensor`` or ``numpy.ndarray`` or
            ``MutableSequence``): Specifies the sequence to update.
        index0 (int): Specifies the index of the first value to swap.
        index1 (int): Specifies the index of the second value to swap.

    Returns:
    -------
        ``torch.Tensor`` or ``numpy.ndarray`` or ``MutableSequence``:
            The updated sequence.

    Example usage:

    .. code-block:: pycon

        >>> from redcat.utils.common import swap2
        >>> seq = [1, 2, 3, 4, 5]
        >>> swap2(seq, 2, 0)
        >>> seq
        [3, 2, 1, 4, 5]
    """
    tmp = copy.deepcopy(sequence[index0])
    sequence[index0] = sequence[index1]
    sequence[index1] = tmp
    return sequence
