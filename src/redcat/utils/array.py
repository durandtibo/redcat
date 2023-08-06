from __future__ import annotations

__all__ = ["check_data_and_dim"]

from numpy import ndarray
from torch import Tensor


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
        >>> from redcat.array import check_data_and_dim
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
