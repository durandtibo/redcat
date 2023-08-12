from __future__ import annotations

__all__ = ["permute_along_dim", "to_array"]

from collections.abc import Sequence

import numpy as np
import torch
from numpy import ndarray

from redcat.base import BaseBatch


def permute_along_dim(array: ndarray, permutation: ndarray, dim: int = 0) -> ndarray:
    r"""Permutes the values of a array along a given dimension.

    Args:
    ----
        array (``numpy.ndarray``): Specifies the array to permute.
        permutation (``numpy.ndarray`` of type long and shape
            ``(dimension,)``): Specifies the permutation to use on the
            array. The dimension of this array should be compatible
            with the shape of the array to permute.
        dim (int, optional): Specifies the dimension used to permute the
            array. Default: ``0``

    Returns:
    -------
        ``numpy.ndarray``: The permuted array.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from redcat.utils.array import permute_along_dim
        >>> permute_along_dim(np.arange(4), permutation=np.array([0, 2, 1, 3]))
        array([0, 2, 1, 3])
        >>> permute_along_dim(
        ...     np.arange(20).reshape(4, 5),
        ...     permutation=np.array([0, 2, 1, 3]),
        ... )
        array([[ 0,  1,  2,  3,  4],
               [10, 11, 12, 13, 14],
               [ 5,  6,  7,  8,  9],
               [15, 16, 17, 18, 19]])
        >>> permute_along_dim(
        ...     np.arange(20).reshape(4, 5),
        ...     permutation=np.array([0, 4, 2, 1, 3]),
        ...     dim=1,
        ... )
        array([[ 0,  4,  2,  1,  3],
               [ 5,  9,  7,  6,  8],
               [10, 14, 12, 11, 13],
               [15, 19, 17, 16, 18]])
        >>> permute_along_dim(
        ...     np.arange(20).reshape(2, 2, 5),
        ...     permutation=np.array([0, 4, 2, 1, 3]),
        ...     dim=2,
        ... )
        array([[[ 0,  4,  2,  1,  3],
                [ 5,  9,  7,  6,  8]],
               [[10, 14, 12, 11, 13],
                [15, 19, 17, 16, 18]]])
    """
    return np.swapaxes(np.swapaxes(array, 0, dim)[permutation], 0, dim)


def to_array(data: Sequence | torch.Tensor | ndarray) -> ndarray:
    r"""Converts the input to a ``numpy.ndarray``.

    Args:
    ----
        data (``BaseBatch`` or ``Sequence`` or ``torch.Tensor`` or
            ``numpy.ndarray``): Specifies the data to convert to an
            array.

    Returns:
    -------
        ``numpy.ndarray``: An array.

    Example usage:

    .. code-block:: pycon

        >>> from redcat.utils.array import to_array
        >>> x = to_array([1, 2, 3, 4, 5])
        >>> x
        array([1, 2, 3, 4, 5])
    """
    if isinstance(data, BaseBatch):
        data = data.data
    if not isinstance(data, ndarray):
        data = np.asarray(data)
    return data
