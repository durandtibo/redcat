from __future__ import annotations

__all__ = ["get_div_rounding_operator", "permute_along_axis", "to_array"]

from collections.abc import Sequence
from typing import Callable

import numpy as np
import torch
from numpy import ndarray

from redcat.base import BaseBatch


def get_div_rounding_operator(mode: str | None) -> Callable:
    r"""Gets the rounding operator for a division.

    Args:
    ----
        mode (str or ``None``, optional): Specifies the
            type of rounding applied to the result.
            - ``None``: true division.
            - ``"floor"``: floor division.
            Default: ``None``

    Returns:
    -------
        ``Callable``: The rounding operator for a division

    Example usage:

    .. code-block:: pycon

        >>> from redcat.utils.array import get_div_rounding_operator
        >>> get_div_rounding_operator(None)
        <ufunc 'divide'>
    """
    if mode is None:
        return np.true_divide
    if mode == "floor":
        return np.floor_divide
    raise RuntimeError(f"Incorrect `rounding_mode` {mode}. Valid values are: None and 'floor'")


def permute_along_axis(array: ndarray, permutation: ndarray, axis: int = 0) -> ndarray:
    r"""Permutes the values of a array along a given axis.

    Args:
    ----
        array (``numpy.ndarray``): Specifies the array to permute.
        permutation (``numpy.ndarray`` of type long and shape
            ``(dimension,)``): Specifies the permutation to use on the
            array. The dimension of this array should be compatible
            with the shape of the array to permute.
        axis (int, optional): Specifies the axis used to permute the
            array. Default: ``0``

    Returns:
    -------
        ``numpy.ndarray``: The permuted array.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from redcat.utils.array import permute_along_axis
        >>> permute_along_axis(np.arange(4), permutation=np.array([0, 2, 1, 3]))
        array([0, 2, 1, 3])
        >>> permute_along_axis(
        ...     np.arange(20).reshape(4, 5),
        ...     permutation=np.array([0, 2, 1, 3]),
        ... )
        array([[ 0,  1,  2,  3,  4],
               [10, 11, 12, 13, 14],
               [ 5,  6,  7,  8,  9],
               [15, 16, 17, 18, 19]])
        >>> permute_along_axis(
        ...     np.arange(20).reshape(4, 5),
        ...     permutation=np.array([0, 4, 2, 1, 3]),
        ...     axis=1,
        ... )
        array([[ 0,  4,  2,  1,  3],
               [ 5,  9,  7,  6,  8],
               [10, 14, 12, 11, 13],
               [15, 19, 17, 16, 18]])
        >>> permute_along_axis(
        ...     np.arange(20).reshape(2, 2, 5),
        ...     permutation=np.array([0, 4, 2, 1, 3]),
        ...     axis=2,
        ... )
        array([[[ 0,  4,  2,  1,  3],
                [ 5,  9,  7,  6,  8]],
               [[10, 14, 12, 11, 13],
                [15, 19, 17, 16, 18]]])
    """
    return np.swapaxes(np.swapaxes(array, 0, axis)[permutation], 0, axis)


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
