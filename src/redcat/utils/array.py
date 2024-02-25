r"""Contain utility functions for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = [
    "arrays_share_data",
    "get_data_base",
    "get_div_rounding_operator",
    "permute_along_axis",
    "to_array",
]

from typing import TYPE_CHECKING, Callable

import numpy as np

from redcat.base import BaseBatch

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


def get_div_rounding_operator(mode: str | None) -> Callable:
    r"""Get the rounding operator for a division.

    Args:
        mode: Specifies the type of rounding applied to the result.
            - ``None``: true division.
            - ``"floor"``: floor division.

    Returns:
        The rounding operator for a division

    Example usage:

    ```pycon
    >>> from redcat.utils.array import get_div_rounding_operator
    >>> get_div_rounding_operator(None)
    <ufunc 'divide'>

    ```
    """
    if mode is None:
        return np.true_divide
    if mode == "floor":
        return np.floor_divide
    msg = f"Incorrect `rounding_mode` {mode}. Valid values are: None and 'floor'"
    raise RuntimeError(msg)


def permute_along_axis(array: np.ndarray, permutation: np.ndarray, axis: int = 0) -> np.ndarray:
    r"""Permutes the values of a array along a given axis.

    Args:
        array: Specifies the array to permute.
        permutation: Specifies the permutation to use on the array.
            The dimension of this array should be compatible with the
            shape of the array to permute.
        axis: Specifies the axis used to permute the array.

    Returns:
        The permuted array.

    Example usage:

    ```pycon
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

    ```
    """
    return np.swapaxes(np.swapaxes(array, 0, axis)[permutation], 0, axis)


def to_array(data: BaseBatch | Sequence | torch.Tensor | np.ndarray) -> np.ndarray:
    r"""Convert the input to a ``numpy.ndarray``.

    Args:
        data: Specifies the data to convert to an array.

    Returns:
        A NumPy array.

    Example usage:

    ```pycon
    >>> from redcat.utils.array import to_array
    >>> x = to_array([1, 2, 3, 4, 5])
    >>> x
    array([1, 2, 3, 4, 5])

    ```
    """
    if isinstance(data, BaseBatch):
        data = data.data
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    return data


def arrays_share_data(x: np.ndarray, y: np.ndarray) -> bool:
    r"""Indicate if two arrays share the same data.

    Args:
        x: Specifies the first array.
        y: Specifies the second array.

    Returns:
        ``True`` if the two arrays share the same data, otherwise ``False``.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat.utils.array import arrays_share_data
    >>> x = np.ones((2, 3))
    >>> arrays_share_data(x, x)
    True
    >>> arrays_share_data(x, x.copy())
    False
    >>> y = x[1:]
    >>> arrays_share_data(x, y)
    True

    ```
    """
    return get_data_base(x) is get_data_base(y)


def get_data_base(array: np.ndarray) -> np.ndarray:
    r"""Return the base array that owns the actual data.

    Args:
        array: Specifies the input array.

    Returns:
        The array that owns the actual data.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat.utils.array import get_data_base
    >>> x = np.ones((2, 3))
    >>> get_data_base(x)
    array([[1., 1., 1.],
           [1., 1., 1.]])
    >>> y = x[1:]
    >>> get_data_base(y)
    array([[1., 1., 1.],
           [1., 1., 1.]])

    ```
    """
    while isinstance(array.base, np.ndarray):
        array = array.base
    return array
