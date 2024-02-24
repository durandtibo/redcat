from __future__ import annotations

__all__ = [
    "arrays_share_data",
    "check_data_and_axis",
    "check_same_batch_axis",
    "get_batch_axes",
    "get_data_base",
]

from collections.abc import Iterable, Mapping
from typing import Any

from numpy import ndarray


def arrays_share_data(x: ndarray, y: ndarray) -> bool:
    r"""Indicate if two arrays share the same data.

    Args:
        x: Specifies the first array.
        y: Specifies the second array.

    Returns:
        ``True`` if the two arrays share the same data, otherwise ``False``.

    Example usage:

    ```pycon
    >>> from redcat import ba
    >>> from redcat.ba import arrays_share_data
    >>> x = ba.ones((2, 3))
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


def check_same_batch_axis(axes: set[int]) -> None:
    r"""Check the batch axes are the same.

    Args:
        axes: Specifies the batch axes to check.

    Raises:
        RuntimeError: if there are more than one batch axis.

    Example usage:

    ```pycon
    >>> from redcat.ba import check_same_batch_axis
    >>> check_same_batch_axis({0})

    ```
    """
    if len(axes) != 1:
        raise RuntimeError(f"The batch axes do not match. Received multiple values: {axes}")


def check_data_and_axis(data: ndarray, batch_axis: int) -> None:
    r"""Check if the array ``data`` and ``batch_axis`` are correct.

    Args:
        data: Specifies the array in the batch.
        batch_axis: Specifies the batch axis in the array object.

    Raises:
        RuntimeError: if one of the input is incorrect.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat.ba.utils import check_data_and_axis
    >>> check_data_and_axis(np.ones((2, 3)), batch_axis=0)

    ```
    """
    ndim = data.ndim
    if ndim < 1:
        raise RuntimeError(f"data needs at least 1 axis (received: {ndim})")
    if batch_axis < 0 or batch_axis >= ndim:
        raise RuntimeError(
            f"Incorrect `batch_axis` ({batch_axis}) but the value should be in [0, {ndim - 1}]"
        )


def get_batch_axes(args: Iterable[Any], kwargs: Mapping[str, Any] | None = None) -> set[int]:
    r"""Return batch axes from the inputs.

    Args:
        args: Variable length argument list.
        kwargs: Arbitrary keyword arguments.

    Returns:
        The batch axes.

    Example usage:

    ```pycon
    >>> from redcat import ba
    >>> from redcat.ba import get_batch_axes
    >>> get_batch_axes(
    ...     args=(ba.ones((2, 3)), ba.ones((2, 6))),
    ...     kwargs={"batch": ba.ones((2, 4))},
    ... )
    {0}

    ```
    """
    kwargs = kwargs or {}
    axes = {val.batch_axis for val in args if hasattr(val, "batch_axis")}
    axes.update({val.batch_axis for val in kwargs.values() if hasattr(val, "batch_axis")})
    return axes


def get_data_base(array: ndarray) -> ndarray:
    r"""Return the base array that owns the actual data.

    Args:
        array: Specifies the input array.

    Returns:
        The array that owns the actual data.

    Example usage:

    ```pycon
    >>> from redcat import ba
    >>> from redcat.ba import get_data_base
    >>> x = ba.ones((2, 3))
    >>> get_data_base(x)
    array([[1., 1., 1.],
           [1., 1., 1.]])
    >>> y = x[1:]
    >>> get_data_base(y)
    array([[1., 1., 1.],
           [1., 1., 1.]])

    ```
    """
    while isinstance(array.base, ndarray):
        array = array.base
    return array
