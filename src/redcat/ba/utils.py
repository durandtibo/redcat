r"""Contain utility functions for ``BatchedArray``."""

from __future__ import annotations

__all__ = [
    "check_data_and_axis",
    "check_same_batch_axis",
    "get_batch_axes",
]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from numpy import ndarray


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
        msg = f"The batch axes do not match. Received multiple values: {axes}"
        raise RuntimeError(msg)


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
    >>> from redcat.ba import check_data_and_axis
    >>> check_data_and_axis(np.ones((2, 3)), batch_axis=0)

    ```
    """
    ndim = data.ndim
    if ndim < 1:
        msg = f"data needs at least 1 axis (received: {ndim})"
        raise RuntimeError(msg)
    if batch_axis < 0 or batch_axis >= ndim:
        msg = f"Incorrect `batch_axis` ({batch_axis}) but the value should be in [0, {ndim - 1}]"
        raise RuntimeError(msg)


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
