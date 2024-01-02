from __future__ import annotations

__all__ = ["check_same_batch_axis", "get_batch_axes"]

from collections.abc import Iterable, Mapping
from typing import Any


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


def get_batch_axes(args: Iterable[Any], kwargs: Mapping[str, Any] | None = None) -> set[int]:
    r"""Return batch axes from the inputs.

    Args:
        args: Variable length argument list.
        kwargs: Arbitrary keyword arguments.

    Returns:
        The batch axes.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> import torch
    >>> from redcat import BatchedArray
    >>> from redcat.ba import get_batch_axes
    >>> get_batch_axes(
    ...     args=(BatchedArray(torch.ones(2, 3)), BatchedArray(torch.ones(2, 6))),
    ...     kwargs={"batch": BatchedArray(torch.ones(2, 4))},
    ... )
    {0}

    ```
    """
    kwargs = kwargs or {}
    axes = {val._batch_dim for val in args if hasattr(val, "_batch_dim")}
    axes.update({val._batch_dim for val in kwargs.values() if hasattr(val, "_batch_dim")})
    return axes
