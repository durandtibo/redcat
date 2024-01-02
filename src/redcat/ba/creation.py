r"""Contain functions to create ``BatchedArray`` objects.

The functions in this module are designed to be a plug-and-play
replacement of their associated numpy functions.

Notes:
- https://numpy.org/doc/stable/reference/routines.array-creation.html
"""

from __future__ import annotations

__all__ = ["array", "ones", "zeros", "empty", "full"]

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from redcat.ba import BatchedArray


def array(
    data: ArrayLike, dtype: DTypeLike | None = None, *, batch_axis: int = 0, **kwargs
) -> BatchedArray:
    r"""Create an array.

    Equivalent of ``numpy.array`` for ``BatchedArray``.

    Args:
        data: An array, any object exposing the array interface,
            an object whose ``__array__`` method returns an array,
            or any (nested) sequence. If object is a scalar,
            a 0-dimensional array containing object is returned.
        dtype: The desired data-type for the array. If not given,
            NumPy will try to use a default dtype that can represent
            the values (by applying promotion rules when necessary.)
        batch_axis: Specifies the batch axis in the array object.
        **kwargs: See the documentation of ``numpy.array``

    Returns:
        An array object satisfying the specified requirements.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba
    >>> batch = ba.array(np.arange(10).reshape(2, 5))
    >>> batch
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]], batch_axis=0)

    ```
    """
    return BatchedArray(np.array(data, dtype=dtype, **kwargs), batch_axis=batch_axis)


def empty(
    shape: int | Sequence[int],
    dtype: DTypeLike | None = None,
    order: str = "C",
    *,
    batch_axis: int = 0,
    **kwargs,
) -> BatchedArray:
    r"""Return a new array of given shape and type, without initializing
    entries.

    Equivalent of ``numpy.empty`` for ``BatchedArray``.

    Args:
        shape: Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        dtype: The desired data-type for the array.
            Default is ``numpy.float64``.
        order: Whether to store multi-dimensional data in row-major
            (C-style) or column-major (Fortran-style) order in memory.
            Default is ``C``.
        batch_axis: Specifies the batch axis in the array object.
        **kwargs: See the documentation of ``numpy.empty``

    Returns:
        An array object satisfying the specified requirements.

    Example usage:

    ```pycon
    >>> from redcat import ba
    >>> batch = ba.empty((2, 3))
    >>> batch
    array([...], batch_axis=0)

    ```
    """
    return BatchedArray(np.empty(shape, dtype=dtype, order=order, **kwargs), batch_axis=batch_axis)


def full(
    shape: int | Sequence[int],
    fill_value: float | ArrayLike,
    dtype: DTypeLike | None = None,
    order: str = "C",
    *,
    batch_axis: int = 0,
    **kwargs,
) -> BatchedArray:
    r"""Return a new array of given shape and type, filled with
    ``fill_value``.

    Equivalent of ``numpy.full`` for ``BatchedArray``.

    Args:
        shape: Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        fill_value: Specifies the fill value.
        dtype: The desired data-type for the array.
            Default is ``numpy.float64``.
        order: Whether to store multi-dimensional data in row-major
            (C-style) or column-major (Fortran-style) order in memory.
            Default is ``C``.
        batch_axis: Specifies the batch axis in the array object.
        **kwargs: See the documentation of ``numpy.full``

    Returns:
        An array object satisfying the specified requirements.

    Example usage:

    ```pycon
    >>> from redcat import ba
    >>> batch = ba.full((2, 3), fill_value=42)
    >>> batch
    array([[42, 42, 42],
           [42, 42, 42]], batch_axis=0)

    ```
    """
    return BatchedArray(
        np.full(shape, fill_value=fill_value, dtype=dtype, order=order, **kwargs),
        batch_axis=batch_axis,
    )


def ones(
    shape: int | Sequence[int],
    dtype: DTypeLike | None = None,
    order: str = "C",
    *,
    batch_axis: int = 0,
    **kwargs,
) -> BatchedArray:
    r"""Return a new array of given shape and type, filled with ones.

    Equivalent of ``numpy.ones`` for ``BatchedArray``.

    Args:
        shape: Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        dtype: The desired data-type for the array.
            Default is ``numpy.float64``.
        order: Whether to store multi-dimensional data in row-major
            (C-style) or column-major (Fortran-style) order in memory.
            Default is ``C``.
        batch_axis: Specifies the batch axis in the array object.
        **kwargs: See the documentation of ``numpy.ones``

    Returns:
        An array object satisfying the specified requirements.

    Example usage:

    ```pycon
    >>> from redcat import ba
    >>> batch = ba.ones((2, 3))
    >>> batch
    array([[1., 1., 1.],
           [1., 1., 1.]], batch_axis=0)

    ```
    """
    return BatchedArray(np.ones(shape, dtype=dtype, order=order, **kwargs), batch_axis=batch_axis)


def zeros(
    shape: int | Sequence[int],
    dtype: DTypeLike | None = None,
    order: str = "C",
    *,
    batch_axis: int = 0,
    **kwargs,
) -> BatchedArray:
    r"""Return a new array of given shape and type, filled with zeros.

    Equivalent of ``numpy.zeros`` for ``BatchedArray``.

    Args:
        shape: Shape of the new array, e.g., ``(2, 3)`` or ``2``.
        dtype: The desired data-type for the array.
            Default is ``numpy.float64``.
        order: Whether to store multi-dimensional data in row-major
            (C-style) or column-major (Fortran-style) order in memory.
            Default is ``C``.
        batch_axis: Specifies the batch axis in the array object.
        **kwargs: See the documentation of ``numpy.zeros``

    Returns:
        An array object satisfying the specified requirements.

    Example usage:

    ```pycon
    >>> from redcat import ba
    >>> batch = ba.zeros((2, 3))
    >>> batch
    array([[0., 0., 0.],
           [0., 0., 0.]], batch_axis=0)

    ```
    """
    return BatchedArray(np.zeros(shape, dtype=dtype, order=order, **kwargs), batch_axis=batch_axis)
