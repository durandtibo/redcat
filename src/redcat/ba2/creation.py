r"""Contain functions to create ``BatchedArray`` objects.

The functions in this module are designed to be a plug-and-play
replacement of their associated numpy functions.

Notes and links:
    - [https://numpy.org/doc/stable/reference/routines.array-creation.html](https://numpy.org/doc/stable/reference/routines.array-creation.html)
"""

from __future__ import annotations

__all__ = [
    "array",
    "empty",
    "empty_like",
    "full",
    "full_like",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
]

from collections.abc import Sequence
from typing import Any, Literal, SupportsIndex, TypeVar, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from redcat.ba2 import BatchedArray

TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")

OrderACFK = Literal["A", "C", "F", "K"]
ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]


def array(
    data: ArrayLike | Sequence, dtype: DTypeLike = None, *, batch_axis: int = 0, **kwargs: Any
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
    dtype: DTypeLike = None,
    order: str = "C",
    *,
    batch_axis: int = 0,
    **kwargs: Any,
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


def empty_like(
    a: TBatchedArray,
    dtype: DTypeLike = None,
    order: OrderACFK = "K",
    subok: bool = True,
    shape: ShapeLike = None,
) -> TBatchedArray:
    r"""Return an array of zeros with the same shape and type as a given
    array.

    Equivalent of ``numpy.empty_like`` for ``BatchedArray``.

    Args:
        a: The shape and data-type of a define these same attributes
            of the returned array.
        dtype: Overrides the data type of the result.
        order: Overrides the memory layout of the result. ‘C’ means
            C-order, ‘F’ means F-order, ‘A’ means ‘F’ if ``a`` is
            Fortran contiguous, ‘C’ otherwise. ‘K’ means match the
            layout of ``a`` as closely as possible.
        subok: If True, then the newly created array will use the
            sub-class type of ``a``, otherwise it will be a base-class
            array. Defaults to True.
        shape: Overrides the shape of the result. If order=’K’ and the
            number of dimensions is unchanged, will try to keep order,
            otherwise, order=’C’ is implied.

    Returns:
        Array of zeros with the same shape and type as ``a``.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.empty_like(ba2.ones((2, 5)))
    >>> batch
    array([[...]], batch_axis=0)

    ```
    """
    return BatchedArray(
        np.empty_like(a.data, dtype=dtype, order=order, subok=subok, shape=shape),
        batch_axis=a.batch_axis,
    )


def full(
    shape: int | Sequence[int],
    fill_value: float | ArrayLike,
    dtype: DTypeLike = None,
    order: str = "C",
    *,
    batch_axis: int = 0,
    **kwargs: Any,
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


def full_like(
    a: TBatchedArray,
    fill_value: ArrayLike,
    dtype: DTypeLike = None,
    order: OrderACFK = "K",
    subok: bool = True,
    shape: ShapeLike = None,
) -> TBatchedArray:
    r"""Return an array of ones with the same shape and type as a given
    array.

    Equivalent of ``numpy.ones_like`` for ``BatchedArray``.

    Args:
        a: The shape and data-type of a define these same attributes
            of the returned array.
        dtype: Overrides the data type of the result.
        order: Overrides the memory layout of the result. ‘C’ means
            C-order, ‘F’ means F-order, ‘A’ means ‘F’ if ``a`` is
            Fortran contiguous, ‘C’ otherwise. ‘K’ means match the
            layout of ``a`` as closely as possible.
        subok: If True, then the newly created array will use the
            sub-class type of ``a``, otherwise it will be a base-class
            array. Defaults to True.
        shape: Overrides the shape of the result. If order=’K’ and the
            number of dimensions is unchanged, will try to keep order,
            otherwise, order=’C’ is implied.

    Returns:
        Array of ones with the same shape and type as ``a``.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.full_like(ba2.zeros((2, 5)), fill_value=2)
    >>> batch
    array([[2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.]], batch_axis=0)

    ```
    """
    return BatchedArray(
        np.full_like(
            a.data, fill_value=fill_value, dtype=dtype, order=order, subok=subok, shape=shape
        ),
        batch_axis=a.batch_axis,
    )


def ones(
    shape: int | Sequence[int],
    dtype: DTypeLike = None,
    order: str = "C",
    *,
    batch_axis: int = 0,
    **kwargs: Any,
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


def ones_like(
    a: TBatchedArray,
    dtype: DTypeLike = None,
    order: OrderACFK = "K",
    subok: bool = True,
    shape: ShapeLike = None,
) -> TBatchedArray:
    r"""Return an array of ones with the same shape and type as a given
    array.

    Equivalent of ``numpy.ones_like`` for ``BatchedArray``.

    Args:
        a: The shape and data-type of a define these same attributes
            of the returned array.
        dtype: Overrides the data type of the result.
        order: Overrides the memory layout of the result. ‘C’ means
            C-order, ‘F’ means F-order, ‘A’ means ‘F’ if ``a`` is
            Fortran contiguous, ‘C’ otherwise. ‘K’ means match the
            layout of ``a`` as closely as possible.
        subok: If True, then the newly created array will use the
            sub-class type of ``a``, otherwise it will be a base-class
            array. Defaults to True.
        shape: Overrides the shape of the result. If order=’K’ and the
            number of dimensions is unchanged, will try to keep order,
            otherwise, order=’C’ is implied.

    Returns:
        Array of ones with the same shape and type as ``a``.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.ones_like(ba2.zeros((2, 5)))
    >>> batch
    array([[1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.]], batch_axis=0)

    ```
    """
    return BatchedArray(
        np.ones_like(a.data, dtype=dtype, order=order, subok=subok, shape=shape),
        batch_axis=a.batch_axis,
    )


def zeros(
    shape: int | Sequence[int],
    dtype: DTypeLike = None,
    order: str = "C",
    *,
    batch_axis: int = 0,
    **kwargs: Any,
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


def zeros_like(
    a: TBatchedArray,
    dtype: DTypeLike = None,
    order: OrderACFK = "K",
    subok: bool = True,
    shape: ShapeLike = None,
) -> TBatchedArray:
    r"""Return an array of zeros with the same shape and type as a given
    array.

    Equivalent of ``numpy.zeros_like`` for ``BatchedArray``.

    Args:
        a: The shape and data-type of a define these same attributes
            of the returned array.
        dtype: Overrides the data type of the result.
        order: Overrides the memory layout of the result. ‘C’ means
            C-order, ‘F’ means F-order, ‘A’ means ‘F’ if ``a`` is
            Fortran contiguous, ‘C’ otherwise. ‘K’ means match the
            layout of ``a`` as closely as possible.
        subok: If True, then the newly created array will use the
            sub-class type of ``a``, otherwise it will be a base-class
            array. Defaults to True.
        shape: Overrides the shape of the result. If order=’K’ and the
            number of dimensions is unchanged, will try to keep order,
            otherwise, order=’C’ is implied.

    Returns:
        Array of zeros with the same shape and type as ``a``.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat import ba2
    >>> batch = ba2.zeros_like(ba2.ones((2, 5)))
    >>> batch
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]], batch_axis=0)

    ```
    """
    return BatchedArray(
        np.zeros_like(a.data, dtype=dtype, order=order, subok=subok, shape=shape),
        batch_axis=a.batch_axis,
    )
