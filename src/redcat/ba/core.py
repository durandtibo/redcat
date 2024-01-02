r"""Implement an extension of ``numpy.ndarray`` to represent a batch.

Notes:
    - https://numpy.org/doc/stable/user/basics.subclassing.html
"""

from __future__ import annotations

__all__ = ["BatchedArray"]

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from coola import objects_are_allclose, objects_are_equal
from numpy import ndarray
from numpy.typing import ArrayLike

from redcat.ba.utils import check_data_and_axis, check_same_batch_axis, get_batch_axes

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")


class BatchedArray(ndarray):
    def __new__(cls, data: ArrayLike, batch_axis: int = 0) -> TBatchedArray:
        obj = np.array(data, copy=False, subok=True).view(cls)
        check_data_and_axis(obj, batch_axis)
        obj.batch_axis = batch_axis
        return obj

    def __array_finalize__(self, obj: BatchedArray | None) -> None:
        # if obj is None:
        #     return
        self.batch_axis = getattr(obj, "batch_axis", 0)

    def __repr__(self) -> str:
        return repr(self.__array__())[:-1] + f", batch_axis={self.batch_axis})"

    def __str__(self) -> str:
        return str(self.__array__()) + f"\nwith batch_axis={self.batch_axis}"

    @property
    def batch_size(self) -> int:
        return self.shape[self.batch_axis]

    def allclose(
        self, other: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
    ) -> bool:
        if not isinstance(other, self.__class__) or self.batch_axis != other.batch_axis:
            return False
        return objects_are_allclose(
            self.__array__(), other.__array__(), rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    def allequal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__) or self.batch_axis != other.batch_axis:
            return False
        return objects_are_equal(self.__array__(), other.__array__())

    def add(self, other: ndarray | int | float, alpha: int | float = 1.0) -> TBatchedArray:
        r"""Add the input ``other``, scaled by ``alpha``, to the ``self``
        array.

        Similar to ``out = self + alpha * other``

        Args:
            other: Specifies the other value to add to the current
                array.
            alpha: Specifies the scale of the array to add.

        Returns:
            A new array containing the addition of the two arrays.

        Example usage:

        ```pycon
        >>> from redcat import ba
        >>> array = ba.ones((2, 3))
        >>> out = array.add(ba.full((2, 3), 2.0))
        >>> array
        array([[1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)
        >>> out
        array([[3., 3., 3.],
               [3., 3., 3.]], batch_axis=0)

        ```
        """
        self._check_valid_axes((self, other))
        return self.__add__(other * alpha if alpha != 1 else other)

    def add_(self, other: ndarray | int | float, alpha: int | float = 1.0) -> None:
        r"""Add the input ``other``, scaled by ``alpha``, to the ``self``
        array.

        Similar to ``self += alpha * other`` (in-place)

        Args:
            other: Specifies the other value to add to the current
                array.
            alpha: Specifies the scale of the array to add.

        Example usage:

        ```pycon
        >>> from redcat import ba
        >>> array = ba.ones((2, 3))
        >>> array.add_(ba.full((2, 3), 2.0))
        >>> array
        array([[3., 3., 3.],
               [3., 3., 3.]], batch_axis=0)

        ```
        """
        self._check_valid_axes((self, other))
        self.__iadd__(other * alpha if alpha != 1 else other)

    def _check_valid_axes(self, arrays: Sequence) -> None:
        r"""Check if the axes are valid/compatible.

        Args:
            arrays: Specifies the arrays to check.
        """
        check_same_batch_axis(get_batch_axes(arrays))
