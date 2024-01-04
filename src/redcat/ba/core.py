r"""Implement an extension of ``numpy.ndarray`` to represent a batch.

Notes:
    - https://numpy.org/doc/stable/user/basics.subclassing.html
    - [N-dimensional array (ndarray)](https://numpy.org/doc/1.26/reference/arrays.ndarray.html)
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

    ###############################
    #     Creation operations     #
    ###############################

    def empty_like(self, *args: Any, **kwargs: Any) -> TBatchedArray:
        r"""Return an array without initializing entries, with the same
        shape as the current array.

        Args:
            *args: See the documentation of ``numpy.empty_like``.
            **kwargs: See the documentation of ``numpy.empty_like``.

        Returns:
            An array filled without initializing entries, with the same
                shape as the current array.

        Example usage:

        ```pycon
        >>> from redcat import ba
        >>> array = ba.zeros((2, 3))
        >>> array.empty_like()
        array([...], batch_axis=0)

        ```
        """
        return np.empty_like(self, *args, **kwargs)

    def full_like(self, fill_value: float | ArrayLike, *args: Any, **kwargs: Any) -> TBatchedArray:
        r"""Return an array filled with the scalar value ``1``, with the
        same shape as the current array.

        Args:
            fill_value: Specifies the fill value.
            *args: See the documentation of ``numpy.full_like``.
            **kwargs: See the documentation of ``numpy.full_like``.

        Returns:
            An array filled with the scalar value ``1``, with the same
                shape as the current array.

        Example usage:

        ```pycon
        >>> from redcat import ba
        >>> array = ba.zeros((2, 3))
        >>> array.full_like(42.0)
        array([[42., 42., 42.],
               [42., 42., 42.]], batch_axis=0)

        ```
        """
        return np.full_like(self, *args, fill_value=fill_value, **kwargs)

    def new_full(
        self, fill_value: float | ArrayLike, batch_size: int | None = None, **kwargs: Any
    ) -> TBatchedArray:
        r"""Return an array filled with the scalar value ``fill_value``,
        with the same shape as the current array.

        By default, the array in the returned array has the same
        shape, ``numpy.dtype``  as the current array.

        Args:
            batch_size: Specifies the batch size. If ``None``,
                the batch size of the current batch is used.
            **kwargs: See the documentation of
                ``numpy.new_full``.

        Returns:
            A batch filled with the scalar value ``fill_value``.

        Example usage:

        ```pycon
        >>> from redcat import ba
        >>> array = ba.ones((2, 3))
        >>> array.new_full(42.0)
        array([[42., 42., 42.],
               [42., 42., 42.]], batch_axis=0)
        >>> array.new_full(fill_value=42.0, batch_size=5)
        array([[42., 42., 42.],
               [42., 42., 42.],
               [42., 42., 42.],
               [42., 42., 42.],
               [42., 42., 42.]], batch_axis=0)

        ```
        """
        shape = list(self.shape)
        if batch_size is not None:
            shape[self.batch_axis] = batch_size
        return self.full_like(fill_value=fill_value, shape=shape, **kwargs)

    def new_ones(self, batch_size: int | None = None, **kwargs: Any) -> TBatchedArray:
        r"""Return an array filled with the scalar value ``1``, with the
        same shape as the current array.

        By default, the array in the returned array has the same
        shape, ``numpy.dtype``  as the current array.

        Args:
            batch_size: Specifies the batch size. If ``None``,
                the batch size of the current batch is used.
            **kwargs: See the documentation of
                ``numpy.new_ones``.

        Returns:
            A batch filled with the scalar value ``1``.

        Example usage:

        ```pycon
        >>> from redcat import ba
        >>> array = ba.zeros((2, 3))
        >>> array.new_ones()
        array([[1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)
        >>> array.new_ones(batch_size=5)
        array([[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)

        ```
        """
        shape = list(self.shape)
        if batch_size is not None:
            shape[self.batch_axis] = batch_size
        return self.ones_like(shape=shape, **kwargs)

    def new_zeros(self, batch_size: int | None = None, **kwargs: Any) -> TBatchedArray:
        r"""Return an array filled with the scalar value ``0``, with the
        same shape as the current array.

        By default, the array in the returned array has the same
        shape, ``numpy.dtype``  as the current array.

        Args:
            batch_size: Specifies the batch size. If ``None``,
                the batch size of the current batch is used.
            **kwargs: See the documentation of
                ``numpy.new_zeros``.

        Returns:
            A batch filled with the scalar value ``0``.

        Example usage:

        ```pycon
        >>> from redcat import ba
        >>> array = ba.ones((2, 3))
        >>> array.new_zeros()
        array([[0., 0., 0.],
               [0., 0., 0.]], batch_axis=0)
        >>> array.new_zeros(batch_size=5)
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]], batch_axis=0)

        ```
        """
        shape = list(self.shape)
        if batch_size is not None:
            shape[self.batch_axis] = batch_size
        return self.zeros_like(shape=shape, **kwargs)

    def ones_like(self, *args: Any, **kwargs: Any) -> TBatchedArray:
        r"""Return an array filled with the scalar value ``1``, with the
        same shape as the current array.

        Args:
            *args: See the documentation of ``numpy.ones_like``.
            **kwargs: See the documentation of ``numpy.ones_like``.

        Returns:
            An array filled with the scalar value ``1``, with the same
                shape as the current array.

        Example usage:

        ```pycon
        >>> from redcat import ba
        >>> array = ba.zeros((2, 3))
        >>> array.ones_like()
        array([[1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)

        ```
        """
        return np.ones_like(self, *args, **kwargs)

    def zeros_like(self, *args: Any, **kwargs: Any) -> TBatchedArray:
        r"""Return an array filled with the scalar value ``0``, with the
        same shape as the current array.

        Args:
            *args: See the documentation of ``numpy.zeros_like``.
            **kwargs: See the documentation of ``numpy.zeros_like``.

        Returns:
            An array filled with the scalar value ``0``, with the same
                shape as the current array.

        Example usage:

        ```pycon
        >>> from redcat import ba
        >>> array = ba.ones((2, 3))
        >>> array.zeros_like()
        array([[0., 0., 0.],
               [0., 0., 0.]], batch_axis=0)

        ```
        """
        return np.zeros_like(self, *args, **kwargs)

    #################################
    #     Comparison operations     #
    #################################

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

    ##################################################
    #     Mathematical | arithmetical operations     #
    ##################################################

    def add(self, other: ndarray | float, alpha: float = 1.0) -> TBatchedArray:
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

    def add_(self, other: ndarray | float, alpha: float = 1.0) -> None:
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

    def sub(self, other: ndarray | float, alpha: float = 1) -> TBatchedArray:
        r"""Subtract the input ``other``, scaled by ``alpha``, to the
        ``self`` array.

        Similar to ``out = self - alpha * other``

        Args:
            other: Specifies the value to subtract.
            alpha: Specifies the scale of the array to substract.

        Returns:
            A new array containing the diffence of the two batchs.

        Example usage:

        ```pycon
        >>> from redcat import ba
        >>> batch = ba.ones((2, 3))
        >>> out = batch.sub(ba.full((2, 3), 2.0))
        >>> batch
        array([[1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)
        >>> out
        array([[-1., -1., -1.],
               [-1., -1., -1.]], batch_axis=0)

        ```
        """
        self._check_valid_axes((self, other))
        return self.__sub__(other * alpha if alpha != 1 else other)

    def sub_(
        self,
        other: ndarray | float,
        alpha: float = 1,
    ) -> None:
        r"""Subtract the input ``other``, scaled by ``alpha``, to the
        ``self`` array.

        Similar to ``self -= alpha * other`` (in-place)

        Args:
            other: Specifies the value to subtract.
            alpha: Specifies the scale of the array to substract.

        Example usage:

        ```pycon
        >>> from redcat import ba
        >>> batch = ba.ones((2, 3))
        >>> batch.sub_(ba.full((2, 3), 2.0))
        >>> batch
        array([[-1., -1., -1.],
               [-1., -1., -1.]], batch_axis=0)

        ```
        """
        self._check_valid_axes((self, other))
        self.__isub__(other * alpha if alpha != 1 else other)

    ################################################
    #     Mathematical | advanced arithmetical     #
    ################################################

    def argsort_along_batch(self, *args: Any, **kwargs: Any) -> TBatchedArray:
        r"""Sort the elements of the batch along the batch axis in
        monotonic order by value.

        Args:
            args: See the documentation of ``numpy.argsort``.
                ``axis`` should not be passed.
            kwargs: See the documentation of ``numpy.argsort``.
                ``axis`` should not be passed.

        Returns:
            The indices that sort the batch along the batch dimension.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.argsort_along_batch()
        array([[0, 0],
               [1, 1],
               [2, 2],
               [3, 3],
               [4, 4]], batch_axis=0)
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.argsort_along_batch()
        array([[0, 1, 2, 3, 4],
               [0, 1, 2, 3, 4]], batch_axis=1)

        ```
        """
        return np.argsort(self, *args, axis=self.batch_axis, **kwargs)

    def _check_valid_axes(self, arrays: Sequence) -> None:
        r"""Check if the axes are valid/compatible.

        Args:
            arrays: Specifies the arrays to check.
        """
        check_same_batch_axis(get_batch_axes(arrays))
