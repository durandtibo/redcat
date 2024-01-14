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
from coola import objects_are_allclose
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

    def allequal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__) or self.batch_axis != other.batch_axis:
            return False
        return np.array_equal(self.__array__(), other.__array__(), equal_nan)

    def eq(self, other: ndarray | bool | float) -> TBatchedArray:
        r"""Return element-wise equality ``(self == other)`` array.

        Args:
            other: Specifies the array to compare.

        Returns:
            A array containing the element-wise equality.

        Example usage:

        ```pycon
        >>> from redcat.ba import BatchedArray
        >>> x1 = BatchedArray(np.array([[1, 3, 4], [0, 2, 2]]))
        >>> x2 = BatchedArray(np.array([[5, 3, 2], [0, 1, 2]]))
        >>> x1.eq(x2)
        array([[False,  True, False],
               [ True, False,  True]], batch_axis=0)
        >>> x1.eq(np.array([[5, 3, 2], [0, 1, 2]]))
        array([[False,  True, False],
               [ True, False,  True]], batch_axis=0)
        >>> x1.eq(2)
        array([[False, False, False],
               [False,  True,  True]], batch_axis=0)

        ```
        """
        return self.__eq__(other)

    def ge(self, other: ndarray | bool | float) -> TBatchedArray:
        r"""Return ``self >= other`` element-wise.

        Args:
            other: Specifies the value to compare with.

        Returns:
            An array containing the element-wise comparison.

        Example usage:

        ```pycon
        >>> from redcat.ba import BatchedArray
        >>> x1 = BatchedArray(np.array([[1, 3, 4], [0, 2, 2]]))
        >>> x2 = BatchedArray(np.array([[5, 3, 2], [0, 1, 2]]))
        >>> x1.ge(x2)
        array([[False,  True,  True],
               [ True,  True,  True]], batch_axis=0)
        >>> x1.ge(np.array([[5, 3, 2], [0, 1, 2]]))
        array([[False,  True,  True],
               [ True,  True,  True]], batch_axis=0)
        >>> x1.ge(2)
        array([[False,  True,  True],
               [False,  True,  True]], batch_axis=0)

        ```
        """
        return self.__ge__(other)

    def gt(self, other: ndarray | bool | float) -> TBatchedArray:
        r"""Return ``self > other`` element-wise.

        Args:
            other: Specifies the batch to compare.

        Returns:
            An array containing the element-wise comparison.

        Example usage:

        ```pycon
        >>> from redcat.ba import BatchedArray
        >>> x1 = BatchedArray(np.array([[1, 3, 4], [0, 2, 2]]))
        >>> x2 = BatchedArray(np.array([[5, 3, 2], [0, 1, 2]]))
        >>> x1.gt(x2)
        array([[False, False,  True],
               [False,  True, False]], batch_axis=0)
        >>> x1.gt(np.array([[5, 3, 2], [0, 1, 2]]))
        array([[False, False,  True],
               [False,  True, False]], batch_axis=0)
        >>> x1.gt(2)
        array([[False,  True,  True],
               [False, False, False]], batch_axis=0)

        ```
        """
        return self.__gt__(other)

    def le(self, other: ndarray | bool | float) -> TBatchedArray:
        r"""Return ``self <= other`` element-wise.

        Args:
            other: Specifies the batch to compare.

        Returns:
            An array containing the element-wise comparison.

        Example usage:

        ```pycon

        >>> from redcat.ba import BatchedArray
        >>> x1 = BatchedArray(np.array([[1, 3, 4], [0, 2, 2]]))
        >>> x2 = BatchedArray(np.array([[5, 3, 2], [0, 1, 2]]))
        >>> x1.le(x2)
        array([[ True,  True, False],
               [ True, False,  True]], batch_axis=0)
        >>> x1.le(np.array([[5, 3, 2], [0, 1, 2]]))
        array([[ True,  True, False],
               [ True, False,  True]], batch_axis=0)
        >>> x1.le(2)
        array([[ True, False, False],
               [ True,  True,  True]], batch_axis=0)

        ```
        """
        return self.__le__(other)

    def lt(self, other: ndarray | bool | float) -> TBatchedArray:
        r"""Return ``self < other`` element-wise.

        Args:
            other: Specifies the batch to compare.

        Returns:
            An array containing the element-wise comparison.

        Example usage:

        ```pycon
        >>> from redcat.ba import BatchedArray
        >>> x1 = BatchedArray(np.array([[1, 3, 4], [0, 2, 2]]))
        >>> x2 = BatchedArray(np.array([[5, 3, 2], [0, 1, 2]]))
        >>> x1.lt(x2)
        array([[ True, False, False],
               [False, False, False]], batch_axis=0)
        >>> x1.lt(np.array([[5, 3, 2], [0, 1, 2]]))
        array([[ True, False, False],
               [False, False, False]], batch_axis=0)
        >>> x1.lt(2)
        array([[ True, False, False],
               [ True, False, False]], batch_axis=0)

        ```
        """
        return self.__lt__(other)

    def ne(self, other: ndarray | bool | float) -> TBatchedArray:
        r"""Return ``self != other`` element-wise.

        Args:
            other: Specifies the batch to compare.

        Returns:
            An array containing the element-wise comparison.

        Example usage:

        ```pycon
        >>> from redcat.ba import BatchedArray
        >>> x1 = BatchedArray(np.array([[1, 3, 4], [0, 2, 2]]))
        >>> x2 = BatchedArray(np.array([[5, 3, 2], [0, 1, 2]]))
        >>> x1.ne(x2)
        array([[ True, False,  True],
               [False,  True, False]], batch_axis=0)
        >>> x1.ne(np.array([[5, 3, 2], [0, 1, 2]]))
        array([[ True, False,  True],
               [False,  True, False]], batch_axis=0)
        >>> x1.ne(2)
        array([[ True,  True,  True],
               [ True, False, False]], batch_axis=0)

        ```
        """
        return self.__ne__(other)

    def isclose(
        self, other: ndarray, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
    ) -> TBatchedArray:
        r"""Return a boolean batch where two arrays are element-wise
        equal within a tolerance.

        Returns:
            A batch containing a boolean array that is ``True`` where
                the current batch is infinite and ``False`` elsewhere.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch1 = BatchedArray(np.array([[1.0, 0.0, 2.0], [0.0, -2.0, -1.0]]))
        >>> batch2 = BatchedArray(np.array([[1.001, 0.5, 2.0], [0.0, -2.5, -0.5]]))
        >>> batch1.isclose(batch2, atol=0.01)
        array([[ True, False,  True],
               [ True, False, False]], batch_axis=0)

        ```
        """
        return np.isclose(self, other, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def isinf(self) -> TBatchedArray:
        r"""Indicate if each element of the batch is infinite (positive
        or negative infinity) or not.

        Returns:
            A batch containing a boolean array that is ``True`` where
                the current batch is infinite and ``False`` elsewhere.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
        >>> batch.isinf()
        array([[False, False, True],
               [False, False, True]], batch_axis=0)

        ```
        """
        return np.isinf(self)

    def isneginf(self) -> TBatchedArray:
        r"""Indicate if each element of the batch is negative infinity or
        not.

        Returns:
            A batch containing a boolean array that is ``True`` where
                the current batch is negative infinity and ``False``
                elsewhere.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
        >>> batch.isneginf()
        array([[False, False, False],
               [False, False,  True]], batch_axis=0)

        ```
        """
        return np.isneginf(self)

    def isposinf(self) -> TBatchedArray:
        r"""Indicate if each element of the batch is positive infinity or
        not.

        Returns:
            A batch containing a boolean array that is ``True`` where
                the current batch is positive infinity and ``False``
                elsewhere.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
        >>> batch.isposinf()
        array([[False, False,   True],
               [False, False,  False]], batch_axis=0)

        ```
        """
        return np.isposinf(self)

    def isnan(self) -> TBatchedArray:
        r"""Indicate if each element in the batch is NaN or not.

        Returns:
            A batch containing a boolean array that is ``True`` where
                the current batch is infinite and ``False`` elsewhere.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.array([[1.0, 0.0, float("nan")], [float("nan"), -2.0, -1.0]]))
        >>> batch.isnan()
        array([[False, False,  True],
               [ True, False, False]], batch_axis=0)

        ```
        """
        return np.isnan(self)

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
