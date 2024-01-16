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
from numpy.typing import ArrayLike

from redcat.ba.utils import check_data_and_axis, check_same_batch_axis, get_batch_axes
from redcat.types import RNGType
from redcat.utils.array import to_array
from redcat.utils.random import randperm

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchedArray = TypeVar("TBatchedArray", np.ndarray, "BatchedArray")


class BatchedArray(np.ndarray):
    def __new__(cls, data: ArrayLike, batch_axis: int = 0) -> TBatchedArray:
        _data = np.array(data, copy=False, subok=True).view(cls)
        _baseclass = getattr(data, "_baseclass", type(_data))
        check_data_and_axis(_data, batch_axis)
        _data.batch_axis = batch_axis
        _data._baseclass = _baseclass
        return _data

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

    def eq(self, other: np.ndarray | bool | float) -> TBatchedArray:
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

    def ge(self, other: np.ndarray | bool | float) -> TBatchedArray:
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

    def gt(self, other: np.ndarray | bool | float) -> TBatchedArray:
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

    def le(self, other: np.ndarray | bool | float) -> TBatchedArray:
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

    def lt(self, other: np.ndarray | bool | float) -> TBatchedArray:
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

    def ne(self, other: np.ndarray | bool | float) -> TBatchedArray:
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
        self, other: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
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

    def add(self, other: np.ndarray | float, alpha: float = 1.0) -> TBatchedArray:
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

    def add_(self, other: np.ndarray | float, alpha: float = 1.0) -> None:
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

    def sub(self, other: np.ndarray | float, alpha: float = 1) -> TBatchedArray:
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
        other: np.ndarray | float,
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
        r"""Return the indices that would sort the batch along the batch
        dimension.

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

    def cumprod_along_batch(self, *args: Any, **kwargs: Any) -> TBatchedArray:
        r"""Return the cumulative product of elements along a batch axis.

        Args:
            args: See the documentation of ``numpy.cumprod``.
                ``axis`` should not be passed.
            kwargs: See the documentation of ``numpy.cumprod``.
                ``axis`` should not be passed.

        Returns:
            The cumulative product of elements along a batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.cumprod_along_batch()
        array([[  0,   1],
               [  0,   3],
               [  0,  15],
               [  0, 105],
               [  0, 945]], batch_axis=0)
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.cumprod_along_batch()
        array([[    0,     0,     0,     0,     0],
               [    5,    30,   210,  1680, 15120]], batch_axis=1)

        ```
        """
        return self.cumprod(*args, axis=self.batch_axis, **kwargs)

    def cumsum_along_batch(self, *args: Any, **kwargs: Any) -> TBatchedArray:
        r"""Return the cumulative sum of elements along a batch axis.

        Args:
            args: See the documentation of ``numpy.cumsum``.
                ``axis`` should not be passed.
            kwargs: See the documentation of ``numpy.cumsum``.
                ``axis`` should not be passed.

        Returns:
            The cumulative sum of elements along a batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.cumsum_along_batch()
        array([[ 0,  1],
               [ 2,  4],
               [ 6,  9],
               [12, 16],
               [20, 25]], batch_axis=0)
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.cumsum_along_batch()
        array([[ 0,  1,  3,  6, 10],
               [ 5, 11, 18, 26, 35]], batch_axis=1)

        ```
        """
        return self.cumsum(*args, axis=self.batch_axis, **kwargs)

    def permute_along_axis(
        self, permutation: np.ndarray | Sequence, axis: int = 0
    ) -> TBatchedArray:
        r"""Permute the values of an array along a given axis.

        Args:
            permutation: Specifies the permutation to use on the array.
                The dimension of this array should be compatible with the
                shape of the array to permute.
            axis: Specifies the axis used to permute the array.

        Returns:
            The permuted array.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(4))
        >>> batch.permute_along_axis(permutation=np.array([0, 2, 1, 3]))
        array([0, 2, 1, 3], batch_axis=0)
        >>> batch = BatchedArray(np.arange(20).reshape(4, 5))
        >>> batch.permute_along_axis(permutation=np.array([0, 2, 1, 3]))
        array([[ 0,  1,  2,  3,  4],
               [10, 11, 12, 13, 14],
               [ 5,  6,  7,  8,  9],
               [15, 16, 17, 18, 19]], batch_axis=0)
        >>> batch = BatchedArray(np.arange(20).reshape(4, 5))
        >>> batch.permute_along_axis(permutation=np.array([0, 4, 2, 1, 3]), axis=1)
        array([[ 0,  4,  2,  1,  3],
               [ 5,  9,  7,  6,  8],
               [10, 14, 12, 11, 13],
               [15, 19, 17, 16, 18]], batch_axis=0)
        >>> batch = BatchedArray(np.arange(20).reshape(2, 2, 5))
        >>> batch.permute_along_axis(permutation=np.array([0, 4, 2, 1, 3]), axis=2)
        array([[[ 0,  4,  2,  1,  3],
                [ 5,  9,  7,  6,  8]],
               [[10, 14, 12, 11, 13],
                [15, 19, 17, 16, 18]]], batch_axis=0)

        ```
        """
        permutation = np.asarray(permutation)
        return self.swapaxes(0, axis)[permutation].swapaxes(0, axis)

    def permute_along_batch(self, permutation: np.ndarray | Sequence) -> TBatchedArray:
        r"""Permute the values of an array along the batch axis.

        Args:
            permutation: Specifies the permutation to use on the array.
                The dimension of this array should be compatible with the
                shape of the array to permute.

        Returns:
            The permuted array along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(4))
        >>> batch.permute_along_batch(permutation=np.array([0, 2, 1, 3]))
        array([0, 2, 1, 3], batch_axis=0)
        >>> batch = BatchedArray(np.arange(20).reshape(4, 5))
        >>> batch.permute_along_batch(permutation=np.array([0, 2, 1, 3]))
        array([[ 0,  1,  2,  3,  4],
               [10, 11, 12, 13, 14],
               [ 5,  6,  7,  8,  9],
               [15, 16, 17, 18, 19]], batch_axis=0)
        >>> batch = BatchedArray(np.arange(20).reshape(4, 5), batch_axis=1)
        >>> batch.permute_along_batch(permutation=np.array([0, 4, 2, 1, 3]))
        array([[ 0,  4,  2,  1,  3],
               [ 5,  9,  7,  6,  8],
               [10, 14, 12, 11, 13],
               [15, 19, 17, 16, 18]], batch_axis=1)

        ```
        """
        return self.permute_along_axis(permutation, axis=self.batch_axis)

    def shuffle_along_axis(self, axis: int, generator: RNGType | None = None) -> TBatchedArray:
        r"""Shuffle the batch along a given axis.

        Args:
            axis: Specifies the shuffle axis.
            generator: Specifies the pseudorandom number generator for
                sampling or the random seed for the random number
                generator.

        Returns:
            A new batch with shuffled data along the given axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.shuffle_along_axis(axis=0)
        array([[...]], batch_axis=0)

        ```
        """
        return self.permute_along_axis(to_array(randperm(self.shape[axis], generator)), axis=axis)

    def shuffle_along_batch(self, generator: RNGType | None = None) -> TBatchedArray:
        r"""Shuffle the batch along the batch axis.

        Args:
            generator: Specifies the pseudorandom number generator for
                sampling or the random seed for the random number
                generator.

        Returns:
            A new batch with shuffled data along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.shuffle_along_batch()
        array([[...]], batch_axis=0)

        ```
        """
        return self.shuffle_along_axis(axis=self.batch_axis, generator=generator)

    def sort_along_batch(self, *args: Any, **kwargs: Any) -> None:
        r"""Sort an array in-place along the batch axis.

        Args:
            args: See the documentation of ``numpy.ndarray.sort``.
                ``axis`` should not be passed.
            kwargs: See the documentation of ``numpy.ndarray.sort``.
                ``axis`` should not be passed.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.array([[4, 5], [2, 3], [6, 7], [8, 9], [0, 1]]))
        >>> batch.sort_along_batch()
        >>> batch
        array([[0, 1],
               [2, 3],
               [4, 5],
               [6, 7],
               [8, 9]], batch_axis=0)
        >>> batch = BatchedArray(np.array([[4, 1, 3, 0, 2], [9, 6, 8, 5, 7]]), batch_axis=1)
        >>> batch.sort_along_batch()
        >>> batch
        array([[0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9]], batch_axis=1)

        ```
        """
        self.sort(*args, axis=self.batch_axis, **kwargs)

    #####################
    #     Reduction     #
    #####################

    def argmax(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Return indices of the maximum values along an axis.

        Args:
            args: See the documentation of ``numpy.argmax``.
            kwargs: See the documentation of ``numpy.argmax``.

        Returns:
            The indices of the maximum values along an axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.argmax(axis=0)
        array([4, 4])
        >>> batch.argmax(axis=0, keepdims=True)
        array([[4, 4]])
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.argmax(axis=1)
        array([4, 4])

        ```
        """
        return self.get_ndarray().argmax(*args, **kwargs)

    def argmax_along_batch(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Return indices of the maximum values along the batch axis.

        Args:
            args: See the documentation of ``numpy.argmax``.
                ``axis`` should not be passed.
            kwargs: See the documentation of ``numpy.argmax``.
                ``axis`` should not be passed.

        Returns:
            The indices of the maximum values along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.argmax_along_batch()
        array([4, 4])
        >>> batch.argmax_along_batch(keepdims=True)
        array([[4, 4]])
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.argmax_along_batch()
        array([4, 4])

        ```
        """
        return self.argmax(*args, axis=self.batch_axis, **kwargs)

    def argmin(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Return indices of the minimum values along an axis.

        Args:
            args: See the documentation of ``numpy.argmin``.
            kwargs: See the documentation of ``numpy.argmin``.

        Returns:
            The indices of the minimum values along an axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.argmin(axis=0)
        array([0, 0])
        >>> batch.argmin(axis=0, keepdims=True)
        array([[0, 0]])
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.argmin(axis=1)
        array([0, 0])

        ```
        """
        return self.get_ndarray().argmin(*args, **kwargs)

    def argmin_along_batch(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Return indices of the minimum values along the batch axis.

        Args:
            args: See the documentation of ``numpy.argmin``.
                ``axis`` should not be passed.
            kwargs: See the documentation of ``numpy.argmin``.
                ``axis`` should not be passed.

        Returns:
            The indices of the minimum values along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.argmin_along_batch()
        array([0, 0])
        >>> batch.argmin_along_batch(keepdims=True)
        array([[0, 0]])
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.argmin_along_batch()
        array([0, 0])

        ```
        """
        return self.argmin(*args, axis=self.batch_axis, **kwargs)

    def max(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Return the maximum along a given axis.

        Args:
            args: See the documentation of ``numpy.max``.
            kwargs: See the documentation of ``numpy.max``.

        Returns:
            The maximum values along an axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.max(axis=0)
        array([8, 9])
        >>> batch.max(axis=0, keepdims=True)
        array([[8, 9]])
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.max(axis=1)
        array([4, 9])

        ```
        """
        return self.get_ndarray().max(*args, **kwargs)

    def max_along_batch(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Return the maximum along the batch axis.

        Args:
            args: See the documentation of ``numpy.max``.
                ``axis`` should not be passed.
            kwargs: See the documentation of ``numpy.max``.
                ``axis`` should not be passed.

        Returns:
            The maximum values along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.max_along_batch()
        array([8, 9])
        >>> batch.max_along_batch(keepdims=True)
        array([[8, 9]])
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.max_along_batch()
        array([4, 9])

        ```
        """
        return self.max(*args, axis=self.batch_axis, **kwargs)

    def mean(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Return the mean along a given axis.

        Args:
            args: See the documentation of ``numpy.mean``.
            kwargs: See the documentation of ``numpy.mean``.

        Returns:
            The mean values along an axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.mean(axis=0)
        array([4., 5.])
        >>> batch.mean(axis=0, keepdims=True)
        array([[4., 5.]])
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.mean(axis=1)
        array([2., 7.])

        ```
        """
        return self.get_ndarray().mean(*args, **kwargs)

    def mean_along_batch(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Return the mean along the batch axis.

        Args:
            args: See the documentation of ``numpy.mean``.
                ``axis`` should not be passed.
            kwargs: See the documentation of ``numpy.mean``.
                ``axis`` should not be passed.

        Returns:
            The mean values along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.mean_along_batch()
        array([4., 5.])
        >>> batch.mean_along_batch(keepdims=True)
        array([[4., 5.]])
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.mean_along_batch()
        array([2., 7.])

        ```
        """
        return self.mean(*args, axis=self.batch_axis, **kwargs)

    def median(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Return the median along a given axis.

        Args:
            args: See the documentation of ``numpy.median``.
            kwargs: See the documentation of ``numpy.median``.

        Returns:
            The median values along an axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.median(axis=0)
        array([4., 5.])
        >>> batch.median(axis=0, keepdims=True)
        array([[4., 5.]])
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.median(axis=1)
        array([2., 7.])

        ```
        """
        return np.median(self.get_ndarray(), *args, **kwargs)

    def median_along_batch(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Return the median along the batch axis.

        Args:
            args: See the documentation of ``numpy.median``.
                ``axis`` should not be passed.
            kwargs: See the documentation of ``numpy.median``.
                ``axis`` should not be passed.

        Returns:
            The median values along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.median_along_batch()
        array([4., 5.])
        >>> batch.median_along_batch(keepdims=True)
        array([[4., 5.]])
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.median_along_batch()
        array([2., 7.])

        ```
        """
        return self.median(*args, axis=self.batch_axis, **kwargs)

    def min(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Return the minimum along a given axis.

        Args:
            args: See the documentation of ``numpy.min``.
            kwargs: See the documentation of ``numpy.min``.

        Returns:
            The minimum values along an axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.min(axis=0)
        array([0, 1])
        >>> batch.min(axis=0, keepdims=True)
        array([[0, 1]])
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.min(axis=1)
        array([0, 5])

        ```
        """
        return self.get_ndarray().min(*args, **kwargs)

    def min_along_batch(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Return the minimum along the batch axis.

        Args:
            args: See the documentation of ``numpy.min``.
                ``axis`` should not be passed.
            kwargs: See the documentation of ``numpy.min``.
                ``axis`` should not be passed.

        Returns:
            The minimum values along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.min_along_batch()
        array([0, 1])
        >>> batch.min_along_batch(keepdims=True)
        array([[0, 1]])
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.min_along_batch()
        array([0, 5])

        ```
        """
        return self.min(*args, axis=self.batch_axis, **kwargs)

    def nanmean(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Compute the arithmetic mean along the specified axis,
        ignoring NaNs.

        Args:
            args: See the documentation of ``numpy.nanmean``.
            kwargs: See the documentation of ``numpy.nanmean``.

        Returns:
            The nanmean values along an axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nanmean(axis=0)
        array([2. , 4. , 3.5])
        >>> batch.nanmean(axis=0, keepdims=True)
        array([[2. , 4. , 3.5]])
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
        >>> batch.nanmean(axis=1)
        array([1.5, 4. ])

        ```
        """
        return np.nanmean(self.get_ndarray(), *args, **kwargs)

    def nanmean_along_batch(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Compute the arithmetic mean along the batch axis, ignoring
        NaNs.

        Args:
            args: See the documentation of ``numpy.nanmean``.
                ``axis`` should not be passed.
            kwargs: See the documentation of ``numpy.nanmean``.
                ``axis`` should not be passed.

        Returns:
            The nanmean values along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nanmean_along_batch()
        array([2. , 4. , 3.5])
        >>> batch.nanmean_along_batch(keepdims=True)
        array([[2. , 4. , 3.5]])
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
        >>> batch.nanmean_along_batch()
        array([1.5, 4. ])

        ```
        """
        return self.nanmean(*args, axis=self.batch_axis, **kwargs)

    def nanmedian(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Compute the median along the specified axis, ignoring NaNs.

        Args:
            args: See the documentation of ``numpy.nanmedian``.
            kwargs: See the documentation of ``numpy.nanmedian``.

        Returns:
            The median along the specified axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nanmedian(axis=0)
        array([2. , 4. , 3.5])
        >>> batch.nanmedian(axis=0, keepdims=True)
        array([[2. , 4. , 3.5]])
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
        >>> batch.nanmedian(axis=1)
        array([1.5, 4. ])

        ```
        """
        return np.nanmedian(self.get_ndarray(), *args, **kwargs)

    def nanmedian_along_batch(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Compute the median along the batch axis, ignoring NaNs.

        Args:
            args: See the documentation of ``numpy.nanmedian``.
                ``axis`` should not be passed.
            kwargs: See the documentation of ``numpy.nanmedian``.
                ``axis`` should not be passed.

        Returns:
            The median along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nanmedian_along_batch()
        array([2. , 4. , 3.5])
        >>> batch.nanmedian_along_batch(keepdims=True)
        array([[2. , 4. , 3.5]])
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
        >>> batch.nanmedian_along_batch()
        array([1.5, 4. ])

        ```
        """
        return self.nanmedian(*args, axis=self.batch_axis, **kwargs)

    def nanprod(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Compute the product along the specified axis, ignoring NaNs.

        Args:
            args: See the documentation of ``numpy.nanprod``.
            kwargs: See the documentation of ``numpy.nanprod``.

        Returns:
            The product along the specified axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nanprod(axis=0)
        array([ 3., 4., 10.])
        >>> batch.nanprod(axis=0, keepdims=True)
        array([[ 3., 4., 10.]])
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
        >>> batch.nanprod(axis=1)
        array([ 2., 60.])

        ```
        """
        return np.nanprod(self.get_ndarray(), *args, **kwargs)

    def nanprod_along_batch(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Compute the product along the batch axis, ignoring NaNs.

        Args:
            args: See the documentation of ``numpy.nanprod``.
                ``axis`` should not be passed.
            kwargs: See the documentation of ``numpy.nanprod``.
                ``axis`` should not be passed.

        Returns:
            The product along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nanprod_along_batch()
        array([ 3., 4., 10.])
        >>> batch.nanprod_along_batch(keepdims=True)
        array([[ 3., 4., 10.]])
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
        >>> batch.nanprod_along_batch()
        array([ 2., 60.])

        ```
        """
        return self.nanprod(*args, axis=self.batch_axis, **kwargs)

    def nansum(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Compute the sum along the specified axis, ignoring NaNs.

        Args:
            args: See the documentation of ``numpy.nansum``.
            kwargs: See the documentation of ``numpy.nansum``.

        Returns:
            The sum along the specified axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nansum(axis=0)
        array([4., 4., 7.])
        >>> batch.nansum(axis=0, keepdims=True)
        array([[4., 4., 7.]])
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
        >>> batch.nansum(axis=1)
        array([ 3., 12.])

        ```
        """
        return np.nansum(self.get_ndarray(), *args, **kwargs)

    def nansum_along_batch(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Compute the sum along the batch axis, ignoring NaNs.

        Args:
            args: See the documentation of ``numpy.nansum``.
                ``axis`` should not be passed.
            kwargs: See the documentation of ``numpy.nansum``.
                ``axis`` should not be passed.

        Returns:
            The sum along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nansum_along_batch()
        array([4., 4., 7.])
        >>> batch.nansum_along_batch(keepdims=True)
        array([[4., 4., 7.]])
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
        >>> batch.nansum_along_batch()
        array([ 3., 12.])

        ```
        """
        return self.nansum(*args, axis=self.batch_axis, **kwargs)

    def prod(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Compute the product along the specified axis.

        Args:
            args: See the documentation of ``numpy.prod``.
            kwargs: See the documentation of ``numpy.prod``.

        Returns:
            The product along the specified axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]))
        >>> batch.prod(axis=0)
        array([ 3, 12, 10])
        >>> batch.prod(axis=0, keepdims=True)
        array([[ 3, 12, 10]])
        >>> batch = BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]), batch_axis=1)
        >>> batch.prod(axis=1)
        array([ 6, 60])

        ```
        """
        return np.prod(self.get_ndarray(), *args, **kwargs)

    def prod_along_batch(self, *args: Any, **kwargs: Any) -> np.ndarray:
        r"""Compute the product along the batch axis.

        Args:
            args: See the documentation of ``numpy.prod``.
                ``axis`` should not be passed.
            kwargs: See the documentation of ``numpy.prod``.
                ``axis`` should not be passed.

        Returns:
            The product along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba import BatchedArray
        >>> batch = BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]))
        >>> batch.prod_along_batch()
        array([ 3, 12, 10])
        >>> batch.prod_along_batch(keepdims=True)
        array([[ 3, 12, 10]])
        >>> batch = BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]), batch_axis=1)
        >>> batch.prod_along_batch()
        array([ 6, 60])

        ```
        """
        return self.prod(*args, axis=self.batch_axis, **kwargs)

    def _check_valid_axes(self, arrays: Sequence) -> None:
        r"""Check if the axes are valid/compatible.

        Args:
            arrays: Specifies the arrays to check.
        """
        check_same_batch_axis(get_batch_axes(arrays))

    def get_ndarray(self) -> np.ndarray:
        return self.view(np.ndarray)
