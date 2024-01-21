from __future__ import annotations

__all__ = ["BatchedArray"]

from collections.abc import Sequence
from typing import Any, Literal, TypeVar

import numpy as np
from coola import objects_are_allclose, objects_are_equal
from numpy.typing import ArrayLike, DTypeLike

from redcat.ba import check_data_and_axis, check_same_batch_axis, get_batch_axes

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")


class BatchedArray(np.lib.mixins.NDArrayOperatorsMixin):  # (BaseBatch[np.ndarray]):
    r"""Implement a wrapper around a NumPy array to track the batch
    axis."""

    def __init__(self, data: ArrayLike, batch_axis: int = 0, check: bool = True) -> None:
        self._data = np.array(data, copy=False, subok=True)
        self._batch_axis = batch_axis
        if check:
            check_data_and_axis(self._data, self._batch_axis)

    ################################
    #     Core functionalities     #
    ################################

    @property
    def batch_size(self) -> int:
        return self._data.shape[self._batch_axis]

    @property
    def data(self) -> np.ndarray:
        r"""The underlying numpy array."""
        return self._data

    def allclose(
        self, other: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
    ) -> bool:
        if not isinstance(other, self.__class__) or self.batch_axis != other.batch_axis:
            return False
        return objects_are_allclose(
            self.data, other.data, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    def allequal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__) or self.batch_axis != other.batch_axis:
            return False
        return objects_are_equal(self.data, other.data, equal_nan=equal_nan)

    ######################################
    #     Additional functionalities     #
    ######################################

    def __array__(self, dtype: DTypeLike = None, /) -> np.ndarray:
        return self._data.__array__(dtype)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "inner"],
        *inputs: Any,
        **kwargs: Any,
    ) -> TBatchedArray | tuple[TBatchedArray, ...]:
        args = []
        batch_axes = set()
        for inp in inputs:
            if isinstance(inp, self.__class__):
                batch_axes.add(inp.batch_axis)
                inp = inp.data
            args.append(inp)
        check_same_batch_axis(batch_axes)

        results = self._data.__array_ufunc__(ufunc, method, *args, **kwargs)
        if ufunc.nout == 1:
            return self._create_new_batch(results)
        return tuple(self._create_new_batch(res) for res in results)

    def __repr__(self) -> str:
        return repr(self._data)[:-1] + f", batch_axis={self._batch_axis})"

    def __str__(self) -> str:
        return str(self._data) + f"\nwith batch_axis={self._batch_axis}"

    @property
    def batch_axis(self) -> int:
        r"""The batch axis in the array."""
        return self._batch_axis

    #########################
    #     Memory layout     #
    #########################

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Tuple of array dimensions."""
        return self._data.shape

    #####################
    #     Data type     #
    #####################

    @property
    def dtype(self) -> np.dtype:
        r"""Data-type of the array’s elements."""
        return self._data.dtype

    ###################################
    #     Arithmetical operations     #
    ###################################

    def __iadd__(self, other: Any) -> TBatchedArray:
        self._check_valid_axes((self, other))
        self._data.__iadd__(self._get_data(other))
        return self

    def __ifloordiv__(self, other: Any) -> TBatchedArray:
        self._check_valid_axes((self, other))
        self._data.__ifloordiv__(self._get_data(other))
        return self

    def __imod__(self, other: Any) -> TBatchedArray:
        self._check_valid_axes((self, other))
        self._data.__imod__(self._get_data(other))
        return self

    def __imul__(self, other: Any) -> TBatchedArray:
        self._check_valid_axes((self, other))
        self._data.__imul__(self._get_data(other))
        return self

    def __isub__(self, other: Any) -> TBatchedArray:
        self._check_valid_axes((self, other))
        self._data.__isub__(self._get_data(other))
        return self

    def __itruediv__(self, other: Any) -> TBatchedArray:
        self._check_valid_axes((self, other))
        self._data.__itruediv__(self._get_data(other))
        return self

    def add(
        self,
        other: BatchedArray | np.ndarray | float,
        alpha: float = 1.0,
    ) -> TBatchedArray:
        r"""Adds the input ``other``, scaled by ``alpha``, to the
        ``self`` batch.

        Similar to ``out = self + alpha * other``

        Args:
            other: Specifies the other value to add to the current
                batch.
            alpha: Specifies the scale of the batch to add.

        Returns:
            A new batch containing the addition of the two batches.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.ones((2, 3))
        >>> out = batch.add(ba2.full((2, 3), 2.0))
        >>> batch
        array([[1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)
        >>> out
        array([[3., 3., 3.],
               [3., 3., 3.]], batch_axis=0)

        ```
        """
        return self.__add__(other * alpha)

    def add_(
        self,
        other: BatchedArray | np.ndarray | float,
        alpha: float = 1.0,
    ) -> None:
        r"""Adds the input ``other``, scaled by ``alpha``, to the
        ``self`` batch.

        Similar to ``self += alpha * other`` (in-place)

        Args:
            other: Specifies the other value to add to the current
                batch.
            alpha: Specifies the scale of the batch to add.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.ones((2, 3))
        >>> batch.add_(ba2.full((2, 3), 2.0))
        >>> batch
        array([[3., 3., 3.],
               [3., 3., 3.]], batch_axis=0)

        ```
        """
        self.__iadd__(other * alpha)

    def floordiv(self, divisor: BatchedArray | np.ndarray | float) -> TBatchedArray:
        r"""Return the largest integer smaller or equal to the division
        of the inputs.

        The current batch is the dividend/numerator.

        Args:
            divisor: Specifies the divisor/denominator.

        Returns:
            The largest integer smaller or equal to the division of
                the inputs.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.ones((2, 3))
        >>> out = batch.floordiv(ba2.full((2, 3), 2.0))
        >>> batch
        array([[1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)
        >>> out
        array([[0., 0., 0.],
               [0., 0., 0.]], batch_axis=0)

        ```
        """
        return self.__floordiv__(divisor)

    def floordiv_(self, divisor: BatchedArray | np.ndarray | float) -> None:
        r"""Return the largest integer smaller or equal to the division
        of the inputs.

        The current batch is the dividend/numerator.

        Args:
            divisor: Specifies the divisor/denominator.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.ones((2, 3))
        >>> batch.floordiv_(ba2.full((2, 3), 2.0))
        >>> batch
        array([[0., 0., 0.],
               [0., 0., 0.]], batch_axis=0)

        ```
        """
        self.__ifloordiv__(divisor)

    def fmod(self, divisor: BatchedArray | np.ndarray | float) -> TBatchedArray:
        r"""Computes the element-wise remainder of division.

        The current batch is the dividend.

        Args:
            divisor: Specifies the divisor.

        Returns:
            A new batch containing the element-wise remainder of
                division.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.ones((2, 3))
        >>> out = batch.fmod(ba2.full((2, 3), 2.0))
        >>> batch
        array([[1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)
        >>> out
        array([[1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)

        ```
        """
        return self.__mod__(divisor)

    def fmod_(self, divisor: BatchedArray | np.ndarray | float) -> None:
        r"""Computes the element-wise remainder of division.

        The current batch is the dividend.

        Args:
            divisor: Specifies the divisor.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.ones((2, 3))
        >>> batch.fmod_(ba2.full((2, 3), 2.0))
        >>> batch
        array([[1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)

        ```
        """
        self.__imod__(divisor)

    def mul(self, other: BatchedArray | np.ndarray | float) -> TBatchedArray:
        r"""Multiplies the ``self`` batch by the input ``other`.

        Similar to ``out = self * other``

        Args:
            other: Specifies the value to multiply.

        Returns:
            A new batch containing the multiplication of the two
                batches.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.ones((2, 3))
        >>> out = batch.mul(ba2.full((2, 3), 2.0))
        >>> batch
        array([[1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)
        >>> out
        array([[2., 2., 2.],
               [2., 2., 2.]], batch_axis=0)

        ```
        """
        return self.__mul__(other)

    def mul_(self, other: BatchedArray | np.ndarray | float) -> None:
        r"""Multiplies the ``self`` batch by the input ``other`.

        Similar to ``self *= other`` (in-place)

        Args:
            other: Specifies the value to multiply.

        Returns:
            A new batch containing the multiplication of the two
                batches.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.ones((2, 3))
        >>> batch.mul_(ba2.full((2, 3), 2.0))
        >>> batch
        array([[2., 2., 2.],
               [2., 2., 2.]], batch_axis=0)

        ```
        """
        self.__imul__(other)

    def neg(self) -> TBatchedArray:
        r"""Returns a new batch with the negative of the elements.

        Returns:
            A new batch with the negative of the elements.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.ones((2, 3))
        >>> out = batch.neg()
        >>> batch
        array([[1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)
        >>> out
        array([[-1., -1., -1.],
               [-1., -1., -1.]], batch_axis=0)

        ```
        """
        return self.__neg__()

    def sub(
        self,
        other: BatchedArray | np.ndarray | float,
        alpha: float = 1,
    ) -> TBatchedArray:
        r"""Subtracts the input ``other``, scaled by ``alpha``, to the
        ``self`` batch.

        Similar to ``out = self - alpha * other``

        Args:
            other: Specifies the value to subtract.
            alpha: Specifies the scale of the batch to substract.

        Returns:
            A new batch containing the diffence of the two batches.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.ones((2, 3))
        >>> out = batch.sub(ba2.full((2, 3), 2.0))
        >>> batch
        array([[1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)
        >>> out
        array([[-1., -1., -1.],
               [-1., -1., -1.]], batch_axis=0)

        ```
        """
        return self.__sub__(other * alpha)

    def sub_(
        self,
        other: BatchedArray | np.ndarray | float,
        alpha: float = 1.0,
    ) -> None:
        r"""Subtracts the input ``other``, scaled by ``alpha``, to the
        ``self`` batch.

        Similar to ``self -= alpha * other`` (in-place)

        Args:
            other: Specifies the value to subtract.
            alpha: Specifies the scale of the batch to substract.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.ones((2, 3))
        >>> batch.sub_(ba2.full((2, 3), 2.0))
        >>> batch
        array([[-1., -1., -1.],
               [-1., -1., -1.]], batch_axis=0)

        ```
        """
        self.__isub__(other * alpha)

    def truediv(self, divisor: BatchedArray | np.ndarray | float) -> TBatchedArray:
        r"""Return the division of the inputs.

        The current batch is the dividend/numerator.

        Args:
            divisor: Specifies the divisor/denominator.

        Returns:
            The division of the inputs.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.ones((2, 3))
        >>> out = batch.truediv(ba2.full((2, 3), 2.0))
        >>> batch
        array([[1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)
        >>> out
        array([[0.5, 0.5, 0.5],
               [0.5, 0.5, 0.5]], batch_axis=0)

        ```
        """
        return self.__truediv__(divisor)

    def truediv_(self, divisor: BatchedArray | np.ndarray | float) -> None:
        r"""Return the division of the inputs.

        The current batch is the dividend/numerator.

        Args:
            divisor: Specifies the divisor/denominator.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.ones((2, 3))
        >>> batch.truediv_(ba2.full((2, 3), 2.0))
        >>> batch
        array([[0.5, 0.5, 0.5],
               [0.5, 0.5, 0.5]], batch_axis=0)

        ```
        """
        self.__itruediv__(divisor)

    #################
    #     Other     #
    #################

    def _check_valid_axes(self, arrays: Sequence) -> None:
        r"""Checks if the dimensions are valid.

        Args:
            arrays: Specifies the sequence of arrays/batches to check.
        """
        check_same_batch_axis(get_batch_axes(arrays))

    def _create_new_batch(self, data: np.ndarray) -> TBatchedArray:
        return self.__class__(data, **self._get_kwargs())

    def _get_kwargs(self) -> dict:
        return {"batch_axis": self._batch_axis}

    def _get_data(self, data: Any) -> Any:
        if isinstance(data, self.__class__):
            data = data.data
        return data
