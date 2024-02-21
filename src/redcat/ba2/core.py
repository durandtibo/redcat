from __future__ import annotations

__all__ = ["BatchedArray", "implements"]

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, SupportsIndex, TypeVar, Union, overload

import numpy as np
from coola import objects_are_allclose, objects_are_equal
from numpy.typing import ArrayLike, DTypeLike

from redcat.ba import check_data_and_axis, check_same_batch_axis, get_batch_axes
from redcat.base import BaseBatch
from redcat.utils.array import to_array

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")

IndexType = Union[int, slice, list[int], np.ndarray, None]
OrderACFK = Literal["A", "C", "F", "K"]
ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]
SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]

HANDLED_FUNCTIONS = {}


class BatchedArray(BaseBatch[np.ndarray], np.lib.mixins.NDArrayOperatorsMixin):
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

    def append(self, other: TBatchedArray | np.ndarray) -> None:
        self.concatenate_along_batch_([other])

    def chunk_along_batch(self, chunks: int) -> tuple[TBatchedArray, ...]:
        return self.chunk(chunks, self._batch_axis)

    def clone(self) -> TBatchedArray:
        return self._create_new_batch(self._data.copy())

    def extend(self, other: Iterable[TBatchedArray | np.ndarray]) -> None:
        self.concatenate_along_batch_(other)

    def index_select_along_batch(self, index: np.ndarray | Sequence[int]) -> TBatchedArray:
        return self.index_select(index=index, axis=self._batch_axis)

    def permute_along_batch(self, permutation: np.ndarray | Sequence[int]) -> TBatchedArray:
        return self.permute_along_axis(permutation, axis=self._batch_axis)

    def permute_along_batch_(self, permutation: np.ndarray | Sequence[int]) -> None:
        self.permute_along_axis_(permutation, axis=self._batch_axis)

    def select_along_batch(self, index: int) -> np.ndarray:
        return self.select(index=index, axis=self._batch_axis)

    def shuffle_along_batch(self, rng: np.random.Generator | None = None) -> TBatchedArray:
        return self.shuffle_along_axis(axis=self._batch_axis, rng=rng)

    def shuffle_along_batch_(self, rng: np.random.Generator | None = None) -> None:
        self.shuffle_along_axis_(axis=self._batch_axis, rng=rng)

    def slice_along_batch(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> TBatchedArray:
        return self.slice_along_axis(self._batch_axis, start, stop, step)

    def split_along_batch(
        self, split_size_or_sections: int | Sequence[int]
    ) -> tuple[TBatchedArray, ...]:
        return self.split_along_axis(split_size_or_sections, axis=self._batch_axis)

    def summary(self) -> str:
        dims = ", ".join([f"{key}={value}" for key, value in self._get_kwargs().items()])
        return f"{self.__class__.__qualname__}(dtype={self.dtype}, shape={self.shape}, {dims})"

    def to_data(self) -> np.ndarray:
        return self._data

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

    def __array_function__(
        self,
        func: Callable,
        types: tuple,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> TBatchedArray | tuple[TBatchedArray, ...]:
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

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
    def ndim(self) -> int:
        r"""Number of array dimensions."""
        return self._data.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Tuple of array dimensions."""
        return self._data.shape

    @property
    def size(self) -> int:
        r"""Number of elements in the array."""
        return self._data.size

    #####################
    #     Data type     #
    #####################

    @property
    def dtype(self) -> np.dtype:
        r"""Data-type of the array`s elements."""
        return self._data.dtype

    ###############################
    #     Creation operations     #
    ###############################

    def copy(self, order: OrderACFK = "C") -> TBatchedArray:
        r"""Return a copy of the array.

        Args:
            order: Controls the memory layout of the copy.
                `C` means C-order, `F` means F-order, `A` means `F`
                if the current array is Fortran contiguous, `C`
                otherwise. `K` means  match the layout of current
                array as closely as possible.

        Returns:
            A copy of the array.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat import ba2
        >>> array = ba2.ones((2, 3))
        >>> x = array.copy()
        >>> x += 1
        >>> array
        array([[1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)
        >>> x
        array([[2., 2., 2.],
               [2., 2., 2.]], batch_axis=0)

        ```
        """
        return self._create_new_batch(self._data.copy(order=order))

    def empty_like(
        self,
        dtype: DTypeLike = None,
        order: OrderACFK = "K",
        subok: bool = True,
        shape: ShapeLike = None,
        batch_size: int | None = None,
    ) -> TBatchedArray:
        r"""Return an array without initializing entries, with the same
        shape as the current array.

        Args:
            dtype: Overrides the data type of the result.
            order: Overrides the memory layout of the result. `C` means
                C-order, `F` means F-order, `A` means `F` if ``self``
                is Fortran contiguous, `C` otherwise. `K` means match
                the layout of ``self`` as closely as possible.
            subok: If True, then the newly created array will use the
                sub-class type of ``self``, otherwise it will be a
                base-class array.
            shape: Overrides the shape of the result. If order=`K` and
                thenumber of dimensions is unchanged, will try to keep
                order, otherwise, order=`C` is implied.
            batch_size: Overrides the batch size. If ``None``,
                the batch size of the current batch is used.

        Returns:
            Array of zeros with the same shape and type as ``self``.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat import ba2
        >>> array = ba2.ones((2, 3))
        >>> array.empty_like().shape
        (2, 3)
        >>> array.empty_like(batch_size=5).shape
        (5, 3)

        ```
        """
        shape = list(shape or self.shape)
        if batch_size is not None:
            shape[self.batch_axis] = batch_size
        return self._create_new_batch(
            np.empty_like(self._data, dtype=dtype, order=order, subok=subok, shape=shape)
        )

    def full_like(
        self,
        fill_value: float | ArrayLike,
        dtype: DTypeLike = None,
        order: OrderACFK = "K",
        subok: bool = True,
        shape: ShapeLike = None,
        batch_size: int | None = None,
    ) -> TBatchedArray:
        r"""Return an array filled with the scalar value ``1``, with the
        same shape as the current array.

        Args:
            fill_value: Specifies the fill value.
            dtype: Overrides the data type of the result.
            order: Overrides the memory layout of the result. `C` means
                C-order, `F` means F-order, `A` means `F` if ``self``
                is Fortran contiguous, `C` otherwise. `K` means match
                the layout of ``self`` as closely as possible.
            subok: If True, then the newly created array will use the
                sub-class type of ``self``, otherwise it will be a
                base-class array.
            shape: Overrides the shape of the result. If order=`K` and
                thenumber of dimensions is unchanged, will try to keep
                order, otherwise, order=`C` is implied.
            batch_size: Overrides the batch size. If ``None``,
                the batch size of the current batch is used.

        Returns:
            An array filled with the scalar value ``1``, with the same
                shape as the current array.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> array = ba2.ones((2, 3))
        >>> array.full_like(42.0)
        array([[42., 42., 42.],
               [42., 42., 42.]], batch_axis=0)
        >>> array.full_like(fill_value=42.0, batch_size=5)
        array([[42., 42., 42.],
               [42., 42., 42.],
               [42., 42., 42.],
               [42., 42., 42.],
               [42., 42., 42.]], batch_axis=0)

        ```
        """
        shape = list(shape or self.shape)
        if batch_size is not None:
            shape[self.batch_axis] = batch_size
        return self._create_new_batch(
            np.full_like(
                self._data,
                fill_value=fill_value,
                dtype=dtype,
                order=order,
                subok=subok,
                shape=shape,
            )
        )

    def ones_like(
        self,
        dtype: DTypeLike = None,
        order: OrderACFK = "K",
        subok: bool = True,
        shape: ShapeLike = None,
        batch_size: int | None = None,
    ) -> TBatchedArray:
        r"""Return an array filled with the scalar value ``1``, with the
        same shape as the current array.

        Args:
            dtype: Overrides the data type of the result.
            order: Overrides the memory layout of the result. `C` means
                C-order, `F` means F-order, `A` means `F` if ``self``
                is Fortran contiguous, `C` otherwise. `K` means match
                the layout of ``self`` as closely as possible.
            subok: If True, then the newly created array will use the
                sub-class type of ``self``, otherwise it will be a
                base-class array.
            shape: Overrides the shape of the result. If order=`K` and
                thenumber of dimensions is unchanged, will try to keep
                order, otherwise, order=`C` is implied.
            batch_size: Overrides the batch size. If ``None``,
                the batch size of the current batch is used.

        Returns:
            An array filled with the scalar value ``1``, with the same
                shape as the current array.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> array = ba2.zeros((2, 3))
        >>> array.ones_like()
        array([[1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)
        >>> array.ones_like(batch_size=5)
        array([[1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.],
               [1., 1., 1.]], batch_axis=0)

        ```
        """
        shape = list(shape or self.shape)
        if batch_size is not None:
            shape[self.batch_axis] = batch_size
        return self._create_new_batch(
            np.ones_like(self._data, dtype=dtype, order=order, subok=subok, shape=shape)
        )

    def zeros_like(
        self,
        dtype: DTypeLike = None,
        order: OrderACFK = "K",
        subok: bool = True,
        shape: ShapeLike = None,
        batch_size: int | None = None,
    ) -> TBatchedArray:
        r"""Return an array filled with the scalar value ``0``, with the
        same shape as the current array.

        Args:
            dtype: Overrides the data type of the result.
            order: Overrides the memory layout of the result. `C` means
                C-order, `F` means F-order, `A` means `F` if ``self``
                is Fortran contiguous, `C` otherwise. `K` means match
                the layout of ``self`` as closely as possible.
            subok: If True, then the newly created array will use the
                sub-class type of ``self``, otherwise it will be a
                base-class array.
            shape: Overrides the shape of the result. If order=`K` and
                thenumber of dimensions is unchanged, will try to keep
                order, otherwise, order=`C` is implied.
            batch_size: Overrides the batch size. If ``None``,
                the batch size of the current batch is used.

        Returns:
            An array filled with the scalar value ``0``, with the same
                shape as the current array.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> array = ba2.ones((2, 3))
        >>> array.zeros_like()
        array([[0., 0., 0.],
               [0., 0., 0.]], batch_axis=0)
        >>> array.zeros_like(batch_size=5)
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]], batch_axis=0)

        ```
        """
        shape = list(shape or self.shape)
        if batch_size is not None:
            shape[self.batch_axis] = batch_size
        return self._create_new_batch(
            np.zeros_like(self._data, dtype=dtype, order=order, subok=subok, shape=shape)
        )

    ################################
    #     Comparison operators     #
    ################################

    ##################################
    #     Arithmetical operators     #
    ##################################

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

    def add(self, other: BatchedArray | np.ndarray | float, alpha: float = 1.0) -> TBatchedArray:
        r"""Adds the input ``other``, scaled by ``alpha``, to the
        ``self`` batch.

        Similar to ``out = self + alpha * other``

        Args:
            other: Specifies the other value to add to the current
                batch.
            alpha: Specifies the scale of the batch to add.

        Returns:
            A new batch containing the addition of the two batches.

        Raises:
            RuntimeError: if the batch axes are different.

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

        Raises:
            RuntimeError: if the batch axes are different.

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

        Raises:
            RuntimeError: if the batch axes are different.

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

        Raises:
            RuntimeError: if the batch axes are different.

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

        Raises:
            RuntimeError: if the batch axes are different.

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

        Raises:
            RuntimeError: if the batch axes are different.

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

        Raises:
            RuntimeError: if the batch axes are different.

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

        Raises:
            RuntimeError: if the batch axes are different.

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

        Raises:
            RuntimeError: if the batch axes are different.

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

        Raises:
            RuntimeError: if the batch axes are different.

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

        Raises:
            RuntimeError: if the batch axes are different.

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

        Raises:
            RuntimeError: if the batch axes are different.

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

    ########################################################
    #     Array manipulation routines | Joining arrays     #
    ########################################################

    @overload
    def concatenate(
        self, arrays: Iterable[TBatchedArray | np.ndarray], axis: None = ...
    ) -> np.ndarray: ...  # pragma: no cover

    @overload
    def concatenate(
        self, arrays: Iterable[TBatchedArray | np.ndarray], axis: int = ...
    ) -> TBatchedArray: ...  # pragma: no cover

    def concatenate(
        self, arrays: Iterable[TBatchedArray | np.ndarray], axis: int | None = 0
    ) -> TBatchedArray | np.ndarray:
        r"""Join a sequence of arrays along an existing axis.

        Args:
            arrays: The arrays must have the same shape, except in the
                dimension corresponding to axis.
            axis: The axis along which the arrays will be joined.
                If axis is None, arrays are flattened before use.

        Returns:
            The concatenated array.

        Raises:
            RuntimeError: if the batch axes are different.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.batched_array([[0, 1, 2], [4, 5, 6]])
        >>> out = batch.concatenate([ba2.batched_array([[10, 11, 12], [13, 14, 15]])])
        >>> batch
        array([[0, 1, 2],
               [4, 5, 6]], batch_axis=0)
        >>> out
        array([[ 0,  1,  2],
               [ 4,  5,  6],
               [10, 11, 12],
               [13, 14, 15]], batch_axis=0)

        ```
        """
        arr = [self._data]
        batch_axes = {self.batch_axis}
        for a in arrays:
            if isinstance(a, self.__class__):
                batch_axes.add(a.batch_axis)
                a = a.data
            arr.append(a)
        check_same_batch_axis(batch_axes)
        out = np.concatenate(arr, axis=axis)
        if axis is None:
            return out
        return self._create_new_batch(out)

    def concatenate_(self, arrays: Iterable[TBatchedArray | np.ndarray], axis: int = 0) -> None:
        r"""Join a sequence of arrays along an existing axis in-place.

        Args:
            arrays: The arrays must have the same shape, except in the
                dimension corresponding to axis.
            axis: The axis along which the arrays will be joined.

        Raises:
            RuntimeError: if the batch axes are different.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.batched_array([[0, 1, 2], [4, 5, 6]])
        >>> batch.concatenate_([ba2.batched_array([[10, 11, 12], [13, 14, 15]])])
        >>> batch
        array([[ 0,  1,  2],
               [ 4,  5,  6],
               [10, 11, 12],
               [13, 14, 15]], batch_axis=0)

        ```
        """
        self._data = self.concatenate(arrays, axis).data

    def concatenate_along_batch(
        self, arrays: Iterable[TBatchedArray | np.ndarray]
    ) -> TBatchedArray:
        r"""Join a sequence of arrays along the batch axis.

        Args:
            arrays: The arrays must have the same shape, except in the
                dimension corresponding to axis.

        Returns:
            The concatenated array.

        Raises:
            RuntimeError: if the batch axes are different.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.batched_array([[0, 1, 2], [4, 5, 6]])
        >>> out = batch.concatenate_along_batch([ba2.batched_array([[10, 11, 12], [13, 14, 15]])])
        >>> batch
        array([[0, 1, 2],
               [4, 5, 6]], batch_axis=0)
        >>> out
        array([[ 0,  1,  2],
               [ 4,  5,  6],
               [10, 11, 12],
               [13, 14, 15]], batch_axis=0)

        ```
        """
        return self.concatenate(arrays, axis=self._batch_axis)

    def concatenate_along_batch_(self, arrays: Iterable[TBatchedArray | np.ndarray]) -> None:
        r"""Join a sequence of arrays along the batch axis in-place.

        Args:
            arrays: The arrays must have the same shape, except in the
                dimension corresponding to axis.

        Raises:
            RuntimeError: if the batch axes are different.

        Example usage:

        ```pycon
        >>> from redcat import ba2
        >>> batch = ba2.batched_array([[0, 1, 2], [4, 5, 6]])
        >>> batch.concatenate_along_batch_([ba2.batched_array([[10, 11, 12], [13, 14, 15]])])
        >>> batch
        array([[ 0,  1,  2],
               [ 4,  5,  6],
               [10, 11, 12],
               [13, 14, 15]], batch_axis=0)

        ```
        """
        self.concatenate_(arrays, axis=self._batch_axis)

    ##########################################################
    #     Array manipulation routines | Splitting arrays     #
    ##########################################################

    def chunk(self, chunks: int, axis: int = 0) -> tuple[TBatchedArray, ...]:
        r"""Split an array into the specified number of chunks. Each
        chunk is a view of the input array.

        Args:
            chunks: Specifies the number of chunks.
            axis: Specifies the axis along which to split the array.

        Returns:
            The array split into chunks along the given axis.

        Raises:
            RuntimeError: if the number of chunks is incorrect

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.chunk(chunks=3)
        (array([[0, 1], [2, 3]], batch_axis=0),
         array([[4, 5], [6, 7]], batch_axis=0),
         array([[8, 9]], batch_axis=0))

        ```
        """
        if chunks == 0:
            raise RuntimeError("chunk expects `chunks` to be greater than 0, got: 0")
        return tuple(
            self._create_new_batch(chunk)
            for chunk in np.array_split(self._data, indices_or_sections=chunks, axis=axis)
        )

    @overload
    def index_select(
        self, index: np.ndarray | Sequence[int], axis: None = ...
    ) -> np.ndarray: ...  # pragma: no cover

    @overload
    def index_select(
        self, index: np.ndarray | Sequence[int], axis: int = ...
    ) -> TBatchedArray: ...  # pragma: no cover

    def index_select(
        self, index: np.ndarray | Sequence[int], axis: int | None = None
    ) -> TBatchedArray | np.ndarray:
        r"""Returns a new array which indexes the input array along the
        given axis using the entries in ``index``.

        Args:
            index: The 1-D array containing the indices to index.
            axis: The axis over which to select values. By default,
                the flattened input array is used.

        Returns:
            A new array which indexes the input array along the
                given axis using the entries in ``index``.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.index_select([2, 4], axis=0)
        array([[4, 5],
               [8, 9]], batch_axis=0)
        >>> batch.index_select(np.array([4, 3, 2, 1, 0]), axis=0)
        array([[8, 9],
               [6, 7],
               [4, 5],
               [2, 3],
               [0, 1]], batch_axis=0)

        ```
        """
        data = np.take(self._data, indices=to_array(index), axis=axis)
        if axis is None:
            return data
        return self._create_new_batch(data)

    def select(self, index: int, axis: int) -> np.ndarray:
        r"""Select the data along the given axis at the given index.

        Args:
            index: Specifies the index to select.
            axis: Specifies the index axis.

        Returns:
            The batch sliced along the given axis at the given index.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.select(index=2, axis=0)
        array([4, 5])

        ```
        """
        return np.take(self._data, indices=index, axis=axis)

    def slice_along_axis(
        self,
        axis: int = 0,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> TBatchedArray:
        r"""Slice the batch in a given axis.

        Args:
            axis: Specifies the axis along which to slice the array.
            start: Specifies the index where the slicing starts.
            stop: Specifies the index where the slicing stops.
                ``None`` means last.
            step: Specifies the increment between each index for
                slicing.

        Returns:
            A slice of the current batch along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.slice_along_axis(start=2)
        array([[4, 5],
               [6, 7],
               [8, 9]], batch_axis=0)
        >>> batch.slice_along_axis(stop=3)
        array([[0, 1],
               [2, 3],
               [4, 5]], batch_axis=0)
        >>> batch.slice_along_axis(step=2)
        array([[0, 1],
               [4, 5],
               [8, 9]], batch_axis=0)

        ```
        """
        if axis == 0:
            data = self._data[start:stop:step]
        elif axis == 1:
            data = self._data[:, start:stop:step]
        else:
            data = self._data.swapaxes(0, axis)[start:stop:step].swapaxes(0, axis)
        return self._create_new_batch(data)

    def split_along_axis(
        self, split_size_or_sections: int | Sequence[int], axis: int = 0
    ) -> tuple[TBatchedArray, ...]:
        r"""Splits the batch into chunks along a given axis.

        Notes:
            This function has a slightly different behavior as
                ``numpy.split``.

        Args:
            split_size_or_sections: Specifies the size of a single
                chunk or list of sizes for each chunk.
            axis: Specifies the axis along which to split the array.

        Returns:
            The batch split into chunks along the given axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.split_along_axis(2, axis=0)
        (array([[0, 1], [2, 3]], batch_axis=0),
         array([[4, 5], [6, 7]], batch_axis=0),
         array([[8, 9]], batch_axis=0))

        ```
        """
        if isinstance(split_size_or_sections, int):
            split_size_or_sections = tuple(
                range(split_size_or_sections, self.batch_size, split_size_or_sections)
            )
        else:
            split_size_or_sections = np.cumsum(split_size_or_sections)[:-1]
        return tuple(
            self._create_new_batch(chunk)
            for chunk in np.array_split(self._data, split_size_or_sections, axis=axis)
        )

    ##############################################################
    #     Array manipulation routines | Rearranging elements     #
    ##############################################################

    def __getitem__(self, index: IndexType) -> np.ndarray:
        if isinstance(index, BatchedArray):
            index = index.data
        return self._data[index]

    def __setitem__(
        self, index: IndexType, value: bool | float | np.ndarray | BatchedArray
    ) -> None:
        if isinstance(index, BatchedArray):
            index = index.data
        if isinstance(value, BatchedArray):
            value = value.data
        self._data[index] = value

    def permute_along_axis(
        self, permutation: np.ndarray | Sequence[int], axis: int
    ) -> TBatchedArray:
        r"""Permute the data/batch along a given axis.

        Args:
            permutation: Specifies the permutation to use on the data.
                The dimension of the permutation input should be
                compatible with the shape of the data.
            axis: Specifies the axis where the permutation is computed.

        Returns:
            A new batch with permuted data.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.permute_along_axis([2, 1, 3, 0, 4], axis=0)
        array([[4, 5],
               [2, 3],
               [6, 7],
               [0, 1],
               [8, 9]], batch_axis=0)

        ```
        """
        return self.index_select(index=permutation, axis=axis)

    def permute_along_axis_(self, permutation: np.ndarray | Sequence[int], axis: int) -> None:
        r"""Permutes the data/batch along a given dimension.

        Args:
            permutation: Specifies the permutation to use on the data.
                The dimension of the permutation input should be
                compatible with the shape of the data.
            axis: Specifies the axis where the permutation is computed.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.permute_along_axis_([2, 1, 3, 0, 4], axis=0)
        >>> batch
        array([[4, 5],
               [2, 3],
               [6, 7],
               [0, 1],
               [8, 9]], batch_axis=0)
        """
        self._data = np.take(self._data, indices=to_array(permutation), axis=axis)

    def shuffle_along_axis(
        self, axis: int, rng: np.random.Generator | None = None
    ) -> TBatchedArray:
        r"""Shuffle the data/batch along a given axis.

        Args:
            axis: Specifies the shuffle axis.
            rng: Specifies the pseudorandom number generator.

        Returns:
            A new batch with shuffled data along a given axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.shuffle_along_axis(axis=0)
        array([[...]], batch_axis=0)

        ```
        """
        rng = setup_rng(rng)
        return self.index_select(axis=axis, index=rng.permutation(self._data.shape[axis]))

    def shuffle_along_axis_(self, axis: int, rng: np.random.Generator | None = None) -> None:
        r"""Shuffle the data/batch along a given axis.

        Args:
            axis: Specifies the shuffle axis.
            rng: Specifies the pseudorandom number generator.

        Returns:
            A new batch with shuffled data along a given axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.shuffle_along_axis_(axis=0)
        >>> batch
        array([[...]], batch_axis=0)

        ```
        """
        self._data = self.shuffle_along_axis(axis=axis, rng=rng).data

    ##############################################
    #     Math | Sums, products, differences     #
    ##############################################

    @overload
    def cumprod(
        self,
        axis: None = ...,
        dtype: DTypeLike = ...,
        out: np.ndarray | None = ...,
    ) -> np.ndarray: ...  # pragma: no cover

    @overload
    def cumprod(
        self,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: np.ndarray | None = ...,
    ) -> TBatchedArray: ...  # pragma: no cover

    def cumprod(
        self,
        axis: SupportsIndex | None = None,
        dtype: DTypeLike = None,
        out: np.ndarray | None = None,
    ) -> TBatchedArray | np.ndarray:
        r"""Return the cumulative product of elements along a given axis.

        Args:
            axis: Axis along which the cumulative product is computed.
                By default, the input is flattened.
            dtype: Type of the returned array and of the accumulator
                in which the elements are multiplied. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.
            out: Alternative output array in which to place the result.
                It must have the same shape and buffer length as the
                expected output but the type will be cast if necessary.

        Returns:
            The cumulative product of elements along a given axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.cumprod(axis=0)
        array([[  0,   1],
               [  0,   3],
               [  0,  15],
               [  0, 105],
               [  0, 945]], batch_axis=0)
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.cumprod(axis=1)
        array([[    0,     0,     0,     0,     0],
               [    5,    30,   210,  1680, 15120]], batch_axis=1)

        ```
        """
        x = self._data.cumprod(axis=axis, dtype=dtype, out=out)
        if out is not None:
            return out
        if axis is not None:
            x = self._create_new_batch(x)
        return x

    def cumprod_along_batch(self, dtype: DTypeLike = None) -> TBatchedArray:
        r"""Return the cumulative product of elements along the batch
        axis.

        Args:
            dtype: Type of the returned array and of the accumulator
                in which the elements are multiplied. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.

        Returns:
            The cumulative product of elements along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.cumprod_along_batch()
        array([[  0,   1],
               [  0,   3],
               [  0,  15],
               [  0, 105],
               [  0, 945]], batch_axis=0)

        ```
        """
        return self.cumprod(axis=self._batch_axis, dtype=dtype)

    @overload
    def cumsum(
        self,
        axis: None = ...,
        dtype: DTypeLike = ...,
        out: np.ndarray | None = ...,
    ) -> np.ndarray: ...  # pragma: no cover

    @overload
    def cumsum(
        self,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: np.ndarray | None = ...,
    ) -> TBatchedArray: ...  # pragma: no cover

    def cumsum(
        self,
        axis: SupportsIndex | None = None,
        dtype: DTypeLike = None,
        out: np.ndarray | None = None,
    ) -> TBatchedArray | np.ndarray:
        r"""Return the cumulative sum of elements along a given axis.

        Args:
            axis: Axis along which the cumulative sum is computed.
                By default, the input is flattened.
            dtype: Type of the returned array and of the accumulator
                in which the elements are summed. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.
            out: Alternative output array in which to place the result.
                It must have the same shape and buffer length as the
                expected output but the type will be cast if necessary.

        Returns:
            The cumulative sum of elements along a given axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.cumsum(axis=0)
        array([[ 0,  1],
               [ 2,  4],
               [ 6,  9],
               [12, 16],
               [20, 25]], batch_axis=0)
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >>> batch.cumsum(axis=1)
        array([[ 0,  1,  3,  6, 10],
               [ 5, 11, 18, 26, 35]], batch_axis=1)

        ```
        """
        x = self._data.cumsum(axis=axis, dtype=dtype, out=out)
        if out is not None:
            return out
        if axis is not None:
            x = self._create_new_batch(x)
        return x

    def cumsum_along_batch(self, dtype: DTypeLike = None) -> TBatchedArray:
        r"""Return the cumulative sum of elements along the batch axis.

        Args:
            dtype: Type of the returned array and of the accumulator
                in which the elements are summed. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.

        Returns:
            The cumulative sum of elements along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
        >>> batch.cumsum_along_batch()
        array([[ 0,  1],
               [ 2,  4],
               [ 6,  9],
               [12, 16],
               [20, 25]], batch_axis=0)

        ```
        """
        return self.cumsum(axis=self._batch_axis, dtype=dtype)

    @overload
    def nancumprod(
        self,
        axis: None = ...,
        dtype: DTypeLike = ...,
        out: np.ndarray | None = ...,
    ) -> np.ndarray: ...  # pragma: no cover

    @overload
    def nancumprod(
        self,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: np.ndarray | None = ...,
    ) -> TBatchedArray: ...  # pragma: no cover

    def nancumprod(
        self,
        axis: SupportsIndex | None = None,
        dtype: DTypeLike = None,
        out: np.ndarray | None = None,
    ) -> TBatchedArray | np.ndarray:
        r"""Return the cumulative product of elements along a given axis
        treating Not a Numbers (NaNs) as one.

        Args:
            axis: Axis along which the cumulative product is computed.
                By default, the input is flattened.
            dtype: Type of the returned array and of the accumulator
                in which the elements are multiplied. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.
            out: Alternative output array in which to place the result.
                It must have the same shape and buffer length as the
                expected output but the type will be cast if necessary.

        Returns:
            The cumulative product of elements along a given axis
                treating Not a Numbers (NaNs) as one.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nancumprod(axis=0)
        array([[ 1.,  1.,  2.],
               [ 3.,  4., 10.]], batch_axis=0)
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
        >>> batch.nancumprod(axis=1)
        array([[ 1.,  1.,  2.],
               [ 3., 12., 60.]], batch_axis=1)

        ```
        """
        x = np.nancumprod(self._data, axis=axis, dtype=dtype, out=out)
        if out is not None:
            return out
        if axis is not None:
            x = self._create_new_batch(x)
        return x

    def nancumprod_along_batch(self, dtype: DTypeLike = None) -> TBatchedArray:
        r"""Return the cumulative product of elements along the batch
        axis treating Not a Numbers (NaNs) as one.

        Args:
            dtype: Type of the returned array and of the accumulator
                in which the elements are multiplied. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.

        Returns:
            The cumulative product of elements along the batch axis
                treating Not a Numbers (NaNs) as one.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nancumprod_along_batch()
        array([[ 1.,  1.,  2.],
               [ 3.,  4., 10.]], batch_axis=0)

        ```
        """
        return self.nancumprod(axis=self._batch_axis, dtype=dtype)

    @overload
    def nancumsum(
        self,
        axis: None = ...,
        dtype: DTypeLike = ...,
        out: np.ndarray | None = ...,
    ) -> np.ndarray: ...  # pragma: no cover

    @overload
    def nancumsum(
        self,
        axis: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: np.ndarray | None = ...,
    ) -> TBatchedArray: ...  # pragma: no cover

    def nancumsum(
        self,
        axis: SupportsIndex | None = None,
        dtype: DTypeLike = None,
        out: np.ndarray | None = None,
    ) -> TBatchedArray | np.ndarray:
        r"""Return the cumulative sum of elements along a given axis
        treating Not a Numbers (NaNs) as zero.

        Args:
            axis: Axis along which the cumulative product is computed.
                By default, the input is flattened.
            dtype: Type of the returned array and of the accumulator
                in which the elements are summed. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.
            out: Alternative output array in which to place the result.
                It must have the same shape and buffer length as the
                expected output but the type will be cast if necessary.

        Returns:
            The cumulative sum of elements along a given axis
                treating Not a Numbers (NaNs) as zero.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nancumsum(axis=0)
        array([[1., 0., 2.],
               [4., 4., 7.]], batch_axis=0)
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
        >>> batch.nancumsum(axis=1)
        array([[ 1.,  1.,  3.],
               [ 3.,  7., 12.]], batch_axis=1)

        ```
        """
        x = np.nancumsum(self._data, axis=axis, dtype=dtype, out=out)
        if out is not None:
            return out
        if axis is not None:
            x = self._create_new_batch(x)
        return x

    def nancumsum_along_batch(self, dtype: DTypeLike = None) -> TBatchedArray:
        r"""Return the cumulative sum of elements along the batch axis
        treating Not a Numbers (NaNs) as zero.

        Args:
            dtype: Type of the returned array and of the accumulator
                in which the elements are summed. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.

        Returns:
            The cumulative sum of elements along the batch axis
                treating Not a Numbers (NaNs) as zero.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nancumsum_along_batch()
        array([[1., 0., 2.],
               [4., 4., 7.]], batch_axis=0)

        ```
        """
        return self.nancumsum(axis=self._batch_axis, dtype=dtype)

    def nanprod(
        self,
        axis: SupportsIndex | None = None,
        dtype: DTypeLike = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
    ) -> TBatchedArray | np.ndarray:
        r"""Return the product of elements along a given axis treating
        Not a Numbers (NaNs) as one.

        Args:
            axis: Axis along which the cumulative product is computed.
                By default, the input is flattened.
            dtype: Type of the returned array and of the accumulator
                in which the elements are multiplied. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.
            out: Alternative output array in which to place the result.
                It must have the same shape and buffer length as the
                expected output but the type will be cast if necessary.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the original array.

        Returns:
            The product of elements along a given axis treating Not a
                Numbers (NaNs) as one.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
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
        x = np.nanprod(self._data, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        if out is not None:
            return out
        return x

    def nanprod_along_batch(self, dtype: DTypeLike = None, keepdims: bool = False) -> TBatchedArray:
        r"""Return the product of elements along the batch axis treating
        Not a Numbers (NaNs) as one.

        Args:
            dtype: Type of the returned array and of the accumulator
                in which the elements are multiplied. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the original array.

        Returns:
            The product of elements along the batch axis treating Not a
                Numbers (NaNs) as one.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nanprod_along_batch()
        array([ 3., 4., 10.])

        ```
        """
        return self.nanprod(axis=self._batch_axis, dtype=dtype, keepdims=keepdims)

    def nansum(
        self,
        axis: SupportsIndex | None = None,
        dtype: DTypeLike = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
    ) -> TBatchedArray | np.ndarray:
        r"""Return the sum of elements along a given axis treating Not a
        Numbers (NaNs) as zero.

        Args:
            axis: Axis along which the cumulative product is computed.
                By default, the input is flattened.
            dtype: Type of the returned array and of the accumulator
                in which the elements are summed. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.
            out: Alternative output array in which to place the result.
                It must have the same shape and buffer length as the
                expected output but the type will be cast if necessary.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the original array.

        Returns:
            The sum of elements along a given axis treating Not a
                Numbers (NaNs) as zero.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
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
        x = np.nansum(self._data, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        if out is not None:
            return out
        return x

    def nansum_along_batch(self, dtype: DTypeLike = None, keepdims: bool = False) -> TBatchedArray:
        r"""Return the sum of elements along the batch axis treating Not
        a Numbers (NaNs) as zero.

        Args:
            dtype: Type of the returned array and of the accumulator
                in which the elements are summed. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the original array.

        Returns:
            The sum of elements along the batch axis treating Not a
                Numbers (NaNs) as zero.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nansum_along_batch()
        array([4., 4., 7.])

        ```
        """
        return self.nansum(axis=self._batch_axis, dtype=dtype, keepdims=keepdims)

    def prod(
        self,
        axis: SupportsIndex | None = None,
        dtype: DTypeLike = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
    ) -> TBatchedArray | np.ndarray:
        r"""Return the product of elements along a given axis.

        Args:
            axis: Axis along which the cumulative product is computed.
                By default, the input is flattened.
            dtype: Type of the returned array and of the accumulator
                in which the elements are multiplied. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.
            out: Alternative output array in which to place the result.
                It must have the same shape and buffer length as the
                expected output but the type will be cast if necessary.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the original array.

        Returns:
            The product of elements along a given axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
        >>> batch.prod(axis=0)
        array([ 3, 24, 10])
        >>> batch.prod(axis=0, keepdims=True)
        array([[ 3, 24, 10]])
        >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
        >>> batch.prod(axis=1)
        array([12, 60])

        ```
        """
        x = self._data.prod(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        if out is not None:
            return out
        return x

    def prod_along_batch(self, dtype: DTypeLike = None, keepdims: bool = False) -> TBatchedArray:
        r"""Return the product of elements along the batch axis.

        Args:
            dtype: Type of the returned array and of the accumulator
                in which the elements are multiplied. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the original array.

        Returns:
            The product of elements along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
        >>> batch.prod_along_batch()
        array([ 3, 24, 10])

        ```
        """
        return self.prod(axis=self._batch_axis, dtype=dtype, keepdims=keepdims)

    def sum(
        self,
        axis: SupportsIndex | None = None,
        dtype: DTypeLike = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
    ) -> TBatchedArray | np.ndarray:
        r"""Return the sum of elements along a given axis.

        Args:
            axis: Axis along which the cumulative product is computed.
                By default, the input is flattened.
            dtype: Type of the returned array and of the accumulator
                in which the elements are summed. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.
            out: Alternative output array in which to place the result.
                It must have the same shape and buffer length as the
                expected output but the type will be cast if necessary.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the original array.

        Returns:
            The sum of elements along a given axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
        >>> batch.sum(axis=0)
        array([ 4, 10, 7])
        >>> batch.sum(axis=0, keepdims=True)
        array([[ 4, 10, 7]])
        >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]), batch_axis=1)
        >>> batch.sum(axis=1)
        array([ 9, 12])

        ```
        """
        x = self._data.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        if out is not None:
            return out
        return x

    def sum_along_batch(self, dtype: DTypeLike = None, keepdims: bool = False) -> TBatchedArray:
        r"""Return the sum of elements along the batch axis.

        Args:
            dtype: Type of the returned array and of the accumulator
                in which the elements are summed. If dtype is not
                specified, it defaults to the dtype of ``self``,
                unless a has an integer dtype with a precision less
                than that of  the default platform integer.
                In that case, the default platform integer is used.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the original array.

        Returns:
            The sum of elements along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
        >>> batch.sum_along_batch()
        array([ 4, 10, 7])

        ```
        """
        return self.sum(axis=self._batch_axis, dtype=dtype, keepdims=keepdims)

    ################
    #     Sort     #
    ################

    def sort(self, axis: SupportsIndex | None = -1, kind: SortKind | None = None) -> None:
        r"""Sort an array in-place.

        Args:
            axis: Axis along which to sort.
            kind: Sorting algorithm. The default is `quicksort`.
                Note that both `stable` and `mergesort` use timsort
                under the covers and, in general, the actual
                implementation will vary with datatype.
                The `mergesort` option is retained for backwards
                compatibility.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
        >>> batch.sort()
        >>> batch
        array([[1, 2, 6],
               [3, 4, 5]], batch_axis=0)

        ```
        """
        self._data.sort(axis=axis, kind=kind)

    def sort_along_batch(self, kind: str | None = None) -> None:
        r"""Sort an array in-place along the batch dimension.

        Args:
            kind: Sorting algorithm. The default is `quicksort`.
                Note that both `stable` and `mergesort` use timsort
                under the covers and, in general, the actual
                implementation will vary with datatype.
                The `mergesort` option is retained for backwards
                compatibility.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
        >>> batch.sort_along_batch()
        >>> batch
        array([[1, 4, 2],
               [3, 6, 5]], batch_axis=0)

        ```
        """
        self.sort(axis=self._batch_axis, kind=kind)

    def argmax(
        self,
        axis: SupportsIndex | None = None,
        out: np.ndarray | None = None,
        *,
        keepdims: bool = False,
    ) -> np.ndarray:
        r"""Return the indices of the maximum values along an axis.

        Args:
            axis: By default, the index is into the flattened array,
                otherwise along the specified axis.
            out: If provided, the result will be inserted into this
                array. It should be of the appropriate shape and dtype.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the array.

        Returns:
            The indices of the maximum values along an axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
        >>> batch.argmax()
        1
        >>> batch.argmax(keepdims=True)
        array([[1]])

        ```
        """
        return self._data.argmax(axis=axis, out=out, keepdims=keepdims)

    def argmax_along_batch(
        self,
        out: np.ndarray | None = None,
        *,
        keepdims: bool = False,
    ) -> np.ndarray:
        r"""Return the indices of the maximum values along the batch
        axis.

        Args:
            axis: By default, the index is into the flattened array,
                otherwise along the specified axis.
            out: If provided, the result will be inserted into this
                array. It should be of the appropriate shape and dtype.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the array.

        Returns:
            The indices of the maximum values along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
        >>> batch.argmax_along_batch()
        array([1, 0, 1])
        >>> batch.argmax_along_batch(keepdims=True)
        array([[1, 0, 1]])

        ```
        """
        return self.argmax(axis=self._batch_axis, out=out, keepdims=keepdims)

    def argmin(
        self,
        axis: SupportsIndex | None = None,
        out: np.ndarray | None = None,
        *,
        keepdims: bool = False,
    ) -> np.ndarray:
        r"""Return the indices of the minimum values along an axis.

        Args:
            axis: By default, the index is into the flattened array,
                otherwise along the specified axis.
            out: If provided, the result will be inserted into this
                array. It should be of the appropriate shape and dtype.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the array.

        Returns:
            The indices of the minimum values along an axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
        >>> batch.argmin()
        0
        >>> batch.argmin(keepdims=True)
        array([[0]])

        ```
        """
        return self._data.argmin(axis=axis, out=out, keepdims=keepdims)

    def argmin_along_batch(
        self,
        out: np.ndarray | None = None,
        *,
        keepdims: bool = False,
    ) -> np.ndarray:
        r"""Return the indices of the minimum values along the batch
        axis.

        Args:
            axis: By default, the index is into the flattened array,
                otherwise along the specified axis.
            out: If provided, the result will be inserted into this
                array. It should be of the appropriate shape and dtype.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the array.

        Returns:
            The indices of the minimum values along the batch axis.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, 6, 2], [3, 4, 5]]))
        >>> batch.argmin_along_batch()
        array([0, 1, 0])
        >>> batch.argmin_along_batch(keepdims=True)
        array([[0, 1, 0]])

        ```
        """
        return self.argmin(axis=self._batch_axis, out=out, keepdims=keepdims)

    def nanargmax(
        self,
        axis: SupportsIndex | None = None,
        out: np.ndarray | None = None,
        *,
        keepdims: bool = False,
    ) -> np.ndarray:
        r"""Return the indices of the maximum values along an axis
        ignoring NaNs.

        Args:
            axis: By default, the index is into the flattened array,
                otherwise along the specified axis.
            out: If provided, the result will be inserted into this
                array. It should be of the appropriate shape and dtype.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the array.

        Returns:
            The indices of the maximum values along an axis ignoring
                NaNs.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nanargmax()
        5
        >>> batch.nanargmax(keepdims=True)
        array([[5]])

        ```
        """
        return np.nanargmax(self._data, axis=axis, out=out, keepdims=keepdims)

    def nanargmax_along_batch(
        self,
        out: np.ndarray | None = None,
        *,
        keepdims: bool = False,
    ) -> np.ndarray:
        r"""Return the indices of the maximum values along the batch axis
        ignoring NaNs.

        Args:
            axis: By default, the index is into the flattened array,
                otherwise along the specified axis.
            out: If provided, the result will be inserted into this
                array. It should be of the appropriate shape and dtype.
            keepdims: If this is set to True, the axes which are
                reduced are left in the result as dimensions with size
                one. With this option, the result will broadcast
                correctly against the array.

        Returns:
            The indices of the maximum values along the batch axis
                ignoring NaNs.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba2 import BatchedArray
        >>> batch = BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))
        >>> batch.nanargmax_along_batch()
        array([1, 1, 1])
        >>> batch.nanargmax_along_batch(keepdims=True)
        array([[1, 1, 1]])

        ```
        """
        return self.nanargmax(axis=self._batch_axis, out=out, keepdims=keepdims)

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


def implements(numpy_function: Callable) -> Callable:
    """Register an __array_function__ implementation for BatchedArray
    objects."""

    def decorator(func: Callable) -> Callable:
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


def setup_rng(rng: np.random.Generator | None) -> np.random.Generator:
    r"""Set up a random number generator.

    Args:
        rng: A random number generator. If ``None``, a random number
            generator is instantiated.

    Returns:
        A random number generator.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat.ba2.core import setup_rng
    >>> rng = setup_rng(None)
    >>> rng.permutation(4)
    array([...])

    ```
    """
    if rng is None:
        return np.random.default_rng()
    return rng
