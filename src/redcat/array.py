from __future__ import annotations

__all__ = ["BatchedArray"]

from collections.abc import Callable, Iterable, Sequence
from itertools import chain
from typing import Any, TypeVar

import numpy as np
from coola import objects_are_allclose, objects_are_equal
from numpy import ndarray

from redcat.utils.array import check_batch_dims, check_data_and_dim, get_batch_dims

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")

HANDLED_FUNCTIONS = {}


class BatchedArray(np.lib.mixins.NDArrayOperatorsMixin):  # (BaseBatch[ndarray]):
    r"""Implements a batched array to easily manipulate a batch of
    examples.

    Args:
    ----
        data (array_like): Specifies the data for the array. It can
            be a list, tuple, NumPy ndarray, scalar, and other types.
        batch_dim (int, optional): Specifies the batch dimension
            in the ``numpy.ndarray`` object. Default: ``0``
        kwargs: Keyword arguments that are passed to
            ``numpy.asarray``.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from redcat import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5))
    """

    def __init__(self, data: Any, *, batch_dim: int = 0, **kwargs) -> None:
        super().__init__()
        self._data = np.asarray(data, **kwargs)
        check_data_and_dim(self._data, batch_dim)
        self._batch_dim = int(batch_dim)

    def __repr__(self) -> str:
        return repr(self._data)[:-1] + f", batch_dim={self._batch_dim})"

    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs, **kwargs) -> TBatchedArray:
        # if method != "__call__":
        #     raise NotImplementedError
        batch_dims = get_batch_dims(inputs, kwargs)
        check_batch_dims(batch_dims)
        args = [a._data if hasattr(a, "_data") else a for a in inputs]
        return self.__class__(ufunc(*args, **kwargs), batch_dim=batch_dims.pop())

    def __array_function__(
        self,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> TBatchedArray:
        # if func not in HANDLED_FUNCTIONS:
        #     return NotImplemented
        #     # Note: this allows subclasses that don't override
        #     # __array_function__ to handle BatchedArray objects
        # if not all(issubclass(t, BatchedArray) for t in types):
        #     return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    @property
    def batch_dim(self) -> int:
        r"""int: The batch dimension in the ``numpy.ndarray`` object."""
        return self._batch_dim

    @property
    def batch_size(self) -> int:
        return self._data.shape[self._batch_dim]

    @property
    def data(self) -> ndarray:
        return self._data

    @property
    def dtype(self) -> np.dtype:
        r"""``numpy.dtype``: The data type."""
        return self._data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        r"""``tuple``: The shape of the array."""
        return self._data.shape

    def dim(self) -> int:
        r"""Gets the number of dimensions.

        Returns:
        -------
            int: The number of dimensions

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.dim()
            2
        """
        return self._data.ndim

    def ndimension(self) -> int:
        r"""Gets the number of dimensions.

        Returns:
        -------
            int: The number of dimensions

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.ndimension()
            2
        """
        return self.dim()

    def numel(self) -> int:
        r"""Gets the total number of elements in the array.

        Returns:
        -------
            int: The total number of elements

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.numel()
            6
        """
        return np.prod(self._data.shape).item()

    ###############################
    #     Creation operations     #
    ###############################

    def clone(self, *args, **kwargs) -> TBatchedArray:
        r"""Creates a copy of the current batch.

        Args:
        ----
            *args: See the documentation of ``numpy.copy``
            **kwargs: See the documentation of ``numpy.copy``

        Returns:
        -------
            ``BatchedArray``: A copy of the current batch.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch_copy = batch.clone()
            >>> batch_copy
            array([[1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
        """
        return self._create_new_batch(self._data.copy(*args, **kwargs))

    def copy(self, *args, **kwargs) -> TBatchedArray:
        r"""Creates a copy of the current batch.

        Args:
        ----
            *args: See the documentation of ``numpy.copy``
            **kwargs: See the documentation of ``numpy.copy``

        Returns:
        -------
            ``BatchedArray``: A copy of the current batch.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch_copy = batch.copy()
            >>> batch_copy
            array([[1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
        """
        return self.clone(*args, **kwargs)

    def empty_like(self, *args, **kwargs) -> TBatchedArray:
        r"""Creates an uninitialized batch, with the same shape as the
        current batch.

        Args:
        ----
            *args: See the documentation of ``numpy.empty_like``
            **kwargs: See the documentation of
                ``numpy.empty_like``

        Returns:
        -------
            ``BatchedArray``: A uninitialized batch with the same
                shape as the current batch.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.empty_like()  # doctest:+ELLIPSIS
            array([[...]], batch_dim=0)
        """
        return self._create_new_batch(np.empty_like(self._data, *args, **kwargs))

    def full_like(self, *args, **kwargs) -> TBatchedArray:
        r"""Creates a batch filled with a given scalar value, with the
        same shape as the current batch.

        Args:
        ----
            *args: See the documentation of ``numpy.full_like``
            **kwargs: See the documentation of
                ``numpy.full_like``

        Returns:
        -------
            ``BatchedArray``: A batch filled with the scalar
                value, with the same shape as the current batch.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.full_like(42)
            array([[42., 42., 42.],
                   [42., 42., 42.]], batch_dim=0)
        """
        return self._create_new_batch(np.full_like(self._data, *args, **kwargs))

    def new_full(
        self,
        fill_value: float | int | bool,
        batch_size: int | None = None,
        **kwargs,
    ) -> TBatchedArray:
        r"""Creates a batch filled with a scalar value.

        By default, the array in the returned batch has the same
        shape, ``numpy.dtype`` as the array in the current batch.

        Args:
        ----
            fill_value (float or int or bool): Specifies the number
                to fill the batch with.
            batch_size (int or ``None``): Specifies the batch size.
                If ``None``, the batch size of the current batch is
                used. Default: ``None``.
            **kwargs: See the documentation of
                ``numpy.new_full``.

        Returns:
        -------
            ``BatchedArray``: A batch filled with the scalar value.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.new_full(42)
            array([[42., 42., 42.],
                   [42., 42., 42.]], batch_dim=0)
            >>> batch.new_full(42, batch_size=5)
            array([[42., 42., 42.],
                   [42., 42., 42.],
                   [42., 42., 42.],
                   [42., 42., 42.],
                   [42., 42., 42.]], batch_dim=0)
        """
        shape = list(self._data.shape)
        if batch_size is not None:
            shape[self._batch_dim] = batch_size
        kwargs["dtype"] = kwargs.get("dtype", self.dtype)
        return self._create_new_batch(np.full(shape, fill_value=fill_value, **kwargs))

    def new_ones(
        self,
        batch_size: int | None = None,
        **kwargs,
    ) -> BatchedArray:
        r"""Creates a batch filled with the scalar value ``1``.

        By default, the array in the returned batch has the same
        shape, ``numpy.dtype`` as the array in the current batch.

        Args:
        ----
            batch_size (int or ``None``): Specifies the batch size.
                If ``None``, the batch size of the current batch is
                used. Default: ``None``.
            **kwargs: See the documentation of
                ``numpy.new_ones``.

        Returns:
        -------
            ``BatchedArray``: A batch filled with the scalar
                value ``1``.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.zeros((2, 3)))
            >>> batch.new_ones()
            array([[1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
            >>> batch.new_ones(batch_size=5)
            array([[1., 1., 1.],
                   [1., 1., 1.],
                   [1., 1., 1.],
                   [1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
        """
        shape = list(self._data.shape)
        if batch_size is not None:
            shape[self._batch_dim] = batch_size
        kwargs["dtype"] = kwargs.get("dtype", self.dtype)
        return self._create_new_batch(np.ones(shape, **kwargs))

    def new_zeros(
        self,
        batch_size: int | None = None,
        **kwargs,
    ) -> TBatchedArray:
        r"""Creates a batch filled with the scalar value ``0``.

        By default, the array in the returned batch has the same
        shape, ``numpy.dtype``  as the array in the current batch.

        Args:
        ----
            batch_size (int or ``None``): Specifies the batch size.
                If ``None``, the batch size of the current batch is
                used. Default: ``None``.
            **kwargs: See the documentation of
                ``numpy.new_zeros``.

        Returns:
        -------
            ``BatchedArray``: A batch filled with the scalar
                value ``0``.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.new_zeros()
            array([[0., 0., 0.],
                   [0., 0., 0.]], batch_dim=0)
            >>> batch.new_zeros(batch_size=5)
            array([[0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.]], batch_dim=0)
        """
        shape = list(self._data.shape)
        if batch_size is not None:
            shape[self._batch_dim] = batch_size
        kwargs["dtype"] = kwargs.get("dtype", self.dtype)
        return self._create_new_batch(np.zeros(shape, **kwargs))

    def ones_like(self, *args, **kwargs) -> TBatchedArray:
        r"""Creates a batch filled with the scalar value ``1``, with the
        same shape as the current batch.

        Args:
        ----
            *args: See the documentation of ``numpy.ones_like``
            **kwargs: See the documentation of
                ``numpy.ones_like``

        Returns:
        -------
            ``BatchedArray``: A batch filled with the scalar
                value ``1``, with the same shape as the current
                batch.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.ones_like()
            array([[1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
        """
        return self._create_new_batch(np.ones_like(self._data, *args, **kwargs))

    def zeros_like(self, *args, **kwargs) -> TBatchedArray:
        r"""Creates a batch filled with the scalar value ``0``, with the
        same shape as the current batch.

        Args:
        ----
            *args: See the documentation of ``numpy.zeros_like``
            **kwargs: See the documentation of
                ``numpy.zeros_like``

        Returns:
        -------
            ``BatchedArray``: A batch filled with the scalar
                value ``0``, with the same shape as the current
                batch.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.zeros_like()
            array([[0., 0., 0.],
                   [0., 0., 0.]], batch_dim=0)
        """
        return self._create_new_batch(np.zeros_like(self._data, *args, **kwargs))

    #################################
    #     Conversion operations     #
    #################################

    def astype(
        self, dtype: np.dtype | type[int] | type[float] | type[bool], *args, **kwargs
    ) -> TBatchedArray:
        r"""Moves and/or casts the data.

        Args:
        ----
            *args: See the documentation of ``numpy.astype``
            **kwargs: See the documentation of ``numpy.astype``

        Returns:
        -------
            ``BatchedArray``: A new batch with the data after
                dtype and/or device conversion.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.astype(dtype=bool)
            array([[  True,  True,  True],
                   [  True,  True,  True]], batch_dim=0)
        """
        return self._create_new_batch(self._data.astype(dtype, *args, **kwargs))

    def to(self, *args, **kwargs) -> TBatchedArray:
        r"""Moves and/or casts the data.

        Args:
        ----
            *args: See the documentation of ``numpy.astype``
            **kwargs: See the documentation of ``numpy.astype``

        Returns:
        -------
            ``BatchedArray``: A new batch with the data after
                dtype and/or device conversion.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.to(dtype=bool)
            array([[  True,  True,  True],
                   [  True,  True,  True]], batch_dim=0)
        """
        return self._create_new_batch(self._data.astype(*args, **kwargs))

    #################################
    #     Comparison operations     #
    #################################

    def __ge__(self, other: Any) -> TBatchedArray:
        return self.ge(other)

    def __gt__(self, other: Any) -> TBatchedArray:
        return self.gt(other)

    def __le__(self, other: Any) -> TBatchedArray:
        return self.le(other)

    def __lt__(self, other: Any) -> TBatchedArray:
        return self.lt(other)

    def allclose(
        self, other: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
    ) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self._batch_dim != other.batch_dim:
            return False
        if self._data.shape != other.data.shape:
            return False
        return objects_are_allclose(
            self._data, other.data, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    def eq(self, other: BatchedArray | ndarray | bool | int | float) -> TBatchedArray:
        r"""Computes element-wise equality.

        Args:
        ----
            other: Specifies the batch to compare.

        Returns:
        -------
            ``BatchedArray``: A batch containing the element-wise
                equality.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch1 = BatchedArray(np.array([[1, 3, 4], [0, 2, 2]]))
            >>> batch2 = BatchedArray(np.array([[5, 3, 2], [0, 1, 2]]))
            >>> batch1.eq(batch2)
            array([[False,  True, False],
                   [ True, False,  True]], batch_dim=0)
            >>> batch1.eq(np.array([[5, 3, 2], [0, 1, 2]]))
            array([[False,  True, False],
                   [ True, False,  True]], batch_dim=0)
            >>> batch1.eq(2)
            array([[False, False, False],
                   [False,  True,  True]], batch_dim=0)
        """
        return np.equal(self, other)

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self._batch_dim != other.batch_dim:
            return False
        return objects_are_equal(self._data, other.data)

    def ge(self, other: BatchedArray | ndarray | bool | int | float) -> TBatchedArray:
        r"""Computes ``self >= other`` element-wise.

        Args:
        ----
            other: Specifies the value to compare
                with.

        Returns:
        -------
            ``BatchedArray``: A batch containing the element-wise
                comparison.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch1 = BatchedArray(np.array([[1, 3, 4], [0, 2, 2]]))
            >>> batch2 = BatchedArray(np.array([[5, 3, 2], [0, 1, 2]]))
            >>> batch1.ge(batch2)
            array([[False,  True,  True],
                   [ True,  True,  True]], batch_dim=0)
            >>> batch1.ge(np.array([[5, 3, 2], [0, 1, 2]]))
            array([[False,  True,  True],
                   [ True,  True,  True]], batch_dim=0)
            >>> batch1.ge(2)
            array([[False,  True,  True],
                   [False,  True,  True]], batch_dim=0)
        """
        return np.greater_equal(self, other)

    def gt(self, other: BatchedArray | ndarray | bool | int | float) -> TBatchedArray:
        r"""Computes ``self > other`` element-wise.

        Args:
        ----
            other: Specifies the batch to compare.

        Returns:
        -------
            ``BatchedArray``: A batch containing the element-wise
                comparison.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch1 = BatchedArray(np.array([[1, 3, 4], [0, 2, 2]]))
            >>> batch2 = BatchedArray(np.array([[5, 3, 2], [0, 1, 2]]))
            >>> batch1.gt(batch2)
            array([[False, False,  True],
                   [False,  True, False]], batch_dim=0)
            >>> batch1.gt(np.array([[5, 3, 2], [0, 1, 2]]))
            array([[False, False,  True],
                   [False,  True, False]], batch_dim=0)
            >>> batch1.gt(2)
            array([[False,  True,  True],
                   [False, False, False]], batch_dim=0)
        """
        return np.greater(self, other)

    def isinf(self) -> TBatchedArray:
        r"""Indicates if each element of the batch is infinite (positive
        or negative infinity) or not.

        Returns:
        -------
            ``BatchedArray``:  A batch containing a boolean array
                that is ``True`` where the current batch is infinite
                and ``False`` elsewhere.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
            >>> batch.isinf()
            array([[False, False, True],
                   [False, False, True]], batch_dim=0)
        """
        return np.isinf(self)

    def isneginf(self) -> TBatchedArray:
        r"""Indicates if each element of the batch is negative infinity
        or not.

        Returns:
        -------
            BatchedArray:  A batch containing a boolean array
                that is ``True`` where the current batch is negative
                infinity and ``False`` elsewhere.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
            >>> batch.isneginf()
            array([[False, False, False],
                   [False, False,  True]], batch_dim=0)
        """
        return np.isneginf(self)

    def isposinf(self) -> TBatchedArray:
        r"""Indicates if each element of the batch is positive infinity
        or not.

        Returns:
        -------
            ``BatchedArray``:  A batch containing a boolean array
                that is ``True`` where the current batch is positive
                infinity and ``False`` elsewhere.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
            >>> batch.isposinf()
            array([[False, False,   True],
                   [False, False,  False]], batch_dim=0)
        """
        return np.isposinf(self)

    def isnan(self) -> TBatchedArray:
        r"""Indicates if each element in the batch is NaN or not.

        Returns:
        -------
            ``BatchedArray``:  A batch containing a boolean array
                that is ``True`` where the current batch is infinite
                and ``False`` elsewhere.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[1.0, 0.0, float("nan")], [float("nan"), -2.0, -1.0]]))
            >>> batch.isnan()
            array([[False, False,  True],
                   [ True, False, False]], batch_dim=0)
        """
        return np.isnan(self)

    def le(self, other: BatchedArray | ndarray | bool | int | float) -> TBatchedArray:
        r"""Computes ``self <= other`` element-wise.

        Args:
        ----
            other: Specifies the batch to compare.

        Returns:
        -------
            ``BatchedArray``: A batch containing the element-wise
                comparison.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch1 = BatchedArray(np.array([[1, 3, 4], [0, 2, 2]]))
            >>> batch2 = BatchedArray(np.array([[5, 3, 2], [0, 1, 2]]))
            >>> batch1.le(batch2)
            array([[ True,  True, False],
                   [ True, False,  True]], batch_dim=0)
            >>> batch1.le(np.array([[5, 3, 2], [0, 1, 2]]))
            array([[ True,  True, False],
                   [ True, False,  True]], batch_dim=0)
            >>> batch1.le(2)
            array([[ True, False, False],
                   [ True,  True,  True]], batch_dim=0)
        """
        return np.less_equal(self, other)

    def lt(self, other: BatchedArray | ndarray | bool | int | float) -> TBatchedArray:
        r"""Computes ``self < other`` element-wise.

        Args:
        ----
            other: Specifies the batch to compare.

        Returns:
        -------
            ``BatchedArray``: A batch containing the element-wise
                comparison.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch1 = BatchedArray(np.array([[1, 3, 4], [0, 2, 2]]))
            >>> batch2 = BatchedArray(np.array([[5, 3, 2], [0, 1, 2]]))
            >>> batch1.lt(batch2)
            array([[ True, False, False],
                   [False, False, False]], batch_dim=0)
            >>> batch1.lt(np.array([[5, 3, 2], [0, 1, 2]]))
            array([[ True, False, False],
                  [False, False, False]], batch_dim=0)
            >>> batch1.lt(2)
            array([[ True, False, False],
                   [ True, False, False]], batch_dim=0)
        """
        return np.less(self, other)

    #################
    #     dtype     #
    #################

    ##################################################
    #     Mathematical | arithmetical operations     #
    ##################################################

    def add(
        self,
        other: BatchedArray | ndarray | int | float,
        alpha: int | float = 1.0,
    ) -> TBatchedArray:
        r"""Adds the input ``other``, scaled by ``alpha``, to the
        ``self`` batch.

        Similar to ``out = self + alpha * other``

        Args:
        ----
            other (``BatchedArray`` or ``numpy.ndarray`` or int or
                float): Specifies the other value to add to the
                current batch.
            alpha (int or float, optional): Specifies the scale of the
                batch to add. Default: ``1.0``

        Returns:
        -------
            ``BatchedArray``: A new batch containing the addition of
                the two batches.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> out = batch.add(BatchedArray(np.full((2, 3), 2.0)))
            >>> batch
            array([[1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
            >>> out
            array([[3., 3., 3.],
                   [3., 3., 3.]], batch_dim=0)
        """
        batch_dims = get_batch_dims((self, other))
        check_batch_dims(batch_dims)
        if isinstance(other, BatchedArray):
            other = other.data
        return self.__class__(np.add(self.data, other * alpha), batch_dim=batch_dims.pop())

    def add_(
        self,
        other: BatchedArray | ndarray | int | float,
        alpha: int | float = 1.0,
    ) -> None:
        r"""Adds the input ``other``, scaled by ``alpha``, to the
        ``self`` batch.

        Similar to ``self += alpha * other`` (in-place)

        Args:
        ----
            other (``BatchedArray`` or ``numpy.ndarray`` or int or
                float): Specifies the other value to add to the
                current batch.
            alpha (int or float, optional): Specifies the scale of the
                batch to add. Default: ``1.0``

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.add_(BatchedArray(np.full((2, 3), 2.0)))
            >>> batch
            array([[3., 3., 3.],
                   [3., 3., 3.]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, other)))
        if isinstance(other, BatchedArray):
            other = other.data
        self._data = np.add(self._data.data, other * alpha)

    # def permute_along_batch(self, permutation: IndicesType) -> TBatchedArray:
    #     return self.permute_along_dim(permutation, dim=self._batch_dim)
    #
    # def permute_along_batch_(self, permutation: IndicesType) -> None:
    #     self.permute_along_dim_(permutation, dim=self._batch_dim)

    ##########################################################
    #    Indexing, slicing, joining, mutating operations     #
    ##########################################################

    # def append(self, other: BaseBatch) -> None:
    #     pass

    def cat(
        self,
        arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
        dim: int = 0,
    ) -> TBatchedArray:
        r"""Concatenates the data of the batch(es) to the current batch
        along a given dimension and creates a new batch.

        Args:
        ----
            arrays (``BatchedArray`` or ``numpy.ndarray`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Returns:
        -------
            ``BatchedArray``: A batch with the concatenated data
                in the given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
            >>> batch.cat(BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])))
            array([[ 0,  1,  2],
                   [ 4,  5,  6],
                   [10, 11, 12],
                   [13, 14, 15]], batch_dim=0)
        """
        return self.concatenate(arrays, dim)

    def concatenate(
        self,
        arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
        axis: int = 0,
    ) -> TBatchedArray:
        r"""Concatenates the data of the batch(es) to the current batch
        along a given dimension and creates a new batch.

        Args:
        ----
            arrays (``BatchedArray`` or ``numpy.ndarray`` or
                ``Iterable``): Specifies the batch(es) to concatenate.
            axis (int, optional): Specifies the axis along which the
                arrays will be concatenated. Default: ``0``

        Returns:
        -------
            ``BatchedArray``: A batch with the concatenated data
                in the given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
            >>> batch.concatenate(BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])))
            array([[ 0,  1,  2],
                   [ 4,  5,  6],
                   [10, 11, 12],
                   [13, 14, 15]], batch_dim=0)
        """
        if isinstance(arrays, (BatchedArray, ndarray)):
            arrays = [arrays]
        return np.concatenate(list(chain([self], arrays)), axis=axis)

    # def chunk_along_batch(self, chunks: int) -> tuple[TBatchedArray, ...]:
    #     pass
    #
    # def extend(self, other: Iterable[BaseBatch]) -> None:
    #     pass
    #
    # def index_select_along_batch(self, index: Tensor | Sequence[int]) -> BaseBatch:
    #     pass
    #
    # def slice_along_batch(
    #     self, start: int = 0, stop: int | None = None, step: int = 1
    # ) -> TBatchedArray:
    #     pass
    #
    # def split(
    #     self, split_size_or_sections: int | Sequence[int], dim: int = 0
    # ) -> tuple[TBatchedArray, ...]:
    #     r"""Splits the batch into chunks along a given dimension.
    #
    #     Args:
    #     ----
    #         split_size_or_sections (int or sequence): Specifies the
    #             size of a single chunk or list of sizes for each chunk.
    #         dim (int, optional): Specifies the dimension along which
    #             to split the array. Default: ``0``
    #
    #     Returns:
    #     -------
    #         tuple: The batch split into chunks along the given
    #             dimension.
    #
    #     Example usage:
    #
    #     .. code-block:: pycon
    #
    #         >>> import torch
    #         >>> from redcat import BatchedArray
    #         >>> batch = BatchedArray(torch.arange(10).view(5, 2))
    #         >>> batch.split(2, dim=0)
    #         (array([[0, 1], [2, 3]], batch_dim=0),
    #          array([[4, 5], [6, 7]], batch_dim=0),
    #          array([[8, 9]], batch_dim=0))
    #     """
    #     if isinstance(split_size_or_sections, int):
    #         split_size_or_sections = np.arange(
    #             split_size_or_sections, self._data.shape[dim], split_size_or_sections
    #         )
    #     return np.split(self, split_size_or_sections)
    #
    # def split_along_batch(
    #     self, split_size_or_sections: int | Sequence[int]
    # ) -> tuple[TBatchedArray, ...]:
    #     return self.split(split_size_or_sections, dim=self._batch_dim)

    #################
    #     Other     #
    #################

    def summary(self) -> str:
        dims = ", ".join([f"{key}={value}" for key, value in self._get_kwargs().items()])
        return f"{self.__class__.__qualname__}(dtype={self.dtype}, shape={self.shape}, {dims})"

    def _create_new_batch(self, data: ndarray) -> TBatchedArray:
        return self.__class__(data, **self._get_kwargs())

    def _get_kwargs(self) -> dict:
        return {"batch_dim": self._batch_dim}

    # TODO: remove later. Temporary hack because BatchedArray is not a BaseBatch yet
    def __eq__(self, other: Any) -> bool:
        return self.equal(other)


def implements(np_function: Callable) -> Callable:
    r"""Register an `__array_function__` implementation for
    ``BatchedArray`` objects.

    Args:
    ----
        np_function (``Callable``):  Specifies the numpy function
            to override.

    Returns:
    -------
        ``Callable``: The decorated function.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from redcat.array import BatchedArray, implements
        >>> @implements(np.sum)
        ... def mysum(input: BatchedArray, *args, **kwargs) -> np.ndarray:
        ...     return np.sum(input.data, *args, **kwargs)
        ...
        >>> np.sum(BatchedArray(np.ones((2, 3))))
        6.0
    """

    def decorator(func: Callable) -> Callable:
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


@implements(np.concatenate)
def concatenate(arrays: Sequence[BatchedArray | ndarray], axis: int = 0) -> BatchedArray:
    r"""See ``numpy.concatenate`` documentation."""
    batch_dims = get_batch_dims(arrays)
    check_batch_dims(batch_dims)
    return BatchedArray(
        np.concatenate(
            [array._data if hasattr(array, "_data") else array for array in arrays], axis=axis
        ),
        batch_dim=batch_dims.pop(),
    )


@implements(np.isneginf)
def isneginf(x: BatchedArray) -> BatchedArray:
    r"""See ``np.isneginf`` documentation."""
    return x.__class__(np.isneginf(x.data), batch_dim=x.batch_dim)


@implements(np.isposinf)
def isposinf(x: BatchedArray) -> BatchedArray:
    r"""See ``np.isposinf`` documentation."""
    return x.__class__(np.isposinf(x.data), batch_dim=x.batch_dim)


@implements(np.sum)
def numpysum(input: BatchedArray, *args, **kwargs) -> ndarray:  # noqa: A002
    r"""See ``np.sum`` documentation.

    Use the name ``numpysum`` to avoid shadowing `sum` python builtin.
    """
    return np.sum(input.data, *args, **kwargs)
