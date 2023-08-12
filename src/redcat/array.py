from __future__ import annotations

__all__ = ["BatchedArray"]

from collections.abc import Callable, Iterable, Sequence
from itertools import chain
from typing import Any, TypeVar, overload

import numpy as np
from coola import objects_are_allclose, objects_are_equal
from numpy import ndarray

from redcat.return_types import ValuesIndicesTuple
from redcat.types import RNGType
from redcat.utils.array import permute_along_dim, to_array
from redcat.utils.common import check_batch_dims, check_data_and_dim, get_batch_dims
from redcat.utils.random import randperm

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
        return self.__class__(np.isneginf(self._data), batch_dim=self._batch_dim)

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
        return self.__class__(np.isposinf(self._data), batch_dim=self._batch_dim)

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

    def bool(self) -> TBatchedArray:
        r"""Converts the current batch to bool data type.

        Returns:
        -------
            ``BatchedArray``: The current batch to bool data type.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.bool()
            array([[ True,  True,  True],
                   [ True,  True,  True]], batch_dim=0)
        """
        return self._create_new_batch(self._data.astype(bool))

    def double(self) -> TBatchedArray:
        r"""Converts the current batch to double (``float64``) data type.

        Returns:
        -------
            ``BatchedArray``: The current batch to double data type.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.double()
            array([[1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
        """
        return self._create_new_batch(self._data.astype(np.double))

    def float(self) -> TBatchedArray:
        r"""Converts the current batch to float (``float32``) data type.

        Returns:
        -------
            ``BatchedArray``: The current batch to float data type.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.float()
            array([[1., 1., 1.],
                   [1., 1., 1.]], dtype=float32, batch_dim=0)
        """
        return self._create_new_batch(self._data.astype(np.single))

    def int(self) -> TBatchedArray:
        r"""Converts the current batch to int (``int32``) data type.

        Returns:
        -------
            ``BatchedArray``: The current batch to int data type.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.int()
            array([[1, 1, 1],
                   [1, 1, 1]], dtype=int32, batch_dim=0)
        """
        return self._create_new_batch(self._data.astype(np.intc))

    def long(self) -> TBatchedArray:
        r"""Converts the current batch to long (``int64``) data type.

        Returns:
        -------
            ``BatchedArray``: The current batch to long data type.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.long()
            array([[1, 1, 1],
                   [1, 1, 1]], batch_dim=0)
        """
        return self._create_new_batch(self._data.astype(np.int_))

    ##################################################
    #     Mathematical | arithmetical operations     #
    ##################################################

    def __add__(self, other: Any) -> TBatchedArray:
        return self.add(other)

    def __iadd__(self, other: Any) -> TBatchedArray:
        self.add_(other)
        return self

    def __floordiv__(self, other: Any) -> TBatchedArray:
        return self.div(other, rounding_mode="floor")

    def __ifloordiv__(self, other: Any) -> TBatchedArray:
        self.div_(other, rounding_mode="floor")
        return self

    def __mul__(self, other: Any) -> TBatchedArray:
        return self.mul(other)

    def __imul__(self, other: Any) -> TBatchedArray:
        self.mul_(other)
        return self

    def __neg__(self) -> TBatchedArray:
        return self.neg()

    def __sub__(self, other: Any) -> TBatchedArray:
        return self.sub(other)

    def __isub__(self, other: Any) -> TBatchedArray:
        self.sub_(other)
        return self

    def __truediv__(self, other: Any) -> TBatchedArray:
        return self.div(other)

    def __itruediv__(self, other: Any) -> TBatchedArray:
        self.div_(other)
        return self

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

    def div(
        self,
        other: BatchedArray | ndarray | int | float,
        rounding_mode: str | None = None,
    ) -> TBatchedArray:
        r"""Divides the ``self`` batch by the input ``other`.

        Similar to ``out = self / other``

        Args:
        ----
            other (``BatchedArray`` or ``numpy.ndarray`` or int or
                float): Specifies the dividend.
            rounding_mode (str or ``None``, optional): Specifies the
                type of rounding applied to the result.
                - ``None``: true division.
                - ``"floor"``: floor division.
                Default: ``None``

        Returns:
        -------
            ``BatchedArray``: A new batch containing the division
                of the two batches.

        Example usage:

        .. code-block:: pycon

            >>> import numpy
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> out = batch.div(BatchedArray(numpy.full((2, 3), 2.0)))
            >>> batch
            array([[1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
            >>> out
            array([[0.5, 0.5, 0.5],
                   [0.5, 0.5, 0.5]], batch_dim=0)
        """

        batch_dims = get_batch_dims((self, other))
        check_batch_dims(batch_dims)
        if isinstance(other, BatchedArray):
            other = other.data
        return self.__class__(
            get_div_rounding_operator(rounding_mode)(self.data, other),
            batch_dim=batch_dims.pop(),
        )

    def div_(
        self,
        other: BatchedArray | ndarray | int | float,
        rounding_mode: str | None = None,
    ) -> None:
        r"""Divides the ``self`` batch by the input ``other`.

        Similar to ``self /= other`` (in-place)

        Args:
        ----
            other (``BatchedArray`` or ``numpy.ndarray`` or int or
                float): Specifies the dividend.
            rounding_mode (str or ``None``, optional): Specifies the
                type of rounding applied to the result.
                - ``None``: true division.
                - ``"floor"``: floor division.
                Default: ``None``

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.div_(BatchedArray(np.full((2, 3), 2.0)))
            >>> batch
            array([[0.5, 0.5, 0.5],
                   [0.5, 0.5, 0.5]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, other)))
        if isinstance(other, BatchedArray):
            other = other.data
        self._data = get_div_rounding_operator(rounding_mode)(self.data, other)

    def fmod(
        self,
        divisor: BatchedArray | ndarray | int | float,
    ) -> TBatchedArray:
        r"""Computes the element-wise remainder of division.

        The current batch is the dividend.

        Args:
        ----
            divisor (``BatchedArray`` or ``numpy.ndarray`` or int
                or float): Specifies the divisor.

        Returns:
        -------
            ``BatchedArray``: A new batch containing the
                element-wise remainder of division.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> out = batch.fmod(BatchedArray(np.full((2, 3), 2.0)))
            >>> batch
            array([[1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
            >>> out
            array([[1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
        """
        batch_dims = get_batch_dims((self, divisor))
        check_batch_dims(batch_dims)
        if isinstance(divisor, BatchedArray):
            divisor = divisor.data
        return self.__class__(
            np.fmod(self.data, divisor),
            batch_dim=batch_dims.pop(),
        )

    def fmod_(self, divisor: BatchedArray | ndarray | int | float) -> None:
        r"""Computes the element-wise remainder of division.

        The current batch is the dividend.

        Args:
        ----
            divisor (``BatchedArray`` or ``numpy.ndarray`` or int
                or float): Specifies the divisor.

        Example usage:

        .. code-block:: pycon

            >>> import numpy
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.fmod_(BatchedArray(np.full((2, 3), 2.0)))
            >>> batch
            array([[1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, divisor)))
        if isinstance(divisor, BatchedArray):
            divisor = divisor.data
        self._data = np.fmod(self._data, divisor)

    def mul(self, other: BatchedArray | ndarray | int | float) -> TBatchedArray:
        r"""Multiplies the ``self`` batch by the input ``other`.

        Similar to ``out = self * other``

        Args:
        ----
            other (``BatchedArray`` or ``numpy.ndarray`` or int or
                float): Specifies the value to multiply.

        Returns:
        -------
            ``BatchedArray``: A new batch containing the
                multiplication of the two batches.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> out = batch.mul(BatchedArray(np.full((2, 3), 2.0)))
            >>> batch
            array([[1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
            >>> out
            array([[2., 2., 2.],
                   [2., 2., 2.]], batch_dim=0)
        """
        batch_dims = get_batch_dims((self, other))
        check_batch_dims(batch_dims)
        if isinstance(other, BatchedArray):
            other = other.data
        return self.__class__(self.data * other, batch_dim=batch_dims.pop())

    def mul_(self, other: BatchedArray | ndarray | int | float) -> None:
        r"""Multiplies the ``self`` batch by the input ``other`.

        Similar to ``self *= other`` (in-place)

        Args:
        ----
            other (``BatchedArray`` or ``numpy.ndarray`` or int or
                float): Specifies the value to multiply.

        Returns:
        -------
            ``BatchedArray``: A new batch containing the
                multiplication of the two batches.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.mul_(BatchedArray(np.full((2, 3), 2.0)))
            >>> batch
            array([[2., 2., 2.],
                   [2., 2., 2.]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, other)))
        if isinstance(other, BatchedArray):
            other = other.data
        self._data = self._data * other

    def neg(self) -> TBatchedArray:
        r"""Returns a new batch with the negative of the elements.

        Returns:
        -------
            ``BatchedArray``: A new batch with the negative of
                the elements.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> out = batch.neg()
            >>> batch
            array([[1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
            >>> out
            array([[-1., -1., -1.],
                   [-1., -1., -1.]], batch_dim=0)
        """
        return self.__class__(np.negative(self.data), batch_dim=self._batch_dim)

    def sub(
        self,
        other: BatchedArray | ndarray | int | float,
        alpha: int | float = 1,
    ) -> TBatchedArray:
        r"""Subtracts the input ``other``, scaled by ``alpha``, to the
        ``self`` batch.

        Similar to ``out = self - alpha * other``

        Args:
        ----
            other (``BatchedArray`` or ``numpy.ndarray`` or int or
                float): Specifies the value to subtract.
            alpha (int or float, optional): Specifies the scale of the
                batch to substract. Default: ``1``

        Returns:
        -------
            ``BatchedArray``: A new batch containing the diffence of
                the two batches.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> out = batch.sub(BatchedArray(np.full((2, 3), 2.0)))
            >>> batch
            array([[1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
            >>> out
            array([[-1., -1., -1.],
                   [-1., -1., -1.]], batch_dim=0)
        """
        batch_dims = get_batch_dims((self, other))
        check_batch_dims(batch_dims)
        if isinstance(other, BatchedArray):
            other = other.data
        return self.__class__(np.subtract(self.data, other * alpha), batch_dim=batch_dims.pop())

    def sub_(
        self,
        other: BatchedArray | ndarray | int | float,
        alpha: int | float = 1,
    ) -> None:
        r"""Subtracts the input ``other``, scaled by ``alpha``, to the
        ``self`` batch.

        Similar to ``self -= alpha * other`` (in-place)

        Args:
        ----
            other (``BatchedArray`` or ``numpy.ndarray`` or int or
                float): Specifies the value to subtract.
            alpha (int or float, optional): Specifies the scale of the
                batch to substract. Default: ``1``

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.sub_(BatchedArray(np.full((2, 3), 2.0)))
            >>> batch
            array([[-1., -1., -1.],
                   [-1., -1., -1.]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, other)))
        if isinstance(other, BatchedArray):
            other = other.data
        self._data = np.subtract(self._data.data, other * alpha)

    ###########################################################
    #     Mathematical | advanced arithmetical operations     #
    ###########################################################

    @overload
    def cumsum(self, dim: None, *args, **kwargs) -> ndarray:
        r"""See ``cumsum`` documentation."""

    @overload
    def cumsum(self, dim: int, *args, **kwargs) -> TBatchedArray:
        r"""See ``cumsum`` documentation."""

    def cumsum(self, dim: int | None, *args, **kwargs) -> TBatchedArray | ndarray:
        r"""Computes the cumulative sum of elements of the current batch
        in a given dimension.

        Args:
        ----
            dim (int): Specifies the dimension of the cumulative sum.
            **kwargs: see ``numpy.cumsum`` documentation

        Returns:
        -------
            ``BatchedArray``: A batch with the cumulative sum of
                elements of the current batch in a given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import numpy
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(numpy.arange(10).reshape(2, 5))
            >>> batch.cumsum(dim=0)
            array([[ 0,  1,  2,  3,  4],
                   [ 5,  7,  9, 11, 13]], batch_dim=0)
        """
        out = np.cumsum(self._data, dim, *args, **kwargs)
        if dim is None:
            return out
        return self._create_new_batch(out)

    def cumsum_(self, dim: int) -> None:
        r"""Computes the cumulative sum of elements of the current batch
        in a given dimension.

        Args:
        ----
            dim (int): Specifies the dimension of the cumulative sum.
            **kwargs: see ``numpy.cumsum_`` documentation

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(2, 5))
            >>> batch.cumsum_(dim=0)
            >>> batch
            array([[ 0,  1,  2,  3,  4],
                   [ 5,  7,  9, 11, 13]], batch_dim=0)
        """
        self._data = np.cumsum(self._data, dim)

    def cumsum_along_batch(self, *args, **kwargs) -> TBatchedArray:
        r"""Computes the cumulative sum of elements of the current batch
        in the batch dimension.

        Args:
        ----
            **kwargs: see ``numpy.cumsum`` documentation

        Returns:
        -------
            ``BatchedArray``: A batch with the cumulative sum of
                elements of the current batch in the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(2, 5))
            >>> batch.cumsum_along_batch()
            array([[ 0,  1,  2,  3,  4],
                   [ 5,  7,  9, 11, 13]], batch_dim=0)
        """
        return self.cumsum(self._batch_dim, *args, **kwargs)

    def cumsum_along_batch_(self) -> None:
        r"""Computes the cumulative sum of elements of the current batch
        in the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(2, 5))
            >>> batch.cumsum_along_batch_()
            >>> batch
            array([[ 0,  1,  2,  3,  4],
                   [ 5,  7,  9, 11, 13]], batch_dim=0)
        """
        self.cumsum_(self._batch_dim)

    def logcumsumexp(self, dim: int) -> TBatchedArray:
        r"""Computes the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in a given
        dimension.

        Args:
        ----
            dim (int): Specifies the dimension of the cumulative sum.

        Returns:
        -------
            ``BatchedArray``: A batch with the cumulative
                summation of the exponentiation of elements of the
                current batch in a given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(2, 5).astype(float))
            >>> batch.logcumsumexp(dim=1)
            array([[0.        , 1.31326169, 2.40760596, 3.4401897 , 4.4519144 ],
                   [5.        , 6.31326169, 7.40760596, 8.4401897 , 9.4519144 ]], batch_dim=0)
        """
        return self._create_new_batch(np.log(np.cumsum(np.exp(self._data), axis=dim)))

    def logcumsumexp_(self, dim: int) -> None:
        r"""Computes the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in a given
        dimension.

        Args:
        ----
            dim (int): Specifies the dimension of the cumulative sum.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(2, 5).astype(float))
            >>> batch.logcumsumexp_(dim=1)
            >>> batch
            array([[0.        , 1.31326169, 2.40760596, 3.4401897 , 4.4519144 ],
                   [5.        , 6.31326169, 7.40760596, 8.4401897 , 9.4519144 ]], batch_dim=0)
        """
        self._data = self.logcumsumexp(dim=dim).data

    def logcumsumexp_along_batch(self) -> TBatchedArray:
        r"""Computes the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in the batch
        dimension.

        Returns:
        -------
            ``BatchedArray``: A batch with the cumulative
                summation of the exponentiation of elements of the
                current batch in the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(5, 2).astype(float))
            >>> batch.logcumsumexp_along_batch()
            array([[0.        , 1.        ],
                   [2.12692801, 3.12692801],
                   [4.14293163, 5.14293163],
                   [6.14507794, 7.14507794],
                   [8.14536806, 9.14536806]], batch_dim=0)
        """
        return self.logcumsumexp(self._batch_dim)

    def logcumsumexp_along_batch_(self) -> None:
        r"""Computes the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in the batch
        dimension.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(5, 2).astype(float))
            >>> batch.logcumsumexp_along_batch_()
            >>> batch
            array([[0.        , 1.        ],
                   [2.12692801, 3.12692801],
                   [4.14293163, 5.14293163],
                   [6.14507794, 7.14507794],
                   [8.14536806, 9.14536806]], batch_dim=0)
        """
        self.logcumsumexp_(self._batch_dim)

    def permute_along_batch(self, permutation: Sequence[int] | ndarray) -> TBatchedArray:
        return self.permute_along_dim(permutation, dim=self._batch_dim)

    def permute_along_batch_(self, permutation: Sequence[int] | ndarray) -> None:
        self.permute_along_dim_(permutation, dim=self._batch_dim)

    def permute_along_dim(self, permutation: Sequence[int] | ndarray, dim: int) -> TBatchedArray:
        r"""Permutes the data/batch along a given dimension.

        Args:
        ----
            permutation (sequence or ``numpy.ndarray`` of type int
                and shape ``(dimension,)``): Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.
            dim (int): Specifies the dimension where the permutation
                is computed.

        Returns:
        -------
            ``BatchedArray``: A new batch with permuted data.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
            >>> batch.permute_along_dim([2, 1, 3, 0, 4], dim=0)
            array([[4, 5],
                   [2, 3],
                   [6, 7],
                   [0, 1],
                   [8, 9]], batch_dim=0)
        """
        if not isinstance(permutation, ndarray):
            permutation = np.asarray(permutation)
        return self._create_new_batch(
            permute_along_dim(self._data, permutation=permutation, dim=dim)
        )

    def permute_along_dim_(self, permutation: Sequence[int] | ndarray, dim: int) -> None:
        r"""Permutes the data/batch along a given dimension.

        Args:
        ----
            permutation (sequence or ``numpy.ndarray`` of type int
                and shape ``(dimension,)``): Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.
            dim (int): Specifies the dimension where the permutation
                is computed.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
            >>> batch.permute_along_dim_([2, 1, 3, 0, 4], dim=0)
            >>> batch
            array([[4, 5],
                   [2, 3],
                   [6, 7],
                   [0, 1],
                   [8, 9]], batch_dim=0)
        """
        if not isinstance(permutation, ndarray):
            permutation = np.asarray(permutation)
        self._data = permute_along_dim(self._data, permutation=permutation, dim=dim)

    def shuffle_along_dim(self, dim: int, generator: RNGType | None = None) -> TBatchedArray:
        r"""Shuffles the data/batch along a given dimension.

        Args:
        ----
            dim (int): Specifies the shuffle dimension.
            generator (``numpy.random.Generator`` or
                ``torch.Generator`` or ``random.Random`` or ``None``,
                optional): Specifies the pseudorandom number
                generator for sampling or the random seed for the
                random number generator. Default: ``None``

        Returns:
        -------
            ``BatchedArray``:  A new batch with shuffled data
                along a given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
            >>> batch.shuffle_along_dim(dim=0)  # doctest:+ELLIPSIS
            array([[...]], batch_dim=0)
        """
        return self.permute_along_dim(to_array(randperm(self._data.shape[dim], generator)), dim=dim)

    def shuffle_along_dim_(self, dim: int, generator: RNGType | None = None) -> None:
        r"""Shuffles the data/batch along a given dimension.

        Args:
        ----
            dim (int): Specifies the shuffle dimension.
            generator (``numpy.random.Generator`` or
                ``torch.Generator`` or ``random.Random`` or ``None``,
                optional): Specifies the pseudorandom number
                generator for sampling or the random seed for the
                random number generator. Default: ``None``

        Returns:
        -------
            ``BatchedArray``:  A new batch with shuffled data
                along a given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
            >>> batch.shuffle_along_dim_(dim=0)
            >>> batch  # doctest:+ELLIPSIS
            array([[...]], batch_dim=0)
        """
        self.permute_along_dim_(to_array(randperm(self._data.shape[dim], generator)), dim=dim)

    def sort(
        self,
        dim: int = -1,
        descending: bool = False,
        stable: bool = False,
    ) -> ValuesIndicesTuple:
        r"""Sorts the elements of the batch along a given dimension in
        monotonic order by value.

        Args:
        ----
            descending (bool, optional): Controls the sorting order.
                If ``True``, the elements are sorted in descending
                order by value. Default: ``False``
            stable (bool, optional): Makes the sorting routine stable,
                which guarantees that the order of equivalent elements
                is preserved. Default: ``False``

        Returns:
        -------
            (``BatchedArray``, ``BatchedArray``): A tuple
                two values:
                    - The first batch contains the batch values sorted
                        along the given dimension.
                    - The second batch contains the indices that sort
                        the batch along the given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(2, 5))
            >>> batch.sort(descending=True)
            ValuesIndicesTuple(
              (values): array([[4, 3, 2, 1, 0],
                        [9, 8, 7, 6, 5]], batch_dim=0)
              (indices): array([[4, 3, 2, 1, 0],
                        [4, 3, 2, 1, 0]], batch_dim=0)
            )
        """
        indices = np.argsort(self._data, axis=dim, kind="stable" if stable else "quicksort")
        if descending:
            indices = np.flip(indices, axis=dim)
        return ValuesIndicesTuple(
            values=self._create_new_batch(np.take_along_axis(self._data, indices, dim)),
            indices=self._create_new_batch(indices),
        )

    def sort_along_batch(
        self,
        descending: bool = False,
        stable: bool = False,
    ) -> ValuesIndicesTuple:
        r"""Sorts the elements of the batch along the batch dimension in
        monotonic order by value.

        Args:
        ----
            descending (bool, optional): Controls the sorting order.
                If ``True``, the elements are sorted in descending
                order by value. Default: ``False``
            stable (bool, optional): Makes the sorting routine stable,
                which guarantees that the order of equivalent elements
                is preserved. Default: ``False``

        Returns:
        -------
            (``BatchedArray``, ``BatchedArray``): A tuple
                two values:
                    - The first batch contains the batch values sorted
                        along the given dimension.
                    - The second batch contains the indices that sort
                        the batch along the given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(2, 5))
            >>> batch.sort_along_batch(descending=True)
            ValuesIndicesTuple(
              (values): array([[5, 6, 7, 8, 9],
                        [0, 1, 2, 3, 4]], batch_dim=0)
              (indices): array([[1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0]], batch_dim=0)
            )
        """
        return self.sort(dim=self._batch_dim, descending=descending, stable=stable)

    ################################################
    #     Mathematical | point-wise operations     #
    ################################################

    def abs(self) -> TBatchedArray:
        r"""Computes the absolute value of each element.

        Returns:
        -------
            ``BatchedArray``: A batch with the absolute value of
                each element.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0]]))
            >>> batch.abs()
            array([[2., 0., 2.],
                   [1., 1., 3.]], batch_dim=0)
        """
        return self._create_new_batch(np.abs(self._data))

    def abs_(self) -> None:
        r"""Computes the absolute value of each element.

        In-place version of ``abs()``.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0]]))
            >>> batch.abs_()
            >>> batch
            array([[2., 0., 2.],
                   [1., 1., 3.]], batch_dim=0)
        """
        self._data = np.abs(self._data)

    def clamp(
        self,
        min: int | float | None = None,  # noqa: A002
        max: int | float | None = None,  # noqa: A002
    ) -> TBatchedArray:
        r"""Clamps all elements in ``self`` into the range ``[min,
        max]``.

        Note: ``min`` and ``max`` cannot be both ``None``.

        Args:
        ----
            min (int, float or ``None``, optional): Specifies
                the lower bound. If ``min`` is ``None``,
                there is no lower bound. Default: ``None``
            max (int, float or ``None``, optional): Specifies
                the upper bound. If ``max`` is ``None``,
                there is no upper bound. Default: ``None``

        Returns:
        -------
            ``BatchedArray``: A batch with clamped values.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(2, 5))
            >>> batch.clamp(min=2, max=5)
            array([[2, 2, 2, 3, 4],
                   [5, 5, 5, 5, 5]], batch_dim=0)
            >>> batch.clamp(min=2)
            array([[2, 2, 2, 3, 4],
                   [5, 6, 7, 8, 9]], batch_dim=0)
            >>> batch.clamp(max=7)
            array([[0, 1, 2, 3, 4],
                   [5, 6, 7, 7, 7]], batch_dim=0)
        """
        return self._create_new_batch(np.clip(self._data, a_min=min, a_max=max))

    def clamp_(
        self,
        min: int | float | None = None,  # noqa: A002
        max: int | float | None = None,  # noqa: A002
    ) -> None:
        r"""Clamps all elements in ``self`` into the range ``[min,
        max]``.

        Inplace version of ``clamp``.

        Note: ``min`` and ``max`` cannot be both ``None``.

        Args:
        ----
            min (int, float or ``None``, optional): Specifies
                the lower bound.  If ``min`` is ``None``,
                there is no lower bound. Default: ``None``
            max (int, float or ``None``, optional): Specifies
                the upper bound. If ``max`` is ``None``,
                there is no upper bound. Default: ``None``

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(2, 5))
            >>> batch.clamp_(min=2, max=5)
            >>> batch
            array([[2, 2, 2, 3, 4],
                   [5, 5, 5, 5, 5]], batch_dim=0)
            >>> batch = BatchedArray(np.arange(10).reshape(2, 5))
            >>> batch.clamp_(min=2)
            >>> batch
            array([[2, 2, 2, 3, 4],
                   [5, 6, 7, 8, 9]], batch_dim=0)
            >>> batch = BatchedArray(np.arange(10).reshape(2, 5))
            >>> batch.clamp_(max=7)
            >>> batch
            array([[0, 1, 2, 3, 4],
                   [5, 6, 7, 7, 7]], batch_dim=0)
        """
        self._data = np.clip(self._data, a_min=min, a_max=max)

    ##########################################################
    #    Indexing, slicing, joining, mutating operations     #
    ##########################################################

    # def append(self, other: BaseBatch) -> None:
    #     pass

    def cat(
        self,
        tensors: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
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
        if isinstance(tensors, (BatchedArray, ndarray)):
            tensors = [tensors]
        tensors = list(chain([self], tensors))
        batch_dims = get_batch_dims(tensors)
        check_batch_dims(batch_dims)
        return BatchedArray(
            np.concatenate(
                [tensor._data if hasattr(tensor, "_data") else tensor for tensor in tensors],
                axis=dim,
            ),
            batch_dim=batch_dims.pop(),
        )

    def cat_(
        self,
        arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
        dim: int = 0,
    ) -> None:
        r"""Concatenates the data of the batch(es) to the current batch
        along a given dimension and creates a new batch.

        Args:
        ----
            array (``BatchedArray`` or ``numpy.ndarray`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
            >>> batch.cat_(BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])))
            >>> batch
            array([[ 0,  1,  2],
                   [ 4,  5,  6],
                   [10, 11, 12],
                   [13, 14, 15]], batch_dim=0)
        """
        self._data = self.cat(arrays, dim=dim).data

    def cat_along_batch(
        self, arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray]
    ) -> TBatchedArray:
        r"""Concatenates the data of the batch(es) to the current batch
        along the batch dimension and creates a new batch.

        Args:
        ----
            arrays (``BatchedArray`` or ``numpy.ndarray`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Returns:
        -------
            ``BatchedArray``: A batch with the concatenated data
                in the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
            >>> batch.cat_along_batch(BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])))
            array([[ 0,  1,  2],
                   [ 4,  5,  6],
                   [10, 11, 12],
                   [13, 14, 15]], batch_dim=0)
            >>> batch = BatchedArray(np.array([[0, 4], [1, 5], [2, 6]]))
            >>> batch.cat_along_batch(
            ...     [
            ...         BatchedArray(np.array([[10, 12], [11, 13]])),
            ...         BatchedArray(np.array([[20, 22], [21, 23]])),
            ...     ]
            ... )
            array([[ 0,  4],
                   [ 1,  5],
                   [ 2,  6],
                   [10, 12],
                   [11, 13],
                   [20, 22],
                   [21, 23]], batch_dim=0)
        """
        return self.cat(arrays, dim=self._batch_dim)

    def cat_along_batch_(
        self, arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray]
    ) -> None:
        r"""Concatenates the data of the batch(es) to the current batch
        along the batch dimension and creates a new batch.

        Args:
        ----
            arrays (``BatchedArray`` or ``numpy.ndarray`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
            >>> batch.cat_along_batch_(BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])))
            >>> batch
            array([[ 0,  1,  2],
                   [ 4,  5,  6],
                   [10, 11, 12],
                   [13, 14, 15]], batch_dim=0)
            >>> batch = BatchedArray(np.array([[0, 4], [1, 5], [2, 6]]))
            >>> batch.cat_along_batch_(
            ...     [
            ...         BatchedArray(np.array([[10, 12], [11, 13]])),
            ...         BatchedArray(np.array([[20, 22], [21, 23]])),
            ...     ]
            ... )
            >>> batch
            array([[ 0,  4],
                   [ 1,  5],
                   [ 2,  6],
                   [10, 12],
                   [11, 13],
                   [20, 22],
                   [21, 23]], batch_dim=0)
        """
        self.cat_(arrays, dim=self._batch_dim)

    # def chunk_along_batch(self, chunks: int) -> tuple[TBatchedArray, ...]:
    #     pass
    #
    # def extend(self, other: Iterable[BaseBatch]) -> None:
    #     pass
    #
    # def index_select_along_batch(self, index: ndarray | Sequence[int]) -> BaseBatch:
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
    #         >>> import numpy
    #         >>> from redcat import BatchedArray
    #         >>> batch = BatchedArray(numpy.arange(10).reshape(5, 2))
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
    return arrays[0].cat(arrays[1:], axis)


@overload
def cumsum(a: TBatchedArray, axis: None, *args, **kwargs) -> ndarray:
    r"""See ``np.cumsum`` documentation."""


@overload
def cumsum(a: TBatchedArray, axis: int, *args, **kwargs) -> TBatchedArray:
    r"""See ``np.cumsum`` documentation."""


@implements(np.cumsum)
def cumsum(a: TBatchedArray, axis: int | None = None, *args, **kwargs) -> TBatchedArray | ndarray:
    r"""See ``np.cumsum`` documentation."""
    return a.cumsum(axis, *args, **kwargs)


@implements(np.isneginf)
def isneginf(x: BatchedArray) -> BatchedArray:
    r"""See ``np.isneginf`` documentation."""
    return x.isneginf()


@implements(np.isposinf)
def isposinf(x: BatchedArray) -> BatchedArray:
    r"""See ``np.isposinf`` documentation."""
    return x.isposinf()


@implements(np.sum)
def numpysum(input: BatchedArray, *args, **kwargs) -> ndarray:  # noqa: A002
    r"""See ``np.sum`` documentation.

    Use the name ``numpysum`` to avoid shadowing `sum` python builtin.
    """
    return np.sum(input.data, *args, **kwargs)


def get_div_rounding_operator(mode: str | None) -> Callable:
    r"""Gets the rounding operator for a division.

    Args:
    ----
        mode (str or ``None``, optional): Specifies the
            type of rounding applied to the result.
            - ``None``: true division.
            - ``"floor"``: floor division.
            Default: ``None``

    Returns:
    -------
        ``Callable``: The rounding operator for a division

    Example usage:

    .. code-block:: pycon

        >>> from redcat.array import get_div_rounding_operator
        >>> get_div_rounding_operator(None)
        <ufunc 'divide'>
    """
    if mode is None:
        return np.true_divide
    if mode == "floor":
        return np.floor_divide
    raise RuntimeError(f"Incorrect `rounding_mode` {mode}. Valid values are: None and 'floor'")
