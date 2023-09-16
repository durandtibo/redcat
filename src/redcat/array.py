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
from redcat.utils.array import get_div_rounding_operator, permute_along_axis, to_array
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
        check_batch_dims(get_batch_dims(inputs, kwargs))
        args = [a._data if hasattr(a, "_data") else a for a in inputs]
        return self.__class__(ufunc(*args, **kwargs), batch_dim=self._batch_dim)

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
    def ndim(self) -> int:
        r"""``int``: The number of dimensions."""
        return self._data.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        r"""``tuple``: The shape of the array."""
        return self._data.shape

    @property
    def size(self) -> tuple[int, ...]:
        r"""``tuple``: The total number of elements in the array."""
        return self._data.size

    ###############################
    #     Creation operations     #
    ###############################

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
        return self._create_new_batch(self._data.copy(*args, **kwargs))

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
            >>> batch.empty_like()
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
        return self._create_new_batch(np.isneginf(self._data))

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
        return self._create_new_batch(np.isposinf(self._data))

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
        self._check_valid_dims((self, other))
        if isinstance(other, BatchedArray):
            other = other.data
        return self._create_new_batch(np.add(self.data, other * alpha))

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
        self._check_valid_dims((self, other))
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

        self._check_valid_dims((self, other))
        if isinstance(other, BatchedArray):
            other = other.data
        return self._create_new_batch(get_div_rounding_operator(rounding_mode)(self.data, other))

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
        self._check_valid_dims((self, other))
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
        self._check_valid_dims((self, divisor))
        if isinstance(divisor, BatchedArray):
            divisor = divisor.data
        return self._create_new_batch(np.fmod(self.data, divisor))

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
        self._check_valid_dims((self, divisor))
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
        self._check_valid_dims((self, other))
        if isinstance(other, BatchedArray):
            other = other.data
        return self._create_new_batch(self._data * other)

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
        self._check_valid_dims((self, other))
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
        return self._create_new_batch(np.negative(self.data))

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
        self._check_valid_dims((self, other))
        if isinstance(other, BatchedArray):
            other = other.data
        return self._create_new_batch(np.subtract(self.data, other * alpha))

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
        self._check_valid_dims((self, other))
        if isinstance(other, BatchedArray):
            other = other.data
        self._data = np.subtract(self._data.data, other * alpha)

    ###########################################################
    #     Mathematical | advanced arithmetical operations     #
    ###########################################################

    @overload
    def cumsum(self, axis: None, *args, **kwargs) -> ndarray:
        r"""See ``numpy.cumsum`` documentation."""

    @overload
    def cumsum(self, axis: int, *args, **kwargs) -> TBatchedArray:
        r"""See ``numpy.cumsum`` documentation."""

    def cumsum(self, axis: int | None, *args, **kwargs) -> TBatchedArray | ndarray:
        r"""Computes the cumulative sum of elements of the current batch
        in a given axis.

        Args:
        ----
            axis (int): Specifies the axis of the cumulative sum.
            *args: See the documentation of ``numpy.cumsum``
            **kwargs: See the documentation of ``numpy.cumsum``

        Returns:
        -------
            ``BatchedArray``: A batch with the cumulative sum of
                elements of the current batch in a given axis.

        Example usage:

        .. code-block:: pycon

            >>> import numpy
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(numpy.arange(10).reshape(2, 5))
            >>> batch.cumsum(axis=0)
            array([[ 0,  1,  2,  3,  4],
                   [ 5,  7,  9, 11, 13]], batch_dim=0)
        """
        out = np.cumsum(self._data, axis, *args, **kwargs)
        if axis is None:
            return out
        return self._create_new_batch(out)

    def cumsum_(self, axis: int, *args, **kwargs) -> None:
        r"""Computes the cumulative sum of elements of the current batch
        in a given axis.

        Args:
        ----
            axis (int): Specifies the axis of the cumulative sum.
            *args: See the documentation of ``numpy.cumsum``
            **kwargs: See the documentation of ``numpy.cumsum``

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(2, 5))
            >>> batch.cumsum_(axis=0)
            >>> batch
            array([[ 0,  1,  2,  3,  4],
                   [ 5,  7,  9, 11, 13]], batch_dim=0)
        """
        self._data = np.cumsum(self._data, axis, *args, **kwargs)

    def cumsum_along_batch(self, *args, **kwargs) -> TBatchedArray:
        r"""Computes the cumulative sum of elements of the current batch
        in the batch axis.

        Args:
        ----
            *args: See the documentation of ``numpy.cumsum``
            **kwargs: See the documentation of ``numpy.cumsum``

        Returns:
        -------
            ``BatchedArray``: A batch with the cumulative sum of
                elements of the current batch in the batch axis.

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

    def cumsum_along_batch_(self, *args, **kwargs) -> None:
        r"""Computes the cumulative sum of elements of the current batch
        in the batch axis.

        Args:
        ----
            *args: See the documentation of ``numpy.cumsum``
            **kwargs: See the documentation of ``numpy.cumsum``

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
        self.cumsum_(self._batch_dim, *args, **kwargs)

    def logcumsumexp(self, axis: int) -> TBatchedArray:
        r"""Computes the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in a given axis.

        Args:
        ----
            axis (int): Specifies the axis of the cumulative sum.

        Returns:
        -------
            ``BatchedArray``: A batch with the cumulative
                summation of the exponentiation of elements of the
                current batch in a given axis.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(2, 5).astype(float))
            >>> batch.logcumsumexp(axis=1)
            array([[0.        , 1.31326169, 2.40760596, 3.4401897 , 4.4519144 ],
                   [5.        , 6.31326169, 7.40760596, 8.4401897 , 9.4519144 ]], batch_dim=0)
        """
        return self._create_new_batch(np.log(np.cumsum(np.exp(self._data), axis=axis)))

    def logcumsumexp_(self, axis: int) -> None:
        r"""Computes the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in a given axis.

        Args:
        ----
            axis (int): Specifies the axis of the cumulative sum.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(2, 5).astype(float))
            >>> batch.logcumsumexp_(axis=1)
            >>> batch
            array([[0.        , 1.31326169, 2.40760596, 3.4401897 , 4.4519144 ],
                   [5.        , 6.31326169, 7.40760596, 8.4401897 , 9.4519144 ]], batch_dim=0)
        """
        self._data = self.logcumsumexp(axis).data

    def logcumsumexp_along_batch(self) -> TBatchedArray:
        r"""Computes the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in the batch
        axis.

        Returns:
        -------
            ``BatchedArray``: A batch with the cumulative
                summation of the exponentiation of elements of the
                current batch in the batch axis.

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
        axis.

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
        return self.permute_along_axis(permutation, axis=self._batch_dim)

    def permute_along_batch_(self, permutation: Sequence[int] | ndarray) -> None:
        self.permute_along_axis_(permutation, axis=self._batch_dim)

    def permute_along_axis(self, permutation: Sequence[int] | ndarray, axis: int) -> TBatchedArray:
        r"""Permutes the data/batch along a given axis.

        Args:
        ----
            permutation (``Sequence`` or ``numpy.ndarray`` of type int
                and shape ``(axis,)``): Specifies the permutation
                to use on the data. The axis of the permutation
                input should be compatible with the shape of the data.
            axis (int): Specifies the axis where the permutation
                is computed.

        Returns:
        -------
            ``BatchedArray``: A new batch with permuted data.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
            >>> batch.permute_along_axis([2, 1, 3, 0, 4], axis=0)
            array([[4, 5],
                   [2, 3],
                   [6, 7],
                   [0, 1],
                   [8, 9]], batch_dim=0)
        """
        return self._create_new_batch(
            permute_along_axis(self._data, permutation=to_array(permutation), axis=axis)
        )

    def permute_along_axis_(self, permutation: Sequence[int] | ndarray, axis: int) -> None:
        r"""Permutes the data/batch along a given axis.

        Args:
        ----
            permutation (``Sequence`` or ``numpy.ndarray`` of type int
                and shape ``(n,)``): Specifies the permutation
                to use on the data. The axis of the permutation
                input should be compatible with the shape of the data.
            axis (int): Specifies the axis where the permutation
                is computed.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
            >>> batch.permute_along_axis_([2, 1, 3, 0, 4], axis=0)
            >>> batch
            array([[4, 5],
                   [2, 3],
                   [6, 7],
                   [0, 1],
                   [8, 9]], batch_dim=0)
        """
        self._data = self.permute_along_axis(permutation=permutation, axis=axis).data

    def shuffle_along_axis(self, axis: int, generator: RNGType | None = None) -> TBatchedArray:
        r"""Shuffles the data/batch along a given axis.

        Args:
        ----
            axis (int): Specifies the shuffle axis.
            generator (``numpy.random.Generator`` or
                ``torch.Generator`` or ``random.Random`` or ``None``,
                optional): Specifies the pseudorandom number
                generator for sampling or the random seed for the
                random number generator. Default: ``None``

        Returns:
        -------
            ``BatchedArray``:  A new batch with shuffled data
                along a given axis.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
            >>> batch.shuffle_along_axis(axis=0)
            array([[...]], batch_dim=0)
        """
        return self.permute_along_axis(
            to_array(randperm(self._data.shape[axis], generator)), axis=axis
        )

    def shuffle_along_axis_(self, axis: int, generator: RNGType | None = None) -> None:
        r"""Shuffles the data/batch along a given axis.

        Args:
        ----
            axis (int): Specifies the shuffle axis.
            generator (``numpy.random.Generator`` or
                ``torch.Generator`` or ``random.Random`` or ``None``,
                optional): Specifies the pseudorandom number
                generator for sampling or the random seed for the
                random number generator. Default: ``None``

        Returns:
        -------
            ``BatchedArray``:  A new batch with shuffled data
                along a given axis.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.arange(10).reshape(5, 2))
            >>> batch.shuffle_along_axis_(axis=0)
            >>> batch
            array([[...]], batch_dim=0)
        """
        self.permute_along_axis_(to_array(randperm(self._data.shape[axis], generator)), axis=axis)

    def sort(
        self,
        axis: int = -1,
        descending: bool = False,
        stable: bool = False,
    ) -> ValuesIndicesTuple:
        r"""Sorts the elements of the batch along a given axis in
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
                        along the given axis.
                    - The second batch contains the indices that sort
                        the batch along the given axis.

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
        indices = np.argsort(self._data, axis=axis, kind="stable" if stable else "quicksort")
        if descending:
            indices = np.flip(indices, axis=axis)
        return ValuesIndicesTuple(
            values=self._create_new_batch(np.take_along_axis(self._data, indices, axis)),
            indices=self._create_new_batch(indices),
        )

    def sort_along_batch(
        self,
        descending: bool = False,
        stable: bool = False,
    ) -> ValuesIndicesTuple:
        r"""Sorts the elements of the batch along the batch axis in
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
                        along the given axis.
                    - The second batch contains the indices that sort
                        the batch along the given axis.

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
        return self.sort(axis=self._batch_dim, descending=descending, stable=stable)

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

    def concatenate(
        self,
        arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
        axis: int = 0,
    ) -> TBatchedArray:
        r"""Concatenates the data of the batch(es) to the current batch
        along a given axis and creates a new batch.

        Args:
        ----
            arrays (``BatchedArray`` or ``numpy.ndarray`` or
                ``Iterable``): Specifies the batch(es) to concatenate.
            axis (int, optional): Specifies the concatenation axis.
                Default: ``0``

        Returns:
        -------
            ``BatchedArray``: A batch with the concatenated data
                in the given axis.

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
        arrays = list(chain([self], arrays))
        self._check_valid_dims(arrays)
        return self._create_new_batch(
            np.concatenate(
                [array.data if isinstance(array, BatchedArray) else array for array in arrays],
                axis=axis,
            ),
        )

    def concatenate_(
        self,
        arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
        axis: int = 0,
    ) -> None:
        r"""Concatenates the data of the batch(es) to the current batch
        along a given axis and creates a new batch.

        Args:
        ----
            array (``BatchedArray`` or ``numpy.ndarray`` or
                ``Iterable``): Specifies the batch(es) to concatenate.
            axis (int, optional): Specifies the concatenation axis.
                Default: ``0``

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
            >>> batch.concatenate_(BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])))
            >>> batch
            array([[ 0,  1,  2],
                   [ 4,  5,  6],
                   [10, 11, 12],
                   [13, 14, 15]], batch_dim=0)
        """
        self._data = self.concatenate(arrays, axis=axis).data

    def concatenate_along_batch(
        self, arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray]
    ) -> TBatchedArray:
        r"""Concatenates the data of the batch(es) to the current batch
        along the batch axis and creates a new batch.

        Args:
        ----
            arrays (``BatchedArray`` or ``numpy.ndarray`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Returns:
        -------
            ``BatchedArray``: A batch with the concatenated data
                in the batch axis.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
            >>> batch.concatenate_along_batch(BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])))
            array([[ 0,  1,  2],
                   [ 4,  5,  6],
                   [10, 11, 12],
                   [13, 14, 15]], batch_dim=0)
            >>> batch = BatchedArray(np.array([[0, 4], [1, 5], [2, 6]]))
            >>> batch.concatenate_along_batch(
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
        return self.concatenate(arrays, axis=self._batch_dim)

    def concatenate_along_batch_(
        self, arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray]
    ) -> None:
        r"""Concatenates the data of the batch(es) to the current batch
        along the batch axis and creates a new batch.

        Args:
        ----
            arrays (``BatchedArray`` or ``numpy.ndarray`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
            >>> batch.concatenate_along_batch_(BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])))
            >>> batch
            array([[ 0,  1,  2],
                   [ 4,  5,  6],
                   [10, 11, 12],
                   [13, 14, 15]], batch_dim=0)
            >>> batch = BatchedArray(np.array([[0, 4], [1, 5], [2, 6]]))
            >>> batch.concatenate_along_batch_(
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
        self.concatenate_(arrays, axis=self._batch_dim)

    #################
    #     Other     #
    #################

    def summary(self) -> str:
        dims = ", ".join([f"{key}={value}" for key, value in self._get_kwargs().items()])
        return f"{self.__class__.__qualname__}(dtype={self.dtype}, shape={self.shape}, {dims})"

    def _check_valid_dims(self, arrays: Sequence) -> None:
        r"""Checks if the dimensions are valid.

        Args:
        ----
            arrays (``Sequence``): Specifies the sequence of
                arrays/batches to check.
        """
        check_batch_dims(get_batch_dims(arrays))

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
    return arrays[0].concatenate(arrays[1:], axis)


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
