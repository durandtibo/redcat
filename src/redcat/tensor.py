from __future__ import annotations

__all__ = ["BatchedTensor", "check_data_and_dim", "check_batch_dims", "get_batch_dims"]

import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import chain
from typing import Any, TypeVar, Union, overload

import torch
from torch import Tensor

from redcat.base import BaseBatch
from redcat.utils.tensor import permute_along_dim

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchedTensor = TypeVar("TBatchedTensor", bound="BatchedTensor")

HANDLED_FUNCTIONS = {}

IndexType = Union[int, slice, list[int], Tensor, None]


class BatchedTensor(BaseBatch[Tensor]):
    r"""Implements a batched tensor to easily manipulate a batch of
    examples.

    Args:
        data (array_like): Specifies the data for the tensor. It can
            be a torch.Tensor, list, tuple, NumPy ndarray, scalar,
            and other types.
        batch_dim (int, optional): Specifies the batch dimension
            in the ``torch.Tensor`` object. Default: ``0``
        kwargs: Keyword arguments that are passed to
            ``torch.as_tensor``.
    """

    def __init__(self, data: Any, *, batch_dim: int = 0, **kwargs) -> None:
        super().__init__()
        self._data = torch.as_tensor(data, **kwargs)
        check_data_and_dim(self._data, batch_dim)
        self._batch_dim = int(batch_dim)

    def __repr__(self) -> str:
        return repr(self._data)[:-1] + f", batch_dim={self._batch_dim})"

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> TBatchedTensor:
        # print(func, types, args, kwargs)
        kwargs = kwargs or {}
        if handled_func := HANDLED_FUNCTIONS.get(func, None):
            return handled_func(*args, **kwargs)

        batch_dims = get_batch_dims(args, kwargs)
        check_batch_dims(batch_dims)
        args = [a._data if hasattr(a, "_data") else a for a in args]
        return cls(func(*args, **kwargs), batch_dim=batch_dims.pop())

    @property
    def batch_dim(self) -> int:
        r"""int: The batch dimension in the ``torch.Tensor`` object."""
        return self._batch_dim

    @property
    def batch_size(self) -> int:
        return self._data.shape[self._batch_dim]

    @property
    def data(self) -> Tensor:
        r"""``torch.Tensor``: The data in the batch."""
        return self._data

    @property
    def device(self) -> torch.device:
        r"""``torch.device``: The device where the batch data/tensor is."""
        return self._data.device

    @property
    def shape(self) -> torch.Size:
        r"""``torch.Size``: The shape of the tensor."""
        return self._data.shape

    def dim(self) -> int:
        r"""Gets the number of dimensions.

        Returns:
            int: The number of dimensions

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.ones(2, 3)).dim()
            2
        """
        return self._data.dim()

    def ndimension(self) -> int:
        r"""Gets the number of dimensions.

        Returns:
            int: The number of dimensions

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.ones(2, 3)).ndimension()
            2
        """
        return self.dim()

    def numel(self) -> int:
        r"""Gets the total number of elements in the tensor.

        Returns:
            int: The total number of elements

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.ones(2, 3)).numel()
            6
        """
        return self._data.numel()

    #################################
    #     Conversion operations     #
    #################################

    def contiguous(
        self, memory_format: torch.memory_format = torch.contiguous_format
    ) -> TBatchedTensor:
        r"""Creates a batch with a contiguous representation of the data.

        Args:
            memory_format (``torch.memory_format``, optional):
                Specifies the desired memory format.
                Default: ``torch.contiguous_format``

        Returns:
            ``BatchedTensor``: A new batch with a contiguous
                representation of the data.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3)).contiguous()
            >>> batch.is_contiguous()
            True
        """
        return self._create_new_batch(self._data.contiguous(memory_format=memory_format))

    def is_contiguous(self, memory_format: torch.memory_format = torch.contiguous_format) -> bool:
        r"""Indicates if a batch as a contiguous representation of the
        data.

        Args:
            memory_format (``torch.memory_format``, optional):
                Specifies the desired memory format.
                Default: ``torch.contiguous_format``

        Returns:
            bool: ``True`` if the data are stored with a contiguous
                tensor, otherwise ``False``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.ones(2, 3)).is_contiguous()
            True
        """
        return self._data.is_contiguous(memory_format=memory_format)

    def to(self, *args, **kwargs) -> TBatchedTensor:
        r"""Moves and/or casts the data.

        Args:
            *args: See the documentation of ``torch.Tensor.to``
            **kwargs: See the documentation of ``torch.Tensor.to``

        Returns:
            ``BatchedTensor``: A new batch with the data after
                dtype and/or device conversion.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch_cuda = batch.to(device=torch.device('cuda:0'))
            >>> batch_bool = batch.to(dtype=torch.bool)
            >>> batch_bool
            tensor([[True, True, True],
                    [True, True, True]], batch_dim=0)
        """
        return self._create_new_batch(self._data.to(*args, **kwargs))

    ###############################
    #     Creation operations     #
    ###############################

    def clone(self, *args, **kwargs) -> TBatchedTensor:
        r"""Creates a copy of the current batch.

        Args:
            *args: See the documentation of ``torch.Tensor.clone``
            **kwargs: See the documentation of ``torch.Tensor.clone``

        Returns:
            ``BatchedTensor``: A copy of the current batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch_copy = batch.clone()
            >>> batch_copy
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
        """
        return self._create_new_batch(self._data.clone(*args, **kwargs))

    def empty_like(self, *args, **kwargs) -> TBatchedTensor:
        r"""Creates an uninitialized batch, with the same shape as the
        current batch.

        Args:
            *args: See the documentation of ``torch.Tensor.empty_like``
            **kwargs: See the documentation of
                ``torch.Tensor.empty_like``

        Returns:
            ``BatchedTensor``: A uninitialized batch with the same
                shape as the current batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.empty_like(42)
            tensor([[0., 0., 0.],
                    [0., 0., 0.]], batch_dim=0)
        """
        return self._create_new_batch(torch.empty_like(self._data, *args, **kwargs))

    def full_like(self, *args, **kwargs) -> TBatchedTensor:
        r"""Creates a batch filled with a given scalar value, with the
        same shape as the current batch.

        Args:
            *args: See the documentation of ``torch.Tensor.full_like``
            **kwargs: See the documentation of
                ``torch.Tensor.full_like``

        Returns:
            ``BatchedTensor``: A batch filled with the scalar
                value, with the same shape as the current batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.full_like(42)
            tensor([[42., 42., 42.],
                    [42., 42., 42.]], batch_dim=0)
        """
        return self._create_new_batch(torch.full_like(self._data, *args, **kwargs))

    def new_full(
        self,
        fill_value: float | int | bool,
        batch_size: int | None = None,
        **kwargs,
    ) -> TBatchedTensor:
        r"""Creates a batch filled with a scalar value.

        By default, the tensor in the returned batch has the same
        shape, ``torch.dtype`` and ``torch.device`` as the tensor in
        the current batch.

        Args:
            fill_value (float or int or bool): Specifies the number
                to fill the batch with.
            batch_size (int or ``None``): Specifies the batch size.
                If ``None``, the batch size of the current batch is
                used. Default: ``None``.
            **kwargs: See the documentation of
                ``torch.Tensor.new_full``.

        Returns:
            ``BatchedTensor``: A batch filled with the scalar value.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.new_full(42)
            tensor([[42., 42., 42.],
                    [42., 42., 42.]], batch_dim=0)
            >>> batch.new_full(42, batch_size=5)
            tensor([[42., 42., 42.],
                    [42., 42., 42.],
                    [42., 42., 42.],
                    [42., 42., 42.],
                    [42., 42., 42.]], batch_dim=0)
        """
        shape = list(self._data.shape)
        if batch_size is not None:
            shape[self._batch_dim] = batch_size
        kwargs["dtype"] = kwargs.get("dtype", self.dtype)
        kwargs["device"] = kwargs.get("device", self.device)
        return self._create_new_batch(torch.full(size=shape, fill_value=fill_value, **kwargs))

    def new_ones(
        self,
        batch_size: int | None = None,
        **kwargs,
    ) -> BatchedTensor:
        r"""Creates a batch filled with the scalar value ``1``.

        By default, the tensor in the returned batch has the same
        shape, ``torch.dtype`` and ``torch.device`` as the tensor in
        the current batch.

        Args:
            batch_size (int or ``None``): Specifies the batch size.
                If ``None``, the batch size of the current batch is
                used. Default: ``None``.
            **kwargs: See the documentation of
                ``torch.Tensor.new_ones``.

        Returns:
            ``BatchedTensor``: A batch filled with the scalar
                value ``1``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.zeros(2, 3))
            >>> batch.new_ones()
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
            >>> batch.new_ones(batch_size=5)
            tensor([[1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
        """
        shape = list(self._data.shape)
        if batch_size is not None:
            shape[self._batch_dim] = batch_size
        kwargs["dtype"] = kwargs.get("dtype", self.dtype)
        kwargs["device"] = kwargs.get("device", self.device)
        return self._create_new_batch(torch.ones(*shape, **kwargs))

    def new_zeros(
        self,
        batch_size: int | None = None,
        **kwargs,
    ) -> TBatchedTensor:
        r"""Creates a batch filled with the scalar value ``0``.

        By default, the tensor in the returned batch has the same
        shape, ``torch.dtype`` and ``torch.device`` as the tensor
        in the current batch.

        Args:
            batch_size (int or ``None``): Specifies the batch size.
                If ``None``, the batch size of the current batch is
                used. Default: ``None``.
            **kwargs: See the documentation of
                ``torch.Tensor.new_zeros``.

        Returns:
            ``BatchedTensor``: A batch filled with the scalar
                value ``0``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.new_zeros()
            tensor([[0., 0., 0.],
                    [0., 0., 0.]], batch_dim=0)
            >>> batch.new_zeros(batch_size=5)
            tensor([[0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]], batch_dim=0)
        """
        shape = list(self._data.shape)
        if batch_size is not None:
            shape[self._batch_dim] = batch_size
        kwargs["dtype"] = kwargs.get("dtype", self.dtype)
        kwargs["device"] = kwargs.get("device", self.device)
        return self._create_new_batch(torch.zeros(*shape, **kwargs))

    def ones_like(self, *args, **kwargs) -> TBatchedTensor:
        r"""Creates a batch filled with the scalar value ``1``, with the same
        shape as the current batch.

        Args:
            *args: See the documentation of ``torch.Tensor.ones_like``
            **kwargs: See the documentation of
                ``torch.Tensor.ones_like``

        Returns:
            ``BatchedTensor``: A batch filled with the scalar
                value ``1``, with the same shape as the current
                batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.ones_like()
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
        """
        return self._create_new_batch(torch.ones_like(self._data, *args, **kwargs))

    def zeros_like(self, *args, **kwargs) -> TBatchedTensor:
        r"""Creates a batch filled with the scalar value ``0``, with the same
        shape as the current batch.

        Args:
            *args: See the documentation of ``torch.Tensor.zeros_like``
            **kwargs: See the documentation of
                ``torch.Tensor.zeros_like``

        Returns:
            ``BatchedTensor``: A batch filled with the scalar
                value ``0``, with the same shape as the current
                batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.zeros_like()
            tensor([[0., 0., 0.],
                    [0., 0., 0.]], batch_dim=0)
        """
        return self._create_new_batch(torch.zeros_like(self._data, *args, **kwargs))

    #################################
    #     Comparison operations     #
    #################################

    def __eq__(self, other: Any) -> TBatchedTensor:
        return self.eq(other)

    def __ge__(self, other: Any) -> TBatchedTensor:
        return self.ge(other)

    def __gt__(self, other: Any) -> TBatchedTensor:
        return self.gt(other)

    def __le__(self, other: Any) -> TBatchedTensor:
        return self.le(other)

    def __lt__(self, other: Any) -> TBatchedTensor:
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
        return self._data.allclose(other.data, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def eq(self, other: BatchedTensor | Tensor | bool | int | float) -> TBatchedTensor:
        r"""Computes element-wise equality.

        Args:
            other: Specifies the batch to compare.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                equality.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch1 = BatchedTensor(torch.tensor([[1, 3, 4], [0, 2, 2]]))
            >>> batch2 = BatchedTensor(torch.tensor([[5, 3, 2], [0, 1, 2]]))
            >>> batch1.eq(batch2)
            tensor([[False,  True, False],
                    [ True, False,  True]], batch_dim=0)
            >>> batch1.eq(torch.tensor([[5, 3, 2], [0, 1, 2]]))
            tensor([[False,  True, False],
                    [ True, False,  True]], batch_dim=0)
            >>> batch1.eq(2)
            tensor([[False, False, False],
                    [False,  True,  True]], batch_dim=0)
        """
        return torch.eq(self, other)

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self._batch_dim != other.batch_dim:
            return False
        return self._data.equal(other.data)

    def ge(self, other: BatchedTensor | Tensor | bool | int | float) -> TBatchedTensor:
        r"""Computes ``self >= other`` element-wise.

        Args:
            other: Specifies the value to compare
                with.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                comparison.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch1 = BatchedTensor(torch.tensor([[1, 3, 4], [0, 2, 2]]))
            >>> batch2 = BatchedTensor(torch.tensor([[5, 3, 2], [0, 1, 2]]))
            >>> batch1.ge(batch2)
            tensor([[False,  True,  True],
                    [ True,  True,  True]], batch_dim=0)
            >>> batch1.ge(torch.tensor([[5, 3, 2], [0, 1, 2]]))
            tensor([[False,  True,  True],
                    [ True,  True,  True]], batch_dim=0)
            >>> batch1.ge(2)
            tensor([[False,  True,  True],
                    [False,  True,  True]], batch_dim=0)
        """
        return torch.ge(self, other)

    def gt(self, other: BatchedTensor | Tensor | bool | int | float) -> TBatchedTensor:
        r"""Computes ``self > other`` element-wise.

        Args:
            other: Specifies the batch to compare.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                comparison.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch1 = BatchedTensor(torch.tensor([[1, 3, 4], [0, 2, 2]]))
            >>> batch2 = BatchedTensor(torch.tensor([[5, 3, 2], [0, 1, 2]]))
            >>> batch1.gt(batch2)
            tensor([[False, False,  True],
                    [False,  True, False]], batch_dim=0)
            >>> batch1.gt(torch.tensor([[5, 3, 2], [0, 1, 2]]))
            tensor([[False, False,  True],
                    [False,  True, False]], batch_dim=0)
            >>> batch1.gt(2)
            tensor([[False,  True,  True],
                    [False, False, False]], batch_dim=0)
        """
        return torch.gt(self, other)

    def isinf(self) -> TBatchedTensor:
        r"""Indicates if each element of the batch is infinite (positive
        or negative infinity) or not.

        Returns:
            BatchedTensor:  A batch containing a boolean tensor
                that is ``True`` where the current batch is infinite
                and ``False`` elsewhere.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[1.0, 0.0, float('inf')], [-1.0, -2.0, float('-inf')]])
            ... )
            >>> batch.isinf()
            tensor([[False, False, True],
                    [False, False, True]], batch_dim=0)
        """
        return torch.isinf(self)

    def isneginf(self) -> TBatchedTensor:
        r"""Indicates if each element of the batch is negative infinity
        or not.

        Returns:
            BatchedTensor:  A batch containing a boolean tensor
                that is ``True`` where the current batch is negative
                infinity and ``False`` elsewhere.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[1.0, 0.0, float('inf')], [-1.0, -2.0, float('-inf')]])
            ... )
            >>> batch.isneginf()
            tensor([[False, False, False],
                    [False, False,  True]], batch_dim=0)
        """
        return torch.isneginf(self)

    def isposinf(self) -> TBatchedTensor:
        r"""Indicates if each element of the batch is positive infinity
        or not.

        Returns:
            BatchedTensor:  A batch containing a boolean tensor
                that is ``True`` where the current batch is positive
                infinity and ``False`` elsewhere.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[1.0, 0.0, float('inf')], [-1.0, -2.0, float('-inf')]])
            ... )
            >>> batch.isposinf()
            tensor([[False, False,   True],
                    [False, False,  False]], batch_dim=0)
        """
        return torch.isposinf(self)

    def isnan(self) -> TBatchedTensor:
        r"""Indicates if each element in the batch is NaN or not.

        Returns:
            BatchedTensor:  A batch containing a boolean tensor
                that is ``True`` where the current batch is infinite
                and ``False`` elsewhere.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[1.0, 0.0, float('nan')], [float('nan'), -2.0, -1.0]])
            ... )
            >>> batch.isnan()
            tensor([[False, False,  True],
                    [ True, False, False]], batch_dim=0)
        """
        return torch.isnan(self)

    def le(self, other: BatchedTensor | Tensor | bool | int | float) -> TBatchedTensor:
        r"""Computes ``self <= other`` element-wise.

        Args:
            other: Specifies the batch to compare.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                comparison.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch1 = BatchedTensor(torch.tensor([[1, 3, 4], [0, 2, 2]]))
            >>> batch2 = BatchedTensor(torch.tensor([[5, 3, 2], [0, 1, 2]]))
            >>> batch1.le(batch2)
            tensor([[ True,  True, False],
                    [ True, False,  True]], batch_dim=0)
            >>> batch1.le(torch.tensor([[5, 3, 2], [0, 1, 2]]))
            tensor([[ True,  True, False],
                    [ True, False,  True]], batch_dim=0)
            >>> batch1.le(2)
            tensor([[ True, False, False],
                    [ True,  True,  True]], batch_dim=0)
        """
        return torch.le(self, other)

    def lt(self, other: BatchedTensor | Tensor | bool | int | float) -> TBatchedTensor:
        r"""Computes ``self < other`` element-wise.

        Args:
            other: Specifies the batch to compare.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                comparison.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch1 = BatchedTensor(torch.tensor([[1, 3, 4], [0, 2, 2]]))
            >>> batch2 = BatchedTensor(torch.tensor([[5, 3, 2], [0, 1, 2]]))
            >>> batch1.lt(batch2)
            tensor([[ True, False, False],
                    [False, False, False]], batch_dim=0)
            >>> batch1.lt(torch.tensor([[5, 3, 2], [0, 1, 2]]))
            tensor([[ True, False, False],
                    [False, False, False]], batch_dim=0)
            >>> batch1.lt(2)
            tensor([[ True, False, False],
                    [ True, False, False]], batch_dim=0)
        """
        return torch.lt(self, other)

    #################
    #     dtype     #
    #################

    @property
    def dtype(self) -> torch.dtype:
        r"""``torch.dtype``: The data type."""
        return self._data.dtype

    def bool(self) -> TBatchedTensor:
        r"""Converts the current batch to bool data type.

        Returns:
            ``BatchedTensor``: The current batch to bool data type.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.bool().dtype
            torch.bool
        """
        return self._create_new_batch(self._data.bool())

    def double(self) -> TBatchedTensor:
        r"""Converts the current batch to double data type.

        Returns:
            ``BatchedTensor``: The current batch to double data type.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.double().dtype
            torch.float64
        """
        return self._create_new_batch(self._data.double())

    def float(self) -> TBatchedTensor:
        r"""Converts the current batch to float data type.

        Returns:
            ``BatchedTensor``: The current batch to float data type.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.float().dtype
            torch.float32
        """
        return self._create_new_batch(self._data.float())

    def int(self) -> TBatchedTensor:
        r"""Converts the current batch to int data type.

        Returns:
            ``BatchedTensor``: The current batch to int data type.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.int().dtype
            torch.int32
        """
        return self._create_new_batch(self._data.int())

    def long(self) -> TBatchedTensor:
        r"""Converts the current batch to long data type.

        Returns:
            ``BatchedTensor``: The current batch to long data type.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.long().dtype
            torch.int64
        """
        return self._create_new_batch(self._data.long())

    ##################################################
    #     Mathematical | arithmetical operations     #
    ##################################################

    def __add__(self, other: Any) -> TBatchedTensor:
        return self.add(other)

    def __iadd__(self, other: Any) -> TBatchedTensor:
        self.add_(other)
        return self

    def __mul__(self, other: Any) -> TBatchedTensor:
        return self.mul(other)

    def __imul__(self, other: Any) -> TBatchedTensor:
        self.mul_(other)
        return self

    def __neg__(self) -> TBatchedTensor:
        return self.neg()

    def __sub__(self, other: Any) -> TBatchedTensor:
        return self.sub(other)

    def __isub__(self, other: Any) -> TBatchedTensor:
        self.sub_(other)
        return self

    def __truediv__(self, other: Any) -> TBatchedTensor:
        return self.div(other)

    def __itruediv__(self, other: Any) -> TBatchedTensor:
        self.div_(other)
        return self

    def add(
        self,
        other: BatchedTensor | Tensor | int | float,
        alpha: int | float = 1.0,
    ) -> TBatchedTensor:
        r"""Adds the input ``other``, scaled by ``alpha``, to the ``self``
        batch.

        Similar to ``out = self + alpha * other``

        Args:
            other (``BatchedTensor`` or ``torch.Tensor`` or int or
                float): Specifies the other value to add to the
                current batch.
            alpha (int or float, optional): Specifies the scale of the
                batch to add. Default: ``1.0``

        Returns:
            ``BatchedTensor``: A new batch containing the addition of
                the two batches.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> out = batch.add(BatchedTensor(torch.full((2, 3), 2.0)))
            >>> batch
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
            >>> out
            tensor([[3., 3., 3.],
                    [3., 3., 3.]], batch_dim=0)
        """
        return torch.add(self, other, alpha=alpha)

    def add_(
        self,
        other: BatchedTensor | Tensor | int | float,
        alpha: int | float = 1.0,
    ) -> None:
        r"""Adds the input ``other``, scaled by ``alpha``, to the ``self``
        batch.

        Similar to ``self += alpha * other`` (in-place)

        Args:
            other (``BatchedTensor`` or ``torch.Tensor`` or int or
                float): Specifies the other value to add to the
                current batch.
            alpha (int or float, optional): Specifies the scale of the
                batch to add. Default: ``1.0``

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.add_(BatchedTensor(torch.full((2, 3), 2.0)))
            >>> batch
            tensor([[3., 3., 3.],
                    [3., 3., 3.]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, other)))
        self._data.add_(other, alpha=alpha)

    def div(
        self,
        other: BatchedTensor | Tensor | int | float,
        rounding_mode: str | None = None,
    ) -> TBatchedTensor:
        r"""Divides the ``self`` batch by the input ``other`.

        Similar to ``out = self / other``

        Args:
            other (``BatchedTensor`` or ``torch.Tensor`` or int or
                float): Specifies the dividend.
            rounding_mode (str or ``None``, optional): Specifies the
                type of rounding applied to the result.
                - ``None``: true division.
                - ``"trunc"``: rounds the results of the division
                    towards zero.
                - ``"floor"``: floor division.
                Default: ``None``

        Returns:
            ``BatchedTensor``: A new batch containing the division
                of the two batches.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> out = batch.div(BatchedTensor(torch.full((2, 3), 2.0)))
            >>> batch
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
            >>> out
            tensor([[0.5000, 0.5000, 0.5000],
                    [0.5000, 0.5000, 0.5000]], batch_dim=0)
        """
        return torch.div(self, other, rounding_mode=rounding_mode)

    def div_(
        self,
        other: BatchedTensor | Tensor | int | float,
        rounding_mode: str | None = None,
    ) -> None:
        r"""Divides the ``self`` batch by the input ``other`.

        Similar to ``self /= other`` (in-place)

        Args:
            other (``BatchedTensor`` or ``torch.Tensor`` or int or
                float): Specifies the dividend.
            rounding_mode (str or ``None``, optional): Specifies the
                type of rounding applied to the result.
                - ``None``: true division.
                - ``"trunc"``: rounds the results of the division
                    towards zero.
                - ``"floor"``: floor division.
                Default: ``None``

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.div_(BatchedTensor(torch.full((2, 3), 2.0)))
            >>> batch
            tensor([[0.5000, 0.5000, 0.5000],
                    [0.5000, 0.5000, 0.5000]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, other)))
        self._data.div_(other, rounding_mode=rounding_mode)

    def fmod(
        self,
        divisor: BatchedTensor | Tensor | int | float,
    ) -> TBatchedTensor:
        r"""Computes the element-wise remainder of division.

        The current batch is the dividend.

        Args:
            divisor (``BatchedTensor`` or ``torch.Tensor`` or int
                or float): Specifies the divisor.

        Returns:
            ``BatchedTensor``: A new batch containing the
                element-wise remainder of division.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> out = batch.fmod(BatchedTensor(torch.full((2, 3), 2.0)))
            >>> batch
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
            >>> out
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
        """
        return torch.fmod(self, divisor)

    def fmod_(self, divisor: BatchedTensor | Tensor | int | float) -> None:
        r"""Computes the element-wise remainder of division.

        The current batch is the dividend.

        Args:
            divisor (``BatchedTensor`` or ``torch.Tensor`` or int
                or float): Specifies the divisor.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.fmod_(BatchedTensor(torch.full((2, 3), 2.0)))
            >>> batch
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, divisor)))
        self._data.fmod_(divisor)

    def mul(self, other: BatchedTensor | Tensor | int | float) -> TBatchedTensor:
        r"""Multiplies the ``self`` batch by the input ``other`.

        Similar to ``out = self * other``

        Args:
            other (``BatchedTensor`` or ``torch.Tensor`` or int or
                float): Specifies the value to multiply.

        Returns:
            ``BatchedTensor``: A new batch containing the
                multiplication of the two batches.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> out = batch.mul(BatchedTensor(torch.full((2, 3), 2.0)))
            >>> batch
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
            >>> out
            tensor([[2., 2., 2.],
                    [2., 2., 2.]], batch_dim=0)
        """
        return torch.mul(self, other)

    def mul_(self, other: BatchedTensor | Tensor | int | float) -> None:
        r"""Multiplies the ``self`` batch by the input ``other`.

        Similar to ``self *= other`` (in-place)

        Args:
            other (``BatchedTensor`` or ``torch.Tensor`` or int or
                float): Specifies the value to multiply.

        Returns:
            ``BatchedTensor``: A new batch containing the
                multiplication of the two batches.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.mul_(BatchedTensor(torch.full((2, 3), 2.0)))
            >>> batch
            tensor([[2., 2., 2.],
                    [2., 2., 2.]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, other)))
        self._data.mul_(other)

    def neg(self) -> TBatchedTensor:
        r"""Returns a new batch with the negative of the elements.

        Returns:
            ``BatchedTensor``: A new batch with the negative of
                the elements.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> out = batch.neg()
            >>> batch
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
            >>> out
            tensor([[-1., -1., -1.],
                    [-1., -1., -1.]], batch_dim=0)
        """
        return torch.neg(self)

    def sub(
        self,
        other: BatchedTensor | Tensor | int | float,
        alpha: int | float = 1,
    ) -> TBatchedTensor:
        r"""Subtracts the input ``other``, scaled by ``alpha``, to the ``self``
        batch.

        Similar to ``out = self - alpha * other``

        Args:
            other (``BatchedTensor`` or ``torch.Tensor`` or int or
                float): Specifies the value to subtract.
            alpha (int or float, optional): Specifies the scale of the
                batch to substract. Default: ``1``

        Returns:
            ``BatchedTensor``: A new batch containing the diffence of
                the two batches.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> out = batch.sub(BatchedTensor(torch.full((2, 3), 2.0)))
            >>> batch
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
            >>> out
            tensor([[-1., -1., -1.],
                    [-1., -1., -1.]], batch_dim=0)
        """
        return torch.sub(self, other, alpha=alpha)

    def sub_(
        self,
        other: BatchedTensor | Tensor | int | float,
        alpha: int | float = 1,
    ) -> None:
        r"""Subtracts the input ``other``, scaled by ``alpha``, to the ``self``
        batch.

        Similar to ``self -= alpha * other`` (in-place)

        Args:
            other (``BatchedTensor`` or ``torch.Tensor`` or int or
                float): Specifies the value to subtract.
            alpha (int or float, optional): Specifies the scale of the
                batch to substract. Default: ``1``

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.sub_(BatchedTensor(torch.full((2, 3), 2.0)))
            >>> batch
            tensor([[-1., -1., -1.],
                    [-1., -1., -1.]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, other)))
        self._data.sub_(other, alpha=alpha)

    ###########################################################
    #     Mathematical | advanced arithmetical operations     #
    ###########################################################

    def cumsum(self, dim: int, **kwargs) -> TBatchedTensor:
        r"""Computes the cumulative sum of elements of the current batch
        in a given dimension.

        Args:
            dim (int): Specifies the dimension of the cumulative sum.
            **kwargs: see ``torch.cumsum`` documentation

        Returns:
            ``BatchedTensor``: A batch with the cumulative sum of
                elements of the current batch in a given dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(2, 5)).cumsum(dim=1)
            tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  7,  9, 11, 13]], batch_dim=0)
        """
        return torch.cumsum(self, dim=dim, **kwargs)

    def cumsum_(self, dim: int, **kwargs) -> None:
        r"""Computes the cumulative sum of elements of the current batch
        in a given dimension.

        Args:
            dim (int): Specifies the dimension of the cumulative sum.
            **kwargs: see ``torch.cumsum_`` documentation

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.cumsum_(dim=1)
            >>> batch
            tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  7,  9, 11, 13]], batch_dim=0)
        """
        self._data.cumsum_(dim=dim, **kwargs)

    def cumsum_along_batch(self, **kwargs) -> TBatchedTensor:
        r"""Computes the cumulative sum of elements of the current batch
        in the batch dimension.

        Args:
            **kwargs: see ``torch.cumsum`` documentation

        Returns:
            ``BatchedTensor``: A batch with the cumulative sum of
                elements of the current batch in the batch dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(2, 5)).cumsum_along_batch()
            tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  7,  9, 11, 13]], batch_dim=0)
        """
        return self.cumsum(self._batch_dim, **kwargs)

    def cumsum_along_batch_(self) -> None:
        r"""Computes the cumulative sum of elements of the current batch
        in the batch dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.cumsum_along_batch_()
            >>> batch
            tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  7,  9, 11, 13]], batch_dim=0)
        """
        self.cumsum_(self._batch_dim)

    def logcumsumexp(self, dim: int) -> TBatchedTensor:
        r"""Computes the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in a given
        dimension.

        Args:
            dim (int): Specifies the dimension of the cumulative sum.

        Returns:
            ``BatchedTensor``: A batch with the cumulative
                summation of the exponentiation of elements of the
                current batch in a given dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(2, 5).float()).logcumsumexp(dim=1)
            tensor([[0.0000, 1.3133, 2.4076, 3.4402, 4.4519],
                    [5.0000, 6.3133, 7.4076, 8.4402, 9.4519]], batch_dim=0)
        """
        return torch.logcumsumexp(self, dim=dim)

    def logcumsumexp_(self, dim: int) -> None:
        r"""Computes the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in a given
        dimension.

        Args:
            dim (int): Specifies the dimension of the cumulative sum.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5).float())
            >>> batch.logcumsumexp_(dim=1)
            >>> batch
            tensor([[0.0000, 1.3133, 2.4076, 3.4402, 4.4519],
                    [5.0000, 6.3133, 7.4076, 8.4402, 9.4519]], batch_dim=0)
        """
        self._data = self._data.logcumsumexp(dim=dim)

    def logcumsumexp_along_batch(self) -> TBatchedTensor:
        r"""Computes the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in the batch
        dimension.

        Returns:
            ``BatchedTensor``: A batch with the cumulative
                summation of the exponentiation of elements of the
                current batch in the batch dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(5, 2).float()).logcumsumexp_along_batch()
            tensor([[0.0000, 1.0000],
                    [2.1269, 3.1269],
                    [4.1429, 5.1429],
                    [6.1451, 7.1451],
                    [8.1454, 9.1454]], batch_dim=0)
        """
        return self.logcumsumexp(self._batch_dim)

    def logcumsumexp_along_batch_(self) -> None:
        r"""Computes the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in the batch
        dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2).float())
            >>> batch.logcumsumexp_along_batch_()
            >>> batch
            tensor([[0.0000, 1.0000],
                    [2.1269, 3.1269],
                    [4.1429, 5.1429],
                    [6.1451, 7.1451],
                    [8.1454, 9.1454]], batch_dim=0)
        """
        self.logcumsumexp_(self._batch_dim)

    def permute_along_batch(self, permutation: Sequence[int] | Tensor) -> TBatchedTensor:
        return self.permute_along_dim(permutation, dim=self._batch_dim)

    def permute_along_batch_(self, permutation: Sequence[int] | Tensor) -> None:
        self.permute_along_dim_(permutation, dim=self._batch_dim)

    def permute_along_dim(self, permutation: Sequence[int] | Tensor, dim: int) -> TBatchedTensor:
        r"""Permutes the data/batch along a given dimension.

        Args:
            permutation (sequence or ``torch.Tensor`` of type long
                and shape ``(dimension,)``): Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.
            dim (int): Specifies the dimension where the permutation
                is computed.

        Returns:
            ``BatchedTensor``: A new batch with permuted data.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(5, 2)).permute_along_dim([2, 1, 3, 0, 4], dim=0)
            tensor([[4, 5],
                    [2, 3],
                    [6, 7],
                    [0, 1],
                    [8, 9]], batch_dim=0)
        """
        if not torch.is_tensor(permutation):
            permutation = torch.as_tensor(permutation)
        return self._create_new_batch(
            permute_along_dim(tensor=self._data, permutation=permutation, dim=dim)
        )

    def permute_along_dim_(self, permutation: Sequence[int] | Tensor, dim: int) -> None:
        r"""Permutes the data/batch along a given dimension.

        Args:
            permutation (sequence or ``torch.Tensor`` of type long
                and shape ``(dimension,)``): Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.
            dim (int): Specifies the dimension where the permutation
                is computed.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.permute_along_dim_([2, 1, 3, 0, 4], dim=0)
            >>> batch
            tensor([[4, 5],
                    [2, 3],
                    [6, 7],
                    [0, 1],
                    [8, 9]], batch_dim=0)
        """
        if not torch.is_tensor(permutation):
            permutation = torch.as_tensor(permutation)
        self._data = permute_along_dim(tensor=self._data, permutation=permutation, dim=dim)

    def shuffle_along_dim(
        self, dim: int, generator: torch.Generator | None = None
    ) -> TBatchedTensor:
        r"""Shuffles the data/batch along a given dimension.

        Args:
            generator (``torch.Generator`` or ``None``, optional):
                Specifies an optional random generator.
                Default: ``None``

        Returns:
            ``BatchedTensor``:  A new batch with shuffled data
                along a given dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(5, 2)).shuffle_along_dim(dim=0)
            tensor([[4, 5],
                    [2, 3],
                    [6, 7],
                    [0, 1],
                    [8, 9]], batch_dim=0)
        """
        return self.permute_along_dim(
            torch.randperm(self._data.shape[dim], generator=generator), dim=dim
        )

    def shuffle_along_dim_(self, dim: int, generator: torch.Generator | None = None) -> None:
        r"""Shuffles the data/batch along a given dimension.

        Args:
            generator (``torch.Generator`` or ``None``, optional):
                Specifies an optional random generator.
                Default: ``None``

        Returns:
            ``BatchedTensor``:  A new batch with shuffled data
                along a given dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.shuffle_along_dim_(dim=0)
            >>> batch
            tensor([[4, 5],
                    [2, 3],
                    [6, 7],
                    [0, 1],
                    [8, 9]], batch_dim=0)
        """
        self.permute_along_dim_(torch.randperm(self._data.shape[dim], generator=generator), dim=dim)

    def sort(
        self,
        dim: int = -1,
        descending: bool = False,
        stable: bool = False,
    ) -> torch.return_types.sort:
        r"""Sorts the elements of the batch along a given dimension in
        monotonic order by value.

        Args:
            descending (bool, optional): Controls the sorting order.
                If ``True``, the elements are sorted in descending
                order by value. Default: ``False``
            stable (bool, optional): Makes the sorting routine stable,
                which guarantees that the order of equivalent elements
                is preserved. Default: ``False``

        Returns:
            (``BatchedTensor``, ``BatchedTensor``): A tuple
                two values:
                    - The first batch contains the batch values sorted
                        along the given dimension.
                    - The second batch contains the indices that sort
                        the batch along the given dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.rand(2, 5)).sort()
            torch.return_types.sort(
            values=tensor([[0.0239, 0.1395, 0.1742, 0.2742, 0.3203],
                    [0.1096, 0.1745, 0.5360, 0.8954, 0.9036]], batch_dim=0),
            indices=tensor([[0, 1, 2, 3, 4],
                    [2, 0, 3, 4, 1]], batch_dim=0))
        """
        return torch.sort(self, dim=dim, descending=descending, stable=stable)

    def sort_along_batch(
        self,
        descending: bool = False,
        stable: bool = False,
    ) -> torch.return_types.sort:
        r"""Sorts the elements of the batch along the batch dimension in
        monotonic order by value.

        Args:
            descending (bool, optional): Controls the sorting order.
                If ``True``, the elements are sorted in descending
                order by value. Default: ``False``
            stable (bool, optional): Makes the sorting routine stable,
                which guarantees that the order of equivalent elements
                is preserved. Default: ``False``

        Returns:
            (``BatchedTensor``, ``BatchedTensor``): A tuple
                two values:
                    - The first batch contains the batch values sorted
                        along the given dimension.
                    - The second batch contains the indices that sort
                        the batch along the given dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.rand(2, 5)).sort_along_batch()
            torch.return_types.sort(
            values=tensor([[0.0091, 0.5615, 0.5453, 0.1468, 0.5192],
                    [0.4122, 0.8932, 0.8783, 0.6494, 0.7763]], batch_dim=0),
            indices=tensor([[0, 0, 1, 1, 0],
                    [1, 1, 0, 0, 1]], batch_dim=0))
        """
        return self.sort(dim=self._batch_dim, descending=descending, stable=stable)

    ################################################
    #     Mathematical | point-wise operations     #
    ################################################

    def abs(self) -> TBatchedTensor:
        r"""Computes the absolute value of each element.

        Return:
            ``BatchedTensor``: A batch with the absolute value of
                each element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0]]))
            >>> batch.abs()
            tensor([[2., 0., 2.],
                    [1., 1., 3.]], batch_dim=0)
        """
        return torch.abs(self)

    def abs_(self) -> None:
        r"""Computes the absolute value of each element.

        In-place version of ``abs()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0]]))
            >>> batch.abs_()
            >>> batch
            tensor([[2., 0., 2.],
                    [1., 1., 3.]], batch_dim=0)
        """
        self._data.abs_()

    def clamp(
        self,
        min: int | float | None = None,  # noqa: A002
        max: int | float | None = None,  # noqa: A002
    ) -> TBatchedTensor:
        r"""Clamps all elements in ``self`` into the range ``[min, max]``.

        Note: ``min`` and ``max`` cannot be both ``None``.

        Args:
            min (int, float or ``None``, optional): Specifies
                the lower bound. If ``min`` is ``None``,
                there is no lower bound. Default: ``None``
            max (int, float or ``None``, optional): Specifies
                the upper bound. If ``max`` is ``None``,
                there is no upper bound. Default: ``None``

        Returns:
            ``BatchedTensor``: A batch with clamped values.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.clamp(min=2, max=5)
            tensor([[2, 2, 2, 3, 4],
                    [5, 5, 5, 5, 5]], batch_dim=0)
            >>> batch.clamp(min=2)
            tensor([[2, 2, 2, 3, 4],
                    [5, 6, 7, 8, 9]])
            >>> batch.clamp(max=7)
            tensor([[0, 1, 2, 3, 4],
                    [5, 6, 7, 7, 7]], batch_dim=0)
        """
        return torch.clamp(self, min=min, max=max)

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
            min (int, float or ``None``, optional): Specifies
                the lower bound.  If ``min`` is ``None``,
                there is no lower bound. Default: ``None``
            max (int, float or ``None``, optional): Specifies
                the upper bound. If ``max`` is ``None``,
                there is no upper bound. Default: ``None``

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.clamp_(min=2, max=5)
            >>> batch
            tensor([[2, 2, 2, 3, 4],
                    [5, 5, 5, 5, 5]], batch_dim=0)
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.clamp_(min=2)
            >>> batch
            tensor([[2, 2, 2, 3, 4],
                    [5, 6, 7, 8, 9]], batch_dim=0)
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.clamp_(max=7)
            >>> batch
            tensor([[0, 1, 2, 3, 4],
                    [5, 6, 7, 7, 7]], batch_dim=0)
        """
        self._data.clamp_(min=min, max=max)

    def exp(self) -> TBatchedTensor:
        r"""Computes the exponential of the elements.

        Return:
            ``BatchedTensor``: A batch with the exponential of the
                elements of the current batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
            >>> batch.exp()
            tensor([[  2.7183,   7.3891,  20.0855],
                    [ 54.5981, 148.4132, 403.4288]], batch_dim=0)
        """
        return torch.exp(self)

    def exp_(self) -> None:
        r"""Computes the exponential of the elements.

        In-place version of ``exp()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
            >>> batch.exp_()
            >>> batch
            tensor([[  2.7183,   7.3891,  20.0855],
                    [ 54.5981, 148.4132, 403.4288]], batch_dim=0)
        """
        self._data.exp_()

    def log(self) -> BatchedTensor:
        r"""Computes the natural logarithm of the elements.

        Return:
            ``BatchedTensor``: A batch with the natural
                logarithm of the elements of the current batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]])
            ... )
            >>> batch.log()
            tensor([[  -inf, 0.0000, 0.6931, 1.0986, 1.3863],
                    [1.6094, 1.7918, 1.9459, 2.0794, 2.1972]], batch_dim=0)
        """
        return torch.log(self)

    def log_(self) -> None:
        r"""Computes the natural logarithm of the elements.

        In-place version of ``log()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]])
            ... )
            >>> batch.log_()
            >>> batch
            tensor([[  -inf, 0.0000, 0.6931, 1.0986, 1.3863],
                    [1.6094, 1.7918, 1.9459, 2.0794, 2.1972]], batch_dim=0)
        """
        self._data.log_()

    def log1p(self) -> BatchedTensor:
        r"""Computes the natural logarithm of ``self + 1``.

        Return:
            ``BatchedTensor``: A batch with the natural
                logarithm of ``self + 1``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]))
            >>> batch.log1p()
            tensor([[0.0000, 0.6931, 1.0986, 1.3863, 1.6094],
                    [1.7918, 1.9459, 2.0794, 2.1972, 2.3026]], batch_dim=0)
        """
        return torch.log1p(self)

    def log1p_(self) -> None:
        r"""Computes the natural logarithm of ``self + 1``.

        In-place version of ``log1p()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]])
            ... )
            >>> batch.log1p_()
            >>> batch
            tensor([[0.0000, 0.6931, 1.0986, 1.3863, 1.6094],
                    [1.7918, 1.9459, 2.0794, 2.1972, 2.3026]], batch_dim=0)
        """
        self._data.log1p_()

    def maximum(self, other: BatchedTensor | Tensor) -> TBatchedTensor:
        r"""Computes the element-wise maximum of ``self`` and ``other``.

        Args:
            other (``BatchedTensor`` or ``torch.Tensor``): Specifies
                a batch.

        Returns:
            ``BatchedTensor``: The batch with the element-wise
                maximum of ``self`` and ``other``

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(6).view(2, 3))
            >>> batch.maximum(BatchedTensor(torch.tensor([[1, 0, 2], [4, 5, 3]])))
            tensor([[1, 1, 2],
                    [4, 5, 5]], batch_dim=0)
        """
        return torch.maximum(self, other)

    def minimum(self, other: BatchedTensor | Tensor) -> TBatchedTensor:
        r"""Computes the element-wise minimum of ``self`` and ``other``.

        Args:
            other (``BatchedTensor`` or ``torch.Tensor``): Specifies
                a batch.

        Returns:
            ``BatchedTensor``: The batch with the element-wise
                minimum of ``self`` and ``other``

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(6).view(2, 3))
            >>> batch.minimum(BatchedTensor(torch.tensor([[1, 0, 2], [4, 5, 3]])))
            tensor([[0, 0, 2],
                    [3, 4, 3]], batch_dim=0)
        """
        return torch.minimum(self, other)

    def pow(self, exponent: int | float | BatchedTensor) -> TBatchedTensor:
        r"""Computes the power of each element with the given exponent.

        Args:
            exponent (int or float or ``BatchedTensor``): Specifies
                the exponent value. ``exponent`` can be either a single
                numeric number or a ``BatchedTensor`` with the same
                number of elements.

        Return:
            ``BatchedTensor``: A batch with the power of each
                element with the given exponent.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
            >>> batch.pow(2)
            tensor([[ 0.,  1.,  4.],
                    [ 9., 16., 25.]], batch_dim=0)
        """
        return torch.pow(self, exponent)

    def pow_(self, exponent: int | float | BatchedTensor) -> None:
        r"""Computes the power of each element with the given exponent.

        In-place version of ``pow(exponent)``.

        Args:
            exponent (int or float or ``BatchedTensor``): Specifies
                the exponent value. ``exponent`` can be either a
                single numeric number or a ``BatchedTensor``
                with the same number of elements.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
            >>> batch.pow_(2)
            >>> batch
            tensor([[ 0.,  1.,  4.],
                    [ 9., 16., 25.]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, exponent)))
        self._data.pow_(exponent)

    def rsqrt(self) -> TBatchedTensor:
        r"""Computes the reciprocal of the square-root of each element.

        Return:
            ``BatchedTensor``: A batch with the square-root of
                each element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.tensor([[1.0, 4.0], [16.0, 25.0]])).rsqrt()
            tensor([[1.0000, 0.5000],
                    [0.2500, 0.2000]], batch_dim=0)
        """
        return torch.rsqrt(self)

    def rsqrt_(self) -> None:
        r"""Computes the reciprocal of the square-root of each element.

        In-place version of ``rsqrt()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[1.0, 4.0], [16.0, 25.0]]))
            >>> batch.rsqrt_()
            >>> batch
            tensor([[1.0000, 0.5000],
                    [0.2500, 0.2000]], batch_dim=0)
        """
        self._data.rsqrt_()

    def sqrt(self) -> TBatchedTensor:
        r"""Computes the square-root of each element.

        Return:
            ``BatchedTensor``: A batch with the square-root of
                each element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]))
            >>> batch.sqrt()
            tensor([[0., 1., 2.],
                    [3., 4., 5.]], batch_dim=0)
        """
        return torch.sqrt(self)

    def sqrt_(self) -> None:
        r"""Computes the square-root of each element.

        In-place version of ``sqrt()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]))
            >>> batch.sqrt_()
            >>> batch
            tensor([[0., 1., 2.],
                    [3., 4., 5.]], batch_dim=0)
        """
        self._data.sqrt_()

    ################################
    #     Reduction operations     #
    ################################

    def max(self, *args, **kwargs) -> Tensor | torch.return_types.max:
        r"""Computes the maximum of all elements.

        Args:
            *args: See the documentation of ``torch.Tensor.max``
            **kwargs: See the documentation of ``torch.Tensor.max``

        Returns:
            ``torch.Tensor`` or ``torch.return_types.max``:
                The maximum of all elements.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(2, 5)).max()
            tensor(9)
            >>> BatchedTensor(torch.arange(10).view(2, 5)).max(dim=1)
            torch.return_types.max(
            values=tensor([4, 9]),
            indices=tensor([4, 4]))
            >>> BatchedTensor(torch.arange(10).view(2, 5)).max(dim=1, keepdim=True)
            torch.return_types.max(
            values=tensor([[4], [9]]),
            indices=tensor([[4], [4]]))
        """
        return torch.max(self, *args, **kwargs)

    def max_along_batch(self, keepdim: bool = False) -> torch.return_types.max:
        r"""Computes the maximum values along the batch dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the batch dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.return_types.max``: A batch with
                the maximum values along the batch dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(
            ...     torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
            ... ).max_along_batch()
            torch.return_types.max(
            values=tensor([4, 9]),
            indices=tensor([4, 4]))
            >>> BatchedTensor(
            ...     torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
            ... ).max_along_batch(keepdim=True)
            torch.return_types.max(
            values=tensor([[4], [9]]),
            indices=tensor([[4], [4]]))
        """
        return self.max(dim=self._batch_dim, keepdim=keepdim)

    def mean(self, *args, **kwargs) -> Tensor:
        r"""Computes the mean of all elements.

        Args:
            *args: See the documentation of ``torch.Tensor.mean``
            **kwargs: See the documentation of ``torch.Tensor.mean``

        Returns:
            ``torch.Tensor``: The mean of all elements.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(2, 5).float()).mean()
            tensor(4.5)
            >>> BatchedTensor(torch.arange(10).view(2, 5).float()).mean(dim=1)
            tensor([2.0, 7.0])
            >>> BatchedTensor(torch.arange(10).view(2, 5).float()).mean(dim=1, keepdim=True)
            tensor([[2.0], [7.0]])
        """
        return torch.mean(self, *args, **kwargs)

    def mean_along_batch(self, keepdim: bool = False) -> Tensor:
        r"""Computes the mean values along the batch dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the batch dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.Tensor``: A batch with
                the mean values along the batch dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(5, 2).float()).mean_along_batch()
            tensor([4.0, 5.0])
            >>> BatchedTensor(torch.arange(10).view(5, 2).float()).mean_along_batch(keepdim=True)
            tensor([[4.0], [5.0]])
        """
        return self.mean(dim=self._batch_dim, keepdim=keepdim)

    def median(self, *args, **kwargs) -> Tensor | torch.return_types.median:
        r"""Computes the median of all elements.

        Args:
            *args: See the documentation of ``torch.Tensor.median``
            **kwargs: See the documentation of ``torch.Tensor.median``

        Returns:
            ``torch.Tensor`` or ``torch.return_types.median``:
                The median of all elements or per dimension.
                The first tensor will be populated with the median
                values and the second tensor, which must have dtype
                long, with their indices in the dimension dim of input.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(2, 5)).median()
            tensor(4)
            >>> BatchedTensor(torch.arange(10).view(2, 5)).median(dim=1)
            torch.return_types.median(
            values=tensor([2, 7]),
            indices=tensor([2, 2]))
            >>> BatchedTensor(torch.arange(10).view(2, 5)).median(dim=1, keepdim=True)
            torch.return_types.median(
            values=tensor([[2], [7]]),
            indices=tensor([[2], [2]]))
        """
        return torch.median(self, *args, **kwargs)

    def median_along_batch(self, keepdim: bool = False) -> torch.return_types.median:
        r"""Computes the median values along the batch dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.return_types.median``:  The first tensor will
                be populated with the median values and the second
                tensor, which must have dtype long, with their indices
                in the batch dimension of input.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(
            ...     torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
            ... ).median_along_batch()
            torch.return_types.median(
            values=tensor([2, 7]),
            indices=tensor([2, 2]))
        """
        return self.median(dim=self._batch_dim, keepdim=keepdim)

    def min(self, *args, **kwargs) -> Tensor | torch.return_types.min:
        r"""Computes the minimum of all elements.

        Args:
            *args: See the documentation of ``torch.Tensor.min``
            **kwargs: See the documentation of ``torch.Tensor.min``

        Returns:
            ``torch.Tensor`` or ``torch.return_types.min``:
                The minimum of all elements.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(2, 5)).min()
            tensor(0)
            >>> BatchedTensor(torch.arange(10).view(2, 5)).min(dim=1)
            torch.return_types.min(
            values=tensor([0, 5]),
            indices=tensor([0, 0]))
            >>> BatchedTensor(torch.arange(10).view(2, 5)).min(dim=1, keepdim=True)
            torch.return_types.min(
            values=tensor([[0], [5]]),
            indices=tensor([[0], [0]]))
        """
        return torch.min(self, *args, **kwargs)

    def min_along_batch(self, keepdim: bool = False) -> torch.return_types.min:
        r"""Computes the minimum values along the batch dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the batch dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.return_types.min``: A batch with
                the minimum values along the batch dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(
            ...     torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
            ... ).min_along_batch()
            torch.return_types.min(
            values=tensor([0, 5]),
            indices=tensor([0, 0]))
            >>> BatchedTensor(
            ...     torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
            ... ).min_along_batch(keepdim=True)
            torch.return_types.min(
            values=tensor([[0], [5]]),
            indices=tensor([[0], [0]]))
        """
        return self.min(dim=self._batch_dim, keepdim=keepdim)

    def nanmean(self, *args, **kwargs) -> Tensor:
        r"""Computes the mean of all elements.

        Args:
            *args: See the documentation of ``torch.Tensor.nanmean``
            **kwargs: See the documentation of ``torch.Tensor.nanmean``

        Returns:
            ``torch.Tensor``: The mean of all elements.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(
            ...     torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])
            ... ).nanmean()
            tensor(4.0)
            >>> BatchedTensor(
            ...     torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])
            ... ).nanmean(dim=1)
            tensor([2.0, 6.5])
            >>> BatchedTensor(
            ...     torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])
            ... ).nanmean(dim=1, keepdim=True)
            tensor([[2.0], [6.5]])
        """
        return torch.nanmean(self, *args, **kwargs)

    def nanmean_along_batch(self, keepdim: bool = False) -> Tensor:
        r"""Computes the mean values along the batch dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the batch dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.Tensor``: A batch with
                the mean values along the batch dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(
            ...     torch.tensor([[0., 5.], [1., 6.], [2., 7.], [3., 8.], [4., float("nan")]])
            ... ).nanmean_along_batch()
            tensor([2.0, 6.5])
            >>> BatchedTensor(
            ...     torch.tensor([[0., 5.], [1., 6.], [2., 7.], [3., 8.], [4., float("nan")]])
            ... ).nanmean_along_batch(keepdim=True)
            tensor([[2.0, 6.5]])
        """
        return self.nanmean(dim=self._batch_dim, keepdim=keepdim)

    def nanmedian(self, *args, **kwargs) -> Tensor | torch.return_types.nanmedian:
        r"""Computes the median of all elements.

        Args:
            *args: See the documentation of ``torch.Tensor.nanmedian``
            **kwargs: See the documentation of ``torch.Tensor.nanmedian``

        Returns:
            ``torch.Tensor`` or ``torch.return_types.nanmedian``:
                The median of all elements or per dimension.
                The first tensor will be populated with the median
                values and the second tensor, which must have dtype
                long, with their indices in the dimension dim of input.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(
            ...     torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])
            ... ).nanmedian()
            tensor(4.0)
            >>> BatchedTensor(
            ...     torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])
            ... ).nanmedian(dim=1)
            torch.return_types.nanmedian(
            values=tensor([2., 6.]),
            indices=tensor([2, 1]))
            >>> BatchedTensor(
            ...     torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])
            ... ).nanmedian(dim=1, keepdim=True)
            torch.return_types.nanmedian(
            values=tensor([[2.], [6.]]),
            indices=tensor([[2], [1]]))
        """
        return torch.nanmedian(self, *args, **kwargs)

    def nanmedian_along_batch(self, keepdim: bool = False) -> torch.return_types.nanmedian:
        r"""Computes the median values along the batch dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.return_types.nanmedian``:  The first tensor will
                be populated with the median values and the second
                tensor, which must have dtype long, with their indices
                in the batch dimension of input.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(
            ...     torch.tensor([[0., 5.], [1., 6.], [2., 7.], [3., 8.], [4., float("nan")]])
            ... ).nanmedian_along_batch()
            torch.return_types.nanmedian(
            values=tensor([2., 6.]),
            indices=tensor([2, 1]))
        """
        return self.nanmedian(dim=self._batch_dim, keepdim=keepdim)

    def nansum(self, *args, **kwargs) -> Tensor:
        r"""Computes the sum of all elements.

        Args:
            *args: See the documentation of ``torch.Tensor.nansum``
            **kwargs: See the documentation of ``torch.Tensor.nansum``

        Returns:
            ``torch.Tensor``: The sum of all elements.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(
            ...     torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])
            ... ).nansum()
            tensor(36.)
            >>> BatchedTensor(
            ...     torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])
            ... ).nansum(dim=1)
            tensor([10., 26.])
            >>> BatchedTensor(
            ...     torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])
            ... ).nansum(dim=1, keepdim=True)
            tensor([[10.], [26.]])
        """
        return torch.nansum(self, *args, **kwargs)

    def nansum_along_batch(self, keepdim: bool = False) -> Tensor:
        r"""Computes the sum values along the batch dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.Tensor``: A tensor with the sum values along the
                batch dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(
            ...     torch.tensor([[0., 5.], [1., 6.], [2., 7.], [3., 8.], [4., float("nan")]])
            ... ).nansum_along_batch()
            tensor([20., 26.])
        """
        return self.nansum(dim=self._batch_dim, keepdim=keepdim)

    def prod(self, *args, **kwargs) -> Tensor:
        r"""Computes the product of all elements.

        Args:
            *args: See the documentation of ``torch.Tensor.nansum``
            **kwargs: See the documentation of ``torch.Tensor.nansum``

        Returns:
            ``torch.Tensor``: The product of all elements.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(
            ...     torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]])
            ... ).prod()
            tensor(362880)
            >>> BatchedTensor(
            ...     torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]])
            ... ).prod(dim=1)
            tensor([ 120, 3024])
            >>> BatchedTensor(
            ...     torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]])
            ... ).prod(dim=1, keepdim=True)
            tensor([[ 120], [3024]])
        """
        return torch.prod(self, *args, **kwargs)

    def prod_along_batch(self, keepdim: bool = False) -> Tensor:
        r"""Computes the product values along the batch dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the batch dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.Tensor``: A batch with
                the product values along the batch dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(
            ...     torch.tensor([[1, 6], [2, 7], [3, 8], [4, 9], [5, 1]])
            ... ).prod_along_batch()
            tensor([ 120, 3024])
            >>> BatchedTensor(
            ...     torch.tensor([[1, 6], [2, 7], [3, 8], [4, 9], [5, 1]])
            ... ).prod_along_batch(keepdim=True)
            tensor([[ 120, 3024]])
        """
        return self.prod(dim=self._batch_dim, keepdim=keepdim)

    def sum(self, *args, **kwargs) -> Tensor:
        r"""Computes the sum of all elements.

        Args:
            *args: See the documentation of ``torch.Tensor.sum``
            **kwargs: See the documentation of ``torch.Tensor.sum``

        Returns:
            ``torch.Tensor``: The sum of all elements.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(2, 5)).sum()
            tensor(45)
            >>> BatchedTensor(torch.arange(10).view(2, 5)).sum(dim=1)
            tensor([10, 35])
            >>> BatchedTensor(torch.arange(10).view(2, 5)).sum(dim=1, keepdim=True)
            tensor([[10], [35]])
        """
        return torch.sum(self, *args, **kwargs)

    def sum_along_batch(self, keepdim: bool = False) -> Tensor:
        r"""Computes the sum values along the batch dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.Tensor``: A tensor with the sum values along the
                batch dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(5, 2)).sum_along_batch()
            tensor([20, 25])
        """
        return self.sum(dim=self._batch_dim, keepdim=keepdim)

    ###########################################
    #     Mathematical | trigo operations     #
    ###########################################

    def acos(self) -> TBatchedTensor:
        r"""Computes the inverse cosine (arccos) of each element.

        Return:
            ``BatchedTensor``: A batch with the inverse cosine
                (arccos) of each element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.acos()
            tensor([[3.1416, 1.5708, 0.0000],
                    [2.0944, 1.5708, 1.0472]], batch_dim=0)
        """
        return torch.acos(self)

    def acos_(self) -> None:
        r"""Computes the inverse cosine (arccos) of each element.

        In-place version of ``acos()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.acos_()
            >>> batch
            tensor([[-1.5708,  0.0000,  1.5708],
                    [-0.5236,  0.0000,  0.5236]], batch_dim=0)
        """
        self._data.acos_()

    def acosh(self) -> TBatchedTensor:
        r"""Computes the inverse hyperbolic cosine (arccosh) of each
        element.

        Return:
            ``BatchedTensor``: A batch with the inverse hyperbolic
                cosine (arccosh) of each element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            >>> batch.acosh()
            tensor([[0.0000, 1.3170, 1.7627],
                    [2.0634, 2.2924, 2.4779]], batch_dim=0)
        """
        return torch.acosh(self)

    def acosh_(self) -> None:
        r"""Computes the inverse hyperbolic cosine (arccosh) of each
        element.

        Return:
            ``BatchedTensor``: A batch with the inverse hyperbolic
                cosine (arccosh) of each element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            >>> batch.acosh_()
            >>> batch
            tensor([[0.0000, 1.3170, 1.7627],
                    [2.0634, 2.2924, 2.4779]], batch_dim=0)
        """
        self._data.acosh_()

    def asin(self) -> TBatchedTensor:
        r"""Computes the inverse cosine (arcsin) of each element.

        Return:
            ``BatchedTensor``: A batch with the inverse sine
                (arcsin) of each element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.asin()
            tensor([[-1.5708,  0.0000,  1.5708],
                    [-0.5236,  0.0000,  0.5236]], batch_dim=0)
        """
        return torch.asin(self)

    def asin_(self) -> None:
        r"""Computes the inverse sine (arcsin) of each element.

        In-place version of ``asin()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.asin_()
            >>> batch
            tensor([[3.1416, 1.5708, 0.0000],
                    [2.0944, 1.5708, 1.0472]], batch_dim=0)
        """
        self._data.asin_()

    def asinh(self) -> TBatchedTensor:
        r"""Computes the inverse hyperbolic sine (arcsinh) of each
        element.

        Return:
            ``BatchedTensor``: A batch with the inverse hyperbolic
                sine (arcsinh) of each element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.asinh()
            tensor([[-0.8814,  0.0000,  0.8814],
                    [-0.4812,  0.0000,  0.4812]], batch_dim=0)
        """
        return torch.asinh(self)

    def asinh_(self) -> None:
        r"""Computes the inverse hyperbolic sine (arcsinh) of each
        element.

        In-place version of ``asinh()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.asinh_()
            >>> batch
            tensor([[-0.8814,  0.0000,  0.8814],
                    [-0.4812,  0.0000,  0.4812]], batch_dim=0)
        """
        self._data.asinh_()

    def atan(self) -> TBatchedTensor:
        r"""Computes the inverse tangent of each element.

        Return:
            ``BatchedTensor``: A batch with the inverse tangent
                of each element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]]))
            >>> batch.atan()
            tensor([[ 0.0000,  0.7854,  1.1071],
                    [-1.1071, -0.7854,  0.0000]], batch_dim=0)
        """
        return torch.atan(self)

    def atan_(self) -> None:
        r"""Computes the inverse tangent of each element.

        In-place version of ``atan()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]]))
            >>> batch.atan_()
            >>> batch
            tensor([[ 0.0000,  0.7854,  1.1071],
                    [-1.1071, -0.7854,  0.0000]], batch_dim=0)
        """
        self._data.atan_()

    def atanh(self) -> TBatchedTensor:
        r"""Computes the inverse hyperbolic tangent of each element.

        Return:
            ``BatchedTensor``: A batch with the inverse hyperbolic
                tangent of each element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-0.5, 0.0, 0.5], [-0.1, 0.0, 0.1]]))
            >>> batch.atanh()
            tensor([[-0.5493,  0.0000,  0.5493],
                    [-0.1003,  0.0000,  0.1003]], batch_dim=0)
        """
        return torch.atanh(self)

    def atanh_(self) -> None:
        r"""Computes the inverse hyperbolic tangent of each element.

        In-place version of ``atanh()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-0.5, 0.0, 0.5], [-0.1, 0.0, 0.1]]))
            >>> batch.atanh_()
            >>> batch
            tensor([[-0.5493,  0.0000,  0.5493],
                    [-0.1003,  0.0000,  0.1003]], batch_dim=0)
        """
        self._data.atanh_()

    def cos(self) -> TBatchedTensor:
        r"""Computes the cosine of each element.

        Return:
            ``BatchedTensor``: A batch with the cosine of each
                element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> import math
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]])
            ... )
            >>> batch.cos()
            tensor([[ 1.0000e+00, -4.3711e-08, -1.0000e+00],
                    [ 1.0000e+00,  1.1925e-08,  1.0000e+00]], batch_dim=0)
        """
        return torch.cos(self)

    def cos_(self) -> None:
        r"""Computes the cosine of each element.

        In-place version of ``cos()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> import math
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]])
            ... )
            >>> batch.cos()
            >>> batch
            tensor([[ 1.0000e+00, -4.3711e-08, -1.0000e+00],
                    [ 1.0000e+00,  1.1925e-08,  1.0000e+00]], batch_dim=0)
        """
        self._data.cos_()

    def cosh(self) -> TBatchedTensor:
        r"""Computes the hyperbolic cosine (cosh) of each element.

        Return:
            ``BatchedTensor``: A batch with the hyperbolic
                cosine (arccosh) of each element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            >>> batch.cosh()
            tensor([[0.0000, 1.3170, 1.7627],
                    [2.0634, 2.2924, 2.4779]], batch_dim=0)
        """
        return torch.cosh(self)

    def cosh_(self) -> None:
        r"""Computes the hyperbolic cosine (arccosh) of each element.

        In-place version of ``cosh()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            >>> batch.cosh_()
            >>> batch
            tensor([[0.0000, 1.3170, 1.7627],
                    [2.0634, 2.2924, 2.4779]], batch_dim=0)
        """
        self._data.cosh_()

    def sin(self) -> TBatchedTensor:
        r"""Computes the sine of each element.

        Return:
            ``BatchedTensor``: A batch with the sine of each
                element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> import math
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]])
            ... )
            >>> batch.sin()
            tensor([[ 0.0000e+00,  1.0000e+00, -8.7423e-08],
                    [ 1.7485e-07, -1.0000e+00,  0.0000e+00]], batch_dim=0)
        """
        return torch.sin(self)

    def sin_(self) -> None:
        r"""Computes the sine of each element.

        In-place version of ``sin()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> import math
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]])
            ... )
            >>> batch.sin_()
            >>> batch
            tensor([[ 0.0000e+00,  1.0000e+00, -8.7423e-08],
                    [ 1.7485e-07, -1.0000e+00,  0.0000e+00]], batch_dim=0)
        """
        self._data.sin_()

    def sinh(self) -> TBatchedTensor:
        r"""Computes the hyperbolic sine of each element.

        Return:
            ``BatchedTensor``: A batch with the hyperbolic sine of
                each element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.sinh()
            tensor([[-1.1752,  0.0000,  1.1752],
                    [-0.5211,  0.0000,  0.5211]], batch_dim=0)
        """
        return torch.sinh(self)

    def sinh_(self) -> None:
        r"""Computes the hyperbolic sine of each element.

        In-place version of ``sinh()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.sinh_()
            >>> batch
            tensor([[-1.1752,  0.0000,  1.1752],
                    [-0.5211,  0.0000,  0.5211]], batch_dim=0)
        """
        self._data.sinh_()

    def tan(self) -> TBatchedTensor:
        r"""Computes the tangent of each element.

        Return:
            ``BatchedTensor``: A batch with the tangent of each
                element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> import math
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor(
            ...         [
            ...             [0.0, 0.25 * math.pi, math.pi],
            ...             [2 * math.pi, 1.75 * math.pi, -0.25 * math.pi]
            ...         ]
            ...     )
            ... )
            >>> batch.tan()
            tensor([[ 0.0000e+00,  1.0000e+00,  8.7423e-08],
                    [ 1.7485e-07, -1.0000e+00, -1.0000e+00]], batch_dim=0)
        """
        return torch.tan(self)

    def tan_(self) -> None:
        r"""Computes the tangent of each element.

        In-place version of ``tan()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> import math
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor(
            ...         [
            ...             [0.0, 0.25 * math.pi, math.pi],
            ...             [2 * math.pi, 1.75 * math.pi, -0.25 * math.pi]
            ...         ]
            ...     )
            ... )
            >>> batch.tan_()
            >>> batch
            tensor([[ 0.0000e+00,  1.0000e+00,  8.7423e-08],
                    [ 1.7485e-07, -1.0000e+00, -1.0000e+00]], batch_dim=0)
        """
        self._data.tan_()

    def tanh(self) -> TBatchedTensor:
        r"""Computes the tangent of each element.

        Return:
            ``BatchedTensor``: A batch with the tangent of each
                element.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]]))
            >>> batch.tanh()
            tensor([[ 0.0000,  0.7616,  0.9640],
                    [-0.9640, -0.7616,  0.0000]], batch_dim=0)
        """
        return torch.tanh(self)

    def tanh_(self) -> None:
        r"""Computes the tangent of each element.

        In-place version of ``tanh()``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]]))
            >>> batch.tanh_()
            >>> batch
            tensor([[ 0.0000,  0.7616,  0.9640],
                    [-0.9640, -0.7616,  0.0000]], batch_dim=0)
        """
        self._data.tanh_()

    #############################################
    #     Mathematical | logical operations     #
    #############################################

    def logical_and(self, other: BatchedTensor | Tensor) -> TBatchedTensor:
        r"""Computes the element-wise logical AND.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BatchedTensor`` or ``torch.Tensor``):
                Specifies the batch or tensor to compute
                logical AND with.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                logical AND.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch1 = BatchedTensor(
            ...     torch.tensor([[True, True, False, False], [True, False, True, False]])
            ... )
            >>> batch2 = BatchedTensor(
            ...     torch.tensor([[True, False, True, False], [True, True, True, True]])
            ... )
            >>> batch1.logical_and(batch2)
            tensor([[ True, False, False, False],
                    [ True, False,  True, False]], batch_dim=0)
        """
        return torch.logical_and(self, other)

    def logical_and_(self, other: BatchedTensor | Tensor) -> None:
        r"""Computes the element-wise logical AND.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BatchedTensor`` or ``torch.Tensor``):
                Specifies the batch or tensor to compute
                logical AND with.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch1 = BatchedTensor(
            ...     torch.tensor([[True, True, False, False], [True, False, True, False]])
            ... )
            >>> batch2 = BatchedTensor(
            ...     torch.tensor([[True, False, True, False], [True, True, True, True]])
            ... )
            >>> batch1.logical_and_(batch2)
            >>> batch1
            tensor([[ True, False, False, False],
                    [ True, False,  True, False]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, other)))
        self._data.logical_and_(other)

    def logical_not(self) -> TBatchedTensor:
        r"""Computes the element-wise logical NOT of the current batch.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                logical NOT of the current batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[True, True, False, False], [True, False, True, False]])
            ... )
            >>> batch.logical_not()
            tensor([[False, False,  True,  True],
                    [False,  True, False,  True]], batch_dim=0)
        """
        return torch.logical_not(self)

    def logical_not_(self) -> None:
        r"""Computes the element-wise logical NOT of the current batch.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[True, True, False, False], [True, False, True, False]])
            ... )
            >>> batch.logical_not_()
            >>> batch
            tensor([[False, False,  True,  True],
                    [False,  True, False,  True]], batch_dim=0)
        """
        self._data.logical_not_()

    def logical_or(self, other: BatchedTensor | Tensor) -> TBatchedTensor:
        r"""Computes the element-wise logical OR.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BatchedTensor`` or ``torch.Tensor``):
                Specifies the batch to compute logical OR with.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                logical OR.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch1 = BatchedTensor(
            ...     torch.tensor([[True, True, False, False], [True, False, True, False]])
            ... )
            >>> batch2 = BatchedTensor(
            ...     torch.tensor([[True, False, True, False], [True, True, True, True]])
            ... )
            >>> batch1.logical_or(batch2)
            tensor([[ True,  True,  True, False],
                    [ True,  True,  True,  True]], batch_dim=0)
        """
        return torch.logical_or(self, other)

    def logical_or_(self, other: BatchedTensor | Tensor) -> None:
        r"""Computes the element-wise logical OR.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BatchedTensor`` or ``torch.Tensor``):
                Specifies the batch to compute logical OR with.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch1 = BatchedTensor(
            ...     torch.tensor([[True, True, False, False], [True, False, True, False]])
            ... )
            >>> batch2 = BatchedTensor(
            ...     torch.tensor([[True, False, True, False], [True, True, True, True]])
            ... )
            >>> batch1.logical_or_(batch2)
            >>> batch1
            tensor([[ True,  True,  True, False],
                    [ True,  True,  True,  True]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, other)))
        self._data.logical_or_(other)

    def logical_xor(self, other: BatchedTensor | Tensor) -> TBatchedTensor:
        r"""Computes the element-wise logical XOR.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BatchedTensor`` or ``torch.Tensor``):
                Specifies the batch to compute logical XOR with.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                logical XOR.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch1 = BatchedTensor(
            ...     torch.tensor([[True, True, False, False], [True, False, True, False]])
            ... )
            >>> batch2 = BatchedTensor(
            ...     torch.tensor([[True, False, True, False], [True, True, True, True]])
            ... )
            >>> batch1.logical_xor(batch2)
            tensor([[False,  True,  True, False],
                    [False,  True, False,  True]], batch_dim=0)
        """
        return torch.logical_xor(self, other)

    def logical_xor_(self, other: BatchedTensor | Tensor) -> None:
        r"""Computes the element-wise logical XOR.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BatchedTensor`` or ``torch.Tensor``):
                Specifies the batch to compute logical XOR with.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch1 = BatchedTensor(
            ...     torch.tensor([[True, True, False, False], [True, False, True, False]])
            ... )
            >>> batch2 = BatchedTensor(
            ...     torch.tensor([[True, False, True, False], [True, True, True, True]])
            ... )
            >>> batch1.logical_xor_(batch2)
            >>> batch1
            tensor([[False,  True,  True, False],
                    [False,  True, False,  True]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, other)))
        self._data.logical_xor_(other)

    ##########################################################
    #    Indexing, slicing, joining, mutating operations     #
    ##########################################################

    def __getitem__(self, index: IndexType) -> Tensor:
        if isinstance(index, BatchedTensor):
            index = index.data
        return self._data[index]

    def __setitem__(
        self, index: IndexType, value: bool | int | float | Tensor | BatchedTensor
    ) -> None:
        if isinstance(index, BatchedTensor):
            index = index.data
        if isinstance(value, BatchedTensor):
            value = value.data
        self._data[index] = value

    def append(self, other: BatchedTensor) -> None:
        self.cat_along_batch_(other)

    def cat(
        self,
        tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
        dim: int = 0,
    ) -> TBatchedTensor:
        r"""Concatenates the data of the batch(es) to the current batch
        along a given dimension and creates a new batch.

        Args:
            tensors (``BatchedTensor`` or ``torch.Tensor`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Returns:
            ``BatchedTensor``: A batch with the concatenated data
                in the batch dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]])).cat(
            ...     BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]]))
            ... )
            tensor([[ 0,  1,  2],
                    [ 4,  5,  6],
                    [10, 11, 12],
                    [13, 14, 15]], batch_dim=0)
        """
        if isinstance(tensors, (BatchedTensor, Tensor)):
            tensors = [tensors]
        return torch.cat(list(chain([self], tensors)), dim=dim)

    def cat_(
        self,
        tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
        dim: int = 0,
    ) -> None:
        r"""Concatenates the data of the batch(es) to the current batch
        along a given dimension and creates a new batch.

        Args:
            tensor (``BatchedTensor`` or ``torch.Tensor`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
            >>> batch.cat_(
            ...     BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]]))
            ... )
            >>> batch
            tensor([[ 0,  1,  2],
                    [ 4,  5,  6],
                    [10, 11, 12],
                    [13, 14, 15]], batch_dim=0)
        """
        self._data = self.cat(tensors, dim=dim).data

    def cat_along_batch(
        self, tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor]
    ) -> TBatchedTensor:
        r"""Concatenates the data of the batch(es) to the current batch
        along the batch dimension and creates a new batch.

        Args:
            tensors (``BatchedTensor`` or ``torch.Tensor`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Returns:
            ``BatchedTensor``: A batch with the concatenated data
                in the batch dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]])).cat_along_batch(
            ...     BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]]))
            ... )
            tensor([[ 0,  1,  2],
                    [ 4,  5,  6],
                    [10, 11, 12],
                    [13, 14, 15]], batch_dim=0)
            >>> BatchedTensor(torch.tensor([[0, 4], [1, 5], [2, 6]])).cat_along_batch(
            ...     [
            ...         BatchedTensor(torch.tensor([[10, 12], [11, 13]])),
            ...         BatchedTensor(torch.tensor([[20, 22], [21, 23]])),
            ...     ]
            ... )
            tensor([[ 0,  4],
                    [ 1,  5],
                    [ 2,  6],
                    [10, 12],
                    [11, 13],
                    [20, 22],
                    [21, 23]], batch_dim=0)
        """
        return self.cat(tensors, dim=self._batch_dim)

    def cat_along_batch_(
        self, tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor]
    ) -> None:
        r"""Concatenates the data of the batch(es) to the current batch
        along the batch dimension and creates a new batch.

        Args:
            tensors (``BatchedTensor`` or ``torch.Tensor`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
            >>> batch.cat_along_batch_(
            ...     BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]]))
            ... )
            >>> batch
            tensor([[ 0,  1,  2],
                    [ 4,  5,  6],
                    [10, 11, 12],
                    [13, 14, 15]], batch_dim=0)
            >>> BatchedTensor(torch.tensor([[0, 4], [1, 5], [2, 6]]))
            >>> batch.cat_along_batch_(
            ...     [
            ...         BatchedTensor(torch.tensor([[10, 12], [11, 13]])),
            ...         BatchedTensor(torch.tensor([[20, 22], [21, 23]])),
            ...     ]
            ... )
            >>> batch
            tensor([[ 0,  4],
                    [ 1,  5],
                    [ 2,  6],
                    [10, 12],
                    [11, 13],
                    [20, 22],
                    [21, 23]], batch_dim=0)
        """
        self.cat_(tensors, dim=self._batch_dim)

    def chunk(self, chunks: int, dim: int = 0) -> tuple[TBatchedTensor, ...]:
        r"""Splits the batch into chunks along a given dimension.

        Args:
            chunks (int): Specifies the number of chunks.
            dim (int, optional): Specifies the dimension along which
                to split the tensor. Default: ``0``

        Returns:
            tuple: The batch split into chunks along the given
                dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(5, 2)).chunk(chunks=3)
            (tensor([[0, 1], [2, 3]], batch_dim=0),
             tensor([[4, 5], [6, 7]], batch_dim=0),
             tensor([[8, 9]], batch_dim=0))
        """
        return torch.chunk(self, chunks, dim=dim)

    def chunk_along_batch(self, chunks: int) -> tuple[TBatchedTensor, ...]:
        return self.chunk(chunks, self._batch_dim)

    def extend(self, other: Iterable[BatchedTensor]) -> None:
        self.cat_along_batch_(other)

    def index_select(self, dim: int, index: Tensor | Sequence[int]) -> TBatchedTensor:
        r"""Selects data at the given indices along a given dimension.

        Args:
            dim (int): Specifies the index dimension.
            index (``torch.Tensor`` or list or tuple): Specifies the
                indices to select.

        Returns:
            ``BatchedTensor``: A new batch which indexes ``self``
                along the batch dimension using the entries in
                ``index``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.index_select(0, [2, 4])
            tensor([[4, 5],
                    [8, 9]], batch_dim=0)
            >>> batch.index_select(0, torch.tensor([4, 3, 2, 1, 0]))
            tensor([[8, 9],
                    [6, 7],
                    [4, 5],
                    [2, 3],
                    [0, 1]], batch_dim=0)
        """
        if not torch.is_tensor(index):
            index = torch.as_tensor(index)
        return self._create_new_batch(self._data.index_select(dim, index))

    def index_select_along_batch(self, index: Tensor | Sequence[int]) -> TBatchedTensor:
        return self.index_select(self._batch_dim, index)

    def masked_fill(
        self, mask: BatchedTensor | Tensor, value: bool | int | float
    ) -> TBatchedTensor:
        r"""Fills elements of ``self`` batch with ``value`` where ``mask`` is
        ``True``.

        Args:
            mask (``BatchedTensor`` or ``torch.Tensor``):
                Specifies the batch of boolean masks.
            value (number): Specifies the value to fill in with.

        Returns:
            ``BatchedTensor``: A new batch with the updated values.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> mask = BatchedTensor(
            ...     torch.tensor(
            ...         [
            ...             [False, False],
            ...             [False, True],
            ...             [True, False],
            ...             [True, True],
            ...             [False, False],
            ...         ]
            ...     )
            ... )
            >>> batch.masked_fill(mask, 42)
            tensor([[ 0,  1],
                    [ 2, 42],
                    [42,  5],
                    [42, 42],
                    [ 8,  9]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, mask)))
        if isinstance(mask, BatchedTensor):
            mask = mask.data
        return self._create_new_batch(self._data.masked_fill(mask.data, value))

    def select(self, dim: int, index: int) -> Tensor:
        r"""Selects the batch along the batch dimension at the given
        index.

        Args:
            dim (int): Specifies the index dimension.
            index (int): Specifies the index to select.

        Returns:
            ``Tensor``: The batch sliced along the batch
                dimension at the given index.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(5, 2)).select(dim=0, index=2)
            tensor([4, 5])
        """
        return torch.select(self._data, dim=dim, index=index)

    def select_along_batch(self, index: int) -> Tensor:
        return self._data.select(self._batch_dim, index)

    def slice_along_batch(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> TBatchedTensor:
        return self.slice_along_dim(self._batch_dim, start, stop, step)

    def slice_along_dim(
        self,
        dim: int = 0,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> TBatchedTensor:
        r"""Slices the batch in a given dimension.

        Args:
            dim (int, optional): Specifies the dimension along which
                to slice the tensor. Default: ``0``
            start (int, optional): Specifies the index where the
                slicing of object starts. Default: ``0``
            stop (int, optional): Specifies the index where the
                slicing of object stops. ``None`` means last.
                Default: ``None``
            step (int, optional): Specifies the increment between
                each index for slicing. Default: ``1``

        Returns:
            ``BatchedTensor``: A slice of the current batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(5, 2)).slice_along_dim(start=2)
            tensor([[4, 5],
                    [6, 7],
                    [8, 9]], batch_dim=0)
            >>> BatchedTensor(torch.arange(10).view(5, 2)).slice_along_dim(stop=3)
            tensor([[0, 1],
                    [2, 3],
                    [4, 5]], batch_dim=0)
            >>> BatchedTensor(torch.arange(10).view(5, 2)).slice_along_dim(step=2)
            tensor([[0, 1],
                    [4, 5],
                    [8, 9]], batch_dim=0)
        """
        if dim == 0:
            data = self._data[start:stop:step]
        elif dim == 1:
            data = self._data[:, start:stop:step]
        else:
            data = self._data.transpose(0, dim)[start:stop:step].transpose(0, dim)
        return self._create_new_batch(data)

    def split(
        self, split_size_or_sections: int | Sequence[int], dim: int = 0
    ) -> tuple[TBatchedTensor, ...]:
        r"""Splits the batch into chunks along a given dimension.

        Args:
            split_size_or_sections (int or sequence): Specifies the
                size of a single chunk or list of sizes for each chunk.
            dim (int, optional): Specifies the dimension along which
                to split the tensor. Default: ``0``

        Returns:
            tuple: The batch split into chunks along the given
                dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(5, 2)).split(2, dim=0)
            (tensor([[0, 1], [2, 3]], batch_dim=0),
             tensor([[4, 5], [6, 7]], batch_dim=0),
             tensor([[8, 9]], batch_dim=0))
        """
        return torch.split(self, split_size_or_sections, dim=dim)

    def split_along_batch(
        self, split_size_or_sections: int | Sequence[int]
    ) -> tuple[BatchedTensor, ...]:
        return self.split(split_size_or_sections, dim=self._batch_dim)

    def take_along_batch(
        self, indices: BaseBatch[Tensor | Sequence] | Tensor | Sequence
    ) -> TBatchedTensor:
        r"""Takes values along the batch dimension.

        Args:
            indices (``BaseBatch`` or ``Tensor`` or sequence):
                Specifies the indices to take along the batch
                dimension.

        Returns:
            ``BaseBatch``: The batch with the selected data.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(5, 2)).take_along_batch(
            ...     BatchedTensor(torch.tensor([[3, 2], [0, 3], [1, 4]]))
            ... )
            tensor([[6, 5],
                    [0, 7],
                    [2, 9]], batch_dim=0)
        """
        return self.take_along_dim(indices, dim=self._batch_dim)

    def take_along_dim(
        self,
        indices: BaseBatch[Tensor | Sequence] | Tensor | Sequence,
        dim: int | None = None,
    ) -> TBatchedTensor:
        r"""Takes values along the batch dimension.

        Args:
            indices (``BaseBatch`` or ``Tensor`` or sequence):
                Specifies the indices to take along the batch
                dimension.

        Returns:
            ``BaseBatch``: The batch with the selected data.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(5, 2)).take_along_dim(
            ...     BatchedTensor(torch.tensor([[3, 2], [0, 3], [1, 4]])), dim=0
            ... )
            tensor([[6, 5],
                    [0, 7],
                    [2, 9]], batch_dim=0)
        """
        if isinstance(indices, Sequence):
            indices = torch.as_tensor(indices)
        return torch.take_along_dim(self, indices, dim=dim)

    def unsqueeze(self, dim: int) -> TBatchedTensor:
        r"""Returns a new batch with a dimension of size one inserted at
        the specified position.

        The returned tensor shares the same underlying data with this
        batch.

        Args:
            dim (int): Specifies the dimension at which to insert the
                singleton dimension.

        Returns:
            ``BatchedTensor``: A new batch with an added singleton
                dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.ones(2, 3)).unsqueeze(dim=0)
            tensor([[[1., 1., 1.],
                    [1., 1., 1.]]], batch_dim=1)
            >>> BatchedTensor(torch.ones(2, 3)).unsqueeze(dim=1)
            tensor([[[1., 1., 1.]],
                    [[1., 1., 1.]]], batch_dim=0)
            >>> BatchedTensor(torch.ones(2, 3)).unsqueeze(dim=-1)
            tensor([[[1.],
                     [1.],
                     [1.]],
                    [[1.],
                     [1.],
                     [1.]]], batch_dim=0)
        """
        return self.__class__(
            self._data.unsqueeze(dim=dim),
            batch_dim=self._batch_dim + 1
            if self._batch_dim >= dim and dim >= 0
            else self._batch_dim,
        )

    def view(self, *shape: tuple[int, ...]) -> Tensor:
        r"""Creates a new tensor with the same data as the ``self`` batch but
        with a new shape.

        Args:
            shape (tuple): Specifies the desired shape.

        Retunrs:
            ``torch.Tensor``: A new view of the tensor in the batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.ones(2, 6)).view(2, 3, 2)
            tensor([[[1., 1.],
                     [1., 1.],
                     [1., 1.]],
                    [[1., 1.],
                     [1., 1.],
                     [1., 1.]]])
        """
        return self._data.view(*shape)

    def view_as(self, other: BatchedTensor | Tensor) -> TBatchedTensor:
        r"""Creates a new batch with the same data as the ``self`` batch but the
        shape of ``other``.

        The returned batch shares the same data and must have the
        same number of elements, but the data may have a different
        size.

        Args:
            other (``BatchedTensor``): Specifies the batch with
                the target shape.

        Returns:
            ``BatchedTensor``: A new batch with the shape of
                ``other``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch1 = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch2 = BatchedTensor(torch.zeros(2, 5, 1))
            >>> batch1.view_as(batch2)
            tensor([[[0],
                     [1],
                     [2],
                     [3],
                     [4]],
                    [[5],
                     [6],
                     [7],
                     [8],
                     [9]]], batch_dim=0)
        """
        check_batch_dims(get_batch_dims((self, other)))
        return self._create_new_batch(self._data.view_as(other.data))

    ########################
    #     mini-batches     #
    ########################

    #################
    #     Other     #
    #################

    def apply(self, fn: Callable[[Tensor], Tensor]) -> TBatchedTensor:
        r"""Apply a function to transform the tensor of the current
        batch.

        Args:
            fn (``Callable``): Specifies the function to be applied to
                the tensor. It is the responsibility of the user to
                verify the function applies a valid transformation of
                the data.

        Returns:
            ``BatchedTensor``: The transformed batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.apply(lambda tensor: tensor + 2)
            tensor([[ 2,  3,  4,  5,  6],
                    [ 7,  8,  9, 10, 11]], batch_dim=0)
        """
        return self._create_new_batch(fn(self._data))

    def apply_(self, fn: Callable[[Tensor], Tensor]) -> TBatchedTensor:
        r"""Apply a function to transform the tensor of the current
        batch.

        In-place version of ``apply``.

        Args:
            fn (``Callable``): Specifies the function to be applied to
                the tensor. It is the responsibility of the user to
                verify the function applies a valid transformation of
                the data.

        Returns:
            ``BatchedTensor``: The transformed batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.apply_(lambda tensor: tensor + 2)
            >>> batch
            tensor([[ 2,  3,  4,  5,  6],
                    [ 7,  8,  9, 10, 11]], batch_dim=0)
        """
        self._data = fn(self._data)

    def _create_new_batch(self, data: Tensor) -> TBatchedTensor:
        return self.__class__(data, **self._get_kwargs())

    def _get_kwargs(self) -> dict:
        return {"batch_dim": self._batch_dim}


def check_data_and_dim(data: Tensor, batch_dim: int) -> None:
    r"""Checks if the tensor ``data`` and ``batch_dim`` are correct.

    Args:
        data (``torch.Tensor``): Specifies the tensor in the batch.
        batch_dim (int): Specifies the batch dimension in the
            ``torch.Tensor`` object.

    Raises:
        RuntimeError: if one of the input is incorrect.
    """
    if data.dim() < 1:
        raise RuntimeError(f"data needs at least 1 dimensions (received: {data.dim()})")
    if batch_dim < 0 or batch_dim >= data.dim():
        raise RuntimeError(
            f"Incorrect batch_dim ({batch_dim}) but the value should be in [0, {data.dim() - 1}]"
        )


def check_batch_dims(dims: set[int]) -> None:
    r"""Gets the batch dimensions from the inputs.

    Args:
        dims (set): Specifies the batch dims to check.

    Raises:
        RuntimeError if there are more than one batch dimension.
    """
    if len(dims) != 1:
        raise RuntimeError(f"The batch dimensions do not match. Received multiple values: {dims}")


def get_batch_dims(args: Iterable[Any], kwargs: Mapping[str, Any] | None = None) -> set[int]:
    r"""Gets the batch dimensions from the inputs.

    Args:
        args: Variable length argument list.
        kwargs: Arbitrary keyword arguments.

    Returns:
        set: The batch dimensions.
    """
    kwargs = kwargs or {}
    dims = {val._batch_dim for val in args if hasattr(val, "_batch_dim")}
    dims.update({val._batch_dim for val in kwargs.values() if hasattr(val, "_batch_dim")})
    return dims


def implements(torch_function: Callable) -> Callable:
    """Register a torch function override for BatchedTensor."""

    def decorator(func: Callable) -> Callable:
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements(torch.cat)
def cat(
    tensors: tuple[BatchedTensor | Tensor, ...] | list[BatchedTensor | Tensor],
    dim: int = 0,
) -> BatchedTensor:
    r"""See ``torch.cat`` documentation."""
    batch_dims = get_batch_dims(tensors)
    check_batch_dims(batch_dims)
    return BatchedTensor(
        torch.cat(
            [tensor._data if hasattr(tensor, "_data") else tensor for tensor in tensors], dim=dim
        ),
        batch_dim=batch_dims.pop(),
    )


@implements(torch.chunk)
def chunk(tensor: BatchedTensor, chunks: int, dim: int = 0) -> tuple[BatchedTensor, ...]:
    r"""See ``torch.chunk`` documentation."""
    return tuple(
        BatchedTensor(chunk, batch_dim=tensor.batch_dim)
        for chunk in tensor.data.chunk(chunks, dim=dim)
    )


# Use the name `torchmax` to avoid shadowing `max` python builtin.
@implements(torch.max)
def torchmax(
    input: BatchedTensor, *args, **kwargs  # noqa: A002
) -> Tensor | torch.return_types.max:
    r"""See ``torch.max`` documentation."""
    return torch.max(input.data, *args, **kwargs)


@implements(torch.maximum)
def maximum(input: BatchedTensor, other: BatchedTensor | Tensor) -> BatchedTensor:  # noqa: A002
    r"""See ``torch.maximum`` documentation."""
    check_batch_dims(get_batch_dims((input, other)))
    if isinstance(other, BatchedTensor):
        other = other.data
    return BatchedTensor(torch.maximum(input.data, other), batch_dim=input.batch_dim)


@implements(torch.mean)
def mean(input: BatchedTensor, *args, **kwargs) -> Tensor:  # noqa: A002
    r"""See ``torch.mean`` documentation."""
    return torch.mean(input.data, *args, **kwargs)


@implements(torch.median)
def median(
    input: BatchedTensor, *args, **kwargs  # noqa: A002
) -> Tensor | torch.return_types.median:
    r"""See ``torch.median`` documentation."""
    return torch.median(input.data, *args, **kwargs)


# Use the name `torchmin` to avoid shadowing `min` python builtin.
@implements(torch.min)
def torchmin(
    input: BatchedTensor, *args, **kwargs  # noqa: A002
) -> Tensor | torch.return_types.min:
    r"""See ``torch.min`` documentation."""
    return torch.min(input.data, *args, **kwargs)


@implements(torch.minimum)
def minimum(input: BatchedTensor, other: BatchedTensor | Tensor) -> BatchedTensor:  # noqa: A002
    r"""See ``torch.minimum`` documentation."""
    check_batch_dims(get_batch_dims((input, other)))
    if isinstance(other, BatchedTensor):
        other = other.data
    return BatchedTensor(torch.minimum(input.data, other), batch_dim=input.batch_dim)


@implements(torch.nanmean)
def nanmean(input: BatchedTensor, *args, **kwargs) -> Tensor:  # noqa: A002
    r"""See ``torch.nanmean`` documentation."""
    return torch.nanmean(input.data, *args, **kwargs)


@implements(torch.nanmedian)
def nanmedian(
    input: BatchedTensor, *args, **kwargs  # noqa: A002
) -> Tensor | torch.return_types.nanmedian:
    r"""See ``torch.nanmedian`` documentation."""
    return torch.nanmedian(input.data, *args, **kwargs)


@implements(torch.nansum)
def nansum(input: BatchedTensor, *args, **kwargs) -> Tensor:  # noqa: A002
    r"""See ``torch.nansum`` documentation."""
    return torch.nansum(input.data, *args, **kwargs)


@implements(torch.prod)
def prod(input: BatchedTensor, *args, **kwargs) -> Tensor:  # noqa: A002
    r"""See ``torch.prod`` documentation."""
    return torch.prod(input.data, *args, **kwargs)


@implements(torch.select)
def select(input: BatchedTensor, dim: int, index: int) -> Tensor:  # noqa: A002
    r"""See ``torch.select`` documentation."""
    return torch.select(input.data, dim=dim, index=index)


@implements(torch.sort)
def sort(input: BatchedTensor, *args, **kwargs) -> torch.return_types.sort:  # noqa: A002
    r"""See ``torch.sort`` documentation."""
    values, indices = torch.sort(input.data, *args, **kwargs)
    return torch.return_types.sort(
        [
            BatchedTensor(data=values, batch_dim=input.batch_dim),
            BatchedTensor(data=indices, batch_dim=input.batch_dim),
        ]
    )


@implements(torch.split)
def split(
    tensor: BatchedTensor, split_size_or_sections: int | Sequence[int], dim: int = 0
) -> tuple[BatchedTensor, ...]:
    r"""See ``torch.split`` documentation."""
    return tuple(
        BatchedTensor(chunk, batch_dim=tensor.batch_dim)
        for chunk in tensor.data.split(split_size_or_sections, dim=dim)
    )


@implements(torch.sum)
def torchsum(input: BatchedTensor, *args, **kwargs) -> Tensor:  # noqa: A002
    r"""See ``torch.sum`` documentation.

    Use the name `torchsum` to avoid shadowing `sum` python builtin.
    """
    return torch.sum(input.data, *args, **kwargs)


@overload
def take_along_dim(
    input: BatchedTensor | Tensor,  # noqa: A002
    indices: BatchedTensor | Tensor,
) -> Tensor:
    r"""See ``torch.take_along_dim`` documentation."""


@overload
def take_along_dim(
    input: BatchedTensor | Tensor, indices: BatchedTensor | Tensor, dim: int  # noqa: A002
) -> BatchedTensor:
    r"""See ``torch.take_along_dim`` documentation."""


@implements(torch.take_along_dim)
def take_along_dim(
    input: BatchedTensor | Tensor,  # noqa: A002
    indices: BatchedTensor | Tensor,
    dim: int | None = None,
) -> BatchedTensor | Tensor:
    r"""See ``torch.take_along_dim`` documentation."""
    batch_dims = get_batch_dims((input, indices))
    check_batch_dims(batch_dims)
    if isinstance(input, BatchedTensor):
        input = input.data  # noqa: A001
    if isinstance(indices, BatchedTensor):
        indices = indices.data
    if dim is None:
        return torch.take_along_dim(input, indices)
    return BatchedTensor(
        torch.take_along_dim(input, indices, dim=dim),
        batch_dim=batch_dims.pop(),
    )
