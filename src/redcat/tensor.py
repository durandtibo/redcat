r"""Contain the implementation of the ``BatchedTensor``.

``BatchedTensor`` is a custom ``torch.Tensor`` container to make
batch manipulation easier.
"""

from __future__ import annotations

__all__ = ["BatchedTensor"]

import functools
from itertools import chain
from typing import TYPE_CHECKING, Any, overload

import torch
from coola import objects_are_allclose, objects_are_equal
from torch import Tensor

from redcat.base import BaseBatch
from redcat.utils.common import check_batch_dims, check_data_and_dim, get_batch_dims
from redcat.utils.tensor import to_tensor

if TYPE_CHECKING:
    import sys
    from collections.abc import Callable, Iterable, Sequence

    import numpy as np

    from redcat.types import IndexType

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


HANDLED_FUNCTIONS = {}


class BatchedTensor(BaseBatch[Tensor]):
    r"""Implement a batched tensor to easily manipulate a batch of
    examples.

    Args:
        data: Specifies the data for the tensor. It can
            be a torch.Tensor, list, tuple, NumPy ndarray, scalar,
            and other types.
        batch_dim: Specifies the batch dimension
            in the ``torch.Tensor`` object.
        kwargs: Keyword arguments that are passed to
            ``torch.as_tensor``.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
        >>> batch
        tensor([[0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9]], batch_dim=0)
    """

    def __init__(self, data: Any, *, batch_dim: int = 0, **kwargs: Any) -> None:
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
    ) -> Self:
        kwargs = kwargs or {}
        if handled_func := HANDLED_FUNCTIONS.get(func):
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
        r"""``torch.device``: The device where the batch data/tensor
        is."""
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        r"""``torch.dtype``: The data type."""
        return self._data.dtype

    @property
    def ndim(self) -> int:
        r"""``int``: The number of dimensions."""
        return self._data.ndim

    @property
    def shape(self) -> torch.Size:
        r"""``torch.Size``: The shape of the tensor."""
        return self._data.shape

    def dim(self) -> int:
        r"""Get the number of dimensions.

        Returns:
            int: The number of dimensions

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.dim()
            2
        """
        return self._data.dim()

    def ndimension(self) -> int:
        r"""Get the number of dimensions.

        Returns:
            int: The number of dimensions

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.ndimension()
            2
        """
        return self.dim()

    def numel(self) -> int:
        r"""Get the total number of elements in the tensor.

        Returns:
            int: The total number of elements

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.numel()
            6
        """
        return self._data.numel()

    #################################
    #     Conversion operations     #
    #################################

    def contiguous(self, memory_format: torch.memory_format = torch.contiguous_format) -> Self:
        r"""Create a batch with a contiguous representation of the data.

        Args:
            memory_format (``torch.memory_format``, optional):
                Specifies the desired memory format.
                Default: ``torch.contiguous_format``

        Returns:
            ``BatchedTensor``: A new batch with a contiguous
                representation of the data.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3)).contiguous()
            >>> batch.is_contiguous()
            True
        """
        return self._create_new_batch(self._data.contiguous(memory_format=memory_format))

    def is_contiguous(self, memory_format: torch.memory_format = torch.contiguous_format) -> bool:
        r"""Indicate if a batch as a contiguous representation of the
        data.

        Args:
            memory_format (``torch.memory_format``, optional):
                Specifies the desired memory format.
                Default: ``torch.contiguous_format``

        Returns:
            bool: ``True`` if the data are stored with a contiguous
                tensor, otherwise ``False``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.is_contiguous()
            True
        """
        return self._data.is_contiguous(memory_format=memory_format)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        r"""Move and/or cast the data.

        Args:
            *args: See the documentation of ``torch.Tensor.to``
            **kwargs: See the documentation of ``torch.Tensor.to``

        Returns:
            ``BatchedTensor``: A new batch with the data after
                dtype and/or device conversion.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.to(dtype=torch.bool)
            tensor([[True, True, True],
                    [True, True, True]], batch_dim=0)
        """
        return self._create_new_batch(self._data.to(*args, **kwargs))

    def to_data(self) -> Tensor:
        return self._data

    ###############################
    #     Creation operations     #
    ###############################

    def clone(self, *args: Any, **kwargs: Any) -> Self:
        r"""Create a copy of the current batch.

        Args:
            *args: See the documentation of ``torch.Tensor.clone``
            **kwargs: See the documentation of ``torch.Tensor.clone``

        Returns:
            ``BatchedTensor``: A copy of the current batch.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch_copy = batch.clone()
            >>> batch_copy
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
        """
        return self._create_new_batch(self._data.clone(*args, **kwargs))

    def empty_like(self, *args: Any, **kwargs: Any) -> Self:
        r"""Create an uninitialized batch, with the same shape as the
        current batch.

        Args:
            *args: See the documentation of ``torch.Tensor.empty_like``
            **kwargs: See the documentation of
                ``torch.Tensor.empty_like``

        Returns:
            ``BatchedTensor``: A uninitialized batch with the same
                shape as the current batch.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.empty_like()
            tensor([[...]], batch_dim=0)
        """
        return self._create_new_batch(torch.empty_like(self._data, *args, **kwargs))

    def full_like(self, *args: Any, **kwargs: Any) -> Self:
        r"""Create a batch filled with a given scalar value, with the
        same shape as the current batch.

        Args:
            *args: See the documentation of ``torch.Tensor.full_like``
            **kwargs: See the documentation of
                ``torch.Tensor.full_like``

        Returns:
            ``BatchedTensor``: A batch filled with the scalar
                value, with the same shape as the current batch.

        Example usage:

        .. code-block:: pycon

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
        fill_value: float | bool,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> Self:
        r"""Create a batch filled with a scalar value.

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

        .. code-block:: pycon

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
        **kwargs: Any,
    ) -> BatchedTensor:
        r"""Create a batch filled with the scalar value ``1``.

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

        .. code-block:: pycon

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
        **kwargs: Any,
    ) -> Self:
        r"""Create a batch filled with the scalar value ``0``.

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

        .. code-block:: pycon

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

    def ones_like(self, *args: Any, **kwargs: Any) -> Self:
        r"""Create a batch filled with the scalar value ``1``, with the
        same shape as the current batch.

        Args:
            *args: See the documentation of ``torch.Tensor.ones_like``
            **kwargs: See the documentation of
                ``torch.Tensor.ones_like``

        Returns:
            ``BatchedTensor``: A batch filled with the scalar
                value ``1``, with the same shape as the current
                batch.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.ones_like()
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
        """
        return self._create_new_batch(torch.ones_like(self._data, *args, **kwargs))

    def zeros_like(self, *args: Any, **kwargs: Any) -> Self:
        r"""Create a batch filled with the scalar value ``0``, with the
        same shape as the current batch.

        Args:
            *args: See the documentation of ``torch.Tensor.zeros_like``
            **kwargs: See the documentation of
                ``torch.Tensor.zeros_like``

        Returns:
            ``BatchedTensor``: A batch filled with the scalar
                value ``0``, with the same shape as the current
                batch.

        Example usage:

        .. code-block:: pycon

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

    def __eq__(self, other: object) -> Self:
        return self.eq(other)

    def __ge__(self, other: Any) -> Self:
        return self.ge(other)

    def __gt__(self, other: Any) -> Self:
        return self.gt(other)

    def __le__(self, other: Any) -> Self:
        return self.le(other)

    def __lt__(self, other: Any) -> Self:
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

    def allequal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self._batch_dim != other.batch_dim:
            return False
        return objects_are_equal(self._data, other.data)

    def eq(self, other: BatchedTensor | Tensor | bool | float) -> Self:
        r"""Compute element-wise equality.

        Args:
            other: Specifies the batch to compare.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                equality.

        Example usage:

        .. code-block:: pycon

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
        r"""Indicate if two batches are equal or not.

        Args:
            other: Specifies the value to compare.

        Returns:
            ``True`` if the batches have the same size, elements and
                same batch dimension, ``False`` otherwise.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> BatchedTensor(torch.ones(2, 3)).equal(BatchedTensor(torch.zeros(2, 3)))
        False

        ```
        """
        return self.allequal(other)

    def ge(self, other: BatchedTensor | Tensor | bool | float) -> Self:
        r"""Compute ``self >= other`` element-wise.

        Args:
            other: Specifies the value to compare
                with.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                comparison.

        Example usage:

        .. code-block:: pycon

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

    def gt(self, other: BatchedTensor | Tensor | bool | float) -> Self:
        r"""Compute ``self > other`` element-wise.

        Args:
            other: Specifies the batch to compare.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                comparison.

        Example usage:

        .. code-block:: pycon

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

    def isinf(self) -> Self:
        r"""Indicate if each element of the batch is infinite (positive
        or negative infinity) or not.

        Returns:
            ``BatchedTensor``:  A batch containing a boolean tensor
                that is ``True`` where the current batch is infinite
                and ``False`` elsewhere.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]])
            ... )
            >>> batch.isinf()
            tensor([[False, False, True],
                    [False, False, True]], batch_dim=0)
        """
        return torch.isinf(self)

    def isneginf(self) -> Self:
        r"""Indicate if each element of the batch is negative infinity or
        not.

        Returns:
            ``BatchedTensor``:  A batch containing a boolean tensor
                that is ``True`` where the current batch is negative
                infinity and ``False`` elsewhere.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]])
            ... )
            >>> batch.isneginf()
            tensor([[False, False, False],
                    [False, False,  True]], batch_dim=0)
        """
        return torch.isneginf(self)

    def isposinf(self) -> Self:
        r"""Indicate if each element of the batch is positive infinity or
        not.

        Returns:
            ``BatchedTensor``:  A batch containing a boolean tensor
                that is ``True`` where the current batch is positive
                infinity and ``False`` elsewhere.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]])
            ... )
            >>> batch.isposinf()
            tensor([[False, False,   True],
                    [False, False,  False]], batch_dim=0)
        """
        return torch.isposinf(self)

    def isnan(self) -> Self:
        r"""Indicate if each element in the batch is NaN or not.

        Returns:
            ``BatchedTensor``:  A batch containing a boolean tensor
                that is ``True`` where the current batch is infinite
                and ``False`` elsewhere.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[1.0, 0.0, float("nan")], [float("nan"), -2.0, -1.0]])
            ... )
            >>> batch.isnan()
            tensor([[False, False,  True],
                    [ True, False, False]], batch_dim=0)
        """
        return torch.isnan(self)

    def le(self, other: BatchedTensor | Tensor | bool | float) -> Self:
        r"""Compute ``self <= other`` element-wise.

        Args:
            other: Specifies the batch to compare.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                comparison.

        Example usage:

        .. code-block:: pycon

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

    def lt(self, other: BatchedTensor | Tensor | bool | float) -> Self:
        r"""Compute ``self < other`` element-wise.

        Args:
            other: Specifies the batch to compare.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                comparison.

        Example usage:

        .. code-block:: pycon

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

    def bool(self) -> Self:
        r"""Convert the current batch to bool data type.

        Returns:
            ``BatchedTensor``: The current batch to bool data type.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.bool().dtype
            torch.bool
        """
        return self._create_new_batch(self._data.bool())

    def double(self) -> Self:
        r"""Convert the current batch to double (``float64``) data type.

        Returns:
            ``BatchedTensor``: The current batch to double data type.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.double().dtype
            torch.float64
        """
        return self._create_new_batch(self._data.double())

    def float(self) -> Self:
        r"""Convert the current batch to float (``float32``) data type.

        Returns:
            ``BatchedTensor``: The current batch to float data type.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.float().dtype
            torch.float32
        """
        return self._create_new_batch(self._data.float())

    def int(self) -> Self:
        r"""Convert the current batch to int (``int32``) data type.

        Returns:
            ``BatchedTensor``: The current batch to int data type.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.int().dtype
            torch.int32
        """
        return self._create_new_batch(self._data.int())

    def long(self) -> Self:
        r"""Convert the current batch to long (``int64``) data type.

        Returns:
            ``BatchedTensor``: The current batch to long data type.

        Example usage:

        .. code-block:: pycon

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

    def __add__(self, other: Any) -> Self:
        return self.add(other)

    def __iadd__(self, other: Any) -> Self:
        self.add_(other)
        return self

    def __floordiv__(self, other: Any) -> Self:
        return self.div(other, rounding_mode="floor")

    def __ifloordiv__(self, other: Any) -> Self:
        self.div_(other, rounding_mode="floor")
        return self

    def __mul__(self, other: Any) -> Self:
        return self.mul(other)

    def __imul__(self, other: Any) -> Self:
        self.mul_(other)
        return self

    def __neg__(self) -> Self:
        return self.neg()

    def __sub__(self, other: Any) -> Self:
        return self.sub(other)

    def __isub__(self, other: Any) -> Self:
        self.sub_(other)
        return self

    def __truediv__(self, other: Any) -> Self:
        return self.div(other)

    def __itruediv__(self, other: Any) -> Self:
        self.div_(other)
        return self

    def add(
        self,
        other: BatchedTensor | Tensor | float,
        alpha: float = 1.0,
    ) -> Self:
        r"""Add the input ``other``, scaled by ``alpha``, to the ``self``
        batch.

        Similar to ``out = self + alpha * other``

        Args:
            other: Specifies the other value to add to the
                current batch.
            alpha: Specifies the scale of the batch to add.

        Returns:
            A new batch containing the addition of the two batches.

        Example usage:

        ```pycon
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

        ```
        """
        return torch.add(self, other, alpha=alpha)

    def add_(
        self,
        other: BatchedTensor | Tensor | float,
        alpha: float = 1.0,
    ) -> None:
        r"""Add the input ``other``, scaled by ``alpha``, to the ``self``
        batch.

        Similar to ``self += alpha * other`` (in-place)

        Args:
            other: Specifies the other value to add to the
                current batch.
            alpha: Specifies the scale of the batch to add.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.ones(2, 3))
        >>> batch.add_(BatchedTensor(torch.full((2, 3), 2.0)))
        >>> batch
        tensor([[3., 3., 3.],
                [3., 3., 3.]], batch_dim=0)

        ```
        """
        self._check_valid_dims((self, other))
        self._data.add_(other, alpha=alpha)

    def div(
        self,
        other: BatchedTensor | Tensor | float,
        rounding_mode: str | None = None,
    ) -> Self:
        r"""Divide the ``self`` batch by the input ``other`.

        Similar to ``out = self / other``

        Args:
            other: Specifies the dividend.
            rounding_mode: Specifies the type of rounding applied to
                the result.
                - ``None``: true division.
                - ``"trunc"``: rounds the results of the division
                    towards zero.
                - ``"floor"``: floor division.

        Returns:
            A new batch containing the division of the two batches.

        Example usage:

        ```pycon
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

        ```
        """
        return torch.div(self, other, rounding_mode=rounding_mode)

    def div_(
        self,
        other: BatchedTensor | Tensor | float,
        rounding_mode: str | None = None,
    ) -> None:
        r"""Divide the ``self`` batch by the input ``other`.

        Similar to ``self /= other`` (in-place)

        Args:
            other: Specifies the dividend.
            rounding_mode: Specifies the type of rounding applied to
                the result.
                - ``None``: true division.
                - ``"trunc"``: rounds the results of the division
                    towards zero.
                - ``"floor"``: floor division.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.ones(2, 3))
        >>> batch.div_(BatchedTensor(torch.full((2, 3), 2.0)))
        >>> batch
        tensor([[0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000]], batch_dim=0)

        ```
        """
        self._check_valid_dims((self, other))
        self._data.div_(other, rounding_mode=rounding_mode)

    def fmod(
        self,
        divisor: BatchedTensor | Tensor | float,
    ) -> Self:
        r"""Compute the element-wise remainder of division.

        The current batch is the dividend.

        Args:
            divisor: Specifies the divisor.

        Returns:
            A new batch containing the element-wise remainder of
                division.

        Example usage:

        ```pycon

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

        ```
        """
        return torch.fmod(self, divisor)

    def fmod_(self, divisor: BatchedTensor | Tensor | float) -> None:
        r"""Compute the element-wise remainder of division.

        The current batch is the dividend.

        Args:
            divisor: Specifies the divisor.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.ones(2, 3))
        >>> batch.fmod_(BatchedTensor(torch.full((2, 3), 2.0)))
        >>> batch
        tensor([[1., 1., 1.],
                [1., 1., 1.]], batch_dim=0)

        ```
        """
        self._check_valid_dims((self, divisor))
        self._data.fmod_(divisor)

    def mul(self, other: BatchedTensor | Tensor | float) -> Self:
        r"""Multiply the ``self`` batch by the input ``other`.

        Similar to ``out = self * other``

        Args:
            other: Specifies the value to multiply.

        Returns:
            A new batch containing the multiplication of the two
                batches.

        Example usage:

        .. code-block:: pycon

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

    def mul_(self, other: BatchedTensor | Tensor | float) -> None:
        r"""Multiply the ``self`` batch by the input ``other`.

        Similar to ``self *= other`` (in-place)

        Args:
            other: Specifies the value to multiply.

        Returns:
            A new batch containing the multiplication of the two
                batches.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.ones(2, 3))
        >>> batch.mul_(BatchedTensor(torch.full((2, 3), 2.0)))
        >>> batch
        tensor([[2., 2., 2.],
                [2., 2., 2.]], batch_dim=0)

        ```
        """
        self._check_valid_dims((self, other))
        self._data.mul_(other)

    def neg(self) -> Self:
        r"""Return a new batch with the negative of the elements.

        Returns:
            A new batch with the negative of the elements.

        Example usage:

        ```pycon
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

        ```
        """
        return torch.neg(self)

    def sub(
        self,
        other: BatchedTensor | Tensor | float,
        alpha: float = 1,
    ) -> Self:
        r"""Subtract the input ``other``, scaled by ``alpha``, to the
        ``self`` batch.

        Similar to ``out = self - alpha * other``

        Args:
            other: Specifies the value to subtract.
            alpha: Specifies the scale of the batch to substract.

        Returns:
            A new batch containing the diffence of the two batches.

        Example usage:

        ```pycon
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

        ```
        """
        return torch.sub(self, other, alpha=alpha)

    def sub_(
        self,
        other: BatchedTensor | Tensor | float,
        alpha: float = 1,
    ) -> None:
        r"""Subtract the input ``other``, scaled by ``alpha``, to the
        ``self`` batch.

        Similar to ``self -= alpha * other`` (in-place)

        Args:
            other: Specifies the value to subtract.
            alpha: Specifies the scale of the batch to substract.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.ones(2, 3))
        >>> batch.sub_(BatchedTensor(torch.full((2, 3), 2.0)))
        >>> batch
        tensor([[-1., -1., -1.],
                [-1., -1., -1.]], batch_dim=0)

        ```
        """
        self._check_valid_dims((self, other))
        self._data.sub_(other, alpha=alpha)

    ###########################################################
    #     Mathematical | advanced arithmetical operations     #
    ###########################################################

    def argsort(self, dim: int = -1, **kwargs: Any) -> Self:
        r"""Return the indices that sort the batch along a given
        dimension in monotonic order by value.

        Args:
            dim: Specifies the dimension to sort
                along. Default: ``-1``
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ``BatchedTensor``: The indices that sort the batch along
                the given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.argsort(descending=True)
            tensor([[4, 3, 2, 1, 0],
                    [4, 3, 2, 1, 0]], batch_dim=0)
        """
        return self._create_new_batch(torch.argsort(self._data, dim=dim, **kwargs))

    def argsort_along_batch(self, **kwargs: Any) -> Self:
        r"""Sorts the elements of the batch along the batch dimension in
        monotonic order by value.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ``BatchedTensor``: The indices that sort the batch along
                the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.argsort_along_batch(descending=True)
            tensor([[4, 4],
                    [3, 3],
                    [2, 2],
                    [1, 1],
                    [0, 0]], batch_dim=0)
        """
        return self.argsort(dim=self._batch_dim, **kwargs)

    def cumprod(self, dim: int, *args: Any, **kwargs: Any) -> Self:
        r"""Compute the cumulative product of elements of the current
        batch in a given dimension.

        Args:
            dim: Specifies the dimension of the cumulative sum.
            *args: See the documentation of ``torch.Tensor.cumprod``
            **kwargs: See the documentation of ``torch.Tensor.cumprod``

        Returns:
            ``BatchedTensor``: A batch with the cumulative product of
                elements of the current batch in a given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.cumprod(dim=0)
            tensor([[ 0,  1,  2,  3,  4],
                    [ 0,  6, 14, 24, 36]], batch_dim=0)
        """
        return torch.cumprod(self, dim, *args, **kwargs)

    def cumprod_(self, dim: int, *args: Any, **kwargs: Any) -> None:
        r"""Compute the cumulative product of elements of the current
        batch in a given dimension.

        Args:
            dim: Specifies the dimension of the cumulative product.
            *args: See the documentation of ``torch.Tensor.cumprod``
            **kwargs: See the documentation of ``torch.Tensor.cumprod``

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.cumprod_(dim=0)
            >>> batch
            tensor([[ 0,  1,  2,  3,  4],
                    [ 0,  6, 14, 24, 36]], batch_dim=0)
        """
        self._data.cumprod_(dim, *args, **kwargs)

    def cumprod_along_batch(self, *args: Any, **kwargs: Any) -> Self:
        r"""Compute the cumulative product of elements of the current
        batch in the batch dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.cumprod``
            **kwargs: See the documentation of ``torch.Tensor.cumprod``

        Returns:
            ``BatchedTensor``: A batch with the cumulative product of
                elements of the current batch in the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.cumprod_along_batch()
            tensor([[ 0,  1,  2,  3,  4],
                    [ 0,  6, 14, 24, 36]], batch_dim=0)
        """
        return self.cumprod(self._batch_dim, *args, **kwargs)

    def cumprod_along_batch_(self, *args: Any, **kwargs: Any) -> None:
        r"""Compute the cumulative product of elements of the current
        batch in the batch dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.cumprod``
            **kwargs: See the documentation of ``torch.Tensor.cumprod``

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.cumprod_along_batch_()
            >>> batch
            tensor([[ 0,  1,  2,  3,  4],
                    [ 0,  6, 14, 24, 36]], batch_dim=0)
        """
        self.cumprod_(self._batch_dim, *args, **kwargs)

    def cumsum(self, dim: int, **kwargs: Any) -> Self:
        r"""Compute the cumulative sum of elements of the current batch
        in a given dimension.

        Args:
            dim: Specifies the dimension of the cumulative sum.
            **kwargs: see ``torch.cumsum`` documentation

        Returns:
            ``BatchedTensor``: A batch with the cumulative sum of
                elements of the current batch in a given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.cumsum(dim=0)
            tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  7,  9, 11, 13]], batch_dim=0)
        """
        return torch.cumsum(self, dim=dim, **kwargs)

    def cumsum_(self, dim: int, **kwargs: Any) -> None:
        r"""Compute the cumulative sum of elements of the current batch
        in a given dimension.

        Args:
            dim: Specifies the dimension of the cumulative sum.
            **kwargs: see ``torch.cumsum_`` documentation

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.cumsum_(dim=0)
            >>> batch
            tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  7,  9, 11, 13]], batch_dim=0)
        """
        self._data.cumsum_(dim=dim, **kwargs)

    def cumsum_along_batch(self, **kwargs: Any) -> Self:
        r"""Compute the cumulative sum of elements of the current batch
        in the batch dimension.

        Args:
            **kwargs: see ``torch.cumsum`` documentation

        Returns:
            ``BatchedTensor``: A batch with the cumulative sum of
                elements of the current batch in the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.cumsum_along_batch()
            tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  7,  9, 11, 13]], batch_dim=0)
        """
        return self.cumsum(self._batch_dim, **kwargs)

    def cumsum_along_batch_(self) -> None:
        r"""Compute the cumulative sum of elements of the current batch
        in the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.cumsum_along_batch_()
            >>> batch
            tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  7,  9, 11, 13]], batch_dim=0)
        """
        self.cumsum_(self._batch_dim)

    def logcumsumexp(self, dim: int) -> Self:
        r"""Compute the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in a given
        dimension.

        Args:
            dim: Specifies the dimension of the cumulative sum.

        Returns:
            ``BatchedTensor``: A batch with the cumulative
                summation of the exponentiation of elements of the
                current batch in a given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5).float())
            >>> batch.logcumsumexp(dim=1)
            tensor([[0.0000, 1.3133, 2.4076, 3.4402, 4.4519],
                    [5.0000, 6.3133, 7.4076, 8.4402, 9.4519]], batch_dim=0)
        """
        return torch.logcumsumexp(self, dim=dim)

    def logcumsumexp_(self, dim: int) -> None:
        r"""Compute the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in a given
        dimension.

        Args:
            dim: Specifies the dimension of the cumulative sum.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5).float())
            >>> batch.logcumsumexp_(dim=1)
            >>> batch
            tensor([[0.0000, 1.3133, 2.4076, 3.4402, 4.4519],
                    [5.0000, 6.3133, 7.4076, 8.4402, 9.4519]], batch_dim=0)
        """
        self._data = self._data.logcumsumexp(dim=dim)

    def logcumsumexp_along_batch(self) -> Self:
        r"""Compute the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in the batch
        dimension.

        Returns:
            ``BatchedTensor``: A batch with the cumulative
                summation of the exponentiation of elements of the
                current batch in the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2).float())
            >>> batch.logcumsumexp_along_batch()
            tensor([[0.0000, 1.0000],
                    [2.1269, 3.1269],
                    [4.1429, 5.1429],
                    [6.1451, 7.1451],
                    [8.1454, 9.1454]], batch_dim=0)
        """
        return self.logcumsumexp(self._batch_dim)

    def logcumsumexp_along_batch_(self) -> None:
        r"""Compute the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in the batch
        dimension.

        Example usage:

        .. code-block:: pycon

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

    def permute_along_batch(self, permutation: Sequence[int] | Tensor) -> Self:
        return self.permute_along_dim(permutation, dim=self._batch_dim)

    def permute_along_batch_(self, permutation: Sequence[int] | Tensor) -> None:
        self.permute_along_dim_(permutation, dim=self._batch_dim)

    def permute_along_dim(self, permutation: Sequence[int] | Tensor, dim: int) -> Self:
        r"""Permutes the data/batch along a given dimension.

        Args:
            permutation: Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.
            dim: Specifies the dimension where the permutation
                is computed.

        Returns:
            A new batch with permuted data.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
        >>> batch.permute_along_dim([2, 1, 3, 0, 4], dim=0)
        tensor([[4, 5],
                [2, 3],
                [6, 7],
                [0, 1],
                [8, 9]], batch_dim=0)

        ```
        """
        return self.index_select(dim=dim, index=permutation)

    def permute_along_dim_(self, permutation: Sequence[int] | Tensor, dim: int) -> None:
        r"""Permutes the data/batch along a given dimension.

        Args:
            permutation: Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.
            dim: Specifies the dimension where the permutation
                is computed.

        Example usage:

        ```pycon
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

        ```
        """
        self._data = self.permute_along_dim(dim=dim, permutation=permutation).data

    def shuffle_along_dim(self, dim: int, generator: torch.Generator | None = None) -> Self:
        r"""Shuffles the data/batch along a given dimension.

        Args:
            dim: Specifies the shuffle dimension.
            generator (``torch.Generator`` or ``None``, optional):
                Specifies an optional random generator.
                Default: ``None``

        Returns:
            A new batch with shuffled data along a given dimension.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
        >>> batch.shuffle_along_dim(dim=0)
        tensor([[...]], batch_dim=0)

        ```
        """
        return self.index_select(
            dim=dim, index=torch.randperm(self._data.shape[dim], generator=generator)
        )

    def shuffle_along_dim_(self, dim: int, generator: torch.Generator | None = None) -> None:
        r"""Shuffles the data/batch along a given dimension.

        Args:
            dim: Specifies the shuffle dimension.
            generator (``torch.Generator`` or ``None``, optional):
                Specifies an optional random generator.
                Default: ``None``

        Returns:
            ``BatchedTensor``:  A new batch with shuffled data
                along a given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.shuffle_along_dim_(dim=0)
            >>> batch
            tensor([[...]], batch_dim=0)
        """
        self._data = self.shuffle_along_dim(dim=dim, generator=generator).data

    def sort(self, *args: Any, **kwargs: Any) -> torch.return_types.sort:
        r"""Sorts the elements of the batch along a given dimension in
        monotonic order by value.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            (``BatchedTensor``, ``BatchedTensor``): A tuple of
                two values:
                    - The first batch contains the batch values sorted
                        along the given dimension.
                    - The second batch contains the indices that sort
                        the batch along the given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.sort(descending=True)
            torch.return_types.sort(
            values=tensor([[4, 3, 2, 1, 0],
                    [9, 8, 7, 6, 5]], batch_dim=0),
            indices=tensor([[4, 3, 2, 1, 0],
                    [4, 3, 2, 1, 0]], batch_dim=0))
        """
        out = torch.sort(self._data, *args, **kwargs)
        return type(out)([self._create_new_batch(o) for o in out])

    def sort_along_batch(self, *args: Any, **kwargs: Any) -> torch.return_types.sort:
        r"""Sorts the elements of the batch along the batch dimension in
        monotonic order by value.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            (``BatchedTensor``, ``BatchedTensor``): A tuple
                two values:
                    - The first batch contains the batch values sorted
                        along the given dimension.
                    - The second batch contains the indices that sort
                        the batch along the given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.sort_along_batch(descending=True)
            torch.return_types.sort(
            values=tensor([[5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4]], batch_dim=0),
            indices=tensor([[1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0]], batch_dim=0))
        """
        return self.sort(self._batch_dim, *args, **kwargs)

    ################################################
    #     Mathematical | point-wise operations     #
    ################################################

    def abs(self) -> Self:
        r"""Compute the absolute value of each element.

        Returns:
            ``BatchedTensor``: A batch with the absolute value of
                each element.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0]]))
            >>> batch.abs()
            tensor([[2., 0., 2.],
                    [1., 1., 3.]], batch_dim=0)
        """
        return self._create_new_batch(self._data.abs())

    def abs_(self) -> None:
        r"""Compute the absolute value of each element.

        In-place version of ``abs()``.

        Example usage:

        .. code-block:: pycon

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
        min: float | None = None,  # noqa: A002
        max: float | None = None,  # noqa: A002
    ) -> Self:
        r"""Clamps all elements in ``self`` into the range ``[min,
        max]``.

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

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.clamp(min=2, max=5)
            tensor([[2, 2, 2, 3, 4],
                    [5, 5, 5, 5, 5]], batch_dim=0)
            >>> batch.clamp(min=2)
            tensor([[2, 2, 2, 3, 4],
                    [5, 6, 7, 8, 9]], batch_dim=0)
            >>> batch.clamp(max=7)
            tensor([[0, 1, 2, 3, 4],
                    [5, 6, 7, 7, 7]], batch_dim=0)
        """
        return torch.clamp(self, min=min, max=max)

    def clamp_(
        self,
        min: float | None = None,  # noqa: A002
        max: float | None = None,  # noqa: A002
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

        .. code-block:: pycon

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

    def exp(self) -> Self:
        r"""Compute the exponential of the elements.

        Returns:
            ``BatchedTensor``: A batch with the exponential of the
                elements of the current batch.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
            >>> batch.exp()
            tensor([[  1.0000,   2.7183,   7.3891],
                    [ 20.0855,  54.5981, 148.4132]], batch_dim=0)
        """
        return torch.exp(self)

    def exp_(self) -> None:
        r"""Compute the exponential of the elements.

        In-place version of ``exp()``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
            >>> batch.exp_()
            >>> batch
            tensor([[  1.0000,   2.7183,   7.3891],
                    [ 20.0855,  54.5981, 148.4132]], batch_dim=0)
        """
        self._data.exp_()

    def log(self) -> BatchedTensor:
        r"""Compute the natural logarithm of the elements.

        Returns:
            ``BatchedTensor``: A batch with the natural
                logarithm of the elements of the current batch.

        Example usage:

        .. code-block:: pycon

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
        r"""Compute the natural logarithm of the elements.

        In-place version of ``log()``.

        Example usage:

        .. code-block:: pycon

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

    def log10(self) -> BatchedTensor:
        r"""Compute the logarithm to the base 10 of the elements.

        Returns:
            ``BatchedTensor``: A batch with the logarithm to the
                base 10 of the elements of the current batch.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]])
            ... )
            >>> batch.log10()
            tensor([[  -inf, 0.0000, 0.3010, 0.4771, 0.6021],
                    [0.6990, 0.7782, 0.8451, 0.9031, 0.9542]], batch_dim=0)
        """
        return torch.log10(self)

    def log10_(self) -> None:
        r"""Compute the logarithm to the base 10 of the elements.

        In-place version of ``log10()``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]])
            ... )
            >>> batch.log10_()
            >>> batch
            tensor([[  -inf, 0.0000, 0.3010, 0.4771, 0.6021],
                    [0.6990, 0.7782, 0.8451, 0.9031, 0.9542]], batch_dim=0)
        """
        self._data.log10_()

    def log1p(self) -> BatchedTensor:
        r"""Compute the natural logarithm of ``self + 1``.

        Returns:
            ``BatchedTensor``: A batch with the natural
                logarithm of ``self + 1``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]])
            ... )
            >>> batch.log1p()
            tensor([[0.0000, 0.6931, 1.0986, 1.3863, 1.6094],
                    [1.7918, 1.9459, 2.0794, 2.1972, 2.3026]], batch_dim=0)
        """
        return torch.log1p(self)

    def log1p_(self) -> None:
        r"""Compute the natural logarithm of ``self + 1``.

        In-place version of ``log1p()``.

        Example usage:

        .. code-block:: pycon

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

    def log2(self) -> BatchedTensor:
        r"""Compute the logarithm to the base 2 of the elements.

        Returns:
            ``BatchedTensor``: A batch with the logarithm to the
                base 2 of the elements of the current batch.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]])
            ... )
            >>> batch.log2()
            tensor([[  -inf, 0.0000, 1.0000, 1.5850, 2.0000],
                    [2.3219, 2.5850, 2.8074, 3.0000, 3.1699]], batch_dim=0)
        """
        return torch.log2(self)

    def log2_(self) -> None:
        r"""Compute the logarithm to the base 2 of the elements.

        In-place version of ``log2()``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]])
            ... )
            >>> batch.log2_()
            >>> batch
            tensor([[  -inf, 0.0000, 1.0000, 1.5850, 2.0000],
                    [2.3219, 2.5850, 2.8074, 3.0000, 3.1699]], batch_dim=0)
        """
        self._data.log2_()

    def maximum(self, other: BatchedTensor | Tensor) -> Self:
        r"""Compute the element-wise maximum of ``self`` and ``other``.

        Args:
            other (``BatchedTensor`` or ``torch.Tensor``): Specifies
                a batch.

        Returns:
            ``BatchedTensor``: The batch with the element-wise
                maximum of ``self`` and ``other``

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(6).view(2, 3))
            >>> batch.maximum(BatchedTensor(torch.tensor([[1, 0, 2], [4, 5, 3]])))
            tensor([[1, 1, 2],
                    [4, 5, 5]], batch_dim=0)
        """
        self._check_valid_dims((self, other))
        if isinstance(other, BatchedTensor):
            other = other.data
        return self._create_new_batch(torch.maximum(self._data, other))

    def minimum(self, other: BatchedTensor | Tensor) -> Self:
        r"""Compute the element-wise minimum of ``self`` and ``other``.

        Args:
            other (``BatchedTensor`` or ``torch.Tensor``): Specifies
                a batch.

        Returns:
            ``BatchedTensor``: The batch with the element-wise
                minimum of ``self`` and ``other``

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(6).view(2, 3))
            >>> batch.minimum(BatchedTensor(torch.tensor([[1, 0, 2], [4, 5, 3]])))
            tensor([[0, 0, 2],
                    [3, 4, 3]], batch_dim=0)
        """
        self._check_valid_dims((self, other))
        if isinstance(other, BatchedTensor):
            other = other.data
        return self._create_new_batch(torch.minimum(self._data, other))

    def pow(self, exponent: float | BatchedTensor) -> Self:
        r"""Compute the power of each element with the given exponent.

        Args:
            exponent (int or float or ``BatchedTensor``): Specifies
                the exponent value. ``exponent`` can be either a single
                numeric number or a ``BatchedTensor`` with the same
                number of elements.

        Returns:
            ``BatchedTensor``: A batch with the power of each
                element with the given exponent.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
            >>> batch.pow(2)
            tensor([[ 0.,  1.,  4.],
                    [ 9., 16., 25.]], batch_dim=0)
        """
        return torch.pow(self, exponent)

    def pow_(self, exponent: float | BatchedTensor) -> None:
        r"""Compute the power of each element with the given exponent.

        In-place version of ``pow(exponent)``.

        Args:
            exponent (int or float or ``BatchedTensor``): Specifies
                the exponent value. ``exponent`` can be either a
                single numeric number or a ``BatchedTensor``
                with the same number of elements.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
            >>> batch.pow_(2)
            >>> batch
            tensor([[ 0.,  1.,  4.],
                    [ 9., 16., 25.]], batch_dim=0)
        """
        self._check_valid_dims((self, exponent))
        self._data.pow_(exponent)

    def rsqrt(self) -> Self:
        r"""Compute the reciprocal of the square-root of each element.

        Returns:
            ``BatchedTensor``: A batch with the square-root of
                each element.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[1.0, 4.0], [16.0, 25.0]]))
            >>> batch.rsqrt()
            tensor([[1.0000, 0.5000],
                    [0.2500, 0.2000]], batch_dim=0)
        """
        return torch.rsqrt(self)

    def rsqrt_(self) -> None:
        r"""Compute the reciprocal of the square-root of each element.

        In-place version of ``rsqrt()``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[1.0, 4.0], [16.0, 25.0]]))
            >>> batch.rsqrt_()
            >>> batch
            tensor([[1.0000, 0.5000],
                    [0.2500, 0.2000]], batch_dim=0)
        """
        self._data.rsqrt_()

    def sqrt(self) -> Self:
        r"""Compute the square-root of each element.

        Returns:
            ``BatchedTensor``: A batch with the square-root of
                each element.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]))
            >>> batch.sqrt()
            tensor([[0., 1., 2.],
                    [3., 4., 5.]], batch_dim=0)
        """
        return torch.sqrt(self)

    def sqrt_(self) -> None:
        r"""Compute the square-root of each element.

        In-place version of ``sqrt()``.

        Example usage:

        .. code-block:: pycon

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

    def amax(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the maximum value of all elements or along a
        dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.amax``
            **kwargs: See the documentation of ``torch.Tensor.amax``

        Returns:
            ``torch.Tensor``: The maximum value of all elements or
                along a dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.amax(dim=1)
            tensor([4, 9])
            >>> batch.amax(dim=1, keepdim=True)
            tensor([[4], [9]])
            >>> batch.amax(dim=(0, 1))
            tensor(9)
            >>> batch.amax(dim=None)
            tensor(9)
        """
        return self._data.amax(*args, **kwargs)

    def amax_along_batch(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the maximum along the batch dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.amax``
            **kwargs: See the documentation of ``torch.Tensor.amax``

        Returns:
            ``torch.Tensor``: The maximum along the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.amax_along_batch()
            tensor([8, 9])
            >>> batch.amax_along_batch(keepdim=True)
            tensor([[8, 9]])
        """
        return self.amax(self._batch_dim, *args, **kwargs)

    def amin(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the minimum value of all elements or along a
        dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.amin``
            **kwargs: See the documentation of ``torch.Tensor.amin``

        Returns:
            ``torch.Tensor``: The minimum value of all elements or
                along a dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.amin(dim=1)
            tensor([0, 5])
            >>> batch.amin(dim=1, keepdim=True)
            tensor([[0], [5]])
            >>> batch.amin(dim=(0, 1))
            tensor(0)
            >>> batch.amin(dim=None)
            tensor(0)
        """
        return self._data.amin(*args, **kwargs)

    def amin_along_batch(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the minimum along the batch dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.amin``
            **kwargs: See the documentation of ``torch.Tensor.amin``

        Returns:
            ``torch.Tensor``: The minimum along the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.amin_along_batch()
            tensor([0, 1])
            >>> batch.amin_along_batch(keepdim=True)
            tensor([[0, 1]])
        """
        return self.amin(self._batch_dim, *args, **kwargs)

    def argmax(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the indices of the maximum value of all elements or
        along a specific dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.argmax``
            **kwargs: See the documentation of ``torch.Tensor.argmax``

        Returns:
            ``torch.Tensor``: The indices of the maximum value of all
                elements or along a specific dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.argmax(dim=1)
            tensor([4, 4])
            >>> batch.argmax(dim=1, keepdim=True)
            tensor([[4], [4]])
            >>> batch.argmax()
            tensor(9)
        """
        return self._data.argmax(*args, **kwargs)

    def argmax_along_batch(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the indices of the maximum value along the batch
        dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.argmax``
            **kwargs: See the documentation of ``torch.Tensor.argmax``

        Returns:
            ``torch.Tensor``: The indices of the maximum value along
                the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.argmax_along_batch()
            tensor([4, 4])
            >>> batch.argmax_along_batch(keepdim=True)
            tensor([[4, 4]])
        """
        return self.argmax(self._batch_dim, *args, **kwargs)

    def argmin(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the indices of the minimum value of all elements or
        along a specific dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.argmin``
            **kwargs: See the documentation of ``torch.Tensor.argmin``

        Returns:
            ``torch.Tensor``: The indices of the minimum value of all
                elements or along a specific dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.argmin(dim=1)
            tensor([0, 0])
            >>> batch.argmin(dim=1, keepdim=True)
            tensor([[0], [0]])
            >>> batch.argmin()
            tensor(0)
        """
        return self._data.argmin(*args, **kwargs)

    def argmin_along_batch(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the indices of the minimum value along the batch
        dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.argmin``
            **kwargs: See the documentation of ``torch.Tensor.argmin``

        Returns:
            ``torch.Tensor``: The indices of the minimum value along
                the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.argmin_along_batch()
            tensor([0, 0])
            >>> batch.argmin_along_batch(keepdim=True)
            tensor([[0, 0]])
        """
        return self.argmin(self._batch_dim, *args, **kwargs)

    def max(self, *args: Any, **kwargs: Any) -> Tensor | torch.return_types.max:
        r"""Compute the maximum of all elements or along a dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.max``
            **kwargs: See the documentation of ``torch.Tensor.max``

        Returns:
            ``torch.Tensor`` or ``torch.return_types.max``:
                The maximum of all elements or along a dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.max()
            tensor(9)
            >>> batch.max(dim=1)
            torch.return_types.max(
            values=tensor([4, 9]),
            indices=tensor([4, 4]))
            >>> batch.max(dim=1, keepdim=True)
            torch.return_types.max(
            values=tensor([[4], [9]]),
            indices=tensor([[4], [4]]))
        """
        return self._data.max(*args, **kwargs)

    def max_along_batch(self, *args: Any, **kwargs: Any) -> torch.return_types.max:
        r"""Compute the maximum values along the batch dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.max``
            **kwargs: See the documentation of ``torch.Tensor.max``

        Returns:
            ``torch.return_types.max``: A batch with
                the maximum values along the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]))
            >>> batch.max_along_batch()
            torch.return_types.max(
            values=tensor([4, 9]),
            indices=tensor([4, 4]))
            >>> batch.max_along_batch(keepdim=True)
            torch.return_types.max(
            values=tensor([[4, 9]]),
            indices=tensor([[4, 4]]))
        """
        return self.max(self._batch_dim, *args, **kwargs)

    def mean(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the mean of all elements.

        Args:
            *args: See the documentation of ``torch.Tensor.mean``
            **kwargs: See the documentation of ``torch.Tensor.mean``

        Returns:
            ``torch.Tensor``: The mean of all elements.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5).float())
            >>> batch.mean()
            tensor(4.5000)
            >>> batch.mean(dim=1)
            tensor([2., 7.])
            >>> batch.mean(dim=1, keepdim=True)
            tensor([[2.], [7.]])
        """
        return self._data.mean(*args, **kwargs)

    def mean_along_batch(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the mean values along the batch dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.mean``
            **kwargs: See the documentation of ``torch.Tensor.mean``

        Returns:
            ``torch.Tensor``: A batch with
                the mean values along the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2).float())
            >>> batch.mean_along_batch()
            tensor([4., 5.])
            >>> batch.mean_along_batch(keepdim=True)
            tensor([[4., 5.]])
        """
        return self.mean(self._batch_dim, *args, **kwargs)

    def median(self, *args: Any, **kwargs: Any) -> Tensor | torch.return_types.median:
        r"""Compute the median of all elements.

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

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.median()
            tensor(4)
            >>> batch.median(dim=1)
            torch.return_types.median(
            values=tensor([2, 7]),
            indices=tensor([2, 2]))
            >>> batch.median(dim=1, keepdim=True)
            torch.return_types.median(
            values=tensor([[2], [7]]),
            indices=tensor([[2], [2]]))
        """
        return self._data.median(*args, **kwargs)

    def median_along_batch(self, *args: Any, **kwargs: Any) -> torch.return_types.median:
        r"""Compute the median values along the batch dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.median``
            **kwargs: See the documentation of ``torch.Tensor.median``

        Returns:
            ``torch.return_types.median``:  The first tensor will
                be populated with the median values and the second
                tensor, which must have dtype long, with their indices
                in the batch dimension of input.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]))
            >>> batch.median_along_batch()
            torch.return_types.median(
            values=tensor([2, 7]),
            indices=tensor([2, 2]))
        """
        return self.median(self._batch_dim, *args, **kwargs)

    def min(self, *args: Any, **kwargs: Any) -> Tensor | torch.return_types.min:
        r"""Compute the minimum of all elements or along a dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.min``
            **kwargs: See the documentation of ``torch.Tensor.min``

        Returns:
            ``torch.Tensor`` or ``torch.return_types.min``:
                The minimum of all elements or along a dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.min()
            tensor(0)
            >>> batch.min(dim=1)
            torch.return_types.min(
            values=tensor([0, 5]),
            indices=tensor([0, 0]))
            >>> batch.min(dim=1, keepdim=True)
            torch.return_types.min(
            values=tensor([[0], [5]]),
            indices=tensor([[0], [0]]))
        """
        return self._data.min(*args, **kwargs)

    def min_along_batch(self, *args: Any, **kwargs: Any) -> torch.return_types.min:
        r"""Compute the minimum values along the batch dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.min``
            **kwargs: See the documentation of ``torch.Tensor.min``

        Returns:
            ``torch.return_types.min``: A batch with
                the minimum values along the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]))
            >>> batch.min_along_batch()
            torch.return_types.min(
            values=tensor([0, 5]),
            indices=tensor([0, 0]))
            >>> batch.min_along_batch(keepdim=True)
            torch.return_types.min(
            values=tensor([[0, 5]]),
            indices=tensor([[0, 0]]))
        """
        return self.min(self._batch_dim, *args, **kwargs)

    def nanmean(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the mean of all elements.

        Args:
            *args: See the documentation of ``torch.Tensor.nanmean``
            **kwargs: See the documentation of ``torch.Tensor.nanmean``

        Returns:
            ``torch.Tensor``: The mean of all elements.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
            >>> batch.nanmean()
            tensor(4.)
            >>> batch.nanmean(dim=1)
            tensor([2.0000, 6.5000])
            >>> batch.nanmean(dim=1, keepdim=True)
            tensor([[2.0000], [6.5000]])
        """
        return self._data.nanmean(*args, **kwargs)

    def nanmean_along_batch(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the mean values along the batch dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.nanmean``
            **kwargs: See the documentation of ``torch.Tensor.nanmean``

        Returns:
            ``torch.Tensor``: A batch with
                the mean values along the batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]])
            ... )
            >>> batch.nanmean_along_batch()
            tensor([2.0000, 6.5000])
            >>> batch.nanmean_along_batch(keepdim=True)
            tensor([[2.0000, 6.5000]])
        """
        return self.nanmean(self._batch_dim, *args, **kwargs)

    def nanmedian(self, *args: Any, **kwargs: Any) -> Tensor | torch.return_types.nanmedian:
        r"""Compute the median of all elements.

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

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
            >>> batch.nanmedian()
            tensor(4.)
            >>> batch.nanmedian(dim=1)
            torch.return_types.nanmedian(
            values=tensor([2., 6.]),
            indices=tensor([2, 1]))
            >>> batch.nanmedian(dim=1, keepdim=True)
            torch.return_types.nanmedian(
            values=tensor([[2.], [6.]]),
            indices=tensor([[2], [1]]))
        """
        return self._data.nanmedian(*args, **kwargs)

    def nanmedian_along_batch(self, *args: Any, **kwargs: Any) -> torch.return_types.nanmedian:
        r"""Compute the median values along the batch dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.nanmedian``
            **kwargs: See the documentation of ``torch.Tensor.nanmedian``

        Returns:
            ``torch.return_types.nanmedian``:  The first tensor will
                be populated with the median values and the second
                tensor, which must have dtype long, with their indices
                in the batch dimension of input.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]])
            ... )
            >>> batch.nanmedian_along_batch()
            torch.return_types.nanmedian(
            values=tensor([2., 6.]),
            indices=tensor([2, 1]))
        """
        return self.nanmedian(self._batch_dim, *args, **kwargs)

    def nansum(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the sum of all elements.

        Args:
            *args: See the documentation of ``torch.Tensor.nansum``
            **kwargs: See the documentation of ``torch.Tensor.nansum``

        Returns:
            ``torch.Tensor``: The sum of all elements.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
            >>> batch.nansum()
            tensor(36.)
            >>> batch.nansum(dim=1)
            tensor([10., 26.])
            >>> batch.nansum(dim=1, keepdim=True)
            tensor([[10.], [26.]])
        """
        return self._data.nansum(*args, **kwargs)

    def nansum_along_batch(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the sum values along the batch dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.nansum``
            **kwargs: See the documentation of ``torch.Tensor.nansum``

        Returns:
            ``torch.Tensor``: A tensor with the sum values along the
                batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]])
            ... )
            >>> batch.nansum_along_batch()
            tensor([10., 26.])
        """
        return self.nansum(self._batch_dim, *args, **kwargs)

    def prod(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the product of all elements.

        Args:
            *args: See the documentation of ``torch.Tensor.prod``
            **kwargs: See the documentation of ``torch.Tensor.prod``

        Returns:
            ``torch.Tensor``: The product of all elements.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]]))
            >>> batch.prod()
            tensor(362880)
            >>> batch.prod(dim=1)
            tensor([ 120, 3024])
            >>> batch.prod(dim=1, keepdim=True)
            tensor([[ 120], [3024]])
        """
        return self._data.prod(*args, **kwargs)

    def prod_along_batch(self, keepdim: bool = False) -> Tensor:
        r"""Compute the product values along the batch dimension.

        Args:
            keepdim: Indicates if the output tensor has ``dim``
                retained or not.

        Returns:
            A tensor with the product values along the batch dimension.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.tensor([[1, 6], [2, 7], [3, 8], [4, 9], [5, 1]]))
        >>> batch.prod_along_batch()
        tensor([ 120, 3024])
        >>> batch.prod_along_batch(keepdim=True)
        tensor([[ 120, 3024]])

        ```
        """
        return self.prod(dim=self._batch_dim, keepdim=keepdim)

    def sum(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the sum of all elements.

        Args:
            *args: See the documentation of ``torch.Tensor.sum``
            **kwargs: See the documentation of ``torch.Tensor.sum``

        Returns:
            ``torch.Tensor``: The sum of all elements.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.sum()
            tensor(45)
            >>> batch.sum(dim=1)
            tensor([10, 35])
            >>> batch.sum(dim=1, keepdim=True)
            tensor([[10], [35]])
        """
        return self._data.sum(*args, **kwargs)

    def sum_along_batch(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Compute the sum values along the batch dimension.

        Args:
            *args: See the documentation of ``torch.Tensor.sum``
            **kwargs: See the documentation of ``torch.Tensor.sum``

        Returns:
            ``torch.Tensor``: A tensor with the sum values along the
                batch dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.sum_along_batch()
            tensor([20, 25])
        """
        return self.sum(self._batch_dim, *args, **kwargs)

    ###########################################
    #     Mathematical | trigo operations     #
    ###########################################

    def acos(self) -> Self:
        r"""Compute the inverse cosine (arccos) of each element.

        Returns:
            ``BatchedTensor``: A batch with the inverse cosine
                (arccos) of each element.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.acos()
            tensor([[3.1416, 1.5708, 0.0000],
                    [2.0944, 1.5708, 1.0472]], batch_dim=0)
        """
        return torch.acos(self)

    def acos_(self) -> None:
        r"""Compute the inverse cosine (arccos) of each element.

        In-place version of ``acos()``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.acos_()
            >>> batch
            tensor([[3.1416, 1.5708, 0.0000],
                    [2.0944, 1.5708, 1.0472]], batch_dim=0)
        """
        self._data.acos_()

    def acosh(self) -> Self:
        r"""Compute the inverse hyperbolic cosine (arccosh) of each
        element.

        Returns:
            ``BatchedTensor``: A batch with the inverse hyperbolic
                cosine (arccosh) of each element.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            >>> batch.acosh()
            tensor([[0.0000, 1.3170, 1.7627],
                    [2.0634, 2.2924, 2.4779]], batch_dim=0)
        """
        return torch.acosh(self)

    def acosh_(self) -> None:
        r"""Compute the inverse hyperbolic cosine (arccosh) of each
        element.

        Returns:
            ``BatchedTensor``: A batch with the inverse hyperbolic
                cosine (arccosh) of each element.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            >>> batch.acosh_()
            >>> batch
            tensor([[0.0000, 1.3170, 1.7627],
                    [2.0634, 2.2924, 2.4779]], batch_dim=0)
        """
        self._data.acosh_()

    def asin(self) -> Self:
        r"""Compute the inverse cosine (arcsin) of each element.

        Returns:
            ``BatchedTensor``: A batch with the inverse sine
                (arcsin) of each element.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.asin()
            tensor([[-1.5708,  0.0000,  1.5708],
                    [-0.5236,  0.0000,  0.5236]], batch_dim=0)
        """
        return torch.asin(self)

    def asin_(self) -> None:
        r"""Compute the inverse sine (arcsin) of each element.

        In-place version of ``asin()``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.asin_()
            >>> batch
            tensor([[-1.5708,  0.0000,  1.5708],
                    [-0.5236,  0.0000,  0.5236]], batch_dim=0)
        """
        self._data.asin_()

    def asinh(self) -> Self:
        r"""Compute the inverse hyperbolic sine (arcsinh) of each
        element.

        Returns:
            ``BatchedTensor``: A batch with the inverse hyperbolic
                sine (arcsinh) of each element.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.asinh()
            tensor([[-0.8814,  0.0000,  0.8814],
                    [-0.4812,  0.0000,  0.4812]], batch_dim=0)
        """
        return torch.asinh(self)

    def asinh_(self) -> None:
        r"""Compute the inverse hyperbolic sine (arcsinh) of each
        element.

        In-place version of ``asinh()``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.asinh_()
            >>> batch
            tensor([[-0.8814,  0.0000,  0.8814],
                    [-0.4812,  0.0000,  0.4812]], batch_dim=0)
        """
        self._data.asinh_()

    def atan(self) -> Self:
        r"""Compute the inverse tangent of each element.

        Returns:
            ``BatchedTensor``: A batch with the inverse tangent
                of each element.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]]))
            >>> batch.atan()
            tensor([[ 0.0000,  0.7854,  1.1071],
                    [-1.1071, -0.7854,  0.0000]], batch_dim=0)
        """
        return torch.atan(self)

    def atan_(self) -> None:
        r"""Compute the inverse tangent of each element.

        In-place version of ``atan()``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]]))
            >>> batch.atan_()
            >>> batch
            tensor([[ 0.0000,  0.7854,  1.1071],
                    [-1.1071, -0.7854,  0.0000]], batch_dim=0)
        """
        self._data.atan_()

    def atanh(self) -> Self:
        r"""Compute the inverse hyperbolic tangent of each element.

        Returns:
            ``BatchedTensor``: A batch with the inverse hyperbolic
                tangent of each element.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-0.5, 0.0, 0.5], [-0.1, 0.0, 0.1]]))
            >>> batch.atanh()
            tensor([[-0.5493,  0.0000,  0.5493],
                    [-0.1003,  0.0000,  0.1003]], batch_dim=0)
        """
        return torch.atanh(self)

    def atanh_(self) -> None:
        r"""Compute the inverse hyperbolic tangent of each element.

        In-place version of ``atanh()``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-0.5, 0.0, 0.5], [-0.1, 0.0, 0.1]]))
            >>> batch.atanh_()
            >>> batch
            tensor([[-0.5493,  0.0000,  0.5493],
                    [-0.1003,  0.0000,  0.1003]], batch_dim=0)
        """
        self._data.atanh_()

    def cos(self) -> Self:
        r"""Compute the cosine of each element.

        Returns:
            ``BatchedTensor``: A batch with the cosine of each
                element.

        Example usage:

        .. code-block:: pycon

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
        r"""Compute the cosine of each element.

        In-place version of ``cos()``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> import math
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]])
            ... )
            >>> batch.cos_()
            >>> batch
            tensor([[ 1.0000e+00, -4.3711e-08, -1.0000e+00],
                    [ 1.0000e+00,  1.1925e-08,  1.0000e+00]], batch_dim=0)
        """
        self._data.cos_()

    def cosh(self) -> Self:
        r"""Compute the hyperbolic cosine (cosh) of each element.

        Returns:
            ``BatchedTensor``: A batch with the hyperbolic
                cosine (arccosh) of each element.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[1.0, 2.1, 3.2], [4.0, 5.1, 6.2]]))
            >>> batch.cosh()
            tensor([[  1.5431,   4.1443,  12.2866],
                    [ 27.3082,  82.0140, 246.3755]], batch_dim=0)
        """
        return torch.cosh(self)

    def cosh_(self) -> None:
        r"""Compute the hyperbolic cosine (arccosh) of each element.

        In-place version of ``cosh()``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[1.0, 2.1, 3.2], [4.0, 5.1, 6.2]]))
            >>> batch.cosh_()
            >>> batch
            tensor([[  1.5431,   4.1443,  12.2866],
                    [ 27.3082,  82.0140, 246.3755]], batch_dim=0)
        """
        self._data.cosh_()

    def sin(self) -> Self:
        r"""Compute the sine of each element.

        Returns:
            ``BatchedTensor``: A batch with the sine of each
                element.

        Example usage:

        .. code-block:: pycon

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
        r"""Compute the sine of each element.

        In-place version of ``sin()``.

        Example usage:

        .. code-block:: pycon

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

    def sinh(self) -> Self:
        r"""Compute the hyperbolic sine of each element.

        Returns:
            ``BatchedTensor``: A batch with the hyperbolic sine of
                each element.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.sinh()
            tensor([[-1.1752,  0.0000,  1.1752],
                    [-0.5211,  0.0000,  0.5211]], batch_dim=0)
        """
        return torch.sinh(self)

    def sinh_(self) -> None:
        r"""Compute the hyperbolic sine of each element.

        In-place version of ``sinh()``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
            >>> batch.sinh_()
            >>> batch
            tensor([[-1.1752,  0.0000,  1.1752],
                    [-0.5211,  0.0000,  0.5211]], batch_dim=0)
        """
        self._data.sinh_()

    def tan(self) -> Self:
        r"""Compute the tangent of each element.

        Returns:
            ``BatchedTensor``: A batch with the tangent of each
                element.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> import math
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor(
            ...         [[0.0, 0.25 * math.pi, math.pi], [2 * math.pi, 1.75 * math.pi, -0.25 * math.pi]]
            ...     )
            ... )
            >>> batch.tan()
            tensor([[ 0.0000e+00,  1.0000e+00,  8.7423e-08],
                    [ 1.7485e-07, -1.0000e+00, -1.0000e+00]], batch_dim=0)
        """
        return torch.tan(self)

    def tan_(self) -> None:
        r"""Compute the tangent of each element.

        In-place version of ``tan()``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> import math
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(
            ...     torch.tensor(
            ...         [[0.0, 0.25 * math.pi, math.pi], [2 * math.pi, 1.75 * math.pi, -0.25 * math.pi]]
            ...     )
            ... )
            >>> batch.tan_()
            >>> batch
            tensor([[ 0.0000e+00,  1.0000e+00,  8.7423e-08],
                    [ 1.7485e-07, -1.0000e+00, -1.0000e+00]], batch_dim=0)
        """
        self._data.tan_()

    def tanh(self) -> Self:
        r"""Compute the tangent of each element.

        Returns:
            ``BatchedTensor``: A batch with the tangent of each
                element.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]]))
            >>> batch.tanh()
            tensor([[ 0.0000,  0.7616,  0.9640],
                    [-0.9640, -0.7616,  0.0000]], batch_dim=0)
        """
        return torch.tanh(self)

    def tanh_(self) -> None:
        r"""Compute the tangent of each element.

        In-place version of ``tanh()``.

        Example usage:

        .. code-block:: pycon

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

    def logical_and(self, other: BatchedTensor | Tensor) -> Self:
        r"""Compute the element-wise logical AND.

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

        .. code-block:: pycon

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
        r"""Compute the element-wise logical AND.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BatchedTensor`` or ``torch.Tensor``):
                Specifies the batch or tensor to compute
                logical AND with.

        Example usage:

        .. code-block:: pycon

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
        self._check_valid_dims((self, other))
        self._data.logical_and_(other)

    def logical_not(self) -> Self:
        r"""Compute the element-wise logical NOT of the current batch.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                logical NOT of the current batch.

        Example usage:

        .. code-block:: pycon

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
        r"""Compute the element-wise logical NOT of the current batch.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Example usage:

        .. code-block:: pycon

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

    def logical_or(self, other: BatchedTensor | Tensor) -> Self:
        r"""Compute the element-wise logical OR.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BatchedTensor`` or ``torch.Tensor``):
                Specifies the batch to compute logical OR with.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                logical OR.

        Example usage:

        .. code-block:: pycon

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
        r"""Compute the element-wise logical OR.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BatchedTensor`` or ``torch.Tensor``):
                Specifies the batch to compute logical OR with.

        Example usage:

        .. code-block:: pycon

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
        self._check_valid_dims((self, other))
        self._data.logical_or_(other)

    def logical_xor(self, other: BatchedTensor | Tensor) -> Self:
        r"""Compute the element-wise logical XOR.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BatchedTensor`` or ``torch.Tensor``):
                Specifies the batch to compute logical XOR with.

        Returns:
            ``BatchedTensor``: A batch containing the element-wise
                logical XOR.

        Example usage:

        .. code-block:: pycon

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
        r"""Compute the element-wise logical XOR.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BatchedTensor`` or ``torch.Tensor``):
                Specifies the batch to compute logical XOR with.

        Example usage:

        .. code-block:: pycon

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
        self._check_valid_dims((self, other))
        self._data.logical_xor_(other)

    ##########################################################
    #    Indexing, slicing, joining, mutating operations     #
    ##########################################################

    def __getitem__(self, index: IndexType) -> Tensor:
        if isinstance(index, BatchedTensor):
            index = index.data
        return self._data[index]

    def __setitem__(self, index: IndexType, value: bool | float | Tensor | BatchedTensor) -> None:
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
    ) -> Self:
        r"""Concatenate the data of the batches to the current batch
        along a given dimension and creates a new batch.

        Args:
            tensors: Specifies the batches to concatenate.
                Non-empty tensors provided must have the same shape,
                except in the cat dimension.
            dim: The dimension over which the tensors are concatenated.

        Returns:
            A batch with the concatenated data in the batch dimension.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
        >>> batch.cat(BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])))
        tensor([[ 0,  1,  2],
                [ 4,  5,  6],
                [10, 11, 12],
                [13, 14, 15]], batch_dim=0)

        ```
        """
        if isinstance(tensors, (BatchedTensor, Tensor)):
            tensors = [tensors]
        tensors = list(chain([self], tensors))
        self._check_valid_dims(tensors)
        return self._create_new_batch(
            torch.cat(
                [
                    tensor.data if isinstance(tensor, BatchedTensor) else tensor
                    for tensor in tensors
                ],
                dim=dim,
            ),
        )

    def cat_(
        self,
        tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
        dim: int = 0,
    ) -> None:
        r"""Concatenate the data of the batches to the current batch
        along a given dimension and creates a new batch.

        Args:
            tensors: Specifies the tensors to concatenate.
                Non-empty tensors provided must have the same shape,
                except in the cat dimension.
            dim: The dimension over which the tensors are concatenated.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
        >>> batch.cat_(BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])))
        >>> batch
        tensor([[ 0,  1,  2],
                [ 4,  5,  6],
                [10, 11, 12],
                [13, 14, 15]], batch_dim=0)

        ```
        """
        self._data = self.cat(tensors, dim=dim).data

    def cat_along_batch(
        self, tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor]
    ) -> Self:
        r"""Concatenates the data of the batches to the current batch
        along the batch dimension and creates a new batch.

        Args:
            tensors: Specifies the batches to concatenate.

        Returns:
            A batch with the concatenated data in the batch dimension.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
        >>> batch.cat_along_batch(BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])))
        tensor([[ 0,  1,  2],
                [ 4,  5,  6],
                [10, 11, 12],
                [13, 14, 15]], batch_dim=0)
        >>> batch = BatchedTensor(torch.tensor([[0, 4], [1, 5], [2, 6]]))
        >>> batch.cat_along_batch(
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

        ```
        """
        return self.cat(tensors, dim=self._batch_dim)

    def cat_along_batch_(
        self, tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor]
    ) -> None:
        r"""Concatenates the data of the batches to the current batch
        along the batch dimension and creates a new batch.

        Args:
            tensors: Specifies the batches to concatenate.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
        >>> batch.cat_along_batch_(BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])))
        >>> batch
        tensor([[ 0,  1,  2],
                [ 4,  5,  6],
                [10, 11, 12],
                [13, 14, 15]], batch_dim=0)
        >>> batch = BatchedTensor(torch.tensor([[0, 4], [1, 5], [2, 6]]))
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

        ```
        """
        self.cat_(tensors, dim=self._batch_dim)

    def chunk(self, chunks: int, dim: int = 0) -> tuple[Self, ...]:
        r"""Split the batch into chunks along a given dimension.

        Args:
            chunks: Specifies the number of chunks.
            dim: Specifies the dimension along which
                to split the tensor. Default: ``0``

        Returns:
            tuple: The batch split into chunks along the given
                dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.chunk(chunks=3)
            (tensor([[0, 1], [2, 3]], batch_dim=0),
             tensor([[4, 5], [6, 7]], batch_dim=0),
             tensor([[8, 9]], batch_dim=0))
        """
        return tuple(self._create_new_batch(chunk) for chunk in self._data.chunk(chunks, dim=dim))

    def chunk_along_batch(self, chunks: int) -> tuple[Self, ...]:
        return self.chunk(chunks, self._batch_dim)

    def extend(self, other: Iterable[BatchedTensor]) -> None:
        self.cat_along_batch_(other)

    def index_select(self, dim: int, index: Tensor | Sequence[int]) -> Self:
        r"""Select data at the given indices along a given dimension.

        Args:
            dim: Specifies the index dimension.
            index (``torch.Tensor`` or list or tuple): Specifies the
                indices to select.

        Returns:
            ``BatchedTensor``: A new batch which indexes ``self``
                along the batch dimension using the entries in
                ``index``.

        Example usage:

        .. code-block:: pycon

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
        return self._create_new_batch(self._data.index_select(dim, to_tensor(index)))

    def index_select_along_batch(self, index: Tensor | Sequence[int]) -> Self:
        return self.index_select(self._batch_dim, index)

    def masked_fill(self, mask: BatchedTensor | Tensor, value: bool | float) -> Self:
        r"""Fill elements of ``self`` batch with ``value`` where ``mask``
        is ``True``.

        Args:
            mask (``BatchedTensor`` or ``torch.Tensor``):
                Specifies the batch of boolean masks.
            value (number): Specifies the value to fill in with.

        Returns:
            ``BatchedTensor``: A new batch with the updated values.

        Example usage:

        .. code-block:: pycon

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
        self._check_valid_dims((self, mask))
        return self._create_new_batch(self._data.masked_fill(to_tensor(mask.data), value))

    def select(self, dim: int, index: int) -> Tensor:
        r"""Select the batch along the batch dimension at the given
        index.

        Args:
            dim: Specifies the index dimension.
            index: Specifies the index to select.

        Returns:
            ``Tensor``: The batch sliced along the batch
                dimension at the given index.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.select(dim=0, index=2)
            tensor([4, 5])
        """
        return self._data.select(dim=dim, index=index)

    def select_along_batch(self, index: int) -> Tensor:
        return self.select(self._batch_dim, index)

    def slice_along_batch(self, start: int = 0, stop: int | None = None, step: int = 1) -> Self:
        return self.slice_along_dim(self._batch_dim, start, stop, step)

    def slice_along_dim(
        self,
        dim: int = 0,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Self:
        r"""Slices the batch in a given dimension.

        Args:
            dim: Specifies the dimension along which
                to slice the tensor. Default: ``0``
            start: Specifies the index where the
                slicing of object starts. Default: ``0``
            stop: Specifies the index where the
                slicing of object stops. ``None`` means last.
                Default: ``None``
            step: Specifies the increment between
                each index for slicing. Default: ``1``

        Returns:
            ``BatchedTensor``: A slice of the current batch.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.slice_along_dim(start=2)
            tensor([[4, 5],
                    [6, 7],
                    [8, 9]], batch_dim=0)
            >>> batch.slice_along_dim(stop=3)
            tensor([[0, 1],
                    [2, 3],
                    [4, 5]], batch_dim=0)
            >>> batch.slice_along_dim(step=2)
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

    def split(self, split_size_or_sections: int | Sequence[int], dim: int = 0) -> tuple[Self, ...]:
        r"""Split the batch into chunks along a given dimension.

        Args:
            split_size_or_sections (int or sequence): Specifies the
                size of a single chunk or list of sizes for each chunk.
            dim: Specifies the dimension along which
                to split the tensor. Default: ``0``

        Returns:
            tuple: The batch split into chunks along the given
                dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.split(2, dim=0)
            (tensor([[0, 1], [2, 3]], batch_dim=0),
             tensor([[4, 5], [6, 7]], batch_dim=0),
             tensor([[8, 9]], batch_dim=0))
        """
        return tuple(
            self._create_new_batch(chunk)
            for chunk in self._data.split(split_size_or_sections, dim=dim)
        )

    def split_along_batch(
        self, split_size_or_sections: int | Sequence[int]
    ) -> tuple[BatchedTensor, ...]:
        return self.split(split_size_or_sections, dim=self._batch_dim)

    def take_along_batch(self, indices: BaseBatch | np.ndarray | Tensor | Sequence) -> Self:
        r"""Take values along the batch dimension.

        Args:
            indices (``BaseBatch`` or ``Tensor`` or sequence):
                Specifies the indices to take along the batch
                dimension.

        Returns:
            ``BatchedTensor``: The batch with the selected data.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.take_along_batch(BatchedTensor(torch.tensor([[3, 2], [0, 3], [1, 4]])))
            tensor([[6, 5],
                    [0, 7],
                    [2, 9]], batch_dim=0)
        """
        return self.take_along_dim(indices, dim=self._batch_dim)

    @overload
    def take_along_dim(
        self,
        indices: BaseBatch | np.ndarray | Tensor | Sequence,
        dim: None = None,
    ) -> Tensor: ...  # pragma: no cover

    @overload
    def take_along_dim(
        self,
        indices: BaseBatch | np.ndarray | Tensor | Sequence,
        dim: int,
    ) -> Self: ...  # pragma: no cover

    def take_along_dim(
        self,
        indices: BaseBatch | np.ndarray | Tensor | Sequence,
        dim: int | None = None,
    ) -> Self | Tensor:
        r"""Take values along the batch dimension.

        Args:
            indices: Specifies the indices to take along the batch
                dimension.
            dim: Specifies the dimension to select along.

        Returns:
            The batch with the selected data.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
        >>> batch.take_along_dim(BatchedTensor(torch.tensor([[3, 2], [0, 3], [1, 4]])), dim=0)
        tensor([[6, 5],
                [0, 7],
                [2, 9]], batch_dim=0)

        ```
        """
        self._check_valid_dims((self, indices))
        indices = to_tensor(indices).long()
        if dim is None:
            return self._data.take_along_dim(indices)
        return self._create_new_batch(self._data.take_along_dim(indices, dim=dim))

    def unsqueeze(self, dim: int) -> Self:
        r"""Return a new batch with a dimension of size one inserted at
        the specified position.

        The returned tensor shares the same underlying data with this
        batch.

        Args:
            dim: Specifies the dimension at which to insert the
                singleton dimension.

        Returns:
            A new batch with an added singleton dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.unsqueeze(dim=0)
            tensor([[[1., 1., 1.],
                    [1., 1., 1.]]], batch_dim=1)
            >>> batch.unsqueeze(dim=1)
            tensor([[[1., 1., 1.]],
                    [[1., 1., 1.]]], batch_dim=0)
            >>> batch.unsqueeze(dim=-1)
            tensor([[[1.],
                     [1.],
                     [1.]],
                    [[1.],
                     [1.],
                     [1.]]], batch_dim=0)
        """
        return self.__class__(
            self._data.unsqueeze(dim=dim),
            batch_dim=(
                self._batch_dim + 1 if self._batch_dim >= dim and dim >= 0 else self._batch_dim
            ),
        )

    def view(self, *shape: tuple[int, ...]) -> Tensor:
        r"""Create a new tensor with the same data as the ``self`` batch
        but with a new shape.

        Args:
            shape (tuple): Specifies the desired shape.

        Returns:
            ``torch.Tensor``: A new view of the tensor in the batch.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 6))
            >>> batch.view(2, 3, 2)
            tensor([[[1., 1.],
                     [1., 1.],
                     [1., 1.]],
                    [[1., 1.],
                     [1., 1.],
                     [1., 1.]]])
        """
        return self._data.view(*shape)

    def view_as(self, other: BatchedTensor | Tensor) -> Self:
        r"""Create a new batch with the same data as the ``self`` batch
        but the shape of ``other``.

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

        .. code-block:: pycon

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
        self._check_valid_dims((self, other))
        return self._create_new_batch(self._data.view_as(other.data))

    ########################
    #     mini-batches     #
    ########################

    #################
    #     Other     #
    #################

    def apply(self, fn: Callable[[Tensor], Tensor]) -> Self:
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

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.apply(lambda tensor: tensor + 2)
            tensor([[ 2,  3,  4,  5,  6],
                    [ 7,  8,  9, 10, 11]], batch_dim=0)
        """
        return self._create_new_batch(fn(self._data))

    def apply_(self, fn: Callable[[Tensor], Tensor]) -> None:
        r"""Apply a function to transform the tensor of the current
        batch.

        In-place version of ``apply``.

        Args:
            fn (``Callable``): Specifies the function to be applied to
                the tensor. It is the responsibility of the user to
                verify the function applies a valid transformation of
                the data.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.apply_(lambda tensor: tensor + 2)
            >>> batch
            tensor([[ 2,  3,  4,  5,  6],
                    [ 7,  8,  9, 10, 11]], batch_dim=0)
        """
        self._data = fn(self._data)

    def summary(self) -> str:
        dims = ", ".join([f"{key}={value}" for key, value in self._get_kwargs().items()])
        return (
            f"{self.__class__.__qualname__}(dtype={self.dtype}, shape={self.shape}, "
            f"device={self.device}, {dims})"
        )

    def _check_valid_dims(self, tensors: Sequence) -> None:
        r"""Check if the dimensions are valid.

        Args:
            tensors (``Sequence``): Specifies the sequence of
                tensors/batches to check.
        """
        check_batch_dims(get_batch_dims(tensors))

    def _create_new_batch(self, data: Tensor) -> Self:
        r"""Create a new batch given a ``torch.Tensor``.

        Args:
            data (``torch.Tensor``): Specifies the data to put in the
                batch.

        Returns:
            ``BatchedTensor``: The new batch.
        """
        return self.__class__(data, **self._get_kwargs())

    def _get_kwargs(self) -> dict:
        return {"batch_dim": self._batch_dim}


def implements(torch_function: Callable) -> Callable:
    """Register a torch function override for BatchedTensor.

    Args:
        torch_function (``Callable``):  Specifies the torch function
            to override.

    Returns:
        ``Callable``: The decorated function.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from redcat.tensor import BatchedTensor, implements
        >>> @implements(torch.sum)
        ... def torchsum(input: BatchedTensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        ...     return torch.sum(input.data, *args, **kwargs)
        ...
        >>> torch.sum(BatchedTensor(torch.ones(2, 3)))
        tensor(6.)
    """

    def decorator(func: Callable) -> Callable:
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements(torch.amax)
def amax(input: BatchedTensor, *args: Any, **kwargs: Any) -> Tensor:  # noqa: A002
    r"""See ``torch.amax`` documentation."""
    return input.amax(*args, **kwargs)


@implements(torch.amin)
def amin(input: BatchedTensor, *args: Any, **kwargs: Any) -> Tensor:  # noqa: A002
    r"""See ``torch.amin`` documentation."""
    return input.amin(*args, **kwargs)


@implements(torch.argmax)
def argmax(input: BatchedTensor, *args: Any, **kwargs: Any) -> Tensor:  # noqa: A002
    r"""See ``torch.argmax`` documentation."""
    return input.argmax(*args, **kwargs)


@implements(torch.argmin)
def argmin(input: BatchedTensor, *args: Any, **kwargs: Any) -> Tensor:  # noqa: A002
    r"""See ``torch.argmin`` documentation."""
    return input.argmin(*args, **kwargs)


@implements(torch.argsort)
def argsort(input: BatchedTensor, *args: Any, **kwargs: Any) -> Tensor:  # noqa: A002
    r"""See ``torch.argsort`` documentation."""
    return input.argsort(*args, **kwargs)


@implements(torch.cat)
def cat(
    tensors: tuple[BatchedTensor | Tensor, ...] | list[BatchedTensor | Tensor],
    dim: int = 0,
) -> BatchedTensor:
    r"""See ``torch.cat`` documentation."""
    return tensors[0].cat(tensors[1:], dim=dim)


@implements(torch.chunk)
def chunk(tensor: BatchedTensor, chunks: int, dim: int = 0) -> tuple[BatchedTensor, ...]:
    r"""See ``torch.chunk`` documentation."""
    return tensor.chunk(chunks=chunks, dim=dim)


# Use the name `torchmax` to avoid shadowing `max` python builtin.
@implements(torch.max)
def torchmax(
    input: BatchedTensor, *args: Any, **kwargs: Any  # noqa: A002
) -> Tensor | torch.return_types.max:
    r"""See ``torch.max`` documentation."""
    return input.max(*args, **kwargs)


@implements(torch.maximum)
def maximum(input: BatchedTensor, other: BatchedTensor | Tensor) -> BatchedTensor:  # noqa: A002
    r"""See ``torch.maximum`` documentation."""
    return input.maximum(other)


@implements(torch.mean)
def mean(input: BatchedTensor, *args: Any, **kwargs: Any) -> Tensor:  # noqa: A002
    r"""See ``torch.mean`` documentation."""
    return input.mean(*args, **kwargs)


@implements(torch.median)
def median(
    input: BatchedTensor, *args: Any, **kwargs: Any  # noqa: A002
) -> Tensor | torch.return_types.median:
    r"""See ``torch.median`` documentation."""
    return input.median(*args, **kwargs)


# Use the name `torchmin` to avoid shadowing `min` python builtin.
@implements(torch.min)
def torchmin(
    input: BatchedTensor, *args: Any, **kwargs: Any  # noqa: A002
) -> Tensor | torch.return_types.min:
    r"""See ``torch.min`` documentation."""
    return input.min(*args, **kwargs)


@implements(torch.minimum)
def minimum(input: BatchedTensor, other: BatchedTensor | Tensor) -> BatchedTensor:  # noqa: A002
    r"""See ``torch.minimum`` documentation."""
    return input.minimum(other)


@implements(torch.nanmean)
def nanmean(input: BatchedTensor, *args: Any, **kwargs: Any) -> Tensor:  # noqa: A002
    r"""See ``torch.nanmean`` documentation."""
    return input.nanmean(*args, **kwargs)


@implements(torch.nanmedian)
def nanmedian(
    input: BatchedTensor, *args: Any, **kwargs: Any  # noqa: A002
) -> Tensor | torch.return_types.nanmedian:
    r"""See ``torch.nanmedian`` documentation."""
    return input.nanmedian(*args, **kwargs)


@implements(torch.nansum)
def nansum(input: BatchedTensor, *args: Any, **kwargs: Any) -> Tensor:  # noqa: A002
    r"""See ``torch.nansum`` documentation."""
    return input.nansum(*args, **kwargs)


@implements(torch.prod)
def prod(input: BatchedTensor, *args: Any, **kwargs: Any) -> Tensor:  # noqa: A002
    r"""See ``torch.prod`` documentation."""
    return input.prod(*args, **kwargs)


@implements(torch.select)
def select(input: BatchedTensor, dim: int, index: int) -> Tensor:  # noqa: A002
    r"""See ``torch.select`` documentation."""
    return input.select(dim=dim, index=index)


@implements(torch.sort)
def sort(input: BatchedTensor, *args: Any, **kwargs: Any) -> torch.return_types.sort:  # noqa: A002
    r"""See ``torch.sort`` documentation."""
    return input.sort(*args, **kwargs)


@implements(torch.split)
def split(
    tensor: BatchedTensor, split_size_or_sections: int | Sequence[int], dim: int = 0
) -> tuple[BatchedTensor, ...]:
    r"""See ``torch.split`` documentation."""
    return tensor.split(split_size_or_sections=split_size_or_sections, dim=dim)


@implements(torch.sum)
def torchsum(input: BatchedTensor, *args: Any, **kwargs: Any) -> Tensor:  # noqa: A002
    r"""See ``torch.sum`` documentation.

    Use the name `torchsum` to avoid shadowing `sum` python builtin.
    """
    return input.sum(*args, **kwargs)


@implements(torch.take_along_dim)
def take_along_dim(
    input: BatchedTensor | Tensor,  # noqa: A002
    indices: BatchedTensor | Tensor,
    dim: int | None = None,
) -> BatchedTensor | Tensor:
    r"""See ``torch.take_along_dim`` documentation."""
    return input.take_along_dim(indices, dim)
