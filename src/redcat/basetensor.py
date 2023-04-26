from __future__ import annotations

__all__ = ["BaseBatchedTensor"]

from abc import abstractmethod
from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any, TypeVar, overload

import torch
from torch import Tensor

from redcat.base import BaseBatch
from redcat.utils import IndexType

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchedTensor = TypeVar("TBatchedTensor", bound="BaseBatchedTensor")


class BaseBatchedTensor(BaseBatch[Tensor]):
    def __init__(self, data: Any, **kwargs) -> None:
        super().__init__()
        self._data = torch.as_tensor(data, **kwargs)

    @property
    def data(self) -> Tensor:
        r"""``torch.Tensor``: The data in the batch."""
        return self._data

    @property
    def device(self) -> torch.device:
        r"""``torch.device``: The device where the batch data/tensor is."""
        return self._data.device

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
            ``BaseBatchedTensor``: A new batch with a contiguous
                representation of the data.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3)).contiguous()
            >>> batch.is_contiguous()
            True
        """
        return self.__class__(
            self._data.contiguous(memory_format=memory_format), **self._get_kwargs()
        )

    def is_contiguous(self, memory_format: torch.memory_format = torch.contiguous_format) -> bool:
        r"""Indicates if a batch as a contiguous representation of the data.

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
            ``BaseBatchedTensor``: A new batch with the data after
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
        return self.__class__(self._data.to(*args, **kwargs), **self._get_kwargs())

    ###############################
    #     Creation operations     #
    ###############################

    def clone(self, *args, **kwargs) -> TBatchedTensor:
        r"""Creates a copy of the current batch.

        Args:
            *args: See the documentation of ``torch.Tensor.clone``
            **kwargs: See the documentation of ``torch.Tensor.clone``

        Returns:
            ``BaseBatchedTensor``: A copy of the current batch.

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
        return self.__class__(self._data.clone(*args, **kwargs), **self._get_kwargs())

    def empty_like(self, *args, **kwargs) -> TBatchedTensor:
        r"""Creates an uninitialized batch, with the same shape as the current
        batch.

        Args:
            *args: See the documentation of ``torch.Tensor.empty_like``
            **kwargs: See the documentation of
                ``torch.Tensor.empty_like``

        Returns:
            ``BaseBatchedTensor``: A uninitialized batch with the same
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
        return self.__class__(torch.empty_like(self._data, *args, **kwargs), **self._get_kwargs())

    def full_like(self, *args, **kwargs) -> TBatchedTensor:
        r"""Creates a batch filled with a given scalar value, with the same
        shape as the current batch.

        Args:
            *args: See the documentation of ``torch.Tensor.full_like``
            **kwargs: See the documentation of
                ``torch.Tensor.full_like``

        Returns:
            ``BaseBatchedTensor``: A batch filled with the scalar
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
        return self.__class__(torch.full_like(self._data, *args, **kwargs), **self._get_kwargs())

    def ones_like(self, *args, **kwargs) -> TBatchedTensor:
        r"""Creates a batch filled with the scalar value ``1``, with the same
        shape as the current batch.

        Args:
            *args: See the documentation of ``torch.Tensor.ones_like``
            **kwargs: See the documentation of
                ``torch.Tensor.ones_like``

        Returns:
            ``BaseBatchedTensor``: A batch filled with the scalar
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
        return self.__class__(torch.ones_like(self._data, *args, **kwargs), **self._get_kwargs())

    def zeros_like(self, *args, **kwargs) -> TBatchedTensor:
        r"""Creates a batch filled with the scalar value ``0``, with the same
        shape as the current batch.

        Args:
            *args: See the documentation of ``torch.Tensor.zeros_like``
            **kwargs: See the documentation of
                ``torch.Tensor.zeros_like``

        Returns:
            ``BaseBatchedTensor``: A batch filled with the scalar
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
        return self.__class__(torch.zeros_like(self._data, *args, **kwargs), **self._get_kwargs())

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

    def eq(self, other: BaseBatchedTensor | torch.Tensor | bool | int | float) -> TBatchedTensor:
        r"""Computes element-wise equality.

        Args:
            other: Specifies the batch to compare.

        Returns:
            ``BaseBatchedTensor``: A batch containing the element-wise
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

    def ge(self, other: BaseBatchedTensor | torch.Tensor | bool | int | float) -> TBatchedTensor:
        r"""Computes ``self >= other`` element-wise.

        Args:
            other: Specifies the value to compare
                with.

        Returns:
            ``BaseBatchedTensor``: A batch containing the element-wise
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

    def gt(self, other: BaseBatchedTensor | torch.Tensor | bool | int | float) -> TBatchedTensor:
        r"""Computes ``self > other`` element-wise.

        Args:
            other: Specifies the batch to compare.

        Returns:
            ``BaseBatchedTensor``: A batch containing the element-wise
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
        r"""Indicates if each element of the batch is infinite (positive or
        negative infinity) or not.

        Returns:
            BaseBatchedTensor:  A batch containing a boolean tensor
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
        r"""Indicates if each element of the batch is negative infinity or not.

        Returns:
            BaseBatchedTensor:  A batch containing a boolean tensor
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
        r"""Indicates if each element of the batch is positive infinity or not.

        Returns:
            BaseBatchedTensor:  A batch containing a boolean tensor
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
            BaseBatchedTensor:  A batch containing a boolean tensor
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

    def le(self, other: BaseBatchedTensor | torch.Tensor | bool | int | float) -> TBatchedTensor:
        r"""Computes ``self <= other`` element-wise.

        Args:
            other: Specifies the batch to compare.

        Returns:
            ``BaseBatchedTensor``: A batch containing the element-wise
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

    def lt(self, other: BaseBatchedTensor | torch.Tensor | bool | int | float) -> TBatchedTensor:
        r"""Computes ``self < other`` element-wise.

        Args:
            other: Specifies the batch to compare.

        Returns:
            ``BaseBatchedTensor``: A batch containing the element-wise
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
            ``BaseBatchedTensor``: The current batch to bool data type.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.bool().dtype
            torch.bool
        """
        return self.__class__(self._data.bool(), **self._get_kwargs())

    def double(self) -> TBatchedTensor:
        r"""Converts the current batch to double data type.

        Returns:
            ``BaseBatchedTensor``: The current batch to double data type.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.double().dtype
            torch.float64
        """
        return self.__class__(self._data.double(), **self._get_kwargs())

    def float(self) -> TBatchedTensor:
        r"""Converts the current batch to float data type.

        Returns:
            ``BaseBatchedTensor``: The current batch to float data type.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.float().dtype
            torch.float32
        """
        return self.__class__(self._data.float(), **self._get_kwargs())

    def int(self) -> TBatchedTensor:
        r"""Converts the current batch to int data type.

        Returns:
            ``BaseBatchedTensor``: The current batch to int data type.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.int().dtype
            torch.int32
        """
        return self.__class__(self._data.int(), **self._get_kwargs())

    def long(self) -> TBatchedTensor:
        r"""Converts the current batch to long data type.

        Returns:
            ``BaseBatchedTensor``: The current batch to long data type.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.long().dtype
            torch.int64
        """
        return self.__class__(self._data.long(), **self._get_kwargs())

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
        other: BaseBatchedTensor | Tensor | int | float,
        alpha: int | float = 1.0,
    ) -> TBatchedTensor:
        r"""Adds the input ``other``, scaled by ``alpha``, to the ``self``
        batch.

        Similar to ``out = self + alpha * other``

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor`` or int or
                float): Specifies the other value to add to the
                current batch.
            alpha (int or float, optional): Specifies the scale of the
                batch to add. Default: ``1.0``

        Returns:
            ``BaseBatchedTensor``: A new batch containing the addition of
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

    @abstractmethod
    def add_(
        self,
        other: BaseBatchedTensor | Tensor | int | float,
        alpha: int | float = 1.0,
    ) -> None:
        r"""Adds the input ``other``, scaled by ``alpha``, to the ``self``
        batch.

        Similar to ``self += alpha * other`` (in-place)

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor`` or int or
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

    def div(
        self,
        other: BaseBatchedTensor | torch.Tensor | int | float,
        rounding_mode: str | None = None,
    ) -> TBatchedTensor:
        r"""Divides the ``self`` batch by the input ``other`.

        Similar to ``out = self / other``

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor`` or int or
                float): Specifies the dividend.
            rounding_mode (str or ``None``, optional): Specifies the
                type of rounding applied to the result.
                - ``None``: true division.
                - ``"trunc"``: rounds the results of the division
                    towards zero.
                - ``"floor"``: floor division.
                Default: ``None``

        Returns:
            ``BaseBatchedTensor``: A new batch containing the division
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

    @abstractmethod
    def div_(
        self,
        other: BaseBatchedTensor | torch.Tensor | int | float,
        rounding_mode: str | None = None,
    ) -> None:
        r"""Divides the ``self`` batch by the input ``other`.

        Similar to ``self /= other`` (in-place)

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor`` or int or
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

    def fmod(
        self,
        divisor: BaseBatchedTensor | torch.Tensor | int | float,
    ) -> TBatchedTensor:
        r"""Computes the element-wise remainder of division.

        The current batch is the dividend.

        Args:
            divisor (``BaseBatchedTensor`` or ``torch.Tensor`` or int
                or float): Specifies the divisor.

        Returns:
            ``BaseBatchedTensor``: A new batch containing the
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

    @abstractmethod
    def fmod_(self, divisor: BaseBatchedTensor | torch.Tensor | int | float) -> None:
        r"""Computes the element-wise remainder of division.

        The current batch is the dividend.

        Args:
            divisor (``BaseBatchedTensor`` or ``torch.Tensor`` or int
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

    def mul(self, other: BaseBatchedTensor | torch.Tensor | int | float) -> TBatchedTensor:
        r"""Multiplies the ``self`` batch by the input ``other`.

        Similar to ``out = self * other``

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor`` or int or
                float): Specifies the value to multiply.

        Returns:
            ``BaseBatchedTensor``: A new batch containing the
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

    @abstractmethod
    def mul_(self, other: BaseBatchedTensor | torch.Tensor | int | float) -> None:
        r"""Multiplies the ``self`` batch by the input ``other`.

        Similar to ``self *= other`` (in-place)

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor`` or int or
                float): Specifies the value to multiply.

        Returns:
            ``BaseBatchedTensor``: A new batch containing the
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

    def neg(self) -> TBatchedTensor:
        r"""Returns a new batch with the negative of the elements.

        Returns:
            ``BaseBatchedTensor``: A new batch with the negative of
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
        other: BaseBatchedTensor | torch.Tensor | int | float,
        alpha: int | float = 1,
    ) -> TBatchedTensor:
        r"""Subtracts the input ``other``, scaled by ``alpha``, to the ``self``
        batch.

        Similar to ``out = self - alpha * other``

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor`` or int or
                float): Specifies the value to subtract.
            alpha (int or float, optional): Specifies the scale of the
                batch to substract. Default: ``1``

        Returns:
            ``BaseBatchedTensor``: A new batch containing the diffence of
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

    @abstractmethod
    def sub_(
        self,
        other: BaseBatchedTensor | torch.Tensor | int | float,
        alpha: int | float = 1,
    ) -> None:
        r"""Subtracts the input ``other``, scaled by ``alpha``, to the ``self``
        batch.

        Similar to ``self -= alpha * other`` (in-place)

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor`` or int or
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

    ###########################################################
    #     Mathematical | advanced arithmetical operations     #
    ###########################################################

    def cumsum(self, dim: int, **kwargs) -> TBatchedTensor:
        r"""Computes the cumulative sum of elements of the current batch in a
        given dimension.

        Args:
            dim (int): Specifies the dimmenison of the cumulative sum.
            **kwargs: see ``torch.cumsum`` documentation

        Returns:
            ``BaseBatchedTensor``: A batch with the cumulative sum of
                elements of the current batch in a given dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5)).cumsum(dim=1)
            >>> batch
            tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  7,  9, 11, 13]], batch_dim=0)
        """
        return torch.cumsum(self, dim=dim, **kwargs)

    def cumsum_(self, dim: int, **kwargs) -> None:
        r"""Computes the cumulative sum of elements of the current batch in a
        given dimension.

        Args:
            dim (int): Specifies the dimmenison of the cumulative sum.
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

    @abstractmethod
    def cumsum_along_batch(self, **kwargs) -> TBatchedTensor:
        r"""Computes the cumulative sum of elements of the current batch in the
        batch dimension.

        Args:
            **kwargs: see ``torch.cumsum`` documentation

        Returns:
            ``BaseBatchedTensor``: A batch with the cumulative sum of
                elements of the current batch in the batch dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5)).cumsum_along_batch()
            >>> batch
            tensor([[ 0,  1,  2,  3,  4],
                    [ 5,  7,  9, 11, 13]], batch_dim=0)
        """

    @abstractmethod
    def cumsum_along_batch_(self) -> None:
        r"""Computes the cumulative sum of elements of the current batch in the
        batch dimension.

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

    ################################################
    #     Mathematical | point-wise operations     #
    ################################################

    def abs(self) -> TBatchedTensor:
        r"""Computes the absolute value of each element.

        Return:
            ``BaseBatchedTensor``: A batch with the absolute value of
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
                the lower bound. If ``min_value`` is ``None``,
                there is no lower bound. Default: ``None``
            max (int, float or ``None``, optional): Specifies
                the upper bound. If ``max_value`` is ``None``,
                there is no upper bound. Default: ``None``

        Returns:
            ``BaseBatchedTensor``: A batch with clamped values.

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
        min_value: int | float | None = None,
        max_value: int | float | None = None,
    ) -> None:
        r"""Clamps all elements in ``self`` into the range ``[min_value,
        max_value]``.

        Inplace version of ``clamp``.

        Note: ``min_value`` and ``max_value`` cannot be both ``None``.

        Args:
            min_value (int, float or ``None``, optional): Specifies
                the lower bound.  If ``min_value`` is ``None``,
                there is no lower bound. Default: ``None``
            max_value (int, float or ``None``, optional): Specifies
                the upper bound. If ``max_value`` is ``None``,
                there is no upper bound. Default: ``None``

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.clamp_(min_value=2, max_value=5)
            >>> batch
            tensor([[2, 2, 2, 3, 4],
                    [5, 5, 5, 5, 5]], batch_dim=0)
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.clamp_(min_value=2)
            >>> batch
            tensor([[2, 2, 2, 3, 4],
                    [5, 6, 7, 8, 9]], batch_dim=0)
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.clamp_(max_value=7)
            >>> batch
            tensor([[0, 1, 2, 3, 4],
                    [5, 6, 7, 7, 7]], batch_dim=0)
        """
        self._data.clamp_(min=min_value, max=max_value)

    def exp(self) -> TBatchedTensor:
        r"""Computes the exponential of the elements.

        Return:
            ``BaseBatchedTensor``: A batch with the exponential of the
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

    def log(self) -> BaseBatchedTensor:
        r"""Computes the natural logarithm of the elements.

        Return:
            ``BaseBatchedTensor``: A batch with the natural
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

    def log1p(self) -> BaseBatchedTensor:
        r"""Computes the natural logarithm of ``self + 1``.

        Return:
            ``BaseBatchedTensor``: A batch with the natural
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

    @overload
    def max(self) -> bool | int | float:
        r"""Finds the maximum value in the batch.

        Returns:
            The maximum value in the batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(6).view(2, 3))
            >>> batch.max()
            5
            >>> batch = BatchedTensor(torch.tensor([[False, True, True], [True, False, True]]))
            >>> batch.max()
            True
        """

    @overload
    def max(self, other: BaseBatchedTensor) -> TBatchedTensor:
        r"""Computes the element-wise maximum of ``self`` and ``other``.

        Args:
            other (``BaseBatchedTensor``): Specifies a batch.

        Returns:
            ``BaseBatchedTensor``: The batch with the element-wise
                maximum of ``self`` and ``other``

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(6).view(2, 3))
            >>> batch.max(BatchedTensor(torch.tensor([[1, 0, 2], [4, 5, 3]])))
            tensor([[1, 1, 2],
                    [4, 5, 5]], batch_dim=0)
        """

    def max(
        self, other: BaseBatchedTensor | Tensor | None = None
    ) -> bool | int | float | TBatchedTensor:
        r"""If ``other`` is None, this method finds the maximum value in the
        batch, otherwise it computes the element-wise maximum of ``self`` and
        ``other``.

        Args:
            other (``BaseBatchedTensor`` or ``None``, optional):
                Specifies a batch. Default: ``None``

        Returns:
            The maximum value in the batch or the element-wise maximum
                of ``self`` and ``other``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(6).view(2, 3))
            >>> batch.max()
            5
            >>> batch.max(BatchedTensor(torch.tensor([[1, 0, 2], [4, 5, 3]])))
            tensor([[1, 1, 2],
                    [4, 5, 5]], batch_dim=0)
        """
        if other is None:
            return torch.max(self._data)
        return torch.max(self, other)

    @overload
    def min(self) -> bool | int | float:
        r"""Finds the minimum value in the batch.

        Returns:
            The minimum value in the batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(6).view(2, 3))
            >>> batch.min()
            0
            >>> batch = BatchedTensor(torch.tensor([[False, True, True], [True, False, True]]))
            >>> batch.min()
            False
        """

    @overload
    def min(self, other: BaseBatchedTensor) -> TBatchedTensor:
        r"""Computes the element-wise minimum of ``self`` and ``other``.

        Args:
            other (``BaseBatchedTensor``): Specifies a batch.

        Returns:
            ``BaseBatchedTensor``: The batch with the element-wise
                minimum of ``self`` and ``other``

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(6).view(2, 3))
            >>> batch.min(BatchedTensor(torch.tensor([[1, 0, 2], [4, 5, 3]])))
            tensor([[0, 0, 2],
                    [3, 4, 3]], batch_dim=0)
        """

    def min(
        self, other: BaseBatchedTensor | Tensor | None = None
    ) -> bool | int | float | TBatchedTensor:
        r"""If ``other`` is None, this method finds the minimum value in the
        batch, otherwise it computes the element-wise minimum of ``self`` and
        ``other``.

        Args:
            other (``BaseBatchedTensor`` or ``None``, optional):
                Specifies a batch. Default: ``None``

        Returns:
            The minimum value in the batch or the element-wise minimum
                of ``self`` and ``other``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(6).view(2, 3))
            >>> batch.min()
            0
            >>> batch.min(BatchedTensor(torch.tensor([[1, 0, 2], [4, 5, 3]])))
            tensor([[0, 0, 2],
                    [3, 4, 3]], batch_dim=0)
        """
        if other is None:
            return torch.min(self._data)
        return torch.min(self, other)

    def pow(self, exponent: int | float | BaseBatchedTensor) -> TBatchedTensor:
        r"""Computes the power of each element with the given exponent.

        Args:
            exponent (int or float or ``BaseBatchedTensor``): Specifies
                the exponent value. ``exponent`` can be either a single
                numeric number or a ``BaseBatchedTensor`` with the same
                number of elements.

        Return:
            ``BaseBatchedTensor``: A batch with the power of each
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

    @abstractmethod
    def pow_(self, exponent: int | float | BaseBatchedTensor) -> None:
        r"""Computes the power of each element with the given exponent.

        In-place version of ``pow(exponent)``.

        Args:
            exponent (int or float or ``BaseBatchedTensor``): Specifies
                the exponent value. ``exponent`` can be either a
                single numeric number or a ``BaseBatchedTensor``
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

    def sqrt(self) -> TBatchedTensor:
        r"""Computes the square-root of each element.

        Return:
            ``BaseBatchedTensor``: A batch with the square-root of
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

    ###########################################
    #     Mathematical | trigo operations     #
    ###########################################

    def acos(self) -> TBatchedTensor:
        r"""Computes the inverse cosine (arccos) of each element.

        Return:
            ``BaseBatchedTensor``: A batch with the inverse cosine
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
        r"""Computes the inverse hyperbolic cosine (arccosh) of each element.

        Return:
            ``BaseBatchedTensor``: A batch with the inverse hyperbolic
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
        r"""Computes the inverse hyperbolic cosine (arccosh) of each element.

        Return:
            ``BaseBatchedTensor``: A batch with the inverse hyperbolic
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
            ``BaseBatchedTensor``: A batch with the inverse sine
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
        r"""Computes the inverse hyperbolic sine (arcsinh) of each element.

        Return:
            ``BaseBatchedTensor``: A batch with the inverse hyperbolic
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
        r"""Computes the inverse hyperbolic sine (arcsinh) of each element.

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
            ``BaseBatchedTensor``: A batch with the inverse tangent
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
            ``BaseBatchedTensor``: A batch with the inverse hyperbolic
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
            ``BaseBatchedTensor``: A batch with the cosine of each
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
            ``BaseBatchedTensor``: A batch with the hyperbolic
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
            ``BaseBatchedTensor``: A batch with the sine of each
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
            ``BaseBatchedTensor``: A batch with the hyperbolic sine of
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
            ``BaseBatchedTensor``: A batch with the tangent of each
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
            ``BaseBatchedTensor``: A batch with the tangent of each
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

    def logical_and(self, other: BaseBatchedTensor | Tensor) -> TBatchedTensor:
        r"""Computes the element-wise logical AND.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor``):
                Specifies the batch or tensor to compute
                logical AND with.

        Returns:
            ``BaseBatchedTensor``: A batch containing the element-wise
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

    @abstractmethod
    def logical_and_(self, other: BaseBatchedTensor | Tensor) -> None:
        r"""Computes the element-wise logical AND.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor``):
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

    def logical_not(self) -> TBatchedTensor:
        r"""Computes the element-wise logical NOT of the current batch.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Returns:
            ``BaseBatchedTensor``: A batch containing the element-wise
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

    def logical_or(self, other: BaseBatchedTensor | Tensor) -> TBatchedTensor:
        r"""Computes the element-wise logical OR.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor``):
                Specifies the batch to compute logical OR with.

        Returns:
            ``BaseBatchedTensor``: A batch containing the element-wise
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

    @abstractmethod
    def logical_or_(self, other: BaseBatchedTensor | Tensor) -> None:
        r"""Computes the element-wise logical OR.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor``):
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

    def logical_xor(self, other: BaseBatchedTensor | Tensor) -> TBatchedTensor:
        r"""Computes the element-wise logical XOR.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor``):
                Specifies the batch to compute logical XOR with.

        Returns:
            ``BaseBatchedTensor``: A batch containing the element-wise
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

    @abstractmethod
    def logical_xor_(self, other: BaseBatchedTensor | Tensor) -> None:
        r"""Computes the element-wise logical XOR.

        Zeros are treated as ``False`` and non-zeros are treated as
        ``True``.

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor``):
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

    ##########################################################
    #    Indexing, slicing, joining, mutating operations     #
    ##########################################################

    def __getitem__(self, index: IndexType) -> Tensor:
        return self._data[index]

    def __setitem__(
        self, index: IndexType, value: bool | int | float | Tensor | BaseBatchedTensor
    ) -> None:
        if isinstance(value, BaseBatchedTensor):
            value = value.data
        self._data[index] = value

    def append(self, other: BaseBatchedTensor) -> None:
        r"""Appends a new batch to the current batch along the batch dimension.

        Args:
            other (``TensorSeqBatch``): Specifies the batch to append
                at the end of current batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.append(BatchedTensor(torch.zeros(1, 3)))
            >>> batch.append(BatchedTensor(torch.full((1, 3), 2.0)))
            >>> batch
            tensor([[1., 1., 1.],
                    [1., 1., 1.],
                    [0., 0., 0.],
                    [2., 2., 2.]], batch_dim=0)
        """
        self.cat_along_batch_(other)

    def cat(
        self,
        tensors: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor],
        dim: int = 0,
    ) -> TBatchedTensor:
        r"""Concatenates the data of the batch(es) to the current batch along a
        given dimension and creates a new batch.

        Args:
            tensors (``BaseBatchedTensor`` or ``torch.Tensor`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Returns:
            ``BaseBatchedTensor``: A batch with the concatenated data
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
        if isinstance(tensors, (BaseBatchedTensor, Tensor)):
            tensors = [tensors]
        return torch.cat(list(chain([self], tensors)), dim=dim)

    def cat_(
        self,
        tensors: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor],
        dim: int = 0,
    ) -> None:
        r"""Concatenates the data of the batch(es) to the current batch along a
        given dimension and creates a new batch.

        Args:
            tensor (``BaseBatchedTensor`` or ``torch.Tensor`` or
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

    @abstractmethod
    def cat_along_batch(
        self, tensors: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor]
    ) -> TBatchedTensor:
        r"""Concatenates the data of the batch(es) to the current batch along
        the batch dimension and creates a new batch.

        Args:
            tensors (``BaseBatchedTensor`` or ``torch.Tensor`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Returns:
            ``BaseBatchedTensor``: A batch with the concatenated data
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

    @abstractmethod
    def cat_along_batch_(
        self, tensors: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor]
    ) -> None:
        r"""Concatenates the data of the batch(es) to the current batch along
        the batch dimension and creates a new batch.

        Args:
            tensors (``BaseBatchedTensor`` or ``torch.Tensor`` or
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

    def extend(self, other: Iterable[BaseBatch]) -> None:
        self.cat_along_batch_(other)

    def masked_fill(
        self, mask: BaseBatchedTensor | Tensor, value: bool | int | float
    ) -> TBatchedTensor:
        r"""Fills elements of ``self`` batch with ``value`` where ``mask`` is
        ``True``.

        Args:
            mask (``BaseBatchedTensor`` or ``torch.Tensor``):
                Specifies the batch of boolean masks.
            value (number): Specifies the value to fill in with.

        Returns:
            ``BaseBatchedTensor``: A new batch with the updated values.

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

    def select(self, dim: int, index: int) -> Tensor:
        r"""Selects the batch along the batch dimension at the given index.

        Args:
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
            ``BaseBatchedTensor``: A slice of the current batch.

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
        return self.__class__(data, **self._get_kwargs())

    def sort(
        self,
        dim: int = -1,
        descending: bool = False,
        stable: bool = False,
    ) -> tuple[TBatchedTensor, TBatchedTensor]:
        return torch.sort(self, dim=dim, descending=descending, stable=stable)

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

    @abstractmethod
    def unsqueeze(self, dim: int) -> TBatchedTensor:
        r"""Returns a new batch with a dimension of size one inserted at the
        specified position.

        The returned tensor shares the same underlying data with this
        batch.

        Args:
            dim (int): Specifies the dimension at which to insert the
                singleton dimension.

        Returns:
            ``BaseBatchedTensor``: A new batch with an added singleton
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

    @abstractmethod
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

    @abstractmethod
    def view_as(self, other: BaseBatchedTensor) -> TBatchedTensor:
        r"""Creates a new batch with the same data as the ``self`` batch but the
        shape of ``other``.

        The returned batch shares the same data and must have the
        same number of elements, but the data may have a different
        size.

        Args:
            other (``BaseBatchedTensor``): Specifies the batch with
                the target shape.

        Returns:
            ``BaseBatchedTensor``: A new batch with the shape of
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

    @abstractmethod
    def _get_kwargs(self) -> dict:
        r"""Gets the keyword arguments that are specific to the batched tensor
        implementation.

        Returns:
            dict: The keyword arguments
        """
