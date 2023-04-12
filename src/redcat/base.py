from __future__ import annotations

__all__ = ["BaseBatchedTensor"]

from abc import ABC, abstractmethod
from typing import Any, TypeVar

import torch
from torch import Tensor

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchedTensor = TypeVar("TBatchedTensor", bound="BaseBatchedTensor")


class BaseBatchedTensor(ABC):
    @abstractmethod
    def _get_kwargs(self) -> dict:
        r"""Gets the keyword arguments that are specific to the batched tensor
        implementation.

        Returns:
            dict: The keyword arguments
        """

    def __init__(self, data: Any, **kwargs) -> None:
        super().__init__()
        self._data = torch.as_tensor(data, **kwargs)

    @property
    @abstractmethod
    def batch_size(self) -> int:
        r"""int: The batch size."""

    @property
    def data(self) -> Tensor:
        r"""``torch.Tensor``: The data in the batch."""
        return self._data

    @property
    def device(self) -> torch.device:
        r"""``torch.device``: The device where the batch data/tensor is."""
        return self._data.device

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
            >>> batch_bool.data
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

    @abstractmethod
    def allclose(
        self, other: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
    ) -> bool:
        r"""Indicates if two batches are equal within a tolerance or not.

        Args:
            other: Specifies the value to compare.
            rtol (float, optional): Specifies the relative tolerance
                parameter. Default: ``1e-5``
            atol (float, optional): Specifies the absolute tolerance
                parameter. Default: ``1e-8``
            equal_nan (bool, optional): If ``True``, then two ``NaN``s
                will be considered equal. Default: ``False``

        Returns:
            bool: ``True`` if the batches are equal within a tolerance,
                ``False`` otherwise.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch1 = BatchedTensor(torch.ones(2, 3))
            >>> batch2 = BatchedTensor(torch.full((2, 3), 1.5))
            >>> batch1.allclose(batch2, atol=1, rtol=0)
            True
        """

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

    @abstractmethod
    def equal(self, other: Any) -> bool:
        r"""Indicates if two batches are equal or not.

        Args:
            other: Specifies the value to compare.

        Returns:
            bool: ``True`` if the batches have the same size,
                elements and same batch dimension, ``False`` otherwise.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.ones(2, 3)).equal(BatchedTensor(torch.zeros(2, 3)))
            False
        """

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
        min_value: int | float | None = None,
        max_value: int | float | None = None,
    ) -> TBatchedTensor:
        r"""Clamps all elements in ``self`` into the range ``[min_value,
        max_value]``.

        Note: ``min_value`` and ``max_value`` cannot be both ``None``.

        Args:
            min_value (int, float or ``None``, optional): Specifies
                the lower bound. If ``min_value`` is ``None``,
                there is no lower bound. Default: ``None``
            max_value (int, float or ``None``, optional): Specifies
                the upper bound. If ``max_value`` is ``None``,
                there is no upper bound. Default: ``None``

        Returns:
            ``BaseBatchedTensor``: A batch with clamped values.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(2, 5))
            >>> batch.clamp(min_value=2, max_value=5)
            tensor([[2, 2, 2, 3, 4],
                    [5, 5, 5, 5, 5]], batch_dim=0)
            >>> batch.clamp(min_value=2)
            tensor([[2, 2, 2, 3, 4],
                    [5, 6, 7, 8, 9]])
            >>> batch.clamp(max_value=7)
            tensor([[0, 1, 2, 3, 4],
                    [5, 6, 7, 7, 7]], batch_dim=0)
        """
        return torch.clamp(self, min=min_value, max=max_value)

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
