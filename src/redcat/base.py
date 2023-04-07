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

    ###################################
    #     Arithmetical operations     #
    ###################################

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

    def div(
        self,
        other: BaseBatchedTensor | torch.Tensor | int | float,
        rounding_mode: str | None = None,
    ) -> TBatchedTensor:
        r"""Divides the ``self`` batch by the input ``other`.

        Similar to ``out = self / other`` (in-place)

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
