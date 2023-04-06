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
            ``BatchedTensor``: A new batch with the data after dtype
                and/or device conversion.

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
