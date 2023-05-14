from __future__ import annotations

__all__ = ["BatchList"]

import copy
from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

import torch
from coola import objects_are_equal
from torch import Tensor

from redcat.base import BaseBatch

T = TypeVar("T")
# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchList = TypeVar("TBatchList", bound="BatchList")


class BatchList(BaseBatch[list[T]]):
    r"""Implements a batch object to easily manipulate a list of
    examples.

    Args:
        data (list): Specifies the list of examples.
    """

    def __init__(self, data: list[T]) -> None:
        if not isinstance(data, list):
            raise TypeError(f"Incorrect type. Expect a list but received {type(data)}")
        self._data = data

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(batch_size={self.batch_size:,})"

    @property
    def batch_size(self) -> int:
        r"""int: The batch size."""
        return len(self._data)

    @property
    def data(self) -> list[T]:
        r"""The data in the batch."""
        return self._data

    ###############################
    #     Creation operations     #
    ###############################

    def clone(self, *args, **kwargs) -> TBatchList:
        return self.__class__(data=copy.deepcopy(self._data))

    #################################
    #     Comparison operations     #
    #################################

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
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self.data, other.data)

    ###########################################################
    #     Mathematical | advanced arithmetical operations     #
    ###########################################################

    def permute_along_batch(self, permutation: Sequence[int] | Tensor) -> TBatchList:
        r"""Permutes the data/batch along the batch dimension.

        Args:
            permutation (sequence or ``torch.Tensor`` of type long
                and shape ``(dimension,)``): Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.

        Returns:
            ``BaseBatchedTensor``: A new batch with permuted data.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.permute_along_batch([2, 1, 3, 0, 4])
            tensor([[4, 5],
                    [2, 3],
                    [6, 7],
                    [0, 1],
                    [8, 9]], batch_dim=0)
        """

    def permute_along_batch_(self, permutation: Sequence[int] | torch.Tensor) -> None:
        r"""Permutes the data/batch along the batch dimension.

        Args:
            permutation (sequence or ``torch.Tensor`` of type long
                and shape ``(dimension,)``): Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.permute_along_batch_([2, 1, 3, 0, 4])
            >>> batch
            tensor([[4, 5],
                    [2, 3],
                    [6, 7],
                    [0, 1],
                    [8, 9]], batch_dim=0)
        """

    ################################################
    #     Mathematical | point-wise operations     #
    ################################################

    ###########################################
    #     Mathematical | trigo operations     #
    ###########################################

    ##########################################################
    #    Indexing, slicing, joining, mutating operations     #
    ##########################################################

    def append(self, other: BaseBatch) -> None:
        r"""Appends a new batch to the current batch along the batch
        dimension.

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

    def chunk_along_batch(self, chunks: int) -> tuple[TBatchList, ...]:
        r"""Splits the batch into chunks along the batch dimension.

        Args:
            chunks (int): Specifies the number of chunks.

        Returns:
            tuple: The batch split into chunks along the batch
                dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(5, 2)).chunk_along_batch(chunks=3)
            (tensor([[0, 1], [2, 3]], batch_dim=0),
             tensor([[4, 5], [6, 7]], batch_dim=0),
             tensor([[8, 9]], batch_dim=0))
        """

    def extend(self, other: Iterable[BaseBatch]) -> None:
        r"""Extends the current batch by appending all the batches from
        the iterable.

        This method should be used with batches of similar nature.
        For example, it is possible to extend a batch representing
        data as ``torch.Tensor`` by another batch representing data
        as ``torch.Tensor``, but it is usually not possible to extend
        a batch representing data ``torch.Tensor`` by a batch
        representing data with a dictionary. Please check each
        implementation to know the supported batch implementations.

        Args:
            other (iterable): Specifies the batches to append to the
                current batch.

        Raises:
            TypeError: if there is no available implementation for the
                input batch type.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.extend([BatchedTensor(torch.zeros(1, 3)), BatchedTensor(torch.full((1, 3), 2.0))])
            >>> batch.data
            tensor([[1., 1., 1.],
                    [1., 1., 1.],
                    [0., 0., 0.],
                    [2., 2., 2.]], batch_dim=0)
        """

    def index_select_along_batch(self, index: Tensor | Sequence[int]) -> BaseBatch:
        r"""Selects data at the given indices along the batch dimension.

        Args:
            index (``torch.Tensor`` or list or tuple): Specifies the
                indices to select.

        Returns:
            ``BaseBatch``: A new batch which indexes ``self``
                along the batch dimension using the entries in
                ``index``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
            >>> batch.index_select_along_batch([2, 4])
            tensor([[4, 5],
                    [8, 9]], batch_dim=0)
            >>> batch.index_select_along_batch(torch.tensor([4, 3, 2, 1, 0]))
            tensor([[8, 9],
                    [6, 7],
                    [4, 5],
                    [2, 3],
                    [0, 1]], batch_dim=0)
        """

    def select_along_batch(self, index: int) -> T:
        r"""Selects the batch along the batch dimension at the given
        index.

        Args:
            index (int): Specifies the index to select.

        Returns:
            ``BaseBatch``: The batch sliced along the batch
                dimension at the given index.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(5, 2)).select_along_batch(2)
            tensor([4, 5])
        """

    def slice_along_batch(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> TBatchList:
        r"""Slices the batch in the batch dimension.

        Args:
            start (int, optional): Specifies the index where the
                slicing of object starts. Default: ``0``
            stop (int, optional): Specifies the index where the
                slicing of object stops. ``None`` means last.
                Default: ``None``
            step (int, optional): Specifies the increment between
                each index for slicing. Default: ``1``

        Returns:
            ``BaseBatch``: A slice of the current batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(5, 2)).slice_along_batch(start=2)
            tensor([[4, 5],
                    [6, 7],
                    [8, 9]], batch_dim=0)
            >>> BatchedTensor(torch.arange(10).view(5, 2)).slice_along_batch(stop=3)
            tensor([[0, 1],
                    [2, 3],
                    [4, 5]], batch_dim=0)
            >>> BatchedTensor(torch.arange(10).view(5, 2)).slice_along_batch(step=2)
            tensor([[0, 1],
                    [4, 5],
                    [8, 9]], batch_dim=0)
        """

    def split_along_batch(
        self, split_size_or_sections: int | Sequence[int]
    ) -> tuple[TBatchList, ...]:
        r"""Splits the batch into chunks along the batch dimension.

        Args:
            split_size_or_sections (int or sequence): Specifies the
                size of a single chunk or list of sizes for each chunk.
            deepcopy (bool, optional): If ``True``, a deepcopy of the
                data is performed before to return the chunks.
                If ``False``, each chunk is a view of the original
                batch/tensor. Using deepcopy allows a deterministic
                behavior when in-place operations are performed on
                the data. Default: ``False``

        Returns:
            tuple: The batch split into chunks along the batch
                dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.arange(10).view(5, 2)).split_along_batch(2)
            (tensor([[0, 1], [2, 3]], batch_dim=0),
             tensor([[4, 5], [6, 7]], batch_dim=0),
             tensor([[8, 9]], batch_dim=0))
        """

    def take_along_batch(self, indices: BaseBatch | Tensor | Sequence) -> TBatchList:
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

    ########################
    #     mini-batches     #
    ########################