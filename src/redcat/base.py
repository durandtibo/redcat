r"""Contain the base class to implement a batch."""

from __future__ import annotations

__all__ = ["BaseBatch"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

T = TypeVar("T")
# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatch = TypeVar("TBatch", bound="BaseBatch")


class BaseBatch(Generic[T], ABC):
    r"""Define the base class to implement a batch."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        r"""The batch size."""

    @property
    @abstractmethod
    def data(self) -> T:
        r"""The data in the batch."""

    #################################
    #     Conversion operations     #
    #################################

    @abstractmethod
    def to_data(self) -> Any:
        r"""Return the internal data without the batch wrapper.

        Returns:
            The internal data.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.ones(2, 3))
        >>> data = batch.to_data()
        >>> data
        tensor([[1., 1., 1.], [1., 1., 1.]])

        ```
        """

    ###############################
    #     Creation operations     #
    ###############################

    @abstractmethod
    def clone(self) -> TBatch:
        r"""Create a copy of the current batch.

        Returns:
            A copy of the current batch.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.ones(2, 3))
        >>> batch_copy = batch.clone()
        >>> batch_copy
        tensor([[1., 1., 1.], [1., 1., 1.]], batch_dim=0)

        ```
        """

    #################################
    #     Comparison operations     #
    #################################

    @abstractmethod
    def allclose(
        self, other: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
    ) -> bool:
        r"""Indicate if two batches are equal within a tolerance or not.

        Args:
            other: Specifies the value to compare.
            rtol: Specifies the relative tolerance parameter.
            atol: Specifies the absolute tolerance parameter.
            equal_nan: If ``True``, then two ``NaN``s will be considered equal.

        Returns:
            ``True`` if the batches are equal within a tolerance,
                ``False`` otherwise.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch1 = BatchedTensor(torch.ones(2, 3))
        >>> batch2 = BatchedTensor(torch.full((2, 3), 1.5))
        >>> batch1.allclose(batch2, atol=1, rtol=0)
        True

        ```
        """

    @abstractmethod
    def allequal(self, other: Any) -> bool:
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
        >>> BatchedTensor(torch.ones(2, 3)).allequal(BatchedTensor(torch.zeros(2, 3)))
        False

        ```
        """

    ###########################################################
    #     Mathematical | advanced arithmetical operations     #
    ###########################################################

    @abstractmethod
    def permute_along_batch(self, permutation: Sequence[int] | Tensor) -> TBatch:
        r"""Permutes the data/batch along the batch dimension.

        Args:
            permutation: Specifies the permutation to use on the data.
                The dimension of the permutation input should be
                compatible with the shape of the data.

        Returns:
            A new batch with permuted data.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
        >>> batch.permute_along_batch([2, 1, 3, 0, 4])
        tensor([[4, 5],
                [2, 3],
                [6, 7],
                [0, 1],
                [8, 9]], batch_dim=0)

        ```
        """

    @abstractmethod
    def permute_along_batch_(self, permutation: Sequence[int] | Tensor) -> None:
        r"""Permutes the data/batch along the batch dimension.

        Args:
            permutation: Specifies the permutation to use on the data.
                The dimension of the permutation input should be
                compatible with the shape of the data.

        Example usage:

        ```pycon
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

        ```
        """

    def shuffle_along_batch(self, generator: torch.Generator | None = None) -> TBatch:
        r"""Shuffles the data/batch along the batch dimension.

        Args:
            generator: Specifies an optional pseudo random number
                generator.

        Returns:
            A new batch with shuffled data.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
        >>> batch.shuffle_along_batch()
        tensor([[...]], batch_dim=0)

        ```
        """
        return self.permute_along_batch(torch.randperm(self.batch_size, generator=generator))

    def shuffle_along_batch_(self, generator: torch.Generator | None = None) -> None:
        r"""Shuffles the data/batch along the batch dimension.

        Args:
            generator: Specifies an optional pseudo random number
                generator.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.arange(10).view(5, 2))
        >>> batch.shuffle_along_batch_()
        >>> batch
        tensor([[...]], batch_dim=0)

        ```
        """
        self.permute_along_batch_(torch.randperm(self.batch_size, generator=generator))

    ################################################
    #     Mathematical | point-wise operations     #
    ################################################

    ###########################################
    #     Mathematical | trigo operations     #
    ###########################################

    ##########################################################
    #    Indexing, slicing, joining, mutating operations     #
    ##########################################################

    @abstractmethod
    def append(self, other: BaseBatch) -> None:
        r"""Append a new batch to the current batch along the batch
        dimension.

        Args:
            other: Specifies the batch to append at the end of current
                batch.

        Example usage:

        ```pycon
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

        ```
        """

    @abstractmethod
    def chunk_along_batch(self, chunks: int) -> tuple[TBatch, ...]:
        r"""Split the batch into chunks along the batch dimension.

        Args:
            chunks: Specifies the number of chunks.

        Returns:
            The batch split into chunks along the batch dimension.

        Raises:
            RuntimeError: if the number of chunks is incorrect

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> BatchedTensor(torch.arange(10).view(5, 2)).chunk_along_batch(chunks=3)
        (tensor([[0, 1], [2, 3]], batch_dim=0),
         tensor([[4, 5], [6, 7]], batch_dim=0),
         tensor([[8, 9]], batch_dim=0))

        ```
        """

    @abstractmethod
    def extend(self, other: Iterable[BaseBatch]) -> None:
        r"""Extend the current batch by appending all the batches from
        the iterable.

        This method should be used with batches of similar nature.
        For example, it is possible to extend a batch representing
        data as ``torch.Tensor`` by another batch representing data
        as ``torch.Tensor``, but it is usually not possible to extend
        a batch representing data ``torch.Tensor`` by a batch
        representing data with a dictionary. Please check each
        implementation to know the supported batch implementations.

        Args:
            other: Specifies the batches to append to the current
                batch.

        Raises:
            TypeError: if there is no available implementation for the
                input batch type.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.ones(2, 3))
        >>> batch.extend([BatchedTensor(torch.zeros(1, 3)), BatchedTensor(torch.full((1, 3), 2.0))])
        >>> batch
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [0., 0., 0.],
                [2., 2., 2.]], batch_dim=0)

        ```
        """

    @abstractmethod
    def index_select_along_batch(self, index: Tensor | Sequence[int]) -> BaseBatch:
        r"""Select data at the given indices along the batch dimension.

        Args:
            index: Specifies the indices to select.

        Returns:
            A new batch which indexes ``self`` along the batch
                dimension using the entries in ``index``.

        Example usage:

        ```pycon
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

        ```
        """

    def select_along_batch(self, index: int) -> T:
        r"""Select the batch along the batch dimension at the given
        index.

        Args:
            index: Specifies the index to select.

        Returns:
            The batch sliced along the batch dimension at the given
                index.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> BatchedTensor(torch.arange(10).view(5, 2)).select_along_batch(2)
        tensor([4, 5])

        ```
        """

    @abstractmethod
    def slice_along_batch(self, start: int = 0, stop: int | None = None, step: int = 1) -> TBatch:
        r"""Slices the batch in the batch dimension.

        Args:
            start: Specifies the index where the slicing of object
                starts.
            stop: Specifies the index where the slicing of object
                stops. ``None`` means last.
            step: Specifies the increment between each index for
                slicing.

        Returns:
            A slice of the current batch.

        Example usage:

        ```pycon
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

        ```
        """

    @abstractmethod
    def split_along_batch(self, split_size_or_sections: int | Sequence[int]) -> tuple[TBatch, ...]:
        r"""Split the batch into chunks along the batch dimension.

        Args:
            split_size_or_sections: Specifies the size of a single
                chunk or list of sizes for each chunk.

        Returns:
            The batch split into chunks along the batch dimension.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> BatchedTensor(torch.arange(10).view(5, 2)).split_along_batch(2)
        (tensor([[0, 1], [2, 3]], batch_dim=0),
         tensor([[4, 5], [6, 7]], batch_dim=0),
         tensor([[8, 9]], batch_dim=0))

        ```
        """

    ########################
    #     mini-batches     #
    ########################

    def get_num_minibatches(self, batch_size: int, drop_last: bool = False) -> int:
        r"""Get the number of mini-batches for a given batch size.

        Args:
            batch_size: Specifies the target batch size of the
                mini-batches.
            drop_last: If ``True``, the last batch is dropped if
                it is not full, otherwise it is returned.

        Returns:
            The number of mini-batches.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.arange(10))
        >>> batch.get_num_minibatches(batch_size=4)
        3
        >>> batch.get_num_minibatches(batch_size=4, drop_last=True)
        2

        ```
        """
        if drop_last:
            return self.batch_size // batch_size
        return (self.batch_size + batch_size - 1) // batch_size

    def to_minibatches(
        self,
        batch_size: int,
        drop_last: bool = False,
        deepcopy: bool = False,
    ) -> Iterable[TBatch]:
        r"""Get the mini-batches of the current batch.

        Args:
            batch_size: Specifies the target batch size of the
                mini-batches.
            drop_last: If ``True``, the last batch is dropped if it is
                not full, otherwise it is returned.
            deepcopy: If ``True``, a deepcopy of the batch is performed
                before to return the mini-batches. If ``False``, each
                chunk is a view of the original batch/tensor.
                Using deepcopy allows a deterministic behavior when
                in-place operations are performed on the data.

        Returns:
            The mini-batches.

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.arange(20).view(10, 2))
        >>> list(batch.to_minibatches(batch_size=4))
        [tensor([[0, 1],
                 [2, 3],
                 [4, 5],
                 [6, 7]], batch_dim=0),
         tensor([[ 8,  9],
                 [10, 11],
                 [12, 13],
                 [14, 15]], batch_dim=0),
         tensor([[16, 17],
                 [18, 19]], batch_dim=0)]
        >>> list(batch.to_minibatches(batch_size=4, drop_last=True))
        [tensor([[0, 1],
                 [2, 3],
                 [4, 5],
                 [6, 7]], batch_dim=0),
         tensor([[ 8,  9],
                 [10, 11],
                 [12, 13],
                 [14, 15]], batch_dim=0)]

        ```
        """
        batch = self
        if deepcopy:
            batch = batch.clone()
        if drop_last:
            batch = self.slice_along_batch(
                stop=int(self.get_num_minibatches(batch_size, drop_last) * batch_size)
            )
        return batch.split_along_batch(batch_size)

    #################
    #     Other     #
    #################

    @abstractmethod
    def summary(self) -> str:
        r"""Return a summary of the current batch.

        Returns:
            The summary of the current batch

        Example usage:

        ```pycon
        >>> import torch
        >>> from redcat import BatchedTensor
        >>> batch = BatchedTensor(torch.arange(20).view(10, 2))
        >>> print(batch.summary())
        BatchedTensor(dtype=torch.int64, shape=torch.Size([10, 2]), device=cpu, batch_dim=0)

        ```
        """
