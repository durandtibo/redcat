from __future__ import annotations

__all__ = ["BatchDict", "check_same_batch_size", "check_same_keys", "get_seq_lens"]

from collections.abc import (
    Hashable,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Sequence,
    ValuesView,
)
from typing import Any, TypeVar

import numpy as np
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.format import str_indent, str_mapping
from torch import Tensor

from redcat.base import BaseBatch

TBaseBatch = TypeVar("TBaseBatch", bound=BaseBatch)
# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchDict = TypeVar("TBatchDict", bound="BatchDict")


class BatchDict(BaseBatch[dict[Hashable, TBaseBatch]]):
    r"""Implements a batch object to represent a dictionary of batches.

    Args:
    ----
        data (dict): Specifies the dictionary of batches.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from redcat import BatchDict, BatchList, BatchedTensorSeq
        >>> batch = BatchDict(
        ...     {
        ...         "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
        ...         "key2": BatchList(["a", "b"]),
        ...     }
        ... )
        >>> batch
        BatchDict(
          (key1): tensor([[0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9]], batch_dim=0, seq_dim=1)
          (key2): BatchList(data=['a', 'b'])
        )
    """

    def __init__(self, data: dict[Hashable, TBaseBatch]) -> None:
        if not isinstance(data, dict):
            raise TypeError(f"Incorrect type. Expect a dict but received {type(data)}")
        check_same_batch_size(data)
        self._data = data

    def __repr__(self) -> str:
        data_str = str_indent(str_mapping({key: repr(value) for key, value in self._data.items()}))
        return f"{self.__class__.__qualname__}(\n  {data_str}\n)"

    @property
    def batch_size(self) -> int:
        return next(iter(self._data.values())).batch_size

    @property
    def data(self) -> dict[Hashable, TBaseBatch]:
        return self._data

    #################################
    #     Dictionary operations     #
    #################################

    def __contains__(self, key: Hashable) -> bool:
        return key in self._data

    def __getitem__(self, key: Hashable) -> TBaseBatch:
        return self._data[key]

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __setitem__(self, key: Hashable, value: TBaseBatch) -> None:
        if value.batch_size != self.batch_size:
            raise RuntimeError(
                f"Incorrect batch size. Expected {self.batch_size} but received {value.batch_size}"
            )
        self._data[key] = value

    def get(self, key: Hashable, default: TBaseBatch | None = None) -> TBaseBatch | None:
        return self._data.get(key, default)

    def items(self) -> ItemsView:
        return self._data.items()

    def keys(self) -> KeysView:
        return self._data.keys()

    def values(self) -> ValuesView:
        return self._data.values()

    #################################
    #     Conversion operations     #
    #################################

    def to_data(self) -> dict:
        return {key: value.to_data() for key, value in self._data.items()}

    ###############################
    #     Creation operations     #
    ###############################

    def clone(self) -> TBatchDict:
        return self.__class__({key: batch.clone() for key, batch in self._data.items()})

    #################################
    #     Comparison operations     #
    #################################

    def allclose(
        self, other: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
    ) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_allclose(
            self.data, other.data, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self.data, other.data)

    ###########################################################
    #     Mathematical | advanced arithmetical operations     #
    ###########################################################

    def permute_along_batch(self, permutation: Sequence[int] | Tensor) -> TBatchDict:
        return self.__class__(
            {k: v.permute_along_batch(permutation) for k, v in self._data.items()}
        )

    def permute_along_batch_(self, permutation: Sequence[int] | Tensor) -> None:
        for value in self._data.values():
            value.permute_along_batch_(permutation)

    def permute_along_seq(self, permutation: Sequence[int] | Tensor) -> TBatchDict:
        r"""Permutes the data along the sequence dimension.

        The same permutation is applied on all the sequences. This
        method should be called only if all the sequences have the
        same length.

        This method only permutes the values that implement
        ``permute_along_seq``.

        Args:
        ----
            permutation (sequence or ``torch.Tensor`` of type long
                and shape ``(dimension,)``): Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.

        Returns:
        -------
            ``BatchDict``: A new batch with permuted data.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchDict, BatchList, BatchedTensorSeq
            >>> batch = BatchDict(
            ...     {
            ...         "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            ...         "key2": BatchList(["a", "b"]),
            ...     }
            ... )
            >>> batch.permute_along_seq([2, 1, 3, 0, 4])
            BatchDict(
              (key1): tensor([[2, 1, 3, 0, 4],
                             [7, 6, 8, 5, 9]], batch_dim=0, seq_dim=1)
              (key2): BatchList(data=['a', 'b'])
            )
        """
        out = {}
        for key, val in self._data.items():
            if hasattr(val, "permute_along_seq"):
                val = val.permute_along_seq(permutation)
            out[key] = val
        return self.__class__(out)

    def permute_along_seq_(self, permutation: Sequence[int] | Tensor) -> None:
        r"""Permutes the data along the sequence dimension.

        The same permutation is applied on all the sequences. This
        method should be called only if all the sequences have the
        same length.

        This method only permutes the values that implement
        ``permute_along_seq``.

        Args:
        ----
            permutation (sequence or ``torch.Tensor`` of type long
                and shape ``(dimension,)``): Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchDict, BatchList, BatchedTensorSeq
            >>> batch = BatchDict(
            ...     {
            ...         "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            ...         "key2": BatchList(["a", "b"]),
            ...     }
            ... )
            >>> batch.permute_along_seq_([2, 1, 3, 0, 4])
            >>> batch
            BatchDict(
              (key1): tensor([[2, 1, 3, 0, 4],
                             [7, 6, 8, 5, 9]], batch_dim=0, seq_dim=1)
              (key2): BatchList(data=['a', 'b'])
            )
        """
        for val in self._data.values():
            if hasattr(val, "permute_along_seq_"):
                val.permute_along_seq_(permutation)

    def shuffle_along_seq(self, generator: torch.Generator | None = None) -> TBatchDict:
        r"""Shuffles the data along the sequence dimension.

        This method should be called only if all the sequences have
        the same length.

        Args:
        ----
            generator (``torch.Generator`` or ``None``, optional):
                Specifies an optional random generator.
                Default: ``None``

        Returns:
        -------
            ``BatchDict``:  A new batch with shuffled data.

        Raises:
            RuntimeError if the batch has multiple sequence lengths.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchDict, BatchList, BatchedTensorSeq
            >>> batch = BatchDict(
            ...     {
            ...         "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            ...         "key2": BatchList(["a", "b"]),
            ...     }
            ... )
            >>> batch.shuffle_along_seq()
            BatchDict(
              (key1): tensor([[...]], batch_dim=0, seq_dim=1)
              (key2): BatchList(data=['a', 'b'])
            )
        """
        seq_lens = get_seq_lens(self._data)
        if not seq_lens:
            return self
        if len(seq_lens) > 1:
            raise RuntimeError(
                f"Invalid operation because the batch has multiple sequence lengths: {seq_lens}"
            )
        return self.permute_along_seq(torch.randperm(seq_lens.pop(), generator=generator))

    def shuffle_along_seq_(self, generator: torch.Generator | None = None) -> None:
        r"""Shuffles the data along the sequence dimension.

        This method should be called only if all the sequences have
        the same length.

        Args:
        ----
            generator (``torch.Generator`` or ``None``, optional):
                Specifies an optional random generator.
                Default: ``None``

        Raises:
        ------
            RuntimeError if the batch has multiple sequence lengths.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchDict, BatchList, BatchedTensorSeq
            >>> batch = BatchDict(
            ...     {
            ...         "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            ...         "key2": BatchList(["a", "b"]),
            ...     }
            ... )
            >>> batch.shuffle_along_seq()
            >>> batch
            BatchDict(
              (key1): tensor([[...]], batch_dim=0, seq_dim=1)
              (key2): BatchList(data=['a', 'b'])
            )
        """
        seq_lens = get_seq_lens(self._data)
        if not seq_lens:
            return
        if len(seq_lens) > 1:
            raise RuntimeError(
                f"Invalid operation because the batch has multiple sequence lengths: {seq_lens}"
            )
        self.permute_along_seq_(torch.randperm(seq_lens.pop(), generator=generator))

    ################################################
    #     Mathematical | point-wise operations     #
    ################################################

    ###########################################
    #     Mathematical | trigo operations     #
    ###########################################

    ##########################################################
    #    Indexing, slicing, joining, mutating operations     #
    ##########################################################

    def append(self, other: TBaseBatch) -> None:
        check_same_keys(self.data, other.data)
        for key, value in self._data.items():
            value.append(other[key])

    def cat_along_seq(self, batches: TBaseBatch | Sequence[TBaseBatch]) -> TBatchDict:
        r"""Concatenates the data of the batch(es) to the current batch
        along the sequence dimension and creates a new batch.

        Note that only the sequences are concatenated.

        Args:
        ----
            batches (``BatchDict`` or  ``Sequence``): Specifies the
                batch(es) to concatenate along the sequence dimension.

        Returns:
        -------
            ``BatchDict``: A batch with the concatenated data
                along the sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchDict, BatchList, BatchedTensorSeq
            >>> b = BatchDict(
            ...     {
            ...         "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            ...         "key2": BatchList(["a", "b"]),
            ...     }
            ... )
            >>> b.cat_along_seq(
            ...     BatchDict({"key1": BatchedTensorSeq(torch.tensor([[10, 11, 12], [20, 21, 22]]))})
            ... )
            BatchDict(
              (key1): tensor([[ 0,  1,  2,  3,  4, 10, 11, 12],
                        [ 5,  6,  7,  8,  9, 20, 21, 22]], batch_dim=0, seq_dim=1)
              (key2): BatchList(data=['a', 'b'])
            )
        """
        if isinstance(batches, BatchDict):
            batches = [batches]
        out = {}
        for key, val in self._data.items():
            if hasattr(val, "cat_along_seq"):
                val = val.cat_along_seq([batch[key] for batch in batches])
            out[key] = val
        return self.__class__(out)

    def cat_along_seq_(self, batches: TBaseBatch | Sequence[TBaseBatch]) -> None:
        r"""Concatenates the data of the batch(es) to the current batch
        along the sequence dimension and creates a new batch.

        Note that only the sequences are concatenated.

        Args:
        ----
            batches (``BatchDict`` or  ``Sequence``): Specifies the
                batch(es) to concatenate along the sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchDict, BatchList, BatchedTensorSeq
            >>> b = BatchDict(
            ...     {
            ...         "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            ...         "key2": BatchList(["a", "b"]),
            ...     }
            ... )
            >>> b.cat_along_seq_(
            ...     BatchDict({"key1": BatchedTensorSeq(torch.tensor([[10, 11, 12], [20, 21, 22]]))})
            ... )
            >>> b
            BatchDict(
              (key1): tensor([[ 0,  1,  2,  3,  4, 10, 11, 12],
                        [ 5,  6,  7,  8,  9, 20, 21, 22]], batch_dim=0, seq_dim=1)
              (key2): BatchList(data=['a', 'b'])
            )
        """
        if isinstance(batches, BatchDict):
            batches = [batches]
        for key, val in self._data.items():
            if hasattr(val, "cat_along_seq_"):
                val.cat_along_seq_([batch[key] for batch in batches])

    def chunk_along_batch(self, chunks: int) -> tuple[TBatchDict, ...]:
        keys = self._data.keys()
        batches = []
        for values in zip(*[batch.chunk_along_batch(chunks) for batch in self._data.values()]):
            batches.append(self.__class__({key: value for key, value in zip(keys, values)}))
        return tuple(batches)

    def extend(self, other: Iterable[BatchDict | Sequence[TBatchDict]]) -> None:
        for batch in other:
            self.append(batch)

    def index_select_along_batch(self, index: Tensor | Sequence[int]) -> TBatchDict:
        return self.__class__(
            {key: value.index_select_along_batch(index) for key, value in self._data.items()}
        )

    def index_select_along_seq(self, index: Tensor | Sequence[int]) -> TBatchDict:
        r"""Slices the batch along the sequence dimension at the given
        indices.

        Args:
        ----
            index (``torch.Tensor`` of type long and shape
                ``(seq_len,)`` or ``(batch_size, seq_len,)`` or
                ``Sequence``): Specifies the indices to select.

        Returns:
        -------
            ``BatchDict``: A new batch sliced along the sequence
                dimension at the given indices.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchDict, BatchList, BatchedTensorSeq
            >>> batch = BatchDict(
            ...     {
            ...         "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            ...         "key2": BatchList(["a", "b"]),
            ...     }
            ... )
            >>> batch.index_select_along_seq([2, 4])
            BatchDict(
              (key1): tensor([[2, 4], [7, 9]], batch_dim=0, seq_dim=1)
              (key2): BatchList(data=['a', 'b'])
            )
            >>> batch.index_select_along_seq(torch.tensor([2, 4]))
            BatchDict(
              (key1): tensor([[2, 4], [7, 9]], batch_dim=0, seq_dim=1)
              (key2): BatchList(data=['a', 'b'])
            )
            >>> batch.index_select_along_seq(torch.tensor([[2, 4], [4, 3]]))
            BatchDict(
              (key1): tensor([[2, 4], [9, 8]], batch_dim=0, seq_dim=1)
              (key2): BatchList(data=['a', 'b'])
            )
        """
        out = {}
        for key, val in self._data.items():
            if hasattr(val, "index_select_along_seq"):
                val = val.index_select_along_seq(index)
            out[key] = val
        return self.__class__(out)

    def repeat_along_seq(self, repeats: int) -> TBatchDict:
        r"""Repeats the batch along the sequence dimension.

        Args:
        ----
            repeats (int): Specifies the number of times to repeat
                the batch along the sequence dimension.

        Returns:
        -------
            ``BatchDict``: A repeated version of the input batch.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchDict, BatchList, BatchedTensorSeq
            >>> batch = BatchDict(
            ...     {
            ...         "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            ...         "key2": BatchList(["a", "b"]),
            ...     }
            ... )
            >>> batch.repeat_along_seq(2)
            BatchDict(
              (key1): tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                        [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]], batch_dim=0, seq_dim=1)
              (key2): BatchList(data=['a', 'b'])
            )
        """
        out = {}
        for key, val in self._data.items():
            if hasattr(val, "repeat_along_seq"):
                val = val.repeat_along_seq(repeats)
            out[key] = val
        return self.__class__(out)

    def select_along_batch(self, index: int) -> dict:
        return {key: value.select_along_batch(index) for key, value in self._data.items()}

    def slice_along_batch(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> TBatchDict:
        return self.__class__(
            {key: value.slice_along_batch(start, stop, step) for key, value in self._data.items()}
        )

    def slice_along_seq(self, start: int = 0, stop: int | None = None, step: int = 1) -> TBatchDict:
        r"""Slices the batch in the sequence dimension.

        Args:
        ----
            start (int, optional): Specifies the index where the
                slicing of object starts. Default: ``0``
            stop (int, optional): Specifies the index where the
                slicing of object stops. ``None`` means last.
                Default: ``None``
            step (int, optional): Specifies the increment between
                each index for slicing. Default: ``1``

        Returns:
        -------
            ``BatchedTensorSeq``: A slice of the current batch.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchDict, BatchList, BatchedTensorSeq
            >>> batch = BatchDict(
            ...     {
            ...         "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            ...         "key2": BatchList(["a", "b"]),
            ...     }
            ... )
            >>> batch.slice_along_seq(start=2)
            BatchDict(
              (key1): tensor([[2, 3, 4],
                        [7, 8, 9]], batch_dim=0, seq_dim=1)
              (key2): BatchList(data=['a', 'b'])
            )
            >>> batch.slice_along_seq(stop=3)
            BatchDict(
              (key1): tensor([[0, 1, 2],
                        [5, 6, 7]], batch_dim=0, seq_dim=1)
              (key2): BatchList(data=['a', 'b'])
            )
            >>> batch.slice_along_seq(step=2)
            BatchDict(
              (key1): tensor([[0, 2, 4],
                        [5, 7, 9]], batch_dim=0, seq_dim=1)
              (key2): BatchList(data=['a', 'b'])
            )
        """
        out = {}
        for key, val in self._data.items():
            if hasattr(val, "slice_along_seq"):
                val = val.slice_along_seq(start, stop, step)
            out[key] = val
        return self.__class__(out)

    def split_along_batch(
        self, split_size_or_sections: int | Sequence[int]
    ) -> tuple[TBatchDict, ...]:
        keys = self._data.keys()
        batches = []
        for values in zip(
            *[batch.split_along_batch(split_size_or_sections) for batch in self._data.values()]
        ):
            batches.append(self.__class__({key: value for key, value in zip(keys, values)}))
        return tuple(batches)

    def take_along_seq(self, indices: TBaseBatch | np.ndarray | Tensor | Sequence) -> TBatchDict:
        r"""Takes values along the sequence dimension.

        Args:
        ----
            indices (``BaseBatch`` or ``numpy.ndarray`` or
                ``torch.Tensor`` or `` Specifies the indices to take
                along the batch dimension.

        Returns:
        -------
            ``BatchDict``: The batch with the selected data.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchDict, BatchList, BatchedTensorSeq
            >>> batch = BatchDict(
            ...     {
            ...         "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            ...         "key2": BatchList(["a", "b"]),
            ...     }
            ... )
            >>> batch.take_along_seq(torch.tensor([[3, 0, 1], [2, 3, 4]]))
            BatchDict(
              (key1): tensor([[3, 0, 1],
                        [7, 8, 9]], batch_dim=0, seq_dim=1)
              (key2): BatchList(data=['a', 'b'])
            )
        """
        out = {}
        for key, val in self._data.items():
            if hasattr(val, "take_along_seq"):
                val = val.take_along_seq(indices)
            out[key] = val
        return self.__class__(out)

    ########################
    #     mini-batches     #
    ########################

    #################
    #     Other     #
    #################

    def summary(self) -> str:
        data_str = str_mapping({key: value.summary() for key, value in self._data.items()})
        return f"{self.__class__.__qualname__}(\n  {str_indent(data_str)}\n)"


def check_same_batch_size(data: dict[Hashable, BaseBatch]) -> None:
    r"""Checks if the all the batches in a group have the same batch
    size.

    Args:
    ----
        group (``BaseBatch`` or dict or sequence): Specifies the group
            of batches to check.

    Raises:
    ------
        RuntimeError if there are several batch sizes.

    Example usage:

    .. code-block:: pycon

        >>> from redcat import BatchedTensor, BatchedTensorSeq
        >>> from redcat.batchdict import check_same_batch_size
        >>> check_same_batch_size(
        ...     {
        ...         "key1": BatchedTensorSeq(torch.ones(2, 3)),
        ...         "key2": BatchedTensor(torch.ones(2, 6)),
        ...     }
        ... )
    """
    if not data:
        raise RuntimeError("The dictionary cannot be empty")
    batch_sizes = {val.batch_size for val in data.values()}
    if len(batch_sizes) != 1:
        raise RuntimeError(
            "Incorrect batch size. A single batch size is expected but received several values: "
            f"{batch_sizes}"
        )


def check_same_keys(data1: dict, data2: dict) -> None:
    r"""Checks if the dictionaries have the same keys.

    Args:
    ----
        data1 (dict): Specifies the first dictionary.
        data2 (dict): Specifies the second dictionary.

    Raises:
    ------
        RuntimeError if the keys are different.

    Example usage:

    .. code-block:: pycon

        >>> from redcat import BatchedTensor, BatchedTensorSeq
        >>> from redcat.batchdict import check_same_keys
        >>> check_same_keys(
        ...     {
        ...         "key1": BatchedTensorSeq(torch.ones(2, 3)),
        ...         "key2": BatchedTensor(torch.ones(2, 6)),
        ...     },
        ...     {
        ...         "key1": BatchedTensorSeq(torch.zeros(2, 4)),
        ...         "key2": BatchedTensor(torch.zeros(2, 4)),
        ...     },
        ... )
    """
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    if keys1 != keys2:
        raise RuntimeError(f"Keys do not match: ({keys1} vs {keys2})")


def get_seq_lens(data: dict[Hashable, BaseBatch]) -> set[int]:
    r"""Gets the sequence lengths from the inputs.

    Args:
    ----
        data (dict): Specifies the data with the sequences.

    Returns:
    -------
        set: The sequence lengths.

    Example usage:

    .. code-block:: pycon

        >>> from redcat import BatchedTensor, BatchedTensorSeq
        >>> from redcat.batchdict import get_seq_lens
        >>> get_seq_lens(
        ...     {
        ...         "key1": BatchedTensorSeq(torch.ones(2, 3)),
        ...         "key2": BatchedTensor(torch.ones(2, 6)),
        ...     }
        ... )
        {3}
    """
    return {val.seq_len for val in data.values() if hasattr(val, "seq_len")}
