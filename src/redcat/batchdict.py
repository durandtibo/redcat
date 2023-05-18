from __future__ import annotations

__all__ = ["BatchDict", "check_same_batch_size", "check_same_keys", "get_seq_lens"]

import copy
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, TypeVar

import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.format import str_indent
from torch import Tensor

from redcat.base import BaseBatch
from redcat.utils.format import str_mapping

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchDict = TypeVar("TBatchDict", bound="BatchDict")


class BatchDict(BaseBatch[dict[Hashable, BaseBatch]]):
    r"""Implements a batch object to represent a dictionary of batches.

    Args:
        data (dict): Specifies the dictionary of batches.
    """

    def __init__(self, data: dict[Hashable, BaseBatch]) -> None:
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
    def data(self) -> dict[Hashable, BaseBatch]:
        return self._data

    ###############################
    #     Creation operations     #
    ###############################

    def clone(self) -> TBatchDict:
        return self.__class__(copy.deepcopy(self._data))

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
            permutation (sequence or ``torch.Tensor`` of type long
                and shape ``(dimension,)``): Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.

        Returns:
            ``BatchDict``: A new batch with permuted data.

        Example usage:

        .. code-block:: python

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
              (key1) tensor([[2, 1, 3, 0, 4],
                             [7, 6, 8, 5, 9]], batch_dim=0, seq_dim=1)
              (key2) BatchList(data=['a', 'b'])
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
            permutation (sequence or ``torch.Tensor`` of type long
                and shape ``(dimension,)``): Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.

        Example usage:

        .. code-block:: python

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
              (key1) tensor([[2, 1, 3, 0, 4],
                             [7, 6, 8, 5, 9]], batch_dim=0, seq_dim=1)
              (key2) BatchList(data=['a', 'b'])
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
            generator (``torch.Generator`` or ``None``, optional):
                Specifies an optional random generator.
                Default: ``None``

        Returns:
            ``BatchDict``:  A new batch with shuffled data.

        Raises:
            RuntimeError if the batch has multiple sequence lengths.

        Example usage:

        .. code-block:: python

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
              (key1) tensor([[2, 1, 3, 0, 4],
                             [7, 6, 8, 5, 9]], batch_dim=0, seq_dim=1)
              (key2) BatchList(data=['a', 'b'])
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
            generator (``torch.Generator`` or ``None``, optional):
                Specifies an optional random generator.
                Default: ``None``

        Raises:
            RuntimeError if the batch has multiple sequence lengths.

        Example usage:

        .. code-block:: python

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
              (key1) tensor([[2, 1, 3, 0, 4],
                             [7, 6, 8, 5, 9]], batch_dim=0, seq_dim=1)
              (key2) BatchList(data=['a', 'b'])
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

    def __getitem__(self, key: Hashable) -> BaseBatch:
        return self._data[key]

    def __setitem__(self, key: Hashable, value: BaseBatch) -> None:
        if value.batch_size != self.batch_size:
            raise RuntimeError(
                f"Incorrect batch size. Expected {self.batch_size} but received {value.batch_size}"
            )
        self._data[key] = value

    def append(self, other: BatchDict) -> None:
        check_same_keys(self.data, other.data)
        for key, value in self._data.items():
            value.append(other[key])

    def cat_along_seq(self, batches: BatchDict | Sequence[BatchDict]) -> TBatchDict:
        r"""Concatenates the data of the batch(es) to the current batch
        along the sequence dimension and creates a new batch.

        Note that only the sequences are concatenated.

        Args:
            batches (``BatchDict`` or  ``Sequence``): Specifies the
                batch(es) to concatenate along the sequence dimension.

        Returns:
            ``BatchDict``: A batch with the concatenated data
                along the sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchDict, BatchList, BatchedTensorSeq
            >>> b = BatchDict(
            ...     {
            ...         "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            ...         "key2": BatchList(["a", "b"]),
            ...     }
            ... )
            >>> b.cat_along_seq(
            ...     BatchDict(
            ...         {"key1": BatchedTensorSeq(torch.tensor([[10, 11, 12], [20, 21, 22]]))}
            ...     )
            ... )
            BatchDict(
              (key1) tensor([[ 0,  1,  2,  3,  4, 10, 11, 12],
                             [ 5,  6,  7,  8,  9, 20, 21, 22]], batch_dim=0, seq_dim=1)
              (key2) BatchList(data=['a', 'b'])
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

    def cat_along_seq_(self, batches: BatchDict | Sequence[BatchDict]) -> None:
        r"""Concatenates the data of the batch(es) to the current batch
        along the sequence dimension and creates a new batch.

        Note that only the sequences are concatenated.

        Args:
            batches (``BatchDict`` or  ``Sequence``): Specifies the
                batch(es) to concatenate along the sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchDict, BatchList, BatchedTensorSeq
            >>> b = BatchDict(
            ...     {
            ...         "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            ...         "key2": BatchList(["a", "b"]),
            ...     }
            ... )
            >>> b.cat_along_seq_(
            ...     BatchDict(
            ...         {"key1": BatchedTensorSeq(torch.tensor([[10, 11, 12], [20, 21, 22]]))}
            ...     )
            ... )
            >>> b
            BatchDict(
              (key1) tensor([[ 0,  1,  2,  3,  4, 10, 11, 12],
                             [ 5,  6,  7,  8,  9, 20, 21, 22]], batch_dim=0, seq_dim=1)
              (key2) BatchList(data=['a', 'b'])
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

    def select_along_batch(self, index: int) -> dict:
        return {key: value.select_along_batch(index) for key, value in self._data.items()}

    def slice_along_batch(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> TBatchDict:
        return self.__class__(
            {key: value.slice_along_batch(start, stop, step) for key, value in self._data.items()}
        )

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

    ########################
    #     mini-batches     #
    ########################


def check_same_batch_size(data: dict[Hashable, BaseBatch]) -> None:
    r"""Checks if the all the batches in a group have the same batch
    size.

    Args:
        group (``BaseBatch`` or dict or sequence): Specifies the group
            of batches to check.

    Raises:
        RuntimeError if there are several batch sizes.
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
        data1 (dict): Specifies the first dictionary.
        data2 (dict): Specifies the second dictionary.

    Raises:
        RuntimeError if the keys are different.
    """
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    if keys1 != keys2:
        raise RuntimeError(f"Keys do not match: ({keys1} vs {keys2})")


def get_seq_lens(data: dict[Hashable, BaseBatch]) -> set[int]:
    r"""Gets the sequence lengths from the inputs.

    Args:
        data (dict): Specifies the data with the sequences.

    Returns:
        set: The sequence lengths.
    """
    return {val.seq_len for val in data.values() if hasattr(val, "seq_len")}
