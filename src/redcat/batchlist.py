from __future__ import annotations

__all__ = ["BatchList"]

import copy
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

from coola import objects_are_allclose, objects_are_equal
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
    ----
        data (list): Specifies the list of examples.

    Raises:
    ------
        TypeError if the input is not a list.

    Example usage:

    .. code-block:: pycon

        >>> from redcat import BatchList
        >>> batch = BatchList([1, 2, 3])
        >>> batch
        BatchList(data=[1, 2, 3])
    """

    def __init__(self, data: list[T]) -> None:
        if not isinstance(data, list):
            raise TypeError(f"Incorrect type. Expect a list but received {type(data)}")
        self._data = data

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(data={self._data})"

    @property
    def batch_size(self) -> int:
        return len(self._data)

    @property
    def data(self) -> list[T]:
        return self._data

    #################################
    #     Conversion operations     #
    #################################

    def to_data(self) -> list[T]:
        return self._data

    ###############################
    #     Creation operations     #
    ###############################

    def clone(self) -> TBatchList:
        return self._create_new_batch(copy.deepcopy(self._data))

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

    def permute_along_batch(self, permutation: Sequence[int] | Tensor) -> TBatchList:
        return self._create_new_batch([self._data[i] for i in permutation])

    def permute_along_batch_(self, permutation: Sequence[int] | Tensor) -> None:
        self._data = [self._data[i] for i in permutation]

    ################################################
    #     Mathematical | point-wise operations     #
    ################################################

    ###########################################
    #     Mathematical | trigo operations     #
    ###########################################

    ##########################################################
    #    Indexing, slicing, joining, mutating operations     #
    ##########################################################

    def __getitem__(self, index: int | slice) -> list[T]:
        return self._data[index]

    def __setitem__(self, index: int | slice, value: Any) -> None:
        if isinstance(value, BatchList):
            value = value.data
        self._data[index] = value

    def append(self, other: BatchList | Sequence[T]) -> None:
        if isinstance(other, BatchList):
            other = other.data
        self._data.extend(other)

    def chunk_along_batch(self, chunks: int) -> tuple[TBatchList, ...]:
        if chunks < 1:
            raise RuntimeError(f"chunks has to be greater than 0 but received {chunks}")
        return self.split_along_batch(math.ceil(self.batch_size / chunks))

    def extend(self, other: Iterable[BatchList | Sequence[T]]) -> None:
        for batch in other:
            self.append(batch)

    def index_select_along_batch(self, index: Tensor | Sequence[int]) -> TBatchList:
        return self._create_new_batch([self._data[i] for i in index])

    def select_along_batch(self, index: int) -> T:
        return self._data[index]

    def slice_along_batch(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> TBatchList:
        return self._create_new_batch(self._data[start:stop:step])

    def split_along_batch(
        self, split_size_or_sections: int | Sequence[int]
    ) -> tuple[TBatchList, ...]:
        if isinstance(split_size_or_sections, int):
            return tuple(
                self._create_new_batch(self._data[i : i + split_size_or_sections])
                for i in range(0, self.batch_size, split_size_or_sections)
            )
        i = 0
        output = []
        for size in split_size_or_sections:
            output.append(self._create_new_batch(self._data[i : i + size]))
            i += size
        return tuple(output)

    ########################
    #     mini-batches     #
    ########################

    #################
    #     Other     #
    #################

    def apply(self, fn: Callable[[T], T]) -> TBatchList:
        r"""Apply a function to transform the element in the list of the
        current batch.

        Args:
        ----
            fn (``Callable``): Specifies the function to be applied to
                the element in the list. It is the responsibility of
                the user to verify the function applies a valid
                transformation of the data.

        Returns:
        -------
            ``BatchList``: The transformed batch.

        Example usage:

        .. code-block:: pycon


            >>> from redcat import BatchList
            >>> batch = BatchList([1, 2, 3])
            >>> batch.apply(lambda val: val + 2)
            BatchList(data=[3, 4, 5])
        """
        return self._create_new_batch([fn(val) for val in self._data])

    def apply_(self, fn: Callable[[T], T]) -> None:
        r"""Apply a function to transform the element in the list of the
        current batch.

        In-place version of ``apply``.

        Args:
        ----
            fn (``Callable``): Specifies the function to be applied to
                the element in the list. It is the responsibility of
                the user to verify the function applies a valid
                transformation of the data.

        Example usage:

        .. code-block:: pycon


            >>> from redcat import BatchList
            >>> batch = BatchList([1, 2, 3])
            >>> batch.apply_(lambda val: val + 2)
            >>> batch
            BatchList(data=[3, 4, 5])
        """
        self._data = [fn(val) for val in self._data]

    def summary(self) -> str:
        return f"{self.__class__.__qualname__}(batch_size={self.batch_size})"

    def _create_new_batch(self, data: list[T]) -> TBatchList:
        return self.__class__(data)
