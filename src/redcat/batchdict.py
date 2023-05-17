from __future__ import annotations

__all__ = ["BatchDict", "check_same_batch_size", "check_same_keys"]

import copy
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, TypeVar

from coola import objects_are_allclose, objects_are_equal
from coola.utils.format import str_indent
from torch import Tensor

from redcat.base import BaseBatch

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
        data_str = str_indent(
            # to_torch_mapping_str({key: repr(value) for key, value in self._data.items()})
            str({key: repr(value) for key, value in self._data.items()})
        )
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

    def clone(self, *args, **kwargs) -> TBatchDict:
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

    def chunk_along_batch(self, chunks: int) -> tuple[TBatchDict, ...]:
        pass

    def extend(self, other: Iterable[BatchDict | Sequence[TBatchDict]]) -> None:
        for batch in other:
            self.append(batch)

    def index_select_along_batch(self, index: Tensor | Sequence[int]) -> TBatchDict:
        pass

    def select_along_batch(self, index: int) -> dict:
        pass

    def slice_along_batch(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> TBatchDict:
        pass

    def split_along_batch(
        self, split_size_or_sections: int | Sequence[int]
    ) -> tuple[TBatchDict, ...]:
        pass

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
