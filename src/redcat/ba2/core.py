from __future__ import annotations

__all__ = ["BatchedArray"]

from typing import Any, Literal, TypeVar

import numpy as np
from coola import objects_are_allclose, objects_are_equal
from numpy.typing import ArrayLike, DTypeLike

from redcat.ba import check_data_and_axis, check_same_batch_axis

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")


class BatchedArray:  # (BaseBatch[np.ndarray]):
    r"""Implement a wrapper around a NumPy array to track the batch
    axis."""

    def __init__(self, data: ArrayLike, batch_axis: int = 0, check: bool = True) -> None:
        self._data = np.array(data, copy=False, subok=True)
        self._batch_axis = batch_axis
        if check:
            check_data_and_axis(self._data, self._batch_axis)

    ################################
    #     Core functionalities     #
    ################################

    @property
    def batch_size(self) -> int:
        return self._data.shape[self._batch_axis]

    @property
    def data(self) -> np.ndarray:
        r"""The underlying numpy array."""
        return self._data

    def allclose(
        self, other: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
    ) -> bool:
        if not isinstance(other, self.__class__) or self.batch_axis != other.batch_axis:
            return False
        return objects_are_allclose(
            self.data, other.data, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    def allequal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__) or self.batch_axis != other.batch_axis:
            return False
        return objects_are_equal(self.data, other.data, equal_nan=equal_nan)

    ######################################
    #     Additional functionalities     #
    ######################################

    def __array__(self, dtype: DTypeLike = None, /) -> np.ndarray:
        return self._data.__array__(dtype)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "inner"],
        *inputs: Any,
        **kwargs: Any,
    ) -> TBatchedArray | tuple[TBatchedArray, ...]:
        args = []
        batch_axes = set()
        for inp in inputs:
            if isinstance(inp, self.__class__):
                batch_axes.add(inp.batch_axis)
                inp = inp.data
            args.append(inp)
        check_same_batch_axis(batch_axes)

        results = self._data.__array_ufunc__(ufunc, method, *args, **kwargs)
        if ufunc.nout == 1:
            return self._create_new_batch(results)
        return tuple(self._create_new_batch(res) for res in results)

    def __repr__(self) -> str:
        return repr(self._data)[:-1] + f", batch_axis={self._batch_axis})"

    def __str__(self) -> str:
        return str(self._data) + f"\nwith batch_axis={self._batch_axis}"

    @property
    def batch_axis(self) -> int:
        r"""The batch axis in the array."""
        return self._batch_axis

    #########################
    #     Memory layout     #
    #########################

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Tuple of array dimensions."""
        return self._data.shape

    #####################
    #     Data type     #
    #####################

    @property
    def dtype(self) -> np.dtype:
        r"""Data-type of the array’s elements."""
        return self._data.dtype

    def _create_new_batch(self, data: np.ndarray) -> TBatchedArray:
        return self.__class__(data, **self._get_kwargs())

    def _get_kwargs(self) -> dict:
        return {"batch_axis": self._batch_axis}
