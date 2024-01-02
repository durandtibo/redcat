r"""Implement an extension of ``numpy.ndarray`` to represent a batch.

Notes:
    - https://numpy.org/doc/stable/user/basics.subclassing.html
"""

from __future__ import annotations

__all__ = ["BatchedArray"]

from typing import Any, Literal, TypeVar

import numpy as np
from coola import objects_are_allclose, objects_are_equal
from numpy import ndarray
from numpy.typing import ArrayLike

from redcat.ba.utils import check_data_and_axis, check_same_batch_axis, get_batch_axes

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")


class BatchedArray(ndarray):
    def __new__(cls, data: ArrayLike, batch_axis: int = 0) -> TBatchedArray:
        obj = np.asarray(data).view(cls)
        check_data_and_axis(obj, batch_axis)
        obj.batch_axis = batch_axis
        return obj

    def __array_finalize__(self, obj: BatchedArray | None) -> None:
        # if obj is None:
        #     return
        self.batch_axis = getattr(obj, "batch_axis", 0)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "inner"],
        *args: Any,
        **kwargs: Any,
    ) -> TBatchedArray:
        # if method != "__call__":
        #     raise NotImplementedError
        check_same_batch_axis(get_batch_axes(args, kwargs))
        args = [a.__array__() if isinstance(a, BatchedArray) else a for a in args]
        return self._create_new_batch(super().__array_ufunc__(ufunc, method, *args, **kwargs))

    def __repr__(self) -> str:
        return repr(self.__array__())[:-1] + f", batch_axis={self.batch_axis})"

    def __str__(self) -> str:
        return str(self.__array__()) + f"\nwith batch_axis={self.batch_axis}"

    def _create_new_batch(self, data: ndarray) -> TBatchedArray:
        return self.__class__(data, **self._get_kwargs())

    def _get_kwargs(self) -> dict:
        return {"batch_axis": self.batch_axis}

    def allclose(
        self, other: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
    ) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self.batch_axis != other.batch_axis:
            return False
        return objects_are_allclose(
            self.__array__(), other.__array__(), rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    def allequal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self.batch_axis != other.batch_axis:
            return False
        return objects_are_equal(self.__array__(), other.__array__())
