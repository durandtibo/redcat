from __future__ import annotations

__all__ = ["BatchedArray"]

from collections.abc import Callable, Iterable, Sequence
from itertools import chain
from typing import Any, TypeVar

import numpy as np
from coola import objects_are_allclose, objects_are_equal
from numpy import ndarray

from redcat.utils.array import check_batch_dims, check_data_and_dim, get_batch_dims

# Workaround because Self is not available for python 3.9 and 3.10
# https://peps.python.org/pep-0673/
TBatchedArray = TypeVar("TBatchedArray", bound="BatchedArray")

HANDLED_FUNCTIONS = {}


class BatchedArray:  # (BaseBatch[ndarray]):
    r"""Implements a batched array to easily manipulate a batch of
    examples.

    Args:
    ----
        data (array_like): Specifies the data for the array. It can
            be a list, tuple, NumPy ndarray, scalar, and other types.
        batch_dim (int, optional): Specifies the batch dimension
            in the ``torch.Tensor`` object. Default: ``0``
        kwargs: Keyword arguments that are passed to
            ``torch.as_array``.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from redcat import BatchedArray
        >>> batch = BatchedArray(np.arange(10).reshape(2, 5))
    """

    def __init__(self, data: Any, *, batch_dim: int = 0, **kwargs) -> None:
        super().__init__()
        self._data = np.array(data, **kwargs)
        check_data_and_dim(self._data, batch_dim)
        self._batch_dim = int(batch_dim)

    def __repr__(self) -> str:
        return repr(self._data)[:-1] + f", batch_dim={self._batch_dim})"

    def __array_function__(
        self,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> TBatchedArray:
        # if func not in HANDLED_FUNCTIONS:
        #     return NotImplemented
        #     # Note: this allows subclasses that don't override
        #     # __array_function__ to handle BatchedArray objects
        # if not all(issubclass(t, BatchedArray) for t in types):
        #     return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    @property
    def batch_dim(self) -> int:
        r"""int: The batch dimension in the ``torch.Tensor`` object."""
        return self._batch_dim

    @property
    def batch_size(self) -> int:
        return self._data.shape[self._batch_dim]

    @property
    def data(self) -> ndarray:
        return self._data

    @property
    def dtype(self) -> np.dtype:
        r"""``numpy.dtype``: The data type."""
        return self._data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        r"""``tuple``: The shape of the array."""
        return self._data.shape

    def dim(self) -> int:
        r"""Gets the number of dimensions.

        Returns:
        -------
            int: The number of dimensions

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.dim()
            2
        """
        return self._data.ndim

    def ndimension(self) -> int:
        r"""Gets the number of dimensions.

        Returns:
        -------
            int: The number of dimensions

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.ndimension()
            2
        """
        return self.dim()

    def numel(self) -> int:
        r"""Gets the total number of elements in the array.

        Returns:
        -------
            int: The total number of elements

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch.numel()
            6
        """
        return np.prod(self._data.shape).item()

    ###############################
    #     Creation operations     #
    ###############################

    def clone(self, *args, **kwargs) -> TBatchedArray:
        r"""Creates a copy of the current batch.

        Args:
        ----
            *args: See the documentation of ``numpy.copy``
            **kwargs: See the documentation of ``numpy.copy``

        Returns:
        -------
            ``BatchedArray``: A copy of the current batch.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch_copy = batch.clone()
            >>> batch_copy
            array([[1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
        """
        return self._create_new_batch(self._data.copy(*args, **kwargs))

    def copy(self, *args, **kwargs) -> TBatchedArray:
        r"""Creates a copy of the current batch.

        Args:
        ----
            *args: See the documentation of ``numpy.copy``
            **kwargs: See the documentation of ``numpy.copy``

        Returns:
        -------
            ``BatchedArray``: A copy of the current batch.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.ones((2, 3)))
            >>> batch_copy = batch.copy()
            >>> batch_copy
            array([[1., 1., 1.],
                   [1., 1., 1.]], batch_dim=0)
        """
        return self.clone(*args, **kwargs)

    #################################
    #     Comparison operations     #
    #################################

    def allclose(
        self, other: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
    ) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self._batch_dim != other.batch_dim:
            return False
        if self._data.shape != other.data.shape:
            return False
        return objects_are_allclose(
            self._data, other.data, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self._batch_dim != other.batch_dim:
            return False
        return objects_are_equal(self._data, other.data)

    # def permute_along_batch(self, permutation: IndicesType) -> TBatchedArray:
    #     return self.permute_along_dim(permutation, dim=self._batch_dim)
    #
    # def permute_along_batch_(self, permutation: IndicesType) -> None:
    #     self.permute_along_dim_(permutation, dim=self._batch_dim)

    ##########################################################
    #    Indexing, slicing, joining, mutating operations     #
    ##########################################################

    # def append(self, other: BaseBatch) -> None:
    #     pass

    def cat(
        self,
        arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
        dim: int = 0,
    ) -> TBatchedArray:
        r"""Concatenates the data of the batch(es) to the current batch
        along a given dimension and creates a new batch.

        Args:
        ----
            arrays (``BatchedArray`` or ``numpy.ndarray`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Returns:
        -------
            ``BatchedArray``: A batch with the concatenated data
                in the given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
            >>> batch.cat(BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])))
            array([[ 0,  1,  2],
                   [ 4,  5,  6],
                   [10, 11, 12],
                   [13, 14, 15]], batch_dim=0)
        """
        return self.concatenate(arrays, dim)

    def concatenate(
        self,
        arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
        axis: int = 0,
    ) -> TBatchedArray:
        r"""Concatenates the data of the batch(es) to the current batch
        along a given dimension and creates a new batch.

        Args:
        ----
            arrays (``BatchedArray`` or ``numpy.ndarray`` or
                ``Iterable``): Specifies the batch(es) to concatenate.
            axis (int, optional): Specifies the axis along which the
                arrays will be concatenated. Default: ``0``

        Returns:
        -------
            ``BatchedArray``: A batch with the concatenated data
                in the given dimension.

        Example usage:

        .. code-block:: pycon

            >>> import numpy as np
            >>> from redcat import BatchedArray
            >>> batch = BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
            >>> batch.concatenate(BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])))
            array([[ 0,  1,  2],
                   [ 4,  5,  6],
                   [10, 11, 12],
                   [13, 14, 15]], batch_dim=0)
        """
        if isinstance(arrays, (BatchedArray, ndarray)):
            arrays = [arrays]
        return np.concatenate(list(chain([self], arrays)), axis=axis)

    # def chunk_along_batch(self, chunks: int) -> tuple[TBatchedArray, ...]:
    #     pass
    #
    # def extend(self, other: Iterable[BaseBatch]) -> None:
    #     pass
    #
    # def index_select_along_batch(self, index: Tensor | Sequence[int]) -> BaseBatch:
    #     pass
    #
    # def slice_along_batch(
    #     self, start: int = 0, stop: int | None = None, step: int = 1
    # ) -> TBatchedArray:
    #     pass
    #
    # def split(
    #     self, split_size_or_sections: int | Sequence[int], dim: int = 0
    # ) -> tuple[TBatchedArray, ...]:
    #     r"""Splits the batch into chunks along a given dimension.
    #
    #     Args:
    #     ----
    #         split_size_or_sections (int or sequence): Specifies the
    #             size of a single chunk or list of sizes for each chunk.
    #         dim (int, optional): Specifies the dimension along which
    #             to split the array. Default: ``0``
    #
    #     Returns:
    #     -------
    #         tuple: The batch split into chunks along the given
    #             dimension.
    #
    #     Example usage:
    #
    #     .. code-block:: pycon
    #
    #         >>> import torch
    #         >>> from redcat import BatchedArray
    #         >>> batch = BatchedArray(torch.arange(10).view(5, 2))
    #         >>> batch.split(2, dim=0)
    #         (array([[0, 1], [2, 3]], batch_dim=0),
    #          array([[4, 5], [6, 7]], batch_dim=0),
    #          array([[8, 9]], batch_dim=0))
    #     """
    #     if isinstance(split_size_or_sections, int):
    #         split_size_or_sections = np.arange(
    #             split_size_or_sections, self._data.shape[dim], split_size_or_sections
    #         )
    #     return np.split(self, split_size_or_sections)
    #
    # def split_along_batch(
    #     self, split_size_or_sections: int | Sequence[int]
    # ) -> tuple[TBatchedArray, ...]:
    #     return self.split(split_size_or_sections, dim=self._batch_dim)

    #################
    #     Other     #
    #################

    def summary(self) -> str:
        dims = ", ".join([f"{key}={value}" for key, value in self._get_kwargs().items()])
        return f"{self.__class__.__qualname__}(dtype={self.dtype}, shape={self.shape}, {dims})"

    def _create_new_batch(self, data: ndarray) -> TBatchedArray:
        return self.__class__(data, **self._get_kwargs())

    def _get_kwargs(self) -> dict:
        return {"batch_dim": self._batch_dim}

    # TODO: remove later. Temporary hack because BatchedArray is not a BaseBatch yet
    def __eq__(self, other: Any) -> bool:
        return self.equal(other)


def implements(np_function: Callable) -> Callable:
    r"""Register an `__array_function__` implementation for
    ``BatchedArray`` objects.

    Args:
    ----
        np_function (``Callable``):  Specifies the numpy function
            to override.

    Returns:
    -------
        ``Callable``: The decorated function.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from redcat.array import BatchedArray, implements
        >>> @implements(np.sum)
        ... def mysum(input: BatchedArray, *args, **kwargs) -> np.ndarray:
        ...     return np.sum(input.data, *args, **kwargs)
        ...
        >>> np.sum(BatchedArray(np.ones((2, 3))))
        6.0
    """

    def decorator(func: Callable) -> Callable:
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


@implements(np.concatenate)
def concatenate(arrays: Sequence[BatchedArray | ndarray], axis: int = 0) -> BatchedArray:
    r"""See ``numpy.concatenate`` documentation."""
    batch_dims = get_batch_dims(arrays)
    check_batch_dims(batch_dims)
    return BatchedArray(
        np.concatenate(
            [array._data if hasattr(array, "_data") else array for array in arrays], axis=axis
        ),
        batch_dim=batch_dims.pop(),
    )


@implements(np.sum)
def numpysum(input: BatchedArray, *args, **kwargs) -> ndarray:  # noqa: A002
    r"""See ``np.sum`` documentation.

    Use the name ``numpysum`` to avoid shadowing `sum` python builtin.
    """
    return np.sum(input.data, *args, **kwargs)
