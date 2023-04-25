from __future__ import annotations

__all__ = ["BatchedTensor", "check_data_and_dim"]

import functools
from collections.abc import Callable, Iterable, Sequence
from typing import Any, overload

import torch
from torch import Tensor

from redcat.base import BaseBatch
from redcat.basetensor import BaseBatchedTensor
from redcat.utils import check_batch_dims, get_batch_dims, permute_along_dim

HANDLED_FUNCTIONS = {}


class BatchedTensor(BaseBatchedTensor):
    r"""Implements a batched tensor to easily manipulate a batch of examples.

    Args:
        data (array_like): Specifies the data for the tensor. It can
            be a torch.Tensor, list, tuple, NumPy ndarray, scalar,
            and other types.
        batch_dim (int, optional): Specifies the batch dimension
            in the ``torch.Tensor`` object. Default: ``0``
        kwargs: Keyword arguments that are passed to
            ``torch.as_tensor``.
    """

    def __init__(self, data: Any, *, batch_dim: int = 0, **kwargs) -> None:
        super().__init__(data, **kwargs)
        check_data_and_dim(self._data, batch_dim)
        self._batch_dim = int(batch_dim)

    def __repr__(self) -> str:
        return repr(self._data)[:-1] + f", batch_dim={self._batch_dim})"

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> BatchedTensor:
        # print(func, types, args, kwargs)
        kwargs = kwargs or {}
        if handled_func := HANDLED_FUNCTIONS.get(func, None):
            return handled_func(*args, **kwargs)

        batch_dims = get_batch_dims(args, kwargs)
        check_batch_dims(batch_dims)
        args = [a._data if hasattr(a, "_data") else a for a in args]
        return cls(func(*args, **kwargs), batch_dim=batch_dims.pop())

    @property
    def batch_dim(self) -> int:
        r"""int: The batch dimension in the ``torch.Tensor`` object."""
        return self._batch_dim

    @property
    def batch_size(self) -> int:
        return self._data.shape[self._batch_dim]

    ###############################
    #     Creation operations     #
    ###############################

    def new_full(
        self,
        fill_value: float | int | bool,
        batch_size: int | None = None,
        **kwargs,
    ) -> BatchedTensor:
        r"""Creates a batch filled with a scalar value.

        By default, the tensor in the returned batch has the same
        shape, ``torch.dtype`` and ``torch.device`` as the tensor in
        the current batch.

        Args:
            fill_value (float or int or bool): Specifies the number
                to fill the batch with.
            batch_size (int or ``None``): Specifies the batch size.
                If ``None``, the batch size of the current batch is
                used. Default: ``None``.
            **kwargs: See the documentation of
                ``torch.Tensor.new_full``.

        Returns:
            ``BaseBatchedTensor``: A batch filled with the scalar value.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.new_full(42)
            tensor([[42., 42., 42.],
                    [42., 42., 42.]], batch_dim=0)
            >>> batch.new_full(42, batch_size=5)
            tensor([[42., 42., 42.],
                    [42., 42., 42.],
                    [42., 42., 42.],
                    [42., 42., 42.],
                    [42., 42., 42.]], batch_dim=0)
        """
        shape = list(self._data.shape)
        if batch_size is not None:
            shape[self._batch_dim] = batch_size
        kwargs["dtype"] = kwargs.get("dtype", self.dtype)
        kwargs["device"] = kwargs.get("device", self.device)
        return BatchedTensor(
            torch.full(size=shape, fill_value=fill_value, **kwargs), **self._get_kwargs()
        )

    def new_ones(
        self,
        batch_size: int | None = None,
        **kwargs,
    ) -> BatchedTensor:
        r"""Creates a batch filled with the scalar value ``1``.

        By default, the tensor in the returned batch has the same
        shape, ``torch.dtype`` and ``torch.device`` as the tensor in
        the current batch.

        Args:
            batch_size (int or ``None``): Specifies the batch size.
                If ``None``, the batch size of the current batch is
                used. Default: ``None``.
            **kwargs: See the documentation of
                ``torch.Tensor.new_ones``.

        Returns:
            ``BaseBatchedTensor``: A batch filled with the scalar
                value ``1``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.zeros(2, 3))
            >>> batch.new_ones()
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
            >>> batch.new_ones(batch_size=5)
            tensor([[1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
        """
        shape = list(self._data.shape)
        if batch_size is not None:
            shape[self._batch_dim] = batch_size
        kwargs["dtype"] = kwargs.get("dtype", self.dtype)
        kwargs["device"] = kwargs.get("device", self.device)
        return BatchedTensor(torch.ones(*shape, **kwargs), **self._get_kwargs())

    def new_zeros(
        self,
        batch_size: int | None = None,
        **kwargs,
    ) -> BatchedTensor:
        r"""Creates a batch filled with the scalar value ``0``.

        By default, the tensor in the returned batch has the same
        shape, ``torch.dtype`` and ``torch.device`` as the tensor
        in the current batch.

        Args:
            batch_size (int or ``None``): Specifies the batch size.
                If ``None``, the batch size of the current batch is
                used. Default: ``None``.
            **kwargs: See the documentation of
                ``torch.Tensor.new_zeros``.

        Returns:
            ``BaseBatchedTensor``: A batch filled with the scalar
                value ``0``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch.new_zeros()
            tensor([[0., 0., 0.],
                    [0., 0., 0.]], batch_dim=0)
            >>> batch.new_zeros(batch_size=5)
            tensor([[0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]], batch_dim=0)
        """
        shape = list(self._data.shape)
        if batch_size is not None:
            shape[self._batch_dim] = batch_size
        kwargs["dtype"] = kwargs.get("dtype", self.dtype)
        kwargs["device"] = kwargs.get("device", self.device)
        return BatchedTensor(torch.zeros(*shape, **kwargs), **self._get_kwargs())

    #################################
    #     Comparison operations     #
    #################################

    def allclose(
        self, other: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
    ) -> bool:
        if not isinstance(other, BatchedTensor):
            return False
        if self._batch_dim != other.batch_dim:
            return False
        if self._data.shape != other.data.shape:
            return False
        return self._data.allclose(other.data, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def equal(self, other: Any) -> bool:
        if not isinstance(other, BatchedTensor):
            return False
        if self._batch_dim != other.batch_dim:
            return False
        return self._data.equal(other.data)

    ##################################################
    #     Mathematical | arithmetical operations     #
    ##################################################

    def add_(
        self,
        other: BaseBatchedTensor | Tensor | int | float,
        alpha: int | float = 1.0,
    ) -> None:
        check_batch_dims(get_batch_dims((self, other)))
        self._data.add_(other, alpha=alpha)

    def div_(
        self,
        other: BaseBatchedTensor | torch.Tensor | int | float,
        rounding_mode: str | None = None,
    ) -> None:
        check_batch_dims(get_batch_dims((self, other)))
        self._data.div_(other, rounding_mode=rounding_mode)

    def fmod_(self, divisor: BaseBatchedTensor | torch.Tensor | int | float) -> None:
        check_batch_dims(get_batch_dims((self, divisor)))
        self._data.fmod_(divisor)

    def mul_(self, other: BaseBatchedTensor | torch.Tensor | int | float) -> None:
        check_batch_dims(get_batch_dims((self, other)))
        self._data.mul_(other)

    def sub_(
        self,
        other: BaseBatchedTensor | Tensor | int | float,
        alpha: int | float = 1.0,
    ) -> None:
        check_batch_dims(get_batch_dims((self, other)))
        self._data.sub_(other, alpha=alpha)

    ###########################################################
    #     Mathematical | advanced arithmetical operations     #
    ###########################################################

    def cumsum_along_batch(self, **kwargs) -> BatchedTensor:
        return torch.cumsum(self, dim=self._batch_dim, **kwargs)

    def cumsum_along_batch_(self) -> None:
        self._data.cumsum_(dim=self._batch_dim)

    def permute_along_batch(self, permutation: Sequence[int] | torch.Tensor) -> BatchedTensor:
        if not torch.is_tensor(permutation):
            permutation = torch.tensor(permutation)
        return self.__class__(
            permute_along_dim(tensor=self._data, permutation=permutation, dim=self._batch_dim),
            **self._get_kwargs(),
        )

    def permute_along_batch_(self, permutation: Sequence[int] | torch.Tensor) -> None:
        if not torch.is_tensor(permutation):
            permutation = torch.tensor(permutation)
        self._data = permute_along_dim(
            tensor=self._data, permutation=permutation, dim=self._batch_dim
        )

    ################################################
    #     Mathematical | point-wise operations     #
    ################################################

    def pow_(self, exponent: int | float | BaseBatchedTensor) -> None:
        check_batch_dims(get_batch_dims((self, exponent)))
        self._data.pow_(exponent)

    #############################################
    #     Mathematical | logical operations     #
    #############################################

    def logical_and_(self, other: BaseBatchedTensor | Tensor) -> None:
        check_batch_dims(get_batch_dims((self, other)))
        self._data.logical_and_(other)

    def logical_or_(self, other: BaseBatchedTensor | Tensor) -> None:
        check_batch_dims(get_batch_dims((self, other)))
        self._data.logical_or_(other)

    def logical_xor_(self, other: BaseBatchedTensor | Tensor) -> None:
        check_batch_dims(get_batch_dims((self, other)))
        self._data.logical_xor_(other)

    ##########################################################
    #    Indexing, slicing, joining, mutating operations     #
    ##########################################################

    def cat_along_batch(
        self, tensors: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor]
    ) -> BatchedTensor:
        return self.cat(tensors, dim=self._batch_dim)

    def cat_along_batch_(
        self, tensors: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor]
    ) -> None:
        self.cat_(tensors, dim=self._batch_dim)

    def index_select_along_batch(self, index: torch.Tensor | Sequence[int]) -> BatchedTensor:
        if not torch.is_tensor(index):
            index = torch.tensor(index)
        return self.__class__(self._data.index_select(self._batch_dim, index), **self._get_kwargs())

    def masked_fill(
        self, mask: BaseBatchedTensor | Tensor, value: bool | int | float
    ) -> BatchedTensor:
        check_batch_dims(get_batch_dims((self, mask)))
        if isinstance(mask, BaseBatchedTensor):
            mask = mask.data
        return self.__class__(self._data.masked_fill(mask.data, value), **self._get_kwargs())

    def select_along_batch(self, index: int) -> Tensor:
        return self._data.select(self._batch_dim, index)

    def slice_along_batch(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> BatchedTensor:
        if self._batch_dim == 0:
            data = self._data[start:stop:step]
        elif self._batch_dim == 1:
            data = self._data[:, start:stop:step]
        else:
            data = self._data.transpose(0, self._batch_dim)[start:stop:step].transpose(
                0, self._batch_dim
            )
        return self.__class__(data, **self._get_kwargs())

    def split_along_batch(self, split_size: int, deepcopy: bool = False) -> Iterable[BatchedTensor]:
        data = self._data
        if deepcopy:
            data = data.clone()
        for chunk in data.split(split_size, dim=self._batch_dim):
            yield self.__class__(chunk, **self._get_kwargs())

    def take_along_batch(
        self, indices: BaseBatch[Tensor | Sequence] | Tensor | Sequence
    ) -> BatchedTensor:
        return self.take_along_dim(indices, dim=self._batch_dim)

    def take_along_dim(
        self, indices: BaseBatch[Tensor | Sequence] | Tensor | Sequence, dim: int | None = None
    ) -> BatchedTensor:
        if isinstance(indices, Sequence):
            indices = torch.as_tensor(indices)
        return torch.take_along_dim(self, indices, dim=dim)

    def unsqueeze(self, dim: int) -> BatchedTensor:
        return self.__class__(
            self._data.unsqueeze(dim=dim),
            batch_dim=self._batch_dim + 1
            if self._batch_dim >= dim and dim >= 0
            else self._batch_dim,
        )

    def view(self, *shape: tuple[int, ...]) -> Tensor:
        return self._data.view(*shape)

    def view_as(self, other: BaseBatchedTensor | Tensor) -> BatchedTensor:
        check_batch_dims(get_batch_dims((self, other)))
        return self.__class__(self._data.view_as(other.data), **self._get_kwargs())

    def _get_kwargs(self) -> dict:
        return {"batch_dim": self._batch_dim}


def check_data_and_dim(data: Tensor, batch_dim: int) -> None:
    r"""Checks if the tensor ``data`` and ``batch_dim`` are correct.

    Args:
        data (``torch.Tensor``): Specifies the tensor in the batch.
        batch_dim (int): Specifies the batch dimension in the
            ``torch.Tensor`` object.

    Raises:
        RuntimeError: if one of the input is incorrect.
    """
    if data.dim() < 1:
        raise RuntimeError(f"data needs at least 1 dimensions (received: {data.dim()})")
    if batch_dim < 0 or batch_dim >= data.dim():
        raise RuntimeError(
            f"Incorrect batch_dim ({batch_dim}) but the value should be in [0, {data.dim() - 1}]"
        )


def implements(torch_function: Callable) -> Callable:
    """Register a torch function override for BatchedTensor."""

    def decorator(func: Callable) -> Callable:
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements(torch.cat)
def cat(
    tensors: tuple[BaseBatchedTensor | Tensor, ...] | list[BaseBatchedTensor | Tensor],
    dim: int = 0,
) -> BatchedTensor:
    r"""See ``torch.cat`` documentation."""
    batch_dims = get_batch_dims(tensors)
    check_batch_dims(batch_dims)
    return BatchedTensor(
        torch.cat(
            [tensor._data if hasattr(tensor, "_data") else tensor for tensor in tensors], dim=dim
        ),
        batch_dim=batch_dims.pop(),
    )


@overload
def take_along_dim(
    input: BaseBatchedTensor | Tensor,  # noqa: A002
    indices: BaseBatchedTensor | Tensor,
) -> Tensor:
    r"""See ``torch.take_along_dim`` documentation."""


@overload
def take_along_dim(
    input: BaseBatchedTensor | Tensor, indices: BaseBatchedTensor | Tensor, dim: int  # noqa: A002
) -> BatchedTensor:
    r"""See ``torch.take_along_dim`` documentation."""


@implements(torch.take_along_dim)
def take_along_dim(
    input: BaseBatchedTensor | Tensor,  # noqa: A002
    indices: BaseBatchedTensor | Tensor,
    dim: int | None = None,
) -> BatchedTensor | Tensor:
    r"""See ``torch.take_along_dim`` documentation."""
    batch_dims = get_batch_dims((input, indices))
    check_batch_dims(batch_dims)
    if isinstance(input, BaseBatchedTensor):
        input = input.data  # noqa: A001
    if isinstance(indices, BaseBatchedTensor):
        indices = indices.data
    if dim is None:
        return torch.take_along_dim(input, indices)
    return BatchedTensor(
        torch.take_along_dim(input, indices, dim=dim),
        batch_dim=batch_dims.pop(),
    )
