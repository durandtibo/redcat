from __future__ import annotations

__all__ = ["BatchedTensorSeq", "check_data_and_dims"]

from collections.abc import Callable, Iterable, Sequence
from itertools import chain
from typing import Any

import torch
from torch import Tensor

from redcat.basetensor import BaseBatchedTensor
from redcat.tensor import BatchedTensor
from redcat.utils import (
    align_to_batch_seq,
    align_to_seq_batch,
    check_batch_dims,
    check_seq_dims,
    get_batch_dims,
    get_seq_dims,
    permute_along_dim,
)


class BatchedTensorSeq(BaseBatchedTensor):
    r"""Implements a batched tensor to easily manipulate a batch of sequences.

    Args:
        data (array_like): Specifies the data for the tensor. It can
            be a torch.Tensor, list, tuple, NumPy ndarray, scalar,
            and other types.
        batch_dim (int, optional): Specifies the batch dimension
            in the ``torch.Tensor`` object. Default: ``0``
        seq_dim (int, optional): Specifies the sequence dimension in
            the ``torch.Tensor`` object. Default: ``1``
        kwargs: Keyword arguments that are passed to
            ``torch.as_tensor``.
    """

    def __init__(self, data: Any, *, batch_dim: int = 0, seq_dim: int = 1, **kwargs) -> None:
        super().__init__(data, **kwargs)
        check_data_and_dims(self._data, batch_dim, seq_dim)
        self._batch_dim = int(batch_dim)
        self._seq_dim = int(seq_dim)

    def __repr__(self) -> str:
        return repr(self._data)[:-1] + f", batch_dim={self._batch_dim}, seq_dim={self._seq_dim})"

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> BatchedTensorSeq:
        kwargs = kwargs or {}
        batch_dims = get_batch_dims(args, kwargs)
        check_batch_dims(batch_dims)
        seq_dims = get_seq_dims(args, kwargs)
        check_seq_dims(seq_dims)
        args = [a._data if hasattr(a, "_data") else a for a in args]
        return cls(func(*args, **kwargs), batch_dim=batch_dims.pop(), seq_dim=seq_dims.pop())

    @property
    def batch_dim(self) -> int:
        r"""int: The batch dimension in the ``torch.Tensor`` object."""
        return self._batch_dim

    @property
    def batch_size(self) -> int:
        return self._data.shape[self._batch_dim]

    @property
    def seq_dim(self) -> int:
        r"""int: The sequence dimension in the ``torch.Tensor`` object."""
        return self._seq_dim

    @property
    def seq_len(self) -> int:
        r"""int: The sequence length."""
        return self._data.shape[self._seq_dim]

    ###############################
    #     Creation operations     #
    ###############################

    def new_full(
        self,
        fill_value: float | int | bool,
        batch_size: int | None = None,
        seq_len: int | None = None,
        **kwargs,
    ) -> BatchedTensorSeq:
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
            seq_len (int or ``None``): Specifies the sequence length.
                If ``None``, the sequence length of the current batch
                is used. Default: ``None``.
            **kwargs: See the documentation of
                ``torch.Tensor.new_full``.

        Returns:
            ``BaseBatchedTensor``: A batch filled with the scalar value.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.ones(2, 3))
            >>> batch.new_full(42)
            tensor([[42., 42., 42.],
                    [42., 42., 42.]], batch_dim=0, seq_dim=1)
            >>> batch.new_full(42, batch_size=5)
            tensor([[42., 42., 42.],
                    [42., 42., 42.],
                    [42., 42., 42.],
                    [42., 42., 42.],
                    [42., 42., 42.]], batch_dim=0, seq_dim=1)
            >>> batch.new_full(42, seq_len=5)
            tensor([[42., 42., 42., 42., 42.],
                    [42., 42., 42., 42., 42.]], batch_dim=0, seq_dim=1)
        """
        shape = list(self._data.shape)
        if batch_size is not None:
            shape[self._batch_dim] = batch_size
        if seq_len is not None:
            shape[self._seq_dim] = seq_len
        kwargs["dtype"] = kwargs.get("dtype", self.dtype)
        kwargs["device"] = kwargs.get("device", self.device)
        return BatchedTensorSeq(
            torch.full(size=shape, fill_value=fill_value, **kwargs), **self._get_kwargs()
        )

    def new_ones(
        self,
        batch_size: int | None = None,
        seq_len: int | None = None,
        **kwargs,
    ) -> BatchedTensorSeq:
        r"""Creates a batch filled with the scalar value ``1``.

        By default, the tensor in the returned batch has the same
        shape, ``torch.dtype`` and ``torch.device`` as the tensor in
        the current batch.

        Args:
            batch_size (int or ``None``): Specifies the batch size.
                If ``None``, the batch size of the current batch is
                used. Default: ``None``.
            seq_len (int or ``None``): Specifies the sequence length.
                If ``None``, the sequence length of the current batch
                is used. Default: ``None``.
            **kwargs: See the documentation of
                ``torch.Tensor.new_ones``.

        Returns:
            ``BaseBatchedTensor``: A batch filled with the scalar
                value ``1``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.zeros(2, 3))
            >>> batch.new_ones()
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0, seq_dim=1)
            >>> batch.new_ones(batch_size=5)
            tensor([[1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0, seq_dim=1)
            >>> batch.new_ones(seq_len=5)
            tensor([[1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.]], batch_dim=0, seq_dim=1)
        """
        shape = list(self._data.shape)
        if batch_size is not None:
            shape[self._batch_dim] = batch_size
        if seq_len is not None:
            shape[self._seq_dim] = seq_len
        kwargs["dtype"] = kwargs.get("dtype", self.dtype)
        kwargs["device"] = kwargs.get("device", self.device)
        return BatchedTensorSeq(torch.ones(*shape, **kwargs), **self._get_kwargs())

    def new_zeros(
        self,
        batch_size: int | None = None,
        seq_len: int | None = None,
        **kwargs,
    ) -> BatchedTensorSeq:
        r"""Creates a batch filled with the scalar value ``0``.

        By default, the tensor in the returned batch has the same
        shape, ``torch.dtype`` and ``torch.device`` as the tensor
        in the current batch.

        Args:
            batch_size (int or ``None``): Specifies the batch size.
                If ``None``, the batch size of the current batch is
                used. Default: ``None``.
            seq_len (int or ``None``): Specifies the sequence length.
                If ``None``, the sequence length of the current batch
                is used. Default: ``None``.
            **kwargs: See the documentation of
                ``torch.Tensor.new_zeros``.

        Returns:
            ``BaseBatchedTensor``: A batch filled with the scalar
                value ``0``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.ones(2, 3))
            >>> batch.new_zeros()
            tensor([[0., 0., 0.],
                    [0., 0., 0.]], batch_dim=0, seq_dim=1)
            >>> batch.new_zeros(batch_size=5)
            tensor([[0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]], batch_dim=0, seq_dim=1)
            >>> batch.new_zeros(seq_len=5)
            tensor([[0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]], batch_dim=0, seq_dim=1)
        """
        shape = list(self._data.shape)
        if batch_size is not None:
            shape[self._batch_dim] = batch_size
        if seq_len is not None:
            shape[self._seq_dim] = seq_len
        kwargs["dtype"] = kwargs.get("dtype", self.dtype)
        kwargs["device"] = kwargs.get("device", self.device)
        return BatchedTensorSeq(torch.zeros(*shape, **kwargs), **self._get_kwargs())

    @classmethod
    def from_seq_batch(cls, data: Any, **kwargs) -> BatchedTensorSeq:
        r"""Creates a batch where the first dimension is the sequence dimension
        and the second dimension is the batch dimension.

        Args:
            data (array_like): Specifies the data for the tensor. It can
                be a torch.Tensor, list, tuple, NumPy ndarray, scalar,
                and other types.
            kwargs: Keyword arguments that are passed to
                ``torch.as_tensor``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq.from_seq_batch(torch.ones(2, 3))
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=1, seq_dim=0)
        """
        return cls(data, batch_dim=1, seq_dim=0, **kwargs)

    #################################
    #     Comparison operations     #
    #################################

    def allclose(
        self, other: Any, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False
    ) -> bool:
        if not isinstance(other, BatchedTensorSeq):
            return False
        if self._batch_dim != other.batch_dim or self._seq_dim != other.seq_dim:
            return False
        if self._data.shape != other.data.shape:
            return False
        return self._data.allclose(other.data, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def equal(self, other: Any) -> bool:
        if not isinstance(other, BatchedTensorSeq):
            return False
        if self._batch_dim != other.batch_dim or self._seq_dim != other.seq_dim:
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
        check_seq_dims(get_seq_dims((self, other)))
        self._data.add_(other, alpha=alpha)

    def div_(
        self,
        other: BaseBatchedTensor | torch.Tensor | int | float,
        rounding_mode: str | None = None,
    ) -> None:
        check_batch_dims(get_batch_dims((self, other)))
        check_seq_dims(get_seq_dims((self, other)))
        self._data.div_(other, rounding_mode=rounding_mode)

    def fmod_(self, divisor: BaseBatchedTensor | torch.Tensor | int | float) -> None:
        check_batch_dims(get_batch_dims((self, divisor)))
        check_seq_dims(get_seq_dims((self, divisor)))
        self._data.fmod_(divisor)

    def mul_(self, other: BaseBatchedTensor | torch.Tensor | int | float) -> None:
        check_batch_dims(get_batch_dims((self, other)))
        check_seq_dims(get_seq_dims((self, other)))
        self._data.mul_(other)

    def sub_(
        self,
        other: BaseBatchedTensor | Tensor | int | float,
        alpha: int | float = 1.0,
    ) -> None:
        check_batch_dims(get_batch_dims((self, other)))
        check_seq_dims(get_seq_dims((self, other)))
        self._data.sub_(other, alpha=alpha)

    ###########################################################
    #     Mathematical | advanced arithmetical operations     #
    ###########################################################

    def cumsum_along_batch(self, **kwargs) -> BatchedTensorSeq:
        return torch.cumsum(self, dim=self._batch_dim, **kwargs)

    def cumsum_along_batch_(self) -> None:
        self._data.cumsum_(dim=self._batch_dim)

    def cumsum_along_seq(self, **kwargs) -> BatchedTensorSeq:
        r"""Computes the cumulative sum of elements of the current batch in the
        sequence dimension.

        Args:
            **kwargs: see ``torch.cumsum`` documentation

        Returns:
            ``BatchedTensorSeq``: A batch with the cumulative sum of
                elements of the current batch in the sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5)).cumsum_along_seq()
            >>> batch
            tensor([[ 0,  1,  3,  6, 10],
                    [ 5, 11, 18, 26, 35]], batch_dim=0, seq_dim=1)
        """
        return torch.cumsum(self, dim=self._seq_dim, **kwargs)

    def cumsum_along_seq_(self) -> None:
        r"""Computes the cumulative sum of elements of the current batch in the
        sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.cumsum_along_seq_()
            >>> batch
            tensor([[ 0,  1,  3,  6, 10],
                    [ 5, 11, 18, 26, 35]], batch_dim=0, seq_dim=1)
        """
        self._data.cumsum_(dim=self._seq_dim)

    def permute_along_batch(self, permutation: Sequence[int] | torch.Tensor) -> BatchedTensorSeq:
        if not torch.is_tensor(permutation):
            permutation = torch.tensor(permutation)
        return self.__class__(
            data=permute_along_dim(tensor=self._data, permutation=permutation, dim=self._batch_dim),
            **self._get_kwargs(),
        )

    def permute_along_seq(self, permutation: Sequence[int] | torch.Tensor) -> BatchedTensorSeq:
        r"""Permutes the data along the sequence dimension.

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
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.permute_along_seq([2, 1, 3, 0, 4])
        """
        if not torch.is_tensor(permutation):
            permutation = torch.tensor(permutation)
        return self.__class__(
            data=permute_along_dim(tensor=self._data, permutation=permutation, dim=self._seq_dim),
            **self._get_kwargs(),
        )

    def sort_along_seq(self, descending: bool = False) -> tuple[BatchedTensorSeq, BatchedTensorSeq]:
        r"""Sorts the elements of the batch along the sequence dimension in
        monotonic order by value.

        Args:
            descending (bool, optional): Controls the sorting order.
                If ``True``, the elements are sorted in descending
                order by value. Default: ``False``

        Returns:
            (``BatchedTensorSeq``, ``BatchedTensorSeq``): A tuple with
                two values:
                    - The first batch contains the batch values sorted
                        along the sequence dimension.
                    - The second batch contains the indices that sort
                        the batch along the sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.rand(2, 5))
            >>> batch.sort_along_seq()
            (tensor([[0.2274, 0.4843, 0.4932, 0.8583, 0.9154],
                        [0.0101, 0.0733, 0.5018, 0.6007, 0.6589]], batch_dim=0, seq_dim=1),
             tensor([[2, 3, 4, 1, 0], [4, 3, 1, 0, 2]], batch_dim=0, seq_dim=1))
        """
        values, indices = self._data.sort(dim=self._seq_dim, descending=descending)
        return (
            BatchedTensorSeq(data=values, **self._get_kwargs()),
            BatchedTensorSeq(data=indices, **self._get_kwargs()),
        )

    ################################################
    #     Mathematical | point-wise operations     #
    ################################################

    def pow_(self, exponent: int | float | BaseBatchedTensor) -> None:
        check_batch_dims(get_batch_dims((self, exponent)))
        check_seq_dims(get_seq_dims((self, exponent)))
        self._data.pow_(exponent)

    #############################################
    #     Mathematical | logical operations     #
    #############################################

    def logical_and_(self, other: BaseBatchedTensor | Tensor) -> None:
        check_batch_dims(get_batch_dims((self, other)))
        check_seq_dims(get_seq_dims((self, other)))
        self._data.logical_and_(other)

    def logical_or_(self, other: BaseBatchedTensor | Tensor) -> None:
        check_batch_dims(get_batch_dims((self, other)))
        check_seq_dims(get_seq_dims((self, other)))
        self._data.logical_or_(other)

    def logical_xor_(self, other: BaseBatchedTensor | Tensor) -> None:
        check_batch_dims(get_batch_dims((self, other)))
        check_seq_dims(get_seq_dims((self, other)))
        self._data.logical_xor_(other)

    ################################
    #     Reduction operations     #
    ################################

    def max_along_seq(
        self, keepdim: bool = False
    ) -> tuple[BatchedTensor | BatchedTensorSeq, BatchedTensor | BatchedTensorSeq]:
        r"""Computes the maximum values along the sequence dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not. If ``False``
                the returned type is ``BatchedTensor``, otherwise it
                is ``BatchedTensorSeq``. Default: ``False``

        Returns:
            ``BatchedTensor``: A batch with the maximum values along the
                sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).max_along_seq()
            (tensor([4, 9], batch_dim=0), tensor([4, 4], batch_dim=0))
            >>> BatchedTensorSeq(torch.arange(30).view(2, 5, 3)).max_along_seq(keepdim=True)
            (tensor([[[12, 13, 14]], [[27, 28, 29]]], batch_dim=0, seq_dim=1),
             tensor([[[4, 4, 4]], [[4, 4, 4]]], batch_dim=0, seq_dim=1))
        """
        values, indices = torch.max(self._data, dim=self._seq_dim, keepdim=keepdim)
        if keepdim:
            return (
                BatchedTensorSeq(data=values, **self._get_kwargs()),
                BatchedTensorSeq(data=indices, **self._get_kwargs()),
            )
        batch_dim = self._batch_dim if self._seq_dim > self._batch_dim else self._batch_dim - 1
        return (
            BatchedTensor(data=values, batch_dim=batch_dim),
            BatchedTensor(data=indices, batch_dim=batch_dim),
        )

    def mean_along_seq(self, keepdim: bool = False) -> BatchedTensor | BatchedTensorSeq:
        r"""Computes the mean values along the sequence dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not. If ``False``
                the returned type is ``BatchedTensor``, otherwise it
                is ``BatchedTensorSeq``. Default: ``False``

        Returns:
            ``BatchedTensor`` or ``BatchedTensorSeq``: A batch with
                the mean values along the sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).mean_along_seq()
            (tensor([2, 7], batch_dim=0), tensor([2, 2], batch_dim=0))
            >>> BatchedTensorSeq(torch.arange(30).view(2, 5, 3)).mean_along_seq(keepdim=True)
            (tensor([[[12, 13, 14]], [[27, 28, 29]]], batch_dim=0, seq_dim=1),
             tensor([[[4, 4, 4]], [[4, 4, 4]]], batch_dim=0, seq_dim=1))
        """
        values = torch.mean(
            self._data if self._data.is_floating_point() else self._data.float(),
            dim=self._seq_dim,
            keepdim=keepdim,
        )
        if keepdim:
            return BatchedTensorSeq(data=values, **self._get_kwargs())
        return BatchedTensor(
            data=values,
            batch_dim=self._batch_dim if self._seq_dim > self._batch_dim else self._batch_dim - 1,
        )

    def median_along_seq(
        self, keepdim: bool = False
    ) -> tuple[BatchedTensor | BatchedTensorSeq, BatchedTensor | BatchedTensorSeq]:
        r"""Computes the median values along the sequence dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not. If ``False``
                the returned type is ``BatchedTensor``, otherwise it
                is ``BatchedTensorSeq``. Default: ``False``

        Returns:
            ``BatchedTensor``: A batch with the median values along the
                sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).median_along_seq()
            (tensor([2, 7], batch_dim=0), tensor([2, 2], batch_dim=0))
            >>> BatchedTensorSeq(torch.arange(30).view(2, 5, 3)).median_along_seq(keepdim=True)
            (tensor([[[12, 13, 14]], [[27, 28, 29]]], batch_dim=0, seq_dim=1),
             tensor([[[4, 4, 4]], [[4, 4, 4]]], batch_dim=0, seq_dim=1))
        """
        values, indices = torch.median(self._data, dim=self._seq_dim, keepdim=keepdim)
        if keepdim:
            return (
                BatchedTensorSeq(data=values, **self._get_kwargs()),
                BatchedTensorSeq(data=indices, **self._get_kwargs()),
            )
        batch_dim = self._batch_dim if self._seq_dim > self._batch_dim else self._batch_dim - 1
        return (
            BatchedTensor(data=values, batch_dim=batch_dim),
            BatchedTensor(data=indices, batch_dim=batch_dim),
        )

    def min_along_seq(
        self, keepdim: bool = False
    ) -> tuple[BatchedTensor | BatchedTensorSeq, BatchedTensor | BatchedTensorSeq]:
        r"""Computes the minimum values along the sequence dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not. If ``False``
                the returned type is ``BatchedTensor``, otherwise it
                is ``BatchedTensorSeq``. Default: ``False``

        Returns:
            ``BatchedTensor``: A batch with the minimum values along the
                sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).min_along_seq()
            (tensor([0, 5], batch_dim=0), tensor([0, 0], batch_dim=0))
            >>> BatchedTensorSeq(torch.arange(30).view(2, 5, 3)).min_along_seq(keepdim=True)
            (tensor([[[ 0,  1,  2]], [[15, 16, 17]]], batch_dim=0, seq_dim=1),
             tensor([[[0, 0, 0]], [[0, 0, 0]]], batch_dim=0, seq_dim=1))
        """
        values, indices = torch.min(self._data, dim=self._seq_dim, keepdim=keepdim)
        if keepdim:
            return (
                BatchedTensorSeq(data=values, **self._get_kwargs()),
                BatchedTensorSeq(data=indices, **self._get_kwargs()),
            )
        batch_dim = self._batch_dim if self._seq_dim > self._batch_dim else self._batch_dim - 1
        return (
            BatchedTensor(data=values, batch_dim=batch_dim),
            BatchedTensor(data=indices, batch_dim=batch_dim),
        )

    def sum_along_seq(self, keepdim: bool = False) -> BatchedTensor | BatchedTensorSeq:
        r"""Computes the sum values along the sequence dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not. If ``False``
                the returned type is ``BatchedTensor``, otherwise it
                is ``BatchedTensorSeq``. Default: ``False``

        Returns:
            ``TensorBatch``: A batch with the sum values along the
                sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).sum_along_seq()
            tensor([10, 35], batch_dim=0)
        """
        values = torch.sum(self._data, dim=self._seq_dim, keepdim=keepdim)
        if keepdim:
            return BatchedTensorSeq(data=values, **self._get_kwargs())
        return BatchedTensor(
            data=values,
            batch_dim=self._batch_dim if self._seq_dim > self._batch_dim else self._batch_dim - 1,
        )

    ##########################################################
    #    Indexing, slicing, joining, mutating operations     #
    ##########################################################

    def align_to_batch_seq(self) -> BatchedTensorSeq:
        r"""Aligns the current batch to the batch-sequence format.

        Returns:
            ``BatchedTensorSeq``: The batch in the batch-sequence
                format.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0)
            >>> batch.align_to_batch_seq()
            tensor([[0, 2, 4, 6, 8],
                    [1, 3, 5, 7, 9]], batch_dim=0, seq_dim=1)
        """
        return self.__class__(
            data=align_to_batch_seq(
                tensor=self._data, batch_dim=self._batch_dim, seq_dim=self._seq_dim
            ),
            batch_dim=0,
            seq_dim=1,
        )

    def align_to_seq_batch(self) -> BatchedTensorSeq:
        r"""Aligns the current batch to the sequence-batch format.

        Returns:
            ``BatchedTensorSeq``: The batch in the sequence-batch format.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=0, seq_dim=1)
            >>> batch.align_to_seq_batch()
            tensor([[0, 5],
                    [1, 6],
                    [2, 7],
                    [3, 8],
                    [4, 9]], batch_dim=1, seq_dim=0)
        """
        return self.__class__(
            data=align_to_seq_batch(
                tensor=self._data, batch_dim=self._batch_dim, seq_dim=self._seq_dim
            ),
            batch_dim=1,
            seq_dim=0,
        )

    def cat_along_batch(
        self, other: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor]
    ) -> BatchedTensorSeq:
        if isinstance(other, (BaseBatchedTensor, Tensor)):
            other = [other]
        batches = list(chain([self], other))
        check_batch_dims(get_batch_dims(batches))
        check_seq_dims(get_seq_dims(batches))
        return self.__class__(
            data=torch.cat(
                [batch.data if hasattr(batch, "data") else batch for batch in batches],
                dim=self._batch_dim,
            ),
            **self._get_kwargs(),
        )

    def cat_along_batch_(
        self, other: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor]
    ) -> None:
        if isinstance(other, (BaseBatchedTensor, Tensor)):
            other = [other]
        batches = list(chain([self], other))
        check_batch_dims(get_batch_dims(batches))
        check_seq_dims(get_seq_dims(batches))
        self._data = torch.cat(
            [batch.data if hasattr(batch, "data") else batch for batch in batches],
            dim=self._batch_dim,
        )

    def cat_along_seq(
        self, other: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor]
    ) -> BatchedTensorSeq:
        r"""Concatenates the data of the batch(es) to the current batch along
        the sequence dimension and creates a new batch.

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Returns:
            ``BatchedTensorSeq``: A batch with the concatenated data
                in the sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]])).cat_along_seq(
            ...     BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]]))
            ... )
            tensor([[ 0,  1,  2, 10, 11],
                    [ 4,  5,  6, 12, 13]], batch_dim=0, seq_dim=1)
            >>> BatchedTensorSeq(
            ...     torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1, seq_dim=0,
            ... ).cat_along_seq(
            ...     [
            ...         BatchedTensorSeq(torch.tensor([[10, 12], [11, 13]]), batch_dim=1, seq_dim=0),
            ...         BatchedTensorSeq(torch.tensor([[20, 22], [21, 23]]), batch_dim=1, seq_dim=0),
            ...     ]
            ... )
            tensor([[ 0,  4],
                    [ 1,  5],
                    [ 2,  6],
                    [10, 12],
                    [11, 13],
                    [20, 22],
                    [21, 23]], batch_dim=0, seq_dim=1)
        """
        if isinstance(other, (BaseBatchedTensor, Tensor)):
            other = [other]
        batches = list(chain([self], other))
        check_batch_dims(get_batch_dims(batches))
        check_seq_dims(get_seq_dims(batches))
        return self.__class__(
            data=torch.cat(
                [batch.data if hasattr(batch, "data") else batch for batch in batches],
                dim=self._seq_dim,
            ),
            **self._get_kwargs(),
        )

    def cat_along_seq_(
        self, other: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor]
    ) -> None:
        r"""Concatenates the data of the batch(es) to the current batch along
        the sequence dimension.

        In-place version of ``cat_along_seq()``.

        Args:
            other (``BaseBatchedTensor`` or ``torch.Tensor`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
            >>> batch.cat_along_seq_(
            ...     BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]]))
            ... )
            >>> batch
            tensor([[ 0,  1,  2, 10, 11],
                    [ 4,  5,  6, 12, 13]], batch_dim=0, seq_dim=1)
            >>> batch = BatchedTensorSeq(
            ...     torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1, seq_dim=0,
            ... )
            >>> batch.cat_along_seq(
            ...     [
            ...         BatchedTensorSeq(torch.tensor([[10, 12], [11, 13]]), batch_dim=1, seq_dim=0),
            ...         BatchedTensorSeq(torch.tensor([[20, 22], [21, 23]]), batch_dim=1, seq_dim=0),
            ...     ]
            ... )
            >>> batch
            tensor([[ 0,  4],
                    [ 1,  5],
                    [ 2,  6],
                    [10, 12],
                    [11, 13],
                    [20, 22],
                    [21, 23]], batch_dim=0, seq_dim=1)
        """
        if isinstance(other, (BaseBatchedTensor, Tensor)):
            other = [other]
        batches = list(chain([self], other))
        check_batch_dims(get_batch_dims(batches))
        check_seq_dims(get_seq_dims(batches))
        self._data = torch.cat(
            [batch.data if hasattr(batch, "data") else batch for batch in batches],
            dim=self._seq_dim,
        )

    def _get_kwargs(self) -> dict:
        return {"batch_dim": self._batch_dim, "seq_dim": self._seq_dim}


def check_data_and_dims(data: torch.Tensor, batch_dim: int, seq_dim: int) -> None:
    r"""Checks if the tensor ``data``, ``batch_dim`` and ``seq_dim`` are
    correct.

    Args:
        data (``torch.Tensor``): Specifies the tensor in the batch.
        batch_dim (int): Specifies the batch dimension in the
            ``torch.Tensor`` object.
        seq_dim (int, optional): Specifies the sequence dimension in
            the ``torch.Tensor`` object.

    Raises:
        RuntimeError: if one of the input is incorrect.
    """
    if data.dim() < 2:
        raise RuntimeError(f"data needs at least 2 dimensions (received: {data.dim()})")
    if batch_dim < 0 or batch_dim >= data.dim():
        raise RuntimeError(
            f"Incorrect batch_dim ({batch_dim}) but the value should be in [0, {data.dim() - 1}]"
        )
    if seq_dim < 0 or seq_dim >= data.dim():
        raise RuntimeError(
            f"Incorrect seq_dim ({seq_dim}) but the value should be in [0, {data.dim() - 1}]"
        )
    if batch_dim == seq_dim:
        raise RuntimeError(f"batch_dim ({batch_dim}) and seq_dim ({seq_dim}) have to be different")