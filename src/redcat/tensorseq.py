from __future__ import annotations

__all__ = ["BatchedTensorSeq", "check_data_and_dims", "check_seq_dims", "get_seq_dims"]

import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, overload

import torch
from torch import Tensor

from redcat import BaseBatch, tensor
from redcat.tensor import BatchedTensor, check_batch_dims, get_batch_dims
from redcat.utils.tensor import (
    align_to_batch_seq,
    align_to_seq_batch,
    compute_batch_seq_permutation,
)

HANDLED_FUNCTIONS = {
    torch.max: tensor.torchmax,
    torch.mean: tensor.mean,
    torch.median: tensor.median,
    torch.min: tensor.torchmin,
    torch.nanmean: tensor.nanmean,
    torch.nanmedian: tensor.nanmedian,
    torch.nansum: tensor.nansum,
    torch.prod: tensor.prod,
    torch.sum: tensor.torchsum,
}


class BatchedTensorSeq(BatchedTensor):
    r"""Implements a batched tensor to easily manipulate a batch of
    sequences.

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
        super().__init__(data, batch_dim=batch_dim, **kwargs)
        check_data_and_dims(self._data, batch_dim, seq_dim)
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
        if handled_func := HANDLED_FUNCTIONS.get(func, None):
            return handled_func(*args, **kwargs)

        batch_dims = get_batch_dims(args, kwargs)
        check_batch_dims(batch_dims)
        seq_dims = get_seq_dims(args, kwargs)
        check_seq_dims(seq_dims)
        args = [a._data if hasattr(a, "_data") else a for a in args]
        return cls(func(*args, **kwargs), batch_dim=batch_dims.pop(), seq_dim=seq_dims.pop())

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
        return self._create_new_batch(torch.full(size=shape, fill_value=fill_value, **kwargs))

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
        return self._create_new_batch(torch.ones(*shape, **kwargs))

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
        return self._create_new_batch(torch.zeros(*shape, **kwargs))

    @classmethod
    def from_seq_batch(cls, data: Any, **kwargs) -> BatchedTensorSeq:
        r"""Creates a batch where the first dimension is the sequence
        dimension and the second dimension is the batch dimension.

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
        other: BatchedTensor | Tensor | int | float,
        alpha: int | float = 1.0,
    ) -> None:
        check_seq_dims(get_seq_dims((self, other)))
        super().add_(other, alpha=alpha)

    def div_(
        self,
        other: BatchedTensor | Tensor | int | float,
        rounding_mode: str | None = None,
    ) -> None:
        check_seq_dims(get_seq_dims((self, other)))
        super().div_(other, rounding_mode=rounding_mode)

    def fmod_(self, divisor: BatchedTensor | Tensor | int | float) -> None:
        check_seq_dims(get_seq_dims((self, divisor)))
        super().fmod_(divisor)

    def mul_(self, other: BatchedTensor | Tensor | int | float) -> None:
        check_seq_dims(get_seq_dims((self, other)))
        super().mul_(other)

    def sub_(
        self,
        other: BatchedTensor | Tensor | int | float,
        alpha: int | float = 1.0,
    ) -> None:
        check_seq_dims(get_seq_dims((self, other)))
        super().sub_(other, alpha=alpha)

    ###########################################################
    #     Mathematical | advanced arithmetical operations     #
    ###########################################################

    def cumsum_along_seq(self, **kwargs) -> BatchedTensorSeq:
        r"""Computes the cumulative sum of elements of the current batch
        in the sequence dimension.

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
        return self.cumsum(dim=self._seq_dim, **kwargs)

    def cumsum_along_seq_(self) -> None:
        r"""Computes the cumulative sum of elements of the current batch
        in the sequence dimension.

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
        self.cumsum_(self._seq_dim)

    def logcumsumexp_along_seq(self) -> BatchedTensorSeq:
        r"""Computes the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in the sequence
        dimension.

        Returns:
            ``BatchedTensorSeq``: A batch with the cumulative
                summation of the exponentiation of elements of the
                current batch in the sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5).float()).logcumsumexp_along_seq()
            tensor([[0.0000, 1.3133, 2.4076, 3.4402, 4.4519],
                    [5.0000, 6.3133, 7.4076, 8.4402, 9.4519]], batch_dim=0, seq_dim=1)
        """
        return self.logcumsumexp(self._seq_dim)

    def logcumsumexp_along_seq_(self) -> None:
        r"""Computes the logarithm of the cumulative summation of the
        exponentiation of elements of the current batch in the sequence
        dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5).float())
            >>> batch.logcumsumexp_along_seq_()
            >>> batch
            tensor([[0.0000, 1.3133, 2.4076, 3.4402, 4.4519],
                    [5.0000, 6.3133, 7.4076, 8.4402, 9.4519]], batch_dim=0, seq_dim=1)
        """
        self.logcumsumexp_(self._seq_dim)

    def permute_along_seq(self, permutation: Sequence[int] | Tensor) -> BatchedTensorSeq:
        r"""Permutes the data along the sequence dimension.

        Args:
            permutation (sequence or ``torch.Tensor`` of type long
                and shape ``(dimension,)``): Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.

        Returns:
            ``BatchedTensorSeq``: A new batch with permuted data.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.permute_along_seq([2, 1, 3, 0, 4])
            tensor([[2, 1, 3, 0, 4],
                    [7, 6, 8, 5, 9]], batch_dim=0, seq_dim=1)
        """
        return self.permute_along_dim(permutation, dim=self._seq_dim)

    def permute_along_seq_(self, permutation: Sequence[int] | Tensor) -> None:
        r"""Permutes the data along the sequence dimension.

        Args:
            permutation (sequence or ``torch.Tensor`` of type long
                and shape ``(dimension,)``): Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.permute_along_seq_([2, 1, 3, 0, 4])
            >>> batch
            tensor([[2, 1, 3, 0, 4],
                    [7, 6, 8, 5, 9]], batch_dim=0, seq_dim=1)
        """
        self.permute_along_dim_(permutation, dim=self._seq_dim)

    def shuffle_along_seq(self, generator: torch.Generator | None = None) -> BatchedTensorSeq:
        r"""Shuffles the data along the sequence dimension.

        Args:
            generator (``torch.Generator`` or ``None``, optional):
                Specifies an optional random generator.
                Default: ``None``

        Returns:
            ``BatchedTensorSeq``:  A new batch with shuffled data.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).shuffle_along_seq()
            tensor([[2, 1, 4, 0, 3],
                    [7, 6, 9, 5, 8]], batch_dim=0, seq_dim=1)
        """
        return self.permute_along_seq(torch.randperm(self.seq_len, generator=generator))

    def shuffle_along_seq_(self, generator: torch.Generator | None = None) -> None:
        r"""Shuffles the data along the sequence dimension.

        Args:
            generator (``torch.Generator`` or ``None``, optional):
                Specifies an optional random generator.
                Default: ``None``

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.shuffle_along_seq_()
            >>> batch
            tensor([[2, 1, 4, 0, 3],
                    [7, 6, 9, 5, 8]], batch_dim=0, seq_dim=1)
        """
        self.permute_along_seq_(torch.randperm(self.seq_len, generator=generator))

    def sort_along_seq(
        self, descending: bool = False, stable: bool = False
    ) -> torch.return_types.sort:
        r"""Sorts the elements of the batch along the sequence dimension
        in monotonic order by value.

        Args:
            descending (bool, optional): Controls the sorting order.
                If ``True``, the elements are sorted in descending
                order by value. Default: ``False``
            stable (bool, optional): Makes the sorting routine stable,
                which guarantees that the order of equivalent elements
                is preserved. Default: ``False``

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
            >>> BatchedTensorSeq(torch.rand(2, 5)).sort_along_seq()
            torch.return_types.sort(
            values=tensor([[0.2884, 0.4014, 0.5857, 0.6949, 0.8264],
                    [0.3811, 0.4431, 0.4857, 0.6009, 0.7207]], batch_dim=0, seq_dim=1),
            indices=tensor([[2, 4, 3, 0, 1],
                    [1, 0, 4, 2, 3]], batch_dim=0, seq_dim=1))
        """
        return self.sort(dim=self._seq_dim, descending=descending, stable=stable)

    ################################################
    #     Mathematical | point-wise operations     #
    ################################################

    def pow_(self, exponent: int | float | BatchedTensor) -> None:
        check_seq_dims(get_seq_dims((self, exponent)))
        super().pow_(exponent)

    #############################################
    #     Mathematical | logical operations     #
    #############################################

    def logical_and_(self, other: BatchedTensor | Tensor) -> None:
        check_seq_dims(get_seq_dims((self, other)))
        super().logical_and_(other)

    def logical_or_(self, other: BatchedTensor | Tensor) -> None:
        check_seq_dims(get_seq_dims((self, other)))
        super().logical_or_(other)

    def logical_xor_(self, other: BatchedTensor | Tensor) -> None:
        check_seq_dims(get_seq_dims((self, other)))
        super().logical_xor_(other)

    ################################
    #     Reduction operations     #
    ################################

    def max_along_seq(self, keepdim: bool = False) -> torch.return_types.max:
        r"""Computes the maximum values along the sequence dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.return_types.max``: A batch with
                the maximum values along the sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).max_along_seq()
            torch.return_types.max(
            values=tensor([4, 9]),
            indices=tensor([4, 4]))
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).max_along_seq(keepdim=True)
            torch.return_types.max(
            values=tensor([[4], [9]]),
            indices=tensor([[4], [4]]))
        """
        return self.max(dim=self._seq_dim, keepdim=keepdim)

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
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5).float()).mean_along_seq()
            tensor([2.0, 7.0])
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5).float()).mean_along_seq(keepdim=True)
            tensor([[2.0], [7.0]])
        """
        return self.mean(dim=self._seq_dim, keepdim=keepdim)

    def median_along_seq(self, keepdim: bool = False) -> torch.return_types.median:
        r"""Computes the median values along the sequence dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.return_types.median``:  The first tensor will
                be populated with the median values and the second
                tensor, which must have dtype long, with their indices
                in the sequence dimension of input.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).median_along_seq()
            torch.return_types.median(
            values=tensor([2, 7]),
            indices=tensor([2, 2]))
        """
        return self.median(dim=self._seq_dim, keepdim=keepdim)

    def min_along_seq(self, keepdim: bool = False) -> torch.return_types.min:
        r"""Computes the minimum values along the sequence dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.return_types.min``: A batch with
                the minimum values along the sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).min_along_seq()
            torch.return_types.min(
            values=tensor([0, 5]),
            indices=tensor([0, 0]))
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).min_along_seq(keepdim=True)
            torch.return_types.min(
            values=tensor([[0], [5]]),
            indices=tensor([[0], [0]]))
        """
        return self.min(dim=self._seq_dim, keepdim=keepdim)

    def nanmean_along_seq(self, keepdim: bool = False) -> Tensor:
        r"""Computes the mean values along the sequence dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the batch dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.Tensor``: A batch with
                the mean values along the sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(
            ...     torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])
            ... ).nanmean_along_seq()
            tensor([2.0, 6.5])
            >>> BatchedTensorSeq(
            ...     torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])
            ... ).nanmean_along_seq(keepdim=True)
            tensor([[2.0], [6.5]])
        """
        return self.nanmean(dim=self._seq_dim, keepdim=keepdim)

    def nanmedian_along_seq(self, keepdim: bool = False) -> torch.return_types.nanmedian:
        r"""Computes the median values along the sequence dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.return_types.nanmedian``:  The first tensor will
                be populated with the median values and the second
                tensor, which must have dtype long, with their indices
                in the sequence dimension of input.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(
            ...     torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])
            ... ).nanmedian_along_seq()
            torch.return_types.nanmedian(
            values=tensor([2., 6.]),
            indices=tensor([2, 1]))
        """
        return self.nanmedian(dim=self._seq_dim, keepdim=keepdim)

    def nansum_along_seq(self, keepdim: bool = False) -> Tensor:
        r"""Computes the sum values along the sequence dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.Tensor``: A tensor with the sum values along the
                sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(
            ...     torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])
            ... ).nansum_along_seq()
            tensor([20., 26.])
        """
        return self.nansum(dim=self._seq_dim, keepdim=keepdim)

    def prod_along_seq(self, keepdim: bool = False) -> Tensor:
        r"""Computes the product values along the sequence dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not.
                Default: ``False``

        Returns:
            ``torch.Tensor``: A batch with
                the product values along the sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]])).prod_along_seq()
            tensor([ 120, 3024])
            >>> BatchedTensorSeq(
            ...     torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]])
            ... ).prod_along_seq(keepdim=True)
            tensor([[ 120], [3024]])
        """
        return self.prod(dim=self._seq_dim, keepdim=keepdim)

    def sum_along_seq(self, keepdim: bool = False) -> Tensor:
        r"""Computes the sum values along the sequence dimension.

        Args:
            keepdim (bool): Indicates whether the output tensor has
                the sequence dimension retained or not. If ``False``
                the returned type is ``BatchedTensor``, otherwise it
                is ``BatchedTensorSeq``. Default: ``False``

        Returns:
            ``torch.Tensor``: A tensor with the sum values along the
                sequence dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).sum_along_seq()
            tensor([10, 35])
        """
        return self.sum(dim=self._seq_dim, keepdim=keepdim)

    ##########################################################
    #    Indexing, slicing, joining, mutating operations     #
    ##########################################################

    def align_as(self, other: BatchedTensorSeq) -> BatchedTensorSeq:
        r"""Aligns the current batch with the batch ``other``.

        This method makes sure the batch and sequence dimensions
        are aligned.

        Args:
            other (``BatchedTensorSeq``): Specifies the batch to use to
                align the current batch.

        Returns:
            ``BatchedTensorSeq``: The aligned batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            # batch-sequence -> sequence-batch
            >>> seq_batch = BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).align_as(seq_batch)
            tensor([[0, 5],
                    [1, 6],
                    [2, 7],
                    [3, 8],
                    [4, 9]], batch_dim=0, seq_dim=1)
            # sequence-batch -> batch-sequence
            >>> batch_seq = BatchedTensorSeq(torch.ones(2, 3))
            >>> BatchedTensorSeq.from_seq_batch(
            ...     torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0
            ... ).align_as(batch_seq)
            tensor([[0, 2, 4, 6, 8],
                    [1, 3, 5, 7, 9]], batch_dim=1, seq_dim=0)
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Incorrect type {type(other)}. No implementation available to `align_as` "
                f"{type(self)} with {type(other)}"
            )
        return self.__class__(
            self._data.permute(  # Align only the batch and sequence dims
                *compute_batch_seq_permutation(
                    num_dims=self._data.dim(),
                    old_batch_dim=self.batch_dim,
                    old_seq_dim=self.seq_dim,
                    new_batch_dim=other.batch_dim,
                    new_seq_dim=other.seq_dim,
                )
            ),
            batch_dim=other.batch_dim,
            seq_dim=other.seq_dim,
        )

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
            align_to_batch_seq(tensor=self._data, **self._get_kwargs()),
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
            align_to_seq_batch(tensor=self._data, batch_dim=self._batch_dim, seq_dim=self._seq_dim),
            batch_dim=1,
            seq_dim=0,
        )

    def cat_along_seq(
        self, tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor]
    ) -> BatchedTensorSeq:
        r"""Concatenates the data of the batch(es) to the current batch
        along the sequence dimension and creates a new batch.

        Args:
            tensors (``BatchedTensorSeq`` or ``torch.Tensor`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Returns:
            ``BatchedTensorSeq``: A batch with the concatenated data
                along the sequence dimension.

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
                    [21, 23]], batch_dim=1, seq_dim=0)
        """
        return self.cat(tensors, dim=self._seq_dim)

    def cat_along_seq_(
        self, tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor]
    ) -> None:
        r"""Concatenates the data of the batch(es) to the current batch
        along the sequence dimension.

        In-place version of ``cat_along_seq()``.

        Args:
            tensors (``BatchedTensor`` or ``torch.Tensor`` or
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
                    [21, 23]], batch_dim=1, seq_dim=0)
        """
        self.cat_(tensors, dim=self._seq_dim)

    def chunk_along_seq(self, chunks: int) -> tuple[BatchedTensorSeq, ...]:
        r"""Splits the batch into chunks along the sequence dimension.

        Args:
            chunks (int): Specifies the number of chunks.

        Returns:
            tuple: The batch split into chunks along the sequence
                dimension.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).chunk_along_seq(chunks=3)
            (tensor([[0, 1], [5, 6]], batch_dim=0, seq_dim=1),
             tensor([[2, 3], [7, 8]], batch_dim=0, seq_dim=1),
             tensor([[4], [9]], batch_dim=0, seq_dim=1))
        """
        return self.chunk(chunks, self._seq_dim)

    def index_select_along_seq(self, index: Tensor | Sequence[int]) -> BatchedTensorSeq:
        r"""Slices the batch along the sequence dimension at the given
        indices.

        Args:
            index (``torch.Tensor`` or list or tuple): Specifies the
                indices to select.

        Returns:
            ``BatchedTensorSeq``: A new batch sliced along the sequence
                dimension at the given indices.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.index_select_along_seq([2, 4])
            tensor([[2, 4],
                    [7, 9]], batch_dim=0, seq_dim=1)
            >>> batch.index_select_along_seq(torch.tensor([4, 3, 2, 1, 0]))
            tensor([[4, 3, 2, 1, 0],
                    [9, 8, 7, 6, 5]], batch_dim=0, seq_dim=1)
        """
        return self.index_select(self._seq_dim, index)

    def masked_fill(
        self, mask: BatchedTensor | Tensor, value: bool | int | float
    ) -> BatchedTensorSeq:
        check_batch_dims(get_batch_dims((self, mask)))
        check_seq_dims(get_seq_dims((self, mask)))
        if isinstance(mask, BatchedTensor):
            mask = mask.data
        return self._create_new_batch(self._data.masked_fill(mask.data, value))

    def repeat_along_seq(self, repeats: int) -> BatchedTensorSeq:
        r"""Repeats the batch along the sequence dimension.

        Args:
            repeats (int): Specifies the number of times to repeat
                the batch along the sequence dimension.

        Returns:
            ``BatchedTensorSeq``: A repeated version of the input batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).repeat_along_seq(2)
            tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]], batch_dim=0, seq_dim=1)
        """
        sizes = [1] * self._data.dim()
        sizes[self._seq_dim] = repeats
        return self._create_new_batch(self._data.repeat(*sizes))

    def select_along_seq(self, index: int) -> BatchedTensor:
        r"""Slices the batch along the sequence dimension at the given
        index.

        Args:
            index (int): Specifies the index to select.

        Returns:
            ``BatchedTensorSeq``: The batch sliced along the sequence
                dimension at the given index.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).select_along_seq(2)
            tensor([2, 7], batch_dim=0)
        """
        return BatchedTensor(
            data=self._data.select(self._seq_dim, index),
            batch_dim=self._batch_dim if self._seq_dim > self._batch_dim else self._batch_dim - 1,
        )

    def slice_along_seq(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> BatchedTensorSeq:
        r"""Slices the batch in the sequence dimension.

        Args:
            start (int, optional): Specifies the index where the
                slicing of object starts. Default: ``0``
            stop (int, optional): Specifies the index where the
                slicing of object stops. ``None`` means last.
                Default: ``None``
            step (int, optional): Specifies the increment between
                each index for slicing. Default: ``1``

        Returns:
            ``BatchedTensorSeq``: A slice of the current batch.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]))
            >>> batch.slice_along_seq(start=2)
            tensor([[2, 3, 4],
                    [7, 6, 5]], batch_dim=0, seq_dim=1)
            >>> batch.slice_along_seq(stop=3)
            tensor([[0, 1, 2],
                    [9, 8, 7]], batch_dim=0, seq_dim=1)
            >>> batch.slice_along_seq(step=2)
            tensor([[0, 2, 4],
                    [9, 7, 5]], batch_dim=0, seq_dim=1)
        """
        return self.slice_along_dim(self._seq_dim, start, stop, step)

    def split_along_seq(
        self, split_size_or_sections: int | Sequence[int]
    ) -> tuple[BatchedTensorSeq, ...]:
        return self.split(split_size_or_sections, dim=self._seq_dim)

    def take_along_seq(self, indices: BaseBatch | Tensor | Sequence) -> BatchedTensorSeq:
        r"""Takes values along the sequence dimension.

        Args:
            indices (``BaseBatch`` or ``Tensor`` or sequence):
                Specifies the indices to take along the sequence
                dimension.

        Returns:
            ``BaseBatch``: The sequence with the selected data.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> BatchedTensorSeq(torch.arange(10).view(2, 5)).take_along_seq(
            ...     BatchedTensorSeq(torch.tensor([[3, 0, 1], [2, 3, 4]]))
            ... )
            tensor([[3, 0, 1],
                    [7, 8, 9]], batch_dim=0, seq_dim=1)
        """
        return self.take_along_dim(indices, dim=self._seq_dim)

    def unsqueeze(self, dim: int) -> BatchedTensorSeq:
        return self.__class__(
            self._data.unsqueeze(dim=dim),
            batch_dim=self._batch_dim + 1
            if self._batch_dim >= dim and dim >= 0
            else self._batch_dim,
            seq_dim=self._seq_dim + 1 if self._seq_dim >= dim and dim >= 0 else self._seq_dim,
        )

    def view_as(self, other: BatchedTensor | Tensor) -> BatchedTensorSeq:
        check_batch_dims(get_batch_dims((self, other)))
        check_seq_dims(get_seq_dims((self, other)))
        return self._create_new_batch(self._data.view_as(other.data))

    def _get_kwargs(self) -> dict:
        return {"batch_dim": self._batch_dim, "seq_dim": self._seq_dim}


def check_data_and_dims(data: Tensor, batch_dim: int, seq_dim: int) -> None:
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


def check_seq_dims(dims: set[int]) -> None:
    r"""Gets the sequence dimensions from the inputs.

    Args:
        dims (set): Specifies the sequence dims to check.

    Raises:
        RuntimeError if there are more than one sequence dimension.
    """
    if len(dims) != 1:
        raise RuntimeError(
            f"The sequence dimensions do not match. Received multiple values: {dims}"
        )


def get_seq_dims(args: Iterable[Any, ...], kwargs: Mapping[str, Any] | None = None) -> set[int]:
    r"""Gets the sequence dimensions from the inputs.

    Args:
        args: Variable length argument list.
        kwargs: Arbitrary keyword arguments.

    Returns:
        set: The sequence dimensions.
    """
    kwargs = kwargs or {}
    dims = {val._seq_dim for val in args if hasattr(val, "_seq_dim")}
    dims.update({val._seq_dim for val in kwargs.values() if hasattr(val, "_seq_dim")})
    return dims


def implements(torch_function: Callable) -> Callable:
    """Register a torch function override for BatchedTensor."""

    def decorator(func: Callable) -> Callable:
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements(torch.cat)
def cat(
    tensors: Sequence[BatchedTensor | Tensor],
    dim: int = 0,
) -> BatchedTensorSeq:
    r"""See ``torch.cat`` documentation."""
    batch_dims = get_batch_dims(tensors)
    check_batch_dims(batch_dims)
    seq_dims = get_seq_dims(tensors)
    check_seq_dims(seq_dims)
    return BatchedTensorSeq(
        torch.cat(
            [tensor._data if hasattr(tensor, "_data") else tensor for tensor in tensors], dim=dim
        ),
        batch_dim=batch_dims.pop(),
        seq_dim=seq_dims.pop(),
    )


@implements(torch.chunk)
def chunk(tensor: BatchedTensorSeq, chunks: int, dim: int = 0) -> tuple[BatchedTensorSeq, ...]:
    r"""See ``torch.chunk`` documentation."""
    return tuple(
        BatchedTensorSeq(chunk, batch_dim=tensor.batch_dim, seq_dim=tensor.seq_dim)
        for chunk in tensor.data.chunk(chunks, dim=dim)
    )


@implements(torch.maximum)
def maximum(
    input: BatchedTensorSeq, other: BatchedTensor | Tensor  # noqa: A002
) -> BatchedTensorSeq:
    r"""See ``torch.maximum`` documentation."""
    check_batch_dims(get_batch_dims((input, other)))
    check_seq_dims(get_seq_dims((input, other)))
    if isinstance(other, BatchedTensor):
        other = other.data
    return BatchedTensorSeq(
        torch.maximum(input.data, other), batch_dim=input.batch_dim, seq_dim=input.seq_dim
    )


@implements(torch.minimum)
def minimum(
    input: BatchedTensorSeq, other: BatchedTensor | Tensor  # noqa: A002
) -> BatchedTensorSeq:
    r"""See ``torch.minimum`` documentation."""
    check_batch_dims(get_batch_dims((input, other)))
    check_seq_dims(get_seq_dims((input, other)))
    if isinstance(other, BatchedTensor):
        other = other.data
    return BatchedTensorSeq(
        torch.minimum(input.data, other), batch_dim=input.batch_dim, seq_dim=input.seq_dim
    )


@implements(torch.select)
def select(input: BatchedTensorSeq, dim: int, index: int) -> Tensor:  # noqa: A002
    r"""See ``torch.select`` documentation."""
    return torch.select(input.data, dim=dim, index=index)


@implements(torch.sort)
def sort(input: BatchedTensorSeq, *args, **kwargs) -> torch.return_types.sort:  # noqa: A002
    r"""See ``torch.sort`` documentation."""
    values, indices = torch.sort(input.data, *args, **kwargs)
    return torch.return_types.sort(
        [
            BatchedTensorSeq(data=values, batch_dim=input.batch_dim, seq_dim=input.seq_dim),
            BatchedTensorSeq(data=indices, batch_dim=input.batch_dim, seq_dim=input.seq_dim),
        ]
    )


@implements(torch.split)
def split(
    tensor: BatchedTensorSeq, split_size_or_sections: int | Sequence[int], dim: int = 0
) -> tuple[BatchedTensorSeq, ...]:
    r"""See ``torch.split`` documentation."""
    return tuple(
        BatchedTensorSeq(chunk, batch_dim=tensor.batch_dim, seq_dim=tensor.seq_dim)
        for chunk in tensor.data.split(split_size_or_sections, dim=dim)
    )


@overload
def take_along_dim(
    input: BatchedTensor | Tensor,  # noqa: A002
    indices: BatchedTensor | Tensor,
) -> Tensor:
    r"""See ``torch.take_along_dim`` documentation."""


@overload
def take_along_dim(
    input: BatchedTensor | Tensor, indices: BatchedTensor | Tensor, dim: int  # noqa: A002
) -> BatchedTensorSeq:
    r"""See ``torch.take_along_dim`` documentation."""


@implements(torch.take_along_dim)
def take_along_dim(
    input: BatchedTensor | Tensor,  # noqa: A002
    indices: BatchedTensor | Tensor,
    dim: int | None = None,
) -> BatchedTensorSeq | Tensor:
    r"""See ``torch.take_along_dim`` documentation."""
    batch_dims = get_batch_dims((input, indices))
    check_batch_dims(batch_dims)
    seq_dims = get_seq_dims((input, indices))
    check_seq_dims(seq_dims)
    if isinstance(input, BatchedTensor):
        input = input.data  # noqa: A001
    if isinstance(indices, BatchedTensor):
        indices = indices.data
    if dim is None:
        return torch.take_along_dim(input, indices)
    return BatchedTensorSeq(
        torch.take_along_dim(input, indices, dim=dim),
        batch_dim=batch_dims.pop(),
        seq_dim=seq_dims.pop(),
    )
