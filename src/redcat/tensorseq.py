from __future__ import annotations

__all__ = [
    "BatchedTensorSeq",
    "check_data_and_dims",
    "from_sequences",
]

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import torch
from coola import objects_are_allclose, objects_are_equal
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from redcat import BaseBatch, tensor
from redcat.tensor import BatchedTensor, get_batch_dims
from redcat.utils.common import check_batch_dims, check_seq_dims, get_seq_dims
from redcat.utils.tensor import (
    align_to_batch_seq,
    align_to_seq_batch,
    compute_batch_seq_permutation,
    to_tensor,
)

HANDLED_FUNCTIONS = {
    torch.amax: tensor.amax,
    torch.amin: tensor.amin,
    torch.argmax: tensor.argmax,
    torch.argmin: tensor.argmin,
    torch.argsort: tensor.argsort,
    torch.cat: tensor.cat,
    torch.chunk: tensor.chunk,
    torch.max: tensor.torchmax,
    torch.maximum: tensor.maximum,
    torch.mean: tensor.mean,
    torch.median: tensor.median,
    torch.min: tensor.torchmin,
    torch.minimum: tensor.minimum,
    torch.nanmean: tensor.nanmean,
    torch.nanmedian: tensor.nanmedian,
    torch.nansum: tensor.nansum,
    torch.prod: tensor.prod,
    torch.select: tensor.select,
    torch.sort: tensor.sort,
    torch.split: tensor.split,
    torch.sum: tensor.torchsum,
    torch.take_along_dim: tensor.take_along_dim,
}


class BatchedTensorSeq(BatchedTensor):
    r"""Implements a batched tensor to easily manipulate a batch of
    sequences.

    Args:
    ----
        data (array_like): Specifies the data for the tensor. It can
            be a torch.Tensor, list, tuple, NumPy ndarray, scalar,
            and other types.
        batch_dim (int, optional): Specifies the batch dimension
            in the ``torch.Tensor`` object. Default: ``0``
        seq_dim (int, optional): Specifies the sequence dimension in
            the ``torch.Tensor`` object. Default: ``1``
        kwargs: Keyword arguments that are passed to
            ``torch.as_tensor``.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from redcat import BatchedTensorSeq
        >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
        >>> batch
        tensor([[0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9]], batch_dim=0, seq_dim=1)
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
        r"""``int``: The sequence length."""
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
        ----
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
        -------
            ``BaseBatchedTensor``: A batch filled with the scalar value.

        Example usage:

        .. code-block:: pycon

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
        ----
            batch_size (int or ``None``): Specifies the batch size.
                If ``None``, the batch size of the current batch is
                used. Default: ``None``.
            seq_len (int or ``None``): Specifies the sequence length.
                If ``None``, the sequence length of the current batch
                is used. Default: ``None``.
            **kwargs: See the documentation of
                ``torch.Tensor.new_ones``.

        Returns:
        -------
            ``BaseBatchedTensor``: A batch filled with the scalar
                value ``1``.

        Example usage:

        .. code-block:: pycon

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
        ----
            batch_size (int or ``None``): Specifies the batch size.
                If ``None``, the batch size of the current batch is
                used. Default: ``None``.
            seq_len (int or ``None``): Specifies the sequence length.
                If ``None``, the sequence length of the current batch
                is used. Default: ``None``.
            **kwargs: See the documentation of
                ``torch.Tensor.new_zeros``.

        Returns:
        -------
            ``BaseBatchedTensor``: A batch filled with the scalar
                value ``0``.

        Example usage:

        .. code-block:: pycon

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
        ----
            data (array_like): Specifies the data for the tensor. It can
                be a torch.Tensor, list, tuple, NumPy ndarray, scalar,
                and other types.
            kwargs: Keyword arguments that are passed to
                ``torch.as_tensor``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq.from_seq_batch(torch.ones(2, 3))
            >>> batch
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
        return objects_are_allclose(
            self._data, other.data, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    def equal(self, other: Any) -> bool:
        if not isinstance(other, BatchedTensorSeq):
            return False
        if self._batch_dim != other.batch_dim or self._seq_dim != other.seq_dim:
            return False
        return objects_are_equal(self._data, other.data)

    ##################################################
    #     Mathematical | arithmetical operations     #
    ##################################################

    ###########################################################
    #     Mathematical | advanced arithmetical operations     #
    ###########################################################

    def argsort_along_seq(self, **kwargs) -> BatchedTensorSeq:
        r"""Sorts the elements of the batch along the sequence dimension
        in monotonic order by value.

        Args:
        ----
            **kwargs: Arbitrary keyword arguments.

        Returns:
        -------
            ``BatchedTensor``: The indices that sort the batch along
                the sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.argsort_along_seq(descending=True)
            tensor([[4, 3, 2, 1, 0],
                    [4, 3, 2, 1, 0]], batch_dim=0, seq_dim=1)
        """
        return self.argsort(dim=self._seq_dim, **kwargs)

    def cumprod_along_seq(self, *args, **kwargs) -> BatchedTensorSeq:
        r"""Computes the cumulative product of elements of the current
        batch in the sequence dimension.

        Args:
        ----
            *args: See the documentation of ``torch.Tensor.cumprod``
            **kwargs: See the documentation of ``torch.Tensor.cumprod``

        Returns:
        -------
            ``BatchedTensorSeq``: A batch with the cumulative sum of
                elements of the current batch in the sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5)).cumprod_along_seq()
            >>> batch
            tensor([[    0,     0,     0,     0,     0],
                    [    5,    30,   210,  1680, 15120]], batch_dim=0, seq_dim=1)
        """
        return self.cumprod(self._seq_dim, *args, **kwargs)

    def cumprod_along_seq_(self, *args, **kwargs) -> None:
        r"""Computes the cumulative product of elements of the current
        batch in the sequence dimension.

        Args:
        ----
            *args: See the documentation of ``torch.Tensor.cumprod``
            **kwargs: See the documentation of ``torch.Tensor.cumprod``

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.cumprod_along_seq_()
            >>> batch
            tensor([[    0,     0,     0,     0,     0],
                    [    5,    30,   210,  1680, 15120]], batch_dim=0, seq_dim=1)
        """
        self.cumprod_(self._seq_dim, *args, **kwargs)

    def cumsum_along_seq(self, **kwargs) -> BatchedTensorSeq:
        r"""Computes the cumulative sum of elements of the current batch
        in the sequence dimension.

        Args:
        ----
            **kwargs: see ``torch.cumsum`` documentation

        Returns:
        -------
            ``BatchedTensorSeq``: A batch with the cumulative sum of
                elements of the current batch in the sequence dimension.

        Example usage:

        .. code-block:: pycon

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

        .. code-block:: pycon

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
        -------
            ``BatchedTensorSeq``: A batch with the cumulative
                summation of the exponentiation of elements of the
                current batch in the sequence dimension.

        Example usage:

        .. code-block:: pycon

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

        .. code-block:: pycon

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
        ----
            permutation (sequence or ``torch.Tensor`` of type long
                and shape ``(seq_len,)``): Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.

        Returns:
        -------
            ``BatchedTensorSeq``: A new batch with permuted data.

        Example usage:

        .. code-block:: pycon

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
        ----
            permutation (sequence or ``torch.Tensor`` of type long
                and shape ``(seq_len,)``): Specifies the permutation
                to use on the data. The dimension of the permutation
                input should be compatible with the shape of the data.

        Example usage:

        .. code-block:: pycon

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
        ----
            generator (``torch.Generator`` or ``None``, optional):
                Specifies an optional random generator.
                Default: ``None``

        Returns:
        -------
            ``BatchedTensorSeq``:  A new batch with shuffled data.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.shuffle_along_seq()
            tensor([[...]], batch_dim=0, seq_dim=1)
        """
        return self.permute_along_seq(torch.randperm(self.seq_len, generator=generator))

    def shuffle_along_seq_(self, generator: torch.Generator | None = None) -> None:
        r"""Shuffles the data along the sequence dimension.

        Args:
        ----
            generator (``torch.Generator`` or ``None``, optional):
                Specifies an optional random generator.
                Default: ``None``

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.shuffle_along_seq_()
            >>> batch
            tensor([[...]], batch_dim=0, seq_dim=1)
        """
        self.permute_along_seq_(torch.randperm(self.seq_len, generator=generator))

    def sort_along_seq(self, *args, **kwargs) -> torch.return_types.sort:
        r"""Sorts the elements of the batch along the sequence dimension
        in monotonic order by value.

        Args:
        ----
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
        -------
            (``BatchedTensorSeq``, ``BatchedTensorSeq``): A tuple with
                two values:
                    - The first batch contains the batch values sorted
                        along the sequence dimension.
                    - The second batch contains the indices that sort
                        the batch along the sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.sort_along_seq(descending=True)
            torch.return_types.sort(
            values=tensor([[4, 3, 2, 1, 0],
                    [9, 8, 7, 6, 5]], batch_dim=0, seq_dim=1),
            indices=tensor([[4, 3, 2, 1, 0],
                    [4, 3, 2, 1, 0]], batch_dim=0, seq_dim=1))
        """
        return self.sort(self._seq_dim, *args, **kwargs)

    ################################################
    #     Mathematical | point-wise operations     #
    ################################################

    #############################################
    #     Mathematical | logical operations     #
    #############################################

    ################################
    #     Reduction operations     #
    ################################

    def amax_along_seq(self, *args, **kwargs) -> Tensor:
        r"""Computes the maximum along the sequence dimension.

        Args:
        ----
            *args: See the documentation of ``torch.Tensor.amax``
            **kwargs: See the documentation of ``torch.Tensor.amax``

        Returns:
        -------
            ``torch.Tensor``: The maximum along the sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.amax_along_seq()
            tensor([4, 9])
            >>> batch.amax_along_seq(keepdim=True)
            tensor([[4], [9]])
        """
        return self.amax(self._seq_dim, *args, **kwargs)

    def amin_along_seq(self, *args, **kwargs) -> Tensor:
        r"""Computes the minimum along the sequence dimension.

        Args:
        ----
            *args: See the documentation of ``torch.Tensor.amin``
            **kwargs: See the documentation of ``torch.Tensor.amin``

        Returns:
        -------
            ``torch.Tensor``: The minimum along the sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.amin_along_seq()
            tensor([0, 5])
            >>> batch.amin_along_seq(keepdim=True)
            tensor([[0], [5]])
        """
        return self.amin(self._seq_dim, *args, **kwargs)

    def argmax_along_seq(self, *args, **kwargs) -> Tensor:
        r"""Computes the indices of the maximum value along the sequence
        dimension.

        Args:
        ----
            *args: See the documentation of ``torch.Tensor.argmax``
            **kwargs: See the documentation of ``torch.Tensor.argmax``

        Returns:
        -------
            ``torch.Tensor``: The indices of the maximum value along
                the sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.argmax_along_seq()
            tensor([4, 4])
            >>> batch.argmax_along_seq(keepdim=True)
            tensor([[4], [4]])
        """
        return self.argmax(self._seq_dim, *args, **kwargs)

    def argmin_along_seq(self, *args, **kwargs) -> Tensor:
        r"""Computes the indices of the minimum value along the sequence
        dimension.

        Args:
        ----
            *args: See the documentation of ``torch.Tensor.argmin``
            **kwargs: See the documentation of ``torch.Tensor.argmin``

        Returns:
        -------
            ``torch.Tensor``: The indices of the minimum value along
                the sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.argmin_along_seq()
            tensor([0, 0])
            >>> batch.argmin_along_seq(keepdim=True)
            tensor([[0], [0]])
        """
        return self.argmin(self._seq_dim, *args, **kwargs)

    def max_along_seq(self, *args, **kwargs) -> torch.return_types.max:
        r"""Computes the maximum values along the sequence dimension.

        Args:
        ----
            *args: See the documentation of ``torch.Tensor.max``
            **kwargs: See the documentation of ``torch.Tensor.max``

        Returns:
        -------
            ``torch.return_types.max``: A batch with
                the maximum values along the sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.max_along_seq()
            torch.return_types.max(
            values=tensor([4, 9]),
            indices=tensor([4, 4]))
            >>> batch.max_along_seq(keepdim=True)
            torch.return_types.max(
            values=tensor([[4], [9]]),
            indices=tensor([[4], [4]]))
        """
        return self.max(self._seq_dim, *args, **kwargs)

    def mean_along_seq(self, *args, **kwargs) -> Tensor:
        r"""Computes the mean values along the sequence dimension.

        Args:
        ----
            *args: See the documentation of ``torch.Tensor.mean``
            **kwargs: See the documentation of ``torch.Tensor.mean``

        Returns:
        -------
            ``torch.Tensor``: A tensor with the mean values along
                the sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5).float())
            >>> batch.mean_along_seq()
            tensor([2., 7.])
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5).float())
            >>> batch.mean_along_seq(keepdim=True)
            tensor([[2.], [7.]])
        """
        return self.mean(self._seq_dim, *args, **kwargs)

    def median_along_seq(self, *args, **kwargs) -> torch.return_types.median:
        r"""Computes the median values along the sequence dimension.

        Args:
        ----
            *args: See the documentation of ``torch.Tensor.median``
            **kwargs: See the documentation of ``torch.Tensor.median``

        Returns:
        -------
            ``torch.return_types.median``:  The first tensor will
                be populated with the median values and the second
                tensor, which must have dtype long, with their indices
                in the sequence dimension of input.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.median_along_seq()
            torch.return_types.median(
            values=tensor([2, 7]),
            indices=tensor([2, 2]))
        """
        return self.median(self._seq_dim, *args, **kwargs)

    def min_along_seq(self, *args, **kwargs) -> torch.return_types.min:
        r"""Computes the minimum values along the sequence dimension.

        Args:
        ----
            *args: See the documentation of ``torch.Tensor.min``
            **kwargs: See the documentation of ``torch.Tensor.min``

        Returns:
        -------
            ``torch.return_types.min``: A batch with
                the minimum values along the sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.min_along_seq()
            torch.return_types.min(
            values=tensor([0, 5]),
            indices=tensor([0, 0]))
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.min_along_seq(keepdim=True)
            torch.return_types.min(
            values=tensor([[0], [5]]),
            indices=tensor([[0], [0]]))
        """
        return self.min(self._seq_dim, *args, **kwargs)

    def nanmean_along_seq(self, *args, **kwargs) -> Tensor:
        r"""Computes the mean values along the sequence dimension.

        Args:
        ----
            *args: See the documentation of ``torch.Tensor.nanmean``
            **kwargs: See the documentation of ``torch.Tensor.nanmean``

        Returns:
        -------
            ``torch.Tensor``: A batch with
                the mean values along the sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
            >>> batch.nanmean_along_seq()
            tensor([2.0000, 6.5000])
            >>> batch.nanmean_along_seq(keepdim=True)
            tensor([[2.0000], [6.5000]])
        """
        return self.nanmean(self._seq_dim, *args, **kwargs)

    def nanmedian_along_seq(self, *args, **kwargs) -> torch.return_types.nanmedian:
        r"""Computes the median values along the sequence dimension.

        Args:
        ----
            *args: See the documentation of ``torch.Tensor.nanmedian``
            **kwargs: See the documentation of ``torch.Tensor.nanmedian``

        Returns:
        -------
            ``torch.return_types.nanmedian``:  The first tensor will
                be populated with the median values and the second
                tensor, which must have dtype long, with their indices
                in the sequence dimension of input.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
            >>> batch.nanmedian_along_seq()
            torch.return_types.nanmedian(
            values=tensor([2., 6.]),
            indices=tensor([2, 1]))
        """
        return self.nanmedian(self._seq_dim, *args, **kwargs)

    def nansum_along_seq(self, *args, **kwargs) -> Tensor:
        r"""Computes the sum values along the sequence dimension.

        Args:
        ----
            *args: See the documentation of ``torch.Tensor.nansum``
            **kwargs: See the documentation of ``torch.Tensor.nansum``

        Returns:
        -------
            ``torch.Tensor``: A tensor with the sum values along the
                sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
            >>> batch.nansum_along_seq()
            tensor([10., 26.])
        """
        return self.nansum(self._seq_dim, *args, **kwargs)

    def prod_along_seq(self, *args, **kwargs) -> Tensor:
        r"""Computes the product values along the sequence dimension.

        Args:
        ----
            *args: See the documentation of ``torch.Tensor.prod``
            **kwargs: See the documentation of ``torch.Tensor.prod``

        Returns:
        -------
            ``torch.Tensor``: A batch with
                the product values along the sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]]))
            >>> batch.prod_along_seq()
            tensor([ 120, 3024])
            >>> batch.prod_along_seq(keepdim=True)
            tensor([[ 120], [3024]])
        """
        return self.prod(self._seq_dim, *args, **kwargs)

    def sum_along_seq(self, *args, **kwargs) -> Tensor:
        r"""Computes the sum values along the sequence dimension.

        Args:
        ----
            *args: See the documentation of ``torch.Tensor.sum``
            **kwargs: See the documentation of ``torch.Tensor.sum``

        Returns:
        -------
            ``torch.Tensor``: A tensor with the sum values along the
                sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.sum_along_seq()
            tensor([10, 35])
        """
        return self.sum(self._seq_dim, *args, **kwargs)

    ##########################################################
    #    Indexing, slicing, joining, mutating operations     #
    ##########################################################

    def align_as(self, other: BatchedTensorSeq) -> BatchedTensorSeq:
        r"""Aligns the current batch with the batch ``other``.

        This method makes sure the batch and sequence dimensions
        are aligned.

        Args:
        ----
            other (``BatchedTensorSeq``): Specifies the batch to use to
                align the current batch.

        Returns:
        -------
            ``BatchedTensorSeq``: The aligned batch.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> # batch-sequence -> sequence-batch
            >>> seq_batch = BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.align_as(seq_batch)
            tensor([[0, 5],
                    [1, 6],
                    [2, 7],
                    [3, 8],
                    [4, 9]], batch_dim=1, seq_dim=0)
            >>> # sequence-batch -> batch-sequence
            >>> batch_seq = BatchedTensorSeq(torch.ones(2, 3))
            >>> batch = BatchedTensorSeq.from_seq_batch(torch.arange(10).view(5, 2))
            >>> batch.align_as(batch_seq)
            tensor([[0, 2, 4, 6, 8],
                    [1, 3, 5, 7, 9]], batch_dim=0, seq_dim=1)
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
        -------
            ``BatchedTensorSeq``: The batch in the batch-sequence
                format.

        Example usage:

        .. code-block:: pycon

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
        -------
            ``BatchedTensorSeq``: The batch in the sequence-batch format.

        Example usage:

        .. code-block:: pycon

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
        ----
            tensors (``BatchedTensorSeq`` or ``torch.Tensor`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Returns:
        -------
            ``BatchedTensorSeq``: A batch with the concatenated data
                along the sequence dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
            >>> batch.cat_along_seq(BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]])))
            tensor([[ 0,  1,  2, 10, 11],
                    [ 4,  5,  6, 12, 13]], batch_dim=0, seq_dim=1)
            >>> batch = BatchedTensorSeq(torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1, seq_dim=0)
            >>> batch.cat_along_seq(
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
        ----
            tensors (``BatchedTensor`` or ``torch.Tensor`` or
                ``Iterable``): Specifies the batch(es) to concatenate.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
            >>> batch.cat_along_seq_(BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]])))
            >>> batch
            tensor([[ 0,  1,  2, 10, 11],
                    [ 4,  5,  6, 12, 13]], batch_dim=0, seq_dim=1)
            >>> batch = BatchedTensorSeq(
            ...     torch.tensor([[0, 4], [1, 5], [2, 6]]),
            ...     batch_dim=1,
            ...     seq_dim=0,
            ... )
            >>> batch.cat_along_seq_(
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
        ----
            chunks (int): Specifies the number of chunks.

        Returns:
        -------
            tuple: The batch split into chunks along the sequence
                dimension.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.chunk_along_seq(chunks=3)
            (tensor([[0, 1], [5, 6]], batch_dim=0, seq_dim=1),
             tensor([[2, 3], [7, 8]], batch_dim=0, seq_dim=1),
             tensor([[4], [9]], batch_dim=0, seq_dim=1))
        """
        return self.chunk(chunks, self._seq_dim)

    def index_select_along_seq(self, index: Tensor | Sequence[int]) -> BatchedTensorSeq:
        r"""Slices the batch along the sequence dimension at the given
        indices.

        Args:
        ----
            index (``torch.Tensor`` of type long and shape
                ``(seq_len,)`` or ``(batch_size, seq_len,)`` or
                ``Sequence``): Specifies the indices to select.

        Returns:
        -------
            ``BatchedTensorSeq``: A new batch sliced along the sequence
                dimension at the given indices.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.index_select_along_seq([2, 4])
            tensor([[2, 4],
                    [7, 9]], batch_dim=0, seq_dim=1)
            >>> batch.index_select_along_seq(torch.tensor([4, 3, 2, 1, 0]))
            tensor([[4, 3, 2, 1, 0],
                    [9, 8, 7, 6, 5]], batch_dim=0, seq_dim=1)
            >>> batch.index_select_along_seq(torch.tensor([[2, 1, 3, 0, 4], [4, 3, 2, 1, 0]]))
            tensor([[2, 1, 3, 0, 4],
                    [9, 8, 7, 6, 5]], batch_dim=0, seq_dim=1)
        """
        index = to_tensor(index)
        if index.ndim == 1:
            return self.index_select(self._seq_dim, index)
        data = self.align_to_batch_seq().data
        seq_len = index.shape[1]
        batch_index = torch.arange(self.batch_size).repeat_interleave(seq_len)
        index = index.flatten()
        return self.__class__(
            data[batch_index, index].view(self.batch_size, seq_len, *data.shape[2:])
        ).align_as(self)

    def repeat_along_seq(self, repeats: int) -> BatchedTensorSeq:
        r"""Repeats the batch along the sequence dimension.

        Args:
        ----
            repeats (int): Specifies the number of times to repeat
                the batch along the sequence dimension.

        Returns:
        -------
            ``BatchedTensorSeq``: A repeated version of the input batch.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.repeat_along_seq(2)
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
        ----
            index (int): Specifies the index to select.

        Returns:
        -------
            ``BatchedTensor``: The batch sliced along the sequence
                dimension at the given index.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.select_along_seq(2)
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

    def take_along_seq(
        self, indices: BaseBatch | np.ndarray | Tensor | Sequence
    ) -> BatchedTensorSeq:
        r"""Takes values along the sequence dimension.

        Args:
        ----
            indices (``BaseBatch`` or ``numpy.ndarray`` or
                ``torch.Tensor`` or `` Specifies the indices to take
                along the batch dimension.

        Returns:
        -------
            ``BatchedTensorSeq``: The sequence with the selected data.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from redcat import BatchedTensorSeq
            >>> batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
            >>> batch.take_along_seq(BatchedTensorSeq(torch.tensor([[3, 0, 1], [2, 3, 4]])))
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

    def _check_valid_dims(self, tensors: Sequence) -> None:
        check_batch_dims(get_batch_dims(tensors))
        check_seq_dims(get_seq_dims(tensors))

    def _get_kwargs(self) -> dict:
        return {"batch_dim": self._batch_dim, "seq_dim": self._seq_dim}


def check_data_and_dims(data: Tensor, batch_dim: int, seq_dim: int) -> None:
    r"""Checks if the tensor ``data``, ``batch_dim`` and ``seq_dim`` are
    correct.

    Args:
    ----
        data (``torch.Tensor``): Specifies the tensor in the batch.
        batch_dim (int): Specifies the batch dimension in the
            ``torch.Tensor`` object.
        seq_dim (int, optional): Specifies the sequence dimension in
            the ``torch.Tensor`` object.

    Raises:
    ------
        RuntimeError: if one of the input is incorrect.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from redcat.tensorseq import check_data_and_dims
        >>> check_data_and_dims(torch.ones(2, 3), batch_dim=0, seq_dim=1)
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


# def implements(torch_function: Callable) -> Callable:
#     """Register a torch function override for BatchedTensor."""
#
#     def decorator(func: Callable) -> Callable:
#         functools.update_wrapper(func, torch_function)
#         HANDLED_FUNCTIONS[torch_function] = func
#         return func
#
#     return decorator


def from_sequences(
    sequences: Iterable[torch.Tensor], padding_value: bool | int | float = 0
) -> BatchedTensorSeq:
    r"""Converts variable length sequences to a single padded tensor.

    Args:
    ----
        sequences (iterable): Specifies an iterable over the variable
            length sequences. Each sequence is a ``torch.Tensor`` of
            shape ``(sequence_length, *)``. This function assumes
            trailing dimensions and type of all the tensors in
            sequences are same.
        padding_value (bool or int or float, optional): Specifies the
        padding value. Default: ``0``

    Returns:
    -------
        ``BatchedTensorSeq``: A padded tensor. The underlying data is
            a ``torch.Tensor`` of shape
            ``(batch_size, sequence_length, *)``.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from redcat.tensorseq import from_sequences
        >>> from_sequences([torch.ones(3), torch.ones(5), torch.ones(1), torch.ones(0)])
        tensor([[1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1.],
                [1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.]], batch_dim=0, seq_dim=1)
    """
    return BatchedTensorSeq(
        pad_sequence(list(sequences), padding_value=padding_value, batch_first=True)
    )
