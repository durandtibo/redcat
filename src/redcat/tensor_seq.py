from __future__ import annotations

__all__ = ["BatchedTensorSeq", "check_data_and_dims"]

from collections.abc import Callable
from typing import Any

import torch

from redcat.base import BaseBatchedTensor


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
        batch_dims = {a._batch_dim for a in args if hasattr(a, "_batch_dim")}
        if len(batch_dims) > 1:
            raise RuntimeError(
                f"The batch dimensions do not match. Received multiple values: {batch_dims}"
            )
        seq_dims = {a._seq_dim for a in args if hasattr(a, "_seq_dim")}
        if len(seq_dims) > 1:
            raise RuntimeError(
                f"The sequence dimensions do not match. Received multiple values: {seq_dims}"
            )
        args = [a._data if hasattr(a, "_data") else a for a in args]
        return cls(
            func(*args, **(kwargs or {})), batch_dim=batch_dims.pop(), seq_dim=seq_dims.pop()
        )

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