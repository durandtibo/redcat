from __future__ import annotations

__all__ = ["BatchedTensor", "check_data_and_dim"]

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


class BatchedTensor:
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
        super().__init__()
        self._data = torch.as_tensor(data, **kwargs)
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
        if kwargs is None:
            kwargs = {}
        batch_dims = {a._batch_dim for a in args if hasattr(a, "_batch_dim")}
        if len(batch_dims) > 1:
            raise RuntimeError(
                f"The batch dimensions do not match. Received multiple values: {batch_dims}"
            )
        args = [a._data if hasattr(a, "_data") else a for a in args]
        return cls(func(*args, **kwargs), batch_dim=batch_dims.pop())

    @property
    def batch_dim(self) -> int:
        r"""int: The batch dimension in the ``torch.Tensor`` object."""
        return self._batch_dim

    @property
    def batch_size(self) -> int:
        r"""int: The batch size."""
        return self._data.shape[self._batch_dim]

    @property
    def data(self) -> Tensor:
        r"""``torch.Tensor``: The data in the batch."""
        return self._data

    @property
    def device(self) -> torch.device:
        r"""``torch.device``: The device where the batch data/tensor is."""
        return self._data.device

    #################################
    #     Conversion operations     #
    #################################

    def contiguous(
        self, memory_format: torch.memory_format = torch.contiguous_format
    ) -> BatchedTensor:
        r"""Creates a batch with a contiguous representation of the data.

        Args:
            memory_format (``torch.memory_format``, optional):
                Specifies the desired memory format.
                Default: ``torch.contiguous_format``

        Returns:
            ``BatchedTensor``: A new batch with a contiguous
                representation of the data.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3)).contiguous()
            >>> batch.data.is_contiguous()
            True
        """
        return BatchedTensor(
            data=self._data.contiguous(memory_format=memory_format), **self._get_kwargs()
        )

    def is_contiguous(self, memory_format: torch.memory_format = torch.contiguous_format) -> bool:
        r"""Indicates if a batch as a contiguous representation of the data.

        Args:
            memory_format (``torch.memory_format``, optional):
                Specifies the desired memory format.
                Default: ``torch.contiguous_format``

        Returns:
            bool: ``True`` if the data are stored with a contiguous
                tensor, otherwise ``False``.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.ones(2, 3)).is_contiguous()
            True
        """
        return self._data.is_contiguous(memory_format=memory_format)

    def to(self, *args, **kwargs) -> BatchedTensor:
        r"""Moves and/or casts the data.

        Args:
            *args: see https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch-tensor-to
            **kwargs: see https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch-tensor-to

        Returns:
            ``BatchedTensor``: A new batch with the data after dtype
                and/or device conversion.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> batch_cuda = batch.to(device=torch.device('cuda:0'))
            >>> batch_bool = batch.to(dtype=torch.bool)
            >>> batch_bool.data
            tensor([[True, True, True],
                    [True, True, True]])
        """
        return BatchedTensor(data=self._data.to(*args, **kwargs), **self._get_kwargs())

    #################################
    #     Comparison operations     #
    #################################

    def equal(self, other: Any) -> bool:
        r"""Indicates if two batches are equal or not.

        Args:
            other: Specifies the value to compare.

        Returns:
            bool: ``True`` if the batches have the same size,
                elements and same batch dimension, ``False`` otherwise.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> BatchedTensor(torch.ones(2, 3)).equal(BatchedTensor(torch.zeros(2, 3)))
            False
        """
        if not isinstance(other, BatchedTensor):
            return False
        if self._batch_dim != other.batch_dim:
            return False
        return self._data.equal(other.data)

    ###################################
    #     Arithmetical operations     #
    ###################################

    def add(
        self,
        other: BatchedTensor | Tensor | int | float,
        alpha: int | float = 1,
    ) -> BatchedTensor:
        r"""Adds the input ``other``, scaled by ``alpha``, to the ``self``
        batch.

        Similar to ``out = self + alpha * other``

        Args:
            other (``BatchedTensor`` or ``torch.Tensor`` or int or
                float): Specifies the other value to add to the
                current batch.
            alpha (int or float, optional): Specifies the scale of the
                batch to add. Default: ``1``

        Returns:
            ``BatchedTensor``: A new batch containing the addition of
                the two batches.

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from redcat import BatchedTensor
            >>> batch = BatchedTensor(torch.ones(2, 3))
            >>> out = batch.add(BatchedTensor(torch.ones(2, 3).mul(2)))
            >>> batch
            tensor([[1., 1., 1.],
                    [1., 1., 1.]], batch_dim=0)
            >>> out
            tensor([[3., 3., 3.],
                    [3., 3., 3.]], batch_dim=0)
        """
        return torch.add(self, other, alpha=alpha)

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
