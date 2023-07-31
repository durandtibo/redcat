from __future__ import annotations

__all__ = ["BatchExtenderIterDataPipe", "create_large_batch"]

import logging
from collections.abc import Iterator, Sequence
from typing import TypeVar

from coola.utils.format import str_indent
from torch.utils.data import IterDataPipe

from redcat.base import BaseBatch

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BatchExtenderIterDataPipe(IterDataPipe[BaseBatch[T]]):
    r"""Implements a DataPipe to combine several ``BaseBatch`` object
    into a single ``BaseBatch`` object.

    Args:
    ----
        datapipe (``IterDataPipe``): Specifies the source
            DataPipe. The DataPipe has to return compatible
            ``BaseBatch`` objects.
        buffer_size (int, optional): Specifies the buffer size i.e.
            the number of batches that are combined into a bigger
            batch. Default: ``10``
        drop_last (bool, optional): If ``True``, the last samples are
            dropped if the buffer is not full, otherwise it is
            returned. Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from redcat import BatchedTensor
        >>> from redcat.datapipes.iter import BatchExtender
        >>> datapipe = BatchExtender(
        ...     IterableWrapper([BatchedTensor(torch.ones(2) * i) for i in range(10)]),
        ...     buffer_size=4,
        ... )
        >>> list(datapipe)
        [tensor([0., 0., 1., 1., 2., 2., 3., 3.], batch_dim=0),
         tensor([4., 4., 5., 5., 6., 6., 7., 7.], batch_dim=0),
         tensor([8., 8., 9., 9.], batch_dim=0)]
        >>> datapipe = BatchExtender(
        ...     IterableWrapper([BatchedTensor(torch.ones(2) * i) for i in range(10)]),
        ...     buffer_size=4,
        ...     drop_last=True,
        ... )
        >>> list(datapipe)
        [tensor([0., 0., 1., 1., 2., 2., 3., 3.], batch_dim=0),
         tensor([4., 4., 5., 5., 6., 6., 7., 7.], batch_dim=0)]
    """

    def __init__(
        self,
        datapipe: IterDataPipe[BaseBatch[T]],
        buffer_size: int = 10,
        drop_last: bool = False,
    ) -> None:
        self._datapipe = datapipe
        if buffer_size < 1:
            raise ValueError(
                f"buffer_size should be greater or equal to 1 (received: {buffer_size})"
            )
        self._buffer_size = int(buffer_size)
        self._drop_last = bool(drop_last)

    def __iter__(self) -> Iterator[BaseBatch[T]]:
        buffer: list[BaseBatch[T]] = []
        for batch in self._datapipe:
            buffer.append(batch)
            if len(buffer) == self._buffer_size:
                yield create_large_batch(buffer)
                buffer = []
        if buffer and not self._drop_last:
            yield create_large_batch(buffer)

    def __len__(self) -> int:
        length = len(self._datapipe)
        if self._drop_last:
            return length // self._buffer_size
        return (length + self._buffer_size - 1) // self._buffer_size

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  buffer_size={self._buffer_size:,},\n"
            f"  drop_last={self._drop_last},\n"
            f"  datapipe={str_indent(self._datapipe)},\n)"
        )


def create_large_batch(batches: Sequence[BaseBatch[T]]) -> BaseBatch[T]:
    r"""Creates a large batch from a sequence of batches.

    Args:
        batches: Specifies the sequence of batches.

    Returns:
        ``BaseBatch``: A batch containing all the examples from the input batches.
    """
    batch = batches[0].clone()  # Create a deepcopy to not change the data in the batch
    batch.extend(batches[1:])
    return batch
