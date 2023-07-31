from __future__ import annotations

__all__ = ["BatchShufflerIterDataPipe"]

import logging
from collections.abc import Iterator
from typing import TypeVar

from coola.utils.format import str_indent
from torch.utils.data import IterDataPipe

from redcat.base import BaseBatch
from redcat.utils.tensor import get_torch_generator

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BatchShufflerIterDataPipe(IterDataPipe[BaseBatch[T]]):
    r"""Implements a DataPipe to shuffle data in ``BaseBatch`` objects.

    Args:
    ----
        datapipe (``IterDataPipe``): Specifies the source DataPipe.
            The DataPipe has to return ``BaseBatch`` objects.
        random_seed (int, optional): Specifies the random seed used
            to shuffle the data. Default: ``3770589329299158004``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from redcat import BatchedTensor
        >>> from redcat.datapipes.iter import BatchShuffler
        >>> datapipe = BatchShuffler(
        ...     IterableWrapper([BatchedTensor(torch.arange(4).add(i)) for i in range(2)])
        ... )
        >>> list(datapipe)
        [tensor([3, 0, 1, 2], batch_dim=0), tensor([1, 3, 4, 2], batch_dim=0)]
    """

    def __init__(
        self,
        datapipe: IterDataPipe[BaseBatch[T]],
        random_seed: int = 3770589329299158004,
    ) -> None:
        self._datapipe = datapipe
        self._generator = get_torch_generator(random_seed)

    def __iter__(self) -> Iterator[BaseBatch[T]]:
        for batch in self._datapipe:
            yield batch.shuffle_along_batch(self._generator)

    def __len__(self) -> int:
        return len(self._datapipe)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  random_seed={self.random_seed},\n"
            f"  datapipe={str_indent(self._datapipe)},\n)"
        )

    @property
    def random_seed(self) -> int:
        r"""``int``: The random seed used to initialize the pseudo
        random generator."""
        return self._generator.initial_seed()
