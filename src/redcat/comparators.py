r"""Contain some comparators to use ``BaseBatch`` objects with
``coola.objects_are_equal`` and ``coola.objects_are_allclose``."""

from __future__ import annotations

__all__ = ["BatchEqualityComparator", "BatchEqualHandler"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.comparators import BaseEqualityComparator
from coola.equality.handlers import (
    BaseEqualityHandler,
    SameObjectHandler,
    SameTypeHandler,
)
from coola.equality.testers import EqualityTester

from redcat.base import BaseBatch

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)


class BatchEqualHandler(BaseEqualityHandler):
    r"""Check if the two batches are equal or not.

    This handler returns ``True`` if the two batches are equal,
    otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    Example usage:

    ```pycon
    >>> import torch
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.testers import EqualityTester
    >>> from redcat.comparators import BatchEqualHandler
    >>> from redcat import BatchList
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = BatchEqualHandler()
    >>> handler.handle(BatchList([1, 2, 3]), BatchList([1, 2, 3]), config)
    True
    >>> handler.handle(BatchList([1, 2, 3]), BatchList([1, 2, 4]), config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        object1: BaseBatch,
        object2: BaseBatch,
        config: EqualityConfig,
    ) -> bool:
        object_equal = batch_equal(object1, object2, config)
        if config.show_difference and not object_equal:
            logger.info(f"batches are not equal:\nobject1:\n{object1}\nobject2:\n{object2}")
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


class BatchEqualityComparator(BaseEqualityComparator[BaseBatch]):
    r"""Implement an equality comparator for ``BaseBatch`` objects."""

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(BatchEqualHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> BatchEqualityComparator:
        return self.__class__()

    def equal(self, object1: BaseBatch, object2: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(object1=object1, object2=object2, config=config)


def batch_equal(batch1: BaseBatch, batch2: BaseBatch, config: EqualityConfig) -> bool:
    r"""Indicate if the two batches are equal within a tolerance.

    Args:
        batch1: Specifies the first batch to compare.
        batch2: Specifies the second batch to compare.
        config: Specifies the equality configuration.

    Returns:
        ``True``if the two batches are equal within a tolerance,
            otherwise ``False``.
    """
    if config.equal_nan or config.atol > 0 or config.rtol > 0:
        return batch1.allclose(
            batch2, atol=config.atol, rtol=config.rtol, equal_nan=config.equal_nan
        )
    return batch1.allequal(batch2)


if not EqualityTester.has_comparator(BaseBatch):  # pragma: no cover
    EqualityTester.add_comparator(BaseBatch, BatchEqualityComparator())
