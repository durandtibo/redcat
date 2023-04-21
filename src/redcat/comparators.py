r"""This module implements some comparators to use ``BaseBatch`` objects with
``coola.objects_are_equal`` and ``coola.objects_are_allclose``."""

__all__ = ["BatchEqualityOperator", "BatchAllCloseOperator"]

import logging
from typing import Any

from coola import (
    AllCloseTester,
    BaseAllCloseOperator,
    BaseAllCloseTester,
    BaseEqualityOperator,
    BaseEqualityTester,
    EqualityTester,
)

from redcat.base import BaseBatch

logger = logging.getLogger(__name__)


class BatchEqualityOperator(BaseEqualityOperator[BaseBatch]):
    r"""Implements an equality operator for ``BaseBatch`` objects."""

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: BaseBatch,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if not isinstance(object2, BaseBatch):
            if show_difference:
                logger.info(f"object2 is not a `BaseBatch` object: {type(object2)}")
            return False
        object_equal = object1.equal(object2)
        if show_difference and not object_equal:
            logger.info(
                f"`BaseBatch` objects are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


class BatchAllCloseOperator(BaseAllCloseOperator[BaseBatch]):
    r"""Implements an allclose operator for ``BaseBatch`` objects."""

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: BaseBatch,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if not isinstance(object2, BaseBatch):
            if show_difference:
                logger.info(f"object2 is not a `BaseBatch` object: {type(object2)}")
            return False
        object_equal = object1.allclose(object2, rtol=rtol, atol=atol, equal_nan=equal_nan)
        if show_difference and not object_equal:
            logger.info(
                f"`BaseBatch` objects are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


if not AllCloseTester.has_allclose_operator(BaseBatch):
    AllCloseTester.add_allclose_operator(BaseBatch, BatchAllCloseOperator())  # pragma: no cover
if not EqualityTester.has_equality_operator(BaseBatch):
    EqualityTester.add_equality_operator(BaseBatch, BatchEqualityOperator())  # pragma: no cover
