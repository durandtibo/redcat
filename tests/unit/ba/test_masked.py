r"""Test the interoperability of ``BatchedArray`` with
``numpy.ma.MaskedArray``.

We want to check that ``BatchedArray`` is compatible with
``numpy.ma.MaskedArray`` so it is possible to combine both.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from coola import objects_are_allclose
from numpy import ma

from redcat.ba import BatchedArray, arrays_share_data
from tests.unit.ba.test_array_compat import BATCH_CLASSES, PAIRWISE_FUNCTIONS


@pytest.fixture
def masked_array() -> ma.MaskedArray:
    return ma.array(data=[[1, 2], [3, 4], [5, 6]], mask=[[0, 1], [1, 0], [0, 0]])


def test_constructor(masked_array: ma.MaskedArray) -> None:
    array = BatchedArray(masked_array)
    assert arrays_share_data(array, masked_array)
    assert array.batch_axis == 0


def test_constructor_batch_axis_1(masked_array: ma.MaskedArray) -> None:
    array = BatchedArray(masked_array, batch_axis=1)
    assert arrays_share_data(array, masked_array)
    assert array.batch_axis == 1


@pytest.mark.parametrize("func", PAIRWISE_FUNCTIONS)
@pytest.mark.parametrize("cls", BATCH_CLASSES)
def test_pairwise(func: Callable, cls: type[np.ndarray]) -> None:
    mask = np.random.choice([0, 1], p=[0.8, 0.2], size=(2, 3, 4))
    array1 = ma.array(data=np.random.rand(2, 3, 4), mask=mask)
    array2 = ma.array(data=np.random.rand(2, 3, 4) + 1.0, mask=mask)
    assert objects_are_allclose(
        func(cls(array1), cls(array2)), cls(func(array1, array2)), equal_nan=True
    )


@pytest.mark.parametrize("func", PAIRWISE_FUNCTIONS)
@pytest.mark.parametrize("cls", BATCH_CLASSES)
def test_pairwise_mixed(func: Callable, cls: type[np.ndarray]) -> None:
    mask = np.random.choice([0, 1], p=[0.8, 0.2], size=(2, 3, 4))
    array1 = ma.array(data=np.random.rand(2, 3, 4), mask=mask)
    array2 = np.random.rand(2, 3, 4) + 1.0
    assert objects_are_allclose(
        func(cls(array1), cls(array2)), cls(func(array1, array2)), equal_nan=True
    )


@pytest.mark.parametrize("func", PAIRWISE_FUNCTIONS)
def test_pairwise_batched_array_custom_axis(func: Callable) -> None:
    mask = np.random.choice([0, 1], p=[0.8, 0.2], size=(2, 3, 4))
    array1 = ma.array(data=np.random.rand(2, 3, 4), mask=mask)
    array2 = ma.array(data=np.random.rand(2, 3, 4) + 1.0, mask=mask)
    assert func(BatchedArray(array1, batch_axis=1), BatchedArray(array2, batch_axis=1)).allclose(
        BatchedArray(func(array1, array2), batch_axis=1), equal_nan=True
    )
