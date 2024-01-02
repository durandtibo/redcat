import numpy as np
import pytest
from numpy.typing import DTypeLike

from redcat import ba
from redcat.ba import BatchedArray

DTYPES = (bool, int, float)

###########################
#     Tests for array     #
###########################


def test_array() -> None:
    assert ba.array(np.arange(10).reshape(2, 5)).allequal(
        BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_array_dtype(dtype: DTypeLike) -> None:
    assert ba.array(np.arange(10).reshape(2, 5), dtype=dtype).allequal(
        BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype))
    )


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_array_batch_axis(batch_axis: int) -> None:
    assert ba.array(np.arange(10).reshape(2, 5), batch_axis=batch_axis).allequal(
        BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), batch_axis=batch_axis)
    )


def test_array_copy() -> None:
    array = np.arange(10).reshape(2, 5)
    barray = ba.array(array, copy=False)
    assert barray.allequal(BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])))


##########################
#     Tests for ones     #
##########################


def test_ones_1d() -> None:
    assert ba.ones(5).allequal(BatchedArray(np.array([1.0, 1.0, 1.0, 1.0, 1.0])))


def test_ones_2d() -> None:
    assert ba.ones((2, 3)).allequal(BatchedArray(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])))


@pytest.mark.parametrize("dtype", DTYPES)
def test_ones_dtype(dtype: DTypeLike) -> None:
    assert ba.ones((2, 3), dtype=dtype).allequal(
        BatchedArray(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=dtype))
    )


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_ones_batch_axis(batch_axis: int) -> None:
    assert ba.ones((2, 3), batch_axis=batch_axis).allequal(
        BatchedArray(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), batch_axis=batch_axis)
    )


###########################
#     Tests for zeros     #
###########################


def test_zeros_1d() -> None:
    assert ba.zeros(5).allequal(BatchedArray(np.array([0.0, 0.0, 0.0, 0.0, 0.0])))


def test_zeros_2d() -> None:
    assert ba.zeros((2, 3)).allequal(BatchedArray(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])))


@pytest.mark.parametrize("dtype", DTYPES)
def test_zeros_dtype(dtype: DTypeLike) -> None:
    assert ba.zeros((2, 3), dtype=dtype).allequal(
        BatchedArray(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=dtype))
    )


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_zeros_batch_axis(batch_axis: int) -> None:
    assert ba.zeros((2, 3), batch_axis=batch_axis).allequal(
        BatchedArray(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), batch_axis=batch_axis)
    )
