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


###########################
#     Tests for empty     #
###########################


def test_empty_1d() -> None:
    array = ba.empty(shape=5)
    assert isinstance(array, BatchedArray)
    assert array.shape == (5,)
    assert array.dtype == float
    assert array.batch_axis == 0


def test_empty_2d() -> None:
    array = ba.empty(shape=(2, 3))
    assert isinstance(array, BatchedArray)
    assert array.shape == (2, 3)
    assert array.dtype == float
    assert array.batch_axis == 0


@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_dtype(dtype: DTypeLike) -> None:
    array = ba.empty(shape=(2, 3), dtype=dtype)
    assert isinstance(array, BatchedArray)
    assert array.shape == (2, 3)
    assert array.dtype == dtype
    assert array.batch_axis == 0


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_empty_batch_axis(batch_axis: int) -> None:
    array = ba.empty(shape=(2, 3), batch_axis=batch_axis)
    assert isinstance(array, BatchedArray)
    assert array.shape == (2, 3)
    assert array.dtype == float
    assert array.batch_axis == batch_axis


##########################
#     Tests for full     #
##########################


def test_full_1d() -> None:
    assert ba.full(shape=5, fill_value=42).allequal(BatchedArray(np.array([42, 42, 42, 42, 42])))


def test_full_2d() -> None:
    assert ba.full(shape=(2, 3), fill_value=42).allequal(
        BatchedArray(np.array([[42, 42, 42], [42, 42, 42]]))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_full_dtype(dtype: DTypeLike) -> None:
    assert ba.full(shape=(2, 3), fill_value=42, dtype=dtype).allequal(
        BatchedArray(np.array([[42, 42, 42], [42, 42, 42]], dtype=dtype))
    )


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_full_batch_axis(batch_axis: int) -> None:
    assert ba.full(shape=(2, 3), fill_value=42, batch_axis=batch_axis).allequal(
        BatchedArray(np.array([[42, 42, 42], [42, 42, 42]]), batch_axis=batch_axis)
    )


##########################
#     Tests for ones     #
##########################


def test_ones_1d() -> None:
    assert ba.ones(shape=5).allequal(BatchedArray(np.array([1.0, 1.0, 1.0, 1.0, 1.0])))


def test_ones_2d() -> None:
    assert ba.ones(shape=(2, 3)).allequal(
        BatchedArray(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_ones_dtype(dtype: DTypeLike) -> None:
    assert ba.ones(shape=(2, 3), dtype=dtype).allequal(
        BatchedArray(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=dtype))
    )


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_ones_batch_axis(batch_axis: int) -> None:
    assert ba.ones(shape=(2, 3), batch_axis=batch_axis).allequal(
        BatchedArray(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), batch_axis=batch_axis)
    )


###########################
#     Tests for zeros     #
###########################


def test_zeros_1d() -> None:
    assert ba.zeros(shape=5).allequal(BatchedArray(np.array([0.0, 0.0, 0.0, 0.0, 0.0])))


def test_zeros_2d() -> None:
    assert ba.zeros(shape=(2, 3)).allequal(
        BatchedArray(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_zeros_dtype(dtype: DTypeLike) -> None:
    assert ba.zeros(shape=(2, 3), dtype=dtype).allequal(
        BatchedArray(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=dtype))
    )


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_zeros_batch_axis(batch_axis: int) -> None:
    assert ba.zeros(shape=(2, 3), batch_axis=batch_axis).allequal(
        BatchedArray(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), batch_axis=batch_axis)
    )