import numpy as np
import pytest
from coola import objects_are_equal
from numpy.typing import DTypeLike

from redcat import ba2 as ba
from redcat.ba2 import BatchedArray

DTYPES = (bool, int, float)


###########################
#     Tests for array     #
###########################


def test_batched_array() -> None:
    x = np.arange(10).reshape(2, 5)
    y = ba.batched_array(x)
    assert y.data is not x
    assert objects_are_equal(y, BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])))


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_array_dtype(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        ba.batched_array(np.arange(10).reshape(2, 5), dtype=dtype),
        BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=dtype)),
    )


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_batched_array_batch_axis(batch_axis: int) -> None:
    assert objects_are_equal(
        ba.batched_array(np.arange(10).reshape(2, 5), batch_axis=batch_axis),
        BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), batch_axis=batch_axis),
    )


def test_batched_array_copy_false() -> None:
    x = np.arange(10).reshape(2, 5)
    y = ba.batched_array(x, copy=False)
    assert y.data is x
    assert objects_are_equal(y, BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])))


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


################################
#     Tests for empty_like     #
################################


def test_empty_like_1d() -> None:
    array = ba.empty_like(ba.ones(5))
    assert isinstance(array, BatchedArray)
    assert array.shape == (5,)
    assert array.dtype == float
    assert array.batch_axis == 0


def test_empty_like_2d() -> None:
    array = ba.empty_like(ba.ones(shape=(2, 3)))
    assert isinstance(array, BatchedArray)
    assert array.shape == (2, 3)
    assert array.dtype == float
    assert array.batch_axis == 0


@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_like_dtype(dtype: DTypeLike) -> None:
    array = ba.empty_like(ba.ones(shape=(2, 3)), dtype=dtype)
    assert isinstance(array, BatchedArray)
    assert array.shape == (2, 3)
    assert array.dtype == dtype
    assert array.batch_axis == 0


@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_like_array_dtype(dtype: DTypeLike) -> None:
    array = ba.empty_like(ba.ones(shape=(2, 3), dtype=dtype))
    assert isinstance(array, BatchedArray)
    assert array.shape == (2, 3)
    assert array.dtype == dtype
    assert array.batch_axis == 0


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_empty_like_batch_axis(batch_axis: int) -> None:
    array = ba.empty_like(ba.ones(shape=(2, 3), batch_axis=batch_axis))
    assert isinstance(array, BatchedArray)
    assert array.shape == (2, 3)
    assert array.dtype == float
    assert array.batch_axis == batch_axis


##########################
#     Tests for full     #
##########################


def test_full_1d() -> None:
    assert objects_are_equal(
        ba.full(shape=5, fill_value=42), BatchedArray(np.array([42, 42, 42, 42, 42]))
    )


def test_full_2d() -> None:
    assert objects_are_equal(
        ba.full(shape=(2, 3), fill_value=42), BatchedArray(np.array([[42, 42, 42], [42, 42, 42]]))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_full_dtype(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        ba.full(shape=(2, 3), fill_value=42, dtype=dtype),
        BatchedArray(np.array([[42, 42, 42], [42, 42, 42]], dtype=dtype)),
    )


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_full_batch_axis(batch_axis: int) -> None:
    assert objects_are_equal(
        ba.full(shape=(2, 3), fill_value=42, batch_axis=batch_axis),
        BatchedArray(np.array([[42, 42, 42], [42, 42, 42]]), batch_axis=batch_axis),
    )


###############################
#     Tests for full_like     #
###############################


def test_full_like_1d() -> None:
    assert objects_are_equal(
        ba.full_like(ba.zeros(shape=5), fill_value=42.0), ba.full(shape=5, fill_value=42.0)
    )


def test_full_like_2d() -> None:
    assert objects_are_equal(
        ba.full_like(ba.zeros(shape=(2, 3)), fill_value=42.0),
        ba.full(shape=(2, 3), fill_value=42.0),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_full_like_dtype(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        ba.full_like(ba.zeros(shape=(2, 3)), fill_value=42.0, dtype=dtype),
        ba.full(shape=(2, 3), fill_value=42.0, dtype=dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_full_like_array_dtype(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        ba.full_like(ba.zeros(shape=(2, 3), dtype=dtype), fill_value=42.0),
        ba.full(shape=(2, 3), fill_value=42.0, dtype=dtype),
    )


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_full_like_batch_axis(batch_axis: int) -> None:
    assert objects_are_equal(
        ba.full_like(ba.zeros(shape=(2, 3), batch_axis=batch_axis), fill_value=42.0),
        ba.full(shape=(2, 3), fill_value=42.0, batch_axis=batch_axis),
    )


##########################
#     Tests for ones     #
##########################


def test_ones_1d() -> None:
    assert objects_are_equal(ba.ones(shape=5), BatchedArray(np.array([1.0, 1.0, 1.0, 1.0, 1.0])))


def test_ones_2d() -> None:
    assert objects_are_equal(
        ba.ones(shape=(2, 3)), BatchedArray(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_ones_dtype(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        ba.ones(shape=(2, 3), dtype=dtype),
        BatchedArray(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=dtype)),
    )


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_ones_batch_axis(batch_axis: int) -> None:
    assert objects_are_equal(
        ba.ones(shape=(2, 3), batch_axis=batch_axis),
        BatchedArray(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), batch_axis=batch_axis),
    )


###############################
#     Tests for ones_like     #
###############################


def test_ones_like_1d() -> None:
    assert objects_are_equal(ba.ones_like(ba.zeros(shape=5)), ba.ones(shape=5))


def test_ones_like_2d() -> None:
    assert objects_are_equal(ba.ones_like(ba.zeros(shape=(2, 3))), ba.ones(shape=(2, 3)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_ones_like_dtype(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        ba.ones_like(ba.zeros(shape=(2, 3)), dtype=dtype), ba.ones(shape=(2, 3), dtype=dtype)
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_ones_like_array_dtype(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        ba.ones_like(ba.zeros(shape=(2, 3), dtype=dtype)), ba.ones(shape=(2, 3), dtype=dtype)
    )


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_ones_like_batch_axis(batch_axis: int) -> None:
    assert objects_are_equal(
        ba.ones_like(ba.zeros(shape=(2, 3), batch_axis=batch_axis)),
        ba.ones(shape=(2, 3), batch_axis=batch_axis),
    )


###########################
#     Tests for zeros     #
###########################


def test_zeros_1d() -> None:
    assert objects_are_equal(ba.zeros(shape=5), BatchedArray(np.array([0.0, 0.0, 0.0, 0.0, 0.0])))


def test_zeros_2d() -> None:
    assert objects_are_equal(
        ba.zeros(shape=(2, 3)), BatchedArray(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_zeros_dtype(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        ba.zeros(shape=(2, 3), dtype=dtype),
        BatchedArray(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=dtype)),
    )


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_zeros_batch_axis(batch_axis: int) -> None:
    assert objects_are_equal(
        ba.zeros(shape=(2, 3), batch_axis=batch_axis),
        BatchedArray(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), batch_axis=batch_axis),
    )


################################
#     Tests for zeros_like     #
################################


def test_zeros_like_1d() -> None:
    assert objects_are_equal(ba.zeros_like(ba.ones(shape=5)), ba.zeros(shape=5))


def test_zeros_like_2d() -> None:
    assert objects_are_equal(ba.zeros_like(ba.ones(shape=(2, 3))), ba.zeros(shape=(2, 3)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_zeros_like_dtype(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        ba.zeros_like(ba.ones(shape=(2, 3)), dtype=dtype), ba.zeros(shape=(2, 3), dtype=dtype)
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_zeros_like_array_dtype(dtype: DTypeLike) -> None:
    assert objects_are_equal(
        ba.zeros_like(ba.ones(shape=(2, 3), dtype=dtype)), ba.zeros(shape=(2, 3), dtype=dtype)
    )


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_zeros_like_batch_axis(batch_axis: int) -> None:
    assert objects_are_equal(
        ba.zeros_like(ba.ones(shape=(2, 3), batch_axis=batch_axis)),
        ba.zeros(shape=(2, 3), batch_axis=batch_axis),
    )
