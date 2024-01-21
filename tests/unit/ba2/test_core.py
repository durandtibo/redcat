from __future__ import annotations

import numpy as np
import pytest

from redcat import ba2 as ba
from redcat.ba2 import BatchedArray

DTYPES = (bool, int, float)
NUMERIC_DTYPES = [np.float64, np.int64]


#######################
#     Constructor     #
#######################


def test_batched_array_init() -> None:
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    array = BatchedArray(x)
    assert array.data is x
    assert np.array_equal(array.data, x)
    assert array.batch_axis == 0


def test_batched_array_init_incorrect_data_axis() -> None:
    with pytest.raises(RuntimeError, match=r"data needs at least 1 axis \(received: 0\)"):
        BatchedArray(np.array(2))


def test_batched_array_init_no_check() -> None:
    BatchedArray(np.array(2), check=False)


################################
#     Core functionalities     #
################################


@pytest.mark.parametrize("batch_size", (1, 2))
def test_batched_array_batch_size(batch_size: int) -> None:
    assert BatchedArray(np.arange(batch_size)).batch_size == batch_size


def test_batched_array_data() -> None:
    data = np.ones((2, 3))
    assert BatchedArray(data).data is data


def test_batched_array_allclose_true() -> None:
    assert ba.ones(shape=(2, 3)).allclose(ba.ones(shape=(2, 3)))


def test_batched_array_allclose_false_different_type() -> None:
    assert not BatchedArray(np.ones((2, 3), dtype=float)).allclose(np.zeros((2, 3), dtype=int))


def test_batched_array_allclose_false_different_data() -> None:
    assert not ba.ones(shape=(2, 3)).allclose(BatchedArray(np.zeros((2, 3))))


def test_batched_array_allclose_false_different_shape() -> None:
    assert not ba.ones(shape=(2, 3)).allclose(BatchedArray(np.ones((2, 3, 1))))


def test_batched_array_allclose_false_different_axes() -> None:
    assert not ba.ones(shape=(2, 3)).allclose(BatchedArray(np.ones((2, 3)), batch_axis=1))


@pytest.mark.parametrize(
    ("array", "atol"),
    (
        (ba.full(shape=(2, 3), fill_value=1.5), 1),
        (ba.full(shape=(2, 3), fill_value=1.05), 1e-1),
        (ba.full(shape=(2, 3), fill_value=1.005), 1e-2),
    ),
)
def test_batched_array_allclose_true_atol(array: BatchedArray, atol: float) -> None:
    assert ba.ones((2, 3)).allclose(array, atol=atol, rtol=0)


@pytest.mark.parametrize(
    ("array", "rtol"),
    (
        (ba.full(shape=(2, 3), fill_value=1.5), 1),
        (ba.full(shape=(2, 3), fill_value=1.05), 1e-1),
        (ba.full(shape=(2, 3), fill_value=1.005), 1e-2),
    ),
)
def test_batched_array_allclose_true_rtol(array: BatchedArray, rtol: float) -> None:
    assert ba.ones((2, 3)).allclose(array, rtol=rtol)


def test_batched_array_allequal_true() -> None:
    assert ba.ones(shape=(2, 3)).allequal(ba.ones(shape=(2, 3)))


def test_batched_array_allequal_false_different_type() -> None:
    assert not ba.ones(shape=(2, 3)).allequal(np.ones(shape=(2, 3)))


def test_batched_array_allequal_false_different_data() -> None:
    assert not ba.ones(shape=(2, 3)).allequal(ba.zeros(shape=(2, 3)))


def test_batched_array_allequal_false_different_shape() -> None:
    assert not ba.ones(shape=(2, 3)).allequal(ba.ones(shape=(2, 3, 1)))


def test_batched_array_allequal_false_different_axes() -> None:
    assert not ba.ones(shape=(2, 3), batch_axis=1).allequal(ba.ones(shape=(2, 3)))


def test_batched_array_allequal_equal_nan_false() -> None:
    assert not BatchedArray(np.array([1, np.nan, 3])).allequal(
        BatchedArray(np.array([1, np.nan, 3]))
    )


def test_batched_array_allequal_equal_nan_true() -> None:
    assert BatchedArray(np.array([1, np.nan, 3])).allequal(
        BatchedArray(np.array([1, np.nan, 3])), equal_nan=True
    )


######################################
#     Additional functionalities     #
######################################


def test_batched_array_repr() -> None:
    assert repr(BatchedArray(np.arange(3))) == "array([0, 1, 2], batch_axis=0)"


def test_batched_array_str() -> None:
    assert str(BatchedArray(np.arange(3))) == "[0 1 2]\nwith batch_axis=0"


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_batched_array_batch_axis(batch_axis: int) -> None:
    assert ba.ones(shape=(2, 3), batch_axis=batch_axis).batch_axis == batch_axis


#########################
#     Memory layout     #
#########################


def test_batched_array_shape() -> None:
    assert ba.ones(shape=(2, 3)).shape == (2, 3)


#####################
#     Data type     #
#####################


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_array_dtype(dtype: np.dtype) -> None:
    assert ba.ones(shape=(2, 3), dtype=dtype).dtype == dtype
