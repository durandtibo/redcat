from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike, DTypeLike

from redcat import ba
from redcat.ba import BatchedArray

DTYPES = (bool, int, float)


def test_batched_array_repr() -> None:
    assert repr(BatchedArray(np.arange(3))) == "array([0, 1, 2], batch_axis=0)"


def test_batched_array_str() -> None:
    assert str(BatchedArray(np.arange(3))) == "[0 1 2]\nwith batch_axis=0"


@pytest.mark.parametrize(
    "data",
    [
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
    ],
)
def test_batched_array_explicit_constructor_call(data: ArrayLike) -> None:
    array = BatchedArray(data)
    assert np.array_equal(array, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float))
    assert array.batch_axis == 0


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_batched_array_explicit_constructor_call_batch_axis(batch_axis: int) -> None:
    array = BatchedArray(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float), batch_axis)
    assert np.array_equal(array, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float))
    assert array.batch_axis == batch_axis


def test_batched_array_view_casting() -> None:
    array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    barray = array.view(BatchedArray)
    assert barray.allequal(BatchedArray(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)))


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_batched_array_new_from_template(batch_axis: int) -> None:
    array = BatchedArray(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float), batch_axis)
    assert array[1:].allequal(
        BatchedArray(np.array([[3.0, 4.0], [5.0, 6.0]], dtype=float), batch_axis)
    )


def test_batched_array_init_incorrect_data_axis() -> None:
    with pytest.raises(RuntimeError, match=r"data needs at least 1 axis \(received: 0\)"):
        BatchedArray(np.array(2))


@pytest.mark.parametrize("batch_size", (1, 2))
def test_batched_array_batch_size(batch_size: int) -> None:
    assert BatchedArray(np.arange(batch_size)).batch_size == batch_size


def test_batched_array_dtype() -> None:
    assert BatchedArray(np.ones((2, 3))).dtype == float


def test_batched_array_ndim() -> None:
    assert BatchedArray(np.ones((2, 3))).ndim == 2


def test_batched_array_shape() -> None:
    assert BatchedArray(np.ones((2, 3))).shape == (2, 3)


def test_batched_array_size() -> None:
    assert BatchedArray(np.ones((2, 3))).size == 6


#################################
#     Conversion operations     #
#################################


def test_batched_array_astype() -> None:
    assert (
        BatchedArray(np.ones(shape=(2, 3)))
        .astype(bool)
        .allequal(BatchedArray(np.ones(shape=(2, 3), dtype=bool)))
    )


def test_batched_array_astype_custom_axis() -> None:
    assert (
        BatchedArray(np.ones(shape=(2, 3)), batch_axis=1)
        .astype(bool)
        .allequal(BatchedArray(np.ones(shape=(2, 3), dtype=bool), batch_axis=1))
    )


###############################
#     Creation operations     #
###############################


def test_batched_array_copy() -> None:
    batch = ba.ones(shape=(2, 3))
    clone = batch.copy()
    # batch.add_(1)
    batch += 1
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=2.0))
    assert clone.allequal(ba.ones(shape=(2, 3)))


def test_batched_array_copy_custom_batch_axis() -> None:
    assert ba.ones(shape=(2, 3), batch_axis=1).copy().allequal(ba.ones(shape=(2, 3), batch_axis=1))


#################################
#     Comparison operations     #
#################################


def test_batched_array_allclose_true() -> None:
    assert BatchedArray(np.ones((2, 3))).allclose(BatchedArray(np.ones((2, 3))))


def test_batched_array_allclose_false_different_type() -> None:
    assert not BatchedArray(np.ones((2, 3), dtype=float)).allclose(np.zeros((2, 3), dtype=int))


def test_batched_array_allclose_false_different_data() -> None:
    assert not BatchedArray(np.ones((2, 3))).allclose(BatchedArray(np.zeros((2, 3))))


def test_batched_array_allclose_false_different_shape() -> None:
    assert not BatchedArray(np.ones((2, 3))).allclose(BatchedArray(np.ones((2, 3, 1))))


def test_batched_array_allclose_false_different_batch_axis() -> None:
    assert not BatchedArray(np.ones((2, 3))).allclose(BatchedArray(np.ones((2, 3)), batch_axis=1))


def test_batched_array_allequal_true() -> None:
    assert BatchedArray(np.ones((2, 3))).allequal(BatchedArray(np.ones((2, 3))))


def test_batched_array_allequal_false_different_type() -> None:
    assert not BatchedArray(np.ones((2, 3), dtype=float)).allequal(np.ones((2, 3), dtype=int))


def test_batched_array_allequal_false_different_data() -> None:
    assert not BatchedArray(np.ones((2, 3))).allequal(BatchedArray(np.zeros((2, 3))))


def test_batched_array_allequal_false_different_shape() -> None:
    assert not BatchedArray(np.ones((2, 3))).allequal(BatchedArray(np.ones((2, 3, 1))))


def test_batched_array_allequal_false_different_batch_axis() -> None:
    assert not BatchedArray(np.ones((2, 3)), batch_axis=1).allequal(BatchedArray(np.ones((2, 3))))


##################################################
#     Mathematical | arithmetical operations     #
##################################################


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_add(other: np.ndarray | int | float) -> None:
    assert ba.ones((2, 3)).add(other).allequal(ba.full(shape=(2, 3), fill_value=3.0))


@pytest.mark.parametrize("dtype", [float, int])
def test_batched_array_add_alpha_2(dtype: DTypeLike) -> None:
    assert (
        ba.ones(shape=(2, 3), dtype=dtype)
        .add(ba.full(shape=(2, 3), fill_value=2.0, dtype=dtype), alpha=2)
        .allequal(ba.full(shape=(2, 3), fill_value=5.0, dtype=dtype))
    )


def test_batched_array_add_batch_axis_1() -> None:
    assert (
        ba.ones(shape=(2, 3), batch_axis=1)
        .add(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
        .allequal(ba.full(shape=(2, 3), fill_value=3.0, batch_axis=1))
    )


def test_batched_array_add_incorrect_batch_axis() -> None:
    batch = ba.ones(shape=(2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.add(ba.ones(shape=(2, 2), batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_add_(other: np.ndarray | int | float) -> None:
    batch = ba.ones(shape=(2, 3))
    batch.add_(other)
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=3.0))


@pytest.mark.parametrize("dtype", [float, int])
def test_batched_array_add__alpha_2(dtype: DTypeLike) -> None:
    batch = ba.ones(shape=(2, 3), dtype=int)
    batch.add_(ba.full(shape=(2, 3), fill_value=2.0, dtype=int), alpha=2)
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=5.0, dtype=int))


def test_batched_array_add__custom_batch_axis() -> None:
    batch = ba.ones(shape=(2, 3), batch_axis=1)
    batch.add_(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=3.0, batch_axis=1))


def test_batched_array_add__incorrect_batch_axis() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.add_(ba.ones(shape=(2, 2), batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_sub(other: np.ndarray | int | float) -> None:
    assert ba.ones((2, 3)).sub(other).allequal(ba.full(shape=(2, 3), fill_value=-1.0))


@pytest.mark.parametrize("dtype", [float, int])
def test_batched_array_sub_alpha_2(dtype: DTypeLike) -> None:
    assert (
        ba.ones(shape=(2, 3), dtype=int)
        .sub(ba.full(shape=(2, 3), fill_value=2, dtype=int), alpha=2)
        .allequal(ba.full(shape=(2, 3), fill_value=-3, dtype=int))
    )


def test_batched_array_sub_custom_batch_axiss() -> None:
    assert (
        ba.ones(shape=(2, 3), batch_axis=1)
        .sub(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
        .allequal(ba.full(shape=(2, 3), fill_value=-1.0, batch_axis=1))
    )


def test_batched_array_sub_incorrect_batch_axis() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.sub(ba.ones(shape=(2, 2), batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_sub_(other: np.ndarray | int | float) -> None:
    batch = ba.ones((2, 3))
    batch.sub_(other)
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=-1.0))


@pytest.mark.parametrize("dtype", [float, int])
def test_batched_array_sub__alpha_2(dtype: DTypeLike) -> None:
    batch = ba.ones(shape=(2, 3), dtype=dtype)
    batch.sub_(ba.full(shape=(2, 3), fill_value=2, dtype=dtype), alpha=2)
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=-3, dtype=dtype))


def test_batched_array_sub__custom_batch_axis() -> None:
    batch = ba.ones(shape=(2, 3), batch_axis=1)
    batch.sub_(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=-1.0, batch_axis=1))


def test_batched_array_sub__incorrect_batch_axis() -> None:
    batch = ba.ones(shape=(2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.sub_(ba.ones(shape=(2, 2), batch_axis=1))
