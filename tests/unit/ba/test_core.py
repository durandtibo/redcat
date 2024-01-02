from __future__ import annotations

from typing import Any

import numpy as np
import pytest

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
def test_batched_array_explicit_constructor_call(data: Any) -> None:
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
