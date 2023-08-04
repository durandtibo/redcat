from __future__ import annotations

from typing import Any

import numpy as np
from pytest import mark, raises

from redcat.array import BatchedArray, check_data_and_dim


def test_batched_array_repr() -> None:
    assert repr(BatchedArray(np.arange(3))) == "array([0, 1, 2], batch_dim=0)"


@mark.parametrize(
    "data",
    (
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
    ),
)
def test_batched_array_init_data(data: Any) -> None:
    assert np.array_equal(
        BatchedArray(data).data, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    )


def test_batched_array_init_incorrect_data_dim() -> None:
    with raises(RuntimeError, match=r"data needs at least 1 dimensions \(received: 0\)"):
        BatchedArray(np.array(2))


@mark.parametrize("batch_dim", (-1, 1, 2))
def test_batched_array_init_incorrect_batch_dim(batch_dim: int) -> None:
    with raises(
        RuntimeError, match=r"Incorrect batch_dim \(.*\) but the value should be in \[0, 0\]"
    ):
        BatchedArray(np.ones((2,)), batch_dim=batch_dim)


@mark.parametrize("batch_size", (1, 2))
def test_batched_array_batch_size(batch_size: int) -> None:
    assert BatchedArray(np.arange(batch_size)).batch_size == batch_size


def test_batched_array_data() -> None:
    assert np.array_equal(BatchedArray(np.arange(3)).data, np.array([0, 1, 2]))


def test_batched_array_dtype() -> None:
    assert BatchedArray(np.ones((2, 3))).dtype == float


def test_batched_array_shape() -> None:
    assert BatchedArray(np.ones((2, 3))).shape == (2, 3)


def test_batched_array_dim() -> None:
    assert BatchedArray(np.ones((2, 3))).dim() == 2


def test_batched_array_ndimension() -> None:
    assert BatchedArray(np.ones((2, 3))).ndimension() == 2


def test_batched_array_numel() -> None:
    assert BatchedArray(np.ones((2, 3))).numel() == 6


###############################
#     Creation operations     #
###############################


def test_batched_array_clone() -> None:
    batch = BatchedArray(np.ones((2, 3)))
    clone = batch.clone()
    batch._data += 1
    assert batch.equal(BatchedArray(np.full((2, 3), 2.0)))
    assert clone.equal(BatchedArray(np.ones((2, 3))))


def test_batched_array_clone_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.ones((2, 3)), batch_dim=1)
        .clone()
        .equal(BatchedArray(np.ones((2, 3)), batch_dim=1))
    )


def test_batched_array_copy() -> None:
    batch = BatchedArray(np.ones((2, 3)))
    clone = batch.copy()
    batch._data += 1
    assert batch.equal(BatchedArray(np.full((2, 3), 2.0)))
    assert clone.equal(BatchedArray(np.ones((2, 3))))


def test_batched_array_copy_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.ones((2, 3)), batch_dim=1)
        .copy()
        .equal(BatchedArray(np.ones((2, 3)), batch_dim=1))
    )


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


def test_batched_array_allclose_false_different_batch_dim() -> None:
    assert not BatchedArray(np.ones((2, 3))).allclose(BatchedArray(np.ones((2, 3)), batch_dim=1))


@mark.parametrize(
    "batch,atol",
    (
        (BatchedArray(np.ones((2, 3)) + 0.5), 1),
        (BatchedArray(np.ones((2, 3)) + 0.05), 1e-1),
        (BatchedArray(np.ones((2, 3)) + 5e-3), 1e-2),
    ),
)
def test_batched_array_allclose_true_atol(batch: BatchedArray, atol: float) -> None:
    assert BatchedArray(np.ones((2, 3))).allclose(batch, atol=atol, rtol=0)


@mark.parametrize(
    "batch,rtol",
    (
        (BatchedArray(np.ones((2, 3)) + 0.5), 1),
        (BatchedArray(np.ones((2, 3)) + 0.05), 1e-1),
        (BatchedArray(np.ones((2, 3)) + 5e-3), 1e-2),
    ),
)
def test_batched_array_allclose_true_rtol(batch: BatchedArray, rtol: float) -> None:
    assert BatchedArray(np.ones((2, 3))).allclose(batch, rtol=rtol)


def test_batched_array_equal_true() -> None:
    assert BatchedArray(np.ones((2, 3))).equal(BatchedArray(np.ones((2, 3))))


def test_batched_array_equal_false_different_type() -> None:
    assert not BatchedArray(np.ones((2, 3), dtype=float)).equal(np.ones((2, 3), dtype=int))


def test_batched_array_equal_false_different_data() -> None:
    assert not BatchedArray(np.ones((2, 3))).equal(BatchedArray(np.zeros((2, 3))))


def test_batched_array_equal_false_different_shape() -> None:
    assert not BatchedArray(np.ones((2, 3))).equal(BatchedArray(np.ones((2, 3, 1))))


def test_batched_array_equal_false_different_batch_dim() -> None:
    assert not BatchedArray(np.ones((2, 3)), batch_dim=1).equal(BatchedArray(np.ones((2, 3))))


#################
#     Other     #
#################


def test_batched_array_summary() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5)).summary()
        == "BatchedArray(dtype=int64, shape=(2, 5), batch_dim=0)"
    )


########################################
#     Tests for check_data_and_dim     #
########################################


def test_check_data_and_dim_correct() -> None:
    check_data_and_dim(np.ones((2, 3)), batch_dim=0)
    # will fail if an exception is raised


def test_check_data_and_dim_incorrect_data_dim() -> None:
    with raises(RuntimeError, match=r"data needs at least 1 dimensions \(received: 0\)"):
        check_data_and_dim(np.array(2), batch_dim=0)


@mark.parametrize("batch_dim", (-1, 2, 3))
def test_check_data_and_dim_incorrect_batch_dim(batch_dim: int) -> None:
    with raises(
        RuntimeError, match=r"Incorrect batch_dim \(.*\) but the value should be in \[0, 1\]"
    ):
        check_data_and_dim(np.ones((2, 3)), batch_dim=batch_dim)


###############################
#     Tests for numpy.sum     #
###############################


def test_numpy_sum() -> None:
    assert np.array_equal(np.sum(BatchedArray(np.arange(10).reshape(2, 5))), np.array(45))


def test_numpy_sum_dim_1() -> None:
    assert np.array_equal(
        np.sum(BatchedArray(np.arange(10).reshape(2, 5)), axis=1), np.array([10, 35])
    )


def test_numpy_sum_keepdim() -> None:
    assert np.array_equal(
        np.sum(BatchedArray(np.arange(10).reshape(2, 5)), axis=1, keepdims=True),
        np.array([[10], [35]]),
    )
