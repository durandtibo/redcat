from __future__ import annotations

import numpy as np
import torch
from pytest import mark, raises

from redcat import BatchedArray, BatchedTensor, BatchedTensorSeq
from redcat.utils.common import (
    check_batch_dims,
    check_data_and_dim,
    get_batch_dims,
    swap2,
)

######################################
#     Tests for check_batch_dims     #
######################################


def test_check_batch_dims_correct() -> None:
    check_batch_dims({0})


def test_check_batch_dims_incorrect() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        check_batch_dims({0, 1})


########################################
#     Tests for check_data_and_dim     #
########################################


@mark.parametrize("array", (np.ones((2, 3)), torch.ones(2, 3)))
def test_check_data_and_dim_correct(array: np.ndarray | torch.Tensor) -> None:
    check_data_and_dim(array, batch_dim=0)
    # will fail if an exception is raised


@mark.parametrize("array", (np.array(2), torch.tensor(2)))
def test_check_data_and_dim_incorrect_data_dim(array: np.ndarray | torch.Tensor) -> None:
    with raises(RuntimeError, match=r"data needs at least 1 dimensions \(received: 0\)"):
        check_data_and_dim(np.array(2), batch_dim=0)


@mark.parametrize("array", (np.ones((2, 3)), torch.ones(2, 3)))
@mark.parametrize("batch_dim", (-1, 2, 3))
def test_check_data_and_dim_incorrect_batch_dim(
    array: np.ndarray | torch.Tensor, batch_dim: int
) -> None:
    with raises(
        RuntimeError, match=r"Incorrect batch_dim \(.*\) but the value should be in \[0, 1\]"
    ):
        check_data_and_dim(np.ones((2, 3)), batch_dim=batch_dim)


####################################
#     Tests for get_batch_dims     #
####################################


def test_get_batch_dims_1_array() -> None:
    assert get_batch_dims(
        (BatchedArray(np.ones((2, 3))), BatchedArray(np.ones((2, 3)))),
        {"val": BatchedArray(np.ones((2, 3)))},
    ) == {0}


def test_get_batch_dims_2_array() -> None:
    assert get_batch_dims(
        (BatchedArray(np.ones((2, 3))), BatchedArray(np.ones((2, 3)), batch_dim=1)),
        {"val": BatchedArray(np.ones((2, 3)))},
    ) == {0, 1}


def test_get_batch_dims_1_tensor() -> None:
    assert get_batch_dims(
        (BatchedTensor(torch.ones(2, 3)), BatchedTensor(torch.ones(2, 3))),
        {"val": BatchedTensorSeq(torch.ones(2, 3))},
    ) == {0}


def test_get_batch_dims_2_tensor() -> None:
    assert get_batch_dims(
        (BatchedTensor(torch.ones(2, 3)), BatchedTensor(torch.ones(2, 3), batch_dim=1)),
        {"val": BatchedTensorSeq(torch.ones(2, 3))},
    ) == {0, 1}


def test_get_batch_dims_empty() -> None:
    assert get_batch_dims(tuple()) == set()


###########################
#     Tests for swap2     #
###########################


def test_swap2_list() -> None:
    seq = [1, 2, 3, 4, 5]
    swap2(seq, 0, 2)
    assert seq == [3, 2, 1, 4, 5]


def test_swap2_tensor_1d() -> None:
    tensor = torch.tensor([1, 2, 3, 4, 5])
    swap2(tensor, 0, 2)
    assert tensor.equal(torch.tensor([3, 2, 1, 4, 5]))


def test_swap2_tensor_2d() -> None:
    tensor = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    swap2(tensor, 0, 2)
    assert tensor.equal(torch.tensor([[4, 5], [2, 3], [0, 1], [6, 7], [8, 9]]))


def test_swap2_ndarray_1d() -> None:
    array = np.array([1, 2, 3, 4, 5])
    swap2(array, 0, 2)
    assert np.array_equal(array, np.array([3, 2, 1, 4, 5]))


def test_swap2_ndarray_2d() -> None:
    array = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    swap2(array, 0, 2)
    assert np.array_equal(array, np.array([[4, 5], [2, 3], [0, 1], [6, 7], [8, 9]]))
