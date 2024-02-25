from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from redcat import BatchedTensor, BatchedTensorSeq
from redcat.ba import BatchedArray
from redcat.utils.common import (
    check_batch_dims,
    check_data_and_dim,
    check_seq_dims,
    get_batch_dims,
    get_data,
    get_seq_dims,
    swap2,
)

######################################
#     Tests for check_batch_dims     #
######################################


def test_check_batch_dims_correct() -> None:
    check_batch_dims({0})


def test_check_batch_dims_incorrect() -> None:
    with pytest.raises(RuntimeError, match=r"The batch dimensions do not match."):
        check_batch_dims({0, 1})


########################################
#     Tests for check_data_and_dim     #
########################################


@pytest.mark.parametrize("array", [np.ones((2, 3)), torch.ones(2, 3)])
def test_check_data_and_dim_correct(array: np.ndarray | torch.Tensor) -> None:
    check_data_and_dim(array, batch_dim=0)
    # will fail if an exception is raised


@pytest.mark.parametrize("array", [np.array(2), torch.tensor(2)])
def test_check_data_and_dim_incorrect_data_dim(array: np.ndarray | torch.Tensor) -> None:
    with pytest.raises(RuntimeError, match=r"data needs at least 1 dimensions \(received: 0\)"):
        check_data_and_dim(np.array(2), batch_dim=0)


@pytest.mark.parametrize("array", [np.ones((2, 3)), torch.ones(2, 3)])
@pytest.mark.parametrize("batch_dim", [-1, 2, 3])
def test_check_data_and_dim_incorrect_batch_dim(
    array: np.ndarray | torch.Tensor, batch_dim: int
) -> None:
    with pytest.raises(
        RuntimeError, match=r"Incorrect batch_dim \(.*\) but the value should be in \[0, 1\]"
    ):
        check_data_and_dim(np.ones((2, 3)), batch_dim=batch_dim)


####################################
#     Tests for check_seq_dims     #
####################################


def test_check_seq_dims_correct() -> None:
    check_seq_dims({0})


def test_check_seq_dims_incorrect() -> None:
    with pytest.raises(RuntimeError, match=r"The sequence dimensions do not match."):
        check_seq_dims({0, 1})


####################################
#     Tests for get_batch_dims     #
####################################


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
    assert get_batch_dims(()) == set()


##############################
#     Tests for get_data     #
##############################


def test_get_data_int() -> None:
    assert get_data(42) == 42


@pytest.mark.parametrize("data", [np.ones((2, 3)), BatchedArray(np.ones((2, 3)))])
def test_get_data_array(data: Any) -> None:
    assert np.array_equal(get_data(data), np.ones((2, 3)))


@pytest.mark.parametrize("data", [torch.ones(2, 3), BatchedTensor(torch.ones(2, 3))])
def test_get_data_tensor(data: Any) -> None:
    assert get_data(data).equal(torch.ones(2, 3))


##################################
#     Tests for get_seq_dims     #
##################################


def test_get_seq_dims() -> None:
    assert get_seq_dims(
        (BatchedTensorSeq(torch.ones(2, 3)), BatchedTensorSeq(torch.ones(2, 3))),
        {"val": BatchedTensorSeq(torch.ones(2, 3))},
    ) == {1}


def test_get_seq_dims_2() -> None:
    assert get_seq_dims(
        (BatchedTensorSeq(torch.ones(2, 3)), BatchedTensorSeq.from_seq_batch(torch.ones(2, 3))),
        {"val": BatchedTensorSeq.from_seq_batch(torch.ones(2, 3))},
    ) == {0, 1}


def test_get_seq_dims_empty() -> None:
    assert get_seq_dims(()) == set()


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
