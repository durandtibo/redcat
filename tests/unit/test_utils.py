from unittest.mock import patch

import numpy as np
import torch
from pytest import raises

from redcat import BatchedTensor, BatchedTensorSeq
from redcat.utils import (
    align_to_batch_first,
    align_to_batch_seq,
    align_to_seq_batch,
    check_batch_dims,
    check_seq_dims,
    compute_batch_seq_permutation,
    get_available_devices,
    get_batch_dims,
    get_seq_dims,
    permute_along_dim,
    swap2,
)

##########################################
#     Tests for align_to_batch_first     #
##########################################


def test_align_to_batch_first_no_permutation() -> None:
    assert align_to_batch_first(torch.arange(10).view(5, 2), batch_dim=0).equal(
        torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    )


def test_align_to_batch_first_permute_dims() -> None:
    assert align_to_batch_first(torch.arange(10).view(5, 2), batch_dim=1).equal(
        torch.tensor([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]])
    )


def test_align_to_batch_first_permute_dims_extra_dims() -> None:
    assert align_to_batch_first(torch.arange(20).view(2, 5, 2), batch_dim=1).equal(
        torch.tensor(
            [
                [[0, 1], [10, 11]],
                [[2, 3], [12, 13]],
                [[4, 5], [14, 15]],
                [[6, 7], [16, 17]],
                [[8, 9], [18, 19]],
            ]
        )
    )


########################################
#     Tests for align_to_batch_seq     #
########################################


def test_align_to_batch_seq_no_permutation() -> None:
    assert align_to_batch_seq(torch.arange(10).view(5, 2), batch_dim=0, seq_dim=1).equal(
        torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    )


def test_align_to_batch_seq_permute_dims() -> None:
    assert align_to_batch_seq(torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0).equal(
        torch.tensor([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]])
    )


def test_align_to_batch_seq_permute_dims_extra_dims() -> None:
    assert align_to_batch_seq(torch.arange(20).view(2, 5, 2), batch_dim=1, seq_dim=2).equal(
        torch.tensor(
            [
                [[0, 10], [1, 11]],
                [[2, 12], [3, 13]],
                [[4, 14], [5, 15]],
                [[6, 16], [7, 17]],
                [[8, 18], [9, 19]],
            ]
        )
    )


########################################
#     Tests for align_to_seq_batch     #
########################################


def test_align_to_seq_batch_no_permutation() -> None:
    assert align_to_seq_batch(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0).equal(
        torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    )


def test_align_to_seq_batch_permute_dims() -> None:
    assert align_to_seq_batch(torch.arange(10).view(2, 5), batch_dim=0, seq_dim=1).equal(
        torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
    )


def test_align_to_seq_batch_permute_dims_extra_dims() -> None:
    assert align_to_seq_batch(torch.arange(20).view(2, 5, 2), batch_dim=1, seq_dim=2).equal(
        torch.tensor(
            [
                [[0, 10], [2, 12], [4, 14], [6, 16], [8, 18]],
                [[1, 11], [3, 13], [5, 15], [7, 17], [9, 19]],
            ]
        )
    )


######################################
#     Tests for check_batch_dims     #
######################################


def test_check_batch_dims_correct() -> None:
    check_batch_dims({0})


def test_check_batch_dims_incorrect() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        check_batch_dims({0, 1})


####################################
#     Tests for check_seq_dims     #
####################################


def test_check_seq_dims_correct() -> None:
    check_seq_dims({0})


def test_check_seq_dims_incorrect() -> None:
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        check_seq_dims({0, 1})


###################################################
#     Tests for compute_batch_seq_permutation     #
###################################################


def test_compute_batch_seq_permutation_batch_seq_to_seq_batch() -> None:
    assert compute_batch_seq_permutation(5, 0, 1, 1, 0) == [1, 0, 2, 3, 4]


def test_compute_batch_seq_permutation_batch_seq_to_seq_batch_dim_2() -> None:
    assert compute_batch_seq_permutation(2, 0, 1, 1, 0) == [1, 0]


def test_compute_batch_seq_permutation_seq_batch_to_batch_seq() -> None:
    assert compute_batch_seq_permutation(5, 1, 0, 0, 1) == [1, 0, 2, 3, 4]


def test_compute_batch_seq_permutation_batch_seq_to_batch_seq() -> None:
    assert compute_batch_seq_permutation(5, 0, 1, 0, 1) == [0, 1, 2, 3, 4]


def test_compute_batch_seq_permutation_batch_dim_2() -> None:
    assert compute_batch_seq_permutation(5, 0, 1, 2, 0) == [1, 2, 0, 3, 4]


def test_compute_batch_seq_permutation_seq_dim_2() -> None:
    assert compute_batch_seq_permutation(5, 0, 1, 0, 2) == [0, 2, 1, 3, 4]


def test_compute_batch_seq_permutation_update_seq_dim() -> None:
    assert compute_batch_seq_permutation(5, 0, 1, 1, 2) == [2, 0, 1, 3, 4]


def test_compute_batch_seq_permutation_incorrect_old_dims() -> None:
    with raises(RuntimeError, match=r"Incorrect old_batch_dim (.*) and old_seq_dim (.*)."):
        compute_batch_seq_permutation(5, 1, 1, 0, 1)


def test_compute_batch_seq_permutation_incorrect_new_dims() -> None:
    with raises(RuntimeError, match=r"Incorrect new_batch_dim (.*) and new_seq_dim (.*)."):
        compute_batch_seq_permutation(5, 0, 1, 1, 1)


###########################################
#     Tests for get_available_devices     #
###########################################


@patch("torch.cuda.is_available", lambda *args, **kwargs: False)
def test_get_available_devices_cpu() -> None:
    assert get_available_devices() == ("cpu",)


@patch("torch.cuda.is_available", lambda *args, **kwargs: True)
@patch("torch.cuda.device_count", lambda *args, **kwargs: 1)
def test_get_available_devices_cpu_and_gpu() -> None:
    assert get_available_devices() == ("cpu", "cuda:0")


####################################
#     Tests for get_batch_dims     #
####################################


def test_get_batch_dims() -> None:
    assert get_batch_dims(
        (BatchedTensor(torch.ones(2, 3)), BatchedTensor(torch.ones(2, 3))),
        {"val": BatchedTensorSeq(torch.ones(2, 3))},
    ) == {0}


def test_get_batch_dims_2() -> None:
    assert get_batch_dims(
        (BatchedTensor(torch.ones(2, 3)), BatchedTensor(torch.ones(2, 3), batch_dim=1)),
        {"val": BatchedTensorSeq(torch.ones(2, 3))},
    ) == {0, 1}


def test_get_batch_dims_empty() -> None:
    assert get_batch_dims(tuple(), dict()) == set()


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
    assert get_seq_dims(tuple(), dict()) == set()


#######################################
#     Tests for permute_along_dim     #
#######################################


def test_permute_along_dim_1d() -> None:
    assert permute_along_dim(tensor=torch.arange(4), permutation=torch.tensor([0, 2, 1, 3])).equal(
        torch.tensor([0, 2, 1, 3])
    )


def test_permute_along_dim_2d_dim_0() -> None:
    assert permute_along_dim(
        tensor=torch.arange(20).view(4, 5), permutation=torch.tensor([0, 2, 1, 3])
    ).equal(
        torch.tensor([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [15, 16, 17, 18, 19]])
    )


def test_permute_along_dim_2d_dim_1() -> None:
    assert permute_along_dim(
        tensor=torch.arange(20).view(4, 5), permutation=torch.tensor([0, 4, 2, 1, 3]), dim=1
    ).equal(
        torch.tensor([[0, 4, 2, 1, 3], [5, 9, 7, 6, 8], [10, 14, 12, 11, 13], [15, 19, 17, 16, 18]])
    )


def test_permute_along_dim_3d_dim_2() -> None:
    assert permute_along_dim(
        tensor=torch.arange(20).view(2, 2, 5), permutation=torch.tensor([0, 4, 2, 1, 3]), dim=2
    ).equal(
        torch.tensor(
            [[[0, 4, 2, 1, 3], [5, 9, 7, 6, 8]], [[10, 14, 12, 11, 13], [15, 19, 17, 16, 18]]]
        )
    )


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
