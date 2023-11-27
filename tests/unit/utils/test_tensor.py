from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from coola.testing import numpy_available, torch_available
from pytest import mark, raises
from torch import Tensor

from redcat import BatchedTensor, BatchList
from redcat.utils.tensor import (
    align_to_batch_first,
    align_to_batch_seq,
    align_to_seq_batch,
    compute_batch_seq_permutation,
    get_torch_generator,
    permute_along_dim,
    to_tensor,
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


#########################################
#     Tests for get_torch_generator     #
#########################################


def test_get_torch_generator_same_seed() -> None:
    assert torch.randn(4, 6, generator=get_torch_generator(1)).equal(
        torch.randn(4, 6, generator=get_torch_generator(1))
    )


def test_get_torch_generator_different_seeds() -> None:
    assert not torch.randn(4, 6, generator=get_torch_generator(1)).equal(
        torch.randn(4, 6, generator=get_torch_generator(2))
    )


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


###############################
#     Tests for to_tensor     #
###############################


@torch_available
@mark.parametrize(
    "data",
    (
        torch.tensor([3, 1, 2, 0, 1]),
        [3, 1, 2, 0, 1],
        (3, 1, 2, 0, 1),
        BatchList([3, 1, 2, 0, 1]),
        BatchedTensor(torch.tensor([3, 1, 2, 0, 1])),
    ),
)
def test_to_tensor_long(data: Sequence | Tensor) -> None:
    assert to_tensor(data).equal(torch.tensor([3, 1, 2, 0, 1], dtype=torch.long))


@torch_available
@mark.parametrize(
    "data",
    (
        torch.tensor([3.0, 1.0, 2.0, 0.0, 1.0]),
        [3.0, 1.0, 2.0, 0.0, 1.0],
        (3.0, 1.0, 2.0, 0.0, 1.0),
        BatchList([3.0, 1.0, 2.0, 0.0, 1.0]),
        BatchedTensor(torch.tensor([3.0, 1.0, 2.0, 0.0, 1.0])),
    ),
)
def test_to_tensor_float(data: Sequence | Tensor) -> None:
    assert to_tensor(data).equal(torch.tensor([3.0, 1.0, 2.0, 0.0, 1.0], dtype=torch.float))


@numpy_available
def test_to_tensor_numpy() -> None:
    assert to_tensor(np.array([3, 1, 2, 0, 1])).equal(
        torch.tensor([3, 1, 2, 0, 1], dtype=torch.long)
    )
