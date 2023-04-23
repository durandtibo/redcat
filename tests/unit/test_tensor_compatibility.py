from typing import Union

import torch
from pytest import mark

from redcat import BaseBatchedTensor, BatchedTensor, BatchedTensorSeq

##################################################
#     Mathematical | arithmetical operations     #
##################################################


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.ones(2, 3)),
        BatchedTensor(torch.ones(2, 3)),
        torch.ones(2, 3),
        1,
        1.0,
    ),
)
def test_torch_add(other: Union[BaseBatchedTensor, torch.Tensor, int, float]) -> None:
    assert torch.add(BatchedTensor(torch.zeros(2, 3)), other).equal(BatchedTensor(torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.ones(2, 3)),
        BatchedTensor(torch.ones(2, 3)),
        torch.ones(2, 3),
        1,
        1.0,
    ),
)
def test_torch_add_alpha(other: Union[BaseBatchedTensor, torch.Tensor, int, float]) -> None:
    assert torch.add(BatchedTensor(torch.ones(2, 3)), other, alpha=2.0).equal(
        BatchedTensor(torch.full((2, 3), 3.0))
    )


def test_torch_add_tensor() -> None:
    assert torch.add(torch.zeros(2, 3), BatchedTensor(torch.ones(2, 3))).equal(
        BatchedTensor(torch.ones(2, 3))
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_torch_div(other: Union[BaseBatchedTensor, torch.Tensor, int, float]) -> None:
    assert torch.div(BatchedTensor(torch.ones(2, 3)), other).equal(
        BatchedTensor(torch.full((2, 3), 0.5))
    )


def test_torch_div_tensor() -> None:
    assert torch.div(torch.ones(2, 3), BatchedTensor(torch.full((2, 3), 2.0))).equal(
        BatchedTensor(torch.full((2, 3), 0.5))
    )


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_torch_fmod(other: Union[BaseBatchedTensor, torch.Tensor, int, float]) -> None:
    assert torch.fmod(BatchedTensor(torch.ones(2, 3)), other).equal(BatchedTensor(torch.ones(2, 3)))


def test_torch_fmod_tensor() -> None:
    assert torch.fmod(torch.ones(2, 3), BatchedTensor(torch.full((2, 3), 2.0))).equal(
        BatchedTensor(torch.ones(2, 3))
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.ones(2, 3)),
        BatchedTensor(torch.ones(2, 3)),
        torch.ones(2, 3),
        1,
        1.0,
    ),
)
def test_torch_mul(other: Union[BaseBatchedTensor, torch.Tensor, int, float]) -> None:
    assert torch.mul(BatchedTensor(torch.ones(2, 3)), other).equal(BatchedTensor(torch.ones(2, 3)))


def test_torch_mul_tensor() -> None:
    assert torch.mul(torch.ones(2, 3), BatchedTensor(torch.ones(2, 3))).equal(
        BatchedTensor(torch.ones(2, 3))
    )


def test_torch_neg() -> None:
    assert torch.neg(BatchedTensor(torch.full((2, 3), 2.0))).equal(
        BatchedTensor(torch.full((2, 3), -2.0))
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.ones(2, 3)),
        BatchedTensor(torch.ones(2, 3)),
        torch.ones(2, 3),
        1,
        1.0,
    ),
)
def test_torch_sub(other: Union[BaseBatchedTensor, torch.Tensor, int, float]) -> None:
    assert torch.sub(BatchedTensor(torch.full((2, 3), 2.0)), other).equal(
        BatchedTensor(torch.ones(2, 3))
    )


def test_torch_sub_tensor() -> None:
    assert torch.sub(torch.full((2, 3), 2.0), BatchedTensor(torch.ones(2, 3))).equal(
        BatchedTensor(torch.ones(2, 3))
    )
