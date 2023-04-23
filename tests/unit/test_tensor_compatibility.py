import math
from typing import Union

import torch
from pytest import mark

from redcat import BaseBatchedTensor, BatchedTensor, BatchedTensorSeq


def test_torch_abs() -> None:
    tensor = torch.tensor([[0.0, 1.0, 2.0], [0.0, -1.0, -2.0]])
    output = torch.abs(BatchedTensor(tensor))
    assert output.equal(BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])))
    assert output.data.equal(torch.abs(tensor))


def test_torch_acos() -> None:
    tensor = torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]])
    output = torch.acos(BatchedTensor(tensor))
    assert output.allclose(
        BatchedTensor(
            torch.tensor(
                [[math.pi, math.pi / 2, 0.0], [2 * math.pi / 3, math.pi / 2, math.pi / 3]],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )
    assert output.data.allclose(torch.acos(tensor), atol=1e-6)


def test_torch_acosh() -> None:
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    output = torch.acosh(BatchedTensor(tensor))
    assert output.allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [0.0, 1.3169578969248166, 1.762747174039086],
                    [2.0634370688955608, 2.2924316695611777, 2.477888730288475],
                ],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )
    assert output.data.allclose(torch.acosh(tensor), atol=1e-6)


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


def test_torch_asin() -> None:
    tensor = torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]])
    output = torch.asin(BatchedTensor(tensor))
    assert output.allclose(
        BatchedTensor(
            torch.tensor(
                [[-math.pi / 2, 0.0, math.pi / 2], [-math.pi / 6, 0.0, math.pi / 6]],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )
    assert output.data.allclose(torch.asin(tensor), atol=1e-6)


def test_torch_asinh() -> None:
    tensor = torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]])
    output = torch.asinh(BatchedTensor(tensor))
    assert output.allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [-0.8813735842704773, 0.0, 0.8813735842704773],
                    [-0.4812118113040924, 0.0, 0.4812118113040924],
                ],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )
    assert output.data.allclose(torch.asinh(tensor), atol=1e-6)


def test_torch_atan() -> None:
    tensor = torch.tensor([[0.0, 1.0, math.sqrt(3.0)], [-math.sqrt(3.0), -1.0, 0.0]])
    output = torch.atan(BatchedTensor(tensor))
    assert output.allclose(
        BatchedTensor(
            torch.tensor(
                [[0.0, math.pi / 4, math.pi / 3], [-math.pi / 3, -math.pi / 4, 0.0]],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )
    assert output.data.allclose(torch.atan(tensor), atol=1e-6)


def test_torch_atanh() -> None:
    tensor = torch.tensor([[-0.5, 0.0, 0.5], [-0.1, 0.0, 0.1]])
    output = torch.atanh(BatchedTensor(tensor))
    assert output.allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [-0.5493061443340549, 0.0, 0.5493061443340549],
                    [-0.10033534773107558, 0.0, 0.10033534773107558],
                ],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )
    assert output.data.allclose(torch.atanh(tensor), atol=1e-6)


def test_torch_cumsum_dim_0() -> None:
    tensor = torch.ones(2, 3)
    output = torch.cumsum(BatchedTensor(tensor), dim=0)
    assert output.equal(BatchedTensor(torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])))
    assert output.data.equal(torch.cumsum(tensor, dim=0))


def test_torch_cumsum_dim_1() -> None:
    tensor = torch.ones(2, 3)
    output = torch.cumsum(BatchedTensor(tensor), dim=1)
    assert output.equal(BatchedTensor(torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])))
    assert output.data.equal(torch.cumsum(tensor, dim=1))


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
