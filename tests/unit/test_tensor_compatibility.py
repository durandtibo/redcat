import math
from collections.abc import Callable
from functools import partial
from typing import Union

import torch
from pytest import mark
from torch import Tensor

from redcat import BaseBatchedTensor, BatchedTensor, BatchedTensorSeq

UNARY_FUNCTIONS = (
    partial(torch.clamp, min=0.1, max=0.5),
    partial(torch.cumsum, dim=0),
    partial(torch.cumsum, dim=1),
    partial(torch.unsqueeze, dim=-1),
    partial(torch.unsqueeze, dim=0),
    torch.abs,
    torch.acos,
    torch.acosh,
    torch.asin,
    torch.asinh,
    torch.atan,
    torch.atanh,
    torch.cos,
    torch.cosh,
    torch.exp,
    torch.isinf,
    torch.isnan,
    torch.isneginf,
    torch.isposinf,
    torch.log,
    torch.log1p,
    torch.logical_not,
    # torch.max,
    # torch.min,
    torch.neg,
    torch.sin,
    torch.sinh,
    torch.sqrt,
    torch.tan,
    torch.tanh,
)

PAIRWISE_FUNCTIONS = (
    # partial(torch.max, dim=0),
    # partial(torch.max, dim=1),
    # partial(torch.min, dim=0),
    # partial(torch.min, dim=1),
    torch.add,
    torch.div,
    torch.eq,
    torch.fmod,
    torch.ge,
    torch.gt,
    torch.le,
    torch.logical_and,
    torch.logical_or,
    torch.logical_xor,
    torch.lt,
    torch.mul,
    torch.sub,
)


@mark.parametrize("func", UNARY_FUNCTIONS)
def test_same_behaviour_unary(func: Callable) -> None:
    tensor = torch.rand(2, 3).mul(2.0)
    assert func(BatchedTensor(tensor)).data.allclose(func(tensor), equal_nan=True)


@mark.parametrize("func", PAIRWISE_FUNCTIONS)
def test_same_behaviour_pairwise(func: Callable) -> None:
    tensor1 = torch.rand(2, 3)
    tensor2 = torch.rand(2, 3) + 1.0
    assert func(BatchedTensor(tensor1), BatchedTensor(tensor2)).data.allclose(
        func(tensor1, tensor2), equal_nan=True
    )


def test_torch_abs() -> None:
    assert torch.abs(BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [0.0, -1.0, -2.0]]))).equal(
        BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]))
    )


def test_torch_acos() -> None:
    assert torch.acos(BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))).allclose(
        BatchedTensor(
            torch.tensor(
                [[math.pi, math.pi / 2, 0.0], [2 * math.pi / 3, math.pi / 2, math.pi / 3]],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_torch_acosh() -> None:
    assert torch.acosh(BatchedTensor(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))).allclose(
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
def test_torch_add(other: Union[BaseBatchedTensor, Tensor, int, float]) -> None:
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
def test_torch_add_alpha(other: Union[BaseBatchedTensor, Tensor, int, float]) -> None:
    assert torch.add(BatchedTensor(torch.ones(2, 3)), other, alpha=2.0).equal(
        BatchedTensor(torch.full((2, 3), 3.0))
    )


def test_torch_add_tensor() -> None:
    assert torch.add(torch.zeros(2, 3), BatchedTensor(torch.ones(2, 3))).equal(
        BatchedTensor(torch.ones(2, 3))
    )


def test_torch_asin() -> None:
    assert torch.asin(BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))).allclose(
        BatchedTensor(
            torch.tensor(
                [[-math.pi / 2, 0.0, math.pi / 2], [-math.pi / 6, 0.0, math.pi / 6]],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_torch_asinh() -> None:
    assert torch.asinh(BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))).allclose(
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


def test_torch_atan() -> None:
    assert torch.atan(
        BatchedTensor(torch.tensor([[0.0, 1.0, math.sqrt(3.0)], [-math.sqrt(3.0), -1.0, 0.0]]))
    ).allclose(
        BatchedTensor(
            torch.tensor(
                [[0.0, math.pi / 4, math.pi / 3], [-math.pi / 3, -math.pi / 4, 0.0]],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_torch_atanh() -> None:
    assert torch.atanh(BatchedTensor(torch.tensor([[-0.5, 0.0, 0.5], [-0.1, 0.0, 0.1]]))).allclose(
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


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.tensor([[4, 5], [14, 15]])),
        BatchedTensor(torch.tensor([[4, 5], [14, 15]])),
        torch.tensor([[4, 5], [14, 15]]),
    ),
)
def test_torch_cat(other: Union[BaseBatchedTensor, Tensor]) -> None:
    assert torch.cat(
        tensors=[
            BatchedTensor(torch.tensor([[0, 1, 2], [10, 11, 12]])),
            other,
        ],
        dim=1,
    ).equal(
        BatchedTensor(
            torch.tensor(
                [[0, 1, 2, 4, 5], [10, 11, 12, 14, 15]],
            )
        ),
    )


def test_torch_clamp() -> None:
    assert torch.clamp(BatchedTensor(torch.arange(10).view(2, 5)), min=2, max=5).equal(
        BatchedTensor(torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]]))
    )


def test_torch_cos() -> None:
    assert torch.cos(
        BatchedTensor(
            torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]])
        )
    ).allclose(
        BatchedTensor(torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, 1.0]], dtype=torch.float)),
        atol=1e-6,
    )


def test_torch_cosh() -> None:
    assert torch.cosh(BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))).allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [1.5430806348152437, 1.0, 1.5430806348152437],
                    [1.1276259652063807, 1.0, 1.1276259652063807],
                ],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_torch_cumsum_dim_0() -> None:
    assert torch.cumsum(BatchedTensor(torch.ones(2, 3)), dim=0).equal(
        BatchedTensor(torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    )


def test_torch_cumsum_dim_1() -> None:
    assert torch.cumsum(BatchedTensor(torch.ones(2, 3)), dim=1).equal(
        BatchedTensor(torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
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
def test_torch_div(other: Union[BaseBatchedTensor, Tensor, int, float]) -> None:
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
        BatchedTensor(torch.full((2, 5), 5.0)),
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.full((2, 1), 5.0)),
        5,
        5.0,
    ),
)
def test_torch_eq(other: Union[BaseBatchedTensor, Tensor, int, float]) -> None:
    assert torch.eq(BatchedTensor(torch.arange(10).view(2, 5)), other).equal(
        BatchedTensor(
            torch.tensor(
                [[False, False, False, False, False], [True, False, False, False, False]],
                dtype=torch.bool,
            ),
        )
    )


def test_torch_eq_tensor() -> None:
    assert torch.eq(torch.arange(10).view(2, 5), BatchedTensor(torch.full((2, 5), 5.0))).equal(
        BatchedTensor(
            torch.tensor(
                [[False, False, False, False, False], [True, False, False, False, False]],
                dtype=torch.bool,
            ),
        )
    )


def test_torch_exp() -> None:
    assert torch.exp(
        BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
    ).allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [2.7182817459106445, 7.389056205749512, 20.08553695678711],
                    [54.598148345947266, 148.4131622314453, 403.4288024902344],
                ]
            )
        ),
        atol=1e-6,
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
def test_torch_fmod(other: Union[BaseBatchedTensor, Tensor, int, float]) -> None:
    assert torch.fmod(BatchedTensor(torch.ones(2, 3)), other).equal(BatchedTensor(torch.ones(2, 3)))


def test_torch_fmod_tensor() -> None:
    assert torch.fmod(torch.ones(2, 3), BatchedTensor(torch.full((2, 3), 2.0))).equal(
        BatchedTensor(torch.ones(2, 3))
    )


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 5), 5.0)),
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.full((2, 1), 5.0)),
        5,
        5.0,
    ),
)
def test_torch_ge(other: Union[BaseBatchedTensor, Tensor, int, float]) -> None:
    assert torch.ge(BatchedTensor(torch.arange(10).view(2, 5)), other).equal(
        BatchedTensor(
            torch.tensor(
                [[False, False, False, False, False], [True, True, True, True, True]],
                dtype=torch.bool,
            ),
        )
    )


def test_torch_ge_tensor() -> None:
    assert torch.ge(torch.arange(10).view(2, 5), BatchedTensor(torch.full((2, 5), 5.0))).equal(
        BatchedTensor(
            torch.tensor(
                [[False, False, False, False, False], [True, True, True, True, True]],
                dtype=torch.bool,
            ),
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 5), 5.0)),
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.full((2, 1), 5.0)),
        5,
        5.0,
    ),
)
def test_torch_gt(other: Union[BaseBatchedTensor, Tensor, int, float]) -> None:
    assert torch.gt(BatchedTensor(torch.arange(10).view(2, 5)), other).equal(
        BatchedTensor(
            torch.tensor(
                [[False, False, False, False, False], [False, True, True, True, True]],
                dtype=torch.bool,
            ),
        )
    )


def test_torch_gt_tensor() -> None:
    assert torch.gt(torch.arange(10).view(2, 5), BatchedTensor(torch.full((2, 5), 5.0))).equal(
        BatchedTensor(
            torch.tensor(
                [[False, False, False, False, False], [False, True, True, True, True]],
                dtype=torch.bool,
            ),
        )
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
def test_torch_mul(other: Union[BaseBatchedTensor, Tensor, int, float]) -> None:
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
def test_torch_sub(other: Union[BaseBatchedTensor, Tensor, int, float]) -> None:
    assert torch.sub(BatchedTensor(torch.full((2, 3), 2.0)), other).equal(
        BatchedTensor(torch.ones(2, 3))
    )


def test_torch_sub_tensor() -> None:
    assert torch.sub(torch.full((2, 3), 2.0), BatchedTensor(torch.ones(2, 3))).equal(
        BatchedTensor(torch.ones(2, 3))
    )
