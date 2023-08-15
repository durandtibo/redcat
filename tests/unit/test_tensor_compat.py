from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
from coola import objects_are_allclose, objects_are_equal
from pytest import mark
from torch import Tensor

from redcat import BatchedTensor, BatchedTensorSeq

BATCH_CLASSES = (BatchedTensor, BatchedTensorSeq)

UNARY_FUNCTIONS = (
    # torch.arctan2,
    partial(torch.argsort, dim=0),
    partial(torch.argsort, dim=1),
    partial(torch.clamp, min=0.1, max=0.5),
    partial(torch.cumprod, dim=0),
    partial(torch.cumprod, dim=1),
    partial(torch.cumsum, dim=0),
    partial(torch.cumsum, dim=1),
    partial(torch.unsqueeze, dim=-1),
    partial(torch.unsqueeze, dim=0),
    partial(torch.clamp, min=0.0),
    partial(torch.clamp, max=1.0),
    partial(torch.clamp, min=0.0, max=1.0),
    torch.abs,
    torch.acos,
    torch.acosh,
    torch.angle,
    torch.arccos,
    torch.arccosh,
    torch.arcsin,
    torch.arcsinh,
    torch.arctan,
    torch.arctanh,
    torch.asin,
    torch.asinh,
    torch.atan,
    torch.atanh,
    torch.ceil,
    torch.cos,
    torch.cosh,
    torch.deg2rad,
    torch.digamma,
    torch.erf,
    torch.erfc,
    torch.erfinv,
    torch.exp,
    torch.fix,
    torch.floor,
    torch.floor,
    torch.frac,
    torch.isfinite,
    torch.isinf,
    torch.isnan,
    torch.isneginf,
    torch.isposinf,
    torch.isreal,
    torch.lgamma,
    torch.log,
    torch.log10,
    torch.log1p,
    torch.log2,
    torch.logical_not,
    torch.logit,
    torch.nan_to_num,
    torch.neg,
    torch.negative,
    torch.positive,
    torch.rad2deg,
    torch.real,
    torch.reciprocal,
    torch.round,
    torch.rsqrt,
    torch.sigmoid,
    torch.sign,
    torch.sin,
    torch.sinc,
    torch.sinh,
    torch.sqrt,
    torch.sqrt,
    torch.square,
    torch.tan,
    torch.tanh,
    torch.trunc,
)

PAIRWISE_FUNCTIONS = (
    torch.add,
    torch.div,
    torch.eq,
    torch.floor_divide,
    torch.fmod,
    torch.ge,
    torch.greater,
    torch.greater_equal,
    torch.gt,
    torch.le,
    torch.less,
    torch.less_equal,
    torch.logaddexp,
    torch.logical_and,
    torch.logical_or,
    torch.logical_xor,
    torch.lt,
    torch.maximum,
    torch.minimum,
    torch.mul,
    torch.ne,
    torch.nextafter,
    torch.not_equal,
    torch.remainder,
    torch.sub,
    torch.true_divide,
)

BATCH_TO_TENSOR = (
    partial(torch.argmax, dim=0),
    partial(torch.argmax, dim=1),
    partial(torch.argmin, dim=0),
    partial(torch.argmin, dim=1),
    partial(torch.select, dim=0, index=0),
    partial(torch.select, dim=1, index=0),
    partial(torch.amax, dim=0),
    partial(torch.amax, dim=1),
    partial(torch.amin, dim=0),
    partial(torch.amin, dim=1),
    partial(torch.max, dim=0),
    partial(torch.max, dim=1),
    partial(torch.mean, dim=0),
    partial(torch.mean, dim=1),
    partial(torch.median, dim=0),
    partial(torch.median, dim=1),
    partial(torch.min, dim=0),
    partial(torch.min, dim=1),
    partial(torch.nanmean, dim=0),
    partial(torch.nanmean, dim=1),
    partial(torch.nanmedian, dim=0),
    partial(torch.nanmedian, dim=1),
    partial(torch.prod, dim=0),
    partial(torch.prod, dim=1),
    partial(torch.sum, dim=0),
    partial(torch.sum, dim=1),
    partial(torch.nansum, dim=0),
    partial(torch.nansum, dim=1),
    torch.argmax,
    torch.argmin,
    torch.max,
    torch.mean,
    torch.median,
    torch.min,
    torch.nanmean,
    torch.nanmedian,
    torch.nansum,
    torch.prod,
    torch.sum,
)


@mark.parametrize("func", UNARY_FUNCTIONS)
@mark.parametrize("cls", BATCH_CLASSES)
def test_unary(func: Callable, cls: type[BatchedTensor]) -> None:
    tensor = torch.rand(2, 3).mul(2.0)
    assert func(cls(tensor)).allclose(cls(func(tensor)), equal_nan=True)


@mark.parametrize("func", UNARY_FUNCTIONS)
def test_unary_batched_tensor_custom_dims(func: Callable) -> None:
    tensor = torch.rand(2, 3).mul(2.0)
    assert func(BatchedTensor(tensor, batch_dim=1)).allclose(
        BatchedTensor(func(tensor), batch_dim=1), equal_nan=True
    )


@mark.parametrize("func", UNARY_FUNCTIONS)
def test_unary_batched_tensor_seq_custom_dims(func: Callable) -> None:
    tensor = torch.rand(2, 3).mul(2.0)
    assert func(BatchedTensorSeq.from_seq_batch(tensor)).allclose(
        BatchedTensorSeq.from_seq_batch(func(tensor)), equal_nan=True
    )


@mark.parametrize("func", PAIRWISE_FUNCTIONS)
@mark.parametrize("cls", BATCH_CLASSES)
def test_pairwise(func: Callable, cls: type[BatchedTensor]) -> None:
    tensor1 = torch.rand(2, 3)
    tensor2 = torch.rand(2, 3) + 1.0
    assert func(cls(tensor1), cls(tensor2)).allclose(cls(func(tensor1, tensor2)), equal_nan=True)


@mark.parametrize("func", PAIRWISE_FUNCTIONS)
def test_pairwise_batched_tensor_custom_dims(func: Callable) -> None:
    tensor1 = torch.rand(2, 3)
    tensor2 = torch.rand(2, 3) + 1.0
    assert func(BatchedTensor(tensor1, batch_dim=1), BatchedTensor(tensor2, batch_dim=1)).allclose(
        BatchedTensor(func(tensor1, tensor2), batch_dim=1), equal_nan=True
    )


@mark.parametrize("func", PAIRWISE_FUNCTIONS)
def test_pairwise_batched_tensor_seq_custom_dims(func: Callable) -> None:
    tensor1 = torch.rand(2, 3)
    tensor2 = torch.rand(2, 3) + 1.0
    assert func(
        BatchedTensorSeq.from_seq_batch(tensor1), BatchedTensorSeq.from_seq_batch(tensor2)
    ).allclose(BatchedTensorSeq.from_seq_batch(func(tensor1, tensor2)), equal_nan=True)


@mark.parametrize("func", BATCH_TO_TENSOR)
@mark.parametrize("cls", BATCH_CLASSES)
def test_batch_to_tensor(func: Callable, cls: type[BatchedTensor]) -> None:
    tensor = torch.rand(2, 3)
    assert objects_are_allclose(func(cls(tensor)), func(tensor), equal_nan=True)


@mark.parametrize("func", BATCH_TO_TENSOR)
def test_batch_to_tensor_batched_tensor_custom_dims(func: Callable) -> None:
    tensor = torch.rand(2, 3)
    assert objects_are_allclose(
        func(BatchedTensor(tensor, batch_dim=1)), func(tensor), equal_nan=True
    )


@mark.parametrize("func", BATCH_TO_TENSOR)
def test_batch_to_tensor_batched_tensor_seq_custom_dims(func: Callable) -> None:
    tensor = torch.rand(2, 3)
    assert objects_are_allclose(
        func(BatchedTensorSeq.from_seq_batch(tensor)), func(tensor), equal_nan=True
    )


def test_same_behaviour_take_along_dim() -> None:  # TODO: update
    tensor = torch.rand(4, 6)
    indices = torch.randint(0, 3, size=(4, 6))
    assert torch.take_along_dim(BatchedTensor(tensor), indices=indices).data.equal(
        torch.take_along_dim(tensor, indices=indices)
    )


def test_same_behaviour_take_along_dim_batch() -> None:  # TODO: update
    tensor = torch.rand(4, 6)
    indices = torch.randint(0, 3, size=(4, 6))
    assert torch.take_along_dim(BatchedTensor(tensor), indices=BatchedTensor(indices)).data.equal(
        torch.take_along_dim(tensor, indices=indices)
    )


def test_same_behaviour_take_along_dim_tensor() -> None:  # TODO: update
    tensor = torch.rand(4, 6)
    indices = torch.randint(0, 3, size=(4, 6))
    assert torch.take_along_dim(tensor, indices=BatchedTensor(indices)).data.equal(
        torch.take_along_dim(tensor, indices=indices)
    )


def test_same_behaviour_take_along_dim_0() -> None:  # TODO: update
    tensor = torch.rand(4, 6)
    indices = torch.randint(0, 3, size=(4, 6))
    assert torch.take_along_dim(
        BatchedTensor(tensor), indices=BatchedTensor(indices), dim=0
    ).data.equal(torch.take_along_dim(tensor, indices=indices, dim=0))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.tensor([[4, 5], [14, 15]])),
        torch.tensor([[4, 5], [14, 15]]),
    ),
)
def test_torch_cat(other: BatchedTensor | Tensor) -> None:
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


@mark.parametrize("cls", BATCH_CLASSES)
def test_torch_sort(cls: type[BatchedTensor]) -> None:
    x = torch.rand(6, 10)
    assert objects_are_equal(
        torch.sort(cls(x)), torch.return_types.sort(cls(y) for y in torch.sort(x))
    )
