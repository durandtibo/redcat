from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal

from redcat import BatchedTensor, BatchedTensorSeq
from redcat.utils.tensor import get_available_devices

if TYPE_CHECKING:
    from collections.abc import Callable

BATCH_CLASSES = (
    BatchedTensor,
    partial(BatchedTensor, batch_dim=1),
    BatchedTensorSeq,
    BatchedTensorSeq.from_seq_batch,
)
DTYPES = (torch.float, torch.long)
SHAPES = ((2, 3), (2, 3, 4), (2, 3, 4, 5))


@pytest.mark.parametrize("cls", BATCH_CLASSES)
@pytest.mark.parametrize("device", get_available_devices())
def test_tensor_device(cls: type[BatchedTensor], device: str) -> None:
    tensor = torch.ones(2, 3, device=device)
    assert cls(tensor).device == tensor.device


@pytest.mark.parametrize("cls", BATCH_CLASSES)
def test_tensor_dim(cls: type[BatchedTensor]) -> None:
    tensor = torch.ones(2, 3)
    assert cls(tensor).dim() == tensor.dim()


@pytest.mark.parametrize("cls", BATCH_CLASSES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tensor_dtype(cls: type[BatchedTensor], dtype: torch.dtype) -> None:
    tensor = torch.ones(2, 3, dtype=dtype)
    assert cls(tensor).dtype == tensor.dtype


@pytest.mark.parametrize("cls", BATCH_CLASSES)
def test_tensor_is_contiguous(cls: type[BatchedTensor]) -> None:
    tensor = torch.rand(2, 3)
    assert cls(tensor).is_contiguous() == tensor.is_contiguous()


@pytest.mark.parametrize("cls", BATCH_CLASSES)
@pytest.mark.parametrize("shape", SHAPES)
def test_tensor_ndim(cls: type[BatchedTensor], shape: tuple[int, ...]) -> None:
    tensor = torch.ones(*shape)
    assert cls(tensor).ndim == tensor.ndim


@pytest.mark.parametrize("cls", BATCH_CLASSES)
@pytest.mark.parametrize("shape", SHAPES)
def test_tensor_ndimension(cls: type[BatchedTensor], shape: tuple[int, ...]) -> None:
    tensor = torch.ones(*shape)
    assert cls(tensor).ndimension() == tensor.ndimension()


@pytest.mark.parametrize("cls", BATCH_CLASSES)
@pytest.mark.parametrize("shape", SHAPES)
def test_tensor_numel(cls: type[BatchedTensor], shape: tuple[int, ...]) -> None:
    tensor = torch.ones(*shape)
    assert cls(tensor).numel() == tensor.numel()


@pytest.mark.parametrize("cls", BATCH_CLASSES)
@pytest.mark.parametrize("shape", SHAPES)
def test_tensor_shape(cls: type[BatchedTensor], shape: tuple[int, ...]) -> None:
    tensor = torch.rand(*shape)
    assert cls(tensor).shape == tensor.shape


@pytest.mark.parametrize("cls", BATCH_CLASSES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", get_available_devices())
def test_tensor_to(cls: type[BatchedTensor], dtype: torch.dtype, device: str) -> None:
    device = torch.device(device)
    tensor = torch.ones(2, 3, dtype=dtype, device=device)
    batch = cls(tensor)
    assert batch.dtype == tensor.dtype
    assert batch.device == tensor.device


UNARY_FUNCTIONS = (
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
    partial(torch.div, rounding_mode="floor"),
    partial(torch.div, rounding_mode="trunc"),
    partial(torch.div, rounding_mode=None),
    torch.add,
    torch.arctan2,
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
    partial(torch.amax, dim=0),
    partial(torch.amax, dim=1),
    partial(torch.amin, dim=0),
    partial(torch.amin, dim=1),
    partial(torch.argmax, dim=0),
    partial(torch.argmax, dim=1),
    partial(torch.argmin, dim=0),
    partial(torch.argmin, dim=1),
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
    partial(torch.nansum, dim=0),
    partial(torch.nansum, dim=1),
    partial(torch.prod, dim=0),
    partial(torch.prod, dim=1),
    partial(torch.select, dim=0, index=0),
    partial(torch.select, dim=1, index=0),
    partial(torch.sum, dim=0),
    partial(torch.sum, dim=1),
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


BATCH_TUPLE = (
    partial(torch.sort, dim=0),
    partial(torch.sort, dim=1),
)

BATCH_TO_TENSOR_TUPLE = (
    partial(torch.max, dim=0),
    partial(torch.max, dim=1),
    partial(torch.min, dim=0),
    partial(torch.min, dim=1),
    partial(torch.median, dim=0),
    partial(torch.median, dim=1),
    partial(torch.nanmedian, dim=0),
    partial(torch.nanmedian, dim=1),
)


@pytest.mark.parametrize("func", UNARY_FUNCTIONS)
@pytest.mark.parametrize("cls", BATCH_CLASSES)
def test_unary(func: Callable, cls: type[BatchedTensor]) -> None:
    tensor = torch.rand(2, 3).mul(2.0)
    assert func(cls(tensor)).allclose(cls(func(tensor)), equal_nan=True)


@pytest.mark.parametrize("func", PAIRWISE_FUNCTIONS)
@pytest.mark.parametrize("cls", BATCH_CLASSES)
def test_pairwise(func: Callable, cls: type[BatchedTensor]) -> None:
    tensor1 = torch.rand(2, 3)
    tensor2 = torch.rand(2, 3) + 1.0
    assert func(cls(tensor1), cls(tensor2)).allclose(cls(func(tensor1, tensor2)), equal_nan=True)


@pytest.mark.parametrize("func", BATCH_TO_TENSOR)
@pytest.mark.parametrize("cls", BATCH_CLASSES)
def test_batch_to_tensor(func: Callable, cls: type[BatchedTensor]) -> None:
    tensor = torch.rand(2, 3)
    assert objects_are_allclose(func(cls(tensor)), func(tensor), equal_nan=True)


@pytest.mark.parametrize("func", BATCH_TUPLE)
@pytest.mark.parametrize("cls", BATCH_CLASSES)
def test_batch_tuple(func: Callable, cls: type[BatchedTensor]) -> None:
    x = torch.rand(6, 10)
    assert objects_are_equal(tuple(func(cls(x))), tuple(cls(y) for y in func(x)))


@pytest.mark.parametrize("func", BATCH_TO_TENSOR_TUPLE)
@pytest.mark.parametrize("cls", BATCH_CLASSES)
def test_batch_to_tensor_tuple(func: Callable, cls: type[BatchedTensor]) -> None:
    x = torch.rand(6, 10)
    assert objects_are_equal(func(cls(x)), func(x))


@pytest.mark.parametrize("cls", BATCH_CLASSES)
def test_torch_cat(cls: type[BatchedTensor]) -> None:
    assert torch.cat(
        tensors=[
            cls(torch.tensor([[0, 1, 2], [10, 11, 12]])),
            torch.tensor([[4, 5], [14, 15]]),
        ],
        dim=1,
    ).equal(
        cls(
            torch.tensor(
                [[0, 1, 2, 4, 5], [10, 11, 12, 14, 15]],
            )
        ),
    )


@pytest.mark.parametrize("cls", BATCH_CLASSES)
def test_take_along_dim(cls: type[BatchedTensor]) -> None:
    tensor = torch.rand(4, 6)
    indices = torch.randint(0, 3, size=(4, 6))
    assert torch.take_along_dim(cls(tensor), indices=indices).equal(
        torch.take_along_dim(tensor, indices=indices)
    )
