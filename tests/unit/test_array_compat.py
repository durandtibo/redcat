from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
from pytest import mark

from redcat import BatchedArray

BATCH_CLASSES = (BatchedArray,)

UNARY_FUNCTIONS = (
    # partial(np.select, dim=0, index=0),
    # np.arctan2,
    # np.max,
    # np.min,
    # partial(np.clip, a_min=0.1, a_max=0.5),
    partial(np.cumsum, axis=0),
    partial(np.cumsum, axis=1),
    partial(np.cumsum, axis=0, dtype=float),
    partial(np.cumsum, axis=0, dtype=int),
    # partial(np.cumprod, axis=0),
    # partial(np.cumprod, axis=1),
    # partial(np.unsqueeze, dim=-1),
    # partial(np.unsqueeze, dim=0),
    # np.abs,
    # np.acos,
    # np.acosh,
    # np.angle,
    # np.arccos,
    # np.arccosh,
    # np.arcsin,
    # np.arcsinh,
    # np.arctan,
    # np.arctanh,
    # np.asin,
    # np.asinh,
    # np.atan,
    # np.atanh,
    # np.ceil,
    # np.cos,
    # np.cosh,
    # np.deg2rad,
    # np.digamma,
    # np.erf,
    # np.erfc,
    # np.erfinv,
    # np.exp,
    # np.fix,
    # np.floor,
    # np.floor,
    # np.frac,
    # np.isfinite,
    # np.isinf,
    # np.isnan,
    # np.isneginf,
    # np.isposinf,
    # np.isreal,
    # np.lgamma,
    # np.log,
    # np.log10,
    # np.log1p,
    # np.log2,
    # np.logical_not,
    # np.logit,
    # np.nan_to_num,
    # np.neg,
    # np.negative,
    # np.positive,
    # np.rad2deg,
    # np.real,
    # np.reciprocal,
    # np.round,
    # np.rsqrt,
    # np.sigmoid,
    # np.sign,
    # np.sin,
    # np.sinc,
    # np.sinh,
    # np.sqrt,
    # np.sqrt,
    # np.square,
    # np.tan,
    # np.tanh,
    # np.trunc,
)

PAIRWISE_FUNCTIONS = (
    # partial(np.max, dim=0),
    # partial(np.max, dim=1),
    # partial(np.min, dim=0),
    # partial(np.min, dim=1),
    np.add,
    np.divide,
    np.equal,
    np.floor_divide,
    np.fmod,
    np.greater,
    np.greater_equal,
    np.less,
    np.less_equal,
    np.logaddexp,
    np.logical_and,
    np.logical_or,
    np.logical_xor,
    np.maximum,
    np.minimum,
    np.multiply,
    np.not_equal,
    np.nextafter,
    np.not_equal,
    np.remainder,
    np.subtract,
    np.true_divide,
)


@mark.parametrize("func", UNARY_FUNCTIONS)
@mark.parametrize("cls", BATCH_CLASSES)
def test_unary(func: Callable, cls: type[BatchedArray]) -> None:
    array = np.random.rand(2, 3) * 2.0
    assert func(cls(array)).allclose(cls(func(array)), equal_nan=True)


@mark.parametrize("func", UNARY_FUNCTIONS)
def test_unary_batched_array_custom_dims(func: Callable) -> None:
    array = np.random.rand(2, 3) * 2.0
    assert func(BatchedArray(array, batch_dim=1)).allclose(
        BatchedArray(func(array), batch_dim=1), equal_nan=True
    )


@mark.parametrize("func", PAIRWISE_FUNCTIONS)
@mark.parametrize("cls", BATCH_CLASSES)
def test_pairwise(func: Callable, cls: type[BatchedArray]) -> None:
    tensor1 = np.random.rand(2, 3)
    tensor2 = np.random.rand(2, 3) + 1.0
    assert func(cls(tensor1), cls(tensor2)).allclose(cls(func(tensor1, tensor2)), equal_nan=True)


@mark.parametrize("func", PAIRWISE_FUNCTIONS)
def test_pairwise_batched_tensor_custom_dims(func: Callable) -> None:
    tensor1 = np.random.rand(2, 3)
    tensor2 = np.random.rand(2, 3) + 1.0
    assert func(BatchedArray(tensor1, batch_dim=1), BatchedArray(tensor2, batch_dim=1)).allclose(
        BatchedArray(func(tensor1, tensor2), batch_dim=1), equal_nan=True
    )
