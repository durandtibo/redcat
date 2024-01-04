from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
from pytest import mark

from redcat.ba import BatchedArray

BATCH_CLASSES = (BatchedArray,)
DTYPES = (int, float)
SHAPES = ((2, 3), (2, 3, 4), (2, 3, 4, 5))


@mark.parametrize("cls", BATCH_CLASSES)
@mark.parametrize("dtype", DTYPES)
def test_array_dtype(cls: type[np.ndarray], dtype: np.dtype) -> None:
    array = np.ones((2, 3), dtype=dtype)
    assert cls(array).dtype == array.dtype


@mark.parametrize("cls", BATCH_CLASSES)
@mark.parametrize("shape", SHAPES)
def test_array_ndim(cls: type[np.ndarray], shape: tuple[int, ...]) -> None:
    array = np.ones(shape)
    assert cls(array).ndim == array.ndim


@mark.parametrize("cls", BATCH_CLASSES)
@mark.parametrize("shape", SHAPES)
def test_array_shape(cls: type[np.ndarray], shape: tuple[int, ...]) -> None:
    array = np.ones(shape)
    assert cls(array).shape == array.shape


@mark.parametrize("cls", BATCH_CLASSES)
@mark.parametrize("shape", SHAPES)
def test_array_size(cls: type[np.ndarray], shape: tuple[int, ...]) -> None:
    array = np.ones(shape)
    assert cls(array).size == array.size


UNARY_FUNCTIONS = (
    # np.arctan2,
    # np.max,
    # np.min,
    # np.rsqrt,
    # np.sigmoid,
    # np.sinc,
    # partial(np.clip, a_min=0.1, a_max=0.5),
    # partial(np.select, axis=0, index=0),
    # partial(np.unsqueeze, axis=-1),
    # partial(np.unsqueeze, axis=0),
    np.abs,
    np.angle,
    np.arccos,
    np.arccosh,
    np.arcsin,
    np.arcsinh,
    np.arctan,
    np.arctanh,
    np.ceil,
    np.cos,
    np.cosh,
    np.deg2rad,
    np.exp,
    np.exp2,
    np.floor,
    np.floor,
    np.isfinite,
    np.isinf,
    np.isnan,
    np.isneginf,
    np.isposinf,
    np.isreal,
    np.log,
    np.log10,
    np.log1p,
    np.log2,
    np.logical_not,
    np.nan_to_num,
    np.negative,
    np.positive,
    np.rad2deg,
    np.real,
    np.reciprocal,
    np.round,
    np.sign,
    np.sin,
    np.sinh,
    np.sqrt,
    np.sqrt,
    np.square,
    np.tan,
    np.tanh,
    np.trunc,
    partial(np.argsort, axis=0),
    partial(np.argsort, axis=1),
    partial(np.cumprod, axis=0),
    partial(np.cumprod, axis=0, dtype=float),
    partial(np.cumprod, axis=0, dtype=int),
    partial(np.cumprod, axis=1),
    partial(np.cumsum, axis=0),
    partial(np.cumsum, axis=0, dtype=float),
    partial(np.cumsum, axis=0, dtype=int),
    partial(np.cumsum, axis=1),
    partial(np.max, axis=2),
    partial(np.min, axis=2),
)

PAIRWISE_FUNCTIONS = (
    np.add,
    np.divide,
    # np.divmod,
    np.equal,
    np.float_power,
    np.floor_divide,
    np.fmod,
    np.greater,
    np.greater_equal,
    np.heaviside,
    np.less,
    np.less_equal,
    np.logaddexp,
    np.logaddexp,
    np.logaddexp2,
    np.logical_and,
    np.logical_or,
    np.logical_xor,
    np.maximum,
    np.minimum,
    np.mod,
    np.multiply,
    np.nextafter,
    np.not_equal,
    np.not_equal,
    np.power,
    np.remainder,
    np.subtract,
    np.true_divide,
)


@mark.parametrize("func", UNARY_FUNCTIONS)
@mark.parametrize("cls", BATCH_CLASSES)
def test_unary(func: Callable, cls: type[np.ndarray]) -> None:
    array = np.random.rand(2, 3, 4) * 2.0
    assert func(cls(array)).allclose(cls(func(array)), equal_nan=True)


@mark.parametrize("func", UNARY_FUNCTIONS)
def test_unary_batched_array_custom_axis(func: Callable) -> None:
    array = np.random.rand(2, 3, 4) * 2.0
    assert func(BatchedArray(array, batch_axis=1)).allclose(
        BatchedArray(func(array), batch_axis=1), equal_nan=True
    )


@mark.parametrize("func", PAIRWISE_FUNCTIONS)
@mark.parametrize("cls", BATCH_CLASSES)
def test_pairwise(func: Callable, cls: type[np.ndarray]) -> None:
    array1 = np.random.rand(2, 3, 4)
    array2 = np.random.rand(2, 3, 4) + 1.0
    assert func(cls(array1), cls(array2)).allclose(cls(func(array1, array2)), equal_nan=True)


@mark.parametrize("func", PAIRWISE_FUNCTIONS)
def test_pairwise_batched_array_custom_axis(func: Callable) -> None:
    array1 = np.random.rand(2, 3)
    array2 = np.random.rand(2, 3) + 1.0
    assert func(BatchedArray(array1, batch_axis=1), BatchedArray(array2, batch_axis=1)).allclose(
        BatchedArray(func(array1, array2), batch_axis=1), equal_nan=True
    )


# TODO: uncomment after the bug is fixed
# @mark.parametrize("func", PAIRWISE_FUNCTIONS)
# def test_pairwise_batched_array_incorrect_axis(func: Callable) -> None:
#     array1 = np.random.rand(2, 3)
#     array2 = np.random.rand(2, 3) + 1.0
#     with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
#         func(BatchedArray(array1), BatchedArray(array2, batch_axis=1))
