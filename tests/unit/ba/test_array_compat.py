from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
from coola import objects_are_allclose
from pytest import mark

from redcat.ba import BatchedArray
from redcat.ba.testing import FunctionCheck, normal_arrays, uniform_arrays

BATCH_CLASSES = (BatchedArray, partial(BatchedArray, batch_axis=1))
DTYPES = (int, float)
SHAPES = ((2, 3), (2, 3, 4), (2, 3, 4, 5))

SHAPE = (4, 10)


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


TRIGONOMETRIC_FUNCTIONS = [
    FunctionCheck.create_ufunc(np.cos, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(
        np.arccos, arrays=uniform_arrays(shape=SHAPE, n=1, low=-1.0, high=1.0)
    ),
    FunctionCheck.create_ufunc(np.cosh, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(
        np.arccosh, arrays=uniform_arrays(shape=SHAPE, n=1, low=1.0, high=100.0)
    ),
    FunctionCheck.create_ufunc(np.sin, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(
        np.arcsin, arrays=uniform_arrays(shape=SHAPE, n=1, low=-1.0, high=1.0)
    ),
    FunctionCheck.create_ufunc(np.sinh, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.arcsinh, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.tan, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.arctan, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.tanh, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(
        np.arctanh, arrays=uniform_arrays(shape=SHAPE, n=1, low=-0.999, high=0.999)
    ),
]

FLOATING_FUNCTIONS = [
    FunctionCheck.create_ufunc(np.ceil, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.copysign, arrays=normal_arrays(shape=SHAPE, n=2)),
    FunctionCheck.create_ufunc(np.fabs, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.floor, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.fmod, arrays=normal_arrays(shape=SHAPE, n=2)),
    FunctionCheck.create_ufunc(np.frexp, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.isfinite, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.isinf, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.isnan, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(
        np.isnat,
        arrays=(
            np.array(
                [
                    ["2007-07-13", "2007-07-14"],
                    ["2006-01-13", "2006-01-14"],
                    ["2010-08-13", "2010-08-14"],
                ],
                dtype="datetime64",
            ),
        ),
    ),
    FunctionCheck.create_ufunc(
        np.ldexp, arrays=(np.random.rand(*SHAPE), np.random.randint(0, 10, size=SHAPE))
    ),
    FunctionCheck.create_ufunc(np.modf, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.nextafter, arrays=normal_arrays(shape=SHAPE, n=2)),
    FunctionCheck.create_ufunc(np.signbit, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.spacing, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.trunc, arrays=normal_arrays(shape=SHAPE, n=1)),
]

FUNCTIONS = TRIGONOMETRIC_FUNCTIONS + FLOATING_FUNCTIONS


@mark.parametrize("func_check", FUNCTIONS)
@mark.parametrize("cls", BATCH_CLASSES)
def test_array_checks(func_check: FunctionCheck, cls: type[np.ndarray]) -> None:
    func = func_check.function
    arrays = func_check.get_arrays()
    outputs = func(*[cls(arr) for arr in arrays])
    outs = func(*arrays)
    expected_outputs = tuple([cls(out) for out in outs]) if func_check.nout > 1 else cls(outs)
    assert objects_are_allclose(outputs, expected_outputs)


UNARY_FUNCTIONS = [
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
    np.absolute,
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
]

PAIRWISE_FUNCTIONS = [
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
    np.isclose,
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
]

IN1_TO_NDARRAY_FUNCTIONS = [
    np.argmax,
    np.argmin,
    np.max,
    np.mean,
    np.median,
    np.min,
    partial(np.argmax, axis=0),
    partial(np.argmax, axis=1),
    partial(np.argmax, axis=None),
    partial(np.argmin, axis=0),
    partial(np.argmin, axis=1),
    partial(np.argmin, axis=None),
    partial(np.cumprod, axis=None),
    partial(np.cumsum, axis=None),
    partial(np.max, axis=0),
    partial(np.max, axis=1),
    partial(np.max, axis=None),
    partial(np.mean, axis=0),
    partial(np.mean, axis=1),
    partial(np.mean, axis=None),
    partial(np.median, axis=0),
    partial(np.median, axis=1),
    partial(np.median, axis=None),
    partial(np.min, axis=0),
    partial(np.min, axis=1),
    partial(np.min, axis=None),
    partial(np.nancumprod, axis=None),
    partial(np.nancumsum, axis=None),
    # partial(np.argsort, axis=None),
    # partial(np.sort, axis=None),
]

IN1_FUNCTIONS = UNARY_FUNCTIONS
IN2_FUNCTIONS = PAIRWISE_FUNCTIONS


@mark.parametrize("func", IN1_FUNCTIONS)
@mark.parametrize("cls", BATCH_CLASSES)
def test_unary(func: Callable, cls: type[np.ndarray]) -> None:
    array = np.random.rand(2, 3) * 2.0
    assert objects_are_allclose(func(cls(array)), cls(func(array)), equal_nan=True)


@mark.parametrize("func", IN1_TO_NDARRAY_FUNCTIONS)
@mark.parametrize("cls", BATCH_CLASSES)
def test_unary_to_ndarray(func: Callable, cls: type[np.ndarray]) -> None:
    array = np.random.rand(2, 3) * 2.0
    assert objects_are_allclose(func(cls(array)), func(array), equal_nan=True)


@mark.parametrize("func", UNARY_FUNCTIONS)
def test_unary_batched_array_custom_axis(func: Callable) -> None:
    array = np.random.rand(2, 3, 4) * 2.0
    assert objects_are_allclose(
        func(BatchedArray(array, batch_axis=1)),
        BatchedArray(func(array), batch_axis=1),
        equal_nan=True,
    )


@mark.parametrize("func", IN2_FUNCTIONS)
@mark.parametrize("cls", BATCH_CLASSES)
def test_pairwise(func: Callable, cls: type[np.ndarray]) -> None:
    array1 = np.random.rand(2, 3, 4)
    array2 = np.random.rand(2, 3, 4) + 1.0
    assert objects_are_allclose(
        func(cls(array1), cls(array2)), cls(func(array1, array2)), equal_nan=True
    )


@mark.parametrize("func", IN2_FUNCTIONS)
def test_pairwise_batched_array_custom_axis(func: Callable) -> None:
    array1 = np.random.rand(2, 3)
    array2 = np.random.rand(2, 3) + 1.0
    assert objects_are_allclose(
        func(BatchedArray(array1, batch_axis=1), BatchedArray(array2, batch_axis=1)),
        BatchedArray(func(array1, array2), batch_axis=1),
        equal_nan=True,
    )


# TODO: uncomment after the bug is fixed
# @mark.parametrize("func", PAIRWISE_FUNCTIONS)
# def test_pairwise_batched_array_incorrect_axis(func: Callable) -> None:
#     array1 = np.random.rand(2, 3)
#     array2 = np.random.rand(2, 3) + 1.0
#     with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
#         func(BatchedArray(array1), BatchedArray(array2, batch_axis=1))
