from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from coola import objects_are_allclose
from pytest import mark

from redcat.ba import BatchedArray
from redcat.ba.testing import (
    FunctionCheck,
    normal_arrays,
    uniform_arrays,
    uniform_int_arrays,
)

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


MATH_FUNCS = [
    FunctionCheck(
        partial(np.cumprod, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
    FunctionCheck(
        partial(np.cumsum, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
    FunctionCheck(
        partial(np.nancumprod, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
    FunctionCheck(
        partial(np.nancumsum, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
    FunctionCheck(np.sort, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(partial(np.sort, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(np.argsort, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(
        partial(np.argsort, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
]

MATH_UFUNCS = [
    FunctionCheck.create_ufunc(np.add),
    FunctionCheck.create_ufunc(np.subtract),
    FunctionCheck.create_ufunc(np.multiply),
    FunctionCheck.create_ufunc(
        np.divide, arrays=uniform_arrays(shape=SHAPE, n=2, low=1.0, high=5.0)
    ),
    FunctionCheck.create_ufunc(
        np.true_divide, arrays=uniform_arrays(shape=SHAPE, n=2, low=1.0, high=5.0)
    ),
    FunctionCheck.create_ufunc(
        np.floor_divide, arrays=uniform_arrays(shape=SHAPE, n=2, low=1.0, high=5.0)
    ),
    FunctionCheck.create_ufunc(
        np.remainder, arrays=uniform_arrays(shape=SHAPE, n=2, low=1.0, high=5.0)
    ),
    FunctionCheck.create_ufunc(np.mod, arrays=uniform_arrays(shape=SHAPE, n=2, low=1.0, high=5.0)),
    FunctionCheck.create_ufunc(np.fmod, arrays=uniform_arrays(shape=SHAPE, n=2, low=1.0, high=5.0)),
    FunctionCheck.create_ufunc(
        np.divmod, arrays=uniform_arrays(shape=SHAPE, n=2, low=1.0, high=5.0)
    ),
    FunctionCheck.create_ufunc(np.logaddexp),
    FunctionCheck.create_ufunc(np.logaddexp2),
    FunctionCheck.create_ufunc(np.negative),
    FunctionCheck.create_ufunc(np.positive),
    FunctionCheck.create_ufunc(
        np.power, arrays=(np.random.rand(*SHAPE), np.random.randint(low=1, high=5, size=SHAPE))
    ),
    FunctionCheck.create_ufunc(
        np.float_power,
        arrays=(np.random.rand(*SHAPE), np.random.randint(low=1, high=5, size=SHAPE)),
    ),
    FunctionCheck.create_ufunc(np.abs),
    FunctionCheck.create_ufunc(np.absolute),
    FunctionCheck.create_ufunc(np.fabs),
    FunctionCheck.create_ufunc(np.rint),
    FunctionCheck.create_ufunc(np.sign),
    FunctionCheck.create_ufunc(np.heaviside),
    FunctionCheck.create_ufunc(np.conj),
    FunctionCheck.create_ufunc(np.conjugate),
    FunctionCheck.create_ufunc(np.exp),
    FunctionCheck.create_ufunc(np.exp2),
    FunctionCheck.create_ufunc(np.expm1),
    FunctionCheck.create_ufunc(np.log, arrays=uniform_arrays(shape=SHAPE, n=1, low=1e-8)),
    FunctionCheck.create_ufunc(np.log10, arrays=uniform_arrays(shape=SHAPE, n=1, low=1e-8)),
    FunctionCheck.create_ufunc(np.log1p, arrays=uniform_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.log2, arrays=uniform_arrays(shape=SHAPE, n=1, low=1e-8)),
    FunctionCheck.create_ufunc(np.sqrt, arrays=uniform_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.square),
    FunctionCheck.create_ufunc(np.cbrt),
    FunctionCheck.create_ufunc(np.reciprocal),
    FunctionCheck.create_ufunc(np.gcd, arrays=uniform_int_arrays(shape=SHAPE, n=2)),
    FunctionCheck.create_ufunc(np.lcm, arrays=uniform_int_arrays(shape=SHAPE, n=2)),
]

TRIGONOMETRIC_UFUNCS = [
    FunctionCheck.create_ufunc(
        np.arccos, arrays=uniform_arrays(shape=SHAPE, n=1, low=-1.0, high=1.0)
    ),
    FunctionCheck.create_ufunc(
        np.arccosh, arrays=uniform_arrays(shape=SHAPE, n=1, low=1.0, high=100.0)
    ),
    FunctionCheck.create_ufunc(
        np.arcsin, arrays=uniform_arrays(shape=SHAPE, n=1, low=-1.0, high=1.0)
    ),
    FunctionCheck.create_ufunc(np.arcsinh),
    FunctionCheck.create_ufunc(np.arctan),
    FunctionCheck.create_ufunc(
        np.arctanh, arrays=uniform_arrays(shape=SHAPE, n=1, low=-0.999, high=0.999)
    ),
    FunctionCheck.create_ufunc(np.cos),
    FunctionCheck.create_ufunc(np.cosh),
    FunctionCheck.create_ufunc(np.deg2rad),
    FunctionCheck.create_ufunc(np.degrees),
    FunctionCheck.create_ufunc(np.hypot),
    FunctionCheck.create_ufunc(np.rad2deg),
    FunctionCheck.create_ufunc(np.radians),
    FunctionCheck.create_ufunc(np.sin),
    FunctionCheck.create_ufunc(np.sinh),
    FunctionCheck.create_ufunc(np.tan),
    FunctionCheck.create_ufunc(np.tanh),
]

BIT_UFUNCS = [
    FunctionCheck.create_ufunc(np.bitwise_and, arrays=uniform_int_arrays(shape=SHAPE, n=2)),
    FunctionCheck.create_ufunc(np.bitwise_or, arrays=uniform_int_arrays(shape=SHAPE, n=2)),
    FunctionCheck.create_ufunc(np.bitwise_xor, arrays=uniform_int_arrays(shape=SHAPE, n=2)),
    FunctionCheck.create_ufunc(np.invert, arrays=uniform_int_arrays(shape=SHAPE, n=1)),
    FunctionCheck.create_ufunc(np.left_shift, arrays=uniform_int_arrays(shape=SHAPE, n=2)),
    FunctionCheck.create_ufunc(np.right_shift, arrays=uniform_int_arrays(shape=SHAPE, n=2)),
]

COMPARISON_UFUNCS = [
    FunctionCheck.create_ufunc(np.equal),
    FunctionCheck.create_ufunc(np.fmax),
    FunctionCheck.create_ufunc(np.fmin),
    FunctionCheck.create_ufunc(np.greater),
    FunctionCheck.create_ufunc(np.greater_equal),
    FunctionCheck.create_ufunc(np.less),
    FunctionCheck.create_ufunc(np.less_equal),
    FunctionCheck.create_ufunc(np.logical_and),
    FunctionCheck.create_ufunc(np.logical_not),
    FunctionCheck.create_ufunc(np.logical_or),
    FunctionCheck.create_ufunc(np.logical_xor),
    FunctionCheck.create_ufunc(np.maximum),
    FunctionCheck.create_ufunc(np.minimum),
    FunctionCheck.create_ufunc(np.not_equal),
]

FLOATING_UFUNCS = [
    FunctionCheck.create_ufunc(np.ceil),
    FunctionCheck.create_ufunc(np.copysign),
    FunctionCheck.create_ufunc(np.fabs),
    FunctionCheck.create_ufunc(np.floor),
    FunctionCheck.create_ufunc(np.fmod),
    FunctionCheck.create_ufunc(np.frexp),
    FunctionCheck.create_ufunc(np.isfinite),
    FunctionCheck.create_ufunc(np.isinf),
    # FunctionCheck(np.isneginf, nin=1, nout=1),
    # FunctionCheck(np.isposinf, nin=1, nout=1),
    FunctionCheck.create_ufunc(np.isnan),
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
    FunctionCheck.create_ufunc(np.modf),
    FunctionCheck.create_ufunc(np.nextafter),
    FunctionCheck.create_ufunc(np.signbit),
    FunctionCheck.create_ufunc(np.spacing),
    FunctionCheck.create_ufunc(np.trunc),
]

FUNCS = MATH_FUNCS
UFUNCS = MATH_UFUNCS + TRIGONOMETRIC_UFUNCS + BIT_UFUNCS + COMPARISON_UFUNCS + FLOATING_UFUNCS
FUNCTIONS = FUNCS + UFUNCS

IN1_FUNCTIONS = [fc for fc in FUNCTIONS if fc.nin == 1]
IN2_FUNCTIONS = [fc for fc in FUNCTIONS if fc.nin == 2]

BATCH_TO_ARRAY_FUNCS = [
    FunctionCheck(np.cumprod, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(np.cumsum, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(np.diff, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(np.nancumprod, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(np.nancumsum, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(np.nanprod, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(
        partial(np.nanprod, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
    FunctionCheck(np.nansum, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(
        partial(np.nansum, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
    FunctionCheck(np.prod, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(partial(np.prod, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(np.sum, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(partial(np.sum, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(np.argmax, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(
        partial(np.argmax, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
    FunctionCheck(np.argmin, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(
        partial(np.argmin, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
    FunctionCheck(np.nanargmax, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(
        partial(np.nanargmax, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
    FunctionCheck(np.nanargmin, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(
        partial(np.nanargmin, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
    FunctionCheck(np.nanmin, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(
        partial(np.nanmin, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
    FunctionCheck(np.nanmax, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(
        partial(np.nanmax, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
    FunctionCheck(np.min, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(partial(np.min, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(np.max, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(partial(np.max, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(np.mean, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(partial(np.mean, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(np.median, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(
        partial(np.median, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
    FunctionCheck(np.nanmean, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(
        partial(np.nanmean, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
    FunctionCheck(np.nanmedian, nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)),
    FunctionCheck(
        partial(np.nanmedian, axis=0), nin=1, nout=1, arrays=normal_arrays(shape=SHAPE, n=1)
    ),
]


@mark.parametrize("func_check", FUNCTIONS)
@mark.parametrize("cls", BATCH_CLASSES)
def test_batched_array(func_check: FunctionCheck, cls: type[np.ndarray]) -> None:
    func = func_check.function
    arrays = func_check.get_arrays()
    outputs = func(*[cls(arr) for arr in arrays])
    outs = func(*arrays)
    expected_outputs = tuple([cls(out) for out in outs]) if func_check.nout > 1 else cls(outs)
    assert objects_are_allclose(outputs, expected_outputs)


@mark.parametrize("func_check", IN2_FUNCTIONS)
@mark.parametrize("cls", BATCH_CLASSES)
def test_in2_batched_array_with_array(func_check: FunctionCheck, cls: type[np.ndarray]) -> None:
    func = func_check.function
    arrays = func_check.get_arrays()
    outputs1 = func(cls(arrays[0]), arrays[1])
    outputs2 = func(arrays[0], cls(arrays[1]))
    outs = func(*arrays)
    expected_outputs = tuple([cls(out) for out in outs]) if func_check.nout > 1 else cls(outs)
    assert objects_are_allclose(outputs1, expected_outputs)
    assert objects_are_allclose(outputs2, expected_outputs)


@mark.parametrize("func_check", IN2_FUNCTIONS)
def test_in2_batched_array_incorrect_axes(func_check: FunctionCheck) -> None:
    func = func_check.function
    arrays = func_check.get_arrays()
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        func(BatchedArray(arrays[0]), BatchedArray(arrays[1], batch_axis=1))


@mark.parametrize("func_check", BATCH_TO_ARRAY_FUNCS)
@mark.parametrize("cls", BATCH_CLASSES)
def test_batched_array_to_array(func_check: FunctionCheck, cls: type[np.ndarray]) -> None:
    func = func_check.function
    arrays = func_check.get_arrays()
    outputs = func(*[cls(arr) for arr in arrays])
    assert objects_are_allclose(outputs, func(*arrays))
