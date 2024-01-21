from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from coola import objects_are_equal

from redcat.ba.testing import FunctionCheck, normal_arrays, uniform_arrays


@dataclass
class ArraySeqShape:
    in_shape: int | Sequence[int]
    out_shape: tuple[int, ...]
    n: int


ARRAY_SEQ_SHAPES = [
    pytest.param(ArraySeqShape(in_shape=6, out_shape=(6,), n=1), id="1 1d array"),
    pytest.param(ArraySeqShape(in_shape=4, out_shape=(4,), n=2), id="2 1d arrays"),
    pytest.param(ArraySeqShape(in_shape=4, out_shape=(4,), n=3), id="3 1d arrays"),
    pytest.param(ArraySeqShape(in_shape=(2, 3), out_shape=(2, 3), n=1), id="1 2d array"),
    pytest.param(ArraySeqShape(in_shape=(2, 3), out_shape=(2, 3), n=2), id="2 2d arrays"),
    pytest.param(ArraySeqShape(in_shape=(2, 3), out_shape=(2, 3), n=3), id="3 2d arrays"),
    pytest.param(ArraySeqShape(in_shape=(2, 3, 4), out_shape=(2, 3, 4), n=1), id="1 3d array"),
    pytest.param(ArraySeqShape(in_shape=(2, 3, 4), out_shape=(2, 3, 4), n=2), id="2 3d arrays"),
    pytest.param(ArraySeqShape(in_shape=(2, 3, 4), out_shape=(2, 3, 4), n=3), id="3 3d arrays"),
    pytest.param(ArraySeqShape(in_shape=(6,), out_shape=(6,), n=1), id="tuple shape"),
    pytest.param(ArraySeqShape(in_shape=[6], out_shape=(6,), n=1), id="list shape"),
]

FUCTION_ARRAYS = [normal_arrays, uniform_arrays]


@pytest.mark.parametrize("array_seq_shape", ARRAY_SEQ_SHAPES)
def test_make_randn_arrays(array_seq_shape: ArraySeqShape) -> None:
    arrays = normal_arrays(shape=array_seq_shape.in_shape, n=array_seq_shape.n)
    assert len(arrays) == array_seq_shape.n
    assert all([arr.shape == array_seq_shape.out_shape for arr in arrays])


@pytest.mark.parametrize("array_seq_shape", ARRAY_SEQ_SHAPES)
def test_uniform_arrays(array_seq_shape: ArraySeqShape) -> None:
    arrays = uniform_arrays(shape=array_seq_shape.in_shape, n=array_seq_shape.n)
    assert len(arrays) == array_seq_shape.n
    assert all([arr.shape == array_seq_shape.out_shape for arr in arrays])
    assert all([arr.min() >= 0.0 for arr in arrays])
    assert all([arr.max() < 1.0 for arr in arrays])


@pytest.mark.parametrize(("low", "high"), [(-1.0, 1.0), (2.0, 5.0), (1.0, 100)])
def test_uniform_arrays_low_high(low: float, high: float) -> None:
    arrays = uniform_arrays(shape=(4, 10), n=2, low=low, high=high)
    assert len(arrays) == 2
    assert all([arr.shape == (4, 10) for arr in arrays])
    assert all([arr.min() >= low for arr in arrays])
    assert all([arr.max() < high for arr in arrays])


@pytest.mark.parametrize("function", FUCTION_ARRAYS)
def test_uniform_arrays_same_random_seed(function: Callable) -> None:
    assert objects_are_equal(
        function(shape=(4, 10), n=2, rng=np.random.default_rng(1)),
        function(shape=(4, 10), n=2, rng=np.random.default_rng(1)),
    )


@pytest.mark.parametrize("function", FUCTION_ARRAYS)
def test_uniform_arrays_different_random_seeds(function: Callable) -> None:
    assert not objects_are_equal(
        function(shape=(4, 10), n=2, rng=np.random.default_rng(1)),
        function(shape=(4, 10), n=2, rng=np.random.default_rng(2)),
    )


#########################
#     FunctionCheck     #
#########################


def test_function_check_add() -> None:
    check = FunctionCheck(np.add, nin=2, nout=1)
    assert check.function is np.add
    assert check.nin == 2
    assert check.nout == 1


def test_function_check_get_arrays() -> None:
    arrays = (np.random.rand(2, 3), np.random.randn(2, 3))
    assert objects_are_equal(
        FunctionCheck(np.add, nin=2, nout=1, arrays=arrays).get_arrays(), arrays
    )


def test_function_check_get_arrays_none() -> None:
    arrays = FunctionCheck(np.add, nin=2, nout=1).get_arrays()
    assert len(arrays) == 2
    assert all([arr.shape == (4, 10) for arr in arrays])


def test_function_check_create_ufunc_add() -> None:
    check = FunctionCheck.create_ufunc(np.add)
    assert check.function is np.add
    assert check.nin == 2
    assert check.nout == 1
    arrays = check.get_arrays()
    assert len(arrays) == 2
    assert all([arr.shape == (4, 10) for arr in arrays])


def test_function_check_create_ufunc_add_arrays() -> None:
    arrays = (np.random.rand(2, 3), np.random.randn(2, 3))
    check = FunctionCheck.create_ufunc(np.add, arrays)
    assert check.function is np.add
    assert check.nin == 2
    assert check.nout == 1
    objects_are_equal(check.get_arrays(), arrays)
