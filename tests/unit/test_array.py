from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any
from unittest.mock import patch

import numpy as np
from coola import objects_are_equal
from numpy import ndarray
from pytest import mark, raises

from redcat.array import BatchedArray, get_div_rounding_operator

DTYPES = (bool, int, float)


def test_batched_array_repr() -> None:
    assert repr(BatchedArray(np.arange(3))) == "array([0, 1, 2], batch_dim=0)"


@mark.parametrize(
    "data",
    (
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
    ),
)
def test_batched_array_init_data(data: Any) -> None:
    assert np.array_equal(
        BatchedArray(data).data, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    )


def test_batched_array_init_incorrect_data_dim() -> None:
    with raises(RuntimeError, match=r"data needs at least 1 dimensions \(received: 0\)"):
        BatchedArray(np.array(2))


@mark.parametrize("batch_dim", (-1, 1, 2))
def test_batched_array_init_incorrect_batch_dim(batch_dim: int) -> None:
    with raises(
        RuntimeError, match=r"Incorrect batch_dim \(.*\) but the value should be in \[0, 0\]"
    ):
        BatchedArray(np.ones((2,)), batch_dim=batch_dim)


@mark.parametrize("batch_size", (1, 2))
def test_batched_array_batch_size(batch_size: int) -> None:
    assert BatchedArray(np.arange(batch_size)).batch_size == batch_size


def test_batched_array_data() -> None:
    assert np.array_equal(BatchedArray(np.arange(3)).data, np.array([0, 1, 2]))


def test_batched_array_dtype() -> None:
    assert BatchedArray(np.ones((2, 3))).dtype == float


def test_batched_array_shape() -> None:
    assert BatchedArray(np.ones((2, 3))).shape == (2, 3)


def test_batched_array_dim() -> None:
    assert BatchedArray(np.ones((2, 3))).dim() == 2


def test_batched_array_ndimension() -> None:
    assert BatchedArray(np.ones((2, 3))).ndimension() == 2


def test_batched_array_numel() -> None:
    assert BatchedArray(np.ones((2, 3))).numel() == 6


#################################
#     Conversion operations     #
#################################


def test_batched_array_astype() -> None:
    assert (
        BatchedArray(np.ones((2, 3))).astype(bool).equal(BatchedArray(np.ones((2, 3), dtype=bool)))
    )


def test_batched_array_astype_custom_dim() -> None:
    assert (
        BatchedArray(np.ones((2, 3)), batch_dim=1)
        .astype(bool)
        .equal(BatchedArray(np.ones((2, 3), dtype=bool), batch_dim=1))
    )


def test_batched_array_to() -> None:
    assert (
        BatchedArray(np.ones((2, 3)))
        .to(dtype=bool)
        .equal(BatchedArray(np.ones((2, 3), dtype=bool)))
    )


def test_batched_array_to_custom_dim() -> None:
    assert (
        BatchedArray(np.ones((2, 3)), batch_dim=1)
        .to(dtype=bool)
        .equal(BatchedArray(np.ones((2, 3), dtype=bool), batch_dim=1))
    )


###############################
#     Creation operations     #
###############################


def test_batched_array_clone() -> None:
    batch = BatchedArray(np.ones((2, 3)))
    clone = batch.clone()
    batch.add_(1)
    assert batch.equal(BatchedArray(np.full((2, 3), 2.0)))
    assert clone.equal(BatchedArray(np.ones((2, 3))))


def test_batched_array_clone_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.ones((2, 3)), batch_dim=1)
        .clone()
        .equal(BatchedArray(np.ones((2, 3)), batch_dim=1))
    )


def test_batched_array_copy() -> None:
    batch = BatchedArray(np.ones((2, 3)))
    clone = batch.copy()
    batch.add_(1)
    assert batch.equal(BatchedArray(np.full((2, 3), 2.0)))
    assert clone.equal(BatchedArray(np.ones((2, 3))))


def test_batched_array_copy_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.ones((2, 3)), batch_dim=1)
        .copy()
        .equal(BatchedArray(np.ones((2, 3)), batch_dim=1))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_array_empty_like(dtype: np.dtype) -> None:
    batch = BatchedArray(np.zeros((2, 3), dtype=dtype)).empty_like()
    assert isinstance(batch, BatchedArray)
    assert batch.data.shape == (2, 3)
    assert batch.dtype == dtype


@mark.parametrize("dtype", DTYPES)
def test_batched_array_empty_like_target_dtype(dtype: np.dtype) -> None:
    batch = BatchedArray(np.zeros((2, 3))).empty_like(dtype=dtype)
    assert isinstance(batch, BatchedArray)
    assert batch.data.shape == (2, 3)
    assert batch.dtype == dtype


def test_batched_array_empty_like_custom_batch_dim() -> None:
    batch = BatchedArray(np.zeros((3, 2)), batch_dim=1).empty_like()
    assert isinstance(batch, BatchedArray)
    assert batch.data.shape == (3, 2)
    assert batch.batch_dim == 1


@mark.parametrize("fill_value", (1.5, 2.0, -1.0))
def test_batched_array_full_like(fill_value: float) -> None:
    assert (
        BatchedArray(np.zeros((2, 3)))
        .full_like(fill_value)
        .equal(BatchedArray(np.full((2, 3), fill_value=fill_value)))
    )


def test_batched_array_full_like_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.zeros((3, 2)), batch_dim=1)
        .full_like(fill_value=2.0)
        .equal(BatchedArray(np.full((3, 2), fill_value=2.0), batch_dim=1))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_array_full_like_dtype(dtype: np.dtype) -> None:
    assert (
        BatchedArray(np.zeros((2, 3), dtype=dtype))
        .full_like(fill_value=2.0)
        .equal(BatchedArray(np.full((2, 3), fill_value=2.0, dtype=dtype)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_array_full_like_target_dtype(dtype: np.dtype) -> None:
    assert (
        BatchedArray(np.zeros((2, 3)))
        .full_like(fill_value=2.0, dtype=dtype)
        .equal(BatchedArray(np.full((2, 3), fill_value=2.0, dtype=dtype)))
    )


@mark.parametrize("fill_value", (1, 2.0, True))
def test_batched_array_new_full_fill_value(fill_value: float | int | bool) -> None:
    assert (
        BatchedArray(np.zeros((2, 3)))
        .new_full(fill_value)
        .equal(BatchedArray(np.full((2, 3), fill_value, dtype=float)))
    )


def test_batched_array_new_full_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.zeros((3, 2)), batch_dim=1)
        .new_full(2.0)
        .equal(BatchedArray(np.full((3, 2), 2.0), batch_dim=1))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_array_new_full_dtype(dtype: np.dtype) -> None:
    assert (
        BatchedArray(np.zeros((2, 3), dtype=dtype))
        .new_full(2.0)
        .equal(BatchedArray(np.full((2, 3), 2.0, dtype=dtype)))
    )


@mark.parametrize("batch_size", (1, 2))
def test_batched_array_new_full_custom_batch_size(batch_size: int) -> None:
    assert (
        BatchedArray(np.zeros((2, 3)))
        .new_full(2.0, batch_size=batch_size)
        .equal(BatchedArray(np.full((batch_size, 3), 2.0)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_array_new_full_custom_dtype(dtype: np.dtype) -> None:
    assert (
        BatchedArray(np.zeros((2, 3)))
        .new_full(2.0, dtype=dtype)
        .equal(BatchedArray(np.full((2, 3), 2.0, dtype=dtype)))
    )


def test_batched_array_new_ones() -> None:
    assert BatchedArray(np.zeros((2, 3))).new_ones().equal(BatchedArray(np.ones((2, 3))))


def test_batched_array_new_ones_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.zeros((3, 2)), batch_dim=1)
        .new_ones()
        .equal(BatchedArray(np.ones((3, 2)), batch_dim=1))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_array_new_ones_dtype(dtype: np.dtype) -> None:
    assert (
        BatchedArray(np.zeros((2, 3), dtype=dtype))
        .new_ones()
        .equal(BatchedArray(np.ones((2, 3), dtype=dtype)))
    )


@mark.parametrize("batch_size", (1, 2))
def test_batched_array_new_ones_custom_batch_size(batch_size: int) -> None:
    assert (
        BatchedArray(np.zeros((2, 3)))
        .new_ones(batch_size=batch_size)
        .equal(BatchedArray(np.ones((batch_size, 3))))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_array_new_ones_custom_dtype(dtype: np.dtype) -> None:
    assert (
        BatchedArray(np.zeros((2, 3)))
        .new_ones(dtype=dtype)
        .equal(BatchedArray(np.ones((2, 3), dtype=dtype)))
    )


def test_batched_array_new_zeros() -> None:
    assert BatchedArray(np.ones((2, 3))).new_zeros().equal(BatchedArray(np.zeros((2, 3))))


def test_batched_array_new_zeros_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.ones((3, 2)), batch_dim=1)
        .new_zeros()
        .equal(BatchedArray(np.zeros((3, 2)), batch_dim=1))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_array_new_zeros_dtype(dtype: np.dtype) -> None:
    assert (
        BatchedArray(np.ones((2, 3), dtype=dtype))
        .new_zeros()
        .equal(BatchedArray(np.zeros((2, 3), dtype=dtype)))
    )


@mark.parametrize("batch_size", (1, 2))
def test_batched_array_new_zeros_custom_batch_size(batch_size: int) -> None:
    assert (
        BatchedArray(np.ones((2, 3)))
        .new_zeros(batch_size=batch_size)
        .equal(BatchedArray(np.zeros((batch_size, 3))))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_array_new_zeros_custom_dtype(dtype: np.dtype) -> None:
    assert (
        BatchedArray(np.ones((2, 3)))
        .new_zeros(dtype=dtype)
        .equal(BatchedArray(np.zeros((2, 3), dtype=dtype)))
    )


def test_batched_array_ones_like() -> None:
    assert BatchedArray(np.zeros((2, 3))).ones_like().equal(BatchedArray(np.ones((2, 3))))


def test_batched_array_ones_like_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.zeros((3, 2)), batch_dim=1)
        .ones_like()
        .equal(BatchedArray(np.ones((3, 2)), batch_dim=1))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_array_ones_like_dtype(dtype: np.dtype) -> None:
    assert (
        BatchedArray(np.zeros((2, 3), dtype=dtype))
        .ones_like()
        .equal(BatchedArray(np.ones((2, 3), dtype=dtype)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_array_ones_like_target_dtype(dtype: np.dtype) -> None:
    assert (
        BatchedArray(np.zeros((2, 3)))
        .ones_like(dtype=dtype)
        .equal(BatchedArray(np.ones((2, 3), dtype=dtype)))
    )


def test_batched_array_zeros_like() -> None:
    assert BatchedArray(np.ones((2, 3))).zeros_like().equal(BatchedArray(np.zeros((2, 3))))


def test_batched_array_zeros_like_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.ones((3, 2)), batch_dim=1)
        .zeros_like()
        .equal(BatchedArray(np.zeros((3, 2)), batch_dim=1))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_array_zeros_like_dtype(dtype: np.dtype) -> None:
    assert (
        BatchedArray(np.ones((2, 3), dtype=dtype))
        .zeros_like()
        .equal(BatchedArray(np.zeros((2, 3), dtype=dtype)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_array_zeros_like_target_dtype(dtype: np.dtype) -> None:
    assert (
        BatchedArray(np.ones((2, 3)))
        .zeros_like(dtype=dtype)
        .equal(BatchedArray(np.zeros((2, 3), dtype=dtype)))
    )


#################################
#     Comparison operations     #
#################################


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 5), 5.0)),
        np.full((2, 5), 5.0),
        BatchedArray(np.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_array__ge__(other: BatchedArray | ndarray | int | float) -> None:
    assert (BatchedArray(np.arange(10).reshape(2, 5)) >= other).equal(
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [True, True, True, True, True]],
                dtype=bool,
            ),
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 5), 5.0)),
        np.full((2, 5), 5.0),
        BatchedArray(np.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_array__gt__(other: BatchedArray | ndarray | int | float) -> None:
    assert (BatchedArray(np.arange(10).reshape(2, 5)) > other).equal(
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [False, True, True, True, True]],
                dtype=bool,
            ),
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 5), 5.0)),
        np.full((2, 5), 5.0),
        BatchedArray(np.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_array__le__(other: BatchedArray | ndarray | int | float) -> None:
    assert (BatchedArray(np.arange(10).reshape(2, 5)) <= other).equal(
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [True, False, False, False, False]],
                dtype=bool,
            ),
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 5), 5.0)),
        np.full((2, 5), 5.0),
        BatchedArray(np.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_array__lt__(other: BatchedArray | ndarray | int | float) -> None:
    assert (BatchedArray(np.arange(10).reshape(2, 5)) < other).equal(
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [False, False, False, False, False]],
                dtype=bool,
            ),
        )
    )


def test_batched_array_allclose_true() -> None:
    assert BatchedArray(np.ones((2, 3))).allclose(BatchedArray(np.ones((2, 3))))


def test_batched_array_allclose_false_different_type() -> None:
    assert not BatchedArray(np.ones((2, 3), dtype=float)).allclose(np.zeros((2, 3), dtype=int))


def test_batched_array_allclose_false_different_data() -> None:
    assert not BatchedArray(np.ones((2, 3))).allclose(BatchedArray(np.zeros((2, 3))))


def test_batched_array_allclose_false_different_shape() -> None:
    assert not BatchedArray(np.ones((2, 3))).allclose(BatchedArray(np.ones((2, 3, 1))))


def test_batched_array_allclose_false_different_batch_dim() -> None:
    assert not BatchedArray(np.ones((2, 3))).allclose(BatchedArray(np.ones((2, 3)), batch_dim=1))


@mark.parametrize(
    "batch,atol",
    (
        (BatchedArray(np.ones((2, 3)) + 0.5), 1),
        (BatchedArray(np.ones((2, 3)) + 0.05), 1e-1),
        (BatchedArray(np.ones((2, 3)) + 5e-3), 1e-2),
    ),
)
def test_batched_array_allclose_true_atol(batch: BatchedArray, atol: float) -> None:
    assert BatchedArray(np.ones((2, 3))).allclose(batch, atol=atol, rtol=0)


@mark.parametrize(
    "batch,rtol",
    (
        (BatchedArray(np.ones((2, 3)) + 0.5), 1),
        (BatchedArray(np.ones((2, 3)) + 0.05), 1e-1),
        (BatchedArray(np.ones((2, 3)) + 5e-3), 1e-2),
    ),
)
def test_batched_array_allclose_true_rtol(batch: BatchedArray, rtol: float) -> None:
    assert BatchedArray(np.ones((2, 3))).allclose(batch, rtol=rtol)


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 5), 5.0)),
        np.full((2, 5), 5.0),
        BatchedArray(np.full((2, 1), 5.0)),
        5,
        5.0,
    ),
)
def test_batched_array_eq(other: BatchedArray | ndarray | int | float) -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .eq(other)
        .equal(
            BatchedArray(
                np.array(
                    [[False, False, False, False, False], [True, False, False, False, False]],
                    dtype=bool,
                ),
            )
        )
    )


def test_batched_array_eq_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_dim=1)
        .eq(BatchedArray(np.full((2, 5), 5), batch_dim=1))
        .equal(
            BatchedArray(
                np.array(
                    [[False, False, False, False, False], [True, False, False, False, False]],
                    dtype=bool,
                ),
                batch_dim=1,
            )
        )
    )


def test_batched_array_equal_true() -> None:
    assert BatchedArray(np.ones((2, 3))).equal(BatchedArray(np.ones((2, 3))))


def test_batched_array_equal_false_different_type() -> None:
    assert not BatchedArray(np.ones((2, 3), dtype=float)).equal(np.ones((2, 3), dtype=int))


def test_batched_array_equal_false_different_data() -> None:
    assert not BatchedArray(np.ones((2, 3))).equal(BatchedArray(np.zeros((2, 3))))


def test_batched_array_equal_false_different_shape() -> None:
    assert not BatchedArray(np.ones((2, 3))).equal(BatchedArray(np.ones((2, 3, 1))))


def test_batched_array_equal_false_different_batch_dim() -> None:
    assert not BatchedArray(np.ones((2, 3)), batch_dim=1).equal(BatchedArray(np.ones((2, 3))))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 5), 5.0)),
        np.full((2, 5), 5.0),
        BatchedArray(np.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_array_ge(other: BatchedArray | ndarray | int | float) -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .ge(other)
        .equal(
            BatchedArray(
                np.array(
                    [[False, False, False, False, False], [True, True, True, True, True]],
                    dtype=bool,
                ),
            )
        )
    )


def test_batched_array_ge_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_dim=1)
        .ge(BatchedArray(np.full((2, 5), 5.0), batch_dim=1))
        .equal(
            BatchedArray(
                np.array(
                    [[False, False, False, False, False], [True, True, True, True, True]],
                    dtype=bool,
                ),
                batch_dim=1,
            )
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 5), 5.0)),
        np.full((2, 5), 5.0),
        BatchedArray(np.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_array_gt(other: BatchedArray | ndarray | int | float) -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .gt(other)
        .equal(
            BatchedArray(
                np.array(
                    [[False, False, False, False, False], [False, True, True, True, True]],
                    dtype=bool,
                ),
            )
        )
    )


def test_batched_array_gt_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_dim=1)
        .gt(BatchedArray(np.full((2, 5), 5.0), batch_dim=1))
        .equal(
            BatchedArray(
                np.array(
                    [[False, False, False, False, False], [False, True, True, True, True]],
                    dtype=bool,
                ),
                batch_dim=1,
            )
        )
    )


def test_batched_array_isinf() -> None:
    assert (
        BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
        .isinf()
        .equal(BatchedArray(np.array([[False, False, True], [False, False, True]], dtype=bool)))
    )


def test_batched_array_isinf_custom_batch_dim() -> None:
    assert (
        BatchedArray(
            np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]),
            batch_dim=1,
        )
        .isinf()
        .equal(
            BatchedArray(
                np.array([[False, False, True], [False, False, True]], dtype=bool),
                batch_dim=1,
            )
        )
    )


def test_batched_array_isneginf() -> None:
    assert (
        BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
        .isneginf()
        .equal(BatchedArray(np.array([[False, False, False], [False, False, True]], dtype=bool)))
    )


def test_batched_array_isneginf_custom_batch_dim() -> None:
    assert (
        BatchedArray(
            np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]),
            batch_dim=1,
        )
        .isneginf()
        .equal(
            BatchedArray(
                np.array([[False, False, False], [False, False, True]], dtype=bool),
                batch_dim=1,
            )
        )
    )


def test_batched_array_isposinf() -> None:
    assert (
        BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
        .isposinf()
        .equal(BatchedArray(np.array([[False, False, True], [False, False, False]], dtype=bool)))
    )


def test_batched_array_isposinf_custom_batch_dim() -> None:
    assert (
        BatchedArray(
            np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]),
            batch_dim=1,
        )
        .isposinf()
        .equal(
            BatchedArray(
                np.array([[False, False, True], [False, False, False]], dtype=bool),
                batch_dim=1,
            )
        )
    )


def test_batched_array_isnan() -> None:
    assert (
        BatchedArray(np.array([[1.0, 0.0, float("nan")], [float("nan"), -2.0, -1.0]]))
        .isnan()
        .equal(BatchedArray(np.array([[False, False, True], [True, False, False]], dtype=bool)))
    )


def test_batched_array_isnan_custom_batch_dim() -> None:
    assert (
        BatchedArray(
            np.array([[1.0, 0.0, float("nan")], [float("nan"), -2.0, -1.0]]),
            batch_dim=1,
        )
        .isnan()
        .equal(
            BatchedArray(
                np.array([[False, False, True], [True, False, False]], dtype=bool),
                batch_dim=1,
            )
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 5), 5.0)),
        np.full((2, 5), 5.0),
        BatchedArray(np.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_array_le(other: BatchedArray | ndarray | int | float) -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .le(other)
        .equal(
            BatchedArray(
                np.array(
                    [[True, True, True, True, True], [True, False, False, False, False]],
                    dtype=bool,
                )
            )
        )
    )


def test_batched_array_le_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_dim=1)
        .le(BatchedArray(np.full((2, 5), 5.0), batch_dim=1))
        .equal(
            BatchedArray(
                np.array(
                    [[True, True, True, True, True], [True, False, False, False, False]],
                    dtype=bool,
                ),
                batch_dim=1,
            )
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 5), 5.0)),
        np.full((2, 5), 5.0),
        BatchedArray(np.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_array_lt(other: BatchedArray | ndarray | int | float) -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .lt(other)
        .equal(
            BatchedArray(
                np.array(
                    [[True, True, True, True, True], [False, False, False, False, False]],
                    dtype=bool,
                ),
            )
        )
    )


def test_batched_array_lt_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_dim=1)
        .lt(BatchedArray(np.full((2, 5), 5.0), batch_dim=1))
        .equal(
            BatchedArray(
                np.array(
                    [[True, True, True, True, True], [False, False, False, False, False]],
                    dtype=bool,
                ),
                batch_dim=1,
            )
        )
    )


#################
#     dtype     #
#################


def test_batched_array_bool() -> None:
    assert BatchedArray(np.ones((2, 3))).bool().equal(BatchedArray(np.ones((2, 3), dtype=bool)))


def test_batched_array_bool_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.ones((2, 3)), batch_dim=1)
        .bool()
        .equal(BatchedArray(np.ones((2, 3), dtype=bool), batch_dim=1))
    )


def test_batched_array_double() -> None:
    assert (
        BatchedArray(np.ones((2, 3), dtype=bool))
        .double()
        .equal(BatchedArray(np.ones((2, 3), dtype=float)))
    )


def test_batched_array_double_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.ones((2, 3), dtype=bool), batch_dim=1)
        .double()
        .equal(BatchedArray(np.ones((2, 3), dtype=float), batch_dim=1))
    )


def test_batched_array_float() -> None:
    assert (
        BatchedArray(np.ones((2, 3), dtype=int))
        .float()
        .equal(BatchedArray(np.ones((2, 3), dtype=np.single)))
    )


def test_batched_array_float_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.ones((2, 3), dtype=int), batch_dim=1)
        .float()
        .equal(BatchedArray(np.ones((2, 3), dtype=np.single), batch_dim=1))
    )


def test_batched_array_int() -> None:
    assert BatchedArray(np.ones((2, 3))).int().equal(BatchedArray(np.ones((2, 3), dtype=np.intc)))


def test_batched_array_int_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.ones((2, 3)), batch_dim=1)
        .int()
        .equal(BatchedArray(np.ones((2, 3), dtype=np.intc), batch_dim=1))
    )


def test_batched_array_long() -> None:
    assert BatchedArray(np.ones((2, 3))).long().equal(BatchedArray(np.ones((2, 3), dtype=int)))


def test_batched_array_long_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.ones((2, 3)), batch_dim=1)
        .long()
        .equal(BatchedArray(np.ones((2, 3), dtype=int), batch_dim=1))
    )


##################################################
#     Mathematical | arithmetical operations     #
##################################################


@mark.parametrize(
    "other",
    (
        BatchedArray(np.ones((2, 3))),
        np.ones((2, 3)),
        BatchedArray(np.ones((2, 1))),
        1,
        1.0,
    ),
)
def test_batched_array__add__(other: BatchedArray | ndarray | int | float) -> None:
    assert (BatchedArray(np.zeros((2, 3))) + other).equal(BatchedArray(np.ones((2, 3))))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.ones((2, 3))),
        np.ones((2, 3)),
        BatchedArray(np.ones((2, 1))),
        1,
        1.0,
    ),
)
def test_batched_array__iadd__(other: BatchedArray | ndarray | int | float) -> None:
    batch = BatchedArray(np.zeros((2, 3)))
    batch += other
    assert batch.equal(BatchedArray(np.ones((2, 3))))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array__floordiv__(other: BatchedArray | ndarray | int | float) -> None:
    assert (BatchedArray(np.ones((2, 3))) // other).equal(BatchedArray(np.zeros((2, 3))))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array__ifloordiv__(other: BatchedArray | ndarray | int | float) -> None:
    batch = BatchedArray(np.ones((2, 3)))
    batch //= other
    assert batch.equal(BatchedArray(np.zeros((2, 3))))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array__mul__(other: BatchedArray | ndarray | int | float) -> None:
    assert (BatchedArray(np.ones((2, 3))) * other).equal(BatchedArray(np.full((2, 3), 2.0)))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array__imul__(other: BatchedArray | ndarray | int | float) -> None:
    batch = BatchedArray(np.ones((2, 3)))
    batch *= other
    assert batch.equal(BatchedArray(np.full((2, 3), 2.0)))


def test_batched_array__neg__() -> None:
    assert (-BatchedArray(np.ones((2, 3)))).equal(BatchedArray(-np.ones((2, 3))))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array__sub__(other: BatchedArray | ndarray | int | float) -> None:
    assert (BatchedArray(np.ones((2, 3))) - other).equal(BatchedArray(-np.ones((2, 3))))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array__isub__(other: BatchedArray | ndarray | int | float) -> None:
    batch = BatchedArray(np.ones((2, 3)))
    batch -= other
    assert batch.equal(BatchedArray(-np.ones((2, 3))))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array__truediv__(other: BatchedArray | ndarray | int | float) -> None:
    assert (BatchedArray(np.ones((2, 3))) / other).equal(BatchedArray(np.full((2, 3), 0.5)))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array__itruediv__(other: BatchedArray | ndarray | int | float) -> None:
    batch = BatchedArray(np.ones((2, 3)))
    batch /= other
    assert batch.equal(BatchedArray(np.full((2, 3), 0.5)))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array_add(other: BatchedArray | ndarray | int | float) -> None:
    assert BatchedArray(np.ones((2, 3))).add(other).equal(BatchedArray(np.full((2, 3), 3.0)))


def test_batched_array_add_alpha_2_float() -> None:
    assert (
        BatchedArray(np.ones((2, 3)))
        .add(BatchedArray(np.full((2, 3), 2.0, dtype=float)), alpha=2.0)
        .equal(BatchedArray(np.full((2, 3), 5.0, dtype=float)))
    )


def test_batched_array_add_alpha_2_int() -> None:
    assert (
        BatchedArray(np.ones((2, 3), dtype=int))
        .add(BatchedArray(np.full((2, 3), 2.0, dtype=int)), alpha=2)
        .equal(BatchedArray(np.full((2, 3), 5.0, dtype=int)))
    )


def test_batched_array_add_batch_dim_1() -> None:
    assert (
        BatchedArray(np.ones((2, 3)), batch_dim=1)
        .add(BatchedArray(np.full((2, 3), 2.0, dtype=float), batch_dim=1))
        .equal(BatchedArray(np.full((2, 3), 3.0, dtype=float), batch_dim=1))
    )


def test_batched_array_add_incorrect_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 2)))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.add(BatchedArray(np.ones((2, 2)), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array_add_(other: BatchedArray | ndarray | int | float) -> None:
    batch = BatchedArray(np.ones((2, 3)))
    batch.add_(other)
    assert batch.equal(BatchedArray(np.full((2, 3), 3.0)))


def test_batched_array_add__alpha_2_float() -> None:
    batch = BatchedArray(np.ones((2, 3)))
    batch.add_(BatchedArray(np.full((2, 3), 2.0)), alpha=2.0)
    assert batch.equal(BatchedArray(np.full((2, 3), 5.0)))


def test_batched_array_add__alpha_2_int() -> None:
    batch = BatchedArray(np.ones((2, 3), dtype=int))
    batch.add_(BatchedArray(np.full((2, 3), 2, dtype=int)), alpha=2)
    assert batch.equal(BatchedArray(np.full((2, 3), 5, dtype=int)))


def test_batched_array_add__custom_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 3)), batch_dim=1)
    batch.add_(BatchedArray(np.full((2, 3), 2.0), batch_dim=1))
    assert batch.equal(BatchedArray(np.full((2, 3), 3.0), batch_dim=1))


def test_batched_array_add__incorrect_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 2)))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.add_(BatchedArray(np.ones((2, 2)), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array_div(other: BatchedArray | ndarray | int | float) -> None:
    assert BatchedArray(np.ones((2, 3))).div(other).equal(BatchedArray(np.full((2, 3), 0.5)))


def test_batched_array_div_rounding_mode_floor() -> None:
    assert (
        BatchedArray(np.ones((2, 3)))
        .div(BatchedArray(np.full((2, 3), 2.0)), rounding_mode="floor")
        .equal(BatchedArray(np.zeros((2, 3))))
    )


def test_batched_array_div_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.ones((2, 3)), batch_dim=1)
        .div(BatchedArray(np.full((2, 3), 2.0), batch_dim=1))
        .equal(BatchedArray(np.full((2, 3), 0.5), batch_dim=1))
    )


def test_batched_array_div_incorrect_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 2)))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.div(BatchedArray(np.ones((2, 2)), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array_div_(other: BatchedArray | ndarray | int | float) -> None:
    batch = BatchedArray(np.ones((2, 3)))
    batch.div_(other)
    assert batch.equal(BatchedArray(np.full((2, 3), 0.5)))


def test_batched_array_div__rounding_mode_floor() -> None:
    batch = BatchedArray(np.ones((2, 3)))
    batch.div_(BatchedArray(np.full((2, 3), 2.0)), rounding_mode="floor")
    assert batch.equal(BatchedArray(np.zeros((2, 3))))


def test_batched_array_div__custom_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 3)), batch_dim=1)
    batch.div_(BatchedArray(np.full((2, 3), 2.0), batch_dim=1))
    assert batch.equal(BatchedArray(np.full((2, 3), 0.5), batch_dim=1))


def test_batched_array_div__incorrect_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 3)))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.div_(BatchedArray(np.ones((2, 3)), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array_fmod(other: BatchedArray | ndarray | int | float) -> None:
    assert BatchedArray(np.ones((2, 3))).fmod(other).equal(BatchedArray(np.ones((2, 3))))


def test_batched_array_fmod_custom_dims() -> None:
    assert (
        BatchedArray(np.ones((2, 3)), batch_dim=1)
        .fmod(BatchedArray(np.full((2, 3), 2.0), batch_dim=1))
        .equal(BatchedArray(np.ones((2, 3)), batch_dim=1))
    )


def test_batched_array_fmod_incorrect_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 2)))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.fmod(BatchedArray(np.ones((2, 2)), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array_fmod_(other: BatchedArray | ndarray | int | float) -> None:
    batch = BatchedArray(np.ones((2, 3)))
    batch.fmod_(other)
    assert batch.equal(BatchedArray(np.ones((2, 3))))


def test_batched_array_fmod__custom_dims() -> None:
    batch = BatchedArray(np.ones((2, 3)), batch_dim=1)
    batch.fmod_(BatchedArray(np.full((2, 3), 2.0), batch_dim=1))
    assert batch.equal(BatchedArray(np.ones((2, 3)), batch_dim=1))


def test_batched_array_fmod__incorrect_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 2)))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.fmod_(BatchedArray(np.ones((2, 2)), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array_mul(other: BatchedArray | ndarray | int | float) -> None:
    assert BatchedArray(np.ones((2, 3))).mul(other).equal(BatchedArray(np.full((2, 3), 2.0)))


def test_batched_array_mul_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.ones((2, 3)), batch_dim=1)
        .mul(BatchedArray(np.full((2, 3), 2.0), batch_dim=1))
        .equal(BatchedArray(np.full((2, 3), 2.0), batch_dim=1))
    )


def test_batched_array_mul_incorrect_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 2)))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.mul(BatchedArray(np.ones((2, 2)), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array_mul_(other: BatchedArray | ndarray | int | float) -> None:
    batch = BatchedArray(np.ones((2, 3)))
    batch.mul_(other)
    assert batch.equal(BatchedArray(np.full((2, 3), 2.0)))


def test_batched_array_mul__custom_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 3)), batch_dim=1)
    batch.mul_(BatchedArray(np.full((2, 3), 2.0), batch_dim=1))
    assert batch.equal(BatchedArray(np.full((2, 3), 2.0), batch_dim=1))


def test_batched_array_mul__incorrect_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 2)))
    with raises(RuntimeError):
        batch.mul_(BatchedArray(np.ones((2, 2)), batch_dim=1))


def test_batched_array_neg() -> None:
    assert BatchedArray(np.ones((2, 3))).neg().equal(BatchedArray(-np.ones((2, 3))))


def test_batched_array_custom_batch_dim() -> None:
    assert (
        BatchedArray(np.ones((2, 3)), batch_dim=1)
        .neg()
        .equal(BatchedArray(-np.ones((2, 3)), batch_dim=1))
    )


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array_sub(other: BatchedArray | ndarray | int | float) -> None:
    assert BatchedArray(np.ones((2, 3))).sub(other).equal(BatchedArray(-np.ones((2, 3))))


def test_batched_array_sub_alpha_2_float() -> None:
    assert (
        BatchedArray(np.ones((2, 3)))
        .sub(BatchedArray(np.full((2, 3), 2.0)), alpha=2.0)
        .equal(BatchedArray(-np.full((2, 3), 3.0)))
    )


def test_batched_array_sub_alpha_2_int() -> None:
    assert (
        BatchedArray(np.ones((2, 3), dtype=int))
        .sub(BatchedArray(np.full((2, 3), 2, dtype=int)), alpha=2)
        .equal(BatchedArray(np.full((2, 3), -3, dtype=int)))
    )


def test_batched_array_sub_custom_batch_dims() -> None:
    assert (
        BatchedArray(np.ones((2, 3)), batch_dim=1)
        .sub(BatchedArray(np.full((2, 3), 2.0), batch_dim=1))
        .equal(BatchedArray(-np.ones((2, 3)), batch_dim=1))
    )


def test_batched_array_sub_incorrect_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 2)))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.sub(BatchedArray(np.ones((2, 2)), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_array_sub_(other: BatchedArray | ndarray | int | float) -> None:
    batch = BatchedArray(np.ones((2, 3)))
    batch.sub_(other)
    assert batch.equal(BatchedArray(-np.ones((2, 3))))


def test_batched_array_sub__alpha_2_float() -> None:
    batch = BatchedArray(np.ones((2, 3)))
    batch.sub_(BatchedArray(np.full((2, 3), 2.0)), alpha=2.0)
    assert batch.equal(BatchedArray(np.full((2, 3), -3.0)))


def test_batched_array_sub__alpha_2_int() -> None:
    batch = BatchedArray(np.ones((2, 3), dtype=int))
    batch.sub_(BatchedArray(np.full((2, 3), 2, dtype=int)), alpha=2)
    assert batch.equal(BatchedArray(np.full((2, 3), -3, dtype=int)))


def test_batched_array_sub__custom_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 3)), batch_dim=1)
    batch.sub_(BatchedArray(np.full((2, 3), 2.0), batch_dim=1))
    assert batch.equal(BatchedArray(-np.ones((2, 3)), batch_dim=1))


def test_batched_array_sub__incorrect_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 2)))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.sub_(BatchedArray(np.ones((2, 2)), batch_dim=1))


###########################################################
#     Mathematical | advanced arithmetical operations     #
###########################################################


def test_batched_array_cumsum() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .cumsum(dim=0)
        .equal(BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])))
    )


def test_batched_array_cumsum_dim_1() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .cumsum(dim=1)
        .equal(BatchedArray(np.array([[0, 1, 3, 6, 10], [5, 11, 18, 26, 35]])))
    )


def test_batched_array_cumsum_dim_none() -> None:
    assert np.array_equal(
        BatchedArray(np.arange(10).reshape(2, 5)).cumsum(dim=None),
        np.array([0, 1, 3, 6, 10, 15, 21, 28, 36, 45]),
    )


def test_batched_array_cumsum_custom_dims() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2), batch_dim=1)
        .cumsum(dim=1)
        .equal(BatchedArray(np.array([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), batch_dim=1))
    )


def test_batched_array_cumsum_dtype() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .cumsum(dim=0, dtype=np.intc)
        .equal(BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]], dtype=np.intc)))
    )


def test_batched_array_cumsum_() -> None:
    batch = BatchedArray(np.arange(10).reshape(2, 5))
    batch.cumsum_(dim=0)
    assert batch.equal(BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])))


def test_batched_array_cumsum__custom_dims() -> None:
    batch = BatchedArray(np.arange(10).reshape(5, 2), batch_dim=1)
    batch.cumsum_(dim=1)
    assert batch.equal(
        BatchedArray(np.array([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), batch_dim=1)
    )


def test_batched_array_cumsum_along_batch() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .cumsum_along_batch()
        .equal(BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])))
    )


def test_batched_array_cumsum_along_batch_custom_dims() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2), batch_dim=1)
        .cumsum_along_batch()
        .equal(BatchedArray(np.array([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), batch_dim=1))
    )


def test_batched_array_cumsum_along_batch_dtype() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .cumsum_along_batch(dtype=np.intc)
        .equal(BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]], dtype=np.intc)))
    )


def test_batched_array_cumsum_along_batch_() -> None:
    batch = BatchedArray(np.arange(10).reshape(2, 5))
    batch.cumsum_along_batch_()
    assert batch.equal(BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])))


def test_batched_array_cumsum_along_batch__custom_dims() -> None:
    batch = BatchedArray(np.arange(10).reshape(5, 2), batch_dim=1)
    batch.cumsum_along_batch_()
    assert batch.equal(
        BatchedArray(np.array([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), batch_dim=1)
    )


def test_batched_array_logcumsumexp_dim_0() -> None:
    assert (
        BatchedArray(np.arange(10, dtype=float).reshape(5, 2))
        .logcumsumexp(dim=0)
        .allclose(
            BatchedArray(
                np.array(
                    [
                        [0.0, 1.0],
                        [2.1269280110429727, 3.1269280110429727],
                        [4.142931628499899, 5.142931628499899],
                        [6.145077938960783, 7.145077938960783],
                        [8.145368056908488, 9.145368056908488],
                    ]
                )
            )
        )
    )


def test_batched_array_logcumsumexp_dim_1() -> None:
    assert (
        BatchedArray(np.arange(10, dtype=float).reshape(2, 5))
        .logcumsumexp(dim=1)
        .allclose(
            BatchedArray(
                np.array(
                    [
                        [
                            0.0,
                            1.3132616875182228,
                            2.40760596444438,
                            3.4401896985611953,
                            4.451914395937593,
                        ],
                        [
                            5.0,
                            6.313261687518223,
                            7.407605964444381,
                            8.440189698561195,
                            9.451914395937592,
                        ],
                    ]
                )
            )
        )
    )


def test_batched_array_logcumsumexp_custom_dims() -> None:
    assert (
        BatchedArray(np.arange(10, dtype=float).reshape(5, 2), batch_dim=1)
        .logcumsumexp(dim=0)
        .allclose(
            BatchedArray(
                np.array(
                    [
                        [0.0, 1.0],
                        [2.1269280110429727, 3.1269280110429727],
                        [4.142931628499899, 5.142931628499899],
                        [6.145077938960783, 7.145077938960783],
                        [8.145368056908488, 9.145368056908488],
                    ]
                ),
                batch_dim=1,
            )
        )
    )


def test_batched_array_logcumsumexp__dim_0() -> None:
    batch = BatchedArray(np.arange(10, dtype=float).reshape(5, 2))
    batch.logcumsumexp_(dim=0)
    assert batch.allclose(
        BatchedArray(
            np.array(
                [
                    [0.0, 1.0],
                    [2.1269280110429727, 3.1269280110429727],
                    [4.142931628499899, 5.142931628499899],
                    [6.145077938960783, 7.145077938960783],
                    [8.145368056908488, 9.145368056908488],
                ]
            )
        )
    )


def test_batched_array_logcumsumexp__dim_1() -> None:
    batch = BatchedArray(np.arange(10, dtype=float).reshape(2, 5))
    batch.logcumsumexp_(dim=1)
    assert batch.allclose(
        BatchedArray(
            np.array(
                [
                    [
                        0.0,
                        1.3132616875182228,
                        2.40760596444438,
                        3.4401896985611953,
                        4.451914395937593,
                    ],
                    [
                        5.0,
                        6.313261687518223,
                        7.407605964444381,
                        8.440189698561195,
                        9.451914395937592,
                    ],
                ]
            )
        )
    )


def test_batched_array_logcumsumexp__custom_dims() -> None:
    batch = BatchedArray(np.arange(10, dtype=float).reshape(5, 2), batch_dim=1)
    batch.logcumsumexp_(dim=0)
    assert batch.allclose(
        BatchedArray(
            np.array(
                [
                    [0.0, 1.0],
                    [2.1269280110429727, 3.1269280110429727],
                    [4.142931628499899, 5.142931628499899],
                    [6.145077938960783, 7.145077938960783],
                    [8.145368056908488, 9.145368056908488],
                ]
            ),
            batch_dim=1,
        )
    )


def test_batched_array_logcumsumexp_along_batch() -> None:
    assert (
        BatchedArray(np.arange(10, dtype=float).reshape(5, 2))
        .logcumsumexp_along_batch()
        .allclose(
            BatchedArray(
                np.array(
                    [
                        [0.0, 1.0],
                        [2.1269280110429727, 3.1269280110429727],
                        [4.142931628499899, 5.142931628499899],
                        [6.145077938960783, 7.145077938960783],
                        [8.145368056908488, 9.145368056908488],
                    ]
                )
            )
        )
    )


def test_batched_array_logcumsumexp_along_batch_custom_dims() -> None:
    assert (
        BatchedArray(np.arange(10, dtype=float).reshape(2, 5), batch_dim=1)
        .logcumsumexp_along_batch()
        .allclose(
            BatchedArray(
                np.array(
                    [
                        [
                            0.0,
                            1.3132616875182228,
                            2.40760596444438,
                            3.4401896985611953,
                            4.451914395937593,
                        ],
                        [
                            5.0,
                            6.313261687518223,
                            7.407605964444381,
                            8.440189698561195,
                            9.451914395937592,
                        ],
                    ]
                ),
                batch_dim=1,
            )
        )
    )


def test_batched_array_logcumsumexp_along_batch_() -> None:
    batch = BatchedArray(np.arange(10, dtype=float).reshape(5, 2))
    batch.logcumsumexp_along_batch_()
    assert batch.allclose(
        BatchedArray(
            np.array(
                [
                    [0.0, 1.0],
                    [2.1269280110429727, 3.1269280110429727],
                    [4.142931628499899, 5.142931628499899],
                    [6.145077938960783, 7.145077938960783],
                    [8.145368056908488, 9.145368056908488],
                ]
            )
        )
    )


def test_batched_array_logcumsumexp_along_batch__custom_dims() -> None:
    batch = BatchedArray(np.arange(10, dtype=float).reshape(2, 5), batch_dim=1)
    batch.logcumsumexp_along_batch_()
    assert batch.allclose(
        BatchedArray(
            np.array(
                [
                    [
                        0.0,
                        1.3132616875182228,
                        2.40760596444438,
                        3.4401896985611953,
                        4.451914395937593,
                    ],
                    [
                        5.0,
                        6.313261687518223,
                        7.407605964444381,
                        8.440189698561195,
                        9.451914395937592,
                    ],
                ]
            ),
            batch_dim=1,
        )
    )


@mark.parametrize("permutation", (np.array([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_array_permute_along_batch(permutation: Sequence[int] | ndarray) -> None:
    assert (
        BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
        .permute_along_batch(permutation)
        .equal(BatchedArray(np.array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))
    )


def test_batched_array_permute_along_batch_custom_dims() -> None:
    assert (
        BatchedArray(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), batch_dim=1)
        .permute_along_batch(np.array([2, 1, 3, 0]))
        .equal(BatchedArray(np.array([[2, 1, 3, 0], [6, 5, 7, 4]]), batch_dim=1))
    )


@mark.parametrize("permutation", (np.array([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_array_permute_along_batch_(permutation: Sequence[int] | ndarray) -> None:
    batch = BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch.permute_along_batch_(permutation)
    assert batch.equal(BatchedArray(np.array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))


def test_batched_array_permute_along_batch__custom_dims() -> None:
    batch = BatchedArray(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), batch_dim=1)
    batch.permute_along_batch_(np.array([2, 1, 3, 0]))
    assert batch.equal(BatchedArray(np.array([[2, 1, 3, 0], [6, 5, 7, 4]]), batch_dim=1))


@mark.parametrize("permutation", (np.array([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_array_permute_along_dim_0(permutation: Sequence[int] | ndarray) -> None:
    assert (
        BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
        .permute_along_dim(permutation, dim=0)
        .equal(BatchedArray(np.array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))
    )


@mark.parametrize("permutation", (np.array([2, 4, 1, 3, 0]), [2, 4, 1, 3, 0], (2, 4, 1, 3, 0)))
def test_batched_array_permute_along_dim_1(permutation: Sequence[int] | ndarray) -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .permute_along_dim(permutation, dim=1)
        .equal(BatchedArray(np.array([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])))
    )


def test_batched_array_permute_along_dim_custom_dims() -> None:
    assert (
        BatchedArray(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), batch_dim=1)
        .permute_along_dim(np.array([2, 1, 3, 0]), dim=1)
        .equal(BatchedArray(np.array([[2, 1, 3, 0], [6, 5, 7, 4]]), batch_dim=1))
    )


@mark.parametrize("permutation", (np.array([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_array_permute_along_dim__0(permutation: Sequence[int] | ndarray) -> None:
    batch = BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch.permute_along_dim_(permutation, dim=0)
    assert batch.equal(BatchedArray(np.array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))


@mark.parametrize("permutation", (np.array([2, 4, 1, 3, 0]), [2, 4, 1, 3, 0], (2, 4, 1, 3, 0)))
def test_batched_array_permute_along_seq__1(permutation: Sequence[int] | ndarray) -> None:
    batch = BatchedArray(np.arange(10).reshape(2, 5))
    batch.permute_along_dim_(permutation, dim=1)
    assert batch.equal(BatchedArray(np.array([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])))


def test_batched_array_permute_along_dim__custom_dims() -> None:
    batch = BatchedArray(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), batch_dim=1)
    batch.permute_along_dim_(np.array([2, 1, 3, 0]), dim=1)
    assert batch.equal(BatchedArray(np.array([[2, 1, 3, 0], [6, 5, 7, 4]]), batch_dim=1))


@patch("redcat.array.randperm", lambda *args, **kwargs: np.array([2, 1, 3, 0]))
def test_batched_array_shuffle_along_dim() -> None:
    assert (
        BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
        .shuffle_along_dim(dim=0)
        .equal(BatchedArray(np.array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))
    )


@patch("redcat.array.randperm", lambda *args, **kwargs: np.array([2, 1, 3, 0]))
def test_batched_array_shuffle_along_dim_custom_dims() -> None:
    assert (
        BatchedArray(np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]), batch_dim=1)
        .shuffle_along_dim(dim=1)
        .equal(BatchedArray(np.array([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]]), batch_dim=1))
    )


def test_batched_array_shuffle_along_dim_same_random_seed() -> None:
    batch = BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert batch.shuffle_along_dim(dim=0, generator=np.random.default_rng(1)).equal(
        batch.shuffle_along_dim(dim=0, generator=np.random.default_rng(1))
    )


def test_batched_array_shuffle_along_dim_different_random_seeds() -> None:
    batch = BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert not batch.shuffle_along_dim(dim=0, generator=np.random.default_rng(1)).equal(
        batch.shuffle_along_dim(dim=0, generator=np.random.default_rng(2))
    )


@patch("redcat.array.randperm", lambda *args, **kwargs: np.array([2, 1, 3, 0]))
def test_batched_array_shuffle_along_dim_() -> None:
    batch = BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch.shuffle_along_dim_(dim=0)
    assert batch.equal(BatchedArray(np.array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))


@patch("redcat.array.randperm", lambda *args, **kwargs: np.array([2, 1, 3, 0]))
def test_batched_array_shuffle_along_dim__custom_dims() -> None:
    batch = BatchedArray(np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]), batch_dim=1)
    batch.shuffle_along_dim_(dim=1)
    assert batch.equal(
        BatchedArray(np.array([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]]), batch_dim=1)
    )


def test_batched_array_shuffle_along_dim__same_random_seed() -> None:
    batch1 = BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch1.shuffle_along_dim_(dim=0, generator=np.random.default_rng(1))
    batch2 = BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch2.shuffle_along_dim_(dim=0, generator=np.random.default_rng(1))
    assert batch1.equal(batch2)


def test_batched_array_shuffle_along_dim__different_random_seeds() -> None:
    batch1 = BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch1.shuffle_along_dim_(dim=0, generator=np.random.default_rng(1))
    batch2 = BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch2.shuffle_along_dim_(dim=0, generator=np.random.default_rng(2))
    assert not batch1.equal(batch2)


##########################################################
#    Indexing, slicing, joining, mutating operations     #
##########################################################


@mark.parametrize(
    "arrays",
    (
        BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])),
        np.array([[10, 11, 12], [13, 14, 15]]),
        [BatchedArray(np.array([[10, 11, 12], [13, 14, 15]]))],
        (BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])),),
    ),
)
def test_batched_array_cat_dim_0(
    arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
) -> None:
    assert (
        BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
        .cat(arrays, dim=0)
        .equal(BatchedArray(np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]])))
    )


@mark.parametrize(
    "arrays",
    (
        BatchedArray(np.array([[10, 11], [12, 13]])),
        np.array([[10, 11], [12, 13]]),
        [BatchedArray(np.array([[10, 11], [12, 13]]))],
        (BatchedArray(np.array([[10, 11], [12, 13]])),),
    ),
)
def test_batched_array_cat_dim_1(
    arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
) -> None:
    assert (
        BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
        .cat(arrays, dim=1)
        .equal(BatchedArray(np.array([[0, 1, 2, 10, 11], [4, 5, 6, 12, 13]])))
    )


def test_batched_array_cat_custom_dims() -> None:
    assert (
        BatchedArray(np.array([[0, 4], [1, 5], [2, 6]]), batch_dim=1)
        .cat(BatchedArray(np.array([[10, 12], [11, 13], [14, 15]]), batch_dim=1), dim=1)
        .equal(
            BatchedArray(
                np.array([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
                batch_dim=1,
            )
        )
    )


def test_batched_array_cat_empty() -> None:
    assert BatchedArray(np.ones((2, 3))).cat([]).equal(BatchedArray(np.ones((2, 3))))


def test_batched_array_cat_incorrect_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 3)))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat([BatchedArray(np.ones((2, 3)), batch_dim=1)])


@mark.parametrize(
    "arrays",
    (
        BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])),
        np.array([[10, 11, 12], [13, 14, 15]]),
        [BatchedArray(np.array([[10, 11, 12], [13, 14, 15]]))],
        (BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])),),
    ),
)
def test_batched_array_cat__dim_0(
    arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
) -> None:
    batch = BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
    batch.cat_(arrays, dim=0)
    assert batch.equal(BatchedArray(np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]])))


@mark.parametrize(
    "arrays",
    (
        BatchedArray(np.array([[10, 11], [12, 13]])),
        np.array([[10, 11], [12, 13]]),
        [BatchedArray(np.array([[10, 11], [12, 13]]))],
        (BatchedArray(np.array([[10, 11], [12, 13]])),),
    ),
)
def test_batched_array_cat__dim_1(
    arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
) -> None:
    batch = BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
    batch.cat_(arrays, dim=1)
    assert batch.equal(BatchedArray(np.array([[0, 1, 2, 10, 11], [4, 5, 6, 12, 13]])))


def test_batched_array_cat__custom_dims() -> None:
    batch = BatchedArray(np.array([[0, 4], [1, 5], [2, 6]]), batch_dim=1)
    batch.cat_(BatchedArray(np.array([[10, 12], [11, 13], [14, 15]]), batch_dim=1), dim=1)
    assert batch.equal(
        BatchedArray(
            np.array([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
            batch_dim=1,
        )
    )


def test_batched_array_cat__empty() -> None:
    batch = BatchedArray(np.ones((2, 3)))
    batch.cat_([])
    assert batch.equal(BatchedArray(np.ones((2, 3))))


def test_batched_array_cat__incorrect_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 2)))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_([BatchedArray(np.zeros((2, 2)), batch_dim=1)])


@mark.parametrize(
    "other",
    (
        BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])),
        np.array([[10, 11, 12], [13, 14, 15]]),
        [BatchedArray(np.array([[10, 11, 12], [13, 14, 15]]))],
        (BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])),),
    ),
)
def test_batched_array_cat_along_batch(
    other: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
) -> None:
    assert (
        BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
        .cat_along_batch(other)
        .equal(BatchedArray(np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]])))
    )


def test_batched_array_cat_along_batch_custom_dims() -> None:
    assert (
        BatchedArray(np.array([[0, 4], [1, 5], [2, 6]]), batch_dim=1)
        .cat_along_batch(BatchedArray(np.array([[10, 12], [11, 13], [14, 15]]), batch_dim=1))
        .equal(
            BatchedArray(
                np.array([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
                batch_dim=1,
            )
        )
    )


def test_batched_array_cat_along_batch_custom_dims_2() -> None:
    assert (
        BatchedArray(np.ones((2, 3, 4)), batch_dim=2)
        .cat_along_batch(BatchedArray(np.ones((2, 3, 1)), batch_dim=2))
        .equal(BatchedArray(np.ones((2, 3, 5)), batch_dim=2))
    )


def test_batched_array_cat_along_batch_multiple() -> None:
    assert (
        BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
        .cat_along_batch(
            [
                BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])),
                BatchedArray(np.array([[20, 21, 22]])),
                np.array([[30, 31, 32]]),
            ]
        )
        .equal(
            BatchedArray(
                np.array(
                    [[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15], [20, 21, 22], [30, 31, 32]]
                )
            )
        )
    )


def test_batched_array_cat_along_batch_empty() -> None:
    assert BatchedArray(np.ones((2, 3))).cat_along_batch([]).equal(BatchedArray(np.ones((2, 3))))


def test_batched_array_cat_along_batch_incorrect_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 2)))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_along_batch([BatchedArray(np.zeros((2, 2)), batch_dim=1)])


@mark.parametrize(
    "other",
    (
        BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])),
        np.array([[10, 11, 12], [13, 14, 15]]),
        [BatchedArray(np.array([[10, 11, 12], [13, 14, 15]]))],
        (BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])),),
    ),
)
def test_batched_array_cat_along_batch_(
    other: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
) -> None:
    batch = BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
    batch.cat_along_batch_(other)
    assert batch.equal(BatchedArray(np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]])))


def test_batched_array_cat_along_batch__custom_dims() -> None:
    batch = BatchedArray(np.array([[0, 4], [1, 5], [2, 6]]), batch_dim=1)
    batch.cat_along_batch_(BatchedArray(np.array([[10, 12], [11, 13], [14, 15]]), batch_dim=1))
    assert batch.equal(
        BatchedArray(
            np.array([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
            batch_dim=1,
        )
    )


def test_batched_array_cat_along_batch__custom_dims_2() -> None:
    batch = BatchedArray(np.ones((2, 3, 4)), batch_dim=2)
    batch.cat_along_batch_(BatchedArray(np.ones((2, 3, 1)), batch_dim=2))
    assert batch.equal(BatchedArray(np.ones((2, 3, 5)), batch_dim=2))


def test_batched_array_cat_along_batch__multiple() -> None:
    batch = BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
    batch.cat_along_batch_(
        [
            BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])),
            BatchedArray(np.array([[20, 21, 22]])),
            np.array([[30, 31, 32]]),
        ]
    )
    assert batch.equal(
        BatchedArray(
            np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15], [20, 21, 22], [30, 31, 32]])
        )
    )


def test_batched_array_cat_along_batch__empty() -> None:
    batch = BatchedArray(np.ones((2, 3)))
    batch.cat_along_batch_([])
    assert batch.equal(BatchedArray(np.ones((2, 3))))


def test_batched_array_cat_along_batch__incorrect_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 2)))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_along_batch_([BatchedArray(np.zeros((2, 2)), batch_dim=1)])


#################
#     Other     #
#################


def test_batched_array_summary() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5)).summary()
        == "BatchedArray(dtype=int64, shape=(2, 5), batch_dim=0)"
    )


###############################
#     Tests for numpy.add     #
###############################


@mark.parametrize(
    "other",
    (
        BatchedArray(np.full((2, 3), 2.0)),
        np.full((2, 3), 2.0),
        BatchedArray(np.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_numpy_add(other: BatchedArray | ndarray | int | float) -> None:
    assert np.add(BatchedArray(np.ones((2, 3))), other).equal(BatchedArray(np.full((2, 3), 3.0)))


def test_numpy_add_custom_dims() -> None:
    assert np.add(
        BatchedArray(np.ones((2, 3)), batch_dim=1), BatchedArray(np.full((2, 3), 2.0), batch_dim=1)
    ).equal(BatchedArray(np.full((2, 3), 3.0), batch_dim=1))


#######################################
#     Tests for numpy.concatenate     #
#######################################


@mark.parametrize(
    "other",
    (
        BatchedArray(np.array([[10, 11, 12], [13, 14, 15]])),
        np.array([[10, 11, 12], [13, 14, 15]]),
    ),
)
def test_numpy_concatenate_axis_0(other: BatchedArray | ndarray) -> None:
    assert objects_are_equal(
        np.concatenate(
            [BatchedArray(np.array([[0, 1, 2], [4, 5, 6]])), other],
            axis=0,
        ),
        BatchedArray(np.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]])),
    )


@mark.parametrize(
    "other",
    (
        BatchedArray(np.array([[4, 5], [14, 15]])),
        np.array([[4, 5], [14, 15]]),
    ),
)
def test_numpy_concatenate_axis_1(other: BatchedArray | ndarray) -> None:
    assert objects_are_equal(
        np.concatenate(
            [BatchedArray(np.array([[0, 1, 2], [10, 11, 12]])), other],
            axis=1,
        ),
        BatchedArray(np.array([[0, 1, 2, 4, 5], [10, 11, 12, 14, 15]])),
    )


def test_numpy_concatenate_array() -> None:
    assert objects_are_equal(
        np.concatenate(
            [
                np.array([[0, 1, 2], [10, 11, 12]]),
                np.array([[4, 5], [14, 15]]),
            ],
            axis=1,
        ),
        np.array(
            [[0, 1, 2, 4, 5], [10, 11, 12, 14, 15]],
        ),
    )


def test_numpy_concatenate_custom_dims() -> None:
    assert objects_are_equal(
        np.concatenate(
            [
                BatchedArray(np.ones((2, 3)), batch_dim=1),
                BatchedArray(np.ones((2, 3)), batch_dim=1),
            ]
        ),
        BatchedArray(np.ones((4, 3)), batch_dim=1),
    )


def test_numpy_concatenate_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        np.concatenate(
            [
                BatchedArray(np.ones((2, 2))),
                BatchedArray(np.zeros((2, 2)), batch_dim=1),
            ]
        )


##################################
#     Tests for numpy.cumsum     #
##################################


def test_numpy_cumsum_axis_0() -> None:
    assert np.cumsum(BatchedArray(np.arange(10).reshape(2, 5)), axis=0).equal(
        BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]]))
    )


def test_numpy_cumsum_axis_1() -> None:
    assert np.cumsum(BatchedArray(np.arange(10).reshape(2, 5)), axis=1).equal(
        BatchedArray(np.array([[0, 1, 3, 6, 10], [5, 11, 18, 26, 35]]))
    )


def test_numpy_cumsum_axis_none() -> None:
    assert np.array_equal(
        np.cumsum(BatchedArray(np.arange(10).reshape(2, 5))),
        np.array([0, 1, 3, 6, 10, 15, 21, 28, 36, 45]),
    )


def test_numpy_cumsum_custom_axiss() -> None:
    assert np.cumsum(BatchedArray(np.arange(10).reshape(5, 2), batch_dim=1), axis=1).equal(
        BatchedArray(np.array([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), batch_dim=1)
    )


def test_numpy_cumsum_dtype() -> None:
    assert np.cumsum(BatchedArray(np.arange(10).reshape(2, 5)), axis=0, dtype=np.intc).equal(
        BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]], dtype=np.intc))
    )


####################################
#     Tests for numpy.isneginf     #
####################################


def test_numpy_isneginf() -> None:
    assert np.isneginf(
        BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
    ).equal(BatchedArray(np.array([[False, False, False], [False, False, True]], dtype=bool)))


def test_numpy_isneginf_custom_dims() -> None:
    assert np.isneginf(
        BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]), batch_dim=1)
    ).equal(
        BatchedArray(
            np.array([[False, False, False], [False, False, True]], dtype=bool), batch_dim=1
        )
    )


####################################
#     Tests for numpy.isposinf     #
####################################


def test_numpy_isposinf() -> None:
    assert np.isposinf(
        BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
    ).equal(BatchedArray(np.array([[False, False, True], [False, False, False]], dtype=bool)))


def test_numpy_isposinf_custom_dims() -> None:
    assert np.isposinf(
        BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]), batch_dim=1)
    ).equal(
        BatchedArray(
            np.array([[False, False, True], [False, False, False]], dtype=bool), batch_dim=1
        )
    )


###############################
#     Tests for numpy.sum     #
###############################


def test_numpy_sum() -> None:
    assert np.array_equal(np.sum(BatchedArray(np.arange(10).reshape(2, 5))), np.array(45))


def test_numpy_sum_dim_1() -> None:
    assert np.array_equal(
        np.sum(BatchedArray(np.arange(10).reshape(2, 5)), axis=1), np.array([10, 35])
    )


def test_numpy_sum_keepdim() -> None:
    assert np.array_equal(
        np.sum(BatchedArray(np.arange(10).reshape(2, 5)), axis=1, keepdims=True),
        np.array([[10], [35]]),
    )


###############################################
#     Tests for get_div_rounding_operator     #
###############################################


def test_get_div_rounding_operator_mode_none() -> None:
    assert get_div_rounding_operator(None) == np.true_divide


def test_get_div_rounding_operator_mode_floor() -> None:
    assert get_div_rounding_operator("floor") == np.floor_divide


def test_get_div_rounding_operator_mode_incorrect() -> None:
    with raises(RuntimeError, match="Incorrect `rounding_mode`"):
        get_div_rounding_operator("incorrect")
