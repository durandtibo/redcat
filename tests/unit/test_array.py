from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from coola import objects_are_equal
from numpy import ndarray
from pytest import mark, raises

from redcat.array import BatchedArray

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


###############################
#     Creation operations     #
###############################


def test_batched_array_clone() -> None:
    batch = BatchedArray(np.ones((2, 3)))
    clone = batch.clone()
    batch._data += 1
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
    batch._data += 1
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
def test_batched_array_concatenate_dim_0(
    arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
) -> None:
    assert (
        BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
        .concatenate(arrays, axis=0)
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
def test_batched_array_concatenate_dim_1(
    arrays: BatchedArray | ndarray | Iterable[BatchedArray | ndarray],
) -> None:
    assert (
        BatchedArray(np.array([[0, 1, 2], [4, 5, 6]]))
        .concatenate(arrays, axis=1)
        .equal(BatchedArray(np.array([[0, 1, 2, 10, 11], [4, 5, 6, 12, 13]])))
    )


def test_batched_array_concatenate_custom_dims() -> None:
    assert (
        BatchedArray(np.array([[0, 4], [1, 5], [2, 6]]), batch_dim=1)
        .concatenate(BatchedArray(np.array([[10, 12], [11, 13], [14, 15]]), batch_dim=1), axis=1)
        .equal(
            BatchedArray(
                np.array([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
                batch_dim=1,
            )
        )
    )


def test_batched_array_concatenate_empty() -> None:
    assert BatchedArray(np.ones((2, 3))).concatenate([]).equal(BatchedArray(np.ones((2, 3))))


def test_batched_array_concatenate_incorrect_batch_dim() -> None:
    batch = BatchedArray(np.ones((2, 3)))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.concatenate([BatchedArray(np.ones((2, 3)), batch_dim=1)])


#################
#     Other     #
#################


def test_batched_array_summary() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5)).summary()
        == "BatchedArray(dtype=int64, shape=(2, 5), batch_dim=0)"
    )


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
                BatchedArray(np.ones((2, 3))),
                BatchedArray(np.zeros((2, 3)), batch_dim=1),
            ]
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
