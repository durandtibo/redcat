from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import patch

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from redcat import ba
from redcat.ba import BatchedArray
from tests.conftest import future_test
from tests.unit.ba.test_core import NUMERIC_DTYPES

#################################
#     Comparison operations     #
#################################


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5.0),
        5,
        5.0,
    ],
)
def test_eq(other: np.ndarray | int | float) -> None:
    assert ba.equal(BatchedArray(np.arange(10).reshape(2, 5)), other).allequal(
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [True, False, False, False, False]],
                dtype=bool,
            ),
        ),
    )


def test_eq_custom_batch_axis() -> None:
    assert ba.equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1),
        ba.full(shape=(2, 5), fill_value=5, batch_axis=1),
    ).allequal(
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [True, False, False, False, False]],
                dtype=bool,
            ),
            batch_axis=1,
        )
    )


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5.0),
        5,
        5.0,
    ],
)
def test_ge(other: np.ndarray | float) -> None:
    assert ba.greater_equal(BatchedArray(np.arange(10).reshape(2, 5)), other).allequal(
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [True, True, True, True, True]],
                dtype=bool,
            ),
        )
    )


def test_ge_custom_batch_axis() -> None:
    assert ba.greater_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1),
        ba.full(shape=(2, 5), fill_value=5, batch_axis=1),
    ).allequal(
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [True, True, True, True, True]],
                dtype=bool,
            ),
            batch_axis=1,
        )
    )


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5.0),
        5,
        5.0,
    ],
)
def test_gt(other: np.ndarray | float) -> None:
    assert ba.greater(BatchedArray(np.arange(10).reshape(2, 5)), other).allequal(
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [False, True, True, True, True]],
                dtype=bool,
            ),
        )
    )


def test_gt_custom_batch_axis() -> None:
    assert ba.greater(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1),
        ba.full(shape=(2, 5), fill_value=5, batch_axis=1),
    ).allequal(
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [False, True, True, True, True]],
                dtype=bool,
            ),
            batch_axis=1,
        )
    )


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5.0),
        5,
        5.0,
    ],
)
def test_le(other: np.ndarray | float) -> None:
    assert ba.less_equal(BatchedArray(np.arange(10).reshape(2, 5)), other).allequal(
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [True, False, False, False, False]],
                dtype=bool,
            )
        )
    )


def test_le_custom_batch_axis() -> None:
    assert ba.less_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1),
        ba.full(shape=(2, 5), fill_value=5, batch_axis=1),
    ).allequal(
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [True, False, False, False, False]],
                dtype=bool,
            ),
            batch_axis=1,
        )
    )


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5.0),
        5,
        5.0,
    ],
)
def test_lt(other: np.ndarray | float) -> None:
    assert ba.less(BatchedArray(np.arange(10).reshape(2, 5)), other).allequal(
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [False, False, False, False, False]],
                dtype=bool,
            ),
        )
    )


def test_lt_custom_batch_axis() -> None:
    assert ba.less(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1),
        ba.full(shape=(2, 5), fill_value=5, batch_axis=1),
    ).allequal(
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [False, False, False, False, False]],
                dtype=bool,
            ),
            batch_axis=1,
        )
    )


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5.0),
        5,
        5.0,
    ],
)
def test_ne(other: np.ndarray | int | float) -> None:
    assert ba.not_equal(BatchedArray(np.arange(10).reshape(2, 5)), other).allequal(
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [False, True, True, True, True]],
                dtype=bool,
            ),
        )
    )


def test_ne_custom_batch_axis() -> None:
    assert ba.not_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1),
        ba.full(shape=(2, 5), fill_value=5, batch_axis=1),
    ).allequal(
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [False, True, True, True, True]],
                dtype=bool,
            ),
            batch_axis=1,
        )
    )


def test_allclose_true() -> None:
    assert ba.allclose(ba.ones(shape=(2, 3)), ba.ones(shape=(2, 3)))


def test_allclose_false_different_type() -> None:
    assert not ba.allclose(ba.ones(shape=(2, 3)), np.ones(shape=(2, 3)))


def test_allclose_false_different_data() -> None:
    assert not ba.allclose(ba.ones(shape=(2, 3)), ba.zeros(shape=(2, 3)))


def test_allclose_false_different_shape() -> None:
    assert not ba.allclose(ba.ones(shape=(2, 3)), ba.ones(shape=(2, 3, 1)))


def test_allclose_false_different_batch_axis() -> None:
    assert not ba.allclose(ba.ones(shape=(2, 3), batch_axis=1), ba.ones(shape=(2, 3)))


@pytest.mark.parametrize(
    ("array", "atol"),
    (
        (ba.ones((2, 3)) + 0.5, 1),
        (ba.ones((2, 3)) + 0.05, 1e-1),
        (ba.ones((2, 3)) + 5e-3, 1e-2),
    ),
)
def test_allclose_true_atol(array: BatchedArray, atol: float) -> None:
    assert ba.allclose(ba.ones((2, 3)), array, atol=atol, rtol=0)


@pytest.mark.parametrize(
    ("array", "rtol"),
    (
        (ba.ones((2, 3)) + 0.5, 1),
        (ba.ones((2, 3)) + 0.05, 1e-1),
        (ba.ones((2, 3)) + 5e-3, 1e-2),
    ),
)
def test_allclose_true_rtol(array: BatchedArray, rtol: float) -> None:
    assert ba.allclose(ba.ones((2, 3)), array, rtol=rtol)


def test_allclose_equal_nan_false() -> None:
    assert not ba.allclose(
        BatchedArray(np.array([1, np.nan, 3])), BatchedArray(np.array([1, np.nan, 3]))
    )


def test_allclose_equal_nan_true() -> None:
    assert ba.allclose(
        BatchedArray(np.array([1, np.nan, 3])),
        BatchedArray(np.array([1, np.nan, 3])),
        equal_nan=True,
    )


def test_allclose_true_numpy() -> None:
    assert ba.allclose(np.ones(shape=(2, 3)), np.ones(shape=(2, 3)))


def test_array_equal_true() -> None:
    assert ba.array_equal(ba.ones(shape=(2, 3)), ba.ones(shape=(2, 3)))


def test_array_equal_false_different_type() -> None:
    assert not ba.array_equal(ba.ones(shape=(2, 3)), np.ones(shape=(2, 3)))


def test_array_equal_false_different_data() -> None:
    assert not ba.array_equal(ba.ones(shape=(2, 3)), ba.zeros(shape=(2, 3)))


def test_array_equal_false_different_shape() -> None:
    assert not ba.array_equal(ba.ones(shape=(2, 3)), ba.ones(shape=(2, 3, 1)))


def test_array_equal_false_different_batch_axis() -> None:
    assert not ba.array_equal(ba.ones(shape=(2, 3), batch_axis=1), ba.ones(shape=(2, 3)))


def test_array_equal_equal_nan_false() -> None:
    assert not ba.array_equal(
        BatchedArray(np.array([1, np.nan, 3])), BatchedArray(np.array([1, np.nan, 3]))
    )


def test_array_equal_equal_nan_true() -> None:
    assert ba.array_equal(
        BatchedArray(np.array([1, np.nan, 3])),
        BatchedArray(np.array([1, np.nan, 3])),
        equal_nan=True,
    )


def test_array_equal_true_numpy() -> None:
    assert ba.array_equal(np.ones(shape=(2, 3)), np.ones(shape=(2, 3)))


###########################################
#     Item selection and manipulation     #
###########################################


def test_argsort_1d() -> None:
    assert objects_are_equal(
        ba.argsort_along_batch(BatchedArray(np.array([4, 1, 3, 2, 0]))),
        BatchedArray(np.asarray([4, 1, 3, 2, 0])),
    )


def test_argsort_2d_axis_0() -> None:
    assert objects_are_equal(
        ba.argsort(BatchedArray(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=0),
        BatchedArray(np.asarray([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
    )


def test_argsort_custom_axis() -> None:
    assert objects_are_equal(
        ba.argsort(BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)),
        BatchedArray(np.asarray([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]), batch_axis=1),
    )


def test_argsort_along_batch() -> None:
    assert objects_are_equal(
        ba.argsort_along_batch(BatchedArray(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]))),
        BatchedArray(np.asarray([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
    )


def test_argsort_along_batch_custom_axis() -> None:
    assert objects_are_equal(
        ba.argsort_along_batch(
            BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)
        ),
        BatchedArray(np.asarray([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]), batch_axis=1),
    )


@future_test
def test_cumprod() -> None:
    assert objects_are_equal(
        ba.cumprod(BatchedArray(np.arange(10).reshape(2, 5))),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    )


def test_cumprod_axis_0() -> None:
    assert objects_are_equal(
        ba.cumprod(BatchedArray(np.arange(10).reshape(2, 5)), axis=0),
        BatchedArray(np.asarray([[0, 1, 2, 3, 4], [0, 6, 14, 24, 36]])),
    )


def test_cumprod_axis_1() -> None:
    assert objects_are_equal(
        ba.cumprod(BatchedArray(np.arange(10).reshape(2, 5)), axis=1),
        BatchedArray(np.array([[0, 0, 0, 0, 0], [5, 30, 210, 1680, 15120]])),
    )


def test_cumprod_custom_axis() -> None:
    assert objects_are_equal(
        ba.cumprod(BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1), axis=0),
        BatchedArray(np.array([[0, 1], [0, 3], [0, 15], [0, 105], [0, 945]]), batch_axis=1),
    )


def test_cumprod_along_batch() -> None:
    assert objects_are_equal(
        ba.cumprod_along_batch(BatchedArray(np.arange(10).reshape(2, 5))),
        BatchedArray(np.array([[0, 1, 2, 3, 4], [0, 6, 14, 24, 36]])),
    )


def test_cumprod_along_batch_custom_axis() -> None:
    assert objects_are_equal(
        ba.cumprod_along_batch(BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1)),
        BatchedArray(np.array([[0, 0], [2, 6], [4, 20], [6, 42], [8, 72]]), batch_axis=1),
    )


@future_test
def test_cumsum() -> None:
    assert objects_are_equal(
        ba.cumsum(BatchedArray(np.arange(10).reshape(2, 5))),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    )


def test_cumsum_axis_0() -> None:
    assert objects_are_equal(
        ba.cumsum(BatchedArray(np.arange(10).reshape(2, 5)), axis=0),
        BatchedArray(np.asarray([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])),
    )


def test_cumsum_axis_1() -> None:
    assert objects_are_equal(
        ba.cumsum(BatchedArray(np.arange(10).reshape(2, 5)), axis=1),
        BatchedArray(np.array([[0, 1, 3, 6, 10], [5, 11, 18, 26, 35]])),
    )


def test_cumsum_custom_axis() -> None:
    assert objects_are_equal(
        ba.cumsum(BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1), axis=0),
        BatchedArray(np.array([[0, 1], [2, 4], [6, 9], [12, 16], [20, 25]]), batch_axis=1),
    )


def test_cumsum_along_batch() -> None:
    assert objects_are_equal(
        ba.cumsum_along_batch(BatchedArray(np.arange(10).reshape(2, 5))),
        BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])),
    )


def test_cumsum_along_batch_custom_axis() -> None:
    assert objects_are_equal(
        ba.cumsum_along_batch(BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1)),
        BatchedArray(np.array([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), batch_axis=1),
    )


def test_nancumprod() -> None:
    assert objects_are_equal(
        ba.nancumprod(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))),
        np.array([1.0, 1.0, 2.0, 6.0, 24.0, 120.0]),
    )


def test_nancumprod_axis_0() -> None:
    assert objects_are_equal(
        ba.nancumprod(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=0),
        BatchedArray(np.asarray([[1.0, 1.0, 2.0], [3.0, 4.0, 10.0]])),
    )


def test_nancumprod_axis_1() -> None:
    assert objects_are_equal(
        ba.nancumprod(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=1),
        BatchedArray(np.array([[1.0, 1.0, 2.0], [3.0, 12.0, 60.0]])),
    )


def test_nancumprod_custom_axes() -> None:
    assert objects_are_equal(
        ba.nancumprod(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1), axis=0),
        BatchedArray(np.array([[1.0, 1.0, 2.0], [3.0, 4.0, 10.0]]), batch_axis=1),
    )


def test_nancumprod_out() -> None:
    out = ba.zeros((2, 3))
    assert (
        ba.nancumprod(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=0, out=out) is out
    )
    assert objects_are_equal(out, BatchedArray(np.asarray([[1.0, 1.0, 2.0], [3.0, 4.0, 10.0]])))


def test_nancumprod_out_array() -> None:
    out = np.zeros(6)
    assert ba.nancumprod(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), out=out) is out
    assert objects_are_equal(out, np.asarray([1.0, 1.0, 2.0, 6.0, 24.0, 120.0]))


def test_nancumprod_along_batch() -> None:
    assert objects_are_equal(
        ba.nancumprod_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))),
        BatchedArray(np.array([[1.0, 1.0, 2.0], [3.0, 4.0, 10.0]])),
    )


def test_nancumprod_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nancumprod_along_batch(
            BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
        ),
        BatchedArray(np.array([[1.0, 1.0, 2.0], [3.0, 12.0, 60.0]]), batch_axis=1),
    )


def test_nancumprod_along_batch_out() -> None:
    out = ba.zeros((2, 3))
    assert (
        ba.nancumprod_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), out=out)
        is out
    )
    assert objects_are_equal(out, BatchedArray(np.asarray([[1.0, 1.0, 2.0], [3.0, 4.0, 10.0]])))


def test_nancumsum() -> None:
    assert objects_are_equal(
        ba.nancumsum(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))),
        np.array([1.0, 1.0, 3.0, 6.0, 10.0, 15.0]),
    )


def test_nancumsum_axis_0() -> None:
    assert objects_are_equal(
        ba.nancumsum(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=0),
        BatchedArray(np.asarray([[1.0, 0.0, 2.0], [4.0, 4.0, 7.0]])),
    )


def test_nancumsum_axis_1() -> None:
    assert objects_are_equal(
        ba.nancumsum(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=1),
        BatchedArray(np.array([[1.0, 1.0, 3.0], [3.0, 7.0, 12.0]])),
    )


def test_nancumsum_custom_axes() -> None:
    assert objects_are_equal(
        ba.nancumsum(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1), axis=0),
        BatchedArray(np.array([[1.0, 0.0, 2.0], [4.0, 4.0, 7.0]]), batch_axis=1),
    )


def test_nancumsum_along_batch() -> None:
    assert objects_are_equal(
        ba.nancumsum_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))),
        BatchedArray(np.array([[1.0, 0.0, 2.0], [4.0, 4.0, 7.0]])),
    )


def test_nancumsum_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nancumsum_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)),
        BatchedArray(np.array([[1.0, 1.0, 3.0], [3.0, 7.0, 12.0]]), batch_axis=1),
    )


def test_nancumsum_out() -> None:
    out = ba.zeros((2, 3))
    assert ba.nancumsum(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=0, out=out) is out
    assert objects_are_equal(out, BatchedArray(np.asarray([[1.0, 0.0, 2.0], [4.0, 4.0, 7.0]])))


def test_nancumsum_array() -> None:
    out = np.zeros(6)
    assert ba.nancumsum(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), out=out) is out
    assert objects_are_equal(out, np.asarray([1.0, 1.0, 3.0, 6.0, 10.0, 15.0]))


def test_nancumsum_along_batch_out() -> None:
    out = ba.zeros((2, 3))
    assert (
        ba.nancumsum_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), out=out)
        is out
    )
    assert objects_are_equal(out, BatchedArray(np.asarray([[1.0, 0.0, 2.0], [4.0, 4.0, 7.0]])))


@pytest.mark.parametrize("permutation", [np.array([0, 2, 1, 3]), [0, 2, 1, 3], (0, 2, 1, 3)])
def test_permute_along_axis_1d(permutation: np.ndarray | Sequence) -> None:
    assert objects_are_equal(
        ba.permute_along_axis(BatchedArray(np.arange(4)), permutation),
        BatchedArray(np.array([0, 2, 1, 3])),
    )


def test_permute_along_axis_2d_axis_0() -> None:
    assert objects_are_equal(
        ba.permute_along_axis(
            BatchedArray(np.arange(20).reshape(4, 5)), permutation=np.array([0, 2, 1, 3])
        ),
        BatchedArray(
            np.array([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [15, 16, 17, 18, 19]])
        ),
    )


def test_permute_along_axis_2d_axis_1() -> None:
    assert objects_are_equal(
        ba.permute_along_axis(
            BatchedArray(np.arange(20).reshape(4, 5)), permutation=np.array([0, 4, 2, 1, 3]), axis=1
        ),
        BatchedArray(
            np.array([[0, 4, 2, 1, 3], [5, 9, 7, 6, 8], [10, 14, 12, 11, 13], [15, 19, 17, 16, 18]])
        ),
    )


def test_permute_along_axis_3d_axis_2() -> None:
    assert objects_are_equal(
        ba.permute_along_axis(
            BatchedArray(np.arange(20).reshape(2, 2, 5)),
            permutation=np.array([0, 4, 2, 1, 3]),
            axis=2,
        ),
        BatchedArray(
            np.array(
                [[[0, 4, 2, 1, 3], [5, 9, 7, 6, 8]], [[10, 14, 12, 11, 13], [15, 19, 17, 16, 18]]]
            )
        ),
    )


def test_permute_along_custom_axes() -> None:
    assert objects_are_equal(
        ba.permute_along_axis(
            BatchedArray(np.arange(20).reshape(4, 5), batch_axis=1),
            permutation=np.array([0, 2, 1, 3]),
        ),
        BatchedArray(
            np.array(
                [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [15, 16, 17, 18, 19]]
            ),
            batch_axis=1,
        ),
    )


@pytest.mark.parametrize("permutation", [np.array([0, 2, 1, 3]), [0, 2, 1, 3], (0, 2, 1, 3)])
def test_permute_along_batch_1d(permutation: np.ndarray | Sequence) -> None:
    assert objects_are_equal(
        ba.permute_along_batch(BatchedArray(np.arange(4)), permutation),
        BatchedArray(np.array([0, 2, 1, 3])),
    )


def test_permute_along_batch_2d() -> None:
    assert objects_are_equal(
        ba.permute_along_batch(
            BatchedArray(np.arange(20).reshape(4, 5)), permutation=np.array([0, 2, 1, 3])
        ),
        BatchedArray(
            np.array([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [15, 16, 17, 18, 19]])
        ),
    )


def test_permute_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.permute_along_batch(
            BatchedArray(np.arange(20).reshape(4, 5), batch_axis=1),
            permutation=np.array([0, 4, 2, 1, 3]),
        ),
        BatchedArray(
            np.array(
                [[0, 4, 2, 1, 3], [5, 9, 7, 6, 8], [10, 14, 12, 11, 13], [15, 19, 17, 16, 18]]
            ),
            batch_axis=1,
        ),
    )


@patch("redcat.ba.core.randperm", lambda *args, **kwargs: np.array([2, 1, 3, 0]))
def test_shuffle_along_axis() -> None:
    assert objects_are_equal(
        ba.shuffle_along_axis(
            BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])), axis=0
        ),
        BatchedArray(np.array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])),
    )


@patch("redcat.ba.core.randperm", lambda *args, **kwargs: np.array([2, 1, 3, 0]))
def test_shuffle_along_axis_custom_axes() -> None:
    assert objects_are_equal(
        ba.shuffle_along_axis(
            BatchedArray(np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]), batch_axis=1),
            axis=1,
        ),
        BatchedArray(np.array([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]]), batch_axis=1),
    )


def test_shuffle_along_axis_same_random_seed() -> None:
    batch = BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert objects_are_equal(
        ba.shuffle_along_axis(batch, axis=0, generator=np.random.default_rng(1)),
        ba.shuffle_along_axis(batch, axis=0, generator=np.random.default_rng(1)),
    )


def test_shuffle_along_axis_different_random_seeds() -> None:
    batch = BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert not objects_are_equal(
        ba.shuffle_along_axis(batch, axis=0, generator=np.random.default_rng(1)),
        ba.shuffle_along_axis(batch, axis=0, generator=np.random.default_rng(2)),
    )


@patch("redcat.ba.core.randperm", lambda *args, **kwargs: np.array([2, 1, 3, 0]))
def test_shuffle_along_batch() -> None:
    assert objects_are_equal(
        ba.shuffle_along_batch(
            BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
        ),
        BatchedArray(np.array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])),
    )


@patch("redcat.ba.core.randperm", lambda *args, **kwargs: np.array([2, 1, 3, 0]))
def test_shuffle_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.shuffle_along_batch(
            BatchedArray(np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]), batch_axis=1)
        ),
        BatchedArray(np.array([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]]), batch_axis=1),
    )


def test_shuffle_along_batch_same_random_seed() -> None:
    batch = BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert objects_are_equal(
        ba.shuffle_along_batch(batch, generator=np.random.default_rng(1)),
        ba.shuffle_along_batch(batch, generator=np.random.default_rng(1)),
    )


def test_shuffle_along_batch_different_random_seeds() -> None:
    batch = BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert not objects_are_equal(
        ba.shuffle_along_batch(batch, generator=np.random.default_rng(1)),
        ba.shuffle_along_batch(batch, generator=np.random.default_rng(2)),
    )


def test_sort() -> None:
    assert objects_are_equal(
        ba.sort(BatchedArray(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]))),
        BatchedArray(np.asarray([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])),
    )


def test_sort_axis_0() -> None:
    assert objects_are_equal(
        ba.sort(BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=0),
        BatchedArray(np.asarray([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])),
    )


def test_sort_axis_1() -> None:
    assert objects_are_equal(
        ba.sort(
            BatchedArray(
                np.asarray(
                    [
                        [[0, 1], [-2, 3], [-4, 5], [-6, 7], [-8, 9]],
                        [[10, -11], [12, -13], [14, -15], [16, -17], [18, -19]],
                    ]
                )
            ),
            axis=1,
        ),
        BatchedArray(
            np.asarray(
                [
                    [[-8, 1], [-6, 3], [-4, 5], [-2, 7], [0, 9]],
                    [[10, -19], [12, -17], [14, -15], [16, -13], [18, -11]],
                ]
            )
        ),
    )


def test_sort_custom_axes() -> None:
    assert objects_are_equal(
        ba.sort(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), batch_axis=1), axis=0
        ),
        BatchedArray(np.asarray([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]), batch_axis=1),
    )


def test_sort_along_batch() -> None:
    assert objects_are_equal(
        ba.sort_along_batch(BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]))),
        BatchedArray(np.asarray([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])),
    )


def test_sort_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.sort_along_batch(
            BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)
        ),
        BatchedArray(np.asarray([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]), batch_axis=1),
    )


#####################
#     Pointwise     #
#####################


def test_absolute() -> None:
    assert objects_are_equal(
        ba.absolute(BatchedArray(np.asarray([[-1, 2], [0, 0], [1, -2]]))),
        BatchedArray(np.asarray([[1, 2], [0, 0], [1, 2]])),
    )


def test_absolute_custom_axes() -> None:
    assert objects_are_equal(
        ba.absolute(BatchedArray(np.asarray([[-1, 2], [0, 0], [1, -2]]), batch_axis=1)),
        BatchedArray(np.asarray([[1, 2], [0, 0], [1, 2]]), batch_axis=1),
    )


def test_clip() -> None:
    assert objects_are_equal(
        ba.clip(BatchedArray(np.asarray([[-1, 2], [0, 0], [1, -2]])), a_min=-1.0, a_max=1.5),
        BatchedArray(np.asarray([[-1.0, 1.5], [0.0, 0.0], [1.0, -1.0]])),
    )


def test_clip_custom_axes() -> None:
    assert objects_are_equal(
        ba.clip(
            BatchedArray(np.asarray([[-1, 2], [0, 0], [1, -2]]), batch_axis=1),
            a_min=-1.0,
            a_max=1.5,
        ),
        BatchedArray(np.asarray([[-1.0, 1.5], [0.0, 0.0], [1.0, -1.0]]), batch_axis=1),
    )


def test_exp() -> None:
    assert objects_are_allclose(
        ba.exp(BatchedArray(np.asarray([[1, 2, 3], [4, 5, 6]]))),
        BatchedArray(
            np.asarray(
                [
                    [2.7182817459106445, 7.389056205749512, 20.08553695678711],
                    [54.598148345947266, 148.4131622314453, 403.4288024902344],
                ]
            )
        ),
        atol=1e-6,
    )


def test_exp_custom_axes() -> None:
    assert objects_are_allclose(
        ba.exp(BatchedArray(np.asarray([[1, 2, 3], [4, 5, 6]]), batch_axis=1)),
        BatchedArray(
            np.asarray(
                [
                    [2.7182817459106445, 7.389056205749512, 20.08553695678711],
                    [54.598148345947266, 148.4131622314453, 403.4288024902344],
                ]
            ),
            batch_axis=1,
        ),
        atol=1e-6,
    )


def test_exp2() -> None:
    assert objects_are_allclose(
        ba.exp2(BatchedArray(np.asarray([[1, 2, 3], [4, 5, 6]]))),
        BatchedArray(np.asarray([[2.0, 4.0, 8.0], [16.0, 32.0, 64.0]])),
        atol=1e-6,
    )


def test_exp2_custom_axes() -> None:
    assert objects_are_allclose(
        ba.exp2(BatchedArray(np.asarray([[1, 2, 3], [4, 5, 6]]), batch_axis=1)),
        BatchedArray(
            np.asarray([[2.0, 4.0, 8.0], [16.0, 32.0, 64.0]]),
            batch_axis=1,
        ),
        atol=1e-6,
    )


def test_expm1() -> None:
    assert objects_are_allclose(
        ba.expm1(BatchedArray(np.asarray([[0, 1, 2], [3, 4, 5]]))),
        BatchedArray(
            np.asarray(
                [
                    [0.0, 1.7182818284590453, 6.38905609893065],
                    [19.085536923187668, 53.598150033144236, 147.4131591025766],
                ]
            )
        ),
        atol=1e-6,
    )


def test_expm1_custom_axes() -> None:
    assert objects_are_allclose(
        ba.expm1(BatchedArray(np.asarray([[0, 1, 2], [3, 4, 5]]), batch_axis=1)),
        BatchedArray(
            np.asarray(
                [
                    [0.0, 1.7182818284590453, 6.38905609893065],
                    [19.085536923187668, 53.598150033144236, 147.4131591025766],
                ]
            ),
            batch_axis=1,
        ),
        atol=1e-6,
    )


def test_fabs() -> None:
    assert objects_are_equal(
        ba.fabs(BatchedArray(np.asarray([[-1, 2], [0, 0], [1, -2]]))),
        BatchedArray(np.asarray([[1.0, 2.0], [0.0, 0.0], [1.0, 2.0]])),
    )


def test_fabs_custom_axes() -> None:
    assert objects_are_equal(
        ba.fabs(BatchedArray(np.asarray([[-1, 2], [0, 0], [1, -2]]), batch_axis=1)),
        BatchedArray(np.asarray([[1.0, 2.0], [0.0, 0.0], [1.0, 2.0]]), batch_axis=1),
    )


@pytest.mark.parametrize(
    "exponent",
    (BatchedArray(np.full((2, 5), 2.0)), np.full((2, 5), 2.0), 2, 2.0),
)
def test_float_power(exponent: BatchedArray | np.ndarray) -> None:
    assert objects_are_allclose(
        np.float_power(BatchedArray(np.arange(10).reshape(2, 5)), exponent),
        BatchedArray(np.array([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=float)),
    )


def test_float_power_custom_axes() -> None:
    assert objects_are_allclose(
        np.float_power(
            BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1),
            BatchedArray(np.full((2, 5), 2.0), batch_axis=1),
        ),
        BatchedArray(np.array([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=float), batch_axis=1),
    )


@future_test
def test_float_power_different_axes() -> None:
    with pytest.raises(RuntimeError, match="The batch axes do not match."):
        np.float_power(
            BatchedArray(np.arange(10).reshape(2, 5)),
            BatchedArray(np.full((2, 5), 2.0), batch_axis=1),
        )


@pytest.mark.parametrize(
    "other",
    (
        BatchedArray(np.array([[2, 0, 1], [0, 1, 0]])),
        np.array([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_fmax(other: BatchedArray | np.ndarray) -> None:
    assert objects_are_allclose(
        np.fmax(
            BatchedArray(np.array([[0, 1, 2], [-2, -1, 0]])),
            other,
        ),
        BatchedArray(np.array([[2, 1, 2], [0, 1, 0]])),
    )


def test_fmax_custom_axes() -> None:
    assert objects_are_allclose(
        np.fmax(
            BatchedArray(np.array([[0, 1, 2], [-2, -1, 0]]), batch_axis=1),
            BatchedArray(np.array([[2, 0, 1], [0, 1, 0]]), batch_axis=1),
        ),
        BatchedArray(np.array([[2, 1, 2], [0, 1, 0]]), batch_axis=1),
    )


@future_test
def test_fmax_different_axes() -> None:
    with pytest.raises(RuntimeError, match="The batch axes do not match."):
        np.fmax(
            BatchedArray(np.array([[0, 1, 2], [-2, -1, 0]])),
            BatchedArray(np.array([[2, 0, 1], [0, 1, 0]]), batch_axis=1),
        )


@pytest.mark.parametrize(
    "other",
    (
        BatchedArray(np.array([[2, 0, 1], [0, 1, 0]])),
        np.array([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_fmin(other: BatchedArray | np.ndarray) -> None:
    assert objects_are_allclose(
        np.fmin(BatchedArray(np.array([[0, 1, 2], [-2, -1, 0]])), other),
        BatchedArray(np.array([[0, 0, 1], [-2, -1, 0]])),
    )


def test_fmin_custom_axes() -> None:
    assert objects_are_allclose(
        np.fmin(
            BatchedArray(np.array([[0, 1, 2], [-2, -1, 0]]), batch_axis=1),
            BatchedArray(np.array([[2, 0, 1], [0, 1, 0]]), batch_axis=1),
        ),
        BatchedArray(np.array([[0, 0, 1], [-2, -1, 0]]), batch_axis=1),
    )


@future_test
def test_fmin_different_axes() -> None:
    with pytest.raises(RuntimeError, match="The batch axes do not match."):
        np.fmin(
            BatchedArray(np.array([[0, 1, 2], [-2, -1, 0]])),
            BatchedArray(np.array([[2, 0, 1], [0, 1, 0]]), batch_axis=1),
        )


def test_log() -> None:
    assert objects_are_allclose(
        ba.log(BatchedArray(np.asarray([[1, 2, 3], [4, 5, 6]]))),
        BatchedArray(
            np.asarray(
                [
                    [0.0, 0.6931471824645996, 1.0986123085021973],
                    [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                ]
            )
        ),
        atol=1e-6,
    )


def test_log_custom_axes() -> None:
    assert objects_are_allclose(
        ba.log(BatchedArray(np.asarray([[1, 2, 3], [4, 5, 6]]), batch_axis=1)),
        BatchedArray(
            np.asarray(
                [
                    [0.0, 0.6931471824645996, 1.0986123085021973],
                    [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                ]
            ),
            batch_axis=1,
        ),
        atol=1e-6,
    )


def test_log10() -> None:
    assert objects_are_allclose(
        ba.log10(BatchedArray(np.asarray([[1, 2, 3], [4, 5, 6]]))),
        BatchedArray(
            np.asarray(
                [
                    [0.0, 0.3010300099849701, 0.4771212637424469],
                    [0.6020600199699402, 0.6989700198173523, 0.778151273727417],
                ]
            )
        ),
        atol=1e-6,
    )


def test_log10_custom_axes() -> None:
    assert objects_are_allclose(
        ba.log10(BatchedArray(np.asarray([[1, 2, 3], [4, 5, 6]]), batch_axis=1)),
        BatchedArray(
            np.asarray(
                [
                    [0.0, 0.3010300099849701, 0.4771212637424469],
                    [0.6020600199699402, 0.6989700198173523, 0.778151273727417],
                ]
            ),
            batch_axis=1,
        ),
        atol=1e-6,
    )


def test_log1p() -> None:
    assert objects_are_allclose(
        ba.log1p(BatchedArray(np.asarray([[0, 1, 2], [3, 4, 5]]))),
        BatchedArray(
            np.asarray(
                [
                    [0.0, 0.6931471824645996, 1.0986123085021973],
                    [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                ]
            )
        ),
        atol=1e-6,
    )


def test_log1p_custom_axes() -> None:
    assert objects_are_allclose(
        ba.log1p(BatchedArray(np.asarray([[0, 1, 2], [3, 4, 5]]), batch_axis=1)),
        BatchedArray(
            np.asarray(
                [
                    [0.0, 0.6931471824645996, 1.0986123085021973],
                    [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                ]
            ),
            batch_axis=1,
        ),
        atol=1e-6,
    )


def test_log2() -> None:
    assert objects_are_allclose(
        ba.log2(BatchedArray(np.asarray([[1, 2, 3], [4, 5, 6]]))),
        BatchedArray(
            np.asarray(
                [[0.0, 1.0, 1.5849624872207642], [2.0, 2.321928024291992, 2.5849626064300537]]
            )
        ),
        atol=1e-6,
    )


def test_log2_custom_axes() -> None:
    assert objects_are_allclose(
        ba.log2(BatchedArray(np.asarray([[1, 2, 3], [4, 5, 6]]), batch_axis=1)),
        BatchedArray(
            np.asarray(
                [[0.0, 1.0, 1.5849624872207642], [2.0, 2.321928024291992, 2.5849626064300537]]
            ),
            batch_axis=1,
        ),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "other",
    (
        BatchedArray(np.array([[2, 0, 1], [0, 1, 0]])),
        np.array([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_maximum(other: BatchedArray | np.ndarray) -> None:
    assert objects_are_allclose(
        np.maximum(
            BatchedArray(np.array([[0, 1, 2], [-2, -1, 0]])),
            other,
        ),
        BatchedArray(np.array([[2, 1, 2], [0, 1, 0]])),
    )


def test_maximum_custom_axes() -> None:
    assert objects_are_allclose(
        np.maximum(
            BatchedArray(np.array([[0, 1, 2], [-2, -1, 0]]), batch_axis=1),
            BatchedArray(np.array([[2, 0, 1], [0, 1, 0]]), batch_axis=1),
        ),
        BatchedArray(np.array([[2, 1, 2], [0, 1, 0]]), batch_axis=1),
    )


@future_test
def test_maximum_different_axes() -> None:
    with pytest.raises(RuntimeError, match="The batch axes do not match."):
        np.maximum(
            BatchedArray(np.array([[0, 1, 2], [-2, -1, 0]])),
            BatchedArray(np.array([[2, 0, 1], [0, 1, 0]]), batch_axis=1),
        )


@pytest.mark.parametrize(
    "other",
    (
        BatchedArray(np.array([[2, 0, 1], [0, 1, 0]])),
        np.array([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_minimum(other: BatchedArray | np.ndarray) -> None:
    assert objects_are_allclose(
        np.minimum(BatchedArray(np.array([[0, 1, 2], [-2, -1, 0]])), other),
        BatchedArray(np.array([[0, 0, 1], [-2, -1, 0]])),
    )


def test_minimum_custom_axes() -> None:
    assert objects_are_allclose(
        np.minimum(
            BatchedArray(np.array([[0, 1, 2], [-2, -1, 0]]), batch_axis=1),
            BatchedArray(np.array([[2, 0, 1], [0, 1, 0]]), batch_axis=1),
        ),
        BatchedArray(np.array([[0, 0, 1], [-2, -1, 0]]), batch_axis=1),
    )


@future_test
def test_minimum_different_axes() -> None:
    with pytest.raises(RuntimeError, match="The batch axes do not match."):
        np.minimum(
            BatchedArray(np.array([[0, 1, 2], [-2, -1, 0]])),
            BatchedArray(np.array([[2, 0, 1], [0, 1, 0]]), batch_axis=1),
        )


@pytest.mark.parametrize(
    "exponent",
    (BatchedArray(np.full((2, 5), 2.0)), np.full((2, 5), 2.0), 2, 2.0),
)
def test_power(exponent: BatchedArray | np.ndarray) -> None:
    assert objects_are_allclose(
        np.power(BatchedArray(np.arange(10, dtype=float).reshape(2, 5)), exponent),
        BatchedArray(np.array([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=float)),
    )


def test_power_custom_axes() -> None:
    assert objects_are_allclose(
        np.power(
            BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1),
            BatchedArray(np.full((2, 5), 2.0), batch_axis=1),
        ),
        BatchedArray(np.array([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=float), batch_axis=1),
    )


@future_test
def test_power_different_axes() -> None:
    with pytest.raises(RuntimeError, match="The batch axes do not match."):
        np.power(
            BatchedArray(np.arange(10).reshape(2, 5)),
            BatchedArray(np.full((2, 5), 2.0), batch_axis=1),
        )


def test_sign() -> None:
    assert objects_are_allclose(
        ba.sign(BatchedArray(np.asarray([[-1, 2], [0, 0], [1, -2]]))),
        BatchedArray(np.asarray([[-1, 1], [0, 0], [1, -1]])),
        atol=1e-6,
    )


def test_sign_custom_axes() -> None:
    assert objects_are_allclose(
        ba.sign(BatchedArray(np.asarray([[-1, 2], [0, 0], [1, -2]]), batch_axis=1)),
        BatchedArray(
            np.asarray([[-1, 1], [0, 0], [1, -1]]),
            batch_axis=1,
        ),
        atol=1e-6,
    )


def test_sqrt() -> None:
    assert objects_are_allclose(
        ba.sqrt(BatchedArray(np.asarray([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]))),
        BatchedArray(np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])),
        atol=1e-6,
    )


def test_sqrt_custom_axes() -> None:
    assert objects_are_allclose(
        ba.sqrt(BatchedArray(np.asarray([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]), batch_axis=1)),
        BatchedArray(
            np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            batch_axis=1,
        ),
        atol=1e-6,
    )


def test_square() -> None:
    assert objects_are_allclose(
        ba.square(BatchedArray(np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))),
        BatchedArray(np.asarray([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]])),
        atol=1e-6,
    )


def test_square_custom_axes() -> None:
    assert objects_are_allclose(
        ba.square(BatchedArray(np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), batch_axis=1)),
        BatchedArray(
            np.asarray([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]),
            batch_axis=1,
        ),
        atol=1e-6,
    )


#####################
#     Reduction     #
#####################


def test_argmax_1d() -> None:
    assert objects_are_equal(
        ba.argmax(BatchedArray(np.asarray([4, 1, 2, 5, 3])), axis=0),
        np.int64(3),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_argmax_2d(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.argmax(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype)), axis=0
        ),
        np.asarray([3, 0]),
    )


def test_argmax_axis_none() -> None:
    assert objects_are_equal(
        ba.argmax(BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=None),
        np.int64(1),
    )


def test_argmax_custom_axes() -> None:
    assert objects_are_equal(
        ba.argmax(
            BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1), axis=1
        ),
        np.asarray([3, 0]),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_argmax_along_batch(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.argmax_along_batch(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype))
        ),
        np.asarray([3, 0]),
    )


def test_argmax_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.argmax_along_batch(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), keepdims=True
        ),
        np.asarray([[3, 0]]),
    )


def test_argmax_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.argmax_along_batch(
            BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)
        ),
        np.asarray([3, 0]),
    )


def test_argmin_1d() -> None:
    assert objects_are_equal(
        ba.argmin(BatchedArray(np.asarray([4, 1, 2, 5, 3])), axis=0),
        np.int64(1),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_argmin_2d(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.argmin(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype)), axis=0
        ),
        np.asarray([1, 2]),
    )


def test_argmin_axis_none() -> None:
    assert objects_are_equal(
        ba.argmin(BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=None),
        np.int64(2),
    )


def test_argmin_custom_axes() -> None:
    assert objects_are_equal(
        ba.argmin(
            BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1), axis=1
        ),
        np.asarray([1, 2]),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_argmin_along_batch(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.argmin_along_batch(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype))
        ),
        np.asarray([1, 2]),
    )


def test_argmin_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.argmin_along_batch(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), keepdims=True
        ),
        np.asarray([[1, 2]]),
    )


def test_argmin_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.argmin_along_batch(
            BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)
        ),
        np.asarray([1, 2]),
    )


def test_max_1d() -> None:
    assert objects_are_equal(
        ba.max(BatchedArray(np.asarray([4, 1, 2, 5, 3])), axis=0),
        np.int64(5),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_max_2d(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.max(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype)), axis=0
        ),
        np.asarray([5, 9], dtype=dtype),
    )


def test_max_axis_none() -> None:
    assert objects_are_equal(
        ba.max(BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=None),
        np.int64(9),
    )


def test_max_custom_axes() -> None:
    assert objects_are_equal(
        ba.max(BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1), axis=1),
        np.asarray([5, 9]),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_max_along_batch(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.max_along_batch(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype))
        ),
        np.asarray([5, 9], dtype=dtype),
    )


def test_max_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.max_along_batch(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), keepdims=True
        ),
        np.asarray([[5, 9]]),
    )


def test_max_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.max_along_batch(
            BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)
        ),
        np.asarray([5, 9]),
    )


def test_mean_1d() -> None:
    assert objects_are_equal(
        ba.mean(BatchedArray(np.asarray([4, 1, 2, 5, 3])), axis=0),
        np.float64(3.0),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_mean_2d(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.mean(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype)), axis=0
        ),
        np.asarray([3.0, 7.0]),
    )


def test_mean_axis_none() -> None:
    assert objects_are_equal(
        ba.mean(BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=None),
        np.float64(5.0),
    )


def test_mean_custom_axes() -> None:
    assert objects_are_equal(
        ba.mean(BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1), axis=1),
        np.asarray([3.0, 7.0]),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_mean_along_batch(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.mean_along_batch(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype))
        ),
        np.asarray([3.0, 7.0]),
    )


def test_mean_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.mean_along_batch(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), keepdims=True
        ),
        np.asarray([[3.0, 7.0]]),
    )


def test_mean_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.mean_along_batch(
            BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)
        ),
        np.asarray([3.0, 7.0]),
    )


def test_median_1d() -> None:
    assert objects_are_equal(
        ba.median(BatchedArray(np.asarray([4, 1, 2, 5, 3])), axis=0),
        np.float64(3.0),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_median_2d(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.median(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype)), axis=0
        ),
        np.asarray([3.0, 7.0]),
    )


def test_median_axis_none() -> None:
    assert objects_are_equal(
        ba.median(BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=None),
        np.float64(5.0),
    )


def test_median_custom_axes() -> None:
    assert objects_are_equal(
        ba.median(
            BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1), axis=1
        ),
        np.asarray([3.0, 7.0]),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_median_along_batch(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.median_along_batch(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype))
        ),
        np.asarray([3.0, 7.0]),
    )


def test_median_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.median_along_batch(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), keepdims=True
        ),
        np.asarray([[3.0, 7.0]]),
    )


def test_median_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.median_along_batch(
            BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)
        ),
        np.asarray([3.0, 7.0]),
    )


def test_min_1d() -> None:
    assert objects_are_equal(
        ba.min(BatchedArray(np.asarray([4, 1, 2, 5, 3])), axis=0),
        np.int64(1),
    )


def test_min_2d() -> None:
    assert objects_are_equal(
        ba.min(BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=0),
        np.asarray([1, 5]),
    )


def test_min_axis_none() -> None:
    assert objects_are_equal(
        ba.min(BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=None),
        np.int64(1),
    )


def test_min_custom_axes() -> None:
    assert objects_are_equal(
        ba.min(BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1), axis=1),
        np.asarray([1, 5]),
    )


def test_min_along_batch() -> None:
    assert objects_are_equal(
        ba.min_along_batch(BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]))),
        np.asarray([1, 5]),
    )


def test_min_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.min_along_batch(
            BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), keepdims=True
        ),
        np.asarray([[1, 5]]),
    )


def test_min_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.min_along_batch(
            BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)
        ),
        np.asarray([1, 5]),
    )


def test_nanargmax_1d() -> None:
    assert objects_are_equal(
        ba.nanargmax(BatchedArray(np.array([1, np.nan, 2])), axis=0),
        np.int64(2),
    )


def test_nanargmax_2d() -> None:
    assert objects_are_equal(
        ba.nanargmax(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=0),
        np.asarray([1, 1, 1]),
    )


def test_nanargmax_axis_none() -> None:
    assert objects_are_equal(
        ba.nanargmax(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=None),
        np.int64(5),
    )


def test_nanargmax_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanargmax(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1), axis=1),
        np.asarray([2, 2]),
    )


def test_nanargmax_along_batch() -> None:
    assert objects_are_equal(
        ba.nanargmax_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))),
        np.asarray([1, 1, 1]),
    )


def test_nanargmax_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.nanargmax_along_batch(
            BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), keepdims=True
        ),
        np.asarray([[1, 1, 1]]),
    )


def test_nanargmax_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanargmax_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)),
        np.asarray([2, 2]),
    )


def test_nanargmin_1d() -> None:
    assert objects_are_equal(
        ba.nanargmin(BatchedArray(np.array([1, np.nan, 2])), axis=0),
        np.int64(0),
    )


def test_nanargmin_2d() -> None:
    assert objects_are_equal(
        ba.nanargmin(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=0),
        np.asarray([0, 1, 0]),
    )


def test_nanargmin_axis_none() -> None:
    assert objects_are_equal(
        ba.nanargmin(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=None),
        np.int64(0),
    )


def test_nanargmin_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanargmin(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1), axis=1),
        np.asarray([0, 0]),
    )


def test_nanargmin_along_batch() -> None:
    assert objects_are_equal(
        ba.nanargmin_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))),
        np.asarray([0, 1, 0]),
    )


def test_nanargmin_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.nanargmin_along_batch(
            BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), keepdims=True
        ),
        np.asarray([[0, 1, 0]]),
    )


def test_nanargmin_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanargmin_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)),
        np.asarray([0, 0]),
    )


def test_nanmax_1d() -> None:
    assert objects_are_equal(
        ba.nanmax(BatchedArray(np.array([1, np.nan, 2])), axis=0),
        np.float64(2.0),
    )


def test_nanmax_2d() -> None:
    assert objects_are_equal(
        ba.nanmax(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=0),
        np.asarray([3.0, 4.0, 5.0]),
    )


def test_nanmax_axis_none() -> None:
    assert objects_are_equal(
        ba.nanmax(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=None),
        np.float64(5.0),
    )


def test_nanmax_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanmax(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1), axis=1),
        np.asarray([2.0, 5.0]),
    )


def test_nanmax_along_batch() -> None:
    assert objects_are_equal(
        ba.nanmax_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))),
        np.asarray([3.0, 4.0, 5.0]),
    )


def test_nanmax_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.nanmax_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), keepdims=True),
        np.asarray([[3.0, 4.0, 5.0]]),
    )


def test_nanmax_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanmax_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)),
        np.asarray([2.0, 5.0]),
    )


def test_nanmean_1d() -> None:
    assert objects_are_equal(
        ba.nanmean(BatchedArray(np.array([1, np.nan, 2])), axis=0),
        np.float64(1.5),
    )


def test_nanmean_2d() -> None:
    assert objects_are_equal(
        ba.nanmean(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=0),
        np.asarray([2.0, 4.0, 3.5]),
    )


def test_nanmean_axis_none() -> None:
    assert objects_are_equal(
        ba.nanmean(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=None),
        np.float64(3.0),
    )


def test_nanmean_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanmean(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1), axis=1),
        np.asarray([1.5, 4.0]),
    )


def test_nanmean_along_batch() -> None:
    assert objects_are_equal(
        ba.nanmean_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))),
        np.asarray([2.0, 4.0, 3.5]),
    )


def test_nanmean_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.nanmean_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), keepdims=True),
        np.asarray([[2.0, 4.0, 3.5]]),
    )


def test_nanmean_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanmean_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)),
        np.asarray([1.5, 4.0]),
    )


def test_nanmedian_1d() -> None:
    assert objects_are_equal(
        ba.nanmedian(BatchedArray(np.array([1, np.nan, 2])), axis=0),
        np.float64(1.5),
    )


def test_nanmedian_2d() -> None:
    assert objects_are_equal(
        ba.nanmedian(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=0),
        np.asarray([2.0, 4.0, 3.5]),
    )


def test_nanmedian_axis_none() -> None:
    assert objects_are_equal(
        ba.nanmedian(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=None),
        np.float64(3.0),
    )


def test_nanmedian_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanmedian(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1), axis=1),
        np.asarray([1.5, 4.0]),
    )


def test_nanmedian_along_batch() -> None:
    assert objects_are_equal(
        ba.nanmedian_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))),
        np.asarray([2.0, 4.0, 3.5]),
    )


def test_nanmedian_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.nanmedian_along_batch(
            BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), keepdims=True
        ),
        np.asarray([[2.0, 4.0, 3.5]]),
    )


def test_nanmedian_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanmedian_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)),
        np.asarray([1.5, 4.0]),
    )


def test_nanmin_1d() -> None:
    assert objects_are_equal(
        ba.nanmin(BatchedArray(np.array([1, np.nan, 2])), axis=0),
        np.float64(1.0),
    )


def test_nanmin_2d() -> None:
    assert objects_are_equal(
        ba.nanmin(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=0),
        np.asarray([1.0, 4.0, 2.0]),
    )


def test_nanmin_axis_none() -> None:
    assert objects_are_equal(
        ba.nanmin(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=None),
        np.float64(1.0),
    )


def test_nanmin_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanmin(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1), axis=1),
        np.asarray([1.0, 3.0]),
    )


def test_nanmin_along_batch() -> None:
    assert objects_are_equal(
        ba.nanmin_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))),
        np.asarray([1.0, 4.0, 2.0]),
    )


def test_nanmin_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.nanmin_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), keepdims=True),
        np.asarray([[1.0, 4.0, 2.0]]),
    )


def test_nanmin_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanmin_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)),
        np.asarray([1.0, 3.0]),
    )


def test_nanprod_1d() -> None:
    assert objects_are_equal(
        ba.nanprod(BatchedArray(np.array([1, np.nan, 2])), axis=0),
        np.float64(2.0),
    )


def test_nanprod_2d() -> None:
    assert objects_are_equal(
        ba.nanprod(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=0),
        np.asarray([3.0, 4.0, 10.0]),
    )


def test_nanprod_axis_none() -> None:
    assert objects_are_equal(
        ba.nanprod(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=None),
        np.float64(120.0),
    )


def test_nanprod_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanprod(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1), axis=1),
        np.asarray([2.0, 60.0]),
    )


def test_nanprod_along_batch() -> None:
    assert objects_are_equal(
        ba.nanprod_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))),
        np.asarray([3.0, 4.0, 10.0]),
    )


def test_nanprod_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.nanprod_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), keepdims=True),
        np.asarray([[3.0, 4.0, 10.0]]),
    )


def test_nanprod_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanprod_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)),
        np.asarray([2.0, 60.0]),
    )


def test_nansum_1d() -> None:
    assert objects_are_equal(
        ba.nansum(BatchedArray(np.array([1, np.nan, 2])), axis=0),
        np.float64(3.0),
    )


def test_nansum_2d() -> None:
    assert objects_are_equal(
        ba.nansum(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=0),
        np.asarray([4.0, 4.0, 7.0]),
    )


def test_nansum_axis_none() -> None:
    assert objects_are_equal(
        ba.nansum(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=None),
        np.float64(15.0),
    )


def test_nansum_custom_axes() -> None:
    assert objects_are_equal(
        ba.nansum(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1), axis=1),
        np.asarray([3.0, 12]),
    )


def test_nansum_along_batch() -> None:
    assert objects_are_equal(
        ba.nansum_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))),
        np.asarray([4.0, 4.0, 7.0]),
    )


def test_nansum_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.nansum_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), keepdims=True),
        np.asarray([[4.0, 4.0, 7.0]]),
    )


def test_nansum_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nansum_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)),
        np.asarray([3.0, 12]),
    )


def test_prod_1d() -> None:
    assert objects_are_equal(
        ba.prod(BatchedArray(np.array([1, 3, 2])), axis=0),
        np.int64(6),
    )


def test_prod_2d() -> None:
    assert objects_are_equal(
        ba.prod(BatchedArray(np.array([[1, 3, 2], [3, 4, 5]])), axis=0),
        np.asarray([3, 12, 10]),
    )


def test_prod_axis_none() -> None:
    assert objects_are_equal(
        ba.prod(BatchedArray(np.array([[1, 3, 2], [3, 4, 5]])), axis=None),
        np.int64(360),
    )


def test_prod_custom_axes() -> None:
    assert objects_are_equal(
        ba.prod(BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]), batch_axis=1), axis=1),
        np.asarray([6, 60]),
    )


def test_prod_along_batch() -> None:
    assert objects_are_equal(
        ba.prod_along_batch(BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]))),
        np.asarray([3, 12, 10]),
    )


def test_prod_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.prod_along_batch(BatchedArray(np.array([[1, 3, 2], [3, 4, 5]])), keepdims=True),
        np.asarray([[3, 12, 10]]),
    )


def test_prod_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.prod_along_batch(BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]), batch_axis=1)),
        np.asarray([6, 60]),
    )


def test_sum_1d() -> None:
    assert objects_are_equal(
        ba.sum(BatchedArray(np.array([1, 3, 2])), axis=0),
        np.int64(6),
    )


def test_sum_2d() -> None:
    assert objects_are_equal(
        ba.sum(BatchedArray(np.array([[1, 3, 2], [3, 4, 5]])), axis=0),
        np.asarray([4, 7, 7]),
    )


def test_sum_axis_none() -> None:
    assert objects_are_equal(
        ba.sum(BatchedArray(np.array([[1, 3, 2], [3, 4, 5]])), axis=None),
        np.int64(18),
    )


def test_sum_custom_axes() -> None:
    assert objects_are_equal(
        ba.sum(BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]), batch_axis=1), axis=1),
        np.asarray([6, 12]),
    )


def test_sum_along_batch() -> None:
    assert objects_are_equal(
        ba.sum_along_batch(BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]))),
        np.asarray([4, 7, 7]),
    )


def test_sum_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.sum_along_batch(BatchedArray(np.array([[1, 3, 2], [3, 4, 5]])), keepdims=True),
        np.asarray([[4, 7, 7]]),
    )


def test_sum_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.sum_along_batch(BatchedArray(np.array([[1, 3, 2], [3, 4, 5]]), batch_axis=1)),
        np.asarray([6, 12]),
    )
