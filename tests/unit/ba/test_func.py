from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from redcat import ba
from redcat.ba import BatchedArray
from tests.conftest import future_test

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
