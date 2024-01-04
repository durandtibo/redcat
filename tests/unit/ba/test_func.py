from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from redcat import ba
from redcat.ba import BatchedArray

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
