from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from redcat import ba2 as ba
from tests.unit.ba2.test_core import NUMERIC_DTYPES

########################
#    Tests for add     #
########################


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_add(other: np.ndarray | int | float) -> None:
    assert ba.add(ba.ones((2, 3)), other).allequal(ba.full(shape=(2, 3), fill_value=3.0))


def test_batched_array_add_batch_axis_1() -> None:
    assert (
        ba.ones(shape=(2, 3), batch_axis=1)
        .add(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
        .allequal(ba.full(shape=(2, 3), fill_value=3.0, batch_axis=1))
    )


def test_batched_array_add_different_axes() -> None:
    batch = ba.ones(shape=(2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.add(ba.ones(shape=(2, 2), batch_axis=1))


###########################
#    Tests for divide     #
###########################


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_divide(other: np.ndarray | int | float) -> None:
    assert ba.divide(ba.ones((2, 3)), other).allequal(ba.full(shape=(2, 3), fill_value=0.5))


def test_batched_array_divide_custom_axes() -> None:
    assert ba.divide(
        ba.ones(shape=(2, 3), batch_axis=1), ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    ).allequal(ba.full(shape=(2, 3), fill_value=0.5, batch_axis=1))


def test_batched_array_divide_incorrect_batch_axis() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        ba.divide(batch, ba.ones(shape=(2, 2), batch_axis=1))


#################################
#    Tests for floor_divide     #
#################################


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_floor_divide(other: np.ndarray | int | float) -> None:
    assert ba.floor_divide(ba.ones((2, 3)), other).allequal(ba.zeros(shape=(2, 3)))


def test_batched_array_floor_divide_custom_axes() -> None:
    assert ba.floor_divide(
        ba.ones(shape=(2, 3), batch_axis=1), ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    ).allequal(ba.zeros(shape=(2, 3), batch_axis=1))


def test_batched_array_floor_divide_incorrect_batch_axis() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        ba.floor_divide(batch, ba.ones(shape=(2, 2), batch_axis=1))


#############################
#    Tests for multiply     #
#############################


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_multiply(other: np.ndarray | int | float) -> None:
    assert ba.multiply(ba.ones((2, 3)), other).allequal(ba.full(shape=(2, 3), fill_value=2.0))


def test_batched_array_multiply_custom_axes() -> None:
    assert ba.multiply(
        ba.ones(shape=(2, 3), batch_axis=1), ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    ).allequal(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))


def test_batched_array_multiply_incorrect_batch_axis() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        ba.multiply(batch, ba.ones(shape=(2, 2), batch_axis=1))


#############################
#    Tests for subtract     #
#############################


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_subtract(other: np.ndarray | int | float) -> None:
    assert ba.subtract(ba.ones((2, 3)), other).allequal(ba.full(shape=(2, 3), fill_value=-1.0))


def test_batched_array_subtract_custom_axes() -> None:
    assert ba.subtract(
        ba.ones(shape=(2, 3), batch_axis=1), ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    ).allequal(ba.full(shape=(2, 3), fill_value=-1.0, batch_axis=1))


def test_batched_array_subtract_incorrect_batch_axis() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        ba.subtract(batch, ba.ones(shape=(2, 2), batch_axis=1))


################################
#    Tests for true_divide     #
################################


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_true_divide(other: np.ndarray | int | float) -> None:
    assert ba.true_divide(ba.ones((2, 3)), other).allequal(ba.full(shape=(2, 3), fill_value=0.5))


def test_batched_array_true_divide_custom_axes() -> None:
    assert ba.true_divide(
        ba.ones(shape=(2, 3), batch_axis=1), ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    ).allequal(ba.full(shape=(2, 3), fill_value=0.5, batch_axis=1))


def test_batched_array_true_divide_incorrect_batch_axis() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        ba.true_divide(batch, ba.ones(shape=(2, 2), batch_axis=1))


############################
#    Tests for cumprod     #
############################


def test_batched_array_cumprod() -> None:
    assert objects_are_equal(
        ba.cumprod(ba.BatchedArray(np.arange(10).reshape(2, 5) + 1)),
        np.array([1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]),
    )


def test_batched_array_cumprod_axis_0() -> None:
    assert objects_are_equal(
        ba.cumprod(ba.BatchedArray(np.arange(10).reshape(2, 5)), axis=0),
        ba.array([[0, 1, 2, 3, 4], [0, 6, 14, 24, 36]]),
    )


def test_batched_array_cumprod_axis_1() -> None:
    assert objects_are_equal(
        ba.cumprod(ba.BatchedArray(np.arange(10).reshape(2, 5)), axis=1),
        ba.array([[0, 0, 0, 0, 0], [5, 30, 210, 1680, 15120]]),
    )


def test_batched_array_cumprod_custom_axes() -> None:
    assert objects_are_equal(
        ba.cumprod(ba.BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1), axis=0),
        ba.array([[0, 1], [0, 3], [0, 15], [0, 105], [0, 945]], batch_axis=1),
    )


def test_batched_array_cumprod_out() -> None:
    out = np.zeros((5, 2), dtype=np.int64)
    assert ba.cumprod(ba.BatchedArray(np.arange(10).reshape(5, 2)), axis=0, out=out) is out
    assert objects_are_equal(out, np.array([[0, 1], [0, 3], [0, 15], [0, 105], [0, 945]]))


def test_batched_array_cumprod_out_array() -> None:
    out = np.zeros(10)
    assert ba.cumprod(ba.BatchedArray(np.arange(10).reshape(2, 5) + 1), out=out) is out
    assert objects_are_equal(
        out, np.array([1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0])
    )


########################################
#    Tests for cumprod_along_batch     #
########################################


def test_batched_array_cumprod_along_batch() -> None:
    assert objects_are_equal(
        ba.cumprod_along_batch(ba.BatchedArray(np.arange(10).reshape(2, 5))),
        ba.array([[0, 1, 2, 3, 4], [0, 6, 14, 24, 36]]),
    )


def test_batched_array_cumprod_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.cumprod_along_batch(ba.BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1)),
        ba.array([[0, 0], [2, 6], [4, 20], [6, 42], [8, 72]], batch_axis=1),
    )


###########################
#    Tests for cumsum     #
###########################


def test_batched_array_cumsum() -> None:
    assert objects_are_equal(
        ba.cumsum(ba.BatchedArray(np.arange(10).reshape(2, 5) + 1)),
        np.array([1, 3, 6, 10, 15, 21, 28, 36, 45, 55]),
    )


def test_batched_array_cumsum_axis_0() -> None:
    assert objects_are_equal(
        ba.cumsum(ba.BatchedArray(np.arange(10).reshape(2, 5)), axis=0),
        ba.array([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]]),
    )


def test_batched_array_cumsum_axis_1() -> None:
    assert objects_are_equal(
        ba.cumsum(ba.BatchedArray(np.arange(10).reshape(2, 5)), axis=1),
        ba.array([[0, 1, 3, 6, 10], [5, 11, 18, 26, 35]]),
    )


def test_batched_array_cumsum_custom_axes() -> None:
    assert objects_are_equal(
        ba.cumsum(ba.BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1), axis=0),
        ba.array([[0, 1], [2, 4], [6, 9], [12, 16], [20, 25]], batch_axis=1),
    )


def test_batched_array_cumsum_out() -> None:
    out = np.zeros((5, 2), dtype=np.int64)
    assert ba.cumsum(ba.BatchedArray(np.arange(10).reshape(5, 2)), axis=0, out=out) is out
    assert objects_are_equal(out, np.array([[0, 1], [2, 4], [6, 9], [12, 16], [20, 25]]))


def test_batched_array_cumsum_out_array() -> None:
    out = np.zeros(10)
    assert ba.cumsum(ba.BatchedArray(np.arange(10).reshape(2, 5) + 1), out=out) is out
    assert objects_are_equal(
        out, np.array([1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0, 55.0])
    )


#######################################
#    Tests for cumsum_along_batch     #
#######################################


def test_batched_array_cumsum_along_batch() -> None:
    assert objects_are_equal(
        ba.cumsum_along_batch(ba.BatchedArray(np.arange(10).reshape(2, 5))),
        ba.array([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]]),
    )


def test_batched_array_cumsum_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.cumsum_along_batch(ba.BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1)),
        ba.array([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]], batch_axis=1),
    )


#########################
#    Tests for diff     #
#########################


def test_batched_array_diff() -> None:
    assert objects_are_equal(
        ba.diff(ba.array([[9, 3, 7, 4, 0], [6, 6, 2, 3, 3]])),
        np.array([[-6, 4, -3, -4], [0, -4, 1, 0]]),
    )


def test_batched_array_diff_axis_0() -> None:
    assert objects_are_equal(
        ba.diff(ba.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]]), axis=0),
        np.array([[0, -1], [1, 7], [-7, -9], [6, 7]]),
    )


def test_batched_array_diff_axis_1() -> None:
    assert objects_are_equal(
        ba.diff(ba.array([[9, 3, 7, 4, 0], [6, 6, 2, 3, 3]]), axis=1),
        np.array([[-6, 4, -3, -4], [0, -4, 1, 0]]),
    )


def test_batched_array_diff_n_0() -> None:
    assert objects_are_equal(
        ba.diff(ba.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]]), axis=0, n=0),
        np.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]]),
    )


def test_batched_array_diff_n_2() -> None:
    assert objects_are_equal(
        ba.diff(ba.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]]), axis=0, n=2),
        np.array([[1, 8], [-8, -16], [13, 16]]),
    )


def test_batched_array_diff_prepend() -> None:
    assert objects_are_equal(
        ba.diff(
            ba.BatchedArray(np.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]])),
            axis=0,
            prepend=np.array([[-1, -2]]),
        ),
        np.array([[7, 5], [0, -1], [1, 7], [-7, -9], [6, 7]]),
    )


def test_batched_array_diff_append() -> None:
    assert objects_are_equal(
        ba.diff(
            ba.BatchedArray(np.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]])),
            axis=0,
            append=np.array([[-1, -2]]),
        ),
        np.array([[0, -1], [1, 7], [-7, -9], [6, 7], [-7, -9]]),
    )


def test_batched_array_diff_prepend_append() -> None:
    assert objects_are_equal(
        ba.diff(
            ba.BatchedArray(np.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]])),
            axis=0,
            prepend=np.array([[-1, -2]]),
            append=np.array([[-1, -2]]),
        ),
        np.array([[7, 5], [0, -1], [1, 7], [-7, -9], [6, 7], [-7, -9]]),
    )


def test_batched_array_diff_custom_axes() -> None:
    assert objects_are_equal(
        ba.diff(ba.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]], batch_axis=1), axis=0),
        np.array([[0, -1], [1, 7], [-7, -9], [6, 7]]),
    )


######################################
#     Tests for diff_along_batch     #
######################################


def test_batched_array_diff_along_batch() -> None:
    assert objects_are_equal(
        ba.diff_along_batch(ba.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]])),
        np.array([[0, -1], [1, 7], [-7, -9], [6, 7]]),
    )


def test_batched_array_diff_along_batch_n_0() -> None:
    assert objects_are_equal(
        ba.diff_along_batch(ba.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]]), n=0),
        np.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]]),
    )


def test_batched_array_diff_along_batch_n_2() -> None:
    assert objects_are_equal(
        ba.diff_along_batch(ba.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]]), n=2),
        np.array([[1, 8], [-8, -16], [13, 16]]),
    )


def test_batched_array_diff_along_batch_prepend() -> None:
    assert objects_are_equal(
        ba.diff_along_batch(
            ba.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]]), prepend=np.array([[-1, -2]])
        ),
        np.array([[7, 5], [0, -1], [1, 7], [-7, -9], [6, 7]]),
    )


def test_batched_array_diff_along_batch_append() -> None:
    assert objects_are_equal(
        ba.diff_along_batch(
            ba.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]]), append=np.array([[-1, -2]])
        ),
        np.array([[0, -1], [1, 7], [-7, -9], [6, 7], [-7, -9]]),
    )


def test_batched_array_diff_along_batch_prepend_append() -> None:
    assert objects_are_equal(
        ba.diff_along_batch(
            ba.array([[6, 3], [6, 2], [7, 9], [0, 0], [6, 7]]),
            prepend=np.array([[-1, -2]]),
            append=np.array([[-1, -2]]),
        ),
        np.array([[7, 5], [0, -1], [1, 7], [-7, -9], [6, 7], [-7, -9]]),
    )


def test_batched_array_diff_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.diff_along_batch(ba.array([[9, 3, 7, 4, 0], [6, 6, 2, 3, 3]], batch_axis=1)),
        np.array([[-6, 4, -3, -4], [0, -4, 1, 0]]),
    )


###############################
#    Tests for nancumprod     #
###############################


def test_batched_array_nancumprod() -> None:
    assert objects_are_equal(
        ba.nancumprod(ba.array([[1, np.nan, 2], [3, 4, 5]])),
        np.array([1.0, 1.0, 2.0, 6.0, 24.0, 120.0]),
    )


def test_batched_array_nancumprod_axis_0() -> None:
    assert objects_are_equal(
        ba.nancumprod(ba.array([[1, np.nan, 2], [3, 4, 5]]), axis=0),
        ba.array([[1.0, 1.0, 2.0], [3.0, 4.0, 10.0]]),
    )


def test_batched_array_nancumprod_axis_1() -> None:
    assert objects_are_equal(
        ba.nancumprod(ba.array([[1, np.nan, 2], [3, 4, 5]]), axis=1),
        ba.array([[1.0, 1.0, 2.0], [3.0, 12.0, 60.0]]),
    )


def test_batched_array_nancumprod_custom_axes() -> None:
    assert objects_are_equal(
        ba.nancumprod(ba.array([[1, np.nan, 2], [3, 4, 5]], batch_axis=1), axis=0),
        ba.array([[1.0, 1.0, 2.0], [3.0, 4.0, 10.0]], batch_axis=1),
    )


def test_batched_array_nancumprod_out_1d() -> None:
    out = np.zeros(6)
    assert ba.nancumprod(ba.array([[1, np.nan, 2], [3, 4, 5]]), out=out) is out
    assert objects_are_equal(out, np.array([1.0, 1.0, 2.0, 6.0, 24.0, 120.0]))


def test_batched_array_nancumprod_out_2d() -> None:
    out = np.zeros((2, 3))
    assert ba.nancumprod(ba.array([[1, np.nan, 2], [3, 4, 5]]), axis=0, out=out) is out
    assert objects_are_equal(out, np.array([[1.0, 1.0, 2.0], [3.0, 4.0, 10.0]]))


###########################################
#    Tests for nancumprod_along_batch     #
###########################################


def test_batched_array_nancumprod_along_batch() -> None:
    assert objects_are_equal(
        ba.nancumprod_along_batch(ba.array([[1, np.nan, 2], [3, 4, 5]])),
        ba.array([[1.0, 1.0, 2.0], [3.0, 4.0, 10.0]]),
    )


def test_batched_array_nancumprod_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nancumprod_along_batch(ba.array([[1, np.nan, 2], [3, 4, 5]], batch_axis=1)),
        ba.array([[1.0, 1.0, 2.0], [3.0, 12.0, 60.0]], batch_axis=1),
    )


##############################
#    Tests for nancumsum     #
##############################


def test_batched_array_nancumsum() -> None:
    assert objects_are_equal(
        ba.nancumsum(ba.array([[1, np.nan, 2], [3, 4, 5]])),
        np.array([1.0, 1.0, 3.0, 6.0, 10.0, 15.0]),
    )


def test_batched_array_nancumsum_axis_0() -> None:
    assert objects_are_equal(
        ba.nancumsum(ba.array([[1, np.nan, 2], [3, 4, 5]]), axis=0),
        ba.array([[1.0, 0.0, 2.0], [4.0, 4.0, 7.0]]),
    )


def test_batched_array_nancumsum_axis_1() -> None:
    assert objects_are_equal(
        ba.nancumsum(ba.array([[1, np.nan, 2], [3, 4, 5]]), axis=1),
        ba.array([[1.0, 1.0, 3.0], [3.0, 7.0, 12.0]]),
    )


def test_batched_array_nancumsum_custom_axes() -> None:
    assert objects_are_equal(
        ba.nancumsum(ba.array([[1, np.nan, 2], [3, 4, 5]], batch_axis=1), axis=0),
        ba.array([[1.0, 0.0, 2.0], [4.0, 4.0, 7.0]], batch_axis=1),
    )


def test_batched_array_nancumsum_out_1d() -> None:
    out = np.zeros(6)
    assert ba.nancumsum(ba.array([[1, np.nan, 2], [3, 4, 5]]), out=out) is out
    assert objects_are_equal(out, np.array([1.0, 1.0, 3.0, 6.0, 10.0, 15.0]))


def test_batched_array_nancumsum_out_2d() -> None:
    out = np.zeros((2, 3))
    assert ba.nancumsum(ba.array([[1, np.nan, 2], [3, 4, 5]]), axis=0, out=out) is out
    assert objects_are_equal(out, np.array([[1.0, 0.0, 2.0], [4.0, 4.0, 7.0]]))


##########################################
#    Tests for nancumsum_along_batch     #
##########################################


def test_batched_array_nancumsum_along_batch() -> None:
    assert objects_are_equal(
        ba.nancumsum_along_batch(ba.array([[1, np.nan, 2], [3, 4, 5]])),
        ba.array([[1.0, 0.0, 2.0], [4.0, 4.0, 7.0]]),
    )


def test_batched_array_nancumsum_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nancumsum_along_batch(ba.array([[1, np.nan, 2], [3, 4, 5]], batch_axis=1)),
        ba.array([[1.0, 1.0, 3.0], [3.0, 7.0, 12.0]], batch_axis=1),
    )


############################
#    Tests for nanprod     #
############################


def test_batched_array_nanprod_1d() -> None:
    assert objects_are_equal(
        ba.nanprod(ba.array([1, np.nan, 2]), axis=0),
        np.float64(2.0),
    )


def test_batched_array_nanprod_2d() -> None:
    assert objects_are_equal(
        ba.nanprod(ba.array([[1, np.nan, 2], [3, 4, 5]]), axis=0),
        np.array([3.0, 4.0, 10.0]),
    )


def test_batched_array_nanprod_axis_none() -> None:
    assert objects_are_equal(
        ba.nanprod(ba.array([[1, np.nan, 2], [3, 4, 5]]), axis=None),
        np.float64(120.0),
    )


def test_batched_array_nanprod_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanprod(ba.array([[1, np.nan, 2], [3, 4, 5]], batch_axis=1), axis=1),
        np.array([2.0, 60.0]),
    )


def test_batched_array_nanprod_out() -> None:
    out = np.array(0.0)
    assert ba.nanprod(ba.array([[1, np.nan, 2], [3, 4, 5]]), out=out) is out
    assert objects_are_equal(out, np.array(120.0))


def test_batched_array_nanprod_out_axis() -> None:
    out = np.zeros(2)
    assert ba.nanprod(ba.array([[1, np.nan, 2], [3, 4, 5]]), axis=1, out=out) is out
    assert objects_are_equal(out, np.array([2.0, 60.0]))


########################################
#    Tests for nanprod_along_batch     #
########################################


def test_batched_array_nanprod_along_batch() -> None:
    assert objects_are_equal(
        ba.nanprod_along_batch(ba.array([[1, np.nan, 2], [3, 4, 5]])),
        np.array([3.0, 4.0, 10.0]),
    )


def test_batched_array_nanprod_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.nanprod_along_batch(ba.array([[1, np.nan, 2], [3, 4, 5]]), keepdims=True),
        np.array([[3.0, 4.0, 10.0]]),
    )


def test_batched_array_nanprod_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanprod_along_batch(ba.array([[1, np.nan, 2], [3, 4, 5]], batch_axis=1)),
        np.array([2.0, 60.0]),
    )


###########################
#    Tests for nansum     #
###########################


def test_batched_array_nansum_1d() -> None:
    assert objects_are_equal(
        ba.nansum(ba.array([1, np.nan, 2]), axis=0),
        np.float64(3.0),
    )


def test_batched_array_nansum_2d() -> None:
    assert objects_are_equal(
        ba.nansum(ba.array([[1, np.nan, 2], [3, 4, 5]]), axis=0),
        np.array([4.0, 4.0, 7.0]),
    )


def test_batched_array_nansum_axis_none() -> None:
    assert objects_are_equal(
        ba.nansum(ba.array([[1, np.nan, 2], [3, 4, 5]]), axis=None),
        np.float64(15.0),
    )


def test_batched_array_nansum_custom_axes() -> None:
    assert objects_are_equal(
        ba.nansum(ba.array([[1, np.nan, 2], [3, 4, 5]], batch_axis=1), axis=1),
        np.array([3.0, 12]),
    )


def test_batched_array_nansum_out() -> None:
    out = np.array(0.0)
    assert ba.nansum(ba.array([[1, np.nan, 2], [3, 4, 5]]), out=out) is out
    assert objects_are_equal(out, np.array(15.0))


def test_batched_array_nansum_out_axis() -> None:
    out = np.zeros(2)
    assert ba.nansum(ba.array([[1, np.nan, 2], [3, 4, 5]]), axis=1, out=out) is out
    assert objects_are_equal(out, np.array([3.0, 12.0]))


#######################################
#    Tests for nansum_along_batch     #
#######################################


def test_batched_array_nansum_along_batch() -> None:
    assert objects_are_equal(
        ba.nansum_along_batch(ba.array([[1, np.nan, 2], [3, 4, 5]])),
        np.array([4.0, 4.0, 7.0]),
    )


def test_batched_array_nansum_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.nansum_along_batch(ba.array([[1, np.nan, 2], [3, 4, 5]]), keepdims=True),
        np.array([[4.0, 4.0, 7.0]]),
    )


def test_batched_array_nansum_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nansum_along_batch(ba.array([[1, np.nan, 2], [3, 4, 5]], batch_axis=1)),
        np.array([3.0, 12]),
    )


#########################
#    Tests for prod     #
#########################


def test_batched_array_prod_1d() -> None:
    assert objects_are_equal(
        ba.prod(ba.array([1, 6, 2]), axis=0),
        np.int64(12),
    )


def test_batched_array_prod_2d() -> None:
    assert objects_are_equal(
        ba.prod(ba.array([[1, 6, 2], [3, 4, 5]]), axis=0),
        np.array([3, 24, 10]),
    )


def test_batched_array_prod_float() -> None:
    assert objects_are_equal(
        ba.prod(ba.array([[1.0, 6.0, 2.0], [3.0, 4.0, 5.0]]), axis=0),
        np.array([3.0, 24.0, 10.0]),
    )


def test_batched_array_prod_axis_none() -> None:
    assert objects_are_equal(
        ba.prod(ba.array([[1, 6, 2], [3, 4, 5]]), axis=None),
        np.int64(720),
    )


def test_batched_array_prod_custom_axes() -> None:
    assert objects_are_equal(
        ba.prod(ba.array([[1, 6, 2], [3, 4, 5]], batch_axis=1), axis=1),
        np.array([12, 60]),
    )


def test_batched_array_prod_out() -> None:
    out = np.array(0)
    assert ba.prod(ba.array([[1, 6, 2], [3, 4, 5]]), out=out) is out
    assert objects_are_equal(out, np.array(720))


def test_batched_array_prod_out_axis() -> None:
    out = np.zeros(2)
    assert ba.prod(ba.array([[1, 6, 2], [3, 4, 5]]), axis=1, out=out) is out
    assert objects_are_equal(out, np.array([12.0, 60.0]))


#####################################
#    Tests for prod_along_batch     #
#####################################


def test_batched_array_prod_along_batch() -> None:
    assert objects_are_equal(
        ba.prod_along_batch(ba.array([[1, 6, 2], [3, 4, 5]])),
        np.array([3, 24, 10]),
    )


def test_batched_array_prod_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.prod_along_batch(ba.array([[1, 6, 2], [3, 4, 5]]), keepdims=True),
        np.array([[3, 24, 10]]),
    )


def test_batched_array_prod_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.prod_along_batch(ba.array([[1, 6, 2], [3, 4, 5]], batch_axis=1)),
        np.array([12, 60]),
    )


########################
#    Tests for sum     #
########################


def test_batched_array_sum_1d() -> None:
    assert objects_are_equal(ba.sum(ba.array([1, 6, 2]), axis=0), np.int64(9))


def test_batched_array_sum_2d() -> None:
    assert objects_are_equal(
        ba.sum(ba.array([[1, 6, 2], [3, 4, 5]]), axis=0),
        np.array([4, 10, 7]),
    )


def test_batched_array_sum_float() -> None:
    assert objects_are_equal(
        ba.sum(ba.array([[1.0, 6.0, 2.0], [3.0, 4.0, 5.0]]), axis=0),
        np.array([4.0, 10.0, 7.0]),
    )


def test_batched_array_sum_axis_none() -> None:
    assert objects_are_equal(
        ba.sum(ba.array([[1, 6, 2], [3, 4, 5]]), axis=None),
        np.int64(21),
    )


def test_batched_array_sum_custom_axes() -> None:
    assert objects_are_equal(
        ba.sum(ba.array([[1, 6, 2], [3, 4, 5]], batch_axis=1), axis=1),
        np.array([9, 12]),
    )


def test_batched_array_sum_out() -> None:
    out = np.array(0)
    assert ba.sum(ba.array([[1, 6, 2], [3, 4, 5]]), out=out) is out
    assert objects_are_equal(out, np.array(21))


def test_batched_array_sum_out_axis() -> None:
    out = np.zeros(2)
    assert ba.sum(ba.array([[1, 6, 2], [3, 4, 5]]), axis=1, out=out) is out
    assert objects_are_equal(out, np.array([9.0, 12.0]))


####################################
#    Tests for sum_along_batch     #
####################################


def test_batched_array_sum_along_batch() -> None:
    assert objects_are_equal(
        ba.sum_along_batch(ba.array([[1, 6, 2], [3, 4, 5]])),
        np.array([4, 10, 7]),
    )


def test_batched_array_sum_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.sum_along_batch(ba.array([[1, 6, 2], [3, 4, 5]]), keepdims=True),
        np.array([[4, 10, 7]]),
    )


def test_batched_array_sum_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.sum_along_batch(ba.array([[1, 6, 2], [3, 4, 5]], batch_axis=1)),
        np.array([9, 12]),
    )


########################
#    Tests for max     #
########################


def test_batched_array_max_1d() -> None:
    assert objects_are_equal(
        ba.max(ba.array([4, 1, 2, 5, 3]), axis=0),
        np.int64(5),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_batched_array_max_2d(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.max(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype), axis=0),
        np.array([5, 9], dtype=dtype),
    )


def test_batched_array_max_axis_none() -> None:
    assert objects_are_equal(
        ba.max(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), axis=None),
        np.int64(9),
    )


def test_batched_array_max_out_axis_none() -> None:
    out = np.array(0, dtype=int)
    assert ba.max(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), out=out) is out
    assert objects_are_equal(out, np.array(9, dtype=int))


def test_batched_array_max_out_axis_0() -> None:
    out = np.array([0, 0], dtype=int)
    assert ba.max(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), axis=0, out=out) is out
    assert objects_are_equal(out, np.array([5, 9], dtype=int))


def test_batched_array_max_custom_axes() -> None:
    assert objects_are_equal(
        ba.max(ba.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]], batch_axis=1), axis=1),
        np.array([5, 9]),
    )


####################################
#    Tests for max_along_batch     #
####################################


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_batched_array_max_along_batch(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.max_along_batch(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype)),
        np.array([5, 9], dtype=dtype),
    )


def test_batched_array_max_along_batch_out() -> None:
    out = np.array([0, 0], dtype=int)
    assert ba.max_along_batch(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), out=out) is out
    assert objects_are_equal(out, np.array([5, 9], dtype=int))


def test_batched_array_max_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.max_along_batch(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), keepdims=True),
        np.array([[5, 9]]),
    )


def test_batched_array_max_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.max_along_batch(ba.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]], batch_axis=1)),
        np.array([5, 9]),
    )


########################
#    Tests for min     #
########################


def test_batched_array_min_1d() -> None:
    assert objects_are_equal(
        ba.min(ba.array([4, 1, 2, 5, 3]), axis=0),
        np.int64(1),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_batched_array_min_2d(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.min(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype), axis=0),
        np.array([1, 5], dtype=dtype),
    )


def test_batched_array_min_axis_none() -> None:
    assert objects_are_equal(
        ba.min(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), axis=None),
        np.int64(1),
    )


def test_batched_array_min_out_axis_none() -> None:
    out = np.array(0, dtype=int)
    assert ba.min(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), out=out) is out
    assert objects_are_equal(out, np.array(1, dtype=int))


def test_batched_array_min_out_axis_0() -> None:
    out = np.array([0, 0], dtype=int)
    assert ba.min(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), axis=0, out=out) is out
    assert objects_are_equal(out, np.array([1, 5], dtype=int))


def test_batched_array_min_custom_axes() -> None:
    assert objects_are_equal(
        ba.min(ba.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]], batch_axis=1), axis=1),
        np.array([1, 5]),
    )


####################################
#    Tests for min_along_batch     #
####################################


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_batched_array_min_along_batch(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.min_along_batch(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype)),
        np.array([1, 5], dtype=dtype),
    )


def test_batched_array_min_along_batch_out() -> None:
    out = np.array([0, 0], dtype=int)
    assert ba.min_along_batch(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), out=out) is out
    assert objects_are_equal(out, np.array([1, 5], dtype=int))


def test_batched_array_min_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.min_along_batch(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), keepdims=True),
        np.array([[1, 5]]),
    )


def test_batched_array_min_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.min_along_batch(ba.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]], batch_axis=1)),
        np.array([1, 5]),
    )
