from __future__ import annotations

import numpy as np
from coola import objects_are_equal

from redcat import ba2
from redcat.ba2 import BatchedArray

############################
#    Tests for cumprod     #
############################


def test_batched_array_cumprod() -> None:
    assert objects_are_equal(
        ba2.cumprod(BatchedArray(np.arange(10).reshape(2, 5) + 1)),
        np.array([1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]),
    )


def test_batched_array_cumprod_axis_0() -> None:
    assert objects_are_equal(
        ba2.cumprod(BatchedArray(np.arange(10).reshape(2, 5)), axis=0),
        BatchedArray(np.asarray([[0, 1, 2, 3, 4], [0, 6, 14, 24, 36]])),
    )


def test_batched_array_cumprod_axis_1() -> None:
    assert objects_are_equal(
        ba2.cumprod(BatchedArray(np.arange(10).reshape(2, 5)), axis=1),
        BatchedArray(np.array([[0, 0, 0, 0, 0], [5, 30, 210, 1680, 15120]])),
    )


def test_batched_array_cumprod_custom_axes() -> None:
    assert objects_are_equal(
        ba2.cumprod(BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1), axis=0),
        BatchedArray(np.array([[0, 1], [0, 3], [0, 15], [0, 105], [0, 945]]), batch_axis=1),
    )


def test_batched_array_cumprod_out() -> None:
    out = np.zeros((5, 2), dtype=np.int64)
    assert ba2.cumprod(BatchedArray(np.arange(10).reshape(5, 2)), axis=0, out=out) is out
    assert objects_are_equal(out, np.asarray([[0, 1], [0, 3], [0, 15], [0, 105], [0, 945]]))


def test_batched_array_cumprod_out_array() -> None:
    out = np.zeros(10)
    assert ba2.cumprod(BatchedArray(np.arange(10).reshape(2, 5) + 1), out=out) is out
    assert objects_are_equal(
        out, np.asarray([1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0])
    )


########################################
#    Tests for cumprod_along_batch     #
########################################


def test_batched_array_cumprod_along_batch() -> None:
    assert objects_are_equal(
        ba2.cumprod_along_batch(BatchedArray(np.arange(10).reshape(2, 5))),
        BatchedArray(np.array([[0, 1, 2, 3, 4], [0, 6, 14, 24, 36]])),
    )


def test_batched_array_cumprod_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba2.cumprod_along_batch(BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1)),
        BatchedArray(np.array([[0, 0], [2, 6], [4, 20], [6, 42], [8, 72]]), batch_axis=1),
    )


###########################
#    Tests for cumsum     #
###########################


def test_batched_array_cumsum() -> None:
    assert objects_are_equal(
        ba2.cumsum(BatchedArray(np.arange(10).reshape(2, 5) + 1)),
        np.array([1, 3, 6, 10, 15, 21, 28, 36, 45, 55]),
    )


def test_batched_array_cumsum_axis_0() -> None:
    assert objects_are_equal(
        ba2.cumsum(BatchedArray(np.arange(10).reshape(2, 5)), axis=0),
        BatchedArray(np.asarray([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])),
    )


def test_batched_array_cumsum_axis_1() -> None:
    assert objects_are_equal(
        ba2.cumsum(BatchedArray(np.arange(10).reshape(2, 5)), axis=1),
        BatchedArray(np.array([[0, 1, 3, 6, 10], [5, 11, 18, 26, 35]])),
    )


def test_batched_array_cumsum_custom_axes() -> None:
    assert objects_are_equal(
        ba2.cumsum(BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1), axis=0),
        BatchedArray(np.array([[0, 1], [2, 4], [6, 9], [12, 16], [20, 25]]), batch_axis=1),
    )


def test_batched_array_cumsum_out() -> None:
    out = np.zeros((5, 2), dtype=np.int64)
    assert ba2.cumsum(BatchedArray(np.arange(10).reshape(5, 2)), axis=0, out=out) is out
    assert objects_are_equal(out, np.asarray([[0, 1], [2, 4], [6, 9], [12, 16], [20, 25]]))


def test_batched_array_cumsum_out_array() -> None:
    out = np.zeros(10)
    assert ba2.cumsum(BatchedArray(np.arange(10).reshape(2, 5) + 1), out=out) is out
    assert objects_are_equal(
        out, np.asarray([1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0, 55.0])
    )


#######################################
#    Tests for cumsum_along_batch     #
#######################################


def test_batched_array_cumsum_along_batch() -> None:
    assert objects_are_equal(
        ba2.cumsum_along_batch(BatchedArray(np.arange(10).reshape(2, 5))),
        BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])),
    )


def test_batched_array_cumsum_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba2.cumsum_along_batch(BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1)),
        BatchedArray(np.array([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), batch_axis=1),
    )


###############################
#    Tests for nancumprod     #
###############################


def test_batched_array_nancumprod() -> None:
    assert objects_are_equal(
        ba2.nancumprod(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))),
        np.array([1.0, 1.0, 2.0, 6.0, 24.0, 120.0]),
    )


def test_batched_array_nancumprod_axis_0() -> None:
    assert objects_are_equal(
        ba2.nancumprod(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=0),
        BatchedArray(np.asarray([[1.0, 1.0, 2.0], [3.0, 4.0, 10.0]])),
    )


def test_batched_array_nancumprod_axis_1() -> None:
    assert objects_are_equal(
        ba2.nancumprod(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=1),
        BatchedArray(np.array([[1.0, 1.0, 2.0], [3.0, 12.0, 60.0]])),
    )


def test_batched_array_nancumprod_custom_axes() -> None:
    assert objects_are_equal(
        ba2.nancumprod(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1), axis=0),
        BatchedArray(np.array([[1.0, 1.0, 2.0], [3.0, 4.0, 10.0]]), batch_axis=1),
    )


def test_batched_array_nancumprod_out_1d() -> None:
    out = np.zeros(6)
    assert ba2.nancumprod(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), out=out) is out
    assert objects_are_equal(out, np.asarray([1.0, 1.0, 2.0, 6.0, 24.0, 120.0]))


def test_batched_array_nancumprod_out_2d() -> None:
    out = np.zeros((2, 3))
    assert (
        ba2.nancumprod(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]])), axis=0, out=out) is out
    )
    assert objects_are_equal(out, np.asarray([[1.0, 1.0, 2.0], [3.0, 4.0, 10.0]]))


###########################################
#    Tests for nancumprod_along_batch     #
###########################################


def test_batched_array_nancumprod_along_batch() -> None:
    assert objects_are_equal(
        ba2.nancumprod_along_batch(BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))),
        BatchedArray(np.array([[1.0, 1.0, 2.0], [3.0, 4.0, 10.0]])),
    )


def test_batched_array_nancumprod_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba2.nancumprod_along_batch(
            BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
        ),
        BatchedArray(np.array([[1.0, 1.0, 2.0], [3.0, 12.0, 60.0]]), batch_axis=1),
    )
