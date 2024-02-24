from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from redcat import ba2 as ba
from tests.unit.ba2.test_core import NUMERIC_DTYPES

#########################
#    Tests for mean     #
#########################


def test_batched_array_mean_1d() -> None:
    assert objects_are_equal(
        ba.mean(ba.array([4, 1, 2, 5, 3]), axis=0),
        np.float64(3.0),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_batched_array_mean_2d(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.mean(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype), axis=0),
        np.array([3.0, 7.0]),
    )


def test_batched_array_mean_axis_none() -> None:
    assert objects_are_equal(
        ba.mean(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), axis=None),
        np.float64(5.0),
    )


def test_batched_array_mean_out_axis_none() -> None:
    out = np.array(0.0)
    assert ba.mean(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), out=out) is out
    assert objects_are_equal(out, np.array(5.0))


def test_batched_array_mean_out_axis_0() -> None:
    out = np.array([0.0, 0.0])
    assert ba.mean(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), axis=0, out=out) is out
    assert objects_are_equal(out, np.array([3.0, 7.0]))


def test_batched_array_mean_custom_axes() -> None:
    assert objects_are_equal(
        ba.mean(ba.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]], batch_axis=1), axis=1),
        np.array([3.0, 7.0]),
    )


#####################################
#    Tests for mean_along_batch     #
#####################################


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_batched_array_mean_along_batch(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.mean_along_batch(
            ba.BatchedArray(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype))
        ),
        np.array([3.0, 7.0]),
    )


def test_batched_array_mean_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.mean_along_batch(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), keepdims=True),
        np.array([[3.0, 7.0]]),
    )


def test_batched_array_mean_along_batch_out() -> None:
    out = np.array([0.0, 0.0])
    assert ba.mean_along_batch(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), out=out) is out
    assert objects_are_equal(out, np.array([3.0, 7.0]))


def test_batched_array_mean_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.mean_along_batch(
            ba.BatchedArray(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)
        ),
        np.array([3.0, 7.0]),
    )


###########################
#    Tests for median     #
###########################


def test_batched_array_median_1d() -> None:
    assert objects_are_equal(
        ba.median(ba.array([4, 1, 2, 5, 3]), axis=0),
        np.float64(3.0),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_batched_array_median_2d(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.median(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype), axis=0),
        np.array([3.0, 7.0]),
    )


def test_batched_array_median_axis_none() -> None:
    assert objects_are_equal(
        ba.median(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), axis=None),
        np.float64(5.0),
    )


def test_batched_array_median_out_axis_none() -> None:
    out = np.array(0.0)
    assert ba.median(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), out=out) is out
    assert objects_are_equal(out, np.array(5.0))


def test_batched_array_median_out_axis_0() -> None:
    out = np.array([0.0, 0.0])
    assert ba.median(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), axis=0, out=out) is out
    assert objects_are_equal(out, np.array([3.0, 7.0]))


def test_batched_array_median_custom_axes() -> None:
    assert objects_are_equal(
        ba.median(ba.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]], batch_axis=1), axis=1),
        np.array([3.0, 7.0]),
    )


#######################################
#    Tests for median_along_batch     #
#######################################


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_batched_array_median_along_batch(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.median_along_batch(
            ba.BatchedArray(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype))
        ),
        np.array([3.0, 7.0]),
    )


def test_batched_array_median_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.median_along_batch(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), keepdims=True),
        np.array([[3.0, 7.0]]),
    )


def test_batched_array_median_along_batch_out() -> None:
    out = np.array([0.0, 0.0])
    assert ba.median_along_batch(ba.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), out=out) is out
    assert objects_are_equal(out, np.array([3.0, 7.0]))


def test_batched_array_median_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.median_along_batch(
            ba.BatchedArray(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)
        ),
        np.array([3.0, 7.0]),
    )


############################
#    Tests for nanmean     #
############################


def test_batched_array_nanmean_1d() -> None:
    assert objects_are_equal(
        ba.nanmean(ba.array([4, 1, 2, 5, np.nan]), axis=0),
        np.float64(3.0),
    )


def test_batched_array_nanmean_2d() -> None:
    assert objects_are_equal(
        ba.nanmean(ba.array([[1, np.nan, 2], [3, 4, 5]]), axis=0),
        np.array([2.0, 4.0, 3.5]),
    )


def test_batched_array_nanmean_axis_none() -> None:
    assert objects_are_equal(
        ba.nanmean(ba.array([[1, np.nan, 2], [3, 4, 5]]), axis=None),
        np.float64(3.0),
    )


def test_batched_array_nanmean_out_axis_none() -> None:
    out = np.array(0.0)
    assert ba.nanmean(ba.array([[1, np.nan, 2], [3, 4, 5]]), out=out) is out
    assert objects_are_equal(out, np.array(3.0))


def test_batched_array_nanmean_out_axis_0() -> None:
    out = np.array([0.0, 0.0, 0.0])
    assert ba.nanmean(ba.array([[1, np.nan, 2], [3, 4, 5]]), axis=0, out=out) is out
    assert objects_are_equal(out, np.array([2.0, 4.0, 3.5]))


def test_batched_array_nanmean_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanmean(ba.array([[1, np.nan, 2], [3, 4, 5]], batch_axis=1), axis=1),
        np.array([1.5, 4.0]),
    )


########################################
#    Tests for nanmean_along_batch     #
########################################


def test_batched_array_nanmean_along_batch() -> None:
    assert objects_are_equal(
        ba.nanmean_along_batch(ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]))),
        np.array([2.0, 4.0, 3.5]),
    )


def test_batched_array_nanmean_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.nanmean_along_batch(ba.array([[1, np.nan, 2], [3, 4, 5]]), keepdims=True),
        np.array([[2.0, 4.0, 3.5]]),
    )


def test_batched_array_nanmean_along_batch_out() -> None:
    out = np.array([0.0, 0.0, 0.0])
    assert ba.nanmean_along_batch(ba.array([[1, np.nan, 2], [3, 4, 5]]), out=out) is out
    assert objects_are_equal(out, np.array([2.0, 4.0, 3.5]))


def test_batched_array_nanmean_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanmean_along_batch(
            ba.BatchedArray(np.array([[1, np.nan, 2], [3, 4, 5]]), batch_axis=1)
        ),
        np.array([1.5, 4.0]),
    )
