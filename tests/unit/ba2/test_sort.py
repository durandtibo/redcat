from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from redcat import ba2 as ba
from redcat.ba2.core import SortKind
from tests.unit.ba2.test_core import NUMERIC_DTYPES, SORT_KINDS

###################
#     argsort     #
###################


def test_argsort() -> None:
    assert objects_are_equal(
        ba.argsort(ba.BatchedArray(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]))),
        ba.BatchedArray(np.asarray([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]])),
    )


@pytest.mark.parametrize("kind", SORT_KINDS)
def test_argsort_kind(kind: SortKind) -> None:
    assert objects_are_equal(
        ba.argsort(ba.BatchedArray(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])), kind=kind),
        ba.BatchedArray(np.asarray([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]])),
    )


def test_argsort_axis_0() -> None:
    assert objects_are_equal(
        ba.argsort(ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=0),
        ba.BatchedArray(np.asarray([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
    )


def test_argsort_axis_1() -> None:
    assert objects_are_equal(
        ba.argsort(
            ba.BatchedArray(
                np.asarray(
                    [
                        [[0, 1], [-2, 3], [-4, 5], [-6, 7], [-8, 9]],
                        [[10, -11], [12, -13], [14, -15], [16, -17], [18, -19]],
                    ]
                )
            ),
            axis=1,
        ),
        ba.BatchedArray(
            np.asarray(
                [[[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]], [[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]]]
            )
        ),
    )


def test_argsort_custom_axes() -> None:
    assert objects_are_equal(
        ba.argsort(
            ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), batch_axis=1),
            axis=0,
        ),
        ba.BatchedArray(np.asarray([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]), batch_axis=1),
    )


###############################
#     argsort_along_batch     #
###############################


def test_argsort_along_batch() -> None:
    assert objects_are_equal(
        ba.argsort_along_batch(
            ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]))
        ),
        ba.BatchedArray(np.asarray([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
    )


@pytest.mark.parametrize("kind", SORT_KINDS)
def test_argsort_along_batch_kind(kind: SortKind) -> None:
    assert objects_are_equal(
        ba.argsort_along_batch(
            ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), kind=kind
        ),
        ba.BatchedArray(np.asarray([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
    )


def test_argsort_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.argsort_along_batch(
            ba.BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)
        ),
        ba.BatchedArray(np.asarray([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]), batch_axis=1),
    )


################
#     sort     #
################


def test_sort() -> None:
    batch = ba.BatchedArray(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]))
    assert objects_are_equal(
        ba.sort(batch), ba.BatchedArray(np.asarray([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]))
    )


@pytest.mark.parametrize("kind", SORT_KINDS)
def test_sort_kind(kind: SortKind) -> None:
    batch = ba.BatchedArray(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]))
    assert objects_are_equal(
        ba.sort(batch, kind=kind), ba.BatchedArray(np.asarray([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]))
    )


def test_sort_axis_0() -> None:
    batch = ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]))
    assert objects_are_equal(
        ba.sort(batch, axis=0),
        ba.BatchedArray(np.asarray([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])),
    )


def test_sort_axis_1() -> None:
    batch = ba.BatchedArray(
        np.asarray(
            [
                [[0, 1], [-2, 3], [-4, 5], [-6, 7], [-8, 9]],
                [[10, -11], [12, -13], [14, -15], [16, -17], [18, -19]],
            ]
        )
    )
    assert objects_are_equal(
        ba.sort(batch, axis=1),
        ba.BatchedArray(
            np.asarray(
                [
                    [[-8, 1], [-6, 3], [-4, 5], [-2, 7], [0, 9]],
                    [[10, -19], [12, -17], [14, -15], [16, -13], [18, -11]],
                ]
            )
        ),
    )


def test_sort_custom_axes() -> None:
    batch = ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), batch_axis=1)
    assert objects_are_equal(
        ba.sort(batch, axis=0),
        ba.BatchedArray(np.asarray([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]), batch_axis=1),
    )


############################
#     sort_along_batch     #
############################


def test_sort_along_batch() -> None:
    batch = ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]))
    assert objects_are_equal(
        ba.sort_along_batch(batch),
        ba.BatchedArray(np.asarray([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])),
    )


@pytest.mark.parametrize("kind", SORT_KINDS)
def test_sort_along_batch_kind(kind: SortKind) -> None:
    batch = ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]))
    assert objects_are_equal(
        ba.sort_along_batch(batch, kind=kind),
        ba.BatchedArray(np.asarray([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])),
    )


def test_sort_along_batch_custom_axes() -> None:
    batch = ba.BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)

    assert objects_are_equal(
        ba.sort_along_batch(batch),
        ba.BatchedArray(np.asarray([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]), batch_axis=1),
    )


##################
#     argmax     #
##################


def test_argmax_1d() -> None:
    assert objects_are_equal(
        ba.argmax(ba.BatchedArray(np.asarray([4, 1, 2, 5, 3])), axis=0),
        np.int64(3),
    )


def test_argmax_2d() -> None:
    assert objects_are_equal(
        ba.argmax(ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=0),
        np.asarray([3, 0]),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_argmax_dtype(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.argmax(
            ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype)),
            axis=0,
        ),
        np.asarray([3, 0]),
    )


def test_argmax_axis_none() -> None:
    assert objects_are_equal(
        ba.argmax(ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=None),
        np.int64(1),
    )


def test_argmax_custom_axes() -> None:
    assert objects_are_equal(
        ba.argmax(
            ba.BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1), axis=1
        ),
        np.asarray([3, 0]),
    )


def test_argmax_out_axis_none() -> None:
    out = np.array(0, dtype=int)
    assert (
        ba.argmax(
            ba.BatchedArray(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=None, out=out
        )
        is out
    )
    assert objects_are_equal(out, np.array(1, dtype=int))


def test_argmax_out_axis_0() -> None:
    out = np.array([0, 0], dtype=int)
    assert (
        ba.argmax(
            ba.BatchedArray(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=0, out=out
        )
        is out
    )
    assert objects_are_equal(out, np.array([3, 0], dtype=int))


##############################
#     argmax_along_batch     #
##############################


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_argmax_along_batch(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.argmax_along_batch(
            ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype))
        ),
        np.asarray([3, 0]),
    )


def test_argmax_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.argmax_along_batch(
            ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), keepdims=True
        ),
        np.asarray([[3, 0]]),
    )


def test_argmax_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.argmax_along_batch(
            ba.BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)
        ),
        np.asarray([3, 0]),
    )


def test_argmax_along_batch_out() -> None:
    out = np.array([0, 0], dtype=int)
    assert (
        ba.argmax_along_batch(
            ba.BatchedArray(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), out=out
        )
        is out
    )
    assert objects_are_equal(out, np.array([3, 0], dtype=int))


##################
#     argmin     #
##################


def test_argmin_1d() -> None:
    assert objects_are_equal(
        ba.argmin(ba.BatchedArray(np.asarray([4, 1, 2, 5, 3])), axis=0),
        np.int64(1),
    )


def test_argmin_2d() -> None:
    assert objects_are_equal(
        ba.argmin(ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=0),
        np.asarray([1, 2]),
    )


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_argmin_dtype(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.argmin(
            ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype)),
            axis=0,
        ),
        np.asarray([1, 2]),
    )


def test_argmin_axis_none() -> None:
    assert objects_are_equal(
        ba.argmin(ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=None),
        np.int64(2),
    )


def test_argmin_custom_axes() -> None:
    assert objects_are_equal(
        ba.argmin(
            ba.BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1), axis=1
        ),
        np.asarray([1, 2]),
    )


def test_argmin_out_axis_none() -> None:
    out = np.array(0, dtype=int)
    assert (
        ba.argmin(
            ba.BatchedArray(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=None, out=out
        )
        is out
    )
    assert objects_are_equal(out, np.array(2, dtype=int))


def test_argmin_out_axis_0() -> None:
    out = np.array([0, 0], dtype=int)
    assert (
        ba.argmin(
            ba.BatchedArray(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), axis=0, out=out
        )
        is out
    )
    assert objects_are_equal(out, np.array([1, 2], dtype=int))


##############################
#     argmin_along_batch     #
##############################


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_argmin_along_batch(dtype: np.dtype) -> None:
    assert objects_are_equal(
        ba.argmin_along_batch(
            ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]], dtype=dtype))
        ),
        np.asarray([1, 2]),
    )


def test_argmin_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.argmin_along_batch(
            ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), keepdims=True
        ),
        np.asarray([[1, 2]]),
    )


def test_argmin_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.argmin_along_batch(
            ba.BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)
        ),
        np.asarray([1, 2]),
    )


def test_argmin_along_batch_out() -> None:
    out = np.array([0, 0], dtype=int)
    assert (
        ba.argmin_along_batch(
            ba.BatchedArray(np.array([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])), out=out
        )
        is out
    )
    assert objects_are_equal(out, np.array([1, 2], dtype=int))


#####################
#     nanargmax     #
#####################


def test_nanargmax_1d() -> None:
    assert objects_are_equal(
        ba.nanargmax(ba.BatchedArray(np.asarray([4, 1, np.nan, 5, 3])), axis=0),
        np.int64(3),
    )


def test_nanargmax_2d() -> None:
    assert objects_are_equal(
        ba.nanargmax(
            ba.BatchedArray(np.asarray([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]])), axis=0
        ),
        np.asarray([3, 0]),
    )


def test_nanargmax_dtype() -> None:
    assert objects_are_equal(
        ba.nanargmax(
            ba.BatchedArray(np.asarray([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]])), axis=0
        ),
        np.asarray([3, 0]),
    )


def test_nanargmax_axis_none() -> None:
    assert objects_are_equal(
        ba.nanargmax(
            ba.BatchedArray(np.asarray([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]])),
            axis=None,
        ),
        np.int64(1),
    )


def test_nanargmax_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanargmax(
            ba.BatchedArray(np.asarray([[4, 1, np.nan, 5, 3], [9, 7, 5, 6, np.nan]]), batch_axis=1),
            axis=1,
        ),
        np.asarray([3, 0]),
    )


def test_nanargmax_out_axis_none() -> None:
    out = np.array(0, dtype=int)
    assert (
        ba.nanargmax(
            ba.BatchedArray(np.array([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]])),
            axis=None,
            out=out,
        )
        is out
    )
    assert objects_are_equal(out, np.array(1, dtype=int))


def test_nanargmax_out_axis_0() -> None:
    out = np.array([0, 0], dtype=int)
    assert (
        ba.nanargmax(
            ba.BatchedArray(np.array([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]])),
            axis=0,
            out=out,
        )
        is out
    )
    assert objects_are_equal(out, np.array([3, 0], dtype=int))


#################################
#     nanargmax_along_batch     #
#################################


def test_nanargmax_along_batch() -> None:
    assert objects_are_equal(
        ba.nanargmax_along_batch(
            ba.BatchedArray(np.asarray([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]]))
        ),
        np.asarray([3, 0]),
    )


def test_nanargmax_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.nanargmax_along_batch(
            ba.BatchedArray(np.asarray([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]])),
            keepdims=True,
        ),
        np.asarray([[3, 0]]),
    )


def test_nanargmax_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanargmax_along_batch(
            ba.BatchedArray(np.asarray([[4, 1, np.nan, 5, 3], [9, 7, 5, 6, np.nan]]), batch_axis=1)
        ),
        np.asarray([3, 0]),
    )


def test_nanargmax_along_batch_out() -> None:
    out = np.array([0, 0], dtype=int)
    assert (
        ba.nanargmax_along_batch(
            ba.BatchedArray(np.array([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]])), out=out
        )
        is out
    )
    assert objects_are_equal(out, np.array([3, 0], dtype=int))


#####################
#     nanargmin     #
#####################


def test_nanargmin_1d() -> None:
    assert objects_are_equal(
        ba.nanargmin(ba.BatchedArray(np.asarray([4, 1, np.nan, 5, 3])), axis=0),
        np.int64(1),
    )


def test_nanargmin_2d() -> None:
    assert objects_are_equal(
        ba.nanargmin(
            ba.BatchedArray(np.asarray([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]])), axis=0
        ),
        np.asarray([1, 2]),
    )


def test_nanargmin_dtype() -> None:
    assert objects_are_equal(
        ba.nanargmin(
            ba.BatchedArray(np.asarray([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]])), axis=0
        ),
        np.asarray([1, 2]),
    )


def test_nanargmin_axis_none() -> None:
    assert objects_are_equal(
        ba.nanargmin(
            ba.BatchedArray(np.asarray([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]])),
            axis=None,
        ),
        np.int64(2),
    )


def test_nanargmin_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanargmin(
            ba.BatchedArray(np.asarray([[4, 1, np.nan, 5, 3], [9, 7, 5, 6, np.nan]]), batch_axis=1),
            axis=1,
        ),
        np.asarray([1, 2]),
    )


def test_nanargmin_out_axis_none() -> None:
    out = np.array(0, dtype=int)
    assert (
        ba.nanargmin(
            ba.BatchedArray(np.array([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]])),
            axis=None,
            out=out,
        )
        is out
    )
    assert objects_are_equal(out, np.array(2, dtype=int))


def test_nanargmin_out_axis_0() -> None:
    out = np.array([0, 0], dtype=int)
    assert (
        ba.nanargmin(
            ba.BatchedArray(np.array([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]])),
            axis=0,
            out=out,
        )
        is out
    )
    assert objects_are_equal(out, np.array([1, 2], dtype=int))


#################################
#     nanargmin_along_batch     #
#################################


def test_nanargmin_along_batch() -> None:
    assert objects_are_equal(
        ba.nanargmin_along_batch(
            ba.BatchedArray(np.asarray([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]]))
        ),
        np.asarray([1, 2]),
    )


def test_nanargmin_along_batch_keepdims() -> None:
    assert objects_are_equal(
        ba.nanargmin_along_batch(
            ba.BatchedArray(np.asarray([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]])),
            keepdims=True,
        ),
        np.asarray([[1, 2]]),
    )


def test_nanargmin_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.nanargmin_along_batch(
            ba.BatchedArray(np.asarray([[4, 1, np.nan, 5, 3], [9, 7, 5, 6, np.nan]]), batch_axis=1)
        ),
        np.asarray([1, 2]),
    )


def test_nanargmin_along_batch_out() -> None:
    out = np.array([0, 0], dtype=int)
    assert (
        ba.nanargmin_along_batch(
            ba.BatchedArray(np.array([[4, 9], [1, np.nan], [2, 5], [5, 6], [np.nan, 8]])), out=out
        )
        is out
    )
    assert objects_are_equal(out, np.array([1, 2], dtype=int))
