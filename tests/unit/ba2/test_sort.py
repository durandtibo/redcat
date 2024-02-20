from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from redcat import ba2 as ba
from tests.unit.ba2.test_core import SORT_KINDS

################
#     Sort     #
################


def test_batched_array_sort() -> None:
    batch = ba.BatchedArray(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]))
    assert objects_are_equal(
        ba.sort(batch), ba.BatchedArray(np.asarray([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]))
    )


@pytest.mark.parametrize("kind", SORT_KINDS)
def test_batched_array_sort_kind(kind: str) -> None:
    batch = ba.BatchedArray(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]))
    assert objects_are_equal(
        ba.sort(batch, kind=kind), ba.BatchedArray(np.asarray([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]))
    )


def test_batched_array_sort_axis_0() -> None:
    batch = ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]))
    assert objects_are_equal(
        ba.sort(batch, axis=0),
        ba.BatchedArray(np.asarray([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])),
    )


def test_batched_array_sort_axis_1() -> None:
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


def test_batched_array_sort_custom_axes() -> None:
    batch = ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), batch_axis=1)
    assert objects_are_equal(
        ba.sort(batch, axis=0),
        ba.BatchedArray(np.asarray([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]), batch_axis=1),
    )


def test_batched_array_sort_along_batch() -> None:
    batch = ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]))
    assert objects_are_equal(
        ba.sort_along_batch(batch),
        ba.BatchedArray(np.asarray([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])),
    )


@pytest.mark.parametrize("kind", SORT_KINDS)
def test_batched_array_sort_along_batch_kind(kind: str) -> None:
    batch = ba.BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]))
    assert objects_are_equal(
        ba.sort_along_batch(batch, kind=kind),
        ba.BatchedArray(np.asarray([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])),
    )


def test_batched_array_sort_along_batch_custom_axes() -> None:
    batch = ba.BatchedArray(np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1)

    assert objects_are_equal(
        ba.sort_along_batch(batch),
        ba.BatchedArray(np.asarray([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]), batch_axis=1),
    )
