from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from coola import objects_are_equal

from redcat import ba2
from redcat.ba2 import BatchedArray

################################
#    Tests for concatenate     #
################################


@pytest.mark.parametrize(
    "arrays",
    [
        [
            ba2.batched_array([[0, 1, 2], [4, 5, 6]]),
            ba2.batched_array([[10, 11, 12], [13, 14, 15]]),
        ],
        [
            ba2.batched_array([[0, 1, 2], [4, 5, 6]]),
            ba2.batched_array([[10, 11, 12]]),
            ba2.batched_array([[13, 14, 15]]),
        ],
    ],
)
def test_concatenate_axis_0(arrays: Sequence[BatchedArray]) -> None:
    assert objects_are_equal(
        np.concatenate(arrays, axis=0),
        ba2.batched_array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]),
    )


@pytest.mark.parametrize(
    "arrays",
    [
        [
            ba2.batched_array([[0, 1, 2], [10, 11, 12]]),
            ba2.batched_array([[4, 5], [14, 15]]),
        ],
        [
            ba2.batched_array([[0, 1, 2], [10, 11, 12]]),
            ba2.batched_array([[4], [14]]),
            ba2.batched_array([[5], [15]]),
        ],
    ],
)
def test_concatenate_axis_1(arrays: Sequence[BatchedArray]) -> None:
    assert objects_are_equal(
        np.concatenate(arrays, axis=1),
        ba2.batched_array([[0, 1, 2, 4, 5], [10, 11, 12, 14, 15]]),
    )


def test_concatenate_array() -> None:
    assert objects_are_equal(
        np.concatenate(
            [np.array([[0, 1, 2], [10, 11, 12]]), np.array([[4, 5], [14, 15]])],
            axis=1,
        ),
        np.array([[0, 1, 2, 4, 5], [10, 11, 12, 14, 15]]),
    )


def test_concatenate_mix_array() -> None:
    with pytest.raises(
        TypeError, match="no implementation found for 'numpy.concatenate' on types that implement"
    ):
        np.concatenate(
            [ba2.batched_array([[0, 1, 2], [10, 11, 12]]), np.array([[4, 5], [14, 15]])],
            axis=1,
        )


def test_concatenate_axis_none() -> None:
    assert objects_are_equal(
        np.concatenate(
            [
                ba2.batched_array([[0, 1, 2], [4, 5, 6]]),
                ba2.batched_array([[10, 11, 12], [13, 14, 15]]),
            ],
            axis=None,
        ),
        np.array([0, 1, 2, 4, 5, 6, 10, 11, 12, 13, 14, 15]),
    )


def test_concatenate_custom_axes() -> None:
    assert objects_are_equal(
        np.concatenate([ba2.ones((2, 3), batch_axis=1), ba2.ones((2, 3), batch_axis=1)]),
        ba2.ones((4, 3), batch_axis=1),
    )


def test_concatenate_incorrect_batch_axis() -> None:
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        np.concatenate([ba2.ones((2, 2)), ba2.zeros((2, 2), batch_axis=1)])


#############################################
#     Tests for concatenate_along_batch     #
#############################################


@pytest.mark.parametrize(
    "arrays",
    [
        [
            ba2.batched_array([[0, 1, 2], [4, 5, 6]]),
            ba2.batched_array([[10, 11, 12], [13, 14, 15]]),
        ],
        [
            ba2.batched_array([[0, 1, 2], [4, 5, 6]]),
            ba2.batched_array([[10, 11, 12]]),
            ba2.batched_array([[13, 14, 15]]),
        ],
    ],
)
def test_concatenate_along_batch_axis_0(arrays: Sequence[BatchedArray]) -> None:
    assert objects_are_equal(
        ba2.concatenate_along_batch(arrays),
        ba2.batched_array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]),
    )


@pytest.mark.parametrize(
    "arrays",
    [
        [
            ba2.batched_array([[0, 1, 2], [10, 11, 12]], batch_axis=1),
            ba2.batched_array([[4, 5], [14, 15]], batch_axis=1),
        ],
        [
            ba2.batched_array([[0, 1, 2], [10, 11, 12]], batch_axis=1),
            ba2.batched_array([[4], [14]], batch_axis=1),
            ba2.batched_array([[5], [15]], batch_axis=1),
        ],
    ],
)
def test_concatenate_along_batch_axis_1(arrays: Sequence[BatchedArray]) -> None:
    assert objects_are_equal(
        ba2.concatenate_along_batch(arrays),
        ba2.batched_array([[0, 1, 2, 4, 5], [10, 11, 12, 14, 15]], batch_axis=1),
    )


def test_concatenate_along_batch_incorrect_batch_axis() -> None:
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        ba2.concatenate_along_batch([ba2.ones((2, 2)), ba2.zeros((2, 2), batch_axis=1)])
