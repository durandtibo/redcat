from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola import objects_are_equal

from redcat import ba

if TYPE_CHECKING:
    from collections.abc import Sequence

################################
#    Tests for concatenate     #
################################


@pytest.mark.parametrize(
    "arrays",
    [
        [
            ba.array([[0, 1, 2], [4, 5, 6]]),
            ba.array([[10, 11, 12], [13, 14, 15]]),
        ],
        [
            ba.array([[0, 1, 2], [4, 5, 6]]),
            ba.array([[10, 11, 12]]),
            ba.array([[13, 14, 15]]),
        ],
    ],
)
def test_concatenate_axis_0(arrays: Sequence[ba.BatchedArray]) -> None:
    assert objects_are_equal(
        np.concatenate(arrays, axis=0),
        ba.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]),
    )


@pytest.mark.parametrize(
    "arrays",
    [
        [
            ba.array([[0, 1, 2], [10, 11, 12]]),
            ba.array([[4, 5], [14, 15]]),
        ],
        [
            ba.array([[0, 1, 2], [10, 11, 12]]),
            ba.array([[4], [14]]),
            ba.array([[5], [15]]),
        ],
    ],
)
def test_concatenate_axis_1(arrays: Sequence[ba.BatchedArray]) -> None:
    assert objects_are_equal(
        np.concatenate(arrays, axis=1),
        ba.array([[0, 1, 2, 4, 5], [10, 11, 12, 14, 15]]),
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
            [ba.array([[0, 1, 2], [10, 11, 12]]), np.array([[4, 5], [14, 15]])],
            axis=1,
        )


def test_concatenate_axis_none() -> None:
    assert objects_are_equal(
        np.concatenate(
            [
                ba.array([[0, 1, 2], [4, 5, 6]]),
                ba.array([[10, 11, 12], [13, 14, 15]]),
            ],
            axis=None,
        ),
        np.array([0, 1, 2, 4, 5, 6, 10, 11, 12, 13, 14, 15]),
    )


def test_concatenate_custom_axes() -> None:
    assert objects_are_equal(
        np.concatenate([ba.ones((2, 3), batch_axis=1), ba.ones((2, 3), batch_axis=1)]),
        ba.ones((4, 3), batch_axis=1),
    )


def test_concatenate_incorrect_batch_axis() -> None:
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        np.concatenate([ba.ones((2, 2)), ba.zeros((2, 2), batch_axis=1)])


#############################################
#     Tests for concatenate_along_batch     #
#############################################


@pytest.mark.parametrize(
    "arrays",
    [
        [
            ba.array([[0, 1, 2], [4, 5, 6]]),
            ba.array([[10, 11, 12], [13, 14, 15]]),
        ],
        [
            ba.array([[0, 1, 2], [4, 5, 6]]),
            ba.array([[10, 11, 12]]),
            ba.array([[13, 14, 15]]),
        ],
    ],
)
def test_concatenate_along_batch_axis_0(arrays: Sequence[ba.BatchedArray]) -> None:
    assert objects_are_equal(
        ba.concatenate_along_batch(arrays),
        ba.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]),
    )


@pytest.mark.parametrize(
    "arrays",
    [
        [
            ba.array([[0, 1, 2], [10, 11, 12]], batch_axis=1),
            ba.array([[4, 5], [14, 15]], batch_axis=1),
        ],
        [
            ba.array([[0, 1, 2], [10, 11, 12]], batch_axis=1),
            ba.array([[4], [14]], batch_axis=1),
            ba.array([[5], [15]], batch_axis=1),
        ],
    ],
)
def test_concatenate_along_batch_axis_1(arrays: Sequence[ba.BatchedArray]) -> None:
    assert objects_are_equal(
        ba.concatenate_along_batch(arrays),
        ba.array([[0, 1, 2, 4, 5], [10, 11, 12, 14, 15]], batch_axis=1),
    )


def test_concatenate_along_batch_incorrect_batch_axis() -> None:
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        ba.concatenate_along_batch([ba.ones((2, 2)), ba.zeros((2, 2), batch_axis=1)])
