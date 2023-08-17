from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import Mock

from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available, is_torch_available
from pytest import mark

from redcat import BatchList
from redcat.utils.array import permute_along_axis, to_array

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()


if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


#######################################
#     Tests for permute_along_axis     #
#######################################


@numpy_available
def test_permute_along_axis_1d() -> None:
    assert np.array_equal(
        permute_along_axis(np.arange(4), permutation=np.array([0, 2, 1, 3])), np.array([0, 2, 1, 3])
    )


@numpy_available
def test_permute_along_axis_2d_axis_0() -> None:
    assert np.array_equal(
        permute_along_axis(np.arange(20).reshape(4, 5), permutation=np.array([0, 2, 1, 3])),
        np.array([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [15, 16, 17, 18, 19]]),
    )


@numpy_available
def test_permute_along_axis_2d_axis_1() -> None:
    assert np.array_equal(
        permute_along_axis(
            np.arange(20).reshape(4, 5), permutation=np.array([0, 4, 2, 1, 3]), axis=1
        ),
        np.array([[0, 4, 2, 1, 3], [5, 9, 7, 6, 8], [10, 14, 12, 11, 13], [15, 19, 17, 16, 18]]),
    )


@numpy_available
def test_permute_along_axis_3d_axis_2() -> None:
    assert np.array_equal(
        permute_along_axis(
            np.arange(20).reshape(2, 2, 5), permutation=np.array([0, 4, 2, 1, 3]), axis=2
        ),
        np.array(
            [[[0, 4, 2, 1, 3], [5, 9, 7, 6, 8]], [[10, 14, 12, 11, 13], [15, 19, 17, 16, 18]]]
        ),
    )


##############################
#     Tests for to_array     #
##############################


@numpy_available
@mark.parametrize(
    "data",
    (
        np.array([3, 1, 2, 0, 1]),
        [3, 1, 2, 0, 1],
        (3, 1, 2, 0, 1),
        BatchList([3, 1, 2, 0, 1]),
        # TODO: uncomment after BatchedArray is a BaseBatch
        # BatchedArray(np.array([3, 1, 2, 0, 1])),
    ),
)
def test_to_array_long(data: Sequence | np.ndarray) -> None:
    assert np.array_equal(to_array(data), np.array([3, 1, 2, 0, 1]))


@numpy_available
@mark.parametrize(
    "data",
    (
        np.array([3.0, 1.0, 2.0, 0.0, 1.0]),
        [3.0, 1.0, 2.0, 0.0, 1.0],
        (3.0, 1.0, 2.0, 0.0, 1.0),
        BatchList([3.0, 1.0, 2.0, 0.0, 1.0]),
        # TODO: uncomment after BatchedArray is a BaseBatch
        # BatchedArray(np.array([3.0, 1.0, 2.0, 0.0, 1.0])),
    ),
)
def test_to_array_float(data: Sequence | np.ndarray) -> None:
    assert np.array_equal(to_array(data), np.array([3.0, 1.0, 2.0, 0.0, 1.0]))


@torch_available
def test_to_array_torch() -> None:
    assert np.array_equal(to_array(torch.tensor([3, 1, 2, 0, 1])), np.array([3, 1, 2, 0, 1]))
