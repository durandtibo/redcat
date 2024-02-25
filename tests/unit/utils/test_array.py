from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available, is_torch_available

from redcat import BatchList
from redcat.ba import BatchedArray
from redcat.utils.array import (
    arrays_share_data,
    get_data_base,
    get_div_rounding_operator,
    permute_along_axis,
    to_array,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()


if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


###############################################
#     Tests for get_div_rounding_operator     #
###############################################


def test_get_div_rounding_operator_mode_none() -> None:
    assert get_div_rounding_operator(None) == np.true_divide


def test_get_div_rounding_operator_mode_floor() -> None:
    assert get_div_rounding_operator("floor") == np.floor_divide


def test_get_div_rounding_operator_mode_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect `rounding_mode`"):
        get_div_rounding_operator("incorrect")


########################################
#     Tests for permute_along_axis     #
########################################


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
@pytest.mark.parametrize(
    "data",
    (
        np.array([3, 1, 2, 0, 1]),
        [3, 1, 2, 0, 1],
        (3, 1, 2, 0, 1),
        BatchList([3, 1, 2, 0, 1]),
        BatchedArray(np.array([3, 1, 2, 0, 1])),
    ),
)
def test_to_array_long(data: Sequence | np.ndarray) -> None:
    assert np.array_equal(to_array(data), np.array([3, 1, 2, 0, 1]))


@numpy_available
@pytest.mark.parametrize(
    "data",
    (
        np.array([3.0, 1.0, 2.0, 0.0, 1.0]),
        [3.0, 1.0, 2.0, 0.0, 1.0],
        (3.0, 1.0, 2.0, 0.0, 1.0),
        BatchList([3.0, 1.0, 2.0, 0.0, 1.0]),
        BatchedArray(np.array([3.0, 1.0, 2.0, 0.0, 1.0])),
    ),
)
def test_to_array_float(data: Sequence | np.ndarray) -> None:
    assert np.array_equal(to_array(data), np.array([3.0, 1.0, 2.0, 0.0, 1.0]))


@torch_available
def test_to_array_torch() -> None:
    assert np.array_equal(to_array(torch.tensor([3, 1, 2, 0, 1])), np.array([3, 1, 2, 0, 1]))


#######################################
#     Tests for arrays_share_data     #
#######################################


def test_arrays_share_data_true() -> None:
    x = np.ones((2, 3))
    assert arrays_share_data(x, x)


def test_arrays_share_data_true_slice() -> None:
    x = np.ones((2, 3))
    assert arrays_share_data(x, x[1:])


def test_arrays_share_data_false() -> None:
    x = np.ones((2, 3))
    assert not arrays_share_data(x, x.copy())


###################################
#     Tests for get_data_base     #
###################################


def test_get_data_base_true() -> None:
    x = np.ones((2, 3))
    assert get_data_base(x) is x


def test_get_data_base_true_slice() -> None:
    x = np.ones((2, 3))
    assert get_data_base(x[1:]) is x


def test_get_data_base_false() -> None:
    x = np.ones((2, 3))
    assert get_data_base(x.copy()) is not x
