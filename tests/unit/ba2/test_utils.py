from __future__ import annotations

import numpy as np
import pytest

from redcat import ba
from redcat.ba import (
    arrays_share_data,
    check_data_and_axis,
    check_same_batch_axis,
    get_batch_axes,
    get_data_base,
)

#######################################
#     Tests for arrays_share_data     #
#######################################


def test_arrays_share_data_true() -> None:
    x = ba.ones((2, 3))
    assert arrays_share_data(x, x)


def test_arrays_share_data_true_slice() -> None:
    x = ba.ones((2, 3))
    assert arrays_share_data(x, x[1:])


def test_arrays_share_data_false() -> None:
    x = ba.ones((2, 3))
    assert not arrays_share_data(x, x.copy())


###########################################
#     Tests for check_same_batch_axis     #
###########################################


def test_check_same_batch_axis_correct() -> None:
    check_same_batch_axis({0})


def test_check_same_batch_axis_incorrect() -> None:
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        check_same_batch_axis({0, 1})


########################################
#     Tests for check_data_and_axis     #
########################################


@pytest.mark.parametrize("array", [np.ones((2, 3)), np.zeros((2, 3))])
def test_check_data_and_axis_correct(array: np.ndarray) -> None:
    check_data_and_axis(array, batch_axis=0)
    # will fail if an exception is raised


@pytest.mark.parametrize("array", [np.array(2), np.array(5)])
def test_check_data_and_axis_incorrect_data_axis(array: np.ndarray) -> None:
    with pytest.raises(RuntimeError, match=r"data needs at least 1 axis \(received: 0\)"):
        check_data_and_axis(np.array(2), batch_axis=0)


@pytest.mark.parametrize("array", [np.ones((2, 3)), np.zeros((2, 3))])
@pytest.mark.parametrize("batch_axis", [-1, 2, 3])
def test_check_data_and_axis_incorrect_batch_axis(array: np.ndarray, batch_axis: int) -> None:
    with pytest.raises(
        RuntimeError, match=r"Incorrect `batch_axis` \(.*\) but the value should be in \[0, 1\]"
    ):
        check_data_and_axis(np.ones((2, 3)), batch_axis=batch_axis)


####################################
#     Tests for get_batch_axes     #
####################################


def test_get_batch_axes_1_array() -> None:
    assert get_batch_axes((ba.ones((2, 3)), ba.ones(2)), {"batch": ba.ones((2, 4))}) == {0}


def test_get_batch_axes_2_array() -> None:
    assert get_batch_axes(
        (ba.ones((2, 3), batch_axis=1), ba.ones(2)), {"batch": ba.ones((2, 4))}
    ) == {0, 1}


def test_get_batch_axes_empty() -> None:
    assert get_batch_axes(tuple()) == set()


###################################
#     Tests for get_data_base     #
###################################


def test_get_data_base_true() -> None:
    x = ba.ones((2, 3))
    assert get_data_base(x) is x.base


def test_get_data_base_true_slice() -> None:
    x = ba.ones((2, 3))
    assert get_data_base(x[1:]) is x.base


def test_get_data_base_false() -> None:
    x = ba.ones((2, 3))
    assert get_data_base(x.copy()) is not x
