from __future__ import annotations

import numpy as np
import pytest

from redcat.ba import (
    BatchedArray,
    check_data_and_axis,
    check_same_batch_axis,
    get_batch_axes,
)

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
    assert get_batch_axes(
        (BatchedArray(np.ones((2, 3))), BatchedArray(np.ones((2, 3)))),
        {"val": BatchedArray(np.ones((2, 3)))},
    ) == {0}


def test_get_batch_axes_2_array() -> None:
    assert get_batch_axes(
        (BatchedArray(np.ones((2, 3))), BatchedArray(np.ones((2, 3)), batch_axis=1)),
        {"val": BatchedArray(np.ones((2, 3)))},
    ) == {0, 1}


def test_get_batch_axes_empty() -> None:
    assert get_batch_axes(tuple()) == set()
