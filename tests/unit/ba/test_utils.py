from __future__ import annotations

import numpy as np
import pytest

from redcat import BatchedArray
from redcat.ba import check_same_batch_axis, get_batch_axes

###########################################
#     Tests for check_same_batch_axis     #
###########################################


def test_check_same_batch_axis_correct() -> None:
    check_same_batch_axis({0})


def test_check_same_batch_axis_incorrect() -> None:
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        check_same_batch_axis({0, 1})


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
        (BatchedArray(np.ones((2, 3))), BatchedArray(np.ones((2, 3)), batch_dim=1)),
        {"val": BatchedArray(np.ones((2, 3)))},
    ) == {0, 1}


def test_get_batch_axes_empty() -> None:
    assert get_batch_axes(tuple()) == set()
