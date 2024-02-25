import numpy as np
import pytest
import torch
from coola.testing import numpy_available, torch_available

from redcat import BatchList
from redcat.utils.collection import to_list

#############################
#     Tests for to_list     #
#############################


@pytest.mark.parametrize("data", [[], [1, 2, 3], ["a", "b", "c", "d"]])
def test_to_list_list(data: list) -> None:
    assert to_list(data) is data


def test_to_list_tuple() -> None:
    assert to_list((1, 2, 3)) == [1, 2, 3]


@numpy_available
def test_to_list_array() -> None:
    assert to_list(np.array([1, 2, 3])) == [1, 2, 3]


@torch_available
def test_to_list_tensor() -> None:
    assert to_list(torch.tensor([1, 2, 3])) == [1, 2, 3]


def test_to_list_batch() -> None:
    assert to_list(BatchList([1, 2, 3])) == [1, 2, 3]
