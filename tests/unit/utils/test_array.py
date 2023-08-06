from __future__ import annotations

import numpy as np
import torch
from pytest import mark, raises

from redcat.utils.array import check_data_and_dim

########################################
#     Tests for check_data_and_dim     #
########################################


@mark.parametrize("array", (np.ones((2, 3)), torch.ones(2, 3)))
def test_check_data_and_dim_correct(array: np.ndarray | torch.Tensor) -> None:
    check_data_and_dim(array, batch_dim=0)
    # will fail if an exception is raised


@mark.parametrize("array", (np.array(2), torch.tensor(2)))
def test_check_data_and_dim_incorrect_data_dim(array: np.ndarray | torch.Tensor) -> None:
    with raises(RuntimeError, match=r"data needs at least 1 dimensions \(received: 0\)"):
        check_data_and_dim(np.array(2), batch_dim=0)


@mark.parametrize("array", (np.ones((2, 3)), torch.ones(2, 3)))
@mark.parametrize("batch_dim", (-1, 2, 3))
def test_check_data_and_dim_incorrect_batch_dim(
    array: np.ndarray | torch.Tensor, batch_dim: int
) -> None:
    with raises(
        RuntimeError, match=r"Incorrect batch_dim \(.*\) but the value should be in \[0, 1\]"
    ):
        check_data_and_dim(np.ones((2, 3)), batch_dim=batch_dim)
