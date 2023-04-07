from unittest.mock import patch

import torch

from redcat.utils import align_to_batch_first, get_available_devices

##########################################
#     Tests for align_to_batch_first     #
##########################################


def test_align_to_batch_first_no_permutation() -> None:
    assert align_to_batch_first(torch.arange(10).view(5, 2), batch_dim=0).equal(
        torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    )


def test_align_to_batch_first_permute_dims() -> None:
    assert align_to_batch_first(torch.arange(10).view(5, 2), batch_dim=1).equal(
        torch.tensor([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]])
    )


def test_align_to_batch_first_permute_dims_extra_dims() -> None:
    assert align_to_batch_first(torch.arange(20).view(2, 5, 2), batch_dim=1).equal(
        torch.tensor(
            [
                [[0, 1], [10, 11]],
                [[2, 3], [12, 13]],
                [[4, 5], [14, 15]],
                [[6, 7], [16, 17]],
                [[8, 9], [18, 19]],
            ]
        )
    )


###########################################
#     Tests for get_available_devices     #
###########################################


@patch("torch.cuda.is_available", lambda *args, **kwargs: False)
def test_get_available_devices_cpu() -> None:
    assert get_available_devices() == ("cpu",)


@patch("torch.cuda.is_available", lambda *args, **kwargs: True)
@patch("torch.cuda.device_count", lambda *args, **kwargs: 1)
def test_get_available_devices_cpu_and_gpu() -> None:
    assert get_available_devices() == ("cpu", "cuda:0")
