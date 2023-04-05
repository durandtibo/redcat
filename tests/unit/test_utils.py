from unittest.mock import patch

from redcat.utils import get_available_devices

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
