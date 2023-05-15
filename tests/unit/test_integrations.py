from unittest.mock import patch

from pytest import raises

from redcat.integrations import (
    check_polars,
    check_torch,
    is_polars_available,
    is_torch_available,
)

##################
#     polars     #
##################


def test_check_polars_with_package() -> None:
    with patch("redcat.integrations.is_polars_available", lambda *args: True):
        check_polars()


def test_check_polars_without_package() -> None:
    with patch("redcat.integrations.is_polars_available", lambda *args: False):
        with raises(RuntimeError, match="`polars` package is required but not installed."):
            check_polars()


def test_is_polars_available() -> None:
    assert isinstance(is_polars_available(), bool)


#################
#     torch     #
#################


def test_check_torch_with_package() -> None:
    with patch("redcat.integrations.is_torch_available", lambda *args: True):
        check_torch()


def test_check_torch_without_package() -> None:
    with patch("redcat.integrations.is_torch_available", lambda *args: False):
        with raises(RuntimeError, match="`torch` package is required but not installed."):
            check_torch()


def test_is_torch_available() -> None:
    assert isinstance(is_torch_available(), bool)
