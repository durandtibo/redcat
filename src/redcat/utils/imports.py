__all__ = [
    "check_polars",
    "check_torch",
    "is_polars_available",
    "is_torch_available",
]

from importlib.util import find_spec

##################
#     polars     #
##################


def check_polars() -> None:
    r"""Checks if the ``polars`` package is installed.

    Raises
    ------
        RuntimeError if the ``polars`` package is not installed.
    """
    if not is_polars_available():
        raise RuntimeError(
            "`polars` package is required but not installed. "
            "You can install `polars` package with the command:\n\n"
            "pip install polars\n"
        )


def is_polars_available() -> bool:
    r"""Indicates if the ``polars`` package is installed or not.

    https://www.pola.rs/
    """
    return find_spec("polars") is not None


#################
#     torch     #
#################


def check_torch() -> None:
    r"""Checks if the ``torch`` package is installed.

    Raises
    ------
        RuntimeError if the ``torch`` package is not installed.
    """
    if not is_torch_available():
        raise RuntimeError(
            "`torch` package is required but not installed. "
            "You can install `torch` package with the command:\n\n"
            "pip install torch\n"
        )


def is_torch_available() -> bool:
    r"""Indicates if the ``torch`` package is installed or not.

    https://pytorch.org/
    """
    return find_spec("torch") is not None
