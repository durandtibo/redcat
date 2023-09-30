import importlib

from packaging.version import Version
from pytest import mark

TORCH_GREATER_EQUAL_1_13 = Version(importlib.import_module("torch").__version__) >= Version(
    "1.13.0"
)

torch_greater_equal_1_13 = mark.skipif(
    not TORCH_GREATER_EQUAL_1_13, reason="Requires torch>=1.13.0"
)
