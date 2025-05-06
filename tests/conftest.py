__all__ = ["TORCH_GREATER_EQUAL_1_13", "torch_greater_equal_1_13"]

import operator

import pytest
from feu import compare_version

TORCH_GREATER_EQUAL_1_13 = compare_version("torch", operator.ge, "1.13.0")

torch_greater_equal_1_13 = pytest.mark.skipif(
    not TORCH_GREATER_EQUAL_1_13, reason="Requires torch>=1.13.0"
)

future_test = pytest.mark.skipif(True, reason="Future tests")
