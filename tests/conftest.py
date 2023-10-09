__all__ = ["torch_greater_equal_1_13", "TORCH_GREATER_EQUAL_1_13"]

import operator

from feu import compare_version
from pytest import mark

TORCH_GREATER_EQUAL_1_13 = compare_version("torch", operator.ge, "1.13.0")

torch_greater_equal_1_13 = mark.skipif(
    not TORCH_GREATER_EQUAL_1_13, reason="Requires torch>=1.13.0"
)
