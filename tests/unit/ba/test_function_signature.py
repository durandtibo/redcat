from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np
import pytest

from redcat import ba

FUNCTIONS = [
    (np.argsort, ba.argsort),
    (np.cumprod, ba.cumprod),
    (np.cumsum, ba.cumsum),
    (np.nancumprod, ba.nancumprod),
    (np.nancumsum, ba.nancumsum),
]


@pytest.mark.parametrize(("func1", "func2"), FUNCTIONS)
def test_function_signature(func1: Callable, func2: Callable) -> None:
    params1 = inspect.signature(func1).parameters
    params2 = inspect.signature(func2).parameters
    assert list(params1.keys()) == list(params2.keys())
