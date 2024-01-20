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
    (np.sort, ba.sort),
]

U_FUNCTIONS = [
    (np.absolute, ba.absolute),
    (np.clip, ba.clip),
    (np.exp, ba.exp),
    (np.exp2, ba.exp2),
    (np.expm1, ba.expm1),
    (np.float_power, ba.float_power),
    (np.fmax, ba.fmax),
    (np.fmin, ba.fmin),
    (np.log, ba.log),
    (np.log10, ba.log10),
    (np.log1p, ba.log1p),
    (np.log2, ba.log2),
    (np.maximum, ba.maximum),
    (np.minimum, ba.minimum),
    (np.power, ba.power),
    (np.sign, ba.sign),
    (np.sqrt, ba.sqrt),
    (np.square, ba.square),
]


@pytest.mark.parametrize(("func1", "func2"), FUNCTIONS)
def test_function_signature(func1: Callable, func2: Callable) -> None:
    params1 = inspect.signature(func1).parameters
    params2 = inspect.signature(func2).parameters
    assert list(params1.keys()) == list(params2.keys())
