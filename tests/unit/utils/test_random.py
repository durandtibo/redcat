from __future__ import annotations

from unittest.mock import Mock

from coola import objects_are_equal
from coola.testing import numpy_available, torch_available
from coola.utils import is_numpy_available, is_torch_available
from pytest import mark

from redcat.utils.random import randperm
from redcat.utils.tensor import get_torch_generator

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()


if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

##############################
#     Tests for randperm     #
##############################


@torch_available
@mark.parametrize("n", (1, 2, 4))
def test_randperm_torch(n: int) -> None:
    out = randperm(n, get_torch_generator(42))
    assert torch.is_tensor(out)
    assert out.shape == (n,)
    assert len(set(out.tolist())) == n


@torch_available
def test_randperm_torch_same_random_seed() -> None:
    assert objects_are_equal(
        randperm(100, get_torch_generator(1)), randperm(100, get_torch_generator(1))
    )


@torch_available
def test_randperm_torch_different_random_seeds() -> None:
    assert not objects_are_equal(
        randperm(100, get_torch_generator(1)), randperm(100, get_torch_generator(2))
    )


@numpy_available
@mark.parametrize("n", (1, 2, 4))
def test_randperm_numpy(n: int) -> None:
    out = randperm(n, np.random.default_rng(42))
    assert isinstance(out, np.ndarray)
    assert out.shape == (n,)
    assert len(set(out.tolist())) == n


@numpy_available
def test_randperm_numpy_same_random_seed() -> None:
    assert objects_are_equal(
        randperm(100, np.random.default_rng(1)), randperm(100, np.random.default_rng(1))
    )


@numpy_available
def test_randperm_numpy_different_random_seeds() -> None:
    assert not objects_are_equal(
        randperm(100, np.random.default_rng(1)), randperm(100, np.random.default_rng(2))
    )


@mark.parametrize("n", (1, 2, 4))
def test_randperm_int(n: int) -> None:
    out = randperm(n, 42)
    assert isinstance(out, list)
    assert len(out) == n
    assert len(set(out)) == n


@numpy_available
def test_randperm_int_same_random_seed() -> None:
    assert objects_are_equal(randperm(100, 1), randperm(100, 1))


@numpy_available
def test_randperm_int_different_random_seeds() -> None:
    assert not objects_are_equal(randperm(100, 1), randperm(100, 2))


@mark.parametrize("n", (1, 2, 4))
def test_randperm_none(n: int) -> None:
    out = randperm(n)
    assert isinstance(out, list)
    assert len(out) == n
    assert len(set(out)) == n
