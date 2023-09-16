from __future__ import annotations

__all__ = ["get_random_rng", "randperm"]

import random
from typing import overload
from unittest.mock import Mock

from coola.utils import is_numpy_available, is_torch_available

from redcat.types import RNGType

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()


if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


def get_random_rng(rng_or_seed: random.Random | int | None = None) -> random.Random:
    r"""Gets a random number generator.

    Args:
    ----
        rng_or_seed (``random.Random`` or int or ``None``):
            Specifies the pseudorandom number generator for sampling
            or the random seed for the random number generator.

    Returns:
    -------
        ``random.Random``: The initialized random number generator.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from redcat.utils.random import get_random_rng
        >>> get_random_rng(42)
        <random.Random object at 0x...>
    """
    if isinstance(rng_or_seed, random.Random):
        return rng_or_seed
    if rng_or_seed is None:
        return random.Random()
    if isinstance(rng_or_seed, int):
        return random.Random(rng_or_seed)
    raise RuntimeError(f"Invalid `rng_or_seed`: {rng_or_seed}")


@overload
def randperm(n: int, rng: np.random.Generator) -> np.ndarray:
    r"""Creates a random permutation of integers from ``0`` to ``n - 1``.

    Args:
    ----
        n (int): Specifies the number of items.
        rng_or_seed (``numpy.random.Generator``): Specifies the
            pseudorandom number generator for sampling.

    Returns:
    -------
        ``numpy.ndarray``: A random permutation of integers from
            ``0`` to ``n - 1``.
    """


@overload
def randperm(n: int, rng: torch.Generator) -> torch.Tensor:
    r"""Creates a random permutation of integers from ``0`` to ``n - 1``.

    Args:
    ----
        n (int): Specifies the number of items.
        rng_or_seed (``torch.Generator``): Specifies the pseudorandom
            number generator for sampling.

    Returns:
    -------
        ``torch.Tensor``: A random permutation of integers from
            ``0`` to ``n - 1``.
    """


@overload
def randperm(n: int, generator: random.Random | int | None = None) -> list[int]:
    r"""Creates a random permutation of integers from ``0`` to ``n - 1``.

    Args:
    ----
        n (int): Specifies the number of items.
        rng_or_seed (``random.Random`` or int or ``None``, optional):
            Specifies the pseudorandom number generator for sampling
            or the random seed for the random number generator.
            Default: ``None``

    Returns:
    -------
        ``list``: A random permutation of integers from
            ``0`` to ``n - 1``.
    """


def randperm(
    n: int, rng_or_seed: RNGType | int | None = None
) -> torch.Tensor | np.ndarray | list[int]:
    r"""Creates a random permutation of integers from ``0`` to ``n - 1``.

    Args:
    ----
        n (int): Specifies the number of items.
        rng_or_seed (``numpy.random.Generator`` or ``torch.Generator``
            or ``random.Random`` or int or ``None``, optional):
            Specifies the pseudorandom number generator for sampling
            or the random seed for the random number generator.
            Default: ``None``

    Returns:
    -------
        ``numpy.ndarray`` or ``torch.Tensor`` or ``list``: A random
            permutation of integers from ``0`` to ``n - 1``.

    Example usage:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from redcat.utils.random import randperm
        >>> randperm(10, np.random.default_rng(42))
        array([...])

    .. code-block:: pycon

        >>> from redcat.utils.tensor import get_torch_generator
        >>> from redcat.utils.random import randperm
        >>> randperm(10, get_torch_generator(42))
        tensor([...])

    .. code-block:: pycon

        >>> from redcat.utils.random import randperm
        >>> randperm(10, 42)
        [...]
    """
    if isinstance(rng_or_seed, torch.Generator):
        return torch.randperm(n, generator=rng_or_seed)
    if isinstance(rng_or_seed, np.random.Generator):
        return rng_or_seed.permutation(n)
    out = list(range(n))
    get_random_rng(rng_or_seed).shuffle(out)
    return out
