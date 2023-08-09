from __future__ import annotations

__all__ = ["randperm"]

import random
from typing import overload
from unittest.mock import Mock

from coola.utils import is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()


if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


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
def randperm(n: int, generator: int | None = None) -> list[int]:
    r"""Creates a random permutation of integers from ``0`` to ``n - 1``.

    Args:
    ----
        n (int): Specifies the number of items.
        rng_or_seed (int or ``None``): Specifies the random seed for the
            random number generator.

    Returns:
    -------
        ``list``: A random permutation of integers from
            ``0`` to ``n - 1``.
    """


def randperm(
    n: int, rng_or_seed: torch.Generator | np.random.Generator | int | None = None
) -> torch.Tensor | np.ndarray | list[int]:
    r"""Creates a random permutation of integers from ``0`` to ``n - 1``.

    Args:
    ----
        n (int): Specifies the number of items.
        rng_or_seed (``numpy.random.Generator`` or ``torch.Generator``
            or int or ``None``): Specifies the pseudorandom number
            generator for sampling or the random seed for the random
            number generator. Default: ``None``

    Returns:
    -------
        ``numpy.ndarray`` or ``torch.Tensor`` or ``list``: A random
            permutation of integers from ``0`` to ``n - 1``.

    Example usage with ``numpy``:

    .. code-block:: pycon

        >>> import numpy as np
        >>> from redcat.utils.random import randperm
        >>> randperm(10, np.random.default_rng(42))  # doctest:+ELLIPSIS
        array([...])

    Example usage with ``torch``:

    .. code-block:: pycon

        >>> from redcat.utils.tensor import get_torch_generator
        >>> from redcat.utils.random import randperm
        >>> randperm(10, get_torch_generator(42))  # doctest:+ELLIPSIS
        tensor([...])

    Example usage:

    .. code-block:: pycon

        >>> from redcat.utils.random import randperm
        >>> randperm(10, 42)  # doctest:+ELLIPSIS
        [...]
    """
    if isinstance(rng_or_seed, torch.Generator):
        return torch.randperm(n, generator=rng_or_seed)
    if isinstance(rng_or_seed, np.random.Generator):
        return rng_or_seed.permutation(n)
    out = list(range(n))
    random.Random(rng_or_seed).shuffle(out)
    return out
