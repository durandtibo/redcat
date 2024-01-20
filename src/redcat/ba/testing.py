from __future__ import annotations

__all__ = ["FunctionCheck", "make_rand_arrays", "make_randn_arrays"]

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

# Default shape value
SHAPE = (4, 10)


def make_rand_arrays(shape: int | Sequence[int], n: int) -> tuple[np.ndarray, ...]:
    r"""Make a tuple of arrays filled with random values sampled from a
    uniform distribution U(0,1).

    Args:
        shape: The dimensions of the returned arrays, must be
            non-negative.
        n: The number of arrays.

    Returns:
        A tuple of arrays filled with random values sampled from a
            uniform distribution U(0,1).

    Example usage:

    ```pycon
    >>> from redcat.ba.testing import make_rand_arrays
    >>> arrays = make_rand_arrays(shape=(2, 3), n=1)
    >>> arrays
    (array([[...]]),)
    >>> arrays = make_rand_arrays(shape=(2, 3), n=2)
    >>> arrays
    (array([[...]]), array([[...]]))

    ```
    """
    if isinstance(shape, int):
        shape = (shape,)
    return tuple(np.random.rand(*shape) for _ in range(n))


def make_randn_arrays(shape: int | Sequence[int], n: int) -> tuple[np.ndarray, ...]:
    r"""Make a tuple of arrays filled with random values sampled from a
    Normal distribution N(0,1).

    Args:
        shape: The dimensions of the returned arrays, must be
            non-negative.
        n: The number of arrays.

    Returns:
        A tuple of arrays filled with random values sampled from
            a Normal distribution N(0,1).

    Example usage:

    ```pycon
    >>> from redcat.ba.testing import make_randn_arrays
    >>> arrays = make_randn_arrays(shape=(2, 3), n=1)
    >>> arrays
    (array([[...]]),)
    >>> arrays = make_randn_arrays(shape=(2, 3), n=2)
    >>> arrays
    (array([[...]]), array([[...]]))

    ```
    """
    if isinstance(shape, int):
        shape = (shape,)
    return tuple(np.random.randn(*shape) for _ in range(n))


@dataclass
class FunctionCheck:
    function: Callable
    nin: int
    nout: int
    arrays: tuple[np.ndarray, ...] | None = None

    def get_arrays(self) -> tuple[np.ndarray, ...]:
        r"""Get the input arrays.

        Returns:
            The input arrays.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba.testing import FunctionCheck
        >>> check = FunctionCheck(np.add, nin=2, nout=1)
        >>> arrays = check.get_arrays()
        >>> arrays
        (array([[...]]), array([[...]]))

        ```
        """
        if self.arrays is None:
            return make_rand_arrays(shape=SHAPE, n=self.nin)
        return self.arrays

    @classmethod
    def create_ufunc(
        cls, ufunc: np.ufunc, arrays: tuple[np.ndarray, ...] | None = None
    ) -> FunctionCheck:
        r"""Instantiate a ``FunctionCheck`` from a universal function
        (``ufunc``).

        Args:
            ufunc: The universal function.
            arrays: Specifies the input arrays.

        Returns:
            The instantiated ``FunctionCheck``.

        Example usage:

        ```pycon
        >>> import numpy as np
        >>> from redcat.ba.testing import FunctionCheck
        >>> check = FunctionCheck.create_ufunc(np.add)
        >>> check.nin
        2
        >>> check.nout
        1

        ```
        """
        return cls(
            function=ufunc,
            nin=ufunc.nin,
            nout=ufunc.nout,
            arrays=arrays,
        )
