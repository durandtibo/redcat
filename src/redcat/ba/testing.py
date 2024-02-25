r"""Contain testing utility functions."""

from __future__ import annotations

__all__ = ["FunctionCheck", "normal_arrays", "uniform_arrays", "uniform_int_arrays"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


SHAPE = (4, 10)  # Default shape value


def normal_arrays(
    shape: int | Sequence[int],
    n: int,
    loc: float = 0.0,
    scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, ...]:
    r"""Make a tuple of arrays filled with random values sampled from a
    Normal distribution.

    Args:
        shape: The dimensions of the returned arrays, must be
            non-negative.
        n: The number of arrays.
        loc: Mean (“centre”) of the distribution.
        scale: Standard deviation (spread or “width”) of the
            distribution. Must be non-negative.
        rng: A pseudo-random number generator.

    Returns:
        A tuple of arrays filled with random values sampled from
            a Normal distribution.

    Example usage:

    ```pycon
    >>> from redcat.ba.testing import normal_arrays
    >>> arrays = normal_arrays(shape=(2, 3), n=1)
    >>> arrays
    (array([[...]]),)
    >>> arrays = normal_arrays(shape=(2, 3), n=2)
    >>> arrays
    (array([[...]]), array([[...]]))

    ```
    """
    if rng is None:
        rng = np.random.default_rng()
    return tuple(rng.normal(loc=loc, scale=scale, size=shape) for _ in range(n))


def uniform_arrays(
    shape: int | Sequence[int],
    n: int,
    low: float = 0.0,
    high: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, ...]:
    r"""Make a tuple of arrays filled with random values sampled from a
    uniform distribution.

    Args:
        shape: The dimensions of the returned arrays, must be
            non-negative.
        n: The number of arrays.
        low: Lower boundary of the output interval. All values
            generated will be greater than or equal to low.
            The default value is 0.0.
        high: Upper boundary of the output interval. All values
            generated will be less than high. The high limit may be
            included in the returned array of floats due to
            floating-point rounding in the equation
            ``low + (high-low) * random_sample()``. ``high - low``
            must be non-negative. The default value is 1.0.
        rng: A pseudo-random number generator.

    Returns:
        A tuple of arrays filled with random values sampled from a
            uniform distribution.

    Example usage:

    ```pycon
    >>> from redcat.ba.testing import uniform_arrays
    >>> arrays = uniform_arrays(shape=(2, 3), n=1)
    >>> arrays
    (array([[...]]),)
    >>> arrays = uniform_arrays(shape=(2, 3), n=2)
    >>> arrays
    (array([[...]]), array([[...]]))

    ```
    """
    if rng is None:
        rng = np.random.default_rng()
    return tuple(rng.uniform(low=low, high=high, size=shape) for _ in range(n))


def uniform_int_arrays(
    shape: int | Sequence[int],
    n: int,
    low: int = 0,
    high: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, ...]:
    r"""Make a tuple of arrays filled with random values sampled from a
    discrete uniform distribution.

    Args:
        shape: The dimensions of the returned arrays, must be
            non-negative.
        n: The number of arrays.
        low: Lower boundary of the output interval. All values
            generated will be greater than or equal to low.
        high: Upper boundary of the output interval. All values
            generated will be less than high. The high limit may be
            included in the returned array of floats due to
            floating-point rounding in the equation
            ``low + (high-low) * random_sample()``. ``high - low``
            must be non-negative.
        rng: A pseudo-random number generator.

    Returns:
        A tuple of arrays filled with random values sampled from a
            uniform distribution.

    Example usage:

    ```pycon
    >>> from redcat.ba.testing import uniform_arrays
    >>> arrays = uniform_int_arrays(shape=(2, 3), n=1)
    >>> arrays
    (array([[...]]),)
    >>> arrays = uniform_int_arrays(shape=(2, 3), n=2)
    >>> arrays
    (array([[...]]), array([[...]]))

    ```
    """
    if rng is None:
        rng = np.random.default_rng()
    return tuple(
        np.floor(rng.uniform(low=low, high=high, size=shape)).astype(int) for _ in range(n)
    )


@dataclass
class FunctionCheck:
    r"""Implement a class to easily check NumPy functions.

    Args:
        function: The function to check.
        nin: The number of input arrays.
        nout: The number of output arrays.
        arrays: The arrays to use as input of the function.
            If ``None``, arrays are automatically generated with values
            sampled from a Normal distribution.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from redcat.ba.testing import FunctionCheck
    >>> check = FunctionCheck(np.add, nin=2, nout=1)
    >>> check
    FunctionCheck(function=<ufunc 'add'>, nin=2, nout=1, arrays=None)
    >>> check = FunctionCheck(np.add, nin=2, nout=1, arrays=(np.ones((2, 3)), np.ones((2, 3))))
    >>> check
    FunctionCheck(function=<ufunc 'add'>, nin=2, nout=1,
    arrays=(array([[1., 1., 1.], [1., 1., 1.]]), array([[1., 1., 1.], [1., 1., 1.]])))

    ```
    """

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
            return normal_arrays(shape=SHAPE, n=self.nin)
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
