from __future__ import annotations

__all__ = ["ValuesIndicesTuple"]

from collections import namedtuple
from typing import Any

from coola import objects_are_equal
from coola.utils import str_indent, str_mapping


class ValuesIndicesTuple(namedtuple("TupleValuesIndices", ["values", "indices"])):
    r"""Implements a namedtuple to represent the pair ``(values,
    indices)``.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from redcat.return_types import ValuesIndicesTuple
        >>> x = ValuesIndicesTuple(torch.ones(6), torch.arange(6))
        >>> x
        ValuesIndicesTuple(
          (values): tensor([1., 1., 1., 1., 1., 1.])
          (indices): tensor([0, 1, 2, 3, 4, 5])
        )
        >>> x.values
        tensor([1., 1., 1., 1., 1., 1.])
        >>> x.indices
        tensor([0, 1, 2, 3, 4, 5])
    """

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ValuesIndicesTuple):
            return False
        return objects_are_equal(self.values, other.values) and objects_are_equal(
            self.indices, other.indices
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        args = str_indent(str_mapping({"values": self.values, "indices": self.indices}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"
