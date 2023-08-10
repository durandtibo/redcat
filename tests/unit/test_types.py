import torch
from coola import objects_are_equal

from redcat.types import SortReturnType

####################################
#     Tests for SortReturnType     #
####################################


def test_sort_return_type_tuple() -> None:
    assert objects_are_equal(
        tuple(SortReturnType(torch.ones(2, 4), torch.ones(2, 4))),
        (torch.ones(2, 4), torch.ones(2, 4)),
    )


def test_sort_return_type_eq_true() -> None:
    assert SortReturnType(torch.ones(2, 4), torch.ones(2, 4)) == SortReturnType(
        torch.ones(2, 4), torch.ones(2, 4)
    )


def test_sort_return_type_eq_false_different_values() -> None:
    assert SortReturnType(torch.ones(2, 4), torch.ones(2, 4)) != SortReturnType(
        torch.zeros(2, 4), torch.ones(2, 4)
    )


def test_sort_return_type_eq_false_different_indices() -> None:
    assert SortReturnType(torch.ones(2, 4), torch.ones(2, 4)) != SortReturnType(
        torch.ones(2, 4), torch.zeros(2, 4)
    )


def test_sort_return_type_eq_false_different_types() -> None:
    assert SortReturnType(torch.ones(2, 4), torch.ones(2, 4)) != (
        torch.ones(2, 4),
        torch.zeros(2, 4),
    )
