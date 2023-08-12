import torch
from coola import objects_are_equal

from redcat.return_types import ValuesIndicesTuple

########################################
#     Tests for ValuesIndicesTuple     #
########################################


def test_values_indices_tuple_str() -> None:
    assert str(ValuesIndicesTuple(torch.ones(2, 4), torch.ones(2, 4))).startswith(
        "ValuesIndicesTuple("
    )


def test_values_indices_tuple_values() -> None:
    assert objects_are_equal(
        ValuesIndicesTuple(torch.ones(2, 4), torch.zeros(2, 4)).values, torch.ones(2, 4)
    )


def test_values_indices_tuple_indices() -> None:
    assert objects_are_equal(
        ValuesIndicesTuple(torch.ones(2, 4), torch.zeros(2, 4)).indices, torch.zeros(2, 4)
    )


def test_values_indices_tuple_tuple() -> None:
    assert objects_are_equal(
        tuple(ValuesIndicesTuple(torch.ones(2, 4), torch.ones(2, 4))),
        (torch.ones(2, 4), torch.ones(2, 4)),
    )


def test_values_indices_tuple_eq_true() -> None:
    assert ValuesIndicesTuple(torch.ones(2, 4), torch.ones(2, 4)) == ValuesIndicesTuple(
        torch.ones(2, 4), torch.ones(2, 4)
    )


def test_values_indices_tuple_eq_false_different_values() -> None:
    assert ValuesIndicesTuple(torch.ones(2, 4), torch.ones(2, 4)) != ValuesIndicesTuple(
        torch.zeros(2, 4), torch.ones(2, 4)
    )


def test_values_indices_tuple_eq_false_different_indices() -> None:
    assert ValuesIndicesTuple(torch.ones(2, 4), torch.ones(2, 4)) != ValuesIndicesTuple(
        torch.ones(2, 4), torch.zeros(2, 4)
    )


def test_values_indices_tuple_eq_false_different_types() -> None:
    assert ValuesIndicesTuple(torch.ones(2, 4), torch.ones(2, 4)) != (
        torch.ones(2, 4),
        torch.zeros(2, 4),
    )
