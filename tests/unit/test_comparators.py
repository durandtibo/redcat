import logging

import torch
from coola import AllCloseTester, EqualityTester
from pytest import LogCaptureFixture, mark

from redcat import BaseBatch, BatchedTensor
from redcat.comparators import BatchAllCloseOperator, BatchEqualityOperator


def test_registered_batch_comparators() -> None:
    assert isinstance(EqualityTester.registry[BaseBatch], BatchEqualityOperator)
    assert isinstance(AllCloseTester.registry[BaseBatch], BatchAllCloseOperator)


###########################################
#     Tests for BatchEqualityOperator     #
###########################################


def test_batch_equality_operator_str() -> None:
    assert str(BatchEqualityOperator()) == "BatchEqualityOperator()"


def test_batch_equality_operator_equal_true() -> None:
    assert BatchEqualityOperator().equal(
        EqualityTester(), BatchedTensor(torch.ones(2, 3)), BatchedTensor(torch.ones(2, 3))
    )


def test_batch_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert BatchEqualityOperator().equal(
            tester=EqualityTester(),
            object1=BatchedTensor(torch.ones(2, 3)),
            object2=BatchedTensor(torch.ones(2, 3)),
            show_difference=True,
        )
        assert not caplog.messages


def test_batch_equality_operator_equal_false_different_value() -> None:
    assert not BatchEqualityOperator().equal(
        EqualityTester(), BatchedTensor(torch.ones(2, 3)), BatchedTensor(torch.zeros(2, 3))
    )


def test_batch_equality_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not BatchEqualityOperator().equal(
            tester=EqualityTester(),
            object1=BatchedTensor(torch.ones(2, 3)),
            object2=BatchedTensor(torch.zeros(2, 3)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("`BaseBatch` objects are different")


def test_batch_equality_operator_equal_false_different_type() -> None:
    assert not BatchEqualityOperator().equal(EqualityTester(), BatchedTensor(torch.ones(2, 3)), 42)


def test_batch_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not BatchEqualityOperator().equal(
            tester=EqualityTester(),
            object1=BatchedTensor(torch.ones(2, 3)),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a `BaseBatch` object")


###########################################
#     Tests for BatchAllCloseOperator     #
###########################################


def test_batch_allclose_operator_str() -> None:
    assert str(BatchAllCloseOperator()) == "BatchAllCloseOperator()"


def test_batch_allclose_operator_allclose_true() -> None:
    assert BatchAllCloseOperator().allclose(
        AllCloseTester(), BatchedTensor(torch.ones(2, 3)), BatchedTensor(torch.ones(2, 3))
    )


def test_batch_allclose_operator_allclose_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert BatchAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=BatchedTensor(torch.ones(2, 3)),
            object2=BatchedTensor(torch.ones(2, 3)),
            show_difference=True,
        )
        assert not caplog.messages


def test_batch_allclose_operator_allclose_false_different_value() -> None:
    assert not BatchAllCloseOperator().allclose(
        AllCloseTester(), BatchedTensor(torch.ones(2, 3)), BatchedTensor(torch.zeros(2, 3))
    )


def test_batch_allclose_operator_allclose_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not BatchAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=BatchedTensor(torch.ones(2, 3)),
            object2=BatchedTensor(torch.zeros(2, 3)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("`BaseBatch` objects are different")


def test_batch_allclose_operator_allclose_false_different_type() -> None:
    assert not BatchAllCloseOperator().allclose(
        AllCloseTester(), BatchedTensor(torch.ones(2, 3)), 42
    )


def test_batch_allclose_operator_allclose_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not BatchAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=BatchedTensor(torch.ones(2, 3)),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a `BaseBatch` object")


@mark.parametrize(
    "tensor,atol",
    (
        (BatchedTensor(torch.ones(2, 3).add(0.5)), 1),
        (BatchedTensor(torch.ones(2, 3).add(0.05)), 1e-1),
        (BatchedTensor(torch.ones(2, 3).add(5e-3)), 1e-2),
    ),
)
def test_batch_allclose_operator_allclose_true_atol(tensor: BatchedTensor, atol: float) -> None:
    assert BatchAllCloseOperator().allclose(
        AllCloseTester(), BatchedTensor(torch.ones(2, 3)), tensor, atol=atol, rtol=0
    )


@mark.parametrize(
    "tensor,rtol",
    (
        (BatchedTensor(torch.ones(2, 3).add(0.5)), 1),
        (BatchedTensor(torch.ones(2, 3).add(0.05)), 1e-1),
        (BatchedTensor(torch.ones(2, 3).add(5e-3)), 1e-2),
    ),
)
def test_batch_allclose_operator_allclose_true_rtol(tensor: BatchedTensor, rtol: float) -> None:
    assert BatchAllCloseOperator().allclose(
        AllCloseTester(), BatchedTensor(torch.ones(2, 3)), tensor, rtol=rtol
    )
