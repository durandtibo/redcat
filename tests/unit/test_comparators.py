from __future__ import annotations

import logging

import pytest
from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler
from coola.equality.testers import EqualityTester

from redcat import BaseBatch, BatchList
from redcat.comparators import BatchEqualHandler, BatchEqualityComparator
from tests.unit.helpers import ExamplePair


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


BATCH_EQUAL = [
    pytest.param(
        ExamplePair(object1=BatchList([]), object2=BatchList([])),
        id="list empty",
    ),
    pytest.param(
        ExamplePair(object1=BatchList([1, 2, 3]), object2=BatchList([1, 2, 3])),
        id="list int",
    ),
    pytest.param(
        ExamplePair(
            object1=BatchList([1.0, 2.0, 3.0, 4.0]), object2=BatchList([1.0, 2.0, 3.0, 4.0])
        ),
        id="list float",
    ),
    pytest.param(
        ExamplePair(object1=BatchList(["a", "b", "c"]), object2=BatchList(["a", "b", "c"])),
        id="list str",
    ),
]
BATCH_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            object1=BatchList([1, 2, 3]),
            object2=BatchList([1, 2, 4]),
            expected_message="batches are not equal:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            object1=BatchList([1, 2, 3]),
            object2=BatchList([1, 2, 3, 4]),
            expected_message="batches are not equal:",
        ),
        id="different batch sizes",
    ),
    pytest.param(
        ExamplePair(
            object1=BatchList([1, 2, 3]),
            object2=[1, 2, 3, 4],
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]
BATCH_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(object1=BatchList([1, 2, 3]), object2=BatchList([1, 2, 4]), atol=1.0),
        id="atol=1",
    ),
]


def test_registered_batch_comparators() -> None:
    assert isinstance(EqualityTester.registry[BaseBatch], BatchEqualityComparator)


#######################################
#     Tests for BatchEqualHandler     #
#######################################


def test_batch_equal_handler_eq_true() -> None:
    assert BatchEqualHandler() == BatchEqualHandler()


def test_batch_equal_handler_eq_false() -> None:
    assert BatchEqualHandler() != FalseHandler()


def test_batch_equal_handler_repr() -> None:
    assert repr(BatchEqualHandler()).startswith("BatchEqualHandler(")


def test_batch_equal_handler_str() -> None:
    assert str(BatchEqualHandler()).startswith("BatchEqualHandler(")


def test_batch_equal_handler_handle_true(config: EqualityConfig) -> None:
    assert BatchEqualHandler().handle(BatchList([1, 2, 3]), BatchList([1, 2, 3]), config)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (BatchList([1, 2, 3]), BatchList([1, 2, 4])),
        (BatchList([1, 2, 3]), BatchList([1, 2, 3, 4])),
        (BatchList([1, 2, 3]), BatchList([1.0, 2.0, 3.0])),
    ],
)
def test_batch_equal_handler_handle_false(
    object1: BaseBatch, object2: BaseBatch, config: EqualityConfig
) -> None:
    assert not BatchEqualHandler().handle(object1, object2, config)


def test_batch_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = BatchEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            object1=BatchList([1, 2, 3]), object2=BatchList([1, 2, 4]), config=config
        )
        assert caplog.messages[-1].startswith("batches are not equal:")


def test_batch_equal_handler_set_next_handler() -> None:
    BatchEqualHandler().set_next_handler(FalseHandler())


#############################################
#     Tests for BatchEqualityComparator     #
#############################################


def test_batch_equality_comparator_str() -> None:
    assert str(BatchEqualityComparator()) == "BatchEqualityComparator()"


def test_batch_equality_comparator__eq__true() -> None:
    assert BatchEqualityComparator() == BatchEqualityComparator()


def test_batch_equality_comparator__eq__false() -> None:
    assert BatchEqualityComparator() != 123


def test_batch_equality_comparator_clone() -> None:
    op = BatchEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_batch_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    obj = BatchList([1, 2, 3])
    assert BatchEqualityComparator().equal(obj, obj, config)


@pytest.mark.parametrize("example", BATCH_EQUAL)
def test_batch_equality_comparator_equal_yes(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = BatchEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", BATCH_EQUAL)
def test_batch_equality_comparator_equal_yes_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = BatchEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", BATCH_NOT_EQUAL)
def test_batch_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = BatchEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", BATCH_NOT_EQUAL)
def test_batch_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = BatchEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_batch_equality_comparator_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert (
        BatchEqualityComparator().equal(
            object1=BatchList([1, 2, float("nan")]),
            object2=BatchList([1, 2, float("nan")]),
            config=config,
        )
        == equal_nan
    )


@pytest.mark.parametrize("example", BATCH_EQUAL_TOLERANCE)
def test_batch_equality_comparator_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert BatchEqualityComparator().equal(
        object1=example.object1, object2=example.object2, config=config
    )
