from unittest.mock import Mock

from coola import objects_are_equal
from pytest import mark, raises

from redcat import BaseBatch, BatchDict, BatchList
from redcat.batchdict import check_batch_size


def test_batch_dict_init_data_1_item() -> None:
    assert objects_are_equal(
        BatchDict({"key": BatchList([1, 2, 3, 4])}).data, {"key": BatchList([1, 2, 3, 4])}
    )


def test_batch_dict_init_data_2_items() -> None:
    assert objects_are_equal(
        BatchDict({"key1": BatchList([1, 2, 3, 4]), "key2": BatchList(["a", "b", "c", "d"])}).data,
        {"key1": BatchList([1, 2, 3, 4]), "key2": BatchList(["a", "b", "c", "d"])},
    )


def test_batch_dict_init_data_different_batch_sizes() -> None:
    with raises(
        RuntimeError,
        match="Incorrect batch size. A single batch size is expected but received several values:",
    ):
        BatchDict({"key1": BatchList([1, 2, 3, 4]), "key2": BatchList(["a", "b", "c"])})


def test_batch_dict_init_data_incorrect_type() -> None:
    with raises(TypeError, match="Incorrect type. Expect a dict but received"):
        BatchDict(BatchList([1, 2, 3, 4]))


def test_batch_dict_str() -> None:
    assert str(BatchDict({"key": BatchList([1, 2, 3, 4])})).startswith("BatchDict(")


@mark.parametrize("batch_size", (1, 2))
def test_batch_dict_batch_size(batch_size: int) -> None:
    assert (
        BatchDict(
            {"key1": BatchList([1] * batch_size), "key2": BatchList(["a"] * batch_size)}
        ).batch_size
        == batch_size
    )


###############################
#     Creation operations     #
###############################


def test_batch_dict_clone() -> None:
    batch = BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})
    clone = batch.clone()
    batch.data["key2"][1] = "d"
    assert batch.equal(
        BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "d", "c"])})
    )
    assert clone.equal(
        BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})
    )


#################################
#     Comparison operations     #
#################################


def test_batch_dict_allclose_true() -> None:
    assert BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "d", "c"])}).allclose(
        BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "d", "c"])})
    )


def test_batch_dict_allclose_false_different_type() -> None:
    assert not BatchDict(
        {"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "d", "c"])}
    ).allclose(["a", "b", "c"])


def test_batch_dict_allclose_false_different_data() -> None:
    assert not BatchDict({"key": BatchList(["a", "d", "c"])}).allclose(
        BatchDict({"key": BatchList(["a", "d", "c", "d"])})
    )


@mark.parametrize(
    "batch,atol",
    (
        (BatchDict({"key": BatchList([0.5, 1.5, 2.5, 3.5])}), 1.0),
        (BatchDict({"key": BatchList([0.05, 1.05, 2.05, 3.05])}), 1e-1),
        (BatchDict({"key": BatchList([0.005, 1.005, 2.005, 3.005])}), 1e-2),
    ),
)
def test_batch_dict_allclose_true_atol(batch: BatchList, atol: float) -> None:
    assert BatchDict({"key": BatchList([0.0, 1.0, 2.0, 3.0])}).allclose(batch, atol=atol, rtol=0)


@mark.parametrize(
    "batch,rtol",
    (
        (BatchDict({"key": BatchList([1.5, 2.5, 3.5])}), 1.0),
        (BatchDict({"key": BatchList([1.05, 2.05, 3.05])}), 1e-1),
        (BatchDict({"key": BatchList([1.005, 2.005, 3.005])}), 1e-2),
    ),
)
def test_batch_dict_allclose_true_rtol(batch: BatchList, rtol: float) -> None:
    assert BatchDict({"key": BatchList([1.0, 2.0, 3.0])}).allclose(batch, rtol=rtol)


def test_batch_dict_equal_true() -> None:
    assert BatchDict({"key": BatchList(["a", "b", "c"])}).equal(
        BatchDict({"key": BatchList(["a", "b", "c"])})
    )


def test_batch_dict_equal_false_different_type() -> None:
    assert not BatchDict({"key": BatchList(["a", "b", "c"])}).equal(["a", "b", "c"])


def test_batch_dict_equal_false_different_data() -> None:
    assert not BatchDict({"key": BatchList(["a", "b", "c"])}).equal(BatchList(["a", "b", "c", "d"]))


######################################
#     Tests for check_batch_size     #
######################################


def test_check_batch_size_1_item() -> None:
    check_batch_size({"key": Mock(spec=BaseBatch, batch_size=42)})
    # will fail if an exception is raised


def test_check_batch_size_2_items_same_batch_size() -> None:
    check_batch_size(
        {"key1": Mock(spec=BaseBatch, batch_size=42), "key2": Mock(spec=BaseBatch, batch_size=42)}
    )
    # will fail if an exception is raised


def test_check_batch_size_2_items_different_batch_size() -> None:
    with raises(
        RuntimeError,
        match="Incorrect batch size. A single batch size is expected but received several values:",
    ):
        check_batch_size(
            {
                "key1": Mock(spec=BaseBatch, batch_size=42),
                "key2": Mock(spec=BaseBatch, batch_size=4),
            }
        )
