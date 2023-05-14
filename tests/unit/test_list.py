from pytest import mark, raises

from redcat import BatchList


@mark.parametrize("data", ([1, 2, 3, 4], ["a", "b", "c"]))
def test_batch_list_init_data(data: list) -> None:
    assert BatchList(data).data == data


def test_batch_list_init_data_incorrect_type() -> None:
    with raises(TypeError, match="Incorrect type. Expect a list but received"):
        BatchList((1, 2, 3, 4))


def test_batch_list_str() -> None:
    assert str(BatchList(["a", "b", "c"])).startswith("BatchList(")


@mark.parametrize("batch_size", (1, 2))
def test_batch_list_batch_size(batch_size: int) -> None:
    assert BatchList(["a"] * batch_size).batch_size == batch_size


###############################
#     Creation operations     #
###############################


def test_batch_list_clone() -> None:
    batch = BatchList(["a", "b", "c"])
    clone = batch.clone()
    batch.data[1] = "d"
    assert batch.equal(BatchList(["a", "d", "c"]))
    assert clone.equal(BatchList(["a", "b", "c"]))


#################################
#     Comparison operations     #
#################################


def test_batch_list_equal_true() -> None:
    assert BatchList(["a", "b", "c"]).equal(BatchList(["a", "b", "c"]))


def test_batch_list_equal_false_different_type() -> None:
    assert not BatchList(["a", "b", "c"]).equal(["a", "b", "c"])


def test_batch_list_equal_false_different_data() -> None:
    assert not BatchList(["a", "b", "c"]).equal(BatchList(["a", "b", "c", "d"]))
