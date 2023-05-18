from __future__ import annotations

from collections.abc import Iterable, Sequence
from unittest.mock import patch

import torch
from coola import objects_are_equal
from pytest import mark, raises
from torch import Tensor

from redcat import BatchList
from redcat.utils import get_torch_generator


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
    batch[1] = "d"
    assert batch.equal(BatchList(["a", "d", "c"]))
    assert clone.equal(BatchList(["a", "b", "c"]))


#################################
#     Comparison operations     #
#################################


def test_batch_list_allclose_true() -> None:
    assert BatchList(["a", "b", "c"]).allclose(BatchList(["a", "b", "c"]))


def test_batch_list_allclose_false_different_type() -> None:
    assert not BatchList(["a", "b", "c"]).allclose(["a", "b", "c"])


def test_batch_list_allclose_false_different_data() -> None:
    assert not BatchList(["a", "b", "c"]).allclose(BatchList(["a", "b", "c", "d"]))


@mark.parametrize(
    "batch,atol",
    (
        (BatchList([0.5, 1.5, 2.5, 3.5]), 1.0),
        (BatchList([0.05, 1.05, 2.05, 3.05]), 1e-1),
        (BatchList([0.005, 1.005, 2.005, 3.005]), 1e-2),
    ),
)
def test_batch_list_allclose_true_atol(batch: BatchList, atol: float) -> None:
    assert BatchList([0.0, 1.0, 2.0, 3.0]).allclose(batch, atol=atol, rtol=0)


@mark.parametrize(
    "batch,rtol",
    (
        (BatchList([1.5, 2.5, 3.5]), 1.0),
        (BatchList([1.05, 2.05, 3.05]), 1e-1),
        (BatchList([1.005, 2.005, 3.005]), 1e-2),
    ),
)
def test_batch_list_allclose_true_rtol(batch: BatchList, rtol: float) -> None:
    assert BatchList([1.0, 2.0, 3.0]).allclose(batch, rtol=rtol)


def test_batch_list_equal_true() -> None:
    assert BatchList(["a", "b", "c"]).equal(BatchList(["a", "b", "c"]))


def test_batch_list_equal_false_different_type() -> None:
    assert not BatchList(["a", "b", "c"]).equal(["a", "b", "c"])


def test_batch_list_equal_false_different_data() -> None:
    assert not BatchList(["a", "b", "c"]).equal(BatchList(["a", "b", "c", "d"]))


###########################################################
#     Mathematical | advanced arithmetical operations     #
###########################################################


@mark.parametrize("permutation", (torch.tensor([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batch_list_permute_along_batch(permutation: Sequence[int] | Tensor) -> None:
    assert (
        BatchList(["a", "b", "c", "d"])
        .permute_along_batch(permutation)
        .equal(BatchList(["c", "b", "d", "a"]))
    )


@mark.parametrize("permutation", (torch.tensor([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batch_list_permute_along_batch_(permutation: Sequence[int] | Tensor) -> None:
    batch = BatchList(["a", "b", "c", "d"])
    batch.permute_along_batch_(permutation)
    assert batch.equal(BatchList(["c", "b", "d", "a"]))


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batch_list_shuffle_along_batch() -> None:
    assert (
        BatchList(["a", "b", "c", "d"]).shuffle_along_batch().equal(BatchList(["c", "b", "d", "a"]))
    )


def test_batch_list_shuffle_along_batch_same_random_seed() -> None:
    batch = BatchList(["a", "b", "c", "d", "e"])
    assert batch.shuffle_along_batch(get_torch_generator(1)).equal(
        batch.shuffle_along_batch(get_torch_generator(1))
    )


def test_batch_list_shuffle_along_batch_different_random_seeds() -> None:
    batch = BatchList(["a", "b", "c", "d", "e"])
    assert not batch.shuffle_along_batch(get_torch_generator(1)).equal(
        batch.shuffle_along_batch(get_torch_generator(2))
    )


def test_batch_list_shuffle_along_batch_multiple_shuffle() -> None:
    batch = BatchList(["a", "b", "c", "d", "e"])
    generator = get_torch_generator(1)
    assert not batch.shuffle_along_batch(generator).equal(batch.shuffle_along_batch(generator))


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batch_list_shuffle_along_batch_() -> None:
    batch = BatchList(["a", "b", "c", "d"])
    batch.shuffle_along_batch_()
    assert batch.equal(BatchList(["c", "b", "d", "a"]))


def test_batch_list_shuffle_along_batch__same_random_seed() -> None:
    batch1 = BatchList(["a", "b", "c", "d", "e"])
    batch1.shuffle_along_batch_(get_torch_generator(1))
    batch2 = BatchList(["a", "b", "c", "d", "e"])
    batch2.shuffle_along_batch_(get_torch_generator(1))
    assert batch1.equal(batch2)


def test_batch_list_shuffle_along_batch__different_random_seeds() -> None:
    batch1 = BatchList(["a", "b", "c", "d", "e"])
    batch1.shuffle_along_batch_(get_torch_generator(1))
    batch2 = BatchList(["a", "b", "c", "d", "e"])
    batch2.shuffle_along_batch_(get_torch_generator(2))
    assert not batch1.equal(batch2)


################################################
#     Mathematical | point-wise operations     #
################################################

###########################################
#     Mathematical | trigo operations     #
###########################################

##########################################################
#    Indexing, slicing, joining, mutating operations     #
##########################################################


def test_batch_list__getitem___int() -> None:
    batch = BatchList(["a", "b", "c", "d", "e"])
    assert batch[0] == "a"


def test_batch_list__getitem___slice() -> None:
    batch = BatchList(["a", "b", "c", "d", "e"])
    assert batch[0:2] == ["a", "b"]


def test_batch_list__setitem___int() -> None:
    batch = BatchList(["a", "b", "c", "d", "e"])
    batch[0] = 7
    assert batch.equal(BatchList([7, "b", "c", "d", "e"]))


@mark.parametrize("value", ([1, 2], (1, 2), BatchList([1, 2])))
def test_batch_list__setitem___slice(value: Sequence | BatchList) -> None:
    batch = BatchList(["a", "b", "c", "d", "e"])
    batch[1:3] = value
    assert batch.equal(BatchList(["a", 1, 2, "d", "e"]))


@mark.parametrize("other", (BatchList(["d", "e"]), ["d", "e"], ("d", "e")))
def test_batch_list_append(other: BatchList | Tensor) -> None:
    batch = BatchList(["a", "b", "c"])
    batch.append(other)
    assert batch.equal(BatchList(["a", "b", "c", "d", "e"]))


def test_batch_list_chunk_along_batch_5() -> None:
    assert objects_are_equal(
        BatchList([i for i in range(5)]).chunk_along_batch(chunks=5),
        (BatchList([0]), BatchList([1]), BatchList([2]), BatchList([3]), BatchList([4])),
    )


def test_batch_list_chunk_along_batch_3() -> None:
    assert objects_are_equal(
        BatchList([i for i in range(5)]).chunk_along_batch(3),
        (BatchList([0, 1]), BatchList([2, 3]), BatchList([4])),
    )


def test_batch_list_chunk_along_batch_incorrect_chunks() -> None:
    with raises(RuntimeError, match="chunks has to be greater than 0 but received"):
        BatchList([i for i in range(5)]).chunk_along_batch(chunks=0)


@mark.parametrize(
    "other",
    (
        [BatchList(["d"]), BatchList(["e"])],
        [BatchList(["d", "e"])],
        [BatchList(["d"]), ["e"]],
    ),
)
def test_batch_list_extend(
    other: Iterable[BatchList | list],
) -> None:
    batch = BatchList(["a", "b", "c"])
    batch.extend(other)
    assert batch.equal(BatchList(["a", "b", "c", "d", "e"]))


@mark.parametrize("index", (torch.tensor([2, 0]), [2, 0], (2, 0)))
def test_batch_list_index_select_along_batch(index: Tensor | Sequence[int]) -> None:
    assert (
        BatchList(["a", "b", "c", "d", "e"])
        .index_select_along_batch(index)
        .equal(BatchList(["c", "a"]))
    )


def test_batch_list_select_along_batch() -> None:
    assert BatchList(["a", "b", "c", "d", "e"]).select_along_batch(2) == "c"


def test_batch_list_slice_along_batch() -> None:
    assert (
        BatchList(["a", "b", "c", "d", "e"])
        .slice_along_batch()
        .equal(BatchList(["a", "b", "c", "d", "e"]))
    )


def test_batch_list_slice_along_batch_start_2() -> None:
    assert (
        BatchList(["a", "b", "c", "d", "e"])
        .slice_along_batch(start=2)
        .equal(BatchList(["c", "d", "e"]))
    )


def test_batch_list_slice_along_batch_stop_3() -> None:
    assert (
        BatchList(["a", "b", "c", "d", "e"])
        .slice_along_batch(stop=3)
        .equal(BatchList(["a", "b", "c"]))
    )


def test_batch_list_slice_along_batch_stop_100() -> None:
    assert (
        BatchList(["a", "b", "c", "d", "e"])
        .slice_along_batch(stop=100)
        .equal(BatchList(["a", "b", "c", "d", "e"]))
    )


def test_batch_list_slice_along_batch_step_2() -> None:
    assert (
        BatchList(["a", "b", "c", "d", "e"])
        .slice_along_batch(step=2)
        .equal(BatchList(["a", "c", "e"]))
    )


def test_batch_list_slice_along_batch_start_1_stop_4_step_2() -> None:
    assert (
        BatchList(["a", "b", "c", "d", "e"])
        .slice_along_batch(start=1, stop=4, step=2)
        .equal(BatchList(["b", "d"]))
    )


def test_batch_list_split_along_batch_split_size_1() -> None:
    assert objects_are_equal(
        BatchList([i for i in range(5)]).split_along_batch(1),
        (BatchList([0]), BatchList([1]), BatchList([2]), BatchList([3]), BatchList([4])),
    )


def test_batch_list_split_along_batch_split_size_2() -> None:
    assert objects_are_equal(
        BatchList([i for i in range(5)]).split_along_batch(2),
        (BatchList([0, 1]), BatchList([2, 3]), BatchList([4])),
    )


def test_batch_list_split_along_batch_split_size_list() -> None:
    assert objects_are_equal(
        BatchList([i for i in range(8)]).split_along_batch([2, 2, 3, 1]),
        (BatchList([0, 1]), BatchList([2, 3]), BatchList([4, 5, 6]), BatchList([7])),
    )


def test_batch_list_split_along_batch_split_size_list_empty() -> None:
    assert objects_are_equal(BatchList([i for i in range(8)]).split_along_batch([]), tuple())


########################
#     mini-batches     #
########################


@mark.parametrize("batch_size,num_minibatches", ((1, 10), (2, 5), (3, 4), (4, 3)))
def test_batch_list_get_num_minibatches_drop_last_false(
    batch_size: int, num_minibatches: int
) -> None:
    assert BatchList([i for i in range(10)]).get_num_minibatches(batch_size) == num_minibatches


@mark.parametrize("batch_size,num_minibatches", ((1, 10), (2, 5), (3, 3), (4, 2)))
def test_batch_list_get_num_minibatches_drop_last_true(
    batch_size: int, num_minibatches: int
) -> None:
    assert (
        BatchList([i for i in range(10)]).get_num_minibatches(batch_size, drop_last=True)
        == num_minibatches
    )


def test_batch_list_to_minibatches_10_batch_size_2() -> None:
    assert objects_are_equal(
        tuple(BatchList([i for i in range(10)]).to_minibatches(batch_size=2)),
        (
            BatchList([0, 1]),
            BatchList([2, 3]),
            BatchList([4, 5]),
            BatchList([6, 7]),
            BatchList([8, 9]),
        ),
    )


def test_batch_list_to_minibatches_10_batch_size_3() -> None:
    assert objects_are_equal(
        tuple(BatchList([i for i in range(10)]).to_minibatches(batch_size=3)),
        (
            BatchList([0, 1, 2]),
            BatchList([3, 4, 5]),
            BatchList([6, 7, 8]),
            BatchList([9]),
        ),
    )


def test_batch_list_to_minibatches_10_batch_size_4() -> None:
    assert objects_are_equal(
        tuple(BatchList([i for i in range(10)]).to_minibatches(batch_size=4)),
        (
            BatchList([0, 1, 2, 3]),
            BatchList([4, 5, 6, 7]),
            BatchList([8, 9]),
        ),
    )


def test_batch_list_to_minibatches_drop_last_true_10_batch_size_2() -> None:
    assert objects_are_equal(
        tuple(BatchList([i for i in range(10)]).to_minibatches(batch_size=2, drop_last=True)),
        (
            BatchList([0, 1]),
            BatchList([2, 3]),
            BatchList([4, 5]),
            BatchList([6, 7]),
            BatchList([8, 9]),
        ),
    )


def test_batch_list_to_minibatches_drop_last_true_10_batch_size_3() -> None:
    assert objects_are_equal(
        tuple(BatchList([i for i in range(10)]).to_minibatches(batch_size=3, drop_last=True)),
        (
            BatchList([0, 1, 2]),
            BatchList([3, 4, 5]),
            BatchList([6, 7, 8]),
        ),
    )


def test_batch_list_to_minibatches_drop_last_true_10_batch_size_4() -> None:
    assert objects_are_equal(
        tuple(BatchList([i for i in range(10)]).to_minibatches(batch_size=4, drop_last=True)),
        (
            BatchList([0, 1, 2, 3]),
            BatchList([4, 5, 6, 7]),
        ),
    )


#################
#     Other     #
#################


def test_batch_list_apply() -> None:
    assert BatchList([1, 2, 3]).apply(lambda val: val + 2).equal(BatchList([3, 4, 5]))


def test_batch_list_apply_() -> None:
    batch = BatchList([1, 2, 3])
    batch.apply_(lambda val: val + 2)
    assert batch.equal(BatchList([3, 4, 5]))
