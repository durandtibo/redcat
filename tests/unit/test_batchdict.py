from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import torch
from coola import objects_are_equal
from torch import Tensor

from redcat import BaseBatch, BatchDict, BatchedTensorSeq, BatchList
from redcat.batchdict import check_same_batch_size, check_same_keys, get_seq_lens
from redcat.utils.tensor import get_torch_generator

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


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
    with pytest.raises(
        RuntimeError,
        match="Incorrect batch size. A single batch size is expected but received several values:",
    ):
        BatchDict({"key1": BatchList([1, 2, 3, 4]), "key2": BatchList(["a", "b", "c"])})


def test_batch_dict_init_data_empty() -> None:
    with pytest.raises(RuntimeError, match="The dictionary cannot be empty"):
        BatchDict({})


def test_batch_dict_init_data_incorrect_type() -> None:
    with pytest.raises(TypeError, match="Incorrect type. Expect a dict but received"):
        BatchDict(BatchList([1, 2, 3, 4]))


def test_batch_dict_str() -> None:
    assert str(BatchDict({"key": BatchList([1, 2, 3, 4])})).startswith("BatchDict(")


@pytest.mark.parametrize("batch_size", [1, 2])
def test_batch_dict_batch_size(batch_size: int) -> None:
    assert (
        BatchDict(
            {"key1": BatchList([1] * batch_size), "key2": BatchList(["a"] * batch_size)}
        ).batch_size
        == batch_size
    )


#################################
#     Dictionary operations     #
#################################


def test_batch_dict__contains__true() -> None:
    assert "key1" in BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})


def test_batch_dict__contains__false() -> None:
    assert "key2" not in BatchDict({"key1": BatchList([1, 2, 3])})


def test_batch_dict__getitem__() -> None:
    assert BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})[
        "key2"
    ].allequal(BatchList(["a", "b", "c"]))


def test_batch_dict__getitem__missing_key() -> None:
    with pytest.raises(KeyError):
        BatchDict({"key1": BatchList([1, 2, 3])})["key2"]


def test_batch_dict__iter__1() -> None:
    assert list(BatchDict({"key1": BatchList([1, 2, 3])})) == ["key1"]


def test_batch_dict__iter__2() -> None:
    assert list(BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})) == [
        "key1",
        "key2",
    ]


def test_batch_dict__len__1() -> None:
    assert len(BatchDict({"key1": BatchList([1, 2, 3])})) == 1


def test_batch_dict__len__2() -> None:
    assert len(BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})) == 2


def test_batch_dict__setitem__update_value() -> None:
    batch = BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})
    batch["key2"] = BatchList(["d", "e", "f"])
    assert batch.allequal(
        BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["d", "e", "f"])})
    )


def test_batch_dict__setitem__new_key() -> None:
    batch = BatchDict({"key1": BatchList([1, 2, 3])})
    batch["key2"] = BatchList(["a", "b", "c"])
    assert batch.allequal(
        BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})
    )


def test_batch_dict__setitem__incorrect_batch_size() -> None:
    batch = BatchDict({"key1": BatchList([1, 2, 3])})
    with pytest.raises(RuntimeError, match="Incorrect batch size."):
        batch["key2"] = BatchList(["a", "b", "c", "d"])


def test_batch_dict_get() -> None:
    assert (
        BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})
        .get("key2")
        .allequal(BatchList(["a", "b", "c"]))
    )


def test_batch_dict_get_missing_key() -> None:
    assert BatchDict({"key1": BatchList([1, 2, 3])}).get("key2") is None


def test_batch_dict_get_missing_key_custom_default() -> None:
    assert (
        BatchDict({"key1": BatchList([1, 2, 3])})
        .get("key2", default=BatchList(["a", "b", "c"]))
        .allequal(BatchList(["a", "b", "c"]))
    )


def test_batch_dict_items() -> None:
    assert objects_are_equal(
        list(
            BatchDict(
                {"key1": BatchList([1, 2, 3, 4]), "key2": BatchList(["a", "b", "c", "d"])}
            ).items()
        ),
        [("key1", BatchList([1, 2, 3, 4])), ("key2", BatchList(["a", "b", "c", "d"]))],
    )


def test_batch_dict_keys() -> None:
    assert list(
        BatchDict({"key1": BatchList([1, 2, 3, 4]), "key2": BatchList(["a", "b", "c", "d"])}).keys()
    ) == ["key1", "key2"]


def test_batch_dict_values() -> None:
    assert objects_are_equal(
        list(
            BatchDict(
                {"key1": BatchList([1, 2, 3, 4]), "key2": BatchList(["a", "b", "c", "d"])}
            ).values()
        ),
        [BatchList([1, 2, 3, 4]), BatchList(["a", "b", "c", "d"])],
    )


#################################
#     Conversion operations     #
#################################


def test_batch_dict_to_data() -> None:
    assert objects_are_equal(
        BatchDict(
            {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
        ).to_data(),
        {"key1": torch.arange(10).view(2, 5), "key2": ["a", "b"]},
    )


###############################
#     Creation operations     #
###############################


def test_batch_dict_clone() -> None:
    batch = BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})
    clone = batch.clone()
    batch["key2"][1] = "d"
    assert batch.allequal(
        BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "d", "c"])})
    )
    assert clone.allequal(
        BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})
    )


#################################
#     Comparison operations     #
#################################


def test_batch_dict_allclose_true() -> None:
    assert BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])}).allclose(
        BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})
    )


def test_batch_dict_allclose_false_different_type() -> None:
    assert not BatchDict(
        {"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])}
    ).allclose(["a", "b", "c"])


def test_batch_dict_allclose_false_different_data() -> None:
    assert not BatchDict({"key": BatchList(["a", "b", "c"])}).allclose(
        BatchDict({"key": BatchList(["a", "b", "c", "d"])})
    )


@pytest.mark.parametrize(
    ("batch", "atol"),
    [
        (BatchDict({"key": BatchList([0.5, 1.5, 2.5, 3.5])}), 1.0),
        (BatchDict({"key": BatchList([0.05, 1.05, 2.05, 3.05])}), 1e-1),
        (BatchDict({"key": BatchList([0.005, 1.005, 2.005, 3.005])}), 1e-2),
    ],
)
def test_batch_dict_allclose_true_atol(batch: BatchList, atol: float) -> None:
    assert BatchDict({"key": BatchList([0.0, 1.0, 2.0, 3.0])}).allclose(batch, atol=atol, rtol=0)


@pytest.mark.parametrize(
    ("batch", "rtol"),
    [
        (BatchDict({"key": BatchList([1.5, 2.5, 3.5])}), 1.0),
        (BatchDict({"key": BatchList([1.05, 2.05, 3.05])}), 1e-1),
        (BatchDict({"key": BatchList([1.005, 2.005, 3.005])}), 1e-2),
    ],
)
def test_batch_dict_allclose_true_rtol(batch: BatchList, rtol: float) -> None:
    assert BatchDict({"key": BatchList([1.0, 2.0, 3.0])}).allclose(batch, rtol=rtol)


def test_batch_dict_allequal_true() -> None:
    assert BatchDict({"key": BatchList(["a", "b", "c"])}).allequal(
        BatchDict({"key": BatchList(["a", "b", "c"])})
    )


def test_batch_dict_allequal_false_different_type() -> None:
    assert not BatchDict({"key": BatchList(["a", "b", "c"])}).allequal(["a", "b", "c"])


def test_batch_dict_allequal_false_different_data() -> None:
    assert not BatchDict({"key": BatchList(["a", "b", "c"])}).allequal(
        BatchList(["a", "b", "c", "d"])
    )


###########################################################
#     Mathematical | advanced arithmetical operations     #
###########################################################


@pytest.mark.parametrize("permutation", [torch.tensor([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)])
def test_batch_dict_permute_along_batch(permutation: Sequence[int] | Tensor) -> None:
    assert (
        BatchDict({"key1": BatchList([1, 2, 3, 4]), "key2": BatchList(["a", "b", "c", "d"])})
        .permute_along_batch(permutation)
        .allequal(
            BatchDict({"key1": BatchList([3, 2, 4, 1]), "key2": BatchList(["c", "b", "d", "a"])})
        )
    )


@pytest.mark.parametrize("permutation", [torch.tensor([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)])
def test_batch_dict_permute_along_batch_(permutation: Sequence[int] | Tensor) -> None:
    batch = BatchDict({"key1": BatchList([1, 2, 3, 4]), "key2": BatchList(["a", "b", "c", "d"])})
    batch.permute_along_batch_(permutation)
    assert batch.allequal(
        BatchDict({"key1": BatchList([3, 2, 4, 1]), "key2": BatchList(["c", "b", "d", "a"])})
    )


@pytest.mark.parametrize(
    "permutation", [torch.tensor([2, 4, 1, 3, 0]), [2, 4, 1, 3, 0], (2, 4, 1, 3, 0)]
)
def test_batch_dict_permute_along_seq(permutation: Sequence[int] | Tensor) -> None:
    assert (
        BatchDict(
            {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
        )
        .permute_along_seq(permutation)
        .allequal(
            BatchDict(
                {
                    "key1": BatchedTensorSeq(torch.tensor([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])),
                    "key2": BatchList(["a", "b"]),
                }
            )
        )
    )


def test_batch_dict_permute_along_seq_no_seq() -> None:
    assert (
        BatchDict({"key": BatchList(["a", "b"])})
        .permute_along_seq((2, 4, 1, 3, 0))
        .allequal(BatchDict({"key": BatchList(["a", "b"])}))
    )


@pytest.mark.parametrize(
    "permutation", [torch.tensor([2, 4, 1, 3, 0]), [2, 4, 1, 3, 0], (2, 4, 1, 3, 0)]
)
def test_batch_dict_permute_along_seq_(permutation: Sequence[int] | Tensor) -> None:
    batch = BatchDict(
        {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
    )
    batch.permute_along_seq_(permutation)
    assert batch.allequal(
        BatchDict(
            {
                "key1": BatchedTensorSeq(torch.tensor([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])),
                "key2": BatchList(["a", "b"]),
            }
        )
    )


def test_batch_dict_permute_along_seq__no_seq() -> None:
    batch = BatchDict({"key": BatchList(["a", "b"])})
    batch.permute_along_seq_((2, 4, 1, 3, 0))
    assert batch.allequal(BatchDict({"key": BatchList(["a", "b"])}))


@patch(
    "redcat.base.torch.randperm",
    lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]),  # noqa: ARG005
)
def test_batch_dict_shuffle_along_batch() -> None:
    assert (
        BatchDict({"key1": BatchList([1, 2, 3, 4]), "key2": BatchList(["a", "b", "c", "d"])})
        .shuffle_along_batch()
        .allequal(
            BatchDict({"key1": BatchList([3, 2, 4, 1]), "key2": BatchList(["c", "b", "d", "a"])})
        )
    )


def test_batch_dict_shuffle_along_batch_same_random_seed() -> None:
    batch = BatchDict(
        {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
    )
    assert batch.shuffle_along_batch(get_torch_generator(1)).allequal(
        batch.shuffle_along_batch(get_torch_generator(1))
    )


def test_batch_dict_shuffle_along_batch_different_random_seeds() -> None:
    batch = BatchDict(
        {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
    )
    assert not batch.shuffle_along_batch(get_torch_generator(1)).allequal(
        batch.shuffle_along_batch(get_torch_generator(2))
    )


def test_batch_dict_shuffle_along_batch_multiple_shuffle() -> None:
    batch = BatchDict(
        {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
    )
    generator = get_torch_generator(1)
    assert not batch.shuffle_along_batch(generator).allequal(batch.shuffle_along_batch(generator))


@patch(
    "redcat.base.torch.randperm",
    lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]),  # noqa: ARG005
)
def test_batch_dict_shuffle_along_batch_() -> None:
    batch = BatchDict({"key1": BatchList([1, 2, 3, 4]), "key2": BatchList(["a", "b", "c", "d"])})
    batch.shuffle_along_batch_()
    assert batch.allequal(
        BatchDict({"key1": BatchList([3, 2, 4, 1]), "key2": BatchList(["c", "b", "d", "a"])})
    )


def test_batch_dict_shuffle_along_batch__same_random_seed() -> None:
    batch1 = BatchDict(
        {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
    )
    batch1.shuffle_along_batch_(get_torch_generator(1))
    batch2 = BatchDict(
        {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
    )
    batch2.shuffle_along_batch_(get_torch_generator(1))
    assert batch1.allequal(batch2)


def test_batch_dict_shuffle_along_batch__different_random_seeds() -> None:
    batch1 = BatchDict(
        {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
    )
    batch1.shuffle_along_batch_(get_torch_generator(1))
    batch2 = BatchDict(
        {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
    )
    batch2.shuffle_along_batch_(get_torch_generator(2))
    assert not batch1.allequal(batch2)


@patch(
    "redcat.base.torch.randperm",
    lambda *args, **kwargs: torch.tensor([2, 4, 1, 3, 0]),  # noqa: ARG005
)
def test_batch_dict_shuffle_along_seq_mix() -> None:
    assert (
        BatchDict(
            {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
        )
        .shuffle_along_seq()
        .allequal(
            BatchDict(
                {
                    "key1": BatchedTensorSeq(torch.tensor([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])),
                    "key2": BatchList(["a", "b"]),
                }
            )
        )
    )


@patch(
    "redcat.base.torch.randperm",
    lambda *args, **kwargs: torch.tensor([2, 4, 1, 3, 0]),  # noqa: ARG005
)
def test_batch_dict_shuffle_along_seq_no_seq() -> None:
    assert (
        BatchDict({"key1": BatchList([1, 2]), "key2": BatchList(["a", "b"])})
        .shuffle_along_seq()
        .allequal(BatchDict({"key1": BatchList([1, 2]), "key2": BatchList(["a", "b"])}))
    )


def test_batch_dict_shuffle_along_seq_multiple_sequence_lengths() -> None:
    batch = BatchDict(
        {
            "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            "key2": BatchedTensorSeq(torch.ones(2, 3)),
        }
    )
    with pytest.raises(
        RuntimeError, match="Invalid operation because the batch has multiple sequence lengths"
    ):
        batch.shuffle_along_seq()


def test_batch_dict_shuffle_along_seq_same_random_seed() -> None:
    batch = BatchDict(
        {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
    )
    assert batch.shuffle_along_seq(get_torch_generator(1)).allequal(
        batch.shuffle_along_seq(get_torch_generator(1))
    )


def test_batch_dict_shuffle_along_seq_different_random_seeds() -> None:
    batch = BatchDict(
        {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
    )
    assert not batch.shuffle_along_seq(get_torch_generator(1)).allequal(
        batch.shuffle_along_seq(get_torch_generator(2))
    )


def test_batch_dict_shuffle_along_seq_multiple_shuffle() -> None:
    batch = BatchDict(
        {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
    )
    generator = get_torch_generator(1)
    assert not batch.shuffle_along_seq(generator).allequal(batch.shuffle_along_seq(generator))


@patch(
    "redcat.base.torch.randperm",
    lambda *args, **kwargs: torch.tensor([2, 4, 1, 3, 0]),  # noqa: ARG005
)
def test_batch_dict_shuffle_along_seq__mix() -> None:
    batch = BatchDict(
        {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
    )
    batch.shuffle_along_seq_()
    assert batch.allequal(
        BatchDict(
            {
                "key1": BatchedTensorSeq(torch.tensor([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])),
                "key2": BatchList(["a", "b"]),
            }
        )
    )


@patch(
    "redcat.base.torch.randperm",
    lambda *args, **kwargs: torch.tensor([2, 4, 1, 3, 0]),  # noqa: ARG005
)
def test_batch_dict_shuffle_along_seq__no_seq() -> None:
    batch = BatchDict({"key1": BatchList([1, 2]), "key2": BatchList(["a", "b"])})
    batch.shuffle_along_seq_()
    assert batch.allequal(BatchDict({"key1": BatchList([1, 2]), "key2": BatchList(["a", "b"])}))


def test_batch_dict_shuffle_along_seq__multiple_sequence_lengths() -> None:
    batch = BatchDict(
        {
            "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            "key2": BatchedTensorSeq(torch.ones(2, 3)),
        }
    )
    with pytest.raises(
        RuntimeError, match="Invalid operation because the batch has multiple sequence lengths"
    ):
        batch.shuffle_along_seq_()


def test_batch_dict_shuffle_along_seq__same_random_seed() -> None:
    batch1 = BatchDict(
        {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
    )
    batch1.shuffle_along_seq_(get_torch_generator(1))
    batch2 = BatchDict(
        {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
    )
    batch2.shuffle_along_seq_(get_torch_generator(1))
    assert batch1.allequal(batch2)


def test_batch_dict_shuffle_along_seq__different_random_seeds() -> None:
    batch1 = BatchDict(
        {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
    )
    batch1.shuffle_along_seq_(get_torch_generator(1))
    batch2 = BatchDict(
        {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
    )
    batch2.shuffle_along_seq_(get_torch_generator(2))
    assert not batch1.allequal(batch2)


################################################
#     Mathematical | point-wise operations     #
################################################

###########################################
#     Mathematical | trigo operations     #
###########################################

##########################################################
#    Indexing, slicing, joining, mutating operations     #
##########################################################


def test_batch_dict_append_1_item() -> None:
    batch = BatchDict({"key1": BatchList([1, 2, 3])})
    batch.append(BatchDict({"key1": BatchList(["a", "b"])}))
    assert batch.allequal(BatchDict({"key1": BatchList([1, 2, 3, "a", "b"])}))


def test_batch_dict_append_2_items() -> None:
    batch = BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})
    batch.append(BatchDict({"key1": BatchList([4, 5]), "key2": BatchList(["d", "e"])}))
    assert batch.allequal(
        BatchDict(
            {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
        )
    )


def test_batch_dict_append_missing_key() -> None:
    batch = BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})
    with pytest.raises(RuntimeError, match="Keys do not match"):
        batch.append(BatchDict({"key2": BatchList(["a", "b"])}))


@pytest.mark.parametrize(
    "other",
    [
        BatchDict({"key1": BatchedTensorSeq(torch.tensor([[10, 11, 12], [20, 21, 22]]))}),
        BatchDict(
            {
                "key1": BatchedTensorSeq(torch.tensor([[10, 11, 12], [20, 21, 22]])),
                "key2": BatchList(["a", "b"]),
            }
        ),
        [BatchDict({"key1": BatchedTensorSeq(torch.tensor([[10, 11, 12], [20, 21, 22]]))})],
        (BatchDict({"key1": BatchedTensorSeq(torch.tensor([[10, 11, 12], [20, 21, 22]]))}),),
        [
            BatchDict({"key1": BatchedTensorSeq(torch.tensor([[10, 11], [20, 21]]))}),
            BatchDict({"key1": BatchedTensorSeq(torch.tensor([[12], [22]]))}),
        ],
    ],
)
def test_batched_tensor_seq_cat_along_seq(other: BatchDict | Sequence[BatchDict]) -> None:
    assert (
        BatchDict(
            {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
        )
        .cat_along_seq(other)
        .allequal(
            BatchDict(
                {
                    "key1": BatchedTensorSeq(
                        torch.tensor([[0, 1, 2, 3, 4, 10, 11, 12], [5, 6, 7, 8, 9, 20, 21, 22]])
                    ),
                    "key2": BatchList(["a", "b"]),
                }
            )
        )
    )


def test_batch_dict_cat_along_seq_empty() -> None:
    assert (
        BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})
        .cat_along_seq([])
        .allequal(BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])}))
    )


@pytest.mark.parametrize(
    "other",
    [
        BatchDict({"key1": BatchedTensorSeq(torch.tensor([[10, 11, 12], [20, 21, 22]]))}),
        BatchDict(
            {
                "key1": BatchedTensorSeq(torch.tensor([[10, 11, 12], [20, 21, 22]])),
                "key2": BatchList(["a", "b"]),
            }
        ),
        [BatchDict({"key1": BatchedTensorSeq(torch.tensor([[10, 11, 12], [20, 21, 22]]))})],
        (BatchDict({"key1": BatchedTensorSeq(torch.tensor([[10, 11, 12], [20, 21, 22]]))}),),
        [
            BatchDict({"key1": BatchedTensorSeq(torch.tensor([[10, 11], [20, 21]]))}),
            BatchDict({"key1": BatchedTensorSeq(torch.tensor([[12], [22]]))}),
        ],
    ],
)
def test_batched_tensor_seq_cat_along_seq_(other: BatchDict | Sequence[BatchDict]) -> None:
    batch = BatchDict(
        {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
    )
    batch.cat_along_seq_(other)
    assert batch.allequal(
        BatchDict(
            {
                "key1": BatchedTensorSeq(
                    torch.tensor([[0, 1, 2, 3, 4, 10, 11, 12], [5, 6, 7, 8, 9, 20, 21, 22]])
                ),
                "key2": BatchList(["a", "b"]),
            }
        )
    )


def test_batch_dict_cat_along_seq__empty() -> None:
    batch = BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})
    batch.cat_along_seq_([])
    assert batch.allequal(
        BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})
    )


def test_batch_dict_chunk_along_batch_5() -> None:
    assert objects_are_equal(
        BatchDict(
            {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
        ).chunk_along_batch(chunks=5),
        (
            BatchDict({"key1": BatchList([1]), "key2": BatchList(["a"])}),
            BatchDict({"key1": BatchList([2]), "key2": BatchList(["b"])}),
            BatchDict({"key1": BatchList([3]), "key2": BatchList(["c"])}),
            BatchDict({"key1": BatchList([4]), "key2": BatchList(["d"])}),
            BatchDict({"key1": BatchList([5]), "key2": BatchList(["e"])}),
        ),
    )


def test_batch_dict_chunk_along_batch_3() -> None:
    assert objects_are_equal(
        BatchDict(
            {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
        ).chunk_along_batch(chunks=3),
        (
            BatchDict({"key1": BatchList([1, 2]), "key2": BatchList(["a", "b"])}),
            BatchDict({"key1": BatchList([3, 4]), "key2": BatchList(["c", "d"])}),
            BatchDict({"key1": BatchList([5]), "key2": BatchList(["e"])}),
        ),
    )


def test_batch_dict_chunk_along_batch_1_item() -> None:
    assert objects_are_equal(
        BatchDict({"key": BatchList([1, 2, 3, 4, 5])}).chunk_along_batch(chunks=3),
        (
            BatchDict({"key": BatchList([1, 2])}),
            BatchDict({"key": BatchList([3, 4])}),
            BatchDict({"key": BatchList([5])}),
        ),
    )


def test_batch_dict_chunk_along_batch_incorrect_chunks() -> None:
    with pytest.raises(RuntimeError, match="chunks has to be greater than 0 but received"):
        BatchDict(
            {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
        ).chunk_along_batch(chunks=0)


@pytest.mark.parametrize(
    "other",
    [
        [
            BatchDict({"key1": BatchList([4]), "key2": BatchList(["d"])}),
            BatchDict({"key1": BatchList([5]), "key2": BatchList(["e"])}),
        ],
        [BatchDict({"key1": BatchList([4, 5]), "key2": BatchList(["d", "e"])})],
    ],
)
def test_batch_dict_extend(
    other: Iterable[BatchDict],
) -> None:
    batch = BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})
    batch.extend(other)
    assert batch.allequal(
        BatchDict(
            {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
        )
    )


@pytest.mark.parametrize("index", [torch.tensor([2, 0]), [2, 0], (2, 0)])
def test_batch_dict_index_select_along_batch(index: Tensor | Sequence[int]) -> None:
    assert (
        BatchDict(
            {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
        )
        .index_select_along_batch(index)
        .allequal(BatchDict({"key1": BatchList([3, 1]), "key2": BatchList(["c", "a"])}))
    )


@pytest.mark.parametrize(
    "index", [torch.tensor([2, 0]), torch.tensor([[2, 0], [2, 0]]), [2, 0], (2, 0)]
)
def test_batch_dict_index_select_along_seq(index: Tensor | Sequence[int]) -> None:
    assert (
        BatchDict(
            {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
        )
        .index_select_along_seq(index)
        .allequal(
            BatchDict(
                {
                    "key1": BatchedTensorSeq(torch.tensor([[2, 0], [7, 5]])),
                    "key2": BatchList(["a", "b"]),
                }
            )
        )
    )


def test_batch_dict_index_select_along_seq_missing() -> None:
    assert (
        BatchDict({"key1": BatchList([1, 2]), "key2": BatchList(["a", "b"])})
        .index_select_along_seq([2, 0])
        .allequal(BatchDict({"key1": BatchList([1, 2]), "key2": BatchList(["a", "b"])}))
    )


def test_batched_tensor_seq_repeat_along_seq_2() -> None:
    assert (
        BatchDict(
            {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
        )
        .repeat_along_seq(2)
        .allequal(
            BatchDict(
                {
                    "key1": BatchedTensorSeq(
                        torch.tensor(
                            [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]]
                        )
                    ),
                    "key2": BatchList(["a", "b"]),
                }
            )
        )
    )


def test_batched_tensor_seq_repeat_along_seq_3() -> None:
    assert (
        BatchDict(
            {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
        )
        .repeat_along_seq(3)
        .allequal(
            BatchDict(
                {
                    "key1": BatchedTensorSeq(
                        torch.tensor(
                            [
                                [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                                [5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9],
                            ]
                        )
                    ),
                    "key2": BatchList(["a", "b"]),
                }
            )
        )
    )


def test_batch_dict_repeat_along_seq_empty() -> None:
    assert (
        BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])})
        .repeat_along_seq(2)
        .allequal(BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])}))
    )


def test_batch_dict_select_along_batch() -> None:
    assert BatchDict(
        {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
    ).select_along_batch(2) == {"key1": 3, "key2": "c"}


def test_batch_dict_slice_along_batch() -> None:
    assert (
        BatchDict(
            {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
        )
        .slice_along_batch()
        .allequal(
            BatchDict(
                {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
            )
        )
    )


def test_batch_dict_slice_along_batch_start_2() -> None:
    assert (
        BatchDict(
            {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
        )
        .slice_along_batch(start=2)
        .allequal(BatchDict({"key1": BatchList([3, 4, 5]), "key2": BatchList(["c", "d", "e"])}))
    )


def test_batch_dict_slice_along_batch_stop_3() -> None:
    assert (
        BatchDict(
            {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
        )
        .slice_along_batch(stop=3)
        .allequal(BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])}))
    )


def test_batch_dict_slice_along_batch_stop_100() -> None:
    assert (
        BatchDict(
            {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
        )
        .slice_along_batch(stop=100)
        .allequal(
            BatchDict(
                {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
            )
        )
    )


def test_batch_dict_slice_along_batch_step_2() -> None:
    assert (
        BatchDict(
            {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
        )
        .slice_along_batch(step=2)
        .allequal(BatchDict({"key1": BatchList([1, 3, 5]), "key2": BatchList(["a", "c", "e"])}))
    )


def test_batch_dict_slice_along_batch_start_1_stop_4_step_2() -> None:
    assert (
        BatchDict(
            {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
        )
        .slice_along_batch(start=1, stop=4, step=2)
        .allequal(BatchDict({"key1": BatchList([2, 4]), "key2": BatchList(["b", "d"])}))
    )


def test_batch_dict_slice_along_seq() -> None:
    assert (
        BatchDict(
            {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
        )
        .slice_along_seq()
        .allequal(
            BatchDict(
                {
                    "key1": BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])),
                    "key2": BatchList(["a", "b"]),
                }
            )
        )
    )


def test_batch_dict_slice_along_seq_start_2() -> None:
    assert (
        BatchDict(
            {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
        )
        .slice_along_seq(start=2)
        .allequal(
            BatchDict(
                {
                    "key1": BatchedTensorSeq(torch.tensor([[2, 3, 4], [7, 8, 9]])),
                    "key2": BatchList(["a", "b"]),
                }
            )
        )
    )


def test_batch_dict_slice_along_seq_stop_3() -> None:
    assert (
        BatchDict(
            {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
        )
        .slice_along_seq(stop=3)
        .allequal(
            BatchDict(
                {
                    "key1": BatchedTensorSeq(torch.tensor([[0, 1, 2], [5, 6, 7]])),
                    "key2": BatchList(["a", "b"]),
                }
            )
        )
    )


def test_batch_dict_slice_along_seq_stop_100() -> None:
    assert (
        BatchDict(
            {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
        )
        .slice_along_seq(stop=100)
        .allequal(
            BatchDict(
                {
                    "key1": BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])),
                    "key2": BatchList(["a", "b"]),
                }
            )
        )
    )


def test_batch_dict_slice_along_seq_step_2() -> None:
    assert (
        BatchDict(
            {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
        )
        .slice_along_seq(step=2)
        .allequal(
            BatchDict(
                {
                    "key1": BatchedTensorSeq(torch.tensor([[0, 2, 4], [5, 7, 9]])),
                    "key2": BatchList(["a", "b"]),
                }
            )
        )
    )


def test_batch_dict_slice_along_seq_start_1_stop_4_step_2() -> None:
    assert (
        BatchDict(
            {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
        )
        .slice_along_seq(start=1, stop=4, step=2)
        .allequal(
            BatchDict(
                {
                    "key1": BatchedTensorSeq(torch.tensor([[1, 3], [6, 8]])),
                    "key2": BatchList(["a", "b"]),
                }
            )
        )
    )


def test_batch_dict_split_along_batch_5() -> None:
    assert objects_are_equal(
        BatchDict(
            {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
        ).split_along_batch(split_size_or_sections=1),
        (
            BatchDict({"key1": BatchList([1]), "key2": BatchList(["a"])}),
            BatchDict({"key1": BatchList([2]), "key2": BatchList(["b"])}),
            BatchDict({"key1": BatchList([3]), "key2": BatchList(["c"])}),
            BatchDict({"key1": BatchList([4]), "key2": BatchList(["d"])}),
            BatchDict({"key1": BatchList([5]), "key2": BatchList(["e"])}),
        ),
    )


def test_batch_dict_split_along_batch_3() -> None:
    assert objects_are_equal(
        BatchDict(
            {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
        ).split_along_batch(split_size_or_sections=2),
        (
            BatchDict({"key1": BatchList([1, 2]), "key2": BatchList(["a", "b"])}),
            BatchDict({"key1": BatchList([3, 4]), "key2": BatchList(["c", "d"])}),
            BatchDict({"key1": BatchList([5]), "key2": BatchList(["e"])}),
        ),
    )


def test_batch_dict_split_along_batch_1_item() -> None:
    assert objects_are_equal(
        BatchDict({"key": BatchList([1, 2, 3, 4, 5])}).split_along_batch(split_size_or_sections=2),
        (
            BatchDict({"key": BatchList([1, 2])}),
            BatchDict({"key": BatchList([3, 4])}),
            BatchDict({"key": BatchList([5])}),
        ),
    )


def test_batch_dict_split_along_batch_split_size_list() -> None:
    assert objects_are_equal(
        BatchDict(
            {
                "key1": BatchList([1, 2, 3, 4, 5, 6, 7, 8]),
                "key2": BatchList(["a", "b", "c", "d", "e", "f", "g", "h"]),
            }
        ).split_along_batch([2, 2, 3, 1]),
        (
            BatchDict({"key1": BatchList([1, 2]), "key2": BatchList(["a", "b"])}),
            BatchDict({"key1": BatchList([3, 4]), "key2": BatchList(["c", "d"])}),
            BatchDict({"key1": BatchList([5, 6, 7]), "key2": BatchList(["e", "f", "g"])}),
            BatchDict({"key1": BatchList([8]), "key2": BatchList(["h"])}),
        ),
    )


def test_batch_dict_split_along_batch_split_size_list_empty() -> None:
    assert objects_are_equal(
        BatchDict(
            {
                "key1": BatchList([1, 2, 3, 4, 5, 6, 7, 8]),
                "key2": BatchList(["a", "b", "c", "d", "e", "f", "g", "h"]),
            }
        ).split_along_batch([]),
        (),
    )


# def test_batch_dict_split_along_batch_incorrect_split_size() -> None:
#     with pytest.raises(RuntimeError, match="chunks has to be greater than 0 but received"):
#         BatchDict(
#             {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
#         ).split_along_batch(split_size_or_sections=0)


def test_batch_dict_take_along_seq() -> None:
    assert (
        BatchDict(
            {
                "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
                "key2": BatchList(["a", "b"]),
            }
        )
        .take_along_seq(torch.tensor([[3, 0, 1], [2, 3, 4]]))
        .allequal(
            BatchDict(
                {
                    "key1": BatchedTensorSeq(torch.tensor([[3, 0, 1], [7, 8, 9]])),
                    "key2": BatchList(["a", "b"]),
                }
            )
        )
    )


def test_batch_dict_take_along_seq_empty() -> None:
    assert (
        BatchDict({"key1": BatchList([1, 2]), "key2": BatchList(["a", "b"])})
        .take_along_seq(torch.tensor([[3, 0, 1], [2, 3, 4]]))
        .allequal(BatchDict({"key1": BatchList([1, 2]), "key2": BatchList(["a", "b"])}))
    )


########################
#     mini-batches     #
########################


@pytest.mark.parametrize(("batch_size", "num_minibatches"), [(1, 10), (2, 5), (3, 4), (4, 3)])
def test_batch_dict_get_num_minibatches_drop_last_false(
    batch_size: int, num_minibatches: int
) -> None:
    assert (
        BatchDict(
            {
                "key1": BatchList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "key2": BatchList(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
            }
        ).get_num_minibatches(batch_size)
        == num_minibatches
    )


@pytest.mark.parametrize(("batch_size", "num_minibatches"), [(1, 10), (2, 5), (3, 3), (4, 2)])
def test_batch_dict_get_num_minibatches_drop_last_true(
    batch_size: int, num_minibatches: int
) -> None:
    assert (
        BatchDict(
            {
                "key1": BatchList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "key2": BatchList(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
            }
        ).get_num_minibatches(batch_size, drop_last=True)
        == num_minibatches
    )


def test_batch_dict_to_minibatches_10_batch_size_2() -> None:
    assert objects_are_equal(
        tuple(
            BatchDict(
                {
                    "key1": BatchList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    "key2": BatchList(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
                }
            ).to_minibatches(batch_size=2)
        ),
        (
            BatchDict({"key1": BatchList([1, 2]), "key2": BatchList(["a", "b"])}),
            BatchDict({"key1": BatchList([3, 4]), "key2": BatchList(["c", "d"])}),
            BatchDict({"key1": BatchList([5, 6]), "key2": BatchList(["e", "f"])}),
            BatchDict({"key1": BatchList([7, 8]), "key2": BatchList(["g", "h"])}),
            BatchDict({"key1": BatchList([9, 10]), "key2": BatchList(["i", "j"])}),
        ),
    )


def test_batch_dict_to_minibatches_10_batch_size_3() -> None:
    assert objects_are_equal(
        tuple(
            BatchDict(
                {
                    "key1": BatchList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    "key2": BatchList(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
                }
            ).to_minibatches(batch_size=3)
        ),
        (
            BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])}),
            BatchDict({"key1": BatchList([4, 5, 6]), "key2": BatchList(["d", "e", "f"])}),
            BatchDict({"key1": BatchList([7, 8, 9]), "key2": BatchList(["g", "h", "i"])}),
            BatchDict({"key1": BatchList([10]), "key2": BatchList(["j"])}),
        ),
    )


def test_batch_dict_to_minibatches_10_batch_size_4() -> None:
    assert objects_are_equal(
        tuple(
            BatchDict(
                {
                    "key1": BatchList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    "key2": BatchList(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
                }
            ).to_minibatches(batch_size=4)
        ),
        (
            BatchDict({"key1": BatchList([1, 2, 3, 4]), "key2": BatchList(["a", "b", "c", "d"])}),
            BatchDict({"key1": BatchList([5, 6, 7, 8]), "key2": BatchList(["e", "f", "g", "h"])}),
            BatchDict({"key1": BatchList([9, 10]), "key2": BatchList(["i", "j"])}),
        ),
    )


def test_batch_dict_to_minibatches_drop_last_true_10_batch_size_2() -> None:
    assert objects_are_equal(
        tuple(
            BatchDict(
                {
                    "key1": BatchList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    "key2": BatchList(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
                }
            ).to_minibatches(batch_size=2, drop_last=True)
        ),
        (
            BatchDict({"key1": BatchList([1, 2]), "key2": BatchList(["a", "b"])}),
            BatchDict({"key1": BatchList([3, 4]), "key2": BatchList(["c", "d"])}),
            BatchDict({"key1": BatchList([5, 6]), "key2": BatchList(["e", "f"])}),
            BatchDict({"key1": BatchList([7, 8]), "key2": BatchList(["g", "h"])}),
            BatchDict({"key1": BatchList([9, 10]), "key2": BatchList(["i", "j"])}),
        ),
    )


def test_batch_dict_to_minibatches_drop_last_true_10_batch_size_3() -> None:
    assert objects_are_equal(
        tuple(
            BatchDict(
                {
                    "key1": BatchList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    "key2": BatchList(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
                }
            ).to_minibatches(batch_size=3, drop_last=True)
        ),
        (
            BatchDict({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])}),
            BatchDict({"key1": BatchList([4, 5, 6]), "key2": BatchList(["d", "e", "f"])}),
            BatchDict({"key1": BatchList([7, 8, 9]), "key2": BatchList(["g", "h", "i"])}),
        ),
    )


def test_batch_dict_to_minibatches_drop_last_true_10_batch_size_4() -> None:
    assert objects_are_equal(
        tuple(
            BatchDict(
                {
                    "key1": BatchList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    "key2": BatchList(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
                }
            ).to_minibatches(batch_size=4, drop_last=True)
        ),
        (
            BatchDict({"key1": BatchList([1, 2, 3, 4]), "key2": BatchList(["a", "b", "c", "d"])}),
            BatchDict({"key1": BatchList([5, 6, 7, 8]), "key2": BatchList(["e", "f", "g", "h"])}),
        ),
    )


def test_batch_dict_summary() -> None:
    assert BatchDict(
        {
            "key1": BatchList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "key2": BatchList(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
        }
    ).summary() == (
        "BatchDict(\n  (key1): BatchList(batch_size=10)\n  (key2): BatchList(batch_size=10)\n)"
    )


######################################
#     Tests for check_batch_size     #
######################################


def test_check_batch_size_1_item() -> None:
    check_same_batch_size({"key": Mock(spec=BaseBatch, batch_size=42)})
    # will fail if an exception is raised


def test_check_batch_size_2_items_same_batch_size() -> None:
    check_same_batch_size(
        {"key1": Mock(spec=BaseBatch, batch_size=42), "key2": Mock(spec=BaseBatch, batch_size=42)}
    )
    # will fail if an exception is raised


def test_check_batch_size_empty() -> None:
    with pytest.raises(RuntimeError, match="The dictionary cannot be empty"):
        check_same_batch_size({})


def test_check_batch_size_2_items_different_batch_size() -> None:
    with pytest.raises(
        RuntimeError,
        match="Incorrect batch size. A single batch size is expected but received several values:",
    ):
        check_same_batch_size(
            {
                "key1": Mock(spec=BaseBatch, batch_size=42),
                "key2": Mock(spec=BaseBatch, batch_size=4),
            }
        )


#####################################
#     Tests for check_same_keys     #
#####################################


def test_check_same_keys_same() -> None:
    check_same_keys({"key1": 1, "key2": 2}, {"key2": 20, "key1": 10})
    # will fail if an exception is raised


def test_check_same_keys_different_names() -> None:
    with pytest.raises(RuntimeError, match="Keys do not match"):
        check_same_keys({"key1": 1, "key2": 2}, {"key": 20, "key1": 10})


def test_check_same_keys_missing_key() -> None:
    with pytest.raises(RuntimeError, match="Keys do not match"):
        check_same_keys({"key1": 1, "key2": 2}, {"key1": 10})


##################################
#     Tests for get_seq_lens     #
##################################


def test_get_seq_lens_no_sequence() -> None:
    assert get_seq_lens({"key1": BatchList([1, 2, 3]), "key2": BatchList(["a", "b", "c"])}) == set()


def test_get_seq_lens_mix() -> None:
    assert get_seq_lens(
        {"key1": BatchedTensorSeq(torch.arange(10).view(2, 5)), "key2": BatchList(["a", "b"])}
    ) == {5}


def test_get_seq_lens_two_sequences_same_lengths() -> None:
    assert get_seq_lens(
        {
            "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            "key2": BatchedTensorSeq(torch.ones(2, 5)),
        }
    ) == {5}


def test_get_seq_lens_two_sequences_different_lengths() -> None:
    assert get_seq_lens(
        {
            "key1": BatchedTensorSeq(torch.arange(10).view(2, 5)),
            "key2": BatchedTensorSeq(torch.ones(2, 3)),
        }
    ) == {3, 5}


def test_get_seq_lens_empty() -> None:
    assert get_seq_lens({}) == set()
