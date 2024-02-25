from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from coola import objects_are_equal
from torch.utils.data.datapipes.iter import IterableWrapper

from redcat import BatchDict, BatchedTensor
from redcat.datapipes.iter import MiniBatcher

#################################
#     Tests for MiniBatcher     #
#################################


def test_mini_batcher_str() -> None:
    assert str(MiniBatcher(IterableWrapper([]), batch_size=2)).startswith(
        "MiniBatcherIterDataPipe("
    )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_mini_batcher_batch_size(batch_size: int) -> None:
    assert MiniBatcher(IterableWrapper([]), batch_size=batch_size).batch_size == batch_size


@pytest.mark.parametrize("random_seed", [1, 2])
def test_mini_batcher_random_seed(random_seed: int) -> None:
    assert (
        MiniBatcher(
            BatchedTensor(torch.ones(2, 5)), batch_size=2, random_seed=random_seed
        ).random_seed
        == random_seed
    )


def test_mini_batcher_iter_datapipe() -> None:
    assert objects_are_equal(
        list(
            MiniBatcher(
                IterableWrapper([BatchedTensor(torch.arange(5).add(i * 4)) for i in range(2)]),
                batch_size=2,
            )
        ),
        [
            BatchedTensor(torch.tensor([0, 1])),
            BatchedTensor(torch.tensor([2, 3])),
            BatchedTensor(torch.tensor([4])),
            BatchedTensor(torch.tensor([4, 5])),
            BatchedTensor(torch.tensor([6, 7])),
            BatchedTensor(torch.tensor([8])),
        ],
    )


def test_mini_batcher_iter_datapipe_iter_drop_last_true() -> None:
    assert objects_are_equal(
        list(
            MiniBatcher(
                IterableWrapper([BatchedTensor(torch.arange(5).add(i * 4)) for i in range(2)]),
                batch_size=2,
                drop_last=True,
            )
        ),
        [
            BatchedTensor(torch.tensor([0, 1])),
            BatchedTensor(torch.tensor([2, 3])),
            BatchedTensor(torch.tensor([4, 5])),
            BatchedTensor(torch.tensor([6, 7])),
        ],
    )


def test_mini_batcher_iter_batch_size_2() -> None:
    datapipe = MiniBatcher(
        BatchDict(
            {
                "key1": BatchedTensor(torch.arange(10).view(5, 2)),
                "key2": BatchedTensor(torch.arange(5)),
            }
        ),
        batch_size=2,
    )
    assert objects_are_equal(
        list(datapipe),
        [
            BatchDict(
                {
                    "key1": BatchedTensor(torch.tensor([[0, 1], [2, 3]])),
                    "key2": BatchedTensor(torch.tensor([0, 1])),
                }
            ),
            BatchDict(
                {
                    "key1": BatchedTensor(torch.tensor([[4, 5], [6, 7]])),
                    "key2": BatchedTensor(torch.tensor([2, 3])),
                }
            ),
            BatchDict(
                {
                    "key1": BatchedTensor(torch.tensor([[8, 9]])),
                    "key2": BatchedTensor(torch.tensor([4])),
                }
            ),
        ],
    )


def test_mini_batcher_iter_batch_size_4() -> None:
    datapipe = MiniBatcher(
        BatchDict(
            {
                "key1": BatchedTensor(torch.arange(20).view(10, 2)),
                "key2": BatchedTensor(torch.arange(10)),
            }
        ),
        batch_size=4,
    )
    assert objects_are_equal(
        list(datapipe),
        [
            BatchDict(
                {
                    "key1": BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])),
                    "key2": BatchedTensor(torch.tensor([0, 1, 2, 3])),
                }
            ),
            BatchDict(
                {
                    "key1": BatchedTensor(torch.tensor([[8, 9], [10, 11], [12, 13], [14, 15]])),
                    "key2": BatchedTensor(torch.tensor([4, 5, 6, 7])),
                }
            ),
            BatchDict(
                {
                    "key1": BatchedTensor(torch.tensor([[16, 17], [18, 19]])),
                    "key2": BatchedTensor(torch.tensor([8, 9])),
                }
            ),
        ],
    )


def test_mini_batcher_iter_drop_last_true_batch_size_2() -> None:
    datapipe = MiniBatcher(
        BatchDict(
            {
                "key1": BatchedTensor(torch.arange(10).view(5, 2)),
                "key2": BatchedTensor(torch.arange(5)),
            }
        ),
        batch_size=2,
        drop_last=True,
    )
    assert objects_are_equal(
        list(datapipe),
        [
            BatchDict(
                {
                    "key1": BatchedTensor(torch.tensor([[0, 1], [2, 3]])),
                    "key2": BatchedTensor(torch.tensor([0, 1])),
                }
            ),
            BatchDict(
                {
                    "key1": BatchedTensor(torch.tensor([[4, 5], [6, 7]])),
                    "key2": BatchedTensor(torch.tensor([2, 3])),
                }
            ),
        ],
    )


def test_mini_batcher_iter_drop_last_true_batch_size_4() -> None:
    datapipe = MiniBatcher(
        BatchDict(
            {
                "key1": BatchedTensor(torch.arange(20).view(10, 2)),
                "key2": BatchedTensor(torch.arange(10)),
            }
        ),
        batch_size=4,
        drop_last=True,
    )
    assert objects_are_equal(
        list(datapipe),
        [
            BatchDict(
                {
                    "key1": BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])),
                    "key2": BatchedTensor(torch.tensor([0, 1, 2, 3])),
                }
            ),
            BatchDict(
                {
                    "key1": BatchedTensor(torch.tensor([[8, 9], [10, 11], [12, 13], [14, 15]])),
                    "key2": BatchedTensor(torch.tensor([4, 5, 6, 7])),
                }
            ),
        ],
    )


@patch(
    "redcat.tensor.torch.randperm",
    lambda *args, **kwargs: torch.tensor([5, 4, 6, 3, 7, 2, 8, 1, 9, 0]),
)
def test_mini_batcher_iter_shuffle_true() -> None:
    assert objects_are_equal(
        list(
            MiniBatcher(
                BatchedTensor(torch.arange(10, dtype=torch.long)),
                batch_size=4,
                shuffle=True,
            )
        ),
        [
            BatchedTensor(torch.tensor([5, 4, 6, 3], dtype=torch.long)),
            BatchedTensor(torch.tensor([7, 2, 8, 1], dtype=torch.long)),
            BatchedTensor(torch.tensor([9, 0], dtype=torch.long)),
        ],
    )


def test_mini_batcher_iter_same_random_seed() -> None:
    assert objects_are_equal(
        tuple(
            MiniBatcher(
                BatchDict(
                    {
                        "key1": BatchedTensor(torch.arange(20).view(10, 2)),
                        "key2": BatchedTensor(torch.arange(10)),
                    }
                ),
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
        tuple(
            MiniBatcher(
                BatchDict(
                    {
                        "key1": BatchedTensor(torch.arange(20).view(10, 2)),
                        "key2": BatchedTensor(torch.arange(10)),
                    }
                ),
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
    )


def test_mini_batcher_iter_different_random_seeds() -> None:
    assert not objects_are_equal(
        tuple(
            MiniBatcher(
                BatchDict(
                    {
                        "key1": BatchedTensor(torch.arange(20).view(10, 2)),
                        "key2": BatchedTensor(torch.arange(10)),
                    }
                ),
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
        tuple(
            MiniBatcher(
                BatchDict(
                    {
                        "key1": BatchedTensor(torch.arange(20).view(10, 2)),
                        "key2": BatchedTensor(torch.arange(10)),
                    }
                ),
                batch_size=4,
                shuffle=True,
                random_seed=2,
            )
        ),
    )


def test_mini_batcher_iter_repeat() -> None:
    datapipe = MiniBatcher(
        BatchDict(
            {
                "key1": BatchedTensor(torch.arange(20).view(10, 2)),
                "key2": BatchedTensor(torch.arange(10)),
            }
        ),
        batch_size=4,
        shuffle=True,
        random_seed=1,
    )
    assert not objects_are_equal(tuple(datapipe), tuple(datapipe))


def test_mini_batcher_len() -> None:
    with pytest.raises(
        TypeError, match="MiniBatcherIterDataPipe instance doesn't have valid length"
    ):
        len(MiniBatcher(IterableWrapper([]), batch_size=4))


def test_mini_batcher_len_batch_batch_size_2() -> None:
    assert len(MiniBatcher(BatchedTensor(torch.ones(10, 2)), batch_size=2)) == 5


def test_mini_batcher_len_batch_batch_size_4() -> None:
    assert len(MiniBatcher(BatchedTensor(torch.ones(10, 2)), batch_size=4)) == 3


def test_mini_batcher_len_batch_drop_last_true_batch_size_2() -> None:
    assert len(MiniBatcher(BatchedTensor(torch.ones(10, 2)), batch_size=2, drop_last=True)) == 5


def test_mini_batcher_len_batch_drop_last_true_batch_size_4() -> None:
    assert len(MiniBatcher(BatchedTensor(torch.ones(10, 2)), batch_size=4, drop_last=True)) == 2
