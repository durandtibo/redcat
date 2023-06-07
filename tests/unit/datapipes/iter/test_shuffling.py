from __future__ import annotations

from unittest.mock import Mock, patch

import torch
from coola import objects_are_equal
from pytest import mark, raises
from torch.utils.data.datapipes.iter import IterableWrapper

from redcat import BatchedTensor
from redcat.datapipes.iter import BatchShuffler

###################################
#     Tests for BatchShuffler     #
###################################


def test_batch_shuffler_str() -> None:
    assert str(BatchShuffler(IterableWrapper([]))).startswith("BatchShufflerIterDataPipe(")


@mark.parametrize("random_seed", (1, 2))
def test_batch_shuffler_iter_random_seed(random_seed: int) -> None:
    assert BatchShuffler(IterableWrapper([]), random_seed=random_seed).random_seed == random_seed


@patch("redcat.tensor.torch.randperm", lambda *args, **kwargs: torch.tensor([0, 2, 1, 3]))
def test_batch_shuffler_iter() -> None:
    assert objects_are_equal(
        tuple(
            BatchShuffler(
                IterableWrapper([BatchedTensor(torch.arange(4).add(i)) for i in range(2)]),
            )
        ),
        (
            BatchedTensor(torch.tensor([0, 2, 1, 3])),
            BatchedTensor(torch.tensor([1, 3, 2, 4])),
        ),
    )


def test_batch_shuffler_iter_same_random_seed() -> None:
    source = IterableWrapper([BatchedTensor(torch.arange(5).add(i)) for i in range(3)])
    assert objects_are_equal(
        tuple(BatchShuffler(source, random_seed=1)),
        tuple(BatchShuffler(source, random_seed=1)),
    )


def test_batch_shuffler_iter_different_random_seeds() -> None:
    source = IterableWrapper([BatchedTensor(torch.arange(5).add(i)) for i in range(3)])
    assert not objects_are_equal(
        tuple(BatchShuffler(source, random_seed=1)),
        tuple(BatchShuffler(source, random_seed=2)),
    )


def test_batch_shuffler_iter_repeat() -> None:
    datapipe = BatchShuffler(
        IterableWrapper([BatchedTensor(torch.arange(5).add(i)) for i in range(3)]), random_seed=1
    )
    assert not objects_are_equal(tuple(datapipe), tuple(datapipe))


def test_batch_shuffler_len() -> None:
    assert len(BatchShuffler(Mock(__len__=Mock(return_value=5)))) == 5


def test_batch_shuffler_no_len() -> None:
    datapipe = IterableWrapper(BatchedTensor(torch.arange(5).add(i)) for i in range(3))
    with raises(TypeError, match="object of type .* has no len()"):
        len(BatchShuffler(datapipe))
