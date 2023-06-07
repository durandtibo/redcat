from __future__ import annotations

from unittest.mock import Mock

import torch
from coola import objects_are_equal
from pytest import mark, raises
from torch.utils.data.datapipes.iter import IterableWrapper

from redcat import BatchedTensor
from redcat.datapipes.iter import BatchExtender
from redcat.datapipes.iter.joining import create_large_batch

###################################
#     Tests for BatchExtender     #
###################################


def test_batch_extender_str() -> None:
    assert str(BatchExtender(IterableWrapper([]))).startswith("BatchExtenderIterDataPipe(")


@mark.parametrize("buffer_size", (1, 2, 3))
def test_batch_extender_buffer_size(buffer_size: int) -> None:
    assert BatchExtender(IterableWrapper([]), buffer_size=buffer_size)._buffer_size == buffer_size


@mark.parametrize("buffer_size", (0, -1))
def test_batch_extender_incorrect_buffer_size(buffer_size: int) -> None:
    with raises(ValueError, match="buffer_size should be greater or equal to 1"):
        BatchExtender(IterableWrapper([]), buffer_size=buffer_size)


def test_batch_extender_iter() -> None:
    datapipe = BatchExtender(
        IterableWrapper([BatchedTensor(torch.full((2,), i, dtype=torch.float)) for i in range(10)]),
        buffer_size=4,
    )
    assert objects_are_equal(
        tuple(datapipe),
        (
            BatchedTensor(torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.float)),
            BatchedTensor(torch.tensor([4, 4, 5, 5, 6, 6, 7, 7], dtype=torch.float)),
            BatchedTensor(torch.tensor([8, 8, 9, 9], dtype=torch.float)),
        ),
    )


def test_batch_extender_iter_drop_last() -> None:
    datapipe = BatchExtender(
        IterableWrapper([BatchedTensor(torch.full((2,), i, dtype=torch.float)) for i in range(10)]),
        buffer_size=4,
        drop_last=True,
    )
    assert objects_are_equal(
        tuple(datapipe),
        (
            BatchedTensor(torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.float)),
            BatchedTensor(torch.tensor([4, 4, 5, 5, 6, 6, 7, 7], dtype=torch.float)),
        ),
    )


@mark.parametrize(
    "num_samples,buffer_size,length",
    (
        (1, 1, 1),
        (5, 10, 0),
        (10, 10, 1),
        (11, 10, 1),
        (20, 10, 2),
        (21, 10, 2),
    ),
)
def test_batch_extender_len_drop_last_true(num_samples: int, buffer_size: int, length: int) -> None:
    assert (
        len(
            BatchExtender(
                Mock(__len__=Mock(return_value=num_samples)),
                buffer_size=buffer_size,
                drop_last=True,
            )
        )
        == length
    )


@mark.parametrize(
    "num_samples,buffer_size,length",
    (
        (1, 1, 1),
        (5, 10, 1),
        (10, 10, 1),
        (11, 10, 2),
        (20, 10, 2),
        (21, 10, 3),
    ),
)
def test_batch_extender_len_drop_last_false(
    num_samples: int, buffer_size: int, length: int
) -> None:
    assert (
        len(BatchExtender(Mock(__len__=Mock(return_value=num_samples)), buffer_size=buffer_size))
        == length
    )


def test_batch_extender_no_len() -> None:
    datapipe = IterableWrapper({"key": i} for i in range(5))
    with raises(TypeError, match="object of type .* has no len()"):
        len(BatchExtender(datapipe))


########################################
#     Tests for create_large_batch     #
########################################


def test_create_large_batch() -> None:
    batches = [
        BatchedTensor(torch.ones(2)),
        BatchedTensor(torch.full((2,), 2.0, dtype=torch.float)),
        BatchedTensor(torch.full((2,), 3.0, dtype=torch.float)),
    ]
    batch = create_large_batch(batches)
    assert batch.equal(BatchedTensor(torch.tensor([1, 1, 2, 2, 3, 3], dtype=torch.float)))
    assert objects_are_equal(
        batches,
        [
            BatchedTensor(torch.ones(2)),
            BatchedTensor(torch.full((2,), 2.0, dtype=torch.float)),
            BatchedTensor(torch.full((2,), 3.0, dtype=torch.float)),
        ],
    )
