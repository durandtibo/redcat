from __future__ import annotations

import logging

import numpy as np
import torch

from redcat import BatchDict, BatchedTensor, BatchedTensorSeq, BatchList
from redcat.ba import BatchedArray

logger = logging.getLogger(__name__)


def check_batch_list() -> None:
    logger.info("Checking BatchList...")
    assert BatchList([1, 2, 3, 4, 5]).slice_along_batch(step=2).equal(BatchList([1, 3, 5]))


def check_batch_dict() -> None:
    logger.info("Checking BatchDict...")
    assert (
        BatchDict(
            {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
        )
        .slice_along_batch()
        .equal(
            BatchDict(
                {"key1": BatchList([1, 2, 3, 4, 5]), "key2": BatchList(["a", "b", "c", "d", "e"])}
            )
        )
    )


def check_batched_array() -> None:
    logger.info("Checking BatchedArray...")
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .add(BatchedArray(np.arange(10).reshape(2, 5)))
        .equal(BatchedArray(np.arange(10).reshape(2, 5)).mul(2.0))
    )


def check_batched_tensor() -> None:
    logger.info("Checking BatchedTensor...")
    assert BatchedTensor(torch.arange(10).view(2, 5)).sum().equal(torch.tensor(45))


def check_batched_tensor_seq() -> None:
    logger.info("Checking BatchedTensorSeq...")
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5)).sum_along_seq().equal(torch.tensor([10, 35]))
    )


def main() -> None:
    check_batch_list()
    check_batch_dict()
    check_batched_array()
    check_batched_tensor()
    check_batched_tensor_seq()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
