from __future__ import annotations

import logging

from redcat import BatchDict, BatchList

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


def main() -> None:
    check_batch_list()
    check_batch_dict()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
