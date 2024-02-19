from __future__ import annotations

from collections.abc import Iterable, Sequence
from unittest.mock import Mock, patch

import numpy as np
import pytest
from coola import objects_are_equal
from numpy.typing import DTypeLike

from redcat import ba2 as ba
from redcat.ba2 import BatchedArray
from redcat.ba2.core import IndexType, setup_rng

DTYPES = (bool, int, float)
NUMERIC_DTYPES = [np.float64, np.int64]

MOCK_PERMUTATION4 = Mock(
    return_value=Mock(
        spec=np.random.Generator, permutation=Mock(return_value=np.array([2, 1, 3, 0]))
    )
)


#######################
#     Constructor     #
#######################


def test_batched_array_init() -> None:
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    array = BatchedArray(x)
    assert array.data is x
    assert np.array_equal(array.data, x)
    assert array.batch_axis == 0


def test_batched_array_init_incorrect_data_axis() -> None:
    with pytest.raises(RuntimeError, match=r"data needs at least 1 axis \(received: 0\)"):
        BatchedArray(np.array(2))


def test_batched_array_init_no_check() -> None:
    BatchedArray(np.array(2), check=False)


################################
#     Core functionalities     #
################################


@pytest.mark.parametrize("batch_size", (1, 2))
def test_batched_array_batch_size(batch_size: int) -> None:
    assert BatchedArray(np.arange(batch_size)).batch_size == batch_size


def test_batched_array_data() -> None:
    data = np.ones((2, 3))
    assert BatchedArray(data).data is data


def test_batched_array_allclose_true() -> None:
    assert ba.ones(shape=(2, 3)).allclose(ba.ones(shape=(2, 3)))


def test_batched_array_allclose_false_different_type() -> None:
    assert not BatchedArray(np.ones((2, 3), dtype=float)).allclose(np.zeros((2, 3), dtype=int))


def test_batched_array_allclose_false_different_data() -> None:
    assert not ba.ones(shape=(2, 3)).allclose(BatchedArray(np.zeros((2, 3))))


def test_batched_array_allclose_false_different_shape() -> None:
    assert not ba.ones(shape=(2, 3)).allclose(BatchedArray(np.ones((2, 3, 1))))


def test_batched_array_allclose_false_different_axes() -> None:
    assert not ba.ones(shape=(2, 3)).allclose(BatchedArray(np.ones((2, 3)), batch_axis=1))


@pytest.mark.parametrize(
    ("array", "atol"),
    (
        (ba.full(shape=(2, 3), fill_value=1.5), 1),
        (ba.full(shape=(2, 3), fill_value=1.05), 1e-1),
        (ba.full(shape=(2, 3), fill_value=1.005), 1e-2),
    ),
)
def test_batched_array_allclose_true_atol(array: BatchedArray, atol: float) -> None:
    assert ba.ones((2, 3)).allclose(array, atol=atol, rtol=0)


@pytest.mark.parametrize(
    ("array", "rtol"),
    (
        (ba.full(shape=(2, 3), fill_value=1.5), 1),
        (ba.full(shape=(2, 3), fill_value=1.05), 1e-1),
        (ba.full(shape=(2, 3), fill_value=1.005), 1e-2),
    ),
)
def test_batched_array_allclose_true_rtol(array: BatchedArray, rtol: float) -> None:
    assert ba.ones((2, 3)).allclose(array, rtol=rtol)


def test_batched_array_allequal_true() -> None:
    assert ba.ones(shape=(2, 3)).allequal(ba.ones(shape=(2, 3)))


def test_batched_array_allequal_false_different_type() -> None:
    assert not ba.ones(shape=(2, 3)).allequal(np.ones(shape=(2, 3)))


def test_batched_array_allequal_false_different_data() -> None:
    assert not ba.ones(shape=(2, 3)).allequal(ba.zeros(shape=(2, 3)))


def test_batched_array_allequal_false_different_shape() -> None:
    assert not ba.ones(shape=(2, 3)).allequal(ba.ones(shape=(2, 3, 1)))


def test_batched_array_allequal_false_different_axes() -> None:
    assert not ba.ones(shape=(2, 3), batch_axis=1).allequal(ba.ones(shape=(2, 3)))


def test_batched_array_allequal_equal_nan_false() -> None:
    assert not BatchedArray(np.array([1, np.nan, 3])).allequal(
        BatchedArray(np.array([1, np.nan, 3]))
    )


def test_batched_array_allequal_equal_nan_true() -> None:
    assert BatchedArray(np.array([1, np.nan, 3])).allequal(
        BatchedArray(np.array([1, np.nan, 3])), equal_nan=True
    )


@pytest.mark.parametrize(
    "other",
    (ba.batched_array([[10, 11, 12], [13, 14, 15]]), np.array([[10, 11, 12], [13, 14, 15]])),
)
def test_batched_array_append(
    other: BatchedArray | np.ndarray,
) -> None:
    array = ba.batched_array([[0, 1, 2], [4, 5, 6]])
    array.append(other)
    assert objects_are_equal(
        array, ba.batched_array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]])
    )


def test_batched_array_append_custom_axes() -> None:
    array = ba.batched_array([[0, 4], [1, 5], [2, 6]], batch_axis=1)
    array.append(ba.batched_array([[10, 12], [11, 13], [14, 15]], batch_axis=1))
    assert objects_are_equal(
        array,
        ba.batched_array(
            [[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]],
            batch_axis=1,
        ),
    )


def test_batched_array_append_different_axes() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.append(ba.zeros((2, 2), batch_axis=1))


def test_batched_array_chunk_along_batch_5() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).chunk_along_batch(5),
        (
            ba.batched_array([[0, 1]]),
            ba.batched_array([[2, 3]]),
            ba.batched_array([[4, 5]]),
            ba.batched_array([[6, 7]]),
            ba.batched_array([[8, 9]]),
        ),
    )


def test_batched_array_chunk_along_batch_3() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).chunk_along_batch(3),
        (
            ba.batched_array([[0, 1], [2, 3]]),
            ba.batched_array([[4, 5], [6, 7]]),
            ba.batched_array([[8, 9]]),
        ),
    )


def test_batched_array_chunk_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1).chunk_along_batch(3),
        (
            ba.batched_array([[0, 1], [5, 6]], batch_axis=1),
            ba.batched_array([[2, 3], [7, 8]], batch_axis=1),
            ba.batched_array([[4], [9]], batch_axis=1),
        ),
    )


def test_batched_array_chunk_along_batch_incorrect_chunks() -> None:
    with pytest.raises(RuntimeError, match="chunk expects `chunks` to be greater than 0, got: 0"):
        BatchedArray(np.arange(10).reshape(5, 2)).chunk_along_batch(0)


def test_batched_array_clone() -> None:
    batch = ba.ones(shape=(2, 3))
    batch_cloned = batch.clone()
    batch += 1
    assert batch.data is not batch_cloned.data
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=2.0))
    assert batch_cloned.allequal(ba.ones(shape=(2, 3)))


@pytest.mark.parametrize(
    "arrays",
    (
        [ba.batched_array([[10, 11, 12], [13, 14, 15]])],
        [np.array([[10, 11, 12], [13, 14, 15]])],
        (ba.batched_array([[10, 11, 12], [13, 14, 15]]),),
        [ba.batched_array([[10, 11, 12]]), ba.batched_array([[13, 14, 15]])],
        [ba.batched_array([[10, 11, 12]]), np.array([[13, 14, 15]])],
    ),
)
def test_batched_array_extend(
    arrays: Iterable[BatchedArray | np.ndarray],
) -> None:
    array = ba.batched_array([[0, 1, 2], [4, 5, 6]])
    array.extend(arrays)
    assert objects_are_equal(
        array, ba.batched_array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]])
    )


def test_batched_array_extend_custom_axes() -> None:
    array = ba.batched_array([[0, 4], [1, 5], [2, 6]], batch_axis=1)
    array.extend([ba.batched_array([[10, 12], [11, 13], [14, 15]], batch_axis=1)])
    assert objects_are_equal(
        array,
        ba.batched_array(
            [[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]],
            batch_axis=1,
        ),
    )


def test_batched_array_extend_empty() -> None:
    array = ba.ones((2, 3))
    array.extend([])
    assert objects_are_equal(array, ba.ones((2, 3)))


def test_batched_array_extend_different_axes() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.extend([ba.zeros((2, 2), batch_axis=1)])


@pytest.mark.parametrize("batch_size,num_minibatches", ((1, 10), (2, 5), (3, 4), (4, 3)))
def test_batched_array_get_num_minibatches_drop_last_false(
    batch_size: int, num_minibatches: int
) -> None:
    assert ba.ones(shape=(10, 2)).get_num_minibatches(batch_size) == num_minibatches


@pytest.mark.parametrize("batch_size,num_minibatches", ((1, 10), (2, 5), (3, 3), (4, 2)))
def test_batched_array_get_num_minibatches_drop_last_true(
    batch_size: int, num_minibatches: int
) -> None:
    assert ba.ones(shape=(10, 2)).get_num_minibatches(batch_size, drop_last=True) == num_minibatches


@pytest.mark.parametrize("index", [np.array([2, 0]), [2, 0], (2, 0)])
def test_batched_array_index_select_along_batch(index: np.ndarray | Sequence[int]) -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2))
        .index_select_along_batch(index)
        .allequal(ba.batched_array([[4, 5], [0, 1]]))
    )


def test_batched_array_index_select_along_batch_custom_axes() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        .index_select_along_batch((2, 0))
        .allequal(ba.batched_array([[2, 0], [7, 5]], batch_axis=1))
    )


@pytest.mark.parametrize("permutation", (np.array([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_array_permute_along_batch(permutation: Sequence[int] | np.ndarray) -> None:
    assert (
        ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        .permute_along_batch(permutation)
        .allequal(ba.batched_array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]]))
    )


def test_batched_array_permute_along_batch_custom_axes() -> None:
    assert (
        ba.batched_array([[0, 1, 2, 3], [4, 5, 6, 7]], batch_axis=1)
        .permute_along_batch(np.array([2, 1, 3, 0]))
        .allequal(ba.batched_array([[2, 1, 3, 0], [6, 5, 7, 4]], batch_axis=1))
    )


@pytest.mark.parametrize("permutation", (np.array([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_array_permute_along_batch_(permutation: Sequence[int] | np.ndarray) -> None:
    batch = ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    batch.permute_along_batch_(permutation)
    assert batch.allequal(ba.batched_array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]]))


def test_batched_array_permute_along_batch__custom_axes() -> None:
    batch = ba.batched_array([[0, 1, 2, 3], [4, 5, 6, 7]], batch_axis=1)
    batch.permute_along_batch_(np.array([2, 1, 3, 0]))
    assert batch.allequal(ba.batched_array([[2, 1, 3, 0], [6, 5, 7, 4]], batch_axis=1))


def test_batched_array_select_along_batch() -> None:
    assert objects_are_equal(
        ba.batched_array([[0, 9], [1, 8], [2, 7], [3, 6], [4, 5]]).select_along_batch(2),
        np.array([2, 7]),
    )


def test_batched_array_select_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1).select_along_batch(2),
        np.array([2, 7]),
    )


@patch("redcat.ba2.core.setup_rng", MOCK_PERMUTATION4)
def test_batched_array_shuffle_along_batch() -> None:
    assert (
        ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        .shuffle_along_batch()
        .allequal(ba.batched_array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]]))
    )


@patch("redcat.ba2.core.setup_rng", MOCK_PERMUTATION4)
def test_batched_array_shuffle_along_batch_custom_axes() -> None:
    assert (
        ba.batched_array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], batch_axis=1)
        .shuffle_along_batch()
        .allequal(ba.batched_array([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]], batch_axis=1))
    )


def test_batched_array_shuffle_along_batch_same_random_seed() -> None:
    batch = ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    assert batch.shuffle_along_batch(rng=np.random.default_rng(1)).allequal(
        batch.shuffle_along_batch(rng=np.random.default_rng(1))
    )


def test_batched_array_shuffle_along_batch_different_random_seeds() -> None:
    batch = ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    assert not batch.shuffle_along_batch(rng=np.random.default_rng(1)).allequal(
        batch.shuffle_along_batch(rng=np.random.default_rng(2))
    )


@patch("redcat.ba2.core.setup_rng", MOCK_PERMUTATION4)
def test_batched_array_shuffle_along_batch_() -> None:
    batch = ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    batch.shuffle_along_batch_()
    assert batch.allequal(ba.batched_array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]]))


@patch("redcat.ba2.core.setup_rng", MOCK_PERMUTATION4)
def test_batched_array_shuffle_along_batch__custom_axes() -> None:
    batch = ba.batched_array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], batch_axis=1)
    batch.shuffle_along_batch_()
    assert batch.allequal(
        ba.batched_array([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]], batch_axis=1)
    )


def test_batched_array_shuffle_along_batch__same_random_seed() -> None:
    batch1 = ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    batch1.shuffle_along_batch_(rng=np.random.default_rng(1))
    batch2 = ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    batch2.shuffle_along_batch_(rng=np.random.default_rng(1))
    assert batch1.allequal(batch2)


def test_batched_array_shuffle_along_batch__different_random_seeds() -> None:
    batch1 = ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    batch1.shuffle_along_batch_(rng=np.random.default_rng(1))
    batch2 = ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    batch2.shuffle_along_batch_(rng=np.random.default_rng(2))
    assert not batch1.allequal(batch2)


def test_batched_array_slice_along_batch() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2))
        .slice_along_batch()
        .allequal(ba.batched_array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]))
    )


def test_batched_array_slice_along_batch_start_2() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2))
        .slice_along_batch(start=2)
        .allequal(ba.batched_array([[4, 5], [6, 7], [8, 9]]))
    )


def test_batched_array_slice_along_batch_stop_3() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2))
        .slice_along_batch(stop=3)
        .allequal(ba.batched_array([[0, 1], [2, 3], [4, 5]]))
    )


def test_batched_array_slice_along_batch_stop_100() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2))
        .slice_along_batch(stop=100)
        .allequal(ba.batched_array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]))
    )


def test_batched_array_slice_along_batch_step_2() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2))
        .slice_along_batch(step=2)
        .allequal(ba.batched_array([[0, 1], [4, 5], [8, 9]]))
    )


def test_batched_array_slice_along_batch_start_1_stop_4_step_2() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2))
        .slice_along_batch(start=1, stop=4, step=2)
        .allequal(ba.batched_array([[2, 3], [6, 7]]))
    )


def test_batched_array_slice_along_batch_custom_axis() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        .slice_along_batch(start=2)
        .allequal(ba.batched_array([[2, 3, 4], [7, 8, 9]], batch_axis=1))
    )


def test_batched_array_slice_along_batch_batch_axis_1() -> None:
    assert (
        BatchedArray(np.arange(20).reshape(2, 5, 2), batch_axis=1)
        .slice_along_batch(start=2)
        .allequal(
            ba.batched_array(
                [[[4, 5], [6, 7], [8, 9]], [[14, 15], [16, 17], [18, 19]]],
                batch_axis=1,
            )
        )
    )


def test_batched_array_slice_along_batch_batch_axis_2() -> None:
    assert (
        BatchedArray(np.arange(20).reshape(2, 2, 5), batch_axis=2)
        .slice_along_batch(start=2)
        .allequal(
            ba.batched_array([[[2, 3, 4], [7, 8, 9]], [[12, 13, 14], [17, 18, 19]]], batch_axis=2)
        )
    )


def test_batched_array_split_along_batch_split_size_1() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).split_along_batch(1),
        (
            ba.batched_array([[0, 1]]),
            ba.batched_array([[2, 3]]),
            ba.batched_array([[4, 5]]),
            ba.batched_array([[6, 7]]),
            ba.batched_array([[8, 9]]),
        ),
    )


def test_batched_array_split_along_batch_split_size_2() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).split_along_batch(2),
        (
            ba.batched_array([[0, 1], [2, 3]]),
            ba.batched_array([[4, 5], [6, 7]]),
            ba.batched_array([[8, 9]]),
        ),
    )


def test_batched_array_split_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1).split_along_batch(2),
        (
            ba.batched_array([[0, 1], [5, 6]], batch_axis=1),
            ba.batched_array([[2, 3], [7, 8]], batch_axis=1),
            ba.batched_array([[4], [9]], batch_axis=1),
        ),
    )


def test_batched_array_split_along_batch_split_list() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).split_along_batch([2, 2, 1]),
        (
            ba.batched_array([[0, 1], [2, 3]]),
            ba.batched_array([[4, 5], [6, 7]]),
            ba.batched_array([[8, 9]]),
        ),
    )


def test_batched_array_summary() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5)).summary()
        == "BatchedArray(dtype=int64, shape=(2, 5), batch_axis=0)"
    )


def test_batched_array_to_data() -> None:
    assert objects_are_equal(ba.ones(shape=(2, 3)).to_data(), np.ones(shape=(2, 3)))


def test_batched_array_to_minibatches_10_batch_size_2() -> None:
    assert objects_are_equal(
        list(BatchedArray(np.arange(20).reshape(10, 2)).to_minibatches(batch_size=2)),
        [
            ba.batched_array([[0, 1], [2, 3]]),
            ba.batched_array([[4, 5], [6, 7]]),
            ba.batched_array([[8, 9], [10, 11]]),
            ba.batched_array([[12, 13], [14, 15]]),
            ba.batched_array([[16, 17], [18, 19]]),
        ],
    )


def test_batched_array_to_minibatches_10_batch_size_3() -> None:
    assert objects_are_equal(
        list(BatchedArray(np.arange(20).reshape(10, 2)).to_minibatches(batch_size=3)),
        [
            ba.batched_array([[0, 1], [2, 3], [4, 5]]),
            ba.batched_array([[6, 7], [8, 9], [10, 11]]),
            ba.batched_array([[12, 13], [14, 15], [16, 17]]),
            ba.batched_array([[18, 19]]),
        ],
    )


def test_batched_array_to_minibatches_10_batch_size_4() -> None:
    assert objects_are_equal(
        list(BatchedArray(np.arange(20).reshape(10, 2)).to_minibatches(batch_size=4)),
        [
            ba.batched_array([[0, 1], [2, 3], [4, 5], [6, 7]]),
            ba.batched_array([[8, 9], [10, 11], [12, 13], [14, 15]]),
            ba.batched_array([[16, 17], [18, 19]]),
        ],
    )


def test_batched_array_to_minibatches_drop_last_true_10_batch_size_2() -> None:
    assert objects_are_equal(
        list(
            BatchedArray(np.arange(20).reshape(10, 2)).to_minibatches(batch_size=2, drop_last=True)
        ),
        [
            ba.batched_array([[0, 1], [2, 3]]),
            ba.batched_array([[4, 5], [6, 7]]),
            ba.batched_array([[8, 9], [10, 11]]),
            ba.batched_array([[12, 13], [14, 15]]),
            ba.batched_array([[16, 17], [18, 19]]),
        ],
    )


def test_batched_array_to_minibatches_drop_last_true_10_batch_size_3() -> None:
    assert objects_are_equal(
        list(
            BatchedArray(np.arange(20).reshape(10, 2)).to_minibatches(batch_size=3, drop_last=True)
        ),
        [
            ba.batched_array([[0, 1], [2, 3], [4, 5]]),
            ba.batched_array([[6, 7], [8, 9], [10, 11]]),
            ba.batched_array([[12, 13], [14, 15], [16, 17]]),
        ],
    )


def test_batched_array_to_minibatches_drop_last_true_10_batch_size_4() -> None:
    assert objects_are_equal(
        list(
            BatchedArray(np.arange(20).reshape(10, 2)).to_minibatches(batch_size=4, drop_last=True)
        ),
        [
            ba.batched_array([[0, 1], [2, 3], [4, 5], [6, 7]]),
            ba.batched_array([[8, 9], [10, 11], [12, 13], [14, 15]]),
        ],
    )


def test_batched_array_to_minibatches_custom_dims() -> None:
    assert objects_are_equal(
        list(BatchedArray(np.arange(20).reshape(2, 10), batch_axis=1).to_minibatches(batch_size=3)),
        [
            ba.batched_array([[0, 1, 2], [10, 11, 12]], batch_axis=1),
            ba.batched_array([[3, 4, 5], [13, 14, 15]], batch_axis=1),
            ba.batched_array([[6, 7, 8], [16, 17, 18]], batch_axis=1),
            ba.batched_array([[9], [19]], batch_axis=1),
        ],
    )


def test_batched_array_to_minibatches_deepcopy_true() -> None:
    batch = ba.batched_array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    for item in batch.to_minibatches(batch_size=2, deepcopy=True):
        item.data[0, 0] = 42
    assert batch.allequal(ba.batched_array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]))


def test_batched_array_to_minibatches_deepcopy_false() -> None:
    batch = ba.batched_array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    for item in batch.to_minibatches(batch_size=2):
        item.data[0, 0] = 42
    assert batch.allequal(ba.batched_array([[42, 1], [2, 3], [42, 5], [6, 7], [42, 9]]))


######################################
#     Additional functionalities     #
######################################


def test_batched_array_asarray() -> None:
    assert objects_are_equal(np.asarray(BatchedArray(np.ones((2, 3)))), np.ones((2, 3)))


def test_batched_array_repr() -> None:
    assert repr(BatchedArray(np.arange(3))) == "array([0, 1, 2], batch_axis=0)"


def test_batched_array_str() -> None:
    assert str(BatchedArray(np.arange(3))) == "[0 1 2]\nwith batch_axis=0"


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_batched_array_batch_axis(batch_axis: int) -> None:
    assert ba.ones(shape=(2, 3), batch_axis=batch_axis).batch_axis == batch_axis


#########################
#     Memory layout     #
#########################


def test_batched_array_ndim() -> None:
    assert ba.ones(shape=(2, 3)).ndim == 2


def test_batched_array_shape() -> None:
    assert ba.ones(shape=(2, 3)).shape == (2, 3)


def test_batched_array_size() -> None:
    assert ba.ones(shape=(2, 3)).size == 6


#####################
#     Data type     #
#####################


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_array_dtype(dtype: np.dtype) -> None:
    assert ba.ones(shape=(2, 3), dtype=dtype).dtype == dtype


###############################
#     Creation operations     #
###############################


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_array_empty_like(dtype: np.dtype) -> None:
    array = ba.zeros(shape=(2, 3), dtype=dtype).empty_like()
    assert isinstance(array, BatchedArray)
    assert array.data.shape == (2, 3)
    assert array.dtype == dtype
    assert array.batch_axis == 0


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_array_empty_like_target_dtype(dtype: np.dtype) -> None:
    array = ba.zeros(shape=(2, 3)).empty_like(dtype=dtype)
    assert isinstance(array, BatchedArray)
    assert array.data.shape == (2, 3)
    assert array.dtype == dtype
    assert array.batch_axis == 0


def test_batched_array_empty_like_custom_axes() -> None:
    array = ba.zeros(shape=(3, 2), batch_axis=1).empty_like()
    assert isinstance(array, BatchedArray)
    assert array.data.shape == (3, 2)
    assert array.dtype == float
    assert array.batch_axis == 1


@pytest.mark.parametrize("batch_size", (1, 2))
def test_batched_array_empty_like_custom_batch_size(batch_size: int) -> None:
    array = ba.zeros(shape=(2, 3)).empty_like(batch_size=batch_size)
    assert isinstance(array, BatchedArray)
    assert array.data.shape == (batch_size, 3)
    assert array.dtype == float
    assert array.batch_axis == 0


@pytest.mark.parametrize("fill_value", (1.5, 2.0, -1.0))
def test_batched_array_full_like(fill_value: float) -> None:
    assert (
        ba.zeros(shape=(2, 3))
        .full_like(fill_value)
        .allequal(ba.full(shape=(2, 3), fill_value=fill_value))
    )


def test_batched_array_full_like_custom_axes() -> None:
    assert (
        ba.zeros(shape=(3, 2), batch_axis=1)
        .full_like(fill_value=2.0)
        .allequal(ba.full(shape=(3, 2), fill_value=2.0, batch_axis=1))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_array_full_like_dtype(dtype: np.dtype) -> None:
    assert (
        ba.zeros(shape=(2, 3), dtype=dtype)
        .full_like(fill_value=2.0)
        .allequal(ba.full(shape=(2, 3), fill_value=2.0, dtype=dtype))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_array_full_like_target_dtype(dtype: np.dtype) -> None:
    assert (
        ba.zeros(shape=(2, 3))
        .full_like(fill_value=2.0, dtype=dtype)
        .allequal(ba.full(shape=(2, 3), fill_value=2.0, dtype=dtype))
    )


@pytest.mark.parametrize("batch_size", (1, 2))
def test_batched_array_full_like_custom_batch_size(batch_size: int) -> None:
    assert (
        ba.zeros(shape=(2, 3))
        .full_like(2.0, batch_size=batch_size)
        .allequal(ba.full(shape=(batch_size, 3), fill_value=2.0))
    )


def test_batched_array_ones_like() -> None:
    assert ba.zeros(shape=(2, 3)).ones_like().allequal(ba.ones(shape=(2, 3)))


def test_batched_array_ones_like_custom_axes() -> None:
    assert (
        ba.zeros(shape=(3, 2), batch_axis=1)
        .ones_like()
        .allequal(ba.ones(shape=(3, 2), batch_axis=1))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_array_ones_like_dtype(dtype: np.dtype) -> None:
    assert (
        ba.zeros(shape=(2, 3), dtype=dtype).ones_like().allequal(ba.ones(shape=(2, 3), dtype=dtype))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_array_ones_like_target_dtype(dtype: np.dtype) -> None:
    assert ba.zeros((2, 3)).ones_like(dtype=dtype).allequal(ba.ones(shape=(2, 3), dtype=dtype))


@pytest.mark.parametrize("batch_size", (1, 2))
def test_batched_array_ones_like_custom_batch_size(batch_size: int) -> None:
    assert ba.zeros((2, 3)).ones_like(batch_size=batch_size).allequal(ba.ones((batch_size, 3)))


def test_batched_array_zeros_like() -> None:
    assert ba.ones(shape=(2, 3)).zeros_like().allequal(ba.zeros(shape=(2, 3)))


def test_batched_array_zeros_like_custom_axes() -> None:
    assert (
        ba.ones(shape=(3, 2), batch_axis=1)
        .zeros_like()
        .allequal(ba.zeros(shape=(3, 2), batch_axis=1))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_array_zeros_like_dtype(dtype: np.dtype) -> None:
    assert (
        ba.ones(shape=(2, 3), dtype=dtype)
        .zeros_like()
        .allequal(ba.zeros(shape=(2, 3), dtype=dtype))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_array_zeros_like_target_dtype(dtype: np.dtype) -> None:
    assert (
        ba.ones(shape=(2, 3)).zeros_like(dtype=dtype).allequal(ba.zeros(shape=(2, 3), dtype=dtype))
    )


@pytest.mark.parametrize("batch_size", (1, 2))
def test_batched_array_zeros_like_custom_batch_size(batch_size: int) -> None:
    assert (
        ba.ones(shape=(2, 3)).zeros_like(batch_size=batch_size).allequal(ba.zeros((batch_size, 3)))
    )


################################
#     Comparison operators     #
################################


@pytest.mark.parametrize(
    "other",
    (
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5),
        5,
        5.0,
    ),
)
def test_batched_array__eq__(other: np.ndarray | float) -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)) == other,
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [True, False, False, False, False]],
                dtype=bool,
            ),
        ),
    )


def test_batched_array__eq__custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        == ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1),
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [True, False, False, False, False]],
                dtype=bool,
            ),
            batch_axis=1,
        ),
    )


def test_batched_array__eq__different_axes() -> None:
    x1 = BatchedArray(np.arange(10).reshape(2, 5))
    x2 = ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1.__eq__(x2)


@pytest.mark.parametrize(
    "other",
    (
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5),
        5,
        5.0,
    ),
)
def test_batched_array__ge__(other: np.ndarray | int | float) -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)) >= other,
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [True, True, True, True, True]],
                dtype=bool,
            ),
        ),
    )


def test_batched_array__ge__custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >= ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1),
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [True, True, True, True, True]],
                dtype=bool,
            ),
            batch_axis=1,
        ),
    )


def test_batched_array__ge__different_axes() -> None:
    x1 = BatchedArray(np.arange(10).reshape(2, 5))
    x2 = ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1.__ge__(x2)


@pytest.mark.parametrize(
    "other",
    (
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5),
        5,
        5.0,
    ),
)
def test_batched_array__gt__(other: np.ndarray | int | float) -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)) > other,
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [False, True, True, True, True]],
                dtype=bool,
            ),
        ),
    )


def test_batched_array__gt__custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        > ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1),
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [False, True, True, True, True]],
                dtype=bool,
            ),
            batch_axis=1,
        ),
    )


def test_batched_array__gt__different_axes() -> None:
    x1 = BatchedArray(np.arange(10).reshape(2, 5))
    x2 = ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1.__gt__(x2)


@pytest.mark.parametrize(
    "other",
    (
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5),
        5,
        5.0,
    ),
)
def test_batched_array__le__(other: np.ndarray | int | float) -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)) <= other,
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [True, False, False, False, False]],
                dtype=bool,
            ),
        ),
    )


def test_batched_array__le__custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        <= ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1),
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [True, False, False, False, False]],
                dtype=bool,
            ),
            batch_axis=1,
        ),
    )


def test_batched_array__le__different_axes() -> None:
    x1 = BatchedArray(np.arange(10).reshape(2, 5))
    x2 = ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1.__le__(x2)


@pytest.mark.parametrize(
    "other",
    (
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5),
        5,
        5.0,
    ),
)
def test_batched_array__lt__(other: np.ndarray | int | float) -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)) < other,
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [False, False, False, False, False]],
                dtype=bool,
            ),
        ),
    )


def test_batched_array__lt__custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        < ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1),
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [False, False, False, False, False]],
                dtype=bool,
            ),
            batch_axis=1,
        ),
    )


def test_batched_array__lt__different_axes() -> None:
    x1 = BatchedArray(np.arange(10).reshape(2, 5))
    x2 = ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1.__lt__(x2)


@pytest.mark.parametrize(
    "other",
    (
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5),
        5,
        5.0,
    ),
)
def test_batched_array__ne__(other: np.ndarray | float) -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)) != other,
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [False, True, True, True, True]],
                dtype=bool,
            ),
        ),
    )


def test_batched_array__ne__custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        != ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1),
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [False, True, True, True, True]],
                dtype=bool,
            ),
            batch_axis=1,
        ),
    )


def test_batched_array__ne__different_axes() -> None:
    x1 = BatchedArray(np.arange(10).reshape(2, 5))
    x2 = ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1.__ne__(x2)


##################################
#     Arithmetical operators     #
##################################


@pytest.mark.parametrize(
    "other",
    (
        ba.ones(shape=(2, 3)),
        np.ones(shape=(2, 3)),
        ba.ones(shape=(2, 1)),
        1,
        1.0,
    ),
)
def test_batched_array__add__(other: np.ndarray | int | float) -> None:
    assert (ba.zeros(shape=(2, 3)) + other).allequal(ba.ones(shape=(2, 3)))


def test_batched_array__add___custom_axes() -> None:
    assert (ba.zeros(shape=(2, 3), batch_axis=1) + ba.ones(shape=(2, 3), batch_axis=1)).allequal(
        ba.ones(shape=(2, 3), batch_axis=1)
    )


def test_batched_array__add___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3))
    x2 = ba.ones(shape=(2, 3), batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1 + x2


@pytest.mark.parametrize(
    "other",
    (
        ba.ones(shape=(2, 3)),
        np.ones(shape=(2, 3)),
        ba.ones(shape=(2, 1)),
        1,
        1.0,
    ),
)
def test_batched_array__iadd__(other: np.ndarray | int | float) -> None:
    batch = ba.zeros(shape=(2, 3))
    batch += other
    assert batch.allequal(ba.ones(shape=(2, 3)))


def test_batched_array__iadd___custom_axes() -> None:
    batch = ba.zeros(shape=(2, 3), batch_axis=1)
    batch += ba.ones(shape=(2, 3), batch_axis=1)
    assert batch.allequal(ba.ones(shape=(2, 3), batch_axis=1))


def test_batched_array__iadd___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3))
    x2 = ba.ones(shape=(2, 3), batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1 += x2


@pytest.mark.parametrize(
    "other",
    (
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ),
)
def test_batched_array__floordiv__(other: np.ndarray | int | float) -> None:
    assert (ba.ones(shape=(2, 3)) // other).allequal(ba.zeros(shape=(2, 3)))


def test_batched_array__floordiv__custom_axes() -> None:
    assert (
        ba.ones(shape=(2, 3), batch_axis=1) // ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    ).allequal(ba.zeros(shape=(2, 3), batch_axis=1))


def test_batched_array__floordiv___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3))
    x2 = ba.ones(shape=(2, 3), batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1 // x2


@pytest.mark.parametrize(
    "other",
    (
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ),
)
def test_batched_array__ifloordiv__(other: np.ndarray | int | float) -> None:
    batch = ba.ones(shape=(2, 3))
    batch //= other
    assert batch.allequal(ba.zeros(shape=(2, 3)))


def test_batched_array__ifloordiv___custom_axes() -> None:
    batch = ba.ones(shape=(2, 3), batch_axis=1)
    batch //= ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    assert batch.allequal(ba.zeros(shape=(2, 3), batch_axis=1))


def test_batched_array__ifloordiv___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3))
    x2 = ba.ones(shape=(2, 3), batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1 //= x2


@pytest.mark.parametrize(
    "other",
    (
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ),
)
def test_batched_array__mod__(other: np.ndarray | int | float) -> None:
    assert (ba.ones(shape=(2, 3)) % other).allequal(ba.ones(shape=(2, 3)))


def test_batched_array__mod___custom_axes() -> None:
    assert (
        ba.ones(shape=(2, 3), batch_axis=1) % ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    ).allequal(ba.ones(shape=(2, 3), batch_axis=1))


def test_batched_array__mod___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3))
    x2 = ba.ones(shape=(2, 3), batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1 % x2


@pytest.mark.parametrize(
    "other",
    (
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ),
)
def test_batched_array__imod__(other: np.ndarray | int | float) -> None:
    batch = ba.ones(shape=(2, 3))
    batch %= other
    assert batch.allequal(ba.ones(shape=(2, 3)))


def test_batched_array__imod__custom_axes() -> None:
    batch = ba.ones(shape=(2, 3), batch_axis=1)
    batch %= ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    assert batch.allequal(ba.ones(shape=(2, 3), batch_axis=1))


def test_batched_array__imod___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3))
    x2 = ba.ones(shape=(2, 3), batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1 %= x2


@pytest.mark.parametrize(
    "other",
    (
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ),
)
def test_batched_array__mul__(other: np.ndarray | int | float) -> None:
    assert (ba.ones(shape=(2, 3)) * other).allequal(ba.full(shape=(2, 3), fill_value=2.0))


def test_batched_array__mul___custom_axes() -> None:
    assert (
        ba.ones(shape=(2, 3), batch_axis=1) * ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    ).allequal(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))


def test_batched_array__mul___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3))
    x2 = ba.ones(shape=(2, 3), batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1 * x2


@pytest.mark.parametrize(
    "other",
    (
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ),
)
def test_batched_array__imul__(other: np.ndarray | int | float) -> None:
    batch = ba.ones(shape=(2, 3))
    batch *= other
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=2.0))


def test_batched_array__imul__custom_axes() -> None:
    batch = ba.ones(shape=(2, 3), batch_axis=1)
    batch *= ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))


def test_batched_array__imul___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3))
    x2 = ba.ones(shape=(2, 3), batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1 *= x2


def test_batched_array__neg__() -> None:
    assert (-ba.ones(shape=(2, 3))).allequal(ba.full(shape=(2, 3), fill_value=-1.0))


def test_batched_array__neg__custom_axes() -> None:
    assert (-ba.ones(shape=(2, 3), batch_axis=1)).allequal(
        ba.full(shape=(2, 3), fill_value=-1.0, batch_axis=1)
    )


@pytest.mark.parametrize(
    "other",
    (
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ),
)
def test_batched_array__sub__(other: np.ndarray | int | float) -> None:
    assert (ba.ones(shape=(2, 3)) - other).allequal(ba.full(shape=(2, 3), fill_value=-1.0))


def test_batched_array__sub__custom_axes() -> None:
    assert (
        ba.ones(shape=(2, 3), batch_axis=1) - ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    ).allequal(ba.full(shape=(2, 3), fill_value=-1.0, batch_axis=1))


def test_batched_array__sub___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3))
    x2 = ba.ones(shape=(2, 3), batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1 - x2


@pytest.mark.parametrize(
    "other",
    (
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ),
)
def test_batched_array__isub__(other: np.ndarray | int | float) -> None:
    batch = ba.ones(shape=(2, 3))
    batch -= other
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=-1.0))


def test_batched_array__isub__custom_axes() -> None:
    batch = ba.ones(shape=(2, 3), batch_axis=1)
    batch -= ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=-1.0, batch_axis=1))


def test_batched_array__isub___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3))
    x2 = ba.ones(shape=(2, 3), batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1 -= x2


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array__truediv__(other: np.ndarray | int | float) -> None:
    assert (ba.ones(shape=(2, 3)) / other).allequal(ba.full(shape=(2, 3), fill_value=0.5))


def test_batched_array__truediv__custom_axes() -> None:
    assert (
        ba.ones(shape=(2, 3), batch_axis=1) / ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    ).allequal(ba.full(shape=(2, 3), fill_value=0.5, batch_axis=1))


def test_batched_array__truediv___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3))
    x2 = ba.ones(shape=(2, 3), batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1 / x2


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array__itruediv__(other: np.ndarray | int | float) -> None:
    batch = ba.ones(shape=(2, 3))
    batch /= other
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=0.5))


def test_batched_array__itruediv__custom_axes() -> None:
    batch = ba.ones(shape=(2, 3), batch_axis=1)
    batch /= ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=0.5, batch_axis=1))


def test_batched_array__itruediv___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3))
    x2 = ba.ones(shape=(2, 3), batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1 /= x2


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_add(other: np.ndarray | int | float) -> None:
    assert ba.ones((2, 3)).add(other).allequal(ba.full(shape=(2, 3), fill_value=3.0))


@pytest.mark.parametrize("dtype", [float, int])
def test_batched_array_add_alpha_2(dtype: DTypeLike) -> None:
    assert (
        ba.ones(shape=(2, 3), dtype=dtype)
        .add(ba.full(shape=(2, 3), fill_value=2.0, dtype=dtype), alpha=2)
        .allequal(ba.full(shape=(2, 3), fill_value=5.0, dtype=dtype))
    )


def test_batched_array_add_batch_axis_1() -> None:
    assert (
        ba.ones(shape=(2, 3), batch_axis=1)
        .add(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
        .allequal(ba.full(shape=(2, 3), fill_value=3.0, batch_axis=1))
    )


def test_batched_array_add_different_axes() -> None:
    batch = ba.ones(shape=(2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.add(ba.ones(shape=(2, 2), batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_add_(other: np.ndarray | int | float) -> None:
    batch = ba.ones(shape=(2, 3))
    batch.add_(other)
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=3.0))


@pytest.mark.parametrize("dtype", [float, int])
def test_batched_array_add__alpha_2(dtype: DTypeLike) -> None:
    batch = ba.ones(shape=(2, 3), dtype=dtype)
    batch.add_(ba.full(shape=(2, 3), fill_value=2.0, dtype=dtype), alpha=2)
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=5.0, dtype=dtype))


def test_batched_array_add__custom_axes() -> None:
    batch = ba.ones(shape=(2, 3), batch_axis=1)
    batch.add_(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=3.0, batch_axis=1))


def test_batched_array_add__different_axes() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.add_(ba.ones(shape=(2, 2), batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_floordiv(other: np.ndarray | int | float) -> None:
    assert ba.ones((2, 3)).floordiv(other).allequal(ba.zeros(shape=(2, 3)))


def test_batched_array_floordiv_custom_axes() -> None:
    assert (
        ba.ones(shape=(2, 3), batch_axis=1)
        .floordiv(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
        .allequal(ba.zeros(shape=(2, 3), batch_axis=1))
    )


def test_batched_array_floordiv_incorrect_batch_axis() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.floordiv(ba.ones(shape=(2, 2), batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_floordiv_(other: np.ndarray | int | float) -> None:
    batch = ba.ones((2, 3))
    batch.floordiv_(other)
    assert batch.allequal(ba.zeros(shape=(2, 3)))


def test_batched_array_floordiv__custom_axes() -> None:
    batch = ba.ones(shape=(2, 3), batch_axis=1)
    batch.floordiv_(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
    assert batch.allequal(ba.zeros(shape=(2, 3), batch_axis=1))


def test_batched_array_floordiv__different_axes() -> None:
    batch = ba.ones(shape=(2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.floordiv_(ba.ones(shape=(2, 2), batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_fmod(other: np.ndarray | int | float) -> None:
    assert ba.ones((2, 3)).fmod(other).allequal(ba.ones(shape=(2, 3)))


def test_batched_array_fmod_custom_axes() -> None:
    assert (
        ba.ones(shape=(2, 3), batch_axis=1)
        .fmod(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
        .allequal(ba.ones(shape=(2, 3), batch_axis=1))
    )


def test_batched_array_fmod_incorrect_batch_axis() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.fmod(ba.ones(shape=(2, 2), batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_fmod_(other: np.ndarray | int | float) -> None:
    batch = ba.ones((2, 3))
    batch.fmod_(other)
    assert batch.allequal(ba.ones(shape=(2, 3)))


def test_batched_array_fmod__custom_axes() -> None:
    batch = ba.ones(shape=(2, 3), batch_axis=1)
    batch.fmod_(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
    assert batch.allequal(ba.ones(shape=(2, 3), batch_axis=1))


def test_batched_array_fmod__different_axes() -> None:
    batch = ba.ones(shape=(2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.fmod_(ba.ones(shape=(2, 2), batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_mul(other: np.ndarray | int | float) -> None:
    assert ba.ones((2, 3)).mul(other).allequal(ba.full(shape=(2, 3), fill_value=2.0))


def test_batched_array_mul_custom_axes() -> None:
    assert (
        ba.ones(shape=(2, 3), batch_axis=1)
        .mul(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
        .allequal(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
    )


def test_batched_array_mul_incorrect_batch_axis() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.mul(ba.ones(shape=(2, 2), batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_mul_(other: np.ndarray | int | float) -> None:
    batch = ba.ones((2, 3))
    batch.mul_(other)
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=2.0))


def test_batched_array_mul__custom_axes() -> None:
    batch = ba.ones(shape=(2, 3), batch_axis=1)
    batch.mul_(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))


def test_batched_array_mul__different_axes() -> None:
    batch = ba.ones(shape=(2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.mul_(ba.ones(shape=(2, 2), batch_axis=1))


def test_batched_array_neg() -> None:
    assert ba.ones(shape=(2, 3)).neg().allequal(-ba.ones(shape=(2, 3)))


def test_batched_array_neg_custom_axes() -> None:
    assert ba.ones(shape=(2, 3), batch_axis=1).neg().allequal(-ba.ones(shape=(2, 3), batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_sub(other: np.ndarray | int | float) -> None:
    assert ba.ones((2, 3)).sub(other).allequal(ba.full(shape=(2, 3), fill_value=-1.0))


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_batched_array_sub_alpha_2(dtype: DTypeLike) -> None:
    assert (
        ba.ones(shape=(2, 3), dtype=dtype)
        .sub(ba.full(shape=(2, 3), fill_value=2, dtype=dtype), alpha=2)
        .allequal(ba.full(shape=(2, 3), fill_value=-3, dtype=dtype))
    )


def test_batched_array_sub_custom_axes() -> None:
    assert (
        ba.ones(shape=(2, 3), batch_axis=1)
        .sub(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
        .allequal(ba.full(shape=(2, 3), fill_value=-1.0, batch_axis=1))
    )


def test_batched_array_sub_incorrect_batch_axis() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.sub(ba.ones(shape=(2, 2), batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_sub_(other: np.ndarray | int | float) -> None:
    batch = ba.ones((2, 3))
    batch.sub_(other)
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=-1.0))


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
def test_batched_array_sub__alpha_2(dtype: DTypeLike) -> None:
    batch = ba.ones(shape=(2, 3), dtype=dtype)
    batch.sub_(ba.full(shape=(2, 3), fill_value=2, dtype=dtype), alpha=2)
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=-3, dtype=dtype))


def test_batched_array_sub__custom_axes() -> None:
    batch = ba.ones(shape=(2, 3), batch_axis=1)
    batch.sub_(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=-1.0, batch_axis=1))


def test_batched_array_sub__different_axes() -> None:
    batch = ba.ones(shape=(2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.sub_(ba.ones(shape=(2, 2), batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_truediv(other: np.ndarray | int | float) -> None:
    assert ba.ones((2, 3)).truediv(other).allequal(ba.full(shape=(2, 3), fill_value=0.5))


def test_batched_array_truediv_custom_axes() -> None:
    assert (
        ba.ones(shape=(2, 3), batch_axis=1)
        .truediv(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
        .allequal(ba.full(shape=(2, 3), fill_value=0.5, batch_axis=1))
    )


def test_batched_array_truediv_incorrect_batch_axis() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.truediv(ba.ones(shape=(2, 2), batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 3), fill_value=2.0),
        np.full(shape=(2, 3), fill_value=2.0),
        ba.full(shape=(2, 1), fill_value=2.0),
        2,
        2.0,
    ],
)
def test_batched_array_truediv_(other: np.ndarray | int | float) -> None:
    batch = ba.ones((2, 3))
    batch.truediv_(other)
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=0.5))


def test_batched_array_truediv__custom_axes() -> None:
    batch = ba.ones(shape=(2, 3), batch_axis=1)
    batch.truediv_(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=0.5, batch_axis=1))


def test_batched_array_truediv__different_axes() -> None:
    batch = ba.ones(shape=(2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.truediv_(ba.ones(shape=(2, 2), batch_axis=1))


#######################################
#     Array manipulation routines     #
#######################################


def test_batched_array__getitem___none() -> None:
    batch = BatchedArray(np.arange(10).reshape(2, 5))
    assert objects_are_equal(batch[None], np.array([[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]]))


def test_batched_array__getitem___int() -> None:
    batch = BatchedArray(np.arange(10).reshape(2, 5))
    assert objects_are_equal(batch[0], np.array([0, 1, 2, 3, 4]))


def test_batched_array__range___slice() -> None:
    batch = BatchedArray(np.arange(10).reshape(2, 5))
    assert objects_are_equal(batch[0:2, 2:4], np.array([[2, 3], [7, 8]]))


@pytest.mark.parametrize(
    "index",
    [[2, 0], np.array([2, 0]), ba.batched_array([2, 0])],
)
def test_batched_array__getitem___list_like(index: IndexType) -> None:
    batch = BatchedArray(np.arange(10).reshape(5, 2))
    assert objects_are_equal(batch[index], np.array([[4, 5], [0, 1]]))


def test_batched_array__setitem___int() -> None:
    batch = BatchedArray(np.arange(10).reshape(2, 5))
    batch[0] = 7
    assert objects_are_equal(batch, ba.batched_array([[7, 7, 7, 7, 7], [5, 6, 7, 8, 9]]))


def test_batched_array__setitem___slice() -> None:
    batch = BatchedArray(np.arange(10).reshape(2, 5))
    batch[0:1, 2:4] = 7
    assert objects_are_equal(batch, ba.batched_array([[0, 1, 7, 7, 4], [5, 6, 7, 8, 9]]))


@pytest.mark.parametrize(
    "index",
    [
        [0, 2],
        np.array([0, 2]),
        ba.batched_array([0, 2]),
    ],
)
def test_batched_array__setitem___list_like_index(index: IndexType) -> None:
    batch = BatchedArray(np.arange(10).reshape(5, 2))
    batch[index] = 7
    assert objects_are_equal(batch, ba.batched_array([[7, 7], [2, 3], [7, 7], [6, 7], [8, 9]]))


@pytest.mark.parametrize(
    "value",
    [np.array([[0, -4]]), ba.batched_array([[0, -4]])],
)
def test_batched_array__setitem___array_value(value: np.ndarray | BatchedArray) -> None:
    batch = BatchedArray(np.arange(10).reshape(2, 5))
    batch[1:2, 2:4] = value
    assert objects_are_equal(batch, ba.batched_array([[0, 1, 2, 3, 4], [5, 6, 0, -4, 9]]))


########################################################
#     Array manipulation routines | Joining arrays     #
########################################################


@pytest.mark.parametrize(
    "arrays",
    (
        [ba.batched_array([[10, 11, 12], [13, 14, 15]])],
        [np.array([[10, 11, 12], [13, 14, 15]])],
        (ba.batched_array([[10, 11, 12], [13, 14, 15]]),),
        [ba.batched_array([[10, 11, 12]]), ba.batched_array([[13, 14, 15]])],
        [ba.batched_array([[10, 11, 12]]), np.array([[13, 14, 15]])],
    ),
)
def test_batched_array_concatenate_axis_0(
    arrays: Iterable[BatchedArray | np.ndarray],
) -> None:
    assert objects_are_equal(
        ba.batched_array([[0, 1, 2], [4, 5, 6]]).concatenate(arrays, axis=0),
        ba.batched_array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]),
    )


@pytest.mark.parametrize(
    "arrays",
    (
        [ba.batched_array([[10, 11], [12, 13]])],
        [np.array([[10, 11], [12, 13]])],
        (ba.batched_array([[10, 11], [12, 13]]),),
        [ba.batched_array([[10], [12]]), ba.batched_array([[11], [13]])],
        [ba.batched_array([[10], [12]]), np.array([[11], [13]])],
    ),
)
def test_batched_array_concatenate_axis_1(
    arrays: Iterable[BatchedArray | np.ndarray],
) -> None:
    assert objects_are_equal(
        ba.batched_array([[0, 1, 2], [4, 5, 6]]).concatenate(arrays, axis=1),
        ba.batched_array([[0, 1, 2, 10, 11], [4, 5, 6, 12, 13]]),
    )


def test_batched_array_concatenate_axis_none() -> None:
    assert objects_are_equal(
        ba.batched_array([[0, 1, 2], [4, 5, 6]]).concatenate(
            [ba.batched_array([[10, 11, 12], [13, 14, 15]])], axis=None
        ),
        np.array([0, 1, 2, 4, 5, 6, 10, 11, 12, 13, 14, 15]),
    )


def test_batched_array_concatenate_custom_axes() -> None:
    assert objects_are_equal(
        ba.batched_array([[0, 4], [1, 5], [2, 6]], batch_axis=1).concatenate(
            [ba.batched_array([[10, 12], [11, 13], [14, 15]], batch_axis=1)], axis=1
        ),
        ba.batched_array(
            [[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]],
            batch_axis=1,
        ),
    )


def test_batched_array_concatenate_empty() -> None:
    assert objects_are_equal(ba.ones((2, 3)).concatenate([]), ba.ones((2, 3)))


def test_batched_array_concatenate_different_axes() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.concatenate([ba.zeros((2, 2), batch_axis=1)])


@pytest.mark.parametrize(
    "arrays",
    (
        [ba.batched_array([[10, 11, 12], [13, 14, 15]])],
        [np.array([[10, 11, 12], [13, 14, 15]])],
        (ba.batched_array([[10, 11, 12], [13, 14, 15]]),),
        [ba.batched_array([[10, 11, 12]]), ba.batched_array([[13, 14, 15]])],
        [ba.batched_array([[10, 11, 12]]), np.array([[13, 14, 15]])],
    ),
)
def test_batched_array_concatenate__axis_0(
    arrays: Iterable[BatchedArray | np.ndarray],
) -> None:
    array = ba.batched_array([[0, 1, 2], [4, 5, 6]])
    array.concatenate_(arrays, axis=0)
    assert objects_are_equal(
        array, ba.batched_array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]])
    )


@pytest.mark.parametrize(
    "arrays",
    (
        [ba.batched_array([[10, 11], [12, 13]])],
        [np.array([[10, 11], [12, 13]])],
        (ba.batched_array([[10, 11], [12, 13]]),),
        [ba.batched_array([[10], [12]]), ba.batched_array([[11], [13]])],
        [ba.batched_array([[10], [12]]), np.array([[11], [13]])],
    ),
)
def test_batched_array_concatenate__axis_1(
    arrays: Iterable[BatchedArray | np.ndarray],
) -> None:
    array = ba.batched_array([[0, 1, 2], [4, 5, 6]])
    array.concatenate_(arrays, axis=1)
    assert objects_are_equal(array, ba.batched_array([[0, 1, 2, 10, 11], [4, 5, 6, 12, 13]]))


def test_batched_array_concatenate__custom_axes() -> None:
    array = ba.batched_array([[0, 4], [1, 5], [2, 6]], batch_axis=1)
    array.concatenate_([ba.batched_array([[10, 12], [11, 13], [14, 15]], batch_axis=1)], axis=1)
    assert objects_are_equal(
        array,
        ba.batched_array(
            [[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]],
            batch_axis=1,
        ),
    )


def test_batched_array_concatenate__empty() -> None:
    array = ba.ones((2, 3))
    array.concatenate_([])
    assert objects_are_equal(array, ba.ones((2, 3)))


def test_batched_array_concatenate__different_axes() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.concatenate_([ba.zeros(shape=(2, 2), batch_axis=1)])


@pytest.mark.parametrize(
    "arrays",
    (
        [ba.batched_array([[10, 11, 12], [13, 14, 15]])],
        [np.array([[10, 11, 12], [13, 14, 15]])],
        (ba.batched_array([[10, 11, 12], [13, 14, 15]]),),
        [ba.batched_array([[10, 11, 12]]), ba.batched_array([[13, 14, 15]])],
        [ba.batched_array([[10, 11, 12]]), np.array([[13, 14, 15]])],
    ),
)
def test_batched_array_concatenate_along_batch(
    arrays: Iterable[BatchedArray | np.ndarray],
) -> None:
    assert objects_are_equal(
        ba.batched_array([[0, 1, 2], [4, 5, 6]]).concatenate_along_batch(arrays),
        ba.batched_array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]),
    )


def test_batched_array_concatenate_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.batched_array([[0, 4], [1, 5], [2, 6]], batch_axis=1).concatenate_along_batch(
            [ba.batched_array([[10, 12], [11, 13], [14, 15]], batch_axis=1)]
        ),
        ba.batched_array(
            [[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]],
            batch_axis=1,
        ),
    )


def test_batched_array_concatenate_along_batch_empty() -> None:
    assert objects_are_equal(ba.ones((2, 3)).concatenate_along_batch([]), ba.ones((2, 3)))


def test_batched_array_concatenate_along_batch_different_axes() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.concatenate_along_batch([ba.zeros((2, 2), batch_axis=1)])


@pytest.mark.parametrize(
    "arrays",
    (
        [ba.batched_array([[10, 11, 12], [13, 14, 15]])],
        [np.array([[10, 11, 12], [13, 14, 15]])],
        (ba.batched_array([[10, 11, 12], [13, 14, 15]]),),
        [ba.batched_array([[10, 11, 12]]), ba.batched_array([[13, 14, 15]])],
        [ba.batched_array([[10, 11, 12]]), np.array([[13, 14, 15]])],
    ),
)
def test_batched_array_concatenate_along_batch_(
    arrays: Iterable[BatchedArray | np.ndarray],
) -> None:
    array = ba.batched_array([[0, 1, 2], [4, 5, 6]])
    array.concatenate_along_batch_(arrays)
    assert objects_are_equal(
        array, ba.batched_array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]])
    )


def test_batched_array_concatenate_along_batch__custom_axes() -> None:
    array = ba.batched_array([[0, 4], [1, 5], [2, 6]], batch_axis=1)
    array.concatenate_along_batch_([ba.batched_array([[10, 12], [11, 13], [14, 15]], batch_axis=1)])
    assert objects_are_equal(
        array,
        ba.batched_array(
            [[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]],
            batch_axis=1,
        ),
    )


def test_batched_array_concatenate_along_batch__empty() -> None:
    array = ba.ones((2, 3))
    array.concatenate_along_batch_([])
    assert objects_are_equal(array, ba.ones((2, 3)))


def test_batched_array_concatenate_along_batch__different_axes() -> None:
    batch = ba.ones((2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.concatenate_along_batch_([ba.zeros((2, 2), batch_axis=1)])


##########################################################
#     Array manipulation routines | Splitting arrays     #
##########################################################


def test_batched_array_chunk_3() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).chunk(3),
        (
            ba.batched_array([[0, 1], [2, 3]]),
            ba.batched_array([[4, 5], [6, 7]]),
            ba.batched_array([[8, 9]]),
        ),
    )


def test_batched_array_chunk_5() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).chunk(5),
        (
            ba.batched_array([[0, 1]]),
            ba.batched_array([[2, 3]]),
            ba.batched_array([[4, 5]]),
            ba.batched_array([[6, 7]]),
            ba.batched_array([[8, 9]]),
        ),
    )


def test_batched_array_chunk_custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1).chunk(3, axis=1),
        (
            ba.batched_array([[0, 1], [5, 6]], batch_axis=1),
            ba.batched_array([[2, 3], [7, 8]], batch_axis=1),
            ba.batched_array([[4], [9]], batch_axis=1),
        ),
    )


def test_batched_array_chunk_incorrect_chunks() -> None:
    with pytest.raises(RuntimeError, match="chunk expects `chunks` to be greater than 0, got: 0"):
        BatchedArray(np.arange(10).reshape(5, 2)).chunk(0)


@pytest.mark.parametrize("index", [np.array([2, 0]), [2, 0], (2, 0)])
def test_batched_array_index_select(index: np.ndarray | Sequence[int]) -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).index_select(index=index, axis=0),
        ba.batched_array([[4, 5], [0, 1]]),
    )


def test_batched_array_index_select_axis_1() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)).index_select(index=(2, 0), axis=1),
        ba.batched_array([[2, 0], [7, 5]]),
    )


def test_batched_array_index_select_axis_none() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).index_select(index=[2, 0], axis=None),
        np.array([2, 0]),
    )


def test_batched_array_index_select_custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1).index_select(index=(2, 0), axis=0),
        ba.batched_array([[4, 5], [0, 1]], batch_axis=1),
    )


def test_batched_array_select_axis_0() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(30).reshape(5, 2, 3)).select(index=2, axis=0),
        np.array([[12, 13, 14], [15, 16, 17]]),
    )


def test_batched_array_select_axis_1() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(30).reshape(5, 2, 3)).select(index=0, axis=1),
        np.array([[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20], [24, 25, 26]]),
    )


def test_batched_array_select_axis_2() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(30).reshape(5, 2, 3)).select(index=1, axis=2),
        np.array([[1, 4], [7, 10], [13, 16], [19, 22], [25, 28]]),
    )


def test_batched_array_select_custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(30).reshape(5, 2, 3), batch_axis=1).select(index=2, axis=0),
        np.array([[12, 13, 14], [15, 16, 17]]),
    )


def test_batched_array_slice_along_axis() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2))
        .slice_along_axis()
        .allequal(ba.batched_array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]))
    )


def test_batched_array_slice_along_axis_start_2() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2))
        .slice_along_axis(start=2)
        .allequal(ba.batched_array([[4, 5], [6, 7], [8, 9]]))
    )


def test_batched_array_slice_along_axis_stop_3() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2))
        .slice_along_axis(stop=3)
        .allequal(ba.batched_array([[0, 1], [2, 3], [4, 5]]))
    )


def test_batched_array_slice_along_axis_stop_100() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2))
        .slice_along_axis(stop=100)
        .allequal(ba.batched_array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]))
    )


def test_batched_array_slice_along_axis_step_2() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2))
        .slice_along_axis(step=2)
        .allequal(ba.batched_array([[0, 1], [4, 5], [8, 9]]))
    )


def test_batched_array_slice_along_axis_start_1_stop_4_step_2() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2))
        .slice_along_axis(start=1, stop=4, step=2)
        .allequal(ba.batched_array([[2, 3], [6, 7]]))
    )


def test_batched_array_slice_along_axis_custom_axes() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1)
        .slice_along_axis(start=2)
        .allequal(ba.batched_array([[4, 5], [6, 7], [8, 9]], batch_axis=1))
    )


def test_batched_array_slice_along_axis_batch_axis_1() -> None:
    assert (
        BatchedArray(np.arange(20).reshape(2, 5, 2))
        .slice_along_axis(axis=1, start=2)
        .allequal(ba.batched_array([[[4, 5], [6, 7], [8, 9]], [[14, 15], [16, 17], [18, 19]]]))
    )


def test_batched_array_slice_along_axis_batch_axis_2() -> None:
    assert (
        BatchedArray(np.arange(20).reshape(2, 2, 5))
        .slice_along_axis(axis=2, start=2)
        .allequal(ba.batched_array([[[2, 3, 4], [7, 8, 9]], [[12, 13, 14], [17, 18, 19]]]))
    )


def test_batched_array_split_along_axis_split_size_1() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).split_along_axis(1),
        (
            ba.batched_array([[0, 1]]),
            ba.batched_array([[2, 3]]),
            ba.batched_array([[4, 5]]),
            ba.batched_array([[6, 7]]),
            ba.batched_array([[8, 9]]),
        ),
    )


def test_batched_array_split_along_axis_split_size_2() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).split_along_axis(2),
        (
            ba.batched_array([[0, 1], [2, 3]]),
            ba.batched_array([[4, 5], [6, 7]]),
            ba.batched_array([[8, 9]]),
        ),
    )


def test_batched_array_split_along_axis_custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1).split_along_axis(2, axis=1),
        (
            ba.batched_array([[0, 1], [5, 6]], batch_axis=1),
            ba.batched_array([[2, 3], [7, 8]], batch_axis=1),
            ba.batched_array([[4], [9]], batch_axis=1),
        ),
    )


def test_batched_array_split_along_axis_split_list() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).split_along_axis([2, 2, 1]),
        (
            ba.batched_array([[0, 1], [2, 3]]),
            ba.batched_array([[4, 5], [6, 7]]),
            ba.batched_array([[8, 9]]),
        ),
    )


##############################################################
#     Array manipulation routines | Rearranging elements     #
##############################################################


@pytest.mark.parametrize("permutation", (np.array([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_array_permute_along_axis_0(permutation: Sequence[int] | np.ndarray) -> None:
    assert (
        BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
        .permute_along_axis(permutation, axis=0)
        .allequal(BatchedArray(np.array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))
    )


@pytest.mark.parametrize(
    "permutation", (np.array([2, 4, 1, 3, 0]), [2, 4, 1, 3, 0], (2, 4, 1, 3, 0))
)
def test_batched_array_permute_along_axis_1(permutation: Sequence[int] | np.ndarray) -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .permute_along_axis(permutation, axis=1)
        .allequal(BatchedArray(np.array([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])))
    )


def test_batched_array_permute_along_axis_custom_axes() -> None:
    assert (
        BatchedArray(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), batch_axis=1)
        .permute_along_axis(np.array([2, 1, 3, 0]), axis=1)
        .allequal(BatchedArray(np.array([[2, 1, 3, 0], [6, 5, 7, 4]]), batch_axis=1))
    )


@pytest.mark.parametrize("permutation", (np.array([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_array_permute_along_axis__0(permutation: Sequence[int] | np.ndarray) -> None:
    batch = BatchedArray(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch.permute_along_axis_(permutation, axis=0)
    assert batch.allequal(BatchedArray(np.array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))


@pytest.mark.parametrize(
    "permutation", (np.array([2, 4, 1, 3, 0]), [2, 4, 1, 3, 0], (2, 4, 1, 3, 0))
)
def test_batched_array_permute_along_axis__1(permutation: Sequence[int] | np.ndarray) -> None:
    batch = BatchedArray(np.arange(10).reshape(2, 5))
    batch.permute_along_axis_(permutation, axis=1)
    assert batch.allequal(BatchedArray(np.array([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])))


def test_batched_array_permute_along_axis__custom_axes() -> None:
    batch = BatchedArray(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), batch_axis=1)
    batch.permute_along_axis_(np.array([2, 1, 3, 0]), axis=1)
    assert batch.allequal(BatchedArray(np.array([[2, 1, 3, 0], [6, 5, 7, 4]]), batch_axis=1))


@patch("redcat.ba2.core.setup_rng", MOCK_PERMUTATION4)
def test_batched_array_shuffle_along_axis() -> None:
    assert (
        ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        .shuffle_along_axis(axis=0)
        .allequal(ba.batched_array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]]))
    )


@patch("redcat.ba2.core.setup_rng", MOCK_PERMUTATION4)
def test_batched_array_shuffle_along_axis_custom_axes() -> None:
    assert (
        ba.batched_array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], batch_axis=1)
        .shuffle_along_axis(axis=1)
        .allequal(ba.batched_array([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]], batch_axis=1))
    )


def test_batched_array_shuffle_along_axis_same_random_seed() -> None:
    batch = ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    assert batch.shuffle_along_axis(axis=0, rng=np.random.default_rng(1)).allequal(
        batch.shuffle_along_axis(axis=0, rng=np.random.default_rng(1))
    )


def test_batched_array_shuffle_along_axis_different_random_seeds() -> None:
    batch = ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    assert not batch.shuffle_along_axis(axis=0, rng=np.random.default_rng(1)).allequal(
        batch.shuffle_along_axis(axis=0, rng=np.random.default_rng(2))
    )


@patch("redcat.ba2.core.setup_rng", MOCK_PERMUTATION4)
def test_batched_array_shuffle_along_axis_() -> None:
    batch = ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    batch.shuffle_along_axis_(axis=0)
    assert batch.allequal(ba.batched_array([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]]))


@patch("redcat.ba2.core.setup_rng", MOCK_PERMUTATION4)
def test_batched_array_shuffle_along_axis__custom_axes() -> None:
    batch = ba.batched_array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], batch_axis=1)
    batch.shuffle_along_axis_(axis=1)
    assert batch.allequal(
        ba.batched_array([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]], batch_axis=1)
    )


def test_batched_array_shuffle_along_axis__same_random_seed() -> None:
    batch1 = ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    batch1.shuffle_along_axis_(axis=0, rng=np.random.default_rng(1))
    batch2 = ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    batch2.shuffle_along_axis_(axis=0, rng=np.random.default_rng(1))
    assert batch1.allequal(batch2)


def test_batched_array_shuffle_along_axis__different_random_seeds() -> None:
    batch1 = ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    batch1.shuffle_along_axis_(axis=0, rng=np.random.default_rng(1))
    batch2 = ba.batched_array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    batch2.shuffle_along_axis_(axis=0, rng=np.random.default_rng(2))
    assert not batch1.allequal(batch2)


##############################################
#     Math | Sums, products, differences     #
##############################################


def test_batched_array_cumprod() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5) + 1).cumprod(),
        np.array([1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]),
    )


def test_batched_array_cumprod_axis_0() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)).cumprod(axis=0),
        BatchedArray(np.asarray([[0, 1, 2, 3, 4], [0, 6, 14, 24, 36]])),
    )


def test_batched_array_cumprod_axis_1() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)).cumprod(axis=1),
        BatchedArray(np.array([[0, 0, 0, 0, 0], [5, 30, 210, 1680, 15120]])),
    )


def test_batched_array_cumprod_custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1).cumprod(axis=0),
        BatchedArray(np.array([[0, 1], [0, 3], [0, 15], [0, 105], [0, 945]]), batch_axis=1),
    )


def test_batched_array_cumprod_out() -> None:
    out = np.zeros((5, 2), dtype=np.int64)
    assert BatchedArray(np.arange(10).reshape(5, 2)).cumprod(axis=0, out=out) is out
    assert objects_are_equal(out, np.asarray([[0, 1], [0, 3], [0, 15], [0, 105], [0, 945]]))


def test_batched_array_cumprod_out_array() -> None:
    out = np.zeros(10)
    assert BatchedArray(np.arange(10).reshape(2, 5) + 1).cumprod(out=out) is out
    assert objects_are_equal(
        out, np.asarray([1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0])
    )


def test_batched_array_cumprod_along_batch() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)).cumprod_along_batch(),
        BatchedArray(np.array([[0, 1, 2, 3, 4], [0, 6, 14, 24, 36]])),
    )


def test_batched_array_cumprod_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1).cumprod_along_batch(),
        BatchedArray(np.array([[0, 0], [2, 6], [4, 20], [6, 42], [8, 72]]), batch_axis=1),
    )


def test_batched_array_cumsum() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5) + 1).cumsum(),
        np.array([1, 3, 6, 10, 15, 21, 28, 36, 45, 55]),
    )


def test_batched_array_cumsum_axis_0() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)).cumsum(axis=0),
        BatchedArray(np.asarray([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])),
    )


def test_batched_array_cumsum_axis_1() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)).cumsum(axis=1),
        BatchedArray(np.array([[0, 1, 3, 6, 10], [5, 11, 18, 26, 35]])),
    )


def test_batched_array_cumsum_custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1).cumsum(axis=0),
        BatchedArray(np.array([[0, 1], [2, 4], [6, 9], [12, 16], [20, 25]]), batch_axis=1),
    )


def test_batched_array_cumsum_out() -> None:
    out = np.zeros((5, 2), dtype=np.int64)
    assert BatchedArray(np.arange(10).reshape(5, 2)).cumsum(axis=0, out=out) is out
    assert objects_are_equal(out, np.asarray([[0, 1], [2, 4], [6, 9], [12, 16], [20, 25]]))


def test_batched_array_cumsum_out_array() -> None:
    out = np.zeros(10)
    assert BatchedArray(np.arange(10).reshape(2, 5) + 1).cumsum(out=out) is out
    assert objects_are_equal(
        out, np.asarray([1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0, 55.0])
    )


def test_batched_array_cumsum_along_batch() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)).cumsum_along_batch(),
        BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])),
    )


def test_batched_array_cumsum_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1).cumsum_along_batch(),
        BatchedArray(np.array([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), batch_axis=1),
    )


#################
#     Other     #
#################


def test_setup_rng() -> None:
    rng = np.random.default_rng()
    assert setup_rng(rng) is rng


def test_setup_rng_none() -> None:
    assert isinstance(setup_rng(None), np.random.Generator)
