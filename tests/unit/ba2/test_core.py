from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest
from coola import objects_are_equal
from numpy.typing import DTypeLike

from redcat import ba2 as ba
from redcat.ba2 import BatchedArray

DTYPES = (bool, int, float)
NUMERIC_DTYPES = [np.float64, np.int64]


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
    "other", (ba.array([[10, 11, 12], [13, 14, 15]]), np.array([[10, 11, 12], [13, 14, 15]]))
)
def test_batched_array_append(
    other: BatchedArray | np.ndarray,
) -> None:
    array = ba.array([[0, 1, 2], [4, 5, 6]])
    array.append(other)
    assert objects_are_equal(array, ba.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))


def test_batched_array_append_custom_axes() -> None:
    array = ba.array([[0, 4], [1, 5], [2, 6]], batch_axis=1)
    array.append(ba.array([[10, 12], [11, 13], [14, 15]], batch_axis=1))
    assert objects_are_equal(
        array,
        ba.array(
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
            ba.array([[0, 1]]),
            ba.array([[2, 3]]),
            ba.array([[4, 5]]),
            ba.array([[6, 7]]),
            ba.array([[8, 9]]),
        ),
    )


def test_batched_array_chunk_along_batch_3() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).chunk_along_batch(3),
        (
            ba.array([[0, 1], [2, 3]]),
            ba.array([[4, 5], [6, 7]]),
            ba.array([[8, 9]]),
        ),
    )


def test_batched_array_chunk_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1).chunk_along_batch(3),
        (
            ba.array([[0, 1], [5, 6]], batch_axis=1),
            ba.array([[2, 3], [7, 8]], batch_axis=1),
            ba.array([[4], [9]], batch_axis=1),
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
        [ba.array([[10, 11, 12], [13, 14, 15]])],
        [np.array([[10, 11, 12], [13, 14, 15]])],
        (ba.array([[10, 11, 12], [13, 14, 15]]),),
        [ba.array([[10, 11, 12]]), ba.array([[13, 14, 15]])],
        [ba.array([[10, 11, 12]]), np.array([[13, 14, 15]])],
    ),
)
def test_batched_array_extend(
    arrays: Iterable[BatchedArray | np.ndarray],
) -> None:
    array = ba.array([[0, 1, 2], [4, 5, 6]])
    array.extend(arrays)
    assert objects_are_equal(array, ba.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))


def test_batched_array_extend_custom_axes() -> None:
    array = ba.array([[0, 4], [1, 5], [2, 6]], batch_axis=1)
    array.extend([ba.array([[10, 12], [11, 13], [14, 15]], batch_axis=1)])
    assert objects_are_equal(
        array,
        ba.array(
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


# TODO: uncomment
# @pytest.mark.parametrize("batch_size,num_minibatches", ((1, 10), (2, 5), (3, 4), (4, 3)))
# def test_batched_array_get_num_minibatches_drop_last_false(
#     batch_size: int, num_minibatches: int
# ) -> None:
#     assert ba.ones(shape=(10, 2)).get_num_minibatches(batch_size) == num_minibatches
#
#
# @pytest.mark.parametrize("batch_size,num_minibatches", ((1, 10), (2, 5), (3, 3), (4, 2)))
# def test_batched_array_get_num_minibatches_drop_last_true(
#     batch_size: int, num_minibatches: int
# ) -> None:
#     assert (
#         ba.ones(shape=(10, 2)).get_num_minibatches(batch_size, drop_last=True)
#         == num_minibatches
#     )


def test_batched_array_split_along_batch_split_size_1() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).split_along_batch(1),
        (
            ba.array([[0, 1]]),
            ba.array([[2, 3]]),
            ba.array([[4, 5]]),
            ba.array([[6, 7]]),
            ba.array([[8, 9]]),
        ),
    )


def test_batched_array_split_along_batch_split_size_2() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).split_along_batch(2),
        (
            ba.array([[0, 1], [2, 3]]),
            ba.array([[4, 5], [6, 7]]),
            ba.array([[8, 9]]),
        ),
    )


def test_batched_array_split_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1).split_along_batch(2),
        (
            ba.array([[0, 1], [5, 6]], batch_axis=1),
            ba.array([[2, 3], [7, 8]], batch_axis=1),
            ba.array([[4], [9]], batch_axis=1),
        ),
    )


def test_batched_array_split_along_batch_split_list() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).split_along_batch([2, 2, 1]),
        (
            ba.array([[0, 1], [2, 3]]),
            ba.array([[4, 5], [6, 7]]),
            ba.array([[8, 9]]),
        ),
    )


def test_batched_array_summary() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5)).summary()
        == "BatchedArray(dtype=int64, shape=(2, 5), batch_axis=0)"
    )


def test_batched_array_to_data() -> None:
    assert objects_are_equal(ba.ones(shape=(2, 3)).to_data(), np.ones(shape=(2, 3)))


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


def test_batched_array_shape() -> None:
    assert ba.ones(shape=(2, 3)).shape == (2, 3)


#####################
#     Data type     #
#####################


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_array_dtype(dtype: np.dtype) -> None:
    assert ba.ones(shape=(2, 3), dtype=dtype).dtype == dtype


###################################
#     Arithmetical operations     #
###################################


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


########################################################
#     Array manipulation routines | Joining arrays     #
########################################################


@pytest.mark.parametrize(
    "arrays",
    (
        [ba.array([[10, 11, 12], [13, 14, 15]])],
        [np.array([[10, 11, 12], [13, 14, 15]])],
        (ba.array([[10, 11, 12], [13, 14, 15]]),),
        [ba.array([[10, 11, 12]]), ba.array([[13, 14, 15]])],
        [ba.array([[10, 11, 12]]), np.array([[13, 14, 15]])],
    ),
)
def test_batched_array_concatenate_dim_0(
    arrays: Iterable[BatchedArray | np.ndarray],
) -> None:
    assert objects_are_equal(
        ba.array([[0, 1, 2], [4, 5, 6]]).concatenate(arrays, axis=0),
        ba.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]),
    )


@pytest.mark.parametrize(
    "arrays",
    (
        [ba.array([[10, 11], [12, 13]])],
        [np.array([[10, 11], [12, 13]])],
        (ba.array([[10, 11], [12, 13]]),),
        [ba.array([[10], [12]]), ba.array([[11], [13]])],
        [ba.array([[10], [12]]), np.array([[11], [13]])],
    ),
)
def test_batched_array_concatenate_axis_1(
    arrays: Iterable[BatchedArray | np.ndarray],
) -> None:
    assert objects_are_equal(
        ba.array([[0, 1, 2], [4, 5, 6]]).concatenate(arrays, axis=1),
        ba.array([[0, 1, 2, 10, 11], [4, 5, 6, 12, 13]]),
    )


def test_batched_array_concatenate_axis_none() -> None:
    assert objects_are_equal(
        ba.array([[0, 1, 2], [4, 5, 6]]).concatenate(
            [ba.array([[10, 11, 12], [13, 14, 15]])], axis=None
        ),
        np.array([0, 1, 2, 4, 5, 6, 10, 11, 12, 13, 14, 15]),
        show_difference=True,
    )


def test_batched_array_concatenate_custom_axes() -> None:
    assert objects_are_equal(
        ba.array([[0, 4], [1, 5], [2, 6]], batch_axis=1).concatenate(
            [ba.array([[10, 12], [11, 13], [14, 15]], batch_axis=1)], axis=1
        ),
        ba.array(
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
        [ba.array([[10, 11, 12], [13, 14, 15]])],
        [np.array([[10, 11, 12], [13, 14, 15]])],
        (ba.array([[10, 11, 12], [13, 14, 15]]),),
        [ba.array([[10, 11, 12]]), ba.array([[13, 14, 15]])],
        [ba.array([[10, 11, 12]]), np.array([[13, 14, 15]])],
    ),
)
def test_batched_array_concatenate__dim_0(
    arrays: Iterable[BatchedArray | np.ndarray],
) -> None:
    array = ba.array([[0, 1, 2], [4, 5, 6]])
    array.concatenate_(arrays, axis=0)
    assert objects_are_equal(array, ba.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))


@pytest.mark.parametrize(
    "arrays",
    (
        [ba.array([[10, 11], [12, 13]])],
        [np.array([[10, 11], [12, 13]])],
        (ba.array([[10, 11], [12, 13]]),),
        [ba.array([[10], [12]]), ba.array([[11], [13]])],
        [ba.array([[10], [12]]), np.array([[11], [13]])],
    ),
)
def test_batched_array_concatenate__axis_1(
    arrays: Iterable[BatchedArray | np.ndarray],
) -> None:
    array = ba.array([[0, 1, 2], [4, 5, 6]])
    array.concatenate_(arrays, axis=1)
    assert objects_are_equal(array, ba.array([[0, 1, 2, 10, 11], [4, 5, 6, 12, 13]]))


def test_batched_array_concatenate__custom_axes() -> None:
    array = ba.array([[0, 4], [1, 5], [2, 6]], batch_axis=1)
    array.concatenate_([ba.array([[10, 12], [11, 13], [14, 15]], batch_axis=1)], axis=1)
    assert objects_are_equal(
        array,
        ba.array(
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
        [ba.array([[10, 11, 12], [13, 14, 15]])],
        [np.array([[10, 11, 12], [13, 14, 15]])],
        (ba.array([[10, 11, 12], [13, 14, 15]]),),
        [ba.array([[10, 11, 12]]), ba.array([[13, 14, 15]])],
        [ba.array([[10, 11, 12]]), np.array([[13, 14, 15]])],
    ),
)
def test_batched_array_concatenate_along_batch(
    arrays: Iterable[BatchedArray | np.ndarray],
) -> None:
    assert objects_are_equal(
        ba.array([[0, 1, 2], [4, 5, 6]]).concatenate_along_batch(arrays),
        ba.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]),
    )


def test_batched_array_concatenate_along_batch_custom_axes() -> None:
    assert objects_are_equal(
        ba.array([[0, 4], [1, 5], [2, 6]], batch_axis=1).concatenate_along_batch(
            [ba.array([[10, 12], [11, 13], [14, 15]], batch_axis=1)]
        ),
        ba.array(
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
        [ba.array([[10, 11, 12], [13, 14, 15]])],
        [np.array([[10, 11, 12], [13, 14, 15]])],
        (ba.array([[10, 11, 12], [13, 14, 15]]),),
        [ba.array([[10, 11, 12]]), ba.array([[13, 14, 15]])],
        [ba.array([[10, 11, 12]]), np.array([[13, 14, 15]])],
    ),
)
def test_batched_array_concatenate_along_batch_(
    arrays: Iterable[BatchedArray | np.ndarray],
) -> None:
    array = ba.array([[0, 1, 2], [4, 5, 6]])
    array.concatenate_along_batch_(arrays)
    assert objects_are_equal(array, ba.array([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))


def test_batched_array_concatenate_along_batch__custom_axes() -> None:
    array = ba.array([[0, 4], [1, 5], [2, 6]], batch_axis=1)
    array.concatenate_along_batch_([ba.array([[10, 12], [11, 13], [14, 15]], batch_axis=1)])
    assert objects_are_equal(
        array,
        ba.array(
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
            ba.array([[0, 1], [2, 3]]),
            ba.array([[4, 5], [6, 7]]),
            ba.array([[8, 9]]),
        ),
    )


def test_batched_array_chunk_5() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).chunk(5),
        (
            ba.array([[0, 1]]),
            ba.array([[2, 3]]),
            ba.array([[4, 5]]),
            ba.array([[6, 7]]),
            ba.array([[8, 9]]),
        ),
    )


def test_batched_array_chunk_custom_dims() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1).chunk(3, axis=1),
        (
            ba.array([[0, 1], [5, 6]], batch_axis=1),
            ba.array([[2, 3], [7, 8]], batch_axis=1),
            ba.array([[4], [9]], batch_axis=1),
        ),
    )


def test_batched_array_chunk_incorrect_chunks() -> None:
    with pytest.raises(RuntimeError, match="chunk expects `chunks` to be greater than 0, got: 0"):
        BatchedArray(np.arange(10).reshape(5, 2)).chunk(0)


def test_batched_array_split_along_axis_split_size_1() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).split_along_axis(1),
        (
            ba.array([[0, 1]]),
            ba.array([[2, 3]]),
            ba.array([[4, 5]]),
            ba.array([[6, 7]]),
            ba.array([[8, 9]]),
        ),
    )


def test_batched_array_split_along_axis_split_size_2() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).split_along_axis(2),
        (
            ba.array([[0, 1], [2, 3]]),
            ba.array([[4, 5], [6, 7]]),
            ba.array([[8, 9]]),
        ),
    )


def test_batched_array_split_along_axis_custom_dims() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1).split_along_axis(2, axis=1),
        (
            ba.array([[0, 1], [5, 6]], batch_axis=1),
            ba.array([[2, 3], [7, 8]], batch_axis=1),
            ba.array([[4], [9]], batch_axis=1),
        ),
    )


def test_batched_array_split_along_axis_split_list() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2)).split_along_axis([2, 2, 1]),
        (
            ba.array([[0, 1], [2, 3]]),
            ba.array([[4, 5], [6, 7]]),
            ba.array([[8, 9]]),
        ),
    )
