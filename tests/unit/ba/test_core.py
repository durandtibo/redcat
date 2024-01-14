from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal
from numpy.typing import ArrayLike, DTypeLike

from redcat import ba
from redcat.ba import BatchedArray, arrays_share_data

DTYPES = (bool, int, float)

future_test = pytest.mark.skipif(True, reason="Future tests")


def test_batched_array_repr() -> None:
    assert repr(BatchedArray(np.arange(3))) == "array([0, 1, 2], batch_axis=0)"


def test_batched_array_str() -> None:
    assert str(BatchedArray(np.arange(3))) == "[0 1 2]\nwith batch_axis=0"


def test_batched_array_constructor() -> None:
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    array = BatchedArray(x)
    assert arrays_share_data(array, x)
    assert np.array_equal(array, x)
    assert array.batch_axis == 0


@pytest.mark.parametrize(
    "data",
    [
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
    ],
)
def test_batched_array_explicit_constructor_call(data: ArrayLike) -> None:
    array = BatchedArray(data)
    assert np.array_equal(array, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float))
    assert array.batch_axis == 0


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_batched_array_explicit_constructor_call_batch_axis(batch_axis: int) -> None:
    array = BatchedArray(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float), batch_axis)
    assert np.array_equal(array, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float))
    assert array.batch_axis == batch_axis


def test_batched_array_view_casting() -> None:
    array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    barray = array.view(BatchedArray)
    assert barray.allequal(BatchedArray(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)))


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_batched_array_new_from_template(batch_axis: int) -> None:
    array = BatchedArray(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float), batch_axis)
    assert array[1:].allequal(
        BatchedArray(np.array([[3.0, 4.0], [5.0, 6.0]], dtype=float), batch_axis)
    )


def test_batched_array_init_incorrect_data_axis() -> None:
    with pytest.raises(RuntimeError, match=r"data needs at least 1 axis \(received: 0\)"):
        BatchedArray(np.array(2))


@pytest.mark.parametrize("batch_size", (1, 2))
def test_batched_array_batch_size(batch_size: int) -> None:
    assert BatchedArray(np.arange(batch_size)).batch_size == batch_size


def test_batched_array_dtype() -> None:
    assert ba.ones(shape=(2, 3)).dtype == float


def test_batched_array_ndim() -> None:
    assert ba.ones(shape=(2, 3)).ndim == 2


def test_batched_array_shape() -> None:
    assert ba.ones(shape=(2, 3)).shape == (2, 3)


def test_batched_array_size() -> None:
    assert ba.ones(shape=(2, 3)).size == 6


#################################
#     Conversion operations     #
#################################


def test_batched_array_astype() -> None:
    assert (
        BatchedArray(np.ones(shape=(2, 3)))
        .astype(bool)
        .allequal(BatchedArray(np.ones(shape=(2, 3), dtype=bool)))
    )


def test_batched_array_astype_custom_axis() -> None:
    assert (
        BatchedArray(np.ones(shape=(2, 3)), batch_axis=1)
        .astype(bool)
        .allequal(BatchedArray(np.ones(shape=(2, 3), dtype=bool), batch_axis=1))
    )


###############################
#     Creation operations     #
###############################


def test_batched_array_copy() -> None:
    batch = ba.ones(shape=(2, 3))
    clone = batch.copy()
    # batch.add_(1)
    batch += 1
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=2.0))
    assert clone.allequal(ba.ones(shape=(2, 3)))


def test_batched_array_copy_custom_axes() -> None:
    assert ba.ones(shape=(2, 3), batch_axis=1).copy().allequal(ba.ones(shape=(2, 3), batch_axis=1))


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


@pytest.mark.parametrize("fill_value", (1, 2.0, True))
def test_batched_array_new_full_fill_value(fill_value: float | int | bool) -> None:
    assert (
        ba.zeros((2, 3))
        .new_full(fill_value)
        .allequal(ba.full(shape=(2, 3), fill_value=fill_value, dtype=float))
    )


def test_batched_array_new_full_custom_axes() -> None:
    assert (
        ba.zeros(shape=(3, 2), batch_axis=1)
        .new_full(2.0)
        .allequal(ba.full(shape=(3, 2), fill_value=2.0, batch_axis=1))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_array_new_full_dtype(dtype: np.dtype) -> None:
    assert (
        ba.zeros(shape=(2, 3), dtype=dtype)
        .new_full(2.0)
        .allequal(ba.full(shape=(2, 3), fill_value=2.0, dtype=dtype))
    )


@pytest.mark.parametrize("batch_size", (1, 2))
def test_batched_array_new_full_custom_batch_size(batch_size: int) -> None:
    assert (
        ba.zeros(shape=(2, 3))
        .new_full(2.0, batch_size=batch_size)
        .allequal(ba.full(shape=(batch_size, 3), fill_value=2.0))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_array_new_full_custom_dtype(dtype: np.dtype) -> None:
    assert (
        ba.zeros(shape=(2, 3))
        .new_full(2.0, dtype=dtype)
        .allequal(ba.full(shape=(2, 3), fill_value=2.0, dtype=dtype))
    )


def test_batched_array_new_ones() -> None:
    assert ba.zeros(shape=(2, 3)).new_ones().allequal(ba.ones(shape=(2, 3)))


def test_batched_array_new_ones_custom_axes() -> None:
    assert (
        ba.zeros(shape=(3, 2), batch_axis=1)
        .new_ones()
        .allequal(ba.ones(shape=(3, 2), batch_axis=1))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_array_new_ones_dtype(dtype: np.dtype) -> None:
    assert (
        ba.zeros(shape=(2, 3), dtype=dtype).new_ones().allequal(ba.ones(shape=(2, 3), dtype=dtype))
    )


@pytest.mark.parametrize("batch_size", (1, 2))
def test_batched_array_new_ones_custom_batch_size(batch_size: int) -> None:
    assert ba.zeros((2, 3)).new_ones(batch_size=batch_size).allequal(ba.ones((batch_size, 3)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_array_new_ones_custom_dtype(dtype: np.dtype) -> None:
    assert ba.zeros((2, 3)).new_ones(dtype=dtype).allequal(ba.ones(shape=(2, 3), dtype=dtype))


def test_batched_array_new_zeros() -> None:
    assert ba.ones(shape=(2, 3)).new_zeros().allequal(ba.zeros(shape=(2, 3)))


def test_batched_array_new_zeros_custom_axes() -> None:
    assert ba.ones(shape=(3, 2), batch_axis=1).new_zeros().allequal(ba.zeros((3, 2), batch_axis=1))


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_array_new_zeros_dtype(dtype: np.dtype) -> None:
    assert (
        ba.ones(shape=(2, 3), dtype=dtype).new_zeros().allequal(ba.zeros(shape=(2, 3), dtype=dtype))
    )


@pytest.mark.parametrize("batch_size", (1, 2))
def test_batched_array_new_zeros_custom_batch_size(batch_size: int) -> None:
    assert (
        ba.ones(shape=(2, 3)).new_zeros(batch_size=batch_size).allequal(ba.zeros((batch_size, 3)))
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_batched_array_new_zeros_custom_dtype(dtype: np.dtype) -> None:
    assert (
        ba.ones(shape=(2, 3)).new_zeros(dtype=dtype).allequal(ba.zeros(shape=(2, 3), dtype=dtype))
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


#################################
#     Comparison operations     #
#################################


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
    assert (BatchedArray(np.arange(10).reshape(2, 5)) == other).allequal(
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [True, False, False, False, False]],
                dtype=bool,
            ),
        )
    )


def test_batched_array__eq__custom_axis() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        == ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1)
    ).allequal(
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [True, False, False, False, False]],
                dtype=bool,
            ),
            batch_axis=1,
        )
    )


@future_test
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
    assert (BatchedArray(np.arange(10).reshape(2, 5)) >= other).allequal(
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [True, True, True, True, True]],
                dtype=bool,
            ),
        )
    )


def test_batched_array__ge__custom_axis() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        >= ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1)
    ).allequal(
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [True, True, True, True, True]],
                dtype=bool,
            ),
            batch_axis=1,
        )
    )


@future_test
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
    assert (BatchedArray(np.arange(10).reshape(2, 5)) > other).allequal(
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [False, True, True, True, True]],
                dtype=bool,
            ),
        )
    )


def test_batched_array__gt__custom_axis() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        > ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1)
    ).allequal(
        BatchedArray(
            np.array(
                [[False, False, False, False, False], [False, True, True, True, True]],
                dtype=bool,
            ),
            batch_axis=1,
        )
    )


@future_test
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
    assert (BatchedArray(np.arange(10).reshape(2, 5)) <= other).allequal(
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [True, False, False, False, False]],
                dtype=bool,
            ),
        )
    )


def test_batched_array__le__custom_axis() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        <= ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1)
    ).allequal(
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [True, False, False, False, False]],
                dtype=bool,
            ),
            batch_axis=1,
        )
    )


@future_test
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
    assert (BatchedArray(np.arange(10).reshape(2, 5)) < other).allequal(
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [False, False, False, False, False]],
                dtype=bool,
            ),
        )
    )


def test_batched_array__lt__custom_axis() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        < ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1)
    ).allequal(
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [False, False, False, False, False]],
                dtype=bool,
            ),
            batch_axis=1,
        )
    )


@future_test
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
    assert (BatchedArray(np.arange(10).reshape(2, 5)) != other).allequal(
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [False, True, True, True, True]],
                dtype=bool,
            ),
        )
    )


def test_batched_array__ne__custom_axis() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        != ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1)
    ).allequal(
        BatchedArray(
            np.array(
                [[True, True, True, True, True], [False, True, True, True, True]],
                dtype=bool,
            ),
            batch_axis=1,
        )
    )


@future_test
def test_batched_array__ne__different_axes() -> None:
    x1 = BatchedArray(np.arange(10).reshape(2, 5))
    x2 = ba.full(shape=(2, 5), fill_value=5.0, batch_axis=1)
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        x1.__ne__(x2)


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
        (ba.ones((2, 3)) + 0.5, 1),
        (ba.ones((2, 3)) + 0.05, 1e-1),
        (ba.ones((2, 3)) + 5e-3, 1e-2),
    ),
)
def test_batched_array_allclose_true_atol(array: BatchedArray, atol: float) -> None:
    assert ba.ones((2, 3)).allclose(array, atol=atol, rtol=0)


@pytest.mark.parametrize(
    ("array", "rtol"),
    (
        (ba.ones((2, 3)) + 0.5, 1),
        (ba.ones((2, 3)) + 0.05, 1e-1),
        (ba.ones((2, 3)) + 5e-3, 1e-2),
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
    [
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5.0),
        5,
        5.0,
    ],
)
def test_batched_array_eq(other: np.ndarray | int | float) -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .eq(other)
        .allequal(
            BatchedArray(
                np.array(
                    [[False, False, False, False, False], [True, False, False, False, False]],
                    dtype=bool,
                ),
            )
        )
    )


def test_batched_array_eq_custom_axes() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        .eq(ba.full(shape=(2, 5), fill_value=5, batch_axis=1))
        .allequal(
            BatchedArray(
                np.array(
                    [[False, False, False, False, False], [True, False, False, False, False]],
                    dtype=bool,
                ),
                batch_axis=1,
            )
        )
    )


@future_test
def test_batched_array_eq_different_axes() -> None:
    array = BatchedArray(np.arange(10).reshape(2, 5))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        array.eq(ba.full(shape=(2, 5), fill_value=5, batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5.0),
        5,
        5.0,
    ],
)
def test_batched_array_ge(other: np.ndarray | float) -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .ge(other)
        .allequal(
            BatchedArray(
                np.array(
                    [[False, False, False, False, False], [True, True, True, True, True]],
                    dtype=bool,
                ),
            )
        )
    )


def test_batched_array_ge_custom_axes() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        .ge(ba.full(shape=(2, 5), fill_value=5, batch_axis=1))
        .allequal(
            BatchedArray(
                np.array(
                    [[False, False, False, False, False], [True, True, True, True, True]],
                    dtype=bool,
                ),
                batch_axis=1,
            )
        )
    )


@future_test
def test_batched_array_ge_different_axes() -> None:
    array = BatchedArray(np.arange(10).reshape(2, 5))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        array.ge(ba.full(shape=(2, 5), fill_value=5, batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5.0),
        5,
        5.0,
    ],
)
def test_batched_array_gt(other: np.ndarray | float) -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .gt(other)
        .allequal(
            BatchedArray(
                np.array(
                    [[False, False, False, False, False], [False, True, True, True, True]],
                    dtype=bool,
                ),
            )
        )
    )


def test_batched_array_gt_custom_axes() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        .gt(ba.full(shape=(2, 5), fill_value=5, batch_axis=1))
        .allequal(
            BatchedArray(
                np.array(
                    [[False, False, False, False, False], [False, True, True, True, True]],
                    dtype=bool,
                ),
                batch_axis=1,
            )
        )
    )


@future_test
def test_batched_array_gt_different_axes() -> None:
    array = BatchedArray(np.arange(10).reshape(2, 5))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        array.gt(ba.full(shape=(2, 5), fill_value=5, batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5.0),
        5,
        5.0,
    ],
)
def test_batched_array_le(other: np.ndarray | float) -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .le(other)
        .allequal(
            BatchedArray(
                np.array(
                    [[True, True, True, True, True], [True, False, False, False, False]],
                    dtype=bool,
                )
            )
        )
    )


def test_batched_array_le_custom_axes() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        .le(ba.full(shape=(2, 5), fill_value=5, batch_axis=1))
        .allequal(
            BatchedArray(
                np.array(
                    [[True, True, True, True, True], [True, False, False, False, False]],
                    dtype=bool,
                ),
                batch_axis=1,
            )
        )
    )


@future_test
def test_batched_array_le_different_axes() -> None:
    array = BatchedArray(np.arange(10).reshape(2, 5))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        array.le(ba.full(shape=(2, 5), fill_value=5, batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5.0),
        5,
        5.0,
    ],
)
def test_batched_array_lt(other: np.ndarray | float) -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .lt(other)
        .allequal(
            BatchedArray(
                np.array(
                    [[True, True, True, True, True], [False, False, False, False, False]],
                    dtype=bool,
                ),
            )
        )
    )


def test_batched_array_lt_custom_axes() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        .lt(ba.full(shape=(2, 5), fill_value=5, batch_axis=1))
        .allequal(
            BatchedArray(
                np.array(
                    [[True, True, True, True, True], [False, False, False, False, False]],
                    dtype=bool,
                ),
                batch_axis=1,
            )
        )
    )


@future_test
def test_batched_array_lt_different_axes() -> None:
    array = BatchedArray(np.arange(10).reshape(2, 5))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        array.lt(ba.full(shape=(2, 5), fill_value=5, batch_axis=1))


@pytest.mark.parametrize(
    "other",
    [
        ba.full(shape=(2, 5), fill_value=5.0),
        np.full(shape=(2, 5), fill_value=5.0),
        ba.full(shape=(2, 1), fill_value=5.0),
        5,
        5.0,
    ],
)
def test_batched_array_ne(other: np.ndarray | int | float) -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5))
        .ne(other)
        .allequal(
            BatchedArray(
                np.array(
                    [[True, True, True, True, True], [False, True, True, True, True]],
                    dtype=bool,
                ),
            )
        )
    )


def test_batched_array_ne_custom_axes() -> None:
    assert (
        BatchedArray(np.arange(10).reshape(2, 5), batch_axis=1)
        .ne(ba.full(shape=(2, 5), fill_value=5, batch_axis=1))
        .allequal(
            BatchedArray(
                np.array(
                    [[True, True, True, True, True], [False, True, True, True, True]],
                    dtype=bool,
                ),
                batch_axis=1,
            )
        )
    )


@future_test
def test_batched_array_ne_different_axes() -> None:
    array = BatchedArray(np.arange(10).reshape(2, 5))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        array.ne(ba.full(shape=(2, 5), fill_value=5, batch_axis=1))


@pytest.mark.parametrize(
    "array",
    [
        BatchedArray(np.array([[1.001, 0.5, 2.0], [0.0, -2.5, -0.5]])),
        np.array([[1.001, 0.5, 2.0], [0.0, -2.5, -0.5]]),
    ],
)
def test_batched_array_isclose(array: np.ndarray) -> None:
    assert (
        BatchedArray(np.array([[1.0, 0.0, 2.0], [0.0, -2.0, -1.0]]))
        .isclose(array, atol=0.01)
        .allequal(BatchedArray(np.array([[True, False, True], [True, False, False]], dtype=bool)))
    )


def test_batched_array_isclose_custom_axes() -> None:
    assert (
        BatchedArray(np.array([[1.0, 0.0, 2.0], [0.0, -2.0, -1.0]]), batch_axis=1)
        .isclose(
            BatchedArray(np.array([[1.001, 0.5, 2.0], [0.0, -2.5, -0.5]]), batch_axis=1), atol=0.01
        )
        .allequal(
            BatchedArray(
                np.array([[True, False, True], [True, False, False]], dtype=bool),
                batch_axis=1,
            )
        )
    )


@future_test
def test_batched_array_isclose_different_axes() -> None:
    batch1 = BatchedArray(np.array([[1.0, 0.0, 2.0], [0.0, -2.0, -1.0]]), batch_axis=1)
    batch2 = BatchedArray(np.array([[1.001, 0.5, 2.0], [0.0, -2.5, -0.5]]))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch1.isclose(batch2)


def test_batched_array_isinf() -> None:
    assert (
        BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
        .isinf()
        .allequal(BatchedArray(np.array([[False, False, True], [False, False, True]], dtype=bool)))
    )


def test_batched_array_isinf_custom_axes() -> None:
    assert (
        BatchedArray(
            np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]),
            batch_axis=1,
        )
        .isinf()
        .allequal(
            BatchedArray(
                np.array([[False, False, True], [False, False, True]], dtype=bool),
                batch_axis=1,
            )
        )
    )


def test_batched_array_isneginf() -> None:
    assert (
        BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
        .isneginf()
        .allequal(BatchedArray(np.array([[False, False, False], [False, False, True]], dtype=bool)))
    )


def test_batched_array_isneginf_custom_axes() -> None:
    assert (
        BatchedArray(
            np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]),
            batch_axis=1,
        )
        .isneginf()
        .allequal(
            BatchedArray(
                np.array([[False, False, False], [False, False, True]], dtype=bool),
                batch_axis=1,
            )
        )
    )


def test_batched_array_isposinf() -> None:
    assert (
        BatchedArray(np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
        .isposinf()
        .allequal(BatchedArray(np.array([[False, False, True], [False, False, False]], dtype=bool)))
    )


def test_batched_array_isposinf_custom_axes() -> None:
    assert (
        BatchedArray(
            np.array([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]),
            batch_axis=1,
        )
        .isposinf()
        .allequal(
            BatchedArray(
                np.array([[False, False, True], [False, False, False]], dtype=bool),
                batch_axis=1,
            )
        )
    )


def test_batched_array_isnan() -> None:
    assert (
        BatchedArray(np.array([[1.0, 0.0, float("nan")], [float("nan"), -2.0, -1.0]]))
        .isnan()
        .allequal(BatchedArray(np.array([[False, False, True], [True, False, False]], dtype=bool)))
    )


def test_batched_array_isnan_custom_axes() -> None:
    assert (
        BatchedArray(
            np.array([[1.0, 0.0, float("nan")], [float("nan"), -2.0, -1.0]]),
            batch_axis=1,
        )
        .isnan()
        .allequal(
            BatchedArray(
                np.array([[False, False, True], [True, False, False]], dtype=bool),
                batch_axis=1,
            )
        )
    )


##################################################
#     Mathematical | arithmetical operations     #
##################################################


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


@future_test
def test_batched_array__add___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3), batch_axis=1)
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


@future_test
def test_batched_array__iadd___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3), batch_axis=1)
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


@future_test
def test_batched_array__floordiv___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3), batch_axis=1)
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


@future_test
def test_batched_array__ifloordiv___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3), batch_axis=1)
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
def test_batched_array__mul__(other: np.ndarray | int | float) -> None:
    assert (ba.ones(shape=(2, 3)) * other).allequal(ba.full(shape=(2, 3), fill_value=2.0))


def test_batched_array__mul___custom_axes() -> None:
    assert (
        ba.ones(shape=(2, 3), batch_axis=1) * ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1)
    ).allequal(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))


@future_test
def test_batched_array__mul___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3), batch_axis=1)
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


@future_test
def test_batched_array__imul___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3), batch_axis=1)
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


@future_test
def test_batched_array__sub___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3), batch_axis=1)
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


@future_test
def test_batched_array__isub___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3), batch_axis=1)
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


@future_test
def test_batched_array__truediv___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3), batch_axis=1)
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


@future_test
def test_batched_array__itruediv___different_axes() -> None:
    x1 = ba.zeros(shape=(2, 3), batch_axis=1)
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


def test_batched_array_add_incorrect_batch_axis() -> None:
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
    batch = ba.ones(shape=(2, 3), dtype=int)
    batch.add_(ba.full(shape=(2, 3), fill_value=2.0, dtype=int), alpha=2)
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=5.0, dtype=int))


def test_batched_array_add__custom_axes() -> None:
    batch = ba.ones(shape=(2, 3), batch_axis=1)
    batch.add_(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=3.0, batch_axis=1))


def test_batched_array_add__incorrect_batch_axis() -> None:
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
def test_batched_array_sub(other: np.ndarray | int | float) -> None:
    assert ba.ones((2, 3)).sub(other).allequal(ba.full(shape=(2, 3), fill_value=-1.0))


@pytest.mark.parametrize("dtype", [float, int])
def test_batched_array_sub_alpha_2(dtype: DTypeLike) -> None:
    assert (
        ba.ones(shape=(2, 3), dtype=int)
        .sub(ba.full(shape=(2, 3), fill_value=2, dtype=int), alpha=2)
        .allequal(ba.full(shape=(2, 3), fill_value=-3, dtype=int))
    )


def test_batched_array_sub_custom_axess() -> None:
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


@pytest.mark.parametrize("dtype", [float, int])
def test_batched_array_sub__alpha_2(dtype: DTypeLike) -> None:
    batch = ba.ones(shape=(2, 3), dtype=dtype)
    batch.sub_(ba.full(shape=(2, 3), fill_value=2, dtype=dtype), alpha=2)
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=-3, dtype=dtype))


def test_batched_array_sub__custom_axes() -> None:
    batch = ba.ones(shape=(2, 3), batch_axis=1)
    batch.sub_(ba.full(shape=(2, 3), fill_value=2.0, batch_axis=1))
    assert batch.allequal(ba.full(shape=(2, 3), fill_value=-1.0, batch_axis=1))


def test_batched_array_sub__incorrect_batch_axis() -> None:
    batch = ba.ones(shape=(2, 2))
    with pytest.raises(RuntimeError, match=r"The batch axes do not match."):
        batch.sub_(ba.ones(shape=(2, 2), batch_axis=1))


################################################
#     Mathematical | advanced arithmetical     #
################################################


def test_batched_array_argsort() -> None:
    assert objects_are_equal(
        BatchedArray(np.array([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).argsort(),
        BatchedArray(np.asarray([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]])),
    )


def test_batched_array_argsort_axis_0() -> None:
    assert objects_are_equal(
        BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])).argsort(axis=0),
        BatchedArray(np.asarray([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
    )


def test_batched_array_argsort_axis_1() -> None:
    assert objects_are_equal(
        BatchedArray(
            np.asarray(
                [
                    [[0, 1], [-2, 3], [-4, 5], [-6, 7], [-8, 9]],
                    [[10, -11], [12, -13], [14, -15], [16, -17], [18, -19]],
                ]
            )
        ).argsort(axis=1),
        BatchedArray(
            np.asarray(
                [
                    [[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]],
                    [[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]],
                ]
            )
        ),
    )


def test_batched_array_argsort_custom_axis() -> None:
    assert objects_are_equal(
        BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), batch_axis=1).argsort(
            axis=0
        ),
        BatchedArray(np.asarray([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]), batch_axis=1),
    )


def test_batched_array_argsort_along_batch() -> None:
    assert objects_are_equal(
        BatchedArray(np.asarray([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])).argsort_along_batch(),
        BatchedArray(np.asarray([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
    )


def test_batched_array_argsort_along_batch_custom_axis() -> None:
    assert objects_are_equal(
        BatchedArray(
            np.asarray([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_axis=1
        ).argsort_along_batch(),
        BatchedArray(np.asarray([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]), batch_axis=1),
    )


@future_test
def test_batched_array_cumprod() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)).cumprod(),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
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


def test_batched_array_cumprod_custom_axis() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1).cumprod(axis=0),
        BatchedArray(np.array([[0, 1], [0, 3], [0, 15], [0, 105], [0, 945]]), batch_axis=1),
        show_difference=True,
    )


def test_batched_array_cumprod_along_batch() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)).cumprod_along_batch(),
        BatchedArray(np.array([[0, 1, 2, 3, 4], [0, 6, 14, 24, 36]])),
    )


def test_batched_array_cumprod_along_batch_custom_axis() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1).cumprod_along_batch(),
        BatchedArray(np.array([[0, 0], [2, 6], [4, 20], [6, 42], [8, 72]]), batch_axis=1),
    )


@future_test
def test_batched_array_cumsum() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)).cumsum(),
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
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


def test_batched_array_cumsum_custom_axis() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1).cumsum(axis=0),
        BatchedArray(np.array([[0, 1], [2, 4], [6, 9], [12, 16], [20, 25]]), batch_axis=1),
        show_difference=True,
    )


def test_batched_array_cumsum_along_batch() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(2, 5)).cumsum_along_batch(),
        BatchedArray(np.array([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])),
    )


def test_batched_array_cumsum_along_batch_custom_axis() -> None:
    assert objects_are_equal(
        BatchedArray(np.arange(10).reshape(5, 2), batch_axis=1).cumsum_along_batch(),
        BatchedArray(np.array([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), batch_axis=1),
    )
