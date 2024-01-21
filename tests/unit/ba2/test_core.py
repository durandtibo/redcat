from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal
from numpy._typing import DTypeLike

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
