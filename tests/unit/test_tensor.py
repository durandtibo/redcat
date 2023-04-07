from typing import Any, Union

import numpy as np
import torch
from pytest import mark, raises
from torch.overrides import is_tensor_like

from redcat import BatchedTensor, BatchedTensorSeq
from redcat.base import BaseBatchedTensor
from redcat.tensor import check_data_and_dim
from redcat.utils import get_available_devices

DTYPES = (torch.bool, torch.int, torch.long, torch.float, torch.double)


def test_batched_tensor_is_tensor_like() -> None:
    assert is_tensor_like(BatchedTensor(torch.ones(2, 3)))


@mark.parametrize(
    "data",
    (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
    ),
)
def test_batched_tensor_init_data(data: Any) -> None:
    assert BatchedTensor(data).data.equal(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float)
    )


def test_batched_tensor_init_incorrect_data_dim() -> None:
    with raises(RuntimeError, match=r"data needs at least 1 dimensions \(received: 0\)"):
        BatchedTensor(torch.tensor(2))


@mark.parametrize("batch_dim", (-1, 1, 2))
def test_batched_tensor_init_incorrect_batch_dim(batch_dim: int) -> None:
    with raises(
        RuntimeError, match=r"Incorrect batch_dim \(.*\) but the value should be in \[0, 0\]"
    ):
        BatchedTensor(torch.ones(2), batch_dim=batch_dim)


@mark.parametrize("batch_size", (1, 2))
def test_batched_tensor_batch_size(batch_size: int) -> None:
    assert BatchedTensor(torch.arange(batch_size)).batch_size == batch_size


def test_batched_tensor_data() -> None:
    assert BatchedTensor(torch.arange(3)).data.equal(torch.tensor([0, 1, 2]))


def test_batched_tensor_repr() -> None:
    assert repr(BatchedTensor(torch.arange(3))) == "tensor([0, 1, 2], batch_dim=0)"


@mark.parametrize("device", get_available_devices())
def test_batched_tensor_device(device: str) -> None:
    device = torch.device(device)
    assert BatchedTensor(torch.ones(2, 3, device=device)).device == device


#################################
#     Conversion operations     #
#################################


def test_batched_tensor_contiguous() -> None:
    batch = BatchedTensor(torch.ones(3, 2).transpose(0, 1))
    assert not batch.is_contiguous()
    cont = batch.contiguous()
    assert cont.equal(BatchedTensor(torch.ones(2, 3)))
    assert cont.is_contiguous()


def test_batched_tensor_contiguous_custom_dim() -> None:
    batch = BatchedTensor(torch.ones(3, 2).transpose(0, 1), batch_dim=1)
    assert not batch.is_contiguous()
    cont = batch.contiguous()
    assert cont.equal(BatchedTensor(torch.ones(2, 3), batch_dim=1))
    assert cont.is_contiguous()


def test_batched_tensor_contiguous_memory_format() -> None:
    batch = BatchedTensor(torch.ones(2, 3, 4, 5))
    assert not batch.data.is_contiguous(memory_format=torch.channels_last)
    cont = batch.contiguous(memory_format=torch.channels_last)
    assert cont.equal(BatchedTensor(torch.ones(2, 3, 4, 5)))
    assert cont.is_contiguous(memory_format=torch.channels_last)


def test_batched_tensor_to() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3))
        .to(dtype=torch.bool)
        .equal(BatchedTensor(torch.ones(2, 3, dtype=torch.bool)))
    )


def test_batched_tensor_to_custom_dim() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .to(dtype=torch.bool)
        .equal(BatchedTensor(torch.ones(2, 3, dtype=torch.bool), batch_dim=1))
    )


#################
#     dtype     #
#################


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_dtype(dtype: torch.dtype) -> None:
    assert BatchedTensor(torch.ones(2, 3, dtype=dtype)).dtype == dtype


def test_batched_tensor_bool() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3))
        .bool()
        .equal(BatchedTensor(torch.ones(2, 3, dtype=torch.bool)))
    )


def test_batched_tensor_bool_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .bool()
        .equal(BatchedTensor(torch.ones(2, 3, dtype=torch.bool), batch_dim=1))
    )


def test_batched_tensor_double() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3))
        .double()
        .equal(BatchedTensor(torch.ones(2, 3, dtype=torch.double)))
    )


def test_batched_tensor_double_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .double()
        .equal(BatchedTensor(torch.ones(2, 3, dtype=torch.double), batch_dim=1))
    )


def test_batched_tensor_float() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3, dtype=torch.long))
        .float()
        .equal(BatchedTensor(torch.ones(2, 3, dtype=torch.float)))
    )


def test_batched_tensor_float_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3, dtype=torch.long), batch_dim=1)
        .float()
        .equal(BatchedTensor(torch.ones(2, 3, dtype=torch.float), batch_dim=1))
    )


def test_batched_tensor_int() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3))
        .int()
        .equal(BatchedTensor(torch.ones(2, 3, dtype=torch.int)))
    )


def test_batched_tensor_int_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .int()
        .equal(BatchedTensor(torch.ones(2, 3, dtype=torch.int), batch_dim=1))
    )


def test_batched_tensor_long() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3))
        .long()
        .equal(BatchedTensor(torch.ones(2, 3, dtype=torch.long)))
    )


def test_batched_tensor_long_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .long()
        .equal(BatchedTensor(torch.ones(2, 3, dtype=torch.long), batch_dim=1))
    )


###############################
#     Creation operations     #
###############################


def test_batched_tensor_clone() -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    clone = batch.clone()
    batch.data.add_(1)
    assert batch.equal(BatchedTensor(torch.full((2, 3), 2.0)))
    assert clone.equal(BatchedTensor(torch.ones(2, 3)))


def test_batched_tensor_clone_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .clone()
        .equal(BatchedTensor(torch.ones(2, 3), batch_dim=1))
    )


@mark.parametrize("fill_value", (1.5, 2.0, -1.0))
def test_batched_tensor_full_like(fill_value: float) -> None:
    assert (
        BatchedTensor(torch.zeros(2, 3))
        .full_like(fill_value)
        .equal(BatchedTensor(torch.full((2, 3), fill_value=fill_value)))
    )


def test_batched_tensor_full_like_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.zeros(3, 2), batch_dim=1)
        .full_like(fill_value=2.0)
        .equal(BatchedTensor(torch.full((3, 2), fill_value=2.0), batch_dim=1))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_full_like_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.zeros(2, 3, dtype=dtype))
        .full_like(fill_value=2.0)
        .equal(BatchedTensor(torch.full((2, 3), fill_value=2.0, dtype=dtype)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_full_like_target_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.zeros(2, 3))
        .full_like(fill_value=2.0, dtype=dtype)
        .equal(BatchedTensor(torch.full((2, 3), fill_value=2.0, dtype=dtype)))
    )


@mark.parametrize("fill_value", (1, 2.0, True))
def test_batched_tensor_new_full_fill_value(fill_value: Union[float, int, bool]) -> None:
    assert (
        BatchedTensor(torch.zeros(2, 3))
        .new_full(fill_value)
        .equal(BatchedTensor(torch.full((2, 3), fill_value, dtype=torch.float)))
    )


def test_batched_tensor_new_full_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.zeros(3, 2), batch_dim=1)
        .new_full(2.0)
        .equal(BatchedTensor(torch.full((3, 2), 2.0), batch_dim=1))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_new_full_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.zeros(2, 3, dtype=dtype))
        .new_full(2.0)
        .equal(BatchedTensor(torch.full((2, 3), 2.0, dtype=dtype)))
    )


@mark.parametrize("device", get_available_devices())
def test_batched_tensor_new_full_device(device: str) -> None:
    device = torch.device(device)
    assert (
        BatchedTensor(torch.zeros(2, 3))
        .new_full(2.0, device=device)
        .equal(BatchedTensor(torch.full((2, 3), 2.0, device=device)))
    )


@mark.parametrize("batch_size", (1, 2))
def test_batched_tensor_new_full_custom_batch_size(batch_size: int) -> None:
    assert (
        BatchedTensor(torch.zeros(2, 3))
        .new_full(2.0, batch_size=batch_size)
        .equal(BatchedTensor(torch.full((batch_size, 3), 2.0)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_new_full_custom_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.zeros(2, 3))
        .new_full(2.0, dtype=dtype)
        .equal(BatchedTensor(torch.full((2, 3), 2.0, dtype=dtype)))
    )


def test_batched_tensor_new_ones() -> None:
    assert BatchedTensor(torch.zeros(2, 3)).new_ones().equal(BatchedTensor(torch.ones(2, 3)))


def test_batched_tensor_new_ones_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.zeros(3, 2), batch_dim=1)
        .new_ones()
        .equal(BatchedTensor(torch.ones(3, 2), batch_dim=1))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_new_ones_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.zeros(2, 3, dtype=dtype))
        .new_ones()
        .equal(BatchedTensor(torch.ones(2, 3, dtype=dtype)))
    )


@mark.parametrize("device", get_available_devices())
def test_batched_tensor_new_ones_device(device: str) -> None:
    device = torch.device(device)
    assert (
        BatchedTensor(torch.zeros(2, 3))
        .new_ones(device=device)
        .equal(BatchedTensor(torch.ones(2, 3, device=device)))
    )


@mark.parametrize("batch_size", (1, 2))
def test_batched_tensor_new_ones_custom_batch_size(batch_size: int) -> None:
    assert (
        BatchedTensor(torch.zeros(2, 3))
        .new_ones(batch_size=batch_size)
        .equal(BatchedTensor(torch.ones(batch_size, 3)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_new_ones_custom_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.zeros(2, 3))
        .new_ones(dtype=dtype)
        .equal(BatchedTensor(torch.ones(2, 3, dtype=dtype)))
    )


def test_batched_tensor_new_zeros() -> None:
    assert BatchedTensor(torch.ones(2, 3)).new_zeros().equal(BatchedTensor(torch.zeros(2, 3)))


def test_batched_tensor_new_zeros_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.ones(3, 2), batch_dim=1)
        .new_zeros()
        .equal(BatchedTensor(torch.zeros(3, 2), batch_dim=1))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_new_zeros_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.ones(2, 3, dtype=dtype))
        .new_zeros()
        .equal(BatchedTensor(torch.zeros(2, 3, dtype=dtype)))
    )


@mark.parametrize("device", get_available_devices())
def test_batched_tensor_new_zeros_device(device: str) -> None:
    device = torch.device(device)
    assert (
        BatchedTensor(torch.ones(2, 3))
        .new_zeros(device=device)
        .equal(BatchedTensor(torch.zeros(2, 3, device=device)))
    )


@mark.parametrize("batch_size", (1, 2))
def test_batched_tensor_new_zeros_custom_batch_size(batch_size: int) -> None:
    assert (
        BatchedTensor(torch.ones(2, 3))
        .new_zeros(batch_size=batch_size)
        .equal(BatchedTensor(torch.zeros(batch_size, 3)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_new_zeros_custom_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.ones(2, 3))
        .new_zeros(dtype=dtype)
        .equal(BatchedTensor(torch.zeros(2, 3, dtype=dtype)))
    )


def test_batched_tensor_ones_like() -> None:
    assert BatchedTensor(torch.zeros(2, 3)).ones_like().equal(BatchedTensor(torch.ones(2, 3)))


def test_batched_tensor_ones_like_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.zeros(3, 2), batch_dim=1)
        .ones_like()
        .equal(BatchedTensor(torch.ones(3, 2), batch_dim=1))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_ones_like_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.zeros(2, 3, dtype=dtype))
        .ones_like()
        .equal(BatchedTensor(torch.ones(2, 3, dtype=dtype)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_ones_like_target_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.zeros(2, 3))
        .ones_like(dtype=dtype)
        .equal(BatchedTensor(torch.ones(2, 3, dtype=dtype)))
    )


def test_batched_tensor_zeros_like() -> None:
    assert BatchedTensor(torch.ones(2, 3)).zeros_like().equal(BatchedTensor(torch.zeros(2, 3)))


def test_batched_tensor_zeros_like_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.ones(3, 2), batch_dim=1)
        .zeros_like()
        .equal(BatchedTensor(torch.zeros(3, 2), batch_dim=1))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_zeros_like_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.ones(2, 3, dtype=dtype))
        .zeros_like()
        .equal(BatchedTensor(torch.zeros(2, 3, dtype=dtype)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_zeros_like_target_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.ones(2, 3))
        .zeros_like(dtype=dtype)
        .equal(BatchedTensor(torch.zeros(2, 3, dtype=dtype)))
    )


#################################
#     Comparison operations     #
#################################


def test_batched_tensor_allclose_true() -> None:
    assert BatchedTensor(torch.ones(2, 3)).allclose(BatchedTensor(torch.ones(2, 3)))


def test_batched_tensor_allclose_false_different_type() -> None:
    assert not BatchedTensor(torch.ones(2, 3)).allclose(torch.zeros(2, 3))


def test_batched_tensor_allclose_false_different_data() -> None:
    assert not BatchedTensor(torch.ones(2, 3)).allclose(BatchedTensor(torch.zeros(2, 3)))


def test_batched_tensor_allclose_false_different_shape() -> None:
    assert not BatchedTensor(torch.ones(2, 3)).allclose(BatchedTensor(torch.ones(2, 3, 1)))


def test_batched_tensor_allclose_false_different_batch_dim() -> None:
    assert not BatchedTensor(torch.ones(2, 3)).allclose(
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
    )


@mark.parametrize(
    "batch,atol",
    (
        (BatchedTensor(torch.ones(2, 3) + 0.5), 1),
        (BatchedTensor(torch.ones(2, 3) + 0.05), 1e-1),
        (BatchedTensor(torch.ones(2, 3) + 5e-3), 1e-2),
    ),
)
def test_batched_tensor_allclose_true_atol(batch: BatchedTensor, atol: float) -> None:
    assert BatchedTensor(torch.ones(2, 3)).allclose(batch, atol=atol, rtol=0)


@mark.parametrize(
    "batch,rtol",
    (
        (BatchedTensor(torch.ones(2, 3) + 0.5), 1),
        (BatchedTensor(torch.ones(2, 3) + 0.05), 1e-1),
        (BatchedTensor(torch.ones(2, 3) + 5e-3), 1e-2),
    ),
)
def test_batched_tensor_allclose_true_rtol(batch: BatchedTensor, rtol: float) -> None:
    assert BatchedTensor(torch.ones(2, 3)).allclose(batch, rtol=rtol)


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 5), 5.0)),
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.full((2, 1), 5.0)),
        5,
        5.0,
    ),
)
def test_batched_tensor_eq(other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .eq(other)
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[False, False, False, False, False], [True, False, False, False, False]],
                    dtype=torch.bool,
                ),
            )
        )
    )


def test_batched_tensor_eq_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .eq(BatchedTensor(torch.ones(2, 5).mul(5), batch_dim=1))
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[False, False, False, False, False], [True, False, False, False, False]],
                    dtype=torch.bool,
                ),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_equal_true() -> None:
    assert BatchedTensor(torch.ones(2, 3)).equal(BatchedTensor(torch.ones(2, 3)))


def test_batched_tensor_equal_false_different_type() -> None:
    assert not BatchedTensor(torch.ones(2, 3)).equal(torch.zeros(2, 3))


def test_batched_tensor_equal_false_different_data() -> None:
    assert not BatchedTensor(torch.ones(2, 3)).equal(BatchedTensor(torch.zeros(2, 3)))


def test_batched_tensor_equal_false_different_shape() -> None:
    assert not BatchedTensor(torch.ones(2, 3)).equal(BatchedTensor(torch.ones(2, 3, 1)))


def test_batched_tensor_equal_false_different_batch_dim() -> None:
    assert not BatchedTensor(torch.ones(2, 3), batch_dim=1).equal(BatchedTensor(torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 5), 5.0)),
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_ge(other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .ge(other)
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[False, False, False, False, False], [True, True, True, True, True]],
                    dtype=torch.bool,
                ),
            )
        )
    )


def test_batched_tensor_ge_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .ge(BatchedTensor(torch.full((2, 5), 5.0), batch_dim=1))
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[False, False, False, False, False], [True, True, True, True, True]],
                    dtype=torch.bool,
                ),
                batch_dim=1,
            )
        )
    )


###################################
#     Arithmetical operations     #
###################################


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_add(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert BatchedTensor(torch.ones(2, 3)).add(other).equal(BatchedTensor(torch.ones(2, 3).mul(3)))


def test_batched_tensor_add_alpha_2_float() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3))
        .add(BatchedTensor(torch.full((2, 3), 2.0, dtype=torch.float)), alpha=2.0)
        .equal(BatchedTensor(torch.full((2, 3), 5.0, dtype=torch.float)))
    )


def test_batched_tensor_add_alpha_2_long() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3, dtype=torch.long))
        .add(BatchedTensor(torch.full((2, 3), 2.0, dtype=torch.long)), alpha=2)
        .equal(BatchedTensor(torch.full((2, 3), 5.0, dtype=torch.long)))
    )


def test_batched_tensor_add_batch_dim_1() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .add(BatchedTensor(torch.full((2, 3), 2.0, dtype=torch.float), batch_dim=1))
        .equal(BatchedTensor(torch.full((2, 3), 3.0, dtype=torch.float), batch_dim=1))
    )


def test_batched_tensor_add_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        BatchedTensor(torch.ones(2, 3)).add(BatchedTensor(torch.ones(2, 3), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_div(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (
        BatchedTensor(torch.ones(2, 3)).div(other).equal(BatchedTensor(torch.ones(2, 3).mul(0.5)))
    )


def test_batched_tensor_div_rounding_mode_floor() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3))
        .div(BatchedTensor(torch.ones(2, 3).mul(2)), rounding_mode="floor")
        .equal(BatchedTensor(torch.zeros(2, 3)))
    )


def test_batched_tensor_div_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .div(BatchedTensor(torch.ones(2, 3).mul(2), batch_dim=1))
        .equal(BatchedTensor(torch.ones(2, 3).mul(0.5), batch_dim=1))
    )


def test_batched_tensor_div_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        BatchedTensor(torch.ones(2, 3)).div(BatchedTensor(torch.ones(2, 3), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_mul(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert BatchedTensor(torch.ones(2, 3)).mul(other).equal(BatchedTensor(torch.full((2, 3), 2.0)))


def test_batched_tensor_mul_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .mul(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
        .equal(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
    )


def test_batched_tensor_mul_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        BatchedTensor(torch.ones(2, 3)).mul(BatchedTensor(torch.ones(2, 3), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_sub(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert BatchedTensor(torch.ones(2, 3)).sub(other).equal(BatchedTensor(-torch.ones(2, 3)))


def test_batched_tensor_sub_alpha_2_float() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3))
        .sub(BatchedTensor(torch.full((2, 3), 2.0)), alpha=2.0)
        .equal(BatchedTensor(-torch.ones(2, 3).mul(3)))
    )


def test_batched_tensor_sub_alpha_2_long() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3, dtype=torch.long))
        .sub(BatchedTensor(torch.ones(2, 3, dtype=torch.long).mul(2)), alpha=2)
        .equal(BatchedTensor(-torch.ones(2, 3, dtype=torch.long).mul(3)))
    )


def test_batched_tensor_sub_custom_batch_dims() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .sub(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
        .equal(BatchedTensor(-torch.ones(2, 3), batch_dim=1))
    )


def test_batched_tensor_sub_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        BatchedTensor(torch.ones(2, 3)).sub(BatchedTensor(torch.ones(2, 3), batch_dim=1))


########################################
#     Tests for check_data_and_dim     #
########################################


def test_check_data_and_dim_correct() -> None:
    check_data_and_dim(torch.ones(2, 3), batch_dim=0)
    # will fail if an exception is raised


def test_check_data_and_dim_incorrect_data_dim() -> None:
    with raises(RuntimeError, match=r"data needs at least 1 dimensions \(received: 0\)"):
        check_data_and_dim(torch.tensor(2), batch_dim=0)


@mark.parametrize("batch_dim", (-1, 2, 3))
def test_check_data_and_dim_incorrect_batch_dim(batch_dim: int) -> None:
    with raises(
        RuntimeError, match=r"Incorrect batch_dim \(.*\) but the value should be in \[0, 1\]"
    ):
        check_data_and_dim(torch.ones(2, 3), batch_dim=batch_dim)
