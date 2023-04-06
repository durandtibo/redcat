from typing import Any, Union

import numpy as np
import torch
from pytest import mark, raises

from redcat import BatchedTensorSeq
from redcat.tensor_seq import check_data_and_dims
from redcat.utils import get_available_devices

DTYPES = (torch.bool, torch.int, torch.long, torch.float, torch.double)


@mark.parametrize(
    "data",
    (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
    ),
)
def test_batched_tensor_seq_init_data(data: Any) -> None:
    assert BatchedTensorSeq(data).data.equal(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float)
    )


def test_batched_tensor_seq_init_incorrect_data_dim() -> None:
    with raises(RuntimeError):
        BatchedTensorSeq(torch.ones(2))


@mark.parametrize("batch_dim", (-1, 3, 4))
def test_batched_tensor_seq_init_incorrect_batch_dim(batch_dim: int) -> None:
    with raises(RuntimeError):
        BatchedTensorSeq(torch.ones(2, 3, 4), batch_dim=batch_dim)


@mark.parametrize("seq_dim", (-1, 3, 4))
def test_batched_tensor_seq_init_incorrect_seq_dim(seq_dim: int) -> None:
    with raises(RuntimeError):
        BatchedTensorSeq(torch.ones(2, 3, 4), batch_dim=0, seq_dim=seq_dim)


def test_batched_tensor_seq_init_incorrect_same_batch_and_seq_dims() -> None:
    with raises(RuntimeError):
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=0, seq_dim=0)


def test_batched_tensor_repr() -> None:
    assert repr(BatchedTensorSeq(torch.arange(10).view(2, 5))) == (
        "tensor([[0, 1, 2, 3, 4],\n        [5, 6, 7, 8, 9]], batch_dim=0, seq_dim=1)"
    )


@mark.parametrize("batch_dim", (0, 1))
def test_batched_tensor_seq_batch_dim(batch_dim: int) -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=batch_dim, seq_dim=2).batch_dim == batch_dim
    )


def test_batched_tensor_seq_batch_dim_default() -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).batch_dim == 0


@mark.parametrize("batch_size", (1, 2))
def test_batched_tensor_seq_batch_size(batch_size: int) -> None:
    assert BatchedTensorSeq(torch.ones(batch_size, 3)).batch_size == batch_size


def test_batched_tensor_seq_data() -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).data.equal(torch.ones(2, 3))


@mark.parametrize("seq_dim", (1, 2))
def test_batched_tensor_seq_seq_dim(seq_dim: int) -> None:
    assert BatchedTensorSeq(torch.ones(2, 3, 1), seq_dim=seq_dim).seq_dim == seq_dim


def test_batched_tensor_seq_seq_dim_default() -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).seq_dim == 1


@mark.parametrize("seq_len", (1, 2))
def test_batched_tensor_seq_seq_len(seq_len: int) -> None:
    assert BatchedTensorSeq(torch.ones(2, seq_len)).seq_len == seq_len


@mark.parametrize("device", get_available_devices())
def test_batched_tensor_seq_device(device: str) -> None:
    device = torch.device(device)
    assert BatchedTensorSeq(torch.ones(2, 3, device=device)).device == device


#################################
#     Conversion operations     #
#################################


def test_batched_tensor_seq_contiguous() -> None:
    batch = BatchedTensorSeq(torch.ones(3, 2).transpose(0, 1))
    assert not batch.is_contiguous()
    cont = batch.contiguous()
    assert cont.equal(BatchedTensorSeq(torch.ones(2, 3)))
    assert cont.is_contiguous()


def test_batched_tensor_seq_contiguous_custom_dims() -> None:
    batch = BatchedTensorSeq(torch.ones(3, 2).transpose(0, 1), batch_dim=1, seq_dim=0)
    assert not batch.is_contiguous()
    cont = batch.contiguous()
    assert cont.equal(BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0))
    assert cont.is_contiguous()


def test_batched_tensor_seq_contiguous_memory_format() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 4, 5))
    assert not batch.data.is_contiguous(memory_format=torch.channels_last)
    cont = batch.contiguous(memory_format=torch.channels_last)
    assert cont.equal(BatchedTensorSeq(torch.ones(2, 3, 4, 5)))
    assert cont.is_contiguous(memory_format=torch.channels_last)


def test_batched_tensor_seq_to() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .to(dtype=torch.bool)
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.bool)))
    )


def test_batched_tensor_seq_to_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .to(dtype=torch.bool)
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.bool), batch_dim=1, seq_dim=0))
    )


#################
#     dtype     #
#################


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_seq_dtype(dtype: torch.dtype) -> None:
    assert BatchedTensorSeq(torch.ones(2, 3, dtype=dtype)).dtype == dtype


def test_batched_tensor_seq_bool() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .bool()
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.bool)))
    )


def test_batched_tensor_seq_bool_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .bool()
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.bool), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_double() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .double()
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.double)))
    )


def test_batched_tensor_seq_double_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .double()
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.double), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_float() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3, dtype=torch.long))
        .float()
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.float)))
    )


def test_batched_tensor_seq_float_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3, dtype=torch.long), batch_dim=1, seq_dim=0)
        .float()
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.float), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_int() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .int()
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.int)))
    )


def test_batched_tensor_seq_int_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .int()
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.int), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_long() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .long()
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.long)))
    )


def test_batched_tensor_seq_long_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .long()
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.long), batch_dim=1, seq_dim=0))
    )


#################################
#     Comparison operations     #
#################################


def test_batched_tensor_seq_equal_true() -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).equal(BatchedTensorSeq(torch.ones(2, 3)))


def test_batched_tensor_seq_equal_false_different_type() -> None:
    assert not BatchedTensorSeq(torch.ones(2, 3)).equal(torch.zeros(2, 3))


def test_batched_tensor_seq_equal_false_different_data() -> None:
    assert not BatchedTensorSeq(torch.ones(2, 3)).equal(BatchedTensorSeq(torch.zeros(2, 3)))


def test_batched_tensor_seq_equal_false_different_shape() -> None:
    assert not BatchedTensorSeq(torch.ones(2, 3)).equal(BatchedTensorSeq(torch.ones(2, 3, 1)))


def test_batched_tensor_seq_equal_false_different_batch_dim() -> None:
    assert not BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=1, seq_dim=2).equal(
        BatchedTensorSeq(torch.ones(2, 3, 1), seq_dim=2)
    )


def test_batched_tensor_seq_equal_false_different_seq_dim() -> None:
    assert not BatchedTensorSeq(torch.ones(2, 3, 1), seq_dim=1).equal(
        BatchedTensorSeq(torch.ones(2, 3, 1), seq_dim=2)
    )


###############################
#     Creation operations     #
###############################


def test_batched_tensor_seq_clone() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    clone = batch.clone()
    batch.data.add_(1)
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3).mul(2)))
    assert clone.equal(BatchedTensorSeq(torch.ones(2, 3)))


def test_batched_tensor_seq_clone_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .clone()
        .equal(BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0))
    )


@mark.parametrize("fill_value", (1.5, 2.0, -1.0))
def test_batched_tensor_seq_full_like(fill_value: float) -> None:
    assert (
        BatchedTensorSeq(torch.zeros(2, 3))
        .full_like(fill_value)
        .equal(BatchedTensorSeq(torch.full((2, 3), fill_value=fill_value)))
    )


def test_batched_tensor_seq_full_like_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.zeros(3, 2), batch_dim=1, seq_dim=0)
        .full_like(fill_value=2.0)
        .equal(BatchedTensorSeq(torch.full((3, 2), fill_value=2.0), batch_dim=1, seq_dim=0))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_seq_full_like_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.zeros(2, 3, dtype=dtype))
        .full_like(fill_value=2.0)
        .equal(BatchedTensorSeq(torch.full((2, 3), fill_value=2.0, dtype=dtype)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_seq_full_like_target_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.zeros(2, 3))
        .full_like(fill_value=2.0, dtype=dtype)
        .equal(BatchedTensorSeq(torch.full((2, 3), fill_value=2.0, dtype=dtype)))
    )


def test_batched_tensor_seq_ones_like() -> None:
    assert BatchedTensorSeq(torch.zeros(2, 3)).ones_like().equal(BatchedTensorSeq(torch.ones(2, 3)))


def test_batched_tensor_seq_ones_like_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.zeros(3, 2), batch_dim=1, seq_dim=0)
        .ones_like()
        .equal(BatchedTensorSeq(torch.ones(3, 2), batch_dim=1, seq_dim=0))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_seq_ones_like_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.zeros(2, 3, dtype=dtype))
        .ones_like()
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=dtype)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_seq_ones_like_target_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.zeros(2, 3))
        .ones_like(dtype=dtype)
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=dtype)))
    )


def test_batched_tensor_seq_zeros_like() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3)).zeros_like().equal(BatchedTensorSeq(torch.zeros(2, 3)))
    )


def test_batched_tensor_seq_zeros_like_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(3, 2), batch_dim=1, seq_dim=0)
        .zeros_like()
        .equal(BatchedTensorSeq(torch.zeros(3, 2), batch_dim=1, seq_dim=0))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_seq_zeros_like_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3, dtype=dtype))
        .zeros_like()
        .equal(BatchedTensorSeq(torch.zeros(2, 3, dtype=dtype)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_seq_zeros_like_target_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .zeros_like(dtype=dtype)
        .equal(BatchedTensorSeq(torch.zeros(2, 3, dtype=dtype)))
    )


###############################
#     Creation operations     #
###############################


@mark.parametrize("fill_value", (1, 2.0, True))
def test_batched_tensor_seq_new_full_fill_value(fill_value: Union[float, int, bool]) -> None:
    assert (
        BatchedTensorSeq(torch.zeros(2, 3))
        .new_full(fill_value)
        .equal(BatchedTensorSeq(torch.full((2, 3), fill_value, dtype=torch.float)))
    )


def test_batched_tensor_seq_new_full_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.zeros(3, 2), batch_dim=1, seq_dim=0)
        .new_full(2.0)
        .equal(BatchedTensorSeq(torch.full((3, 2), 2.0), batch_dim=1, seq_dim=0))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_seq_new_full_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.zeros(2, 3, dtype=dtype))
        .new_full(2.0)
        .equal(BatchedTensorSeq(torch.full((2, 3), 2.0, dtype=dtype)))
    )


@mark.parametrize("device", get_available_devices())
def test_batched_tensor_seq_new_full_device(device: str) -> None:
    device = torch.device(device)
    assert (
        BatchedTensorSeq(torch.zeros(2, 3))
        .new_full(2.0, device=device)
        .equal(BatchedTensorSeq(torch.full((2, 3), 2.0, device=device)))
    )


@mark.parametrize("batch_size", (1, 2))
def test_batched_tensor_seq_new_full_custom_batch_size(batch_size: int) -> None:
    assert (
        BatchedTensorSeq(torch.zeros(2, 3))
        .new_full(2.0, batch_size=batch_size)
        .equal(BatchedTensorSeq(torch.full((batch_size, 3), 2.0)))
    )


@mark.parametrize("seq_len", (1, 2))
def test_batched_tensor_seq_new_full_custom_seq_len(seq_len: int) -> None:
    assert (
        BatchedTensorSeq(torch.zeros(2, 3))
        .new_full(2.0, seq_len=seq_len)
        .equal(BatchedTensorSeq(torch.full((2, seq_len), 2.0)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_seq_new_full_custom_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.zeros(2, 3))
        .new_full(2.0, dtype=dtype)
        .equal(BatchedTensorSeq(torch.full((2, 3), 2.0, dtype=dtype)))
    )


def test_batched_tensor_seq_new_ones() -> None:
    assert BatchedTensorSeq(torch.zeros(2, 3)).new_ones().equal(BatchedTensorSeq(torch.ones(2, 3)))


def test_batched_tensor_seq_new_ones_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.zeros(3, 2), batch_dim=1, seq_dim=0)
        .new_ones()
        .equal(BatchedTensorSeq(torch.ones(3, 2), batch_dim=1, seq_dim=0))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_seq_new_ones_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.zeros(2, 3, dtype=dtype))
        .new_ones()
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=dtype)))
    )


@mark.parametrize("device", get_available_devices())
def test_batched_tensor_seq_new_ones_device(device: str) -> None:
    device = torch.device(device)
    assert (
        BatchedTensorSeq(torch.zeros(2, 3))
        .new_ones(device=device)
        .equal(BatchedTensorSeq(torch.ones(2, 3, device=device)))
    )


@mark.parametrize("batch_size", (1, 2))
def test_batched_tensor_seq_new_ones_custom_batch_size(batch_size: int) -> None:
    assert (
        BatchedTensorSeq(torch.zeros(2, 3))
        .new_ones(batch_size=batch_size)
        .equal(BatchedTensorSeq(torch.ones(batch_size, 3)))
    )


@mark.parametrize("seq_len", (1, 2))
def test_batched_tensor_seq_new_ones_custom_seq_len(seq_len: int) -> None:
    assert (
        BatchedTensorSeq(torch.zeros(2, 3))
        .new_ones(seq_len=seq_len)
        .equal(BatchedTensorSeq(torch.ones(2, seq_len)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_seq_new_ones_custom_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.zeros(2, 3))
        .new_ones(dtype=dtype)
        .equal(BatchedTensorSeq(torch.ones(2, 3, dtype=dtype)))
    )


def test_batched_tensor_seq_new_zeros() -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).new_zeros().equal(BatchedTensorSeq(torch.zeros(2, 3)))


def test_batched_tensor_seq_new_zeros_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(3, 2), batch_dim=1, seq_dim=0)
        .new_zeros()
        .equal(BatchedTensorSeq(torch.zeros(3, 2), batch_dim=1, seq_dim=0))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_seq_new_zeros_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3, dtype=dtype))
        .new_zeros()
        .equal(BatchedTensorSeq(torch.zeros(2, 3, dtype=dtype)))
    )


@mark.parametrize("device", get_available_devices())
def test_batched_tensor_seq_new_zeros_device(device: str) -> None:
    device = torch.device(device)
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .new_zeros(device=device)
        .equal(BatchedTensorSeq(torch.zeros(2, 3, device=device)))
    )


@mark.parametrize("batch_size", (1, 2))
def test_batched_tensor_seq_new_zeros_custom_batch_size(batch_size: int) -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .new_zeros(batch_size=batch_size)
        .equal(BatchedTensorSeq(torch.zeros(batch_size, 3)))
    )


@mark.parametrize("seq_len", (1, 2))
def test_batched_tensor_seq_new_zeros_custom_seq_len(seq_len: int) -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .new_zeros(seq_len=seq_len)
        .equal(BatchedTensorSeq(torch.zeros(2, seq_len)))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_seq_new_zeros_custom_dtype(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .new_zeros(dtype=dtype)
        .equal(BatchedTensorSeq(torch.zeros(2, 3, dtype=dtype)))
    )


#########################################
#     Tests for check_data_and_dims     #
#########################################


def test_check_data_and_dims_correct() -> None:
    check_data_and_dims(torch.ones(2, 3), batch_dim=0, seq_dim=1)
    # will fail if an exception is raised


def test_check_data_and_dims_incorrect_data_dim() -> None:
    with raises(RuntimeError):
        check_data_and_dims(torch.ones(2), batch_dim=0, seq_dim=1)


@mark.parametrize("batch_dim", (-1, 3, 4))
def test_check_data_and_dims_incorrect_batch_dim(batch_dim: int) -> None:
    with raises(RuntimeError):
        check_data_and_dims(torch.ones(2, 3, 4), batch_dim=batch_dim, seq_dim=1)


@mark.parametrize("seq_dim", (-1, 3, 4))
def test_check_data_and_dims_incorrect_seq_dim(seq_dim: int) -> None:
    with raises(RuntimeError):
        check_data_and_dims(torch.ones(2, 3, 4), batch_dim=0, seq_dim=seq_dim)


def test_check_data_and_dims_same_batch_and_seq_dims() -> None:
    with raises(RuntimeError):
        check_data_and_dims(torch.ones(2, 3), batch_dim=0, seq_dim=0)
