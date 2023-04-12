import math
from typing import Any, Union

import numpy as np
import torch
from pytest import mark, raises
from torch.overrides import is_tensor_like

from redcat import BatchedTensor, BatchedTensorSeq
from redcat.base import BaseBatchedTensor
from redcat.tensor_seq import check_data_and_dims
from redcat.utils import get_available_devices

DTYPES = (torch.bool, torch.int, torch.long, torch.float, torch.double)


def test_batched_tensor_seq_is_tensor_like() -> None:
    assert is_tensor_like(BatchedTensorSeq(torch.ones(2, 3)))


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
    with raises(RuntimeError, match=r"data needs at least 2 dimensions"):
        BatchedTensorSeq(torch.ones(2))


@mark.parametrize("batch_dim", (-1, 3, 4))
def test_batched_tensor_seq_init_incorrect_batch_dim(batch_dim: int) -> None:
    with raises(
        RuntimeError, match=r"Incorrect batch_dim (.*) but the value should be in \[0, 2\]"
    ):
        BatchedTensorSeq(torch.ones(2, 3, 4), batch_dim=batch_dim)


@mark.parametrize("seq_dim", (-1, 3, 4))
def test_batched_tensor_seq_init_incorrect_seq_dim(seq_dim: int) -> None:
    with raises(RuntimeError, match=r"Incorrect seq_dim (.*) but the value should be in \[0, 2\]"):
        BatchedTensorSeq(torch.ones(2, 3, 4), batch_dim=0, seq_dim=seq_dim)


def test_batched_tensor_seq_init_incorrect_same_batch_and_seq_dims() -> None:
    with raises(RuntimeError, match=r"batch_dim \(0\) and seq_dim \(0\) have to be different"):
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


###############################
#     Creation operations     #
###############################


def test_batched_tensor_seq_clone() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    clone = batch.clone()
    batch.add_(1)
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 2.0)))
    assert clone.equal(BatchedTensorSeq(torch.ones(2, 3)))


def test_batched_tensor_seq_clone_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .clone()
        .equal(BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0))
    )


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_seq_empty_like(dtype: torch.dtype) -> None:
    batch = BatchedTensorSeq(torch.zeros(2, 3, dtype=dtype)).empty_like()
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.data.shape == (2, 3)
    assert batch.dtype == dtype


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_seq_empty_like_target_dtype(dtype: torch.dtype) -> None:
    batch = BatchedTensorSeq(torch.zeros(2, 3)).empty_like(dtype=dtype)
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.data.shape == (2, 3)
    assert batch.dtype == dtype


def test_batched_tensor_seq_empty_like_custom_dims() -> None:
    batch = BatchedTensorSeq(torch.zeros(3, 2), batch_dim=1, seq_dim=0).empty_like()
    assert isinstance(batch, BatchedTensorSeq)
    assert batch.data.shape == (3, 2)
    assert batch.batch_dim == 1
    assert batch.seq_dim == 0


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


@mark.parametrize(
    "data",
    (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
    ),
)
def test_batched_tensor_seq_from_seq_batch(data: Any) -> None:
    assert BatchedTensorSeq.from_seq_batch(data).equal(
        BatchedTensorSeq(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        )
    )


#################################
#     Comparison operations     #
#################################


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensorSeq(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq__eq__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (BatchedTensorSeq(torch.arange(10).view(2, 5)) == other).equal(
        BatchedTensorSeq(
            torch.tensor(
                [[False, False, False, False, False], [True, False, False, False, False]],
                dtype=torch.bool,
            ),
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensorSeq(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq__ge__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (BatchedTensorSeq(torch.arange(10).view(2, 5)) >= other).equal(
        BatchedTensorSeq(
            torch.tensor(
                [[False, False, False, False, False], [True, True, True, True, True]],
                dtype=torch.bool,
            ),
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensorSeq(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq__gt__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (BatchedTensorSeq(torch.arange(10).view(2, 5)) > other).equal(
        BatchedTensorSeq(
            torch.tensor(
                [[False, False, False, False, False], [False, True, True, True, True]],
                dtype=torch.bool,
            ),
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensorSeq(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq__le__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (BatchedTensorSeq(torch.arange(10).view(2, 5)) <= other).equal(
        BatchedTensorSeq(
            torch.tensor(
                [[True, True, True, True, True], [True, False, False, False, False]],
                dtype=torch.bool,
            ),
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensorSeq(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq__lt__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (BatchedTensorSeq(torch.arange(10).view(2, 5)) < other).equal(
        BatchedTensorSeq(
            torch.tensor(
                [[True, True, True, True, True], [False, False, False, False, False]],
                dtype=torch.bool,
            ),
        )
    )


def test_batched_tensor_seq_allclose_true() -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).allclose(BatchedTensorSeq(torch.ones(2, 3)))


def test_batched_tensor_seq_allclose_false_different_type() -> None:
    assert not BatchedTensorSeq(torch.ones(2, 3)).allclose(torch.zeros(2, 3))


def test_batched_tensor_seq_allclose_false_different_data() -> None:
    assert not BatchedTensorSeq(torch.ones(2, 3)).allclose(BatchedTensorSeq(torch.zeros(2, 3)))


def test_batched_tensor_seq_allclose_false_different_shape() -> None:
    assert not BatchedTensorSeq(torch.ones(2, 3)).allclose(BatchedTensorSeq(torch.ones(2, 3, 1)))


def test_batched_tensor_seq_allclose_false_different_batch_dim() -> None:
    assert not BatchedTensorSeq(torch.ones(2, 3, 1)).allclose(
        BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2)
    )


def test_batched_tensor_seq_allclose_false_different_seq_dim() -> None:
    assert not BatchedTensorSeq(torch.ones(2, 3, 1)).allclose(
        BatchedTensorSeq(torch.ones(2, 3, 1), seq_dim=2)
    )


@mark.parametrize(
    "batch,atol",
    (
        (BatchedTensorSeq(torch.ones(2, 3) + 0.5), 1),
        (BatchedTensorSeq(torch.ones(2, 3) + 0.05), 1e-1),
        (BatchedTensorSeq(torch.ones(2, 3) + 5e-3), 1e-2),
    ),
)
def test_batched_tensor_seq_allclose_true_atol(batch: BatchedTensorSeq, atol: float) -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).allclose(batch, atol=atol, rtol=0)


@mark.parametrize(
    "batch,rtol",
    (
        (BatchedTensorSeq(torch.ones(2, 3) + 0.5), 1),
        (BatchedTensorSeq(torch.ones(2, 3) + 0.05), 1e-1),
        (BatchedTensorSeq(torch.ones(2, 3) + 5e-3), 1e-2),
    ),
)
def test_batched_tensor_seq_allclose_true_rtol(batch: BatchedTensorSeq, rtol: float) -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).allclose(batch, rtol=rtol)


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensorSeq(torch.full((2, 1), 5.0)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq_eq(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .eq(other)
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[False, False, False, False, False], [True, False, False, False, False]],
                    dtype=torch.bool,
                ),
            )
        )
    )


def test_batched_tensor_seq_eq_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .eq(BatchedTensorSeq(torch.full((2, 5), 5.0), batch_dim=1, seq_dim=0))
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[False, False, False, False, False], [True, False, False, False, False]],
                    dtype=torch.bool,
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


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


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensorSeq(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq_ge(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .ge(other)
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[False, False, False, False, False], [True, True, True, True, True]],
                    dtype=torch.bool,
                ),
            )
        )
    )


def test_batched_tensor_seq_ge_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .ge(BatchedTensorSeq(torch.full((2, 5), 5.0), batch_dim=1, seq_dim=0))
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[False, False, False, False, False], [True, True, True, True, True]],
                    dtype=torch.bool,
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensorSeq(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq_gt(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .gt(other)
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[False, False, False, False, False], [False, True, True, True, True]],
                    dtype=torch.bool,
                ),
            )
        )
    )


def test_batched_tensor_seq_gt_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .gt(BatchedTensorSeq(torch.full((2, 5), 5.0), batch_dim=1, seq_dim=0))
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[False, False, False, False, False], [False, True, True, True, True]],
                    dtype=torch.bool,
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_isinf() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
        .isinf()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[False, False, True], [False, False, True]], dtype=torch.bool)
            )
        )
    )


def test_batched_tensor_seq_isinf_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]),
            batch_dim=1,
            seq_dim=0,
        )
        .isinf()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[False, False, True], [False, False, True]], dtype=torch.bool),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_isneginf() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
        .isneginf()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[False, False, False], [False, False, True]], dtype=torch.bool)
            )
        )
    )


def test_batched_tensor_seq_isneginf_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]),
            batch_dim=1,
            seq_dim=0,
        )
        .isneginf()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[False, False, False], [False, False, True]], dtype=torch.bool),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_isposinf() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
        .isposinf()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[False, False, True], [False, False, False]], dtype=torch.bool)
            )
        )
    )


def test_batched_tensor_seq_isposinf_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]),
            batch_dim=1,
            seq_dim=0,
        )
        .isposinf()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[False, False, True], [False, False, False]], dtype=torch.bool),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_isnan() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1.0, 0.0, float("nan")], [float("nan"), -2.0, -1.0]]))
        .isnan()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[False, False, True], [True, False, False]], dtype=torch.bool)
            )
        )
    )


def test_batched_tensor_seq_isnan_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[1.0, 0.0, float("nan")], [float("nan"), -2.0, -1.0]]),
            batch_dim=1,
            seq_dim=0,
        )
        .isnan()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[False, False, True], [True, False, False]], dtype=torch.bool),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensorSeq(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq_le(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .le(other)
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[True, True, True, True, True], [True, False, False, False, False]],
                    dtype=torch.bool,
                )
            )
        )
    )


def test_batched_tensor_seq_le_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .le(BatchedTensorSeq(torch.full((2, 5), 5.0), batch_dim=1, seq_dim=0))
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[True, True, True, True, True], [True, False, False, False, False]],
                    dtype=torch.bool,
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensorSeq(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq_lt(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .lt(other)
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[True, True, True, True, True], [False, False, False, False, False]],
                    dtype=torch.bool,
                ),
            )
        )
    )


def test_batched_tensor_seq_lt_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .lt(BatchedTensorSeq(torch.full((2, 5), 5.0), batch_dim=1, seq_dim=0))
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[True, True, True, True, True], [False, False, False, False, False]],
                    dtype=torch.bool,
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
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


##################################################
#     Mathematical | arithmetical operations     #
##################################################


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.ones(2, 3)),
        BatchedTensor(torch.ones(2, 3)),
        torch.ones(2, 3),
        1,
        1.0,
    ),
)
def test_batched_tensor_seq__add__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (BatchedTensorSeq(torch.zeros(2, 3)) + other).equal(BatchedTensorSeq(torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.ones(2, 3)),
        BatchedTensor(torch.ones(2, 3)),
        torch.ones(2, 3),
        1,
        1.0,
    ),
)
def test_batched_tensor_seq__iadd__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    batch = BatchedTensorSeq(torch.zeros(2, 3))
    batch += other
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq__mul__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (BatchedTensorSeq(torch.ones(2, 3)) * other).equal(
        BatchedTensorSeq(torch.full((2, 3), 2.0))
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq__imul__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch *= other
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 2.0)))


def test_batched_tensor_seq__neg__() -> None:
    assert (-BatchedTensorSeq(torch.ones(2, 3))).equal(BatchedTensorSeq(-torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq__sub__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (BatchedTensorSeq(torch.ones(2, 3)) - other).equal(BatchedTensorSeq(-torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq__isub__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch -= other
    assert batch.equal(BatchedTensorSeq(-torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq__truediv__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (BatchedTensorSeq(torch.ones(2, 3)) / other).equal(
        BatchedTensorSeq(torch.ones(2, 3).mul(0.5))
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq__itruediv__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch /= other
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3).mul(0.5)))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_add(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .add(other)
        .equal(BatchedTensorSeq(torch.full((2, 3), 3.0)))
    )


def test_batched_tensor_seq_add_alpha_2_float() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .add(BatchedTensorSeq(torch.full((2, 3), 2.0, dtype=torch.float)), alpha=2.0)
        .equal(BatchedTensorSeq(torch.full((2, 3), 5.0, dtype=torch.float)))
    )


def test_batched_tensor_seq_add_alpha_2_long() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3, dtype=torch.long))
        .add(BatchedTensorSeq(torch.full((2, 3), 2.0, dtype=torch.long)), alpha=2)
        .equal(BatchedTensorSeq(torch.full((2, 3), 5.0, dtype=torch.long)))
    )


def test_batched_tensor_seq_add_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .add(BatchedTensorSeq(torch.full((2, 3), 2.0, dtype=torch.float), batch_dim=1, seq_dim=0))
        .equal(BatchedTensorSeq(torch.full((2, 3), 3.0, dtype=torch.float), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_add_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 3)).add(BatchedTensorSeq(torch.ones(2, 3, 4), batch_dim=2))


def test_batched_tensor_seq_add_incorrect_seq_dim() -> None:
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 3, 4)).add(
            BatchedTensorSeq(torch.ones(2, 3, 4), batch_dim=0, seq_dim=2)
        )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_add_(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.add_(other)
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 3.0)))


def test_batched_tensor_seq_add__alpha_2_float() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.add_(BatchedTensorSeq(torch.full((2, 3), 2.0)), alpha=2.0)
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 5.0)))


def test_batched_tensor_seq_add__alpha_2_long() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, dtype=torch.long))
    batch.add_(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.long).mul(2)), alpha=2)
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.long).mul(5)))


def test_batched_tensor_seq_add__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
    batch.add_(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 3.0), batch_dim=1, seq_dim=0))


def test_batched_tensor_seq_add__incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.add_(BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2))


def test_batched_tensor_seq_add__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.add_(BatchedTensorSeq(torch.zeros(2, 3, 1), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_div(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .div(other)
        .equal(BatchedTensorSeq(torch.full((2, 3), 0.5)))
    )


def test_batched_tensor_seq_div_rounding_mode_floor() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .div(BatchedTensorSeq(torch.full((2, 3), 2.0)), rounding_mode="floor")
        .equal(BatchedTensorSeq(torch.zeros(2, 3)))
    )


def test_batched_tensor_seq_div_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .div(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
        .equal(BatchedTensorSeq(torch.full((2, 3), 0.5), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_div_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 3, 1)).div(
            BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2)
        )


def test_batched_tensor_seq_div_incorrect_seq_dim() -> None:
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 3, 1)).div(BatchedTensorSeq(torch.zeros(2, 1, 3), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_div_(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.div_(other)
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 0.5)))


def test_batched_tensor_seq_div__rounding_mode_floor() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.div_(BatchedTensorSeq(torch.full((2, 3), 2.0)), rounding_mode="floor")
    assert batch.equal(BatchedTensorSeq(torch.zeros(2, 3)))


def test_batched_tensor_seq_div__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
    batch.div_(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 0.5), batch_dim=1, seq_dim=0))


def test_batched_tensor_seq_div__incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.div_(BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2))


def test_batched_tensor_seq_div__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.div_(BatchedTensorSeq(torch.ones(2, 3, 1), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_fmod(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).fmod(other).equal(BatchedTensorSeq(torch.ones(2, 3)))


def test_batched_tensor_seq_fmod_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .fmod(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
        .equal(BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_fmod_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 3, 1)).fmod(
            BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2)
        )


def test_batched_tensor_seq_fmod_incorrect_seq_dim() -> None:
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 3, 1)).fmod(
            BatchedTensorSeq(torch.zeros(2, 1, 3), seq_dim=2)
        )


@mark.parametrize(
    "other", (BatchedTensorSeq(torch.full((2, 3), 2.0)), torch.full((2, 3), 2.0), 2, 2.0)
)
def test_batched_tensor_seq_fmod_(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.fmod_(other)
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3)))


def test_batched_tensor_seq_fmod__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
    batch.fmod_(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0))


def test_batched_tensor_seq_fmod__incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.fmod_(BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2))


def test_batched_tensor_seq_fmod__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.fmod_(BatchedTensorSeq(torch.zeros(2, 1, 3), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_mul(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .mul(other)
        .equal(BatchedTensorSeq(torch.full((2, 3), 2.0)))
    )


def test_batched_tensor_seq_mul_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .mul(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
        .equal(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_mul_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 3, 1)).mul(
            BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2)
        )


def test_batched_tensor_seq_mul_incorrect_seq_dim() -> None:
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 3, 1)).mul(BatchedTensorSeq(torch.zeros(2, 1, 3), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_mul_(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.mul_(other)
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 2.0)))


def test_batched_tensor_seq_mul__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
    batch.mul_(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))


def test_batched_tensor_seq_mul__incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError):
        batch.mul_(BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2))


def test_batched_tensor_seq_mul__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError):
        batch.mul_(BatchedTensorSeq(torch.zeros(2, 1, 3), seq_dim=2))


def test_batched_tensor_seq_neg() -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).neg().equal(BatchedTensorSeq(-torch.ones(2, 3)))


def test_batched_tensor_seq_neg_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .neg()
        .equal(BatchedTensorSeq(-torch.ones(2, 3), batch_dim=1, seq_dim=0))
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_sub(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).sub(other).equal(BatchedTensorSeq(-torch.ones(2, 3)))


def test_batched_tensor_seq_sub_alpha_2_float() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .sub(BatchedTensorSeq(torch.full((2, 3), 2.0)), alpha=2.0)
        .equal(BatchedTensorSeq(-torch.full((2, 3), 3.0)))
    )


def test_batched_tensor_seq_sub_alpha_2_long() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3, dtype=torch.long))
        .sub(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.long).mul(2)), alpha=2)
        .equal(BatchedTensorSeq(-torch.ones(2, 3, dtype=torch.long).mul(3)))
    )


def test_batched_tensor_seq_sub_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .sub(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
        .equal(BatchedTensorSeq(-torch.ones(2, 3), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_sub_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 3, 1)).sub(
            BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2)
        )


def test_batched_tensor_seq_sub_incorrect_seq_dim() -> None:
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 3, 1)).sub(BatchedTensorSeq(torch.zeros(2, 3, 1), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_sub_(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.sub_(other)
    assert batch.equal(BatchedTensorSeq(-torch.ones(2, 3)))


def test_batched_tensor_seq_sub__alpha_2_float() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.sub_(BatchedTensorSeq(torch.full((2, 3), 2.0)), alpha=2.0)
    assert batch.equal(BatchedTensorSeq(-torch.full((2, 3), 3.0)))


def test_batched_tensor_seq_sub__alpha_2_long() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, dtype=torch.long))
    batch.sub_(BatchedTensorSeq(torch.ones(2, 3, dtype=torch.long).mul(2)), alpha=2)
    assert batch.equal(BatchedTensorSeq(-torch.ones(2, 3, dtype=torch.long).mul(3)))


def test_batched_tensor_seq_sub__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
    batch.sub_(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
    assert batch.equal(BatchedTensorSeq(-torch.ones(2, 3), batch_dim=1, seq_dim=0))


def test_batched_tensor_seq_sub__incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.sub_(BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2))


def test_batched_tensor_seq_sub__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.sub_(BatchedTensorSeq(torch.zeros(2, 1, 3), seq_dim=2))


################################################
#     Mathematical | point-wise operations     #
################################################


def test_batched_tensor_seq_abs() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0]], dtype=torch.float))
        .abs()
        .equal(
            BatchedTensorSeq(torch.tensor([[2.0, 0.0, 2.0], [1.0, 1.0, 3.0]], dtype=torch.float))
        )
    )


def test_batched_tensor_seq_abs_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        )
        .abs()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[2.0, 0.0, 2.0], [1.0, 1.0, 3.0]], dtype=torch.float),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_abs_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0]], dtype=torch.float))
    batch.abs_()
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[2.0, 0.0, 2.0], [1.0, 1.0, 3.0]], dtype=torch.float))
    )


def test_batched_tensor_seq_abs__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0]], dtype=torch.float),
        batch_dim=1,
        seq_dim=0,
    )
    batch.abs_()
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[2.0, 0.0, 2.0], [1.0, 1.0, 3.0]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        )
    )


def test_batched_tensor_seq_clamp() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .clamp(min_value=2, max_value=5)
        .equal(BatchedTensorSeq(torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]])))
    )


def test_batched_tensor_seq_clamp_only_max_value() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .clamp(max_value=5)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 5, 5, 5, 5]])))
    )


def test_batched_tensor_seq_clamp_only_min_value() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .clamp(min_value=2)
        .equal(BatchedTensorSeq(torch.tensor([[2, 2, 2, 3, 4], [5, 6, 7, 8, 9]])))
    )


def test_batched_tensor_seq_clamp_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .clamp(min_value=2, max_value=5)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]]), batch_dim=1, seq_dim=0
            )
        )
    )


def test_batched_tensor_seq_clamp_() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch.clamp_(min_value=2, max_value=5)
    assert batch.equal(BatchedTensorSeq(torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]])))


def test_batched_tensor_seq_clamp__only_max_value() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch.clamp_(max_value=5)
    assert batch.equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 5, 5, 5, 5]])))


def test_batched_tensor_seq_clamp__only_min_value() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch.clamp_(min_value=2)
    assert batch.equal(BatchedTensorSeq(torch.tensor([[2, 2, 2, 3, 4], [5, 6, 7, 8, 9]])))


def test_batched_tensor_seq_clamp__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
    batch.clamp_(min_value=2, max_value=5)
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]]), batch_dim=1, seq_dim=0)
    )


def test_batched_tensor_seq_exp() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
        .exp()
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [2.7182817459106445, 7.389056205749512, 20.08553695678711],
                        [54.598148345947266, 148.4131622314453, 403.4288024902344],
                    ]
                )
            )
        )
    )


def test_batched_tensor_seq_exp_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float), batch_dim=1, seq_dim=0
        )
        .exp()
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [2.7182817459106445, 7.389056205749512, 20.08553695678711],
                        [54.598148345947266, 148.4131622314453, 403.4288024902344],
                    ]
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_exp_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
    batch.exp_()
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [2.7182817459106445, 7.389056205749512, 20.08553695678711],
                    [54.598148345947266, 148.4131622314453, 403.4288024902344],
                ]
            )
        )
    )


def test_batched_tensor_seq_exp__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float),
        batch_dim=1,
        seq_dim=0,
    )
    batch.exp_()
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [2.7182817459106445, 7.389056205749512, 20.08553695678711],
                    [54.598148345947266, 148.4131622314453, 403.4288024902344],
                ]
            ),
            batch_dim=1,
            seq_dim=0,
        )
    )


def test_batched_tensor_seq_log() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
        .log()
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [0.0, 0.6931471824645996, 1.0986123085021973],
                        [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                    ]
                )
            )
        )
    )


def test_batched_tensor_seq_log_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float), batch_dim=1, seq_dim=0
        )
        .log()
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [0.0, 0.6931471824645996, 1.0986123085021973],
                        [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                    ]
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_log_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
    batch.log_()
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [0.0, 0.6931471824645996, 1.0986123085021973],
                    [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                ]
            )
        )
    )


def test_batched_tensor_seq_log__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float),
        batch_dim=1,
        seq_dim=0,
    )
    batch.log_()
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [0.0, 0.6931471824645996, 1.0986123085021973],
                    [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                ]
            ),
            batch_dim=1,
            seq_dim=0,
        )
    )


def test_batched_tensor_seq_log1p() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float))
        .log1p()
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [0.0, 0.6931471824645996, 1.0986123085021973],
                        [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                    ]
                )
            )
        )
    )


def test_batched_tensor_seq_log1p_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float), batch_dim=1, seq_dim=0
        )
        .log1p()
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [0.0, 0.6931471824645996, 1.0986123085021973],
                        [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                    ]
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_log1p_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float))
    batch.log1p_()
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [0.0, 0.6931471824645996, 1.0986123085021973],
                    [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                ]
            )
        )
    )


def test_batched_tensor_seq_log1p__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float),
        batch_dim=1,
        seq_dim=0,
    )
    batch.log1p_()
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [0.0, 0.6931471824645996, 1.0986123085021973],
                    [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                ]
            ),
            batch_dim=1,
            seq_dim=0,
        )
    )


@mark.parametrize(
    "exponent",
    (BatchedTensorSeq(torch.full((2, 5), 2.0)), BatchedTensor(torch.full((2, 5), 2.0)), 2, 2.0),
)
def test_batched_tensor_seq_pow(exponent: Union[BaseBatchedTensor, int, float]) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5))
        .pow(exponent)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=torch.float)
            )
        )
    )


def test_batched_tensor_seq_pow_exponent_2_float() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5))
        .pow(2.0)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=torch.float)
            )
        )
    )


def test_batched_tensor_seq_pow_exponent_2_long() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .pow(2)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=torch.long)
            )
        )
    )


def test_batched_tensor_seq_pow_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        )
        .pow(BatchedTensorSeq(torch.ones(2, 3).mul(2), batch_dim=1, seq_dim=0))
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_pow_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 3, 1)).pow(
            BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2)
        )


def test_batched_tensor_seq_pow_incorrect_seq_dim() -> None:
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 3, 1)).pow(BatchedTensorSeq(torch.zeros(2, 3, 1), seq_dim=2))


@mark.parametrize(
    "exponent",
    (BatchedTensorSeq(torch.full((2, 5), 2.0)), BatchedTensor(torch.full((2, 5), 2.0)), 2, 2.0),
)
def test_batched_tensor_seq_pow_(exponent: Union[BaseBatchedTensor, int, float]) -> None:
    batch = BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5))
    batch.pow_(exponent)
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=torch.float))
    )


def test_batched_tensor_seq_pow__exponent_2_float() -> None:
    batch = BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5))
    batch.pow_(2.0)
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=torch.float))
    )


def test_batched_tensor_seq_pow__exponent_2_long() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch.pow_(2)
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=torch.long))
    )


def test_batched_tensor_seq_pow__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float), batch_dim=1, seq_dim=0
    )
    batch.pow_(BatchedTensorSeq(torch.ones(2, 3).mul(2), batch_dim=1, seq_dim=0))
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        )
    )


def test_batched_tensor_seq_pow__incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 3, 1)).pow_(
            BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2)
        )


def test_batched_tensor_seq_pow__incorrect_seq_dim() -> None:
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 3, 1)).pow_(
            BatchedTensorSeq(torch.zeros(2, 3, 1), seq_dim=2)
        )


def test_batched_tensor_seq_sqrt() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float))
        .sqrt()
        .equal(
            BatchedTensorSeq(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float))
        )
    )


def test_batched_tensor_seq_sqrt_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        )
        .sqrt()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_sqrt_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float))
    batch.sqrt_()
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float))
    )


def test_batched_tensor_seq_sqrt__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float),
        batch_dim=1,
        seq_dim=0,
    )
    batch.sqrt_()
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        )
    )


###########################################
#     Mathematical | trigo operations     #
###########################################


def test_batched_tensor_seq_acos() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
        .acos()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [[math.pi, math.pi / 2, 0.0], [2 * math.pi / 3, math.pi / 2, math.pi / 3]],
                    dtype=torch.float,
                )
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_acos_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
            batch_dim=1,
            seq_dim=0,
        )
        .acos()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [[math.pi, math.pi / 2, 0.0], [2 * math.pi / 3, math.pi / 2, math.pi / 3]],
                    dtype=torch.float,
                ),
                batch_dim=1,
                seq_dim=0,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_acos_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
    batch.acos_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [[math.pi, math.pi / 2, 0.0], [2 * math.pi / 3, math.pi / 2, math.pi / 3]],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_acos__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
        batch_dim=1,
        seq_dim=0,
    )
    batch.acos_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [[math.pi, math.pi / 2, 0.0], [2 * math.pi / 3, math.pi / 2, math.pi / 3]],
                dtype=torch.float,
            ),
            batch_dim=1,
            seq_dim=0,
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_acosh() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        .acosh()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [0.0, 1.3169578969248166, 1.762747174039086],
                        [2.0634370688955608, 2.2924316695611777, 2.477888730288475],
                    ],
                    dtype=torch.float,
                )
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_acosh_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            batch_dim=1,
            seq_dim=0,
        )
        .acosh()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [0.0, 1.3169578969248166, 1.762747174039086],
                        [2.0634370688955608, 2.2924316695611777, 2.477888730288475],
                    ],
                    dtype=torch.float,
                ),
                batch_dim=1,
                seq_dim=0,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_acosh_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    batch.acosh_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [0.0, 1.3169578969248166, 1.762747174039086],
                    [2.0634370688955608, 2.2924316695611777, 2.477888730288475],
                ],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_acosh__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        batch_dim=1,
        seq_dim=0,
    )
    batch.acosh_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [0.0, 1.3169578969248166, 1.762747174039086],
                    [2.0634370688955608, 2.2924316695611777, 2.477888730288475],
                ],
                dtype=torch.float,
            ),
            batch_dim=1,
            seq_dim=0,
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_asin() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
        .asin()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [[-math.pi / 2, 0.0, math.pi / 2], [-math.pi / 6, 0.0, math.pi / 6]],
                    dtype=torch.float,
                )
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_asin_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
            batch_dim=1,
            seq_dim=0,
        )
        .asin()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [[-math.pi / 2, 0.0, math.pi / 2], [-math.pi / 6, 0.0, math.pi / 6]],
                    dtype=torch.float,
                ),
                batch_dim=1,
                seq_dim=0,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_asin_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
    batch.asin_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [[-math.pi / 2, 0.0, math.pi / 2], [-math.pi / 6, 0.0, math.pi / 6]],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_asin__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
        batch_dim=1,
        seq_dim=0,
    )
    batch.asin_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [[-math.pi / 2, 0.0, math.pi / 2], [-math.pi / 6, 0.0, math.pi / 6]],
                dtype=torch.float,
            ),
            batch_dim=1,
            seq_dim=0,
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_asinh() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
        .asinh()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [-0.8813735842704773, 0.0, 0.8813735842704773],
                        [-0.4812118113040924, 0.0, 0.4812118113040924],
                    ],
                    dtype=torch.float,
                )
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_asinh_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
            batch_dim=1,
            seq_dim=0,
        )
        .asinh()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [-0.8813735842704773, 0.0, 0.8813735842704773],
                        [-0.4812118113040924, 0.0, 0.4812118113040924],
                    ],
                    dtype=torch.float,
                ),
                batch_dim=1,
                seq_dim=0,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_asinh_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
    batch.asinh_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [-0.8813735842704773, 0.0, 0.8813735842704773],
                    [-0.4812118113040924, 0.0, 0.4812118113040924],
                ],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_asinh__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
        batch_dim=1,
        seq_dim=0,
    )
    batch.asinh_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [-0.8813735842704773, 0.0, 0.8813735842704773],
                    [-0.4812118113040924, 0.0, 0.4812118113040924],
                ],
                dtype=torch.float,
            ),
            batch_dim=1,
            seq_dim=0,
        ),
        atol=1e-6,
    )


#########################################
#     Tests for check_data_and_dims     #
#########################################


def test_check_data_and_dims_correct() -> None:
    check_data_and_dims(torch.ones(2, 3), batch_dim=0, seq_dim=1)
    # will fail if an exception is raised


def test_check_data_and_dims_incorrect_data_dim() -> None:
    with raises(RuntimeError, match=r"data needs at least 2 dimensions"):
        check_data_and_dims(torch.ones(2), batch_dim=0, seq_dim=1)


@mark.parametrize("batch_dim", (-1, 3, 4))
def test_check_data_and_dims_incorrect_batch_dim(batch_dim: int) -> None:
    with raises(
        RuntimeError, match=r"Incorrect batch_dim (.*) but the value should be in \[0, 2\]"
    ):
        check_data_and_dims(torch.ones(2, 3, 4), batch_dim=batch_dim, seq_dim=1)


@mark.parametrize("seq_dim", (-1, 3, 4))
def test_check_data_and_dims_incorrect_seq_dim(seq_dim: int) -> None:
    with raises(RuntimeError, match=r"Incorrect seq_dim (.*) but the value should be in \[0, 2\]"):
        check_data_and_dims(torch.ones(2, 3, 4), batch_dim=0, seq_dim=seq_dim)


def test_check_data_and_dims_same_batch_and_seq_dims() -> None:
    with raises(RuntimeError, match=r"batch_dim \(0\) and seq_dim \(0\) have to be different"):
        check_data_and_dims(torch.ones(2, 3), batch_dim=0, seq_dim=0)
