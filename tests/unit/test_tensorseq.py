import math
from collections.abc import Iterable, Sequence
from typing import Any, Union
from unittest.mock import patch

import numpy as np
import torch
from pytest import mark, raises
from torch import Tensor
from torch.overrides import is_tensor_like

from redcat import BatchedTensor, BatchedTensorSeq
from redcat.basetensor import BaseBatchedTensor
from redcat.tensorseq import check_data_and_dims
from redcat.utils import get_available_devices, get_torch_generator

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


def test_batched_tensor_seq_numel() -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).numel() == 6


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


###########################################################
#     Mathematical | advanced arithmetical operations     #
###########################################################


def test_batched_tensor_seq_cumsum() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .cumsum(dim=0)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])))
    )


def test_batched_tensor_seq_cumsum_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.arange(10).view(5, 2))
        .cumsum(dim=1)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), seq_dim=0, batch_dim=1
            )
        )
    )


def test_batched_tensor_seq_cumsum_dtype() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .cumsum(dim=0, dtype=torch.int)
        .equal(
            BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]], dtype=torch.int))
        )
    )


def test_batched_tensor_seq_cumsum_() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch.cumsum_(dim=0)
    assert batch.equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])))


def test_batched_tensor_seq_cumsum__custom_dims() -> None:
    batch = BatchedTensorSeq.from_seq_batch(torch.arange(10).view(5, 2))
    batch.cumsum_(dim=1)
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), seq_dim=0, batch_dim=1
        )
    )


def test_batched_tensor_seq_cumsum_along_batch() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .cumsum_along_batch()
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])))
    )


def test_batched_tensor_seq_cumsum_along_batch_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.arange(10).view(5, 2))
        .cumsum_along_batch()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), seq_dim=0, batch_dim=1
            )
        )
    )


def test_batched_tensor_seq_cumsum_along_batch_dtype() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .cumsum_along_batch(dtype=torch.int)
        .equal(
            BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]], dtype=torch.int))
        )
    )


def test_batched_tensor_seq_cumsum_along_batch_() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch.cumsum_along_batch_()
    assert batch.equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])))


def test_batched_tensor_seq_cumsum_along_batch__custom_dims() -> None:
    batch = BatchedTensorSeq.from_seq_batch(torch.arange(10).view(5, 2))
    batch.cumsum_along_batch_()
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), seq_dim=0, batch_dim=1
        )
    )


def test_batched_tensor_seq_cumsum_along_seq() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .cumsum_along_seq()
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 3, 6, 10], [5, 11, 18, 26, 35]])))
    )


def test_batched_tensor_seq_cumsum_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.arange(10).view(5, 2))
        .cumsum_along_seq()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0, 1], [2, 4], [6, 9], [12, 16], [20, 25]]), seq_dim=0, batch_dim=1
            )
        )
    )


def test_batched_tensor_seq_cumsum_along_seq_dtype() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .cumsum_along_seq(dtype=torch.int)
        .equal(
            BatchedTensorSeq(torch.tensor([[0, 1, 3, 6, 10], [5, 11, 18, 26, 35]], dtype=torch.int))
        )
    )


def test_batched_tensor_seq_cumsum_along_seq_() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch.cumsum_along_seq_()
    assert batch.equal(BatchedTensorSeq(torch.tensor([[0, 1, 3, 6, 10], [5, 11, 18, 26, 35]])))


def test_batched_tensor_seq_cumsum_along_seq__custom_dims() -> None:
    batch = BatchedTensorSeq.from_seq_batch(torch.arange(10).view(5, 2))
    batch.cumsum_along_seq_()
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[0, 1], [2, 4], [6, 9], [12, 16], [20, 25]]), seq_dim=0, batch_dim=1
        )
    )


@mark.parametrize("permutation", (torch.tensor([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_tensor_seq_permute_along_batch(
    permutation: Union[Sequence[int], torch.Tensor]
) -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
        .permute_along_batch(permutation)
        .equal(BatchedTensorSeq(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))
    )


def test_batched_tensor_seq_permute_along_batch_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), batch_dim=1, seq_dim=0)
        .permute_along_batch(torch.tensor([2, 1, 3, 0]))
        .equal(BatchedTensorSeq(torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4]]), batch_dim=1, seq_dim=0))
    )


@mark.parametrize("permutation", (torch.tensor([2, 4, 1, 3, 0]), [2, 4, 1, 3, 0], (2, 4, 1, 3, 0)))
def test_batched_tensor_seq_permute_along_seq(
    permutation: Union[Sequence[int], torch.Tensor]
) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .permute_along_seq(permutation)
        .equal(BatchedTensorSeq(torch.tensor([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])))
    )


def test_batched_tensor_seq_permute_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0)
        .permute_along_seq(torch.tensor([2, 4, 1, 3, 0]))
        .equal(
            BatchedTensorSeq(
                torch.tensor([[4, 5], [8, 9], [2, 3], [6, 7], [0, 1]]), batch_dim=1, seq_dim=0
            )
        )
    )


@patch("redcat.basetensor.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_seq_shuffle_along_batch() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
        .shuffle_along_batch()
        .equal(BatchedTensorSeq(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))
    )


@patch("redcat.basetensor.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_seq_shuffle_along_batch_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]))
        .shuffle_along_batch()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]]), batch_dim=1, seq_dim=0
            )
        )
    )


def test_batched_tensor_seq_shuffle_along_batch_same_random_seed() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert batch.shuffle_along_batch(get_torch_generator(1)).equal(
        batch.shuffle_along_batch(get_torch_generator(1))
    )


def test_batched_tensor_seq_shuffle_along_batch_different_random_seeds() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert not batch.shuffle_along_batch(get_torch_generator(1)).equal(
        batch.shuffle_along_batch(get_torch_generator(2))
    )


def test_batched_tensor_seq_shuffle_along_batch_multiple_shuffle() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    generator = get_torch_generator(1)
    assert not batch.shuffle_along_batch(generator).equal(batch.shuffle_along_batch(generator))


def test_batched_tensor_seq_sort_along_seq_descending_false() -> None:
    values, indices = BatchedTensorSeq(
        torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])
    ).sort_along_seq()
    assert values.equal(BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])))
    assert indices.equal(BatchedTensorSeq(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]])))


def test_batched_tensor_seq_sort_along_seq_descending_true() -> None:
    values, indices = BatchedTensorSeq(
        torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])
    ).sort_along_seq(descending=True)
    assert values.equal(BatchedTensorSeq(torch.tensor([[5, 4, 3, 2, 1], [9, 8, 7, 6, 5]])))
    assert indices.equal(BatchedTensorSeq(torch.tensor([[3, 0, 4, 2, 1], [0, 4, 1, 3, 2]])))


def test_batched_tensor_seq_sort_along_seq_dim_3() -> None:
    values, indices = BatchedTensorSeq(
        torch.tensor(
            [
                [[0, 1], [-2, 3], [-4, 5], [-6, 7], [-8, 9]],
                [[10, -11], [12, -13], [14, -15], [16, -17], [18, -19]],
            ]
        )
    ).sort_along_seq()
    assert values.equal(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [[-8, 1], [-6, 3], [-4, 5], [-2, 7], [0, 9]],
                    [[10, -19], [12, -17], [14, -15], [16, -13], [18, -11]],
                ]
            )
        )
    )
    assert indices.equal(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]],
                    [[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]],
                ]
            )
        )
    )


def test_batched_tensor_seq_sort_along_seq_seq_dim_0() -> None:
    values, indices = BatchedTensorSeq(
        torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), seq_dim=0, batch_dim=1
    ).sort_along_seq()
    assert values.equal(
        BatchedTensorSeq(
            torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]), seq_dim=0, batch_dim=1
        )
    )
    assert indices.equal(
        BatchedTensorSeq(
            torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]), seq_dim=0, batch_dim=1
        )
    )


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
    "data,max_value",
    (
        (torch.tensor([[False, True, True], [True, False, True]], dtype=torch.bool), True),  # bool
        (torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long), 5),  # long
        (torch.tensor([[4.0, 1.0, 7.0], [3.0, 2.0, 5.0]], dtype=torch.float), 7.0),  # float
    ),
)
def test_batched_tensor_seq_max_global(
    data: torch.Tensor, max_value: Union[bool, int, float]
) -> None:
    assert BatchedTensorSeq(data).max() == max_value


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        torch.tensor([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_batched_tensor_seq_max(other: BaseBatchedTensor | Tensor) -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [-2, -1, 0]]))
        .max(other)
        .equal(BatchedTensorSeq(torch.tensor([[2, 1, 2], [0, 1, 0]])))
    )


def test_batched_tensor_seq_max_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [-2, -1, 0]]), batch_dim=1, seq_dim=0)
        .max(BatchedTensorSeq(torch.tensor([[2, 0, 1], [0, 1, 0]]), batch_dim=1, seq_dim=0))
        .equal(BatchedTensorSeq(torch.tensor([[2, 1, 2], [0, 1, 0]]), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_max_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.max(BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2))


def test_batched_tensor_seq_max_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.max(BatchedTensorSeq(torch.zeros(2, 1, 3), seq_dim=2))


@mark.parametrize(
    "data,min_value",
    (
        (torch.tensor([[False, True, True], [True, False, True]], dtype=torch.bool), False),
        # bool
        (torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long), 0),  # long
        (torch.tensor([[4.0, 1.0, 7.0], [3.0, 2.0, 5.0]], dtype=torch.float), 1.0),  # float
    ),
)
def test_batched_tensor_seq_min_global(
    data: torch.Tensor, min_value: Union[bool, int, float]
) -> None:
    assert BatchedTensorSeq(data).min() == min_value


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        torch.tensor([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_batched_tensor_seq_min(other: BaseBatchedTensor | Tensor) -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [-2, -1, 0]]))
        .min(other)
        .equal(BatchedTensorSeq(torch.tensor([[0, 0, 1], [-2, -1, 0]])))
    )


def test_batched_tensor_seq_min_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [-2, -1, 0]]), batch_dim=1, seq_dim=0)
        .min(BatchedTensorSeq(torch.tensor([[2, 0, 1], [0, 1, 0]]), batch_dim=1, seq_dim=0))
        .equal(BatchedTensorSeq(torch.tensor([[0, 0, 1], [-2, -1, 0]]), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_min_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.min(BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2))


def test_batched_tensor_seq_min_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.min(BatchedTensorSeq(torch.zeros(2, 1, 3), seq_dim=2))


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


def test_batched_tensor_seq_atan() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0.0, 1.0, math.sqrt(3.0)], [-math.sqrt(3.0), -1.0, 0.0]]))
        .atan()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [[0.0, math.pi / 4, math.pi / 3], [-math.pi / 3, -math.pi / 4, 0.0]],
                    dtype=torch.float,
                )
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_atan_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0.0, 1.0, math.sqrt(3.0)], [-math.sqrt(3.0), -1.0, 0.0]]),
            batch_dim=1,
            seq_dim=0,
        )
        .atan()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [[0.0, math.pi / 4, math.pi / 3], [-math.pi / 3, -math.pi / 4, 0.0]],
                    dtype=torch.float,
                ),
                batch_dim=1,
                seq_dim=0,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_atan_() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[0.0, 1.0, math.sqrt(3.0)], [-math.sqrt(3.0), -1.0, 0.0]])
    )
    batch.atan_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [[0.0, math.pi / 4, math.pi / 3], [-math.pi / 3, -math.pi / 4, 0.0]],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_atan__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[0.0, 1.0, math.sqrt(3.0)], [-math.sqrt(3.0), -1.0, 0.0]]),
        batch_dim=1,
        seq_dim=0,
    )
    batch.atan_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [[0.0, math.pi / 4, math.pi / 3], [-math.pi / 3, -math.pi / 4, 0.0]],
                dtype=torch.float,
            ),
            batch_dim=1,
            seq_dim=0,
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_atanh() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[-0.5, 0.0, 0.5], [-0.1, 0.0, 0.1]]))
        .atanh()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [-0.5493061443340549, 0.0, 0.5493061443340549],
                        [-0.10033534773107558, 0.0, 0.10033534773107558],
                    ],
                    dtype=torch.float,
                )
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_atanh_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[-0.5, 0.0, 0.5], [-0.1, 0.0, 0.1]]),
            batch_dim=1,
            seq_dim=0,
        )
        .atanh()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [-0.5493061443340549, 0.0, 0.5493061443340549],
                        [-0.10033534773107558, 0.0, 0.10033534773107558],
                    ],
                    dtype=torch.float,
                ),
                batch_dim=1,
                seq_dim=0,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_atanh_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[-0.5, 0.0, 0.5], [-0.1, 0.0, 0.1]]))
    batch.atanh_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [-0.5493061443340549, 0.0, 0.5493061443340549],
                    [-0.10033534773107558, 0.0, 0.10033534773107558],
                ],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_atanh__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[-0.5, 0.0, 0.5], [-0.1, 0.0, 0.1]]),
        batch_dim=1,
        seq_dim=0,
    )
    batch.atanh_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [-0.5493061443340549, 0.0, 0.5493061443340549],
                    [-0.10033534773107558, 0.0, 0.10033534773107558],
                ],
                dtype=torch.float,
            ),
            batch_dim=1,
            seq_dim=0,
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_cos() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]])
        )
        .cos()
        .allclose(
            BatchedTensorSeq(torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, 1.0]], dtype=torch.float)),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_cos_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]]),
            batch_dim=1,
            seq_dim=0,
        )
        .cos()
        .allclose(
            BatchedTensorSeq(
                torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, 1.0]], dtype=torch.float),
                batch_dim=1,
                seq_dim=0,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_cos_() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]])
    )
    batch.cos_()
    assert batch.allclose(
        BatchedTensorSeq(torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, 1.0]], dtype=torch.float)),
        atol=1e-6,
    )


def test_batched_tensor_seq_cos__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]]),
        batch_dim=1,
        seq_dim=0,
    )
    batch.cos_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, 1.0]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_cosh() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
        .cosh()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [1.5430806348152437, 1.0, 1.5430806348152437],
                        [1.1276259652063807, 1.0, 1.1276259652063807],
                    ],
                    dtype=torch.float,
                )
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_cosh_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
            batch_dim=1,
            seq_dim=0,
        )
        .cosh()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [1.5430806348152437, 1.0, 1.5430806348152437],
                        [1.1276259652063807, 1.0, 1.1276259652063807],
                    ],
                    dtype=torch.float,
                ),
                batch_dim=1,
                seq_dim=0,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_cosh_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
    batch.cosh_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [1.5430806348152437, 1.0, 1.5430806348152437],
                    [1.1276259652063807, 1.0, 1.1276259652063807],
                ],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_cosh__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
        batch_dim=1,
        seq_dim=0,
    )
    batch.cosh_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [1.5430806348152437, 1.0, 1.5430806348152437],
                    [1.1276259652063807, 1.0, 1.1276259652063807],
                ],
                dtype=torch.float,
            ),
            batch_dim=1,
            seq_dim=0,
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_sin() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]])
        )
        .sin()
        .allclose(
            BatchedTensorSeq(torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float)),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_sin_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]]),
            batch_dim=1,
            seq_dim=0,
        )
        .sin()
        .allclose(
            BatchedTensorSeq(
                torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float),
                batch_dim=1,
                seq_dim=0,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_sin_() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]])
    )
    batch.sin_()
    assert batch.allclose(
        BatchedTensorSeq(torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float)),
        atol=1e-6,
    )


def test_batched_tensor_seq_sin__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]]),
        batch_dim=1,
        seq_dim=0,
    )
    batch.sin_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_sinh() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]], dtype=torch.float))
        .sinh()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [-1.175201177597046, 0.0, 1.175201177597046],
                        [-0.5210952758789062, 0.0, 0.5210952758789062],
                    ],
                    dtype=torch.float,
                )
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_sinh_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        )
        .sinh()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [-1.175201177597046, 0.0, 1.175201177597046],
                        [-0.5210952758789062, 0.0, 0.5210952758789062],
                    ],
                    dtype=torch.float,
                ),
                batch_dim=1,
                seq_dim=0,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_sinh_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]], dtype=torch.float))
    batch.sinh_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [-1.175201177597046, 0.0, 1.175201177597046],
                    [-0.5210952758789062, 0.0, 0.5210952758789062],
                ],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_sinh__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]], dtype=torch.float),
        batch_dim=1,
        seq_dim=0,
    )
    batch.sinh_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [-1.175201177597046, 0.0, 1.175201177597046],
                    [-0.5210952758789062, 0.0, 0.5210952758789062],
                ],
                dtype=torch.float,
            ),
            batch_dim=1,
            seq_dim=0,
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_tan() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor(
                [[0.0, 0.25 * math.pi, math.pi], [2 * math.pi, 1.75 * math.pi, -0.25 * math.pi]]
            )
        )
        .tan()
        .allclose(
            BatchedTensorSeq(torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, -1.0]], dtype=torch.float)),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_tan_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor(
                [[0.0, 0.25 * math.pi, math.pi], [2 * math.pi, 1.75 * math.pi, -0.25 * math.pi]]
            ),
            batch_dim=1,
            seq_dim=0,
        )
        .tan()
        .allclose(
            BatchedTensorSeq(
                torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, -1.0]], dtype=torch.float),
                batch_dim=1,
                seq_dim=0,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_tan_() -> None:
    batch = BatchedTensorSeq(
        torch.tensor(
            [[0.0, 0.25 * math.pi, math.pi], [2 * math.pi, 1.75 * math.pi, -0.25 * math.pi]]
        )
    )
    batch.tan_()
    assert batch.allclose(
        BatchedTensorSeq(torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, -1.0]], dtype=torch.float)),
        atol=1e-6,
    )


def test_batched_tensor_seq_tan__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor(
            [[0.0, 0.25 * math.pi, math.pi], [2 * math.pi, 1.75 * math.pi, -0.25 * math.pi]]
        ),
        batch_dim=1,
        seq_dim=0,
    )
    batch.tan_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, -1.0]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_tanh() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]]))
        .tanh()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [0.0, 0.7615941559557649, 0.9640275800758169],
                        [-0.9640275800758169, -0.7615941559557649, 0.0],
                    ],
                    dtype=torch.float,
                )
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_tanh_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]]),
            batch_dim=1,
            seq_dim=0,
        )
        .tanh()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [0.0, 0.7615941559557649, 0.9640275800758169],
                        [-0.9640275800758169, -0.7615941559557649, 0.0],
                    ],
                    dtype=torch.float,
                ),
                batch_dim=1,
                seq_dim=0,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_seq_tanh_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]]))
    batch.tanh_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [0.0, 0.7615941559557649, 0.9640275800758169],
                    [-0.9640275800758169, -0.7615941559557649, 0.0],
                ],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_batched_tensor_seq_tanh__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]]),
        batch_dim=1,
        seq_dim=0,
    )
    batch.tanh_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [0.0, 0.7615941559557649, 0.9640275800758169],
                    [-0.9640275800758169, -0.7615941559557649, 0.0],
                ],
                dtype=torch.float,
            ),
            batch_dim=1,
            seq_dim=0,
        ),
        atol=1e-6,
    )


#############################################
#     Mathematical | logical operations     #
#############################################


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensorSeq(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_seq_logical_and(
    other: BaseBatchedTensor | Tensor, dtype: torch.dtype
) -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
        )
        .logical_and(other)
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[True, False, False, False], [True, False, True, False]], dtype=torch.bool
                )
            )
        )
    )


def test_batched_tensor_seq_logical_and_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor(
                [[True, True, False, False], [True, False, True, False]], dtype=torch.bool
            ),
            batch_dim=1,
            seq_dim=0,
        )
        .logical_and(
            BatchedTensorSeq(
                torch.tensor(
                    [[True, False, True, False], [True, True, True, True]], dtype=torch.bool
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[True, False, False, False], [True, False, True, False]], dtype=torch.bool
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_logical_and_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.zeros(2, 3, 1, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_and(
            BatchedTensorSeq(
                torch.zeros(2, 3, 1, dtype=torch.bool),
                batch_dim=2,
            )
        )


def test_batched_tensor_seq_logical_and_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1, dtype=torch.bool))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.logical_and(BatchedTensorSeq(torch.zeros(2, 3, 1, dtype=torch.bool), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensorSeq(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_seq_logical_and_(
    other: BaseBatchedTensor | Tensor, dtype: torch.dtype
) -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
    )
    batch.logical_and_(other)
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor(
                [[True, False, False, False], [True, False, True, False]], dtype=torch.bool
            )
        )
    )


def test_batched_tensor_seq_logical_and__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=torch.bool),
        batch_dim=1,
        seq_dim=0,
    )
    batch.logical_and_(
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
            batch_dim=1,
            seq_dim=0,
        )
    )
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor(
                [[True, False, False, False], [True, False, True, False]], dtype=torch.bool
            ),
            batch_dim=1,
            seq_dim=0,
        )
    )


def test_batched_tensor_seq_logical_and__incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.zeros(2, 3, 1, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_and_(
            BatchedTensorSeq(
                torch.zeros(2, 3, 1, dtype=torch.bool),
                batch_dim=2,
            )
        )


def test_batched_tensor_seq_logical_and__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1, dtype=torch.bool))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.logical_and_(BatchedTensorSeq(torch.zeros(2, 3, 1, dtype=torch.bool), seq_dim=2))


@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_seq_logical_not(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
        )
        .logical_not()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[False, False, True, True], [False, True, False, True]], dtype=dtype)
            )
        )
    )


def test_batched_tensor_seq_logical_not_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor(
                [[True, True, False, False], [True, False, True, False]], dtype=torch.bool
            ),
            batch_dim=1,
            seq_dim=0,
        )
        .logical_not()
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[False, False, True, True], [False, True, False, True]], dtype=torch.bool
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_seq_logical_not_(dtype: torch.dtype) -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
    )
    batch.logical_not_()
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[False, False, True, True], [False, True, False, True]], dtype=dtype)
        )
    )


def test_batched_tensor_seq_logical_not__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=torch.bool),
        batch_dim=1,
        seq_dim=0,
    )
    batch.logical_not_()
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor(
                [[False, False, True, True], [False, True, False, True]], dtype=torch.bool
            ),
            batch_dim=1,
            seq_dim=0,
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensorSeq(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_seq_logical_or(
    other: BaseBatchedTensor | Tensor, dtype: torch.dtype
) -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
        )
        .logical_or(other)
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[True, True, True, False], [True, True, True, True]], dtype=torch.bool
                )
            )
        )
    )


def test_batched_tensor_seq_logical_or_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor(
                [[True, True, False, False], [True, False, True, False]], dtype=torch.bool
            ),
            batch_dim=1,
            seq_dim=0,
        )
        .logical_or(
            BatchedTensorSeq(
                torch.tensor(
                    [[True, False, True, False], [True, True, True, True]], dtype=torch.bool
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[True, True, True, False], [True, True, True, True]], dtype=torch.bool
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_logical_or_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.zeros(2, 3, 1, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_or(
            BatchedTensorSeq(
                torch.zeros(2, 3, 1, dtype=torch.bool),
                batch_dim=2,
            )
        )


def test_batched_tensor_seq_logical_or_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1, dtype=torch.bool))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.logical_or(BatchedTensorSeq(torch.zeros(2, 3, 1, dtype=torch.bool), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensorSeq(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_seq_logical_or_(
    other: BaseBatchedTensor | Tensor, dtype: torch.dtype
) -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
    )
    batch.logical_or_(other)
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[True, True, True, False], [True, True, True, True]], dtype=torch.bool)
        )
    )


def test_batched_tensor_seq_logical_or__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=torch.bool),
        batch_dim=1,
        seq_dim=0,
    )
    batch.logical_or_(
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
            batch_dim=1,
            seq_dim=0,
        )
    )
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[True, True, True, False], [True, True, True, True]], dtype=torch.bool),
            batch_dim=1,
            seq_dim=0,
        )
    )


def test_batched_tensor_seq_logical_or__incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.zeros(2, 3, 1, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_or_(
            BatchedTensorSeq(
                torch.zeros(2, 3, 1, dtype=torch.bool),
                batch_dim=2,
            )
        )


def test_batched_tensor_seq_logical_or__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1, dtype=torch.bool))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.logical_or_(BatchedTensorSeq(torch.zeros(2, 3, 1, dtype=torch.bool), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensorSeq(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_seq_logical_xor(
    other: BaseBatchedTensor | Tensor, dtype: torch.dtype
) -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
        )
        .logical_xor(other)
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[False, True, True, False], [False, True, False, True]], dtype=torch.bool
                )
            )
        )
    )


def test_batched_tensor_seq_logical_xor_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor(
                [[True, True, False, False], [True, False, True, False]], dtype=torch.bool
            ),
            batch_dim=1,
            seq_dim=0,
        )
        .logical_xor(
            BatchedTensorSeq(
                torch.tensor(
                    [[True, False, True, False], [True, True, True, True]], dtype=torch.bool
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[False, True, True, False], [False, True, False, True]], dtype=torch.bool
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_logical_xor_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.zeros(2, 3, 1, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_xor(
            BatchedTensorSeq(
                torch.zeros(2, 3, 1, dtype=torch.bool),
                batch_dim=2,
            )
        )


def test_batched_tensor_seq_logical_xor_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1, dtype=torch.bool))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.logical_xor(BatchedTensorSeq(torch.zeros(2, 3, 1, dtype=torch.bool), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensorSeq(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_seq_logical_xor_(
    other: BaseBatchedTensor | Tensor, dtype: torch.dtype
) -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
    )
    batch.logical_xor_(other)
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[False, True, True, False], [False, True, False, True]], dtype=torch.bool)
        )
    )


def test_batched_tensor_seq_logical_xor__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=torch.bool),
        batch_dim=1,
        seq_dim=0,
    )
    batch.logical_xor_(
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
            batch_dim=1,
            seq_dim=0,
        )
    )
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor(
                [[False, True, True, False], [False, True, False, True]], dtype=torch.bool
            ),
            batch_dim=1,
            seq_dim=0,
        )
    )


def test_batched_tensor_seq_logical_xor__incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.zeros(2, 3, 1, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_xor_(
            BatchedTensorSeq(
                torch.zeros(2, 3, 1, dtype=torch.bool),
                batch_dim=2,
            )
        )


def test_batched_tensor_seq_logical_xor__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1, dtype=torch.bool))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.logical_xor_(BatchedTensorSeq(torch.zeros(2, 3, 1, dtype=torch.bool), seq_dim=2))


################################
#     Reduction operations     #
################################


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_max_along_seq(dtype: torch.dtype) -> None:
    values, indices = BatchedTensorSeq(torch.arange(10).view(2, 5).to(dtype=dtype)).max_along_seq()
    assert values.equal(BatchedTensor(torch.tensor([4, 9], dtype=dtype)))
    assert indices.equal(BatchedTensor(torch.tensor([4, 4])))


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_max_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    values, indices = BatchedTensorSeq(torch.arange(10).view(2, 5).to(dtype=dtype)).max_along_seq(
        keepdim=True
    )
    assert values.equal(BatchedTensorSeq(torch.tensor([[4], [9]], dtype=dtype)))
    assert indices.equal(BatchedTensorSeq(torch.tensor([[4], [4]])))


def test_batched_tensor_seq_max_along_seq_custom_dims() -> None:
    values, indices = BatchedTensorSeq(
        torch.tensor([[2, 4], [1, 5], [0, 2]]), batch_dim=1, seq_dim=0
    ).max_along_seq()
    assert values.equal(BatchedTensor(torch.tensor([2, 5])))
    assert indices.equal(BatchedTensor(torch.tensor([0, 1])))


def test_batched_tensor_seq_max_along_seq_keepdim_true_custom_dims() -> None:
    values, indices = BatchedTensorSeq.from_seq_batch(torch.arange(10).view(5, 2)).max_along_seq(
        keepdim=True
    )
    assert values.equal(BatchedTensorSeq(torch.tensor([[8, 9]]), batch_dim=1, seq_dim=0))
    assert indices.equal(BatchedTensorSeq(torch.tensor([[4, 4]]), batch_dim=1, seq_dim=0))


def test_batched_tensor_seq_max_along_seq_extra_dims() -> None:
    values, indices = BatchedTensorSeq(
        torch.arange(20).view(2, 5, 2), batch_dim=2, seq_dim=1
    ).max_along_seq()
    assert values.equal(BatchedTensor(torch.tensor([[8, 9], [18, 19]]), batch_dim=1))
    assert indices.equal(BatchedTensor(torch.tensor([[4, 4], [4, 4]]), batch_dim=1))


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_mean_along_seq(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5).to(dtype=dtype))
        .mean_along_seq()
        .equal(BatchedTensor(torch.tensor([2.0, 7.0], dtype=torch.float)))
    )


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_mean_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5).to(dtype=dtype))
        .mean_along_seq(keepdim=True)
        .equal(BatchedTensorSeq(torch.tensor([[2.0], [7.0]], dtype=torch.float)))
    )


def test_batched_tensor_seq_mean_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 4], [1, 2], [2, 6]]), batch_dim=1, seq_dim=0)
        .mean_along_seq()
        .equal(BatchedTensor(torch.tensor([1.0, 4.0], dtype=torch.float)))
    )


def test_batched_tensor_seq_mean_along_seq_keepdim_true_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.arange(10).view(5, 2))
        .mean_along_seq(keepdim=True)
        .equal(
            BatchedTensorSeq(torch.tensor([[4.0, 5.0]], dtype=torch.float), batch_dim=1, seq_dim=0)
        )
    )


def test_batched_tensor_seq_mean_along_seq_extra_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(20).view(2, 5, 2), batch_dim=2, seq_dim=1)
        .mean_along_seq()
        .equal(
            BatchedTensor(torch.tensor([[4.0, 5.0], [14.0, 15.0]], dtype=torch.float), batch_dim=1)
        )
    )


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_median_along_seq(dtype: torch.dtype) -> None:
    values, indices = BatchedTensorSeq(
        torch.arange(10).view(2, 5).to(dtype=dtype)
    ).median_along_seq()
    assert values.equal(BatchedTensor(torch.tensor([2, 7], dtype=dtype)))
    assert indices.equal(BatchedTensor(torch.tensor([2, 2])))


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_median_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    values, indices = BatchedTensorSeq(
        torch.arange(10).view(2, 5).to(dtype=dtype)
    ).median_along_seq(keepdim=True)
    assert values.equal(BatchedTensorSeq(torch.tensor([[2], [7]], dtype=dtype)))
    assert indices.equal(BatchedTensorSeq(torch.tensor([[2], [2]])))


def test_batched_tensor_seq_median_along_seq_custom_dims() -> None:
    values, indices = BatchedTensorSeq(
        torch.tensor([[2, 4], [1, 5], [0, 2]]), batch_dim=1, seq_dim=0
    ).median_along_seq()
    assert values.equal(BatchedTensor(torch.tensor([1, 4])))
    assert indices.equal(BatchedTensor(torch.tensor([1, 0])))


def test_batched_tensor_seq_median_along_seq_keepdim_true_custom_dims() -> None:
    values, indices = BatchedTensorSeq.from_seq_batch(torch.arange(10).view(5, 2)).median_along_seq(
        keepdim=True
    )
    assert values.equal(BatchedTensorSeq(torch.tensor([[4, 5]]), batch_dim=1, seq_dim=0))
    assert indices.equal(BatchedTensorSeq(torch.tensor([[2, 2]]), batch_dim=1, seq_dim=0))


def test_batched_tensor_seq_median_along_seq_extra_dims() -> None:
    values, indices = BatchedTensorSeq(
        torch.arange(20).view(2, 5, 2), batch_dim=2, seq_dim=1
    ).median_along_seq()
    assert values.equal(BatchedTensor(torch.tensor([[4, 5], [14, 15]]), batch_dim=1))
    assert indices.equal(BatchedTensor(torch.tensor([[2, 2], [2, 2]]), batch_dim=1))


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_min_along_seq(dtype: torch.dtype) -> None:
    values, indices = BatchedTensorSeq(torch.arange(10).view(2, 5).to(dtype=dtype)).min_along_seq()
    assert values.equal(BatchedTensor(torch.tensor([0, 5], dtype=dtype)))
    assert indices.equal(BatchedTensor(torch.tensor([0, 0])))


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_min_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    values, indices = BatchedTensorSeq(torch.arange(10).view(2, 5).to(dtype=dtype)).min_along_seq(
        keepdim=True
    )
    assert values.equal(BatchedTensorSeq(torch.tensor([[0], [5]], dtype=dtype)))
    assert indices.equal(BatchedTensorSeq(torch.tensor([[0], [0]])))


def test_batched_tensor_seq_min_along_seq_custom_dims() -> None:
    values, indices = BatchedTensorSeq(
        torch.tensor([[0, 4], [1, 2], [2, 5]]), batch_dim=1, seq_dim=0
    ).min_along_seq()
    assert values.equal(BatchedTensor(torch.tensor([0, 2])))
    assert indices.equal(BatchedTensor(torch.tensor([0, 1])))


def test_batched_tensor_seq_min_along_seq_keepdim_true_custom_dims() -> None:
    values, indices = BatchedTensorSeq.from_seq_batch(torch.arange(10).view(5, 2)).min_along_seq(
        keepdim=True
    )
    assert values.equal(BatchedTensorSeq(torch.tensor([[0, 1]]), batch_dim=1, seq_dim=0))
    assert indices.equal(BatchedTensorSeq(torch.tensor([[0, 0]]), batch_dim=1, seq_dim=0))


def test_batched_tensor_seq_min_along_seq_extra_dims() -> None:
    values, indices = BatchedTensorSeq(
        torch.arange(20).view(2, 5, 2), batch_dim=2, seq_dim=1
    ).min_along_seq()
    assert values.equal(BatchedTensor(torch.tensor([[0, 1], [10, 11]]), batch_dim=1))
    assert indices.equal(BatchedTensor(torch.tensor([[0, 0], [0, 0]]), batch_dim=1))


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_sum_along_seq(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5).to(dtype=dtype))
        .sum_along_seq()
        .equal(BatchedTensor(torch.tensor([10, 35], dtype=dtype)))
    )


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_sum_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5).to(dtype=dtype))
        .sum_along_seq(keepdim=True)
        .equal(BatchedTensorSeq(torch.tensor([[10], [35]], dtype=dtype)))
    )


def test_batched_tensor_seq_sum_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 4], [1, 2], [2, 5]]), batch_dim=1, seq_dim=0)
        .sum_along_seq()
        .equal(BatchedTensor(torch.tensor([3, 11])))
    )


def test_batched_tensor_seq_sum_along_seq_keepdim_true_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.arange(10).view(5, 2))
        .sum_along_seq(keepdim=True)
        .equal(BatchedTensorSeq(torch.tensor([[20, 25]]), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_sum_along_seq_extra_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(20).view(2, 5, 2), batch_dim=2, seq_dim=1)
        .sum_along_seq()
        .equal(BatchedTensor(torch.tensor([[20, 25], [70, 75]]), batch_dim=1))
    )


##########################################################
#    Indexing, slicing, joining, mutating operations     #
##########################################################


def test_batched_tensor_seq__getitem___int() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    assert batch[0].equal(torch.tensor([0, 1, 2, 3, 4]))


def test_batched_tensor_seq__range___range() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    assert batch[0:2, 2:4].equal(torch.tensor([[2, 3], [7, 8]]))


@mark.parametrize(
    "index",
    (
        torch.tensor([[1, 3], [0, 4]]),
        BatchedTensor(torch.tensor([[1, 3], [0, 4]])),
        BatchedTensorSeq(torch.tensor([[1, 3], [0, 4]])),
    ),
)
def test_batched_tensor_seq__getitem___tensor_like(index: Union[Tensor, BaseBatchedTensor]) -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    assert batch[0].equal(torch.tensor([0, 1, 2, 3, 4]))


def test_batched_tensor_seq__setitem___int() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch[0] = 7
    assert batch.equal(BatchedTensorSeq(torch.tensor([[7, 7, 7, 7, 7], [5, 6, 7, 8, 9]])))


@mark.parametrize(
    "value",
    (
        torch.tensor([[0, -4]]),
        BatchedTensor(torch.tensor([[0, -4]])),
        BatchedTensorSeq(torch.tensor([[0, -4]])),
    ),
)
def test_batched_tensor_seq__setitem___range(value: Union[Tensor, BaseBatchedTensor]) -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch[1:2, 2:4] = value
    assert batch.equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 0, -4, 9]])))


def test_batched_tensor_seq_align_to_batch_seq_no_permutation() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .align_to_batch_seq()
        .equal(BatchedTensorSeq(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])))
    )


def test_batched_tensor_seq_align_to_batch_seq_permute_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0)
        .align_to_batch_seq()
        .equal(BatchedTensorSeq(torch.tensor([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]])))
    )


def test_batched_tensor_seq_align_to_batch_seq_permute_dims_extra_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(20).view(2, 5, 2), batch_dim=1, seq_dim=2)
        .align_to_batch_seq()
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [[0, 10], [1, 11]],
                        [[2, 12], [3, 13]],
                        [[4, 14], [5, 15]],
                        [[6, 16], [7, 17]],
                        [[8, 18], [9, 19]],
                    ]
                )
            )
        )
    )


def test_batched_tensor_seq_align_to_seq_batch_no_permutation() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .align_to_seq_batch()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), batch_dim=1, seq_dim=0
            )
        )
    )


def test_batched_tensor_seq_align_to_seq_batch_permute_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=0, seq_dim=1)
        .align_to_seq_batch()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]), batch_dim=1, seq_dim=0
            )
        )
    )


def test_batched_tensor_seq_align_to_seq_batch_permute_dims_extra_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(20).view(2, 5, 2), batch_dim=1, seq_dim=2)
        .align_to_seq_batch()
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [[0, 10], [2, 12], [4, 14], [6, 16], [8, 18]],
                        [[1, 11], [3, 13], [5, 15], [7, 17], [9, 19]],
                    ]
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        torch.tensor([[10, 11, 12], [13, 14, 15]]),
    ),
)
def test_batched_tensor_seq_append(other: BaseBatchedTensor | Tensor) -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.append(other)
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))
    )


def test_batched_tensor_seq_append_custom_dims() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1, seq_dim=0)
    batch.append(
        BatchedTensorSeq(torch.tensor([[10, 12], [11, 13], [14, 15]]), batch_dim=1, seq_dim=0)
    )
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
            batch_dim=1,
            seq_dim=0,
        )
    )


def test_batched_tensor_seq_append_custom_dims_seq_dim_2() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 4), batch_dim=2)
    batch.append(BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2))
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3, 5), batch_dim=2))


def test_batched_tensor_seq_append_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.append(BatchedTensorSeq(torch.zeros(2, 3, 1), batch_dim=2))


def test_batched_tensor_seq_append_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.append(BatchedTensorSeq(torch.zeros(2, 3, 1), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        torch.tensor([[10, 11, 12], [13, 14, 15]]),
        [BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]]))],
        (BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]])),),
    ),
)
def test_batched_tensor_seq_cat_along_batch(
    other: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor],
) -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
        .cat_along_batch(other)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]])))
    )


def test_batched_tensor_seq_cat_along_batch_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1, seq_dim=0)
        .cat_along_batch(
            BatchedTensorSeq(torch.tensor([[10, 12], [11, 13], [14, 15]]), batch_dim=1, seq_dim=0)
        )
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_cat_along_batch_custom_dims_seq_dim_2() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3, 4), batch_dim=2)
        .cat_along_batch(BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2))
        .equal(BatchedTensorSeq(torch.ones(2, 3, 5), batch_dim=2))
    )


def test_batched_tensor_seq_cat_along_batch_multiple() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
        .cat_along_batch(
            [
                BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]])),
                BatchedTensor(torch.tensor([[20, 21, 22]])),
                torch.tensor([[30, 31, 32]]),
            ]
        )
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15], [20, 21, 22], [30, 31, 32]]
                )
            )
        )
    )


def test_batched_tensor_seq_cat_along_batch_empty() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .cat_along_batch([])
        .equal(BatchedTensorSeq(torch.ones(2, 3)))
    )


def test_batched_tensor_seq_cat_along_batch_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_along_batch([BatchedTensorSeq(torch.zeros(2, 3, 1), batch_dim=2)])


def test_batched_tensor_seq_cat_along_batch_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.cat_along_batch([BatchedTensorSeq(torch.zeros(2, 3, 1), seq_dim=2)])


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        torch.tensor([[10, 11, 12], [13, 14, 15]]),
        [BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]]))],
        (BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]])),),
    ),
)
def test_batched_tensor_seq_cat_along_batch_(
    other: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor],
) -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.cat_along_batch_(other)
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))
    )


def test_batched_tensor_seq_cat_along_batch__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1, seq_dim=0)
    batch.cat_along_batch_(
        BatchedTensorSeq(torch.tensor([[10, 12], [11, 13], [14, 15]]), batch_dim=1, seq_dim=0)
    )
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
            batch_dim=1,
            seq_dim=0,
        )
    )


def test_batched_tensor_seq_cat_along_batch__custom_dims_seq_dim_2() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 4), batch_dim=2)
    batch.cat_along_batch_(BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2))
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3, 5), batch_dim=2))


def test_batched_tensor_seq_cat_along_batch__multiple() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.cat_along_batch_(
        [
            BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]])),
            BatchedTensor(torch.tensor([[20, 21, 22]])),
            torch.tensor([[30, 31, 32]]),
        ]
    )
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor(
                [[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15], [20, 21, 22], [30, 31, 32]]
            )
        )
    )


def test_batched_tensor_seq_cat_along_batch__empty() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.cat_along_batch_([])
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3)))


def test_batched_tensor_seq_cat_along_batch__incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_along_batch_([BatchedTensorSeq(torch.zeros(2, 3, 1), batch_dim=2)])


def test_batched_tensor_seq_cat_along_batch__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.cat_along_batch_([BatchedTensorSeq(torch.zeros(2, 3, 1), seq_dim=2)])


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]])),
        BatchedTensor(torch.tensor([[10, 11], [12, 13]])),
        torch.tensor([[10, 11], [12, 13]]),
        [BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]]))],
        (BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]])),),
    ),
)
def test_batched_tensor_seq_cat_along_seq(
    other: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor],
) -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
        .cat_along_seq(other)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 10, 11], [4, 5, 6, 12, 13]])))
    )


def test_batched_tensor_seq_cat_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1, seq_dim=0)
        .cat_along_seq(BatchedTensorSeq(torch.tensor([[10, 12], [11, 13]]), batch_dim=1, seq_dim=0))
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0, 4], [1, 5], [2, 6], [10, 12], [11, 13]]), batch_dim=1, seq_dim=0
            )
        )
    )


def test_batched_tensor_seq_cat_along_seq_custom_dims_seq_dim_2() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3, 4), seq_dim=2)
        .cat_along_seq(BatchedTensorSeq(torch.ones(2, 3, 1), seq_dim=2))
        .equal(BatchedTensorSeq(torch.ones(2, 3, 5), seq_dim=2))
    )


def test_batched_tensor_seq_cat_along_seq_multiple() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
        .cat_along_seq(
            [
                BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]])),
                BatchedTensor(torch.tensor([[20, 21], [22, 23]])),
                torch.tensor([[30, 31, 32], [33, 34, 35]]),
            ]
        )
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[0, 1, 2, 10, 11, 20, 21, 30, 31, 32], [4, 5, 6, 12, 13, 22, 23, 33, 34, 35]]
                )
            )
        )
    )


def test_batched_tensor_seq_cat_along_seq_empty() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .cat_along_seq([])
        .equal(BatchedTensorSeq(torch.ones(2, 3)))
    )


def test_batched_tensor_seq_cat_along_seq_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_along_seq([BatchedTensorSeq(torch.zeros(2, 3, 1), batch_dim=2)])


def test_batched_tensor_seq_cat_along_seq_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.cat_along_seq([BatchedTensorSeq(torch.zeros(2, 3, 1), seq_dim=2)])


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]])),
        BatchedTensor(torch.tensor([[10, 11], [12, 13]])),
        torch.tensor([[10, 11], [12, 13]]),
        [BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]]))],
        (BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]])),),
    ),
)
def test_batched_tensor_seq_cat_along_seq_(
    other: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor],
) -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.cat_along_seq_(other)
    assert batch.equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 10, 11], [4, 5, 6, 12, 13]])))


def test_batched_tensor_seq_cat_along_seq__custom_dims() -> None:
    batch = BatchedTensorSeq.from_seq_batch(torch.tensor([[0, 4], [1, 5], [2, 6]]))
    batch.cat_along_seq_(BatchedTensorSeq.from_seq_batch(torch.tensor([[10, 12], [11, 13]])))
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[0, 4], [1, 5], [2, 6], [10, 12], [11, 13]]), batch_dim=1, seq_dim=0
        )
    )


def test_batched_tensor_seq_cat_along_seq__custom_dims_seq_dim_2() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 4), seq_dim=2)
    batch.cat_along_seq_(BatchedTensorSeq(torch.ones(2, 3, 1), seq_dim=2))
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3, 5), seq_dim=2))


def test_batched_tensor_seq_cat_along_seq__multiple() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.cat_along_seq_(
        [
            BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]])),
            BatchedTensor(torch.tensor([[20, 21], [22, 23]])),
            torch.tensor([[30, 31, 32], [33, 34, 35]]),
        ]
    )
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor(
                [[0, 1, 2, 10, 11, 20, 21, 30, 31, 32], [4, 5, 6, 12, 13, 22, 23, 33, 34, 35]]
            )
        )
    )


def test_batched_tensor_seq_cat_along_seq__empty() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.cat_along_seq_([])
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3)))


def test_batched_tensor_seq_cat_along_seq__incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_along_seq_([BatchedTensorSeq(torch.zeros(2, 3, 1), batch_dim=2)])


def test_batched_tensor_seq_cat_along_seq__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.cat_along_seq_([BatchedTensorSeq(torch.zeros(2, 3, 1), seq_dim=2)])


@mark.parametrize(
    "other",
    (
        [BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]]))],
        [
            BatchedTensorSeq(torch.tensor([[10, 11, 12]])),
            BatchedTensorSeq(torch.tensor([[13, 14, 15]])),
        ],
        (BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]])),),
    ),
)
def test_batched_tensor_seq_extend(
    other: Iterable[BaseBatchedTensor | Tensor],
) -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.extend(other)
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))
    )


def test_batched_tensor_seq_extend_custom_dims() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1, seq_dim=0)
    batch.extend(
        [BatchedTensorSeq(torch.tensor([[10, 12], [11, 13], [14, 15]]), batch_dim=1, seq_dim=0)]
    )
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
            batch_dim=1,
            seq_dim=0,
        )
    )


def test_batched_tensor_seq_extend_custom_dims_seq_dim_2() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 4), batch_dim=2)
    batch.extend(BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=2))
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3, 5), batch_dim=2))


def test_batched_tensor_seq_extend_multiple() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.extend(
        [
            BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]])),
            BatchedTensor(torch.tensor([[20, 21, 22]])),
            torch.tensor([[30, 31, 32]]),
        ]
    )
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor(
                [[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15], [20, 21, 22], [30, 31, 32]]
            )
        )
    )


def test_batched_tensor_seq_extend_empty() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.extend([])
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3)))


def test_batched_tensor_seq_extend_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.extend([BatchedTensorSeq(torch.zeros(2, 3, 1), batch_dim=2)])


def test_batched_tensor_seq_extend_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.extend([BatchedTensorSeq(torch.zeros(2, 3, 1), seq_dim=2)])


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
