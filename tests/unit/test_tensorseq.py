from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from coola import objects_are_equal
from pytest import mark, raises
from torch import Tensor
from torch.overrides import is_tensor_like

from redcat import BaseBatch, BatchedTensor, BatchedTensorSeq, BatchList
from redcat.tensor import IndexType
from redcat.tensorseq import (
    check_data_and_dims,
    check_seq_dims,
    from_sequences,
    get_seq_dims,
)
from redcat.utils.tensor import get_available_devices, get_torch_generator

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


def test_batched_tensor_seq_shape() -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).shape == torch.Size([2, 3])


def test_batched_tensor_seq_dim() -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).dim() == 2


def test_batched_tensor_seq_ndimension() -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).ndimension() == 2


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
def test_batched_tensor_seq_new_full_fill_value(fill_value: float | int | bool) -> None:
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
        BatchedTensorSeq(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq__eq__(other: BatchedTensor | Tensor | int | float) -> None:
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
        BatchedTensorSeq(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq__ge__(other: BatchedTensor | Tensor | int | float) -> None:
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
        BatchedTensorSeq(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq__gt__(other: BatchedTensor | Tensor | int | float) -> None:
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
        BatchedTensorSeq(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq__le__(other: BatchedTensor | Tensor | int | float) -> None:
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
        BatchedTensorSeq(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq__lt__(other: BatchedTensor | Tensor | int | float) -> None:
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
    assert not BatchedTensorSeq(torch.ones(2, 3, dtype=torch.float)).allclose(
        torch.ones(2, 3, dtype=torch.long)
    )


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
def test_batched_tensor_seq_eq(other: BatchedTensor | Tensor | int | float) -> None:
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
    assert not BatchedTensorSeq(torch.ones(2, 3, dtype=torch.float)).equal(
        torch.ones(2, 3, dtype=torch.long)
    )


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
        BatchedTensorSeq(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq_ge(other: BatchedTensor | Tensor | int | float) -> None:
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
        BatchedTensorSeq(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq_gt(other: BatchedTensor | Tensor | int | float) -> None:
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
        BatchedTensorSeq(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq_le(other: BatchedTensor | Tensor | int | float) -> None:
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
        BatchedTensorSeq(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_seq_lt(other: BatchedTensor | Tensor | int | float) -> None:
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
        BatchedTensor(torch.ones(2, 1)),
        torch.ones(2, 3),
        1,
        1.0,
    ),
)
def test_batched_tensor_seq__add__(other: BatchedTensor | Tensor | int | float) -> None:
    assert (BatchedTensorSeq(torch.zeros(2, 3)) + other).equal(BatchedTensorSeq(torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.ones(2, 3)),
        BatchedTensor(torch.ones(2, 3)),
        BatchedTensor(torch.ones(2, 1)),
        torch.ones(2, 3),
        1,
        1.0,
    ),
)
def test_batched_tensor_seq__iadd__(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensorSeq(torch.zeros(2, 3))
    batch += other
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq__floordiv__(other: BatchedTensor | Tensor | int | float) -> None:
    assert (BatchedTensorSeq(torch.ones(2, 3)) // other).equal(BatchedTensorSeq(torch.zeros(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq__ifloordiv__(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch //= other
    assert batch.equal(BatchedTensorSeq(torch.zeros(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq__mul__(other: BatchedTensor | Tensor | int | float) -> None:
    assert (BatchedTensorSeq(torch.ones(2, 3)) * other).equal(
        BatchedTensorSeq(torch.full((2, 3), 2.0))
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq__imul__(other: BatchedTensor | Tensor | int | float) -> None:
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
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq__sub__(other: BatchedTensor | Tensor | int | float) -> None:
    assert (BatchedTensorSeq(torch.ones(2, 3)) - other).equal(BatchedTensorSeq(-torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq__isub__(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch -= other
    assert batch.equal(BatchedTensorSeq(-torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq__truediv__(other: BatchedTensor | Tensor | int | float) -> None:
    assert (BatchedTensorSeq(torch.ones(2, 3)) / other).equal(
        BatchedTensorSeq(torch.full((2, 3), 0.5))
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq__itruediv__(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch /= other
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 0.5)))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_add(other: BatchedTensor | Tensor | int | float) -> None:
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
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.add(BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_add_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.add(BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=0, seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_add_(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.add_(other)
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 3.0)))


def test_batched_tensor_seq_add__alpha_2_float() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.add_(BatchedTensorSeq(torch.full((2, 3), 2.0)), alpha=2.0)
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 5.0)))


def test_batched_tensor_seq_add__alpha_2_long() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, dtype=torch.long))
    batch.add_(BatchedTensorSeq(torch.full((2, 3), 2, dtype=torch.long)), alpha=2)
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 5, dtype=torch.long)))


def test_batched_tensor_seq_add__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
    batch.add_(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 3.0), batch_dim=1, seq_dim=0))


def test_batched_tensor_seq_add__incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.add_(BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_add__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.add_(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_div(other: BatchedTensor | Tensor | int | float) -> None:
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
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.div(BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_div_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.div(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_div_(other: BatchedTensor | Tensor | int | float) -> None:
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
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.div_(BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_div__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.div_(BatchedTensorSeq(torch.ones(2, 2, 2), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_fmod(other: BatchedTensor | Tensor | int | float) -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).fmod(other).equal(BatchedTensorSeq(torch.ones(2, 3)))


def test_batched_tensor_seq_fmod_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .fmod(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
        .equal(BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_fmod_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.fmod(BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_fmod_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.fmod(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_fmod_(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.fmod_(other)
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3)))


def test_batched_tensor_seq_fmod__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
    batch.fmod_(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0))


def test_batched_tensor_seq_fmod__incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.fmod_(BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_fmod__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.fmod_(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_mul(other: BatchedTensor | Tensor | int | float) -> None:
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
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.mul(BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_mul_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.mul(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_mul_(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.mul_(other)
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 2.0)))


def test_batched_tensor_seq_mul__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
    batch.mul_(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))


def test_batched_tensor_seq_mul__incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.mul_(BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_mul__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.mul_(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


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
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_sub(other: BatchedTensor | Tensor | int | float) -> None:
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
        .sub(BatchedTensorSeq(torch.full((2, 3), 2, dtype=torch.long)), alpha=2)
        .equal(BatchedTensorSeq(-torch.full((2, 3), 3, dtype=torch.long)))
    )


def test_batched_tensor_seq_sub_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .sub(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
        .equal(BatchedTensorSeq(-torch.ones(2, 3), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_sub_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.sub(BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_sub_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.sub(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 1), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_seq_sub_(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.sub_(other)
    assert batch.equal(BatchedTensorSeq(-torch.ones(2, 3)))


def test_batched_tensor_seq_sub__alpha_2_float() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.sub_(BatchedTensorSeq(torch.full((2, 3), 2.0)), alpha=2.0)
    assert batch.equal(BatchedTensorSeq(-torch.full((2, 3), 3.0)))


def test_batched_tensor_seq_sub__alpha_2_long() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3, dtype=torch.long))
    batch.sub_(BatchedTensorSeq(torch.full((2, 3), 2, dtype=torch.long)), alpha=2)
    assert batch.equal(BatchedTensorSeq(-torch.full((2, 3), 3, dtype=torch.long)))


def test_batched_tensor_seq_sub__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
    batch.sub_(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
    assert batch.equal(BatchedTensorSeq(-torch.ones(2, 3), batch_dim=1, seq_dim=0))


def test_batched_tensor_seq_sub__incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.sub_(BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_sub__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.sub_(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


###########################################################
#     Mathematical | advanced arithmetical operations     #
###########################################################


def test_batched_tensor_seq_argsort_descending_false() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).argsort(
            descending=False
        ),
        BatchedTensorSeq(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]])),
    )


def test_batched_tensor_seq_argsort_descending_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).argsort(descending=True),
        BatchedTensorSeq(torch.tensor([[3, 0, 4, 2, 1], [0, 4, 1, 3, 2]])),
    )


def test_batched_tensor_seq_argsort_dim_0() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])).argsort(dim=0),
        BatchedTensorSeq(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
    )


def test_batched_tensor_seq_argsort_dim_1() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [[0, 1], [-2, 3], [-4, 5], [-6, 7], [-8, 9]],
                    [[10, -11], [12, -13], [14, -15], [16, -17], [18, -19]],
                ]
            )
        ).argsort(dim=1),
        BatchedTensorSeq(
            torch.tensor(
                [
                    [[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]],
                    [[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]],
                ]
            )
        ),
    )


def test_batched_tensor_seq_argsort_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), seq_dim=0, batch_dim=1
        ).argsort(dim=0),
        BatchedTensorSeq(
            torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]), seq_dim=0, batch_dim=1
        ),
    )


def test_batched_tensor_seq_argsort_along_batch_descending_false() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])
        ).argsort_along_batch(),
        BatchedTensorSeq(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
    )


def test_batched_tensor_seq_argsort_along_batch_descending_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])
        ).argsort_along_batch(descending=True),
        BatchedTensorSeq(torch.tensor([[3, 0], [0, 4], [4, 1], [2, 3], [1, 2]])),
    )


def test_batched_tensor_seq_argsort_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), seq_dim=0, batch_dim=1
        ).argsort_along_batch(),
        BatchedTensorSeq(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]), seq_dim=0, batch_dim=1),
    )


def test_batched_tensor_seq_argsort_along_seq_descending_false() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).argsort_along_seq(),
        BatchedTensorSeq(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]])),
    )


def test_batched_tensor_seq_argsort_along_seq_descending_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).argsort_along_seq(
            descending=True
        ),
        BatchedTensorSeq(torch.tensor([[3, 0, 4, 2, 1], [0, 4, 1, 3, 2]])),
    )


def test_batched_tensor_seq_argsort_along_seq_dim_3() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [[0, 1], [-2, 3], [-4, 5], [-6, 7], [-8, 9]],
                    [[10, -11], [12, -13], [14, -15], [16, -17], [18, -19]],
                ]
            )
        ).argsort_along_seq(),
        BatchedTensorSeq(
            torch.tensor(
                [
                    [[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]],
                    [[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]],
                ]
            )
        ),
    )


def test_batched_tensor_seq_argsort_along_seq_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), seq_dim=0, batch_dim=1
        ).argsort_along_seq(),
        BatchedTensorSeq(
            torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]), seq_dim=0, batch_dim=1
        ),
    )


def test_batched_tensor_seq_cumsum_dim_0() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .cumsum(dim=0)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])))
    )


def test_batched_tensor_seq_cumsum_dim_1() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .cumsum(dim=1)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 3, 6, 10], [5, 11, 18, 26, 35]])))
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


def test_batched_tensor_seq_logcumsumexp_dim_0() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(5, 2))
        .logcumsumexp(dim=0)
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [0.0, 1.0],
                        [2.1269280110429727, 3.1269280110429727],
                        [4.142931628499899, 5.142931628499899],
                        [6.145077938960783, 7.145077938960783],
                        [8.145368056908488, 9.145368056908488],
                    ]
                )
            )
        )
    )


def test_batched_tensor_seq_logcumsumexp_dim_1() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5))
        .logcumsumexp(dim=1)
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [
                            0.0,
                            1.3132616875182228,
                            2.40760596444438,
                            3.4401896985611953,
                            4.451914395937593,
                        ],
                        [
                            5.0,
                            6.313261687518223,
                            7.407605964444381,
                            8.440189698561195,
                            9.451914395937592,
                        ],
                    ]
                )
            )
        )
    )


def test_batched_tensor_seq_logcumsumexp_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.arange(10, dtype=torch.float).view(5, 2))
        .logcumsumexp(dim=0)
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [0.0, 1.0],
                        [2.1269280110429727, 3.1269280110429727],
                        [4.142931628499899, 5.142931628499899],
                        [6.145077938960783, 7.145077938960783],
                        [8.145368056908488, 9.145368056908488],
                    ]
                ),
                seq_dim=0,
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_seq_logcumsumexp__dim_0() -> None:
    batch = BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(5, 2))
    batch.logcumsumexp_(dim=0)
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [0.0, 1.0],
                    [2.1269280110429727, 3.1269280110429727],
                    [4.142931628499899, 5.142931628499899],
                    [6.145077938960783, 7.145077938960783],
                    [8.145368056908488, 9.145368056908488],
                ]
            )
        )
    )


def test_batched_tensor_seq_logcumsumexp__dim_1() -> None:
    batch = BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5))
    batch.logcumsumexp_(dim=1)
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [
                        0.0,
                        1.3132616875182228,
                        2.40760596444438,
                        3.4401896985611953,
                        4.451914395937593,
                    ],
                    [
                        5.0,
                        6.313261687518223,
                        7.407605964444381,
                        8.440189698561195,
                        9.451914395937592,
                    ],
                ]
            )
        )
    )


def test_batched_tensor_seq_logcumsumexp__custom_dims() -> None:
    batch = BatchedTensorSeq.from_seq_batch(torch.arange(10, dtype=torch.float).view(5, 2))
    batch.logcumsumexp_(dim=0)
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [0.0, 1.0],
                    [2.1269280110429727, 3.1269280110429727],
                    [4.142931628499899, 5.142931628499899],
                    [6.145077938960783, 7.145077938960783],
                    [8.145368056908488, 9.145368056908488],
                ]
            ),
            seq_dim=0,
            batch_dim=1,
        )
    )


def test_batched_tensor_seq_logcumsumexp_along_batch() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(5, 2))
        .logcumsumexp_along_batch()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [0.0, 1.0],
                        [2.1269280110429727, 3.1269280110429727],
                        [4.142931628499899, 5.142931628499899],
                        [6.145077938960783, 7.145077938960783],
                        [8.145368056908488, 9.145368056908488],
                    ]
                )
            )
        )
    )


def test_batched_tensor_seq_logcumsumexp_along_batch_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.arange(10, dtype=torch.float).view(2, 5))
        .logcumsumexp_along_batch()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [
                            0.0,
                            1.3132616875182228,
                            2.40760596444438,
                            3.4401896985611953,
                            4.451914395937593,
                        ],
                        [
                            5.0,
                            6.313261687518223,
                            7.407605964444381,
                            8.440189698561195,
                            9.451914395937592,
                        ],
                    ]
                ),
                seq_dim=0,
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_seq_logcumsumexp_along_batch_() -> None:
    batch = BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(5, 2))
    batch.logcumsumexp_along_batch_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [0.0, 1.0],
                    [2.1269280110429727, 3.1269280110429727],
                    [4.142931628499899, 5.142931628499899],
                    [6.145077938960783, 7.145077938960783],
                    [8.145368056908488, 9.145368056908488],
                ]
            )
        )
    )


def test_batched_tensor_seq_logcumsumexp_along_batch__custom_dims() -> None:
    batch = BatchedTensorSeq.from_seq_batch(torch.arange(10, dtype=torch.float).view(2, 5))
    batch.logcumsumexp_along_batch_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [
                        0.0,
                        1.3132616875182228,
                        2.40760596444438,
                        3.4401896985611953,
                        4.451914395937593,
                    ],
                    [
                        5.0,
                        6.313261687518223,
                        7.407605964444381,
                        8.440189698561195,
                        9.451914395937592,
                    ],
                ]
            ),
            seq_dim=0,
            batch_dim=1,
        )
    )


def test_batched_tensor_seq_logcumsumexp_along_seq() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5))
        .logcumsumexp_along_seq()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [
                            0.0,
                            1.3132616875182228,
                            2.40760596444438,
                            3.4401896985611953,
                            4.451914395937593,
                        ],
                        [
                            5.0,
                            6.313261687518223,
                            7.407605964444381,
                            8.440189698561195,
                            9.451914395937592,
                        ],
                    ]
                )
            )
        )
    )


def test_batched_tensor_seq_logcumsumexp_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.arange(10, dtype=torch.float).view(5, 2))
        .logcumsumexp_along_seq()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [0.0, 1.0],
                        [2.1269280110429727, 3.1269280110429727],
                        [4.142931628499899, 5.142931628499899],
                        [6.145077938960783, 7.145077938960783],
                        [8.145368056908488, 9.145368056908488],
                    ]
                ),
                seq_dim=0,
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_seq_logcumsumexp_along_seq_() -> None:
    batch = BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5))
    batch.logcumsumexp_along_seq_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [
                        0.0,
                        1.3132616875182228,
                        2.40760596444438,
                        3.4401896985611953,
                        4.451914395937593,
                    ],
                    [
                        5.0,
                        6.313261687518223,
                        7.407605964444381,
                        8.440189698561195,
                        9.451914395937592,
                    ],
                ]
            )
        )
    )


def test_batched_tensor_seq_logcumsumexp_along_seq__custom_dims() -> None:
    batch = BatchedTensorSeq.from_seq_batch(torch.arange(10, dtype=torch.float).view(5, 2))
    batch.logcumsumexp_along_seq_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [0.0, 1.0],
                    [2.1269280110429727, 3.1269280110429727],
                    [4.142931628499899, 5.142931628499899],
                    [6.145077938960783, 7.145077938960783],
                    [8.145368056908488, 9.145368056908488],
                ]
            ),
            seq_dim=0,
            batch_dim=1,
        )
    )


@mark.parametrize("permutation", (torch.tensor([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_tensor_seq_permute_along_batch(permutation: Sequence[int] | Tensor) -> None:
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


@mark.parametrize("permutation", (torch.tensor([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_tensor_seq_permute_along_batch_(permutation: Sequence[int] | Tensor) -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch.permute_along_batch_(permutation)
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]]))
    )


def test_batched_tensor_seq_permute_along_batch__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), batch_dim=1, seq_dim=0)
    batch.permute_along_batch_(torch.tensor([2, 1, 3, 0]))
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4]]), batch_dim=1, seq_dim=0)
    )


@mark.parametrize("permutation", (torch.tensor([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_tensor_seq_permute_along_dim_0(permutation: Sequence[int] | Tensor) -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
        .permute_along_dim(permutation, dim=0)
        .equal(BatchedTensorSeq(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))
    )


@mark.parametrize("permutation", (torch.tensor([2, 4, 1, 3, 0]), [2, 4, 1, 3, 0], (2, 4, 1, 3, 0)))
def test_batched_tensor_seq_permute_along_dim_1(permutation: Sequence[int] | Tensor) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .permute_along_dim(permutation, dim=1)
        .equal(BatchedTensorSeq(torch.tensor([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])))
    )


def test_batched_tensor_seq_permute_along_dim_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), batch_dim=1, seq_dim=0)
        .permute_along_dim(torch.tensor([2, 1, 3, 0]), dim=1)
        .equal(BatchedTensorSeq(torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4]]), batch_dim=1, seq_dim=0))
    )


@mark.parametrize("permutation", (torch.tensor([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_tensor_seq_permute_along_dim__0(permutation: Sequence[int] | Tensor) -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch.permute_along_dim_(permutation, dim=0)
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]]))
    )


@mark.parametrize("permutation", (torch.tensor([2, 4, 1, 3, 0]), [2, 4, 1, 3, 0], (2, 4, 1, 3, 0)))
def test_batched_tensor_seq_permute_along_dim__1(permutation: Sequence[int] | Tensor) -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch.permute_along_dim_(permutation, dim=1)
    assert batch.equal(BatchedTensorSeq(torch.tensor([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])))


def test_batched_tensor_seq_permute_along_dim__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), batch_dim=1, seq_dim=0)
    batch.permute_along_dim_(torch.tensor([2, 1, 3, 0]), dim=1)
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4]]), batch_dim=1, seq_dim=0)
    )


@mark.parametrize("permutation", (torch.tensor([2, 4, 1, 3, 0]), [2, 4, 1, 3, 0], (2, 4, 1, 3, 0)))
def test_batched_tensor_seq_permute_along_seq(permutation: Sequence[int] | Tensor) -> None:
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


@mark.parametrize("permutation", (torch.tensor([2, 4, 1, 3, 0]), [2, 4, 1, 3, 0], (2, 4, 1, 3, 0)))
def test_batched_tensor_seq_permute_along_seq_(permutation: Sequence[int] | Tensor) -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch.permute_along_seq_(permutation)
    assert batch.equal(BatchedTensorSeq(torch.tensor([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])))


def test_batched_tensor_seq_permute_along_seq__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0)
    batch.permute_along_seq_(torch.tensor([2, 4, 1, 3, 0]))
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[4, 5], [8, 9], [2, 3], [6, 7], [0, 1]]), batch_dim=1, seq_dim=0
        )
    )


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_seq_shuffle_along_batch() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
        .shuffle_along_batch()
        .equal(BatchedTensorSeq(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))
    )


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
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


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_seq_shuffle_along_batch_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch.shuffle_along_batch_()
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]]))
    )


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_seq_shuffle_along_batch__custom_dims() -> None:
    batch = BatchedTensorSeq.from_seq_batch(
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    )
    batch.shuffle_along_batch_()
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]]), batch_dim=1, seq_dim=0
        )
    )


def test_batched_tensor_seq_shuffle_along_batch__same_random_seed() -> None:
    batch1 = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch1.shuffle_along_batch_(get_torch_generator(1))
    batch2 = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch2.shuffle_along_batch_(get_torch_generator(1))
    assert batch1.equal(batch2)


def test_batched_tensor_seq_shuffle_along_batch__different_random_seeds() -> None:
    batch1 = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch1.shuffle_along_batch_(get_torch_generator(1))
    batch2 = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch2.shuffle_along_batch_(get_torch_generator(2))
    assert not batch1.equal(batch2)


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_seq_shuffle_along_dim() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
        .shuffle_along_dim(dim=0)
        .equal(BatchedTensorSeq(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))
    )


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_seq_shuffle_along_dim_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]))
        .shuffle_along_dim(dim=1)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]]), batch_dim=1, seq_dim=0
            )
        )
    )


def test_batched_tensor_seq_shuffle_along_dim_same_random_seed() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert batch.shuffle_along_dim(dim=0, generator=get_torch_generator(1)).equal(
        batch.shuffle_along_dim(dim=0, generator=get_torch_generator(1))
    )


def test_batched_tensor_seq_shuffle_along_dim_different_random_seeds() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert not batch.shuffle_along_dim(dim=0, generator=get_torch_generator(1)).equal(
        batch.shuffle_along_dim(dim=0, generator=get_torch_generator(2))
    )


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_seq_shuffle_along_dim_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch.shuffle_along_dim_(dim=0)
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]]))
    )


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_seq_shuffle_along_dim__custom_dims() -> None:
    batch = BatchedTensorSeq.from_seq_batch(
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    )
    batch.shuffle_along_dim_(dim=1)
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]]), batch_dim=1, seq_dim=0
        )
    )


def test_batched_tensor_seq_shuffle_along_dim__same_random_seed() -> None:
    batch1 = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch1.shuffle_along_dim_(dim=0, generator=get_torch_generator(1))
    batch2 = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch2.shuffle_along_dim_(dim=0, generator=get_torch_generator(1))
    assert batch1.equal(batch2)


def test_batched_tensor_seq_shuffle_along_dim__different_random_seeds() -> None:
    batch1 = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch1.shuffle_along_dim_(dim=0, generator=get_torch_generator(1))
    batch2 = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch2.shuffle_along_dim_(dim=0, generator=get_torch_generator(2))
    assert not batch1.equal(batch2)


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 4, 1, 3, 0]))
def test_batched_tensor_seq_shuffle_along_seq() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .shuffle_along_seq()
        .equal(BatchedTensorSeq(torch.tensor([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])))
    )


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_seq_shuffle_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(
            torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        )
        .shuffle_along_seq()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]]), batch_dim=1, seq_dim=0
            )
        )
    )


def test_batched_tensor_seq_shuffle_along_seq_same_random_seed() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert batch.shuffle_along_seq(get_torch_generator(1)).equal(
        batch.shuffle_along_seq(get_torch_generator(1))
    )


def test_batched_tensor_seq_shuffle_along_seq_different_random_seeds() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert not batch.shuffle_along_seq(get_torch_generator(1)).equal(
        batch.shuffle_along_seq(get_torch_generator(2))
    )


def test_batched_tensor_seq_shuffle_along_seq_multiple_shuffle() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    generator = get_torch_generator(1)
    assert not batch.shuffle_along_seq(generator).equal(batch.shuffle_along_seq(generator))


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 4, 1, 3, 0]))
def test_batched_tensor_seq_shuffle_along_seq_() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch.shuffle_along_seq_()
    assert batch.equal(BatchedTensorSeq(torch.tensor([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])))


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_seq_shuffle_along_seq__custom_dims() -> None:
    batch = BatchedTensorSeq.from_seq_batch(
        torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    )
    batch.shuffle_along_seq_()
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]]), batch_dim=1, seq_dim=0
        )
    )


def test_batched_tensor_seq_shuffle_along_seq__same_random_seed() -> None:
    batch1 = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch1.shuffle_along_seq_(get_torch_generator(1))
    batch2 = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch2.shuffle_along_seq_(get_torch_generator(1))
    assert batch1.equal(batch2)


def test_batched_tensor_seq_shuffle_along_seq__different_random_seeds() -> None:
    batch1 = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch1.shuffle_along_seq_(get_torch_generator(1))
    batch2 = BatchedTensorSeq(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch2.shuffle_along_seq_(get_torch_generator(2))
    assert not batch1.equal(batch2)


def test_batched_tensor_seq_sort() -> None:
    values, indices = BatchedTensorSeq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).sort()
    assert objects_are_equal(
        values, BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]))
    )
    assert objects_are_equal(
        indices, BatchedTensorSeq(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]))
    )


def test_batched_tensor_seq_sort_namedtuple() -> None:
    out = BatchedTensorSeq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).sort()
    assert objects_are_equal(
        out.values, BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]))
    )
    assert objects_are_equal(
        out.indices, BatchedTensorSeq(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]))
    )


def test_batched_tensor_seq_sort_descending_false() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).sort(
                descending=False
            )
        ),
        (
            BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])),
            BatchedTensorSeq(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]])),
        ),
    )


def test_batched_tensor_seq_sort_descending_true() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).sort(descending=True)
        ),
        (
            BatchedTensorSeq(torch.tensor([[5, 4, 3, 2, 1], [9, 8, 7, 6, 5]])),
            BatchedTensorSeq(torch.tensor([[3, 0, 4, 2, 1], [0, 4, 1, 3, 2]])),
        ),
    )


def test_batched_tensor_seq_sort_dim_0() -> None:
    assert objects_are_equal(
        tuple(BatchedTensorSeq(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])).sort(dim=0)),
        (
            BatchedTensorSeq(torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])),
            BatchedTensorSeq(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
        ),
    )


def test_batched_tensor_seq_sort_dim_1() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [[0, 1], [-2, 3], [-4, 5], [-6, 7], [-8, 9]],
                        [[10, -11], [12, -13], [14, -15], [16, -17], [18, -19]],
                    ]
                )
            ).sort(dim=1)
        ),
        (
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [[-8, 1], [-6, 3], [-4, 5], [-2, 7], [0, 9]],
                        [[10, -19], [12, -17], [14, -15], [16, -13], [18, -11]],
                    ]
                )
            ),
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]],
                        [[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]],
                    ]
                )
            ),
        ),
    )


def test_batched_tensor_seq_sort_custom_dims() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(
                torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), seq_dim=0, batch_dim=1
            ).sort(dim=0)
        ),
        (
            BatchedTensorSeq(
                torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]), seq_dim=0, batch_dim=1
            ),
            BatchedTensorSeq(
                torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]), seq_dim=0, batch_dim=1
            ),
        ),
    )


def test_batched_tensor_seq_sort_along_batch() -> None:
    values, indices = BatchedTensorSeq(
        torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])
    ).sort_along_batch()
    assert objects_are_equal(
        values, BatchedTensorSeq(torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]))
    )
    assert objects_are_equal(
        indices, BatchedTensorSeq(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]))
    )


def test_batched_tensor_seq_sort_along_batch_namedtuple() -> None:
    out = BatchedTensorSeq(
        torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])
    ).sort_along_batch()
    assert objects_are_equal(
        out.values, BatchedTensorSeq(torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]))
    )
    assert objects_are_equal(
        out.indices, BatchedTensorSeq(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]))
    )


def test_batched_tensor_seq_sort_along_batch_descending_false() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(
                torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])
            ).sort_along_batch()
        ),
        (
            BatchedTensorSeq(torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])),
            BatchedTensorSeq(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
        ),
    )


def test_batched_tensor_seq_sort_along_batch_descending_true() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(
                torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])
            ).sort_along_batch(descending=True)
        ),
        (
            BatchedTensorSeq(torch.tensor([[5, 9], [4, 8], [3, 7], [2, 6], [1, 5]])),
            BatchedTensorSeq(torch.tensor([[3, 0], [0, 4], [4, 1], [2, 3], [1, 2]])),
        ),
    )


def test_batched_tensor_seq_sort_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(
                torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), seq_dim=0, batch_dim=1
            ).sort_along_batch()
        ),
        (
            BatchedTensorSeq(
                torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]), seq_dim=0, batch_dim=1
            ),
            BatchedTensorSeq(
                torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]), seq_dim=0, batch_dim=1
            ),
        ),
    )


def test_batched_tensor_seq_sort_along_seq() -> None:
    values, indices = BatchedTensorSeq(
        torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])
    ).sort_along_seq()
    assert objects_are_equal(
        values, BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]))
    )
    assert objects_are_equal(
        indices, BatchedTensorSeq(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]))
    )


def test_batched_tensor_seq_sort_along_seq_namedtuple() -> None:
    out = BatchedTensorSeq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).sort_along_seq()
    assert objects_are_equal(
        out.values, BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]))
    )
    assert objects_are_equal(
        out.indices, BatchedTensorSeq(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]))
    )


def test_batched_tensor_seq_sort_along_seq_descending_false() -> None:
    assert objects_are_equal(
        tuple(BatchedTensorSeq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).sort_along_seq()),
        (
            BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])),
            BatchedTensorSeq(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]])),
        ),
    )


def test_batched_tensor_seq_sort_along_seq_descending_true() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).sort_along_seq(
                descending=True
            )
        ),
        (
            BatchedTensorSeq(torch.tensor([[5, 4, 3, 2, 1], [9, 8, 7, 6, 5]])),
            BatchedTensorSeq(torch.tensor([[3, 0, 4, 2, 1], [0, 4, 1, 3, 2]])),
        ),
    )


def test_batched_tensor_seq_sort_along_seq_dim_3() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [[0, 1], [-2, 3], [-4, 5], [-6, 7], [-8, 9]],
                        [[10, -11], [12, -13], [14, -15], [16, -17], [18, -19]],
                    ]
                )
            ).sort_along_seq()
        ),
        (
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [[-8, 1], [-6, 3], [-4, 5], [-2, 7], [0, 9]],
                        [[10, -19], [12, -17], [14, -15], [16, -13], [18, -11]],
                    ]
                )
            ),
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]],
                        [[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]],
                    ]
                )
            ),
        ),
    )


def test_batched_tensor_seq_sort_along_seq_custom_dims() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(
                torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), seq_dim=0, batch_dim=1
            ).sort_along_seq()
        ),
        (
            BatchedTensorSeq(
                torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]), seq_dim=0, batch_dim=1
            ),
            BatchedTensorSeq(
                torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]), seq_dim=0, batch_dim=1
            ),
        ),
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
        .clamp(min=2, max=5)
        .equal(BatchedTensorSeq(torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]])))
    )


def test_batched_tensor_seq_clamp_only_max() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .clamp(max=5)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 5, 5, 5, 5]])))
    )


def test_batched_tensor_seq_clamp_only_min() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .clamp(min=2)
        .equal(BatchedTensorSeq(torch.tensor([[2, 2, 2, 3, 4], [5, 6, 7, 8, 9]])))
    )


def test_batched_tensor_seq_clamp_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .clamp(min=2, max=5)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]]), batch_dim=1, seq_dim=0
            )
        )
    )


def test_batched_tensor_seq_clamp_() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch.clamp_(min=2, max=5)
    assert batch.equal(BatchedTensorSeq(torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]])))


def test_batched_tensor_seq_clamp__only_max() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch.clamp_(max=5)
    assert batch.equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 5, 5, 5, 5]])))


def test_batched_tensor_seq_clamp__only_min() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch.clamp_(min=2)
    assert batch.equal(BatchedTensorSeq(torch.tensor([[2, 2, 2, 3, 4], [5, 6, 7, 8, 9]])))


def test_batched_tensor_seq_clamp__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
    batch.clamp_(min=2, max=5)
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


def test_batched_tensor_seq_log10() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
        .log10()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [0.0, 0.3010300099849701, 0.4771212637424469],
                        [0.6020600199699402, 0.6989700198173523, 0.778151273727417],
                    ]
                )
            )
        )
    )


def test_batched_tensor_seq_log10_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        )
        .log10()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [0.0, 0.3010300099849701, 0.4771212637424469],
                        [0.6020600199699402, 0.6989700198173523, 0.778151273727417],
                    ]
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_log10_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
    batch.log10_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [0.0, 0.3010300099849701, 0.4771212637424469],
                    [0.6020600199699402, 0.6989700198173523, 0.778151273727417],
                ]
            )
        )
    )


def test_batched_tensor_seq_log10__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float),
        batch_dim=1,
        seq_dim=0,
    )
    batch.log10_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [0.0, 0.3010300099849701, 0.4771212637424469],
                    [0.6020600199699402, 0.6989700198173523, 0.778151273727417],
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


def test_batched_tensor_seq_log2() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
        .log2()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [[0.0, 1.0, 1.5849624872207642], [2.0, 2.321928024291992, 2.5849626064300537]]
                )
            )
        )
    )


def test_batched_tensor_seq_log2_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        )
        .log2()
        .allclose(
            BatchedTensorSeq(
                torch.tensor(
                    [[0.0, 1.0, 1.5849624872207642], [2.0, 2.321928024291992, 2.5849626064300537]]
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_log2_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
    batch.log2_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [[0.0, 1.0, 1.5849624872207642], [2.0, 2.321928024291992, 2.5849626064300537]]
            )
        )
    )


def test_batched_tensor_seq_log2__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float),
        batch_dim=1,
        seq_dim=0,
    )
    batch.log2_()
    assert batch.allclose(
        BatchedTensorSeq(
            torch.tensor(
                [[0.0, 1.0, 1.5849624872207642], [2.0, 2.321928024291992, 2.5849626064300537]]
            ),
            batch_dim=1,
            seq_dim=0,
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        torch.tensor([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_batched_tensor_seq_maximum(other: BatchedTensor | Tensor) -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [-2, -1, 0]]))
        .maximum(other)
        .equal(BatchedTensorSeq(torch.tensor([[2, 1, 2], [0, 1, 0]])))
    )


def test_batched_tensor_seq_maximum_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [-2, -1, 0]]), batch_dim=1, seq_dim=0)
        .maximum(BatchedTensorSeq(torch.tensor([[2, 0, 1], [0, 1, 0]]), batch_dim=1, seq_dim=0))
        .equal(BatchedTensorSeq(torch.tensor([[2, 1, 2], [0, 1, 0]]), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_maximum_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.maximum(BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_maximum_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.maximum(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        torch.tensor([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_batched_tensor_seq_minimum(other: BatchedTensor | Tensor) -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [-2, -1, 0]]))
        .minimum(other)
        .equal(BatchedTensorSeq(torch.tensor([[0, 0, 1], [-2, -1, 0]])))
    )


def test_batched_tensor_seq_minimum_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [-2, -1, 0]]), batch_dim=1, seq_dim=0)
        .minimum(BatchedTensorSeq(torch.tensor([[2, 0, 1], [0, 1, 0]]), batch_dim=1, seq_dim=0))
        .equal(BatchedTensorSeq(torch.tensor([[0, 0, 1], [-2, -1, 0]]), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_minimum_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.minimum(BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_minimum_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.minimum(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


@mark.parametrize(
    "exponent",
    (BatchedTensorSeq(torch.full((2, 5), 2.0)), BatchedTensor(torch.full((2, 5), 2.0)), 2, 2.0),
)
def test_batched_tensor_seq_pow(exponent: BatchedTensor | int | float) -> None:
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
        .pow(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_pow_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.pow(BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_pow_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.pow(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


@mark.parametrize(
    "exponent",
    (BatchedTensorSeq(torch.full((2, 5), 2.0)), BatchedTensor(torch.full((2, 5), 2.0)), 2, 2.0),
)
def test_batched_tensor_seq_pow_(exponent: BatchedTensor | int | float) -> None:
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
    batch.pow_(BatchedTensorSeq(torch.full((2, 3), 2.0), batch_dim=1, seq_dim=0))
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        )
    )


def test_batched_tensor_seq_pow__incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 2, 2)).pow_(
            BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2)
        )


def test_batched_tensor_seq_pow__incorrect_seq_dim() -> None:
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        BatchedTensorSeq(torch.ones(2, 2, 2)).pow_(
            BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2)
        )


def test_batched_tensor_seq_rsqrt() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1.0, 4.0], [16.0, 25.0]], dtype=torch.float))
        .rsqrt()
        .equal(BatchedTensorSeq(torch.tensor([[1.0, 0.5], [0.25, 0.2]], dtype=torch.float)))
    )


def test_batched_tensor_seq_rsqrt_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[1.0, 4.0], [16.0, 25.0]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        )
        .rsqrt()
        .equal(
            BatchedTensorSeq(
                torch.tensor([[1.0, 0.5], [0.25, 0.2]], dtype=torch.float),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_rsqrt_() -> None:
    batch = BatchedTensorSeq(torch.tensor([[1.0, 4.0], [16.0, 25.0]], dtype=torch.float))
    batch.rsqrt_()
    assert batch.equal(BatchedTensorSeq(torch.tensor([[1.0, 0.5], [0.25, 0.2]], dtype=torch.float)))


def test_batched_tensor_seq_rsqrt__custom_dims() -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[1.0, 4.0], [16.0, 25.0]], dtype=torch.float),
        batch_dim=1,
        seq_dim=0,
    )
    batch.rsqrt_()
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[1.0, 0.5], [0.25, 0.2]], dtype=torch.float),
            batch_dim=1,
            seq_dim=0,
        )
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
def test_batched_tensor_seq_logical_and(other: BatchedTensor | Tensor, dtype: torch.dtype) -> None:
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
    batch = BatchedTensorSeq(torch.zeros(2, 2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_and(
            BatchedTensorSeq(
                torch.zeros(2, 2, 2, dtype=torch.bool),
                batch_dim=2,
            )
        )


def test_batched_tensor_seq_logical_and_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.logical_and(BatchedTensorSeq(torch.zeros(2, 2, 2, dtype=torch.bool), seq_dim=2))


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
def test_batched_tensor_seq_logical_and_(other: BatchedTensor | Tensor, dtype: torch.dtype) -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
    )
    batch.logical_and_(other)
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[True, False, False, False], [True, False, True, False]], dtype=dtype)
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
    batch = BatchedTensorSeq(torch.zeros(2, 2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_and_(
            BatchedTensorSeq(
                torch.zeros(2, 2, 2, dtype=torch.bool),
                batch_dim=2,
            )
        )


def test_batched_tensor_seq_logical_and__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.logical_and_(BatchedTensorSeq(torch.zeros(2, 2, 2, dtype=torch.bool), seq_dim=2))


@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_seq_logical_not(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
        )
        .logical_not()
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[False, False, True, True], [False, True, False, True]], dtype=torch.bool
                )
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
def test_batched_tensor_seq_logical_or(other: BatchedTensor | Tensor, dtype: torch.dtype) -> None:
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
    batch = BatchedTensorSeq(torch.zeros(2, 2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_or(
            BatchedTensorSeq(
                torch.zeros(2, 2, 2, dtype=torch.bool),
                batch_dim=2,
            )
        )


def test_batched_tensor_seq_logical_or_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.logical_or(BatchedTensorSeq(torch.zeros(2, 2, 2, dtype=torch.bool), seq_dim=2))


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
def test_batched_tensor_seq_logical_or_(other: BatchedTensor | Tensor, dtype: torch.dtype) -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
    )
    batch.logical_or_(other)
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[True, True, True, False], [True, True, True, True]], dtype=dtype)
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
    batch = BatchedTensorSeq(torch.zeros(2, 2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_or_(
            BatchedTensorSeq(
                torch.zeros(2, 2, 2, dtype=torch.bool),
                batch_dim=2,
            )
        )


def test_batched_tensor_seq_logical_or__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.logical_or_(BatchedTensorSeq(torch.zeros(2, 2, 2, dtype=torch.bool), seq_dim=2))


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
def test_batched_tensor_seq_logical_xor(other: BatchedTensor | Tensor, dtype: torch.dtype) -> None:
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
    batch = BatchedTensorSeq(torch.zeros(2, 2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_xor(
            BatchedTensorSeq(
                torch.zeros(2, 2, 2, dtype=torch.bool),
                batch_dim=2,
            )
        )


def test_batched_tensor_seq_logical_xor_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.logical_xor(BatchedTensorSeq(torch.zeros(2, 2, 2, dtype=torch.bool), seq_dim=2))


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
def test_batched_tensor_seq_logical_xor_(other: BatchedTensor | Tensor, dtype: torch.dtype) -> None:
    batch = BatchedTensorSeq(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
    )
    batch.logical_xor_(other)
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[False, True, True, False], [False, True, False, True]], dtype=dtype)
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
    batch = BatchedTensorSeq(torch.zeros(2, 2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_xor_(
            BatchedTensorSeq(
                torch.zeros(2, 2, 2, dtype=torch.bool),
                batch_dim=2,
            )
        )


def test_batched_tensor_seq_logical_xor__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.logical_xor_(BatchedTensorSeq(torch.zeros(2, 2, 2, dtype=torch.bool), seq_dim=2))


################################
#     Reduction operations     #
################################


def test_batched_tensor_seq_amax_dim_none() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(2, 5)).amax(dim=None).equal(torch.tensor(9))


def test_batched_tensor_seq_amax_dim_0() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(5, 2)).amax(dim=0).equal(torch.tensor([8, 9]))


def test_batched_tensor_seq_amax_dim_1() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(2, 5)).amax(dim=1).equal(torch.tensor([4, 9]))


def test_batched_tensor_seq_amax_keepdim_false() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).amax(dim=1), torch.tensor([4, 9])
    )


def test_batched_tensor_seq_amax_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).amax(dim=1, keepdim=True),
        torch.tensor([[4], [9]]),
    )


def test_batched_tensor_seq_amax_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .amax(dim=1)
        .equal(torch.tensor([4, 9]))
    )


def test_batched_tensor_seq_amax_along_batch() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])).amax_along_batch(),
        torch.tensor([4, 9]),
    )


def test_batched_tensor_seq_amax_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])).amax_along_batch(
            keepdim=True
        ),
        torch.tensor([[4, 9]]),
    )


def test_batched_tensor_seq_amax_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0).amax_along_batch(),
        torch.tensor([4, 9]),
    )


def test_batched_tensor_seq_amax_along_seq() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).amax_along_seq(), torch.tensor([4, 9])
    )


def test_batched_tensor_seq_amax_along_seq_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).amax_along_seq(keepdim=True),
        torch.tensor([[4], [9]]),
    )


def test_batched_tensor_seq_amax_along_seq_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]), batch_dim=1, seq_dim=0
        ).amax_along_seq(),
        torch.tensor([4, 9]),
    )


def test_batched_tensor_seq_amin_dim_none() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(2, 5)).amin(dim=None).equal(torch.tensor(0))


def test_batched_tensor_seq_amin_dim_0() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(5, 2)).amin(dim=0).equal(torch.tensor([0, 1]))


def test_batched_tensor_seq_amin_dim_1() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(2, 5)).amin(dim=1).equal(torch.tensor([0, 5]))


def test_batched_tensor_seq_amin_keepdim_false() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).amin(dim=1), torch.tensor([0, 5])
    )


def test_batched_tensor_seq_amin_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).amin(dim=1, keepdim=True),
        torch.tensor([[0], [5]]),
    )


def test_batched_tensor_seq_amin_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .amin(dim=1)
        .equal(torch.tensor([0, 5]))
    )


def test_batched_tensor_seq_amin_along_batch() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])).amin_along_batch(),
        torch.tensor([0, 5]),
    )


def test_batched_tensor_seq_amin_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])).amin_along_batch(
            keepdim=True
        ),
        torch.tensor([[0, 5]]),
    )


def test_batched_tensor_seq_amin_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0).amin_along_batch(),
        torch.tensor([0, 5]),
    )


def test_batched_tensor_seq_amin_along_seq() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).amin_along_seq(), torch.tensor([0, 5])
    )


def test_batched_tensor_seq_amin_along_seq_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).amin_along_seq(keepdim=True),
        torch.tensor([[0], [5]]),
    )


def test_batched_tensor_seq_amin_along_seq_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]), batch_dim=1, seq_dim=0
        ).amin_along_seq(),
        torch.tensor([0, 5]),
    )


def test_batched_tensor_seq_argmax() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(2, 5)).argmax().equal(torch.tensor(9))


def test_batched_tensor_seq_argmax_dim_0() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(5, 2)).argmax(dim=0).equal(torch.tensor([4, 4]))


def test_batched_tensor_seq_argmax_dim_1() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(2, 5)).argmax(dim=1).equal(torch.tensor([4, 4]))


def test_batched_tensor_seq_argmax_keepdim_false() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).argmax(dim=1), torch.tensor([4, 4])
    )


def test_batched_tensor_seq_argmax_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).argmax(dim=1, keepdim=True),
        torch.tensor([[4], [4]]),
    )


def test_batched_tensor_seq_argmax_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .argmax(dim=1)
        .equal(torch.tensor([4, 4]))
    )


def test_batched_tensor_seq_argmax_along_batch() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[0, 9], [1, 6], [2, 7], [3, 8], [4, 5]])
        ).argmax_along_batch(),
        torch.tensor([4, 0]),
    )


def test_batched_tensor_seq_argmax_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.tensor([[0, 9], [1, 6], [2, 7], [3, 8], [4, 5]])).argmax_along_batch(
            keepdim=True
        ),
        torch.tensor([[4, 0]]),
    )


def test_batched_tensor_seq_argmax_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0).argmax_along_batch(),
        torch.tensor([4, 4]),
    )


def test_batched_tensor_seq_argmax_along_seq() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).argmax_along_seq(), torch.tensor([4, 4])
    )


def test_batched_tensor_seq_argmax_along_seq_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).argmax_along_seq(keepdim=True),
        torch.tensor([[4], [4]]),
    )


def test_batched_tensor_seq_argmax_along_seq_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[0, 9], [1, 6], [2, 7], [3, 8], [4, 5]]), batch_dim=1, seq_dim=0
        ).argmax_along_seq(),
        torch.tensor([4, 0]),
    )


def test_batched_tensor_seq_argmin() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(2, 5)).argmin().equal(torch.tensor(0))


def test_batched_tensor_seq_argmin_dim_0() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(5, 2)).argmin(dim=0).equal(torch.tensor([0, 0]))


def test_batched_tensor_seq_argmin_dim_1() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(2, 5)).argmin(dim=1).equal(torch.tensor([0, 0]))


def test_batched_tensor_seq_argmin_keepdim_false() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).argmin(dim=1), torch.tensor([0, 0])
    )


def test_batched_tensor_seq_argmin_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).argmin(dim=1, keepdim=True),
        torch.tensor([[0], [0]]),
    )


def test_batched_tensor_seq_argmin_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .argmin(dim=1)
        .equal(torch.tensor([0, 0]))
    )


def test_batched_tensor_seq_argmin_along_batch() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[0, 9], [1, 6], [2, 7], [3, 8], [4, 5]])
        ).argmin_along_batch(),
        torch.tensor([0, 4]),
    )


def test_batched_tensor_seq_argmin_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.tensor([[0, 9], [1, 6], [2, 7], [3, 8], [4, 5]])).argmin_along_batch(
            keepdim=True
        ),
        torch.tensor([[0, 4]]),
    )


def test_batched_tensor_seq_argmin_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0).argmin_along_batch(),
        torch.tensor([0, 0]),
    )


def test_batched_tensor_seq_argmin_along_seq() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).argmin_along_seq(), torch.tensor([0, 0])
    )


def test_batched_tensor_seq_argmin_along_seq_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).argmin_along_seq(keepdim=True),
        torch.tensor([[0], [0]]),
    )


def test_batched_tensor_seq_argmin_along_seq_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[0, 9], [1, 6], [2, 7], [3, 8], [4, 5]]), batch_dim=1, seq_dim=0
        ).argmin_along_seq(),
        torch.tensor([0, 4]),
    )


def test_batched_tensor_seq_max() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(2, 5)).max().equal(torch.tensor(9))


def test_batched_tensor_seq_max_dim_tuple() -> None:
    values, indices = BatchedTensorSeq(torch.arange(10).view(2, 5)).max(dim=1)
    assert values.equal(torch.tensor([4, 9]))
    assert indices.equal(torch.tensor([4, 4]))


def test_batched_tensor_seq_max_dim_namedtuple() -> None:
    out = BatchedTensorSeq(torch.arange(10).view(2, 5)).max(dim=1)
    assert out.values.equal(torch.tensor([4, 9]))
    assert out.indices.equal(torch.tensor([4, 4]))


def test_batched_tensor_seq_max_keepdim_false() -> None:
    assert objects_are_equal(
        tuple(BatchedTensorSeq(torch.arange(10).view(2, 5)).max(dim=1)),
        (torch.tensor([4, 9]), torch.tensor([4, 4])),
    )


def test_batched_tensor_seq_max_keepdim_true() -> None:
    assert objects_are_equal(
        tuple(BatchedTensorSeq(torch.arange(10).view(2, 5)).max(dim=1, keepdim=True)),
        (torch.tensor([[4], [9]]), torch.tensor([[4], [4]])),
    )


def test_batched_tensor_seq_max_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .max()
        .equal(torch.tensor(9))
    )


def test_batched_tensor_seq_max_along_batch() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(
                torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
            ).max_along_batch()
        ),
        (torch.tensor([4, 9]), torch.tensor([4, 4])),
    )


def test_batched_tensor_seq_max_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(
                torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
            ).max_along_batch(keepdim=True)
        ),
        (torch.tensor([[4, 9]]), torch.tensor([[4, 4]])),
    )


def test_batched_tensor_seq_max_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0).max_along_batch()
        ),
        (torch.tensor([4, 9]), torch.tensor([4, 4])),
    )


def test_batched_tensor_seq_max_along_seq() -> None:
    assert objects_are_equal(
        tuple(BatchedTensorSeq(torch.arange(10).view(2, 5)).max_along_seq()),
        (torch.tensor([4, 9]), torch.tensor([4, 4])),
    )


def test_batched_tensor_seq_max_along_seq_keepdim_true() -> None:
    assert objects_are_equal(
        tuple(BatchedTensorSeq(torch.arange(10).view(2, 5)).max_along_seq(keepdim=True)),
        (torch.tensor([[4], [9]]), torch.tensor([[4], [4]])),
    )


def test_batched_tensor_seq_max_along_seq_custom_dims() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(
                torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]), batch_dim=1, seq_dim=0
            ).max_along_seq()
        ),
        (torch.tensor([4, 9]), torch.tensor([4, 4])),
    )


def test_batched_tensor_seq_mean() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5))
        .mean()
        .equal(torch.tensor(4.5))
    )


def test_batched_tensor_seq_mean_keepdim_false() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5))
        .mean(dim=1)
        .equal(torch.tensor([2.0, 7.0]))
    )


def test_batched_tensor_seq_mean_keepdim_true() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5))
        .mean(dim=1, keepdim=True)
        .equal(torch.tensor([[2.0], [7.0]]))
    )


def test_batched_tensor_seq_mean_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5), batch_dim=1, seq_dim=0)
        .mean()
        .equal(torch.tensor(4.5))
    )


def test_batched_tensor_seq_mean_along_batch() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(5, 2))
        .mean_along_batch()
        .equal(torch.tensor([4.0, 5.0], dtype=torch.float))
    )


def test_batched_tensor_seq_mean_along_batch_keepdim_true() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(5, 2))
        .mean_along_batch(keepdim=True)
        .equal(torch.tensor([[4.0, 5.0]], dtype=torch.float))
    )


def test_batched_tensor_seq_mean_along_batch_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.arange(10, dtype=torch.float).view(2, 5))
        .mean_along_batch()
        .equal(torch.tensor([2.0, 7.0], dtype=torch.float))
    )


def test_batched_tensor_seq_mean_along_seq() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5))
        .mean_along_seq()
        .equal(torch.tensor([2.0, 7.0], dtype=torch.float))
    )


def test_batched_tensor_seq_mean_along_seq_keepdim_true() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5))
        .mean_along_seq(keepdim=True)
        .equal(torch.tensor([[2.0], [7.0]], dtype=torch.float))
    )


def test_batched_tensor_seq_mean_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.arange(10, dtype=torch.float).view(5, 2))
        .mean_along_seq()
        .equal(torch.tensor([4.0, 5.0], dtype=torch.float))
    )


def test_batched_tensor_seq_median() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(2, 5)).median().equal(torch.tensor(4))


def test_batched_tensor_seq_median_keepdim_false() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).median(dim=1),
        torch.return_types.median([torch.tensor([2, 7]), torch.tensor([2, 2])]),
    )


def test_batched_tensor_seq_median_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).median(dim=1, keepdim=True),
        torch.return_types.median([torch.tensor([[2], [7]]), torch.tensor([[2], [2]])]),
    )


def test_batched_tensor_seq_median_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .median()
        .equal(torch.tensor(4))
    )


def test_batched_tensor_seq_median_along_batch() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
        ).median_along_batch(),
        torch.return_types.median([torch.tensor([2, 7]), torch.tensor([2, 2])]),
    )


def test_batched_tensor_seq_median_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])).median_along_batch(
            keepdim=True
        ),
        torch.return_types.median([torch.tensor([[2, 7]]), torch.tensor([[2, 2]])]),
    )


def test_batched_tensor_seq_median_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0).median_along_batch(),
        torch.return_types.median([torch.tensor([2, 7]), torch.tensor([2, 2])]),
    )


def test_batched_tensor_seq_median_along_seq() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).median_along_seq(),
        torch.return_types.median([torch.tensor([2, 7]), torch.tensor([2, 2])]),
    )


def test_batched_tensor_seq_median_along_seq_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).median_along_seq(keepdim=True),
        torch.return_types.median([torch.tensor([[2], [7]]), torch.tensor([[2], [2]])]),
    )


def test_batched_tensor_seq_median_along_seq_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]), batch_dim=1, seq_dim=0
        ).median_along_seq(),
        torch.return_types.median([torch.tensor([2, 7]), torch.tensor([2, 2])]),
    )


def test_batched_tensor_seq_min() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(2, 5)).min().equal(torch.tensor(0))


def test_batched_tensor_seq_min_dim_tuple() -> None:
    values, indices = BatchedTensorSeq(torch.arange(10).view(2, 5)).min(dim=1)
    assert values.equal(torch.tensor([0, 5]))
    assert indices.equal(torch.tensor([0, 0]))


def test_batched_tensor_seq_min_dim_namedtuple() -> None:
    out = BatchedTensorSeq(torch.arange(10).view(2, 5)).min(dim=1)
    assert out.values.equal(torch.tensor([0, 5]))
    assert out.indices.equal(torch.tensor([0, 0]))


def test_batched_tensor_seq_min_keepdim_false() -> None:
    assert objects_are_equal(
        tuple(BatchedTensorSeq(torch.arange(10).view(2, 5)).min(dim=1)),
        (torch.tensor([0, 5]), torch.tensor([0, 0])),
    )


def test_batched_tensor_seq_min_keepdim_true() -> None:
    assert objects_are_equal(
        tuple(BatchedTensorSeq(torch.arange(10).view(2, 5)).min(dim=1, keepdim=True)),
        (torch.tensor([[0], [5]]), torch.tensor([[0], [0]])),
    )


def test_batched_tensor_seq_min_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .min()
        .equal(torch.tensor(0))
    )


def test_batched_tensor_seq_min_along_batch() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(
                torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
            ).min_along_batch()
        ),
        (torch.tensor([0, 5]), torch.tensor([0, 0])),
    )


def test_batched_tensor_seq_min_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(
                torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
            ).min_along_batch(keepdim=True)
        ),
        (torch.tensor([[0, 5]]), torch.tensor([[0, 0]])),
    )


def test_batched_tensor_seq_min_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0).min_along_batch()
        ),
        (torch.tensor([0, 5]), torch.tensor([0, 0])),
    )


def test_batched_tensor_seq_min_along_seq() -> None:
    assert objects_are_equal(
        tuple(BatchedTensorSeq(torch.arange(10).view(2, 5)).min_along_seq()),
        (torch.tensor([0, 5]), torch.tensor([0, 0])),
    )


def test_batched_tensor_seq_min_along_seq_keepdim_true() -> None:
    assert objects_are_equal(
        tuple(BatchedTensorSeq(torch.arange(10).view(2, 5)).min_along_seq(keepdim=True)),
        (torch.tensor([[0], [5]]), torch.tensor([[0], [0]])),
    )


def test_batched_tensor_seq_min_along_seq_custom_dims() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensorSeq(
                torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]), batch_dim=1, seq_dim=0
            ).min_along_seq()
        ),
        (torch.tensor([0, 5]), torch.tensor([0, 0])),
    )


def test_batched_tensor_seq_nanmean() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nanmean()
        .equal(torch.tensor(4.0))
    )


def test_batched_tensor_seq_nanmean_keepdim_false() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nanmean(dim=1)
        .equal(torch.tensor([2.0, 6.5]))
    )


def test_batched_tensor_seq_nanmean_keepdim_true() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nanmean(dim=1, keepdim=True)
        .equal(torch.tensor([[2.0], [6.5]]))
    )


def test_batched_tensor_seq_nanmean_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]), batch_dim=1, seq_dim=0
        )
        .nanmean()
        .equal(torch.tensor(4.0))
    )


def test_batched_tensor_seq_nanmean_along_batch() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]])
        )
        .nanmean_along_batch()
        .equal(torch.tensor([2.0, 6.5]))
    )


def test_batched_tensor_seq_nanmean_along_batch_keepdim_true() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]])
        )
        .nanmean_along_batch(keepdim=True)
        .equal(torch.tensor([[2.0, 6.5]]))
    )


def test_batched_tensor_seq_nanmean_along_batch_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]), batch_dim=1, seq_dim=0
        )
        .nanmean_along_batch()
        .equal(torch.tensor([2.0, 6.5]))
    )


def test_batched_tensor_seq_nanmean_along_seq() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nanmean_along_seq()
        .equal(torch.tensor([2.0, 6.5]))
    )


def test_batched_tensor_seq_nanmean_along_seq_keepdim_true() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nanmean_along_seq(keepdim=True)
        .equal(torch.tensor([[2.0], [6.5]]))
    )


def test_batched_tensor_seq_nanmean_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]]),
            batch_dim=1,
            seq_dim=0,
        )
        .nanmean_along_seq()
        .equal(torch.tensor([2.0, 6.5]))
    )


def test_batched_tensor_seq_nanmedian() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nanmedian()
        .equal(torch.tensor(4.0))
    )


def test_batched_tensor_seq_nanmedian_keepdim_false() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])).nanmedian(
            dim=1
        ),
        torch.return_types.nanmedian([torch.tensor([2.0, 6.0]), torch.tensor([2, 1])]),
    )


def test_batched_tensor_seq_nanmedian_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])).nanmedian(
            dim=1, keepdim=True
        ),
        torch.return_types.nanmedian([torch.tensor([[2.0], [6.0]]), torch.tensor([[2], [1]])]),
    )


def test_batched_tensor_seq_nanmedian_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]), batch_dim=1, seq_dim=0
        )
        .nanmedian()
        .equal(torch.tensor(4.0))
    )


def test_batched_tensor_seq_nanmedian_along_batch() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]])
        ).nanmedian_along_batch(),
        torch.return_types.nanmedian([torch.tensor([2.0, 6.0]), torch.tensor([2, 1])]),
    )


def test_batched_tensor_seq_nanmedian_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]])
        ).nanmedian_along_batch(keepdim=True),
        torch.return_types.nanmedian([torch.tensor([[2.0, 6.0]]), torch.tensor([[2, 1]])]),
    )


def test_batched_tensor_seq_nanmedian_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]), batch_dim=1, seq_dim=0
        ).nanmedian_along_batch(),
        torch.return_types.nanmedian([torch.tensor([2.0, 6.0]), torch.tensor([2, 1])]),
    )


def test_batched_tensor_seq_nanmedian_along_seq() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])
        ).nanmedian_along_seq(),
        torch.return_types.nanmedian([torch.tensor([2.0, 6.0]), torch.tensor([2, 1])]),
    )


def test_batched_tensor_seq_nanmedian_along_seq_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])
        ).nanmedian_along_seq(keepdim=True),
        torch.return_types.nanmedian([torch.tensor([[2.0], [6.0]]), torch.tensor([[2], [1]])]),
    )


def test_batched_tensor_seq_nanmedian_along_seq_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(
            torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]]),
            batch_dim=1,
            seq_dim=0,
        ).nanmedian_along_seq(),
        torch.return_types.nanmedian([torch.tensor([2.0, 6.0]), torch.tensor([2, 1])]),
    )


def test_batched_tensor_seq_nansum() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nansum()
        .equal(torch.tensor(36.0))
    )


def test_batched_tensor_seq_nansum_keepdim_false() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nansum(dim=1)
        .equal(torch.tensor([10.0, 26.0]))
    )


def test_batched_tensor_seq_nansum_keepdim_true() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nansum(dim=1, keepdim=True)
        .equal(torch.tensor([[10.0], [26.0]]))
    )


def test_batched_tensor_seq_nansum_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]), batch_dim=1, seq_dim=0
        )
        .nansum()
        .equal(torch.tensor(36.0))
    )


def test_batched_tensor_seq_nansum_along_batch() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]])
        )
        .nansum_along_batch()
        .equal(torch.tensor([10.0, 26.0]))
    )


def test_batched_tensor_seq_nansum_along_batch_keepdim_true() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]])
        )
        .nansum_along_batch(keepdim=True)
        .equal(torch.tensor([[10.0, 26.0]]))
    )


def test_batched_tensor_seq_nansum_along_batch_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]), batch_dim=1, seq_dim=0
        )
        .nansum_along_batch()
        .equal(torch.tensor([10.0, 26.0]))
    )


def test_batched_tensor_seq_nansum_along_seq() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nansum_along_seq()
        .equal(torch.tensor([10.0, 26.0]))
    )


def test_batched_tensor_seq_nansum_along_seq_keepdim_true() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nansum_along_seq(keepdim=True)
        .equal(torch.tensor([[10.0], [26.0]]))
    )


def test_batched_tensor_seq_nansum_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]]),
            batch_dim=1,
            seq_dim=0,
        )
        .nansum_along_seq()
        .equal(torch.tensor([10.0, 26.0]))
    )


def test_batched_tensor_seq_prod() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]]))
        .prod()
        .equal(torch.tensor(362880))
    )


def test_batched_tensor_seq_prod_keepdim_false() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]]))
        .prod(dim=1)
        .equal(torch.tensor([120, 3024]))
    )


def test_batched_tensor_seq_prod_keepdim_true() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]]))
        .prod(dim=1, keepdim=True)
        .equal(torch.tensor([[120], [3024]]))
    )


def test_batched_tensor_seq_prod_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]]), batch_dim=1, seq_dim=0)
        .prod()
        .equal(torch.tensor(362880))
    )


def test_batched_tensor_seq_prod_along_batch_keepdim_false() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1, 6], [2, 7], [3, 8], [4, 9], [5, 1]]))
        .prod_along_batch()
        .equal(torch.tensor([120, 3024]))
    )


def test_batched_tensor_seq_prod_along_batch_keepdim_true() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1, 6], [2, 7], [3, 8], [4, 9], [5, 1]]))
        .prod_along_batch(keepdim=True)
        .equal(torch.tensor([[120, 3024]]))
    )


def test_batched_tensor_seq_prod_along_batch_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]]), batch_dim=1, seq_dim=0)
        .prod_along_batch()
        .equal(torch.tensor([120, 3024]))
    )


def test_batched_tensor_seq_prod_along_seq_keepdim_false() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]]))
        .prod_along_seq()
        .equal(torch.tensor([120, 3024]))
    )


def test_batched_tensor_seq_prod_along_seq_keepdim_true() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]]))
        .prod_along_seq(keepdim=True)
        .equal(torch.tensor([[120], [3024]]))
    )


def test_batched_tensor_seq_prod_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq(
            torch.tensor([[1, 6], [2, 7], [3, 8], [4, 9], [5, 1]]), batch_dim=1, seq_dim=0
        )
        .prod_along_seq()
        .equal(torch.tensor([120, 3024]))
    )


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_sum(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5).to(dtype=dtype))
        .sum()
        .equal(torch.tensor(45, dtype=dtype))
    )


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_sum_keepdim_false(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5).to(dtype=dtype))
        .sum(dim=1)
        .equal(torch.tensor([10, 35], dtype=dtype))
    )


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_sum_keepdim_true(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5).to(dtype=dtype))
        .sum(dim=1, keepdim=True)
        .equal(torch.tensor([[10], [35]], dtype=dtype))
    )


def test_batched_tensor_seq_sum_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .sum()
        .equal(torch.tensor(45))
    )


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_sum_along_batch(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2).to(dtype=dtype))
        .sum_along_batch()
        .equal(torch.tensor([20, 25], dtype=dtype))
    )


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_sum_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2).to(dtype=dtype))
        .sum_along_batch(keepdim=True)
        .equal(torch.tensor([[20, 25]], dtype=dtype))
    )


def test_batched_tensor_seq_sum_along_batch_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 4], [1, 2], [2, 5]]), batch_dim=1, seq_dim=0)
        .sum_along_batch()
        .equal(torch.tensor([4, 3, 7]))
    )


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_sum_along_seq(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5).to(dtype=dtype))
        .sum_along_seq()
        .equal(torch.tensor([10, 35], dtype=dtype))
    )


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_seq_sum_along_seq_keepdim_true(dtype: torch.dtype) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5).to(dtype=dtype))
        .sum_along_seq(keepdim=True)
        .equal(torch.tensor([[10], [35]], dtype=dtype))
    )


def test_batched_tensor_seq_sum_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 4], [1, 2], [2, 5]]), batch_dim=1, seq_dim=0)
        .sum_along_seq()
        .equal(torch.tensor([3, 11]))
    )


##########################################################
#    Indexing, slicing, joining, mutating operations     #
##########################################################


def test_batched_tensor_seq__getitem___none() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    assert batch[None].equal(torch.tensor([[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]]))


def test_batched_tensor_seq__getitem___int() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    assert batch[0].equal(torch.tensor([0, 1, 2, 3, 4]))


def test_batched_tensor_seq__range___slice() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    assert batch[0:2, 2:4].equal(torch.tensor([[2, 3], [7, 8]]))


@mark.parametrize(
    "index",
    (
        [2, 0],
        torch.tensor([2, 0]),
        BatchedTensor(torch.tensor([2, 0])),
    ),
)
def test_batched_tensor_seq__getitem___list_like(index: IndexType) -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(5, 2))
    assert batch[index].equal(torch.tensor([[4, 5], [0, 1]]))


def test_batched_tensor_seq__setitem___int() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch[0] = 7
    assert batch.equal(BatchedTensorSeq(torch.tensor([[7, 7, 7, 7, 7], [5, 6, 7, 8, 9]])))


def test_batched_tensor_seq__setitem___slice() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch[0:1, 2:4] = 7
    assert batch.equal(BatchedTensorSeq(torch.tensor([[0, 1, 7, 7, 4], [5, 6, 7, 8, 9]])))


@mark.parametrize(
    "index",
    (
        [0, 2],
        torch.tensor([0, 2]),
        BatchedTensor(torch.tensor([0, 2])),
    ),
)
def test_batched_tensor_seq__setitem___list_like_index(index: IndexType) -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(5, 2))
    batch[index] = 7
    assert batch.equal(BatchedTensorSeq(torch.tensor([[7, 7], [2, 3], [7, 7], [6, 7], [8, 9]])))


@mark.parametrize(
    "value",
    (
        torch.tensor([[0, -4]]),
        BatchedTensor(torch.tensor([[0, -4]])),
        BatchedTensorSeq(torch.tensor([[0, -4]])),
    ),
)
def test_batched_tensor_seq__setitem___tensor_value(value: Tensor | BatchedTensor) -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch[1:2, 2:4] = value
    assert batch.equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 0, -4, 9]])))


def test_batched_tensor_seq_align_as_no_permutation() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .align_as(BatchedTensorSeq(torch.ones(2, 3)))
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])))
    )


def test_batched_tensor_seq_align_as_seq_batch() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .align_as(BatchedTensorSeq.from_seq_batch(torch.ones(2, 3)))
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]), batch_dim=1, seq_dim=0
            )
        )
    )


def test_batched_tensor_seq_align_as_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5, 1))
        .align_as(BatchedTensorSeq(torch.ones(2, 3, 4), batch_dim=1, seq_dim=2))
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]]), batch_dim=1, seq_dim=2
            )
        )
    )


def test_batched_tensor_seq_align_as_incorrect_type() -> None:
    with raises(TypeError):
        BatchedTensorSeq(torch.arange(10).view(2, 5)).align_as(torch.ones(2, 5))


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
def test_batched_tensor_seq_append(other: BatchedTensor | Tensor) -> None:
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
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.append(BatchedTensorSeq(torch.zeros(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_append_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.append(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


@mark.parametrize(
    "tensors",
    (
        BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]])),
        BatchedTensor(torch.tensor([[10, 11], [12, 13]])),
        torch.tensor([[10, 11], [12, 13]]),
        [BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]]))],
        (BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]])),),
    ),
)
def test_batched_tensor_seq_cat_dim_1(
    tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
) -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
        .cat(tensors, dim=1)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 10, 11], [4, 5, 6, 12, 13]])))
    )


def test_batched_tensor_seq_cat_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1, seq_dim=0)
        .cat(
            BatchedTensorSeq(torch.tensor([[10, 12], [11, 13], [14, 15]]), batch_dim=1, seq_dim=0),
            dim=1,
        )
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_cat_empty() -> None:
    assert BatchedTensorSeq(torch.ones(2, 3)).cat([]).equal(BatchedTensorSeq(torch.ones(2, 3)))


def test_batched_tensor_seq_cat_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat([BatchedTensorSeq(torch.zeros(2, 2, 2), batch_dim=2)])


def test_batched_tensor_seq_cat_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.cat([BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2)])


@mark.parametrize(
    "tensors",
    (
        BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        torch.tensor([[10, 11, 12], [13, 14, 15]]),
        [BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]]))],
        (BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),),
    ),
)
def test_batched_tensor_seq_cat__dim_0(
    tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
) -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.cat_(tensors, dim=0)
    assert batch.equal(
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))
    )


@mark.parametrize(
    "tensors",
    (
        BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]])),
        BatchedTensor(torch.tensor([[10, 11], [12, 13]])),
        torch.tensor([[10, 11], [12, 13]]),
        [BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]]))],
        (BatchedTensorSeq(torch.tensor([[10, 11], [12, 13]])),),
    ),
)
def test_batched_tensor_seq_cat__dim_1(
    tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
) -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.cat_(tensors, dim=1)
    assert batch.equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 10, 11], [4, 5, 6, 12, 13]])))


def test_batched_tensor_seq_cat__custom_dims() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1, seq_dim=0)
    batch.cat_(
        BatchedTensorSeq(torch.tensor([[10, 12], [11, 13], [14, 15]]), batch_dim=1, seq_dim=0),
        dim=1,
    )
    assert batch.equal(
        BatchedTensorSeq(
            torch.tensor([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]), batch_dim=1, seq_dim=0
        )
    )


def test_batched_tensor_seq_cat__empty() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch.cat_([])
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3)))


def test_batched_tensor_cat__incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_([BatchedTensorSeq(torch.zeros(2, 2, 2), batch_dim=2)])


def test_batched_tensor_seq_cat__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.cat_([BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2)])


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
    other: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
) -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]]))
        .cat_along_batch(other)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]])))
    )


def test_batched_tensor_seq_seq_cat_along_batch_custom_dims() -> None:
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
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_along_batch([BatchedTensorSeq(torch.zeros(2, 2, 2), batch_dim=2)])


def test_batched_tensor_seq_cat_along_batch_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.cat_along_batch([BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2)])


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
    other: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
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
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_along_batch_([BatchedTensorSeq(torch.zeros(2, 2, 2), batch_dim=2)])


def test_batched_tensor_seq_cat_along_batch__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.cat_along_batch_([BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2)])


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
    other: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
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
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_along_seq([BatchedTensorSeq(torch.zeros(2, 2, 2), batch_dim=2)])


def test_batched_tensor_seq_cat_along_seq_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.cat_along_seq([BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2)])


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
    other: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
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
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_along_seq_([BatchedTensorSeq(torch.zeros(2, 2, 2), batch_dim=2)])


def test_batched_tensor_seq_cat_along_seq__incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.cat_along_seq_([BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2)])


def test_batched_tensor_seq_chunk_3() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(5, 2)).chunk(3),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_seq_chunk_5() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(5, 2)).chunk(5),
        (
            BatchedTensorSeq(torch.tensor([[0, 1]])),
            BatchedTensorSeq(torch.tensor([[2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5]])),
            BatchedTensorSeq(torch.tensor([[6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_seq_chunk_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0).chunk(3, dim=1),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [5, 6]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[2, 3], [7, 8]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[4], [9]]), batch_dim=1, seq_dim=0),
        ),
    )


def test_batched_tensor_seq_chunk_along_batch_5() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(5, 2)).chunk_along_batch(5),
        (
            BatchedTensorSeq(torch.tensor([[0, 1]])),
            BatchedTensorSeq(torch.tensor([[2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5]])),
            BatchedTensorSeq(torch.tensor([[6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_seq_chunk_along_batch_3() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(5, 2)).chunk_along_batch(3),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_seq_chunk_along_batch_incorrect_chunks() -> None:
    with raises(RuntimeError, match="chunk expects `chunks` to be greater than 0, got: 0"):
        BatchedTensorSeq(torch.arange(10).view(5, 2)).chunk_along_batch(0)


def test_batched_tensor_seq_chunk_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0).chunk_along_batch(3),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [5, 6]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[2, 3], [7, 8]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[4], [9]]), batch_dim=1, seq_dim=0),
        ),
    )


def test_batched_tensor_seq_chunk_along_seq_5() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).chunk_along_seq(5),
        (
            BatchedTensorSeq(torch.tensor([[0], [5]])),
            BatchedTensorSeq(torch.tensor([[1], [6]])),
            BatchedTensorSeq(torch.tensor([[2], [7]])),
            BatchedTensorSeq(torch.tensor([[3], [8]])),
            BatchedTensorSeq(torch.tensor([[4], [9]])),
        ),
    )


def test_batched_tensor_seq_chunk_along_seq_3() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).chunk_along_seq(3),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [5, 6]])),
            BatchedTensorSeq(torch.tensor([[2, 3], [7, 8]])),
            BatchedTensorSeq(torch.tensor([[4], [9]])),
        ),
    )


def test_batched_tensor_seq_chunk_along_seq_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0).chunk_along_seq(3),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[4, 5], [6, 7]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[8, 9]]), batch_dim=1, seq_dim=0),
        ),
    )


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
    other: Iterable[BatchedTensor | Tensor],
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
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.extend([BatchedTensorSeq(torch.zeros(2, 2, 2), batch_dim=2)])


def test_batched_tensor_seq_extend_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.extend([BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2)])


@mark.parametrize("index", (torch.tensor([2, 0]), [2, 0], (2, 0)))
def test_batched_tensor_seq_index_select_dim_0(index: Tensor | Sequence[int]) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .index_select(0, index)
        .equal(BatchedTensorSeq(torch.tensor([[4, 5], [0, 1]])))
    )


@mark.parametrize("index", (torch.tensor([2, 0]), [2, 0], (2, 0)))
def test_batched_tensor_seq_index_select_dim_1(index: Tensor | list[int] | tuple[int, ...]) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .index_select(1, index)
        .equal(BatchedTensorSeq(torch.tensor([[2, 0], [7, 5]])))
    )


def test_batched_tensor_seq_index_select_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0)
        .index_select(0, (2, 0))
        .equal(BatchedTensorSeq(torch.tensor([[4, 5], [0, 1]]), batch_dim=1, seq_dim=0))
    )


@mark.parametrize("index", (torch.tensor([2, 0]), [2, 0], (2, 0)))
def test_batched_tensor_seq_index_select_along_batch(index: Tensor | Sequence[int]) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .index_select_along_batch(index)
        .equal(BatchedTensorSeq(torch.tensor([[4, 5], [0, 1]])))
    )


def test_batched_tensor_seq_index_select_along_batch_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .index_select_along_batch((2, 0))
        .equal(BatchedTensorSeq(torch.tensor([[2, 0], [7, 5]]), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_index_select_along_batch_extra_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(20).view(2, 2, 5), batch_dim=2, seq_dim=1)
        .index_select_along_batch((2, 0))
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[2, 0], [7, 5]], [[12, 10], [17, 15]]]), batch_dim=2, seq_dim=1
            )
        )
    )


@mark.parametrize("index", (torch.tensor([2, 0]), [2, 0], (2, 0)))
def test_batched_tensor_seq_index_select_along_seq(
    index: Tensor | list[int] | tuple[int, ...]
) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .index_select_along_seq(index)
        .equal(BatchedTensorSeq(torch.tensor([[2, 0], [7, 5]])))
    )


def test_batched_tensor_seq_index_select_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.arange(10).view(5, 2))
        .index_select_along_seq((2, 0))
        .equal(BatchedTensorSeq(torch.tensor([[4, 5], [0, 1]]), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_index_select_along_seq_extra_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(20).view(2, 5, 2), batch_dim=2, seq_dim=1)
        .index_select_along_seq((2, 0))
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[4, 5], [0, 1]], [[14, 15], [10, 11]]]), batch_dim=2, seq_dim=1
            )
        )
    )


@mark.parametrize(
    "mask",
    (
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False, True], [False, False, False, False, False]])
        ),
        BatchedTensor(
            torch.tensor([[True, False, True, False, True], [False, False, False, False, False]])
        ),
        torch.tensor([[True, False, True, False, True], [False, False, False, False, False]]),
    ),
)
def test_batched_tensor_seq_masked_fill(mask: BatchedTensor | Tensor) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .masked_fill(mask, -1)
        .equal(BatchedTensorSeq(torch.tensor([[-1, 1, -1, 3, -1], [5, 6, 7, 8, 9]])))
    )


def test_batched_tensor_seq_masked_fill_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.arange(10).view(5, 2))
        .masked_fill(
            BatchedTensorSeq.from_seq_batch(
                torch.tensor(
                    [[True, False], [False, True], [True, False], [False, True], [True, False]]
                )
            ),
            -1,
        )
        .equal(
            BatchedTensorSeq(
                torch.tensor([[-1, 1], [2, -1], [-1, 5], [6, -1], [-1, 9]]), batch_dim=1, seq_dim=0
            )
        )
    )


def test_batched_tensor_seq_masked_fill_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.masked_fill(BatchedTensorSeq(torch.zeros(2, 2, 2), batch_dim=2), 0)


def test_batched_tensor_seq_masked_fill_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.masked_fill(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2), 0)


def test_batched_tensor_seq_repeat_along_seq_repeats_1() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .repeat_along_seq(repeats=1)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])))
    )


def test_batched_tensor_seq_repeat_along_seq_repeats_2() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .repeat_along_seq(repeats=2)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 5, 6, 7, 8, 9]])
            )
        )
    )


def test_batched_tensor_seq_repeat_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0)
        .repeat_along_seq(repeats=2)
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
                ),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_repeat_along_seq_extra_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(20).view(2, 5, 2))
        .repeat_along_seq(repeats=2)
        .equal(
            BatchedTensorSeq(
                torch.tensor(
                    [
                        [
                            [0, 1],
                            [2, 3],
                            [4, 5],
                            [6, 7],
                            [8, 9],
                            [0, 1],
                            [2, 3],
                            [4, 5],
                            [6, 7],
                            [8, 9],
                        ],
                        [
                            [10, 11],
                            [12, 13],
                            [14, 15],
                            [16, 17],
                            [18, 19],
                            [10, 11],
                            [12, 13],
                            [14, 15],
                            [16, 17],
                            [18, 19],
                        ],
                    ]
                )
            )
        )
    )


def test_batched_tensor_seq_select_dim_0() -> None:
    assert (
        BatchedTensorSeq(torch.arange(30).view(5, 2, 3))
        .select(dim=0, index=2)
        .equal(torch.tensor([[12, 13, 14], [15, 16, 17]]))
    )


def test_batched_tensor_seq_select_dim_1() -> None:
    assert (
        BatchedTensorSeq(torch.arange(30).view(5, 2, 3))
        .select(dim=1, index=0)
        .equal(torch.tensor([[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20], [24, 25, 26]]))
    )


def test_batched_tensor_seq_select_dim_2() -> None:
    assert (
        BatchedTensorSeq(torch.arange(30).view(5, 2, 3))
        .select(dim=2, index=1)
        .equal(torch.tensor([[1, 4], [7, 10], [13, 16], [19, 22], [25, 28]]))
    )


def test_batched_tensor_seq_select_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(30).view(5, 2, 3), batch_dim=1, seq_dim=0)
        .select(dim=0, index=2)
        .equal(torch.tensor([[12, 13, 14], [15, 16, 17]]))
    )


def test_batched_tensor_seq_select_along_batch() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 9], [1, 8], [2, 7], [3, 6], [4, 5]]))
        .select_along_batch(2)
        .equal(torch.tensor([2, 7]))
    )


def test_batched_tensor_seq_select_along_batch_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.arange(10).view(2, 5))
        .select_along_batch(2)
        .equal(torch.tensor([2, 7]))
    )


def test_batched_tensor_seq_select_along_seq() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]))
        .select_along_seq(2)
        .equal(BatchedTensor(torch.tensor([2, 7])))
    )


def test_batched_tensor_seq_select_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0)
        .select_along_seq(2)
        .equal(BatchedTensor(torch.tensor([4, 5])))
    )


def test_batched_tensor_seq_select_along_seq_extra_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(20).view(2, 5, 2), batch_dim=2, seq_dim=1)
        .select_along_seq(2)
        .equal(BatchedTensor(torch.tensor([[4, 5], [14, 15]]), batch_dim=1))
    )


def test_batched_tensor_seq_slice_along_batch() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .slice_along_batch()
        .equal(BatchedTensorSeq(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])))
    )


def test_batched_tensor_seq_slice_along_batch_start_2() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .slice_along_batch(start=2)
        .equal(BatchedTensorSeq(torch.tensor([[4, 5], [6, 7], [8, 9]])))
    )


def test_batched_tensor_seq_slice_along_batch_stop_3() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .slice_along_batch(stop=3)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1], [2, 3], [4, 5]])))
    )


def test_batched_tensor_seq_slice_along_batch_stop_100() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .slice_along_batch(stop=100)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])))
    )


def test_batched_tensor_seq_slice_along_batch_step_2() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .slice_along_batch(step=2)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1], [4, 5], [8, 9]])))
    )


def test_batched_tensor_seq_slice_along_batch_start_1_stop_4_step_2() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .slice_along_batch(start=1, stop=4, step=2)
        .equal(BatchedTensorSeq(torch.tensor([[2, 3], [6, 7]])))
    )


def test_batched_tensor_seq_slice_along_batch_custom_dim() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.arange(10).view(2, 5))
        .slice_along_batch(start=2)
        .equal(BatchedTensorSeq(torch.tensor([[2, 3, 4], [7, 8, 9]]), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_slice_along_batch_batch_dim_1() -> None:
    assert (
        BatchedTensorSeq(torch.arange(20).view(2, 5, 2), batch_dim=1, seq_dim=0)
        .slice_along_batch(start=2)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[4, 5], [6, 7], [8, 9]], [[14, 15], [16, 17], [18, 19]]]),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_slice_along_batch_batch_dim_2() -> None:
    assert (
        BatchedTensorSeq(torch.arange(20).view(2, 2, 5), batch_dim=2)
        .slice_along_batch(start=2)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[2, 3, 4], [7, 8, 9]], [[12, 13, 14], [17, 18, 19]]]), batch_dim=2
            )
        )
    )


def test_batched_tensor_seq_slice_along_dim() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .slice_along_dim()
        .equal(BatchedTensorSeq(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])))
    )


def test_batched_tensor_seq_slice_along_dim_start_2() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .slice_along_dim(start=2)
        .equal(BatchedTensorSeq(torch.tensor([[4, 5], [6, 7], [8, 9]])))
    )


def test_batched_tensor_seq_slice_along_dim_stop_3() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .slice_along_dim(stop=3)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1], [2, 3], [4, 5]])))
    )


def test_batched_tensor_seq_slice_along_dim_stop_100() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .slice_along_dim(stop=100)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])))
    )


def test_batched_tensor_seq_slice_along_dim_step_2() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .slice_along_dim(step=2)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1], [4, 5], [8, 9]])))
    )


def test_batched_tensor_seq_slice_along_dim_start_1_stop_4_step_2() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .slice_along_dim(start=1, stop=4, step=2)
        .equal(BatchedTensorSeq(torch.tensor([[2, 3], [6, 7]])))
    )


def test_batched_tensor_seq_slice_along_dim_batch_dim_1() -> None:
    assert (
        BatchedTensorSeq(torch.arange(20).view(2, 5, 2), batch_dim=1, seq_dim=0)
        .slice_along_dim(start=2, dim=1)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[4, 5], [6, 7], [8, 9]], [[14, 15], [16, 17], [18, 19]]]),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_slice_along_dim_batch_dim_2() -> None:
    assert (
        BatchedTensorSeq(torch.arange(20).view(2, 2, 5), batch_dim=2)
        .slice_along_dim(start=2, dim=2)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[2, 3, 4], [7, 8, 9]], [[12, 13, 14], [17, 18, 19]]]), batch_dim=2
            )
        )
    )


def test_batched_tensor_seq_slice_along_seq() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]))
        .slice_along_seq()
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]])))
    )


def test_batched_tensor_seq_slice_along_seq_start_2() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]))
        .slice_along_seq(start=2)
        .equal(BatchedTensorSeq(torch.tensor([[2, 3, 4], [7, 6, 5]])))
    )


def test_batched_tensor_seq_slice_along_seq_stop_3() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]))
        .slice_along_seq(stop=3)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 2], [9, 8, 7]])))
    )


def test_example_batch_slice_along_seq_stop_100() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]))
        .slice_along_seq(stop=100)
        .equal(BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]])))
    )


def test_batched_tensor_seq_slice_along_seq_step_2() -> None:
    assert (
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]))
        .slice_along_seq(step=2)
        .equal(BatchedTensorSeq(torch.tensor([[0, 2, 4], [9, 7, 5]])))
    )


def test_batched_tensor_seq_slice_along_seq_seq_dim_0() -> None:
    assert (
        BatchedTensorSeq(torch.arange(20).view(5, 2, 2), batch_dim=1, seq_dim=0)
        .slice_along_seq(start=2)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[8, 9], [10, 11]], [[12, 13], [14, 15]], [[16, 17], [18, 19]]]),
                batch_dim=1,
                seq_dim=0,
            )
        )
    )


def test_batched_tensor_seq_slice_along_seq_seq_dim_2() -> None:
    assert (
        BatchedTensorSeq(torch.arange(20).view(2, 2, 5), seq_dim=2)
        .slice_along_seq(start=2)
        .equal(
            BatchedTensorSeq(
                torch.tensor([[[2, 3, 4], [7, 8, 9]], [[12, 13, 14], [17, 18, 19]]]), seq_dim=2
            )
        )
    )


def test_batched_tensor_seq_split_split_size_1() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(5, 2)).split(1),
        (
            BatchedTensorSeq(torch.tensor([[0, 1]])),
            BatchedTensorSeq(torch.tensor([[2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5]])),
            BatchedTensorSeq(torch.tensor([[6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_seq_split_split_size_2() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(5, 2)).split(2),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_seq_split_split_size_list() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(5, 2)).split([2, 2, 1]),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_seq_split_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0).split(2, dim=1),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [5, 6]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[2, 3], [7, 8]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[4], [9]]), batch_dim=1, seq_dim=0),
        ),
    )


def test_batched_tensor_seq_split_along_batch_split_size_1() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(5, 2)).split_along_batch(1),
        (
            BatchedTensorSeq(torch.tensor([[0, 1]])),
            BatchedTensorSeq(torch.tensor([[2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5]])),
            BatchedTensorSeq(torch.tensor([[6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_seq_split_along_batch_split_size_2() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(5, 2)).split_along_batch(2),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_seq_split_along_batch_split_size_list() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(5, 2)).split_along_batch([2, 2, 1]),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_seq_split_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0).split_along_batch(2),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [5, 6]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[2, 3], [7, 8]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[4], [9]]), batch_dim=1, seq_dim=0),
        ),
    )


def test_batched_tensor_seq_split_along_seq_split_size_1() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).split_along_seq(1),
        (
            BatchedTensorSeq(torch.tensor([[0], [5]])),
            BatchedTensorSeq(torch.tensor([[1], [6]])),
            BatchedTensorSeq(torch.tensor([[2], [7]])),
            BatchedTensorSeq(torch.tensor([[3], [8]])),
            BatchedTensorSeq(torch.tensor([[4], [9]])),
        ),
    )


def test_batched_tensor_seq_split_along_seq_split_size_2() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).split_along_seq(2),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [5, 6]])),
            BatchedTensorSeq(torch.tensor([[2, 3], [7, 8]])),
            BatchedTensorSeq(torch.tensor([[4], [9]])),
        ),
    )


def test_batched_tensor_seq_split_along_seq_split_size_list() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(2, 5)).split_along_seq([2, 2, 1]),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [5, 6]])),
            BatchedTensorSeq(torch.tensor([[2, 3], [7, 8]])),
            BatchedTensorSeq(torch.tensor([[4], [9]])),
        ),
    )


def test_batched_tensor_seq_split_along_seq_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensorSeq(torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0).split_along_seq(2),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[4, 5], [6, 7]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[8, 9]]), batch_dim=1, seq_dim=0),
        ),
    )


@mark.parametrize(
    "indices",
    (
        BatchedTensorSeq(torch.tensor([[3, 2], [0, 3], [1, 4]])),
        BatchedTensor(torch.tensor([[3, 2], [0, 3], [1, 4]])),
        torch.tensor([[3, 2], [0, 3], [1, 4]]),
        torch.tensor([[3, 2], [0, 3], [1, 4]], dtype=torch.float),
        [[3, 2], [0, 3], [1, 4]],
        BatchList([[3, 2], [0, 3], [1, 4]]),
    ),
)
def test_batched_tensor_seq_take_along_batch(indices: BaseBatch | Tensor | Sequence) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .take_along_batch(indices)
        .equal(BatchedTensorSeq(torch.tensor([[6, 5], [0, 7], [2, 9]])))
    )


def test_batched_tensor_seq_take_along_batch_empty_indices() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .take_along_batch(BatchedTensorSeq(torch.ones(0, 2, dtype=torch.long)))
        .equal(BatchedTensorSeq(torch.ones(0, 2, dtype=torch.long)))
    )


def test_batched_tensor_seq_take_along_batch_custom_dim() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .take_along_batch(
            BatchedTensorSeq(torch.tensor([[3, 0, 1], [2, 3, 4]]), batch_dim=1, seq_dim=0)
        )
        .equal(BatchedTensorSeq(torch.tensor([[3, 0, 1], [7, 8, 9]]), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_take_along_batch_extra_dim_first() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(1, 5, 2), batch_dim=1, seq_dim=2)
        .take_along_batch(
            BatchedTensorSeq(torch.tensor([[[3, 2], [0, 3], [1, 4]]]), batch_dim=1, seq_dim=2)
        )
        .equal(BatchedTensorSeq(torch.tensor([[[6, 5], [0, 7], [2, 9]]]), batch_dim=1, seq_dim=2))
    )


def test_batched_tensor_seq_take_along_batch_extra_dim_end() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2, 1))
        .take_along_batch(BatchedTensorSeq(torch.tensor([[[3], [2]], [[0], [3]], [[1], [4]]])))
        .equal(BatchedTensorSeq(torch.tensor([[[6], [5]], [[0], [7]], [[2], [9]]])))
    )


def test_batched_tensor_seq_take_along_batch_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.take_along_batch(BatchedTensorSeq(torch.zeros(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_take_along_batch_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.take_along_batch(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


@mark.parametrize(
    "indices",
    (
        BatchedTensorSeq(torch.tensor([[3, 2], [0, 3], [1, 4]])),
        BatchedTensor(torch.tensor([[3, 2], [0, 3], [1, 4]])),
        torch.tensor([[3, 2], [0, 3], [1, 4]]),
        torch.tensor([[3, 2], [0, 3], [1, 4]], dtype=torch.float),
        [[3, 2], [0, 3], [1, 4]],
        BatchList([[3, 2], [0, 3], [1, 4]]),
    ),
)
def test_batched_tensor_seq_take_along_dim(indices: BaseBatch | Tensor | Sequence) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .take_along_dim(indices)
        .equal(torch.tensor([3, 2, 0, 3, 1, 4]))
    )


@mark.parametrize(
    "indices",
    (
        BatchedTensorSeq(torch.tensor([[3, 2], [0, 3], [1, 4]])),
        BatchedTensor(torch.tensor([[3, 2], [0, 3], [1, 4]])),
        torch.tensor([[3, 2], [0, 3], [1, 4]]),
        torch.tensor([[3, 2], [0, 3], [1, 4]], dtype=torch.float),
        [[3, 2], [0, 3], [1, 4]],
        BatchList([[3, 2], [0, 3], [1, 4]]),
    ),
)
def test_batched_tensor_seq_take_along_dim_0(indices: BaseBatch | Tensor | Sequence) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .take_along_dim(indices, dim=0)
        .equal(BatchedTensorSeq(torch.tensor([[6, 5], [0, 7], [2, 9]])))
    )


def test_batched_tensor_seq_take_along_dim_empty_indices() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2))
        .take_along_dim(BatchedTensorSeq(torch.ones(0, 2, dtype=torch.long)), dim=0)
        .equal(BatchedTensorSeq(torch.ones(0, 2, dtype=torch.long)))
    )


def test_batched_tensor_seq_take_along_dim_custom_dim() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5), batch_dim=1, seq_dim=0)
        .take_along_dim(
            BatchedTensorSeq(torch.tensor([[3, 0, 1], [2, 3, 4]]), batch_dim=1, seq_dim=0), dim=1
        )
        .equal(BatchedTensorSeq(torch.tensor([[3, 0, 1], [7, 8, 9]]), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_take_along_dim_extra_dim_first() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(1, 5, 2), batch_dim=1, seq_dim=0)
        .take_along_dim(
            BatchedTensorSeq(torch.tensor([[[3, 2], [0, 3], [1, 4]]]), batch_dim=1, seq_dim=0),
            dim=1,
        )
        .equal(BatchedTensorSeq(torch.tensor([[[6, 5], [0, 7], [2, 9]]]), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_take_along_dim_extra_dim_end() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2, 1))
        .take_along_dim(BatchedTensorSeq(torch.tensor([[[3], [2]], [[0], [3]], [[1], [4]]])), dim=0)
        .equal(BatchedTensorSeq(torch.tensor([[[6], [5]], [[0], [7]], [[2], [9]]])))
    )


def test_batched_tensor_seq_take_along_dim_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.take_along_dim(BatchedTensorSeq(torch.zeros(2, 2, 2), batch_dim=2), dim=0)


def test_batched_tensor_seq_take_along_dim_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.take_along_batch(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


@mark.parametrize(
    "indices",
    (
        BatchedTensorSeq(torch.tensor([[3, 0, 1], [2, 3, 4]])),
        BatchedTensor(torch.tensor([[3, 0, 1], [2, 3, 4]])),
        torch.tensor([[3, 0, 1], [2, 3, 4]]),
        torch.tensor([[3, 0, 1], [2, 3, 4]], dtype=torch.float),
        BatchList([[3, 0, 1], [2, 3, 4]]),
        [[3, 0, 1], [2, 3, 4]],
        np.array([[3, 0, 1], [2, 3, 4]]),
    ),
)
def test_batched_tensor_seq_take_along_seq(indices: BaseBatch | Tensor | Sequence) -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .take_along_seq(indices)
        .equal(BatchedTensorSeq(torch.tensor([[3, 0, 1], [7, 8, 9]])))
    )


def test_batched_tensor_seq_take_along_seq_empty_indices() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .take_along_seq(BatchedTensorSeq(torch.tensor([[], []], dtype=torch.long)))
        .equal(BatchedTensorSeq(torch.tensor([[], []], dtype=torch.long)))
    )


def test_batched_tensor_seq_take_along_seq_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0)
        .take_along_seq(
            BatchedTensorSeq(torch.tensor([[3, 2], [0, 3], [1, 4]]), batch_dim=1, seq_dim=0)
        )
        .equal(BatchedTensorSeq(torch.tensor([[6, 5], [0, 7], [2, 9]]), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_take_along_seq_extra_dim_end() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5, 1))
        .take_along_seq(BatchedTensorSeq(torch.tensor([[[3], [0], [1]], [[2], [3], [4]]])))
        .equal(BatchedTensorSeq(torch.tensor([[[3], [0], [1]], [[7], [8], [9]]])))
    )


def test_batched_tensor_seq_take_along_seq_extra_dim_first() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(1, 2, 5), batch_dim=1, seq_dim=2)
        .take_along_seq(
            BatchedTensorSeq(torch.tensor([[[3, 0, 1], [2, 3, 4]]]), batch_dim=1, seq_dim=2)
        )
        .equal(BatchedTensorSeq(torch.tensor([[[3, 0, 1], [7, 8, 9]]]), batch_dim=1, seq_dim=2))
    )


def test_batched_tensor_seq_take_along_seq_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.take_along_seq(BatchedTensorSeq(torch.zeros(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_take_along_seq_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.take_along_seq(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


def test_batched_tensor_seq_unsqueeze_dim_0() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .unsqueeze(dim=0)
        .equal(BatchedTensorSeq(torch.ones(1, 2, 3), batch_dim=1, seq_dim=2))
    )


def test_batched_tensor_seq_unsqueeze_dim_1() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .unsqueeze(dim=1)
        .equal(BatchedTensorSeq(torch.ones(2, 1, 3), batch_dim=0, seq_dim=2))
    )


def test_batched_tensor_seq_unsqueeze_dim_2() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .unsqueeze(dim=2)
        .equal(BatchedTensorSeq(torch.ones(2, 3, 1)))
    )


def test_batched_tensor_seq_unsqueeze_dim_last() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .unsqueeze(dim=-1)
        .equal(BatchedTensorSeq(torch.ones(2, 3, 1)))
    )


def test_batched_tensor_seq_unsqueeze_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .unsqueeze(dim=0)
        .equal(BatchedTensorSeq(torch.ones(1, 2, 3), batch_dim=2, seq_dim=1))
    )


def test_batched_tensor_seq_view() -> None:
    assert BatchedTensorSeq(torch.ones(2, 6)).view(2, 3, 2).equal(torch.ones(2, 3, 2))


def test_batched_tensor_seq_view_first() -> None:
    assert BatchedTensorSeq(torch.ones(2, 6)).view(1, 2, 6).equal(torch.ones(1, 2, 6))


def test_batched_tensor_seq_view_last() -> None:
    assert BatchedTensorSeq(torch.ones(2, 6)).view(2, 6, 1).equal(torch.ones(2, 6, 1))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.zeros(2, 3, 1)),
        BatchedTensor(torch.zeros(2, 3, 1)),
        torch.zeros(2, 3, 1),
    ),
)
def test_batched_tensor_seq_view_as(other: BatchedTensor | Tensor) -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3))
        .view_as(other)
        .equal(BatchedTensorSeq(torch.ones(2, 3, 1)))
    )


def test_batched_tensor_seq_view_as_custom_dims() -> None:
    assert (
        BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0)
        .view_as(BatchedTensorSeq(torch.zeros(2, 3, 1), batch_dim=1, seq_dim=0))
        .equal(BatchedTensorSeq(torch.ones(2, 3, 1), batch_dim=1, seq_dim=0))
    )


def test_batched_tensor_seq_view_as_incorrect_batch_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.view_as(BatchedTensorSeq(torch.zeros(2, 2, 2), batch_dim=2))


def test_batched_tensor_seq_view_as_incorrect_seq_dim() -> None:
    batch = BatchedTensorSeq(torch.ones(2, 2, 2))
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        batch.view_as(BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2))


########################
#     mini-batches     #
########################


@mark.parametrize("batch_size,num_minibatches", ((1, 10), (2, 5), (3, 4), (4, 3)))
def test_batched_tensor_seq_get_num_minibatches_drop_last_false(
    batch_size: int, num_minibatches: int
) -> None:
    assert BatchedTensorSeq(torch.ones(10, 2)).get_num_minibatches(batch_size) == num_minibatches


@mark.parametrize("batch_size,num_minibatches", ((1, 10), (2, 5), (3, 3), (4, 2)))
def test_batched_tensor_seq_get_num_minibatches_drop_last_true(
    batch_size: int, num_minibatches: int
) -> None:
    assert (
        BatchedTensorSeq(torch.ones(10, 2)).get_num_minibatches(batch_size, drop_last=True)
        == num_minibatches
    )


def test_batched_tensor_seq_to_minibatches_10_batch_size_2() -> None:
    assert objects_are_equal(
        list(BatchedTensorSeq(torch.arange(20).view(10, 2)).to_minibatches(batch_size=2)),
        [
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9], [10, 11]])),
            BatchedTensorSeq(torch.tensor([[12, 13], [14, 15]])),
            BatchedTensorSeq(torch.tensor([[16, 17], [18, 19]])),
        ],
    )


def test_batched_tensor_seq_to_minibatches_10_batch_size_3() -> None:
    assert objects_are_equal(
        list(BatchedTensorSeq(torch.arange(20).view(10, 2)).to_minibatches(batch_size=3)),
        [
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3], [4, 5]])),
            BatchedTensorSeq(torch.tensor([[6, 7], [8, 9], [10, 11]])),
            BatchedTensorSeq(torch.tensor([[12, 13], [14, 15], [16, 17]])),
            BatchedTensorSeq(torch.tensor([[18, 19]])),
        ],
    )


def test_batched_tensor_seq_to_minibatches_10_batch_size_4() -> None:
    assert objects_are_equal(
        list(BatchedTensorSeq(torch.arange(20).view(10, 2)).to_minibatches(batch_size=4)),
        [
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9], [10, 11], [12, 13], [14, 15]])),
            BatchedTensorSeq(torch.tensor([[16, 17], [18, 19]])),
        ],
    )


def test_batched_tensor_seq_to_minibatches_drop_last_true_10_batch_size_2() -> None:
    assert objects_are_equal(
        list(
            BatchedTensorSeq(torch.arange(20).view(10, 2)).to_minibatches(
                batch_size=2, drop_last=True
            )
        ),
        [
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9], [10, 11]])),
            BatchedTensorSeq(torch.tensor([[12, 13], [14, 15]])),
            BatchedTensorSeq(torch.tensor([[16, 17], [18, 19]])),
        ],
    )


def test_batched_tensor_seq_to_minibatches_drop_last_true_10_batch_size_3() -> None:
    assert objects_are_equal(
        list(
            BatchedTensorSeq(torch.arange(20).view(10, 2)).to_minibatches(
                batch_size=3, drop_last=True
            )
        ),
        [
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3], [4, 5]])),
            BatchedTensorSeq(torch.tensor([[6, 7], [8, 9], [10, 11]])),
            BatchedTensorSeq(torch.tensor([[12, 13], [14, 15], [16, 17]])),
        ],
    )


def test_batched_tensor_seq_to_minibatches_drop_last_true_10_batch_size_4() -> None:
    assert objects_are_equal(
        list(
            BatchedTensorSeq(torch.arange(20).view(10, 2)).to_minibatches(
                batch_size=4, drop_last=True
            )
        ),
        [
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9], [10, 11], [12, 13], [14, 15]])),
        ],
    )


def test_batched_tensor_seq_to_minibatches_custom_dims() -> None:
    assert objects_are_equal(
        list(
            BatchedTensorSeq(torch.arange(20).view(2, 10), batch_dim=1, seq_dim=0).to_minibatches(
                batch_size=3
            )
        ),
        [
            BatchedTensorSeq(torch.tensor([[0, 1, 2], [10, 11, 12]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[3, 4, 5], [13, 14, 15]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[6, 7, 8], [16, 17, 18]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[9], [19]]), batch_dim=1, seq_dim=0),
        ],
    )


def test_batched_tensor_seq_to_minibatches_deepcopy_true() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]))
    for item in batch.to_minibatches(batch_size=2, deepcopy=True):
        item.data[0, 0] = 42
    assert batch.equal(BatchedTensorSeq(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])))


def test_batched_tensor_seq_to_minibatches_deepcopy_false() -> None:
    batch = BatchedTensorSeq(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]))
    for item in batch.to_minibatches(batch_size=2):
        item.data[0, 0] = 42
    assert batch.equal(BatchedTensorSeq(torch.tensor([[42, 1], [2, 3], [42, 5], [6, 7], [42, 9]])))


#################
#     Other     #
#################


def test_batched_tensor_seq_apply() -> None:
    assert (
        BatchedTensorSeq(torch.arange(10).view(2, 5))
        .apply(lambda tensor: tensor + 2)
        .equal(BatchedTensorSeq(torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])))
    )


def test_batched_tensor_seq_apply_custom_dims() -> None:
    assert (
        BatchedTensorSeq.from_seq_batch(torch.arange(10).view(2, 5))
        .apply(lambda tensor: tensor + 2)
        .equal(BatchedTensorSeq.from_seq_batch(torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])))
    )


def test_batched_tensor_seq_apply_() -> None:
    batch = BatchedTensorSeq(torch.arange(10).view(2, 5))
    batch.apply_(lambda tensor: tensor + 2)
    assert batch.equal(BatchedTensorSeq(torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])))


def test_batched_tensor_seq_apply__custom_dims() -> None:
    batch = BatchedTensorSeq.from_seq_batch(torch.arange(10).view(2, 5))
    batch.apply_(lambda tensor: tensor + 2)
    assert batch.equal(
        BatchedTensorSeq.from_seq_batch(torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]]))
    )


def test_batched_tensor_seq_summary() -> None:
    assert BatchedTensorSeq(torch.arange(10).view(2, 5)).summary() == (
        "BatchedTensorSeq(dtype=torch.int64, shape=torch.Size([2, 5]), device=cpu, "
        "batch_dim=0, seq_dim=1)"
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


####################################
#     Tests for check_seq_dims     #
####################################


def test_check_seq_dims_correct() -> None:
    check_seq_dims({0})


def test_check_seq_dims_incorrect() -> None:
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        check_seq_dims({0, 1})


##################################
#     Tests for get_seq_dims     #
##################################


def test_get_seq_dims() -> None:
    assert get_seq_dims(
        (BatchedTensorSeq(torch.ones(2, 3)), BatchedTensorSeq(torch.ones(2, 3))),
        {"val": BatchedTensorSeq(torch.ones(2, 3))},
    ) == {1}


def test_get_seq_dims_2() -> None:
    assert get_seq_dims(
        (BatchedTensorSeq(torch.ones(2, 3)), BatchedTensorSeq.from_seq_batch(torch.ones(2, 3))),
        {"val": BatchedTensorSeq.from_seq_batch(torch.ones(2, 3))},
    ) == {0, 1}


def test_get_seq_dims_empty() -> None:
    assert get_seq_dims(tuple()) == set()


################################
#     Tests for torch.amax     #
################################


def test_torch_amax_dim_0() -> None:
    assert torch.amax(BatchedTensorSeq(torch.arange(10).view(5, 2)), dim=0).equal(
        torch.tensor([8, 9])
    )


def test_torch_amax_dim_1() -> None:
    assert objects_are_equal(
        torch.amax(BatchedTensorSeq(torch.arange(10).view(2, 5)), dim=1),
        torch.tensor([4, 9]),
    )


def test_torch_amax_dim_1_keepdim() -> None:
    assert objects_are_equal(
        torch.amax(BatchedTensorSeq(torch.arange(10).view(2, 5)), dim=1, keepdim=True),
        torch.tensor([[4], [9]]),
    )


def test_torch_amax_custom_dims() -> None:
    assert objects_are_equal(
        torch.amax(BatchedTensorSeq.from_seq_batch(torch.arange(10).view(2, 5)), dim=1),
        torch.tensor([4, 9]),
    )


################################
#     Tests for torch.amin     #
################################


def test_torch_amin_dim_0() -> None:
    assert torch.amin(BatchedTensorSeq(torch.arange(10).view(5, 2)), dim=0).equal(
        torch.tensor([0, 1])
    )


def test_torch_amin_dim_1() -> None:
    assert objects_are_equal(
        torch.amin(BatchedTensorSeq(torch.arange(10).view(2, 5)), dim=1),
        torch.tensor([0, 5]),
    )


def test_torch_amin_dim_1_keepdim() -> None:
    assert objects_are_equal(
        torch.amin(BatchedTensorSeq(torch.arange(10).view(2, 5)), dim=1, keepdim=True),
        torch.tensor([[0], [5]]),
    )


def test_torch_amin_custom_dims() -> None:
    assert objects_are_equal(
        torch.amin(BatchedTensorSeq.from_seq_batch(torch.arange(10).view(2, 5)), dim=1),
        torch.tensor([0, 5]),
    )


###############################
#     Tests for torch.cat     #
###############################


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        torch.tensor([[10, 11, 12], [13, 14, 15]]),
    ),
)
def test_torch_cat_dim_0(other: BatchedTensor | Tensor) -> None:
    assert torch.cat(
        tensors=[BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6]])), other],
        dim=0,
    ).equal(
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]])),
    )


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.tensor([[4, 5], [14, 15]])),
        BatchedTensor(torch.tensor([[4, 5], [14, 15]])),
        torch.tensor([[4, 5], [14, 15]]),
    ),
)
def test_torch_cat_dim_1(other: BatchedTensor | Tensor) -> None:
    assert torch.cat(
        tensors=[BatchedTensorSeq(torch.tensor([[0, 1, 2], [10, 11, 12]])), other],
        dim=1,
    ).equal(
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 4, 5], [10, 11, 12, 14, 15]])),
    )


def test_torch_cat_tensor() -> None:
    assert torch.cat(
        tensors=[
            torch.tensor([[0, 1, 2], [10, 11, 12]]),
            torch.tensor([[4, 5], [14, 15]]),
        ],
        dim=1,
    ).equal(
        torch.tensor(
            [[0, 1, 2, 4, 5], [10, 11, 12, 14, 15]],
        )
    )


def test_torch_cat_custom_dims() -> None:
    assert torch.cat(
        [
            BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.ones(2, 3), batch_dim=1, seq_dim=0),
        ]
    ).equal(BatchedTensorSeq(torch.ones(4, 3), batch_dim=1, seq_dim=0))


def test_torch_cat_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        torch.cat(
            [
                BatchedTensorSeq(torch.ones(2, 2, 2)),
                BatchedTensorSeq(torch.zeros(2, 2, 2), batch_dim=2),
            ]
        )


def test_torch_cat_incorrect_seq_dim() -> None:
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        torch.cat(
            [
                BatchedTensorSeq(torch.ones(2, 2, 2)),
                BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2),
            ]
        )


#################################
#     Tests for torch.chunk     #
#################################


def test_torch_chunk_3() -> None:
    assert objects_are_equal(
        torch.chunk(BatchedTensorSeq(torch.arange(10).view(5, 2)), chunks=3),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9]])),
        ),
    )


def test_torch_chunk_5() -> None:
    assert objects_are_equal(
        torch.chunk(BatchedTensorSeq(torch.arange(10).view(5, 2)), chunks=5),
        (
            BatchedTensorSeq(torch.tensor([[0, 1]])),
            BatchedTensorSeq(torch.tensor([[2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5]])),
            BatchedTensorSeq(torch.tensor([[6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9]])),
        ),
    )


def test_torch_chunk_custom_dims() -> None:
    assert objects_are_equal(
        torch.chunk(
            BatchedTensorSeq(torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0), chunks=3
        ),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[4, 5], [6, 7]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[8, 9]]), batch_dim=1, seq_dim=0),
        ),
    )


def test_torch_chunk_dim_1() -> None:
    assert objects_are_equal(
        torch.chunk(BatchedTensorSeq(torch.arange(10).view(2, 5)), chunks=3, dim=1),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [5, 6]])),
            BatchedTensorSeq(torch.tensor([[2, 3], [7, 8]])),
            BatchedTensorSeq(torch.tensor([[4], [9]])),
        ),
    )


###############################
#     Tests for torch.max     #
###############################


def test_torch_max() -> None:
    assert torch.max(BatchedTensorSeq(torch.arange(10).view(2, 5))).equal(torch.tensor(9))


def test_torch_max_dim_1() -> None:
    assert objects_are_equal(
        tuple(torch.max(BatchedTensorSeq(torch.arange(10).view(2, 5)), dim=1)),
        (torch.tensor([4, 9]), torch.tensor([4, 4])),
    )


def test_torch_max_dim_1_keepdim() -> None:
    assert objects_are_equal(
        tuple(torch.max(BatchedTensorSeq(torch.arange(10).view(2, 5)), dim=1, keepdim=True)),
        (torch.tensor([[4], [9]]), torch.tensor([[4], [4]])),
    )


###################################
#     Tests for torch.maximum     #
###################################


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        torch.tensor([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_torch_maximum_other(other: BatchedTensor | Tensor) -> None:
    assert torch.maximum(BatchedTensorSeq(torch.tensor([[0, 1, 2], [-2, -1, 0]])), other).equal(
        BatchedTensorSeq(torch.tensor([[2, 1, 2], [0, 1, 0]]))
    )


def test_torch_maximum_custom_dims() -> None:
    assert torch.maximum(
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [-2, -1, 0]]), batch_dim=1, seq_dim=0),
        BatchedTensorSeq(torch.tensor([[2, 0, 1], [0, 1, 0]]), batch_dim=1, seq_dim=0),
    ).equal(BatchedTensorSeq(torch.tensor([[2, 1, 2], [0, 1, 0]]), batch_dim=1, seq_dim=0))


def test_torch_maximum_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        torch.maximum(
            BatchedTensorSeq(torch.ones(2, 2, 2)),
            BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2),
        )


def test_torch_maximum_incorrect_seq_dim() -> None:
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        torch.maximum(
            BatchedTensorSeq(torch.ones(2, 2, 2)), BatchedTensorSeq(torch.zeros(2, 1, 3), seq_dim=2)
        )


################################
#     Tests for torch.mean     #
################################


def test_torch_mean() -> None:
    assert torch.mean(BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5))).equal(
        torch.tensor(4.5)
    )


def test_torch_mean_dim_1() -> None:
    assert torch.mean(
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5)), dim=1
    ).equal(torch.tensor([2.0, 7.0]))


def test_torch_mean_keepdim() -> None:
    assert torch.mean(
        BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5)), dim=1, keepdim=True
    ).equal(torch.tensor([[2.0], [7.0]]))


##################################
#     Tests for torch.median     #
##################################


def test_torch_median() -> None:
    assert torch.median(BatchedTensorSeq(torch.arange(10).view(2, 5))).equal(torch.tensor(4))


def test_torch_median_dim_1() -> None:
    assert objects_are_equal(
        torch.median(BatchedTensorSeq(torch.arange(10).view(2, 5)), dim=1),
        torch.return_types.median([torch.tensor([2, 7]), torch.tensor([2, 2])]),
    )


def test_torch_median_keepdim() -> None:
    assert objects_are_equal(
        torch.median(BatchedTensorSeq(torch.arange(10).view(2, 5)), dim=1, keepdim=True),
        torch.return_types.median([torch.tensor([[2], [7]]), torch.tensor([[2], [2]])]),
    )


###############################
#     Tests for torch.min     #
###############################


def test_torch_min() -> None:
    assert torch.min(BatchedTensorSeq(torch.arange(10).view(2, 5))).equal(torch.tensor(0))


def test_torch_min_dim_1() -> None:
    assert objects_are_equal(
        tuple(torch.min(BatchedTensorSeq(torch.arange(10).view(2, 5)), dim=1)),
        (torch.tensor([0, 5]), torch.tensor([0, 0])),
    )


def test_torch_min_dim_1_keepdim() -> None:
    assert objects_are_equal(
        tuple(torch.min(BatchedTensorSeq(torch.arange(10).view(2, 5)), dim=1, keepdim=True)),
        (torch.tensor([[0], [5]]), torch.tensor([[0], [0]])),
    )


###################################
#     Tests for torch.minimum     #
###################################


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        torch.tensor([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_torch_minimum(other: BatchedTensor | Tensor) -> None:
    assert torch.minimum(BatchedTensorSeq(torch.tensor([[0, 1, 2], [-2, -1, 0]])), other).equal(
        BatchedTensorSeq(torch.tensor([[0, 0, 1], [-2, -1, 0]]))
    )


def test_torch_minimum_custom_dims() -> None:
    assert torch.minimum(
        BatchedTensorSeq(torch.tensor([[0, 1, 2], [-2, -1, 0]]), batch_dim=1, seq_dim=0),
        BatchedTensorSeq(torch.tensor([[2, 0, 1], [0, 1, 0]]), batch_dim=1, seq_dim=0),
    ).equal(BatchedTensorSeq(torch.tensor([[0, 0, 1], [-2, -1, 0]]), batch_dim=1, seq_dim=0))


def test_torch_minimum_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        torch.minimum(
            BatchedTensorSeq(torch.ones(2, 2, 2)),
            BatchedTensorSeq(torch.ones(2, 2, 2), batch_dim=2),
        )


def test_torch_minimum_incorrect_seq_dim() -> None:
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        torch.minimum(
            BatchedTensorSeq(torch.ones(2, 2, 2)), BatchedTensorSeq(torch.zeros(2, 1, 3), seq_dim=2)
        )


###################################
#     Tests for torch.nanmean     #
###################################


def test_torch_nanmean() -> None:
    assert torch.nanmean(
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
    ).equal(torch.tensor(4.0))


def test_torch_nanmean_dim_1() -> None:
    assert torch.nanmean(
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])), dim=1
    ).equal(torch.tensor([2.0, 6.5]))


def test_torch_nanmean_keepdim() -> None:
    assert torch.nanmean(
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])),
        dim=1,
        keepdim=True,
    ).equal(torch.tensor([[2.0], [6.5]]))


#####################################
#     Tests for torch.nanmedian     #
#####################################


def test_torch_nanmedian() -> None:
    assert torch.nanmedian(
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
    ).equal(torch.tensor(4.0))


def test_torch_nanmedian_dim_1() -> None:
    assert objects_are_equal(
        torch.nanmedian(
            BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])), dim=1
        ),
        torch.return_types.nanmedian([torch.tensor([2.0, 6.0]), torch.tensor([2, 1])]),
    )


def test_torch_nanmedian_keepdim() -> None:
    assert objects_are_equal(
        torch.nanmedian(
            BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])),
            dim=1,
            keepdim=True,
        ),
        torch.return_types.nanmedian([torch.tensor([[2.0], [6.0]]), torch.tensor([[2], [1]])]),
    )


##################################
#     Tests for torch.nansum     #
##################################


def test_torch_nansum() -> None:
    assert torch.nansum(
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
    ).equal(torch.tensor(36.0))


def test_torch_nansum_dim_1() -> None:
    assert torch.nansum(
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])), dim=1
    ).equal(torch.tensor([10.0, 26.0]))


def test_torch_nansum_keepdim() -> None:
    assert torch.nansum(
        BatchedTensorSeq(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])),
        dim=1,
        keepdim=True,
    ).equal(torch.tensor([[10.0], [26.0]]))


################################
#     Tests for torch.prod     #
################################


def test_torch_prod() -> None:
    assert torch.prod(BatchedTensorSeq(torch.arange(10, dtype=torch.float).view(2, 5))).equal(
        torch.tensor(0.0)
    )


def test_torch_prod_dim_1() -> None:
    assert torch.prod(BatchedTensorSeq(torch.arange(10).view(2, 5)), dim=1).equal(
        torch.tensor([0, 15120])
    )


def test_torch_prod_keepdim() -> None:
    assert torch.prod(BatchedTensorSeq(torch.arange(10).view(2, 5)), dim=1, keepdim=True).equal(
        torch.tensor([[0], [15120]])
    )


##################################
#     Tests for torch.select     #
##################################


def test_torch_select_dim_0() -> None:
    assert torch.select(BatchedTensorSeq(torch.arange(30).view(5, 2, 3)), dim=0, index=2).equal(
        torch.tensor([[12, 13, 14], [15, 16, 17]])
    )


def test_torch_select_dim_1() -> None:
    assert torch.select(BatchedTensorSeq(torch.arange(30).view(5, 2, 3)), dim=1, index=0).equal(
        torch.tensor([[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20], [24, 25, 26]])
    )


def test_torch_select_dim_2() -> None:
    assert torch.select(BatchedTensorSeq(torch.arange(30).view(5, 2, 3)), dim=2, index=1).equal(
        torch.tensor([[1, 4], [7, 10], [13, 16], [19, 22], [25, 28]])
    )


################################
#     Tests for torch.sort     #
################################


def test_torch_sort_descending_false() -> None:
    assert objects_are_equal(
        torch.sort(BatchedTensorSeq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]))),
        torch.return_types.sort(
            [
                BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])),
                BatchedTensorSeq(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]])),
            ]
        ),
    )


def test_torch_sort_descending_true() -> None:
    assert objects_are_equal(
        torch.sort(
            BatchedTensorSeq(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])), descending=True
        ),
        torch.return_types.sort(
            [
                BatchedTensorSeq(torch.tensor([[5, 4, 3, 2, 1], [9, 8, 7, 6, 5]])),
                BatchedTensorSeq(torch.tensor([[3, 0, 4, 2, 1], [0, 4, 1, 3, 2]])),
            ]
        ),
    )


def test_torch_sort_dim_0() -> None:
    assert objects_are_equal(
        torch.sort(
            BatchedTensorSeq(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])),
            dim=0,
        ),
        torch.return_types.sort(
            [
                BatchedTensorSeq(torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])),
                BatchedTensorSeq(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
            ]
        ),
    )


def test_torch_sort_dim_1() -> None:
    def test_torch_sort_dim_0() -> None:
        assert objects_are_equal(
            torch.sort(
                BatchedTensorSeq(
                    torch.tensor(
                        [
                            [[0, 1], [-2, 3], [-4, 5], [-6, 7], [-8, 9]],
                            [[10, -11], [12, -13], [14, -15], [16, -17], [18, -19]],
                        ]
                    )
                ),
                dim=1,
            ),
            torch.return_types.sort(
                [
                    BatchedTensorSeq(
                        torch.tensor(
                            [
                                [[-8, 1], [-6, 3], [-4, 5], [-2, 7], [0, 9]],
                                [[10, -19], [12, -17], [14, -15], [16, -13], [18, -11]],
                            ]
                        )
                    ),
                    BatchedTensorSeq(
                        torch.tensor(
                            [
                                [[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]],
                                [[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]],
                            ]
                        )
                    ),
                ]
            ),
        )


def test_torch_sort_custom_dims() -> None:
    values, indices = torch.sort(
        BatchedTensorSeq(
            torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), seq_dim=0, batch_dim=1
        ),
        dim=0,
    )
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


#################################
#     Tests for torch.split     #
#################################


def test_torch_split_size_1() -> None:
    assert objects_are_equal(
        torch.split(BatchedTensorSeq(torch.arange(10).view(5, 2)), 1),
        (
            BatchedTensorSeq(torch.tensor([[0, 1]])),
            BatchedTensorSeq(torch.tensor([[2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5]])),
            BatchedTensorSeq(torch.tensor([[6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9]])),
        ),
    )


def test_torch_split_size_2() -> None:
    assert objects_are_equal(
        torch.split(BatchedTensorSeq(torch.arange(10).view(5, 2)), 2),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9]])),
        ),
    )


def test_torch_split_size_list() -> None:
    assert objects_are_equal(
        torch.split(BatchedTensorSeq(torch.arange(10).view(5, 2)), [2, 2, 1]),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensorSeq(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensorSeq(torch.tensor([[8, 9]])),
        ),
    )


def test_torch_split_custom_dims() -> None:
    assert objects_are_equal(
        torch.split(BatchedTensorSeq(torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0), 2),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [2, 3]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[4, 5], [6, 7]]), batch_dim=1, seq_dim=0),
            BatchedTensorSeq(torch.tensor([[8, 9]]), batch_dim=1, seq_dim=0),
        ),
    )


def test_torch_split_dim_1() -> None:
    assert objects_are_equal(
        torch.split(BatchedTensorSeq(torch.arange(10).view(2, 5)), 2, dim=1),
        (
            BatchedTensorSeq(torch.tensor([[0, 1], [5, 6]])),
            BatchedTensorSeq(torch.tensor([[2, 3], [7, 8]])),
            BatchedTensorSeq(torch.tensor([[4], [9]])),
        ),
    )


###############################
#     Tests for torch.sum     #
###############################


def test_torch_sum() -> None:
    assert torch.sum(BatchedTensorSeq(torch.arange(10).view(2, 5))).equal(torch.tensor(45))


def test_torch_sum_dim_1() -> None:
    assert torch.sum(BatchedTensorSeq(torch.arange(10).view(2, 5)), dim=1).equal(
        torch.tensor([10, 35])
    )


def test_torch_sum_keepdim() -> None:
    assert torch.sum(BatchedTensorSeq(torch.arange(10).view(2, 5)), dim=1, keepdim=True).equal(
        torch.tensor([[10], [35]])
    )


##########################################
#     Tests for torch.take_along_dim     #
##########################################


@mark.parametrize(
    "indices",
    (
        torch.tensor([2, 4, 1, 3]),
        torch.tensor([[2, 4], [1, 3]]),
        torch.tensor([[[2], [4]], [[1], [3]]]),
        BatchedTensor(torch.tensor([2, 4, 1, 3])),
        BatchedTensorSeq(torch.tensor([[2, 4], [1, 3]])),
    ),
)
def test_torch_take_along_dim(indices: BatchedTensor | Tensor) -> None:
    assert torch.take_along_dim(
        BatchedTensorSeq(torch.arange(10).view(2, 5)), indices=indices
    ).equal(torch.tensor([2, 4, 1, 3]))


@mark.parametrize(
    "indices",
    (
        torch.tensor([[2, 4], [1, 3]]),
        BatchedTensor(torch.tensor([[2, 4], [1, 3]]), batch_dim=1),
        BatchedTensorSeq(torch.tensor([[2, 4], [1, 3]]), batch_dim=1, seq_dim=0),
    ),
)
def test_torch_take_along_dim_custom_dims(indices: BatchedTensor | Tensor) -> None:
    assert torch.take_along_dim(
        BatchedTensorSeq(torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0),
        indices=indices,
    ).equal(torch.tensor([2, 4, 1, 3]))


@mark.parametrize(
    "indices",
    (
        torch.tensor([[2, 4], [1, 3]]),
        BatchedTensor(torch.tensor([[2, 4], [1, 3]])),
        BatchedTensorSeq(torch.tensor([[2, 4], [1, 3]])),
    ),
)
def test_torch_take_along_dim_0(indices: BatchedTensor | Tensor) -> None:
    assert torch.take_along_dim(
        BatchedTensorSeq(torch.arange(10).view(5, 2)), indices=indices, dim=0
    ).equal(BatchedTensorSeq(torch.tensor([[4, 9], [2, 7]])))


@mark.parametrize(
    "indices",
    (
        torch.tensor([[2, 4], [1, 3]]),
        BatchedTensor(torch.tensor([[2, 4], [1, 3]]), batch_dim=1),
        BatchedTensorSeq(torch.tensor([[2, 4], [1, 3]]), batch_dim=1, seq_dim=0),
    ),
)
def test_torch_take_along_dim_0_custom_dims(indices: BatchedTensor | Tensor) -> None:
    assert torch.take_along_dim(
        BatchedTensorSeq(torch.arange(10).view(5, 2), batch_dim=1, seq_dim=0),
        indices=indices,
        dim=0,
    ).equal(BatchedTensorSeq(torch.tensor([[4, 9], [2, 7]]), batch_dim=1, seq_dim=0))


def test_torch_take_along_dim_tensor() -> None:
    assert torch.take_along_dim(
        torch.arange(10).view(5, 2), indices=BatchedTensorSeq(torch.tensor([[2, 4], [1, 3]])), dim=0
    ).equal(BatchedTensorSeq(torch.tensor([[4, 9], [2, 7]])))


def test_torch_take_along_dim_tensor2() -> None:
    assert torch.take_along_dim(
        torch.arange(10).view(5, 2), indices=torch.tensor([[2, 4], [1, 3]]), dim=0
    ).equal(torch.tensor([[4, 9], [2, 7]]))


def test_torch_take_along_dim_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        torch.take_along_dim(
            BatchedTensorSeq(torch.ones(2, 2, 2)),
            BatchedTensorSeq(torch.zeros(2, 2, 2), batch_dim=2),
        )


def test_torch_take_along_dim_incorrect_seq_dim() -> None:
    with raises(RuntimeError, match=r"The sequence dimensions do not match."):
        torch.take_along_dim(
            BatchedTensorSeq(torch.ones(2, 2, 2)), BatchedTensorSeq(torch.zeros(2, 2, 2), seq_dim=2)
        )


####################################
#     Tests for from_sequences     #
####################################


@mark.parametrize("dtype", (torch.bool, torch.long, torch.float))
def test_from_sequences(dtype: torch.dtype) -> None:
    assert from_sequences(
        [
            torch.ones(3, dtype=dtype),
            torch.ones(5, dtype=dtype),
            torch.ones(1, dtype=dtype),
            torch.ones(0, dtype=dtype),
        ]
    ).equal(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=dtype,
            )
        )
    )


@mark.parametrize("padding_value", (0.0, -1.0, float("nan")))
def test_from_sequences_padding_value(padding_value: float) -> None:
    assert from_sequences(
        [
            torch.ones(3, dtype=torch.float),
            torch.ones(5, dtype=torch.float),
            torch.ones(1, dtype=torch.float),
            torch.ones(0, dtype=torch.float),
        ],
        padding_value=padding_value,
    ).allclose(
        BatchedTensorSeq(
            torch.tensor(
                [
                    [1, 1, 1, padding_value, padding_value],
                    [1, 1, 1, 1, 1],
                    [1, padding_value, padding_value, padding_value, padding_value],
                    [padding_value, padding_value, padding_value, padding_value, padding_value],
                ],
                dtype=torch.float,
            )
        ),
        equal_nan=True,
    )
