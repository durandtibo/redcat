from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import torch
from coola import objects_are_equal
from pytest import mark, raises
from torch import Tensor
from torch.overrides import is_tensor_like

from redcat import BaseBatch, BatchedTensor, BatchedTensorSeq, BatchList
from redcat.utils.tensor import get_available_devices, get_torch_generator

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from redcat.tensor import IndexType

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


def test_batched_tensor_shape() -> None:
    assert BatchedTensor(torch.ones(2, 3)).shape == torch.Size([2, 3])


def test_batched_tensor_dim() -> None:
    assert BatchedTensor(torch.ones(2, 3)).dim() == 2


def test_batched_tensor_ndimension() -> None:
    assert BatchedTensor(torch.ones(2, 3)).ndimension() == 2


def test_batched_tensor_numel() -> None:
    assert BatchedTensor(torch.ones(2, 3)).numel() == 6


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


def test_batched_tensor_to_data() -> None:
    assert BatchedTensor(torch.ones(2, 3)).to_data().equal(torch.ones(2, 3))


def test_batched_tensor_to_data_custom_dim() -> None:
    assert BatchedTensor(torch.ones(2, 3), batch_dim=1).to_data().equal(torch.ones(2, 3))


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


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_empty_like(dtype: torch.dtype) -> None:
    batch = BatchedTensor(torch.zeros(2, 3, dtype=dtype)).empty_like()
    assert isinstance(batch, BatchedTensor)
    assert batch.data.shape == (2, 3)
    assert batch.dtype == dtype


@mark.parametrize("dtype", DTYPES)
def test_batched_tensor_empty_like_target_dtype(dtype: torch.dtype) -> None:
    batch = BatchedTensor(torch.zeros(2, 3)).empty_like(dtype=dtype)
    assert isinstance(batch, BatchedTensor)
    assert batch.data.shape == (2, 3)
    assert batch.dtype == dtype


def test_batched_tensor_empty_like_custom_batch_dim() -> None:
    batch = BatchedTensor(torch.zeros(3, 2), batch_dim=1).empty_like()
    assert isinstance(batch, BatchedTensor)
    assert batch.data.shape == (3, 2)
    assert batch.batch_dim == 1


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
def test_batched_tensor_new_full_fill_value(fill_value: float | int | bool) -> None:
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


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor__eq__(other: BatchedTensor | Tensor | int | float) -> None:
    assert (BatchedTensor(torch.arange(10).view(2, 5)) == other).equal(
        BatchedTensor(
            torch.tensor(
                [[False, False, False, False, False], [True, False, False, False, False]],
                dtype=torch.bool,
            ),
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor__ge__(other: BatchedTensor | Tensor | int | float) -> None:
    assert (BatchedTensor(torch.arange(10).view(2, 5)) >= other).equal(
        BatchedTensor(
            torch.tensor(
                [[False, False, False, False, False], [True, True, True, True, True]],
                dtype=torch.bool,
            ),
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor__gt__(other: BatchedTensor | Tensor | int | float) -> None:
    assert (BatchedTensor(torch.arange(10).view(2, 5)) > other).equal(
        BatchedTensor(
            torch.tensor(
                [[False, False, False, False, False], [False, True, True, True, True]],
                dtype=torch.bool,
            ),
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor__le__(other: BatchedTensor | Tensor | int | float) -> None:
    assert (BatchedTensor(torch.arange(10).view(2, 5)) <= other).equal(
        BatchedTensor(
            torch.tensor(
                [[True, True, True, True, True], [True, False, False, False, False]],
                dtype=torch.bool,
            ),
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor__lt__(other: BatchedTensor | Tensor | int | float) -> None:
    assert (BatchedTensor(torch.arange(10).view(2, 5)) < other).equal(
        BatchedTensor(
            torch.tensor(
                [[True, True, True, True, True], [False, False, False, False, False]],
                dtype=torch.bool,
            ),
        )
    )


def test_batched_tensor_allclose_true() -> None:
    assert BatchedTensor(torch.ones(2, 3)).allclose(BatchedTensor(torch.ones(2, 3)))


def test_batched_tensor_allclose_false_different_type() -> None:
    assert not BatchedTensor(torch.ones(2, 3, dtype=torch.float)).allclose(
        torch.ones(2, 3, dtype=torch.long)
    )


def test_batched_tensor_allclose_false_different_data() -> None:
    assert not BatchedTensor(torch.ones(2, 3)).allclose(BatchedTensor(torch.zeros(2, 3)))


def test_batched_tensor_allclose_false_different_shape() -> None:
    assert not BatchedTensor(torch.ones(2, 3)).allclose(BatchedTensor(torch.ones(2, 3, 1)))


def test_batched_tensor_allclose_false_different_batch_dim() -> None:
    assert not BatchedTensor(torch.ones(2, 3)).allclose(
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
    )


@mark.parametrize(
    ("batch", "atol"),
    (
        (BatchedTensor(torch.ones(2, 3) + 0.5), 1),
        (BatchedTensor(torch.ones(2, 3) + 0.05), 1e-1),
        (BatchedTensor(torch.ones(2, 3) + 5e-3), 1e-2),
    ),
)
def test_batched_tensor_allclose_true_atol(batch: BatchedTensor, atol: float) -> None:
    assert BatchedTensor(torch.ones(2, 3)).allclose(batch, atol=atol, rtol=0)


@mark.parametrize(
    ("batch", "rtol"),
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
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.full((2, 1), 5.0)),
        5,
        5.0,
    ),
)
def test_batched_tensor_eq(other: BatchedTensor | Tensor | int | float) -> None:
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
        .eq(BatchedTensor(torch.full((2, 5), 5), batch_dim=1))
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


def test_batched_tensor_equal_false_different_dtype() -> None:
    assert not BatchedTensor(torch.ones(2, 3, dtype=torch.float)).equal(
        torch.ones(2, 3, dtype=torch.long)
    )


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
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_ge(other: BatchedTensor | Tensor | int | float) -> None:
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


def test_batched_tensor_ge_custom_batch_dim() -> None:
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


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_gt(other: BatchedTensor | Tensor | int | float) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .gt(other)
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[False, False, False, False, False], [False, True, True, True, True]],
                    dtype=torch.bool,
                ),
            )
        )
    )


def test_batched_tensor_gt_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .gt(BatchedTensor(torch.full((2, 5), 5.0), batch_dim=1))
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[False, False, False, False, False], [False, True, True, True, True]],
                    dtype=torch.bool,
                ),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_isinf() -> None:
    assert (
        BatchedTensor(torch.tensor([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
        .isinf()
        .equal(
            BatchedTensor(
                torch.tensor([[False, False, True], [False, False, True]], dtype=torch.bool)
            )
        )
    )


def test_batched_tensor_isinf_custom_batch_dim() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]),
            batch_dim=1,
        )
        .isinf()
        .equal(
            BatchedTensor(
                torch.tensor([[False, False, True], [False, False, True]], dtype=torch.bool),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_isneginf() -> None:
    assert (
        BatchedTensor(torch.tensor([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
        .isneginf()
        .equal(
            BatchedTensor(
                torch.tensor([[False, False, False], [False, False, True]], dtype=torch.bool)
            )
        )
    )


def test_batched_tensor_isneginf_custom_batch_dim() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]),
            batch_dim=1,
        )
        .isneginf()
        .equal(
            BatchedTensor(
                torch.tensor([[False, False, False], [False, False, True]], dtype=torch.bool),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_isposinf() -> None:
    assert (
        BatchedTensor(torch.tensor([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]))
        .isposinf()
        .equal(
            BatchedTensor(
                torch.tensor([[False, False, True], [False, False, False]], dtype=torch.bool)
            )
        )
    )


def test_batched_tensor_isposinf_custom_batch_dim() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[1.0, 0.0, float("inf")], [-1.0, -2.0, float("-inf")]]),
            batch_dim=1,
        )
        .isposinf()
        .equal(
            BatchedTensor(
                torch.tensor([[False, False, True], [False, False, False]], dtype=torch.bool),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_isnan() -> None:
    assert (
        BatchedTensor(torch.tensor([[1.0, 0.0, float("nan")], [float("nan"), -2.0, -1.0]]))
        .isnan()
        .equal(
            BatchedTensor(
                torch.tensor([[False, False, True], [True, False, False]], dtype=torch.bool)
            )
        )
    )


def test_batched_tensor_isnan_custom_batch_dim() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[1.0, 0.0, float("nan")], [float("nan"), -2.0, -1.0]]),
            batch_dim=1,
        )
        .isnan()
        .equal(
            BatchedTensor(
                torch.tensor([[False, False, True], [True, False, False]], dtype=torch.bool),
                batch_dim=1,
            )
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_le(other: BatchedTensor | Tensor | int | float) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .le(other)
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[True, True, True, True, True], [True, False, False, False, False]],
                    dtype=torch.bool,
                )
            )
        )
    )


def test_batched_tensor_le_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .le(BatchedTensor(torch.full((2, 5), 5.0), batch_dim=1))
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[True, True, True, True, True], [True, False, False, False, False]],
                    dtype=torch.bool,
                ),
                batch_dim=1,
            )
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.full((2, 1), 5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_lt(other: BatchedTensor | Tensor | int | float) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .lt(other)
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[True, True, True, True, True], [False, False, False, False, False]],
                    dtype=torch.bool,
                ),
            )
        )
    )


def test_batched_tensor_lt_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .lt(BatchedTensor(torch.full((2, 5), 5.0), batch_dim=1))
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[True, True, True, True, True], [False, False, False, False, False]],
                    dtype=torch.bool,
                ),
                batch_dim=1,
            )
        )
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


##################################################
#     Mathematical | arithmetical operations     #
##################################################


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.ones(2, 3)),
        torch.ones(2, 3),
        BatchedTensor(torch.ones(2, 1)),
        1,
        1.0,
    ),
)
def test_batched_tensor__add__(other: BatchedTensor | Tensor | int | float) -> None:
    assert (BatchedTensor(torch.zeros(2, 3)) + other).equal(BatchedTensor(torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.ones(2, 3)),
        torch.ones(2, 3),
        BatchedTensor(torch.ones(2, 1)),
        1,
        1.0,
    ),
)
def test_batched_tensor__iadd__(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensor(torch.zeros(2, 3))
    batch += other
    assert batch.equal(BatchedTensor(torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor__floordiv__(other: BatchedTensor | Tensor | int | float) -> None:
    assert (BatchedTensor(torch.ones(2, 3)) // other).equal(BatchedTensor(torch.zeros(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor__ifloordiv__(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch //= other
    assert batch.equal(BatchedTensor(torch.zeros(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor__mul__(other: BatchedTensor | Tensor | int | float) -> None:
    assert (BatchedTensor(torch.ones(2, 3)) * other).equal(BatchedTensor(torch.full((2, 3), 2.0)))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor__imul__(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch *= other
    assert batch.equal(BatchedTensor(torch.full((2, 3), 2.0)))


def test_batched_tensor__neg__() -> None:
    assert (-BatchedTensor(torch.ones(2, 3))).equal(BatchedTensor(-torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor__sub__(other: BatchedTensor | Tensor | int | float) -> None:
    assert (BatchedTensor(torch.ones(2, 3)) - other).equal(BatchedTensor(-torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor__isub__(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch -= other
    assert batch.equal(BatchedTensor(-torch.ones(2, 3)))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor__truediv__(other: BatchedTensor | Tensor | int | float) -> None:
    assert (BatchedTensor(torch.ones(2, 3)) / other).equal(BatchedTensor(torch.full((2, 3), 0.5)))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor__itruediv__(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch /= other
    assert batch.equal(BatchedTensor(torch.full((2, 3), 0.5)))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor_add(other: BatchedTensor | Tensor | int | float) -> None:
    assert BatchedTensor(torch.ones(2, 3)).add(other).equal(BatchedTensor(torch.full((2, 3), 3.0)))


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
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.add(BatchedTensor(torch.ones(2, 2), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor_add_(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.add_(other)
    assert batch.equal(BatchedTensor(torch.full((2, 3), 3.0)))


def test_batched_tensor_add__alpha_2_float() -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.add_(BatchedTensor(torch.full((2, 3), 2.0)), alpha=2.0)
    assert batch.equal(BatchedTensor(torch.full((2, 3), 5.0)))


def test_batched_tensor_add__alpha_2_long() -> None:
    batch = BatchedTensor(torch.ones(2, 3, dtype=torch.long))
    batch.add_(BatchedTensor(torch.full((2, 3), 2, dtype=torch.long)), alpha=2)
    assert batch.equal(BatchedTensor(torch.full((2, 3), 5, dtype=torch.long)))


def test_batched_tensor_add__custom_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 3), batch_dim=1)
    batch.add_(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
    assert batch.equal(BatchedTensor(torch.full((2, 3), 3.0), batch_dim=1))


def test_batched_tensor_add__incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.add_(BatchedTensor(torch.ones(2, 2), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor_div(other: BatchedTensor | Tensor | int | float) -> None:
    assert BatchedTensor(torch.ones(2, 3)).div(other).equal(BatchedTensor(torch.full((2, 3), 0.5)))


def test_batched_tensor_div_rounding_mode_floor() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3))
        .div(BatchedTensor(torch.full((2, 3), 2.0)), rounding_mode="floor")
        .equal(BatchedTensor(torch.zeros(2, 3)))
    )


def test_batched_tensor_div_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .div(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
        .equal(BatchedTensor(torch.full((2, 3), 0.5), batch_dim=1))
    )


def test_batched_tensor_div_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.div(BatchedTensor(torch.ones(2, 2), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor_div_(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.div_(other)
    assert batch.equal(BatchedTensor(torch.full((2, 3), 0.5)))


def test_batched_tensor_div__rounding_mode_floor() -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.div_(BatchedTensor(torch.full((2, 3), 2.0)), rounding_mode="floor")
    assert batch.equal(BatchedTensor(torch.zeros(2, 3)))


def test_batched_tensor_div__custom_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 3), batch_dim=1)
    batch.div_(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
    assert batch.equal(BatchedTensor(torch.full((2, 3), 0.5), batch_dim=1))


def test_batched_tensor_div__incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.div_(BatchedTensor(torch.ones(2, 2), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor_fmod(other: BatchedTensor | Tensor | int | float) -> None:
    assert BatchedTensor(torch.ones(2, 3)).fmod(other).equal(BatchedTensor(torch.ones(2, 3)))


def test_batched_tensor_fmod_custom_dims() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .fmod(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
        .equal(BatchedTensor(torch.ones(2, 3), batch_dim=1))
    )


def test_batched_tensor_fmod_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.fmod(BatchedTensor(torch.ones(2, 2), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor_fmod_(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.fmod_(other)
    assert batch.equal(BatchedTensor(torch.ones(2, 3)))


def test_batched_tensor_fmod__custom_dims() -> None:
    batch = BatchedTensor(torch.ones(2, 3), batch_dim=1)
    batch.fmod_(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
    assert batch.equal(BatchedTensor(torch.ones(2, 3), batch_dim=1))


def test_batched_tensor_fmod__incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.fmod_(BatchedTensor(torch.ones(2, 2), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor_mul(other: BatchedTensor | Tensor | int | float) -> None:
    assert BatchedTensor(torch.ones(2, 3)).mul(other).equal(BatchedTensor(torch.full((2, 3), 2.0)))


def test_batched_tensor_mul_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .mul(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
        .equal(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
    )


def test_batched_tensor_mul_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.mul(BatchedTensor(torch.ones(2, 2), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor_mul_(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.mul_(other)
    assert batch.equal(BatchedTensor(torch.full((2, 3), 2.0)))


def test_batched_tensor_mul__custom_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 3), batch_dim=1)
    batch.mul_(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
    assert batch.equal(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))


def test_batched_tensor_mul__incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 3, 1))
    with raises(RuntimeError):
        batch.mul_(BatchedTensor(torch.ones(2, 3, 1), batch_dim=2))


def test_batched_tensor_neg() -> None:
    assert BatchedTensor(torch.ones(2, 3)).neg().equal(BatchedTensor(-torch.ones(2, 3)))


def test_batched_tensor_custom_batch_dim() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .neg()
        .equal(BatchedTensor(-torch.ones(2, 3), batch_dim=1))
    )


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor_sub(other: BatchedTensor | Tensor | int | float) -> None:
    assert BatchedTensor(torch.ones(2, 3)).sub(other).equal(BatchedTensor(-torch.ones(2, 3)))


def test_batched_tensor_sub_alpha_2_float() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3))
        .sub(BatchedTensor(torch.full((2, 3), 2.0)), alpha=2.0)
        .equal(BatchedTensor(-torch.full((2, 3), 3.0)))
    )


def test_batched_tensor_sub_alpha_2_long() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3, dtype=torch.long))
        .sub(BatchedTensor(torch.full((2, 3), 2, dtype=torch.long)), alpha=2)
        .equal(BatchedTensor(torch.full((2, 3), -3, dtype=torch.long)))
    )


def test_batched_tensor_sub_custom_batch_dims() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .sub(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
        .equal(BatchedTensor(-torch.ones(2, 3), batch_dim=1))
    )


def test_batched_tensor_sub_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.sub(BatchedTensor(torch.ones(2, 2), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        BatchedTensor(torch.full((2, 1), 2.0)),
        2,
        2.0,
    ),
)
def test_batched_tensor_sub_(other: BatchedTensor | Tensor | int | float) -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.sub_(other)
    assert batch.equal(BatchedTensor(-torch.ones(2, 3)))


def test_batched_tensor_sub__alpha_2_float() -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.sub_(BatchedTensor(torch.full((2, 3), 2.0)), alpha=2.0)
    assert batch.equal(BatchedTensor(torch.full((2, 3), -3.0)))


def test_batched_tensor_sub__alpha_2_long() -> None:
    batch = BatchedTensor(torch.ones(2, 3, dtype=torch.long))
    batch.sub_(BatchedTensor(torch.full((2, 3), 2, dtype=torch.long)), alpha=2)
    assert batch.equal(BatchedTensor(torch.full((2, 3), -3, dtype=torch.long)))


def test_batched_tensor_sub__custom_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 3), batch_dim=1)
    batch.sub_(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
    assert batch.equal(BatchedTensor(-torch.ones(2, 3), batch_dim=1))


def test_batched_tensor_sub__incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.sub_(BatchedTensor(torch.ones(2, 2), batch_dim=1))


###########################################################
#     Mathematical | advanced arithmetical operations     #
###########################################################


def test_batched_tensor_argsort_descending_false() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).argsort(descending=False),
        BatchedTensor(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]])),
    )


def test_batched_tensor_argsort_descending_true() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).argsort(descending=True),
        BatchedTensor(torch.tensor([[3, 0, 4, 2, 1], [0, 4, 1, 3, 2]])),
    )


def test_batched_tensor_argsort_dim_0() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])).argsort(dim=0),
        BatchedTensor(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
    )


def test_batched_tensor_argsort_dim_1() -> None:
    assert objects_are_equal(
        BatchedTensor(
            torch.tensor(
                [
                    [[0, 1], [-2, 3], [-4, 5], [-6, 7], [-8, 9]],
                    [[10, -11], [12, -13], [14, -15], [16, -17], [18, -19]],
                ]
            )
        ).argsort(dim=1),
        BatchedTensor(
            torch.tensor(
                [
                    [[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]],
                    [[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]],
                ]
            )
        ),
    )


def test_batched_tensor_argsort_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), batch_dim=1).argsort(
            dim=0
        ),
        BatchedTensor(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]), batch_dim=1),
    )


def test_batched_tensor_argsort_along_batch_descending_false() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])).argsort_along_batch(),
        BatchedTensor(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
    )


def test_batched_tensor_argsort_along_batch_descending_true() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])).argsort_along_batch(
            descending=True
        ),
        BatchedTensor(torch.tensor([[3, 0], [0, 4], [4, 1], [2, 3], [1, 2]])),
    )


def test_batched_tensor_argsort_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensor(
            torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_dim=1
        ).argsort_along_batch(),
        BatchedTensor(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]), batch_dim=1),
    )


def test_batched_tensor_cumprod_dim_0() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .cumprod(dim=0)
        .equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [0, 6, 14, 24, 36]])))
    )


def test_batched_tensor_cumprod_dim_1() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .cumprod(dim=1)
        .equal(BatchedTensor(torch.tensor([[0, 0, 0, 0, 0], [5, 30, 210, 1680, 15120]])))
    )


def test_batched_tensor_cumprod_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2), batch_dim=1)
        .cumprod(dim=1)
        .equal(
            BatchedTensor(torch.tensor([[0, 0], [2, 6], [4, 20], [6, 42], [8, 72]]), batch_dim=1)
        )
    )


def test_batched_tensor_cumprod_dtype() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .cumprod(dim=0, dtype=torch.int)
        .equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [0, 6, 14, 24, 36]], dtype=torch.int)))
    )


def test_batched_tensor_cumprod_() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch.cumprod_(dim=0)
    assert batch.equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [0, 6, 14, 24, 36]])))


def test_batched_tensor_cumprod__custom_dims() -> None:
    batch = BatchedTensor(torch.arange(10).view(5, 2), batch_dim=1)
    batch.cumprod_(dim=1)
    assert batch.equal(
        BatchedTensor(torch.tensor([[0, 0], [2, 6], [4, 20], [6, 42], [8, 72]]), batch_dim=1)
    )


def test_batched_tensor_cumprod_along_batch() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .cumprod_along_batch()
        .equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [0, 6, 14, 24, 36]])))
    )


def test_batched_tensor_cumprod_along_batch_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2), batch_dim=1)
        .cumprod_along_batch()
        .equal(
            BatchedTensor(torch.tensor([[0, 0], [2, 6], [4, 20], [6, 42], [8, 72]]), batch_dim=1)
        )
    )


def test_batched_tensor_cumprod_along_batch_dtype() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .cumprod_along_batch(dtype=torch.int)
        .equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [0, 6, 14, 24, 36]], dtype=torch.int)))
    )


def test_batched_tensor_cumprod_along_batch_() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch.cumprod_along_batch_()
    assert batch.equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [0, 6, 14, 24, 36]])))


def test_batched_tensor_cumprod_along_batch__custom_dims() -> None:
    batch = BatchedTensor(torch.arange(10).view(5, 2), batch_dim=1)
    batch.cumprod_along_batch_()
    assert batch.equal(
        BatchedTensor(torch.tensor([[0, 0], [2, 6], [4, 20], [6, 42], [8, 72]]), batch_dim=1)
    )


def test_batched_tensor_cumsum() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .cumsum(dim=0)
        .equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])))
    )


def test_batched_tensor_cumsum_dim_1() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .cumsum(dim=1)
        .equal(BatchedTensor(torch.tensor([[0, 1, 3, 6, 10], [5, 11, 18, 26, 35]])))
    )


def test_batched_tensor_cumsum_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2), batch_dim=1)
        .cumsum(dim=1)
        .equal(BatchedTensor(torch.tensor([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), batch_dim=1))
    )


def test_batched_tensor_cumsum_dtype() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .cumsum(dim=0, dtype=torch.int)
        .equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]], dtype=torch.int)))
    )


def test_batched_tensor_cumsum_() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch.cumsum_(dim=0)
    assert batch.equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])))


def test_batched_tensor_cumsum__custom_dims() -> None:
    batch = BatchedTensor(torch.arange(10).view(5, 2), batch_dim=1)
    batch.cumsum_(dim=1)
    assert batch.equal(
        BatchedTensor(torch.tensor([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), batch_dim=1)
    )


def test_batched_tensor_cumsum_along_batch() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .cumsum_along_batch()
        .equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])))
    )


def test_batched_tensor_cumsum_along_batch_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2), batch_dim=1)
        .cumsum_along_batch()
        .equal(BatchedTensor(torch.tensor([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), batch_dim=1))
    )


def test_batched_tensor_cumsum_along_batch_dtype() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .cumsum_along_batch(dtype=torch.int)
        .equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]], dtype=torch.int)))
    )


def test_batched_tensor_cumsum_along_batch_() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch.cumsum_along_batch_()
    assert batch.equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])))


def test_batched_tensor_cumsum_along_batch__custom_dims() -> None:
    batch = BatchedTensor(torch.arange(10).view(5, 2), batch_dim=1)
    batch.cumsum_along_batch_()
    assert batch.equal(
        BatchedTensor(torch.tensor([[0, 1], [2, 5], [4, 9], [6, 13], [8, 17]]), batch_dim=1)
    )


def test_batched_tensor_logcumsumexp_dim_0() -> None:
    assert (
        BatchedTensor(torch.arange(10, dtype=torch.float).view(5, 2))
        .logcumsumexp(dim=0)
        .allclose(
            BatchedTensor(
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


def test_batched_tensor_logcumsumexp_dim_1() -> None:
    assert (
        BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5))
        .logcumsumexp(dim=1)
        .allclose(
            BatchedTensor(
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


def test_batched_tensor_logcumsumexp_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10, dtype=torch.float).view(5, 2), batch_dim=1)
        .logcumsumexp(dim=0)
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [
                        [0.0, 1.0],
                        [2.1269280110429727, 3.1269280110429727],
                        [4.142931628499899, 5.142931628499899],
                        [6.145077938960783, 7.145077938960783],
                        [8.145368056908488, 9.145368056908488],
                    ]
                ),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_logcumsumexp__dim_0() -> None:
    batch = BatchedTensor(torch.arange(10, dtype=torch.float).view(5, 2))
    batch.logcumsumexp_(dim=0)
    assert batch.allclose(
        BatchedTensor(
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


def test_batched_tensor_logcumsumexp__dim_1() -> None:
    batch = BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5))
    batch.logcumsumexp_(dim=1)
    assert batch.allclose(
        BatchedTensor(
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


def test_batched_tensor_logcumsumexp__custom_dims() -> None:
    batch = BatchedTensor(torch.arange(10, dtype=torch.float).view(5, 2), batch_dim=1)
    batch.logcumsumexp_(dim=0)
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [0.0, 1.0],
                    [2.1269280110429727, 3.1269280110429727],
                    [4.142931628499899, 5.142931628499899],
                    [6.145077938960783, 7.145077938960783],
                    [8.145368056908488, 9.145368056908488],
                ]
            ),
            batch_dim=1,
        )
    )


def test_batched_tensor_logcumsumexp_along_batch() -> None:
    assert (
        BatchedTensor(torch.arange(10, dtype=torch.float).view(5, 2))
        .logcumsumexp_along_batch()
        .allclose(
            BatchedTensor(
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


def test_batched_tensor_logcumsumexp_along_batch_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5), batch_dim=1)
        .logcumsumexp_along_batch()
        .allclose(
            BatchedTensor(
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
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_logcumsumexp_along_batch_() -> None:
    batch = BatchedTensor(torch.arange(10, dtype=torch.float).view(5, 2))
    batch.logcumsumexp_along_batch_()
    assert batch.allclose(
        BatchedTensor(
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


def test_batched_tensor_logcumsumexp_along_batch__custom_dims() -> None:
    batch = BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5), batch_dim=1)
    batch.logcumsumexp_along_batch_()
    assert batch.allclose(
        BatchedTensor(
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
            batch_dim=1,
        )
    )


@mark.parametrize("permutation", (torch.tensor([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_tensor_permute_along_batch(permutation: Sequence[int] | Tensor) -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
        .permute_along_batch(permutation)
        .equal(BatchedTensor(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))
    )


def test_batched_tensor_permute_along_batch_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), batch_dim=1)
        .permute_along_batch(torch.tensor([2, 1, 3, 0]))
        .equal(BatchedTensor(torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4]]), batch_dim=1))
    )


@mark.parametrize("permutation", (torch.tensor([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_tensor_permute_along_batch_(permutation: Sequence[int] | Tensor) -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch.permute_along_batch_(permutation)
    assert batch.equal(BatchedTensor(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))


def test_batched_tensor_permute_along_batch__custom_dims() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), batch_dim=1)
    batch.permute_along_batch_(torch.tensor([2, 1, 3, 0]))
    assert batch.equal(BatchedTensor(torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4]]), batch_dim=1))


@mark.parametrize("permutation", (torch.tensor([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_tensor_permute_along_dim_0(permutation: Sequence[int] | Tensor) -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
        .permute_along_dim(permutation, dim=0)
        .equal(BatchedTensor(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))
    )


@mark.parametrize("permutation", (torch.tensor([2, 4, 1, 3, 0]), [2, 4, 1, 3, 0], (2, 4, 1, 3, 0)))
def test_batched_tensor_permute_along_dim_1(permutation: Sequence[int] | Tensor) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .permute_along_dim(permutation, dim=1)
        .equal(BatchedTensor(torch.tensor([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])))
    )


def test_batched_tensor_permute_along_dim_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), batch_dim=1)
        .permute_along_dim(torch.tensor([2, 1, 3, 0]), dim=1)
        .equal(BatchedTensor(torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4]]), batch_dim=1))
    )


@mark.parametrize("permutation", (torch.tensor([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_tensor_permute_along_dim__0(permutation: Sequence[int] | Tensor) -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch.permute_along_dim_(permutation, dim=0)
    assert batch.equal(BatchedTensor(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))


@mark.parametrize("permutation", (torch.tensor([2, 4, 1, 3, 0]), [2, 4, 1, 3, 0], (2, 4, 1, 3, 0)))
def test_batched_tensor_permute_along_seq__1(permutation: Sequence[int] | Tensor) -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch.permute_along_dim_(permutation, dim=1)
    assert batch.equal(BatchedTensor(torch.tensor([[2, 4, 1, 3, 0], [7, 9, 6, 8, 5]])))


def test_batched_tensor_permute_along_dim__custom_dims() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), batch_dim=1)
    batch.permute_along_dim_(torch.tensor([2, 1, 3, 0]), dim=1)
    assert batch.equal(BatchedTensor(torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4]]), batch_dim=1))


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_shuffle_along_batch() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
        .shuffle_along_batch()
        .equal(BatchedTensor(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))
    )


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_shuffle_along_batch_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]), batch_dim=1)
        .shuffle_along_batch()
        .equal(
            BatchedTensor(torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]]), batch_dim=1)
        )
    )


def test_batched_tensor_shuffle_along_batch_same_random_seed() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert batch.shuffle_along_batch(get_torch_generator(1)).equal(
        batch.shuffle_along_batch(get_torch_generator(1))
    )


def test_batched_tensor_shuffle_along_batch_different_random_seeds() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert not batch.shuffle_along_batch(get_torch_generator(1)).equal(
        batch.shuffle_along_batch(get_torch_generator(2))
    )


def test_batched_tensor_shuffle_along_batch_multiple_shuffle() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    generator = get_torch_generator(1)
    assert not batch.shuffle_along_batch(generator).equal(batch.shuffle_along_batch(generator))


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_shuffle_along_batch_() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch.shuffle_along_batch_()
    assert batch.equal(BatchedTensor(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_shuffle_along_batch__custom_dims() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]), batch_dim=1)
    batch.shuffle_along_batch_()
    assert batch.equal(
        BatchedTensor(torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]]), batch_dim=1)
    )


def test_batched_tensor_shuffle_along_batch__same_random_seed() -> None:
    batch1 = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch1.shuffle_along_batch_(get_torch_generator(1))
    batch2 = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch2.shuffle_along_batch_(get_torch_generator(1))
    assert batch1.equal(batch2)


def test_batched_tensor_shuffle_along_batch__different_random_seeds() -> None:
    batch1 = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch1.shuffle_along_batch_(get_torch_generator(1))
    batch2 = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch2.shuffle_along_batch_(get_torch_generator(2))
    assert not batch1.equal(batch2)


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_shuffle_along_dim() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
        .shuffle_along_dim(dim=0)
        .equal(BatchedTensor(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))
    )


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_shuffle_along_dim_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]), batch_dim=1)
        .shuffle_along_dim(dim=1)
        .equal(
            BatchedTensor(torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]]), batch_dim=1)
        )
    )


def test_batched_tensor_shuffle_along_dim_same_random_seed() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert batch.shuffle_along_dim(dim=0, generator=get_torch_generator(1)).equal(
        batch.shuffle_along_dim(dim=0, generator=get_torch_generator(1))
    )


def test_batched_tensor_shuffle_along_dim_different_random_seeds() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    assert not batch.shuffle_along_dim(dim=0, generator=get_torch_generator(1)).equal(
        batch.shuffle_along_dim(dim=0, generator=get_torch_generator(2))
    )


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_shuffle_along_dim_() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch.shuffle_along_dim_(dim=0)
    assert batch.equal(BatchedTensor(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))


@patch("redcat.base.torch.randperm", lambda *args, **kwargs: torch.tensor([2, 1, 3, 0]))
def test_batched_tensor_shuffle_along_dim__custom_dims() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]), batch_dim=1)
    batch.shuffle_along_dim_(dim=1)
    assert batch.equal(
        BatchedTensor(torch.tensor([[2, 1, 3, 0], [6, 5, 7, 4], [10, 9, 11, 8]]), batch_dim=1)
    )


def test_batched_tensor_shuffle_along_dim__same_random_seed() -> None:
    batch1 = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch1.shuffle_along_dim_(dim=0, generator=get_torch_generator(1))
    batch2 = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch2.shuffle_along_dim_(dim=0, generator=get_torch_generator(1))
    assert batch1.equal(batch2)


def test_batched_tensor_shuffle_along_dim__different_random_seeds() -> None:
    batch1 = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch1.shuffle_along_dim_(dim=0, generator=get_torch_generator(1))
    batch2 = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch2.shuffle_along_dim_(dim=0, generator=get_torch_generator(2))
    assert not batch1.equal(batch2)


def test_batched_tensor_sort() -> None:
    values, indices = BatchedTensor(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).sort()
    assert objects_are_equal(
        values, BatchedTensor(torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]))
    )
    assert objects_are_equal(
        indices, BatchedTensor(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]))
    )


def test_batched_tensor_sort_namedtuple() -> None:
    out = BatchedTensor(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).sort()
    assert objects_are_equal(
        out.values, BatchedTensor(torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]))
    )
    assert objects_are_equal(
        out.indices, BatchedTensor(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]))
    )


def test_batched_tensor_sort_descending_true() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensor(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])).sort(descending=True)
        ),
        (
            BatchedTensor(torch.tensor([[5, 4, 3, 2, 1], [9, 8, 7, 6, 5]])),
            BatchedTensor(torch.tensor([[3, 0, 4, 2, 1], [0, 4, 1, 3, 2]])),
        ),
    )


def test_batched_tensor_sort_dim_0() -> None:
    assert objects_are_equal(
        tuple(BatchedTensor(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])).sort(dim=0)),
        (
            BatchedTensor(torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])),
            BatchedTensor(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
        ),
    )


def test_batched_tensor_sort_dim_1() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensor(
                torch.tensor(
                    [
                        [[0, 1], [-2, 3], [-4, 5], [-6, 7], [-8, 9]],
                        [[10, -11], [12, -13], [14, -15], [16, -17], [18, -19]],
                    ]
                )
            ).sort(dim=1)
        ),
        (
            BatchedTensor(
                torch.tensor(
                    [
                        [[-8, 1], [-6, 3], [-4, 5], [-2, 7], [0, 9]],
                        [[10, -19], [12, -17], [14, -15], [16, -13], [18, -11]],
                    ]
                )
            ),
            BatchedTensor(
                torch.tensor(
                    [
                        [[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]],
                        [[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]],
                    ]
                )
            ),
        ),
    )


def test_batched_tensor_sort_custom_dims() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensor(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), batch_dim=1).sort(
                dim=0
            )
        ),
        (
            BatchedTensor(torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]), batch_dim=1),
            BatchedTensor(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]), batch_dim=1),
        ),
    )


def test_batched_tensor_sort_along_batch() -> None:
    values, indices = BatchedTensor(
        torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])
    ).sort_along_batch()
    assert objects_are_equal(
        values, BatchedTensor(torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]))
    )
    assert objects_are_equal(
        indices, BatchedTensor(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]))
    )


def test_batched_tensor_sort_along_batch_namedtuple() -> None:
    out = BatchedTensor(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])).sort_along_batch()
    assert objects_are_equal(
        out.values, BatchedTensor(torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]))
    )
    assert objects_are_equal(
        out.indices, BatchedTensor(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]))
    )


def test_batched_tensor_sort_along_batch_descending_false() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensor(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])).sort_along_batch()
        ),
        (
            BatchedTensor(torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])),
            BatchedTensor(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
        ),
    )


def test_batched_tensor_sort_along_batch_descending_true() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensor(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])).sort_along_batch(
                descending=True
            )
        ),
        (
            BatchedTensor(torch.tensor([[5, 9], [4, 8], [3, 7], [2, 6], [1, 5]])),
            BatchedTensor(torch.tensor([[3, 0], [0, 4], [4, 1], [2, 3], [1, 2]])),
        ),
    )


def test_batched_tensor_sort_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensor(
                torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]]), batch_dim=1
            ).sort_along_batch()
        ),
        (
            BatchedTensor(torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]), batch_dim=1),
            BatchedTensor(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]]), batch_dim=1),
        ),
    )


################################################
#     Mathematical | point-wise operations     #
################################################


def test_batched_tensor_abs() -> None:
    assert (
        BatchedTensor(torch.tensor([[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0]], dtype=torch.float))
        .abs()
        .equal(BatchedTensor(torch.tensor([[2.0, 0.0, 2.0], [1.0, 1.0, 3.0]], dtype=torch.float)))
    )


def test_batched_tensor_abs_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0]], dtype=torch.float), batch_dim=1
        )
        .abs()
        .equal(
            BatchedTensor(
                torch.tensor([[2.0, 0.0, 2.0], [1.0, 1.0, 3.0]], dtype=torch.float), batch_dim=1
            )
        )
    )


def test_batched_tensor_abs_() -> None:
    batch = BatchedTensor(torch.tensor([[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0]], dtype=torch.float))
    batch.abs_()
    assert batch.equal(
        BatchedTensor(torch.tensor([[2.0, 0.0, 2.0], [1.0, 1.0, 3.0]], dtype=torch.float))
    )


def test_batched_tensor_abs__custom_batch_dim() -> None:
    batch = BatchedTensor(
        torch.tensor([[-2.0, 0.0, 2.0], [-1.0, 1.0, 3.0]], dtype=torch.float), batch_dim=1
    )
    batch.abs_()
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[2.0, 0.0, 2.0], [1.0, 1.0, 3.0]], dtype=torch.float), batch_dim=1
        )
    )


def test_batched_tensor_clamp() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .clamp(min=2, max=5)
        .equal(BatchedTensor(torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]])))
    )


def test_batched_tensor_clamp_only_max() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .clamp(max=5)
        .equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 5, 5, 5, 5]])))
    )


def test_batched_tensor_clamp_only_min() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .clamp(min=2)
        .equal(BatchedTensor(torch.tensor([[2, 2, 2, 3, 4], [5, 6, 7, 8, 9]])))
    )


def test_batched_tensor_clamp_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .clamp(min=2, max=5)
        .equal(BatchedTensor(torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]]), batch_dim=1))
    )


def test_batched_tensor_clamp_() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch.clamp_(min=2, max=5)
    assert batch.equal(BatchedTensor(torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]])))


def test_batched_tensor_clamp__only_max() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch.clamp_(max=5)
    assert batch.equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 5, 5, 5, 5]])))


def test_batched_tensor_clamp__only_min() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch.clamp_(min=2)
    assert batch.equal(BatchedTensor(torch.tensor([[2, 2, 2, 3, 4], [5, 6, 7, 8, 9]])))


def test_batched_tensor_clamp__custom_dims() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
    batch.clamp_(min=2, max=5)
    assert batch.equal(BatchedTensor(torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]]), batch_dim=1))


def test_batched_tensor_exp() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
        .exp()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [
                        [2.7182817459106445, 7.389056205749512, 20.08553695678711],
                        [54.598148345947266, 148.4131622314453, 403.4288024902344],
                    ]
                )
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_exp_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float), batch_dim=1)
        .exp()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [
                        [2.7182817459106445, 7.389056205749512, 20.08553695678711],
                        [54.598148345947266, 148.4131622314453, 403.4288024902344],
                    ]
                ),
                batch_dim=1,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_exp_() -> None:
    batch = BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
    batch.exp_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [2.7182817459106445, 7.389056205749512, 20.08553695678711],
                    [54.598148345947266, 148.4131622314453, 403.4288024902344],
                ]
            )
        ),
        atol=1e-6,
    )


def test_batched_tensor_exp__custom_dims() -> None:
    batch = BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float), batch_dim=1)
    batch.exp_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [2.7182817459106445, 7.389056205749512, 20.08553695678711],
                    [54.598148345947266, 148.4131622314453, 403.4288024902344],
                ]
            ),
            batch_dim=1,
        ),
        atol=1e-6,
    )


def test_batched_tensor_log() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
        .log()
        .equal(
            BatchedTensor(
                torch.tensor(
                    [
                        [0.0, 0.6931471824645996, 1.0986123085021973],
                        [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                    ]
                )
            )
        )
    )


def test_batched_tensor_log_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float), batch_dim=1)
        .log()
        .equal(
            BatchedTensor(
                torch.tensor(
                    [
                        [0.0, 0.6931471824645996, 1.0986123085021973],
                        [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                    ]
                ),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_log_() -> None:
    batch = BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
    batch.log_()
    assert batch.equal(
        BatchedTensor(
            torch.tensor(
                [
                    [0.0, 0.6931471824645996, 1.0986123085021973],
                    [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                ]
            )
        )
    )


def test_batched_tensor_log__custom_dims() -> None:
    batch = BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float), batch_dim=1)
    batch.log_()
    assert batch.equal(
        BatchedTensor(
            torch.tensor(
                [
                    [0.0, 0.6931471824645996, 1.0986123085021973],
                    [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                ]
            ),
            batch_dim=1,
        )
    )


def test_batched_tensor_log10() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
        .log10()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [
                        [0.0, 0.3010300099849701, 0.4771212637424469],
                        [0.6020600199699402, 0.6989700198173523, 0.778151273727417],
                    ]
                )
            )
        )
    )


def test_batched_tensor_log10_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float), batch_dim=1)
        .log10()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [
                        [0.0, 0.3010300099849701, 0.4771212637424469],
                        [0.6020600199699402, 0.6989700198173523, 0.778151273727417],
                    ]
                ),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_log10_() -> None:
    batch = BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
    batch.log10_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [0.0, 0.3010300099849701, 0.4771212637424469],
                    [0.6020600199699402, 0.6989700198173523, 0.778151273727417],
                ]
            )
        )
    )


def test_batched_tensor_log10__custom_dims() -> None:
    batch = BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float), batch_dim=1)
    batch.log10_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [0.0, 0.3010300099849701, 0.4771212637424469],
                    [0.6020600199699402, 0.6989700198173523, 0.778151273727417],
                ]
            ),
            batch_dim=1,
        )
    )


def test_batched_tensor_log1p() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float))
        .log1p()
        .equal(
            BatchedTensor(
                torch.tensor(
                    [
                        [0.0, 0.6931471824645996, 1.0986123085021973],
                        [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                    ]
                )
            )
        )
    )


def test_batched_tensor_log1p_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float), batch_dim=1)
        .log1p()
        .equal(
            BatchedTensor(
                torch.tensor(
                    [
                        [0.0, 0.6931471824645996, 1.0986123085021973],
                        [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                    ]
                ),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_log1p_() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float))
    batch.log1p_()
    assert batch.equal(
        BatchedTensor(
            torch.tensor(
                [
                    [0.0, 0.6931471824645996, 1.0986123085021973],
                    [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                ]
            )
        )
    )


def test_batched_tensor_log1p__custom_dims() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float), batch_dim=1)
    batch.log1p_()
    assert batch.equal(
        BatchedTensor(
            torch.tensor(
                [
                    [0.0, 0.6931471824645996, 1.0986123085021973],
                    [1.3862943649291992, 1.6094379425048828, 1.7917594909667969],
                ]
            ),
            batch_dim=1,
        )
    )


def test_batched_tensor_log2() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
        .log2()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [[0.0, 1.0, 1.5849624872207642], [2.0, 2.321928024291992, 2.5849626064300537]]
                )
            )
        )
    )


def test_batched_tensor_log2_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float), batch_dim=1)
        .log2()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [[0.0, 1.0, 1.5849624872207642], [2.0, 2.321928024291992, 2.5849626064300537]]
                ),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_log2_() -> None:
    batch = BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
    batch.log2_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [[0.0, 1.0, 1.5849624872207642], [2.0, 2.321928024291992, 2.5849626064300537]]
            )
        )
    )


def test_batched_tensor_log2__custom_dims() -> None:
    batch = BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float), batch_dim=1)
    batch.log2_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [[0.0, 1.0, 1.5849624872207642], [2.0, 2.321928024291992, 2.5849626064300537]]
            ),
            batch_dim=1,
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        torch.tensor([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_batched_tensor_maximum(other: BatchedTensor | Tensor) -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [-2, -1, 0]]))
        .maximum(other)
        .equal(BatchedTensor(torch.tensor([[2, 1, 2], [0, 1, 0]])))
    )


def test_batched_tensor_maximum_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [-2, -1, 0]]), batch_dim=1)
        .maximum(BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]]), batch_dim=1))
        .equal(BatchedTensor(torch.tensor([[2, 1, 2], [0, 1, 0]]), batch_dim=1))
    )


def test_batched_tensor_maximum_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.maximum(BatchedTensor(torch.ones(2, 2), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        torch.tensor([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_batched_tensor_minimum(other: BatchedTensor | Tensor) -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [-2, -1, 0]]))
        .minimum(other)
        .equal(BatchedTensor(torch.tensor([[0, 0, 1], [-2, -1, 0]])))
    )


def test_batched_tensor_minimum_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [-2, -1, 0]]), batch_dim=1)
        .minimum(BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]]), batch_dim=1))
        .equal(BatchedTensor(torch.tensor([[0, 0, 1], [-2, -1, 0]]), batch_dim=1))
    )


def test_batched_tensor_minimum_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.minimum(BatchedTensor(torch.ones(2, 2), batch_dim=1))


@mark.parametrize(
    "exponent",
    (BatchedTensor(torch.full((2, 5), 2.0)), torch.full((2, 5), 2.0), 2, 2.0),
)
def test_batched_tensor_pow(exponent: BatchedTensor | int | float) -> None:
    assert (
        BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5))
        .pow(exponent)
        .equal(
            BatchedTensor(torch.tensor([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=torch.float))
        )
    )


def test_batched_tensor_pow_exponent_2_float() -> None:
    assert (
        BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5))
        .pow(2.0)
        .equal(
            BatchedTensor(torch.tensor([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=torch.float))
        )
    )


def test_batched_tensor_pow_exponent_2_long() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .pow(2)
        .equal(
            BatchedTensor(torch.tensor([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=torch.long))
        )
    )


def test_batched_tensor_pow_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float),
            batch_dim=1,
        )
        .pow(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
        .equal(
            BatchedTensor(
                torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_pow_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.pow(BatchedTensor(torch.ones(2, 2), batch_dim=1))


@mark.parametrize(
    "exponent",
    (BatchedTensor(torch.full((2, 5), 2.0)), torch.full((2, 5), 2.0), 2, 2.0),
)
def test_batched_tensor_pow_(exponent: BatchedTensor | int | float) -> None:
    batch = BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5))
    batch.pow_(exponent)
    assert batch.equal(
        BatchedTensor(torch.tensor([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=torch.float))
    )


def test_batched_tensor_pow__exponent_2_float() -> None:
    batch = BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5))
    batch.pow_(2.0)
    assert batch.equal(
        BatchedTensor(torch.tensor([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=torch.float))
    )


def test_batched_tensor_pow__exponent_2_long() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch.pow_(2)
    assert batch.equal(
        BatchedTensor(torch.tensor([[0, 1, 4, 9, 16], [25, 36, 49, 64, 81]], dtype=torch.long))
    )


def test_batched_tensor_pow__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float), batch_dim=1
    )
    batch.pow_(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float),
            batch_dim=1,
        )
    )


def test_batched_tensor_pow__incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.pow_(BatchedTensor(torch.ones(2, 2), batch_dim=1))


def test_batched_tensor_rsqrt() -> None:
    assert (
        BatchedTensor(torch.tensor([[1.0, 4.0], [16.0, 25.0]], dtype=torch.float))
        .rsqrt()
        .equal(BatchedTensor(torch.tensor([[1.0, 0.5], [0.25, 0.2]], dtype=torch.float)))
    )


def test_batched_tensor_rsqrt_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[1.0, 4.0], [16.0, 25.0]], dtype=torch.float),
            batch_dim=1,
        )
        .rsqrt()
        .equal(
            BatchedTensor(
                torch.tensor([[1.0, 0.5], [0.25, 0.2]], dtype=torch.float),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_rsqrt_() -> None:
    batch = BatchedTensor(torch.tensor([[1.0, 4.0], [16.0, 25.0]], dtype=torch.float))
    batch.rsqrt_()
    assert batch.equal(BatchedTensor(torch.tensor([[1.0, 0.5], [0.25, 0.2]], dtype=torch.float)))


def test_batched_tensor_rsqrt__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[1.0, 4.0], [16.0, 25.0]], dtype=torch.float),
        batch_dim=1,
    )
    batch.rsqrt_()
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[1.0, 0.5], [0.25, 0.2]], dtype=torch.float),
            batch_dim=1,
        )
    )


def test_batched_tensor_sqrt() -> None:
    assert (
        BatchedTensor(torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float))
        .sqrt()
        .equal(BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float)))
    )


def test_batched_tensor_sqrt_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float),
            batch_dim=1,
        )
        .sqrt()
        .equal(
            BatchedTensor(
                torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_sqrt_() -> None:
    batch = BatchedTensor(torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float))
    batch.sqrt_()
    assert batch.equal(
        BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float))
    )


def test_batched_tensor_sqrt__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float),
        batch_dim=1,
    )
    batch.sqrt_()
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float),
            batch_dim=1,
        )
    )


################################
#     Reduction operations     #
################################


def test_batched_tensor_amax_dim_none() -> None:
    assert BatchedTensor(torch.arange(10).view(2, 5)).amax(dim=None).equal(torch.tensor(9))


def test_batched_tensor_amax_dim_0() -> None:
    assert BatchedTensor(torch.arange(10).view(5, 2)).amax(dim=0).equal(torch.tensor([8, 9]))


def test_batched_tensor_amax_dim_1() -> None:
    assert BatchedTensor(torch.arange(10).view(2, 5)).amax(dim=1).equal(torch.tensor([4, 9]))


def test_batched_tensor_amax_keepdim_false() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5)).amax(dim=1), torch.tensor([4, 9])
    )


def test_batched_tensor_amax_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5)).amax(dim=1, keepdim=True),
        torch.tensor([[4], [9]]),
    )


def test_batched_tensor_amax_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .amax(dim=1)
        .equal(torch.tensor([4, 9]))
    )


def test_batched_tensor_amax_along_batch() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])).amax_along_batch(),
        torch.tensor([4, 9]),
    )


def test_batched_tensor_amax_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])).amax_along_batch(
            keepdim=True
        ),
        torch.tensor([[4, 9]]),
    )


def test_batched_tensor_amax_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).amax_along_batch(),
        torch.tensor([4, 9]),
    )


def test_batched_tensor_amin_dim_none() -> None:
    assert BatchedTensor(torch.arange(10).view(2, 5)).amin(dim=None).equal(torch.tensor(0))


def test_batched_tensor_amin_dim_0() -> None:
    assert BatchedTensor(torch.arange(10).view(5, 2)).amin(dim=0).equal(torch.tensor([0, 1]))


def test_batched_tensor_amin_dim_1() -> None:
    assert BatchedTensor(torch.arange(10).view(2, 5)).amin(dim=1).equal(torch.tensor([0, 5]))


def test_batched_tensor_amin_keepdim_false() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5)).amin(dim=1), torch.tensor([0, 5])
    )


def test_batched_tensor_amin_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5)).amin(dim=1, keepdim=True),
        torch.tensor([[0], [5]]),
    )


def test_batched_tensor_amin_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .amin(dim=1)
        .equal(torch.tensor([0, 5]))
    )


def test_batched_tensor_amin_along_batch() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])).amin_along_batch(),
        torch.tensor([0, 5]),
    )


def test_batched_tensor_amin_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])).amin_along_batch(
            keepdim=True
        ),
        torch.tensor([[0, 5]]),
    )


def test_batched_tensor_amin_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).amin_along_batch(),
        torch.tensor([0, 5]),
    )


def test_batched_tensor_argmax() -> None:
    assert BatchedTensor(torch.arange(10).view(2, 5)).argmax().equal(torch.tensor(9))


def test_batched_tensor_argmax_dim_0() -> None:
    assert BatchedTensor(torch.arange(10).view(5, 2)).argmax(dim=0).equal(torch.tensor([4, 4]))


def test_batched_tensor_argmax_dim_1() -> None:
    assert BatchedTensor(torch.arange(10).view(2, 5)).argmax(dim=1).equal(torch.tensor([4, 4]))


def test_batched_tensor_argmax_keepdim_false() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5)).argmax(dim=1), torch.tensor([4, 4])
    )


def test_batched_tensor_argmax_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5)).argmax(dim=1, keepdim=True),
        torch.tensor([[4], [4]]),
    )


def test_batched_tensor_argmax_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .argmax(dim=1)
        .equal(torch.tensor([4, 4]))
    )


def test_batched_tensor_argmax_along_batch() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[0, 9], [1, 6], [2, 7], [3, 8], [4, 5]])).argmax_along_batch(),
        torch.tensor([4, 0]),
    )


def test_batched_tensor_argmax_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[0, 9], [1, 6], [2, 7], [3, 8], [4, 5]])).argmax_along_batch(
            keepdim=True
        ),
        torch.tensor([[4, 0]]),
    )


def test_batched_tensor_argmax_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).argmax_along_batch(),
        torch.tensor([4, 4]),
    )


def test_batched_tensor_argmin() -> None:
    assert BatchedTensor(torch.arange(10).view(2, 5)).argmin().equal(torch.tensor(0))


def test_batched_tensor_argmin_dim_0() -> None:
    assert BatchedTensor(torch.arange(10).view(5, 2)).argmin(dim=0).equal(torch.tensor([0, 0]))


def test_batched_tensor_argmin_dim_1() -> None:
    assert BatchedTensor(torch.arange(10).view(2, 5)).argmin(dim=1).equal(torch.tensor([0, 0]))


def test_batched_tensor_argmin_keepdim_false() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5)).argmin(dim=1), torch.tensor([0, 0])
    )


def test_batched_tensor_argmin_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5)).argmin(dim=1, keepdim=True),
        torch.tensor([[0], [0]]),
    )


def test_batched_tensor_argmin_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .argmin(dim=1)
        .equal(torch.tensor([0, 0]))
    )


def test_batched_tensor_argmin_along_batch() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[0, 9], [1, 6], [2, 7], [3, 8], [4, 5]])).argmin_along_batch(),
        torch.tensor([0, 4]),
    )


def test_batched_tensor_argmin_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[0, 9], [1, 6], [2, 7], [3, 8], [4, 5]])).argmin_along_batch(
            keepdim=True
        ),
        torch.tensor([[0, 4]]),
    )


def test_batched_tensor_argmin_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).argmin_along_batch(),
        torch.tensor([0, 0]),
    )


def test_batched_tensor_max() -> None:
    assert BatchedTensor(torch.arange(10).view(2, 5)).max().equal(torch.tensor(9))


def test_batched_tensor_max_dim_tuple() -> None:
    values, indices = BatchedTensor(torch.arange(10).view(2, 5)).max(dim=1)
    assert values.equal(torch.tensor([4, 9]))
    assert indices.equal(torch.tensor([4, 4]))


def test_batched_tensor_max_dim_namedtuple() -> None:
    out = BatchedTensor(torch.arange(10).view(2, 5)).max(dim=1)
    assert out.values.equal(torch.tensor([4, 9]))
    assert out.indices.equal(torch.tensor([4, 4]))


def test_batched_tensor_max_keepdim_false() -> None:
    assert objects_are_equal(
        tuple(BatchedTensor(torch.arange(10).view(2, 5)).max(dim=1)),
        (torch.tensor([4, 9]), torch.tensor([4, 4])),
    )


def test_batched_tensor_max_keepdim_true() -> None:
    assert objects_are_equal(
        tuple(BatchedTensor(torch.arange(10).view(2, 5)).max(dim=1, keepdim=True)),
        (torch.tensor([[4], [9]]), torch.tensor([[4], [4]])),
    )


def test_batched_tensor_max_custom_dims() -> None:
    assert BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).max().equal(torch.tensor(9))


def test_batched_tensor_max_along_batch() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensor(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])).max_along_batch()
        ),
        (torch.tensor([4, 9]), torch.tensor([4, 4])),
    )


def test_batched_tensor_max_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensor(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])).max_along_batch(
                keepdim=True
            )
        ),
        (torch.tensor([[4, 9]]), torch.tensor([[4, 4]])),
    )


def test_batched_tensor_max_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        tuple(BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).max_along_batch()),
        (torch.tensor([4, 9]), torch.tensor([4, 4])),
    )


def test_batched_tensor_mean() -> None:
    assert (
        BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5))
        .mean()
        .equal(torch.tensor(4.5))
    )


def test_batched_tensor_mean_keepdim_false() -> None:
    assert (
        BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5))
        .mean(dim=1)
        .equal(torch.tensor([2.0, 7.0]))
    )


def test_batched_tensor_mean_keepdim_true() -> None:
    assert (
        BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5))
        .mean(dim=1, keepdim=True)
        .equal(torch.tensor([[2.0], [7.0]]))
    )


def test_batched_tensor_mean_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5), batch_dim=1)
        .mean()
        .equal(torch.tensor(4.5))
    )


def test_batched_tensor_mean_along_batch() -> None:
    assert (
        BatchedTensor(torch.arange(10, dtype=torch.float).view(5, 2))
        .mean_along_batch()
        .equal(torch.tensor([4.0, 5.0], dtype=torch.float))
    )


def test_batched_tensor_mean_along_batch_keepdim_true() -> None:
    assert (
        BatchedTensor(torch.arange(10, dtype=torch.float).view(5, 2))
        .mean_along_batch(keepdim=True)
        .equal(torch.tensor([[4.0, 5.0]], dtype=torch.float))
    )


def test_batched_tensor_mean_along_batch_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5), batch_dim=1)
        .mean_along_batch()
        .equal(torch.tensor([2.0, 7.0], dtype=torch.float))
    )


def test_batched_tensor_median() -> None:
    assert BatchedTensor(torch.arange(10).view(2, 5)).median().equal(torch.tensor(4))


def test_batched_tensor_median_keepdim_false() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5)).median(dim=1),
        torch.return_types.median([torch.tensor([2, 7]), torch.tensor([2, 2])]),
    )


def test_batched_tensor_median_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5)).median(dim=1, keepdim=True),
        torch.return_types.median([torch.tensor([[2], [7]]), torch.tensor([[2], [2]])]),
    )


def test_batched_tensor_median_custom_dims() -> None:
    assert BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).median().equal(torch.tensor(4))


def test_batched_tensor_median_along_batch() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])).median_along_batch(),
        torch.return_types.median([torch.tensor([2, 7]), torch.tensor([2, 2])]),
    )


def test_batched_tensor_median_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])).median_along_batch(
            keepdim=True
        ),
        torch.return_types.median([torch.tensor([[2, 7]]), torch.tensor([[2, 2]])]),
    )


def test_batched_tensor_median_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).median_along_batch(),
        torch.return_types.median([torch.tensor([2, 7]), torch.tensor([2, 2])]),
    )


def test_batched_tensor_min() -> None:
    assert BatchedTensor(torch.arange(10).view(2, 5)).min().equal(torch.tensor(0))


def test_batched_tensor_min_dim_tuple() -> None:
    values, indices = BatchedTensor(torch.arange(10).view(2, 5)).min(dim=1)
    assert values.equal(torch.tensor([0, 5]))
    assert indices.equal(torch.tensor([0, 0]))


def test_batched_tensor_min_dim_namedtuple() -> None:
    out = BatchedTensor(torch.arange(10).view(2, 5)).min(dim=1)
    assert out.values.equal(torch.tensor([0, 5]))
    assert out.indices.equal(torch.tensor([0, 0]))


def test_batched_tensor_min_keepdim_false() -> None:
    assert objects_are_equal(
        tuple(BatchedTensor(torch.arange(10).view(2, 5)).min(dim=1)),
        (torch.tensor([0, 5]), torch.tensor([0, 0])),
    )


def test_batched_tensor_min_keepdim_true() -> None:
    assert objects_are_equal(
        tuple(BatchedTensor(torch.arange(10).view(2, 5)).min(dim=1, keepdim=True)),
        (torch.tensor([[0], [5]]), torch.tensor([[0], [0]])),
    )


def test_batched_tensor_min_custom_dims() -> None:
    assert BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).min().equal(torch.tensor(0))


def test_batched_tensor_min_along_batch() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensor(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])).min_along_batch()
        ),
        (torch.tensor([0, 5]), torch.tensor([0, 0])),
    )


def test_batched_tensor_min_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        tuple(
            BatchedTensor(torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])).min_along_batch(
                keepdim=True
            )
        ),
        (torch.tensor([[0, 5]]), torch.tensor([[0, 0]])),
    )


def test_batched_tensor_min_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        tuple(BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).min_along_batch()),
        (torch.tensor([0, 5]), torch.tensor([0, 0])),
    )


def test_batched_tensor_nanmean() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nanmean()
        .equal(torch.tensor(4.0))
    )


def test_batched_tensor_nanmean_keepdim_false() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nanmean(dim=1)
        .equal(torch.tensor([2.0, 6.5]))
    )


def test_batched_tensor_nanmean_keepdim_true() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nanmean(dim=1, keepdim=True)
        .equal(torch.tensor([[2.0], [6.5]]))
    )


def test_batched_tensor_nanmean_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]), batch_dim=1)
        .nanmean()
        .equal(torch.tensor(4.0))
    )


def test_batched_tensor_nanmean_along_batch() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]])
        )
        .nanmean_along_batch()
        .equal(torch.tensor([2.0, 6.5]))
    )


def test_batched_tensor_nanmean_along_batch_keepdim_true() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]])
        )
        .nanmean_along_batch(keepdim=True)
        .equal(torch.tensor([[2.0, 6.5]]))
    )


def test_batched_tensor_nanmean_along_batch_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]), batch_dim=1)
        .nanmean_along_batch()
        .equal(torch.tensor([2.0, 6.5]))
    )


def test_batched_tensor_nanmedian() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nanmedian()
        .equal(torch.tensor(4.0))
    )


def test_batched_tensor_nanmedian_keepdim_false() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])).nanmedian(dim=1),
        torch.return_types.nanmedian([torch.tensor([2.0, 6.0]), torch.tensor([2, 1])]),
    )


def test_batched_tensor_nanmedian_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])).nanmedian(
            dim=1, keepdim=True
        ),
        torch.return_types.nanmedian([torch.tensor([[2.0], [6.0]]), torch.tensor([[2], [1]])]),
    )


def test_batched_tensor_nanmedian_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]), batch_dim=1)
        .nanmedian()
        .equal(torch.tensor(4.0))
    )


def test_batched_tensor_nanmedian_along_batch() -> None:
    assert objects_are_equal(
        BatchedTensor(
            torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]])
        ).nanmedian_along_batch(),
        torch.return_types.nanmedian([torch.tensor([2.0, 6.0]), torch.tensor([2, 1])]),
    )


def test_batched_tensor_nanmedian_along_batch_keepdim_true() -> None:
    assert objects_are_equal(
        BatchedTensor(
            torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]])
        ).nanmedian_along_batch(keepdim=True),
        torch.return_types.nanmedian([torch.tensor([[2.0, 6.0]]), torch.tensor([[2, 1]])]),
    )


def test_batched_tensor_nanmedian_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensor(
            torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]), batch_dim=1
        ).nanmedian_along_batch(),
        torch.return_types.nanmedian([torch.tensor([2.0, 6.0]), torch.tensor([2, 1])]),
    )


def test_batched_tensor_nansum() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nansum()
        .equal(torch.tensor(36.0))
    )


def test_batched_tensor_nansum_keepdim_false() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nansum(dim=1)
        .equal(torch.tensor([10.0, 26.0]))
    )


def test_batched_tensor_nansum_keepdim_true() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
        .nansum(dim=1, keepdim=True)
        .equal(torch.tensor([[10.0], [26.0]]))
    )


def test_batched_tensor_nansum_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]), batch_dim=1)
        .nansum()
        .equal(torch.tensor(36.0))
    )


def test_batched_tensor_nansum_along_batch() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]])
        )
        .nansum_along_batch()
        .equal(torch.tensor([10.0, 26.0]))
    )


def test_batched_tensor_nansum_along_batch_keepdim_true() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, float("nan")]])
        )
        .nansum_along_batch(keepdim=True)
        .equal(torch.tensor([[10.0, 26.0]]))
    )


def test_batched_tensor_nansum_along_batch_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]), batch_dim=1)
        .nansum_along_batch()
        .equal(torch.tensor([10.0, 26.0]))
    )


def test_batched_tensor_prod() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]]))
        .prod()
        .equal(torch.tensor(362880))
    )


def test_batched_tensor_prod_keepdim_false() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]]))
        .prod(dim=1)
        .equal(torch.tensor([120, 3024]))
    )


def test_batched_tensor_prod_keepdim_true() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]]))
        .prod(dim=1, keepdim=True)
        .equal(torch.tensor([[120], [3024]]))
    )


def test_batched_tensor_prod_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]]), batch_dim=1)
        .prod()
        .equal(torch.tensor(362880))
    )


def test_batched_tensor_prod_along_batch_keepdim_false() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 6], [2, 7], [3, 8], [4, 9], [5, 1]]))
        .prod_along_batch()
        .equal(torch.tensor([120, 3024]))
    )


def test_batched_tensor_prod_along_batch_keepdim_true() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 6], [2, 7], [3, 8], [4, 9], [5, 1]]))
        .prod_along_batch(keepdim=True)
        .equal(torch.tensor([[120, 3024]]))
    )


def test_batched_tensor_prod_along_batch_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 1]]), batch_dim=1)
        .prod_along_batch()
        .equal(torch.tensor([120, 3024]))
    )


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_sum(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5).to(dtype=dtype))
        .sum()
        .equal(torch.tensor(45, dtype=dtype))
    )


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_sum_keepdim_false(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5).to(dtype=dtype))
        .sum(dim=1)
        .equal(torch.tensor([10, 35], dtype=dtype))
    )


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_sum_keepdim_true(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5).to(dtype=dtype))
        .sum(dim=1, keepdim=True)
        .equal(torch.tensor([[10], [35]], dtype=dtype))
    )


def test_batched_tensor_sum_custom_dims() -> None:
    assert BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).sum().equal(torch.tensor(45))


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_sum_along_batch(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2).to(dtype=dtype))
        .sum_along_batch()
        .equal(torch.tensor([20, 25], dtype=dtype))
    )


@mark.parametrize("dtype", (torch.float, torch.long))
def test_batched_tensor_sum_along_batch_keepdim_true(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2).to(dtype=dtype))
        .sum_along_batch(keepdim=True)
        .equal(torch.tensor([[20, 25]], dtype=dtype))
    )


def test_batched_tensor_sum_along_batch_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 4], [1, 2], [2, 5]]), batch_dim=1)
        .sum_along_batch()
        .equal(torch.tensor([4, 3, 7]))
    )


###########################################
#     Mathematical | trigo operations     #
###########################################


def test_batched_tensor_acos() -> None:
    assert (
        BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
        .acos()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [[math.pi, math.pi / 2, 0.0], [2 * math.pi / 3, math.pi / 2, math.pi / 3]],
                    dtype=torch.float,
                )
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_acos_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
            batch_dim=1,
        )
        .acos()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [[math.pi, math.pi / 2, 0.0], [2 * math.pi / 3, math.pi / 2, math.pi / 3]],
                    dtype=torch.float,
                ),
                batch_dim=1,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_acos_() -> None:
    batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
    batch.acos_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [[math.pi, math.pi / 2, 0.0], [2 * math.pi / 3, math.pi / 2, math.pi / 3]],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_batched_tensor_acos__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
        batch_dim=1,
    )
    batch.acos_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [[math.pi, math.pi / 2, 0.0], [2 * math.pi / 3, math.pi / 2, math.pi / 3]],
                dtype=torch.float,
            ),
            batch_dim=1,
        ),
        atol=1e-6,
    )


def test_batched_tensor_acosh() -> None:
    assert (
        BatchedTensor(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        .acosh()
        .allclose(
            BatchedTensor(
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


def test_batched_tensor_acosh_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            batch_dim=1,
        )
        .acosh()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [
                        [0.0, 1.3169578969248166, 1.762747174039086],
                        [2.0634370688955608, 2.2924316695611777, 2.477888730288475],
                    ],
                    dtype=torch.float,
                ),
                batch_dim=1,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_acosh_() -> None:
    batch = BatchedTensor(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    batch.acosh_()
    assert batch.allclose(
        BatchedTensor(
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


def test_batched_tensor_acosh__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        batch_dim=1,
    )
    batch.acosh_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [0.0, 1.3169578969248166, 1.762747174039086],
                    [2.0634370688955608, 2.2924316695611777, 2.477888730288475],
                ],
                dtype=torch.float,
            ),
            batch_dim=1,
        ),
        atol=1e-6,
    )


def test_batched_tensor_asin() -> None:
    assert (
        BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
        .asin()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [[-math.pi / 2, 0.0, math.pi / 2], [-math.pi / 6, 0.0, math.pi / 6]],
                    dtype=torch.float,
                )
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_asin_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
            batch_dim=1,
        )
        .asin()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [[-math.pi / 2, 0.0, math.pi / 2], [-math.pi / 6, 0.0, math.pi / 6]],
                    dtype=torch.float,
                ),
                batch_dim=1,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_asin_() -> None:
    batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
    batch.asin_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [[-math.pi / 2, 0.0, math.pi / 2], [-math.pi / 6, 0.0, math.pi / 6]],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_batched_tensor_asin__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
        batch_dim=1,
    )
    batch.asin_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [[-math.pi / 2, 0.0, math.pi / 2], [-math.pi / 6, 0.0, math.pi / 6]],
                dtype=torch.float,
            ),
            batch_dim=1,
        ),
        atol=1e-6,
    )


def test_batched_tensor_asinh() -> None:
    assert (
        BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
        .asinh()
        .allclose(
            BatchedTensor(
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


def test_batched_tensor_asinh_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
            batch_dim=1,
        )
        .asinh()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [
                        [-0.8813735842704773, 0.0, 0.8813735842704773],
                        [-0.4812118113040924, 0.0, 0.4812118113040924],
                    ],
                    dtype=torch.float,
                ),
                batch_dim=1,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_asinh_() -> None:
    batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
    batch.asinh_()
    assert batch.allclose(
        BatchedTensor(
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


def test_batched_tensor_asinh__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
        batch_dim=1,
    )
    batch.asinh_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [-0.8813735842704773, 0.0, 0.8813735842704773],
                    [-0.4812118113040924, 0.0, 0.4812118113040924],
                ],
                dtype=torch.float,
            ),
            batch_dim=1,
        ),
        atol=1e-6,
    )


def test_batched_tensor_atan() -> None:
    assert (
        BatchedTensor(torch.tensor([[0.0, 1.0, math.sqrt(3.0)], [-math.sqrt(3.0), -1.0, 0.0]]))
        .atan()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [[0.0, math.pi / 4, math.pi / 3], [-math.pi / 3, -math.pi / 4, 0.0]],
                    dtype=torch.float,
                )
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_atan_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[0.0, 1.0, math.sqrt(3.0)], [-math.sqrt(3.0), -1.0, 0.0]]),
            batch_dim=1,
        )
        .atan()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [[0.0, math.pi / 4, math.pi / 3], [-math.pi / 3, -math.pi / 4, 0.0]],
                    dtype=torch.float,
                ),
                batch_dim=1,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_atan_() -> None:
    batch = BatchedTensor(torch.tensor([[0.0, 1.0, math.sqrt(3.0)], [-math.sqrt(3.0), -1.0, 0.0]]))
    batch.atan_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [[0.0, math.pi / 4, math.pi / 3], [-math.pi / 3, -math.pi / 4, 0.0]],
                dtype=torch.float,
            )
        ),
        atol=1e-6,
    )


def test_batched_tensor_atan__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[0.0, 1.0, math.sqrt(3.0)], [-math.sqrt(3.0), -1.0, 0.0]]),
        batch_dim=1,
    )
    batch.atan_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [[0.0, math.pi / 4, math.pi / 3], [-math.pi / 3, -math.pi / 4, 0.0]],
                dtype=torch.float,
            ),
            batch_dim=1,
        ),
        atol=1e-6,
    )


def test_batched_tensor_atanh() -> None:
    assert (
        BatchedTensor(torch.tensor([[-0.5, 0.0, 0.5], [-0.1, 0.0, 0.1]]))
        .atanh()
        .allclose(
            BatchedTensor(
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


def test_batched_tensor_atanh_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[-0.5, 0.0, 0.5], [-0.1, 0.0, 0.1]]),
            batch_dim=1,
        )
        .atanh()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [
                        [-0.5493061443340549, 0.0, 0.5493061443340549],
                        [-0.10033534773107558, 0.0, 0.10033534773107558],
                    ],
                    dtype=torch.float,
                ),
                batch_dim=1,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_atanh_() -> None:
    batch = BatchedTensor(torch.tensor([[-0.5, 0.0, 0.5], [-0.1, 0.0, 0.1]]))
    batch.atanh_()
    assert batch.allclose(
        BatchedTensor(
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


def test_batched_tensor_atanh__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[-0.5, 0.0, 0.5], [-0.1, 0.0, 0.1]]),
        batch_dim=1,
    )
    batch.atanh_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [-0.5493061443340549, 0.0, 0.5493061443340549],
                    [-0.10033534773107558, 0.0, 0.10033534773107558],
                ],
                dtype=torch.float,
            ),
            batch_dim=1,
        ),
        atol=1e-6,
    )


def test_batched_tensor_cos() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]])
        )
        .cos()
        .allclose(
            BatchedTensor(torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, 1.0]], dtype=torch.float)),
            atol=1e-6,
        )
    )


def test_batched_tensor_cos_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]]),
            batch_dim=1,
        )
        .cos()
        .allclose(
            BatchedTensor(
                torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, 1.0]], dtype=torch.float),
                batch_dim=1,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_cos_() -> None:
    batch = BatchedTensor(
        torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]])
    )
    batch.cos_()
    assert batch.allclose(
        BatchedTensor(torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, 1.0]], dtype=torch.float)),
        atol=1e-6,
    )


def test_batched_tensor_cos__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]]),
        batch_dim=1,
    )
    batch.cos_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor([[1.0, 0.0, -1.0], [1.0, 0.0, 1.0]], dtype=torch.float),
            batch_dim=1,
        ),
        atol=1e-6,
    )


def test_batched_tensor_cosh() -> None:
    assert (
        BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
        .cosh()
        .allclose(
            BatchedTensor(
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


def test_batched_tensor_cosh_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
            batch_dim=1,
        )
        .cosh()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [
                        [1.5430806348152437, 1.0, 1.5430806348152437],
                        [1.1276259652063807, 1.0, 1.1276259652063807],
                    ],
                    dtype=torch.float,
                ),
                batch_dim=1,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_cosh_() -> None:
    batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]))
    batch.cosh_()
    assert batch.allclose(
        BatchedTensor(
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


def test_batched_tensor_cosh__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]),
        batch_dim=1,
    )
    batch.cosh_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [1.5430806348152437, 1.0, 1.5430806348152437],
                    [1.1276259652063807, 1.0, 1.1276259652063807],
                ],
                dtype=torch.float,
            ),
            batch_dim=1,
        ),
        atol=1e-6,
    )


def test_batched_tensor_sin() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]])
        )
        .sin()
        .allclose(
            BatchedTensor(torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float)),
            atol=1e-6,
        )
    )


def test_batched_tensor_sin_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]]),
            batch_dim=1,
        )
        .sin()
        .allclose(
            BatchedTensor(
                torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float),
                batch_dim=1,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_sin_() -> None:
    batch = BatchedTensor(
        torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]])
    )
    batch.sin_()
    assert batch.allclose(
        BatchedTensor(torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float)),
        atol=1e-6,
    )


def test_batched_tensor_sin__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[0.0, 0.5 * math.pi, math.pi], [2 * math.pi, 1.5 * math.pi, 0.0]]),
        batch_dim=1,
    )
    batch.sin_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float),
            batch_dim=1,
        ),
        atol=1e-6,
    )


def test_batched_tensor_sinh() -> None:
    assert (
        BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]], dtype=torch.float))
        .sinh()
        .allclose(
            BatchedTensor(
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


def test_batched_tensor_sinh_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]], dtype=torch.float),
            batch_dim=1,
        )
        .sinh()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [
                        [-1.175201177597046, 0.0, 1.175201177597046],
                        [-0.5210952758789062, 0.0, 0.5210952758789062],
                    ],
                    dtype=torch.float,
                ),
                batch_dim=1,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_sinh_() -> None:
    batch = BatchedTensor(torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]], dtype=torch.float))
    batch.sinh_()
    assert batch.allclose(
        BatchedTensor(
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


def test_batched_tensor_sinh__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]], dtype=torch.float),
        batch_dim=1,
    )
    batch.sinh_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [-1.175201177597046, 0.0, 1.175201177597046],
                    [-0.5210952758789062, 0.0, 0.5210952758789062],
                ],
                dtype=torch.float,
            ),
            batch_dim=1,
        ),
        atol=1e-6,
    )


def test_batched_tensor_tan() -> None:
    assert (
        BatchedTensor(
            torch.tensor(
                [[0.0, 0.25 * math.pi, math.pi], [2 * math.pi, 1.75 * math.pi, -0.25 * math.pi]]
            )
        )
        .tan()
        .allclose(
            BatchedTensor(torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, -1.0]], dtype=torch.float)),
            atol=1e-6,
        )
    )


def test_batched_tensor_tan_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor(
                [[0.0, 0.25 * math.pi, math.pi], [2 * math.pi, 1.75 * math.pi, -0.25 * math.pi]]
            ),
            batch_dim=1,
        )
        .tan()
        .allclose(
            BatchedTensor(
                torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, -1.0]], dtype=torch.float),
                batch_dim=1,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_tan_() -> None:
    batch = BatchedTensor(
        torch.tensor(
            [[0.0, 0.25 * math.pi, math.pi], [2 * math.pi, 1.75 * math.pi, -0.25 * math.pi]]
        )
    )
    batch.tan_()
    assert batch.allclose(
        BatchedTensor(torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, -1.0]], dtype=torch.float)),
        atol=1e-6,
    )


def test_batched_tensor_tan__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor(
            [[0.0, 0.25 * math.pi, math.pi], [2 * math.pi, 1.75 * math.pi, -0.25 * math.pi]]
        ),
        batch_dim=1,
    )
    batch.tan_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, -1.0]], dtype=torch.float),
            batch_dim=1,
        ),
        atol=1e-6,
    )


def test_batched_tensor_tanh() -> None:
    assert (
        BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]]))
        .tanh()
        .allclose(
            BatchedTensor(
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


def test_batched_tensor_tanh_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]]),
            batch_dim=1,
        )
        .tanh()
        .allclose(
            BatchedTensor(
                torch.tensor(
                    [
                        [0.0, 0.7615941559557649, 0.9640275800758169],
                        [-0.9640275800758169, -0.7615941559557649, 0.0],
                    ],
                    dtype=torch.float,
                ),
                batch_dim=1,
            ),
            atol=1e-6,
        )
    )


def test_batched_tensor_tanh_() -> None:
    batch = BatchedTensor(torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]]))
    batch.tanh_()
    assert batch.allclose(
        BatchedTensor(
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


def test_batched_tensor_tanh__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]]),
        batch_dim=1,
    )
    batch.tanh_()
    assert batch.allclose(
        BatchedTensor(
            torch.tensor(
                [
                    [0.0, 0.7615941559557649, 0.9640275800758169],
                    [-0.9640275800758169, -0.7615941559557649, 0.0],
                ],
                dtype=torch.float,
            ),
            batch_dim=1,
        ),
        atol=1e-6,
    )


#############################################
#     Mathematical | logical operations     #
#############################################


@mark.parametrize(
    "other",
    (
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_logical_and(other: BatchedTensor | Tensor, dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(
            torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
        )
        .logical_and(other)
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[True, False, False, False], [True, False, True, False]], dtype=torch.bool
                )
            )
        )
    )


def test_batched_tensor_logical_and_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor(
                [[True, True, False, False], [True, False, True, False]], dtype=torch.bool
            ),
            batch_dim=1,
        )
        .logical_and(
            BatchedTensor(
                torch.tensor(
                    [[True, False, True, False], [True, True, True, True]], dtype=torch.bool
                ),
                batch_dim=1,
            )
        )
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[True, False, False, False], [True, False, True, False]], dtype=torch.bool
                ),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_logical_and_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.zeros(2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_and(
            BatchedTensor(
                torch.zeros(2, 2, dtype=torch.bool),
                batch_dim=1,
            )
        )


@mark.parametrize(
    "other",
    (
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_logical_and_(other: BatchedTensor | Tensor, dtype: torch.dtype) -> None:
    batch = BatchedTensor(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
    )
    batch.logical_and_(other)
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[True, False, False, False], [True, False, True, False]], dtype=dtype)
        )
    )


def test_batched_tensor_logical_and__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=torch.bool),
        batch_dim=1,
    )
    batch.logical_and_(
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
            batch_dim=1,
        )
    )
    assert batch.equal(
        BatchedTensor(
            torch.tensor(
                [[True, False, False, False], [True, False, True, False]], dtype=torch.bool
            ),
            batch_dim=1,
        )
    )


def test_batched_tensor_logical_and__incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.zeros(2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_and_(
            BatchedTensor(
                torch.zeros(2, 2, dtype=torch.bool),
                batch_dim=1,
            )
        )


@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_logical_not(dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(
            torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
        )
        .logical_not()
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[False, False, True, True], [False, True, False, True]], dtype=torch.bool
                )
            )
        )
    )


def test_batched_tensor_logical_not_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor(
                [[True, True, False, False], [True, False, True, False]], dtype=torch.bool
            ),
            batch_dim=1,
        )
        .logical_not()
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[False, False, True, True], [False, True, False, True]], dtype=torch.bool
                ),
                batch_dim=1,
            )
        )
    )


@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_logical_not_(dtype: torch.dtype) -> None:
    batch = BatchedTensor(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
    )
    batch.logical_not_()
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[False, False, True, True], [False, True, False, True]], dtype=dtype)
        )
    )


def test_batched_tensor_logical_not__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=torch.bool),
        batch_dim=1,
    )
    batch.logical_not_()
    assert batch.equal(
        BatchedTensor(
            torch.tensor(
                [[False, False, True, True], [False, True, False, True]], dtype=torch.bool
            ),
            batch_dim=1,
        )
    )


@mark.parametrize(
    "other",
    (
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_logical_or(other: BatchedTensor | Tensor, dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(
            torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
        )
        .logical_or(other)
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[True, True, True, False], [True, True, True, True]], dtype=torch.bool
                )
            )
        )
    )


def test_batched_tensor_logical_or_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor(
                [[True, True, False, False], [True, False, True, False]], dtype=torch.bool
            ),
            batch_dim=1,
        )
        .logical_or(
            BatchedTensor(
                torch.tensor(
                    [[True, False, True, False], [True, True, True, True]], dtype=torch.bool
                ),
                batch_dim=1,
            )
        )
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[True, True, True, False], [True, True, True, True]], dtype=torch.bool
                ),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_logical_or_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.zeros(2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_or(
            BatchedTensor(
                torch.zeros(2, 2, dtype=torch.bool),
                batch_dim=1,
            )
        )


@mark.parametrize(
    "other",
    (
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_logical_or_(other: BatchedTensor | Tensor, dtype: torch.dtype) -> None:
    batch = BatchedTensor(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
    )
    batch.logical_or_(other)
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[True, True, True, False], [True, True, True, True]], dtype=dtype)
        )
    )


def test_batched_tensor_logical_or__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=torch.bool),
        batch_dim=1,
    )
    batch.logical_or_(
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
            batch_dim=1,
        )
    )
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[True, True, True, False], [True, True, True, True]], dtype=torch.bool),
            batch_dim=1,
        )
    )


def test_batched_tensor_logical_or__incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.zeros(2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_or_(
            BatchedTensor(
                torch.zeros(2, 2, dtype=torch.bool),
                batch_dim=1,
            )
        )


@mark.parametrize(
    "other",
    (
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_logical_xor(other: BatchedTensor | Tensor, dtype: torch.dtype) -> None:
    assert (
        BatchedTensor(
            torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
        )
        .logical_xor(other)
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[False, True, True, False], [False, True, False, True]], dtype=torch.bool
                )
            )
        )
    )


def test_batched_tensor_logical_xor_custom_dims() -> None:
    assert (
        BatchedTensor(
            torch.tensor(
                [[True, True, False, False], [True, False, True, False]], dtype=torch.bool
            ),
            batch_dim=1,
        )
        .logical_xor(
            BatchedTensor(
                torch.tensor(
                    [[True, False, True, False], [True, True, True, True]], dtype=torch.bool
                ),
                batch_dim=1,
            )
        )
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[False, True, True, False], [False, True, False, True]], dtype=torch.bool
                ),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_logical_xor_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.zeros(2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_xor(
            BatchedTensor(
                torch.zeros(2, 2, dtype=torch.bool),
                batch_dim=1,
            )
        )


@mark.parametrize(
    "other",
    (
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_logical_xor_(other: BatchedTensor | Tensor, dtype: torch.dtype) -> None:
    batch = BatchedTensor(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
    )
    batch.logical_xor_(other)
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[False, True, True, False], [False, True, False, True]], dtype=dtype)
        )
    )


def test_batched_tensor_logical_xor__custom_dims() -> None:
    batch = BatchedTensor(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=torch.bool),
        batch_dim=1,
    )
    batch.logical_xor_(
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
            batch_dim=1,
        )
    )
    assert batch.equal(
        BatchedTensor(
            torch.tensor(
                [[False, True, True, False], [False, True, False, True]], dtype=torch.bool
            ),
            batch_dim=1,
        )
    )


def test_batched_tensor_logical_xor__incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.zeros(2, 2, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_xor_(
            BatchedTensor(
                torch.zeros(2, 2, dtype=torch.bool),
                batch_dim=1,
            )
        )


##########################################################
#    Indexing, slicing, joining, mutating operations     #
##########################################################


def test_batched_tensor__getitem___none() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    assert batch[None].equal(torch.tensor([[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]]))


def test_batched_tensor__getitem___int() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    assert batch[0].equal(torch.tensor([0, 1, 2, 3, 4]))


def test_batched_tensor__range___slice() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    assert batch[0:2, 2:4].equal(torch.tensor([[2, 3], [7, 8]]))


@mark.parametrize(
    "index",
    (
        [2, 0],
        torch.tensor([2, 0]),
        BatchedTensor(torch.tensor([2, 0])),
    ),
)
def test_batched_tensor__getitem___list_like(index: IndexType) -> None:
    batch = BatchedTensor(torch.arange(10).view(5, 2))
    assert batch[index].equal(torch.tensor([[4, 5], [0, 1]]))


def test_batched_tensor__setitem___int() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch[0] = 7
    assert batch.equal(BatchedTensor(torch.tensor([[7, 7, 7, 7, 7], [5, 6, 7, 8, 9]])))


def test_batched_tensor__setitem___slice() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch[0:1, 2:4] = 7
    assert batch.equal(BatchedTensor(torch.tensor([[0, 1, 7, 7, 4], [5, 6, 7, 8, 9]])))


@mark.parametrize(
    "index",
    (
        [0, 2],
        torch.tensor([0, 2]),
        BatchedTensor(torch.tensor([0, 2])),
    ),
)
def test_batched_tensor__setitem___list_like_index(index: IndexType) -> None:
    batch = BatchedTensor(torch.arange(10).view(5, 2))
    batch[index] = 7
    assert batch.equal(BatchedTensor(torch.tensor([[7, 7], [2, 3], [7, 7], [6, 7], [8, 9]])))


@mark.parametrize(
    "value",
    (
        torch.tensor([[0, -4]]),
        BatchedTensor(torch.tensor([[0, -4]])),
    ),
)
def test_batched_tensor__setitem___tensor_value(value: Tensor | BatchedTensor) -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch[1:2, 2:4] = value
    assert batch.equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 0, -4, 9]])))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        torch.tensor([[10, 11, 12], [13, 14, 15]]),
    ),
)
def test_batched_tensor_append(other: BatchedTensor | Tensor) -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.append(other)
    assert batch.equal(
        BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))
    )


def test_batched_tensor_append_custom_dims() -> None:
    batch = BatchedTensor(torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1)
    batch.append(BatchedTensor(torch.tensor([[10, 12], [11, 13], [14, 15]]), batch_dim=1))
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
            batch_dim=1,
        )
    )


def test_batched_tensor_append_custom_dims_2() -> None:
    batch = BatchedTensor(torch.ones(2, 3, 4), batch_dim=2)
    batch.append(BatchedTensor(torch.ones(2, 3, 1), batch_dim=2))
    assert batch.equal(BatchedTensor(torch.ones(2, 3, 5), batch_dim=2))


def test_batched_tensor_append_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.append(BatchedTensor(torch.zeros(2, 2), batch_dim=1))


@mark.parametrize(
    "tensors",
    (
        BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        torch.tensor([[10, 11, 12], [13, 14, 15]]),
        [BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]]))],
        (BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),),
    ),
)
def test_batched_tensor_cat_dim_0(
    tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
) -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
        .cat(tensors, dim=0)
        .equal(BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]])))
    )


@mark.parametrize(
    "tensors",
    (
        BatchedTensor(torch.tensor([[10, 11], [12, 13]])),
        torch.tensor([[10, 11], [12, 13]]),
        [BatchedTensor(torch.tensor([[10, 11], [12, 13]]))],
        (BatchedTensor(torch.tensor([[10, 11], [12, 13]])),),
    ),
)
def test_batched_tensor_cat_dim_1(
    tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
) -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
        .cat(tensors, dim=1)
        .equal(BatchedTensor(torch.tensor([[0, 1, 2, 10, 11], [4, 5, 6, 12, 13]])))
    )


def test_batched_tensor_cat_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1)
        .cat(BatchedTensor(torch.tensor([[10, 12], [11, 13], [14, 15]]), batch_dim=1), dim=1)
        .equal(
            BatchedTensor(
                torch.tensor([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_cat_empty() -> None:
    assert BatchedTensor(torch.ones(2, 3)).cat([]).equal(BatchedTensor(torch.ones(2, 3)))


def test_batched_tensor_cat_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat([BatchedTensor(torch.zeros(2, 2), batch_dim=1)])


@mark.parametrize(
    "tensors",
    (
        BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        torch.tensor([[10, 11, 12], [13, 14, 15]]),
        [BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]]))],
        (BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),),
    ),
)
def test_batched_tensor_cat__dim_0(
    tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
) -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.cat_(tensors, dim=0)
    assert batch.equal(
        BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))
    )


@mark.parametrize(
    "tensors",
    (
        BatchedTensor(torch.tensor([[10, 11], [12, 13]])),
        torch.tensor([[10, 11], [12, 13]]),
        [BatchedTensor(torch.tensor([[10, 11], [12, 13]]))],
        (BatchedTensor(torch.tensor([[10, 11], [12, 13]])),),
    ),
)
def test_batched_tensor_cat__dim_1(
    tensors: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
) -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.cat_(tensors, dim=1)
    assert batch.equal(BatchedTensor(torch.tensor([[0, 1, 2, 10, 11], [4, 5, 6, 12, 13]])))


def test_batched_tensor_cat__custom_dims() -> None:
    batch = BatchedTensor(torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1)
    batch.cat_(BatchedTensor(torch.tensor([[10, 12], [11, 13], [14, 15]]), batch_dim=1), dim=1)
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
            batch_dim=1,
        )
    )


def test_batched_tensor_cat__empty() -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.cat_([])
    assert batch.equal(BatchedTensor(torch.ones(2, 3)))


def test_batched_tensor_cat__incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_([BatchedTensor(torch.zeros(2, 2), batch_dim=1)])


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        torch.tensor([[10, 11, 12], [13, 14, 15]]),
        [BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]]))],
        (BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),),
    ),
)
def test_batched_tensor_cat_along_batch(
    other: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
) -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
        .cat_along_batch(other)
        .equal(BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]])))
    )


def test_batched_tensor_cat_along_batch_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1)
        .cat_along_batch(BatchedTensor(torch.tensor([[10, 12], [11, 13], [14, 15]]), batch_dim=1))
        .equal(
            BatchedTensor(
                torch.tensor([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_cat_along_batch_custom_dims_2() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3, 4), batch_dim=2)
        .cat_along_batch(BatchedTensor(torch.ones(2, 3, 1), batch_dim=2))
        .equal(BatchedTensor(torch.ones(2, 3, 5), batch_dim=2))
    )


def test_batched_tensor_cat_along_batch_multiple() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
        .cat_along_batch(
            [
                BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
                BatchedTensor(torch.tensor([[20, 21, 22]])),
                torch.tensor([[30, 31, 32]]),
            ]
        )
        .equal(
            BatchedTensor(
                torch.tensor(
                    [[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15], [20, 21, 22], [30, 31, 32]]
                )
            )
        )
    )


def test_batched_tensor_cat_along_batch_empty() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3)).cat_along_batch([]).equal(BatchedTensor(torch.ones(2, 3)))
    )


def test_batched_tensor_cat_along_batch_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_along_batch([BatchedTensor(torch.zeros(2, 2), batch_dim=1)])


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        torch.tensor([[10, 11, 12], [13, 14, 15]]),
        [BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]]))],
        (BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),),
    ),
)
def test_batched_tensor_cat_along_batch_(
    other: BatchedTensor | Tensor | Iterable[BatchedTensor | Tensor],
) -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.cat_along_batch_(other)
    assert batch.equal(
        BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))
    )


def test_batched_tensor_cat_along_batch__custom_dims() -> None:
    batch = BatchedTensor(torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1)
    batch.cat_along_batch_(BatchedTensor(torch.tensor([[10, 12], [11, 13], [14, 15]]), batch_dim=1))
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
            batch_dim=1,
        )
    )


def test_batched_tensor_cat_along_batch__custom_dims_2() -> None:
    batch = BatchedTensor(torch.ones(2, 3, 4), batch_dim=2)
    batch.cat_along_batch_(BatchedTensor(torch.ones(2, 3, 1), batch_dim=2))
    assert batch.equal(BatchedTensor(torch.ones(2, 3, 5), batch_dim=2))


def test_batched_tensor_cat_along_batch__multiple() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.cat_along_batch_(
        [
            BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
            BatchedTensor(torch.tensor([[20, 21, 22]])),
            torch.tensor([[30, 31, 32]]),
        ]
    )
    assert batch.equal(
        BatchedTensor(
            torch.tensor(
                [[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15], [20, 21, 22], [30, 31, 32]]
            )
        )
    )


def test_batched_tensor_cat_along_batch__empty() -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.cat_along_batch_([])
    assert batch.equal(BatchedTensor(torch.ones(2, 3)))


def test_batched_tensor_cat_along_batch__incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_along_batch_([BatchedTensor(torch.zeros(2, 2), batch_dim=1)])


def test_batched_tensor_chunk_3() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(5, 2)).chunk(3),
        (
            BatchedTensor(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensor(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_chunk_5() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(5, 2)).chunk(5),
        (
            BatchedTensor(torch.tensor([[0, 1]])),
            BatchedTensor(torch.tensor([[2, 3]])),
            BatchedTensor(torch.tensor([[4, 5]])),
            BatchedTensor(torch.tensor([[6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_chunk_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).chunk(3, dim=1),
        (
            BatchedTensor(torch.tensor([[0, 1], [5, 6]]), batch_dim=1),
            BatchedTensor(torch.tensor([[2, 3], [7, 8]]), batch_dim=1),
            BatchedTensor(torch.tensor([[4], [9]]), batch_dim=1),
        ),
    )


def test_batched_tensor_chunk_along_batch_5() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(5, 2)).chunk_along_batch(5),
        (
            BatchedTensor(torch.tensor([[0, 1]])),
            BatchedTensor(torch.tensor([[2, 3]])),
            BatchedTensor(torch.tensor([[4, 5]])),
            BatchedTensor(torch.tensor([[6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_chunk_along_batch_3() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(5, 2)).chunk_along_batch(3),
        (
            BatchedTensor(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensor(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_chunk_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).chunk_along_batch(3),
        (
            BatchedTensor(torch.tensor([[0, 1], [5, 6]]), batch_dim=1),
            BatchedTensor(torch.tensor([[2, 3], [7, 8]]), batch_dim=1),
            BatchedTensor(torch.tensor([[4], [9]]), batch_dim=1),
        ),
    )


def test_batched_tensor_chunk_along_batch_incorrect_chunks() -> None:
    with raises(RuntimeError, match="chunk expects `chunks` to be greater than 0, got: 0"):
        BatchedTensor(torch.arange(10).view(5, 2)).chunk_along_batch(0)


@mark.parametrize(
    "other",
    (
        [BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]]))],
        [
            BatchedTensor(torch.tensor([[10, 11, 12]])),
            BatchedTensor(torch.tensor([[13, 14, 15]])),
        ],
        (BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),),
    ),
)
def test_batched_tensor_extend(other: Iterable[BatchedTensor | Tensor]) -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.extend(other)
    assert batch.equal(
        BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]]))
    )


def test_batched_tensor_extend_custom_dims() -> None:
    batch = BatchedTensor(torch.tensor([[0, 4], [1, 5], [2, 6]]), batch_dim=1)
    batch.extend(BatchedTensor(torch.tensor([[10, 12], [11, 13], [14, 15]]), batch_dim=1))
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[0, 4, 10, 12], [1, 5, 11, 13], [2, 6, 14, 15]]),
            batch_dim=1,
        )
    )


def test_batched_tensor_extend_custom_dims_2() -> None:
    batch = BatchedTensor(torch.ones(2, 3, 4), batch_dim=2)
    batch.extend(BatchedTensor(torch.ones(2, 3, 1), batch_dim=2))
    assert batch.equal(BatchedTensor(torch.ones(2, 3, 5), batch_dim=2))


def test_batched_tensor_extend_multiple() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.extend(
        [
            BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
            BatchedTensor(torch.tensor([[20, 21, 22]])),
            torch.tensor([[30, 31, 32]]),
        ]
    )
    assert batch.equal(
        BatchedTensor(
            torch.tensor(
                [[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15], [20, 21, 22], [30, 31, 32]]
            )
        )
    )


def test_batched_tensor_extend_empty() -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.extend([])
    assert batch.equal(BatchedTensor(torch.ones(2, 3)))


def test_batched_tensor_extend_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.extend([BatchedTensor(torch.zeros(2, 2), batch_dim=1)])


@mark.parametrize("index", (torch.tensor([2, 0]), [2, 0], (2, 0)))
def test_batched_tensor_index_select_along_batch(index: Tensor | Sequence[int]) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .index_select_along_batch(index)
        .equal(BatchedTensor(torch.tensor([[4, 5], [0, 1]])))
    )


def test_batched_tensor_index_select_along_batch_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .index_select_along_batch((2, 0))
        .equal(BatchedTensor(torch.tensor([[2, 0], [7, 5]]), batch_dim=1))
    )


@mark.parametrize(
    "mask",
    (
        BatchedTensor(
            torch.tensor([[True, False, True, False, True], [False, False, False, False, False]])
        ),
        torch.tensor([[True, False, True, False, True], [False, False, False, False, False]]),
    ),
)
def test_batched_tensor_masked_fill(mask: BatchedTensor | Tensor) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .masked_fill(mask, -1)
        .equal(BatchedTensor(torch.tensor([[-1, 1, -1, 3, -1], [5, 6, 7, 8, 9]])))
    )


def test_batched_tensor_masked_fill_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2), batch_dim=1)
        .masked_fill(
            BatchedTensor(
                torch.tensor(
                    [[True, False], [False, True], [True, False], [False, True], [True, False]]
                ),
                batch_dim=1,
            ),
            -1,
        )
        .equal(
            BatchedTensor(torch.tensor([[-1, 1], [2, -1], [-1, 5], [6, -1], [-1, 9]]), batch_dim=1)
        )
    )


def test_batched_tensor_masked_fill_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.masked_fill(BatchedTensor(torch.zeros(2, 2), batch_dim=1), 0)


def test_batched_tensor_select_dim_0() -> None:
    assert (
        BatchedTensor(torch.arange(30).view(5, 2, 3))
        .select(dim=0, index=2)
        .equal(torch.tensor([[12, 13, 14], [15, 16, 17]]))
    )


def test_batched_tensor_select_dim_1() -> None:
    assert (
        BatchedTensor(torch.arange(30).view(5, 2, 3))
        .select(dim=1, index=0)
        .equal(torch.tensor([[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20], [24, 25, 26]]))
    )


def test_batched_tensor_select_dim_2() -> None:
    assert (
        BatchedTensor(torch.arange(30).view(5, 2, 3))
        .select(dim=2, index=1)
        .equal(torch.tensor([[1, 4], [7, 10], [13, 16], [19, 22], [25, 28]]))
    )


def test_batched_tensor_select_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(30).view(5, 2, 3), batch_dim=1)
        .select(dim=0, index=2)
        .equal(torch.tensor([[12, 13, 14], [15, 16, 17]]))
    )


def test_batched_tensor_select_along_batch() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 9], [1, 8], [2, 7], [3, 6], [4, 5]]))
        .select_along_batch(2)
        .equal(torch.tensor([2, 7]))
    )


def test_batched_tensor_select_along_batch_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .select_along_batch(2)
        .equal(torch.tensor([2, 7]))
    )


def test_batched_tensor_slice_along_batch() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .slice_along_batch()
        .equal(BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])))
    )


def test_batched_tensor_slice_along_batch_start_2() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .slice_along_batch(start=2)
        .equal(BatchedTensor(torch.tensor([[4, 5], [6, 7], [8, 9]])))
    )


def test_batched_tensor_slice_along_batch_stop_3() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .slice_along_batch(stop=3)
        .equal(BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5]])))
    )


def test_batched_tensor_slice_along_batch_stop_100() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .slice_along_batch(stop=100)
        .equal(BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])))
    )


def test_batched_tensor_slice_along_batch_step_2() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .slice_along_batch(step=2)
        .equal(BatchedTensor(torch.tensor([[0, 1], [4, 5], [8, 9]])))
    )


def test_batched_tensor_slice_along_batch_start_1_stop_4_step_2() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .slice_along_batch(start=1, stop=4, step=2)
        .equal(BatchedTensor(torch.tensor([[2, 3], [6, 7]])))
    )


def test_batched_tensor_slice_along_batch_custom_dim() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .slice_along_batch(start=2)
        .equal(BatchedTensor(torch.tensor([[2, 3, 4], [7, 8, 9]]), batch_dim=1))
    )


def test_batched_tensor_slice_along_batch_batch_dim_1() -> None:
    assert (
        BatchedTensor(torch.arange(20).view(2, 5, 2), batch_dim=1)
        .slice_along_batch(start=2)
        .equal(
            BatchedTensor(
                torch.tensor([[[4, 5], [6, 7], [8, 9]], [[14, 15], [16, 17], [18, 19]]]),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_slice_along_batch_batch_dim_2() -> None:
    assert (
        BatchedTensor(torch.arange(20).view(2, 2, 5), batch_dim=2)
        .slice_along_batch(start=2)
        .equal(
            BatchedTensor(
                torch.tensor([[[2, 3, 4], [7, 8, 9]], [[12, 13, 14], [17, 18, 19]]]), batch_dim=2
            )
        )
    )


def test_batched_tensor_slice_along_dim() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .slice_along_dim()
        .equal(BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])))
    )


def test_batched_tensor_slice_along_dim_start_2() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .slice_along_dim(start=2)
        .equal(BatchedTensor(torch.tensor([[4, 5], [6, 7], [8, 9]])))
    )


def test_batched_tensor_slice_along_dim_stop_3() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .slice_along_dim(stop=3)
        .equal(BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5]])))
    )


def test_batched_tensor_slice_along_dim_stop_100() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .slice_along_dim(stop=100)
        .equal(BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])))
    )


def test_batched_tensor_slice_along_dim_step_2() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .slice_along_dim(step=2)
        .equal(BatchedTensor(torch.tensor([[0, 1], [4, 5], [8, 9]])))
    )


def test_batched_tensor_slice_along_dim_start_1_stop_4_step_2() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .slice_along_dim(start=1, stop=4, step=2)
        .equal(BatchedTensor(torch.tensor([[2, 3], [6, 7]])))
    )


def test_batched_tensor_slice_along_dim_batch_dim_1() -> None:
    assert (
        BatchedTensor(torch.arange(20).view(2, 5, 2), batch_dim=1)
        .slice_along_dim(start=2, dim=1)
        .equal(
            BatchedTensor(
                torch.tensor([[[4, 5], [6, 7], [8, 9]], [[14, 15], [16, 17], [18, 19]]]),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_slice_along_dim_batch_dim_2() -> None:
    assert (
        BatchedTensor(torch.arange(20).view(2, 2, 5), batch_dim=2)
        .slice_along_dim(start=2, dim=2)
        .equal(
            BatchedTensor(
                torch.tensor([[[2, 3, 4], [7, 8, 9]], [[12, 13, 14], [17, 18, 19]]]), batch_dim=2
            )
        )
    )


def test_batched_tensor_split_split_size_1() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(5, 2)).split(1),
        (
            BatchedTensor(torch.tensor([[0, 1]])),
            BatchedTensor(torch.tensor([[2, 3]])),
            BatchedTensor(torch.tensor([[4, 5]])),
            BatchedTensor(torch.tensor([[6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_split_split_size_2() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(5, 2)).split(2),
        (
            BatchedTensor(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensor(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_split_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).split(2, dim=1),
        (
            BatchedTensor(torch.tensor([[0, 1], [5, 6]]), batch_dim=1),
            BatchedTensor(torch.tensor([[2, 3], [7, 8]]), batch_dim=1),
            BatchedTensor(torch.tensor([[4], [9]]), batch_dim=1),
        ),
    )


def test_batched_tensor_split_split_list() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(5, 2)).split([2, 2, 1]),
        (
            BatchedTensor(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensor(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_split_along_batch_split_size_1() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(5, 2)).split_along_batch(1),
        (
            BatchedTensor(torch.tensor([[0, 1]])),
            BatchedTensor(torch.tensor([[2, 3]])),
            BatchedTensor(torch.tensor([[4, 5]])),
            BatchedTensor(torch.tensor([[6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_split_along_batch_split_size_2() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(5, 2)).split_along_batch(2),
        (
            BatchedTensor(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensor(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ),
    )


def test_batched_tensor_split_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).split_along_batch(2),
        (
            BatchedTensor(torch.tensor([[0, 1], [5, 6]]), batch_dim=1),
            BatchedTensor(torch.tensor([[2, 3], [7, 8]]), batch_dim=1),
            BatchedTensor(torch.tensor([[4], [9]]), batch_dim=1),
        ),
    )


def test_batched_tensor_split_along_batch_split_list() -> None:
    assert objects_are_equal(
        BatchedTensor(torch.arange(10).view(5, 2)).split_along_batch([2, 2, 1]),
        (
            BatchedTensor(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensor(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ),
    )


@mark.parametrize(
    "indices",
    (
        BatchedTensor(torch.tensor([[3, 2], [0, 3], [1, 4]])),
        BatchedTensor(torch.tensor([[3, 2], [0, 3], [1, 4]], dtype=torch.float)),
        torch.tensor([[3, 2], [0, 3], [1, 4]]),
        BatchList([[3, 2], [0, 3], [1, 4]]),
        [[3, 2], [0, 3], [1, 4]],
        np.array([[3, 2], [0, 3], [1, 4]]),
    ),
)
def test_batched_tensor_take_along_batch(indices: BaseBatch | Tensor | Sequence) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .take_along_batch(indices)
        .equal(BatchedTensor(torch.tensor([[6, 5], [0, 7], [2, 9]])))
    )


def test_batched_tensor_take_along_batch_empty_indices() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .take_along_batch(BatchedTensor(torch.ones(0, 2, dtype=torch.long)))
        .equal(BatchedTensor(torch.ones(0, 2, dtype=torch.long)))
    )


def test_batched_tensor_take_along_batch_custom_dim() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .take_along_batch(BatchedTensor(torch.tensor([[3, 0, 1], [2, 3, 4]]), batch_dim=1))
        .equal(BatchedTensor(torch.tensor([[3, 0, 1], [7, 8, 9]]), batch_dim=1))
    )


def test_batched_tensor_take_along_batch_extra_dim_first() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(1, 5, 2), batch_dim=1)
        .take_along_batch(BatchedTensor(torch.tensor([[[3, 2], [0, 3], [1, 4]]]), batch_dim=1))
        .equal(BatchedTensor(torch.tensor([[[6, 5], [0, 7], [2, 9]]]), batch_dim=1))
    )


def test_batched_tensor_take_along_batch_extra_dim_end() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2, 1))
        .take_along_batch(BatchedTensor(torch.tensor([[[3], [2]], [[0], [3]], [[1], [4]]])))
        .equal(BatchedTensor(torch.tensor([[[6], [5]], [[0], [7]], [[2], [9]]])))
    )


def test_batched_tensor_take_along_batch_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.take_along_batch(BatchedTensor(torch.zeros(2, 2), batch_dim=1))


@mark.parametrize(
    "indices",
    (
        BatchedTensor(torch.tensor([[3, 2], [0, 3], [1, 4]])),
        BatchedTensor(torch.tensor([[3, 2], [0, 3], [1, 4]], dtype=torch.float)),
        torch.tensor([[3, 2], [0, 3], [1, 4]]),
        BatchList([[3, 2], [0, 3], [1, 4]]),
        [[3, 2], [0, 3], [1, 4]],
        np.array([[3, 2], [0, 3], [1, 4]]),
    ),
)
def test_batched_tensor_take_along_dim(indices: BaseBatch | Tensor | Sequence) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .take_along_dim(indices)
        .equal(torch.tensor([3, 2, 0, 3, 1, 4]))
    )


@mark.parametrize(
    "indices",
    (
        BatchedTensor(torch.tensor([[3, 2], [0, 3], [1, 4]])),
        BatchedTensor(torch.tensor([[3, 2], [0, 3], [1, 4]], dtype=torch.float)),
        torch.tensor([[3, 2], [0, 3], [1, 4]]),
        BatchList([[3, 2], [0, 3], [1, 4]]),
        [[3, 2], [0, 3], [1, 4]],
        np.array([[3, 2], [0, 3], [1, 4]]),
    ),
)
def test_batched_tensor_take_along_dim_0(indices: BaseBatch | Tensor | Sequence) -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .take_along_dim(indices, dim=0)
        .equal(BatchedTensor(torch.tensor([[6, 5], [0, 7], [2, 9]])))
    )


def test_batched_tensor_take_along_dim_empty_indices() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2))
        .take_along_dim(BatchedTensor(torch.ones(0, 2, dtype=torch.long)), dim=0)
        .equal(BatchedTensor(torch.ones(0, 2, dtype=torch.long)))
    )


def test_batched_tensor_take_along_dim_custom_dim() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .take_along_dim(BatchedTensor(torch.tensor([[3, 0, 1], [2, 3, 4]]), batch_dim=1), dim=1)
        .equal(BatchedTensor(torch.tensor([[3, 0, 1], [7, 8, 9]]), batch_dim=1))
    )


def test_batched_tensor_take_along_dim_extra_dim_first() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(1, 5, 2), batch_dim=1)
        .take_along_dim(BatchedTensor(torch.tensor([[[3, 2], [0, 3], [1, 4]]]), batch_dim=1), dim=1)
        .equal(BatchedTensor(torch.tensor([[[6, 5], [0, 7], [2, 9]]]), batch_dim=1))
    )


def test_batched_tensor_take_along_dim_extra_dim_end() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(5, 2, 1))
        .take_along_dim(BatchedTensor(torch.tensor([[[3], [2]], [[0], [3]], [[1], [4]]])), dim=0)
        .equal(BatchedTensor(torch.tensor([[[6], [5]], [[0], [7]], [[2], [9]]])))
    )


def test_batched_tensor_take_along_dim_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.take_along_dim(BatchedTensor(torch.zeros(2, 2), batch_dim=1), dim=0)


def test_batched_tensor_unsqueeze_dim_0() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3))
        .unsqueeze(dim=0)
        .equal(BatchedTensor(torch.ones(1, 2, 3), batch_dim=1))
    )


def test_batched_tensor_unsqueeze_dim_1() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3)).unsqueeze(dim=1).equal(BatchedTensor(torch.ones(2, 1, 3)))
    )


def test_batched_tensor_unsqueeze_dim_2() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3)).unsqueeze(dim=2).equal(BatchedTensor(torch.ones(2, 3, 1)))
    )


def test_batched_tensor_unsqueeze_dim_last() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3)).unsqueeze(dim=-1).equal(BatchedTensor(torch.ones(2, 3, 1)))
    )


def test_batched_tensor_unsqueeze_custom_dims() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .unsqueeze(dim=0)
        .equal(BatchedTensor(torch.ones(1, 2, 3), batch_dim=2))
    )


def test_batched_tensor_view() -> None:
    assert BatchedTensor(torch.ones(2, 6)).view(2, 3, 2).equal(torch.ones(2, 3, 2))


def test_batched_tensor_view_first() -> None:
    assert BatchedTensor(torch.ones(2, 6)).view(1, 2, 6).equal(torch.ones(1, 2, 6))


def test_batched_tensor_view_last() -> None:
    assert BatchedTensor(torch.ones(2, 6)).view(2, 6, 1).equal(torch.ones(2, 6, 1))


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.zeros(2, 3, 1)),
        BatchedTensor(torch.zeros(2, 3, 1)),
        torch.zeros(2, 3, 1),
    ),
)
def test_batched_tensor_view_as(other: BatchedTensor | Tensor) -> None:
    assert BatchedTensor(torch.ones(2, 3)).view_as(other).equal(BatchedTensor(torch.ones(2, 3, 1)))


def test_batched_tensor_view_as_custom_dims() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .view_as(BatchedTensor(torch.zeros(2, 3, 1), batch_dim=1))
        .equal(BatchedTensor(torch.ones(2, 3, 1), batch_dim=1))
    )


def test_batched_tensor_view_as_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 2))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.view_as(BatchedTensor(torch.zeros(2, 2), batch_dim=1))


########################
#     mini-batches     #
########################


@mark.parametrize(("batch_size", "num_minibatches"), ((1, 10), (2, 5), (3, 4), (4, 3)))
def test_batched_tensor_get_num_minibatches_drop_last_false(
    batch_size: int, num_minibatches: int
) -> None:
    assert BatchedTensor(torch.ones(10, 2)).get_num_minibatches(batch_size) == num_minibatches


@mark.parametrize(("batch_size", "num_minibatches"), ((1, 10), (2, 5), (3, 3), (4, 2)))
def test_batched_tensor_get_num_minibatches_drop_last_true(
    batch_size: int, num_minibatches: int
) -> None:
    assert (
        BatchedTensor(torch.ones(10, 2)).get_num_minibatches(batch_size, drop_last=True)
        == num_minibatches
    )


def test_batched_tensor_to_minibatches_10_batch_size_2() -> None:
    assert objects_are_equal(
        list(BatchedTensor(torch.arange(20).view(10, 2)).to_minibatches(batch_size=2)),
        [
            BatchedTensor(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensor(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensor(torch.tensor([[8, 9], [10, 11]])),
            BatchedTensor(torch.tensor([[12, 13], [14, 15]])),
            BatchedTensor(torch.tensor([[16, 17], [18, 19]])),
        ],
    )


def test_batched_tensor_to_minibatches_10_batch_size_3() -> None:
    assert objects_are_equal(
        list(BatchedTensor(torch.arange(20).view(10, 2)).to_minibatches(batch_size=3)),
        [
            BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5]])),
            BatchedTensor(torch.tensor([[6, 7], [8, 9], [10, 11]])),
            BatchedTensor(torch.tensor([[12, 13], [14, 15], [16, 17]])),
            BatchedTensor(torch.tensor([[18, 19]])),
        ],
    )


def test_batched_tensor_to_minibatches_10_batch_size_4() -> None:
    assert objects_are_equal(
        list(BatchedTensor(torch.arange(20).view(10, 2)).to_minibatches(batch_size=4)),
        [
            BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])),
            BatchedTensor(torch.tensor([[8, 9], [10, 11], [12, 13], [14, 15]])),
            BatchedTensor(torch.tensor([[16, 17], [18, 19]])),
        ],
    )


def test_batched_tensor_to_minibatches_drop_last_true_10_batch_size_2() -> None:
    assert objects_are_equal(
        list(
            BatchedTensor(torch.arange(20).view(10, 2)).to_minibatches(batch_size=2, drop_last=True)
        ),
        [
            BatchedTensor(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensor(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensor(torch.tensor([[8, 9], [10, 11]])),
            BatchedTensor(torch.tensor([[12, 13], [14, 15]])),
            BatchedTensor(torch.tensor([[16, 17], [18, 19]])),
        ],
    )


def test_batched_tensor_to_minibatches_drop_last_true_10_batch_size_3() -> None:
    assert objects_are_equal(
        list(
            BatchedTensor(torch.arange(20).view(10, 2)).to_minibatches(batch_size=3, drop_last=True)
        ),
        [
            BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5]])),
            BatchedTensor(torch.tensor([[6, 7], [8, 9], [10, 11]])),
            BatchedTensor(torch.tensor([[12, 13], [14, 15], [16, 17]])),
        ],
    )


def test_batched_tensor_to_minibatches_drop_last_true_10_batch_size_4() -> None:
    assert objects_are_equal(
        list(
            BatchedTensor(torch.arange(20).view(10, 2)).to_minibatches(batch_size=4, drop_last=True)
        ),
        [
            BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])),
            BatchedTensor(torch.tensor([[8, 9], [10, 11], [12, 13], [14, 15]])),
        ],
    )


def test_batched_tensor_to_minibatches_custom_dims() -> None:
    assert objects_are_equal(
        list(BatchedTensor(torch.arange(20).view(2, 10), batch_dim=1).to_minibatches(batch_size=3)),
        [
            BatchedTensor(torch.tensor([[0, 1, 2], [10, 11, 12]]), batch_dim=1),
            BatchedTensor(torch.tensor([[3, 4, 5], [13, 14, 15]]), batch_dim=1),
            BatchedTensor(torch.tensor([[6, 7, 8], [16, 17, 18]]), batch_dim=1),
            BatchedTensor(torch.tensor([[9], [19]]), batch_dim=1),
        ],
    )


def test_batched_tensor_to_minibatches_deepcopy_true() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]))
    for item in batch.to_minibatches(batch_size=2, deepcopy=True):
        item.data[0, 0] = 42
    assert batch.equal(BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])))


def test_batched_tensor_to_minibatches_deepcopy_false() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]))
    for item in batch.to_minibatches(batch_size=2):
        item.data[0, 0] = 42
    assert batch.equal(BatchedTensor(torch.tensor([[42, 1], [2, 3], [42, 5], [6, 7], [42, 9]])))


#################
#     Other     #
#################


def test_batched_tensor_apply() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .apply(lambda tensor: tensor + 2)
        .equal(BatchedTensor(torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])))
    )


def test_batched_tensor_apply_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .apply(lambda tensor: tensor + 2)
        .equal(BatchedTensor(torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]]), batch_dim=1))
    )


def test_batched_tensor_apply_() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch.apply_(lambda tensor: tensor + 2)
    assert batch.equal(BatchedTensor(torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])))


def test_batched_tensor_apply__custom_dims() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
    batch.apply_(lambda tensor: tensor + 2)
    assert batch.equal(
        BatchedTensor(torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]]), batch_dim=1)
    )


def test_batched_tensor_summary() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5)).summary()
        == "BatchedTensor(dtype=torch.int64, shape=torch.Size([2, 5]), device=cpu, batch_dim=0)"
    )


################################
#     Tests for torch.amax     #
################################


def test_torch_amax_dim_0() -> None:
    assert torch.amax(BatchedTensor(torch.arange(10).view(5, 2)), dim=0).equal(torch.tensor([8, 9]))


def test_torch_amax_dim_1() -> None:
    assert objects_are_equal(
        torch.amax(BatchedTensor(torch.arange(10).view(2, 5)), dim=1),
        torch.tensor([4, 9]),
    )


def test_torch_amax_dim_1_keepdim() -> None:
    assert objects_are_equal(
        torch.amax(BatchedTensor(torch.arange(10).view(2, 5)), dim=1, keepdim=True),
        torch.tensor([[4], [9]]),
    )


def test_torch_amax_custom_dims() -> None:
    assert objects_are_equal(
        torch.amax(BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1), dim=1),
        torch.tensor([4, 9]),
    )


################################
#     Tests for torch.amin     #
################################


def test_torch_amin_dim_0() -> None:
    assert torch.amin(BatchedTensor(torch.arange(10).view(5, 2)), dim=0).equal(torch.tensor([0, 1]))


def test_torch_amin_dim_1() -> None:
    assert objects_are_equal(
        torch.amin(BatchedTensor(torch.arange(10).view(2, 5)), dim=1),
        torch.tensor([0, 5]),
    )


def test_torch_amin_dim_1_keepdim() -> None:
    assert objects_are_equal(
        torch.amin(BatchedTensor(torch.arange(10).view(2, 5)), dim=1, keepdim=True),
        torch.tensor([[0], [5]]),
    )


def test_torch_amin_custom_dims() -> None:
    assert objects_are_equal(
        torch.amin(BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1), dim=1),
        torch.tensor([0, 5]),
    )


###############################
#     Tests for torch.cat     #
###############################


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        torch.tensor([[10, 11, 12], [13, 14, 15]]),
    ),
)
def test_torch_cat_dim_0(other: BatchedTensor | Tensor) -> None:
    assert torch.cat(
        tensors=[BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]])), other],
        dim=0,
    ).equal(
        BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6], [10, 11, 12], [13, 14, 15]])),
    )


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.tensor([[4, 5], [14, 15]])),
        torch.tensor([[4, 5], [14, 15]]),
    ),
)
def test_torch_cat_dim_1(other: BatchedTensor | Tensor) -> None:
    assert torch.cat(
        tensors=[BatchedTensor(torch.tensor([[0, 1, 2], [10, 11, 12]])), other],
        dim=1,
    ).equal(
        BatchedTensor(torch.tensor([[0, 1, 2, 4, 5], [10, 11, 12, 14, 15]])),
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
            BatchedTensor(torch.ones(2, 3), batch_dim=1),
            BatchedTensor(torch.ones(2, 3), batch_dim=1),
        ]
    ).equal(BatchedTensor(torch.ones(4, 3), batch_dim=1))


def test_torch_cat_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        torch.cat(
            [
                BatchedTensor(torch.ones(2, 2)),
                BatchedTensor(torch.zeros(2, 2), batch_dim=1),
            ]
        )


#################################
#     Tests for torch.chunk     #
#################################


def test_torch_chunk_3() -> None:
    assert objects_are_equal(
        torch.chunk(BatchedTensor(torch.arange(10).view(5, 2)), chunks=3),
        (
            BatchedTensor(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensor(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ),
    )


def test_torch_chunk_5() -> None:
    assert objects_are_equal(
        torch.chunk(BatchedTensor(torch.arange(10).view(5, 2)), chunks=5),
        (
            BatchedTensor(torch.tensor([[0, 1]])),
            BatchedTensor(torch.tensor([[2, 3]])),
            BatchedTensor(torch.tensor([[4, 5]])),
            BatchedTensor(torch.tensor([[6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ),
    )


def test_torch_chunk_custom_dims() -> None:
    assert objects_are_equal(
        torch.chunk(BatchedTensor(torch.arange(10).view(5, 2), batch_dim=1), chunks=3),
        (
            BatchedTensor(torch.tensor([[0, 1], [2, 3]]), batch_dim=1),
            BatchedTensor(torch.tensor([[4, 5], [6, 7]]), batch_dim=1),
            BatchedTensor(torch.tensor([[8, 9]]), batch_dim=1),
        ),
    )


def test_torch_chunk_dim_1() -> None:
    assert objects_are_equal(
        torch.chunk(BatchedTensor(torch.arange(10).view(2, 5)), chunks=3, dim=1),
        (
            BatchedTensor(torch.tensor([[0, 1], [5, 6]])),
            BatchedTensor(torch.tensor([[2, 3], [7, 8]])),
            BatchedTensor(torch.tensor([[4], [9]])),
        ),
    )


################################
#     Tests for torch.max     #
################################


def test_torch_max() -> None:
    assert torch.max(BatchedTensor(torch.arange(10).view(2, 5))).equal(torch.tensor(9))


def test_torch_max_dim_1() -> None:
    assert objects_are_equal(
        tuple(torch.max(BatchedTensor(torch.arange(10).view(2, 5)), dim=1)),
        (torch.tensor([4, 9]), torch.tensor([4, 4])),
    )


def test_torch_max_dim_1_keepdim() -> None:
    assert objects_are_equal(
        tuple(torch.max(BatchedTensor(torch.arange(10).view(2, 5)), dim=1, keepdim=True)),
        (torch.tensor([[4], [9]]), torch.tensor([[4], [4]])),
    )


###################################
#     Tests for torch.maximum     #
###################################


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        torch.tensor([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_torch_maximum_other(other: BatchedTensor | Tensor) -> None:
    assert torch.maximum(BatchedTensor(torch.tensor([[0, 1, 2], [-2, -1, 0]])), other).equal(
        BatchedTensor(torch.tensor([[2, 1, 2], [0, 1, 0]]))
    )


def test_torch_maximum_custom_dims() -> None:
    assert torch.maximum(
        BatchedTensor(torch.tensor([[0, 1, 2], [-2, -1, 0]]), batch_dim=1),
        BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]]), batch_dim=1),
    ).equal(BatchedTensor(torch.tensor([[2, 1, 2], [0, 1, 0]]), batch_dim=1))


def test_torch_maximum_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        torch.maximum(
            BatchedTensor(torch.ones(2, 2)),
            BatchedTensor(torch.ones(2, 2), batch_dim=1),
        )


################################
#     Tests for torch.mean     #
################################


def test_torch_mean() -> None:
    assert torch.mean(BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5))).equal(
        torch.tensor(4.5)
    )


def test_torch_mean_dim_1() -> None:
    assert torch.mean(BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5)), dim=1).equal(
        torch.tensor([2.0, 7.0])
    )


def test_torch_mean_keepdim() -> None:
    assert torch.mean(
        BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5)), dim=1, keepdim=True
    ).equal(
        torch.tensor([[2.0], [7.0]]),
    )


##################################
#     Tests for torch.median     #
##################################


def test_torch_median() -> None:
    assert torch.median(BatchedTensor(torch.arange(10).view(2, 5))).equal(torch.tensor(4))


def test_torch_median_dim_1() -> None:
    assert objects_are_equal(
        torch.median(BatchedTensor(torch.arange(10).view(2, 5)), dim=1),
        torch.return_types.median([torch.tensor([2, 7]), torch.tensor([2, 2])]),
    )


def test_torch_median_keepdim() -> None:
    assert objects_are_equal(
        torch.median(BatchedTensor(torch.arange(10).view(2, 5)), dim=1, keepdim=True),
        torch.return_types.median([torch.tensor([[2], [7]]), torch.tensor([[2], [2]])]),
    )


###############################
#     Tests for torch.min     #
###############################


def test_torch_min() -> None:
    assert torch.min(BatchedTensor(torch.arange(10).view(2, 5))).equal(torch.tensor(0))


def test_torch_min_dim_1() -> None:
    assert objects_are_equal(
        tuple(torch.min(BatchedTensor(torch.arange(10).view(2, 5)), dim=1)),
        (torch.tensor([0, 5]), torch.tensor([0, 0])),
    )


def test_torch_min_dim_1_keepdim() -> None:
    assert objects_are_equal(
        tuple(torch.min(BatchedTensor(torch.arange(10).view(2, 5)), dim=1, keepdim=True)),
        (torch.tensor([[0], [5]]), torch.tensor([[0], [0]])),
    )


###################################
#     Tests for torch.minimum     #
###################################


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        torch.tensor([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_torch_minimum(other: BatchedTensor | Tensor) -> None:
    assert torch.minimum(BatchedTensor(torch.tensor([[0, 1, 2], [-2, -1, 0]])), other).equal(
        BatchedTensor(torch.tensor([[0, 0, 1], [-2, -1, 0]]))
    )


def test_torch_minimum_custom_dims() -> None:
    assert torch.minimum(
        BatchedTensor(torch.tensor([[0, 1, 2], [-2, -1, 0]]), batch_dim=1),
        BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]]), batch_dim=1),
    ).equal(BatchedTensor(torch.tensor([[0, 0, 1], [-2, -1, 0]]), batch_dim=1))


def test_torch_minimum_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        torch.minimum(
            BatchedTensor(torch.ones(2, 2)),
            BatchedTensor(torch.ones(2, 2), batch_dim=1),
        )


###################################
#     Tests for torch.nanmean     #
###################################


def test_torch_nanmean() -> None:
    assert torch.nanmean(
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
    ).equal(torch.tensor(4.0))


def test_torch_nanmean_dim_1() -> None:
    assert torch.nanmean(
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])), dim=1
    ).equal(torch.tensor([2.0, 6.5]))


def test_torch_nanmean_keepdim() -> None:
    assert torch.nanmean(
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])),
        dim=1,
        keepdim=True,
    ).equal(torch.tensor([[2.0], [6.5]]))


#####################################
#     Tests for torch.nanmedian     #
#####################################


def test_torch_nanmedian() -> None:
    assert torch.nanmedian(
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
    ).equal(torch.tensor(4.0))


def test_torch_nanmedian_dim_1() -> None:
    assert objects_are_equal(
        torch.nanmedian(
            BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])), dim=1
        ),
        torch.return_types.nanmedian([torch.tensor([2.0, 6.0]), torch.tensor([2, 1])]),
    )


def test_torch_nanmedian_keepdim() -> None:
    assert objects_are_equal(
        torch.nanmedian(
            BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])),
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
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]]))
    ).equal(torch.tensor(36.0))


def test_torch_nansum_dim_1() -> None:
    assert torch.nansum(
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])), dim=1
    ).equal(torch.tensor([10.0, 26.0]))


def test_torch_nansum_keepdim() -> None:
    assert torch.nansum(
        BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, float("nan")]])),
        dim=1,
        keepdim=True,
    ).equal(torch.tensor([[10.0], [26.0]]))


################################
#     Tests for torch.prod     #
################################


def test_torch_prod() -> None:
    assert torch.prod(BatchedTensor(torch.arange(10, dtype=torch.float).view(2, 5))).equal(
        torch.tensor(0.0)
    )


def test_torch_prod_dim_1() -> None:
    assert torch.prod(BatchedTensor(torch.arange(10).view(2, 5)), dim=1).equal(
        torch.tensor([0, 15120])
    )


def test_torch_prod_keepdim() -> None:
    assert torch.prod(BatchedTensor(torch.arange(10).view(2, 5)), dim=1, keepdim=True).equal(
        torch.tensor([[0], [15120]])
    )


##################################
#     Tests for torch.select     #
##################################


def test_torch_select_dim_0() -> None:
    assert torch.select(BatchedTensor(torch.arange(30).view(5, 2, 3)), dim=0, index=2).equal(
        torch.tensor([[12, 13, 14], [15, 16, 17]])
    )


def test_torch_select_dim_1() -> None:
    assert torch.select(BatchedTensor(torch.arange(30).view(5, 2, 3)), dim=1, index=0).equal(
        torch.tensor([[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20], [24, 25, 26]])
    )


def test_torch_select_dim_2() -> None:
    assert torch.select(BatchedTensor(torch.arange(30).view(5, 2, 3)), dim=2, index=1).equal(
        torch.tensor([[1, 4], [7, 10], [13, 16], [19, 22], [25, 28]])
    )


################################
#     Tests for torch.sort     #
################################


def test_torch_sort_descending_false() -> None:
    assert objects_are_equal(
        torch.sort(
            BatchedTensor(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])),
        ),
        torch.return_types.sort(
            [
                BatchedTensor(torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])),
                BatchedTensor(torch.tensor([[1, 2, 4, 0, 3], [2, 3, 1, 4, 0]])),
            ]
        ),
    )


def test_torch_sort_descending_true() -> None:
    assert objects_are_equal(
        torch.sort(
            BatchedTensor(torch.tensor([[4, 1, 2, 5, 3], [9, 7, 5, 6, 8]])), descending=True
        ),
        torch.return_types.sort(
            [
                BatchedTensor(torch.tensor([[5, 4, 3, 2, 1], [9, 8, 7, 6, 5]])),
                BatchedTensor(torch.tensor([[3, 0, 4, 2, 1], [0, 4, 1, 3, 2]])),
            ]
        ),
    )


def test_torch_sort_dim_0() -> None:
    assert objects_are_equal(
        torch.sort(
            BatchedTensor(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]])),
            dim=0,
        ),
        torch.return_types.sort(
            [
                BatchedTensor(torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]])),
                BatchedTensor(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]])),
            ]
        ),
    )


def test_torch_sort_dim_1() -> None:
    assert objects_are_equal(
        torch.sort(
            BatchedTensor(
                torch.tensor(
                    [
                        [[0, 1], [-2, 3], [-4, 5], [-6, 7], [-8, 9]],
                        [[10, -11], [12, -13], [14, -15], [16, -17], [18, -19]],
                    ]
                ),
            ),
            dim=1,
        ),
        torch.return_types.sort(
            [
                BatchedTensor(
                    torch.tensor(
                        [
                            [[-8, 1], [-6, 3], [-4, 5], [-2, 7], [0, 9]],
                            [[10, -19], [12, -17], [14, -15], [16, -13], [18, -11]],
                        ]
                    )
                ),
                BatchedTensor(
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
    assert objects_are_equal(
        torch.sort(
            BatchedTensor(torch.tensor([[4, 9], [1, 7], [2, 5], [5, 6], [3, 8]]), batch_dim=1),
            dim=0,
        ),
        torch.return_types.sort(
            [
                BatchedTensor(torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9]]), batch_dim=1),
                BatchedTensor(torch.tensor([[1, 2], [2, 3], [4, 1], [0, 4], [3, 0]]), batch_dim=1),
            ]
        ),
    )


#################################
#     Tests for torch.split     #
#################################


def test_torch_split_size_1() -> None:
    assert objects_are_equal(
        torch.split(BatchedTensor(torch.arange(10).view(5, 2)), 1),
        (
            BatchedTensor(torch.tensor([[0, 1]])),
            BatchedTensor(torch.tensor([[2, 3]])),
            BatchedTensor(torch.tensor([[4, 5]])),
            BatchedTensor(torch.tensor([[6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ),
    )


def test_torch_split_size_2() -> None:
    assert objects_are_equal(
        torch.split(BatchedTensor(torch.arange(10).view(5, 2)), 2),
        (
            BatchedTensor(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensor(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ),
    )


def test_torch_split_size_list() -> None:
    assert objects_are_equal(
        torch.split(BatchedTensor(torch.arange(10).view(5, 2)), [2, 2, 1]),
        (
            BatchedTensor(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensor(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ),
    )


def test_torch_split_custom_dims() -> None:
    assert objects_are_equal(
        torch.split(BatchedTensor(torch.arange(10).view(5, 2), batch_dim=1), 2),
        (
            BatchedTensor(torch.tensor([[0, 1], [2, 3]]), batch_dim=1),
            BatchedTensor(torch.tensor([[4, 5], [6, 7]]), batch_dim=1),
            BatchedTensor(torch.tensor([[8, 9]]), batch_dim=1),
        ),
    )


def test_torch_split_dim_1() -> None:
    assert objects_are_equal(
        torch.split(BatchedTensor(torch.arange(10).view(2, 5)), 2, dim=1),
        (
            BatchedTensor(torch.tensor([[0, 1], [5, 6]])),
            BatchedTensor(torch.tensor([[2, 3], [7, 8]])),
            BatchedTensor(torch.tensor([[4], [9]])),
        ),
    )


###############################
#     Tests for torch.sum     #
###############################


def test_torch_sum() -> None:
    assert torch.sum(BatchedTensor(torch.arange(10).view(2, 5))).equal(torch.tensor(45))


def test_torch_sum_dim_1() -> None:
    assert torch.sum(BatchedTensor(torch.arange(10).view(2, 5)), dim=1).equal(
        torch.tensor([10, 35])
    )


def test_torch_sum_keepdim() -> None:
    assert torch.sum(BatchedTensor(torch.arange(10).view(2, 5)), dim=1, keepdim=True).equal(
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
    ),
)
def test_torch_take_along_dim(indices: BatchedTensor | Tensor) -> None:
    assert torch.take_along_dim(BatchedTensor(torch.arange(10).view(2, 5)), indices=indices).equal(
        torch.tensor([2, 4, 1, 3])
    )


@mark.parametrize(
    "indices",
    (
        torch.tensor([[2, 4], [1, 3]]),
        BatchedTensor(torch.tensor([[2, 4], [1, 3]]), batch_dim=1),
    ),
)
def test_torch_take_along_dim_custom_dims(indices: BatchedTensor | Tensor) -> None:
    assert torch.take_along_dim(
        BatchedTensor(torch.arange(10).view(5, 2), batch_dim=1),
        indices=indices,
    ).equal(torch.tensor([2, 4, 1, 3]))


@mark.parametrize(
    "indices",
    (
        torch.tensor([[2, 4], [1, 3]]),
        BatchedTensor(torch.tensor([[2, 4], [1, 3]])),
    ),
)
def test_torch_take_along_dim_0(indices: BatchedTensor | Tensor) -> None:
    assert torch.take_along_dim(
        BatchedTensor(torch.arange(10).view(5, 2)), indices=indices, dim=0
    ).equal(BatchedTensor(torch.tensor([[4, 9], [2, 7]])))


@mark.parametrize(
    "indices",
    (
        torch.tensor([[2, 4], [1, 3]]),
        BatchedTensor(torch.tensor([[2, 4], [1, 3]]), batch_dim=1),
    ),
)
def test_torch_take_along_dim_0_custom_dims(indices: BatchedTensor | Tensor) -> None:
    assert torch.take_along_dim(
        BatchedTensor(torch.arange(10).view(5, 2), batch_dim=1),
        indices=indices,
        dim=0,
    ).equal(BatchedTensor(torch.tensor([[4, 9], [2, 7]]), batch_dim=1))


def test_torch_take_along_dim_tensor() -> None:
    assert torch.take_along_dim(
        torch.arange(10).view(5, 2), indices=BatchedTensor(torch.tensor([[2, 4], [1, 3]])), dim=0
    ).equal(BatchedTensor(torch.tensor([[4, 9], [2, 7]])))


def test_torch_take_along_dim_tensor2() -> None:
    assert torch.take_along_dim(
        torch.arange(10).view(5, 2), indices=torch.tensor([[2, 4], [1, 3]]), dim=0
    ).equal(torch.tensor([[4, 9], [2, 7]]))


def test_torch_take_along_dim_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        torch.take_along_dim(
            BatchedTensor(torch.ones(2, 2)),
            BatchedTensor(torch.zeros(2, 2), batch_dim=1),
        )
