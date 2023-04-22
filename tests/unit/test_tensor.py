import math
from collections.abc import Iterable, Sequence
from typing import Any, Union
from unittest.mock import patch

import numpy as np
import torch
from coola import objects_are_equal
from pytest import mark, raises
from torch import Tensor
from torch.overrides import is_tensor_like

from redcat import BaseBatch, BaseBatchedTensor, BatchedTensor, BatchedTensorSeq
from redcat.tensor import check_data_and_dim
from redcat.utils import get_available_devices, get_torch_generator

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
def test_batched_tensor__eq__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
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
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor__ge__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
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
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor__gt__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
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
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor__le__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
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
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor__lt__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
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
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_gt(other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]) -> None:
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
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensorSeq(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_le(other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]) -> None:
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
        BatchedTensorSeq(torch.full((2, 5), 5.0)),
        torch.full((2, 5), 5.0),
        BatchedTensor(torch.ones(2, 1).mul(5)),
        5,
        5.0,
    ),
)
def test_batched_tensor_lt(other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]) -> None:
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
        BatchedTensorSeq(torch.ones(2, 3)),
        BatchedTensor(torch.ones(2, 3)),
        torch.ones(2, 3),
        1,
        1.0,
    ),
)
def test_batched_tensor__add__(
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
def test_batched_tensor__iadd__(
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
def test_batched_tensor__mul__(
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
def test_batched_tensor__imul__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch *= other
    assert batch.equal(BatchedTensorSeq(torch.full((2, 3), 2.0)))


def test_batched_tensor__neg__() -> None:
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
def test_batched_tensor__sub__(
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
def test_batched_tensor__isub__(
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
def test_batched_tensor__truediv__(
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
def test_batched_tensor__itruediv__(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    batch = BatchedTensorSeq(torch.ones(2, 3))
    batch /= other
    assert batch.equal(BatchedTensorSeq(torch.ones(2, 3).mul(0.5)))


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
def test_batched_tensor_add_(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.add_(other)
    assert batch.equal(BatchedTensor(torch.full((2, 3), 3.0)))


def test_batched_tensor_add__alpha_2_float() -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.add_(BatchedTensor(torch.full((2, 3), 2.0)), alpha=2.0)
    assert batch.equal(BatchedTensor(torch.full((2, 3), 5.0)))


def test_batched_tensor_add__alpha_2_long() -> None:
    batch = BatchedTensor(torch.ones(2, 3, dtype=torch.long))
    batch.add_(BatchedTensor(torch.ones(2, 3, dtype=torch.long).mul(2)), alpha=2)
    assert batch.equal(BatchedTensor(torch.ones(2, 3, dtype=torch.long).mul(5)))


def test_batched_tensor_add__custom_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 3), batch_dim=1)
    batch.add_(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
    assert batch.equal(BatchedTensor(torch.full((2, 3), 3.0), batch_dim=1))


def test_batched_tensor_add__incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.add_(BatchedTensor(torch.ones(2, 3), batch_dim=1))


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
def test_batched_tensor_div_(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
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
    batch = BatchedTensor(torch.ones(2, 3))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.div_(BatchedTensor(torch.ones(2, 3), batch_dim=1))


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
def test_batched_tensor_fmod(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    assert BatchedTensor(torch.ones(2, 3)).fmod(other).equal(BatchedTensor(torch.ones(2, 3)))


def test_batched_tensor_fmod_custom_dims() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .fmod(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
        .equal(BatchedTensor(torch.ones(2, 3), batch_dim=1))
    )


def test_batched_tensor_fmod_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        BatchedTensor(torch.ones(2, 3)).fmod(BatchedTensor(torch.ones(2, 3), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.full((2, 3), 2.0)),
        BatchedTensor(torch.full((2, 3), 2.0)),
        torch.full((2, 3), 2.0),
        2,
        2.0,
    ),
)
def test_batched_tensor_fmod_(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.fmod_(other)
    assert batch.equal(BatchedTensor(torch.ones(2, 3)))


def test_batched_tensor_fmod__custom_dims() -> None:
    batch = BatchedTensor(torch.ones(2, 3), batch_dim=1)
    batch.fmod_(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
    assert batch.equal(BatchedTensor(torch.ones(2, 3), batch_dim=1))


def test_batched_tensor_fmod__incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.fmod_(BatchedTensor(torch.ones(2, 3), batch_dim=1))


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
def test_batched_tensor_mul_(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
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
        .equal(BatchedTensor(-torch.full((2, 3), 3.0)))
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
def test_batched_tensor_sub_(
    other: Union[BaseBatchedTensor, torch.Tensor, bool, int, float]
) -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.sub_(other)
    assert batch.equal(BatchedTensor(-torch.ones(2, 3)))


def test_batched_tensor_sub__alpha_2_float() -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    batch.sub_(BatchedTensor(torch.full((2, 3), 2.0)), alpha=2.0)
    assert batch.equal(BatchedTensor(-torch.full((2, 3), 3.0)))


def test_batched_tensor_sub__alpha_2_long() -> None:
    batch = BatchedTensor(torch.ones(2, 3, dtype=torch.long))
    batch.sub_(BatchedTensor(torch.ones(2, 3, dtype=torch.long).mul(2)), alpha=2)
    assert batch.equal(BatchedTensor(-torch.ones(2, 3, dtype=torch.long).mul(3)))


def test_batched_tensor_sub__custom_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 3), batch_dim=1)
    batch.sub_(BatchedTensor(torch.full((2, 3), 2.0), batch_dim=1))
    assert batch.equal(BatchedTensor(-torch.ones(2, 3), batch_dim=1))


def test_batched_tensor_sub__incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.sub_(BatchedTensor(torch.ones(2, 3, 1), batch_dim=2))


###########################################################
#     Mathematical | advanced arithmetical operations     #
###########################################################


def test_batched_tensor_cumsum() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .cumsum(dim=0)
        .equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 7, 9, 11, 13]])))
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


@mark.parametrize("permutation", (torch.tensor([2, 1, 3, 0]), [2, 1, 3, 0], (2, 1, 3, 0)))
def test_batched_tensor_permute_along_batch(
    permutation: Union[Sequence[int], torch.Tensor]
) -> None:
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
def test_batched_tensor_permute_along_batch_(
    permutation: Union[Sequence[int], torch.Tensor]
) -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]))
    batch.permute_along_batch_(permutation)
    assert batch.equal(BatchedTensor(torch.tensor([[6, 7, 8], [3, 4, 5], [9, 10, 11], [0, 1, 2]])))


def test_batched_tensor_permute_along_batch__custom_dims() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), batch_dim=1)
    batch.permute_along_batch_(torch.tensor([2, 1, 3, 0]))
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
        .clamp(min_value=2, max_value=5)
        .equal(BatchedTensor(torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]])))
    )


def test_batched_tensor_clamp_only_max_value() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .clamp(max_value=5)
        .equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 5, 5, 5, 5]])))
    )


def test_batched_tensor_clamp_only_min_value() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5))
        .clamp(min_value=2)
        .equal(BatchedTensor(torch.tensor([[2, 2, 2, 3, 4], [5, 6, 7, 8, 9]])))
    )


def test_batched_tensor_clamp_custom_dims() -> None:
    assert (
        BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
        .clamp(min_value=2, max_value=5)
        .equal(BatchedTensor(torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]]), batch_dim=1))
    )


def test_batched_tensor_clamp_() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch.clamp_(min_value=2, max_value=5)
    assert batch.equal(BatchedTensor(torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]])))


def test_batched_tensor_clamp__only_max_value() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch.clamp_(max_value=5)
    assert batch.equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 5, 5, 5, 5]])))


def test_batched_tensor_clamp__only_min_value() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch.clamp_(min_value=2)
    assert batch.equal(BatchedTensor(torch.tensor([[2, 2, 2, 3, 4], [5, 6, 7, 8, 9]])))


def test_batched_tensor_clamp__custom_dims() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1)
    batch.clamp_(min_value=2, max_value=5)
    assert batch.equal(BatchedTensor(torch.tensor([[2, 2, 2, 3, 4], [5, 5, 5, 5, 5]]), batch_dim=1))


def test_batched_tensor_exp() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
        .exp()
        .equal(
            BatchedTensor(
                torch.tensor(
                    [
                        [2.7182817459106445, 7.389056205749512, 20.08553695678711],
                        [54.598148345947266, 148.4131622314453, 403.4288024902344],
                    ]
                )
            )
        )
    )


def test_batched_tensor_exp_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float), batch_dim=1)
        .exp()
        .equal(
            BatchedTensor(
                torch.tensor(
                    [
                        [2.7182817459106445, 7.389056205749512, 20.08553695678711],
                        [54.598148345947266, 148.4131622314453, 403.4288024902344],
                    ]
                ),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_exp_() -> None:
    batch = BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float))
    batch.exp_()
    assert batch.equal(
        BatchedTensor(
            torch.tensor(
                [
                    [2.7182817459106445, 7.389056205749512, 20.08553695678711],
                    [54.598148345947266, 148.4131622314453, 403.4288024902344],
                ]
            )
        )
    )


def test_batched_tensor_exp__custom_dims() -> None:
    batch = BatchedTensor(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float), batch_dim=1)
    batch.exp_()
    assert batch.equal(
        BatchedTensor(
            torch.tensor(
                [
                    [2.7182817459106445, 7.389056205749512, 20.08553695678711],
                    [54.598148345947266, 148.4131622314453, 403.4288024902344],
                ]
            ),
            batch_dim=1,
        )
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


@mark.parametrize(
    "data,max_value",
    (
        (torch.tensor([[False, True, True], [True, False, True]], dtype=torch.bool), True),  # bool
        (torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long), 5),  # long
        (torch.tensor([[4.0, 1.0, 7.0], [3.0, 2.0, 5.0]], dtype=torch.float), 7.0),  # float
    ),
)
def test_batched_tensor_max_global(data: torch.Tensor, max_value: Union[bool, int, float]) -> None:
    assert BatchedTensor(data).max() == max_value


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        BatchedTensorSeq(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        torch.tensor([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_batched_tensor_max(other: BaseBatchedTensor | Tensor) -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [-2, -1, 0]]))
        .max(other)
        .equal(BatchedTensor(torch.tensor([[2, 1, 2], [0, 1, 0]])))
    )


def test_batched_tensor_max_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [-2, -1, 0]]), batch_dim=1)
        .max(BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]]), batch_dim=1))
        .equal(BatchedTensor(torch.tensor([[2, 1, 2], [0, 1, 0]]), batch_dim=1))
    )


def test_batched_tensor_max_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.max(BatchedTensor(torch.ones(2, 3, 1), batch_dim=2))


@mark.parametrize(
    "data,min_value",
    (
        (torch.tensor([[False, True, True], [True, False, True]], dtype=torch.bool), False),
        # bool
        (torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long), 0),  # long
        (torch.tensor([[4.0, 1.0, 7.0], [3.0, 2.0, 5.0]], dtype=torch.float), 1.0),  # float
    ),
)
def test_batched_tensor_min_global(data: torch.Tensor, min_value: Union[bool, int, float]) -> None:
    assert BatchedTensor(data).min() == min_value


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]])),
        torch.tensor([[2, 0, 1], [0, 1, 0]]),
    ),
)
def test_batched_tensor_min(other: BaseBatchedTensor | Tensor) -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [-2, -1, 0]]))
        .min(other)
        .equal(BatchedTensor(torch.tensor([[0, 0, 1], [-2, -1, 0]])))
    )


def test_batched_tensor_min_custom_dims() -> None:
    assert (
        BatchedTensor(torch.tensor([[0, 1, 2], [-2, -1, 0]]), batch_dim=1)
        .min(BatchedTensor(torch.tensor([[2, 0, 1], [0, 1, 0]]), batch_dim=1))
        .equal(BatchedTensor(torch.tensor([[0, 0, 1], [-2, -1, 0]]), batch_dim=1))
    )


def test_batched_tensor_min_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 3, 1))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.min(BatchedTensor(torch.ones(2, 3, 1), batch_dim=2))


@mark.parametrize(
    "exponent",
    (BatchedTensor(torch.full((2, 5), 2.0)), BatchedTensorSeq(torch.full((2, 5), 2.0)), 2, 2.0),
)
def test_batched_tensor_pow(exponent: Union[BaseBatchedTensor, int, float]) -> None:
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
        .pow(BatchedTensor(torch.ones(2, 3).mul(2), batch_dim=1))
        .equal(
            BatchedTensor(
                torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float),
                batch_dim=1,
            )
        )
    )


def test_batched_tensor_pow_incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        BatchedTensor(torch.ones(2, 3, 1)).pow(BatchedTensor(torch.ones(2, 3, 1), batch_dim=2))


@mark.parametrize(
    "exponent",
    (BatchedTensorSeq(torch.full((2, 5), 2.0)), BatchedTensor(torch.full((2, 5), 2.0)), 2, 2.0),
)
def test_batched_tensor_pow_(exponent: Union[BaseBatchedTensor, int, float]) -> None:
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
    batch.pow_(BatchedTensor(torch.ones(2, 3).mul(2), batch_dim=1))
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]], dtype=torch.float),
            batch_dim=1,
        )
    )


def test_batched_tensor_pow__incorrect_batch_dim() -> None:
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        BatchedTensor(torch.ones(2, 3, 1)).pow_(BatchedTensor(torch.ones(2, 3, 1), batch_dim=2))


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
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        BatchedTensorSeq(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_logical_and(other: BaseBatchedTensor | Tensor, dtype: torch.dtype) -> None:
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
    batch = BatchedTensor(torch.zeros(2, 3, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_and(
            BatchedTensor(
                torch.zeros(2, 3, dtype=torch.bool),
                batch_dim=1,
            )
        )


@mark.parametrize(
    "other",
    (
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        BatchedTensorSeq(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_logical_and_(other: BaseBatchedTensor | Tensor, dtype: torch.dtype) -> None:
    batch = BatchedTensor(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
    )
    batch.logical_and_(other)
    assert batch.equal(
        BatchedTensor(
            torch.tensor(
                [[True, False, False, False], [True, False, True, False]], dtype=torch.bool
            )
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
    batch = BatchedTensor(torch.zeros(2, 3, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_and_(
            BatchedTensor(
                torch.zeros(2, 3, dtype=torch.bool),
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
                torch.tensor([[False, False, True, True], [False, True, False, True]], dtype=dtype)
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
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        BatchedTensorSeq(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_logical_or(other: BaseBatchedTensor | Tensor, dtype: torch.dtype) -> None:
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
    batch = BatchedTensor(torch.zeros(2, 3, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_or(
            BatchedTensor(
                torch.zeros(2, 3, dtype=torch.bool),
                batch_dim=1,
            )
        )


@mark.parametrize(
    "other",
    (
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        BatchedTensorSeq(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_logical_or_(other: BaseBatchedTensor | Tensor, dtype: torch.dtype) -> None:
    batch = BatchedTensor(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
    )
    batch.logical_or_(other)
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[True, True, True, False], [True, True, True, True]], dtype=torch.bool)
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
    batch = BatchedTensor(torch.zeros(2, 3, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_or_(
            BatchedTensor(
                torch.zeros(2, 3, dtype=torch.bool),
                batch_dim=1,
            )
        )


@mark.parametrize(
    "other",
    (
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        BatchedTensorSeq(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_logical_xor(other: BaseBatchedTensor | Tensor, dtype: torch.dtype) -> None:
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
    batch = BatchedTensor(torch.zeros(2, 3, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_xor(
            BatchedTensor(
                torch.zeros(2, 3, dtype=torch.bool),
                batch_dim=1,
            )
        )


@mark.parametrize(
    "other",
    (
        BatchedTensor(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool)
        ),
        torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        BatchedTensor(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        BatchedTensorSeq(torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float)),
        torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float),
    ),
)
@mark.parametrize("dtype", (torch.bool, torch.float, torch.long))
def test_batched_tensor_logical_xor_(other: BaseBatchedTensor | Tensor, dtype: torch.dtype) -> None:
    batch = BatchedTensor(
        torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=dtype)
    )
    batch.logical_xor_(other)
    assert batch.equal(
        BatchedTensor(
            torch.tensor([[False, True, True, False], [False, True, False, True]], dtype=torch.bool)
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
    batch = BatchedTensor(torch.zeros(2, 3, dtype=torch.bool))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.logical_xor_(
            BatchedTensor(
                torch.zeros(2, 3, dtype=torch.bool),
                batch_dim=1,
            )
        )


##########################################################
#    Indexing, slicing, joining, mutating operations     #
##########################################################


def test_batched_tensor__getitem___int() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    assert batch[0].equal(torch.tensor([0, 1, 2, 3, 4]))


def test_batched_tensor__range___range() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    assert batch[0:2, 2:4].equal(torch.tensor([[2, 3], [7, 8]]))


@mark.parametrize(
    "index",
    (
        torch.tensor([[1, 3], [0, 4]]),
        BatchedTensor(torch.tensor([[1, 3], [0, 4]])),
        BatchedTensor(torch.tensor([[1, 3], [0, 4]])),
    ),
)
def test_batched_tensor__getitem___tensor_like(index: Union[Tensor, BaseBatchedTensor]) -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    assert batch[0].equal(torch.tensor([0, 1, 2, 3, 4]))


def test_batched_tensor__setitem___int() -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch[0] = 7
    assert batch.equal(BatchedTensor(torch.tensor([[7, 7, 7, 7, 7], [5, 6, 7, 8, 9]])))


@mark.parametrize(
    "value",
    (
        torch.tensor([[0, -4]]),
        BatchedTensor(torch.tensor([[0, -4]])),
        BatchedTensorSeq(torch.tensor([[0, -4]])),
    ),
)
def test_batched_tensor__setitem___range(value: Union[Tensor, BaseBatchedTensor]) -> None:
    batch = BatchedTensor(torch.arange(10).view(2, 5))
    batch[1:2, 2:4] = value
    assert batch.equal(BatchedTensor(torch.tensor([[0, 1, 2, 3, 4], [5, 6, 0, -4, 9]])))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        torch.tensor([[10, 11, 12], [13, 14, 15]]),
    ),
)
def test_batched_tensor_append(other: BaseBatchedTensor | Tensor) -> None:
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


def test_batched_tensor_append_custom_dims_seq_dim_2() -> None:
    batch = BatchedTensor(torch.ones(2, 3, 4), batch_dim=2)
    batch.append(BatchedTensor(torch.ones(2, 3, 1), batch_dim=2))
    assert batch.equal(BatchedTensor(torch.ones(2, 3, 5), batch_dim=2))


def test_batched_tensor_append_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.append(BatchedTensor(torch.zeros(2, 3), batch_dim=1))


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        torch.tensor([[10, 11, 12], [13, 14, 15]]),
        [BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]]))],
        (BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),),
    ),
)
def test_batched_tensor_cat_along_batch(
    other: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor],
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


def test_batched_tensor_cat_along_batch_custom_dims_seq_dim_2() -> None:
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
                BatchedTensorSeq(torch.tensor([[20, 21, 22]])),
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
    batch = BatchedTensor(torch.ones(2, 3))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_along_batch([BatchedTensor(torch.zeros(2, 3), batch_dim=1)])


@mark.parametrize(
    "other",
    (
        BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        BatchedTensorSeq(torch.tensor([[10, 11, 12], [13, 14, 15]])),
        torch.tensor([[10, 11, 12], [13, 14, 15]]),
        [BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]]))],
        (BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),),
    ),
)
def test_batched_tensor_cat_along_batch_(
    other: BaseBatchedTensor | Tensor | Iterable[BaseBatchedTensor | Tensor],
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


def test_batched_tensor_cat_along_batch__custom_dims_seq_dim_2() -> None:
    batch = BatchedTensor(torch.ones(2, 3, 4), batch_dim=2)
    batch.cat_along_batch_(BatchedTensor(torch.ones(2, 3, 1), batch_dim=2))
    assert batch.equal(BatchedTensor(torch.ones(2, 3, 5), batch_dim=2))


def test_batched_tensor_cat_along_batch__multiple() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.cat_along_batch_(
        [
            BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
            BatchedTensorSeq(torch.tensor([[20, 21, 22]])),
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
    batch = BatchedTensor(torch.ones(2, 3))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.cat_along_batch_([BatchedTensor(torch.zeros(2, 3), batch_dim=1)])


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
def test_batched_tensor_extend(other: Iterable[BaseBatchedTensor | Tensor]) -> None:
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


def test_batched_tensor_extend_custom_dims_seq_dim_2() -> None:
    batch = BatchedTensor(torch.ones(2, 3, 4), batch_dim=2)
    batch.extend(BatchedTensor(torch.ones(2, 3, 1), batch_dim=2))
    assert batch.equal(BatchedTensor(torch.ones(2, 3, 5), batch_dim=2))


def test_batched_tensor_extend_multiple() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1, 2], [4, 5, 6]]))
    batch.extend(
        [
            BatchedTensor(torch.tensor([[10, 11, 12], [13, 14, 15]])),
            BatchedTensorSeq(torch.tensor([[20, 21, 22]])),
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
    batch = BatchedTensor(torch.ones(2, 3))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.extend([BatchedTensor(torch.zeros(2, 3), batch_dim=1)])


@mark.parametrize("index", (torch.tensor([2, 0]), [2, 0], (2, 0)))
def test_batched_tensor_index_select_along_batch(index: Union[torch.Tensor, Sequence[int]]) -> None:
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
        BatchedTensorSeq(
            torch.tensor([[True, False, True, False, True], [False, False, False, False, False]])
        ),
        BatchedTensor(
            torch.tensor([[True, False, True, False, True], [False, False, False, False, False]])
        ),
        torch.tensor([[True, False, True, False, True], [False, False, False, False, False]]),
    ),
)
def test_batched_tensor_masked_fill(mask: BaseBatchedTensor | Tensor) -> None:
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
    batch = BatchedTensor(torch.ones(2, 3))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.masked_fill(BatchedTensor(torch.zeros(2, 3), batch_dim=1), 0)


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


def test_batched_tensor_split_along_batch_split_size_1() -> None:
    assert objects_are_equal(
        list(BatchedTensor(torch.arange(10).view(5, 2)).split_along_batch(1)),
        [
            BatchedTensor(torch.tensor([[0, 1]])),
            BatchedTensor(torch.tensor([[2, 3]])),
            BatchedTensor(torch.tensor([[4, 5]])),
            BatchedTensor(torch.tensor([[6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ],
    )


def test_batched_tensor_split_along_batch_split_size_2() -> None:
    assert objects_are_equal(
        list(BatchedTensor(torch.arange(10).view(5, 2)).split_along_batch(2)),
        [
            BatchedTensor(torch.tensor([[0, 1], [2, 3]])),
            BatchedTensor(torch.tensor([[4, 5], [6, 7]])),
            BatchedTensor(torch.tensor([[8, 9]])),
        ],
    )


def test_batched_tensor_split_along_batch_custom_dims() -> None:
    assert objects_are_equal(
        list(BatchedTensor(torch.arange(10).view(2, 5), batch_dim=1).split_along_batch(2)),
        [
            BatchedTensor(torch.tensor([[0, 1], [5, 6]]), batch_dim=1),
            BatchedTensor(torch.tensor([[2, 3], [7, 8]]), batch_dim=1),
            BatchedTensor(torch.tensor([[4], [9]]), batch_dim=1),
        ],
    )


def test_batched_tensor_split_along_batch_deepcopy_true() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]))
    for item in batch.split_along_batch(split_size=1, deepcopy=True):
        item[0, 0] = 42
    assert batch.equal(BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])))


def test_batched_tensor_split_along_batch_deepcopy_false() -> None:
    batch = BatchedTensor(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]))
    for item in batch.split_along_batch(split_size=1):
        item[0, 0] = 42
    assert batch.equal(BatchedTensor(torch.tensor([[42, 1], [42, 3], [42, 5], [42, 7], [42, 9]])))


@mark.parametrize(
    "indices",
    (
        BatchedTensorSeq(torch.tensor([[3, 2], [0, 3], [1, 4]])),
        BatchedTensor(torch.tensor([[3, 2], [0, 3], [1, 4]])),
        torch.tensor([[3, 2], [0, 3], [1, 4]]),
        [[3, 2], [0, 3], [1, 4]],
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
    batch = BatchedTensor(torch.ones(2, 3))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.take_along_batch(BatchedTensor(torch.zeros(1, 3), batch_dim=1))


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


@mark.parametrize(
    "other",
    (
        BatchedTensorSeq(torch.zeros(2, 3, 1)),
        BatchedTensor(torch.zeros(2, 3, 1)),
        torch.zeros(2, 3, 1),
    ),
)
def test_batched_tensor_view_as(other: BaseBatchedTensor | Tensor) -> None:
    assert BatchedTensor(torch.ones(2, 3)).view_as(other).equal(BatchedTensor(torch.ones(2, 3, 1)))


def test_batched_tensor_view_as_custom_dims() -> None:
    assert (
        BatchedTensor(torch.ones(2, 3), batch_dim=1)
        .view_as(BatchedTensor(torch.zeros(2, 3, 1), batch_dim=1))
        .equal(BatchedTensor(torch.ones(2, 3, 1), batch_dim=1))
    )


def test_batched_tensor_view_as_incorrect_batch_dim() -> None:
    batch = BatchedTensor(torch.ones(2, 3))
    with raises(RuntimeError, match=r"The batch dimensions do not match."):
        batch.view_as(BatchedTensor(torch.zeros(2, 1), batch_dim=1))


########################
#     mini-batches     #
########################


@mark.parametrize("batch_size,num_minibatches", ((1, 10), (2, 5), (3, 4), (4, 3)))
def test_batched_tensor_get_num_minibatches_drop_last_false(
    batch_size: int, num_minibatches: int
) -> None:
    assert BatchedTensor(torch.ones(10, 2)).get_num_minibatches(batch_size) == num_minibatches


@mark.parametrize("batch_size,num_minibatches", ((1, 10), (2, 5), (3, 3), (4, 2)))
def test_batched_tensor_get_num_minibatches_drop_last_true(
    batch_size: int, num_minibatches: int
) -> None:
    assert (
        BatchedTensor(torch.ones(10, 2)).get_num_minibatches(batch_size, drop_last=True)
        == num_minibatches
    )


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
