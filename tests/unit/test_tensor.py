from typing import Union

import torch
from pytest import mark, raises
from torch.overrides import is_tensor_like

from redcat import BatchedTensor
from redcat.tensor import check_data_and_dim
from redcat.utils import get_available_devices


def test_batched_tensor_is_tensor_like() -> None:
    assert is_tensor_like(BatchedTensor(torch.ones(2, 3)))


@mark.parametrize("batch_dim", (-1, 1, 2))
def test_batched_tensor_incorrect_batch_dim(batch_dim: int) -> None:
    with raises(RuntimeError):
        BatchedTensor(torch.ones(2), batch_dim=batch_dim)


@mark.parametrize("batch_size", (1, 2))
def test_batched_tensor_batch_size(batch_size: int) -> None:
    assert BatchedTensor(torch.arange(batch_size)).batch_size == batch_size


def test_batched_tensor_data() -> None:
    assert BatchedTensor(torch.arange(3)).data.equal(torch.tensor([0, 1, 2]))


def test_batched_tensor_repr() -> None:
    assert repr(BatchedTensor(torch.arange(3))) == "tensor([0, 1, 2], batch_dim=0)"


@mark.parametrize("device", get_available_devices())
def test_batched_tensor_device(device: str)->None:
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


#################################
#     Comparison operations     #
#################################


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


###################################
#     Arithmetical operations     #
###################################


@mark.parametrize(
    "other", (BatchedTensor(torch.ones(2, 3).mul(2)), torch.ones(2, 3).mul(2), 2, 2.0)
)
def test_batched_tensor_add(other: Union[BatchedTensor, torch.Tensor, bool, int, float]) -> None:
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
    with raises(RuntimeError):
        BatchedTensor(torch.ones(2, 3)).add(BatchedTensor(torch.ones(2, 3), batch_dim=1))


########################################
#     Tests for check_data_and_dim     #
########################################


def test_check_data_and_dim_correct() -> None:
    check_data_and_dim(torch.ones(2, 3), batch_dim=0)
    # will fail if an exception is raised


def test_check_data_and_dim_incorrect_data_dim() -> None:
    with raises(RuntimeError):
        check_data_and_dim(torch.tensor(2), batch_dim=0)


@mark.parametrize("batch_dim", (-1, 2, 3))
def test_check_data_and_dim_incorrect_batch_dim(batch_dim: int) -> None:
    with raises(RuntimeError):
        check_data_and_dim(torch.ones(2, 3), batch_dim=batch_dim)
