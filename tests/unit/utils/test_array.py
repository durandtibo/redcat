
import numpy as np

from redcat.utils.array import permute_along_dim

#######################################
#     Tests for permute_along_dim     #
#######################################


def test_permute_along_dim_1d() -> None:
    assert np.array_equal(
        permute_along_dim(np.arange(4), permutation=np.array([0, 2, 1, 3])), np.array([0, 2, 1, 3])
    )


def test_permute_along_dim_2d_dim_0() -> None:
    assert np.array_equal(
        permute_along_dim(np.arange(20).reshape(4, 5), permutation=np.array([0, 2, 1, 3])),
        np.array([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [15, 16, 17, 18, 19]]),
    )


def test_permute_along_dim_2d_dim_1() -> None:
    assert np.array_equal(
        permute_along_dim(np.arange(20).reshape(4, 5), permutation=np.array([0, 4, 2, 1, 3]), dim=1),
        np.array([[0, 4, 2, 1, 3], [5, 9, 7, 6, 8], [10, 14, 12, 11, 13], [15, 19, 17, 16, 18]]),
    )


def test_permute_along_dim_3d_dim_2() -> None:
    assert np.array_equal(
        permute_along_dim(
            np.arange(20).reshape(2, 2, 5), permutation=np.array([0, 4, 2, 1, 3]), dim=2
        ),
        np.array(
            [[[0, 4, 2, 1, 3], [5, 9, 7, 6, 8]], [[10, 14, 12, 11, 13], [15, 19, 17, 16, 18]]]
        ),
    )
