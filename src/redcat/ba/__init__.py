r"""Contain the implementation of ``BatchedArray`` and its associated
functions.

``BatchedArray`` is a custom NumPy array container to make batch
manipulation easier.
"""

from __future__ import annotations

__all__ = [
    "BatchedArray",
    "add",
    "argmax",
    "argmax_along_batch",
    "argmin",
    "argmin_along_batch",
    "argsort",
    "argsort_along_batch",
    "array",
    "check_data_and_axis",
    "check_same_batch_axis",
    "concatenate",
    "concatenate_along_batch",
    "cumprod",
    "cumprod_along_batch",
    "cumsum",
    "cumsum_along_batch",
    "diff",
    "diff_along_batch",
    "divide",
    "empty",
    "empty_like",
    "floor_divide",
    "full",
    "full_like",
    "get_batch_axes",
    "max",
    "max_along_batch",
    "mean",
    "mean_along_batch",
    "median",
    "median_along_batch",
    "min",
    "min_along_batch",
    "multiply",
    "nanargmax",
    "nanargmax_along_batch",
    "nanargmin",
    "nanargmin_along_batch",
    "nancumprod",
    "nancumprod_along_batch",
    "nancumsum",
    "nancumsum_along_batch",
    "nanmax",
    "nanmax_along_batch",
    "nanmean",
    "nanmean_along_batch",
    "nanmedian",
    "nanmedian_along_batch",
    "nanmin",
    "nanmin_along_batch",
    "nanprod",
    "nanprod_along_batch",
    "nansum",
    "nansum_along_batch",
    "ones",
    "ones_like",
    "prod",
    "prod_along_batch",
    "sort",
    "sort_along_batch",
    "subtract",
    "sum",
    "sum_along_batch",
    "true_divide",
    "zeros",
    "zeros_like",
]

from redcat.ba.core import BatchedArray
from redcat.ba.creation import (
    array,
    empty,
    empty_like,
    full,
    full_like,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from redcat.ba.joining import concatenate, concatenate_along_batch
from redcat.ba.math import (
    add,
    cumprod,
    cumprod_along_batch,
    cumsum,
    cumsum_along_batch,
    diff,
    diff_along_batch,
    divide,
    floor_divide,
    max,
    max_along_batch,
    min,
    min_along_batch,
    multiply,
    nancumprod,
    nancumprod_along_batch,
    nancumsum,
    nancumsum_along_batch,
    nanmax,
    nanmax_along_batch,
    nanmin,
    nanmin_along_batch,
    nanprod,
    nanprod_along_batch,
    nansum,
    nansum_along_batch,
    prod,
    prod_along_batch,
    subtract,
    sum,
    sum_along_batch,
    true_divide,
)
from redcat.ba.sort import (
    argmax,
    argmax_along_batch,
    argmin,
    argmin_along_batch,
    argsort,
    argsort_along_batch,
    nanargmax,
    nanargmax_along_batch,
    nanargmin,
    nanargmin_along_batch,
    sort,
    sort_along_batch,
)
from redcat.ba.stats import (
    mean,
    mean_along_batch,
    median,
    median_along_batch,
    nanmean,
    nanmean_along_batch,
    nanmedian,
    nanmedian_along_batch,
)
from redcat.ba.utils import check_data_and_axis, check_same_batch_axis, get_batch_axes
