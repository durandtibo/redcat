# BatchedArray

This page shows the supported operations for `BatchedArray` and `BatchedArraySeq`

## Core functionalities

| name                       | `BatchedArray`     | `BatchedArraySeq` |
|----------------------------|--------------------|-------------------|
| `batch_size`               | :white_check_mark: |                   |
| `data`                     | :white_check_mark: |                   |
| `allclose`                 | :white_check_mark: |                   |
| `allequal`                 | :white_check_mark: |                   |
| `append`                   | :white_check_mark: |                   |
| `chunk_along_batch`        | :white_check_mark: |                   |
| `clone`                    | :white_check_mark: |                   |
| `extend`                   | :white_check_mark: |                   |
| `get_num_minibatches`      | :white_check_mark: |                   |
| `index_select_along_batch` | :white_check_mark: |                   |
| `permute_along_batch`      | :white_check_mark: |                   |
| `permute_along_batch_`     | :white_check_mark: |                   |
| `select_along_batch`       | :white_check_mark: |                   |
| `shuffle_along_batch`      | :white_check_mark: |                   |
| `shuffle_along_batch_`     | :white_check_mark: |                   |
| `slice_along_batch`        | :white_check_mark: |                   |
| `split_along_batch`        | :white_check_mark: |                   |
| `summary`                  | :white_check_mark: |                   |
| `to_data`                  | :white_check_mark: |                   |
| `to_minibatches`           | :white_check_mark: |                   |

## Array creation

| name         | `redcat.ba`        | `BatchedArray`     | `BatchedArraySeq` |
|--------------|--------------------|--------------------|-------------------|
| `array`      | :white_check_mark: | :x:                |                   |
| `copy`       |                    | :white_check_mark: |                   |
| `empty`      | :white_check_mark: |                    |                   |
| `empty_like` | :white_check_mark: | :white_check_mark: |                   |
| `full`       | :white_check_mark: |                    |                   |
| `full_like`  | :white_check_mark: | :white_check_mark: |                   |
| `ones`       | :white_check_mark: |                    |                   |
| `ones_like`  | :white_check_mark: | :white_check_mark: |                   |
| `zeros`      | :white_check_mark: |                    |                   |
| `zeros_like` | :white_check_mark: | :white_check_mark: |                   |

## Logic functions

[doc](https://numpy.org/doc/stable/reference/routines.logic.html)

| name                   | `np`               | `redcat.ba` | `BatchedArray`     | `BatchedArraySeq` |
|------------------------|--------------------|-------------|--------------------|-------------------|
| **Comparison**         |                    |             |                    |                   |
| `__eq__`               | :x:                | :x:         | :white_check_mark: |                   |
| `__ge__`               | :x:                | :x:         | :white_check_mark: |                   |
| `__gt__`               | :x:                | :x:         | :white_check_mark: |                   |
| `__le__`               | :x:                | :x:         | :white_check_mark: |                   |
| `__lt__`               | :x:                | :x:         | :white_check_mark: |                   |
| `__ne__`               | :x:                | :x:         | :white_check_mark: |                   |
| `allclose`             |                    |             |                    |                   |
| `array_equal`          |                    |             |                    |                   |
| `array_equiv`          |                    |             |                    |                   |
| `equal`                | :white_check_mark: |             |                    |                   |
| `greater_equal`        | :white_check_mark: |             |                    |                   |
| `greater`              | :white_check_mark: |             |                    |                   |
| `less_equal`           | :white_check_mark: |             |                    |                   |
| `less`                 | :white_check_mark: |             |                    |                   |
| `not_equal`            | :white_check_mark: |             |                    |                   |
| **Array contents**     |                    |             |                    |                   |
| `isclose`              | :white_check_mark: |             |                    |                   |
| `isfinite`             | :white_check_mark: |             |                    |                   |
| `isinf`                | :white_check_mark: |             |                    |                   |
| `isnan`                | :white_check_mark: |             |                    |                   |
| `isnat`                | :white_check_mark: |             |                    |                   |
| `isneginf`             | :white_check_mark: |             |                    |                   |
| `isposinf`             | :white_check_mark: |             |                    |                   |
| **Logical operations** |                    |             |                    |                   |
| `logical_and`          | :white_check_mark: |             |                    |                   |
| `logical_not`          | :white_check_mark: |             |                    |                   |
| `logical_or`           | :white_check_mark: |             |                    |                   |
| `logical_xor`          | :white_check_mark: |             |                    |                   |

## Array manipulation

[doc](https://numpy.org/doc/stable/reference/routines.array-manipulation.html)

| name                       | `redcat.np`        | `redcat.ba`        | `BatchedArray`     | `BatchedArraySeq` |
|----------------------------|--------------------|--------------------|--------------------|-------------------|
| `__getitem__`              |                    |                    | :white_check_mark: |                   |
| `__setitem__`              |                    |                    | :white_check_mark: |                   |
| **Joining arrays**         |                    |                    |                    |                   |
| `concatenate`              | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `concatenate_`             | :x:                | :x:                | :white_check_mark: |                   |
| `concatenate_along_batch`  | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `concatenate_along_batch_` | :x:                | :x:                | :white_check_mark: |                   |
| `concatenate_along_seq`    | :x:                |                    | :x:                |                   |
| `concatenate_along_seq_`   | :x:                | :x:                | :x:                |                   |
| **Slicing arrays**         |                    |                    |                    |                   |
| `chunk`                    | :x:                |                    | :white_check_mark: |                   |
| `index_select`             | :x:                |                    | :white_check_mark: |                   |
| `select`                   | :x:                |                    | :white_check_mark: |                   |
| `slice_along_axis`         | :x:                |                    | :white_check_mark: |                   |
| `slice_along_seq`          | :x:                |                    | :x:                |                   |
| `split_along_axis`         | :x:                |                    | :white_check_mark: |                   |
| `split_along_seq`          | :x:                |                    | :x:                |                   |
| **Rearranging elements**   |                    |                    |                    |                   |
| `permute_along_axis`       | :x:                |                    | :white_check_mark: |                   |
| `permute_along_axis_`      | :x:                |                    | :white_check_mark: |                   |
| `permute_along_seq`        | :x:                |                    | :x:                |                   |
| `permute_along_seq_`       | :x:                |                    | :x:                |                   |
| `shuffle_along_axis`       | :x:                |                    | :white_check_mark: |                   |
| `shuffle_along_axis_`      | :x:                |                    | :white_check_mark: |                   |
| `shuffle_along_seq`        | :x:                |                    | :x:                |                   |
| `shuffle_along_seq_`       | :x:                |                    | :x:                |                   |

## Math

[doc](https://numpy.org/doc/stable/reference/routines.math.html)

| name                            | `np`               | `redcat.ba`        | `BatchedArray`     | `BatchedArraySeq` |
|---------------------------------|--------------------|--------------------|--------------------|-------------------|
| **Arithmetic**                  |                    |                    |                    |                   |
| `__add__`                       | :x:                | :x:                | :white_check_mark: |                   |
| `__iadd__`                      | :x:                | :x:                | :white_check_mark: |                   |
| `__floordiv__`                  | :x:                | :x:                | :white_check_mark: |                   |
| `__ifloordiv__`                 | :x:                | :x:                | :white_check_mark: |                   |
| `__mul__`                       | :x:                | :x:                | :white_check_mark: |                   |
| `__imul__`                      | :x:                | :x:                | :white_check_mark: |                   |
| `__neg__`                       | :x:                | :x:                | :white_check_mark: |                   |
| `__sub__`                       | :x:                | :x:                | :white_check_mark: |                   |
| `__isub__`                      | :x:                | :x:                | :white_check_mark: |                   |
| `__truediv__`                   | :x:                | :x:                | :white_check_mark: |                   |
| `__itruediv__`                  | :x:                | :x:                | :white_check_mark: |                   |
| `add`                           | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `add_`                          | :x:                | :x:                | :white_check_mark: |                   |
| `divide`                        | :white_check_mark: | :white_check_mark: |                    |                   |
| `divmod`                        | :white_check_mark: |                    | :white_check_mark: |                   |
| `divmod_`                       | :x:                | :x:                | :white_check_mark: |                   |
| `floordiv`                      | :x:                |                    | :white_check_mark: |                   |
| `floordiv_`                     | :x:                | :x:                | :white_check_mark: |                   |
| `floor_divide`                  | :white_check_mark: | :white_check_mark: |                    |                   |
| `fmod`                          | :white_check_mark: |                    | :white_check_mark: |                   |
| `fmod_`                         | :x:                | :x:                | :white_check_mark: |                   |
| `mul`                           | :x:                |                    | :white_check_mark: |                   |
| `mul_`                          | :x:                | :x:                | :white_check_mark: |                   |
| `multiply`                      | :white_check_mark: | :white_check_mark: |                    |                   |
| `sub`                           | :x:                |                    | :white_check_mark: |                   |
| `sub_`                          | :x:                | :x:                | :white_check_mark: |                   |
| `substract`                     | :white_check_mark: | :white_check_mark: |                    |                   |
| `truediv`                       | :x:                | :x:                | :white_check_mark: |                   |
| `truediv_`                      | :x:                | :x:                | :white_check_mark: |                   |
| `true_divide`                   | :white_check_mark: | :white_check_mark: |                    |                   |
| **Sums, products, differences** |                    |                    |                    |                   |
| `cumprod_along_batch`           | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `cumprod_along_seq`             | :x:                |                    | :x:                |                   |
| `cumprod`                       | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `cumsum_along_batch`            | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `cumsum_along_seq`              | :x:                |                    | :x:                |                   |
| `cumsum`                        | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `diff_along_batch`              | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `diff_along_seq`                | :x:                |                    | :x:                |                   |
| `diff`                          | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `nancumprod_along_batch`        | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `nancumprod_along_seq`          | :x:                |                    | :x:                |                   |
| `nancumprod`                    | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `nancumsum_along_batch`         | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `nancumsum_along_seq`           | :x:                |                    | :x:                |                   |
| `nancumsum`                     | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `nanprod_along_batch`           | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `nanprod_along_seq`             | :x:                |                    | :x:                |                   |
| `nanprod`                       | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `nansum_along_batch`            | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `nansum_along_seq`              | :x:                |                    | :x:                |                   |
| `nansum`                        | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `prod_along_batch`              | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `prod_along_seq`                | :x:                |                    | :x:                |                   |
| `prod`                          | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `sum_along_batch`               | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `sum_along_seq`                 | :x:                |                    | :x:                |                   |
| `sum`                           | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `trapz`                         |                    |                    |                    |                   |
| **Trigonometric functions**     |                    |                    |                    |                   |
| **Hyperbolic functions**        |                    |                    |                    |                   |
| **Exponents and logarithms**    |                    |                    |                    |                   |
| **Rounding**                    |                    |                    |                    |                   |
| **Floating point routines**     |                    |                    |                    |                   |
| **Rational routines**           |                    |                    |                    |                   |
| **Extrema Finding**             |                    |                    |                    |                   |
| `fmax`                          | :white_check_mark: |                    |                    |                   |
| `fmin`                          | :white_check_mark: |                    |                    |                   |
| `max_along_batch`               | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `max_along_seq`                 | :x:                |                    | :x:                |                   |
| `max`                           | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `maximum`                       | :white_check_mark: |                    |                    |                   |
| `min_along_batch`               | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `min_along_seq`                 | :x:                |                    | :x:                |                   |
| `min`                           | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `minimum`                       | :white_check_mark: |                    |                    |                   |
| `nanmax_along_batch`            | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `nanmax_along_seq`              | :x:                |                    | :x:                |                   |
| `nanmax`                        | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `nanmin_along_batch`            | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `nanmin_along_seq`              | :x:                |                    | :x:                |                   |
| `nanmin`                        | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |

## Sorting, searching, and counting

| name                    | `np`               | `redcat.ba`        | `BatchedArray`     | `BatchedArraySeq` |
|-------------------------|--------------------|--------------------|--------------------|-------------------|
| **Sorting**             |                    |                    |                    |                   |
| `argsort_along_batch`   | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `argsort_along_seq`     | :x:                |                    | :x:                |                   |
| `argsort`               | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `sort_along_batch`      | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `sort_along_seq`        | :x:                |                    | :x:                |                   |
| `sort`                  | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| **Searching**           |                    |                    |                    |                   |
| `argmax_along_batch`    | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `argmax_along_seq`      | :x:                |                    | :x:                |                   |
| `argmax`                | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `argmin_along_batch`    | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `argmin_along_seq`      | :x:                |                    | :x:                |                   |
| `argmin`                | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `nanargmax_along_batch` | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `nanargmax_along_seq`   | :x:                |                    | :x:                |                   |
| `nanargmax`             | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `nanargmin_along_batch` | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `nanargmin_along_seq`   | :x:                |                    | :x:                |                   |
| `nanargmin`             | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |

## Statistics

| name                      | `np`               | `redcat.ba`        | `BatchedArray`     | `BatchedArraySeq` |
|---------------------------|--------------------|--------------------|--------------------|-------------------|
| `mean_along_batch`        | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `mean_along_seq`          | :x:                |                    | :x:                |                   |
| `mean`                    | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `median_along_batch`      | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `median_along_seq`        | :x:                |                    | :x:                |                   |
| `median`                  | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `nanmean_along_batch`     | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `nanmean_along_seq`       | :x:                |                    | :x:                |                   |
| `nanmean`                 | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `nanmedian_along_batch`   | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `nanmedian_along_seq`     | :x:                |                    | :x:                |                   |
| `nanmedian`               | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `nanquantile_along_batch` | :x:                |                    |                    |                   |
| `nanquantile_along_seq`   | :x:                |                    | :x:                |                   |
| `nanquantile`             |                    |                    |                    |                   |
| `nanstd_along_batch`      | :x:                |                    |                    |                   |
| `nanstd_along_seq`        | :x:                |                    | :x:                |                   |
| `nanstd`                  |                    |                    |                    |                   |
| `nanvar_along_batch`      | :x:                |                    |                    |                   |
| `nanvar_along_seq`        | :x:                |                    | :x:                |                   |
| `nanvar`                  |                    |                    |                    |                   |
| `quantile_along_batch`    | :x:                |                    |                    |                   |
| `quantile_along_seq`      | :x:                |                    | :x:                |                   |
| `quantile`                |                    |                    |                    |                   |
| `std_along_batch`         | :x:                |                    |                    |                   |
| `std_along_seq`           | :x:                |                    | :x:                |                   |
| `std`                     |                    |                    |                    |                   |
| `var_along_batch`         | :x:                |                    |                    |                   |
| `var_along_seq`           | :x:                |                    | :x:                |                   |
| `var`                     |                    |                    |                    |                   |
