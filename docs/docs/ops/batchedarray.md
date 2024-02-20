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

| name            | `redcat.ba`        | `BatchedArray`     | `BatchedArraySeq` |
|-----------------|--------------------|--------------------|-------------------|
| `batched_array` | :white_check_mark: |                    |                   |
| `empty`         | :white_check_mark: |                    |                   |
| `empty_like`    | :white_check_mark: | :white_check_mark: |                   |
| `full`          | :white_check_mark: |                    |                   |
| `full_like`     | :white_check_mark: | :white_check_mark: |                   |
| `ones`          | :white_check_mark: |                    |                   |
| `ones_like`     | :white_check_mark: | :white_check_mark: |                   |
| `zeros`         | :white_check_mark: |                    |                   |
| `zeros_like`    | :white_check_mark: | :white_check_mark: |                   |

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
| `slice_along_seq`          | :x:                |                    |                    |                   |
| `split_along_axis`         | :x:                |                    | :white_check_mark: |                   |
| `split_along_seq`          | :x:                |                    |                    |                   |
| **Rearranging elements**   |                    |                    |                    |                   |
| `permute_along_axis`       | :x:                |                    | :white_check_mark: |                   |
| `permute_along_axis_`      | :x:                |                    | :white_check_mark: |                   |
| `permute_along_seq`        | :x:                |                    |                    |                   |
| `permute_along_seq_`       | :x:                |                    |                    |                   |
| `shuffle_along_axis`       | :x:                |                    | :white_check_mark: |                   |
| `shuffle_along_axis_`      | :x:                |                    | :white_check_mark: |                   |
| `shuffle_along_seq`        | :x:                |                    |                    |                   |
| `shuffle_along_seq_`       | :x:                |                    |                    |                   |

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
| `mul`                           | :white_check_mark: |                    | :white_check_mark: |                   |
| `mul_`                          | :x:                | :x:                | :white_check_mark: |                   |
| `multiply`                      | :white_check_mark: | :white_check_mark: |                    |                   |
| `sub`                           | :white_check_mark: |                    | :white_check_mark: |                   |
| `sub_`                          | :x:                | :x:                | :white_check_mark: |                   |
| `substract`                     | :white_check_mark: | :white_check_mark: |                    |                   |
| `truediv`                       | :white_check_mark: |                    | :white_check_mark: |                   |
| `truediv_`                      | :x:                | :x:                | :white_check_mark: |                   |
| `true_divide`                   | :white_check_mark: | :white_check_mark: |                    |                   |
| **Sums, products, differences** |                    |                    |                    |                   |
| `cumprod`                       | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `cumprod_along_batch`           | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `cumprod_along_seq`             | :x:                |                    | :x:                |                   |
| `cumsum`                        | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `cumsum_along_batch`            | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `cumsum_along_seq`              | :x:                |                    | :x:                |                   |
| `diff`                          |                    |                    |                    |                   |
| `ediff1d`                       |                    |                    |                    |                   |
| `nancumprod`                    | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `nancumprod_along_batch`        | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `nancumprod_along_seq`          | :x:                |                    | :x:                |                   |
| `nancumsum`                     | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `nancumsum_along_batch`         | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `nancumsum_along_seq`           | :x:                |                    | :x:                |                   |
| `nanprod`                       | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `nanprod_along_batch`           | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `nanprod_along_seq`             | :x:                |                    | :x:                |                   |
| `nansum`                        | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `nansum_along_batch`            | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `nansum_along_seq`              | :x:                |                    | :x:                |                   |
| `prod`                          | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `prod_along_batch`              | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `prod_along_seq`                | :x:                |                    | :x:                |                   |
| `sum`                           | :white_check_mark: | :white_check_mark: | :white_check_mark: |                   |
| `sum_along_batch`               | :x:                | :white_check_mark: | :white_check_mark: |                   |
| `sum_along_seq`                 | :x:                |                    | :x:                |                   |
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
| `max_along_batch`               | :x:                |                    |                    |                   |
| `max_along_seq`                 | :x:                |                    |                    |                   |
| `max`                           | :white_check_mark: |                    |                    |                   |
| `maximum`                       | :white_check_mark: |                    |                    |                   |
| `min_along_batch`               | :x:                |                    |                    |                   |
| `min_along_seq`                 | :x:                |                    |                    |                   |
| `min`                           | :white_check_mark: |                    |                    |                   |
| `minimum`                       | :white_check_mark: |                    |                    |                   |
| `nanmax_along_batch`            | :x:                |                    |                    |                   |
| `nanmax_along_seq`              | :x:                |                    |                    |                   |
| `nanmax`                        | :white_check_mark: |                    |                    |                   |
| `nanmin_along_batch`            | :x:                |                    |                    |                   |
| `nanmin_along_seq`              | :x:                |                    |                    |                   |
| `nanmin`                        | :white_check_mark: |                    |                    |                   |

## Sort

| name                                 | `np` | `redcat.ba` | `BatchedArray` | `BatchedArraySeq` |
|--------------------------------------|------|-------------|----------------|-------------------|
| **Sorting, searching, and counting** |      |             |                |                   |
| `sort`                               |      |             |                |                   |
| `sort_along_batch`                   | :x:  |             |                |                   |
| **Searching**                        |      |             |                |                   |
| `argmax`                             |      |             |                |                   |
| `argmax_along_batch`                 | :x:  |             |                |                   |
| `argmin`                             |      |             |                |                   |
| `argmin_along_batch`                 | :x:  |             |                |                   |
| `nanargmax`                          |      |             |                |                   |
| `nanargmax_along_batch`              | :x:  |             |                |                   |
| `nanargmin`                          |      |             |                |                   |
| `nanargmin_along_batch`              | :x:  |             |                |                   |
