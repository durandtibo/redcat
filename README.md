# redcat

<p align="center">
    <a href="https://github.com/durandtibo/redcat/actions">
        <img alt="CI" src="https://github.com/durandtibo/redcat/workflows/CI/badge.svg">
    </a>
    <a href="https://durandtibo.github.io/redcat/">
        <img alt="Documentation" src="https://github.com/durandtibo/redcat/workflows/Documentation/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/redcat/actions">
        <img alt="Nightly Tests" src="https://github.com/durandtibo/redcat/workflows/Nightly%20Tests/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/redcat/actions">
        <img alt="Nightly Package Tests" src="https://github.com/durandtibo/redcat/workflows/Nightly%20Package%20Tests/badge.svg">
    </a>
    <br/>
    <a href="https://codecov.io/gh/durandtibo/redcat">
        <img alt="Codecov" src="https://codecov.io/gh/durandtibo/redcat/branch/main/graph/badge.svg">
    </a>
    <a href="https://codeclimate.com/github/durandtibo/redcat/maintainability">
        <img src="https://api.codeclimate.com/v1/badges/0987ab26fe4d52025085/maintainability" />
    </a>
    <a href="https://codeclimate.com/github/durandtibo/redcat/test_coverage">
        <img src="https://api.codeclimate.com/v1/badges/0987ab26fe4d52025085/test_coverage" />
    </a>
    <br/>
    <a href="https://pypi.org/project/redcat/">
        <img alt="PYPI version" src="https://img.shields.io/pypi/v/redcat">
    </a>
    <a href="https://pypi.org/project/redcat/">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/redcat.svg">
    </a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
        <img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/redcat">
    </a>
    <a href="https://github.com/psf/black">
        <img  alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/%20style-google-3666d6.svg">
    </a>
    <br/>
    <a href="https://pepy.tech/project/redcat">
        <img  alt="Downloads" src="https://static.pepy.tech/badge/redcat">
    </a>
    <a href="https://pepy.tech/project/redcat">
        <img  alt="Monthly downloads" src="https://static.pepy.tech/badge/redcat/month">
    </a>
    <br/>
</p>

<p align="center">
<img height="242" src="assets/redcat.png" alt="logo"/>
</p>



---

## Philosophy

- `BatchedTensor` and `BatchedTensorSeq` must behave like `torch.Tensor`
- `BatchedArray` and `BatchedArraySeq` must behave like `numpy.ndarray`
- `BatchedTensor` (resp. `BatchedTensorSeq`) does not have to behave like `BatchedArray` (
  resp. `BatchedArraySeq`)

## Installation

We highly recommend installing
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
`redcat` can be installed from pip using the following command:

```shell
pip install redcat
```

To make the package as slim as possible, only the minimal packages required to use `redcat` are
installed.
To include all the dependencies, you can use the following command:

```shell
pip install redcat[all]
```

Please check the [get started page](https://durandtibo.github.io/redcat/get_started) to see how to
install only some specific dependencies or other alternatives to install the library.
The following is the corresponding `redcat` versions and supported Python, PyTorch and NumPy
versions.

| `redcat` | `coola`            | `numpy`        | `torch`       | `python`      |
|----------|--------------------|----------------|---------------|---------------|
| `0.0.17` | `>=0.0.20,<0.0.25` | `>=1.22,<1.27` | `>=1.11,<2.2` | `>=3.9,<3.12` |
| `0.0.16` | `>=0.0.20,<0.0.24` | `>=1.22,<1.27` | `>=1.11,<2.1` | `>=3.9,<3.12` |
| `0.0.15` | `>=0.0.20,<0.0.24` | `>=1.22,<1.27` | `>=1.11,<2.1` | `>=3.9,<3.12` |
| `0.0.14` | `>=0.0.20,<0.0.24` | `>=1.22,<1.27` | `>=1.11,<2.1` | `>=3.9,<3.12` |
| `0.0.13` | `>=0.0.20,<0.0.24` | `>=1.22,<1.27` | `>=1.11,<2.1` | `>=3.9,<3.12` |

<details>
    <summary>older versions</summary>

| `redcat` | `coola`            | `numpy`        | `torch`       | `python`      |
|----------|--------------------|----------------|---------------|---------------|
| `0.0.12` | `>=0.0.20,<0.0.24` | `>=1.22,<1.27` | `>=1.11,<2.1` | `>=3.9,<3.12` |
| `0.0.11` | `>=0.0.20,<0.0.24` | `>=1.22,<1.27` | `>=1.11,<2.1` | `>=3.9,<3.12` |
| `0.0.10` | `>=0.0.20,<0.0.24` | `>=1.22,<1.27` | `>=1.11,<2.1` | `>=3.9,<3.12` |
| `0.0.9`  | `>=0.0.20,<0.0.24` | `>=1.22,<1.27` | `>=1.11,<2.1` | `>=3.9,<3.12` |
| `0.0.8`  | `>=0.0.20,<0.0.21` | `>=1.21,<1.26` | `>=1.11,<2.1` | `>=3.9,<3.12` |

</details>

## Contributing

Please check the instructions in [CONTRIBUTING.md](.github/CONTRIBUTING.md).

## API stability

:warning: While `redcat` is in development stage, no API is guaranteed to be stable from one
release to the next.
In fact, it is very likely that the API will change multiple times before a stable 1.0.0 release.
In practice, this means that upgrading `redcat` to a new version will possibly break any code that
was using the old version of `redcat`.

## License

`redcat` is licensed under BSD 3-Clause "New" or "Revised" license available in [LICENSE](LICENSE)
file.

---

*The logo was generated with [Fooocus](https://github.com/lllyasviel/Fooocus)*
