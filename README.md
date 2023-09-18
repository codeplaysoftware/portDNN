# The portDNN neural network acceleration library

## Table of Contents

  * [Supported Platforms](#supported-platforms)
  * [Getting Started with portDNN](#getting-started-with-portDNN)
  * [Support](#support)
  * [Cross-compilation with ComputeCpp](#cross-compilation-with-computecpp)
  * [Contributions](#contributions)
  * [Citation](#citation)

portDNN is a library implementing various neural network algorithms such as
pooling and convolution written using SYCL and C++.

portDNN currently supports the following operations:

* 2D convolutions
* 2D depthwise convolutions
* 2D max & average pooling
* Relu and tanh activations

The convolution operations have several implementations, including tiled and
Winograd kernels. The supported data format is NHWC.

The project is maintained by [Codeplay Software][codeplay developer].

## Supported Platforms

The master branch of portDNN is regularly tested with the "Supported" hardware
listed on [the ComputeCpp Supported Platforms page][supported platforms].
portDNN may also work on other hardware and platforms assuming they implement
SPIR or SPIR-V support. portDNN is primarily tested on Ubuntu 16.04 LTS with
the corresponding default package versions. portDNN will generally match the
most recently released ComputeCpp, though it is likely to be compatible with
other versions. We test against the most recent version.

## Getting Started with portDNN

### Pre-requisites

* CMake (version 3.5.1 and above)
* OpenCL 1.2-capable hardware and drivers with SPIR 1.2 or SPIR-V support
* OpenCL ICD Loader
* OpenCL headers
* gcc (version 5.4 and above)
* [ComputeCpp][codeplay developer]
* Building documentation requires Doxygen and Graphviz/Dot. Tested
  against versions 1.8.11 and 2.38.0 respectively.

### Building portDNN

portDNN uses CMake as its build system. There are provisions in the CMake
files for downloading portDNN's dependencies automatically, for finding
other dependencies and for selecting which bits of portDNN to build. All
these configuration options can be found in the main CMakeLists.txt for the
project and will show up in the CMake GUI if you use it. By default, the
tests and library will be built, but not the benchmarks.

It is recommended to leave the option `SNN_DOWNLOAD_MISSING_DEPS` set to
on. This will automatically download the source libraries necessary for
portDNN to build and run (such as Google Test, Google benchmark and
the Eigen linear algebra library). Even if you already have these on your
machine, downloading them as part of the portDNN means a more consistent
configuration.

#### Building with ComputeCpp

You will need to provide the location of the ComputeCpp install you are
using in the variable `ComputeCpp_DIR`. It should point to the folder
where `bin/`, `lib/` etc. are. This should be the only argument that is
mandatory, everything else should be optional. The default build type is
Release, though this can be overridden.

ComputeCpp with portDNN does not currently support USM. If you build with
ComputeCpp you must disable USM support.

The following command shows how to compile portDNN.

```bash
# Setup build environment
mkdir build && cd build
cmake .. -DComputeCpp_DIR=/path/to/computecpp -DSNN_ENABLE_USM=OFF
# Compile portDNN
make -j$(nproc)
```

#### Building with DPC++

You will need to provide the location of the DPC++ compiler to CMake to
build with DPC++.

DPC++ does support USM. USM support will be automatically built unless you
disable it with `-DSNN_ENABLE_USM=OFF`.

```bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=/path/to/llvm/bin/clang++ -DSNN_BUILD_BENCHMARKS=OFF -DSNN_BENCH_SYCLBLAS=OFF 
# Compile portDNN
make -j$(nproc)
```

### Undefined reference linker errors

portDNN exposes optional features (`double` and `half` data types, `NCHW` data format, USM support), 
that can be enabled and disabled when building the library.

Attempting to use those feature in an application that links to a build of portDNN that doesn't support them may 
cause `undefined reference` error at link time. Please ensure that your build of portDNN has the required features enabled.

You can refer to [OPTIONS.md](docs/OPTIONS.md) for a full list of the supported CMake options.



### Sample Code

The "samples" directory contains sample code for the 2D convolution and pooling
operations offered by portDNN. These binaries are compiled when building portDNN
using CMake.

### Running the portDNN Tests

The portDNN tests are compiled when building portDNN using CMake.
The following command shows how to run the tests.

```bash
# Run the tests
ctest
# If compiled with benchmark support, run just the benchmarks
ctest -C Benchmark -E test
```

## Support

### Bug reports and Issues

Bug reports are vital to provide feedback to the developers about what is going
wrong with the project, you can raise these using the ["Issues"][issues]
feature in GitHub.

Please make sure that your bug report contains the following information:

* A clear and descriptive title.
* The output of
  `clinfo | grep -E "Platform ID|Name|Vendor|[Vv]ersion|Profile|Extensions"`.
* The output of `computecpp_info`.
* The exact steps and commands to run to reproduce the bug.
* The exact error text shown (if applicable), otherwise the behaviour you
  expected and what you encountered instead.
* Should the problem arise outside the project's test suite then please provide
  a minimal test to allow us to reproduce the problem.

## Cross-compilation with ComputeCpp

portDNN supports cross-compilation targeting a number of devices. However,
because of the two-step compilation process used in ComputeCpp, standard
CMake toolchain files won't provide enough information to portDNN's build
scripts to work properly.

To that end, two toolchains are available. The first, gcc-generic.cmake,
will likely work with any prebuilt GCC toolchain (it is not compatible
with those installed through package managers). The second is designed to
work with the poky toolchain available as part of the Yocto Linux system.

The first step is to download ComputeCpp for both the host machine you are
running on and for the platform you would like to target. You should make
sure to match the ComputeCpp version for both downloads. Both are required
so that the host can run the compiler binary, while the tools can link
using the target device library. Similarly, acquire a GCC toolchain for
the platform you are targeting. Lastly you should download the OpenCL
headers. They are standard across all platforms, but you cannot specify
the default package-managed location of `/usr/include` for them, as that
will cause conflicts with other system headers. An easy fix is to download
the headers [from GitHub][ocl headers].

Toolchain files cannot make use cache variables set by the user when
running CMake, as the cache does not exist when the toolchain is executed.
Environment variables are available to toolchain files, however, so they
are used to pass information to the toolchain. The gcc-generic.cmake
toolchain relies on the following environment variables:

```cmake
SNN_TARGET_TRIPLE # the triple of the platform you are targeting
SNN_TOOLCHAIN_DIR # The root directory of the GCC you downloaded
SNN_SYSROOT_DIR   # The system root, probably (but not necessarily)
                  # ${SNN_TOOLCHAIN_DIR}/${SNN_TARGET_TRIPLE}/libc
```

CMake can then be invoked in a build directory as follows:

```bash
cmake -DComputeCpp_DIR=/path/to/computecpp \
      -DComputeCpp_HOST_DIR=/path/to/host/computecpp \
      -DOpenCL_INCLUDE_DIR=/path/to/opencl/headers \
      `# For cross-compiling, check documentation for your platform` \
      -DCOMPUTECPP_BITCODE=[(spir[32|64]|spirv[32|64]|ptx64)] \
      -DSNN_BUILD_DOCUMENTATION=OFF \
      `# Next options let you install the tests to a zippable folder` \
      -DSNN_BUILD_TESTS=ON \
      -DSNN_BUILD_BENCHMARKS=ON \
      -DSNN_INSTALL_TESTS=ON \
      -DSNN_INSTALL_BENCHMARKS=ON \
      `# This is the most important part - tells CMake to crosscompile` \
      -DCMAKE_TOOLCHAIN_FILE=$PWD/../cmake/toolchains/(gcc-generic|arm-gcc-poky).cmake \
      -DCMAKE_INSTALL_PREFIX=packaged-binaries \
      -GNinja ../
```

The process for the poky toolchain is similar, save that you only need to
provide the `SNN_SYSROOT_DIR` environment variable. It should be set to
point to the directory named `sysroots` in the poky toolchain. You will
likely want `COMPUTECPP_BITCODE=spir32`. Otherwise, these instructions
should still work.

## Contributions

Please see the file [CONTRIBUTING.md](CONTRIBUTING.md) for further details if
you would like to contribute code, build systems, bug fixes or similar.

## Citation

If you use portDNN in your research, please cite the library as follows:

> Rod Burns, John Lawson, Duncan McBain, and Daniel Soutar. 2019. *Accelerated
> Neural Networks on OpenCL Devices Using portDNN.* In Proceedings of the
> International Workshop on OpenCL (IWOCL'19). ACM, New York, NY, USA, Article
> 10, 4 pages. DOI: https://doi.org/10.1145/3318170.3318183

```bibtex
@inproceedings{Burns:2019:ANN:3318170.3318183,
 author = {Burns, Rod and Lawson, John and McBain, Duncan and Soutar, Daniel},
 title = {Accelerated Neural Networks on OpenCL Devices Using portDNN},
 booktitle = {Proceedings of the International Workshop on OpenCL},
 series = {IWOCL'19},
 year = {2019},
 isbn = {978-1-4503-6230-6},
 location = {Boston, MA, USA},
 pages = {10:1--10:4},
 articleno = {10},
 numpages = {4},
 url = {http://doi.acm.org/10.1145/3318170.3318183},
 doi = {10.1145/3318170.3318183},
 acmid = {3318183},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {GPGPU, OpenCL, SYCL, machine learning, neural networks},
}
```

[supported platforms]: https://developer.codeplay.com/products/computecpp/ce/guides/platform-support
[issues]: https://github.com/codeplaysoftware/portDNN/issues
[ocl headers]: https://github.com/KhronosGroup/OpenCL-Headers
[codeplay developer]: https://developer.codeplay.com
