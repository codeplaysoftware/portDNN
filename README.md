# SYCL-DNN Readme

## Introduction

This is the README document for SYCL-DNN, a library implementing various
neural network algorithms, written using the SYCL API.

## Contents

* bench/
    - Some benchmarks, used to track the performance of SYCL-DNN.
* cmake/
    - Contains helper files and functions for downloading dependencies
      and performing common tasks (like adding tests, libraries etc.).
* CMakeLists.txt
    - The root CMakeLists.txt file, refer to this when building SYCL-DNN.
* CONTRIBUTING.md
    - Information about how to contribute bug reports, code, or other works
      to this project.
* hooks/
    - Contains scripts that are suitable for adding to the .git/hooks folder
      (e.g. making sure that the repo is correctly formatted).
* include/
    - The directory under which the public interface is stored.
* LICENSE
    - The license this sofware is available under: Apache 2.0
* README.md
    - This readme file.
* src/
    - The source files of SYCL-DNN are kept in here.
* test/
    - All tests are bundled in this folder and are the mark of correctness
      when doing pre-merge tests.

## Requirements

* SYCL-DNN will generally match the most recently released ComputeCpp, though
  it is likely that it will be compatible with multiple versions concurrently
  as the SYCL interface is fixed. We test against the most recent version.

* OpenCL 1.2-capable hardware and drivers with SPIR 1.2/SPIR-V/PTX support

* A C++-11 compiler and STL implementation

* CMake version 3.2.2

## Setup

SYCL-DNN uses CMake as its build system. There are provisions in the CMake
files for downloading SYCL-DNN's dependencies automatically, for finding
other dependencies and for selecting which bits of SYCL-DNN to build. All
these configuration options can be found in the main CMakeLists.txt for the
project and will show up in the CMake GUI if you use it. By default, the
tests and library will be built, but not the benchmarks.

It is recommended to leave the option `SNN_DOWNLOAD_MISSING_DEPS` set to
on. This will automatically download the source libraries necessary for
SYCL-DNN to build and run (currently Google Test, Google benchmark and
the Eigen linear algebra library). Even if you already have these on your
machine, downloading them as part of the SYCL-DNN means a more consistent
configuration.

You will need to provide the location of the ComputeCpp install you are
using in the variable `COMPUTECPP_PACKAGE_ROOT_DIR`. It should point to
the folder where bin/, lib/ etc. are. This should be the only argument
that is mandatory, everything else should be optional. The default build
type is Release, though this can be overridden.

The following example shows how to specify the ComputeCpp location, then
build and run the tests.

```sh
mkdir build && cd build
cmake .. -DCOMPUTECPP_PACKAGE_ROOT_DIR=/path/to/computecpp
make -j`nproc`
# Can then run tests
ctest
# If compiled with benchmark support, can run benchmarks exclusively
ctest -C Benchmark -E test
```

## Troubleshooting

The master branch of SYCL-DNN should always compile and tests should always
pass on our supported platforms. Ideally we should be writing portable,
standards-compliant SYCL code, and as such it should pass tests on all
compatible OpenCL hardware. See CONTRIBUTING.md for details about creating
bug reports for this project.

## Maintainers

This project is written and maintained by
[Codeplay Software Ltd](https://www.codeplay.com/).
Please get in touch if you have any issues or questions - you can reach us at
[sycl@codeplay.com](mailto:sycl@codeplay.com).

## Contributions

Please see the file CONTRIBUTIONS.md for further details if you would like to
contribute code, build systems, bug fixes or similar.
