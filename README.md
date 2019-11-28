# The SYCL-DNN neural network acceleration library

SYCL-DNN is a library implementing various neural network algorithms written
using the SYCL API, developed by Codeplay Software.

## Project layout

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

## Supported operations

SYCL-DNN is still in development, but at the moment we support the following
operations:

* 2D convolutions
* 2D depthwise convolutions
* 2D max & average pooling
* Relu and tanh activations

The convolution operations have several implementations, including tiled and
Winograd kernels. The supported data format is NHWC.

## Citation

If you use SYCL-DNN in your research, please cite the library as follows:

> Rod Burns, John Lawson, Duncan McBain, and Daniel Soutar. 2019. *Accelerated
> Neural Networks on OpenCL Devices Using SYCL-DNN.* In Proceedings of the
> International Workshop on OpenCL (IWOCL'19). ACM, New York, NY, USA, Article
> 10, 4 pages. DOI: https://doi.org/10.1145/3318170.3318183

```bibtex
@inproceedings{Burns:2019:ANN:3318170.3318183,
 author = {Burns, Rod and Lawson, John and McBain, Duncan and Soutar, Daniel},
 title = {Accelerated Neural Networks on OpenCL Devices Using SYCL-DNN},
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

## Requirements

SYCL-DNN is primarily tested on Ubuntu 16.04 LTS with the corresponding
default package versions, as summarized below.

* SYCL-DNN will generally match the most recently released ComputeCpp, though
  it is likely to be compatible with other versions. We test against the most
  recent version.

* Hardware capable of running SYCL code. For ComputeCpp this could be OpenCL
  1.2-capable hardware and drivers with SPIR 1.2/SPIR-V/PTX support.

* A C++-11 compiler and STL implementation. We test against GCC 5.4.0.

* CMake. Tested against version 3.5.1.

* Building documentation requires Doxygen and Graphviz/Dot. Tested
  against versions 1.8.11 and 2.38.0 respectively.

## Building SYCL-DNN

SYCL-DNN uses CMake as its build system. There are provisions in the CMake
files for downloading SYCL-DNN's dependencies automatically, for finding
other dependencies and for selecting which bits of SYCL-DNN to build. All
these configuration options can be found in the main CMakeLists.txt for the
project and will show up in the CMake GUI if you use it. By default, the
tests and library will be built, but not the benchmarks.

It is recommended to leave the option `SNN_DOWNLOAD_MISSING_DEPS` set to
on. This will automatically download the source libraries necessary for
SYCL-DNN to build and run (such as Google Test, Google benchmark and
the Eigen linear algebra library). Even if you already have these on your
machine, downloading them as part of the SYCL-DNN means a more consistent
configuration.

You will need to provide the location of the ComputeCpp install you are
using in the variable `ComputeCpp_DIR`. It should point to the folder
where `bin/`, `lib/` etc. are. This should be the only argument that is
mandatory, everything else should be optional. The default build type is
Release, though this can be overridden.

The following example shows how to specify the ComputeCpp location, then
build and run the tests.

```bash
# Setup build environment
mkdir build && cd build
cmake .. -DComputeCpp_DIR=/path/to/computecpp
# Compile SYCL-DNN
make -j`nproc`
# Run the tests
ctest
# If compiled with benchmark support, run just the benchmarks
ctest -C Benchmark -E test
```

## Cross-compilation with ComputeCpp

SYCL-DNN supports cross-compilation targeting a number of devices. However,
because of the two-step compilation process used in ComputeCpp, standard
CMake toolchain files won't provide enough information to SYCL-DNN's build
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
the headers [from GitHub](https://github.com/KhronosGroup/OpenCL-Headers).

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

Please see the file CONTRIBUTING.md for further details if you would like to
contribute code, build systems, bug fixes or similar.
