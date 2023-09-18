# portDNN Configuration Options

portDNN's CMake provides a number of configuration options to control which
parts of the code will be built, tested and benchmarked. They are described
here.

## Build options

Option                             | Type     | Default   | Description
---------------------------------- | -------- | --------- | -----------
`CMAKE_BUILD_TYPE`                 | `STRING` | `Release` | Release, RelWithDebInfo etc. Controls compiler flags.
`SNN_FASTBUILD`                    | `BOOL`   | `OFF`     | Disables default-set `CMAKE_BUILD_TYPE` when `ON`
`SNN_BUILD_TESTS`                  | `BOOL`   | `ON`      | Enables the portDNN test suite
`SNN_BUILD_SAMPLES`                | `BOOL`   | `ON`      | Builds portDNN's sample code
`SNN_BUILD_BENCHMARKS`             | `BOOL`   | `ON`      | Builds portDNN's benchmarks
`SNN_BUILD_EXTENDED_BENCHMARKS`    | `BOOL`   | `OFF`     | `OFF` disables batch sizes 2, 8, 16,     64.
`SNN_BUILD_LARGE_BATCH_BENCHMARKS` | `BOOL`   | `OFF`     | `OFF` disables batch sizes    8, 16, 32, 64. Slow.
`SNN_BUILD_INTERNAL_BENCHMARKS`    | `BOOL`   | `OFF`     | Builds a large tiled convolution benchmark.
`SNN_BUILD_DOCUMENTATION`          | `BOOL`   | `ON`      | Generates HTML documentation. Requires Doxygen.
`SNN_VISIBILITY_HIDDEN`            | `BOOL`   | `ON`      | Hides library symbols by default, exporting select functions
`SNN_FORCE_COLOUR_DIAGNOSTICS`     | `BOOL`   | `ON`      | Forces compilers to output error messages in colour
`SNN_HIGH_MEM_JOB_LIMIT`           | `INT`    | `8`       | Number of concurrent build jobs for high memory targets (Ninja only)
`SNN_DEVICE_TRIPLE`                | `LIST`   | `spir64`  | Sets the DPC++ device triple(s). Semicolon-separated if multiple flags are passed
`SNN_DPCPP_ARCH`                   | `STRING` | Empty     | Sets the specific device architecture for DPC++ builds
`SNN_DPCPP_USER_FLAGS`             | `LIST`   | Empty     | Sets the extra compiler flags to pass to DPC++. Semicolon-separated if multiple flags are passed

## Download options

Option                      | Type     | Default         | Description
--------------------------- | -------- | --------------- | -----------
`SNN_DOWNLOAD_BENCHMARK`    | `BOOL`   | `ON`            | Download and build Google Benchmark, rather than look locally
`BENCHMARK_GIT_TAG`         | `STRING` | `v1.3.0`        | Commit-ish object (e.g. tag, branch, commit hash)
`SNN_DOWNLOAD_GTEST`        | `BOOL`   | `OFF`           | Download and build Google Test, rather than look locally
`GTEST_GIT_TAG`             | `STRING` | `release-1.10.0`| Commit-ish object (e.g. tag, branch, commit hash)
`SNN_DOWNLOAD_EIGEN`        | `BOOL`   | `ON`            | Download Eigen, rather than look locally
`EIGEN_REPO`                | `STRING` | Eigen GitLab    | The remote to download Eigen from
`EIGEN_GIT_TAG`             | `STRING` | `00de5707`      | Commit-ish object (e.g. tag, branch, commit hash)
`EIGEN_INCLUDE_DIR`         | `STRING` | build/eigen     | The Eigen include directory. Can be set by download.
`SNN_DOWNLOAD_SYCLBLAS`     | `BOOL`   | `ON`            | Download SYCLBLAS, rather than look locally
`sycl_blas_REPO`            | `STRING` | Codeplay GitHub | The remote to download SYCLBLAS from
`sycl_blas_GIT_TAG`         | `STRING` | `dd2455c`       | Commit-ish object (e.g. tag, branch, commit hash)
`SNN_DOWNLOAD_MISSING_DEPS` | `BOOL`   | `ON`            | Downloads any `-NOTFOUND` dependencies

## Test options

Option                      | Type   | Default | Description
--------------------------- | ------ | ------- | -----------
`SNN_TEST_EIGEN`            | `BOOL` | `OFF`   | Test the Eigen backend
`SNN_TEST_EIGEN_MATMULS`    | `BOOL` | `OFF`   | Use Eigen matmuls to test convolutions (very slow)
`SNN_TEST_SYCLBLAS`         | `BOOL` | `OFF`   | Test the SYCLBLAS backend
`SNN_TEST_SYCLBLAS_MATMULS` | `BOOL` | `OFF`   | Use SYCLBLAS matmuls to test convolutions (very slow)

## Benchmark options

Option                  | Type   | Default | Description
----------------------- | ------ | ------- | -----------
`SNN_BENCH_EIGEN`       | `BOOL` | `OFF`   | Build benchmarks with Eigen matmul support
`SNN_BENCH_SYCLBLAS`    | `BOOL` | `ON`    | Build benchmarks with SYCLBLAS support
`SNN_BENCH_MKLDNN`      | `BOOL` | `OFF`   | Build MKLDNN benchmarks
`SNN_BENCH_ARM_COMPUTE` | `BOOL` | `OFF`   | Build ARM Compute Library benchmarks
`SNN_BENCH_SNN`         | `BOOL` | `OFF`   | Build benchmarks with portDNN matmul support

## Eigen options

Option                     | Type     | Default | Description
-------------------------- | -------- | ------- | -----------
`SNN_EIGEN_LOCAL_MEM`      | `BOOL`   | `ON`    | Only compile Eigen kernels with local memory support
`SNN_EIGEN_NO_LOCAL_MEM`   | `BOOL`   | `OFF`   | Only compile Eigen kernels without local memory support
`SNN_EIGEN_COMPRESS_NAMES` | `BOOL`   | `OFF`   | Turns kernel names into hashes to avoid OpenCL driver bugs
`SNN_EIGEN_NO_BARRIER`     | `BOOL`   | `OFF`   | Use barrier-free matmul. Implies `NO_LOCAL_MEM`.

## ComputeCpp options

Option                  | Type     | Default  | Description
----------------------- | -------- | -------- | -----------
`ComputeCpp_DIR`        | `PATH`   | unset    | The ComputeCpp installation location
`ComputeCpp_HOST_DIR`   | `PATH`   | unset    | Crosscompiling only. ComputeCpp location for host arch.
`COMPUTECPP_USER_FLAGS` | `STRING` | unset    | Flags to be appended to `compute++` command line
`COMPUTECPP_BITCODE`    | `STRING` | `spir64` | Format of bitcode to be emitted by compiler (spirv64, nvptx64 etc.)

## Kernel options

Option                      | Type     | Default  | Description
--------------------------- | -------- | -------- | -----------
`SNN_ENABLE_DOUBLE`         | `BOOL`   | `OFF`    | Compiles kernels that operate on double-precision floats
`SNN_ENABLE_HALF`           | `BOOL`   | `OFF`    | Compiles kernels that operate on OpenCL half-precision floats
`SNN_ENABLE_64BIT_INDICES`  | `BOOL`   | `OFF`    | Enable 64-bit index types to allow large (> 2bn element) tensors
`SNN_CONV2D_STATIC_KERNELS` | `BOOL`   | `OFF`    | Enable compilation of static sizes of direct convolutions
`SNN_REGISTER_TILE_SPECIALIZATIONS` | `BOOL` | `OFF` | Specialises register tiles to help compiler keep data in registers

## Install options

Option                   | Type   | Default      | Description
------------------------ | ------ | ------------ | -----------
`CMAKE_INSTALL_PREFIX`   | `PATH` | `/usr/local` | Prefix added in front of all install paths
`SNN_INSTALL_TESTS`      | `BOOL` | `OFF`        | Controls installing tests as part of install target
`SNN_INSTALL_BENCHMARKS` | `BOOL` | `OFF`        | Controls installing benchmarks as part of install target
`SNN_INSTALL_SAMPLES`    | `BOOL` | `OFF`        | Controls installing samples as part of install target
