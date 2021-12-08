# Copyright 2018 Codeplay Software Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use these files except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
cmake_minimum_required(VERSION 3.3)

project(sycldnn LANGUAGES C CXX VERSION 0.6.0)

# Configuration options controlling automatic downloading of dependencies.
option(SNN_DOWNLOAD_GTEST
  "Download and build google test, rather than use the system version" ON)
option(SNN_DOWNLOAD_BENCHMARK
  "Download and build google benchmark, rather than use the system version" ON)
option(SNN_DOWNLOAD_EIGEN "Download Eigen headers" ON)
option(SNN_DOWNLOAD_SYCLBLAS "Download SYCL-BLAS repository" ON)
option(SNN_DOWNLOAD_MISSING_DEPS
  "Download any dependencies which cannot be found" ON)

# Configuration options controlling which SYCL-DNN components are built.
option(SNN_BUILD_TESTS "Whether or not to build unit tests" ON)
option(SNN_BUILD_BENCHMARKS "Whether or not to build benchmarks" ON)
option(SNN_BUILD_INTERNAL_BENCHMARKS
  "Whether or not to build internal benchmarks" OFF)
option(SNN_BUILD_EXTENDED_BENCHMARKS
  "Whether or not to build a more comprehesive (but extremely time consuming) set of benchmarks" OFF)
option(SNN_BUILD_LARGE_BATCH_BENCHMARKS
  "Whether or not to build benchmarks that use a batch sizes larger than 4" OFF)
option(SNN_BUILD_SAMPLES "Whether or not to build samples" ON)
option(SNN_BUILD_DOCUMENTATION "Whether or not to build documentation" ON)

# Configuration options controlling the installation of test and benchmark
# executables.
option(SNN_INSTALL_TESTS
  "Whether or not to include unit tests when installing" OFF)
option(SNN_INSTALL_BENCHMARKS
  "Whether or not to include benchmarks when installing" OFF)
option(SNN_INSTALL_SAMPLES
  "Whether or not to include samples when installing" OFF)

option(SNN_TRISYCL "Use TriSYCL (default is ComputeCpp)" OFF)
option(SNN_FASTBUILD
  "Disable setting the cmake build type if no flag specified" OFF)

# Testing configuration options
option(SNN_TEST_EIGEN "Test Eigen backend" OFF)
option(SNN_TEST_SYCLBLAS "Test SYCL-BLAS backend" OFF)
option(SNN_TEST_CLBLAST "Test CLBlast backend" OFF)
option(SNN_TEST_EIGEN_MATMULS
  "Enable testing using Eigen matmul in addition to internal matmul" OFF)
option(SNN_TEST_SYCLBLAS_MATMULS
  "Enable testing using SYCL-BLAS matmul in addition to internal matmul" OFF)
option(SNN_TEST_CLBLAST_MATMULS
  "Enable testing using CLBlast matmul in addition to internal matmul" OFF)

# Eigen configuration options.
option(SNN_EIGEN_LOCAL_MEM
  "Only compile the local memory versions of Eigen kernels" ON)
option(SNN_EIGEN_NO_LOCAL_MEM
  "Only compile the no local memory versions of Eigen kernels" OFF)
option(SNN_EIGEN_COMPRESS_NAMES
  "Compress Eigen SYCL kernel names" OFF)
option(SNN_EIGEN_NO_BARRIER
  "Use Eigen matmul which does not use barriers (implies NO_LOCAL_MEM)" OFF)

# Benchmark configuration options
option(SNN_BENCH_EIGEN "Benchmark SYCL-DNN using Eigen" OFF)
option(SNN_BENCH_SYCLBLAS "Benchmark SYCL-DNN using SYCL-BLAS" ON)
option(SNN_BENCH_CLBLAST "Benchmark SYCL-DNN using CLBlast" OFF)
option(SNN_BENCH_SNN "Benchmark SYCL-DNN's own matmul" OFF)
option(SNN_BENCH_MKLDNN "Whether or not to build MKL-DNN benchmarks" OFF)
option(SNN_BENCH_ARM_COMPUTE
  "Whether or not to build ARM compute library benchmarks" OFF)

set(SNN_DATA_TYPES float)
set(SNN_INDEX_TYPES int32_t)
option(SNN_ENABLE_DOUBLE "Enable double support for kernels and tests" OFF)
if(SNN_ENABLE_DOUBLE)
  list(APPEND SNN_DATA_TYPES double)
  add_definitions(-DSNN_USE_DOUBLE=1)
endif()
option(SNN_ENABLE_HALF "Enable half support for kernels and tests" OFF)
if(SNN_ENABLE_HALF)
  list(APPEND SNN_DATA_TYPES cl::sycl::half)
  add_definitions(-DSNN_USE_HALF=1)
endif()
option(SNN_ENABLE_64BIT_INDICES
  "Enable using 64 bit indices for very large tensors" OFF)
if(SNN_ENABLE_64BIT_INDICES)
  list(APPEND SNN_INDEX_TYPES int64_t)
  add_definitions(-DSNN_USE_INT64=1)
endif()
option(SNN_CONV2D_DIRECT_STATIC_KERNELS
  "Enable compiling static sizes of direct conv2d kernels" OFF)
if(SNN_CONV2D_DIRECT_STATIC_KERNELS)
  add_definitions(-DSNN_CONV2D_STATIC_DIRECT=1)
endif()
option(SNN_REGISTER_TILE_SPECIALISATIONS
  "Add specialisations to register tiles to help hoist to registers" OFF)
if(SNN_REGISTER_TILE_SPECIALISATIONS)
  add_definitions(-DSNN_REGISTER_TILE_SPECIALISATIONS=1)
endif()
option(SNN_VISIBILITY_HIDDEN
  "Set default visibility to hidden, reducing the number of exported symbols" ON)
if(SNN_VISIBILITY_HIDDEN)
  set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
  set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
endif()
set(SNN_HIGH_MEM_JOB_LIMIT 8 CACHE STRING
  "Number of concurrent build jobs for high memory targets (Ninja only)")
set_property(GLOBAL PROPERTY JOB_POOLS high_mem=${SNN_HIGH_MEM_JOB_LIMIT})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")
if(NOT SNN_FASTBUILD)
  include(DefaultBuildType)
endif()

if(SNN_TRISYCL)
  find_package(TriSYCL)
else()
  find_package(ComputeCpp)
  option(SNN_COMPUTECPP_USE_SERIAL_MEMOP
    "Replace memory operations (eg memset) in kernels with serial operations." OFF)
  if(NOT SNN_COMPUTECPP_USE_SERIAL_MEMOP)
    list(APPEND COMPUTECPP_USER_FLAGS -no-serial-memop)
  endif()
  list(APPEND COMPUTECPP_USER_FLAGS "-Xclang -cl-mad-enable")
  list(APPEND COMPUTECPP_USER_FLAGS "-Xclang -cl-no-signed-zeros")
endif()

set(CMAKE_DEBUG_POSTFIX "-debug")
set(include_dest "include")
set(library_dest "lib/sycldnn")
set(cmake_config_dest "${library_dest}/cmake")
set(runtime_dest "bin")

add_subdirectory(src)

if(SNN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
if(SNN_BUILD_BENCHMARKS)
  enable_testing()
  add_subdirectory(bench)
endif()

add_library(sycl_dnn SHARED
  $<TARGET_OBJECTS:direct_conv2d>
  $<TARGET_OBJECTS:tiled_conv2d>
  $<TARGET_OBJECTS:im2col_conv2d>
  $<TARGET_OBJECTS:winograd_conv2d>
  $<TARGET_OBJECTS:depthwise_conv2d>
  $<TARGET_OBJECTS:selector_conv2d>
  $<TARGET_OBJECTS:pooling>
  $<TARGET_OBJECTS:bias>
  $<TARGET_OBJECTS:pointwise>
  $<TARGET_OBJECTS:matmul>
  $<TARGET_OBJECTS:transpose>
  $<TARGET_OBJECTS:softmax>
  $<TARGET_OBJECTS:batchnorm>
)
snn_target(TARGET sycl_dnn WITH_SYCL)
set_target_properties(sycl_dnn PROPERTIES
  OUTPUT_NAME "sycldnn"
  VERSION ${sycldnn_VERSION}
  SOVERSION ${sycldnn_VERSION}
)

add_library(sycl_dnn_static STATIC
  $<TARGET_OBJECTS:direct_conv2d>
  $<TARGET_OBJECTS:tiled_conv2d>
  $<TARGET_OBJECTS:im2col_conv2d>
  $<TARGET_OBJECTS:winograd_conv2d>
  $<TARGET_OBJECTS:depthwise_conv2d>
  $<TARGET_OBJECTS:selector_conv2d>
  $<TARGET_OBJECTS:pooling>
  $<TARGET_OBJECTS:bias>
  $<TARGET_OBJECTS:pointwise>
  $<TARGET_OBJECTS:matmul>
  $<TARGET_OBJECTS:transpose>
  $<TARGET_OBJECTS:softmax>
  $<TARGET_OBJECTS:batchnorm>
)
snn_target(TARGET sycl_dnn_static WITH_SYCL)
set_target_properties(sycl_dnn_static PROPERTIES
  OUTPUT_NAME "sycldnn_static"
  VERSION ${sycldnn_VERSION}
)

include(GenerateExportHeader)
generate_export_header(sycl_dnn
  BASE_NAME "SNN"
  EXPORT_FILE_NAME "sycldnn/export.h"
)

include(CMakePackageConfigHelpers)
set(version_file "${CMAKE_CURRENT_BINARY_DIR}/cmake/sycldnn-version.cmake")
write_basic_package_version_file(${version_file}
  VERSION ${sycldnn_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(TARGETS sycl_dnn sycl_dnn_static
  EXPORT sycldnn
  LIBRARY DESTINATION ${library_dest}
  ARCHIVE DESTINATION ${library_dest}
  PUBLIC_HEADER DESTINATION ${include_dest}
  INCLUDES DESTINATION ${include_dest}
)
install(DIRECTORY include/sycldnn DESTINATION ${include_dest})
install(FILES ${version_file} DESTINATION ${cmake_config_dest})
install(FILES ${sycldnn_BINARY_DIR}/sycldnn/export.h DESTINATION ${include_dest}/sycldnn)
install(EXPORT sycldnn
  DESTINATION ${cmake_config_dest}
  NAMESPACE SYCLDNN::
  FILE sycldnn-config.cmake
)

# Optionally build the samples.
if(SNN_BUILD_SAMPLES)
  add_subdirectory(samples)
endif()

# Optionally build the documentation.
if(SNN_BUILD_DOCUMENTATION)
  add_subdirectory(docs)
endif()
