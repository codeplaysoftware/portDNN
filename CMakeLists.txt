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
cmake_minimum_required(VERSION 3.2.2)
project(sycldnn LANGUAGES C CXX VERSION 0.1.0)

option(SNN_DOWNLOAD_GTEST
  "Download and build google test, rather than use the system version" OFF)
option(SNN_DOWNLOAD_BENCHMARK
  "Download and build google benchmark, rather than use the system version" ON)
option(SNN_DOWNLOAD_EIGEN "Download Eigen headers" ON)
option(SNN_DOWNLOAD_MISSING_DEPS
  "Download any dependencies which cannot be found" ON)

option(SNN_BUILD_TESTS "Whether or not to build unit tests" ON)
option(SNN_BUILD_BENCHMARKS "Whether or not to build benchmarks" ON)

option(SNN_TRISYCL "Use TriSYCL (default is ComputeCpp)" OFF)
option(SNN_FASTBUILD
  "Disable setting the cmake build type if no flag specified" OFF)

option(SNN_TEST_EIGEN "Test implementation using Eigen" ON)
option(SNN_EIGEN_LOCAL_MEM
  "Only compile the local memory versions of Eigen kernels" ON)
option(SNN_EIGEN_NO_LOCAL_MEM
  "Only compile the no local memory versions of Eigen kernels" OFF)

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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")
if(NOT SNN_FASTBUILD)
  include(DefaultBuildType)
endif()

if(SNN_TRISYCL)
  find_package(TriSYCL)
else()
  find_package(ComputeCpp)
  list(APPEND COMPUTECPP_USER_FLAGS -no-serial-memop)
endif()

include(HandleEigen)

if(SNN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
if(SNN_BUILD_BENCHMARKS)
  enable_testing()
  add_subdirectory(bench)
endif()

