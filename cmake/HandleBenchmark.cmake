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


# Either finds or downloads benchmark library
#
# Depending on the cmake options this will try to find the benchmark library or
# download the library from github. If the library cannot be found or
# downloaded then the cmake configuration will fail, otherwise the library will
# be available in the `benchmark::benchmark` target.
cmake_minimum_required(VERSION 3.2.2)

if(NOT SNN_DOWNLOAD_BENCHMARK)
  find_package(benchmark)
endif()
if(SNN_DOWNLOAD_BENCHMARK OR (SNN_DOWNLOAD_MISSING_DEPS AND NOT benchmark_FOUND))
  message(STATUS "Downloading benchmark library")
  configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/benchmarkDownload.cmake.in
    benchmark-download/CMakeLists.txt
  )
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${snn_bench_BINARY_DIR}/benchmark-download
  )
  if(result)
    message(FATAL_ERROR "CMake step for benchmark failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${snn_bench_BINARY_DIR}/benchmark-download
  )
  if(result)
    message(FATAL_ERROR "Build step for benchmark failed: ${result}")
  endif()

  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  add_subdirectory(${snn_bench_BINARY_DIR}/benchmark-src
                   ${snn_bench_BINARY_DIR}/benchmark-build
                   EXCLUDE_FROM_ALL)

  mark_as_advanced(benchmark_FOUND)
  set(benchmark_LIBRARIES benchmark)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(benchmark
    DEFAULT_MSG
    benchmark_LIBRARIES
  )
  if(NOT TARGET benchmark::benchmark)
    add_library(benchmark::benchmark ALIAS ${benchmark_LIBRARIES})
  endif()
  if(NOT DEFINED benchmark_FOUND)
    # Earlier versions of cmake only set BENCHMARK_FOUND, not benchmark_FOUND.
    set(benchmark_FOUND ${BENCHMARK_FOUND})
  endif()
endif()
if(NOT benchmark_FOUND)
  message(FATAL_ERROR
    "Could NOT find benchmark, consider setting SNN_DOWNLOAD_MISSING_DEPS.")
endif()
