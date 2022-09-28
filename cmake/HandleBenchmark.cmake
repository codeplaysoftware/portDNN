# Copyright Codeplay Software Ltd.
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
cmake_minimum_required(VERSION 3.10.2)

include(SNNHelpers)
snn_include_guard(HANDLE_BENCHMARK)

if(NOT SNN_DOWNLOAD_BENCHMARK)
  find_package(benchmark)
endif()
if(SNN_DOWNLOAD_BENCHMARK OR (SNN_DOWNLOAD_MISSING_DEPS AND NOT benchmark_FOUND))
  find_package(Threads REQUIRED)
  include(ExternalProject)
  set(BENCHMARK_GIT_TAG "v1.3.0" CACHE STRING
    "Git tag, branch or commit to use for Google benchmark"
  )
  set(benchmark_SOURCE_DIR ${snn_bench_BINARY_DIR}/benchmark-src)
  set(benchmark_BINARY_DIR ${snn_bench_BINARY_DIR}/benchmark-build)
  set(benchmark_LIBNAME ${CMAKE_STATIC_LIBRARY_PREFIX}benchmark${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(benchmark_LIBRARIES ${benchmark_BINARY_DIR}/src/${benchmark_LIBNAME})

  # Need to explicitly declare byproducts of external projects, or Ninja gets
  # confused about dependencies.
  list(APPEND benchmark_BYPRODUCTS "${benchmark_LIBRARIES}")
  if(CMAKE_CROSSCOMPILING)
    set(cmake_toolchain
      "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
    )
  endif()
  ExternalProject_Add(benchmark
    GIT_REPOSITORY    https://github.com/google/benchmark.git
    GIT_TAG           ${BENCHMARK_GIT_TAG}
    GIT_SHALLOW       ON
    GIT_CONFIG        advice.detachedHead=false
    SOURCE_DIR        ${benchmark_SOURCE_DIR}
    BINARY_DIR        ${benchmark_BINARY_DIR}
    CMAKE_ARGS        -DBENCHMARK_ENABLE_TESTING=OFF
                      -DBUILD_SHARED_LIBS=OFF
                      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                      "-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG=1 -Wno-error=unused-but-set-variable"
                      -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                      ${cmake_toolchain}
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
    BUILD_BYPRODUCTS ${benchmark_BYPRODUCTS}
  )
  set(benchmark_INCLUDE_DIR ${benchmark_SOURCE_DIR}/include)
  # Have to explicitly make the include directory to add it to the library
  # target. This will be filled with the headers at build time when the
  # benchmark library is downloaded.
  file(MAKE_DIRECTORY ${benchmark_INCLUDE_DIR})
  set(benchmark_LINK_LIBRARIES Threads::Threads)
  find_library(LIBRT rt)
  if(LIBRT)
    list(APPEND benchmark_LINK_LIBRARIES ${LIBRT})
  endif()
  if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    list(APPEND benchmark_LINK_LIBRARIES Shlwapi)
  endif()

  add_library(benchmark::benchmark IMPORTED STATIC)
  add_dependencies(benchmark::benchmark benchmark)
  set_target_properties(benchmark::benchmark PROPERTIES
    IMPORTED_LOCATION             ${benchmark_LIBRARIES}
    INTERFACE_LINK_LIBRARIES      "${benchmark_LINK_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${benchmark_INCLUDE_DIR}"
  )
  mark_as_advanced(benchmark_FOUND, benchmark_LIBRARIES)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(benchmark
    DEFAULT_MSG
    benchmark_LIBRARIES
  )
  if(NOT DEFINED benchmark_FOUND)
    # Earlier versions of cmake only set BENCHMARK_FOUND, not benchmark_FOUND.
    set(benchmark_FOUND ${BENCHMARK_FOUND})
  endif()
endif()
if(NOT benchmark_FOUND)
  message(FATAL_ERROR
    "Could NOT find benchmark, consider setting SNN_DOWNLOAD_MISSING_DEPS.")
endif()
