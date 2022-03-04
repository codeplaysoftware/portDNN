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


# Try to find the Google Benchmark library and headers.
#
# If the library is found then the `benchmark::benchmark` target will be
# exported with the required include directories.
#
# Sets the following variables:
#   benchmark_FOUND        - system has benchmark lib
#   benchmark_INCLUDE_DIRS - the benchmark include directory
#   benchmark_LIBRARIES    - libraries needed to use benchmark
#
find_path(benchmark_INCLUDE_DIR
  NAMES benchmark/benchmark.h
  DOC "The google benchmark include directory"
)
find_library(benchmark_LIBRARY
  NAMES benchmark
  DOC "The google benchmark library"
)
mark_as_advanced(benchmark_FOUND benchmark_INCLUDE_DIRS benchmark_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(benchmark
  FOUND_VAR benchmark_FOUND
  REQUIRED_VARS benchmark_INCLUDE_DIR benchmark_LIBRARY
)
if(benchmark_FOUND)
  set(benchmark_INCLUDE_DIRS ${benchmark_INCLUDE_DIR})
  set(benchmark_LIBRARIES    ${benchmark_LIBRARY})
endif()

if(benchmark_FOUND AND NOT TARGET benchmark::benchmark)
  add_library(benchmark::benchmark IMPORTED UNKNOWN)
  set_target_properties(benchmark::benchmark PROPERTIES
    IMPORTED_LOCATION ${benchmark_LIBRARY}
  )
  set_target_properties(benchmark::benchmark PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${benchmark_INCLUDE_DIRS}"
  )
endif()
