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


cmake_minimum_required(VERSION 3.10.2)
project(snn_bench_internal)

include(HandleCSV)

snn_bench(
  WITH_SYCL
  TARGET
    tiled_conv2d
  KERNEL_SOURCES
    tiled_conv2d_benchmark.cc
  PUBLIC_LIBRARIES
    bench_info
    bench_main
)

snn_bench(
  WITH_SYCL
  TARGET
    tiled_matmul
  SOURCES
    tiled_matmul_benchmark.cc
  OBJECTS
    $<TARGET_OBJECTS:internal_matmul>
  PUBLIC_LIBRARIES
    bench_info
    CSV::fast-cpp-csv-parser
)
