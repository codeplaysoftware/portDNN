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


cmake_minimum_required(VERSION 3.2.2)
project(snn_bench)
include(HandleBenchmark)
include(SNNHelpers)

add_library(bench_main STATIC bench_main.cc)
target_link_libraries(bench_main benchmark::benchmark)

snn_bench(
  WITH_SYCL
  TARGET
    set_buffer
  SOURCES
    basic_sycl.cc
  PUBLIC_LIBRARIES
    bench_main
)
snn_bench(
  WITH_SYCL
  TARGET
    simple_convolution
  SOURCES
    simple_convolution.cc
  PUBLIC_LIBRARIES
    bench_main
    Eigen::Eigen
    direct_conv2d
)
