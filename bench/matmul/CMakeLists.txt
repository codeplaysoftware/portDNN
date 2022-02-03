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

snn_object_library(
  WITH_SYCL
  TARGET
    matmul_benchmark_functions
  KERNEL_SOURCES
    benchmark_functions.cc
  PUBLIC_LIBRARIES
    benchmark::benchmark
    ${_BACKENDS}
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)

function(snn_matmul_config_lib modelname)
  snn_object_library(
    TARGET
      ${modelname}_matmul_config
    SOURCES
      ${modelname}.cc
    PUBLIC_LIBRARIES
      benchmark::benchmark
    PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
  )
endfunction()

snn_matmul_config_lib(simple)
snn_matmul_config_lib(resnet)
snn_matmul_config_lib(vgg)
snn_matmul_config_lib(mobilenet)

function(snn_matmul_bench modelname)
  snn_bench(
    WITH_SYCL
    TARGET
      ${modelname}_matmul
    OBJECTS
      $<TARGET_OBJECTS:matmul_benchmark_functions>
      $<TARGET_OBJECTS:${modelname}_matmul_config>
    PUBLIC_LIBRARIES
      bench_main
      sycl_dnn
      ${_BACKENDS}
  )
endfunction()

snn_matmul_bench(simple)
snn_matmul_bench(resnet)
snn_matmul_bench(vgg)
snn_matmul_bench(mobilenet)
