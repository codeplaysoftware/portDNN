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

# Creates a git-config.tmp file containing the commit hash and date. A custom
# target ensures that this command is always executed.
add_custom_target(generate_git_config
  ${CMAKE_CURRENT_SOURCE_DIR}/make_git_config.sh >
    ${CMAKE_CURRENT_BINARY_DIR}/git_config.tmp
  BYPRODUCTS
    git_config.tmp
  WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Conditionally copies the new git-config.tmp to git-config.h if the file has
# changed. This pattern is used to prevent constant rebuilds even though the git
# history hasn't changed.
add_custom_command(OUTPUT git_config.h
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different git_config.tmp git_config.h
  DEPENDS
    generate_git_config git_config.tmp
  WORKING_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}
)
add_library(bench_main STATIC bench_main.cc git_config.h)
target_link_libraries(bench_main benchmark::benchmark)
target_include_directories(bench_main PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(SNN_BUILD_EXTENDED_BENCHMARKS)
  list(APPEND _BENCHMARK_DEFINITIONS -DSNN_EXTENDED_BENCHMARKS)
endif()

if(SNN_BUILD_LARGE_BATCH_BENCHMARKS)
  list(APPEND _BENCHMARK_DEFINITIONS -DSNN_LARGE_BATCH_BENCHMARKS)
endif()

if(SNN_BENCH_EIGEN)
  include(HandleEigen)
  list(APPEND _BENCHMARK_DEFINITIONS -DSNN_BENCH_EIGEN)
  list(APPEND _BACKENDS Eigen::Eigen)
endif()

if(SNN_BENCH_SYCLBLAS)
  include(HandleSyclBLAS)
  list(APPEND _BENCHMARK_DEFINITIONS -DSNN_BENCH_SYCLBLAS)
  list(APPEND _BACKENDS SyclBLAS::SyclBLAS)
endif()

snn_bench(
  WITH_SYCL
  TARGET
    set_buffer
  KERNEL_SOURCES
    basic_sycl.cc
  PUBLIC_LIBRARIES
    bench_main
)
snn_bench(
  WITH_SYCL
  TARGET
    simple_convolution
  KERNEL_SOURCES
    conv2d/simple_convolution.cc
  OBJECTS
    $<TARGET_OBJECTS:direct_conv2d>
    $<TARGET_OBJECTS:tiled_conv2d>
    $<TARGET_OBJECTS:im2col_conv2d>
    $<TARGET_OBJECTS:winograd_conv2d>
  PUBLIC_LIBRARIES
    bench_main
    ${_BACKENDS} 
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)
snn_bench(
  WITH_SYCL
  TARGET
    resnet_convolution
  KERNEL_SOURCES
    conv2d/resnet.cc
  OBJECTS
    $<TARGET_OBJECTS:direct_conv2d>
    $<TARGET_OBJECTS:tiled_conv2d>
    $<TARGET_OBJECTS:im2col_conv2d>
    $<TARGET_OBJECTS:winograd_conv2d>
  PUBLIC_LIBRARIES
    bench_main
    ${_BACKENDS} 
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)
snn_bench(
  WITH_SYCL
  TARGET
    vgg_convolution
  KERNEL_SOURCES
    conv2d/vgg.cc
  OBJECTS
    $<TARGET_OBJECTS:direct_conv2d>
    $<TARGET_OBJECTS:tiled_conv2d>
    $<TARGET_OBJECTS:im2col_conv2d>
    $<TARGET_OBJECTS:winograd_conv2d>
  PUBLIC_LIBRARIES
    bench_main
    ${_BACKENDS}
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)
snn_bench(
  WITH_SYCL
  TARGET
    mobilenet_convolution
  KERNEL_SOURCES
    conv2d/mobilenet.cc
  OBJECTS
    $<TARGET_OBJECTS:direct_conv2d>
    $<TARGET_OBJECTS:tiled_conv2d>
    $<TARGET_OBJECTS:im2col_conv2d>
    $<TARGET_OBJECTS:winograd_conv2d>
  PUBLIC_LIBRARIES
    bench_main
    ${_BACKENDS}
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)
snn_bench(
  WITH_SYCL
  TARGET
    ssd_mobilenet_convolution
  KERNEL_SOURCES
    conv2d/ssd_mobilenet.cc
  OBJECTS
    $<TARGET_OBJECTS:direct_conv2d>
    $<TARGET_OBJECTS:tiled_conv2d>
    $<TARGET_OBJECTS:im2col_conv2d>
    $<TARGET_OBJECTS:winograd_conv2d>
  PUBLIC_LIBRARIES
    bench_main
    ${_BACKENDS}
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)
snn_bench(
  WITH_SYCL
  TARGET
    resnet_pooling
  SOURCES
    pooling/resnet.cc
  OBJECTS
    $<TARGET_OBJECTS:pooling>
  PUBLIC_LIBRARIES
    bench_main
    ${_BACKENDS}
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)
snn_bench(
  WITH_SYCL
  TARGET
    vgg_pooling
  SOURCES
    pooling/vgg.cc
  OBJECTS
    $<TARGET_OBJECTS:pooling>
  PUBLIC_LIBRARIES
    bench_main
    ${_BACKENDS}
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)
snn_bench(
  WITH_SYCL
  TARGET
    mobilenet_pooling
  SOURCES
    pooling/mobilenet.cc
  OBJECTS
    $<TARGET_OBJECTS:pooling>
  PUBLIC_LIBRARIES
    bench_main
    ${_BACKENDS}
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)
snn_bench(
  WITH_SYCL
  TARGET
    relu
  SOURCES
    pointwise/relu.cc
  OBJECTS
    $<TARGET_OBJECTS:pointwise>
  PUBLIC_LIBRARIES
    bench_main
    ${_BACKENDS}
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)
snn_bench(
  WITH_SYCL
  TARGET
    tanh
  SOURCES
    pointwise/tanh.cc
  OBJECTS
    $<TARGET_OBJECTS:pointwise>
  PUBLIC_LIBRARIES
    bench_main
    ${_BACKENDS}
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)
snn_bench(
  WITH_SYCL
  TARGET
    mobilenet_depthwise_convolution
  KERNEL_SOURCES
    depthwise_conv2d/mobilenet.cc
  OBJECTS
    $<TARGET_OBJECTS:depthwise_conv2d>
  PUBLIC_LIBRARIES
    bench_main
    ${_BACKENDS}
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)

if(SNN_BUILD_INTERNAL_BENCHMARKS)
  add_subdirectory(internal)
endif()

if(SNN_ARM_COMPUTE_ENABLED)
  find_package(ARMCompute REQUIRED)

  snn_bench(
    TARGET
      arm_resnet_convolution
    SOURCES
      conv2d/arm_resnet.cc
    PUBLIC_LIBRARIES
      bench_main
      ARMCompute::ARMCompute
      ARMCompute::Core
    PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
  )
  snn_bench(
    TARGET
      arm_vgg_convolution
    SOURCES
      conv2d/arm_vgg.cc
    PUBLIC_LIBRARIES
      bench_main
      ARMCompute::ARMCompute
      ARMCompute::Core
    PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
  )
endif()
